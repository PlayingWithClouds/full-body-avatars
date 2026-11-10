"""The avatar's pose in every frame: what the fits found, plus small corrections learned from the
images during training.

The fits match silhouettes and keypoints to within a few pixels; frames that disagree by that
much cannot all be matched by the same Gaussians, and the result is blur and ghosting. So each
training frame's body pose, head pose and expression may move slightly while the Gaussians train.
Frames kept out of training are never rendered there, so their corrections stay zero and they
measure the avatar as the fits pose it.

Every frame's corrections are parameters of their own. A frame is rendered once in a few hundred
steps; held in one tensor, its corrections would keep moving with Adam's momentum in the steps
between, a random walk. Parameters without a gradient are skipped by the optimizer instead.
"""

from __future__ import annotations

import numpy as np
import torch

from avatar_splats.body_models.anny import AnnyBody
from avatar_splats.body_models.hybrid import Avatar, Rig
from avatar_splats.body_models.ict import IctFaceKit
from avatar_splats.fitting.body import POSED_BONES, BodyFit
from avatar_splats.fitting.head import HeadFit, head_vertices

TRANSLATION_SCALE = 10.0
"""Corrections are penalised in radians, expression weights and decimetres, so that a 1 cm shift
costs as much as a turn of 0.1 radian."""


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array)).float()


class _Corrections(torch.nn.Module):
    """A frame's corrections of rotation vectors, translation and expression, all zero at first."""

    def __init__(self, rotations: tuple[int, ...], expressions: int):
        super().__init__()
        self.rotation = torch.nn.Parameter(torch.zeros(rotations))
        self.translation = torch.nn.Parameter(torch.zeros(3))
        self.expression = torch.nn.Parameter(torch.zeros(expressions))

    def prior(self, rotation_mask: torch.Tensor | float = 1.0) -> torch.Tensor:
        return (
            (self.rotation * rotation_mask).square().sum()
            + (TRANSLATION_SCALE * self.translation).square().sum()
            + self.expression.square().sum()
        )


class _HeadVideo(torch.nn.Module):
    """One head video's fitted head poses and expressions, and their corrections."""

    def __init__(self, fit: HeadFit):
        super().__init__()
        self.register_buffer("rotations", _tensor(fit.rotations))
        self.register_buffer("translations", _tensor(fit.translations))
        self.register_buffer("expressions", _tensor(fit.expressions))
        self.register_buffer("eye_rotations", _tensor(fit.eye_rotations))
        self.frames = torch.nn.ModuleList(
            _Corrections((3,), fit.expressions.shape[1]) for _ in range(len(fit.rotations))
        )


class Posing(torch.nn.Module):
    """Computes the avatar's vertices for a frame of any of the videos."""

    def __init__(
        self,
        avatar: Avatar,
        body: AnnyBody,
        head: IctFaceKit,
        body_fit: BodyFit,
        head_fits: dict[str, HeadFit],
    ):
        super().__init__()
        self.rig = Rig(avatar)
        self.body = body
        self.head = head
        self.head_start = avatar.head_start
        frames, bones, _ = body_fit.rotations.shape
        expressions = len(avatar.expression_names)

        self.register_buffer("phenotype", _tensor(body_fit.phenotype)[None])
        self.register_buffer("local_changes", _tensor(body_fit.local_changes)[None])
        self.register_buffer("body_rotations", _tensor(body_fit.rotations))
        self.register_buffer("body_translations", _tensor(body_fit.translation))
        self.register_buffer("posed", torch.zeros(bones, 1))
        self.posed[[body.bone(name) for name in POSED_BONES] + [0]] = 1
        self.body_frames = torch.nn.ModuleList(
            _Corrections((bones, 3), expressions) for _ in range(frames)
        )

        some_fit = next(iter(head_fits.values()))
        self.register_buffer("identity", _tensor(some_fit.identity)[None])
        self.register_buffer("centre", _tensor(some_fit.centre))
        self.heads = torch.nn.ModuleDict(
            {video: _HeadVideo(fit) for video, fit in head_fits.items()}
        )

    def vertices(self, video: str, index: int) -> torch.Tensor:
        """``(V, 3)`` the avatar's vertices in frame ``index`` of ``video``, corrections included.

        For head videos only the head's vertices are meaningful; the others are zero.
        """
        if video == "body":
            correction = self.body_frames[index]
            rotations = self.body_rotations[index] + correction.rotation * self.posed
            translation = self.body_translations[index] + correction.translation
            # The avatar's bind poses are the fitted body's rest skeleton.
            skinning = self.body.skinning(self.rig.bind_poses, rotations[None], translation[None])
            expression = correction.expression.clamp(0, 1)
            return self.rig(skinning, expression[None])[0]

        head = self.heads[video]
        correction = head.frames[index]
        heads = head_vertices(
            self.head,
            self.identity,
            (head.expressions[index] + correction.expression).clamp(0, 1)[None],
            head.eye_rotations[index][None],
            (head.rotations[index] + correction.rotation)[None],
            (head.translations[index] + correction.translation)[None],
            self.centre,
        )[0]
        after = len(self.rig.rest_vertices) - self.head_start - len(heads)
        return torch.cat((heads.new_zeros(self.head_start, 3), heads, heads.new_zeros(after, 3)))

    def prior(self, video: str, index: int) -> torch.Tensor:
        """How far the corrections take frame ``index`` of ``video`` from its fit."""
        if video == "body":
            return self.body_frames[index].prior(self.posed)
        return self.heads[video].frames[index].prior()

    def corrections(self) -> list[torch.nn.Parameter]:
        """The learned corrections, for the optimizer."""
        corrections = list(self.body_frames.parameters())
        for head in self.heads.values():
            corrections += list(head.frames.parameters())
        return corrections
