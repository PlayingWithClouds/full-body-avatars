"""How gravity reshapes the breasts as she leans: corrective shapes an engine adds before
skinning, weighted by where gravity points in her chest's axes.

Anny's breasts only follow the skeleton, so they keep their upright shape when she lies down or
leans forward. Here five shapes are added to the body vertices (before skinning), each weighted by
gravity's direction ``g`` in the chest bone's axes (``breasts.CHEST_BONE``; its rest-pose axes
are the world's), less its upright direction ``DOWN``: with ``d = g - DOWN``,

=================  ===============  ===================================================
shape              weight           where it shows
=================  ===============  ===================================================
breastsForward     max(-d_y, 0)     leaning forward, lying on her front (to 1)
breastsBack        max(d_y, 0)      leaning back, lying on her back (to 1)
breastsLeft        max(d_x, 0)      lying on her left side (to 1)
breastsRight       max(-d_x, 0)     lying on her right side (to 1)
breastsLift        d_z              no longer pulled down: 1 lying, 2 upside down
=================  ===============  ===================================================

(``gravity_weights``). Upright, all are 0: the rest shape is the upright one, which the shape fit
(``breasts``) took from upright close-ups.

**The shapes.** A prior: each vertex moves along its shape's direction (gravity's change) by
``sag`` times its depth in the breast squared (``BreastRegion.tissue``): the tip most, the chest
not at all. Close-ups in other postures (their gravity from the shape fit, taking the camera as
upright) refine it: ``sag`` and smooth residuals (averaged over the mesh as the shape fit's
offsets, faded by the breast region). Her body is taken as symmetric: the right shape is the
left one mirrored, the others are their own mirror images; so every close-up teaches both sides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import tqdm

from avatar_splats.records import ArrayRecord
from avatar_splats.regions.breasts import (
    CHEST_BONE,
    MILLIMETRE,
    BreastFit,
    Scene,
    averaging_matrix,
    data_loss,
    smooth,
)

GRAVITY_SHAPES = ("breastsForward", "breastsBack", "breastsLeft", "breastsRight", "breastsLift")
DOWN = (0.0, 0.0, -1.0)
"""Gravity in the chest's axes when she stands upright."""
DIRECTIONS = np.array([(0, -1, 0), (0, 1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1)], np.float32)
"""Per shape the way the prior moves the breasts: gravity's change that weights it."""
LEFT, RIGHT = 2, 3


def gravity_weights(gravity):
    """``(..., 5)`` the shapes' weights for ``(..., 3)`` unit gravity in the chest's axes; numpy
    arrays or tensors."""
    lib = torch if isinstance(gravity, torch.Tensor) else np
    d = gravity - (gravity.new_tensor(DOWN) if lib is torch else np.asarray(DOWN))
    zero = lib.zeros_like(d[..., 0])
    return lib.stack(
        [lib.maximum(-d[..., 1], zero), lib.maximum(d[..., 1], zero),
         lib.maximum(d[..., 0], zero), lib.maximum(-d[..., 0], zero), d[..., 2]],
        -1,
    )  # fmt: skip


def mirror_map(vertices: np.ndarray) -> np.ndarray:
    """``(B,)`` each vertex's mirror image across x = 0: the vertex nearest its reflection."""
    from scipy.spatial import cKDTree

    reflected = vertices * np.array([-1.0, 1.0, 1.0])
    return cKDTree(vertices).query(reflected)[1]


@dataclass(frozen=True)
class GravitySettings:
    steps: int
    learning_rate: float
    sag: float
    """Metres the prior moves the tip of the breast for a weight of 1."""
    smoothing_steps: int
    min_identity: float
    min_iou: float
    """As the shape fit's: which close-ups teach the shapes."""
    weights: dict[str, float]
    """Per loss its weight: outline, normals, nipples, joints, pose, residuals (per square
    millimetre), sag (on its log's change)."""


class GravityModel(torch.nn.Module):
    def __init__(
        self, rest: torch.Tensor, faces: torch.Tensor, weights: torch.Tensor,
        tissue: torch.Tensor, sag: float, smoothing_steps: int,
    ):  # fmt: skip
        super().__init__()
        device = rest.device
        self.matrix = averaging_matrix(faces, len(rest))
        self.smoothing_steps = smoothing_steps
        self.register_buffer("weights", weights[:, None])
        self.register_buffer("tissue", tissue[:, None])
        self.register_buffer("mirror", torch.from_numpy(mirror_map(rest.cpu().numpy())).to(device))
        self.register_buffer("directions", torch.from_numpy(DIRECTIONS).to(device))
        self.log_sag = torch.nn.Parameter(torch.tensor(float(np.log(sag)), device=device))
        self.log_sag0 = float(np.log(sag))
        self.values = torch.nn.Parameter(
            torch.zeros(len(GRAVITY_SHAPES), len(rest), 3, device=device)
        )

    def mirrored(self, offsets: torch.Tensor) -> torch.Tensor:
        """``(..., B, 3)`` offsets reflected across x = 0."""
        flip = offsets.new_tensor([-1.0, 1.0, 1.0])
        return offsets[..., self.mirror, :] * flip

    def residuals(self) -> torch.Tensor:
        """``(S, B, 3)`` metres: the smooth residuals, faded by the region, symmetric."""
        values = self.values * self.weights
        smoothed = torch.stack(
            [smooth(v, self.matrix, self.smoothing_steps) for v in values]
        ) * self.weights * MILLIMETRE  # fmt: skip
        out = (smoothed + self.mirrored(smoothed)) / 2  # forward, back, lift: their own mirror
        out[RIGHT] = self.mirrored(smoothed[LEFT])
        out[LEFT] = smoothed[LEFT]
        return out

    def prior(self) -> torch.Tensor:
        """``(S, B, 3)`` metres: the tissue moving with gravity's change."""
        reach = torch.exp(self.log_sag) * self.tissue.square()
        return reach[None] * self.directions[:, None, :]

    def shapes(self) -> torch.Tensor:
        return self.prior() + self.residuals()


@dataclass(frozen=True)
class GravityFit(ArrayRecord):
    """The breasts' posture correctives, and how they fit the close-ups."""

    names: np.ndarray
    """``(S,)`` ``GRAVITY_SHAPES``."""
    shapes: np.ndarray
    """``(S, B, 3)`` metres added to the body vertices at weight 1."""
    sag: float
    """The fitted ``GravitySettings.sag``."""
    keys: np.ndarray
    weights: np.ndarray
    """``(N, S)`` each close-up's weights."""
    used: np.ndarray
    """``(N,)`` bool: the close-ups that taught the shapes."""
    normal_before: np.ndarray
    normal_after: np.ndarray
    """``(N,)`` mean degrees between rendered and Sapiens2's normals over the breasts, without
    and with the correctives (each with its close-ups placed)."""
    iou_before: np.ndarray
    iou_after: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    turns: np.ndarray
    """The close-ups' places with the correctives (as ``BreastFit``'s)."""


def fit_gravity(
    body, arrays: dict, bone_labels: list[str], closeups, fit: BreastFit,
    settings: GravitySettings, device: str, progress: bool = True,
) -> GravityFit:  # fmt: skip
    """Fit the posture correctives to ``closeups``, on the avatar's body with ``fit``'s
    breasts, the close-ups placed and their gravity measured as ``fit`` left them.

    Args:
        body: an ``AnnyBody`` on ``device``.
        arrays: the ``.avatar`` file's arrays and ``head_start``.
    """
    s = settings
    weights = torch.from_numpy(fit.weights).float().to(device)
    nipples = torch.from_numpy(fit.nipples).long().to(device)
    scene = Scene(body, arrays, bone_labels, closeups, weights, nipples, 256, device)
    rest = scene.rest + torch.from_numpy(fit.change).float().to(device)
    tissue = torch.from_numpy(fit.tissue).float().to(device)
    model = GravityModel(rest, scene.faces, weights, tissue, s.sag, s.smoothing_steps)
    shape_weights = gravity_weights(torch.from_numpy(fit.gravity).float().to(device))
    places = [torch.nn.Parameter(torch.from_numpy(getattr(fit, k)).float().to(device))
              for k in ("rotations", "translations", "turns")]  # fmt: skip
    identity = np.nan_to_num(closeups.identity, nan=-1.0)
    used = (identity >= s.min_identity) & (fit.iou_after >= s.min_iou)
    on = torch.from_numpy(used).to(device)

    def losses(corrected: bool, scores: bool = False) -> dict:
        offsets = torch.einsum("ns,sbc->nbc", shape_weights, model.shapes()) if corrected else 0
        vertices, transforms = scene.posing(rest + offsets, *places)  # each its own shape
        return scene.losses(vertices, transforms, scores)

    with torch.no_grad():
        before = losses(False, scores=True)
    optimiser = torch.optim.Adam([*places, model.values, model.log_sag], lr=s.learning_rate)
    w = s.weights
    for _ in tqdm(range(s.steps), desc="gravity", disable=not progress, leave=False):
        optimiser.zero_grad()
        parts = losses(True)
        loss = data_loss({k: v[on] for k, v in parts.items()}, w)
        loss = loss + w["pose"] * places[2].square().sum((1, 2)).mean()
        loss = loss + w["residuals"] * model.residuals().div(MILLIMETRE).square().sum(-1).mean()
        loss = loss + w["sag"] * (model.log_sag - model.log_sag0).square()
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        after = losses(True, scores=True)

    def numpy(values):
        return values.detach().cpu().numpy()

    return GravityFit(
        names=np.array(GRAVITY_SHAPES), shapes=numpy(model.shapes()),
        sag=float(torch.exp(model.log_sag).detach()), keys=fit.keys, weights=numpy(shape_weights),
        used=used, normal_before=numpy(before["angle"]), normal_after=numpy(after["angle"]),
        iou_before=numpy(before["iou"]), iou_after=numpy(after["iou"]),
        rotations=numpy(places[0]), translations=numpy(places[1]), turns=numpy(places[2]),
    )  # fmt: skip


BODY_SHAPES_FORMAT = 5
"""The ``.avatar`` format version that brought body shapes (``body_shape_basis``)."""


def with_gravity(file, gravity: GravityFit):
    """A copy of an ``AvatarFile`` with the posture correctives: ``body_shape_basis``
    ``(S, head_start, 3)`` and their ``metadata["body_shapes"]``, format 5."""
    from avatar_splats.export.avatar_file import AvatarFile

    metadata = dict(file.metadata)
    metadata["format_version"] = max(int(metadata["format_version"]), BODY_SHAPES_FORMAT)
    metadata["body_shapes"] = {
        "names": [str(n) for n in gravity.names],
        "driver": {"kind": "gravity", "bone": CHEST_BONE, "down": list(DOWN)},
    }
    arrays = dict(file.arrays, body_shape_basis=gravity.shapes.astype(np.float32))
    return AvatarFile(metadata, arrays)
