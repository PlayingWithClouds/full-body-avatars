"""Optimizing the bound Gaussians against the frames of all three videos.

The number of Gaussians is fixed from the start, a budget the game runtime can afford: they are
spread over the mesh, denser on the head, and only moved, shaped and coloured, never split. The
fits are taken as they are; the Gaussians absorb what they got slightly wrong.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from avatar_splats.body_models.hybrid import BODY, Avatar
from avatar_splats.config import TrainingConfig
from avatar_splats.gaussians import BoundGaussians, render
from avatar_splats.records import ArrayRecord
from avatar_splats.training.losses import psnr, ssim
from avatar_splats.training.views import View, head_gaussians

logger = logging.getLogger(__name__)

MIN_OPACITY = 0.005
"""Gaussians fainter than this after training are dropped."""


@dataclass(frozen=True)
class Evaluation(ArrayRecord):
    """Quality on the frames kept out of training."""

    videos: np.ndarray
    frames: np.ndarray
    psnr: np.ndarray
    """``(N,)`` in dB, within the compared region, against the masked frame on black."""


def render_view(
    gaussians: BoundGaussians,
    faces: torch.Tensor,
    view: View,
    head: torch.Tensor,
    sh_degree: int,
    background: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render a view's frame: ``(H, W, 3)`` image and ``(H, W)`` opacity."""
    means, rotations, scales = gaussians.place(view.vertices, faces)
    chosen = head if view.head_only else slice(None)
    image, alpha, _ = render(
        means[chosen],
        rotations[chosen],
        scales[chosen],
        gaussians.opacities()[chosen],
        gaussians.sh()[chosen],
        view.camera,
        sh_degree,
        background,
    )
    return image, alpha


def target(view: View, background: torch.Tensor) -> torch.Tensor:
    """The frame with everything outside the performer's mask replaced by ``background``."""
    mask = view.mask.unsqueeze(-1)
    return view.image.float() / 255 * mask + background * (1 - mask)


def train(
    views: list[View], avatar: Avatar, config: TrainingConfig
) -> tuple[BoundGaussians, Evaluation]:
    device = config.device
    faces = torch.from_numpy(avatar.faces).to(device)
    parts = torch.from_numpy(avatar.face_parts).to(device)
    density = torch.where(parts == BODY, 1.0, config.head_density)
    gaussians = BoundGaussians.initialize(
        torch.from_numpy(avatar.rest_vertices).to(device),
        faces,
        density,
        config.num_gaussians,
        config.sh_degree,
    ).to(device)
    head = head_gaussians(avatar, gaussians.faces)

    training = [v for v in views if v.frame % config.holdout_every]
    held_out = [v for v in views if not v.frame % config.holdout_every]
    rates = config.learning_rates
    optimizer = torch.optim.Adam(
        [
            {"params": [gaussians.offsets], "lr": rates["offsets"]},
            {"params": [gaussians.rotations], "lr": rates["rotations"]},
            {"params": [gaussians.log_scales], "lr": rates["scales"]},
            {"params": [gaussians.opacity_logits], "lr": rates["opacities"]},
            {"params": [gaussians.sh_dc], "lr": rates["sh_dc"]},
            {"params": [gaussians.sh_rest], "lr": rates["sh_rest"]},
        ],
        eps=1e-15,
    )
    generator = random.Random(0)
    for step in range(config.iterations):
        view = generator.choice(training)
        background = torch.rand(3, device=device)
        sh_degree = 0 if step < config.iterations // 3 else config.sh_degree
        image, alpha = render_view(gaussians, faces, view, head, sh_degree, background)
        terms = _loss(image, alpha, view, background, gaussians, config)
        optimizer.zero_grad()
        sum(terms.values()).backward()
        optimizer.step()
        if step % 1000 == 0 or step == config.iterations - 1:
            logger.info(
                "step %d: %s", step, ", ".join(f"{k} {v.item():.4f}" for k, v in terms.items())
            )

    gaussians = _prune(gaussians)
    return gaussians, evaluate(gaussians, faces, held_out, avatar, config.sh_degree)


@torch.no_grad()
def evaluate(
    gaussians: BoundGaussians,
    faces: torch.Tensor,
    views: list[View],
    avatar: Avatar,
    sh_degree: int,
) -> Evaluation:
    head = head_gaussians(avatar, gaussians.faces)
    black = torch.zeros(3, device=faces.device)
    scores = []
    for view in views:
        image, _ = render_view(gaussians, faces, view, head, sh_degree, black)
        scores.append(psnr(image, target(view, black), view.region))
    evaluation = Evaluation(
        videos=np.array([v.video for v in views]),
        frames=np.array([v.frame for v in views]),
        psnr=np.array(scores),
    )
    by_video = defaultdict(list)
    for video, score in zip(evaluation.videos, evaluation.psnr, strict=True):
        by_video[video].append(score)
    for video, values in by_video.items():
        logger.info("held-out %s: PSNR %.2f dB over %d frames", video, np.mean(values), len(values))
    return evaluation


def _loss(
    image: torch.Tensor,
    alpha: torch.Tensor,
    view: View,
    background: torch.Tensor,
    gaussians: BoundGaussians,
    config: TrainingConfig,
) -> dict[str, torch.Tensor]:
    goal = target(view, background)
    region = view.region
    area = region.sum().clamp_min(1)

    def within(values: torch.Tensor) -> torch.Tensor:
        return (values * region).sum() / area

    mask = F.binary_cross_entropy(alpha.clamp(1e-5, 1 - 1e-5), view.mask, reduction="none")
    return {
        "l1": (1 - config.ssim_weight) * within((image - goal).abs().mean(-1)),
        "ssim": config.ssim_weight * within(1 - ssim(image, goal)),
        "mask": config.mask_weight * within(mask),
        "offsets": config.offset_weight * gaussians.offsets.square().sum(-1).mean(),
        "scales": config.scale_weight
        * F.relu(gaussians.log_scales.exp() - 1).square().sum(-1).mean(),
    }


def _prune(gaussians: BoundGaussians) -> BoundGaussians:
    record = gaussians.record()
    kept = 1 / (1 + np.exp(-record.opacity_logits)) >= MIN_OPACITY
    logger.info("kept %d of %d Gaussians", kept.sum(), len(kept))
    pruned = type(record)(**{name: value[kept] for name, value in vars(record).items()})
    return BoundGaussians(pruned).to(gaussians.faces.device)
