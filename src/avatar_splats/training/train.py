"""Optimizing the avatar's textures and Gaussians against the frames of all three videos.

The avatar is drawn as a game draws it: the textured mesh, then the Gaussians over it (see
``rendering.composite``). The textures are baked from the training frames, and baked again every
``rebake_every`` steps from the poses training has corrected (see ``textures``).

A Gaussian's colour is the texture's at its anchor plus a learned correction that is kept small.
Gaussians overlap; when they disagree in colour, which one shows depends on the order they are
blended in, and that changes as the view turns, so the avatar shimmers. Tied to the texture they
agree, and each re-bake carries them along. Opacity is kept low where Gaussians add nothing, so
the mesh shows through.

Near the edge of the performer's mask the frames mix skin with the backdrop, so colours are not
compared within ``edge_band`` pixels of it; the silhouette still is.

The number of Gaussians is fixed from the start, a budget the game runtime can afford. They are
spread over the mesh by a density per part (denser on the face, none on the hands) times how
well the videos saw each place: where no frame saw the surface squarely, there is nothing for
Gaussians to learn, and the filled-in texture shows alone. They are only moved, shaped and
coloured, never split, and kept no larger than ``max_size``: thin discs lying on the skin,
a detail layer over the texture. Each training frame's pose and
expression is corrected along with them (see ``posing``), and each head video's colours are
matched to the body video's (see ``colour``).
"""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from avatar_splats.body_models.hybrid import (
    BODY,
    HANDS,
    HEAD_PARTS,
    HEAD_SKIN,
    HEAD_TEXTURE,
    Avatar,
)
from avatar_splats.config import TrainingConfig
from avatar_splats.gaussians import BoundGaussians
from avatar_splats.gaussians.bound import SH_C0
from avatar_splats.records import ArrayRecord
from avatar_splats.rendering import MeshImage, composite, render_mesh, sample_texture
from avatar_splats.training.colour import ColourCorrection
from avatar_splats.training.consistency import view_spread
from avatar_splats.training.losses import psnr, ssim
from avatar_splats.training.posing import Posing
from avatar_splats.training.textures import Textures, bake_textures
from avatar_splats.training.views import View, head_gaussians

logger = logging.getLogger(__name__)

MIN_OPACITY = 0.005
"""Gaussians fainter than this after training are dropped."""


@dataclass(frozen=True)
class Evaluation(ArrayRecord):
    """Quality on every frame: training frames as corrected, held-out ones as fitted."""

    videos: np.ndarray
    frames: np.ndarray
    held_out: np.ndarray
    """``(N,)`` bool: the frame was kept out of training."""
    psnr: np.ndarray
    """``(N,)`` in dB over ``scored_pixels``, against the masked frame."""
    view_spread: np.ndarray
    """``(P,)`` for points on the body, the standard deviation of their colour between viewing
    directions all around the avatar (see ``consistency``), 0 to 1."""
    head_view_spread: np.ndarray
    """``(P,)`` the same for points on the head's skin."""


class Scene(torch.nn.Module):
    """Everything drawn: the avatar's mesh, its textures and its Gaussians."""

    def __init__(self, avatar: Avatar, textures: Textures, gaussians: BoundGaussians):
        super().__init__()
        self.textures = textures
        self.gaussians = gaussians
        self.register_buffer("faces", torch.from_numpy(avatar.faces).long())
        self.register_buffer("uvs", torch.from_numpy(avatar.uvs).float())
        self.register_buffer("textures_of_faces", torch.from_numpy(avatar.face_textures).long())
        parts = torch.from_numpy(avatar.face_parts)
        on_head = torch.isin(parts, torch.tensor(HEAD_PARTS))
        self.register_buffer("head_faces", torch.nonzero(on_head).squeeze(-1))
        self.avatar = avatar
        self.set_gaussians(gaussians)

    def set_gaussians(self, gaussians: BoundGaussians) -> None:
        """Draw these Gaussians, coloured by themselves alone."""
        self.gaussians = gaussians
        device = self.uvs.device
        faces, barycentric = gaussians.faces.to(device), gaussians.barycentric.to(device)
        self.register_buffer("head_gaussians", head_gaussians(self.avatar, faces))
        anchors = (barycentric.unsqueeze(-1) * self.uvs[faces]).sum(dim=1)
        self.register_buffer("anchor_uvs", anchors)
        self.register_buffer("colour_base", torch.zeros(gaussians.sh_dc.shape, device=device))

    @torch.no_grad()
    def follow_textures(self) -> None:
        """Make the textures' colours at the Gaussians' anchors the base their colours are
        learned relative to."""
        on_head = self.textures_of_faces[self.gaussians.faces] == HEAD_TEXTURE
        colours = torch.empty(len(self.gaussians), 3, device=self.faces.device)
        for using, texture in zip((~on_head, on_head), self.textures.textures(), strict=True):
            colours[using] = sample_texture(texture, self.anchor_uvs[using])
        self.colour_base[:, 0] = (colours - 0.5) / SH_C0

    @torch.no_grad()
    def fold_colours(self) -> None:
        """Turn the Gaussians' colours from corrections of the texture into colours of their own,
        as the avatar file stores them."""
        self.gaussians.sh_dc += self.colour_base
        self.colour_base.zero_()

    def sh(self) -> torch.Tensor:
        """``(G, K, 3)`` the Gaussians' colours, the texture's included."""
        gaussians = self.gaussians
        return torch.cat((gaussians.sh_dc + self.colour_base, gaussians.sh_rest), dim=1)

    def draw_mesh(self, vertices: torch.Tensor, view: View) -> MeshImage:
        """The textured mesh alone, posed as ``(V, 3)`` vertices, over black. Head videos see the
        head alone."""
        drawn = self.head_faces if view.head_only else slice(None)
        return render_mesh(
            vertices,
            self.faces[drawn],
            self.uvs[drawn],
            self.textures_of_faces[drawn],
            self.textures.textures(),
            view.camera,
            vertices.new_zeros(3),
        )

    def render(
        self,
        vertices: torch.Tensor,
        view: View,
        sh_degree: int,
        occlusion_margin: float,
        mesh: MeshImage | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw the avatar, posed as ``(V, 3)`` vertices, for a view: ``(H, W, 3)`` colours over
        black and ``(H, W)`` opacity. Head videos see the head alone.

        Args:
            mesh: the mesh as ``draw_mesh`` drew it, if already drawn.
        """
        if mesh is None:
            mesh = self.draw_mesh(vertices, view)
        gaussians = self.gaussians
        means, rotations, scales = gaussians.place(vertices, self.faces)
        margins = occlusion_margin * gaussians.triangle_sizes(vertices, self.faces).detach()
        chosen = self.head_gaussians if view.head_only else slice(None)
        return composite(
            means[chosen],
            rotations[chosen],
            scales[chosen],
            gaussians.opacities()[chosen],
            self.sh()[chosen],
            margins[chosen],
            mesh,
            view.camera,
            sh_degree,
        )


def target(view: View, background: torch.Tensor) -> torch.Tensor:
    """The frame with everything outside the performer's mask replaced by ``background``."""
    mask = view.mask.unsqueeze(-1)
    return view.image.float() / 255 * mask + background * (1 - mask)


def scored_pixels(view: View, alpha: torch.Tensor) -> torch.Tensor:
    """``(H, W)`` the pixels quality is measured on: within the compared region, where the
    performer is or the avatar draws something. Empty background on both sides would only push
    the score up."""
    return view.region * torch.maximum(view.mask, (alpha > 0.5).float())


def place_gaussians(avatar: Avatar, textures: Textures, config: TrainingConfig) -> BoundGaussians:
    """Spread the Gaussians over the mesh by part and coverage, with no colour of their own:
    drawn by a ``Scene`` that follows the textures, they start as the texture's colour."""
    device = config.device
    faces = torch.from_numpy(avatar.faces).to(device)
    parts = torch.from_numpy(avatar.face_parts).to(device)
    uvs = torch.from_numpy(avatar.uvs).to(device)
    textures_of_faces = torch.from_numpy(avatar.face_textures).to(device)

    density = torch.full(parts.shape, config.head_density, device=device)
    density[parts == BODY] = 1
    density[parts == HANDS] = config.hand_density
    low, high = config.coverage_range
    seen = ((textures.coverage(uvs, textures_of_faces) - low) / (high - low)).clamp(0, 1)
    logger.info("no Gaussians on %.0f %% of the triangles, too little seen", _share(seen == 0))
    gaussians = BoundGaussians.initialize(
        torch.from_numpy(avatar.rest_vertices).to(device),
        faces,
        density * seen,
        config.num_gaussians,
        config.sh_degree,
    ).to(device)
    with torch.no_grad():
        gaussians.sh_dc.zero_()
    return gaussians


def training_views(views: list[View], config: TrainingConfig) -> list[View]:
    """The views trained on; every ``holdout_every``-th frame is kept out to measure quality."""
    return [v for v in views if v.frame % config.holdout_every]


def train(
    views: list[View], posing: Posing, avatar: Avatar, textures: Textures, config: TrainingConfig
) -> tuple[Scene, ColourCorrection, Evaluation]:
    device = config.device
    scene = Scene(avatar, textures, place_gaussians(avatar, textures, config)).to(device)
    scene.follow_textures()
    gaussians = scene.gaussians
    colours = ColourCorrection(sorted({v.video for v in views}), reference="body").to(device)
    rest = torch.from_numpy(avatar.rest_vertices).to(device)
    rest_sizes = gaussians.triangle_sizes(rest, scene.faces)
    max_log_scales = math.log(config.max_size) - rest_sizes.log()
    max_offsets = config.max_offset / rest_sizes
    max_heights = config.max_height / rest_sizes
    colour_weights = torch.where(
        scene.head_gaussians, config.head_colour_weight, config.colour_weight
    ).unsqueeze(-1)
    log_thicknesses = math.log(config.thickness) - rest_sizes.log()

    def limit() -> None:
        gaussians.lay_flat(max_heights, log_thicknesses)
        gaussians.limit_scales(max_log_scales, config.max_elongation)
        gaussians.limit_offsets(max_offsets)

    limit()

    training = training_views(views, config)
    rates = config.learning_rates
    optimizer = torch.optim.Adam(
        [
            {"params": [gaussians.offsets], "lr": rates["offsets"]},
            {"params": [gaussians.rotations], "lr": rates["rotations"]},
            {"params": [gaussians.log_scales], "lr": rates["scales"]},
            {"params": [gaussians.opacity_logits], "lr": rates["opacities"]},
            {"params": [gaussians.sh_dc], "lr": rates["sh_dc"]},
            {"params": [gaussians.sh_rest], "lr": rates["sh_rest"]},
            {"params": list(colours.parameters()), "lr": rates["colours"]},
            {"params": posing.corrections(), "lr": rates["poses"]},
        ],
        eps=1e-15,
    )
    generator = random.Random(0)
    for step in range(config.iterations):
        if step and not step % config.rebake_every:
            textures.update(bake_textures(avatar, training, posing, config, colours))
            scene.follow_textures()
            logger.info("step %d: textures baked again", step)
        view = generator.choice(training)
        background = torch.rand(3, device=device)
        sh_degree = 0 if step < config.iterations // 3 else config.sh_degree
        vertices = posing.vertices(view.video, view.index)
        colour, alpha = scene.render(vertices, view, sh_degree, config.occlusion_margin)
        image = colours(view.video, colour, alpha) + (1 - alpha).unsqueeze(-1) * background
        terms = _loss(image.clamp(0, 1), alpha, view, background, gaussians, colour_weights, config)
        terms["poses"] = config.pose_prior_weight * posing.prior(view.video, view.index)
        optimizer.zero_grad()
        sum(terms.values()).backward()
        optimizer.step()
        limit()
        if step % 1000 == 0 or step == config.iterations - 1:
            logger.info(
                "step %d: %s", step, ", ".join(f"{k} {v.item():.4f}" for k, v in terms.items())
            )

    for video, matrix in colours.matrices.items():
        logger.info(
            "%s colours: gains %s, offsets %s",
            video,
            np.round(matrix.diagonal().tolist(), 3),
            np.round(colours.offsets[video].tolist(), 3),
        )
    scene.fold_colours()
    scene.set_gaussians(_prune(gaussians))
    return scene, colours, evaluate(scene, views, posing, colours, config)


@torch.no_grad()
def evaluate(
    scene: Scene,
    views: list[View],
    posing: Posing,
    colours: ColourCorrection,
    config: TrainingConfig,
) -> Evaluation:
    black = torch.zeros(3, device=config.device)
    scores = []
    for view in views:
        vertices = posing.vertices(view.video, view.index)
        colour, alpha = scene.render(vertices, view, config.sh_degree, config.occlusion_margin)
        image = colours(view.video, colour, alpha).clamp(0, 1)
        scores.append(psnr(image, target(view, black), scored_pixels(view, alpha)))
    body = next(v for v in views if v.video == "body")
    vertices = posing.vertices("body", body.index)
    spread = view_spread(scene, vertices, body.camera, config.occlusion_margin)
    head_spread = view_spread(
        scene, vertices, body.camera, config.occlusion_margin, parts=(HEAD_SKIN,)
    )
    for part, values in (("body", spread), ("head", head_spread)):
        logger.info(
            "%s colour change between viewing directions: median %.3f, p90 %.3f",
            part, values.median(), values.quantile(0.9),
        )  # fmt: skip
    evaluation = Evaluation(
        videos=np.array([v.video for v in views]),
        frames=np.array([v.frame for v in views]),
        held_out=np.array([not v.frame % config.holdout_every for v in views]),
        psnr=np.array(scores),
        view_spread=spread.cpu().numpy(),
        head_view_spread=head_spread.cpu().numpy(),
    )
    by_video = defaultdict(lambda: defaultdict(list))
    for video, held_out, score in zip(
        evaluation.videos, evaluation.held_out, evaluation.psnr, strict=True
    ):
        by_video[video]["held-out" if held_out else "training"].append(score)
    for video, splits in by_video.items():
        logger.info(
            "%s: PSNR %s",
            video,
            ", ".join(f"{split} {np.mean(s):.2f} dB ({len(s)})" for split, s in splits.items()),
        )
    return evaluation


def _loss(
    image: torch.Tensor,
    alpha: torch.Tensor,
    view: View,
    background: torch.Tensor,
    gaussians: BoundGaussians,
    colour_weights: torch.Tensor,
    config: TrainingConfig,
) -> dict[str, torch.Tensor]:
    """The loss terms of one step.

    Args:
        colour_weights: ``(G, 1)`` how strongly each Gaussian's colour is held to the texture's.
    """
    goal = target(view, background)
    compared = view.region * (1 - mask_edge(view.mask, config.edge_band))

    def within(values: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
        return (values * region).sum() / region.sum().clamp_min(1)

    mask = F.binary_cross_entropy(alpha.clamp(1e-5, 1 - 1e-5), view.mask, reduction="none")
    return {
        "l1": (1 - config.ssim_weight) * within((image - goal).abs().mean(-1), compared),
        "ssim": config.ssim_weight * within(1 - ssim(image, goal), compared),
        "mask": config.mask_weight * within(mask, view.region),
        "offsets": config.offset_weight * gaussians.offsets.square().sum(-1).mean(),
        "colour": (colour_weights * (SH_C0 * gaussians.sh_dc[:, 0]).abs()).mean(),
        "opacity": config.opacity_weight * gaussians.opacities().mean(),
    }


def mask_edge(mask: torch.Tensor, band: int) -> torch.Tensor:
    """``(H, W)`` 1 within ``band`` pixels of the ``(H, W)`` mask's edge, on either side."""
    size = 2 * band + 1
    grown = F.max_pool2d(mask[None, None], size, 1, band)[0, 0]
    shrunk = -F.max_pool2d(-mask[None, None], size, 1, band)[0, 0]
    return ((grown - shrunk) > 0.5).float()


def _share(mask: torch.Tensor) -> float:
    return 100 * mask.float().mean().item()


def _prune(gaussians: BoundGaussians) -> BoundGaussians:
    record = gaussians.record()
    kept = 1 / (1 + np.exp(-record.opacity_logits)) >= MIN_OPACITY
    logger.info("kept %d of %d Gaussians", kept.sum(), len(kept))
    pruned = type(record)(**{name: value[kept] for name, value in vars(record).items()})
    return BoundGaussians(pruned).to(gaussians.faces.device)
