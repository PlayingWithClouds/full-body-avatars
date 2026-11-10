"""Lining the frames up with the textured mesh, so the textures baked from them come out sharp.

The fits place the body and head by silhouettes, keypoints and landmarks, to within ten or twenty
pixels. Frames placed that loosely disagree about where each detail of the skin is, and a texture
baked from them is blurred by as much. So the frames are aligned by their pixels instead: bake the
textures, move each frame's pose and expression, within the corrections ``posing`` allows, until
the textured mesh drawn in that pose matches the frame, and bake again from the moved frames.
Each round's texture is sharper than the last, and so a better target for the next.

The image alone can be matched by sliding the mesh until the parts it matches badly fall
outside the performer, so the mesh's outline is held to her mask too, drawn differentiably as in
fitting. Aligning works on frames of half the size: a pose needs no more, and the bakes use the
full frames.

Each head video's colours are matched to the body video's along the way (see ``colour``).
"""

from __future__ import annotations

import dataclasses
import logging
import random

import torch
import torch.nn.functional as F

from avatar_splats.body_models.hybrid import HEAD_PARTS, Avatar
from avatar_splats.config import TrainingConfig
from avatar_splats.fitting.losses import silhouette_loss
from avatar_splats.fitting.silhouette import SurfaceSamples, render_silhouette
from avatar_splats.rendering import MeshImage, render_mesh
from avatar_splats.training.colour import ColourCorrection
from avatar_splats.training.losses import ssim
from avatar_splats.training.posing import Posing
from avatar_splats.training.textures import Textures, bake_textures
from avatar_splats.training.train import mask_edge, target
from avatar_splats.training.views import View

logger = logging.getLogger(__name__)


class MeshDrawing(torch.nn.Module):
    """Draws the avatar's textured mesh posed for a view, and its silhouette differentiably."""

    def __init__(self, avatar: Avatar, textures: Textures, samples: int, inset: float):
        """Args: the number of points the silhouettes are splatted from, and how far they lie
        inside the surface, in radii of their blobs (see ``fitting.silhouette``)."""
        super().__init__()
        self.textures = textures
        self.register_buffer("faces", torch.from_numpy(avatar.faces).long())
        self.register_buffer("uvs", torch.from_numpy(avatar.uvs).float())
        self.register_buffer("textures_of_faces", torch.from_numpy(avatar.face_textures).long())
        on_head = torch.isin(torch.from_numpy(avatar.face_parts), torch.tensor(HEAD_PARTS))
        self.register_buffer("head_faces", torch.nonzero(on_head).squeeze(-1))
        rest = torch.from_numpy(avatar.rest_vertices).float()
        self.spacing = {}
        for part, faces in (("all", self.faces), ("head", self.faces[self.head_faces])):
            drawn = SurfaceSamples.draw(rest, faces, samples)
            self.register_buffer(f"{part}_sample_faces", drawn.faces)
            self.register_buffer(f"{part}_sample_barycentric", drawn.barycentric)
            self.spacing[part] = drawn.spacing
        self.inset = inset

    def forward(self, vertices: torch.Tensor, view: View) -> MeshImage:
        """The mesh posed as ``(V, 3)`` vertices, over black. Head videos see the head alone."""
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

    def silhouette(self, vertices: torch.Tensor, view: View) -> torch.Tensor:
        """``(H, W)`` the mesh's silhouette in [0, 1], differentiable in ``(V, 3)`` vertices."""
        part = "head" if view.head_only else "all"
        faces = self.faces[self.head_faces] if view.head_only else self.faces
        samples = SurfaceSamples(
            getattr(self, f"{part}_sample_faces"),
            getattr(self, f"{part}_sample_barycentric"),
            self.spacing[part],
        )
        points = samples.points(vertices, faces, self.inset * samples.radius)
        return render_silhouette(points[None], samples.radius, view.camera)[0]


def align(
    views: list[View],
    posing: Posing,
    avatar: Avatar,
    textures: Textures,
    colours: ColourCorrection,
    config: TrainingConfig,
    samples: int,
    inset: float,
) -> None:
    """Align the views' frames and bake the textures from them, in place, over
    ``config.align_rounds`` rounds of ``config.align_steps`` steps. ``samples`` and ``inset``
    draw the silhouettes, as in fitting."""
    drawing = MeshDrawing(avatar, textures, samples, inset).to(textures.body.device)
    small = [smaller(view, config.align_resolution) for view in views]
    for round_ in range(config.align_rounds):
        textures.update(bake_textures(avatar, views, posing, config, colours))
        loss = refine(small, posing, drawing, colours, config, learn_colours=True)
        logger.info("alignment round %d: image loss %.4f", round_, loss)
    textures.update(bake_textures(avatar, views, posing, config, colours))


def refine(
    views: list[View],
    posing: Posing,
    drawing: MeshDrawing,
    colours: ColourCorrection,
    config: TrainingConfig,
    learn_colours: bool,
) -> float:
    """Move the views' poses and expressions, and with ``learn_colours`` the colour
    correction, ``config.align_steps`` steps towards matching the drawn mesh to their frames.
    Returns the mean image loss over the last tenth of the steps."""
    rates = config.learning_rates
    groups = [{"params": posing.corrections(), "lr": rates["poses"]}]
    if learn_colours:
        groups.append({"params": list(colours.parameters()), "lr": rates["colours"]})
    optimizer = torch.optim.Adam(groups, eps=1e-15)
    generator = random.Random(0)
    recent = []
    for step in range(config.align_steps):
        view = generator.choice(views)
        vertices = posing.vertices(view.video, view.index)
        mesh = drawing(vertices, view)
        image = colours(view.video, mesh.image, mesh.mask).clamp(0, 1)
        loss = image_loss(image, mesh.mask, view, config)
        outline = silhouette_loss(
            drawing.silhouette(vertices, view)[None], view.mask[None], view.region[None]
        )
        loss = loss + config.align_silhouette_weight * outline
        prior = config.pose_prior_weight * posing.prior(view.video, view.index)
        optimizer.zero_grad()
        (loss + prior).backward()
        optimizer.step()
        if step >= config.align_steps * 0.9:
            recent.append(loss.item())
    return sum(recent) / max(len(recent), 1)


def smaller(view: View, resolution: int) -> View:
    """The view with its frame shrunk to ``resolution`` pixels on its long side."""
    camera = view.camera
    factor = resolution / max(camera.width, camera.height)
    if factor >= 1:
        return view
    small = camera.scaled(factor)
    size = (small.height, small.width)

    def shrink(image: torch.Tensor) -> torch.Tensor:
        return F.interpolate(image.permute(2, 0, 1)[None], size, mode="area")[0].permute(1, 2, 0)

    return dataclasses.replace(
        view,
        image=shrink(view.image.float()).round().to(view.image.dtype),
        mask=shrink(view.mask[..., None])[..., 0],
        region=shrink(view.region[..., None])[..., 0],
        camera=small,
    )


def image_loss(
    image: torch.Tensor, mask: torch.Tensor, view: View, config: TrainingConfig
) -> torch.Tensor:
    """L1 and SSIM between an ``(H, W, 3)`` drawn mesh and its frame, where both show the
    performer, away from the edge of her mask."""
    goal = target(view, torch.zeros(3, device=image.device))
    compared = view.region * view.mask * mask * (1 - mask_edge(view.mask, config.edge_band))
    total = compared.sum().clamp_min(1)
    l1 = ((image - goal).abs().mean(-1) * compared).sum() / total
    structure = ((1 - ssim(image, goal)) * compared).sum() / total
    return (1 - config.ssim_weight) * l1 + config.ssim_weight * structure
