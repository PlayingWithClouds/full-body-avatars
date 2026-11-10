"""The breasts' shape fitted to close-ups (``closeups``), on an avatar notebook 12 built.

**The region.** Anny marks no breast vertices, but its breast controls move only them: the
region is where raising ``cupsize`` moves the surface (``breast_region``), and each nipple is
where ``nipple-size-incr`` moves it most on its side.

**The shape** is the avatar's, plus, inside the region only:

- Anny's own breast controls refitted: the phenotype's ``cupsize`` and ``firmness`` and the local
  changes ``BREAST_CHANGES``, as the difference they make to Anny's rest surface;
- smooth offsets (free values averaged over the mesh, as notebook 09's shape changes), faded out
  by the region's weight so the region's edge does not move.

Size is the avatar's: the body fit placed the nipples in metres, and a close-up cannot tell a
small breast near the camera from a large one far off; its distance is free instead.

**Per close-up**: the body's place in the camera (a rotation about the breasts' middle and a
shift) and small turns of the spine, clavicles and upper arms (``POSED_BONES``) from her rest pose.
The camera is the source's field of view on the crop.

**Losses**, per close-up: her torso's outline (Sapiens2's torso and clothes, ignoring arms, hair,
face and legs, which are not drawn), the normals (Sapiens2's, over bare skin, weighed up in the
region), the nipples (each found one paired with the nearer of the model's) and the shoulders and
elbows (ViTPose's against the bones' joints). The breasts' masks are not used: their edge on the
chest is a guess of NSFW-API's, the outline and normals say more.

**Upright only.** Breasts sag with gravity: after the close-ups are placed, only those with her
torso upright (``upright``) shape the breasts; the others follow the shape without changing it,
and their gravity, in the chest's axes, is kept for the posture correctives to come.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import roma
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from avatar_splats.body_models.mesh import vertex_normals
from avatar_splats.records import ArrayRecord

BREAST_CHANGES = ("breast-point-incr", "nipple-size-incr", "breast-dist-incr",
                  "breast-volume-vert-up")  # fmt: skip
BREAST_PHENOTYPES = ("cupsize", "firmness")
POSED_BONES = ("spine03", "spine02", "spine01", "clavicle.L", "clavicle.R", "upperarm01.L",
               "upperarm01.R")  # fmt: skip
"""The bones a close-up turns from her rest pose."""
CHEST_BONE = "spine01"
"""The bone whose axes gravity is measured in: the chest's."""
ARM_BONES = ("shoulder", "upperarm", "lowerarm", "wrist", "finger", "metacarpal", "thumb", "hand",
             "upperleg", "lowerleg", "foot", "toe")  # fmt: skip
"""Parts of bone names that are not the torso: their triangles are not drawn."""
TORSO_TARGET = ("Torso", "Upper_Clothing", "Lower_Clothing", "Apparel")
"""Sapiens2 labels inside her torso's outline."""
IGNORED = ("Face_Neck", "Hair", "Eyeglass", "Left_Hand", "Right_Hand", "Left_Lower_Arm",
           "Right_Lower_Arm", "Left_Upper_Arm", "Right_Upper_Arm", "Left_Upper_Leg",
           "Right_Upper_Leg", "Left_Lower_Leg", "Right_Lower_Leg", "Lower_Lip", "Upper_Lip",
           "Lower_Teeth", "Upper_Teeth", "Tongue")  # fmt: skip
"""Labels of what the model does not draw, or draws elsewhere: compared with nothing."""
COCO_JOINTS = {5: "upperarm01.L", 6: "upperarm01.R", 7: "lowerarm01.L", 8: "lowerarm01.R"}
"""COCO shoulders and elbows and the bone whose head is that joint (COCO's left is hers)."""
FRONT = torch.tensor([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
"""World to camera axes for a camera facing her front: x right, y down, z forward."""
MILLIMETRE = 1e-3
BREAST_WIDTH = 0.64
"""A breast's mask's width over the nipples' spread (0.61 to 0.67 in water.mp4's close-ups):
the scale of a close-up showing one nipple."""


@dataclass(frozen=True)
class BreastRegion:
    kept: torch.Tensor
    """``(B,)`` for each of the avatar's body vertices (those before ``head_start``) the index of
    Anny's vertex it is."""
    weights: torch.Tensor
    """``(B,)`` in [0, 1]: how much each body vertex belongs to the breasts."""
    nipples: torch.Tensor
    """``(2,)`` the body vertices of her right (-x) and left (+x) nipple."""


def averaging_matrix(faces: torch.Tensor, count: int) -> torch.Tensor:
    """``(count, count)`` sparse: each vertex's mean with its neighbours over ``(F, 3)`` faces."""
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    loops = torch.arange(count, device=faces.device)[:, None].repeat(1, 2)
    edges = torch.unique(torch.cat([edges, edges.flip(1), loops]), dim=0)
    degree = torch.bincount(edges[:, 0], minlength=count).float()
    return torch.sparse_coo_tensor(
        edges.T, 1 / degree[edges[:, 0]], (count, count), check_invariants=True
    ).coalesce()


def smooth(values: torch.Tensor, matrix: torch.Tensor, steps: int) -> torch.Tensor:
    for _ in range(steps):
        values = torch.sparse.mm(matrix, values)
    return values


def breast_region(
    body, phenotype: np.ndarray, local_changes: np.ndarray, fitted_body: torch.Tensor,
    faces: torch.Tensor, reach: float = 0.15, smoothing: int = 3,
) -> BreastRegion:  # fmt: skip
    """The breasts on an avatar's body.

    Args:
        body: an ``AnnyBody``.
        phenotype, local_changes: the body fit's.
        fitted_body: ``(B, 3)`` the avatar's body vertices as the fit built them (notebook 12's
            ``avatar.npz``, before refinement moved them): each is one of Anny's.
        faces: ``(F, 3)`` the avatar's triangles among its body vertices.
        reach: the share of the largest movement raising ``cupsize`` makes at which a vertex
            belongs wholly to the breasts.
    """
    from avatar_splats.body_models.hybrid import rest_body

    base = rest_body(body, phenotype, local_changes)["vertices"]
    kept = torch.cat([torch.cdist(part, base).argmin(dim=1) for part in fitted_body.split(2048)])

    def moved(name: str, step: float) -> torch.Tensor:
        p, c = phenotype.copy(), local_changes.copy()
        if name in body.phenotype_labels:
            p[body.phenotype_labels.index(name)] += step
        else:
            c[body.local_change_labels.index(name)] += step
        return (rest_body(body, p, c)["vertices"] - base)[kept].norm(dim=1)

    movement = moved("cupsize", 0.2)
    weights = (movement / (reach * movement.max())).clamp(0, 1)
    matrix = averaging_matrix(faces, len(kept))
    weights = smooth(weights[:, None], matrix, smoothing)[:, 0].clamp(0, 1)
    nipple = moved("nipple-size-incr", 0.5)
    x = fitted_body[:, 0]
    nipples = torch.stack(
        [torch.where(x * side > 0, nipple, torch.zeros_like(nipple)).argmax() for side in (-1, 1)]
    )
    return BreastRegion(kept=kept, weights=weights, nipples=nipples)


def torso_faces(faces: np.ndarray, bone_indices: np.ndarray, bone_weights: np.ndarray,
                bone_labels: list[str], head_start: int) -> np.ndarray:  # fmt: skip
    """``(T,)`` the body triangles whose vertices all follow mostly the torso's bones."""
    main = bone_indices[np.arange(len(bone_indices)), bone_weights.argmax(axis=1)]
    limb = np.array([any(part in label for part in ARM_BONES) for label in bone_labels])
    body = (faces < head_start).all(axis=1)
    torso_vertex = ~limb[main]
    return np.flatnonzero(body & torso_vertex[np.minimum(faces, len(main) - 1)].all(axis=1))


@dataclass(frozen=True)
class FitSettings:
    render_size: int
    """Pixels along a close-up's side as rendered and compared."""
    pose_steps: int
    """Steps placing the close-ups with the shape held."""
    shape_steps: int
    """Steps fitting shape and places together."""
    learning_rate: float
    smoothing_steps: int
    """Rounds of averaging the offsets' free values: their smoothness."""
    max_tilt: float
    """Degrees her torso may lean from upright for a close-up to shape the breasts."""
    min_identity: float
    """Likeness to the actor a close-up needs to shape them; unknown counts as too little."""
    min_iou: float
    """The torso outline's IoU a placed close-up needs to shape them: placing failed below."""
    weights: dict[str, float]
    """Per loss its weight: outline, normals, nipples, joints, pose, offsets, controls."""


class BreastModel(torch.nn.Module):
    """The breasts' shape (see the module's docstring) and the close-ups' places."""

    def __init__(
        self, body, region: BreastRegion, rest: torch.Tensor, phenotype: np.ndarray,
        local_changes: np.ndarray, faces: torch.Tensor, count: int, smoothing_steps: int,
    ):  # fmt: skip
        super().__init__()
        device = rest.device
        self.body, self.region, self.smoothing_steps = body, region, smoothing_steps
        self.register_buffer("rest", rest)
        self.register_buffer("phenotype0", torch.from_numpy(phenotype).float().to(device))
        self.register_buffer("changes0", torch.from_numpy(local_changes).float().to(device))
        self.phenotype_index = [body.phenotype_labels.index(n) for n in BREAST_PHENOTYPES]
        self.change_index = [body.local_change_labels.index(n) for n in BREAST_CHANGES]
        controls = len(BREAST_PHENOTYPES) + len(BREAST_CHANGES)
        self.controls = torch.nn.Parameter(torch.zeros(controls, device=device))
        self.offset_values = torch.nn.Parameter(torch.zeros(len(rest), 3, device=device))
        self.matrix = averaging_matrix(faces, len(rest))
        with torch.no_grad():
            self.base = self._anny(self.controls)
        self.rotations = torch.nn.Parameter(torch.zeros(count, 3, device=device))
        self.translations = torch.nn.Parameter(torch.zeros(count, 3, device=device))
        self.turns = torch.nn.Parameter(torch.zeros(count, len(POSED_BONES), 3, device=device))

    def control_values(self, controls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The phenotype and local changes with the breast controls moved by ``controls``
        (logits for the phenotype's [0, 1], inverse tanh for the changes' [-1, 1])."""
        p, c = self.phenotype0.clone(), self.changes0.clone()
        k = len(BREAST_PHENOTYPES)
        start = p[self.phenotype_index].clamp(1e-3, 1 - 1e-3)
        p[self.phenotype_index] = torch.sigmoid(torch.logit(start) + controls[:k])
        start = c[self.change_index].clamp(-0.999, 0.999)
        c[self.change_index] = torch.tanh(torch.atanh(start) + controls[k:])
        return p, c

    def _anny(self, controls: torch.Tensor) -> torch.Tensor:
        p, c = self.control_values(controls)
        zeros = p.new_zeros(1, self.body.num_bones, 3)
        out = self.body(p[None], c[None], zeros, p.new_zeros(1, 3))
        return out["rest_vertices"][0][self.region.kept]

    def offsets(self) -> torch.Tensor:
        """``(B, 3)`` metres: the smooth offsets, faded by the region."""
        w = self.region.weights[:, None]
        return smooth(self.offset_values * w, self.matrix, self.smoothing_steps) * w * MILLIMETRE

    def change(self) -> torch.Tensor:
        """``(B, 3)`` metres: how far each body vertex moves from the avatar's."""
        w = self.region.weights[:, None]
        return w * (self._anny(self.controls) - self.base) + self.offsets()

    def shaped(self) -> torch.Tensor:
        return self.rest + self.change()


@dataclass(frozen=True)
class Posing:
    """Poses the avatar's body vertices for close-ups."""

    body: torch.nn.Module
    """An ``AnnyBody``: its skinning."""
    bind_poses: torch.Tensor
    bone_indices: torch.Tensor
    bone_weights: torch.Tensor
    """The avatar's, for its body vertices."""
    nipples: torch.Tensor
    """``(2,)`` the nipples' body vertices: the body turns about their middle."""

    def __call__(self, rest: torch.Tensor, rotations: torch.Tensor, translations: torch.Tensor,
                 turns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
        """``(N, B, 3)`` the body vertices of ``(B, 3)`` ``rest`` posed for N close-ups, in
        their cameras' axes, and ``(N, J, 4, 4)`` the bones' transforms (rest to camera).

        Args:
            rotations: ``(N, 3)`` the body's turn in each camera, after ``FRONT``.
            translations: ``(N, 3)`` metres: where the nipples' middle is.
            turns: ``(N, len(POSED_BONES), 3)`` the posed bones' rotation vectors.
        """
        from avatar_splats.body_models.skinning import linear_blend_skinning

        count = len(rotations)
        bones = [self.body.bone(n) for n in POSED_BONES]
        pose = rest.new_zeros(count, self.body.num_bones, 3)
        pose[:, bones] = turns
        skinning = self.body.skinning(self.bind_poses, pose, rest.new_zeros(count, 3))
        posed = linear_blend_skinning(
            rest.expand(count, -1, -1), self.bone_indices, self.bone_weights, skinning
        )
        centre = rest[self.nipples].mean(dim=0)
        turn = roma.rotvec_to_rotmat(rotations) @ FRONT.to(rest)
        out = (posed - centre) @ turn.transpose(1, 2) + translations[:, None]
        whole = torch.eye(4, device=rest.device).repeat(count, 1, 1)
        whole[:, :3, :3] = turn
        whole[:, :3, 3] = translations - (turn @ centre[:, None])[..., 0]
        return out, whole[:, None] @ skinning


def clip(points: torch.Tensor, K: torch.Tensor, size: int, near: float = 0.05,
         far: float = 20.0) -> torch.Tensor:  # fmt: skip
    """``(N, V, 4)`` nvdiffrast clip coordinates of ``(N, V, 3)`` camera-axes points for
    ``(N, 3, 3)`` intrinsics of a ``size`` square image."""
    x, y, z = points.unbind(-1)
    fx, fy = K[:, 0, 0, None], K[:, 1, 1, None]
    cx, cy = K[:, 0, 2, None], K[:, 1, 2, None]
    return torch.stack(
        (2 * fx / size * x + (2 * cx / size - 1) * z, 2 * fy / size * y + (2 * cy / size - 1) * z,
         (far + near) / (far - near) * z - 2 * far * near / (far - near), z),
        dim=-1,
    ).contiguous()  # fmt: skip


def project(points: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """``(N, P, 2)`` pixels of ``(N, P, 3)`` camera-axes points."""
    uv = points @ K.transpose(1, 2)
    return uv[..., :2] / uv[..., 2:].clamp_min(1e-6)


@dataclass
class Targets:
    """What the close-ups show, at the render size, on the GPU."""

    K: torch.Tensor  # (N, 3, 3) at the render size
    torso: torch.Tensor  # (N, S, S) 1 inside her torso's outline
    valid: torch.Tensor  # (N, S, S) 1 where compared
    skin: torch.Tensor  # (N, S, S) 1 on bare torso skin (normals compared)
    normals: torch.Tensor  # (N, S, S, 3)
    nipples: torch.Tensor  # (N, 2, 2) render pixels, NaN missing
    joints: torch.Tensor  # (N, 4, 3) COCO shoulders and elbows, render pixels and confidence
    breast_width: torch.Tensor  # (N,) pixels: the width of the breast mask under a lone nipple
    torso_x: torch.Tensor  # (N,) pixels: her torso's middle across the image


def targets(closeups, size: int, device: str) -> Targets:
    """The close-ups' targets at ``size`` pixels."""
    from avatar_splats.preprocess.prepare import PART_NAMES

    factor = size / closeups.images.shape[1]
    labels = torch.from_numpy(closeups.labels).to(device)

    def resized(values: torch.Tensor) -> torch.Tensor:
        return F.interpolate(values[:, None].float(), (size, size), mode="area")[:, 0]

    def labelled(names) -> torch.Tensor:
        return torch.isin(labels, torch.tensor([PART_NAMES.index(n) for n in names], device=device))

    torso = resized(labelled(TORSO_TARGET))
    valid = 1 - resized(labelled(IGNORED))
    skin = (resized(labelled(("Torso",))) > 0.99).float()
    normals = torch.from_numpy(closeups.normals).to(device).float().permute(0, 3, 1, 2)
    normals = F.normalize(F.interpolate(normals, (size, size), mode="area"), dim=1)
    K = torch.from_numpy(closeups.K).to(device).float().clone()
    K[:, :2] *= factor
    joints = torch.from_numpy(closeups.keypoints[:, list(COCO_JOINTS)]).to(device).float().clone()
    joints[..., :2] *= factor
    widths = np.full(len(closeups.keys), np.nan)
    for i, (mask, nipples) in enumerate(zip(closeups.breasts, closeups.nipples, strict=True)):
        found = nipples[np.isfinite(nipples[:, 0])]
        if len(found) != 1:
            continue
        count, parts = cv2.connectedComponents(mask.astype(np.uint8))
        x, y = np.clip(np.round(found[0]).astype(int), 0, mask.shape[0] - 1)
        if parts[y, x]:
            columns = np.flatnonzero((parts == parts[y, x]).any(axis=0))
            width = columns[-1] - columns[0] + 1
            if width >= 0.1 * mask.shape[1]:  # a speck is no breast
                widths[i] = width * factor
    columns = torso.sum(1)
    torso_x = (columns * torch.arange(size, device=device)).sum(1) / columns.sum(1).clamp_min(1)
    return Targets(
        K=K, torso=torso, valid=(valid > 0.5).float(), skin=skin,
        normals=normals.permute(0, 2, 3, 1),
        nipples=torch.from_numpy(closeups.nipples).to(device).float() * factor, joints=joints,
        breast_width=torch.from_numpy(widths).float().to(device), torso_x=torso_x,
    )  # fmt: skip


def nipple_error(projected: torch.Tensor, found: torch.Tensor) -> torch.Tensor:
    """``(N,)`` mean pixels from each found nipple (``(N, 2, 2)``, NaN missing) to the nearer of
    the model's two ``(N, 2, 2)``, the pair assigned as a whole when both are found."""
    known = torch.isfinite(found[..., 0])
    found = torch.nan_to_num(found)
    straight = (projected - found).norm(dim=-1)
    crossed = (projected.flip(1) - found).norm(dim=-1)
    both = known.all(dim=1)
    paired = torch.minimum(straight.sum(1), crossed.sum(1)) / 2
    single = (torch.minimum(straight, crossed) * known).sum(1) / known.sum(1).clamp_min(1)
    return torch.where(both, paired, single)


class Renderer:
    """Draws the torso of posed body vertices: coverage (antialiased, so its edge has gradients),
    normals in the camera's axes and the breast region's weight."""

    def __init__(self, faces: torch.Tensor, all_faces: torch.Tensor, weights: torch.Tensor):
        import nvdiffrast.torch as dr

        self.dr = dr
        self.context = dr.RasterizeCudaContext()
        self.faces = faces.int().contiguous()
        self.all_faces = all_faces
        self.weights = weights

    def __call__(self, vertices: torch.Tensor, K: torch.Tensor, size: int) -> dict:
        dr = self.dr
        positions = clip(vertices, K, size)
        rast, _ = dr.rasterize(self.context, positions, self.faces, (size, size))
        covered = (rast[..., 3:] > 0).float()
        coverage = dr.antialias(covered, rast, positions, self.faces)[..., 0]
        normals = vertex_normals(vertices, self.all_faces)
        values = torch.cat((normals, self.weights[None, :, None].expand(len(vertices), -1, 1)), -1)
        values, _ = dr.interpolate(values.contiguous(), rast, self.faces)
        return {
            "coverage": coverage,
            "covered": covered[..., 0],
            "normals": F.normalize(values[..., :3], dim=-1),
            "region": values[..., 3] * covered[..., 0],
        }


def place_closeups(model: BreastModel, rest: torch.Tensor, target: Targets) -> None:
    """Start each close-up's shift from its nipples: their middle where both are found, at the
    distance their spread implies. With one, the spread is its breast's width over
    ``BREAST_WIDTH``, the middle that far towards her torso's middle; with none, the crop's
    centre, at a distance at which the breasts fill half the crop."""
    K, nipples = target.K, target.nipples
    spread = float((rest[model.region.nipples[0]] - rest[model.region.nipples[1]]).norm())
    with torch.no_grad():
        for i in range(len(K)):
            focal, cx, cy = float(K[i, 0, 0]), float(K[i, 0, 2]), float(K[i, 1, 2])
            found = nipples[i][torch.isfinite(nipples[i][:, 0])]
            width = float(target.breast_width[i])
            if len(found) == 2:
                pixels = float((found[0] - found[1]).norm())
                middle = found.mean(dim=0)
            elif len(found) == 1 and math.isfinite(width):
                pixels = width / BREAST_WIDTH
                side = 1.0 if float(target.torso_x[i]) >= float(found[0, 0]) else -1.0
                middle = found[0] + found.new_tensor([side * pixels / 2, 0.0])
            else:
                half = target.torso.shape[-1] / 2  # the breasts, twice the spread, half the crop
                pixels, middle = half, K.new_tensor([half, half])
            depth = focal * spread / max(pixels, 1.0)
            model.translations[i] = torch.stack(
                [
                    (middle[0] - cx) * depth / focal,
                    (middle[1] - cy) * depth / focal,
                    middle.new_tensor(depth),
                ]  # fmt: skip
            )


def tilt_and_gravity(
    posing: Posing, rest: torch.Tensor, model: BreastModel
) -> tuple[np.ndarray, np.ndarray]:
    """Per close-up the degrees her chest leans from upright and ``(N, 3)`` gravity in the
    chest bone's axes (rest-pose world axes turned with it), taking the camera as upright:
    gravity points down the image."""
    with torch.no_grad():
        _, transforms = posing(rest, model.rotations, model.translations, model.turns)
        chest = transforms[:, posing.body.bone(CHEST_BONE), :3, :3]
        down = rest.new_tensor([0.0, 1.0, 0.0])
        gravity = F.normalize(chest.transpose(1, 2) @ down, dim=-1)
        tilt = torch.rad2deg(torch.acos((-gravity[:, 2]).clamp(-1, 1)))
    return tilt.cpu().numpy(), gravity.cpu().numpy()


@dataclass(frozen=True)
class BreastFit(ArrayRecord):
    """The fitted breasts and the close-ups' places."""

    change: np.ndarray
    """``(B, 3)`` metres added to the avatar's body vertices (those before ``head_start``)."""
    weights: np.ndarray
    """``(B,)`` the breast region (``BreastRegion.weights``)."""
    nipples: np.ndarray
    """``(2,)`` the nipples' body vertices: hers right, left."""
    controls: np.ndarray
    """``(C,)`` the fitted values of ``BREAST_PHENOTYPES`` then ``BREAST_CHANGES``."""
    controls_before: np.ndarray
    keys: np.ndarray
    """``(N,)`` the close-ups' keys."""
    rotations: np.ndarray
    """``(N, 3)`` rotation vectors: the body's turn in each camera, after ``FRONT``."""
    translations: np.ndarray
    """``(N, 3)`` metres: where the nipples' middle is in each camera."""
    turns: np.ndarray
    """``(N, len(POSED_BONES), 3)`` the posed bones' rotation vectors."""
    tilt: np.ndarray
    """``(N,)`` degrees her chest leans from upright."""
    gravity: np.ndarray
    """``(N, 3)`` gravity's direction in the chest bone's axes."""
    shaping: np.ndarray
    """``(N,)`` bool: the close-ups that shaped the breasts."""
    iou_before: np.ndarray
    iou_after: np.ndarray
    """``(N,)`` the torso outline's IoU where compared, with the avatar's shape and the fitted
    one, both with the close-ups placed."""
    normal_before: np.ndarray
    normal_after: np.ndarray
    """``(N,)`` mean degrees between rendered and Sapiens2's normals over the breasts."""
    nipple_before: np.ndarray
    nipple_after: np.ndarray
    """``(N,)`` mean pixels (at the render size) from the found nipples to the model's."""


def fit_breasts(
    body, avatar_arrays: dict, fitted_body: np.ndarray, phenotype: np.ndarray,
    local_changes: np.ndarray, bone_labels: list[str], closeups, settings: FitSettings,
    device: str, progress: bool = True,
) -> BreastFit:  # fmt: skip
    """Fit the breasts' shape to ``closeups``.

    Args:
        body: an ``AnnyBody`` on ``device``.
        avatar_arrays: the ``.avatar`` file's arrays (``rest_vertices``, ``faces``,
            ``bone_indices``, ``bone_weights``, ``bind_poses``) and ``head_start``.
        fitted_body: ``(B, 3)`` the body vertices of notebook 12's ``avatar.npz``.
        phenotype, local_changes: the body fit's.
        bone_labels: the avatar's bones.
    """
    s = settings
    head_start = int(avatar_arrays["head_start"])
    faces_np = avatar_arrays["faces"].astype(np.int64)
    body_faces_np = faces_np[(faces_np < head_start).all(axis=1)]
    rest = torch.from_numpy(avatar_arrays["rest_vertices"][:head_start]).float().to(device)
    body_faces = torch.from_numpy(body_faces_np).to(device)
    region = breast_region(
        body, phenotype, local_changes, torch.from_numpy(fitted_body).float().to(device),
        body_faces,
    )  # fmt: skip
    bone_indices = avatar_arrays["bone_indices"][:head_start].astype(np.int64)
    bone_indices = torch.from_numpy(bone_indices).to(device)
    bone_weights = torch.from_numpy(avatar_arrays["bone_weights"][:head_start]).float().to(device)
    bind_poses = torch.from_numpy(avatar_arrays["bind_poses"]).float().to(device)
    drawn = torso_faces(
        body_faces_np, avatar_arrays["bone_indices"][:head_start],
        avatar_arrays["bone_weights"][:head_start], bone_labels, head_start,
    )  # fmt: skip
    renderer = Renderer(body_faces[drawn], body_faces, region.weights)
    target = targets(closeups, s.render_size, device)
    count = len(closeups.keys)
    model = BreastModel(
        body, region, rest, phenotype, local_changes, body_faces, count, s.smoothing_steps
    )
    place_closeups(model, rest, target)
    joint_bones = [body.bone(n) for n in COCO_JOINTS.values()]
    joint_heads = bind_poses[joint_bones, :3, 3]
    posing = Posing(body, bind_poses, bone_indices, bone_weights, region.nipples)

    def posed(shape: torch.Tensor):
        return posing(shape, model.rotations, model.translations, model.turns)

    def losses(shape_on: torch.Tensor | None, scores: bool = False) -> dict:
        shaped = model.shaped()
        vertices, transforms = posed(shaped.detach())
        if shape_on is not None:  # only these change the shape; the others follow it
            vertices = torch.where(shape_on[:, None, None], posed(shaped)[0], vertices)
        image = renderer(vertices, target.K, s.render_size)
        outline = ((image["coverage"] - target.torso).square() * target.valid).mean((1, 2))
        region_weight = 0.2 + image["region"]
        compared = target.skin * image["covered"] * region_weight
        cosine = (image["normals"] * target.normals).sum(-1)
        normal = ((1 - cosine) * compared).sum((1, 2)) / compared.sum((1, 2)).clamp_min(1)
        nipples = project(vertices[:, region.nipples], target.K)
        nipple = nipple_error(nipples, target.nipples) / s.render_size
        heads = transforms[:, joint_bones, :3, :3] @ joint_heads[None, :, :, None]
        heads = heads[..., 0] + transforms[:, joint_bones, :3, 3]
        sure = (target.joints[..., 2] >= 0.5).float()
        joint = ((project(heads, target.K) - target.joints[..., :2]).norm(dim=-1) * sure).sum(1)
        joint = joint / sure.sum(1).clamp_min(1) / s.render_size
        out = {
            "outline": outline, "normals": normal, "nipples": nipple, "joints": joint,
            "pose": model.turns.square().sum((1, 2)),
        }  # fmt: skip
        if scores:
            inside = image["region"] > 0.5
            hard = (image["covered"] > 0.5) & (target.valid > 0.5)
            torso = (target.torso > 0.5) & (target.valid > 0.5)
            iou = (hard & torso).sum((1, 2)) / (hard | torso).sum((1, 2)).clamp_min(1)
            on = inside & (target.skin > 0.5)
            angle = torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))
            out["iou"] = iou
            out["angle"] = (angle * on).sum((1, 2)) / on.sum((1, 2)).clamp_min(1)
            out["nipple_pixels"] = nipple * s.render_size
        return out

    def total(parts: dict, shaping: bool) -> torch.Tensor:
        w = s.weights
        value = sum(
            w[k] * parts[k].mean() for k in ("outline", "normals", "nipples", "joints", "pose")
        )
        if shaping:
            value = value + w["offsets"] * model.offsets().div(MILLIMETRE).square().sum(1).mean()
            value = value + w["controls"] * model.controls.square().sum()
        return value

    def run(steps: int, parameters: list, shape_on, desc: str) -> None:
        optimiser = torch.optim.Adam(parameters, lr=s.learning_rate)
        for _ in tqdm(range(steps), desc=desc, disable=not progress, leave=False):
            optimiser.zero_grad()
            total(losses(shape_on), shape_on is not None).backward()
            optimiser.step()

    pose = [model.rotations, model.translations, model.turns]
    run(s.pose_steps, pose, None, "placing")
    with torch.no_grad():
        before = losses(None, scores=True)
    tilt, gravity = tilt_and_gravity(posing, rest, model)
    identity = np.nan_to_num(closeups.identity, nan=-1.0)
    placed = before["iou"].cpu().numpy() >= s.min_iou
    shaping = (tilt <= s.max_tilt) & (identity >= s.min_identity) & placed
    shape_on = torch.from_numpy(shaping).to(device)
    run(s.shape_steps, [*pose, model.controls, model.offset_values], shape_on, "shaping")
    with torch.no_grad():
        after = losses(None, scores=True)
        tilt, gravity = tilt_and_gravity(posing, rest, model)
        p, c = model.control_values(model.controls)
        p0, c0 = model.control_values(torch.zeros_like(model.controls))

    def picked(values_p, values_c):
        return torch.cat([values_p[model.phenotype_index], values_c[model.change_index]])

    def numpy(values):
        return values.detach().cpu().numpy()

    return BreastFit(
        change=numpy(model.change()), weights=numpy(region.weights),
        nipples=numpy(region.nipples), controls=numpy(picked(p, c)),
        controls_before=numpy(picked(p0, c0)), keys=closeups.keys,
        rotations=numpy(model.rotations), translations=numpy(model.translations),
        turns=numpy(model.turns), tilt=tilt, gravity=gravity, shaping=shaping,
        iou_before=numpy(before["iou"]), iou_after=numpy(after["iou"]),
        normal_before=numpy(before["angle"]), normal_after=numpy(after["angle"]),
        nipple_before=numpy(before["nipple_pixels"]), nipple_after=numpy(after["nipple_pixels"]),
    )  # fmt: skip


def avatar_body(arrays: dict, device: str) -> dict:
    """The avatar file's body: ``rest`` ``(B, 3)``, ``faces`` among the body vertices, and its
    skinning (``bind_poses``, ``bone_indices``, ``bone_weights``), as tensors on ``device``."""
    head_start = int(arrays["head_start"])
    faces = arrays["faces"].astype(np.int64)
    return {
        "rest": torch.from_numpy(arrays["rest_vertices"][:head_start]).float().to(device),
        "faces": torch.from_numpy(faces[(faces < head_start).all(axis=1)]).to(device),
        "bind_poses": torch.from_numpy(arrays["bind_poses"]).float().to(device),
        "bone_indices": torch.from_numpy(arrays["bone_indices"][:head_start].astype(np.int64)).to(
            device
        ),
        "bone_weights": torch.from_numpy(arrays["bone_weights"][:head_start]).float().to(device),
    }


@torch.no_grad()
def render_closeups(
    body, arrays: dict, bone_labels: list[str], fit: BreastFit, closeups, size: int,
    changed: bool, device: str,
) -> dict[str, np.ndarray]:  # fmt: skip
    """The torso as ``fit`` placed it in each close-up, with the avatar's breasts or (``changed``)
    the fitted ones: ``coverage`` ``(N, S, S)``, ``normals`` ``(N, S, S, 3)`` in the camera's
    axes, ``region`` ``(N, S, S)`` the breasts' weight, and ``nipples`` ``(N, 2, 2)`` pixels."""
    b = avatar_body(arrays, device)
    rest = b["rest"] + (torch.from_numpy(fit.change).to(device) if changed else 0)
    nipples = torch.from_numpy(fit.nipples).long().to(device)
    posing = Posing(body, b["bind_poses"], b["bone_indices"], b["bone_weights"], nipples)
    head_start = int(arrays["head_start"])
    drawn = torso_faces(
        b["faces"].cpu().numpy(), arrays["bone_indices"][:head_start],
        arrays["bone_weights"][:head_start], bone_labels, head_start,
    )  # fmt: skip
    weights = torch.from_numpy(fit.weights).to(device)
    renderer = Renderer(b["faces"][drawn], b["faces"], weights)
    K = torch.from_numpy(closeups.K).float().to(device).clone()
    K[:, :2] *= size / closeups.images.shape[1]
    out = {"coverage": [], "normals": [], "region": [], "nipples": []}
    for start in range(0, len(K), 16):
        part = slice(start, start + 16)
        vertices, _ = posing(
            rest, *(torch.from_numpy(getattr(fit, k)[part]).to(device)
                    for k in ("rotations", "translations", "turns")),
        )  # fmt: skip
        image = renderer(vertices, K[part], size)
        for key in ("coverage", "normals", "region"):
            out[key].append(image[key].cpu().numpy())
        out["nipples"].append(project(vertices[:, nipples], K[part]).cpu().numpy())
    return {key: np.concatenate(values) for key, values in out.items()}


def with_breasts(file, fit: BreastFit):
    """A copy of an ``AvatarFile`` with ``fit``'s breasts: its body vertices moved, and the
    fitted controls under ``metadata["breasts"]``."""
    from avatar_splats.export.avatar_file import AvatarFile

    head_start = int(file.metadata["head_start"])
    rest = file.arrays["rest_vertices"].copy()
    rest[:head_start] += fit.change.astype(rest.dtype)
    names = (*BREAST_PHENOTYPES, *BREAST_CHANGES)
    metadata = dict(file.metadata)
    metadata["breasts"] = {
        "controls": {n: round(float(v), 4) for n, v in zip(names, fit.controls, strict=True)},
        "nipples": [int(v) for v in fit.nipples],
    }
    return AvatarFile(metadata, dict(file.arrays, rest_vertices=rest))
