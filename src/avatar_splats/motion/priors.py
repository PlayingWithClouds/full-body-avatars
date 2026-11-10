"""What one camera cannot see, filled in from what bodies can do: joints turn only so far (and
knees one way), and bodies, two or one, don't pass through each other."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from avatar_splats.body_models.anny import AnnyBody

SPHERES_PER_BONE = 3
"""Spheres along each bone's stretch of the body."""
MIN_BONE_VERTICES = 60
"""Bones moving fewer vertices than this (fingers, toes, the face's) get no spheres: they are
too small to matter for bodies passing through each other."""
SELF_CONTACT_HOPS = 2
"""Spheres on bones this many steps apart in the skeleton or fewer may overlap: a joint's two
sides press into each other as it bends."""


def knee_bend(
    hip: torch.Tensor, knee: torch.Tensor, ankle: torch.Tensor, lateral: torch.Tensor
) -> torch.Tensor:
    """How far ``(..., 3)`` legs bend forwards at the knee: zero for legs bent backwards.

    A knee bends the shin back, so the thigh and shin turn about the body's left; bent the other
    way, they turn about its right. Measured as the squared sine of the angle bent forwards.

    Args:
        lateral: ``(..., 3)`` from the right hip to the left one.
    """
    thigh, shin = knee - hip, ankle - knee
    turn = torch.linalg.cross(thigh, shin) / (thigh.norm(dim=-1) * shin.norm(dim=-1)).clamp_min(
        1e-9
    ).unsqueeze(-1)
    side = torch.nn.functional.normalize(lateral, dim=-1)
    return torch.relu(-(turn * side).sum(-1)).square()


_SPINE = ((-7, 19), (-7, 7), (-7, 7))
_NECK = ((-12.5, 12.5), (-21, 21), (-12.5, 12.5))
JOINT_LIMITS = {
    "spine05": _SPINE, "spine04": _SPINE, "spine03": _SPINE, "spine02": _SPINE, "spine01": _SPINE,
    "neck01": _NECK, "neck02": _NECK, "neck03": _NECK, "head": _NECK,
    "clavicle": ((-10, 10), (-10, 10), (-10, 10)),
    "shoulder01": ((-10, 10), (-10, 10), (-10, 10)),
    "upperarm01": ((-65, 185), (-47.5, 37.5), (-87, 143)),
    "upperarm02": ((-5, 5), (-47.5, 37.5), (-5, 5)),
    "lowerarm01": ((-48, 112), (-5, 5), (-5, 5)),
    "lowerarm02": ((-5, 5), (-85, 85), (-5, 5)),
    "wrist": ((-85, 85), (-5, 5), (-35, 35)),
    "upperleg01": ((-120, 40), (-35, 35), (-70, 40)),  # out and turned: as sitting allows
    "upperleg02": ((-5, 5), (-25, 25), (-5, 5)),
    "lowerleg01": ((-14, 131), (-5, 5), (-5, 5)),
    "lowerleg02": ((-5, 5), (-20, 20), (-5, 5)),
    "foot": ((-25, 55), (-40, 20), (-15, 15)),
    # the thumb: +x folds it across the palm, +z lifts it off the palm
    "finger1-1": ((-25, 20), (-30, 30), (-30, 75)),
    "finger1-2": ((-5, 55), (-5, 5), (-10, 10)),
    "finger1-3": ((-25, 85), (-5, 5), (-5, 5)),
    # the fingers: +x curls them towards the palm, z spreads them
    **{f"finger{f}-1": ((-50, 95), (-5, 5), (-25, 25)) for f in range(2, 6)},
    **{f"finger{f}-2": ((-5, 105), (-5, 5), (-5, 5)) for f in range(2, 6)},
    **{f"finger{f}-3": ((-15, 95), (-5, 5), (-5, 5)) for f in range(2, 6)},
}  # fmt: skip
"""Degrees each bone may turn from Anny's zero pose about its own axes, as (least, most) for
the left side; the right side's mirror them (its y and z turn the other way). x bends (an
elbow's or knee's hinge), y twists about the bone, z bends sideways.

From AAOS range-of-motion tables plus 5 degrees (the hip's sideways and outward turn widened to what
a bent hip allows: the tables measure them straight, lying down, and she sits), measured from Anny's
zero pose, not anatomical neutral: its arms are raised 42 degrees to the side, its elbows bent 43
degrees, its hips 5 and its knees 9. A region's range (the spine's, the neck's with the head; a
limb's twist over its two bones) is shared out over its bones. Directions, checked on the skeleton:
+x flexes the spine, neck, shoulder, elbow, knee and ankle (points the toes) and extends the hip;
for the left side +z raises the arm sideways, brings the leg in and turns the toes in; -y turns the
upper arm and the thigh outwards, +y the sole outwards. Not in the tables, so guesses: the clavicle
and shoulder01 (the shoulder blade, whose motion the shoulder's ranges already include: a little),
the shin's twist, the foot's sideways turn, the thumb's base twist; the wrist's are kept symmetric
(which way is the palm's is not checked). The fingers' are measured from the zero pose's slight
curl, not from straight; the palm's side is where the relaxed thumb's tip is."""


class JointLimits:
    """How far rotations go past ``JOINT_LIMITS``, the bones not listed being free."""

    def __init__(self, body: AnnyBody, rest_bones: torch.Tensor):
        """``rest_bones``: ``(J, 4, 4)`` the rest bone poses, whose rotations are the bones'
        own axes (y along the bone)."""
        self.axes = rest_bones[:, :3, :3]
        low = torch.full((body.num_bones, 3), -torch.inf)
        high = torch.full((body.num_bones, 3), torch.inf)
        for j, label in enumerate(body.bone_labels):
            name, _, side = label.partition(".")
            if name not in JOINT_LIMITS:
                continue
            for axis, (least, most) in enumerate(JOINT_LIMITS[name]):
                if side == "R" and axis > 0:
                    least, most = -most, -least
                low[j, axis], high[j, axis] = least, most
        self.low = torch.deg2rad(low).to(rest_bones.device)
        self.high = torch.deg2rad(high).to(rest_bones.device)

    def local(self, rotations: torch.Tensor) -> torch.Tensor:
        """``(..., J, 3)`` Anny ``local-ref`` rotation vectors (in world axes) in the bones' own
        axes."""
        return torch.einsum("jba,...jb->...ja", self.axes, rotations)

    def __call__(self, rotations: torch.Tensor) -> torch.Tensor:
        """``(...)`` the summed squared excess, in radians, of ``(..., J, 3)`` rotations."""
        local = self.local(rotations)
        excess = torch.relu(self.low - local) + torch.relu(local - self.high)
        return excess.square().sum((-1, -2))


@dataclass(frozen=True)
class BodyProxies:
    """Spheres filling the body, each carried by one bone, to tell when two bodies overlap.

    Each bone's vertices (those it moves most) are split into ``SPHERES_PER_BONE`` slices along
    their longest axis; a sphere sits at each slice's centre, as big as the slice is wide.
    """

    bones: torch.Tensor
    """``(S,)`` the bone carrying each sphere."""
    centres: torch.Tensor
    """``(S, 4)`` the spheres' centres in the rest pose, homogeneous."""
    radii: torch.Tensor
    """``(S,)`` in metres."""

    @classmethod
    def build(cls, body: AnnyBody, rest_vertices: torch.Tensor) -> BodyProxies:
        """Spheres for ``(V, 3)`` rest vertices with Anny's skinning weights."""
        dominant = body.skin_weights.argmax(dim=1)
        bones, centres, radii = [], [], []
        for bone in range(body.num_bones):
            points = rest_vertices[dominant == bone]
            if len(points) < MIN_BONE_VERTICES:
                continue
            centred = points - points.mean(dim=0)
            axis = torch.linalg.svd(centred, full_matrices=False).Vh[0]
            along = centred @ axis
            edges = torch.quantile(
                along, torch.linspace(0, 1, SPHERES_PER_BONE + 1, device=along.device)
            )
            for k in range(SPHERES_PER_BONE):
                part = points[(along >= edges[k]) & (along <= edges[k + 1])]
                centre = part.mean(dim=0)
                bones.append(bone)
                centres.append(centre)
                radii.append((part - centre).norm(dim=1).median())
        return cls(
            torch.tensor(bones, device=rest_vertices.device),
            torch.nn.functional.pad(torch.stack(centres), (0, 1), value=1.0),
            torch.stack(radii),
        )

    def far_pairs(self, body: AnnyBody, hops: int = SELF_CONTACT_HOPS) -> torch.Tensor:
        """``(S, S)`` bool: the pairs of spheres on bones more than ``hops`` steps apart in the
        skeleton, which one body's pose must keep apart."""
        parents = body.model.bone_parents
        chains = []
        for bone in range(body.num_bones):
            chain = [bone]
            while parents[chain[-1]] >= 0:
                chain.append(parents[chain[-1]])
            chains.append({b: i for i, b in enumerate(chain)})
        steps = torch.zeros(body.num_bones, body.num_bones, dtype=torch.long)
        for a in range(body.num_bones):
            for b in range(body.num_bones):
                common = min(chains[b], key=lambda x: chains[b][x] if x in chains[a] else 1 << 30)
                steps[a, b] = chains[a][common] + chains[b][common]
        bones = self.bones.cpu()
        return (steps[bones][:, bones] > hops).to(self.bones.device)

    def place(self, skinning: torch.Tensor) -> torch.Tensor:
        """``(..., S, 4)`` the spheres' posed centres and radii for ``(..., J, 4, 4)`` skinning
        matrices."""
        centres = (skinning[..., self.bones, :, :] @ self.centres.unsqueeze(-1))[..., :3, 0]
        return torch.cat((centres, self.radii.expand(*centres.shape[:-1]).unsqueeze(-1)), dim=-1)


def overlap(spheres: torch.Tensor, slack: float) -> torch.Tensor:
    """How deep the bodies of different people sink into each other, per frame on average.

    Args:
        spheres: ``(P, N, S, 4)`` each person's placed spheres.
        slack: share of the radii two spheres may overlap by for free: skin gives way.

    Returns:
        The squared depths in metres of every pair of overlapping spheres, summed over pairs of
        spheres and people, averaged over frames.
    """
    total = spheres.new_zeros(())
    for p in range(len(spheres)):
        for q in range(p + 1, len(spheres)):
            a, b = spheres[p], spheres[q]
            distance = torch.cdist(a[..., :3], b[..., :3])
            reach = (1 - slack) * (a[..., 3, None] + b[..., None, 3])
            total = total + torch.relu(reach - distance).square().sum((-1, -2)).mean()
    return total


def self_overlap(spheres: torch.Tensor, far: torch.Tensor, slack: float) -> torch.Tensor:
    """How deep a body sinks into itself, per body and frame on average: as ``overlap``, between
    ``(..., S, 4)`` placed spheres of one body, on the ``(S, S)`` ``BodyProxies.far_pairs``."""
    distance = torch.cdist(spheres[..., :3], spheres[..., :3])
    reach = (1 - slack) * (spheres[..., 3, None] + spheres[..., None, 3])
    depth = torch.relu(reach - distance) * far
    return depth.square().sum((-1, -2)).mean() / 2
