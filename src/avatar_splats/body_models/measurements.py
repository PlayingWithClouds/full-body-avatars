"""A body measured as a tailor would: height, weight, the tape-measure circumferences of the
bust, underbust, waist and hips, and the feet's length (a shoe size).

Anny's phenotype is not in centimetres or kilograms (its "weight" moves fat and its "cupsize"
the breasts, on scales of 0 to 1), so known measurements of the performer are matched by
measuring the mesh. A circumference is the perimeter of the convex outline of the body's
horizontal slice at a level, as a tape pulled round it lies; the arms are left out. The weight
is the closed mesh's volume times a body's density.

The levels and the outlines' corners are chosen without gradients, the perimeters and the
volume are differentiable in the vertices, so a fit can be held to the measurements.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import ConvexHull

from avatar_splats.body_models.anny import AnnyBody

BODY_DENSITY = 985.0
"""kg/m³: a body's average density, the lungs' air included."""
SLICE = 0.01
"""Metres: half the thickness of the slab whose vertices make a slice's outline."""
LEVELS = {"bust": (0.66, 0.80), "waist": (0.55, 0.68), "hips": (0.44, 0.57)}
"""Shares of the height above the soles where each circumference is looked for: the bust at
the breasts' foremost point, the waist where the torso is narrowest, the hips where they are
widest."""
UNDERBUST = (0.02, 0.08)
"""Shares of the height below the bust where the underbust is looked for, where the torso is
narrowest: on the ribs, under the breasts. Without it a tape round the bust cannot tell breasts
from a wide chest."""
SCAN_STEPS = 40
"""Levels tried in each range."""


def foot_ends(body: AnnyBody, rest: torch.Tensor) -> list[tuple[int, int, int]]:
    """Per foot, left first, the vertices of the big toe's tip, the little toe's tip and the
    heel, picked on ``(V, 3)`` rest vertices (facing -y, z up): the foremost vertex of the first
    and the fifth toe, the hindmost of the foot's lowest third."""
    dominant = body.skin_weights.argmax(dim=1)

    def of(*bones: str) -> torch.Tensor:
        return torch.isin(dominant, torch.tensor([body.bone(b) for b in bones], device=rest.device))

    def best(score: torch.Tensor, where: torch.Tensor) -> int:
        return int(torch.where(where, score, -torch.inf).argmax())

    ends = []
    for side in "LR":
        toes = [best(-rest[:, 1], of(f"toe{t}-1.{side}", f"toe{t}-2.{side}")) for t in (1, 5)]
        foot = of(f"foot.{side}")
        low = rest[:, 2] <= torch.quantile(rest[foot, 2], 1 / 3)
        ends.append((*toes, best(rest[:, 1], foot & low)))
    return ends


def arm_vertices(body: AnnyBody) -> torch.Tensor:
    """``(V,)`` bool: vertices the arms (from the clavicles down to the fingertips) move most;
    a tape round the bust, waist or hips passes inside them."""
    parents = body.model.bone_parents
    roots = {body.bone("clavicle.L"), body.bone("clavicle.R")}
    in_arm = []
    for bone in range(body.num_bones):
        b = bone
        while b >= 0 and b not in roots:
            b = parents[b]
        in_arm.append(b >= 0)
    in_arm = torch.tensor(in_arm, device=body.skin_weights.device)
    return in_arm[body.skin_weights.argmax(dim=1)]


class BodyMeasurements:
    """Measures ``(V, 3)`` rest vertices of Anny (metres, z up, facing -y)."""

    def __init__(self, body: AnnyBody):
        self.body = body
        self.faces = body.faces
        self.torso = ~arm_vertices(body)

    def volume(self, vertices: torch.Tensor) -> torch.Tensor:
        """The closed mesh's volume in m³ (the divergence theorem)."""
        a, b, c = (vertices[self.faces[:, k]] for k in range(3))
        return (a * torch.linalg.cross(b, c)).sum() / 6

    def _outline(self, vertices: np.ndarray, level: float) -> np.ndarray | None:
        """Indices of the torso's vertices on the convex outline of the slice at ``level``."""
        chosen = np.flatnonzero(self.torso.cpu().numpy() & (np.abs(vertices[:, 2] - level) < SLICE))
        if len(chosen) < 4:
            return None
        return chosen[ConvexHull(vertices[chosen, :2]).vertices]

    @staticmethod
    def _perimeter(points: torch.Tensor) -> torch.Tensor:
        return (points[:, :2] - points.roll(1, dims=0)[:, :2]).norm(dim=1).sum()

    def levels(self, vertices: torch.Tensor) -> dict[str, np.ndarray]:
        """Each circumference's outline (vertex indices, in order round it), found by scanning
        its range of levels, and the feet's ``(2, 2)`` big toe and heel vertices."""
        v = vertices.detach().cpu().numpy()
        floor, height = v[:, 2].min(), np.ptp(v[:, 2])
        torso = self.torso.cpu().numpy()
        outlines = {}
        for name, (low, high) in LEVELS.items():
            levels = floor + height * np.linspace(low, high, SCAN_STEPS)
            if name == "bust":  # where the breasts reach furthest forwards
                band = torso & (v[:, 2] > levels[0]) & (v[:, 2] < levels[-1])
                bust = v[band][np.argmin(v[band, 1]), 2]
                outlines[name] = self._outline(v, bust)
                below = bust - height * np.linspace(*UNDERBUST, SCAN_STEPS)
                outlines["underbust"] = self._extreme(v, below, min)
                continue
            outlines[name] = self._extreme(v, levels, min if name == "waist" else max)
        # the feet: big toe and heel of each
        outlines["foot"] = np.array(
            [(toe, heel) for toe, _, heel in foot_ends(self.body, vertices)]
        )
        return outlines

    def _extreme(self, vertices: np.ndarray, levels: np.ndarray, pick) -> np.ndarray:
        """The outline with the least (``pick`` min) or greatest perimeter among the levels."""
        found = [(o, self._perimeter(torch.from_numpy(vertices[o]))) for o in map(
            lambda level: self._outline(vertices, level), levels
        ) if o is not None]  # fmt: skip
        return pick(found, key=lambda pair: float(pair[1]))[0]

    def __call__(
        self, vertices: torch.Tensor, outlines: dict[str, np.ndarray] | None = None
    ) -> dict[str, torch.Tensor]:
        """Height (m), weight (kg), the circumferences and the feet's length (m) of ``(V, 3)``
        rest vertices; the outlines from ``levels`` (found anew where not given)."""
        outlines = self.levels(vertices) if outlines is None else outlines
        measured = {
            "height": vertices[:, 2].max() - vertices[:, 2].min(),
            "weight": BODY_DENSITY * self.volume(vertices),
        }
        for name, outline in outlines.items():
            points = vertices[torch.from_numpy(outline).to(vertices.device)]
            if name == "foot":  # heel to big toe along the ground, the feet's mean
                measured[name] = (points[:, 0, :2] - points[:, 1, :2]).norm(dim=-1).mean()
            else:
                measured[name] = self._perimeter(points)
        return measured


def measurement_loss(measured: dict[str, torch.Tensor], targets: dict[str, float]) -> torch.Tensor:
    """Summed squared relative errors of the measurements given in ``targets``."""
    return sum(((measured[k] - v) / v).square() for k, v in targets.items())


LOCAL_CHANGE_PRIOR = 0.01
"""Weight of the local changes' squared size against the measurements' squared relative error:
reach the measurements with the phenotype first."""


def fit_measurements(
    body: AnnyBody,
    phenotype: torch.Tensor,
    targets: dict[str, float],
    steps: int = 300,
    rescan_every: int = 25,
    held: tuple[str, ...] = ("gender",),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anny's ``(P,)`` phenotype and ``(L,)`` local changes that give the measured ``targets``
    (``height`` and the circumferences in metres, ``weight`` in kg), from ``phenotype``; the
    ``held`` phenotype values stay as given."""
    measure = BodyMeasurements(body)
    free = torch.tensor(
        [label not in held for label in body.phenotype_labels], device=phenotype.device
    )
    start = torch.logit(phenotype.clamp(1e-3, 1 - 1e-3))
    logits = torch.nn.Parameter(start.clone())
    raw = torch.nn.Parameter(torch.zeros(len(body.local_change_labels), device=phenotype.device))
    optimizer = torch.optim.Adam([logits, raw], lr=0.02)
    zeros = torch.zeros(1, body.num_bones, 3, device=phenotype.device)

    def shaped() -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.where(free, torch.sigmoid(logits), torch.sigmoid(start))
        return values, torch.tanh(raw)

    outlines = None
    for step in range(steps):
        values, changes = shaped()
        vertices = body(values[None], changes[None], zeros, zeros[:, 0])["rest_vertices"][0]
        if step % rescan_every == 0:
            outlines = measure.levels(vertices)
        loss = measurement_loss(measure(vertices, outlines), targets)
        loss = loss + LOCAL_CHANGE_PRIOR * changes.square().sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    values, changes = shaped()
    return values.detach(), changes.detach()
