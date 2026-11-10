"""Genitals and anus for the avatar, generated on the fitted body: Anny has neither, and an
eye-level orbit sees too little of them to fit any.

The crotch is described in surface coordinates: ``s`` runs along the body's midline, from the
belly between the legs to the buttocks, and ``u`` across it, along x. A fine grid in these
coordinates is laid on Anny's surface (ray cast against it, then smoothed of its facets), lifted
along the surface normal into labia, a clitoral hood and an anus with radial folds, and stitched
into a hole cut in Anny's crotch. Each opening is a ring of rings narrowing to it, continued as a
blind canal into the body.

Shapes are blendshapes: ``vulvaSpread`` moves the labia apart; ``vaginaOpen1..4`` and
``anusOpen1..4`` widen the opening and successive quarters of its canal, so an engine can open a
canal as far as something has entered it. All four at 1 open it fully to ``open_radius``.

Material and place are kept apart: every surface vertex keeps its rest ``(u, s)``, which decides
its relief (a labium stays a labium), while a shape moves where on the surface it sits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from avatar_splats.body_models.anny import AnnyBody
from avatar_splats.body_models.mesh import (
    boundary_loops,
    closest_points,
    plane_barycentric,
    read_stl,
)
from avatar_splats.config import GenitalsConfig
from avatar_splats.rendering import rasterize

CM = 0.01
REFERENCE_HIP_WIDTH = 0.18
"""Distance, in metres, between the hip keypoints of the body the proportions below are for."""

SEGMENTS = 48
"""Vertices around each opening's rings and canal."""
OPENING_RINGS = 10
"""Rings from the edge of an opening's disc to the opening."""
DEPTH_STEP = 0.25 * CM
"""Spacing of the canal's rings."""
BLEND_DEPTH = 2.5 * CM
"""Depth over which an opening's shape turns into its canal's."""
ENTRY_LENGTH = 3 * CM
"""Depth over which a canal's rings turn from lying in the skin around the opening to square to
the canal, so a wide opening does not cut through the skin where the canal enters at a slant."""
CAP_LENGTH = 1.0 * CM
"""Length of the rounded end of each canal."""
OPEN_SEGMENTS = 4
"""Blendshapes opening each canal, from the entrance inwards."""
SKINNED_DEPTH = 3 * CM
"""Depth beyond which a canal follows the pelvis alone, not the skin around its opening."""
PELVIS = "spine05"
"""Anny's bone carrying the crotch."""
SURFACE_SMOOTHING = 0.4 * CM
"""Standard deviation of the blur that takes Anny's facets out of the crotch's surface."""
RAY_START = 6 * CM
"""How far outside the surface the rays that find it start."""
ON_SURFACE = 0.8 * CM
"""How far an Anny vertex may be from the generated surface and still count as under it."""
LATTICE_STEP = 0.05 * CM
MIDDLE_DENSITY = 4
"""How much closer the grid's columns are along the midline, for the slit, than at its sides."""
SCULPT_SAMPLES = 400
"""Samples of the sculpt's height along its vulva."""
SCULPT_BLUR = 0.2
"""Standard deviation, as a share of the sculpt's vulva length, of the blur whose difference
from its surface is the detail taken: the labia, the vestibule, not the buttocks around."""
SCULPT_WINDOW = (0.42, 0.58, -0.05)
"""Half axes across and along, and the shift of its centre towards the back, of the ellipse
around the sculpt's vulva its detail is taken from, as shares of its length."""
SCULPT_DEPTH = 0.6 * CM
"""How deep the sculpt's detail may sink below the skin."""
LOOP_RELAXING = 20
"""Rounds of smoothing the grid's edges, staircases of cells, before they are stitched: from a
distant vertex, a step running back flips the triangles fanning out to it."""
BAND_WIDTH = 0.5 * CM
"""How far inside the cut the grid stops. Anny's edges around the cut are straight chords that
sag below the skin by a millimetre or two; a narrower band between them and the grid flips."""
EDGE_FADE = 1.0 * CM
"""Distance from the cut over which the relief and every shape fade out, so the seam holds."""
SURFACE_COLUMN = 0.4
"""Width of the insert's tile taken by its outer surface, which spans the tile's height."""
CANAL_COLUMN = 0.26
"""Width of the tile taken by each canal unrolled: its circumference, opened."""
GAP = 0.04


@dataclass(frozen=True)
class Anatomy:
    """Places and sizes, in metres along the surface, for a body of the reference size."""

    hood_after_turn: float = 1.0 * CM
    """From where the front of the body turns under to the top of the clitoral hood."""
    vulva_length: float = 6.0 * CM
    """From the top of the hood to the back of the vestibule."""
    perineum: float = 2.8 * CM
    """From the back of the vestibule to the anus."""
    before_hood: float = 3.5 * CM
    """How far the generated surface reaches in front of the hood."""
    after_anus: float = 5.0 * CM
    half_width: float = 5.0 * CM
    """Half the width of the region cut out of Anny; the cut keeps to where Anny's surface is
    seen straight from below, so it ends nearer between the thighs."""
    hood_length: float = 0.7 * CM
    hood_width: float = 0.35 * CM
    hood_height: float = 0.65 * CM
    glans_height: float = 0.25 * CM
    vestibule_dip: float = 0.25 * CM
    """How far the vestibule sinks around the introitus, spread."""
    raphe_height: float = 0.04 * CM
    anus_funnel_depth: float = 0.25 * CM
    anus_funnel_width: float = 0.8 * CM
    fold_height: float = 0.03 * CM
    funnel_width: float = 1.2 * CM
    funnel_depth: float = 0.5 * CM
    """How far the skin around an opening draws in as it opens, instead of bunching up."""
    folds: int = 14
    introitus_half_width: float = 0.05 * CM
    introitus_half_length: float = 0.45 * CM
    introitus_disc: float = 1.3 * CM
    anus_radius: float = 0.12 * CM
    anus_disc: float = 1.3 * CM
    vagina_half_width: float = 1.2 * CM
    vagina_half_depth: float = 0.25 * CM
    anal_canal_radius: float = 0.15 * CM
    opening_reach: float = 4.5 * CM
    """How far from an opening the surface stretches as it opens."""
    spread: float = 0.8 * CM
    """How far each side of the vulva moves out, spread."""
    spread_softness: float = 0.4 * CM
    vagina_angle: float = math.radians(15)
    """The vaginal canal's tilt from straight up towards the back."""
    anal_angles: tuple[float, float] = (math.radians(0), math.radians(25))
    """The anal canal's tilt towards the back at its entrance and at its end."""

    def scaled(self, factor: float) -> Anatomy:
        lengths = {
            name: value * factor
            for name, value in self.__dict__.items()
            if isinstance(value, float) and not name.endswith("angle")
        }
        return Anatomy(**{**self.__dict__, **lengths})


@dataclass(frozen=True)
class Insert:
    """The generated genitals, joined to Anny's mesh in its rest pose."""

    removed: torch.Tensor
    """``(F_body,)`` bool: Anny's triangles the insert replaces."""
    border: torch.Tensor
    """``(B,)`` Anny's vertices around the cut."""
    border_vertices: torch.Tensor
    """``(B, 3)`` where they move to: onto the generated surface, which is Anny's smoothed of its
    facets, so the seam has no step."""
    vertices: torch.Tensor
    """``(N, 3)`` new vertices in the rest pose."""
    faces: torch.Tensor
    """``(M, 3)`` triangles indexing Anny's vertices, then the new ones (from ``V_body``)."""
    uvs: torch.Tensor
    """``(M, 3, 2)``: in Anny's texture layout where ``on_body_texture``, else in [0, 1]² of the
    insert's own tile."""
    on_body_texture: torch.Tensor
    """``(M,)`` bool: the triangles bridging to Anny, textured like the skin around them."""
    inside: torch.Tensor
    """``(M,)`` bool: the canals' walls."""
    weights: torch.Tensor
    """``(N, J)`` skinning weights of the new vertices."""
    shape_names: tuple[str, ...]
    shapes: torch.Tensor
    """``(S, N, 3)`` offsets of the new vertices for each shape at weight 1."""


@dataclass
class _Opening:
    name: str
    centre: tuple[float, float]
    """``(u, s)``."""
    rest: tuple[float, float]
    """Half axes of the opening across and along the midline."""
    disc: float
    canal: tuple[float, float]
    """Half axes of the canal at rest across the body and front to back."""
    depth: float
    angles: tuple[float, float]


class _Surface:
    """Anny's crotch as a smooth function of ``(u, s)``: points and outward normals."""

    def __init__(self, body: AnnyBody, vertices: torch.Tensor, hips: torch.Tensor):
        device = vertices.device
        faces = body.faces
        centres = vertices[faces].mean(dim=1)
        hip_centre = hips.mean(dim=0)
        near = (centres[:, 0].abs() < 0.15) & ((centres[:, 1:] - hip_centre[1:]).norm(dim=1) < 0.25)
        self.vertices, self.faces = vertices, faces[near]
        self.triangles = vertices[self.faces]

        angles = torch.linspace(math.radians(-100), math.radians(100), 801, device=device)
        directions = torch.stack((torch.zeros_like(angles), angles.sin(), -angles.cos()), dim=1)
        origin = torch.tensor([0.0, hip_centre[1], hip_centre[2]], device=device)
        distance = _first_hits(origin.expand_as(directions), directions, self.triangles)
        found = distance.isfinite()
        points = origin + distance[found, None] * directions[found]
        points = _smooth_curve(points, 9)
        steps = (points[1:] - points[:-1]).norm(dim=1)
        arc = torch.cat((steps.new_zeros(1), steps.cumsum(0)))
        self.length = float(arc[-1])
        samples = torch.arange(0, self.length, 0.001, device=device)
        self.curve = torch.stack([_interp(samples, arc, points[:, i]) for i in range(3)], dim=1)
        self.curve[:, 0] = 0
        tangent = torch.gradient(self.curve, dim=0)[0]
        normal = torch.stack((torch.zeros_like(tangent[:, 0]), -tangent[:, 2], tangent[:, 1]), 1)
        normal = F.normalize(normal, dim=1)
        outward = ((self.curve - origin) * normal).sum(1, keepdim=True).sign()
        self.curve_normals = normal * outward
        self.samples = samples

    def turn(self) -> float:
        """``s`` where the front of the body turns under: the midline's normal halfway between
        facing forward (-y) and down (-z)."""
        n = self.curve_normals
        facing = torch.atan2(-n[:, 1], -n[:, 2])
        front = torch.nonzero(facing > math.radians(45)).squeeze(-1)
        return float(self.samples[front.max()])

    def build_lattice(self, u_range: tuple[float, float], s_range: tuple[float, float]) -> None:
        """Sample the surface finely over a ``(u, s)`` box, for interpolating later.

        Each row is where the plane across the midline at its ``s`` cuts Anny, measured along
        the cut from the midline: so ``u`` follows the surface down the inner thighs and into
        the buttocks' cleft, where straight rays would leap from one surface to another.
        """
        device = self.curve.device
        us = np.arange(u_range[0], u_range[1] + LATTICE_STEP / 2, LATTICE_STEP)
        ss = np.arange(s_range[0], s_range[1] + LATTICE_STEP / 2, LATTICE_STEP)
        vertices = self.vertices.cpu().numpy().astype(np.float64)
        faces = self.faces.cpu().numpy()
        curve = self.curve.cpu().numpy().astype(np.float64)
        tangents = np.gradient(curve, axis=0)
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
        rows = []
        for s_value in ss:
            index = int(np.clip(round(s_value / 0.001), 0, len(curve) - 1))
            rows.append(_cross_section(vertices, faces, curve[index], tangents[index], us))
        grid_s, grid_u = torch.meshgrid(
            torch.from_numpy(ss).float(), torch.from_numpy(us).float(), indexing="ij"
        )
        self.raw = torch.from_numpy(np.stack(rows)).float().to(device)
        self.smooth = _blur(self.raw, SURFACE_SMOOTHING / LATTICE_STEP)
        self.lattice_coordinates = torch.stack((grid_u, grid_s), -1).to(device)
        middle = len(us) // 2
        self.outward = self.curve_normals[
            (torch.from_numpy(ss) / 0.001).round().long().clamp(0, len(self.curve) - 1)
        ]
        self.middle = middle
        self.u_range, self.s_range = u_range, s_range
        self._set(self.smooth)

    def _set(self, points: torch.Tensor) -> None:
        du = torch.gradient(points, dim=1)[0]
        ds = torch.gradient(points, dim=0)[0]
        normal = F.normalize(torch.cross(ds, du, dim=-1), dim=-1)
        if (normal[:, self.middle] * self.outward).sum() < 0:  # one orientation for the whole
            normal = -normal
        self.lattice = torch.cat((points, normal), -1).permute(2, 0, 1)[None]
        self.points = points

    def __call__(self, u: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(P, 3)`` points and unit normals at ``(P,)`` surface coordinates."""
        x = 2 * (u - self.u_range[0]) / (self.u_range[1] - self.u_range[0]) - 1
        y = 2 * (s - self.s_range[0]) / (self.s_range[1] - self.s_range[0]) - 1
        grid = torch.stack((x, y), -1).view(1, 1, -1, 2)
        values = F.grid_sample(self.lattice, grid, align_corners=True, padding_mode="border")
        values = values[0, :, 0].T
        return values[:, :3], F.normalize(values[:, 3:], dim=-1)

    def coordinates(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(P,)`` ``u``, ``s`` and distance of the nearest sampled point of the surface."""
        flat = self.raw.reshape(-1, 3)
        low, high = flat.amin(dim=0) - ON_SURFACE, flat.amax(dim=0) + ON_SURFACE
        near = ((points >= low) & (points <= high)).all(dim=1)
        distance = torch.full((len(points),), float("inf"), device=points.device)
        nearest = torch.zeros(len(points), dtype=torch.long, device=points.device)
        for chunk in torch.nonzero(near).squeeze(-1).split(256):
            distance[chunk], nearest[chunk] = torch.cdist(points[chunk], flat).min(dim=1)
        coordinates = self.lattice_coordinates.reshape(-1, 2)[nearest]
        return coordinates[:, 0], coordinates[:, 1], distance


def build_insert(body: AnnyBody, rest: dict[str, torch.Tensor], config: GenitalsConfig) -> Insert:
    """Generate the genitals on the fitted body in its rest pose.

    Args:
        rest: the fitted body's rest ``vertices`` ``(V, 3)`` and COCO ``keypoints`` ``(17, 3)``.
    """
    vertices = rest["vertices"]
    hips = rest["keypoints"][[11, 12]]
    scale = float((hips[0] - hips[1]).norm()) / REFERENCE_HIP_WIDTH
    anatomy = Anatomy().scaled(scale)
    surface = _Surface(body, vertices, hips)

    hood = surface.turn() + anatomy.hood_after_turn
    back = hood + anatomy.vulva_length
    anus = back + anatomy.perineum
    sculpt = _Sculpt(config, anatomy.vulva_length, back, vertices.device)
    introitus = back - (config.sculpt_introitus - config.sculpt_back) * sculpt.scale
    places = {"hood": hood, "back": back, "anus": anus, "introitus": introitus}
    s_range = (hood - anatomy.before_hood, anus + anatomy.after_anus)
    margin = 2 * CM
    surface.build_lattice(
        (-anatomy.half_width - margin, anatomy.half_width + margin),
        (s_range[0] - margin, s_range[1] + margin),
    )
    openings = [
        _Opening(
            "vagina",
            (0.0, introitus),
            (anatomy.introitus_half_width, anatomy.introitus_half_length),
            anatomy.introitus_disc,
            (anatomy.vagina_half_width, anatomy.vagina_half_depth),
            config.vagina_depth,
            (anatomy.vagina_angle, anatomy.vagina_angle),
        ),
        _Opening(
            "anus",
            (0.0, anus),
            (anatomy.anus_radius, anatomy.anus_radius),
            anatomy.anus_disc,
            (anatomy.anal_canal_radius, anatomy.anal_canal_radius),
            config.anus_depth,
            anatomy.anal_angles,
        ),
    ]
    removed, border = _cut(body, vertices, surface, anatomy, s_range)
    outline = torch.stack(surface.coordinates(vertices[border])[:2], 1)
    vertices = vertices.clone()
    vertices[border] = surface(outline[:, 0], outline[:, 1])[0]
    mesh = _Topology(anatomy, s_range, outline, openings, config.resolution, vertices.device)
    mesh.stitch(vertices, border, outline)

    def positions(state: dict[str, float]) -> torch.Tensor:
        return mesh.positions(surface, anatomy, places, openings, sculpt, state, config.open_radius)

    rest_positions = positions({})
    names, shapes = ["vulvaSpread"], [positions({"spread": 1.0}) - rest_positions]
    for opening in openings:
        opened = positions({opening.name: 1.0}) - rest_positions
        depth = mesh.depth_of[opening.name]
        for k in range(OPEN_SEGMENTS):
            names.append(f"{opening.name}Open{k + 1}")
            shapes.append(opened * _segment_weight(depth, opening.depth, k)[:, None])

    faces = mesh.oriented_faces(body.faces[~removed])
    return Insert(
        removed=removed,
        border=border,
        border_vertices=vertices[border],
        vertices=rest_positions,
        faces=faces,
        uvs=mesh.uvs(faces, len(vertices), rest_positions, body, vertices),
        on_body_texture=(faces < len(vertices)).any(dim=1),
        inside=mesh.inside(faces, len(vertices)),
        weights=mesh.weights(rest_positions, vertices, body),
        shape_names=tuple(names),
        shapes=torch.stack(shapes),
    )


class _Topology:
    """The insert's vertices and triangles, and what each vertex is: a point of the surface at
    a rest ``(u, s)``, or a point of a canal at a depth and angle."""

    def __init__(
        self,
        anatomy: Anatomy,
        s_range: tuple[float, float],
        outline: torch.Tensor,
        openings: list[_Opening],
        resolution: float,
        device: torch.device,
    ):
        self.anatomy, self.openings, self.s_range = anatomy, openings, s_range
        half = anatomy.half_width
        stretch = math.acosh(MIDDLE_DENSITY)
        step = resolution * math.sinh(stretch) / (half * stretch * MIDDLE_DENSITY)
        t = torch.linspace(-1, 1, 2 * math.ceil(1 / step) + 1, device=device)
        us = half * torch.sinh(stretch * t) / math.sinh(stretch)
        ss = torch.arange(s_range[0], s_range[1] + resolution / 2, resolution, device=device)
        grid_s, grid_u = torch.meshgrid(ss, us, indexing="ij")
        within, distance = _polygon(torch.stack((grid_u, grid_s), -1).reshape(-1, 2), outline)
        inside = (within & (distance > BAND_WIDTH)).view(grid_u.shape)
        for opening in openings:
            centre = torch.tensor([opening.centre], device=device)
            within, distance = _polygon(centre, outline)
            if not within[0] or distance[0] < opening.disc + 3 * resolution:
                raise ValueError(f"the {opening.name}'s disc reaches past the crotch's cut")
        for opening in openings:
            du, ds = grid_u - opening.centre[0], grid_s - opening.centre[1]
            inside &= (du.square() + ds.square()).sqrt() > opening.disc + resolution
        quads = inside[:-1, :-1] & inside[1:, :-1] & inside[:-1, 1:] & inside[1:, 1:]
        quads = _without_pinches(quads)

        rows, columns = len(ss), len(us)
        index = torch.arange(rows * columns, device=device).view(rows, columns)
        corners = torch.stack((index[:-1, :-1], index[:-1, 1:], index[1:, 1:], index[1:, :-1]), -1)[
            quads
        ]
        used = torch.unique(corners)
        remap = torch.full((rows * columns,), -1, dtype=torch.long, device=device)
        remap[used] = torch.arange(len(used), device=device)
        corners = remap[corners]
        faces = [torch.cat((corners[:, [0, 1, 2]], corners[:, [0, 2, 3]]))]
        u = [grid_u.reshape(-1)[used]]
        s = [grid_s.reshape(-1)[used]]
        count = len(used)

        self.surface_count = 0
        self.disc_rings: dict[str, torch.Tensor] = {}
        angles = torch.arange(SEGMENTS, device=device) * (2 * math.pi / SEGMENTS)
        self.angles = angles
        for opening in openings:
            rings = []
            for k in range(OPENING_RINGS + 1):
                t = k / OPENING_RINGS
                a_u = (1 - t) * opening.disc + t * opening.rest[0]
                a_s = (1 - t) * opening.disc + t * opening.rest[1]
                u.append(opening.centre[0] + a_u * angles.cos())
                s.append(opening.centre[1] + a_s * angles.sin())
                rings.append(count + torch.arange(SEGMENTS, device=device))
                count += SEGMENTS
            rings = torch.stack(rings)
            self.disc_rings[opening.name] = rings
            faces.append(_strips(rings))
        self.surface_count = count
        self.u, self.s = torch.cat(u), torch.cat(s)

        self.canal_rings: dict[str, torch.Tensor] = {}
        self.canal_depths: dict[str, torch.Tensor] = {}
        self.caps: dict[str, int] = {}
        for opening in openings:
            depths = torch.arange(DEPTH_STEP, opening.depth + DEPTH_STEP / 2, DEPTH_STEP)
            depths = depths.to(device)
            rings = [self.disc_rings[opening.name][-1]]
            for _ in depths:
                rings.append(count + torch.arange(SEGMENTS, device=device))
                count += SEGMENTS
            rings = torch.stack(rings)
            faces.append(_strips(rings))
            cap = count
            count += 1
            last = rings[-1]
            faces.append(torch.stack((last, last.roll(-1), torch.full_like(last, cap)), 1))
            self.canal_rings[opening.name] = rings
            self.canal_depths[opening.name] = depths
            self.caps[opening.name] = cap
        self.count = count

        self.depth_of = {}
        for opening in openings:
            depth = torch.zeros(count, device=device)
            on_canal = torch.zeros(count, dtype=torch.bool, device=device)
            rings = self.canal_rings[opening.name]
            depth[rings[1:]] = self.canal_depths[opening.name][:, None].expand(-1, SEGMENTS)
            depth[self.caps[opening.name]] = opening.depth
            on_canal[rings[1:]] = True
            on_canal[self.caps[opening.name]] = True
            surface_part = torch.arange(count, device=device) < self.surface_count
            # Surface vertices open with the entrance; the other canal's vertices not at all.
            self.depth_of[opening.name] = torch.where(
                on_canal | surface_part, depth, torch.full_like(depth, float("inf"))
            )

        self.faces = torch.cat(faces)
        self.grid_loops = _grid_loops(self.faces, self.u, self.s, openings, self.surface_count)
        outer, holes = self.grid_loops
        for loop in (outer, *holes.values()):
            for _ in range(LOOP_RELAXING):
                for values in (self.u, self.s):
                    along = values[loop]
                    values[loop] = 0.5 * along + 0.25 * (along.roll(1) + along.roll(-1))
        _, distance = _polygon(torch.stack((self.u, self.s), 1), outline)
        self.edge_fade = _smoothstep(distance / EDGE_FADE)

    def stitch(
        self, body_vertices: torch.Tensor, border: torch.Tensor, outline: torch.Tensor
    ) -> None:
        """Bridge the grid's outer edge to Anny's cut, and each opening's disc to its hole.

        Args:
            border: ``(B,)`` Anny's vertices around the cut, at ``(B, 2)`` ``outline``.
        """
        device = body_vertices.device
        offset = len(body_vertices)
        outer, holes = self.grid_loops
        bridges = [
            _zip(
                outline,
                border,
                torch.stack((self.u[outer], self.s[outer]), 1),
                outer + offset,
            )
        ]
        for opening in self.openings:
            ring = self.disc_rings[opening.name][0]
            hole = holes[opening.name]
            bridges.append(
                _zip(
                    torch.stack((self.u[hole], self.s[hole]), 1),
                    hole + offset,
                    torch.stack((self.u[ring], self.s[ring]), 1),
                    ring + offset,
                )
            )
        self.all_faces = torch.cat([self.faces + offset] + [b.to(device) for b in bridges])

    def positions(
        self,
        surface: _Surface,
        anatomy: Anatomy,
        places: dict[str, float],
        openings: list[_Opening],
        sculpt: _Sculpt,
        state: dict[str, float],
        open_radius: float,
    ) -> torch.Tensor:
        """``(N, 3)`` the new vertices for a state: ``spread`` and each opening's openness."""
        u, s = self.u, self.s
        moved_u, moved_s = u.clone(), s.clone()
        spread = state.get("spread", 0.0)
        if spread:
            along = _window(s, places["hood"] + anatomy.hood_length, places["back"], CM)
            fade = 1 - _smoothstep(
                (u.abs() - 0.7 * anatomy.half_width) / (0.25 * anatomy.half_width)
            )
            moved_u = (
                moved_u
                + spread * anatomy.spread * torch.tanh(u / anatomy.spread_softness) * along * fade
            )
        for opening in openings:
            amount = state.get(opening.name, 0.0)
            if amount:
                moved_u, moved_s = _open_map(
                    moved_u, moved_s, opening, open_radius, anatomy.opening_reach, amount
                )
        fade = self.edge_fade
        moved_u, moved_s = u + fade * (moved_u - u), s + fade * (moved_s - s)
        points, normals = surface(moved_u, moved_s)
        relief = fade * _relief(u, s, anatomy, places, openings, sculpt, state)
        result = torch.empty(self.count, 3, device=u.device)
        result[: self.surface_count] = points + relief[:, None] * normals

        for opening in openings:
            rings = self.canal_rings[opening.name]
            amount = state.get(opening.name, 0.0)
            entrance = result[rings[0]]
            centre, normal = surface(
                torch.tensor([opening.centre[0]], device=u.device),
                torch.tensor([opening.centre[1]], device=u.device),
            )
            lift = _relief(
                centre.new_tensor([opening.centre[0]]), centre.new_tensor([opening.centre[1]]),
                anatomy, places, openings, sculpt, state,
            )  # fmt: skip
            start = centre[0] + normal[0] * lift[0]
            depths = self.canal_depths[opening.name]
            line, tangents = _canal_line(start, opening.angles, opening.depth, depths)
            x_axis = torch.tensor([1.0, 0, 0], device=u.device)
            back = torch.tensor([0.0, 1, 0], device=u.device)
            across = x_axis.expand_as(tangents)
            along = F.normalize(back - (tangents @ back)[:, None] * tangents, dim=1)
            ahead, behind = surface(
                torch.full((2,), opening.centre[0], device=u.device),
                torch.tensor([opening.centre[1] + 1e-3, opening.centre[1] - 1e-3], device=u.device),
            )[0]
            skin = F.normalize(ahead - behind, dim=0)
            turned = _smoothstep(depths / ENTRY_LENGTH)[:, None]
            along = F.normalize((1 - turned) * skin + turned * along, dim=1)
            half_u = (1 - amount) * opening.canal[0] + amount * open_radius
            half_s = (1 - amount) * opening.canal[1] + amount * open_radius
            cap = 1 - _smoothstep((depths - (opening.depth - CAP_LENGTH)) / CAP_LENGTH) * 0.85
            cos, sin = self.angles.cos(), self.angles.sin()
            ring = (
                line[:, None]
                + (half_u * cap)[:, None, None] * cos[None, :, None] * across[:, None]
                + (half_s * cap)[:, None, None] * sin[None, :, None] * along[:, None]
            )
            blend = _smoothstep(depths / BLEND_DEPTH)[:, None, None]
            ring = (1 - blend) * entrance[None] + blend * ring
            result[rings[1:]] = ring
            result[self.caps[opening.name]] = line[-1] + 0.3 * CAP_LENGTH * tangents[-1]
        return result

    def oriented_faces(self, body_faces: torch.Tensor) -> torch.Tensor:
        """The insert's triangles wound like Anny's: across every shared edge, opposite ways."""
        faces = self.all_faces.cpu().numpy().copy()
        reference = body_faces.cpu().numpy()
        directed = {}
        for a, b, c in reference:
            for edge in ((a, b), (b, c), (c, a)):
                directed[edge] = True
        edges_of: dict[tuple[int, int], list[int]] = {}
        for i, (a, b, c) in enumerate(faces):
            for edge in ((a, b), (b, c), (c, a)):
                edges_of.setdefault((min(edge), max(edge)), []).append(i)
        done = np.zeros(len(faces), dtype=bool)
        queue = []
        for i, (a, b, c) in enumerate(faces):
            for x, y in ((a, b), (b, c), (c, a)):
                if (x, y) in directed:
                    faces[i] = faces[i][::-1]
                    done[i] = True
                    break
                if (y, x) in directed:
                    done[i] = True
                    break
            if done[i]:
                queue.append(i)
        while queue:
            i = queue.pop()
            a, b, c = faces[i]
            for x, y in ((a, b), (b, c), (c, a)):
                for j in edges_of[(min(x, y), max(x, y))]:
                    if done[j]:
                        continue
                    p, q, r = faces[j]
                    if (x, y) in ((p, q), (q, r), (r, p)):
                        faces[j] = faces[j][::-1]
                    done[j] = True
                    queue.append(j)
        return torch.from_numpy(faces).to(self.all_faces.device)

    def inside(self, faces: torch.Tensor, offset: int) -> torch.Tensor:
        """``(M,)`` bool: triangles of the canals' walls, past their entrance rings."""
        on_canal = torch.zeros(self.count, dtype=torch.bool, device=faces.device)
        for opening in self.openings:
            on_canal[self.canal_rings[opening.name][1:].reshape(-1)] = True
            on_canal[self.caps[opening.name]] = True
        local = (faces - offset).clamp_min(0)
        return (faces >= offset).all(1) & on_canal[local].all(1)

    def uvs(
        self,
        faces: torch.Tensor,
        offset: int,
        positions: torch.Tensor,
        body: AnnyBody,
        body_vertices: torch.Tensor,
    ) -> torch.Tensor:
        """``(M, 3, 2)`` texture coordinates (see ``Insert.uvs``).

        The surface lies in the left of the tile, ``u`` across and ``s`` down; each canal is
        unrolled in a column to its right, angle across and depth down, at the same scale.
        """
        device = faces.device
        height = self.s_range[1] - self.s_range[0]
        coords = torch.zeros(self.count, 2, device=device)
        surface = slice(0, self.surface_count)
        coords[surface, 0] = SURFACE_COLUMN * (self.u / (SURFACE_COLUMN * height) + 0.5)
        coords[surface, 1] = 1 - (self.s - self.s_range[0]) / height
        canal = torch.full((self.count,), -1, dtype=torch.long, device=device)
        canal_coords = torch.zeros(self.count, 2, device=device)
        for i, opening in enumerate(self.openings):
            rings = self.canal_rings[opening.name]
            depth = torch.cat((torch.zeros(1, device=device), self.canal_depths[opening.name]))
            left = SURFACE_COLUMN + GAP + i * (CANAL_COLUMN + GAP)
            canal_coords[rings, 0] = left + (self.angles / (2 * math.pi) * CANAL_COLUMN)[None]
            canal_coords[rings, 1] = 1 - depth[:, None].expand_as(rings) / height
            cap = self.caps[opening.name]
            canal_coords[cap] = torch.tensor(
                [left + CANAL_COLUMN / 2, 1 - opening.depth / height], device=device
            )
            canal[rings[1:]] = i
            canal[cap] = i

        own = (faces >= offset).all(dim=1)
        local = (faces - offset).clamp_min(0)
        uvs = coords[local]
        on_canal = own & (canal[local] >= 0).any(dim=1)
        walls = canal_coords[local[on_canal]]
        low = walls[..., 0] < walls[..., 0].amax(dim=1, keepdim=True) - CANAL_COLUMN / 2
        walls[..., 0] += low * CANAL_COLUMN  # the triangles closing each ring across its seam
        uvs[on_canal] = walls

        bridge = ~own
        corners = torch.where(
            (faces[bridge] < offset)[..., None],
            body_vertices[faces[bridge].clamp_max(offset - 1)],
            positions[local[bridge]],
        )
        uvs[bridge] = _body_uvs(corners, faces[bridge], offset, body, body_vertices)
        return uvs

    def weights(
        self, positions: torch.Tensor, body_vertices: torch.Tensor, body: AnnyBody
    ) -> torch.Tensor:
        """``(N, J)``: the surface takes the skin's weights under it; the canals turn from their
        entrance's weights to the pelvis's alone over ``SKINNED_DEPTH``."""
        distances, nearest = torch.cdist(positions, body_vertices).topk(4, largest=False)
        blend = 1 / distances.clamp_min(1e-4)
        blend = blend / blend.sum(-1, keepdim=True)
        weights = (body.skin_weights[nearest] * blend[..., None]).sum(1)
        pelvis = torch.zeros(weights.shape[1], device=weights.device)
        pelvis[body.bone(PELVIS)] = 1
        for opening in self.openings:
            rings = self.canal_rings[opening.name]
            entrance = weights[rings[0]]
            depth = self.canal_depths[opening.name]
            t = (depth / SKINNED_DEPTH).clamp(max=1)[:, None, None]
            weights[rings[1:]] = (1 - t) * entrance[None] + t * pelvis
            weights[self.caps[opening.name]] = pelvis
        return weights


def _relief(
    u: torch.Tensor,
    s: torch.Tensor,
    a: Anatomy,
    places: dict[str, float],
    openings: list[_Opening],
    sculpt: _Sculpt,
    state: dict[str, float],
) -> torch.Tensor:
    """``(P,)`` how far, in metres, the surface at rest coordinates is lifted along its normal.

    The vulva's labia and vestibule are the sculpt's; in front, where it stops, the hood and the
    glans under it are generated; so are the perineum's raphe and the anus with its folds.
    """
    hood, back, anus = places["hood"], places["back"], places["anus"]
    spread = state.get("spread", 0.0)
    detail, weight = sculpt(u, s)
    relief = (1 - 0.3 * spread) * detail
    dip = _bump(u, 0.4 * CM) * _bump(s - places["introitus"], 0.8 * CM)
    relief = relief - spread * a.vestibule_dip * dip
    hood_fold = (
        a.hood_height * _bump(u, a.hood_width) * _bump(s - hood - 0.5 * a.hood_length, 0.45 * CM)
    )
    glans = a.glans_height * _bump(u, 0.12 * CM) * _bump(s - hood - 1.2 * a.hood_length, 0.12 * CM)
    relief = relief + (1 - weight) * torch.maximum(hood_fold, glans)

    relief = relief + a.raphe_height * _bump(u, 0.08 * CM) * _window(s, back, anus, 0.5 * CM)
    du, ds = u, s - anus
    distance = (du.square() + ds.square()).sqrt()
    relief = relief - a.anus_funnel_depth * _bump(distance, a.anus_funnel_width)
    folds = torch.cos(a.folds * torch.atan2(ds, du)) * _bump(distance - 0.3 * CM, 0.15 * CM)
    relief = relief + (1 - state.get("anus", 0.0)) * a.fold_height * folds

    for opening in openings:
        amount = state.get(opening.name, 0.0)
        if amount:
            du, ds = u - opening.centre[0], s - opening.centre[1]
            near = amount * _bump((du.square() + ds.square()).sqrt(), a.funnel_width)
            relief = relief * (1 - 0.8 * near) - a.funnel_depth * near
    return relief


class _Sculpt:
    """A sculpted vulva's surface detail, as a relief over the generated surface's ``(u, s)``.

    The sculpt is drawn from above into a height map; the detail is its height less a blurred
    copy, within an ellipse around its vulva, scaled so the vulva is as long as the avatar's.
    """

    def __init__(self, config: GenitalsConfig, vulva_length: float, back: float, device):
        vertices, faces = read_stl(config.sculpt)
        length = config.sculpt_front - config.sculpt_back
        self.scale = vulva_length / length
        """Metres per unit of the sculpt."""
        self.config, self.back = config, back
        step = length / SCULPT_SAMPLES
        low, high = vertices.min(0), vertices.max(0)
        width, height = (np.ceil((high[:2] - low[:2]) / step) + 1).astype(int)
        screen = np.stack(
            ((vertices[:, 0] - low[0]) / step, (high[1] - vertices[:, 1]) / step,
             high[2] + 1 - vertices[:, 2]), 1
        )  # fmt: skip
        fragments = rasterize(
            torch.from_numpy(screen).float().to(device),
            torch.from_numpy(faces).to(device),
            int(width),
            int(height),
        )
        heights = high[2] + 1 - fragments.depth_image()
        heights = torch.where(heights.isfinite(), heights, float(low[2]))
        blurred = _blur(heights[..., None], SCULPT_BLUR * length / step)[..., 0]

        ys = high[1] - torch.arange(int(height), device=device) * step
        xs = low[0] + torch.arange(int(width), device=device) * step
        across, along, shift = SCULPT_WINDOW
        middle = (config.sculpt_back + config.sculpt_front) / 2 + shift * length
        radius = (
            ((xs[None] - config.sculpt_midline) / (across * length)).square()
            + ((ys[:, None] - middle) / (along * length)).square()
        ).sqrt()
        window = 1 - _smoothstep((radius - 0.75) / 0.25)
        detail = (heights - blurred) * self.scale
        # The sculpt's entrance is a deep pit; the canal is the avatar's hole, so the detail
        # sinks no deeper than the vestibule, easing into that floor.
        detail = torch.where(
            detail < 0, -SCULPT_DEPTH * (1 - torch.exp(detail / SCULPT_DEPTH)), detail
        )
        self.maps = torch.stack((detail * window, window))[None].float()
        self.low_x, self.high_y, self.size = (
            float(low[0]),
            float(high[1]),
            (int(width), int(height)),
        )
        self.step = step

    def __call__(self, u: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(P,)`` the detail in metres at surface coordinates, and how much of it is the
        sculpt's (1 well inside its vulva, 0 outside)."""
        x = self.config.sculpt_midline + u / self.scale
        y = self.config.sculpt_back + (self.back - s) / self.scale
        column = (x - self.low_x) / self.step
        row = (self.high_y - y) / self.step
        grid = torch.stack(
            (2 * column / (self.size[0] - 1) - 1, 2 * row / (self.size[1] - 1) - 1), -1
        )
        values = F.grid_sample(self.maps, grid.view(1, 1, -1, 2).float(), align_corners=True)
        return values[0, 0, 0], values[0, 1, 0]


def _open_map(
    u: torch.Tensor,
    s: torch.Tensor,
    opening: _Opening,
    open_radius: float,
    reach: float,
    amount: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Surface coordinates pushed out radially from an opening opened by ``amount``: its edge
    moves from its rest ellipse to a circle of ``open_radius``.

    The skin around it stretches rather than bunching up: each ring moves out just enough to
    keep the area inside it, which pushes less the farther out it is, and the push fades out
    over the outer half of ``reach``.
    """
    du, ds = u - opening.centre[0], s - opening.centre[1]
    radius = (du.square() + ds.square()).sqrt().clamp_min(1e-9)
    cos, sin = du / radius, ds / radius
    edge = 1 / ((cos / opening.rest[0]).square() + (sin / opening.rest[1]).square()).sqrt()
    target = (1 - amount) * edge + amount * open_radius
    outside = radius.clamp_min(edge)
    stretched = (outside.square() + target.square() - edge.square()).sqrt()
    fade = 1 - _smoothstep((radius - reach / 2) / (reach / 2))
    pushed = radius + (stretched - outside) * fade
    return opening.centre[0] + pushed * cos, opening.centre[1] + pushed * sin


def _canal_line(
    start: torch.Tensor, angles: tuple[float, float], length: float, depths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(D, 3)`` points and unit tangents of a canal's centre line at ``depths``: from
    ``start`` upwards, tilted towards the back by an angle turning from the first to the
    second of ``angles``."""
    fine = torch.linspace(0, length, 400, device=start.device)
    tilt = angles[0] + (angles[1] - angles[0]) * _smoothstep(fine / length)
    tangent = torch.stack((torch.zeros_like(tilt), tilt.sin(), tilt.cos()), 1)
    step = fine[1] - fine[0]
    points = start + torch.cat((tangent.new_zeros(1, 3), (tangent[:-1] * step).cumsum(0)))
    index = (depths / step).round().long().clamp(0, len(fine) - 1)
    return points[index], tangent[index]


def _segment_weight(depth: torch.Tensor, length: float, k: int) -> torch.Tensor:
    """``(N,)`` how much opening segment ``k`` moves vertices at ``depth``: hat functions over
    the canal that add up to 1; the surface (depth 0) belongs to the first."""
    nodes = torch.linspace(0, length, OPEN_SEGMENTS, device=depth.device)
    spacing = length / (OPEN_SEGMENTS - 1)
    weight = (1 - (depth - nodes[k]).abs() / spacing).clamp_min(0)
    if k == 0:
        weight = torch.where(depth <= 0, torch.ones_like(weight), weight)
    if k == OPEN_SEGMENTS - 1:
        weight = torch.where(depth.isfinite() & (depth >= length), torch.ones_like(weight), weight)
    return torch.where(depth.isfinite(), weight, torch.zeros_like(weight))


def _cut(
    body: AnnyBody,
    vertices: torch.Tensor,
    surface: _Surface,
    anatomy: Anatomy,
    s_range: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anny's triangles under the generated surface, and the loop of vertices around them."""
    u, s, distance = surface.coordinates(vertices)
    under = (distance < ON_SURFACE) & (_domain(u, s, anatomy.half_width, s_range) <= 1)
    removed = under[body.faces].all(dim=1)
    loops = boundary_loops(body.faces[removed].cpu().numpy())
    outer = max(loops, key=len)
    if len(loops) > 1:
        raise ValueError(f"the crotch's cut is not one hole: {len(loops)} loops around it")
    return removed, torch.from_numpy(outer).to(vertices.device)


def _cross_section(
    vertices: np.ndarray, faces: np.ndarray, centre: np.ndarray, tangent: np.ndarray, us: np.ndarray
) -> np.ndarray:
    """``(U, 3)`` points where the plane through ``centre`` square to ``tangent`` cuts the mesh,
    at distances ``us`` along the cut from where it is nearest ``centre``, positive towards +x.
    Past the ends of the cut, its last points repeat."""
    side = (vertices - centre) @ tangent
    side[np.abs(side) < 1e-12] = 1e-12
    crossed = {}
    neighbours: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face in faces[(np.sign(side[faces]).min(1) < 0) & (np.sign(side[faces]).max(1) > 0)]:
        edges = []
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            if side[a] * side[b] < 0:
                key = (min(a, b), max(a, b))
                t = side[a] / (side[a] - side[b])
                crossed[key] = vertices[a] + t * (vertices[b] - vertices[a])
                edges.append(key)
        if len(edges) == 2:
            neighbours.setdefault(edges[0], []).append(edges[1])
            neighbours.setdefault(edges[1], []).append(edges[0])
    keys = list(crossed)
    points = np.array([crossed[k] for k in keys])
    start = keys[int(np.argmin(np.linalg.norm(points - centre, axis=1)))]

    def walk(first: tuple[int, int]) -> list[np.ndarray]:
        path, previous, current = [crossed[start]], start, first
        while current is not None and current != start and len(path) < 400:
            path.append(crossed[current])
            following = [k for k in neighbours.get(current, []) if k != previous]
            previous, current = current, (following[0] if following else None)
        return path

    branches = [walk(first) for first in neighbours.get(start, [])[:2]]
    while len(branches) < 2:
        branches.append([crossed[start]])
    if branches[0][min(1, len(branches[0]) - 1)][0] < branches[1][min(1, len(branches[1]) - 1)][0]:
        branches.reverse()
    result = np.empty((len(us), 3))
    for sign, branch in ((1, branches[0]), (-1, branches[1])):
        path = np.array(branch)
        arc = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))
        chosen = (us >= 0) if sign > 0 else (us < 0)
        for i in range(3):
            result[chosen, i] = np.interp(sign * us[chosen], arc, path[:, i])
    return result


def _polygon(points: torch.Tensor, polygon: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(P,)`` whether ``(P, 2)`` points are inside a ``(B, 2)`` closed polygon, and how far
    they are from its edges."""
    start, end = polygon, polygon.roll(-1, dims=0)
    p = points[:, None]
    edge = end - start
    t = (((p - start) * edge).sum(-1) / edge.square().sum(-1).clamp_min(1e-20)).clamp(0, 1)
    distance = (p - start - t[..., None] * edge).norm(dim=-1).amin(dim=1)
    crosses = (start[:, 1] > p[..., 1]) != (end[:, 1] > p[..., 1])
    x = start[:, 0] + (p[..., 1] - start[:, 1]) / (edge[:, 1] + 1e-20) * edge[:, 0]
    inside = (crosses & (p[..., 0] < x)).sum(dim=1) % 2 == 1
    return inside, distance


def _domain(u: torch.Tensor, s: torch.Tensor, half: float, s_range: tuple[float, float]):
    """A superellipse over the generated surface: at most 1 inside it."""
    middle, half_length = (s_range[0] + s_range[1]) / 2, (s_range[1] - s_range[0]) / 2
    return (u / half).abs() ** 4 + ((s - middle) / half_length).abs() ** 4


def _without_pinches(quads: torch.Tensor) -> torch.Tensor:
    """The grid's cells without any pair touching at a corner only, which would join the loops
    around them."""
    quads = quads.clone()
    while True:
        a, b = quads[:-1, :-1], quads[:-1, 1:]
        c, d = quads[1:, :-1], quads[1:, 1:]
        pinch = (a & d & ~b & ~c) | (b & c & ~a & ~d)
        if not pinch.any():
            return quads
        rows, columns = torch.nonzero(pinch, as_tuple=True)
        quads[rows, columns] = False
        quads[rows + 1, columns + 1] = False
        quads[rows, columns + 1] = False
        quads[rows + 1, columns] = False


def _grid_loops(
    faces: torch.Tensor,
    u: torch.Tensor,
    s: torch.Tensor,
    openings: list[_Opening],
    surface_count: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The grid's outer loop, and per opening the loop around its hole."""
    grid_faces = faces[(faces < surface_count).all(1)]
    discs = torch.zeros(surface_count, dtype=torch.bool, device=faces.device)
    for opening in openings:
        du, ds = u - opening.centre[0], s - opening.centre[1]
        discs |= (du.square() + ds.square()).sqrt() <= opening.disc + 1e-6
    grid_faces = grid_faces[~discs[grid_faces].any(1)]
    loops = [
        torch.from_numpy(loop).to(faces.device) for loop in boundary_loops(grid_faces.cpu().numpy())
    ]
    outer = max(loops, key=lambda loop: float(u[loop].max() - u[loop].min()))
    holes = {}
    for opening in openings:
        centre = torch.tensor(opening.centre, device=faces.device)

        def distance(loop: torch.Tensor, centre: torch.Tensor = centre) -> float:
            return float((torch.stack((u[loop], s[loop]), 1) - centre).norm(dim=1).mean())

        holes[opening.name] = min((loop for loop in loops if loop is not outer), key=distance)
    return outer, holes


def _zip(
    lower: torch.Tensor, lower_index: torch.Tensor, upper: torch.Tensor, upper_index: torch.Tensor
) -> torch.Tensor:
    """Triangles bridging two loops given by ``(N, 2)`` and ``(M, 2)`` surface coordinates, as
    vertex indices: points are paired by how far along its loop each lies, from the pair
    nearest each other, so long thin loops zip as evenly as round ones."""
    a, b = lower.cpu().numpy(), upper.cpu().numpy()
    a_index, b_index = lower_index, upper_index
    if _signed_area(a) < 0:
        a, a_index = a[::-1], a_index.flip(0)
    if _signed_area(b) < 0:
        b, b_index = b[::-1], b_index.flip(0)
    shift = int(np.argmin(np.linalg.norm(b - a[0], axis=1)))
    b, b_index = np.roll(b, -shift, axis=0), b_index.roll(-shift)

    def along(points: np.ndarray) -> np.ndarray:
        steps = np.linalg.norm(np.diff(points, axis=0, append=points[:1]), axis=1)
        return np.concatenate(([0], np.cumsum(steps))) / steps.sum()

    ta, tb = along(a), along(b)
    n, m = len(a), len(b)
    triangles, i, j = [], 0, 0
    while i < n or j < m:
        if j == m or (i < n and ta[i + 1] <= tb[j + 1]):
            triangles.append((i % n, (i + 1) % n, n + j % m))
            i += 1
        else:
            triangles.append((i % n, n + (j + 1) % m, n + j % m))
            j += 1
    indices = torch.from_numpy(np.array(triangles)).to(lower_index.device)
    return torch.cat((a_index, b_index))[indices]


def _signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * (x * np.roll(y, -1) - np.roll(x, -1) * y).sum())


def _strips(rings: torch.Tensor) -> torch.Tensor:
    """Triangles joining each of ``(R, N)`` rings of vertex indices to the next."""
    a, b = rings[:-1], rings[1:]
    a_next, b_next = a.roll(-1, dims=1), b.roll(-1, dims=1)
    first = torch.stack((a, a_next, b_next), -1).reshape(-1, 3)
    second = torch.stack((a, b_next, b), -1).reshape(-1, 3)
    return torch.cat((first, second))


def _body_uvs(
    corners: torch.Tensor,
    faces: torch.Tensor,
    offset: int,
    body: AnnyBody,
    body_vertices: torch.Tensor,
) -> torch.Tensor:
    """``(T, 3, 2)`` body texture coordinates for ``(T, 3, 3)`` triangles bridging to Anny: from
    the plane of the Anny triangle nearest each one's centre among those around its Anny corners,
    so none spans a seam of Anny's layout."""
    anchors = torch.unique(faces[faces < offset])
    around = torch.nonzero(torch.isin(body.faces, anchors).any(dim=1)).squeeze(-1)
    nearest, _ = closest_points(corners.mean(dim=1), body_vertices, body.faces[around])
    chosen = around[nearest]
    triangle = body_vertices[body.faces[chosen]]
    weights = plane_barycentric(corners, *triangle.unsqueeze(1).unbind(dim=2))
    return weights @ body.uvs[chosen]


def _first_hits(
    origins: torch.Tensor, directions: torch.Tensor, triangles: torch.Tensor, chunk: int = 2048
) -> torch.Tensor:
    """``(R,)`` distance along each ray to the first triangle it hits, infinite where none.

    Args:
        triangles: ``(T, 3, 3)``.
    """
    a, b, c = triangles.unbind(1)
    e1, e2 = b - a, c - a
    distances = []
    for start in range(0, len(origins), chunk):
        o = origins[start : start + chunk, None]
        d = directions[start : start + chunk, None]
        p = torch.cross(d.expand(-1, len(a), -1), e2[None].expand(len(o), -1, -1), dim=-1)
        det = (e1[None] * p).sum(-1)
        inverse = 1 / torch.where(det.abs() < 1e-12, torch.ones_like(det), det)
        t_vec = o - a[None]
        bu = (t_vec * p).sum(-1) * inverse
        q = torch.cross(t_vec, e1[None].expand(len(o), -1, -1), dim=-1)
        bv = (d * q).sum(-1) * inverse
        t = (e2[None] * q).sum(-1) * inverse
        hit = (det.abs() >= 1e-12) & (bu >= -1e-6) & (bv >= -1e-6) & (bu + bv <= 1 + 1e-6)
        hit &= t > 1e-6
        distances.append(torch.where(hit, t, torch.full_like(t, float("inf"))).amin(dim=1))
    return torch.cat(distances)


def _smooth_curve(points: torch.Tensor, width: int) -> torch.Tensor:
    kernel = torch.ones(1, 1, width, device=points.device) / width
    padded = F.pad(points.T[:, None], (width // 2, width // 2), mode="replicate")
    return F.conv1d(padded, kernel)[:, 0].T


def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(np.interp(x.cpu().numpy(), xp.cpu().numpy(), fp.cpu().numpy())).to(x)


def _blur(points: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = math.ceil(3 * sigma)
    offsets = torch.arange(-radius, radius + 1, device=points.device, dtype=points.dtype)
    kernel = torch.exp(-0.5 * (offsets / sigma).square())
    kernel = kernel / kernel.sum()
    channels = points.permute(2, 0, 1)[:, None]
    channels = F.pad(channels, (radius, radius, radius, radius), mode="replicate")
    channels = F.conv2d(channels, kernel.view(1, 1, -1, 1))
    channels = F.conv2d(channels, kernel.view(1, 1, 1, -1))
    return channels[:, 0].permute(1, 2, 0)


def _bump(x: torch.Tensor, width: float) -> torch.Tensor:
    return torch.exp(-0.5 * (x / width).square())


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0, 1)
    return x * x * (3 - 2 * x)


def _window(x: torch.Tensor, start: float, end: float, ramp: float) -> torch.Tensor:
    return _smoothstep((x - start) / ramp + 0.5) * _smoothstep((end - x) / ramp + 0.5)
