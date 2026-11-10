/**
 * The body the hair rests on: capsules for the neck, the shoulder girdle and the upper arms, and
 * the torso as a height field around a vertical axis through it, so long hair lies on the flat of
 * the back and on the chest instead of on rounded stand-ins. Capsule ends hang on bones and the
 * torso on the chest bone, so all follow the pose; sizes and the torso's shape are measured on
 * the avatar's mesh in its bind pose, so they fit whatever body the avatar file holds.
 */

import type { Rig } from '#lib/avatar/rig.ts';
import { invertRigid, multiply4, type Mat4, type Vec3 } from '#lib/math.ts';

export const MAX_CAPSULES = 8;
/** Torso height field size: around the axis, and from top to bottom (1.25 cm apart). */
export const TORSO_ANGLES = 64;
export const TORSO_HEIGHTS = 32;

export interface Capsule {
	boneA: number;
	/** End `a` in bone A's frame. */
	a: Vec3;
	boneB: number;
	b: Vec3;
	radius: number;
}

/** The torso's surface as distances from a vertical axis, in the bind pose's world frame. */
export interface Torso {
	/** The bone it moves with, and that bone's bind transform. */
	bone: number;
	bind: Mat4;
	/** The axis's x and y, and the field's top and bottom z. */
	shape: [number, number, number, number];
	/**
	 * `(TORSO_HEIGHTS * TORSO_ANGLES)` distance of the surface from the axis, rows from the top
	 * down; columns by angle from the back (+y) through the avatar's left (+x).
	 */
	field: Float32Array;
}

export interface Body {
	capsules: Capsule[];
	torso: Torso | null;
}

/** The body of an avatar whose bind pose has bone transforms `bind` and `vertices`. */
export function measureBody(rig: Rig, bind: Mat4[], vertices: Float32Array): Body {
	const has = (name: string) => rig.boneNames.includes(name);
	const at = (name: string): Vec3 => {
		const m = bind[rig.bone(name)];
		return [m[3], m[7], m[11]];
	};
	const world: { boneA: string; a: Vec3; boneB: string; b: Vec3; radius: number }[] = [];
	const measure = (a: Vec3, b: Vec3, reach: number, share: number) =>
		fitRadius(vertices, a, b, reach, share);

	if (has('neck01') && has('head')) {
		const a = at('neck01');
		const b = at('head');
		world.push({ boneA: 'neck01', a, boneB: 'head', b, radius: measure(a, b, 0.09, 0.85) });
	}

	// Arms, to tell torso vertices from arm vertices and as colliders themselves.
	const arms: [Vec3, Vec3][] = [];
	let shoulderHeight = -Infinity;
	for (const side of ['L', 'R']) {
		const names = [`upperarm01.${side}`, `lowerarm01.${side}`, `wrist.${side}`];
		if (!names.every(has)) continue;
		const [shoulder, elbow, wrist] = names.map(at);
		shoulderHeight = Math.max(shoulderHeight, shoulder[2]);
		arms.push([shoulder, elbow], [elbow, wrist]);
		world.push({
			boneA: names[0],
			a: shoulder,
			boneB: names[0],
			b: elbow,
			radius: measure(shoulder, elbow, 0.08, 0.5)
		});
		// The deltoid, thicker than the arm below it.
		const deltoid: Vec3 = [0, 1, 2].map(
			(i) => shoulder[i] + 0.35 * (elbow[i] - shoulder[i])
		) as Vec3;
		world.push({
			boneA: names[0],
			a: shoulder,
			boneB: names[0],
			b: deltoid,
			radius: measure(shoulder, deltoid, 0.09, 0.6)
		});
		const clavicle = `clavicle.${side}`;
		if (has(clavicle) && has('spine01')) {
			// Over the shoulder girdle: from the base of the neck to the shoulder joint.
			const inner: Vec3 = [0, at(clavicle)[1], at(clavicle)[2]];
			world.push({
				boneA: 'spine01',
				a: inner,
				boneB: clavicle,
				b: shoulder,
				radius: measure(inner, shoulder, 0.085, 0.3)
			});
		}
	}

	const capsules = world.slice(0, MAX_CAPSULES).map((c) => {
		const boneA = rig.bone(c.boneA);
		const boneB = rig.bone(c.boneB);
		return {
			boneA,
			a: transform(invertRigid(bind[boneA]), c.a),
			boneB,
			b: transform(invertRigid(bind[boneB]), c.b),
			radius: c.radius
		};
	});

	const spine = ['spine03', 'spine02', 'spine01'].filter(has);
	let torso: Torso | null = null;
	if (spine.length && Number.isFinite(shoulderHeight)) {
		const chest = rig.bone(spine[spine.length - 1]);
		// Up to the base of the neck: the field's top rows are the tops of the shoulders.
		const top = has('neck01') ? at('neck01')[2] : shoulderHeight + 0.05;
		const bottom = at(spine[0])[2] - 0.05;
		torso = {
			bone: chest,
			bind: bind[chest],
			...measureTorso(vertices, top, bottom, arms)
		};
	}
	return { capsules, torso };
}

/**
 * `(MAX_CAPSULES * 8)` the capsules in world space for bones' world transforms `boneWorld`:
 * per capsule end a and radius, then end b and an unused 0.
 */
export function placeCapsules(capsules: Capsule[], boneWorld: Mat4[], out: Float32Array): void {
	out.fill(0);
	capsules.forEach((c, i) => {
		out.set(transform(boneWorld[c.boneA], c.a), 8 * i);
		out[8 * i + 3] = c.radius;
		out.set(transform(boneWorld[c.boneB], c.b), 8 * i + 4);
	});
}

/** The transform from world to the torso field's frame (the bind pose's world) for a pose. */
export function torsoFrame(torso: Torso, boneWorld: Mat4[]): Mat4 {
	return multiply4(torso.bind, invertRigid(boneWorld[torso.bone]));
}

/**
 * Push point `p` (bind-pose world, metres) out of the body as it stands in the bind pose, by
 * `margin`; returns whether it moved. Capsules are given in the bind pose's world frame here.
 */
export function pushOutOfBody(
	p: Vec3,
	capsules: { a: Vec3; b: Vec3; radius: number }[],
	torso: Torso | null,
	margin: number
): boolean {
	let moved = false;
	for (const { a, b, radius } of capsules) {
		const { distance, closest } = segmentDistance(p, a, b);
		if (distance < radius + margin && distance > 1e-6) {
			for (let i = 0; i < 3; i++) {
				p[i] = closest[i] + ((p[i] - closest[i]) * (radius + margin)) / distance;
			}
			moved = true;
		}
	}
	if (torso) {
		const [x, y, top, bottom] = torso.shape;
		if (p[2] < top && p[2] > bottom) {
			const dx = p[0] - x;
			const dy = p[1] - y;
			const r = Math.hypot(dx, dy);
			const angle = Math.atan2(dx, dy);
			const surface = torsoRadius(torso, angle, p[2]);
			if (surface > 0 && r < surface + margin && r > 1e-6) {
				// Out along the surface's normal: sideways off the back, up off the shoulders.
				const step = 0.02;
				const dAngle =
					(torsoRadius(torso, angle + step, p[2]) - torsoRadius(torso, angle - step, p[2])) /
					(2 * step * surface);
				const dz =
					(torsoRadius(torso, angle, p[2] + 0.01) - torsoRadius(torso, angle, p[2] - 0.01)) / 0.02;
				const radial: Vec3 = [dx / r, dy / r, 0];
				const around: Vec3 = [dy / r, -dx / r, 0];
				const n = normalize3([radial[0] - dAngle * around[0], radial[1] - dAngle * around[1], -dz]);
				const depth =
					(surface + margin - r) * n[0] * radial[0] + (surface + margin - r) * n[1] * radial[1];
				for (let i = 0; i < 3; i++) p[i] += n[i] * depth;
				moved = true;
			}
		}
	}
	return moved;
}

/** The torso's distance from its axis at `angle` (radians from the back) and height `z`. */
function torsoRadius(torso: Torso, angle: number, z: number): number {
	const [, , top, bottom] = torso.shape;
	const u = (angle / (2 * Math.PI) + 0.5) * TORSO_ANGLES - 0.5;
	const v = Math.min(
		Math.max(((top - z) / (top - bottom)) * TORSO_HEIGHTS - 0.5, 0),
		TORSO_HEIGHTS - 1
	);
	const i = Math.floor(u);
	const j = Math.min(Math.floor(v), TORSO_HEIGHTS - 2);
	const fu = u - i;
	const fv = Math.min(v - j, 1);
	const at = (a: number, b: number) =>
		torso.field[b * TORSO_ANGLES + (((a % TORSO_ANGLES) + TORSO_ANGLES) % TORSO_ANGLES)];
	return (
		(1 - fv) * ((1 - fu) * at(i, j) + fu * at(i + 1, j)) +
		fv * ((1 - fu) * at(i, j + 1) + fu * at(i + 1, j + 1))
	);
}

function normalize3(v: Vec3): Vec3 {
	const length = Math.hypot(v[0], v[1], v[2]) || 1;
	return [v[0] / length, v[1] / length, v[2] / length];
}

export function transform(m: Mat4, p: Vec3): Vec3 {
	return [
		m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
		m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
		m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11]
	];
}

/**
 * The radius at which a share of the vertices along segment a-b (within `reach` of it, away from
 * its ends) lie inside: the capsule then sits just under the skin.
 */
function fitRadius(vertices: Float32Array, a: Vec3, b: Vec3, reach: number, share: number): number {
	const distances: number[] = [];
	for (let v = 0; v < vertices.length; v += 3) {
		const p: Vec3 = [vertices[v], vertices[v + 1], vertices[v + 2]];
		const { distance, t } = segmentDistance(p, a, b);
		if (t > 0.15 && t < 0.85 && distance < reach) distances.push(distance);
	}
	if (distances.length < 8) return reach * 0.5;
	distances.sort((x, y) => x - y);
	return distances[Math.floor(share * (distances.length - 1))];
}

/** The torso between heights `top` and `bottom` as a height field, arms left out. */
function measureTorso(
	vertices: Float32Array,
	top: number,
	bottom: number,
	arms: [Vec3, Vec3][]
): { shape: [number, number, number, number]; field: Float32Array } {
	const kept: Vec3[] = [];
	for (let v = 0; v < vertices.length; v += 3) {
		const p: Vec3 = [vertices[v], vertices[v + 1], vertices[v + 2]];
		if (p[2] > top + 0.01 || p[2] < bottom - 0.01 || Math.abs(p[0]) > 0.3) continue;
		if (arms.some(([a, b]) => segmentDistance(p, a, b).distance < 0.065)) continue;
		kept.push(p);
	}
	// The axis: halfway between the back and front at mid height.
	let front = Infinity;
	let back = -Infinity;
	const middle = (top + bottom) / 2;
	for (const p of kept) {
		if (Math.abs(p[2] - middle) < 0.03 && Math.abs(p[0]) < 0.05) {
			front = Math.min(front, p[1]);
			back = Math.max(back, p[1]);
		}
	}
	const x = 0;
	const y = Number.isFinite(front) ? (front + back) / 2 : 0;

	const field = new Float32Array(TORSO_ANGLES * TORSO_HEIGHTS);
	for (const p of kept) {
		const angle = Math.atan2(p[0] - x, p[1] - y);
		const i = Math.min(TORSO_ANGLES - 1, Math.floor((angle / (2 * Math.PI) + 0.5) * TORSO_ANGLES));
		const j = Math.floor(((top - p[2]) / (top - bottom)) * TORSO_HEIGHTS);
		if (j < 0 || j >= TORSO_HEIGHTS) continue;
		const r = Math.hypot(p[0] - x, p[1] - y);
		field[j * TORSO_ANGLES + i] = Math.max(field[j * TORSO_ANGLES + i], r);
	}
	// Fill the bins no vertex fell in (the mesh is coarse, the arms were cut out), then round
	// off single vertices.
	for (let pass = 0; pass < 8; pass++) {
		const source = Float32Array.from(field);
		for (let j = 0; j < TORSO_HEIGHTS; j++) {
			for (let i = 0; i < TORSO_ANGLES; i++) {
				if (source[j * TORSO_ANGLES + i] > 0) continue;
				let sum = 0;
				let n = 0;
				for (const [di, dj] of [
					[-1, 0],
					[1, 0],
					[0, -1],
					[0, 1]
				]) {
					const jj = j + dj;
					if (jj < 0 || jj >= TORSO_HEIGHTS) continue;
					const value = source[jj * TORSO_ANGLES + ((i + di + TORSO_ANGLES) % TORSO_ANGLES)];
					if (value > 0) {
						sum += value;
						n++;
					}
				}
				if (n) field[j * TORSO_ANGLES + i] = sum / n;
			}
		}
	}
	const smooth = Float32Array.from(field);
	for (let j = 0; j < TORSO_HEIGHTS; j++) {
		for (let i = 0; i < TORSO_ANGLES; i++) {
			let peak = 0;
			for (let di = -1; di <= 1; di++) {
				for (let dj = -1; dj <= 1; dj++) {
					const jj = Math.min(TORSO_HEIGHTS - 1, Math.max(0, j + dj));
					peak = Math.max(
						peak,
						field[jj * TORSO_ANGLES + ((i + di + TORSO_ANGLES) % TORSO_ANGLES)]
					);
				}
			}
			smooth[j * TORSO_ANGLES + i] = 0.5 * (peak + field[j * TORSO_ANGLES + i]);
		}
	}
	return { shape: [x, y, top, bottom], field: smooth };
}

function segmentDistance(
	p: Vec3,
	a: Vec3,
	b: Vec3
): { distance: number; t: number; closest: Vec3 } {
	const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
	const ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
	const length2 = ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2 || 1;
	const t = (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / length2;
	const c = Math.min(Math.max(t, 0), 1);
	const closest: Vec3 = [a[0] + c * ab[0], a[1] + c * ab[1], a[2] + c * ab[2]];
	return {
		distance: Math.hypot(p[0] - closest[0], p[1] - closest[1], p[2] - closest[2]),
		t,
		closest
	};
}

/**
 * GLSL: `collideBody(p, margin, hit)` pushes world point `p` out of the capsules and the torso by
 * `margin`; `hit` is set if it moved.
 */
export const BODY_GLSL = /* glsl */ `
uniform vec4 capsules[${2 * MAX_CAPSULES}];
uniform int capsuleCount;
uniform sampler2D torsoField;
/** World to the torso field's frame, and back. */
uniform mat4 torsoFrame;
uniform mat4 torsoFrameInverse;
/** The axis's x and y, the field's top and bottom z; top below bottom when there's no torso. */
uniform vec4 torsoShape;

vec3 collideBody(vec3 p, float margin, inout bool hit) {
	for (int i = 0; i < ${MAX_CAPSULES}; i++) {
		if (i >= capsuleCount) break;
		vec4 a = capsules[2 * i];
		vec3 b = capsules[2 * i + 1].xyz;
		vec3 ab = b - a.xyz;
		float t = clamp(dot(p - a.xyz, ab) / max(dot(ab, ab), 1e-8), 0.0, 1.0);
		vec3 closest = a.xyz + t * ab;
		vec3 away = p - closest;
		float distance = length(away);
		float reach = a.w + margin;
		if (distance < reach) {
			p = closest + (distance > 1e-6 ? away / distance : vec3(0.0, 0.0, 1.0)) * reach;
			hit = true;
		}
	}
	vec3 q = (torsoFrame * vec4(p, 1.0)).xyz;
	if (q.z < torsoShape.z && q.z > torsoShape.w) {
		vec2 d = q.xy - torsoShape.xy;
		float r = length(d);
		float height = torsoShape.z - torsoShape.w;
		vec2 uv = vec2(atan(d.x, d.y) / 6.2831853 + 0.5, (torsoShape.z - q.z) / height);
		float surface = texture(torsoField, uv).r;
		if (surface > 0.0 && r < surface + margin && r > 1e-6) {
			// Out along the surface's normal: sideways off the back, up off the shoulders.
			vec2 du = vec2(0.02 / 6.2831853, 0.0);
			vec2 dv = vec2(0.0, 0.01 / height);
			float dAngle = (texture(torsoField, uv + du).r - texture(torsoField, uv - du).r) / (0.04 * surface);
			// v grows downwards.
			float dz = (texture(torsoField, uv - dv).r - texture(torsoField, uv + dv).r) / 0.02;
			vec2 radial = d / r;
			vec2 around = vec2(radial.y, -radial.x);
			vec3 n = normalize(vec3(radial - dAngle * around, -dz));
			q += n * (surface + margin - r) * dot(n.xy, radial);
			p = (torsoFrameInverse * vec4(q, 1.0)).xyz;
			hit = true;
		}
	}
	return p;
}
`;
