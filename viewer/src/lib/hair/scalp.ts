/**
 * The bald head the hair grows from and lies on, in the head bone's frame (x to the avatar's
 * left, y up, z forward, metres): a centre inside the cranium and the head's outer surface as a
 * radius per direction from it, stored in an octahedral map. Hair collides with that surface as
 * with a star-shaped body, which follows the head bone exactly whatever the pose.
 *
 * Built from posed vertices alone: the head is the outermost surface within reach of the
 * cranium, so the mouth, teeth and eyes inside it don't matter.
 */

import { invertRigid, type Mat4, type Vec3 } from '#lib/math.ts';

/** Texels per side of the octahedral radius map. */
export const FIELD_SIZE = 128;
/** Below this elevation (sin) the map is empty: the neck is a capsule instead. */
const LOWEST = -0.82;

export class Scalp {
	constructor(
		/** Cranium centre in the head bone's frame. */
		readonly centre: Vec3,
		/** `(FIELD_SIZE * FIELD_SIZE)` head radius in metres per octahedral texel, 0 where unset. */
		readonly field: Float32Array
	) {}

	/** Head radius along unit direction `d` (bilinear), 0 below the jaw. */
	radius(d: Vec3): number {
		const [u, v] = octEncode(d);
		const x = Math.min(Math.max(u * FIELD_SIZE - 0.5, 0), FIELD_SIZE - 1.001);
		const y = Math.min(Math.max(v * FIELD_SIZE - 0.5, 0), FIELD_SIZE - 1.001);
		const i = Math.floor(x);
		const j = Math.floor(y);
		const fx = x - i;
		const fy = y - j;
		const f = this.field;
		const at = (a: number, b: number) => f[b * FIELD_SIZE + a];
		return (
			(1 - fy) * ((1 - fx) * at(i, j) + fx * at(i + 1, j)) +
			fy * ((1 - fx) * at(i, j + 1) + fx * at(i + 1, j + 1))
		);
	}

	/** The surface point along unit direction `d`, lifted by `clearance`. */
	surface(d: Vec3, clearance = 0): Vec3 {
		const r = this.radius(d) + clearance;
		const c = this.centre;
		return [c[0] + r * d[0], c[1] + r * d[1], c[2] + r * d[2]];
	}

	/** Outward surface normal along unit direction `d`, from the radius map's slope. */
	normal(d: Vec3): Vec3 {
		const r = this.radius(d);
		// Two tangents, and the surface's change along each.
		const a = normalize(cross(Math.abs(d[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0], d));
		const b = cross(d, a);
		const step = 0.03;
		const ra = this.radius(normalize(add(d, scale(a, step))));
		const rb = this.radius(normalize(add(d, scale(b, step))));
		const ta = sub(scale(normalize(add(d, scale(a, step))), ra), scale(d, r));
		const tb = sub(scale(normalize(add(d, scale(b, step))), rb), scale(d, r));
		const n = normalize(cross(ta, tb));
		return dot(n, d) < 0 ? scale(n, -1) : n;
	}

	/**
	 * Push head-local point `p` out of the head to `clearance` above its surface; returns
	 * whether it moved.
	 */
	pushOut(p: Vec3, clearance: number): boolean {
		const q = sub(p, this.centre);
		const r = Math.hypot(q[0], q[1], q[2]);
		if (r < 1e-6) return false;
		const d = scale(q, 1 / r);
		const surface = this.radius(d);
		if (surface <= 0 || r >= surface + clearance) return false;
		const out = this.surface(d, clearance);
		p[0] = out[0];
		p[1] = out[1];
		p[2] = out[2];
		return true;
	}
}

/**
 * The scalp of an avatar posed with `vertices` (`(V * 3)` world positions), its head bone's
 * world transform `head` (row-major).
 */
export function buildScalp(vertices: Float32Array, head: Mat4): Scalp {
	const toLocal = invertRigid(head);
	const count = vertices.length / 3;
	const local = new Float32Array(vertices.length);
	let top = -Infinity;
	for (let v = 0; v < count; v++) {
		const x = vertices[3 * v];
		const y = vertices[3 * v + 1];
		const z = vertices[3 * v + 2];
		for (let i = 0; i < 3; i++) {
			local[3 * v + i] =
				toLocal[4 * i] * x + toLocal[4 * i + 1] * y + toLocal[4 * i + 2] * z + toLocal[4 * i + 3];
		}
		if (Math.hypot(local[3 * v], local[3 * v + 2]) < 0.15) top = Math.max(top, local[3 * v + 1]);
	}

	// The cranium: a sphere fitted to the top 6 cm of the head.
	const cap: number[] = [];
	for (let v = 0; v < count; v++) {
		if (local[3 * v + 1] > top - 0.06 && Math.hypot(local[3 * v], local[3 * v + 2]) < 0.15) {
			cap.push(local[3 * v], local[3 * v + 1], local[3 * v + 2]);
		}
	}
	const centre = fitSphere(cap);

	// The outermost vertex per texel; the jaw and neck below it are left to the capsules.
	const size = FIELD_SIZE;
	const field = new Float32Array(size * size);
	for (let v = 0; v < count; v++) {
		const q: Vec3 = [
			local[3 * v] - centre[0],
			local[3 * v + 1] - centre[1],
			local[3 * v + 2] - centre[2]
		];
		const r = Math.hypot(q[0], q[1], q[2]);
		if (r > 0.2 || r < 0.03 || local[3 * v + 1] < centre[1] - 0.16) continue;
		const [u, w] = octEncode(scale(q, 1 / r));
		const i = Math.min(size - 1, Math.floor(u * size));
		const j = Math.min(size - 1, Math.floor(w * size));
		field[j * size + i] = Math.max(field[j * size + i], r);
	}
	// Fill texels no vertex fell in from their neighbours, then round off the bumps a single
	// vertex makes: a dilation keeps hair out of the gaps, a blur smooths what it lies on.
	for (let pass = 0; pass < 6; pass++) fillHoles(field, size);
	const dilated = filter(field, size, (values) => Math.max(...values));
	const smooth = filter(dilated, size, (values) => values.reduce((a, b) => a + b) / values.length);
	for (let j = 0; j < size; j++) {
		for (let i = 0; i < size; i++) {
			const d = octDecode((i + 0.5) / size, (j + 0.5) / size);
			if (d[1] < LOWEST) smooth[j * size + i] = 0;
		}
	}
	return new Scalp(centre, smooth);
}

/**
 * Octahedral map coordinates in [0, 1]² of unit direction `d`, with the head's up (y) at the
 * map's centre so the crown, where most hair grows, is far from the folds.
 */
export function octEncode(d: Vec3): [number, number] {
	// Map axes: a = x, b = z, up = y.
	const sum = Math.abs(d[0]) + Math.abs(d[1]) + Math.abs(d[2]);
	let a = d[0] / sum;
	let b = d[2] / sum;
	if (d[1] < 0) {
		const fa = (1 - Math.abs(b)) * (a >= 0 ? 1 : -1);
		const fb = (1 - Math.abs(a)) * (b >= 0 ? 1 : -1);
		a = fa;
		b = fb;
	}
	return [a * 0.5 + 0.5, b * 0.5 + 0.5];
}

export function octDecode(u: number, v: number): Vec3 {
	let a = u * 2 - 1;
	let b = v * 2 - 1;
	const up = 1 - Math.abs(a) - Math.abs(b);
	if (up < 0) {
		const fa = (1 - Math.abs(b)) * (a >= 0 ? 1 : -1);
		const fb = (1 - Math.abs(a)) * (b >= 0 ? 1 : -1);
		a = fa;
		b = fb;
	}
	return normalize([a, up, b]);
}

/** GLSL of `octEncode`, for the shaders that collide with the head. */
export const OCT_ENCODE_GLSL = /* glsl */ `
vec2 octEncode(vec3 d) {
	vec2 m = d.xz / (abs(d.x) + abs(d.y) + abs(d.z));
	if (d.y < 0.0) m = (1.0 - abs(m.yx)) * vec2(m.x >= 0.0 ? 1.0 : -1.0, m.y >= 0.0 ? 1.0 : -1.0);
	return m * 0.5 + 0.5;
}
`;

function fillHoles(field: Float32Array, size: number): void {
	const source = Float32Array.from(field);
	for (let j = 0; j < size; j++) {
		for (let i = 0; i < size; i++) {
			if (source[j * size + i] > 0) continue;
			let sum = 0;
			let n = 0;
			for (let dj = -1; dj <= 1; dj++) {
				for (let di = -1; di <= 1; di++) {
					const value = source[clamp(j + dj, size) * size + clamp(i + di, size)];
					if (value > 0) {
						sum += value;
						n++;
					}
				}
			}
			if (n > 0) field[j * size + i] = sum / n;
		}
	}
}

function filter(
	field: Float32Array,
	size: number,
	reduce: (values: number[]) => number
): Float32Array {
	const out = new Float32Array(field.length);
	const values: number[] = [];
	for (let j = 0; j < size; j++) {
		for (let i = 0; i < size; i++) {
			values.length = 0;
			for (let dj = -1; dj <= 1; dj++) {
				for (let di = -1; di <= 1; di++) {
					const value = field[clamp(j + dj, size) * size + clamp(i + di, size)];
					if (value > 0) values.push(value);
				}
			}
			out[j * size + i] = values.length ? reduce(values) : 0;
		}
	}
	return out;
}

function clamp(i: number, size: number): number {
	return Math.min(Math.max(i, 0), size - 1);
}

/** Least-squares sphere through `(N * 3)` points; returns its centre. */
function fitSphere(points: number[]): Vec3 {
	// |p|² = 2 c·p + k, linear in (c, k).
	const a = Array.from({ length: 4 }, () => new Float64Array(5));
	for (let i = 0; i < points.length; i += 3) {
		const row = [2 * points[i], 2 * points[i + 1], 2 * points[i + 2], 1];
		const rhs = points[i] ** 2 + points[i + 1] ** 2 + points[i + 2] ** 2;
		for (let r = 0; r < 4; r++) {
			for (let c = 0; c < 4; c++) a[r][c] += row[r] * row[c];
			a[r][4] += row[r] * rhs;
		}
	}
	for (let i = 0; i < 4; i++) {
		let pivot = i;
		for (let k = i + 1; k < 4; k++) if (Math.abs(a[k][i]) > Math.abs(a[pivot][i])) pivot = k;
		[a[i], a[pivot]] = [a[pivot], a[i]];
		for (let k = 0; k < 4; k++) {
			if (k === i) continue;
			const f = a[k][i] / a[i][i];
			for (let c = i; c < 5; c++) a[k][c] -= f * a[i][c];
		}
	}
	return [a[0][4] / a[0][0], a[1][4] / a[1][1], a[2][4] / a[2][2]];
}

export function add(a: Vec3, b: Vec3): Vec3 {
	return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function sub(a: Vec3, b: Vec3): Vec3 {
	return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function scale(a: Vec3, s: number): Vec3 {
	return [a[0] * s, a[1] * s, a[2] * s];
}

export function dot(a: Vec3, b: Vec3): number {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function cross(a: Vec3, b: Vec3): Vec3 {
	return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function normalize(v: Vec3): Vec3 {
	const length = Math.hypot(v[0], v[1], v[2]) || 1;
	return [v[0] / length, v[1] / length, v[2] / length];
}
