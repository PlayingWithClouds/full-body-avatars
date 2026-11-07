/**
 * The little linear algebra the viewer needs. Matrices are row-major arrays, as in the avatar
 * file and NumPy: element (row, column) of a 4x4 matrix is at `4 * row + column`, and points are
 * column vectors, `p' = M p`.
 */

export type Mat3 = Float64Array; // 9 elements
export type Mat4 = Float64Array; // 16 elements
export type Vec3 = [number, number, number];

export function identity4(): Mat4 {
	const m = new Float64Array(16);
	m[0] = m[5] = m[10] = m[15] = 1;
	return m;
}

export function multiply4(a: Mat4, b: Mat4, out: Mat4 = new Float64Array(16)): Mat4 {
	for (let i = 0; i < 4; i++) {
		for (let j = 0; j < 4; j++) {
			let sum = 0;
			for (let k = 0; k < 4; k++) sum += a[4 * i + k] * b[4 * k + j];
			out[4 * i + j] = sum;
		}
	}
	return out;
}

/** Inverse of a rigid transform (rotation and translation). */
export function invertRigid(m: Mat4): Mat4 {
	const out = identity4();
	for (let i = 0; i < 3; i++) {
		for (let j = 0; j < 3; j++) out[4 * i + j] = m[4 * j + i];
	}
	for (let i = 0; i < 3; i++) {
		out[4 * i + 3] = -(out[4 * i] * m[3] + out[4 * i + 1] * m[7] + out[4 * i + 2] * m[11]);
	}
	return out;
}

/** Rotation matrix of a rotation vector (axis times angle in radians). */
export function rotvecToMatrix(v: Vec3): Mat3 {
	const angle = Math.hypot(v[0], v[1], v[2]);
	const m = new Float64Array(9);
	if (angle < 1e-12) {
		m[0] = m[4] = m[8] = 1;
		return m;
	}
	const [x, y, z] = v.map((c) => c / angle);
	const c = Math.cos(angle);
	const s = Math.sin(angle);
	const t = 1 - c;
	m.set([
		t * x * x + c, t * x * y - s * z, t * x * z + s * y,
		t * x * y + s * z, t * y * y + c, t * y * z - s * x,
		t * x * z - s * y, t * y * z + s * x, t * z * z + c
	]); // prettier-ignore
	return m;
}

export function cross(a: Vec3, b: Vec3): Vec3 {
	return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function normalize(v: Vec3): Vec3 {
	const length = Math.hypot(v[0], v[1], v[2]) || 1;
	return [v[0] / length, v[1] / length, v[2] / length];
}
