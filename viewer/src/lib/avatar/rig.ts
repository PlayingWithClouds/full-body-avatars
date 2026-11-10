/**
 * Posing the avatar's mesh, as `avatar_splats.export.runtime` does: expression blendshapes on
 * the head's rest vertices, body shapes weighted by the pose on the body's (format 5: the breasts'
 * posture correctives), forward kinematics of bind-relative bone rotations, then linear blend
 * skinning.
 */

import { array, type AvatarFile } from '#lib/avatar/file.ts';
import {
	identity4,
	invertRigid,
	multiply4,
	rotvecToMatrix,
	type Mat4,
	type Vec3
} from '#lib/math.ts';

/** Gravity in the chest's axes when the avatar stands upright, as `regions/gravity.py`'s DOWN. */
const UPRIGHT: Vec3 = [0, 0, -1];

/**
 * The posture correctives' weights (forward, back, left, right, lift; `regions/gravity.py`) for
 * unit gravity in the chest bone's axes.
 */
export function gravityWeights(gravity: Vec3): number[] {
	const dx = gravity[0] - UPRIGHT[0];
	const dy = gravity[1] - UPRIGHT[1];
	const dz = gravity[2] - UPRIGHT[2];
	return [Math.max(-dy, 0), Math.max(dy, 0), Math.max(dx, 0), Math.max(-dx, 0), dz];
}

/**
 * Gravity (`down`, in the rest pose's axes) in the axes of a bone whose skinning matrix (row-major
 * 4x4, or its top three rows) is `m`: its rotation transposed, applied to `down`.
 */
export function gravityInBone(m: ArrayLike<number>, down: Vec3): Vec3 {
	const g: Vec3 = [0, 0, 0];
	for (let a = 0; a < 3; a++) {
		g[a] = m[a] * down[0] + m[4 + a] * down[1] + m[8 + a] * down[2];
	}
	const length = Math.hypot(...g) || 1;
	return [g[0] / length, g[1] / length, g[2] / length];
}

export class Rig {
	readonly boneNames: string[];
	readonly expressionNames: string[];
	readonly vertexCount: number;
	private readonly rest: Float32Array;
	private readonly boneIndices: Uint16Array;
	private readonly boneWeights: Float32Array;
	private readonly influences: number;
	private readonly parents: Int32Array;
	private readonly bindPoses: Mat4[];
	private readonly inverseBindPoses: Mat4[];
	/** Each bone's bind pose relative to its parent's (its own for the root). */
	private readonly relative: Mat4[];
	private readonly headStart: number;
	private readonly expressionBasis: Float32Array;
	/** `(S * headStart * 3)` the body shapes, or none before format 5. */
	private readonly bodyShapeBasis: Float32Array | null;
	private readonly bodyShapeBone: number;
	private readonly bodyShapeDown: Vec3;
	/** The body shapes' weights in the last pose. */
	bodyShapeWeights: number[] = [];
	private readonly posed: Float32Array;
	/** The rest vertices with the current expressions, reused between poses. */
	private readonly shaped: Float32Array;
	/** `(J * 12)` the skinning matrices' top three rows, reused between poses. */
	private readonly matrices: Float64Array;
	/** `(J)` each bone's world transform in the last pose, root offset included. */
	boneWorld: Mat4[];

	constructor(file: AvatarFile) {
		this.boneNames = file.metadata.bones;
		this.expressionNames = file.metadata.expressions;
		this.headStart = file.metadata.head_start;
		this.rest = array<Float32Array>(file, 'rest_vertices').data;
		this.vertexCount = this.rest.length / 3;
		const indices = array<Uint16Array>(file, 'bone_indices');
		this.boneIndices = indices.data;
		this.influences = indices.shape[1];
		this.boneWeights = array<Float32Array>(file, 'bone_weights').data;
		this.parents = array<Int32Array>(file, 'bone_parents').data;
		this.expressionBasis = array<Float32Array>(file, 'expression_basis').data;
		const shapes = file.metadata.body_shapes;
		if (shapes && shapes.driver.kind !== 'gravity') {
			throw new Error(`unknown body shape driver ${shapes.driver.kind}`);
		}
		this.bodyShapeBasis = shapes ? array<Float32Array>(file, 'body_shape_basis').data : null;
		this.bodyShapeBone = shapes ? this.boneNames.indexOf(shapes.driver.bone) : -1;
		this.bodyShapeDown = shapes ? shapes.driver.down : [0, 0, -1];

		const bind = array<Float32Array>(file, 'bind_poses').data;
		this.bindPoses = this.boneNames.map((_, j) =>
			Float64Array.from(bind.subarray(16 * j, 16 * j + 16))
		);
		this.inverseBindPoses = this.bindPoses.map(invertRigid);
		this.relative = this.bindPoses.map((pose, j) => {
			const parent = this.parents[j];
			return parent < 0 ? pose : multiply4(this.inverseBindPoses[parent], pose);
		});
		this.posed = new Float32Array(this.rest.length);
		this.shaped = Float32Array.from(this.rest);
		this.matrices = new Float64Array(12 * this.boneNames.length);
		this.boneWorld = this.bindPoses.map((pose) => Float64Array.from(pose));
	}

	/** `(4, 4)` row-major: bone `j`'s frame in the bind pose. */
	bindPose(j: number): Mat4 {
		return Float64Array.from(this.bindPoses[j]);
	}

	/** How far the root must rise from its bind position for the rest pose to stand on z = 0. */
	standingHeight(): number {
		let lowest = Infinity;
		for (let v = 2; v < this.rest.length; v += 3) lowest = Math.min(lowest, this.rest[v]);
		return -lowest;
	}

	/** Index of the bone called `name`. */
	bone(name: string): number {
		const index = this.boneNames.indexOf(name);
		if (index < 0) throw new Error(`no bone ${name}`);
		return index;
	}

	/**
	 * Bind-relative rotations, one 3x3 per bone, from rotation vectors in world axes by bone name
	 * (x to the avatar's left, y to its back, z up); bones not named keep their bind pose.
	 */
	bindRelative(worldRotvecs: Record<string, Vec3>): Float64Array[] {
		return this.boneNames.map((name, j) => {
			const rotvec = worldRotvecs[name];
			const out = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
			if (!rotvec) return out;
			const r = rotvecToMatrix(rotvec);
			const f = this.bindPoses[j];
			// frames^T R frames, with frames the bind pose's rotation.
			for (let a = 0; a < 3; a++) {
				for (let b = 0; b < 3; b++) {
					let sum = 0;
					for (let i = 0; i < 3; i++) {
						for (let k = 0; k < 3; k++) sum += f[4 * i + a] * r[3 * i + k] * f[4 * k + b];
					}
					out[3 * a + b] = sum;
				}
			}
			return out;
		});
	}

	/**
	 * `(J)` bones' world transforms for bind-relative rotations, the root moved by `rootOffset`
	 * from its bind position (in world axes), as `export/runtime.py`'s `forward_kinematics`.
	 */
	world(rotations: Float64Array[], rootOffset: Vec3 = [0, 0, 0]): Mat4[] {
		const world: Mat4[] = [];
		for (let j = 0; j < this.boneNames.length; j++) {
			const local = identity4();
			for (let a = 0; a < 3; a++) {
				for (let b = 0; b < 3; b++) local[4 * a + b] = rotations[j][3 * a + b];
			}
			const parent = this.parents[j];
			const base = parent < 0 ? this.relative[j] : multiply4(world[parent], this.relative[j]);
			const transform = multiply4(base, local);
			if (parent < 0) for (let i = 0; i < 3; i++) transform[4 * i + 3] += rootOffset[i];
			world.push(transform);
		}
		return world;
	}

	/** `(J)` skinning matrices: world transform times inverse bind pose. */
	skinning(world: Mat4[]): Mat4[] {
		return world.map((w, j) => multiply4(w, this.inverseBindPoses[j]));
	}

	/**
	 * The body's rest vertices plus the body shapes weighted for this pose (`matrices`: the
	 * skinning matrices' top rows), into `shaped`; nothing without body shapes.
	 */
	private shapeBody(matrices: Float64Array): void {
		const basis = this.bodyShapeBasis;
		if (!basis) return;
		const body = 3 * this.headStart;
		const shaped = this.shaped;
		shaped.set(this.rest.subarray(0, body), 0);
		const chest = matrices.subarray(12 * this.bodyShapeBone, 12 * this.bodyShapeBone + 12);
		this.bodyShapeWeights = gravityWeights(gravityInBone(chest, this.bodyShapeDown));
		this.bodyShapeWeights.forEach((weight, s) => {
			if (Math.abs(weight) < 1e-3) return;
			const offset = s * body;
			for (let i = 0; i < body; i++) shaped[i] += weight * basis[offset + i];
		});
	}

	/**
	 * `(V * 3)` posed vertices for bind-relative rotations, expression weights and the root's
	 * offset; the returned array is reused by the next call. Also sets `boneWorld`.
	 */
	pose(
		rotations: Float64Array[],
		expressions: Float32Array,
		rootOffset: Vec3 = [0, 0, 0]
	): Float32Array {
		this.boneWorld = this.world(rotations, rootOffset);
		// The top three rows of every skinning matrix, side by side, for the loop below.
		const matrices = this.matrices;
		this.skinning(this.boneWorld).forEach((m, j) => matrices.set(m.subarray(0, 12), 12 * j));

		// Only the head's vertices change with expressions: start them from the rest pose and add
		// each expression that shows (weights below a thousandth move nothing visibly).
		const shaped = this.shaped;
		const head = 3 * this.headStart;
		const headValues = shaped.length - head;
		shaped.set(this.rest.subarray(head), head);
		this.shapeBody(matrices);
		const basis = this.expressionBasis;
		for (let e = 0; e < expressions.length; e++) {
			const weight = expressions[e];
			if (Math.abs(weight) < 1e-3) continue;
			const offset = e * headValues;
			for (let i = 0; i < headValues; i++) shaped[head + i] += weight * basis[offset + i];
		}

		const out = this.posed;
		const k = this.influences;
		const indices = this.boneIndices;
		const weights = this.boneWeights;
		for (let v = 0; v < this.vertexCount; v++) {
			const x = shaped[3 * v];
			const y = shaped[3 * v + 1];
			const z = shaped[3 * v + 2];
			let px = 0;
			let py = 0;
			let pz = 0;
			for (let i = k * v; i < k * v + k; i++) {
				const w = weights[i];
				if (w === 0) continue;
				const m = 12 * indices[i];
				px += w * (matrices[m] * x + matrices[m + 1] * y + matrices[m + 2] * z + matrices[m + 3]);
				py +=
					w * (matrices[m + 4] * x + matrices[m + 5] * y + matrices[m + 6] * z + matrices[m + 7]);
				pz +=
					w * (matrices[m + 8] * x + matrices[m + 9] * y + matrices[m + 10] * z + matrices[m + 11]);
			}
			out[3 * v] = px;
			out[3 * v + 1] = py;
			out[3 * v + 2] = pz;
		}
		return out;
	}
}
