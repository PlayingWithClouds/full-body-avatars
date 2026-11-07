/**
 * Posing the avatar's mesh, as `avatar_splats.export.runtime` does: expression blendshapes on
 * the head's rest vertices, forward kinematics of bind-relative bone rotations, then linear blend
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
	private readonly posed: Float32Array;

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

	/** `(J)` skinning matrices for bind-relative rotations: world transform times inverse bind pose. */
	skinning(rotations: Float64Array[]): Mat4[] {
		const world: Mat4[] = [];
		for (let j = 0; j < this.boneNames.length; j++) {
			const local = identity4();
			for (let a = 0; a < 3; a++) {
				for (let b = 0; b < 3; b++) local[4 * a + b] = rotations[j][3 * a + b];
			}
			const parent = this.parents[j];
			const base = parent < 0 ? this.relative[j] : multiply4(world[parent], this.relative[j]);
			world.push(multiply4(base, local));
		}
		return world.map((w, j) => multiply4(w, this.inverseBindPoses[j]));
	}

	/** `(V * 3)` posed vertices; the returned array is reused by the next call. */
	pose(rotations: Float64Array[], expressions: Float32Array): Float32Array {
		const skinning = this.skinning(rotations);
		const rest = this.rest;
		const basis = this.expressionBasis;
		const headValues = rest.length - 3 * this.headStart;
		const out = this.posed;
		const k = this.influences;

		const shaped = Float32Array.from(rest);
		for (let e = 0; e < expressions.length; e++) {
			const weight = expressions[e];
			if (weight === 0) continue;
			const offset = e * headValues;
			const head = 3 * this.headStart;
			for (let i = 0; i < headValues; i++) shaped[head + i] += weight * basis[offset + i];
		}

		for (let v = 0; v < this.vertexCount; v++) {
			const x = shaped[3 * v];
			const y = shaped[3 * v + 1];
			const z = shaped[3 * v + 2];
			let px = 0;
			let py = 0;
			let pz = 0;
			for (let i = 0; i < k; i++) {
				const w = this.boneWeights[k * v + i];
				if (w === 0) continue;
				const m = skinning[this.boneIndices[k * v + i]];
				px += w * (m[0] * x + m[1] * y + m[2] * z + m[3]);
				py += w * (m[4] * x + m[5] * y + m[6] * z + m[7]);
				pz += w * (m[8] * x + m[9] * y + m[10] * z + m[11]);
			}
			out[3 * v] = px;
			out[3 * v + 1] = py;
			out[3 * v + 2] = pz;
		}
		return out;
	}
}
