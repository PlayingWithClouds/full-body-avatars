/**
 * Motion clips: `.motion` files as written by `avatar_splats.motion.clip.write_motion`, and
 * sampling them at any time.
 *
 * A clip holds, per person and frame, each bone's rotation relative to its bind pose (wxyz
 * quaternions), the root's offset from its bind position (metres, world axes) and the face's
 * expression weights. Bones and expressions are matched to the avatar's by name.
 */

import { array, readContainer, type AvatarArray } from '#lib/avatar/file.ts';
import type { Rig } from '#lib/avatar/rig.ts';
import type { Vec3 } from '#lib/math.ts';

const MAGIC = 'MOTION01';
const SUPPORTED_VERSION = 1;

export interface MotionClip {
	name: string;
	fps: number;
	frames: number;
	people: number;
	boneNames: string[];
	expressionNames: string[];
	/** Vertical field of view of the video's camera. */
	fovYDegrees: number;
	/** `(P, N, J, 4)` wxyz rotations relative to the bind pose. */
	rotations: Float32Array;
	/** `(P, N, 3)` the root's offset from its bind position. */
	rootTranslation: Float32Array;
	/** `(P, N, E)` expression weights. */
	expressions: Float32Array;
	/** `(4, 4)` row-major: where the video's camera was, OpenCV axes. */
	cameraToWorld: Float32Array;
}

export function duration(clip: MotionClip): number {
	return (clip.frames - 1) / clip.fps;
}

export function readMotion(buffer: ArrayBuffer, name: string): MotionClip {
	const { metadata, arrays } = readContainer(buffer, MAGIC, 'motion');
	if (metadata.format_version !== SUPPORTED_VERSION) {
		throw new Error(
			`motion format ${metadata.format_version}, this viewer reads ${SUPPORTED_VERSION}`
		);
	}
	const file = { metadata: metadata as never, arrays };
	const floats = (key: string) => (array(file, key) as AvatarArray<Float32Array>).data;
	return {
		name,
		fps: metadata.fps as number,
		frames: metadata.frames as number,
		people: metadata.people as number,
		boneNames: metadata.bones as string[],
		expressionNames: metadata.expressions as string[],
		fovYDegrees: metadata.fov_y_degrees as number,
		rotations: floats('rotations'),
		rootTranslation: floats('root_translation'),
		expressions: floats('expressions'),
		cameraToWorld: floats('camera_to_world')
	};
}

/**
 * One person's pose in the avatar's order: what animations blend before it is turned into
 * rotation matrices.
 */
export interface Pose {
	/** `(J * 4)` wxyz rotations relative to each bone's bind pose. */
	rotations: Float64Array;
	rootOffset: Vec3;
	expressions: Float32Array;
}

export function bindPose(rig: Rig): Pose {
	const rotations = new Float64Array(4 * rig.boneNames.length);
	for (let j = 0; j < rig.boneNames.length; j++) rotations[4 * j] = 1;
	return {
		rotations,
		rootOffset: [0, 0, 0],
		expressions: new Float32Array(rig.expressionNames.length)
	};
}

/** Samples a clip in the avatar's bone and expression order. */
export class ClipSampler {
	/** For each of the avatar's bones, the clip's bone of the same name, or -1. */
	private readonly bones: Int32Array;
	private readonly expressions: Int32Array;

	constructor(
		readonly clip: MotionClip,
		private readonly rig: Rig
	) {
		this.bones = Int32Array.from(rig.boneNames, (name) => clip.boneNames.indexOf(name));
		this.expressions = Int32Array.from(rig.expressionNames, (name) =>
			clip.expressionNames.indexOf(name)
		);
	}

	/** One person's pose at `time` seconds, interpolated between frames, into `out`. */
	sample(person: number, time: number, out: Pose = bindPose(this.rig)): Pose {
		const { clip } = this;
		const position = Math.min(Math.max(time * clip.fps, 0), clip.frames - 1);
		const first = Math.floor(position);
		const second = Math.min(first + 1, clip.frames - 1);
		const t = position - first;
		const bones = clip.boneNames.length;
		const a = (person * clip.frames + first) * bones;
		const b = (person * clip.frames + second) * bones;
		for (let j = 0; j < this.bones.length; j++) {
			const k = this.bones[j];
			if (k < 0) {
				out.rotations.set([1, 0, 0, 0], 4 * j);
				continue;
			}
			nlerp(clip.rotations, 4 * (a + k), clip.rotations, 4 * (b + k), t, out.rotations, 4 * j);
		}
		const ra = 3 * (person * clip.frames + first);
		const rb = 3 * (person * clip.frames + second);
		for (let i = 0; i < 3; i++) {
			out.rootOffset[i] =
				clip.rootTranslation[ra + i] +
				t * (clip.rootTranslation[rb + i] - clip.rootTranslation[ra + i]);
		}
		const names = clip.expressionNames.length;
		const ea = (person * clip.frames + first) * names;
		const eb = (person * clip.frames + second) * names;
		for (let e = 0; e < this.expressions.length; e++) {
			const k = this.expressions[e];
			out.expressions[e] =
				k < 0
					? 0
					: clip.expressions[ea + k] + t * (clip.expressions[eb + k] - clip.expressions[ea + k]);
		}
		return out;
	}
}

/**
 * Normalized interpolation between two quaternions along the shorter way, written to `out` at
 * `o`. Close enough to slerp for neighbouring frames and short blends.
 */
export function nlerp(
	a: ArrayLike<number>,
	i: number,
	b: ArrayLike<number>,
	k: number,
	t: number,
	out: Float64Array,
	o: number
): void {
	const sign =
		a[i] * b[k] + a[i + 1] * b[k + 1] + a[i + 2] * b[k + 2] + a[i + 3] * b[k + 3] < 0 ? -1 : 1;
	let w = a[i] + t * (sign * b[k] - a[i]);
	let x = a[i + 1] + t * (sign * b[k + 1] - a[i + 1]);
	let y = a[i + 2] + t * (sign * b[k + 2] - a[i + 2]);
	let z = a[i + 3] + t * (sign * b[k + 3] - a[i + 3]);
	const length = Math.hypot(w, x, y, z) || 1;
	w /= length;
	x /= length;
	y /= length;
	z /= length;
	out[o] = w;
	out[o + 1] = x;
	out[o + 2] = y;
	out[o + 3] = z;
}

/** Row-major 3x3 rotation of a wxyz quaternion at `q[i]`. */
export function quaternionMatrix(q: ArrayLike<number>, i: number): Float64Array {
	const w = q[i];
	const x = q[i + 1];
	const y = q[i + 2];
	const z = q[i + 3];
	return Float64Array.of(
		1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
		2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
		2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)
	); // prettier-ignore
}

/** A row-major 3x3 rotation as a wxyz quaternion, written to `out` at `o`. */
export function matrixToQuaternion(m: ArrayLike<number>, out: Float64Array, o: number): void {
	const trace = m[0] + m[4] + m[8];
	let w: number, x: number, y: number, z: number;
	if (trace > 0) {
		const s = 2 * Math.sqrt(trace + 1);
		w = s / 4;
		x = (m[7] - m[5]) / s;
		y = (m[2] - m[6]) / s;
		z = (m[3] - m[1]) / s;
	} else if (m[0] > m[4] && m[0] > m[8]) {
		const s = 2 * Math.sqrt(1 + m[0] - m[4] - m[8]);
		w = (m[7] - m[5]) / s;
		x = s / 4;
		y = (m[1] + m[3]) / s;
		z = (m[2] + m[6]) / s;
	} else if (m[4] > m[8]) {
		const s = 2 * Math.sqrt(1 + m[4] - m[0] - m[8]);
		w = (m[2] - m[6]) / s;
		x = (m[1] + m[3]) / s;
		y = s / 4;
		z = (m[5] + m[7]) / s;
	} else {
		const s = 2 * Math.sqrt(1 + m[8] - m[0] - m[4]);
		w = (m[3] - m[1]) / s;
		x = (m[2] + m[6]) / s;
		y = (m[5] + m[7]) / s;
		z = s / 4;
	}
	out[o] = w;
	out[o + 1] = x;
	out[o + 2] = y;
	out[o + 3] = z;
}
