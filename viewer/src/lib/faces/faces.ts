/**
 * The faces of a video as `avatar-splats faces` converts them (`outputs/faces/<video>.faces.json`):
 * per frame, the avatar's expression weights, the head's rotation and the eyes' turn. This file
 * reads them and places a posed head on a grid cell; `layer.ts` draws the heads.
 */

import { rotvecToMatrix, type Mat3, type Vec3 } from '#lib/math.ts';

export interface FaceFrame {
	/** The frame's index in the video. */
	index: number;
	/** Whether a face was found: elsewhere the face is held by its neighbours. */
	found: boolean;
	/** `(E)` weights in the avatar's expression order. */
	expressions: Float32Array;
	/** The head's rotation vector in world axes, about the head's centre. */
	head: Vec3;
	/** The eyeballs' rotation vectors in world axes: the one on the avatar's +x side, then -x. */
	eyes: [Vec3, Vec3];
}

/** The video's crops around the head: an image of square tiles, row by row in frame order. */
export interface Crops {
	/** The atlas image's file name, next to the faces file. */
	file: string;
	/** Columns and rows of tiles. */
	columns: number;
	rows: number;
}

export interface Faces {
	video: string;
	fps: number;
	frames: FaceFrame[];
	crops: Crops | null;
}

const FORMAT_VERSION = 1;

function isVec3(value: unknown): value is Vec3 {
	return Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);
}

/**
 * Read a faces document, resolving each frame's expression names against the avatar's.
 *
 * @throws when the document is not a faces file, or names an expression the avatar lacks.
 */
export function parseFaces(document: unknown, expressionNames: string[]): Faces {
	const data = document as {
		format_version?: number;
		video?: string;
		fps?: number;
		frames?: Record<string, unknown>[];
		crops?: Crops;
	};
	if (data?.format_version !== FORMAT_VERSION || !Array.isArray(data.frames)) {
		throw new Error('not a faces file (format_version 1 with frames)');
	}
	const index = new Map(expressionNames.map((name, i) => [name, i]));
	const frames = data.frames.map((frame, i): FaceFrame => {
		const eyes = frame.eyes as unknown[];
		if (!isVec3(frame.head) || !Array.isArray(eyes) || !eyes.every(isVec3) || eyes.length !== 2) {
			throw new Error(`frames[${i}]: head and eyes must be rotation vectors`);
		}
		const expressions = new Float32Array(expressionNames.length);
		for (const [name, weight] of Object.entries(frame.expressions as Record<string, number>)) {
			const at = index.get(name);
			if (at === undefined) throw new Error(`frames[${i}]: the avatar has no expression ${name}`);
			expressions[at] = weight;
		}
		return {
			index: Number(frame.index),
			found: frame.found !== false,
			expressions,
			head: frame.head,
			eyes: eyes as [Vec3, Vec3]
		};
	});
	return {
		video: String(data.video ?? ''),
		fps: Number(data.fps ?? 0),
		frames,
		crops: data.crops ?? null
	};
}

/** The `[u0, v0, u1, v1]` of tile `i` in an atlas of `columns` by `rows` tiles (v up). */
export function tileUvs(i: number, { columns, rows }: Crops): [number, number, number, number] {
	const column = i % columns;
	const row = Math.floor(i / columns);
	return [column / columns, 1 - (row + 1) / rows, (column + 1) / columns, 1 - row / rows];
}

/** Columns and rows of a grid of `count` cells that is about `aspect` times wider than tall. */
export function gridShape(count: number, aspect = 16 / 9): { columns: number; rows: number } {
	const columns = Math.max(1, Math.ceil(Math.sqrt(count * aspect)));
	return { columns, rows: Math.max(1, Math.ceil(count / columns)) };
}

/** Where a head is posed: how to turn and move the vertices of the avatar's head. */
export interface HeadPlacement {
	/** The vertices that make up the head; the rest collapse onto `centre`. */
	vertices: Uint32Array;
	/** The point of the rest head it turns about. */
	pivot: Vec3;
	/** Where the pivot goes. */
	centre: Vec3;
	/** Row-major 3x3 head rotation. */
	rotation: Mat3;
	/** Per eyeball: its vertices, its centre in the rest pose and its row-major 3x3 turn. */
	eyes: { vertices: Uint32Array; centre: Vec3; rotation: Mat3 }[];
}

/**
 * Write the head of `posed` (`(V * 3)` vertices of the avatar in the rest pose, with an
 * expression) turned and moved as `placement` says into `out`. The eyeballs turn about their
 * centres first, then the whole head about its pivot. Vertices that are not the head's all go to
 * `placement.centre`, so triangles that touch them draw as nothing.
 */
export function placeHead(posed: Float32Array, out: Float32Array, placement: HeadPlacement): void {
	const { pivot, centre, rotation: r } = placement;
	for (let v = 0; v < out.length; v += 3) {
		out[v] = centre[0];
		out[v + 1] = centre[1];
		out[v + 2] = centre[2];
	}
	const eyeOf = new Map<number, number>();
	placement.eyes.forEach((eye, e) => eye.vertices.forEach((v) => eyeOf.set(v, e)));
	for (const v of placement.vertices) {
		let x = posed[3 * v];
		let y = posed[3 * v + 1];
		let z = posed[3 * v + 2];
		const e = eyeOf.get(v);
		if (e !== undefined) {
			const { centre: c, rotation: g } = placement.eyes[e];
			const [dx, dy, dz] = [x - c[0], y - c[1], z - c[2]];
			x = c[0] + g[0] * dx + g[1] * dy + g[2] * dz;
			y = c[1] + g[3] * dx + g[4] * dy + g[5] * dz;
			z = c[2] + g[6] * dx + g[7] * dy + g[8] * dz;
		}
		const [dx, dy, dz] = [x - pivot[0], y - pivot[1], z - pivot[2]];
		out[3 * v] = centre[0] + r[0] * dx + r[1] * dy + r[2] * dz;
		out[3 * v + 1] = centre[1] + r[3] * dx + r[4] * dy + r[5] * dz;
		out[3 * v + 2] = centre[2] + r[6] * dx + r[7] * dy + r[8] * dz;
	}
}

/** Row-major 3x3 turn of a rotation vector, or none when `enabled` is false. */
export function turn(rotvec: Vec3, enabled = true): Mat3 {
	return rotvecToMatrix(enabled ? rotvec : [0, 0, 0]);
}
