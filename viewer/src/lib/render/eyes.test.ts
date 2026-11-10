import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';
import type { AvatarFile, EyeMetadata } from '#lib/avatar/file.ts';
import { EyeballRig, eyeFrames } from '#lib/render/eyes.ts';

/** Two octahedral eyeballs 12 mm across, at x = ±3 cm, looking along -y, and a triangle of skin. */
function twoEyes(): { file: AvatarFile; eyes: EyeMetadata } {
	const r = 0.012;
	const corners = [
		[r, 0, 0],
		[-r, 0, 0],
		[0, r, 0],
		[0, -r, 0],
		[0, 0, r],
		[0, 0, -r]
	];
	const octahedron = [
		[0, 3, 4],
		[3, 1, 4],
		[1, 2, 4],
		[2, 0, 4],
		[3, 0, 5],
		[1, 3, 5],
		[2, 1, 5],
		[0, 2, 5]
	];
	const centres = [
		[0.03, 0, 1.6],
		[-0.03, 0, 1.6]
	];
	const vertices = centres.flatMap(([x, y, z]) =>
		corners.map(([a, b, c]) => [x + a, y + b, z + c])
	);
	vertices.push([0, 0, 0], [1, 0, 0], [0, 1, 0]);
	const faces = [0, 6].flatMap((start) => octahedron.map((face) => face.map((v) => v + start)));
	faces.push([12, 13, 14]);
	const eyes: EyeMetadata = {
		iris_colour: [0.4, 0.45, 0.4],
		sclera_colour: [0.9, 0.9, 0.9],
		iris_angle: 0.4,
		eyeballs: centres.map((centre, eye) => ({
			side: eye === 0 ? 'left' : 'right',
			centre: centre as [number, number, number],
			radius: r,
			apex: 6 * eye + 3,
			top: 6 * eye + 4
		}))
	};
	const file: AvatarFile = {
		metadata: { parts: ['body', 'head_skin', 'mouth', 'eyes'], eyes } as AvatarFile['metadata'],
		arrays: {
			rest_vertices: { data: Float32Array.from(vertices.flat()), shape: [vertices.length, 3] },
			faces: { data: Uint32Array.from(faces.flat()), shape: [faces.length, 3] },
			face_parts: {
				data: Uint8Array.from(faces.map((_, f) => (f < 16 ? 3 : 0))),
				shape: [faces.length]
			}
		}
	};
	return { file, eyes };
}

describe('eyeballs', () => {
	test('point where their posed vertices say', () => {
		const { file, eyes } = twoEyes();
		const rig = new EyeballRig(file, eyes);
		const posed = Float32Array.from(file.arrays.rest_vertices.data as Float32Array);

		// Turn the first eyeball 30 degrees to its left (about +z) and lift the whole head.
		const turn = new THREE.Matrix4().makeRotationZ(Math.PI / 6);
		const centre = new THREE.Vector3(0.03, 0, 1.6);
		const point = new THREE.Vector3();
		for (let v = 0; v < 6; v++) {
			point
				.fromArray(posed, 3 * v)
				.sub(centre)
				.applyMatrix4(turn)
				.add(centre)
				.toArray(posed, 3 * v);
		}
		for (let v = 0; v < 12; v++) posed[3 * v + 2] += 0.1;

		const frames = eyeFrames();
		rig.pose(posed, frames);
		expect(frames.centres[0].distanceTo(new THREE.Vector3(0.03, 0, 1.7))).toBeLessThan(1e-5);
		const turned = new THREE.Vector3(0, -1, 0).applyMatrix4(turn);
		expect(frames.forwards[0].distanceTo(turned)).toBeLessThan(1e-5);
		expect(frames.forwards[1].distanceTo(new THREE.Vector3(0, -1, 0))).toBeLessThan(1e-5);
		expect(frames.ups[1].distanceTo(new THREE.Vector3(0, 0, 1))).toBeLessThan(1e-5);
	});
});
