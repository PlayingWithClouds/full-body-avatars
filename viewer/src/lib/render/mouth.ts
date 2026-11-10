/**
 * The inside of the mouth: teeth, gums, tongue and the mouth's socket, drawn with stock
 * materials (`export/mouth.py`) instead of the baked texture, which the videos, seeing the inside
 * of the mouth rarely and badly, fill with bits of lips. The surfaces darken with depth behind
 * the front teeth, as little light reaches there; the teeth are rounded off by how squarely they
 * face the eye, and teeth, gums and tongue are wet enough to catch the key light.
 */

import * as THREE from 'three';
import type { AvatarFile, MouthMetadata } from '#lib/avatar/file.ts';
import { LIGHTING } from '#lib/render/lighting.ts';

/** From how far behind the front teeth, in metres, the inside of the mouth darkens, to how far
 * it is darkest, and how dark. */
const DARKENING = { from: 0.01, to: 0.06, darkest: 0.45 };

/** Smooth normals of the mouth's surfaces, recomputed as the mesh is posed. */
export class MouthNormals {
	/** The mouth's triangles, `(M * 3)` vertex indices. */
	private readonly faces: Uint32Array;
	private readonly sums: Float32Array;

	constructor(file: AvatarFile, mouthFaces: number[]) {
		const all = file.arrays.faces.data as Uint32Array;
		this.faces = new Uint32Array(3 * mouthFaces.length);
		mouthFaces.forEach((f, i) => this.faces.set(all.subarray(3 * f, 3 * f + 3), 3 * i));
		this.sums = new Float32Array(file.arrays.rest_vertices.data.length);
	}

	/** Write the posed normal of each corner of the mouth's triangles, in their order. */
	compute(posed: Float32Array, out: Float32Array): void {
		const sums = this.sums;
		const faces = this.faces;
		for (const v of faces) sums[3 * v] = sums[3 * v + 1] = sums[3 * v + 2] = 0;
		for (let i = 0; i < faces.length; i += 3) {
			const a = 3 * faces[i];
			const b = 3 * faces[i + 1];
			const c = 3 * faces[i + 2];
			const ux = posed[b] - posed[a];
			const uy = posed[b + 1] - posed[a + 1];
			const uz = posed[b + 2] - posed[a + 2];
			const vx = posed[c] - posed[a];
			const vy = posed[c + 1] - posed[a + 1];
			const vz = posed[c + 2] - posed[a + 2];
			// The cross product, weighted by the triangle's area.
			const nx = uy * vz - uz * vy;
			const ny = uz * vx - ux * vz;
			const nz = ux * vy - uy * vx;
			for (const v of [a, b, c]) {
				sums[v] += nx;
				sums[v + 1] += ny;
				sums[v + 2] += nz;
			}
		}
		for (let i = 0; i < faces.length; i++) {
			const v = 3 * faces[i];
			const length = Math.hypot(sums[v], sums[v + 1], sums[v + 2]) || 1;
			out[3 * i] = sums[v] / length;
			out[3 * i + 1] = sums[v + 1] / length;
			out[3 * i + 2] = sums[v + 2] / length;
		}
	}
}

/**
 * Per corner of the mouth's triangles: its region's number and its depth behind the front
 * teeth in the rest pose, in metres (the face looks along -y).
 */
export function mouthAttributes(
	file: AvatarFile,
	mouthFaces: number[]
): { region: Float32Array; depth: Float32Array } {
	const faces = file.arrays.faces.data as Uint32Array;
	const rest = file.arrays.rest_vertices.data as Float32Array;
	const regions = file.arrays.mouth_regions.data as Uint8Array;
	const teeth = file.metadata.mouth!.regions.indexOf('teeth');
	let front = Infinity;
	for (const f of mouthFaces) {
		if (regions[f] !== teeth) continue;
		for (let c = 0; c < 3; c++) front = Math.min(front, rest[3 * faces[3 * f + c] + 1]);
	}
	const region = new Float32Array(3 * mouthFaces.length);
	const depth = new Float32Array(3 * mouthFaces.length);
	mouthFaces.forEach((f, i) => {
		for (let c = 0; c < 3; c++) {
			region[3 * i + c] = regions[f];
			depth[3 * i + c] = Math.max(0, rest[3 * faces[3 * f + c] + 1] - front);
		}
	});
	return { region, depth };
}

export function mouthMaterial(mouth: MouthMetadata): THREE.ShaderMaterial {
	const colour = (name: string) => new THREE.Vector3(...mouth.colours[name]);
	const number = (name: string) => mouth.regions.indexOf(name).toFixed(1);
	return new THREE.ShaderMaterial({
		uniforms: {
			teethColour: { value: colour('teeth') },
			gumsColour: { value: colour('gums') },
			tongueColour: { value: colour('tongue') },
			socketColour: { value: colour('socket') },
			innerLipsColour: { value: colour('inner_lips') },
			keyDirection: { value: LIGHTING.key }
		},
		vertexShader: VERTEX,
		fragmentShader: FRAGMENT.replace('TEETH', number('teeth'))
			.replace('GUMS', number('gums'))
			.replace('TONGUE', number('tongue'))
			.replace('INNER_LIPS', number('inner_lips')),
		side: THREE.DoubleSide
	});
}

const VERTEX = /* glsl */ `
attribute float region;
attribute float depth;
varying vec3 vWorld;
varying vec3 vNormal;
varying float vRegion;
varying float vDepth;

void main() {
	vec4 world = modelMatrix * vec4(position, 1.0);
	vWorld = world.xyz;
	vNormal = normal;
	vRegion = region;
	vDepth = depth;
	gl_Position = projectionMatrix * viewMatrix * world;
}
`;

const FRAGMENT = /* glsl */ `
uniform vec3 teethColour;
uniform vec3 gumsColour;
uniform vec3 tongueColour;
uniform vec3 socketColour;
uniform vec3 innerLipsColour;
uniform vec3 keyDirection;
varying vec3 vWorld;
varying vec3 vNormal;
varying float vRegion;
varying float vDepth;

float hash(vec3 p) {
	return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453);
}

void main() {
	vec3 view = normalize(cameraPosition - vWorld);
	vec3 normal = normalize(vNormal);
	normal = dot(normal, view) < 0.0 ? -normal : normal;
	float facing = max(dot(normal, view), 0.0);
	float lit = 0.8 + 0.2 * max(dot(normal, keyDirection), 0.0);
	float wet = pow(max(dot(normal, normalize(keyDirection + view)), 0.0), 60.0);

	vec3 colour;
	float shine;
	if (abs(vRegion - TEETH) < 0.5) {
		// Rounded off at their sides, where they turn away.
		colour = teethColour * mix(0.72, 1.04, sqrt(facing));
		shine = 0.3;
	} else if (abs(vRegion - GUMS) < 0.5) {
		colour = gumsColour * mix(0.85, 1.0, facing);
		shine = 0.12;
	} else if (abs(vRegion - TONGUE) < 0.5) {
		// Papillae: a faint mottle a couple of millimetres across.
		colour = tongueColour * (0.95 + 0.05 * hash(floor(vWorld * 600.0)));
		shine = 0.1;
	} else if (abs(vRegion - INNER_LIPS) < 0.5) {
		colour = innerLipsColour;
		shine = 0.15;
	} else {
		colour = socketColour;
		shine = 0.04;
	}
	float shade = mix(1.0, ${DARKENING.darkest.toFixed(3)}, smoothstep(${DARKENING.from.toFixed(3)}, ${DARKENING.to.toFixed(3)}, vDepth));
	gl_FragColor = vec4(min(colour * lit * shade + shine * wet * shade, vec3(1.0)), 1.0);
}
`;
