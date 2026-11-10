/**
 * Eyelashes and eyebrows: strands bound to the face's triangles (`export/face_hair.py`), so
 * they close with the lids and rise with the brows. Each strand's points are stored in its
 * triangle's frame (first edge, across, normal) in metres from its root; every frame they are
 * placed on the posed triangle and drawn as camera-facing ribbons that taper to a point.
 *
 * A lash is a tenth of a millimetre across, under a pixel at most distances: a ribbon is drawn
 * at least a pixel wide and covers it by the share the real hair would, so lashes fade with
 * distance instead of flickering.
 */

import * as THREE from 'three';
import { array, type AvatarFile } from '#lib/avatar/file.ts';
import { LIGHTING } from '#lib/render/lighting.ts';
import type { PosedAvatar, SceneLayer } from '#lib/render/viewer.ts';

export class FaceHair implements SceneLayer {
	readonly object = new THREE.Group();
	private readonly faces: Uint32Array;
	private readonly triangles: Uint32Array;
	private readonly barycentric: Float32Array;
	private readonly points: Float32Array;
	private readonly count: number;
	private readonly perStrand: number;
	private readonly geometry: THREE.BufferGeometry;
	private readonly material: THREE.ShaderMaterial;
	private readonly copies: { mesh: THREE.Mesh; buffer: THREE.InterleavedBuffer }[] = [];
	/** One strand's posed points, `(P * 3)`. */
	private readonly strand: Float32Array;

	constructor(file: AvatarFile) {
		const metadata = file.metadata.face_hair!;
		this.faces = array<Uint32Array>(file, 'hair_faces').data;
		this.triangles = array<Uint32Array>(file, 'faces').data;
		this.barycentric = array<Float32Array>(file, 'hair_barycentric').data;
		this.points = array<Float32Array>(file, 'hair_points').data;
		this.count = this.faces.length;
		this.perStrand = metadata.points_per_strand;
		this.strand = new Float32Array(3 * this.perStrand);
		const widths = array<Float32Array>(file, 'hair_widths').data;
		const groups = array<Uint8Array>(file, 'hair_groups').data;

		// Two vertices per point, one each side of the strand; two triangles per segment.
		const P = this.perStrand;
		const vertices = 2 * P * this.count;
		const side = new Float32Array(vertices);
		const along = new Float32Array(vertices);
		const width = new Float32Array(vertices);
		const colour = new Float32Array(3 * vertices);
		const index: number[] = [];
		for (let s = 0; s < this.count; s++) {
			const rgb = metadata.groups[groups[s]].colour;
			for (let k = 0; k < P; k++) {
				for (let e = 0; e < 2; e++) {
					const v = 2 * (s * P + k) + e;
					side[v] = e === 0 ? -1 : 1;
					along[v] = k / (P - 1);
					width[v] = widths[s];
					colour.set(rgb, 3 * v);
				}
				if (k < P - 1) {
					const a = 2 * (s * P + k);
					index.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
				}
			}
		}
		this.geometry = new THREE.BufferGeometry();
		this.geometry.setAttribute('side', new THREE.BufferAttribute(side, 1));
		this.geometry.setAttribute('along', new THREE.BufferAttribute(along, 1));
		this.geometry.setAttribute('width', new THREE.BufferAttribute(width, 1));
		this.geometry.setAttribute('colour', new THREE.BufferAttribute(colour, 3));
		this.geometry.setIndex(index);

		this.material = new THREE.ShaderMaterial({
			uniforms: {
				viewportHeight: { value: 1 },
				keyDirection: { value: LIGHTING.key }
			},
			vertexShader: VERTEX,
			fragmentShader: FRAGMENT,
			transparent: true,
			depthWrite: false,
			side: THREE.DoubleSide
		});
	}

	update(_seconds: number, avatars: readonly PosedAvatar[]): boolean {
		let changed = this.ensureCopies(avatars.length);
		avatars.forEach((avatar, copy) => {
			changed = this.place(avatar.vertices, this.copies[copy].buffer) || changed;
		});
		return changed;
	}

	dispose(): void {
		this.geometry.dispose();
		this.material.dispose();
	}

	private ensureCopies(count: number): boolean {
		if (count === this.copies.length) return false;
		for (const { mesh } of this.copies.splice(count)) {
			this.object.remove(mesh);
			mesh.geometry.dispose();
		}
		while (this.copies.length < count) {
			const geometry = new THREE.BufferGeometry();
			for (const name of ['side', 'along', 'width', 'colour']) {
				geometry.setAttribute(name, this.geometry.getAttribute(name));
			}
			geometry.setIndex(this.geometry.getIndex());
			// Each point's position, and the strand's direction there, both written per frame.
			const buffer = new THREE.InterleavedBuffer(
				new Float32Array(6 * 2 * this.count * this.perStrand),
				6
			);
			buffer.setUsage(THREE.DynamicDrawUsage);
			geometry.setAttribute('position', new THREE.InterleavedBufferAttribute(buffer, 3, 0));
			geometry.setAttribute('tangent', new THREE.InterleavedBufferAttribute(buffer, 3, 3));
			const mesh = new THREE.Mesh(geometry, this.material);
			mesh.frustumCulled = false;
			mesh.renderOrder = 2; // after the Gaussians, over the skin
			mesh.onBeforeRender = (renderer) => {
				this.material.uniforms.viewportHeight.value = renderer.getDrawingBufferSize(SIZE).y;
			};
			this.copies.push({ mesh, buffer });
			this.object.add(mesh);
		}
		return true;
	}

	/** Place the strands on the posed triangles; return whether they moved. */
	private place(posed: Float32Array, buffer: THREE.InterleavedBuffer): boolean {
		const data = buffer.array as Float32Array;
		const P = this.perStrand;
		const strand = this.strand;
		const [a, b, c, x, y, z, root] = Array.from({ length: 7 }, () => new THREE.Vector3());
		let moved = false;
		for (let s = 0; s < this.count; s++) {
			const f = this.faces[s];
			a.fromArray(posed, 3 * this.triangles[3 * f]);
			b.fromArray(posed, 3 * this.triangles[3 * f + 1]);
			c.fromArray(posed, 3 * this.triangles[3 * f + 2]);
			// The triangle's frame: first edge, across, normal.
			x.subVectors(b, a);
			z.crossVectors(x, y.subVectors(c, a)).normalize();
			x.normalize();
			y.crossVectors(z, x);
			const weights = this.barycentric.subarray(3 * s, 3 * s + 3);
			root.set(0, 0, 0);
			root.addScaledVector(a, weights[0]).addScaledVector(b, weights[1]);
			root.addScaledVector(c, weights[2]);
			for (let k = 0; k < P; k++) {
				const o = 3 * (s * P + k);
				for (let d = 0; d < 3; d++) {
					strand[3 * k + d] =
						root.getComponent(d) +
						x.getComponent(d) * this.points[o] +
						y.getComponent(d) * this.points[o + 1] +
						z.getComponent(d) * this.points[o + 2];
				}
			}
			for (let k = 0; k < P; k++) {
				// The direction along the strand: to the next point, the last one's from the one before.
				const from = Math.min(k, P - 2);
				let tx = strand[3 * from + 3] - strand[3 * from];
				let ty = strand[3 * from + 4] - strand[3 * from + 1];
				let tz = strand[3 * from + 5] - strand[3 * from + 2];
				const length = Math.hypot(tx, ty, tz) || 1;
				tx /= length;
				ty /= length;
				tz /= length;
				for (let e = 0; e < 2; e++) {
					const at = 6 * (2 * (s * P + k) + e);
					moved ||= data[at] !== strand[3 * k] || data[at + 1] !== strand[3 * k + 1];
					data.set([strand[3 * k], strand[3 * k + 1], strand[3 * k + 2], tx, ty, tz], at);
				}
			}
		}
		if (moved) buffer.needsUpdate = true;
		return moved;
	}
}

const SIZE = new THREE.Vector2();

const VERTEX = /* glsl */ `
attribute vec3 tangent;
attribute float side;
attribute float along;
attribute float width;
attribute vec3 colour;
uniform float viewportHeight;
varying float vCoverage;
varying float vAlong;
varying vec3 vColour;
varying vec3 vTangent;
varying vec3 vWorld;

void main() {
	vec3 view = normalize(cameraPosition - position);
	vec3 across = normalize(cross(tangent, view));
	vec4 clip = projectionMatrix * viewMatrix * vec4(position, 1.0);
	float pixel = 2.0 * clip.w / (projectionMatrix[1][1] * viewportHeight);
	float real = width * (1.0 - 0.9 * along);
	float drawn = max(real, 1.2 * pixel);
	vCoverage = real / drawn;
	vAlong = along;
	vColour = colour;
	vTangent = tangent;
	vWorld = position + across * side * 0.5 * drawn;
	gl_Position = projectionMatrix * viewMatrix * vec4(vWorld, 1.0);
}
`;

const FRAGMENT = /* glsl */ `
uniform vec3 keyDirection;
varying float vCoverage;
varying float vAlong;
varying vec3 vColour;
varying vec3 vTangent;
varying vec3 vWorld;

void main() {
	// Lit as hair is, along its length (Kajiya-Kay): bright where the key light falls across it,
	// a sheen where it would reflect it towards the eye.
	vec3 t = normalize(vTangent);
	vec3 view = normalize(cameraPosition - vWorld);
	float across = sqrt(max(1.0 - pow(dot(t, keyDirection), 2.0), 0.0));
	vec3 half_ = normalize(keyDirection + view);
	float sheen = pow(sqrt(max(1.0 - pow(dot(t, half_), 2.0), 0.0)), 80.0);
	vec3 colour = vColour * (0.75 + 0.35 * across) + 0.06 * sheen;
	gl_FragColor = vec4(colour, vCoverage * (1.0 - 0.3 * vAlong));
}
`;
