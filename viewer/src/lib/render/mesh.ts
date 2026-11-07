/**
 * The avatar's textured mesh: unlit, its colours baked from the videos. One material per texture
 * (body, head); the triangles are grouped by the texture they use.
 */

import * as THREE from 'three';
import { array, type AvatarFile } from '#lib/avatar/file.ts';

export class TexturedMesh {
	readonly object: THREE.Mesh;
	private readonly positions: THREE.BufferAttribute;
	/** The vertex at each corner, in the grouped order. */
	private readonly cornerVertices: Uint32Array;

	constructor(file: AvatarFile) {
		const faces = array<Uint32Array>(file, 'faces').data;
		const uvs = array<Float32Array>(file, 'uvs').data;
		const textureOfFace = array<Uint8Array>(file, 'face_textures').data;
		const faceCount = faces.length / 3;
		const names = file.metadata.textures;

		const order = Array.from({ length: faceCount }, (_, f) => f).sort(
			(a, b) => textureOfFace[a] - textureOfFace[b]
		);
		this.cornerVertices = new Uint32Array(3 * faceCount);
		const cornerUvs = new Float32Array(6 * faceCount);
		order.forEach((face, slot) => {
			for (let c = 0; c < 3; c++) {
				this.cornerVertices[3 * slot + c] = faces[3 * face + c];
				cornerUvs[6 * slot + 2 * c] = uvs[6 * face + 2 * c];
				cornerUvs[6 * slot + 2 * c + 1] = uvs[6 * face + 2 * c + 1];
			}
		});

		const geometry = new THREE.BufferGeometry();
		this.positions = new THREE.BufferAttribute(new Float32Array(9 * faceCount), 3);
		this.positions.setUsage(THREE.DynamicDrawUsage);
		geometry.setAttribute('position', this.positions);
		geometry.setAttribute('uv', new THREE.BufferAttribute(cornerUvs, 2));
		let start = 0;
		names.forEach((_, texture) => {
			const used = order.filter((f) => textureOfFace[f] === texture).length;
			geometry.addGroup(3 * start, 3 * used, texture);
			start += used;
		});

		const materials = names.map(
			(name) =>
				new THREE.MeshBasicMaterial({
					map: texture(array<Uint8Array>(file, name)),
					side: THREE.FrontSide
				})
		);
		this.object = new THREE.Mesh(geometry, materials);
		this.object.frustumCulled = false;
	}

	/** Move the mesh to ``(V * 3)`` posed vertices. */
	setVertices(posed: Float32Array): void {
		const data = this.positions.array as Float32Array;
		const corners = this.cornerVertices;
		for (let i = 0; i < corners.length; i++) {
			const v = 3 * corners[i];
			data[3 * i] = posed[v];
			data[3 * i + 1] = posed[v + 1];
			data[3 * i + 2] = posed[v + 2];
		}
		this.positions.needsUpdate = true;
		this.object.geometry.computeBoundingSphere();
	}

	dispose(): void {
		this.object.geometry.dispose();
		for (const material of this.object.material as THREE.MeshBasicMaterial[]) {
			material.map?.dispose();
			material.dispose();
		}
	}
}

/** A texture from the file's ``(H, W, 3)`` 8-bit rows, top row first (v = 1). */
function texture({ data, shape }: { data: Uint8Array; shape: number[] }): THREE.DataTexture {
	const [height, width] = shape;
	const rgba = new Uint8Array(4 * width * height);
	for (let row = 0; row < height; row++) {
		// three.js puts a data texture's first row at v = 0, the bottom.
		const source = (height - 1 - row) * width;
		for (let x = 0; x < width; x++) {
			rgba.set(data.subarray(3 * (source + x), 3 * (source + x) + 3), 4 * (row * width + x));
			rgba[4 * (row * width + x) + 3] = 255;
		}
	}
	const map = new THREE.DataTexture(rgba, width, height, THREE.RGBAFormat, THREE.UnsignedByteType);
	map.colorSpace = THREE.NoColorSpace;
	map.generateMipmaps = true;
	map.minFilter = THREE.LinearMipmapLinearFilter;
	map.magFilter = THREE.LinearFilter;
	map.anisotropy = 8;
	map.needsUpdate = true;
	return map;
}
