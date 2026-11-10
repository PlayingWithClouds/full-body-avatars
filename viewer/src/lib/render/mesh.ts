/**
 * The avatar's textured mesh. One material per texture (body, head), and the eyes' own
 * (`eyes.ts`) and the mouth's (`mouth.ts`) where the file describes them; the triangles are
 * grouped by material. Copies of the avatar share the materials and texture coordinates
 * (`copy`); only their positions, and so where their eyes look, are their own.
 *
 * Up to format 3 the textures hold the colours baked from the videos and are drawn unlit. From
 * format 4 they may hold albedo instead, lit by the light the file learned, through smooth
 * normals of the posed mesh, and the head's albedo may gain wrinkle maps weighted by each copy's
 * expressions (`skin.ts`).
 */

import * as THREE from 'three';
import { array, type AvatarFile } from '#lib/avatar/file.ts';
import {
	EyeballRig,
	eyeFrames,
	eyeMaterial,
	type EyeFrames,
	type EyeMaterial
} from '#lib/render/eyes.ts';
import { MouthNormals, mouthAttributes, mouthMaterial } from '#lib/render/mouth.ts';
import {
	SkinMaterial,
	cornerNormals,
	wrinkleExpressions,
	wrinkleTexture
} from '#lib/render/skin.ts';

export class TexturedMesh {
	readonly object: THREE.Mesh;
	private readonly positions: THREE.BufferAttribute;
	/** The vertex at each corner, in the grouped order. */
	private readonly cornerVertices: Uint32Array;
	/** Whether this mesh made its materials, rather than sharing another's. */
	private readonly ownsMaterials: boolean;
	private readonly eyeRig: EyeballRig | null;
	private readonly eyes = eyeFrames();
	/** The mouth's normals, and where its corners are in the grouped order. */
	private readonly mouth: { normals: MouthNormals; start: number; count: number } | null;
	/** Every corner's normal, where the skin is lit or the mouth drawn. */
	private readonly normals: THREE.BufferAttribute | null;
	/** Scratch for the skin's vertex normals, `(V * 3)`, where the skin is lit. */
	private readonly normalSums: Float32Array | null;
	/** `(K)` each wrinkle map's expression, and this copy's weights for them. */
	private readonly wrinkleExpressions: Uint32Array | null;
	private readonly wrinkleWeights: Float32Array;

	constructor(file: AvatarFile, original?: TexturedMesh) {
		if (original) {
			const geometry = new THREE.BufferGeometry();
			const corners = original.positions.count;
			this.positions = new THREE.BufferAttribute(new Float32Array(3 * corners), 3);
			this.positions.setUsage(THREE.DynamicDrawUsage);
			geometry.setAttribute('position', this.positions);
			geometry.setAttribute('uv', original.object.geometry.getAttribute('uv'));
			for (const group of original.object.geometry.groups) {
				geometry.addGroup(group.start, group.count, group.materialIndex);
			}
			this.cornerVertices = original.cornerVertices;
			this.eyeRig = original.eyeRig;
			this.mouth = original.mouth;
			this.normalSums = original.normalSums && new Float32Array(original.normalSums.length);
			this.normals = original.normals && normalsAttribute(corners);
			if (this.normals) geometry.setAttribute('normal', this.normals);
			if (this.mouth) {
				for (const name of ['region', 'depth']) {
					geometry.setAttribute(name, original.object.geometry.getAttribute(name));
				}
			}
			this.wrinkleExpressions = original.wrinkleExpressions;
			this.wrinkleWeights = new Float32Array(original.wrinkleWeights.length);
			this.object = new THREE.Mesh(geometry, original.object.material);
			this.object.frustumCulled = false;
			this.ownsMaterials = false;
			this.useOwn();
			return;
		}
		const faces = array<Uint32Array>(file, 'faces').data;
		const uvs = array<Float32Array>(file, 'uvs').data;
		const textureOfFace = array<Uint8Array>(file, 'face_textures').data;
		const faceCount = faces.length / 3;
		const names = file.metadata.textures;
		const eyes = file.metadata.eyes;
		const eyePart = file.metadata.parts?.indexOf('eyes') ?? -1;
		const parts = eyes ? array<Uint8Array>(file, 'face_parts').data : null;
		const mouth = file.metadata.mouth;
		const regions = mouth ? array<Uint8Array>(file, 'mouth_regions').data : null;
		// Materials: one per texture, then the eyes', then the mouth's.
		const eyeMaterialIndex = eyes ? names.length : -1;
		const mouthMaterialIndex = mouth ? names.length + (eyes ? 1 : 0) : -1;
		const materialOf = (f: number) =>
			regions && regions[f] > 0
				? mouthMaterialIndex
				: parts && parts[f] === eyePart
					? eyeMaterialIndex
					: textureOfFace[f];

		const order = Array.from({ length: faceCount }, (_, f) => f).sort(
			(a, b) => materialOf(a) - materialOf(b)
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
		// Files with a learned light hold albedo, lit by it; the head's may have wrinkle maps.
		const { lighting, wrinkles } = file.metadata;
		const head = names.indexOf('head_texture');
		const materials: THREE.Material[] = names.map((name, t) => {
			const map = texture(array<Uint8Array>(file, name));
			const wrinkled =
				wrinkles && t === head
					? {
							metadata: wrinkles,
							maps: wrinkleTexture(array<Uint8Array>(file, 'wrinkle_textures'))
						}
					: null;
			return lighting || wrinkled
				? new SkinMaterial(map, lighting, wrinkled)
				: new THREE.MeshBasicMaterial({ map, side: THREE.FrontSide });
		});
		this.wrinkleExpressions =
			wrinkles && head >= 0
				? wrinkleExpressions(wrinkles.expressions, file.metadata.expressions)
				: null;
		this.wrinkleWeights = new Float32Array(this.wrinkleExpressions?.length ?? 0);
		this.eyeRig = eyes ? new EyeballRig(file, eyes) : null;
		if (eyes && this.eyeRig) materials.push(eyeMaterial(eyes, this.eyeRig));
		if (mouth) materials.push(mouthMaterial(mouth));
		let start = 0;
		materials.forEach((_, material) => {
			const used = order.filter((f) => materialOf(f) === material).length;
			geometry.addGroup(3 * start, 3 * used, material);
			start += used;
		});

		const corners = 3 * faceCount;
		this.mouth = null;
		this.normals = lighting || mouth ? normalsAttribute(corners) : null;
		if (this.normals) geometry.setAttribute('normal', this.normals);
		this.normalSums = lighting ? new Float32Array(array(file, 'rest_vertices').data.length) : null;
		if (mouth) {
			// The mouth's triangles come last in the grouped order, after every other material's.
			const mouthFaces = order.filter((f) => materialOf(f) === mouthMaterialIndex);
			const first = faceCount - mouthFaces.length;
			const { region, depth } = mouthAttributes(file, mouthFaces);
			const regionAll = new Float32Array(corners);
			const depthAll = new Float32Array(corners);
			regionAll.set(region, 3 * first);
			depthAll.set(depth, 3 * first);
			geometry.setAttribute('region', new THREE.BufferAttribute(regionAll, 1));
			geometry.setAttribute('depth', new THREE.BufferAttribute(depthAll, 1));
			this.mouth = {
				normals: new MouthNormals(file, mouthFaces),
				start: 3 * first,
				count: 3 * mouthFaces.length
			};
		}

		this.object = new THREE.Mesh(geometry, materials);
		this.object.frustumCulled = false;
		this.ownsMaterials = true;
		this.useOwn();
	}

	/**
	 * Before each draw of the eyes or the wrinkled skin, point their shared material at this
	 * copy's eyes or wrinkle weights.
	 */
	private useOwn(): void {
		if (!this.eyeRig && !this.wrinkleExpressions) return;
		const own: EyeFrames = this.eyes;
		const weights = this.wrinkleWeights;
		this.object.onBeforeRender = (_renderer, _scene, _camera, _geometry, material) => {
			if (material instanceof SkinMaterial) material.useWrinkles(weights);
			else if ('use' in material) (material as EyeMaterial).use(own);
		};
	}

	/** Light the skin with the file's learned light, or show its albedo unlit; copies follow. */
	setLit(lit: boolean): void {
		for (const material of this.object.material as THREE.Material[]) {
			if (material instanceof SkinMaterial) material.setLit(lit);
		}
	}

	/** Another copy of the avatar's mesh, sharing this one's materials and texture coordinates. */
	copy(file: AvatarFile): TexturedMesh {
		return new TexturedMesh(file, this);
	}

	/**
	 * Move the mesh to ``(V * 3)`` posed vertices, and its wrinkles to the expression weights
	 * they were posed with, if given (otherwise they stay).
	 */
	setVertices(posed: Float32Array, expressions?: Float32Array): void {
		const data = this.positions.array as Float32Array;
		const corners = this.cornerVertices;
		for (let i = 0; i < corners.length; i++) {
			const v = 3 * corners[i];
			data[3 * i] = posed[v];
			data[3 * i + 1] = posed[v + 1];
			data[3 * i + 2] = posed[v + 2];
		}
		this.positions.needsUpdate = true;
		this.eyeRig?.pose(posed, this.eyes);
		if (expressions && this.wrinkleExpressions) {
			this.wrinkleExpressions.forEach((e, k) => (this.wrinkleWeights[k] = expressions[e]));
		}
		if (this.normals) {
			const normals = this.normals.array as Float32Array;
			this.normals.clearUpdateRanges();
			// The lit skin's normals change everywhere; the mouth's own, smoothed over its own
			// triangles only, overwrite theirs.
			if (this.normalSums) cornerNormals(posed, this.cornerVertices, this.normalSums, normals);
			if (this.mouth) {
				const { normals: mouthNormals, start, count } = this.mouth;
				mouthNormals.compute(posed, normals.subarray(3 * start));
				if (!this.normalSums) this.normals.addUpdateRange(3 * start, 3 * count);
			}
			this.normals.needsUpdate = true;
		}
		// The box frames the camera and is recomputed when asked for. The sphere orders the draws
		// every frame: one from every 64th vertex is close enough and much cheaper.
		const geometry = this.object.geometry;
		geometry.boundingBox = null;
		const box = new THREE.Box3();
		const point = new THREE.Vector3();
		for (let v = 0; v < posed.length; v += 3 * 64) box.expandByPoint(point.fromArray(posed, v));
		geometry.boundingSphere = box.getBoundingSphere(geometry.boundingSphere ?? new THREE.Sphere());
	}

	/** Free the geometry, and the materials if this mesh made them: dispose copies first. */
	dispose(): void {
		this.object.geometry.dispose();
		if (!this.ownsMaterials) return;
		for (const material of this.object.material as THREE.Material[]) {
			if (material instanceof THREE.MeshBasicMaterial) material.map?.dispose();
			material.dispose();
		}
	}
}

/** Normals for every corner: the lit skin's and the mouth's are written, the others unused. */
function normalsAttribute(corners: number): THREE.BufferAttribute {
	const normals = new THREE.BufferAttribute(new Float32Array(3 * corners), 3);
	normals.setUsage(THREE.DynamicDrawUsage);
	return normals;
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
