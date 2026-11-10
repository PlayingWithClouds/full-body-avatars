/**
 * The skin's material for files that learned their light (format 4): the textures hold albedo,
 * lit by second-order spherical harmonics of the posed surface's normal, and the head's albedo
 * gains expression-dependent wrinkle detail.
 *
 * The light is fixed in the avatar's world (metres, z up, facing -y), which is the scene's: the
 * posed vertices, and so the normals computed from them, are already in it, copies and motion
 * clips' root offsets included. Copies of the avatar share the material and set its wrinkle
 * weights to their own expressions just before they are drawn, as the eyes do.
 */

import * as THREE from 'three';
import type { LightingMetadata, WrinkleMetadata } from '#lib/avatar/file.ts';

/** Constants of the real spherical harmonics up to degree 2, in the order the file gives them. */
const SH = [0.282095, 0.488603, 1.092548, 0.315392, 0.546274];

/** `(9)` the spherical harmonics' basis at a unit normal `(x, y, z)`. */
export function shBasis(x: number, y: number, z: number): number[] {
	return [
		SH[0],
		SH[1] * y,
		SH[1] * z,
		SH[1] * x,
		SH[2] * x * y,
		SH[2] * y * z,
		SH[3] * (3 * z * z - 1),
		SH[2] * x * z,
		SH[4] * (x * x - y * y)
	];
}

/** RGB the learned light multiplies the albedo by at a unit normal, never below zero. */
export function irradiance(sh: number[][], normal: [number, number, number]): number[] {
	const basis = shBasis(...normal);
	return [0, 1, 2].map((c) =>
		Math.max(
			0,
			basis.reduce((sum, y, k) => sum + sh[k][c] * y, 0)
		)
	);
}

/**
 * Smooth normals of a posed mesh whose triangles are given corner by corner: `corners` holds the
 * vertex of each, three per triangle. Each vertex sums its triangles' area-weighted normals (as
 * `body_models.mesh.vertex_normals`); `out` gets every corner's vertex's unit normal. `sums` is
 * scratch, as long as `posed`.
 */
export function cornerNormals(
	posed: Float32Array,
	corners: Uint32Array,
	sums: Float32Array,
	out: Float32Array
): void {
	sums.fill(0);
	for (let i = 0; i < corners.length; i += 3) {
		const a = 3 * corners[i];
		const b = 3 * corners[i + 1];
		const c = 3 * corners[i + 2];
		const ux = posed[b] - posed[a];
		const uy = posed[b + 1] - posed[a + 1];
		const uz = posed[b + 2] - posed[a + 2];
		const vx = posed[c] - posed[a];
		const vy = posed[c + 1] - posed[a + 1];
		const vz = posed[c + 2] - posed[a + 2];
		const nx = uy * vz - uz * vy;
		const ny = uz * vx - ux * vz;
		const nz = ux * vy - uy * vx;
		for (const v of [a, b, c]) {
			sums[v] += nx;
			sums[v + 1] += ny;
			sums[v + 2] += nz;
		}
	}
	for (let i = 0; i < corners.length; i++) {
		const v = 3 * corners[i];
		const length = Math.hypot(sums[v], sums[v + 1], sums[v + 2]) || 1;
		out[3 * i] = sums[v] / length;
		out[3 * i + 1] = sums[v + 1] / length;
		out[3 * i + 2] = sums[v + 2] / length;
	}
}

/** `(K)` the index among `expressionNames` of each wrinkle map's expression. */
export function wrinkleExpressions(wrinkles: string[], expressionNames: string[]): Uint32Array {
	return Uint32Array.from(wrinkles, (name) => {
		const index = expressionNames.indexOf(name);
		if (index < 0) throw new Error(`wrinkle map for unknown expression ${name}`);
		return index;
	});
}

/** The skin's material: one texture's albedo, its wrinkle maps if any, lit or not. */
export class SkinMaterial extends THREE.ShaderMaterial {
	/** The weights the wrinkle maps are drawn with, shared by every copy that uses the material. */
	private readonly weights: Float32Array;

	constructor(
		map: THREE.Texture,
		lighting: LightingMetadata | undefined,
		wrinkles: { metadata: WrinkleMetadata; maps: THREE.DataArrayTexture } | null
	) {
		const count = wrinkles?.metadata.expressions.length ?? 0;
		const weights = new Float32Array(Math.max(count, 1));
		const defines: Record<string, number | string> = {};
		if (lighting) defines.LIGHTING = '';
		if (count) defines.WRINKLES = count;
		super({
			defines,
			uniforms: {
				map: { value: map },
				lit: { value: true },
				sh: { value: (lighting?.sh ?? []).map((rgb) => new THREE.Vector3(...rgb)) },
				wrinkleMaps: { value: wrinkles?.maps ?? null },
				wrinkleWeights: { value: weights },
				wrinkleRegion: { value: new THREE.Vector4(...(wrinkles?.metadata.region ?? [0, 0, 1, 1])) },
				wrinkleScale: { value: wrinkles?.metadata.scale ?? 0 }
			},
			vertexShader: VERTEX,
			fragmentShader: FRAGMENT,
			side: THREE.FrontSide
		});
		this.weights = weights;
	}

	/** Light the albedo with the learned light, or show it unlit. */
	setLit(lit: boolean): void {
		this.uniforms.lit.value = lit;
		this.uniformsNeedUpdate = true;
	}

	/** Before a copy is drawn: its `(K)` wrinkle weights. */
	useWrinkles(weights: Float32Array): void {
		if (!this.defines.WRINKLES) return;
		this.weights.set(weights);
		// Copies share the material: three.js uploads its uniforms again only when told to.
		this.uniformsNeedUpdate = true;
	}

	dispose(): void {
		this.uniforms.map.value.dispose();
		this.uniforms.wrinkleMaps.value?.dispose();
		super.dispose();
	}
}

/** Wrinkle maps from the file's `(K, S, S, 3)` 8-bit rows, each map's top row first (v = 1). */
export function wrinkleTexture({
	data,
	shape
}: {
	data: Uint8Array;
	shape: number[];
}): THREE.DataArrayTexture {
	const [layers, size] = shape;
	const rgba = new Uint8Array(4 * layers * size * size);
	for (let layer = 0; layer < layers; layer++) {
		const first = layer * size * size;
		for (let row = 0; row < size; row++) {
			// three.js puts a data texture's first row at v = 0, the bottom.
			const source = first + (size - 1 - row) * size;
			const target = first + row * size;
			for (let x = 0; x < size; x++) {
				rgba.set(data.subarray(3 * (source + x), 3 * (source + x) + 3), 4 * (target + x));
				rgba[4 * (target + x) + 3] = 255;
			}
		}
	}
	const maps = new THREE.DataArrayTexture(rgba, size, size, layers);
	maps.colorSpace = THREE.NoColorSpace;
	maps.generateMipmaps = true;
	maps.minFilter = THREE.LinearMipmapLinearFilter;
	maps.magFilter = THREE.LinearFilter;
	maps.needsUpdate = true;
	return maps;
}

const VERTEX = /* glsl */ `
varying vec2 vUv;
#ifdef LIGHTING
varying vec3 vNormal;
#endif

void main() {
	vUv = uv;
#ifdef LIGHTING
	vNormal = mat3(modelMatrix) * normal;
#endif
	gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const FRAGMENT = /* glsl */ `
uniform sampler2D map;
varying vec2 vUv;
#ifdef LIGHTING
uniform bool lit;
uniform vec3 sh[9];
varying vec3 vNormal;
#endif
#ifdef WRINKLES
uniform highp sampler2DArray wrinkleMaps;
uniform float wrinkleWeights[WRINKLES];
uniform vec4 wrinkleRegion; // u0, v0, u1, v1
uniform float wrinkleScale;
#endif

void main() {
	vec3 albedo = texture2D(map, vUv).rgb;
#ifdef WRINKLES
	// Each map's bytes are signed detail around 128, scaled; only inside the region.
	vec2 local = (vUv - wrinkleRegion.xy) / (wrinkleRegion.zw - wrinkleRegion.xy);
	float inside = step(0.0, local.x) * step(local.x, 1.0) * step(0.0, local.y) * step(local.y, 1.0);
	for (int k = 0; k < WRINKLES; k++) {
		if (wrinkleWeights[k] == 0.0) continue;
		vec3 bytes = texture(wrinkleMaps, vec3(clamp(local, 0.0, 1.0), float(k))).rgb * 255.0;
		albedo += inside * wrinkleWeights[k] * (bytes - 128.0) / 127.0 * wrinkleScale;
	}
	albedo = clamp(albedo, 0.0, 1.0);
#endif
#ifdef LIGHTING
	if (lit) {
		vec3 n = normalize(vNormal);
		vec3 light = sh[0] * ${SH[0]}
			+ sh[1] * (${SH[1]} * n.y)
			+ sh[2] * (${SH[1]} * n.z)
			+ sh[3] * (${SH[1]} * n.x)
			+ sh[4] * (${SH[2]} * n.x * n.y)
			+ sh[5] * (${SH[2]} * n.y * n.z)
			+ sh[6] * (${SH[3]} * (3.0 * n.z * n.z - 1.0))
			+ sh[7] * (${SH[2]} * n.x * n.z)
			+ sh[8] * (${SH[4]} * (n.x * n.x - n.y * n.y));
		albedo *= max(light, 0.0);
	}
#endif
	gl_FragColor = vec4(clamp(albedo, 0.0, 1.0), 1.0);
}
`;
