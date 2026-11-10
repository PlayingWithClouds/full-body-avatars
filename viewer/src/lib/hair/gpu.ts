/**
 * The bits of WebGL2 general-purpose computing the hair needs: float data textures and
 * full-screen passes that run a fragment shader once per texel of a render target.
 */

import * as THREE from 'three';

/** An RGBA float texture of `width * height` texels, read with `texelFetch`. */
export function dataTexture(data: Float32Array, width: number, height: number): THREE.DataTexture {
	const padded = data.length === 4 * width * height ? data : new Float32Array(4 * width * height);
	if (padded !== data) padded.set(data);
	const texture = new THREE.DataTexture(padded, width, height, THREE.RGBAFormat, THREE.FloatType);
	texture.minFilter = texture.magFilter = THREE.NearestFilter;
	texture.generateMipmaps = false;
	texture.needsUpdate = true;
	return texture;
}

/** A float render target for computing, `count` textures written at once. */
export function computeTarget(width: number, height: number, count = 1): THREE.WebGLRenderTarget {
	return new THREE.WebGLRenderTarget(width, height, {
		count,
		type: THREE.FloatType,
		format: THREE.RGBAFormat,
		minFilter: THREE.NearestFilter,
		magFilter: THREE.NearestFilter,
		depthBuffer: false,
		generateMipmaps: false
	});
}

export const FULLSCREEN_VERTEX = /* glsl */ `
in vec3 position;
void main() {
	gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

/** A shader run over every texel of a target: one triangle covering the viewport. */
export class FullscreenPass {
	readonly material: THREE.RawShaderMaterial;
	private readonly scene = new THREE.Scene();
	private readonly camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
	private readonly mesh: THREE.Mesh;

	constructor(fragmentShader: string, uniforms: Record<string, THREE.IUniform>) {
		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute(
			'position',
			new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3)
		);
		this.material = new THREE.RawShaderMaterial({
			glslVersion: THREE.GLSL3,
			vertexShader: FULLSCREEN_VERTEX,
			fragmentShader,
			uniforms,
			depthTest: false,
			depthWrite: false
		});
		this.mesh = new THREE.Mesh(geometry, this.material);
		this.mesh.frustumCulled = false;
		this.scene.add(this.mesh);
	}

	get uniforms(): Record<string, THREE.IUniform> {
		return this.material.uniforms;
	}

	run(renderer: THREE.WebGLRenderer, target: THREE.WebGLRenderTarget | null): void {
		renderer.setRenderTarget(target);
		renderer.render(this.scene, this.camera);
	}

	dispose(): void {
		this.mesh.geometry.dispose();
		this.material.dispose();
	}
}

/**
 * Run `draw` with the renderer's target, clear colour and auto-clear restored afterwards, so the
 * hair's passes leave the viewer's state as they found it.
 */
export function preserving(renderer: THREE.WebGLRenderer, draw: () => void): void {
	const target = renderer.getRenderTarget();
	const colour = renderer.getClearColor(new THREE.Color());
	const alpha = renderer.getClearAlpha();
	const autoClear = renderer.autoClear;
	try {
		draw();
	} finally {
		renderer.setRenderTarget(target);
		renderer.setClearColor(colour, alpha);
		renderer.autoClear = autoClear;
	}
}

/** A three.js matrix from a row-major one. */
export function matrix(rowMajor: ArrayLike<number>, out = new THREE.Matrix4()): THREE.Matrix4 {
	const m = rowMajor;
	return out.set(
		m[0], m[1], m[2], m[3],
		m[4], m[5], m[6], m[7],
		m[8], m[9], m[10], m[11],
		m[12], m[13], m[14], m[15]
	); // prettier-ignore
}
