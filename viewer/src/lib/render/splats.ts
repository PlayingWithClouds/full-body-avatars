/**
 * The avatar's Gaussians as one instanced three.js mesh: a quad per Gaussian, placed on its
 * skinned triangle in the vertex shader the way `BoundGaussians.place` does it, projected to an
 * ellipse (EWA splatting, as gsplat does) and blended back to front over the textured mesh.
 *
 * The mesh hides a Gaussian as training hid it: whole, when its centre lies more than the
 * occlusion margin behind the mesh's depth at its pixel. A per-pixel depth test instead would cut
 * splats along the slanted skin under them, and the cut would slide as the view turns.
 *
 * Per-Gaussian data and the posed vertices live in float textures; only the posed vertices change
 * when the avatar moves, and only the draw order when the camera does.
 */

import * as THREE from 'three';
import { array, type AvatarFile } from '#lib/avatar/file.ts';

const TEXTURE_WIDTH = 2048;
/** Gaussians are drawn out to this many standard deviations, as in gsplat. */
const EXTENT = 3;

export class Splats {
	readonly object: THREE.Mesh;
	readonly count: number;
	private readonly material: THREE.ShaderMaterial;
	private readonly order: THREE.InstancedBufferAttribute;
	private readonly vertices: THREE.DataTexture;
	private readonly faces: Uint32Array;
	private readonly gaussianFaces: Uint32Array;
	private readonly barycentric: Float32Array;
	private readonly offsets: Float32Array;
	/** ``(G * 3)`` the Gaussians' centres in the current pose, for sorting. */
	private readonly means: Float32Array;
	private readonly depths: Float32Array;
	private readonly sorted: Float32Array;

	constructor(file: AvatarFile) {
		const gaussianFaces = array<Uint32Array>(file, 'gaussian_faces').data;
		const barycentric = array<Float32Array>(file, 'gaussian_barycentric').data;
		const offsets = array<Float32Array>(file, 'gaussian_offsets').data;
		const rotations = array<Float32Array>(file, 'gaussian_rotations').data;
		const logScales = array<Float32Array>(file, 'gaussian_log_scales').data;
		const opacities = array<Float32Array>(file, 'gaussian_opacities').data;
		const sh = array<Float32Array>(file, 'gaussian_sh');
		const faces = array<Uint32Array>(file, 'faces').data;
		const vertexCount = array<Float32Array>(file, 'rest_vertices').shape[0];
		const count = gaussianFaces.length;
		const shCount = sh.shape[1];

		this.count = count;
		this.faces = faces;
		this.gaussianFaces = gaussianFaces;
		this.barycentric = barycentric;
		this.offsets = offsets;
		this.means = new Float32Array(3 * count);
		this.depths = new Float32Array(count);
		this.sorted = new Float32Array(count);

		const anchor = new Float32Array(4 * count);
		const shape = new Float32Array(4 * count);
		const scale = new Float32Array(4 * count);
		for (let i = 0; i < count; i++) {
			anchor.set(barycentric.subarray(3 * i, 3 * i + 3), 4 * i);
			anchor[4 * i + 3] = gaussianFaces[i];
			shape.set(offsets.subarray(3 * i, 3 * i + 3), 4 * i);
			shape[4 * i + 3] = opacities[i];
			scale.set(logScales.subarray(3 * i, 3 * i + 3), 4 * i);
		}
		const colours = new Float32Array(4 * count * shCount);
		for (let i = 0; i < count * shCount; i++)
			colours.set(sh.data.subarray(3 * i, 3 * i + 3), 4 * i);
		const faceData = new Uint32Array(4 * (faces.length / 3));
		for (let f = 0; f < faces.length / 3; f++)
			faceData.set(faces.subarray(3 * f, 3 * f + 3), 4 * f);

		this.vertices = floatTexture(new Float32Array(4 * vertexCount), vertexCount);
		this.material = new THREE.ShaderMaterial({
			glslVersion: THREE.GLSL3,
			vertexShader: VERTEX,
			fragmentShader: FRAGMENT,
			uniforms: {
				anchors: { value: floatTexture(anchor, count) },
				shapes: { value: floatTexture(shape, count) },
				rotations: { value: floatTexture(padded(rotations, count), count) },
				scales: { value: floatTexture(scale, count) },
				colours: { value: floatTexture(colours, count * shCount) },
				faceVertices: { value: uintTexture(faceData, faces.length / 3) },
				vertices: { value: this.vertices },
				shCount: { value: shCount },
				viewDependent: { value: true },
				antialias: { value: false },
				occlusionMargin: { value: file.metadata.occlusion_margin },
				meshDepth: { value: null },
				testMesh: { value: false },
				near: { value: 0.1 },
				far: { value: 100 },
				viewport: { value: new THREE.Vector2(1, 1) }
			},
			transparent: true,
			// A quad's winding depends on its ellipse's axes; neither side is a back.
			side: THREE.DoubleSide,
			depthTest: false,
			depthWrite: false,
			blending: THREE.CustomBlending,
			blendSrc: THREE.OneFactor,
			blendDst: THREE.OneMinusSrcAlphaFactor,
			blendSrcAlpha: THREE.OneFactor,
			blendDstAlpha: THREE.OneMinusSrcAlphaFactor
		});

		const geometry = new THREE.InstancedBufferGeometry();
		geometry.setAttribute(
			'position',
			new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]), 3)
		);
		geometry.setIndex([0, 1, 2, 0, 2, 3]);
		this.order = new THREE.InstancedBufferAttribute(this.sorted, 1);
		this.order.setUsage(THREE.DynamicDrawUsage);
		geometry.setAttribute('splat', this.order);
		geometry.instanceCount = count;
		this.object = new THREE.Mesh(geometry, this.material);
		this.object.frustumCulled = false;
		this.object.renderOrder = 1;
	}

	/** Move the Gaussians with ``(V * 3)`` posed vertices. */
	setVertices(posed: Float32Array): void {
		const data = this.vertices.image.data as Float32Array;
		for (let v = 0; v < posed.length / 3; v++) data.set(posed.subarray(3 * v, 3 * v + 3), 4 * v);
		this.vertices.needsUpdate = true;
		// The centres, as the vertex shader places them, decide the draw order.
		const { faces, gaussianFaces, barycentric, offsets, means } = this;
		for (let i = 0; i < this.count; i++) {
			const f = 3 * gaussianFaces[i];
			const a = 3 * faces[f];
			const b = 3 * faces[f + 1];
			const c = 3 * faces[f + 2];
			const e1x = posed[b] - posed[a];
			const e1y = posed[b + 1] - posed[a + 1];
			const e1z = posed[b + 2] - posed[a + 2];
			const e2x = posed[c] - posed[a];
			const e2y = posed[c + 1] - posed[a + 1];
			const e2z = posed[c + 2] - posed[a + 2];
			let nx = e1y * e2z - e1z * e2y;
			let ny = e1z * e2x - e1x * e2z;
			let nz = e1x * e2y - e1y * e2x;
			const area = Math.hypot(nx, ny, nz) || 1e-12;
			const size = Math.sqrt(area);
			nx /= area;
			ny /= area;
			nz /= area;
			const edge = Math.hypot(e1x, e1y, e1z) || 1e-12;
			const xx = e1x / edge;
			const xy = e1y / edge;
			const xz = e1z / edge;
			const yx = ny * xz - nz * xy;
			const yy = nz * xx - nx * xz;
			const yz = nx * xy - ny * xx;
			const ox = offsets[3 * i] * size;
			const oy = offsets[3 * i + 1] * size;
			const oz = offsets[3 * i + 2] * size;
			const u = barycentric[3 * i];
			const v = barycentric[3 * i + 1];
			const w = barycentric[3 * i + 2];
			means[3 * i] = u * posed[a] + v * posed[b] + w * posed[c] + xx * ox + yx * oy + nx * oz;
			means[3 * i + 1] =
				u * posed[a + 1] + v * posed[b + 1] + w * posed[c + 1] + xy * ox + yy * oy + ny * oz;
			means[3 * i + 2] =
				u * posed[a + 2] + v * posed[b + 2] + w * posed[c + 2] + xz * ox + yz * oy + nz * oz;
		}
	}

	/**
	 * Rendering options. ``viewDependent``: colours change with the viewing direction (the
	 * spherical harmonics beyond the constant one). ``antialias``: Gaussians smaller than a pixel
	 * fade instead of twinkling, as in gsplat's antialiased mode; training used neither fade.
	 */
	setOptions(options: { viewDependent: boolean; antialias: boolean }): void {
		this.material.uniforms.viewDependent.value = options.viewDependent;
		this.material.uniforms.antialias.value = options.antialias;
	}

	/** Order the Gaussians back to front for a camera; call when it or the pose changed. */
	sort(camera: THREE.Camera): void {
		const position = camera.getWorldPosition(new THREE.Vector3());
		const forward = camera.getWorldDirection(new THREE.Vector3());
		const { means, depths, count } = this;
		let min = Infinity;
		let max = -Infinity;
		for (let i = 0; i < count; i++) {
			const d =
				(means[3 * i] - position.x) * forward.x +
				(means[3 * i + 1] - position.y) * forward.y +
				(means[3 * i + 2] - position.z) * forward.z;
			depths[i] = d;
			if (d < min) min = d;
			if (d > max) max = d;
		}
		// Counting sort on 16-bit depth buckets, farthest first.
		const buckets = 65536;
		const scale = (buckets - 1) / Math.max(max - min, 1e-6);
		const counts = new Uint32Array(buckets);
		const keys = new Uint16Array(count);
		for (let i = 0; i < count; i++) {
			const key = buckets - 1 - Math.floor((depths[i] - min) * scale);
			keys[i] = key;
			counts[key]++;
		}
		for (let k = 1; k < buckets; k++) counts[k] += counts[k - 1];
		for (let i = count - 1; i >= 0; i--) this.sorted[--counts[keys[i]]] = i;
		this.order.needsUpdate = true;
	}

	/** Hide Gaussians behind the mesh, given its depth buffer from ``camera``; null draws all. */
	setMeshDepth(depth: THREE.DepthTexture | null, camera: THREE.PerspectiveCamera): void {
		const uniforms = this.material.uniforms;
		uniforms.meshDepth.value = depth;
		uniforms.testMesh.value = depth !== null;
		uniforms.near.value = camera.near;
		uniforms.far.value = camera.far;
	}

	setViewport(width: number, height: number): void {
		this.material.uniforms.viewport.value.set(width, height);
	}

	dispose(): void {
		this.object.geometry.dispose();
		this.material.dispose();
		for (const uniform of Object.values(this.material.uniforms)) {
			if (uniform.value instanceof THREE.Texture) uniform.value.dispose();
		}
	}
}

function padded(values: Float32Array, count: number): Float32Array {
	return values.length === 4 * count ? values : new Float32Array(values);
}

function rows(count: number): number {
	return Math.max(1, Math.ceil(count / TEXTURE_WIDTH));
}

function floatTexture(data: Float32Array, count: number): THREE.DataTexture {
	const full = new Float32Array(4 * TEXTURE_WIDTH * rows(count));
	full.set(data);
	const texture = new THREE.DataTexture(
		full, TEXTURE_WIDTH, rows(count), THREE.RGBAFormat, THREE.FloatType
	); // prettier-ignore
	texture.needsUpdate = true;
	return texture;
}

function uintTexture(data: Uint32Array, count: number): THREE.DataTexture {
	const full = new Uint32Array(4 * TEXTURE_WIDTH * rows(count));
	full.set(data);
	const texture = new THREE.DataTexture(
		full, TEXTURE_WIDTH, rows(count), THREE.RGBAIntegerFormat, THREE.UnsignedIntType
	); // prettier-ignore
	texture.internalFormat = 'RGBA32UI';
	texture.needsUpdate = true;
	return texture;
}

const VERTEX = /* glsl */ `
precision highp float;
precision highp int;
precision highp usampler2D;

uniform sampler2D anchors;    // barycentric xyz, triangle
uniform sampler2D shapes;     // offset xyz (triangle frame, triangle sizes), opacity
uniform sampler2D rotations;  // wxyz relative to the triangle frame
uniform sampler2D scales;     // log extents in triangle sizes
uniform sampler2D colours;    // spherical harmonics, shCount texels per Gaussian
uniform usampler2D faceVertices;
uniform sampler2D vertices;   // posed vertices
uniform int shCount;
uniform bool viewDependent;
uniform bool antialias;
uniform float occlusionMargin;
uniform sampler2D meshDepth;  // the mesh's depth buffer, when testMesh
uniform bool testMesh;
uniform float near;
uniform float far;
uniform vec2 viewport;

in float splat;
out vec2 corner;
out vec4 colour;

const int WIDTH = ${TEXTURE_WIDTH};
const float EXTENT = ${EXTENT.toFixed(1)};
const float SH_C0 = 0.28209479177387814;
const float SH_C1 = 0.4886025119029199;

vec4 fetch(sampler2D data, int i) { return texelFetch(data, ivec2(i % WIDTH, i / WIDTH), 0); }
vec3 vertexAt(uint i) { return fetch(vertices, int(i)).xyz; }

mat3 quaternionMatrix(vec4 q) {
	q = normalize(q);
	float w = q.x, x = q.y, y = q.z, z = q.w;
	return mat3(
		1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y),
		2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
		2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)
	);
}

void main() {
	int i = int(splat);
	vec4 anchor = fetch(anchors, i);
	vec4 shape = fetch(shapes, i);
	int face = int(anchor.w);
	uvec4 corners = texelFetch(faceVertices, ivec2(face % WIDTH, face / WIDTH), 0);
	vec3 a = vertexAt(corners.x), b = vertexAt(corners.y), c = vertexAt(corners.z);

	// The triangle's frame: first edge, across, normal; its size scales offsets and extents.
	vec3 normal = cross(b - a, c - a);
	float size = sqrt(max(length(normal), 1e-12));
	vec3 x = normalize(b - a);
	vec3 z = normalize(normal);
	mat3 frame = mat3(x, cross(z, x), z);

	vec3 mean = a * anchor.x + b * anchor.y + c * anchor.z + frame * (shape.xyz * size);
	mat3 rotation = frame * quaternionMatrix(fetch(rotations, i));
	vec3 extent = exp(fetch(scales, i).xyz) * size;
	mat3 scaled = rotation * mat3(extent.x, 0.0, 0.0, 0.0, extent.y, 0.0, 0.0, 0.0, extent.z);
	mat3 covariance = scaled * transpose(scaled);

	vec4 view = viewMatrix * vec4(mean, 1.0);
	float depth = -view.z;
	// Discs lie on the skin; one whose skin faces away is culled with it, as in training.
	if (depth < 0.01 || dot(rotation[2], cameraPosition - mean) <= 0.0) {
		gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
		return;
	}
	vec2 focal = vec2(projectionMatrix[0][0], projectionMatrix[1][1]) * viewport * 0.5;
	mat3 jacobian = mat3(
		focal.x / depth, 0.0, 0.0,
		0.0, focal.y / depth, 0.0,
		focal.x * view.x / (depth * depth), focal.y * view.y / (depth * depth), 0.0
	);
	mat3 toView = mat3(viewMatrix);
	mat3 projected = jacobian * toView * covariance * transpose(toView) * transpose(jacobian);
	float ca = projected[0][0] + 0.3, cb = projected[0][1], cc = projected[1][1] + 0.3;

	float middle = 0.5 * (ca + cc);
	float radius = length(vec2(0.5 * (ca - cc), cb));
	float major = middle + radius, minor = max(middle - radius, 0.0);
	vec2 axis = abs(cb) > 1e-9 ? normalize(vec2(cb, major - ca)) : (ca >= cc ? vec2(1, 0) : vec2(0, 1));
	vec2 majorAxis = min(sqrt(major), 1024.0) * axis;
	vec2 minorAxis = min(sqrt(minor), 1024.0) * vec2(axis.y, -axis.x);

	// The depth test against the mesh lets a Gaussian lie up to the margin behind it.
	vec4 centre = projectionMatrix * view;
	if (testMesh) {
		vec2 ndc = centre.xy / centre.w;
		ivec2 pixel = ivec2(floor((ndc * 0.5 + 0.5) * viewport));
		if (all(greaterThanEqual(pixel, ivec2(0))) && all(lessThan(pixel, ivec2(viewport)))) {
			float stored = texelFetch(meshDepth, pixel, 0).r;
			if (stored < 1.0) {
				float meshDistance = 2.0 * near * far / (far + near - (2.0 * stored - 1.0) * (far - near));
				if (depth > meshDistance + occlusionMargin * size) {
					gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
					return;
				}
			}
		}
	}
	corner = position.xy * EXTENT;
	vec2 offset = (corner.x * majorAxis + corner.y * minorAxis) / (0.5 * viewport);
	gl_Position = vec4(centre.xy / centre.w + offset, 0.0, 1.0);

	vec3 direction = normalize(mean - cameraPosition);
	vec3 rgb = SH_C0 * fetch(colours, i * shCount).rgb;
	if (viewDependent && shCount > 1) {
		rgb += SH_C1 * (-direction.y * fetch(colours, i * shCount + 1).rgb
			+ direction.z * fetch(colours, i * shCount + 2).rgb
			- direction.x * fetch(colours, i * shCount + 3).rgb);
	}
	float opacity = shape.w;
	if (antialias) {
		float sharp = (ca - 0.3) * (cc - 0.3) - cb * cb;
		opacity *= sqrt(max(sharp, 0.0) / max(ca * cc - cb * cb, 1e-12));
	}
	colour = vec4(max(rgb + 0.5, 0.0), opacity);
}
`;

const FRAGMENT = /* glsl */ `
precision highp float;

in vec2 corner;
in vec4 colour;
out vec4 fragment;

void main() {
	float distance = dot(corner, corner);
	if (distance > ${(EXTENT * EXTENT).toFixed(1)}) discard;
	float alpha = min(0.99, colour.a * exp(-0.5 * distance));
	if (alpha < 1.0 / 255.0) discard;
	fragment = vec4(colour.rgb * alpha, alpha);
}
`;
