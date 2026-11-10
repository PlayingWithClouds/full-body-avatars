/**
 * Render strands: tens of thousands of hairs interpolated between the simulated guides, drawn as
 * camera-facing ribbons.
 *
 * Whenever the guides move, one pass lays out every render strand's points in a texture: a
 * smooth blend of the four guides around its root, gathered into its clump towards the tip,
 * curled, frizzed, and kept out of the head and body. The ribbons then read their points from
 * that texture, so the three draws of the hair (two for the shadow map, one for the camera) don't
 * redo the work.
 *
 * Fibres are 50-100 µm across, far under a pixel at any sensible distance. A ribbon is drawn at
 * least a pixel wide plus a pixel of margin and covers each pixel by the share the true fibre
 * would (a box filter), so thin hair fades instead of flickering, and thousands of strands add
 * up to the right opacity.
 */

import * as THREE from 'three';
import { BODY_GLSL } from '#lib/hair/colliders.ts';
import { GUIDE_POINTS, STRAND_TEXELS, type Groom, type HairLook } from '#lib/hair/groom.ts';
import { computeTarget, FullscreenPass } from '#lib/hair/gpu.ts';
import { OCT_ENCODE_GLSL } from '#lib/hair/scalp.ts';
import { MELANIN_GLSL, opacityMapGlsl, OPACITY_LAYERS, SHADING_GLSL } from '#lib/hair/shading.ts';
import type { Lighting } from '#lib/render/lighting.ts';

/** Texels per row of the strand textures. */
export const ROW = 4096;

export type StrandPass = 'opacityDepth' | 'opacityLayers' | 'shade';

/** The textures a groom's strands are made from, shared by every avatar wearing it. */
export interface GroomTextures {
	rest: THREE.DataTexture;
	strands: THREE.DataTexture;
	field: THREE.Texture;
	centre: THREE.Vector3;
}

export class Strands {
	readonly mesh: THREE.Mesh;
	readonly materials: Record<StrandPass, THREE.RawShaderMaterial>;
	private readonly target: THREE.WebGLRenderTarget;
	/** Per strand point: light from the key through the hair in front, and the room's share. */
	private readonly lit: THREE.WebGLRenderTarget;
	private readonly expansion: FullscreenPass;
	private readonly lighting: FullscreenPass;
	/** Uniforms every ribbon material shares. */
	private readonly shared: Record<string, THREE.IUniform>;

	/** `body` holds the uniforms of `BODY_GLSL`, which the caller keeps on the avatar's pose. */
	constructor(
		private readonly groom: Groom,
		textures: GroomTextures,
		geometry: THREE.InstancedBufferGeometry,
		body: Record<string, THREE.IUniform>
	) {
		const perRow = Math.floor(ROW / groom.renderPoints);
		this.target = computeTarget(perRow * groom.renderPoints, Math.ceil(groom.strands / perRow));
		this.lit = computeTarget(this.target.width, this.target.height);
		const layout = {
			strandData: { value: textures.strands },
			renderPoints: { value: groom.renderPoints },
			strandsPerRow: { value: perRow },
			strandCount: { value: groom.strands }
		};
		this.expansion = new FullscreenPass(EXPAND_FRAGMENT, {
			...layout,
			guides: { value: null },
			field: { value: textures.field },
			centre: { value: textures.centre },
			head: { value: new THREE.Matrix4() },
			headInverse: { value: new THREE.Matrix4() },
			...body,
			strandLength: { value: groom.shape.length },
			curl: { value: groom.shape.curl },
			curlPitch: { value: groom.shape.curlPitch },
			clumping: { value: groom.shape.clumping },
			frizz: { value: groom.shape.frizz }
		});

		const pigment = {
			eumelanin: { value: 1 },
			pheomelanin: { value: 0.3 },
			variation: { value: 0.3 }
		};
		this.shared = {
			...layout,
			...pigment,
			strandPositions: { value: this.target.texture },
			thickness: { value: 80e-6 },
			widthScale: { value: 1 },
			viewportHeight: { value: 1 },
			orthographic: { value: false },
			viewDirection: { value: new THREE.Vector3() },
			eye: { value: new THREE.Vector3() }
		};
		const opacity = {
			opacityDepth: { value: null },
			opacityLayers: { value: null },
			lightMatrix: { value: new THREE.Matrix4() },
			lightRange: { value: 1 }
		};
		this.lighting = new FullscreenPass(LIGHT_FRAGMENT, {
			...layout,
			...pigment,
			...opacity,
			strandPositions: { value: this.target.texture }
		});
		const make = (pass: StrandPass, options: THREE.ShaderMaterialParameters) =>
			new THREE.RawShaderMaterial({
				glslVersion: THREE.GLSL3,
				vertexShader: `#define ${PASS_DEFINES[pass]}\n${RIBBON_VERTEX}`,
				fragmentShader: `#define ${PASS_DEFINES[pass]}\n${RIBBON_FRAGMENT}`,
				// A ribbon turns to face the viewer, so either winding can face it.
				side: THREE.DoubleSide,
				...options
			});
		const additive = {
			transparent: true,
			depthWrite: false,
			blending: THREE.CustomBlending,
			blendEquation: THREE.AddEquation,
			blendSrc: THREE.OneFactor,
			blendDst: THREE.OneFactor,
			blendSrcAlpha: THREE.OneFactor,
			blendDstAlpha: THREE.OneFactor
		};
		this.materials = {
			opacityDepth: make('opacityDepth', {
				uniforms: this.shared,
				colorWrite: false
			}),
			opacityLayers: make('opacityLayers', {
				uniforms: { ...this.shared, ...opacity, opacityScale: { value: 1 } },
				depthTest: false,
				...additive
			}),
			shade: make('shade', {
				uniforms: {
					...this.shared,
					strandLight: { value: this.lit.texture },
					roughness: { value: 0.35 },
					key: { value: new THREE.Vector3() },
					keyColour: { value: new THREE.Color() },
					fill: { value: new THREE.Vector3() },
					fillColour: { value: new THREE.Color() },
					ambient: { value: new THREE.Color() },
					depthReference: { value: 0 },
					sceneDepth: { value: null },
					cameraNear: { value: 0.1 },
					cameraFar: { value: 100 }
				},
				// The meshes hide hair by a soft test of their depth in the fragment shader.
				depthTest: false,
				...additive
			})
		};
		this.mesh = new THREE.Mesh(geometry, this.materials.shade);
		this.mesh.frustumCulled = false;
	}

	/** Lay out the render strands from guide positions `guides` on head transform `head`. */
	expand(renderer: THREE.WebGLRenderer, guides: THREE.Texture, head: THREE.Matrix4): void {
		const u = this.expansion.uniforms;
		u.guides.value = guides;
		u.head.value.copy(head);
		u.headInverse.value.copy(head).invert();
		this.expansion.run(renderer, this.target);
	}

	/** Light the strand points from the current opacity map. */
	light(renderer: THREE.WebGLRenderer): void {
		this.lighting.run(renderer, this.lit);
	}

	setLook(look: HairLook): void {
		this.shared.thickness.value = look.thickness * 1e-6;
		const u = this.shared;
		const { eumelanin, pheomelanin } = pigment(look);
		u.eumelanin.value = eumelanin;
		u.pheomelanin.value = pheomelanin;
		u.variation.value = look.variation;
		this.materials.shade.uniforms.roughness.value = 0.15 + 0.6 * look.roughness;
	}

	setLighting(lighting: Lighting): void {
		const u = this.materials.shade.uniforms;
		u.key.value.copy(lighting.key);
		u.keyColour.value.copy(lighting.keyColour);
		u.fill.value.copy(lighting.fill);
		u.fillColour.value.copy(lighting.fillColour);
		u.ambient.value.copy(lighting.ambient);
	}

	/** Use the deep opacity map in `depth` and `layers`, made with `light`. */
	setOpacityMap(
		depth: THREE.Texture,
		layers: THREE.Texture,
		light: THREE.OrthographicCamera
	): void {
		const matrix = new THREE.Matrix4().multiplyMatrices(
			light.projectionMatrix,
			light.matrixWorldInverse
		);
		for (const u of [this.materials.opacityLayers.uniforms, this.lighting.uniforms]) {
			u.opacityDepth.value = depth;
			u.opacityLayers.value = layers;
			u.lightMatrix.value.copy(matrix);
			u.lightRange.value = light.far - light.near;
		}
	}

	/**
	 * Get ready to draw `pass` through `camera` onto a target `height` pixels high, with one in
	 * `sampling` of the strands (each standing for that many in the opacity map).
	 */
	prepare(pass: StrandPass, camera: THREE.Camera, height: number, sampling = 1): void {
		this.mesh.material = this.materials[pass];
		const geometry = this.mesh.geometry as THREE.InstancedBufferGeometry;
		geometry.instanceCount = Math.ceil(this.groom.strands / sampling);
		this.materials.opacityLayers.uniforms.opacityScale.value =
			this.groom.strands / geometry.instanceCount;
		const u = this.shared;
		u.widthScale.value = 1;
		u.viewportHeight.value = height;
		u.orthographic.value = (camera as THREE.OrthographicCamera).isOrthographicCamera === true;
		camera.getWorldDirection(u.viewDirection.value);
		camera.getWorldPosition(u.eye.value);
	}

	/**
	 * Shade through perspective `camera`, hidden by the meshes' depth in `sceneDepth`; strands
	 * from `depthReference` metres away on count as the front of the hair.
	 */
	prepareShade(
		camera: THREE.PerspectiveCamera,
		height: number,
		sceneDepth: THREE.Texture,
		depthReference: number
	): void {
		// Far away, where strands are a small fraction of a pixel, fewer and wider strands cover
		// the same: drop to as few as a quarter of them.
		const pixel =
			(2 * Math.max(depthReference, 0.05) * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) /
			height;
		const fraction = Math.min(1, Math.max(0.25, this.shared.thickness.value / pixel / 0.3));
		this.prepare('shade', camera, height, 1 / fraction);
		const drawn = (this.mesh.geometry as THREE.InstancedBufferGeometry).instanceCount;
		this.shared.widthScale.value = this.groom.strands / drawn;
		const u = this.materials.shade.uniforms;
		u.sceneDepth.value = sceneDepth;
		u.cameraNear.value = camera.near;
		u.cameraFar.value = camera.far;
		u.depthReference.value = depthReference;
	}

	dispose(): void {
		this.target.dispose();
		this.lit.dispose();
		this.expansion.dispose();
		this.lighting.dispose();
		for (const material of Object.values(this.materials)) material.dispose();
	}

	get strandCount(): number {
		return this.groom.strands;
	}
}

/**
 * Melanin concentrations per fibre diameter for a look, after Chiang et al. (2016): about 0.3
 * eumelanin is blonde, 1.5 brown, 8 black. Melanin darkens far from linearly, so the slider is
 * squared.
 */
export function pigment(look: HairLook): { eumelanin: number; pheomelanin: number } {
	return {
		eumelanin: (0.1 + 8.5 * look.melanin ** 2) * (1 - 0.85 * look.redness),
		pheomelanin: look.redness * (0.4 + 2.5 * look.melanin)
	};
}

/** The ribbon all strands share: two vertices per point, drawn once per strand. */
export function ribbonGeometry(groom: Groom): THREE.InstancedBufferGeometry {
	const points = groom.renderPoints;
	const geometry = new THREE.InstancedBufferGeometry();
	const position = new Float32Array(points * 2 * 3);
	for (let k = 0; k < points; k++) {
		position.set([k, -1, 0, k, 1, 0], 6 * k);
	}
	const index: number[] = [];
	for (let k = 0; k + 1 < points; k++) {
		const a = 2 * k;
		index.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
	}
	geometry.setAttribute('position', new THREE.BufferAttribute(position, 3));
	geometry.setIndex(index);
	geometry.instanceCount = groom.strands;
	return geometry;
}

const PASS_DEFINES: Record<StrandPass, string> = {
	opacityDepth: 'OPACITY_DEPTH',
	opacityLayers: 'OPACITY_LAYERS',
	shade: 'SHADE'
};

const LAYOUT_GLSL = /* glsl */ `
uniform sampler2D strandData;
uniform int renderPoints;
uniform int strandsPerRow;
uniform int strandCount;

vec4 strandTexel(int strand, int k) {
	int index = strand * ${STRAND_TEXELS} + k;
	return texelFetch(strandData, ivec2(index % ${ROW}, index / ${ROW}), 0);
}

ivec2 pointTexel(int strand, int point) {
	return ivec2((strand % strandsPerRow) * renderPoints + point, strand / strandsPerRow);
}

float hash(float n) {
	return fract(sin(n) * 43758.5453);
}
`;

const EXPAND_FRAGMENT = /* glsl */ `
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D guides;
uniform sampler2D field;
uniform vec3 centre;
uniform mat4 head;
uniform mat4 headInverse;
uniform float strandLength;
uniform float curl;
uniform float curlPitch;
uniform float clumping;
uniform float frizz;

layout(location = 0) out vec4 outPosition;

${LAYOUT_GLSL}
${OCT_ENCODE_GLSL}
${BODY_GLSL}

const int POINTS = ${GUIDE_POINTS};

vec3 guidePoint(int g, int k) {
	return texelFetch(guides, ivec2(clamp(k, 0, POINTS - 1), g), 0).xyz;
}

/** Point and derivative at u (0 to POINTS - 1) along guide g, as a Catmull-Rom spline. */
void guide(int g, float u, out vec3 p, out vec3 dp) {
	int k = min(int(floor(u)), POINTS - 2);
	float t = u - float(k);
	vec3 p0 = guidePoint(g, k - 1);
	vec3 p1 = guidePoint(g, k);
	vec3 p2 = guidePoint(g, k + 1);
	vec3 p3 = guidePoint(g, k + 2);
	vec3 b = p2 - p0;
	vec3 c = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3;
	vec3 d = -p0 + 3.0 * p1 - 3.0 * p2 + p3;
	p = 0.5 * (2.0 * p1 + t * (b + t * (c + t * d)));
	dp = 0.5 * (b + t * (2.0 * c + 3.0 * t * d));
}

/** Smooth noise in [-1, 1]³ along a strand. */
vec3 noise(float seed, float x) {
	float i = floor(x);
	float f = fract(x);
	f = f * f * (3.0 - 2.0 * f);
	vec3 a = vec3(hash(seed * 131.1 + i), hash(seed * 171.3 + i + 17.0), hash(seed * 97.7 + i + 41.0));
	vec3 b = vec3(
		hash(seed * 131.1 + i + 1.0),
		hash(seed * 171.3 + i + 18.0),
		hash(seed * 97.7 + i + 42.0)
	);
	return mix(a, b, f) * 2.0 - 1.0;
}

void main() {
	ivec2 texel = ivec2(gl_FragCoord.xy);
	int strand = texel.y * strandsPerRow + texel.x / renderPoints;
	int point = texel.x % renderPoints;
	if (strand >= strandCount) {
		outPosition = vec4(0.0);
		return;
	}
	vec4 indices = strandTexel(strand, 0);
	vec4 weights = strandTexel(strand, 1);
	vec4 shape = strandTexel(strand, 2);
	vec4 offset = strandTexel(strand, 3);
	float tip = shape.x;
	float seed = shape.y;
	bool flyaway = shape.w > 0.5;
	float s = tip * float(point) / float(renderPoints - 1);
	float u = s * float(POINTS - 1);
	float arc = s * strandLength;

	vec3 p = vec3(0.0);
	vec3 dp = vec3(0.0);
	vec3 clump;
	vec3 dClump;
	for (int k = 0; k < 4; k++) {
		vec3 q;
		vec3 dq;
		guide(int(indices[k]), u, q, dq);
		if (k == 0) {
			clump = q;
			dClump = dq;
		}
		p += weights[k] * q;
		dp += weights[k] * dq;
	}
	mat3 rotation = mat3(head);
	vec3 rootOffset = rotation * offset.xyz;
	p += rootOffset;
	// Strands gather into the clump of their nearest guide, more towards the tip.
	float gather = (flyaway ? 0.2 : 1.0) * clumping * smoothstep(0.0, 0.35, arc) * 0.92;
	p = mix(p, clump + rootOffset * 0.15, gather);
	dp = mix(dp, dClump, gather);

	vec3 tangent = normalize(dp);
	vec3 outward = p - (head * vec4(centre, 1.0)).xyz;
	vec3 side = cross(tangent, outward);
	side = dot(side, side) > 1e-10 ? normalize(side) : normalize(cross(tangent, vec3(0.0, 0.0, 1.0)));
	vec3 normal = cross(side, tangent);

	// Curls: the strands of one clump turn together, so they form ringlets.
	if (curl > 0.0) {
		float phase = 6.2832 * (hash(indices.x * 0.731 + 3.1) + 0.12 * seed);
		float angle = phase + 6.2832 * arc / curlPitch;
		float radius = curl * smoothstep(0.01, 0.045, arc) * (0.8 + 0.4 * hash(seed * 7.7));
		p += radius * (cos(angle) * normal + sin(angle) * side);
	}
	// Frizz: fine kinks, stronger towards the tips; flyaways stray in long slow arcs.
	vec3 kink = noise(seed * 13.7 + 1.0, arc * 14.0);
	p += frizz * 0.0018 * pow(s / max(tip, 1e-3), 1.2) * (kink.x * side + kink.y * normal);
	if (flyaway) {
		vec3 drift = noise(seed * 3.3 + 7.0, arc * 2.5);
		float amount = (0.008 + 0.035 * frizz) * smoothstep(0.02, 0.25, arc);
		p += amount * (drift.x * side + (0.7 + 0.5 * drift.y) * normal);
	}

	bool hit = false;
	vec3 q = (headInverse * vec4(p, 1.0)).xyz - centre;
	float r = length(q);
	vec3 d = q / max(r, 1e-6);
	float surface = texture(field, octEncode(d)).r;
	float clearance = 0.0008 + 0.004 * smoothstep(0.0, 0.03, arc);
	if (surface > 0.0 && r < surface + clearance) {
		p = (head * vec4(centre + d * (surface + clearance), 1.0)).xyz;
	}
	p = collideBody(p, 0.004, hit);
	outPosition = vec4(p, s);
}
`;

/** GLSL: a strand's absorption, its pigment varied by its random seed. */
const PIGMENT_GLSL = /* glsl */ `
uniform float eumelanin;
uniform float pheomelanin;
uniform float variation;

vec3 strandAbsorption(float seed) {
	// Pigment varies from strand to strand, more in amount than in redness.
	float jitter = (hash(seed * 91.7) + hash(seed * 53.3 + 2.0) - 1.0) * 1.2;
	float amount = exp(variation * 0.6 * jitter);
	float red = mix(1.0, exp(0.5 * (hash(seed * 17.1) - 0.5)), variation);
	return melaninAbsorption(eumelanin * amount, pheomelanin * amount * red);
}
`;

/**
 * Per strand point, once per change of the hair: the key light reaching it through the hair in
 * front (deep opacity map), and how much of the room's light reaches it.
 */
const LIGHT_FRAGMENT = /* glsl */ `
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D strandPositions;
layout(location = 0) out vec4 outLight;

${LAYOUT_GLSL}
${MELANIN_GLSL}
${PIGMENT_GLSL}
${opacityMapGlsl()}

void main() {
	ivec2 texel = ivec2(gl_FragCoord.xy);
	int strand = texel.y * strandsPerRow + texel.x / renderPoints;
	int point = texel.x % renderPoints;
	if (strand >= strandCount) {
		outLight = vec4(0.0);
		return;
	}
	vec3 p = texelFetch(strandPositions, texel, 0).xyz;
	vec3 sigma = strandAbsorption(strandTexel(strand, 2).y);
	// Light through hair keeps some of its colour: fibres pass on part of what reaches them.
	vec3 lost = 1.0 - 0.75 * exp(-sigma * 1.2);
	vec3 transmitted = exp(-opticalDepth(p, 1.5, 0.0) * 0.6 * lost);
	// Hair under other hair, and near the roots, sees less of the room.
	float along = float(point) / float(renderPoints - 1);
	float occlusion = mix(0.35, 1.0, strandTexel(strand, 3).w) * mix(0.55, 1.0, smoothstep(0.0, 0.25, along));
	occlusion *= mix(0.45, 1.0, dot(transmitted, vec3(0.3333)));
	outLight = vec4(transmitted, occlusion);
}
`;

const RIBBON_VERTEX = /* glsl */ `
precision highp float;
precision highp int;
precision highp sampler2D;

in vec3 position;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform sampler2D strandPositions;
uniform float thickness;
/** Strand width multiplier: wider when fewer strands are drawn. */
uniform float widthScale;
uniform float viewportHeight;
uniform bool orthographic;
uniform vec3 viewDirection;
uniform vec3 eye;

${LAYOUT_GLSL}

out float vAcross;
out float vWidth;
out float vDepth;
#ifdef SHADE
uniform sampler2D strandLight;
uniform float roughness;
uniform vec3 fill;
uniform vec3 fillColour;
uniform vec3 ambient;
out vec3 vTangent;
out vec3 vWorld;
out vec3 vSigma;
out vec3 vTransmitted;
out vec3 vSoftLight;
${MELANIN_GLSL}
${PIGMENT_GLSL}
${SHADING_GLSL}
#endif

vec3 strandPoint(int strand, int point) {
	return texelFetch(strandPositions, pointTexel(strand, clamp(point, 0, renderPoints - 1)), 0).xyz;
}

void main() {
	int strand = gl_InstanceID;
	int point = int(position.x);
	vec4 here = texelFetch(strandPositions, pointTexel(strand, point), 0);
	vec3 tangent = strandPoint(strand, point + 1) - strandPoint(strand, point - 1);
	tangent = dot(tangent, tangent) > 1e-14 ? normalize(tangent) : vec3(0.0, 0.0, -1.0);
	vec4 shape = strandTexel(strand, 2);
	float along = float(point) / float(renderPoints - 1);

	vec3 toViewer = orthographic ? -viewDirection : normalize(eye - here.xyz);
	vec3 side = cross(tangent, toViewer);
	side = dot(side, side) > 1e-10 ? normalize(side) : normalize(cross(tangent, vec3(0.0, 0.0, 1.0)));
	vec4 view = viewMatrix * vec4(here.xyz, 1.0);
	float pixel = (orthographic ? 1.0 : -view.z) * 2.0 / (projectionMatrix[1][1] * viewportHeight);
	// Hair keeps its diameter most of its length and tapers over its last part.
	float taper = mix(0.7, 1.0, smoothstep(0.0, 0.04, along)) * (1.0 - 0.85 * smoothstep(0.8, 1.0, along));
	float width = thickness * widthScale * shape.z * taper;
	float drawn = width + 1.5 * pixel;
	vec3 world = here.xyz + side * position.y * 0.5 * drawn;
	vec4 viewPosition = viewMatrix * vec4(world, 1.0);
	gl_Position = projectionMatrix * viewPosition;
	vAcross = position.y * 0.5 * drawn / pixel;
	vWidth = width / pixel;
	vDepth = -viewPosition.z;

#ifdef SHADE
	vTangent = tangent;
	vWorld = world;
	vec3 sigma = strandAbsorption(shape.y);
	vSigma = sigma;
	// The soft fill and room light are lit per point, the key light's sharp highlights per pixel.
	vec4 lit = texelFetch(strandLight, pointTexel(strand, point), 0);
	vTransmitted = lit.rgb;
	float occlusion = lit.a;
	vSoftLight = fillColour * occlusion * (
		hairScattering(tangent, toViewer, fill, sigma, roughness) +
		multipleScattering(tangent, toViewer, fill, sigma, vec3(1.0))
	);
	// The room's light: reflected off the cuticle, white, and scattered through the hair.
	vSoftLight += ambient * occlusion * (0.02 + 0.5 * exp(-2.0 * sigma));
#endif
}
`;

const RIBBON_FRAGMENT = /* glsl */ `
precision highp float;
precision highp int;
precision highp sampler2D;

in float vAcross;
in float vWidth;
in float vDepth;

#ifdef OPACITY_DEPTH
layout(location = 0) out vec4 outColour;
#endif

#ifdef OPACITY_LAYERS
uniform sampler2D opacityDepth;
uniform float lightRange;
/** The map is drawn from a sample of the strands; each counts for this many. */
uniform float opacityScale;
layout(location = 0) out vec4 outLayers;
#endif

#ifdef SHADE
${SHADING_GLSL}
uniform vec3 eye;
uniform float roughness;
uniform vec3 key;
uniform vec3 keyColour;
uniform float depthReference;
uniform sampler2D sceneDepth;
uniform float cameraNear;
uniform float cameraFar;
in vec3 vTangent;
in vec3 vWorld;
in vec3 vSigma;
in vec3 vTransmitted;
in vec3 vSoftLight;
layout(location = 0) out vec4 outAccumulated;
layout(location = 1) out vec4 outOpticalDepth;
#endif

#ifdef SHADE
/** View depth in metres of NDC depth z. */
float viewDepth(float z) {
	return 2.0 * cameraNear * cameraFar / (cameraFar + cameraNear - z * (cameraFar - cameraNear));
}
#endif

/**
 * Share of this pixel the fibre covers. A fibre under a pixel wide is spread by a tent filter
 * 1.5 pixels wide, which sums to about its width over the pixels it crosses at any angle (a box
 * filter would leave diagonal strands dotted); a wider one is box-filtered at its edges.
 */
float coverage() {
	float x = abs(vAcross);
	return vWidth < 1.0
		? vWidth * max(0.0, 1.0 - x / 0.75) / 0.75
		: clamp(0.5 + 0.5 * vWidth - x, 0.0, 1.0);
}

void main() {
	float alpha = coverage();
	if (alpha <= 0.0) discard;
	float optical = -log(1.0 - min(alpha, 0.98));

#ifdef OPACITY_DEPTH
	outColour = vec4(0.0);
#endif

#ifdef OPACITY_LAYERS
	float front = texelFetch(opacityDepth, ivec2(gl_FragCoord.xy), 0).r;
	float behind = (gl_FragCoord.z - front) * lightRange;
	const vec4 ENDS = vec4(${OPACITY_LAYERS.join(', ')});
	outLayers = opacityScale * optical * vec4(lessThan(vec4(behind), ENDS));
#endif

#ifdef SHADE
	// Hidden behind the meshes, fading over a few millimetres: hair lying on the skin would
	// otherwise flicker in and out of it in dashes.
	ivec2 pixel = ivec2(gl_FragCoord.xy);
	float z = texelFetch(sceneDepth, pixel, 0).r * 2.0 - 1.0;
	float meshDepth = viewDepth(z);
	// Skin seen at a grazing angle is far behind hair a millimetre above it along the view ray:
	// allow for the skin's depth slope over a couple of pixels.
	float slope = max(
		abs(viewDepth(texelFetch(sceneDepth, pixel + ivec2(1, 0), 0).r * 2.0 - 1.0) - meshDepth),
		abs(viewDepth(texelFetch(sceneDepth, pixel + ivec2(0, 1), 0).r * 2.0 - 1.0) - meshDepth)
	);
	float tolerance = 0.004 + 2.5 * min(slope, 0.02);
	alpha *= 1.0 - smoothstep(tolerance, tolerance + 0.008, vDepth - meshDepth);
	if (alpha <= 0.0) discard;
	optical = -log(1.0 - min(alpha, 0.98));

	vec3 T = normalize(vTangent);
	vec3 V = normalize(eye - vWorld);
	vec3 transmitted = vTransmitted;
	vec3 colour = keyColour * transmitted * hairScattering(T, V, key, vSigma, roughness);
	colour += keyColour * sqrt(transmitted) * multipleScattering(T, V, key, vSigma, transmitted);
	colour += vSoftLight;

	// Weighted, blended order-independent transparency (McGuire and Bavoil 2013), the weights
	// growing over a centimetre of depth towards the viewer so the outermost hair dominates its
	// pixel: measured from the skin behind, or where there is none from the nearest hair. Ten
	// centimetres span all of it, which half floats hold.
	float towards = z < 1.0 ? meshDepth - vDepth : 0.108 - (vDepth - depthReference);
	float weight = alpha * exp(clamp(towards / 0.012, 0.0, 9.0) - 9.0);
	outAccumulated = vec4(colour * weight, weight);
	outOpticalDepth = vec4(optical, 0.0, 0.0, 0.0);
#endif
}
`;
