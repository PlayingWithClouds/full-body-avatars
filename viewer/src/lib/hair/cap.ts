/**
 * The scalp under the hair, tinted the hair's colour: a thin shell over the scalp inside the
 * hairline, fading out across it. A real head has more strands, crossing at more angles, than can
 * be drawn, so between the drawn ones the bald scalp would show through as skin; games paint the
 * scalp under the hair for the same reason. The shell is drawn with the hair, behind all of it.
 */

import * as THREE from 'three';
import { hairDensity, type GroomShape } from '#lib/hair/groom.ts';
import { octDecode, type Scalp } from '#lib/hair/scalp.ts';
import { MELANIN_GLSL } from '#lib/hair/shading.ts';

/** Grid cells per side of the octahedral map the shell is made from. */
const CELLS = 96;
/** Opacity of the shell where hair grows fully. */
const OPACITY = 0.8;

/** The shell's geometry in the head bone's frame, for a groom of `shape`. */
export function capGeometry(scalp: Scalp, shape: GroomShape): THREE.BufferGeometry {
	const size = CELLS + 1;
	const positions = new Float32Array(3 * size * size);
	const normals = new Float32Array(3 * size * size);
	const density = new Float32Array(size * size);
	for (let j = 0; j < size; j++) {
		for (let i = 0; i < size; i++) {
			const d = octDecode(i / CELLS, j / CELLS);
			const v = j * size + i;
			positions.set(scalp.surface(d, 0.0006), 3 * v);
			normals.set(scalp.normal(d), 3 * v);
			density[v] = scalp.radius(d) > 0 ? hairDensity(d, shape) : 0;
		}
	}
	const index: number[] = [];
	for (let j = 0; j < CELLS; j++) {
		for (let i = 0; i < CELLS; i++) {
			const a = j * size + i;
			const corners = [a, a + 1, a + size, a + size + 1];
			if (corners.every((v) => density[v] === 0)) continue;
			index.push(a, a + 1, a + size + 1, a, a + size + 1, a + size);
		}
	}
	const geometry = new THREE.BufferGeometry();
	geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
	geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
	geometry.setAttribute('density', new THREE.BufferAttribute(density, 1));
	geometry.setIndex(index);
	return geometry;
}

/** The shell's material, drawn into the hair's accumulation targets like the strands. */
export function capMaterial(): THREE.RawShaderMaterial {
	return new THREE.RawShaderMaterial({
		glslVersion: THREE.GLSL3,
		vertexShader: CAP_VERTEX,
		fragmentShader: CAP_FRAGMENT,
		uniforms: {
			eumelanin: { value: 1 },
			pheomelanin: { value: 0.3 },
			key: { value: new THREE.Vector3() },
			keyColour: { value: new THREE.Color() },
			ambient: { value: new THREE.Color() },
			sceneDepth: { value: null },
			cameraNear: { value: 0.1 },
			cameraFar: { value: 100 }
		},
		transparent: true,
		depthTest: false,
		depthWrite: false,
		blending: THREE.CustomBlending,
		blendEquation: THREE.AddEquation,
		blendSrc: THREE.OneFactor,
		blendDst: THREE.OneFactor,
		blendSrcAlpha: THREE.OneFactor,
		blendDstAlpha: THREE.OneFactor
	});
}

const CAP_VERTEX = /* glsl */ `
precision highp float;

in vec3 position;
in vec3 normal;
in float density;
uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
out vec3 vNormal;
out float vDensity;
out float vDepth;

void main() {
	vec4 view = viewMatrix * modelMatrix * vec4(position, 1.0);
	gl_Position = projectionMatrix * view;
	vNormal = mat3(modelMatrix) * normal;
	vDensity = density;
	vDepth = -view.z;
}
`;

const CAP_FRAGMENT = /* glsl */ `
precision highp float;
precision highp sampler2D;

uniform float eumelanin;
uniform float pheomelanin;
uniform vec3 key;
uniform vec3 keyColour;
uniform vec3 ambient;
uniform sampler2D sceneDepth;
uniform float cameraNear;
uniform float cameraFar;
in vec3 vNormal;
in float vDensity;
in float vDepth;
layout(location = 0) out vec4 outAccumulated;
layout(location = 1) out vec4 outOpticalDepth;

${MELANIN_GLSL}

/** View depth in metres of window depth d. */
float viewDepth(float d) {
	float z = d * 2.0 - 1.0;
	return 2.0 * cameraNear * cameraFar / (cameraFar + cameraNear - z * (cameraFar - cameraNear));
}

void main() {
	// Hidden by the meshes in front, as the hair is; it lies just over the skin.
	ivec2 pixel = ivec2(gl_FragCoord.xy);
	float meshDepth = viewDepth(texelFetch(sceneDepth, pixel, 0).r);
	// Skin seen at a grazing angle lies far behind along the view ray: allow for its slope.
	float slope = max(
		abs(viewDepth(texelFetch(sceneDepth, pixel + ivec2(1, 0), 0).r) - meshDepth),
		abs(viewDepth(texelFetch(sceneDepth, pixel + ivec2(0, 1), 0).r) - meshDepth)
	);
	float tolerance = 0.004 + 2.5 * min(slope, 0.02);
	float alpha = ${OPACITY} * vDensity * (1.0 - smoothstep(tolerance, tolerance + 0.008, vDepth - meshDepth));
	if (alpha <= 0.001) discard;
	// The hair's roots: their colour, in the shade of the hair over them.
	vec3 albedo = exp(-2.0 * melaninAbsorption(eumelanin, pheomelanin));
	float facing = max(dot(normalize(vNormal), key), 0.0);
	vec3 colour = albedo * (0.4 * ambient + 0.15 * keyColour * facing) + 0.01 * ambient;
	// The faintest weight: any hair over it outweighs it.
	float weight = alpha * exp(-9.5);
	outAccumulated = vec4(colour * weight, weight);
	outOpticalDepth = vec4(-log(1.0 - alpha), 0.0, 0.0, 0.0);
}
`;
