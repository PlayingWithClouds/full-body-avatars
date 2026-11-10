/**
 * The eyes, drawn by a material of their own instead of from the baked texture: averaged over
 * frames that look different ways and blink, the texture ghosts. The file's `eyes` entry gives
 * the iris' colour and size measured from the head video; the iris is drawn around each
 * eyeball's front, with fibres, a lighter ring at the pupil and a dark limbal ring, the sclera
 * around it pinker towards the corners and shaded under the lids, and a glossy cornea over both
 * that catches the studio's lights.
 *
 * Each eyeball turns with its bone and the head, so where it points is read from the posed
 * vertices: its centre, and two of its vertices, its front (`apex`) and its top. Copies of the
 * avatar share the material and set its uniforms to their own eyes just before they are drawn.
 */

import * as THREE from 'three';
import type { AvatarFile, EyeMetadata } from '#lib/avatar/file.ts';
import { LIGHTING } from '#lib/render/lighting.ts';

/** The pupil's radius in shares of the iris'. */
const PUPIL = 0.42;
/** The cornea's radius of curvature in shares of the eyeball's: it bulges out of it. */
const CORNEA = 0.62;
/** Half the eye opening's width in iris radii, from the iris' measured share of the eye's width. */
const OPENING_WIDTH = 2.3;

/** Where each eyeball is, found from the rest pose: its vertices and the points that turn it. */
export class EyeballRig {
	/** Per eyeball, its vertices. */
	private readonly vertices: Uint32Array[];
	/** Per eyeball, its centre minus the mean of its vertices, in the rest pose. */
	private readonly offsets: THREE.Vector3[];
	readonly apex: number[];
	readonly top: number[];
	readonly radius: number;

	constructor(file: AvatarFile, eyes: EyeMetadata) {
		const rest = file.arrays.rest_vertices.data as Float32Array;
		const faces = file.arrays.faces.data as Uint32Array;
		const parts = file.arrays.face_parts.data as Uint8Array;
		const eyePart = file.metadata.parts?.indexOf('eyes') ?? -1;
		const centres = eyes.eyeballs.map((eyeball) => new THREE.Vector3(...eyeball.centre));
		const sets = centres.map(() => new Set<number>());
		const point = new THREE.Vector3();
		for (let f = 0; f < parts.length; f++) {
			if (parts[f] !== eyePart) continue;
			for (let c = 0; c < 3; c++) {
				const v = faces[3 * f + c];
				point.fromArray(rest, 3 * v);
				const nearest = point.distanceTo(centres[0]) < point.distanceTo(centres[1]) ? 0 : 1;
				sets[nearest].add(v);
			}
		}
		this.vertices = sets.map((set) => Uint32Array.from(set));
		this.offsets = centres.map((centre, eye) => centre.clone().sub(mean(rest, this.vertices[eye])));
		this.apex = eyes.eyeballs.map((eyeball) => eyeball.apex);
		this.top = eyes.eyeballs.map((eyeball) => eyeball.top);
		this.radius = eyes.eyeballs.reduce((sum, eyeball) => sum + eyeball.radius, 0) / 2;
	}

	/** The posed eyeballs' frames: centre, unit forward and up, written into `frames`. */
	pose(posed: Float32Array, frames: EyeFrames): void {
		const point = new THREE.Vector3();
		for (let eye = 0; eye < 2; eye++) {
			// The offset from the vertices' mean to the centre turns with the eye; at the turns
			// an eye makes that moves the centre by a fraction of a millimetre.
			const centre = frames.centres[eye]
				.copy(mean(posed, this.vertices[eye]))
				.add(this.offsets[eye]);
			const forward = frames.forwards[eye]
				.fromArray(posed, 3 * this.apex[eye])
				.sub(centre)
				.normalize();
			point.fromArray(posed, 3 * this.top[eye]).sub(centre);
			frames.ups[eye].copy(point.addScaledVector(forward, -point.dot(forward)).normalize());
		}
	}
}

export interface EyeFrames {
	centres: [THREE.Vector3, THREE.Vector3];
	forwards: [THREE.Vector3, THREE.Vector3];
	ups: [THREE.Vector3, THREE.Vector3];
}

export function eyeFrames(): EyeFrames {
	const two = () => [new THREE.Vector3(), new THREE.Vector3()] as [THREE.Vector3, THREE.Vector3];
	return { centres: two(), forwards: two(), ups: two() };
}

/** The eyes' material; `use` points it at one copy's eyes before that copy is drawn. */
export function eyeMaterial(eyes: EyeMetadata, rig: EyeballRig): EyeMaterial {
	const frames = eyeFrames();
	const material = new THREE.ShaderMaterial({
		uniforms: {
			eyeCentre: { value: frames.centres },
			eyeForward: { value: frames.forwards },
			eyeUp: { value: frames.ups },
			eyeRadius: { value: rig.radius },
			irisAngle: { value: eyes.iris_angle },
			irisColour: { value: new THREE.Vector3(...eyes.iris_colour) },
			scleraColour: { value: new THREE.Vector3(...eyes.sclera_colour) },
			keyDirection: { value: LIGHTING.key },
			fillDirection: { value: LIGHTING.fill }
		},
		vertexShader: VERTEX,
		fragmentShader: FRAGMENT,
		side: THREE.FrontSide
	}) as EyeMaterial;
	material.use = (own: EyeFrames) => {
		for (let eye = 0; eye < 2; eye++) {
			frames.centres[eye].copy(own.centres[eye]);
			frames.forwards[eye].copy(own.forwards[eye]);
			frames.ups[eye].copy(own.ups[eye]);
		}
		// Copies share the material: three.js uploads its uniforms again only when told to.
		material.uniformsNeedUpdate = true;
	};
	return material;
}

export type EyeMaterial = THREE.ShaderMaterial & { use(frames: EyeFrames): void };

function mean(positions: Float32Array, vertices: Uint32Array): THREE.Vector3 {
	const sum = new THREE.Vector3();
	for (const v of vertices) {
		sum.x += positions[3 * v];
		sum.y += positions[3 * v + 1];
		sum.z += positions[3 * v + 2];
	}
	return sum.divideScalar(vertices.length);
}

const VERTEX = /* glsl */ `
varying vec3 vWorld;
void main() {
	vec4 world = modelMatrix * vec4(position, 1.0);
	vWorld = world.xyz;
	gl_Position = projectionMatrix * viewMatrix * world;
}
`;

const FRAGMENT = /* glsl */ `
uniform vec3 eyeCentre[2];
uniform vec3 eyeForward[2];
uniform vec3 eyeUp[2];
uniform float eyeRadius;
uniform float irisAngle;
uniform vec3 irisColour;
uniform vec3 scleraColour;
uniform vec3 keyDirection;
uniform vec3 fillDirection;
varying vec3 vWorld;

const float PUPIL = ${PUPIL.toFixed(3)};
const float CORNEA = ${CORNEA.toFixed(3)};
const float OPENING_WIDTH = ${OPENING_WIDTH.toFixed(3)};

float hash(vec2 p) {
	return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
	vec2 i = floor(p);
	vec2 f = fract(p);
	vec2 u = f * f * (3.0 - 2.0 * f);
	return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
		mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

void main() {
	bool first = distance(vWorld, eyeCentre[0]) < distance(vWorld, eyeCentre[1]);
	vec3 centre = first ? eyeCentre[0] : eyeCentre[1];
	vec3 forward = first ? eyeForward[0] : eyeForward[1];
	vec3 up = first ? eyeUp[0] : eyeUp[1];
	vec3 right = cross(forward, up);

	vec3 normal = normalize(vWorld - centre);
	float t = acos(clamp(dot(normal, forward), -1.0, 1.0)) / irisAngle;
	float phi = atan(dot(normal, up), dot(normal, right));
	vec2 around = vec2(cos(phi), sin(phi));

	// The iris: fibres running out from the pupil, crypts between them.
	float fibres = 0.6 * noise(around * 45.0 + vec2(t * 2.5)) + 0.4 * noise(around * 14.0 - vec2(t * 4.0));
	vec3 iris = irisColour * mix(0.65, 1.35, fibres);
	float collarette = smoothstep(PUPIL, PUPIL + 0.08, t) * (1.0 - smoothstep(PUPIL + 0.12, PUPIL + 0.32, t));
	iris = mix(iris, iris * 1.3 + vec3(0.05, 0.04, 0.0), 0.6 * collarette);
	iris *= mix(1.0, 0.45, smoothstep(0.78, 1.0, t));
	iris = mix(vec3(0.015), iris, smoothstep(PUPIL - 0.03, PUPIL + 0.02, t));

	// The sclera, measured in iris radii from the front: the eye's opening reaches about one
	// above and below and OPENING_WIDTH to the sides. Brightest beside the iris, pinker towards
	// the corners, and shaded in a soft band along the lids, which hang over it there.
	float sideways = asin(clamp(abs(dot(normal, right)), 0.0, 1.0)) / irisAngle;
	float height = asin(clamp(dot(normal, up), -1.0, 1.0)) / irisAngle;
	float opening = length(vec2(sideways / OPENING_WIDTH, height));
	vec3 sclera = scleraColour * 1.06;
	sclera = mix(sclera, sclera * vec3(1.0, 0.84, 0.82), 0.6 * smoothstep(1.6, 2.6, sideways));
	float lids = 1.0 - 0.5 * smoothstep(0.55, 1.0, opening);
	vec3 colour = mix(iris, sclera, smoothstep(0.97, 1.05, t)) * mix(1.0, lids, 0.6 + 0.4 * step(1.0, t));

	// A wet cornea over both: sharp catchlights of the key and fill light, and a sheen at grazing
	// angles. Over the iris it curves more tightly than the eyeball.
	vec3 corneaCentre = centre + forward * eyeRadius * (1.0 - CORNEA);
	vec3 surface = normalize(mix(normalize(vWorld - corneaCentre), normal, smoothstep(0.9, 1.15, t)));
	vec3 view = normalize(cameraPosition - vWorld);
	float facing = max(dot(surface, view), 0.0);
	float fresnel = 0.02 + 0.98 * pow(1.0 - facing, 5.0);
	float key = pow(max(dot(surface, normalize(keyDirection + view)), 0.0), 900.0);
	float fill = pow(max(dot(surface, normalize(fillDirection + view)), 0.0), 600.0);
	float sheen = pow(max(dot(surface, normalize(keyDirection + view)), 0.0), 40.0);
	colour += vec3(1.0, 0.98, 0.95) * (1.2 * key + 0.35 * fill + 0.04 * sheen) + 0.2 * fresnel;

	gl_FragColor = vec4(min(colour, vec3(1.0)), 1.0);
}
`;
