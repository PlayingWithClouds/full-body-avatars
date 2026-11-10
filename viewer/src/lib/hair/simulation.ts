/**
 * Guide strands simulated on the GPU, one texel per point (x: point along the guide, y: guide),
 * positions and velocities in world space.
 *
 * Each step integrates gravity, drag and wind, pulls points towards their groomed place (firmly
 * at the root, not at all towards the tips), bends each segment towards its groomed angle to the
 * previous one, and then lays the strand out root to tip with every segment at its rest length
 * (follow-the-leader, Müller et al. 2012, "Fast Simulation of Inextensible Hair and Fur"),
 * pushing each point out of the head and the body capsules as it goes. Follow-the-leader is
 * sequential along a strand, so each texel replays its strand from the root up to itself: with 32
 * points that costs little and keeps the strand exactly inextensible, so it cannot stretch or
 * explode at any time step. The velocity correction of dynamic follow-the-leader then damps the
 * pull the leader exerts.
 */

import * as THREE from 'three';
import { BODY_GLSL } from '#lib/hair/colliders.ts';
import { GUIDE_POINTS, type Groom } from '#lib/hair/groom.ts';
import { computeTarget, FullscreenPass } from '#lib/hair/gpu.ts';
import { OCT_ENCODE_GLSL } from '#lib/hair/scalp.ts';

/** Simulation time step, in seconds; frames run as many as fit their duration. */
export const STEP = 1 / 120;

export interface Dynamics {
	/** Bending stiffness, 0 to 1, from the groom. */
	stiffness: number;
	/** Wind strength in metres per second. */
	wind: number;
}

export class GuideSimulation {
	private targets: [THREE.WebGLRenderTarget, THREE.WebGLRenderTarget];
	private readonly step: FullscreenPass;
	private readonly place: FullscreenPass;
	private time = 0;

	/**
	 * `body` holds the uniforms of `BODY_GLSL`, which the caller keeps on the avatar's pose; the
	 * scalp's radius map is `field`, around `centre` in the head's frame.
	 */
	constructor(
		groom: Groom,
		rest: THREE.DataTexture,
		field: THREE.Texture,
		centre: THREE.Vector3,
		body: Record<string, THREE.IUniform>
	) {
		const make = () => computeTarget(GUIDE_POINTS, groom.guides, 2);
		this.targets = [make(), make()];
		const shared = {
			rest: { value: rest },
			field: { value: field },
			centre: { value: centre },
			head: { value: new THREE.Matrix4() },
			headInverse: { value: new THREE.Matrix4() }
		};
		this.step = new FullscreenPass(STEP_FRAGMENT, {
			...shared,
			positions: { value: null },
			velocities: { value: null },
			...body,
			dt: { value: STEP },
			bend: { value: 0.1 },
			pin: { value: 0.5 },
			gravity: { value: new THREE.Vector3(0, 0, -9.81) },
			wind: { value: new THREE.Vector3() },
			time: { value: 0 }
		});
		this.place = new FullscreenPass(PLACE_FRAGMENT, {
			...shared,
			positions: { value: null },
			carry: { value: new THREE.Matrix4() },
			mode: { value: 0 }
		});
	}

	/** `(GUIDE_POINTS, guides)` world positions of the guides' points. */
	get positions(): THREE.Texture {
		return this.targets[0].textures[0];
	}

	/** Put every guide in its groomed place on head transform `head`, at rest. */
	reset(renderer: THREE.WebGLRenderer, head: THREE.Matrix4): void {
		this.setHead(this.place, head);
		this.place.uniforms.mode.value = 0;
		this.runPlace(renderer);
	}

	/** Move the hair rigidly by `carry` (world to world) and stop it, as when the avatar jumps. */
	carry(renderer: THREE.WebGLRenderer, carry: THREE.Matrix4): void {
		this.place.uniforms.carry.value.copy(carry);
		this.place.uniforms.mode.value = 1;
		this.runPlace(renderer);
	}

	/** Advance one step with the head at `head`. */
	advance(renderer: THREE.WebGLRenderer, head: THREE.Matrix4, dynamics: Dynamics): void {
		const u = this.step.uniforms;
		this.setHead(this.step, head);
		// Stiffer hair bends back faster; per step at 120 Hz.
		u.bend.value = 0.03 + 0.45 * dynamics.stiffness ** 2;
		u.pin.value = 0.25 + 0.5 * dynamics.stiffness;
		this.time += STEP;
		u.time.value = this.time;
		// Wind from the front and a little from the avatar's right, gusting.
		const gust = 0.7 + 0.3 * Math.sin(this.time * 0.9) * Math.sin(this.time * 2.3 + 1);
		u.wind.value
			.set(-0.25, 1, 0.1)
			.normalize()
			.multiplyScalar(dynamics.wind * gust);
		u.positions.value = this.targets[0].textures[0];
		u.velocities.value = this.targets[0].textures[1];
		this.step.run(renderer, this.targets[1]);
		this.targets.reverse();
	}

	dispose(): void {
		for (const target of this.targets) target.dispose();
		this.step.dispose();
		this.place.dispose();
	}

	private runPlace(renderer: THREE.WebGLRenderer): void {
		this.place.uniforms.positions.value = this.targets[0].textures[0];
		this.place.run(renderer, this.targets[1]);
		this.targets.reverse();
	}

	private setHead(pass: FullscreenPass, head: THREE.Matrix4): void {
		pass.uniforms.head.value.copy(head);
		pass.uniforms.headInverse.value.copy(head).invert();
	}
}

const COMMON = /* glsl */ `
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D rest;
uniform sampler2D field;
uniform vec3 centre;
uniform mat4 head;
uniform mat4 headInverse;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outVelocity;

${OCT_ENCODE_GLSL}

/** Head radius along head-frame direction d, 0 below the jaw. */
float headRadius(vec3 d) {
	return texture(field, octEncode(d)).r;
}
`;

const STEP_FRAGMENT = /* glsl */ `
${COMMON}
uniform sampler2D positions;
uniform sampler2D velocities;
uniform float dt;
uniform float bend;
uniform float pin;
uniform vec3 gravity;
uniform vec3 wind;
uniform float time;

${BODY_GLSL}

const int POINTS = ${GUIDE_POINTS};
/** Share of the leader's correction taken back out of the velocity (dynamic FTL). */
const float FTL_DAMPING = 0.9;
const float MAX_SPEED = 6.0;

vec3 collideHead(vec3 p, float clearance, inout bool hit) {
	vec3 q = (headInverse * vec4(p, 1.0)).xyz - centre;
	float r = length(q);
	vec3 d = q / max(r, 1e-6);
	float surface = headRadius(d);
	if (surface > 0.0 && r < surface + clearance) {
		p = (head * vec4(centre + d * (surface + clearance), 1.0)).xyz;
		hit = true;
	}
	return p;
}

/** v turned by the rotation taking unit vector a to unit vector b. */
vec3 turn(vec3 a, vec3 b, vec3 v) {
	vec3 k = cross(a, b);
	float c = dot(a, b);
	if (c < -0.999) return -v;
	return v * c + cross(k, v) + k * dot(k, v) / (1.0 + c);
}

/** How far above the scalp a point groomed at head-frame position r stays. */
float clearanceOf(vec3 r) {
	vec3 q = r - centre;
	float length_ = length(q);
	float surface = headRadius(q / max(length_, 1e-6));
	return clamp(length_ - surface, 0.0015, 0.02);
}

void main() {
	int i = int(gl_FragCoord.x);
	int g = int(gl_FragCoord.y);
	vec3 current = texelFetch(positions, ivec2(i, g), 0).xyz;
	vec4 rest0 = texelFetch(rest, ivec2(0, g), 0);
	vec3 previous = (head * vec4(rest0.xyz, 1.0)).xyz;
	if (i == 0) {
		outPosition = vec4(previous, 1.0);
		outVelocity = vec4(0.0);
		return;
	}
	vec3 restPrevious = rest0.xyz;
	vec3 restDirectionBefore = vec3(0.0);
	vec3 directionBefore = vec3(0.0);
	vec3 result = current;
	vec3 correction = vec3(0.0);
	bool collided = false;
	mat3 rotation = mat3(head);

	for (int j = 1; j < POINTS; j++) {
		if (j > i + 1) break;
		vec4 r = texelFetch(rest, ivec2(j, g), 0);
		vec3 x = texelFetch(positions, ivec2(j, g), 0).xyz;
		vec3 v = texelFetch(velocities, ivec2(j, g), 0).xyz;
		vec3 groomed = (head * vec4(r.xyz, 1.0)).xyz;

		// Air drags the hair towards the wind's speed (still air: to a stop), unevenly along it.
		float flutter = 0.6 + 0.4 * sin(time * 3.1 + x.z * 7.0 + x.x * 5.0 + float(g) * 0.37);
		vec3 force = gravity + 1.5 * (wind * flutter - v);
		vec3 predicted = x + dt * (v + force * dt);
		vec3 p = mix(predicted, groomed, r.w * pin);

		vec3 restSegment = rotation * (r.xyz - restPrevious);
		float segment = length(restSegment);
		vec3 restDirection = restSegment / max(segment, 1e-6);
		vec3 wanted = j >= 2 ? turn(restDirectionBefore, directionBefore, restDirection) : restDirection;
		vec3 toward = p - previous;
		float length_ = length(toward);
		vec3 direction = length_ > 1e-7 ? toward / length_ : wanted;
		direction = normalize(mix(direction, wanted, j == 1 ? 1.0 : bend));
		p = previous + direction * segment;

		bool hit = false;
		p = collideHead(p, clearanceOf(r.xyz), hit);
		p = collideBody(p, 0.005, hit);
		p = previous + normalize(p - previous) * segment;

		if (j == i) {
			result = p;
			collided = hit;
		}
		if (j == i + 1) correction = p - predicted;
		restDirectionBefore = restDirection;
		directionBefore = normalize(p - previous);
		restPrevious = r.xyz;
		previous = p;
	}

	vec3 velocity = (result - current) / dt - FTL_DAMPING * correction / dt;
	// Hair resting on skin, or on the hair under it, grips: it barely slides.
	if (collided) velocity *= 0.5;
	float speed = length(velocity);
	if (speed > MAX_SPEED) velocity *= MAX_SPEED / speed;
	if (any(isnan(result)) || any(isnan(velocity))) {
		result = (head * vec4(texelFetch(rest, ivec2(i, g), 0).xyz, 1.0)).xyz;
		velocity = vec3(0.0);
	}
	outPosition = vec4(result, 1.0);
	outVelocity = vec4(velocity, 0.0);
}
`;

/** Reset (mode 0: groomed place on the head) or carry (mode 1: moved rigidly), at rest. */
const PLACE_FRAGMENT = /* glsl */ `
${COMMON}
uniform sampler2D positions;
uniform mat4 carry;
uniform int mode;

void main() {
	ivec2 texel = ivec2(gl_FragCoord.xy);
	vec3 p = mode == 0
		? (head * vec4(texelFetch(rest, texel, 0).xyz, 1.0)).xyz
		: (carry * vec4(texelFetch(positions, texel, 0).xyz, 1.0)).xyz;
	outPosition = vec4(p, 1.0);
	outVelocity = vec4(0.0);
}
`;
