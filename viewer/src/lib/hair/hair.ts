/**
 * Strand hair on the viewer's avatars, as a scene layer: one groom grown on the avatar's scalp,
 * worn by every copy of the avatar, each copy simulating and drawing its own.
 *
 * Every frame, each copy's guides follow its head bone and simulate on the GPU; its render strands
 * are laid out from them, and a deep opacity map is drawn from the key light for the hair's
 * shadows. When the viewer draws, the layer's full-screen quad, drawn after everything else,
 * first draws all hair into float targets of its own, against the depth of the avatars' meshes
 * (so the head and body hide the hair behind them), and then lays the result over the frame. The
 * Gaussians are drawn by then, so hair in front of them covers them, as it should, and the
 * quad also darkens the skin in the hair's shadow.
 */

import * as THREE from 'three';
import type { Rig } from '#lib/avatar/rig.ts';
import {
	MAX_CAPSULES,
	measureBody,
	placeCapsules,
	TORSO_ANGLES,
	TORSO_HEIGHTS,
	torsoFrame,
	type Body
} from '#lib/hair/colliders.ts';
import {
	GUIDE_POINTS,
	growGroom,
	type Groom,
	type GroomShape,
	type HairLook
} from '#lib/hair/groom.ts';
import { capGeometry, capMaterial } from '#lib/hair/cap.ts';
import { dataTexture, matrix, preserving } from '#lib/hair/gpu.ts';
import { buildScalp, FIELD_SIZE, type Scalp } from '#lib/hair/scalp.ts';
import { opacityMapGlsl } from '#lib/hair/shading.ts';
import { GuideSimulation, STEP } from '#lib/hair/simulation.ts';
import { pigment, ribbonGeometry, ROW, Strands, type GroomTextures } from '#lib/hair/strands.ts';
import { LIGHTING } from '#lib/render/lighting.ts';
import {
	MESH_DEPTH,
	type AvatarViewer,
	type PosedAvatar,
	type SceneLayer
} from '#lib/render/viewer.ts';
import type { Mat4 } from '#lib/math.ts';

/** Deep opacity map resolution. */
const OPACITY_SIZE = 1024;
/** Steps per frame at most: a slow frame drops time rather than falling behind. */
const MAX_STEPS = 4;
/** Seconds the hair keeps simulating after the avatar last moved. */
const SETTLE = 4;
/** Copies whose hair shadows the skin. */
const SHADOWING = 2;

export interface HairStats {
	guides: number;
	strands: number;
	pointsPerStrand: number;
	/** Milliseconds the CPU spent growing the current groom. */
	growMs: number;
}

/** What a groom is grown on: the avatar's scalp and body in its bind pose. */
interface Anatomy {
	scalp: Scalp;
	body: Body;
	/** The torso's height field, for the shaders. */
	torso: THREE.DataTexture;
	bind: Mat4[];
	head: number;
}

export class HairLayer implements SceneLayer {
	readonly object = new THREE.Group();
	private readonly anatomy: Anatomy;
	private groom: Groom | null = null;
	private textures: GroomTextures | null = null;
	private geometry: THREE.InstancedBufferGeometry | null = null;
	private capGeometry: THREE.BufferGeometry | null = null;
	private readonly capMaterial = capMaterial();
	private readonly field: THREE.DataTexture;
	private instances: HairInstance[] = [];
	private readonly hairScene = new THREE.Scene();
	private readonly accumulation: THREE.WebGLRenderTarget;
	/** The meshes' depth as the camera sees them. */
	private readonly sceneDepth: THREE.WebGLRenderTarget;
	private readonly meshDepth = new THREE.MeshBasicMaterial({ colorWrite: false });
	private readonly composite: THREE.Mesh<THREE.BufferGeometry, THREE.RawShaderMaterial>;
	private look: HairLook;
	private simulate = true;
	private wind = 0;
	private growMs = 0;
	/** Something changed that the next frame must show. */
	private changed = true;
	/** The look changed: the strands' lighting, which depends on their pigment, is stale. */
	private relight = false;

	constructor(
		private readonly viewer: AvatarViewer,
		shape: GroomShape,
		look: HairLook
	) {
		this.anatomy = measure(viewer.rig);
		this.field = new THREE.DataTexture(
			this.anatomy.scalp.field,
			FIELD_SIZE,
			FIELD_SIZE,
			THREE.RedFormat,
			THREE.FloatType
		);
		this.field.minFilter = this.field.magFilter = THREE.LinearFilter;
		this.field.needsUpdate = true;
		this.look = look;

		// Half floats: hair blends hundreds of fragments into each pixel, and blending is what
		// drawing it costs most.
		this.accumulation = new THREE.WebGLRenderTarget(1, 1, {
			count: 2,
			type: THREE.HalfFloatType,
			minFilter: THREE.NearestFilter,
			magFilter: THREE.NearestFilter,
			depthBuffer: false
		});
		this.accumulation.textures[1].format = THREE.RedFormat;
		this.sceneDepth = new THREE.WebGLRenderTarget(1, 1, {
			depthTexture: new THREE.DepthTexture(1, 1, THREE.FloatType)
		});
		this.composite = compositeQuad(this.accumulation, this.sceneDepth.depthTexture!);
		this.composite.onBeforeRender = (renderer, _scene, camera) => this.drawHair(renderer, camera);
		this.object.add(this.composite);
		const cap = this.capMaterial.uniforms;
		cap.key.value.copy(LIGHTING.key);
		cap.keyColour.value.copy(LIGHTING.keyColour);
		cap.ambient.value.copy(LIGHTING.ambient);
		this.setShape(shape);
		this.setLook(look);
	}

	/** Grow the hair anew with `shape`. */
	setShape(shape: GroomShape): void {
		const start = performance.now();
		const { scalp, body, bind, head } = this.anatomy;
		const groom = growGroom(shape, scalp, body, bind, bind[head]);
		this.growMs = performance.now() - start;
		this.disposeInstances();
		this.textures?.rest.dispose();
		this.textures?.strands.dispose();
		this.geometry?.dispose();
		this.capGeometry?.dispose();
		this.capGeometry = capGeometry(scalp, shape);
		this.groom = groom;
		const rows = Math.ceil((groom.strands * 4) / ROW);
		this.textures = {
			rest: dataTexture(groom.rest, GUIDE_POINTS, groom.guides),
			strands: dataTexture(groom.strandData, ROW, rows),
			field: this.field,
			centre: new THREE.Vector3(...scalp.centre)
		};
		this.geometry = ribbonGeometry(groom);
		this.changed = true;
	}

	setLook(look: HairLook): void {
		this.look = look;
		for (const instance of this.instances) instance.strands.setLook(look);
		this.composite.material.uniforms.shadowStrength.value = look.shadow;
		const { eumelanin, pheomelanin } = pigment(look);
		this.capMaterial.uniforms.eumelanin.value = eumelanin;
		this.capMaterial.uniforms.pheomelanin.value = pheomelanin;
		this.changed = true;
		this.relight = true;
	}

	setSimulation(simulate: boolean, wind: number): void {
		this.simulate = simulate;
		this.wind = wind;
		for (const instance of this.instances) instance.restless = SETTLE;
		this.changed = true;
	}

	/** Put the hair back in its groomed place. */
	reset(): void {
		for (const instance of this.instances) instance.placed = false;
		this.changed = true;
	}

	stats(): HairStats {
		return {
			guides: this.groom?.guides ?? 0,
			strands: this.groom?.strands ?? 0,
			pointsPerStrand: this.groom?.renderPoints ?? 0,
			growMs: this.growMs
		};
	}

	update(seconds: number, avatars: readonly PosedAvatar[]): boolean {
		const { groom, textures, geometry, capGeometry } = this;
		if (!groom || !textures || !geometry || !capGeometry) return false;
		while (this.instances.length > avatars.length) this.instances.pop()?.dispose(this.hairScene);
		while (this.instances.length < avatars.length) {
			const cap = new THREE.Mesh(capGeometry, this.capMaterial);
			const instance = new HairInstance(groom, textures, geometry, cap, this.anatomy);
			instance.strands.setLook(this.look);
			instance.strands.setLighting(LIGHTING);
			this.hairScene.add(instance.strands.mesh, cap);
			this.instances.push(instance);
			this.changed = true;
		}
		const renderer = this.viewer.renderer;
		let changed = this.changed;
		const relight = this.relight;
		this.changed = this.relight = false;
		preserving(renderer, () => {
			for (let i = 0; i < avatars.length; i++) {
				const moved = this.instances[i].update(renderer, seconds, avatars[i], {
					simulate: this.simulate,
					wind: this.wind,
					stiffness: groom.shape.stiffness
				});
				if (moved) {
					this.instances[i].drawOpacityMap(renderer, this.hairScene, this.instances);
					changed = true;
				} else if (relight) this.instances[i].strands.light(renderer);
			}
		});
		return changed;
	}

	dispose(): void {
		this.disposeInstances();
		this.textures?.rest.dispose();
		this.textures?.strands.dispose();
		this.geometry?.dispose();
		this.capGeometry?.dispose();
		this.capMaterial.dispose();
		this.field.dispose();
		this.anatomy.torso.dispose();
		this.accumulation.dispose();
		this.sceneDepth.depthTexture?.dispose();
		this.sceneDepth.dispose();
		this.meshDepth.dispose();
		this.composite.geometry.dispose();
		this.composite.material.dispose();
	}

	private disposeInstances(): void {
		for (const instance of this.instances) instance.dispose(this.hairScene);
		this.instances = [];
	}

	/** Draw all hair into the accumulation targets, just before the composite shows them. */
	private drawHair(renderer: THREE.WebGLRenderer, camera: THREE.Camera): void {
		const uniforms = this.composite.material.uniforms;
		uniforms.count.value = 0;
		if (this.instances.length === 0) return;
		const size = renderer.getDrawingBufferSize(new THREE.Vector2());
		if (this.accumulation.width !== size.x || this.accumulation.height !== size.y) {
			this.accumulation.setSize(size.x, size.y);
			this.sceneDepth.setSize(size.x, size.y);
		}
		const scene = this.viewer.scene;
		preserving(renderer, () => {
			renderer.setRenderTarget(this.sceneDepth);
			renderer.setClearColor(0x000000, 0);
			renderer.autoClear = false;
			renderer.clear(true, true, false);

			// The meshes' depth, so they hide the hair behind them; shown or not.
			const hidden: THREE.Object3D[] = [];
			scene.traverse((object) => {
				if (object.layers.isEnabled(MESH_DEPTH) && !object.visible) {
					object.visible = true;
					hidden.push(object);
				}
			});
			const mask = camera.layers.mask;
			const background = scene.background;
			camera.layers.set(MESH_DEPTH);
			scene.overrideMaterial = this.meshDepth;
			scene.background = null;
			renderer.render(scene, camera);
			scene.overrideMaterial = null;
			scene.background = background;
			camera.layers.mask = mask;
			for (const object of hidden) object.visible = false;

			const eye = camera.getWorldPosition(new THREE.Vector3());
			const nearest = Math.min(...this.instances.map((instance) => instance.distance(eye)));
			for (const instance of this.instances) {
				instance.strands.prepareShade(
					camera as THREE.PerspectiveCamera,
					size.y,
					this.sceneDepth.depthTexture!,
					nearest
				);
			}
			const cap = this.capMaterial.uniforms;
			cap.sceneDepth.value = this.sceneDepth.depthTexture;
			cap.cameraNear.value = (camera as THREE.PerspectiveCamera).near;
			cap.cameraFar.value = (camera as THREE.PerspectiveCamera).far;
			renderer.setRenderTarget(this.accumulation);
			renderer.clear(true, false, false);
			renderer.render(this.hairScene, camera);
		});

		uniforms.inverseViewProjection.value
			.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)
			.invert();
		this.instances.slice(0, SHADOWING).forEach((instance, i) => {
			uniforms[`opacityDepth${i}`].value = instance.opacityDepth.depthTexture;
			uniforms[`opacityLayers${i}`].value = instance.opacityLayers.texture;
			uniforms[`lightMatrix${i}`].value.multiplyMatrices(
				instance.light.projectionMatrix,
				instance.light.matrixWorldInverse
			);
			uniforms[`lightRange${i}`].value = instance.light.far - instance.light.near;
		});
		uniforms.count.value = Math.min(SHADOWING, this.instances.length);
	}
}

/** One avatar's hair: its simulation, render strands and shadow map. */
class HairInstance {
	readonly strands: Strands;
	readonly opacityDepth: THREE.WebGLRenderTarget;
	readonly opacityLayers: THREE.WebGLRenderTarget;
	readonly light = new THREE.OrthographicCamera();
	/** Seconds of simulation left before the hair counts as settled. */
	restless = SETTLE;
	/** Whether the guides have been put on the head since the groom or a reset. */
	placed = false;
	private readonly simulation: GuideSimulation;
	private readonly head = new THREE.Matrix4();
	private readonly previousHead = new THREE.Matrix4();
	/** The uniforms of `BODY_GLSL`, on this copy's pose. */
	private readonly body: Record<string, THREE.IUniform>;
	/** The body's colliders as last placed, to tell whether the avatar moved. */
	private readonly pose = new Float32Array(8 * MAX_CAPSULES + 16);
	private readonly previousPose = new Float32Array(8 * MAX_CAPSULES + 16);
	private time = 0;

	constructor(
		private readonly groom: Groom,
		textures: GroomTextures,
		geometry: THREE.InstancedBufferGeometry,
		/** The hair-coloured scalp under this copy's hair. */
		readonly cap: THREE.Mesh,
		private readonly anatomy: Anatomy
	) {
		cap.matrixAutoUpdate = cap.matrixWorldAutoUpdate = false;
		cap.frustumCulled = false;
		const torso = anatomy.body.torso;
		this.body = {
			capsules: { value: new Float32Array(8 * MAX_CAPSULES) },
			capsuleCount: { value: anatomy.body.capsules.length },
			torsoField: { value: anatomy.torso },
			torsoFrame: { value: new THREE.Matrix4() },
			torsoFrameInverse: { value: new THREE.Matrix4() },
			// Top below bottom: no torso to collide with.
			torsoShape: { value: new THREE.Vector4(...(torso?.shape ?? [0, 0, -1, 1])) }
		};
		this.simulation = new GuideSimulation(
			groom,
			textures.rest,
			textures.field,
			textures.centre,
			this.body
		);
		this.strands = new Strands(groom, textures, geometry, this.body);
		this.opacityDepth = new THREE.WebGLRenderTarget(OPACITY_SIZE, OPACITY_SIZE, {
			type: THREE.UnsignedByteType,
			depthTexture: new THREE.DepthTexture(OPACITY_SIZE, OPACITY_SIZE, THREE.FloatType)
		});
		this.opacityLayers = new THREE.WebGLRenderTarget(OPACITY_SIZE, OPACITY_SIZE, {
			type: THREE.HalfFloatType,
			depthBuffer: false,
			// Mipmapped, for the skin's shadow: averaged over many strands, not speckled by each.
			generateMipmaps: true,
			minFilter: THREE.LinearMipmapLinearFilter,
			magFilter: THREE.LinearFilter
		});
	}

	/** Distance from `eye` to the nearest of the hair, roughly. */
	distance(eye: THREE.Vector3): number {
		const centre = new THREE.Vector3().setFromMatrixPosition(this.head);
		return Math.max(0, eye.distanceTo(centre) - this.groom.reach - 0.15);
	}

	/** Follow `avatar` for `seconds`; returns whether the hair moved. */
	update(
		renderer: THREE.WebGLRenderer,
		seconds: number,
		avatar: PosedAvatar,
		settings: { simulate: boolean; wind: number; stiffness: number }
	): boolean {
		this.previousHead.copy(this.head);
		this.previousPose.set(this.pose);
		matrix(avatar.boneWorld[this.anatomy.head], this.head);
		this.cap.matrix.copy(this.head);
		this.cap.matrixWorld.copy(this.head);
		const capsules = this.body.capsules.value as Float32Array;
		placeCapsules(this.anatomy.body.capsules, avatar.boneWorld, capsules);
		this.pose.set(capsules);
		const torso = this.anatomy.body.torso;
		if (torso) {
			const frame = torsoFrame(torso, avatar.boneWorld);
			matrix(frame, this.body.torsoFrame.value);
			this.body.torsoFrameInverse.value.copy(this.body.torsoFrame.value).invert();
			this.pose.set(frame, capsules.length);
		}
		const dynamics = { stiffness: settings.stiffness, wind: settings.wind };

		if (!this.placed) {
			this.placed = true;
			this.simulation.reset(renderer, this.head);
			if (settings.simulate) {
				// Let the groom settle under gravity before anyone sees it.
				for (let i = 0; i < 90; i++) {
					this.simulation.advance(renderer, this.head, dynamics);
				}
			}
			this.restless = SETTLE;
			this.lay(renderer);
			return true;
		}

		const moved = !this.head.equals(this.previousHead) || !same(this.pose, this.previousPose);
		if (moved) this.restless = SETTLE;
		if (moved && jumped(this.previousHead, this.head)) {
			// A cut or a teleport: carry the hair along as it lies rather than fling it.
			const carry = this.head.clone().multiply(this.previousHead.clone().invert());
			this.simulation.carry(renderer, carry);
		}

		if (!settings.simulate) {
			if (!moved) return false;
			this.simulation.reset(renderer, this.head);
			this.lay(renderer);
			return true;
		}
		if (this.restless <= 0 && settings.wind <= 0) return false;
		this.restless -= seconds;

		this.time += seconds;
		const steps = Math.min(MAX_STEPS, Math.floor(this.time / STEP));
		this.time = Math.min(this.time - steps * STEP, STEP);
		if (steps === 0) return false;
		const from = decompose(this.previousHead);
		const to = decompose(this.head);
		const head = new THREE.Matrix4();
		for (let k = 1; k <= steps; k++) {
			const t = k / steps;
			head.compose(
				from.position.clone().lerp(to.position, t),
				from.rotation.clone().slerp(to.rotation, t),
				new THREE.Vector3(1, 1, 1)
			);
			this.simulation.advance(renderer, head, dynamics);
		}
		this.lay(renderer);
		return true;
	}

	/** Draw this copy's deep opacity map from the key light; other copies are hidden meanwhile. */
	drawOpacityMap(
		renderer: THREE.WebGLRenderer,
		scene: THREE.Scene,
		instances: HairInstance[]
	): void {
		const centre = new THREE.Vector3().setFromMatrixPosition(this.head);
		centre.z -= 0.35 * this.groom.reach;
		const radius = 0.14 + 0.75 * this.groom.reach;
		const light = this.light;
		light.left = light.bottom = -radius;
		light.right = light.top = radius;
		light.near = 0.05;
		light.far = 2 * radius + 0.1;
		light.position.copy(centre).addScaledVector(LIGHTING.key, radius + 0.05);
		light.up.set(0, 0, 1);
		light.lookAt(centre);
		light.updateProjectionMatrix();
		light.updateMatrixWorld();

		for (const instance of instances) {
			instance.strands.mesh.visible = instance === this;
			instance.cap.visible = false;
		}
		renderer.setClearColor(0x000000, 0);
		renderer.autoClear = false;
		// Half the strands make a map as good: their roots are spread at random.
		this.strands.prepare('opacityDepth', light, OPACITY_SIZE, 2);
		renderer.setRenderTarget(this.opacityDepth);
		renderer.clear(true, true, false);
		renderer.render(scene, light);
		this.strands.setOpacityMap(this.opacityDepth.depthTexture!, this.opacityLayers.texture, light);
		this.strands.prepare('opacityLayers', light, OPACITY_SIZE, 2);
		renderer.setRenderTarget(this.opacityLayers);
		renderer.clear(true, false, false);
		renderer.render(scene, light);
		for (const instance of instances) instance.strands.mesh.visible = instance.cap.visible = true;
		this.strands.light(renderer);
	}

	dispose(scene: THREE.Scene): void {
		scene.remove(this.strands.mesh, this.cap);
		this.simulation.dispose();
		this.strands.dispose();
		this.opacityDepth.depthTexture?.dispose();
		this.opacityDepth.dispose();
		this.opacityLayers.dispose();
	}

	/** Lay out the render strands from the guides where they are now. */
	private lay(renderer: THREE.WebGLRenderer): void {
		this.strands.expand(renderer, this.simulation.positions, this.head);
	}
}

/** The scalp and colliders of `rig`'s avatar, measured in its bind pose. */
function measure(rig: Rig): Anatomy {
	const rotations = rig.bindRelative({});
	const bind = rig.world(rotations);
	// In the bind pose the posed vertices are the rest vertices; copied, as the rig reuses them.
	const vertices = Float32Array.from(
		rig.pose(rotations, new Float32Array(rig.expressionNames.length))
	);
	const head = rig.bone('head');
	const body = measureBody(rig, bind, vertices);
	const torso = body.torso
		? new THREE.DataTexture(
				body.torso.field,
				TORSO_ANGLES,
				TORSO_HEIGHTS,
				THREE.RedFormat,
				THREE.FloatType
			)
		: new THREE.DataTexture(new Float32Array(1), 1, 1, THREE.RedFormat, THREE.FloatType);
	torso.wrapS = THREE.RepeatWrapping;
	torso.minFilter = torso.magFilter = THREE.LinearFilter;
	torso.needsUpdate = true;
	return { scalp: buildScalp(vertices, bind[head]), body, torso, bind, head };
}

function decompose(m: THREE.Matrix4): { position: THREE.Vector3; rotation: THREE.Quaternion } {
	const position = new THREE.Vector3();
	const rotation = new THREE.Quaternion();
	m.decompose(position, rotation, new THREE.Vector3());
	return { position, rotation };
}

/** Whether the head moved too far in one frame for anything but a cut. */
function jumped(from: THREE.Matrix4, to: THREE.Matrix4): boolean {
	const a = decompose(from);
	const b = decompose(to);
	return a.position.distanceTo(b.position) > 0.3 || a.rotation.angleTo(b.rotation) > 1.2;
}

function same(a: Float32Array, b: Float32Array): boolean {
	for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
	return true;
}

/** The full-screen quad that lays the hair over the frame and shades the skin under it. */
function compositeQuad(
	accumulation: THREE.WebGLRenderTarget,
	sceneDepth: THREE.DepthTexture
): THREE.Mesh<THREE.BufferGeometry, THREE.RawShaderMaterial> {
	const geometry = new THREE.BufferGeometry();
	geometry.setAttribute(
		'position',
		new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3)
	);
	const uniforms: Record<string, THREE.IUniform> = {
		accumulated: { value: accumulation.textures[0] },
		optical: { value: accumulation.textures[1] },
		sceneDepth: { value: sceneDepth },
		inverseViewProjection: { value: new THREE.Matrix4() },
		shadowStrength: { value: 0.6 },
		count: { value: 0 }
	};
	for (let i = 0; i < SHADOWING; i++) {
		uniforms[`opacityDepth${i}`] = { value: null };
		uniforms[`opacityLayers${i}`] = { value: null };
		uniforms[`lightMatrix${i}`] = { value: new THREE.Matrix4() };
		uniforms[`lightRange${i}`] = { value: 1 };
	}
	const material = new THREE.RawShaderMaterial({
		glslVersion: THREE.GLSL3,
		vertexShader: /* glsl */ `
			in vec3 position;
			void main() {
				gl_Position = vec4(position.xy, 0.0, 1.0);
			}
		`,
		fragmentShader: COMPOSITE_FRAGMENT,
		uniforms,
		transparent: true,
		depthTest: false,
		depthWrite: false,
		blending: THREE.CustomBlending,
		blendSrc: THREE.OneFactor,
		blendDst: THREE.OneMinusSrcAlphaFactor,
		blendSrcAlpha: THREE.ZeroFactor,
		blendDstAlpha: THREE.OneFactor
	});
	const quad = new THREE.Mesh(geometry, material);
	quad.frustumCulled = false;
	// After the mesh and the Gaussians.
	quad.renderOrder = Number.MAX_SAFE_INTEGER;
	return quad;
}

const COMPOSITE_FRAGMENT = /* glsl */ `
precision highp float;
precision highp sampler2D;

uniform sampler2D accumulated;
uniform sampler2D optical;
uniform sampler2D sceneDepth;
uniform mat4 inverseViewProjection;
uniform float shadowStrength;
uniform int count;
${Array.from({ length: SHADOWING }, (_, i) => opacityMapGlsl(String(i))).join('\n')}

out vec4 outColour;

void main() {
	ivec2 texel = ivec2(gl_FragCoord.xy);
	vec4 sum = texelFetch(accumulated, texel, 0);
	float alpha = 1.0 - exp(-texelFetch(optical, texel, 0).r);
	vec3 colour = sum.a > 0.0 ? sum.rgb / sum.a : vec3(0.0);
	// Linear light to the display's values, the highlights rolled off rather than clipped.
	colour = pow(colour / (1.0 + 0.25 * colour), vec3(1.0 / 2.2));

	// Skin in the hair's shadow.
	float lit = 1.0;
	float depth = texelFetch(sceneDepth, texel, 0).r;
	if (depth < 1.0 && count > 0) {
		vec2 ndc = (vec2(texel) + 0.5) / vec2(textureSize(sceneDepth, 0)) * 2.0 - 1.0;
		vec4 world = inverseViewProjection * vec4(ndc, depth * 2.0 - 1.0, 1.0);
		vec3 p = world.xyz / world.w;
		float hair = opticalDepth0(p, 4.0, 2.0);
		${SHADOWING > 1 ? 'if (count > 1) hair += opticalDepth1(p, 4.0, 2.0);' : ''}
		lit = mix(1.0, exp(-0.3 * hair), shadowStrength);
	}
	outColour = vec4(colour * alpha, 1.0 - (1.0 - alpha) * lit);
}
`;
