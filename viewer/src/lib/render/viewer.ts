/**
 * The avatar in a three.js scene: its textured mesh, its Gaussians over it, an orbit camera.
 * World axes are the avatar's: metres, z up, the avatar facing -y.
 *
 * The scene can hold several copies of the avatar, each posed on its own (a clip with two
 * people plays on two copies): a mesh per copy, and all copies' Gaussians in one sorted draw.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { AvatarFile } from '#lib/avatar/file.ts';
import { Rig } from '#lib/avatar/rig.ts';
import type { Mat4, Vec3 } from '#lib/math.ts';
import { FaceHair } from '#lib/render/face_hair.ts';
import { TexturedMesh } from '#lib/render/mesh.ts';
import { Splats } from '#lib/render/splats.ts';

/**
 * Camera layer of the depth pass that hides Gaussians: whatever is on it (the mesh, and anything
 * else opaque in front of the skin, such as hair) hides the Gaussians behind it.
 */
export const MESH_DEPTH = 1;

export interface Stats {
	gaussians: number;
	triangles: number;
}

/** An avatar as posed for the current frame. */
export interface PosedAvatar {
	rig: Rig;
	/** `(V * 3)` posed vertices in world coordinates. */
	vertices: Float32Array;
	/** `(J)` bones' world transforms, row-major. */
	boneWorld: Mat4[];
}

/**
 * Something drawn with the avatars that follows them, such as hair: added with `addLayer`,
 * updated every animation frame, after the animation has posed the avatars.
 */
export interface SceneLayer {
	readonly object: THREE.Object3D;
	/** Where the layer draws, if the camera should frame it with the avatars. */
	readonly bounds?: THREE.Box3;
	/**
	 * Advance by `seconds` and follow the posed avatars; return whether anything changed, so
	 * the viewer draws the frame.
	 */
	update(seconds: number, avatars: readonly PosedAvatar[]): boolean;
	dispose(): void;
}

/** A pose of one copy, as an animation gives it. */
export interface CopyPose {
	/** `(J)` 3x3 rotations relative to each bone's bind pose, row-major. */
	rotations: Float64Array[];
	expressions: Float32Array;
	/** The root's offset from its bind position, in world axes. */
	rootOffset: Vec3;
}

export class AvatarViewer {
	readonly rig: Rig;
	readonly renderer: THREE.WebGLRenderer;
	readonly scene = new THREE.Scene();
	readonly camera: THREE.PerspectiveCamera;
	private readonly controls: OrbitControls;
	private meshes: TexturedMesh[] = [];
	private splats: Splats;
	private posed: PosedAvatar[] = [];
	private readonly resizeObserver: ResizeObserver;
	private readonly layers: SceneLayer[] = [];
	private readonly timer = new THREE.Timer();
	private animator: ((seconds: number) => boolean) | null = null;
	/** The meshes' depth, which hides the Gaussians behind them. */
	private readonly depthTarget = new THREE.WebGLRenderTarget(1, 1, {
		depthTexture: new THREE.DepthTexture(1, 1, THREE.FloatType)
	});
	private visible = { mesh: true, gaussians: true };
	private splatOptions = { viewDependent: true, antialias: false };
	private frame = 0;
	/** Whether something changed since the last render; the viewer draws only then. */
	private dirty = true;

	constructor(
		private readonly canvas: HTMLCanvasElement,
		readonly file: AvatarFile
	) {
		this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
		this.renderer.setPixelRatio(window.devicePixelRatio);
		// The avatar's colours were fitted to the videos' pixel values: show them unchanged.
		this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
		THREE.ColorManagement.enabled = false;

		this.camera = new THREE.PerspectiveCamera(30, 1, 0.05, 100);
		this.camera.up.set(0, 0, 1);
		this.controls = new OrbitControls(this.camera, canvas);
		this.controls.enableDamping = true;
		this.controls.addEventListener('change', () => (this.dirty = true));

		this.rig = new Rig(file);
		this.splats = new Splats(file);
		this.setCopies(1);
		if (file.metadata.face_hair) this.addLayer(new FaceHair(file));
		this.setBackground('#ffffff');
		this.pose({}, new Float32Array(this.rig.expressionNames.length));
		this.frameAvatar();

		this.resizeObserver = new ResizeObserver(() => this.resize());
		this.resizeObserver.observe(canvas);
		this.resize();
		this.loop();
	}

	/** How many copies of the avatar the scene holds. */
	get copies(): number {
		return this.meshes.length;
	}

	/** Hold `count` copies of the avatar; new copies stand in the bind pose. */
	setCopies(count: number): void {
		if (count === this.meshes.length) return;
		for (const mesh of this.meshes.splice(count)) {
			this.scene.remove(mesh.object);
			mesh.dispose();
		}
		while (this.meshes.length < count) {
			const first = this.meshes[0];
			const mesh = first ? first.copy(this.file) : new TexturedMesh(this.file);
			mesh.object.layers.enable(MESH_DEPTH);
			mesh.object.visible = this.visible.mesh;
			this.meshes.push(mesh);
			this.scene.add(mesh.object);
		}
		this.scene.remove(this.splats.object);
		this.splats.dispose();
		this.splats = new Splats(this.file, count);
		this.splats.object.visible = this.drawSplats;
		this.splats.setOptions(this.splatOptions);
		this.scene.add(this.splats.object);
		const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
		this.splats.setViewport(size.x, size.y);

		const kept = this.posed.slice(0, count);
		this.posed = [];
		for (let copy = 0; copy < count; copy++) {
			const pose = kept[copy];
			if (pose) this.setPosed(copy, pose.vertices, pose.boneWorld);
			else {
				const face = this.neutralFace();
				const vertices = this.rig.pose(this.rig.bindRelative({}), face);
				this.setPosed(copy, vertices, this.rig.boneWorld, face);
			}
		}
		this.dirty = true;
	}

	/**
	 * Pose the first copy by hand: rotation vectors in world axes by bone name, and expression
	 * weights.
	 */
	pose(worldRotvecs: Record<string, Vec3>, expressions: Float32Array): void {
		const vertices = this.rig.pose(this.rig.bindRelative(worldRotvecs), expressions);
		this.setPosed(0, vertices, this.rig.boneWorld, expressions);
		this.dirty = true;
	}

	/** Pose one copy as an animation gives it. */
	poseCopy(copy: number, pose: CopyPose): void {
		const vertices = this.rig.pose(pose.rotations, pose.expressions, pose.rootOffset);
		this.setPosed(copy, vertices, this.rig.boneWorld, pose.expressions);
		this.dirty = true;
	}

	/**
	 * Run `animate` every animation frame, before the layers update: it poses the copies and
	 * returns whether anything moved. Null stops it.
	 */
	setAnimator(animate: ((seconds: number) => boolean) | null): void {
		this.animator = animate;
		this.dirty = true;
	}

	/** The avatars as posed for the current frame. */
	avatars(): PosedAvatar[] {
		return this.posed;
	}

	addLayer(layer: SceneLayer): void {
		this.layers.push(layer);
		this.scene.add(layer.object);
		this.dirty = true;
	}

	removeLayer(layer: SceneLayer): void {
		const index = this.layers.indexOf(layer);
		if (index < 0) return;
		this.layers.splice(index, 1);
		this.scene.remove(layer.object);
		layer.dispose();
		this.dirty = true;
	}

	show(layers: { mesh: boolean; gaussians: boolean }): void {
		this.visible = { ...layers };
		for (const mesh of this.meshes) mesh.object.visible = layers.mesh;
		this.splats.object.visible = this.drawSplats;
		this.dirty = true;
	}

	/** Light the skin with the file's learned light, or show its albedo unlit (format 4). */
	setLit(lit: boolean): void {
		// Copies share the first mesh's materials.
		this.meshes[0]?.setLit(lit);
		this.dirty = true;
	}

	setSplatOptions(options: { viewDependent: boolean; antialias: boolean }): void {
		this.splatOptions = { ...options };
		this.splats.setOptions(options);
		this.dirty = true;
	}

	setBackground(colour: string): void {
		this.scene.background = new THREE.Color(colour);
		this.dirty = true;
	}

	/** Look at all the avatars, and the layers that give bounds, from the front. */
	frameAvatar(): void {
		const box = new THREE.Box3();
		for (const mesh of this.meshes) box.expandByObject(mesh.object);
		for (const layer of this.layers) if (layer.bounds) box.union(layer.bounds);
		const centre = box.getCenter(new THREE.Vector3());
		const size = box.getSize(new THREE.Vector3());
		const extent = Math.max(size.z, size.x / this.camera.aspect);
		const distance = (1.15 * extent) / 2 / Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
		this.controls.target.copy(centre);
		this.camera.position.set(centre.x, centre.y - distance - size.y / 2, centre.z);
		this.controls.update();
		this.dirty = true;
	}

	/**
	 * Look through a camera given as a row-major camera-to-world transform in OpenCV axes (x
	 * right, y down, z forward), with a vertical field of view: the view of a motion clip's video.
	 */
	lookThrough(cameraToWorld: Float32Array, fovDegrees: number): void {
		const m = cameraToWorld;
		const position = new THREE.Vector3(m[3], m[7], m[11]);
		const forward = new THREE.Vector3(m[2], m[6], m[10]);
		// Look at a point a few metres ahead, so the orbit turns about where the people are. The
		// clip's world is levelled, so the camera's roll is close to none and is dropped.
		const target = position.clone().addScaledVector(forward, 3);
		this.camera.fov = fovDegrees;
		this.camera.updateProjectionMatrix();
		this.camera.position.copy(position);
		this.controls.target.copy(target);
		this.controls.update();
		this.dirty = true;
	}

	stats(): Stats {
		const triangles = this.meshes[0]?.object.geometry.getAttribute('position').count ?? 0;
		return {
			gaussians: this.splats.count * this.splats.copies,
			triangles: ((triangles / 3) | 0) * this.meshes.length
		};
	}

	dispose(): void {
		cancelAnimationFrame(this.frame);
		this.resizeObserver.disconnect();
		for (const layer of this.layers) layer.dispose();
		this.controls.dispose();
		for (const mesh of this.meshes.toReversed()) mesh.dispose();
		this.splats.dispose();
		this.depthTarget.dispose();
		this.renderer.dispose();
	}

	private neutralFace(): Float32Array {
		return new Float32Array(this.rig.expressionNames.length);
	}

	/**
	 * Show one copy as `vertices` (copied: the rig reuses its buffer) posed with `expressions`
	 * (if not given, its wrinkles stay as they were), and record its bones.
	 */
	private setPosed(
		copy: number,
		vertices: Float32Array,
		boneWorld: Mat4[],
		expressions?: Float32Array
	): void {
		const own = this.posed[copy]?.vertices;
		const kept = own && own.length === vertices.length ? own : new Float32Array(vertices.length);
		kept.set(vertices);
		this.posed[copy] = { rig: this.rig, vertices: kept, boneWorld };
		this.meshes[copy].setVertices(kept, expressions);
		if (this.splats.count > 0) this.splats.setVertices(kept, copy);
	}

	/** Whether the Gaussians are drawn: when asked for and the file has any. */
	private get drawSplats(): boolean {
		return this.visible.gaussians && this.splats.count > 0;
	}

	private resize(): void {
		const { clientWidth: width, clientHeight: height } = this.canvas;
		if (width === 0 || height === 0) return;
		this.renderer.setSize(width, height, false);
		this.camera.aspect = width / height;
		this.camera.updateProjectionMatrix();
		const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
		this.splats.setViewport(size.x, size.y);
		this.depthTarget.setSize(size.x, size.y);
		this.dirty = true;
	}

	private loop = (): void => {
		this.frame = requestAnimationFrame(this.loop);
		this.controls.update();
		this.timer.update();
		const seconds = Math.min(this.timer.getDelta(), 0.1);
		if (this.animator?.(seconds)) this.dirty = true;
		for (const layer of this.layers) {
			if (layer.update(seconds, this.posed)) this.dirty = true;
		}
		if (!this.dirty) return;
		this.dirty = false;
		const testMesh = this.visible.mesh && this.drawSplats;
		if (this.drawSplats) this.splats.sort(this.camera);
		if (testMesh) {
			this.camera.layers.set(MESH_DEPTH);
			this.renderer.setRenderTarget(this.depthTarget);
			this.renderer.render(this.scene, this.camera);
			this.renderer.setRenderTarget(null);
			this.camera.layers.set(0);
		}
		this.splats.setMeshDepth(testMesh ? this.depthTarget.depthTexture : null, this.camera);
		this.renderer.render(this.scene, this.camera);
	};
}
