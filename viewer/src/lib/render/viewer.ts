/**
 * The avatar in a three.js scene: its textured mesh, its Gaussians over it, an orbit camera.
 * World axes are the avatar's: metres, z up, the avatar facing -y.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { AvatarFile } from '#lib/avatar/file.ts';
import { Rig } from '#lib/avatar/rig.ts';
import type { Vec3 } from '#lib/math.ts';
import { TexturedMesh } from '#lib/render/mesh.ts';
import { Splats } from '#lib/render/splats.ts';

/** Camera layer of the mesh's depth pass. */
const MESH_DEPTH = 1;

export interface Stats {
	gaussians: number;
	triangles: number;
}

export class AvatarViewer {
	readonly rig: Rig;
	private readonly renderer: THREE.WebGLRenderer;
	private readonly scene = new THREE.Scene();
	private readonly camera: THREE.PerspectiveCamera;
	private readonly controls: OrbitControls;
	private readonly mesh: TexturedMesh;
	private readonly splats: Splats;
	private readonly resizeObserver: ResizeObserver;
	/** The mesh's depth, which hides the Gaussians behind it. */
	private readonly depthTarget = new THREE.WebGLRenderTarget(1, 1, {
		depthTexture: new THREE.DepthTexture(1, 1, THREE.FloatType)
	});
	private frame = 0;
	/** Whether something changed since the last render; the viewer draws only then. */
	private dirty = true;

	constructor(
		private readonly canvas: HTMLCanvasElement,
		file: AvatarFile
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
		this.mesh = new TexturedMesh(file);
		this.splats = new Splats(file);
		this.mesh.object.layers.enable(MESH_DEPTH);
		this.scene.add(this.mesh.object, this.splats.object);
		this.setBackground('#ffffff');
		this.pose({}, new Float32Array(this.rig.expressionNames.length));
		this.frameAvatar();

		this.resizeObserver = new ResizeObserver(() => this.resize());
		this.resizeObserver.observe(canvas);
		this.resize();
		this.loop();
	}

	/** Pose the avatar: rotation vectors in world axes by bone name, and expression weights. */
	pose(worldRotvecs: Record<string, Vec3>, expressions: Float32Array): void {
		const posed = this.rig.pose(this.rig.bindRelative(worldRotvecs), expressions);
		this.mesh.setVertices(posed);
		this.splats.setVertices(posed);
		this.dirty = true;
	}

	show(layers: { mesh: boolean; gaussians: boolean }): void {
		this.mesh.object.visible = layers.mesh;
		this.splats.object.visible = layers.gaussians;
		this.dirty = true;
	}

	setSplatOptions(options: { viewDependent: boolean; antialias: boolean }): void {
		this.splats.setOptions(options);
		this.dirty = true;
	}

	setBackground(colour: string): void {
		this.scene.background = new THREE.Color(colour);
		this.dirty = true;
	}

	/** Look at the whole avatar from the front. */
	frameAvatar(): void {
		const box = new THREE.Box3().setFromObject(this.mesh.object);
		const centre = box.getCenter(new THREE.Vector3());
		const height = box.max.z - box.min.z;
		const distance = (1.15 * height) / 2 / Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
		this.controls.target.copy(centre);
		this.camera.position.set(centre.x, centre.y - distance, centre.z);
		this.controls.update();
		this.dirty = true;
	}

	stats(): Stats {
		return {
			gaussians: this.splats.count,
			triangles: (this.mesh.object.geometry.getAttribute('position').count / 3) | 0
		};
	}

	dispose(): void {
		cancelAnimationFrame(this.frame);
		this.resizeObserver.disconnect();
		this.controls.dispose();
		this.mesh.dispose();
		this.splats.dispose();
		this.depthTarget.dispose();
		this.renderer.dispose();
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
		if (!this.dirty) return;
		this.dirty = false;
		const testMesh = this.mesh.object.visible && this.splats.object.visible;
		if (this.splats.object.visible) this.splats.sort(this.camera);
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
