/**
 * A grid of heads beside the avatar, one per frame of a video, each wearing that frame's face:
 * the avatar's head with the frame's expression weights, gaze and head rotation, so that every
 * frame of a video can be compared at once.
 *
 * Only heads are drawn, and only as the textured mesh: a Gaussian layer per frame would repeat
 * the whole body's 29 thousand Gaussians for every one of them, and a head is all there is to
 * compare. The heads are posed on the CPU in the rest pose: expressions through a rig of their
 * own, then a rigid turn about the head's centre, so the grid needs no skeleton.
 */

import * as THREE from 'three';
import { array, type AvatarFile } from '#lib/avatar/file.ts';
import { Rig } from '#lib/avatar/rig.ts';
import {
	gridShape,
	tileUvs,
	type Crops,
	parseFaces,
	placeHead,
	turn,
	type Faces,
	type HeadPlacement
} from '#lib/faces/faces.ts';
import type { Vec3 } from '#lib/math.ts';
import { TexturedMesh } from '#lib/render/mesh.ts';
import type { SceneLayer } from '#lib/render/viewer.ts';

/**
 * Horizontal and vertical distance between heads, in metres: a head is about 0.2 m wide. With
 * the video's crops beside the heads a cell is wider.
 */
const SPACING = [0.27, 0.33];
const SPACING_WITH_CROPS = [0.52, 0.33];
/** The video's crop is a square of this side, this far left of its head, in metres. */
const CROP = { size: 0.24, left: 0.23 };
/** How far in front of the avatar's shoulders (+x) the grid starts, in metres. */
const GAP = 0.9;
/** The label's size in metres, and its distance below the head's centre. */
const LABEL = { width: 0.16, height: 0.04, below: 0.14 };

export interface FaceGridOptions {
	/** Whether heads turn as the video's did, or all face front. */
	rotate: boolean;
	/** Whether to label each head with its frame's index. */
	labels: boolean;
	/** Whether to show the video's crop of each frame beside its head, if there are crops. */
	video: boolean;
}

export class FaceGridLayer implements SceneLayer {
	readonly object = new THREE.Group();
	/** The grid and the avatar's head, to frame the camera on. */
	readonly bounds = new THREE.Box3();
	readonly faces: Faces;
	private readonly original: TexturedMesh;
	private readonly meshes: TexturedMesh[] = [];
	private readonly labels = new THREE.Group();
	private readonly crops = new THREE.Group();
	private readonly rig: Rig;
	/** One head's vertices as placed, reused between frames: the meshes copy them. */
	private readonly placed: Float32Array;
	private readonly placement: Omit<HeadPlacement, 'rotation' | 'centre' | 'eyes'>;
	private readonly eyeSets: { vertices: Uint32Array; centre: Vec3 }[];
	private readonly centres: Vec3[] = [];

	constructor(
		file: AvatarFile,
		document: unknown,
		/** The image of the video's crops that the document describes, if loaded. */
		private readonly atlas: THREE.Texture | null,
		private options: FaceGridOptions
	) {
		this.rig = new Rig(file);
		this.faces = parseFaces(document, this.rig.expressionNames);
		const rest = this.rig.pose(
			this.rig.bindRelative({}),
			new Float32Array(this.rig.expressionNames.length)
		);
		const head = headVertices(file);
		this.placement = { vertices: head.vertices, pivot: head.pivot };
		this.eyeSets = eyeSets(file, rest);

		this.original = new TexturedMesh(file);
		const count = this.faces.frames.length;
		const crops = this.atlas ? this.faces.crops : null;
		const [dx, dy] = crops ? SPACING_WITH_CROPS : SPACING;
		const { columns, rows } = gridShape(count, (16 / 9) * (0.27 / dx));
		for (let i = 0; i < count; i++) {
			// Row by row from the top left, at the avatar's head height, to its +x side.
			const column = i % columns;
			const row = Math.floor(i / columns);
			this.centres.push([
				head.pivot[0] + GAP + column * dx,
				head.pivot[1],
				head.pivot[2] + ((rows - 1) / 2 - row) * dy
			]);
			const mesh = this.original.copy(file);
			dropBody(mesh);
			this.meshes.push(mesh);
			this.object.add(mesh.object);
		}
		this.object.add(this.labels, this.crops);

		this.bounds.setFromPoints([...this.centres, head.pivot].map((c) => new THREE.Vector3(...c)));
		this.bounds.expandByScalar(0.2);
		this.placed = new Float32Array(rest.length);
		this.pose();
		this.makeLabels();
		if (crops) this.makeCrops(crops);
		this.crops.visible = options.video;
	}

	/** Change how the heads show. */
	setOptions(options: FaceGridOptions): void {
		const rotateChanged = options.rotate !== this.options.rotate;
		this.options = { ...options };
		if (rotateChanged) this.pose();
		this.labels.visible = options.labels;
		this.crops.visible = options.video;
	}

	update(): boolean {
		return false;
	}

	dispose(): void {
		for (const mesh of this.meshes) mesh.dispose();
		this.original.dispose();
		for (const crop of this.crops.children as THREE.Mesh[]) {
			crop.geometry.dispose();
			(crop.material as THREE.Material).dispose();
		}
		this.atlas?.dispose();
		for (const label of this.labels.children as THREE.Sprite[]) {
			label.material.map?.dispose();
			label.material.dispose();
		}
	}

	/** Pose every head for its frame, turned or facing front. */
	private pose(): void {
		const rig = this.rig;
		this.faces.frames.forEach((frame, i) => {
			const bound = rig.pose(rig.bindRelative({}), frame.expressions);
			placeHead(bound, this.placed, {
				...this.placement,
				centre: this.centres[i],
				rotation: turn(frame.head, this.options.rotate),
				eyes: this.eyeSets.map((eye, e) => ({ ...eye, rotation: turn(frame.eyes[e]) }))
			});
			this.meshes[i].setVertices(this.placed, frame.expressions);
		});
	}

	/** A plane facing the camera (-y) beside each head, showing that frame's tile of the atlas. */
	private makeCrops(crops: Crops): void {
		this.faces.frames.forEach((_, i) => {
			const [u0, v0, u1, v1] = tileUvs(i, crops);
			const geometry = new THREE.PlaneGeometry(CROP.size, CROP.size);
			geometry.setAttribute(
				'uv',
				new THREE.BufferAttribute(new Float32Array([u0, v1, u1, v1, u0, v0, u1, v0]), 2)
			);
			geometry.rotateX(Math.PI / 2);
			const mesh = new THREE.Mesh(
				geometry,
				new THREE.MeshBasicMaterial({ map: this.atlas, side: THREE.FrontSide })
			);
			const [x, y, z] = this.centres[i];
			mesh.position.set(x - CROP.left, y, z);
			this.crops.add(mesh);
		});
	}

	private makeLabels(): void {
		this.faces.frames.forEach((frame, i) => {
			const canvas = document.createElement('canvas');
			canvas.width = 256;
			canvas.height = 64;
			const context = canvas.getContext('2d')!;
			context.fillStyle = 'rgba(24, 24, 27, 0.75)';
			context.beginPath();
			context.roundRect(48, 8, 160, 52, 14);
			context.fill();
			context.font = '600 40px ui-monospace, monospace';
			context.textAlign = 'center';
			context.textBaseline = 'middle';
			// Frames where no face was found keep their neighbours' face: mark them.
			context.fillStyle = frame.found ? '#e4e4e7' : '#f87171';
			context.fillText(`${frame.index}${frame.found ? '' : ' ?'}`, 128, 34);
			const map = new THREE.CanvasTexture(canvas);
			map.colorSpace = THREE.NoColorSpace;
			const sprite = new THREE.Sprite(
				new THREE.SpriteMaterial({ map, transparent: true, depthWrite: false })
			);
			sprite.scale.set(LABEL.width, LABEL.height, 1);
			const [x, y, z] = this.centres[i];
			sprite.position.set(x, y, z - LABEL.below);
			this.labels.add(sprite);
		});
		this.labels.visible = this.options.labels;
	}
}

/** Drop the body's material group from a mesh: a head grid has no use for the body. */
function dropBody(mesh: TexturedMesh): void {
	const geometry = mesh.object.geometry;
	const kept = geometry.groups.filter((group) => group.materialIndex !== 0);
	geometry.clearGroups();
	for (const group of kept) geometry.addGroup(group.start, group.count, group.materialIndex!);
}

/** The head's vertices and the point it turns about: the mean of its skin's vertices. */
function headVertices(file: AvatarFile): { vertices: Uint32Array; pivot: Vec3 } {
	const faces = array<Uint32Array>(file, 'faces').data;
	const rest = array<Float32Array>(file, 'rest_vertices').data;
	const parts = file.metadata.parts;
	const partOfFace = parts ? array<Uint8Array>(file, 'face_parts').data : null;
	const textures = array<Uint8Array>(file, 'face_textures').data;
	const headStart = file.metadata.head_start;
	const headParts = ['head_skin', 'mouth', 'eyes'].map((name) => parts?.indexOf(name) ?? -1);
	const skin = parts?.indexOf('head_skin') ?? -1;

	const head = new Set<number>();
	const skinVertices = new Set<number>();
	for (let f = 0; f < faces.length / 3; f++) {
		const corners = [faces[3 * f], faces[3 * f + 1], faces[3 * f + 2]];
		// Files without parts: the head texture's triangles on head vertices (no genitals then).
		const isHead = partOfFace
			? headParts.includes(partOfFace[f])
			: textures[f] === 1 && corners.every((v) => v >= headStart);
		if (!isHead) continue;
		for (const v of corners) {
			head.add(v);
			if (!partOfFace || partOfFace[f] === skin) skinVertices.add(v);
		}
	}
	const pivot: Vec3 = [0, 0, 0];
	for (const v of skinVertices) for (let a = 0; a < 3; a++) pivot[a] += rest[3 * v + a];
	return {
		vertices: Uint32Array.from(head),
		pivot: pivot.map((sum) => sum / skinVertices.size) as Vec3
	};
}

/**
 * Each eyeball's vertices and centre, the one on the avatar's +x side first (the order of a
 * face's eye rotations). Empty when the file does not describe the eyes.
 */
function eyeSets(file: AvatarFile, rest: Float32Array): { vertices: Uint32Array; centre: Vec3 }[] {
	const eyes = file.metadata.eyes;
	const eyePart = file.metadata.parts?.indexOf('eyes') ?? -1;
	if (!eyes || eyePart < 0) return [];
	const faces = array<Uint32Array>(file, 'faces').data;
	const parts = array<Uint8Array>(file, 'face_parts').data;
	const balls = eyes.eyeballs.toSorted((a, b) => b.centre[0] - a.centre[0]);
	const sets = balls.map(() => new Set<number>());
	for (let f = 0; f < parts.length; f++) {
		if (parts[f] !== eyePart) continue;
		for (let c = 0; c < 3; c++) {
			const v = faces[3 * f + c];
			const distance = balls.map((ball) =>
				Math.hypot(...ball.centre.map((x, a) => x - rest[3 * v + a]))
			);
			sets[distance[0] < distance[1] ? 0 : 1].add(v);
		}
	}
	return balls.map((ball, e) => ({ vertices: Uint32Array.from(sets[e]), centre: ball.centre }));
}
