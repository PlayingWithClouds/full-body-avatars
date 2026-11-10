/**
 * Running a scripted sequence (`sequence.ts`) on the player: starting each step when its time
 * comes, and layering the cues' expressions and gaze over the poses the player shows.
 *
 * Sequence time runs with the player's speed. Gaze turns the head bone only, by at most
 * `MAX_LOOK` and never more than the cue's envelope allows. It is computed from the head's world
 * transform in the frame before, which is close enough at animation frame rates and keeps posing
 * to one pass.
 */

import type { Rig } from '#lib/avatar/rig.ts';
import { PRESETS, rotations as presetRotations } from '#lib/avatar/poses.ts';
import {
	apply3,
	cross,
	heading,
	multiply3,
	normalize,
	rotation3,
	transpose3,
	yawMatrix,
	type Vec3
} from '#lib/math.ts';
import type { AvatarViewer } from '#lib/render/viewer.ts';
import {
	duration,
	matrixToQuaternion,
	quaternionMatrix,
	type MotionClip,
	type Pose
} from '#lib/animation/clip.ts';
import type { Player } from '#lib/animation/player.ts';
import {
	envelope,
	expressionIndices,
	schedule,
	type Cue,
	type Scheduled,
	type Sequence,
	type Target
} from '#lib/animation/sequence.ts';

/** Most the head turns to look at something, in radians. */
const MAX_LOOK = (60 * Math.PI) / 180;
/** Height of the eyes above the head bone's joint, in metres: where gaze is measured from. */
const EYE_HEIGHT = 0.09;

export class Sequencer {
	time = 0;
	/** Index of the step playing, -1 before the first and after the last. */
	step = -1;
	finished = false;
	private readonly plan: Scheduled[];
	private readonly length: number;
	private readonly rig: Rig;
	private readonly head: number;
	/** The head's rotation relative to its bind pose as last shown, per person, with gaze. */
	private readonly shownHead: Float64Array[] = [];
	private readonly firedCameras = new Set<Cue>();

	/**
	 * Args: `clips` maps the clip names the sequence uses to loaded clips; every one must be
	 * there.
	 */
	constructor(
		readonly sequence: Sequence,
		private readonly clips: Map<string, MotionClip>,
		private readonly player: Player,
		private readonly viewer: AvatarViewer
	) {
		this.rig = viewer.rig;
		this.head = this.rig.bone('head');
		this.plan = schedule(sequence.steps, (name) => duration(this.clipNamed(name)));
		this.length = this.plan.at(-1)!.end;
	}

	get schedule(): readonly Scheduled[] {
		return this.plan;
	}

	get total(): number {
		return this.length;
	}

	start(): void {
		this.time = 0;
		this.step = -1;
		this.finished = false;
		this.firedCameras.clear();
		this.player.mode = 'loop';
		this.startStep(0);
		this.player.onAdvance = (seconds) => this.advance(seconds);
		this.player.overlay = (person, pose) => this.overlay(person, pose);
	}

	stop(): void {
		if (this.player.onAdvance) this.player.onAdvance = null;
		this.player.overlay = null;
		this.finished = true;
	}

	private advance(seconds: number): void {
		this.time += seconds * this.player.speed;
		if (this.time >= this.length) {
			if (!this.sequence.loop) {
				// The last step's motion keeps playing; the sequence is done.
				this.time = this.length;
				this.player.onAdvance = null;
				this.finished = true;
				this.step = -1;
				return;
			}
			this.time -= this.length;
			this.firedCameras.clear();
			this.startStep(0);
		}
		while (this.step + 1 < this.plan.length && this.time >= this.plan[this.step + 1].start) {
			this.startStep(this.step + 1);
		}
		for (const cue of this.sequence.cues) {
			if (cue.kind === 'camera' && this.time >= cue.at && !this.firedCameras.has(cue)) {
				this.firedCameras.add(cue);
				if (cue.view === 'video') this.player.videoView();
				else this.viewer.frameAvatar();
			}
		}
	}

	private startStep(index: number): void {
		this.step = index;
		const { step } = this.plan[index];
		if (step.kind === 'clip') {
			this.player.play(this.clipNamed(step.clip), {
				from: step.from,
				to: step.to ?? undefined,
				speed: step.speed,
				fade: step.fade ?? undefined
			});
		} else if (step.kind === 'pose') {
			// The pose clip is made where the people are, so it is not moved again.
			this.player.play(this.poseClip(step.pose), { fade: step.fade ?? undefined, align: false });
		}
	}

	private clipNamed(name: string): MotionClip {
		const clip = this.clips.get(name);
		if (!clip) throw new Error(`no clip "${name}" is loaded`);
		return clip;
	}

	/**
	 * A still clip of a pose preset for each person showing, standing on the floor where they
	 * are and facing the way they face.
	 */
	private poseClip(name: string): MotionClip {
		const preset = PRESETS[name];
		if (!preset) throw new Error(`no pose "${name}"; there are ${Object.keys(PRESETS).join(', ')}`);
		const rig = this.rig;
		const people = Math.max(this.viewer.copies, 1);
		const bones = rig.boneNames.length;
		const matrices = rig.bindRelative(presetRotations(preset));
		const one = new Float32Array(4 * bones);
		const quaternion = new Float64Array(4);
		matrices.forEach((m, j) => {
			matrixToQuaternion(m, quaternion, 0);
			one.set(quaternion, 4 * j);
		});
		const frames = 2;
		const rotations = new Float32Array(people * frames * 4 * bones);
		const rootTranslation = new Float32Array(people * frames * 3);
		const bind = rig.bindPose(0);
		const rootBind = rotation3(bind);
		for (let person = 0; person < people; person++) {
			const root = this.viewer.avatars()[person]?.boneWorld[0] ?? bind;
			// The root turned about world z to the person's heading: B^T Rz B as its rotation.
			const turn = yawMatrix(heading(rotation3(root)) - heading(rootBind));
			matrixToQuaternion(multiply3(transpose3(rootBind), multiply3(turn, rootBind)), quaternion, 0);
			for (let frame = 0; frame < frames; frame++) {
				const k = person * frames + frame;
				rotations.set(one, 4 * bones * k);
				rotations.set(quaternion, 4 * bones * k);
				rootTranslation.set([root[3] - bind[3], root[7] - bind[7], rig.standingHeight()], 3 * k);
			}
		}
		const cameraToWorld = new Float32Array(16);
		cameraToWorld.set(this.player.clip?.cameraToWorld ?? [], 0);
		return {
			name,
			fps: 1,
			frames,
			people,
			boneNames: rig.boneNames,
			expressionNames: rig.expressionNames,
			fovYDegrees: this.player.clip?.fovYDegrees ?? 30,
			rotations,
			rootTranslation,
			expressions: new Float32Array(people * frames * rig.expressionNames.length),
			cameraToWorld
		};
	}

	/** Layer the active expression and gaze cues over one person's pose. */
	private overlay(person: number, pose: Pose): void {
		for (const cue of this.sequence.cues) {
			if (cue.kind === 'camera' || !applies(cue.character, person)) continue;
			const weight = envelope(this.time - cue.at, cue.duration, cue.fade);
			if (weight === 0) continue;
			if (cue.kind === 'expression') {
				for (const [name, value] of Object.entries(cue.weights)) {
					for (const e of expressionIndices(name, this.rig.expressionNames)) {
						pose.expressions[e] += weight * (value - pose.expressions[e]);
					}
				}
			} else this.look(person, pose, cue.target, weight);
		}
		this.shownHead[person] = pose.rotations.slice(4 * this.head, 4 * this.head + 4);
	}

	/** Turn a person's head towards a target, by `weight` of the way (up to `MAX_LOOK`). */
	private look(person: number, pose: Pose, target: Target, weight: number): void {
		const shown = this.viewer.avatars()[person];
		const before = this.shownHead[person];
		if (!shown || !before) return;
		const world = shown.boneWorld[this.head];
		const rotation = rotation3(world);
		// The head's parent frame in the world: its world rotation without its own last rotation.
		const parent = multiply3(rotation, transpose3(quaternionMatrix(before, 0)));
		// Where the face pointed: the bind pose faces -y, carried along by the head's rotation.
		const bind = this.rig.bindPose(this.head);
		const front = apply3(rotation, apply3(transpose3(rotation3(bind)), [0, -1, 0]));
		const eyes: Vec3 = [world[3], world[7], world[11] + EYE_HEIGHT];
		const point = this.targetPoint(person, target);
		if (!point) return;
		const wanted = normalize([point[0] - eyes[0], point[1] - eyes[1], point[2] - eyes[2]]);
		const axis = cross(front, wanted);
		const sine = Math.hypot(axis[0], axis[1], axis[2]);
		if (sine < 1e-6) return;
		const angle = Math.min(Math.atan2(sine, dot(front, wanted)), MAX_LOOK) * weight;
		const turn = axisAngle(normalize(axis), angle);
		// The head's new rotation relative to its bind pose: P^T C P L, turning it in the world.
		const local = quaternionMatrix(pose.rotations, 4 * this.head);
		const turned = multiply3(transpose3(parent), multiply3(turn, multiply3(parent, local)));
		matrixToQuaternion(turned, pose.rotations, 4 * this.head);
	}

	private targetPoint(person: number, target: Target): Vec3 | null {
		if (target === 'camera') {
			const { x, y, z } = this.viewer.camera.position;
			return [x, y, z];
		}
		if (target === 'partner') {
			const others = this.viewer.avatars().filter((_, i) => i !== person);
			if (!others.length) return null;
			const head = others[0].boneWorld[this.head];
			return [head[3], head[7], head[11] + EYE_HEIGHT];
		}
		return target;
	}
}

function applies(character: number | 'all', person: number): boolean {
	return character === 'all' || character === person;
}

function axisAngle([x, y, z]: Vec3, angle: number): Float64Array {
	const c = Math.cos(angle);
	const s = Math.sin(angle);
	const t = 1 - c;
	return Float64Array.of(
		t * x * x + c, t * x * y - s * z, t * x * z + s * y,
		t * x * y + s * z, t * y * y + c, t * y * z - s * x,
		t * x * z - s * y, t * y * z + s * x, t * z * z + c
	); // prettier-ignore
}

function dot(a: Vec3, b: Vec3): number {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
