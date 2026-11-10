/**
 * Playing motion clips on copies of the avatar, one copy per person in the clip.
 *
 * Switching clips, or a clip ending and the next one starting, never cuts: the outgoing clip keeps
 * playing while the incoming one fades in over `fadeSeconds`, bone rotations interpolated along
 * the shorter way and root offsets and expressions linearly. With `align` on, the incoming clip
 * is first turned about the vertical and moved so its first person starts where and facing the
 * way the outgoing clip's first person is, so the people carry on from where they are instead of
 * sliding back to where the clip was recorded.
 *
 * A clip can be played trimmed and at its own speed (`PlayOptions`), and scripted sequences
 * (`sequence.ts`) drive the player through `onAdvance` and layer expressions and gaze over its
 * poses through `overlay`.
 */

import type { Rig } from '#lib/avatar/rig.ts';
import { heading, multiply3, rotation3, transpose3, yawMatrix, type Vec3 } from '#lib/math.ts';
import type { AvatarViewer } from '#lib/render/viewer.ts';
import {
	bindPose,
	ClipSampler,
	duration,
	nlerp,
	matrixToQuaternion,
	quaternionMatrix,
	type MotionClip,
	type Pose
} from '#lib/animation/clip.ts';

/** What happens when a clip reaches its end. */
export type EndMode = 'loop' | 'once' | 'next';

/** A rotation about world z and a horizontal shift, putting a clip where the people are. */
interface Placement {
	yaw: number;
	shift: Vec3;
}

/** How to play a clip: from and to which time in it, how fast, and how long to fade it in. */
export interface PlayOptions {
	from?: number;
	to?: number;
	speed?: number;
	fade?: number;
	/** Whether to carry the people on from where they are; the player's `align` if left out. */
	align?: boolean;
}

interface Track {
	sampler: ClipSampler;
	/** Seconds into the clip. */
	time: number;
	start: number;
	end: number;
	speed: number;
	placement: Placement;
}

const NO_PLACEMENT: Placement = { yaw: 0, shift: [0, 0, 0] };

export class Player {
	readonly clips: MotionClip[] = [];
	current = -1;
	playing = false;
	speed = 1;
	mode: EndMode = 'loop';
	fadeSeconds = 0.4;
	align = true;
	/** Called first thing every frame while playing, with the seconds since the last. */
	onAdvance: ((seconds: number) => void) | null = null;
	/** Changes each person's pose after it is sampled and blended, before it is shown. */
	overlay: ((person: number, pose: Pose) => void) | null = null;
	private track: Track | null = null;
	private options: PlayOptions = {};
	private fadeLength = 0.4;
	/** The outgoing clip while the current one fades in. */
	private outgoing: Track | null = null;
	private fade = 1;
	private readonly samplers = new Map<MotionClip, ClipSampler>();
	private poses: Pose[] = [];
	private readonly rig: Rig;
	/** The root bone's bind rotation, to turn clips about world z. */
	private readonly rootBind: Float64Array;
	private readonly rootBindPosition: Vec3;

	constructor(private readonly viewer: AvatarViewer) {
		this.rig = viewer.rig;
		const bind = this.rig.bindPose(0);
		this.rootBind = Float64Array.of(
			bind[0], bind[1], bind[2], bind[4], bind[5], bind[6], bind[8], bind[9], bind[10]
		); // prettier-ignore
		this.rootBindPosition = [bind[3], bind[7], bind[11]];
	}

	/** Add a clip to the library; returns its index. */
	add(clip: MotionClip): number {
		this.clips.push(clip);
		this.samplers.set(clip, new ClipSampler(clip, this.rig));
		return this.clips.length - 1;
	}

	get time(): number {
		return this.track?.time ?? 0;
	}

	/** Where the current clip ends, in seconds into it (its length unless trimmed). */
	get duration(): number {
		return this.track?.end ?? 0;
	}

	get clip(): MotionClip | null {
		return this.track?.sampler.clip ?? null;
	}

	/**
	 * Start playing a clip (from the library by index, or any other) from its beginning, or
	 * `options.from`, fading in from what is showing.
	 */
	play(which: number | MotionClip, options: PlayOptions = {}): void {
		const clip = typeof which === 'number' ? this.clips[which] : which;
		let sampler = this.samplers.get(clip);
		if (!sampler) {
			sampler = new ClipSampler(clip, this.rig);
			this.samplers.set(clip, sampler);
		}
		const showing = this.track !== null;
		const align = options.align ?? this.align;
		const placement = showing && align ? this.placementFor(sampler) : NO_PLACEMENT;
		const length = duration(clip);
		const start = Math.min(Math.max(options.from ?? 0, 0), length);
		const end = Math.min(Math.max(options.to ?? length, start), length);
		this.outgoing = showing ? this.track : null;
		this.fade = showing ? 0 : 1;
		this.fadeLength = options.fade ?? this.fadeSeconds;
		this.options = options;
		this.track = { sampler, time: start, start, end, speed: options.speed ?? 1, placement };
		this.current = this.clips.indexOf(clip);
		this.playing = true;
		this.viewer.setCopies(Math.max(clip.people, this.outgoing?.sampler.clip.people ?? 0));
		this.viewer.setAnimator((seconds) => this.update(seconds));
		if (!showing) this.viewer.lookThrough(clip.cameraToWorld, clip.fovYDegrees);
	}

	/** Show the current clip at `time` seconds. */
	seek(time: number): void {
		if (!this.track) return;
		this.track.time = Math.min(Math.max(time, this.track.start), this.track.end);
		this.outgoing = null;
		this.fade = 1;
		this.apply();
	}

	/** Stop playing and go back to a single avatar in its bind pose. */
	stop(): void {
		this.playing = false;
		this.track = this.outgoing = null;
		this.current = -1;
		this.onAdvance = this.overlay = null;
		this.viewer.setAnimator(null);
		this.viewer.setCopies(1);
	}

	/** Look at the clip through its video's camera. */
	videoView(): void {
		const clip = this.clip;
		if (clip) this.viewer.lookThrough(clip.cameraToWorld, clip.fovYDegrees);
	}

	private update(seconds: number): boolean {
		if (!this.track || !this.playing) return false;
		this.onAdvance?.(seconds);
		if (!this.track || !this.playing) return false;
		if (this.outgoing) {
			const outgoing = this.outgoing;
			outgoing.time = Math.min(outgoing.time + seconds * this.speed * outgoing.speed, outgoing.end);
		}
		this.fade = Math.min(this.fade + seconds / Math.max(this.fadeLength, 1e-3), 1);
		if (this.fade >= 1 && this.outgoing) {
			this.outgoing = null;
			this.viewer.setCopies(this.track.sampler.clip.people);
		}
		this.track.time += seconds * this.speed * this.track.speed;
		const end = this.track.end;
		if (this.track.time >= end) {
			const overshoot = this.track.time - end;
			this.track.time = end;
			if (this.mode === 'once') this.playing = false;
			else if (this.mode === 'next' && this.current >= 0) {
				this.play((this.current + 1) % this.clips.length);
				this.track!.time += overshoot;
			} else {
				this.play(this.track.sampler.clip, this.options);
				this.track!.time += overshoot;
			}
		}
		this.apply();
		return true;
	}

	/** Pose every copy for the current time, blending in the outgoing clip while it fades. */
	private apply(): void {
		const track = this.track!;
		const people = this.viewer.copies;
		const weight = smoothstep(this.fade);
		for (let person = 0; person < people; person++) {
			const incoming = this.sampleTrack(track, person, 0);
			const outgoing = this.outgoing ? this.sampleTrack(this.outgoing, person, 1) : null;
			const pose =
				incoming && outgoing ? blend(outgoing, incoming, weight) : (incoming ?? outgoing!);
			this.overlay?.(person, pose);
			this.viewer.poseCopy(person, {
				rotations: Array.from({ length: this.rig.boneNames.length }, (_, j) =>
					quaternionMatrix(pose.rotations, 4 * j)
				),
				rootOffset: pose.rootOffset,
				expressions: pose.expressions
			});
		}
	}

	/** A person's pose in a track, placed; null when the track's clip has fewer people. */
	private sampleTrack(track: Track, person: number, slot: number): Pose | null {
		if (person >= track.sampler.clip.people) return null;
		const key = 2 * person + slot;
		this.poses[key] ??= bindPose(this.rig);
		const pose = track.sampler.sample(person, track.time, this.poses[key]);
		return this.place(pose, track.placement);
	}

	/** Turn a pose about world z by the placement's yaw and shift it, in place. */
	private place(pose: Pose, { yaw, shift }: Placement): Pose {
		if (yaw === 0 && shift.every((v) => v === 0)) return pose;
		const c = Math.cos(yaw);
		const s = Math.sin(yaw);
		// The root's world rotation is B L (bind, then the clip's rotation): turned, Rz B L, so
		// the clip's rotation becomes B^T Rz B L.
		const b = this.rootBind;
		const turn = yawMatrix(yaw);
		const local = multiply3(
			transpose3(b),
			multiply3(turn, multiply3(b, quaternionMatrix(pose.rotations, 0)))
		);
		matrixToQuaternion(local, pose.rotations, 0);
		// The root's world position is the bind position plus the offset: turned and shifted.
		const [bx, by] = this.rootBindPosition;
		const x = bx + pose.rootOffset[0];
		const y = by + pose.rootOffset[1];
		pose.rootOffset[0] = c * x - s * y + shift[0] - bx;
		pose.rootOffset[1] = s * x + c * y + shift[1] - by;
		pose.rootOffset[2] += shift[2];
		return pose;
	}

	/**
	 * Where to put a clip so its first person starts where the showing first person stands,
	 * facing the same way; height stays the clip's own.
	 */
	private placementFor(sampler: ClipSampler): Placement {
		const showing = this.viewer.avatars()[0];
		if (!showing) return NO_PLACEMENT;
		const start = sampler.sample(0, 0, bindPose(this.rig));
		const startWorld = multiply3(this.rootBind, quaternionMatrix(start.rotations, 0));
		const root = showing.boneWorld[0];
		const yaw = heading(rotation3(root)) - heading(startWorld);
		const c = Math.cos(yaw);
		const s = Math.sin(yaw);
		const [bx, by] = this.rootBindPosition;
		const x = bx + start.rootOffset[0];
		const y = by + start.rootOffset[1];
		return { yaw, shift: [root[3] - (c * x - s * y), root[7] - (s * x + c * y), 0] };
	}
}

function smoothstep(t: number): number {
	return t * t * (3 - 2 * t);
}

/** `a` blended towards `b` by `t`, into a new pose. */
function blend(a: Pose, b: Pose, t: number): Pose {
	const rotations = new Float64Array(a.rotations.length);
	for (let j = 0; j < rotations.length; j += 4)
		nlerp(a.rotations, j, b.rotations, j, t, rotations, j);
	const expressions = new Float32Array(a.expressions.length);
	for (let e = 0; e < expressions.length; e++) {
		expressions[e] = a.expressions[e] + t * (b.expressions[e] - a.expressions[e]);
	}
	return {
		rotations,
		rootOffset: [0, 1, 2].map(
			(i) => a.rootOffset[i] + t * (b.rootOffset[i] - a.rootOffset[i])
		) as Vec3,
		expressions
	};
}
