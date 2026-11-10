/**
 * Scripted sequences: a list of steps the avatars perform one after another, and cues layered
 * over them at set times. Written as JSON:
 *
 * ```json
 * {
 *   "name": "greeting",
 *   "loop": false,
 *   "steps": [
 *     { "clip": "walk_in", "from": 0, "to": 4.5, "speed": 1, "repeat": 1, "fade": 0.4 },
 *     { "pose": "Wave", "hold": 2 },
 *     { "wait": 1 }
 *   ],
 *   "cues": [
 *     { "at": 0.5, "for": 3, "character": 0, "expression": { "mouthSmile": 0.7 }, "fade": 0.3 },
 *     { "at": 1, "for": 4, "character": "all", "look": "camera" },
 *     { "at": 0, "camera": "video" }
 *   ]
 * }
 * ```
 *
 * Steps: `clip` plays a motion clip, by the name it was loaded under or as a `.motion` URL,
 * optionally trimmed (`from`, `to` in seconds), faster or slower (`speed`) and `repeat`ed; `pose`
 * holds one of the pose presets for `hold` seconds; `wait` lets whatever plays go on. Each step
 * fades in over `fade` seconds (the player's transition time if left out).
 *
 * Cues start `at` seconds into the sequence and last `for` seconds, fading in and out over
 * `fade`: `expression` sets expression weights (ARKit names like `mouthSmileLeft`, or one name
 * for both sides like `mouthSmile`, or the avatar's own like `mouthSmile_L`); `look` turns the
 * head towards `"camera"`, `"partner"` (the other character's head) or a point `[x, y, z]`;
 * `camera` switches to the `"video"`'s view or a `"front"` view of everyone. `character` is a
 * person's index in the clips, or `"all"` (the default).
 */

import type { Vec3 } from '#lib/math.ts';

export type Step =
	| {
			kind: 'clip';
			clip: string;
			from: number;
			to: number | null;
			speed: number;
			repeat: number;
			fade: number | null;
	  }
	| { kind: 'pose'; pose: string; hold: number; fade: number | null }
	| { kind: 'wait'; seconds: number };

export type Target = 'camera' | 'partner' | Vec3;
export type Character = number | 'all';

export type Cue =
	| {
			kind: 'expression';
			at: number;
			duration: number;
			fade: number;
			character: Character;
			weights: Record<string, number>;
	  }
	| {
			kind: 'look';
			at: number;
			duration: number;
			fade: number;
			character: Character;
			target: Target;
	  }
	| { kind: 'camera'; at: number; view: 'video' | 'front' };

export interface Sequence {
	name: string;
	loop: boolean;
	steps: Step[];
	cues: Cue[];
}

/** A step and when it plays, in seconds from the start of the sequence. */
export interface Scheduled {
	step: Step;
	start: number;
	end: number;
}

const DEFAULT_CUE_FADE = 0.3;

/** Read a sequence from parsed JSON, saying where it is wrong when it is. */
export function parseSequence(json: unknown): Sequence {
	const root = object(json, 'sequence');
	const steps = array(root.steps, 'steps').map((step, i) => parseStep(step, `steps[${i}]`));
	if (!steps.length) throw new Error('steps: a sequence needs at least one step');
	const cues = root.cues === undefined ? [] : array(root.cues, 'cues');
	return {
		name: root.name === undefined ? 'sequence' : string(root.name, 'name'),
		loop: root.loop === undefined ? false : boolean(root.loop, 'loop'),
		steps,
		cues: cues.map((cue, i) => parseCue(cue, `cues[${i}]`))
	};
}

function parseStep(json: unknown, path: string): Step {
	const step = object(json, path);
	const fade = step.fade === undefined ? null : nonNegative(step.fade, `${path}.fade`);
	if ('clip' in step) {
		known(step, ['clip', 'from', 'to', 'speed', 'repeat', 'fade'], path);
		const from = step.from === undefined ? 0 : nonNegative(step.from, `${path}.from`);
		const to = step.to === undefined ? null : nonNegative(step.to, `${path}.to`);
		if (to !== null && to <= from) throw new Error(`${path}.to: must be after from (${from})`);
		const speed = step.speed === undefined ? 1 : positive(step.speed, `${path}.speed`);
		const repeat = step.repeat === undefined ? 1 : positive(step.repeat, `${path}.repeat`);
		return { kind: 'clip', clip: string(step.clip, `${path}.clip`), from, to, speed, repeat, fade };
	}
	if ('pose' in step) {
		known(step, ['pose', 'hold', 'fade'], path);
		return {
			kind: 'pose',
			pose: string(step.pose, `${path}.pose`),
			hold: positive(step.hold, `${path}.hold`),
			fade
		};
	}
	if ('wait' in step) {
		known(step, ['wait'], path);
		return { kind: 'wait', seconds: positive(step.wait, `${path}.wait`) };
	}
	throw new Error(`${path}: a step needs "clip", "pose" or "wait"`);
}

function parseCue(json: unknown, path: string): Cue {
	const cue = object(json, path);
	const at = nonNegative(cue.at, `${path}.at`);
	if ('camera' in cue) {
		known(cue, ['at', 'camera'], path);
		const view = cue.camera;
		if (view !== 'video' && view !== 'front') {
			throw new Error(`${path}.camera: "video" or "front", not ${JSON.stringify(view)}`);
		}
		return { kind: 'camera', at, view };
	}
	const duration = positive(cue.for, `${path}.for`);
	const fade = cue.fade === undefined ? DEFAULT_CUE_FADE : nonNegative(cue.fade, `${path}.fade`);
	const character = parseCharacter(cue.character, `${path}.character`);
	if ('expression' in cue) {
		known(cue, ['at', 'for', 'fade', 'character', 'expression'], path);
		const weights: Record<string, number> = {};
		for (const [name, value] of Object.entries(object(cue.expression, `${path}.expression`))) {
			weights[name] = number(value, `${path}.expression.${name}`);
		}
		return { kind: 'expression', at, duration, fade, character, weights };
	}
	if ('look' in cue) {
		known(cue, ['at', 'for', 'fade', 'character', 'look'], path);
		return {
			kind: 'look',
			at,
			duration,
			fade,
			character,
			target: parseTarget(cue.look, `${path}.look`)
		};
	}
	throw new Error(`${path}: a cue needs "expression", "look" or "camera"`);
}

function parseCharacter(json: unknown, path: string): Character {
	if (json === undefined || json === 'all') return 'all';
	const index = nonNegative(json, path);
	if (!Number.isInteger(index)) throw new Error(`${path}: a person's index, or "all"`);
	return index;
}

function parseTarget(json: unknown, path: string): Target {
	if (json === 'camera' || json === 'partner') return json;
	if (Array.isArray(json) && json.length === 3) {
		return json.map((v, i) => number(v, `${path}[${i}]`)) as Vec3;
	}
	throw new Error(`${path}: "camera", "partner" or a point [x, y, z]`);
}

/**
 * When each step plays. Clips last their (trimmed) length at their speed, times their repeats;
 * `clipLength` gives a clip's full length in seconds.
 */
export function schedule(steps: Step[], clipLength: (clip: string) => number): Scheduled[] {
	let time = 0;
	return steps.map((step) => {
		let length: number;
		if (step.kind === 'clip') {
			const end = Math.min(step.to ?? Infinity, clipLength(step.clip));
			length = (Math.max(end - step.from, 0) / step.speed) * step.repeat;
		} else length = step.kind === 'pose' ? step.hold : step.seconds;
		const scheduled = { step, start: time, end: time + length };
		time += length;
		return scheduled;
	});
}

/**
 * How strongly a cue lasting `duration` seconds applies `elapsed` seconds after it starts: rising
 * over `fade`, holding, and falling over `fade` before its end.
 */
export function envelope(elapsed: number, duration: number, fade: number): number {
	if (elapsed < 0 || elapsed > duration) return 0;
	if (fade <= 0) return 1;
	const rise = Math.min(elapsed / fade, 1);
	const fall = Math.min((duration - elapsed) / fade, 1);
	const t = Math.min(rise, fall);
	return t * t * (3 - 2 * t);
}

/**
 * The avatar's expressions an expression name stands for: the avatar's own name, or ARKit's
 * (`mouthSmileLeft` is `mouthSmile_L`), or a name without a side for both sides.
 */
export function expressionIndices(name: string, expressions: readonly string[]): number[] {
	const exact = expressions.indexOf(name);
	if (exact >= 0) return [exact];
	for (const [side, suffix] of [
		['Left', '_L'],
		['Right', '_R']
	]) {
		if (name.endsWith(side)) {
			const index = expressions.indexOf(name.slice(0, -side.length) + suffix);
			if (index >= 0) return [index];
		}
	}
	return ['_L', '_R'].map((suffix) => expressions.indexOf(name + suffix)).filter((i) => i >= 0);
}

function known(json: Record<string, unknown>, keys: string[], path: string): void {
	for (const key of Object.keys(json)) {
		if (!keys.includes(key)) throw new Error(`${path}: unknown key "${key}"`);
	}
}

function object(json: unknown, path: string): Record<string, unknown> {
	if (typeof json !== 'object' || json === null || Array.isArray(json)) {
		throw new Error(`${path}: expected an object`);
	}
	return json as Record<string, unknown>;
}

function array(json: unknown, path: string): unknown[] {
	if (!Array.isArray(json)) throw new Error(`${path}: expected a list`);
	return json;
}

function string(json: unknown, path: string): string {
	if (typeof json !== 'string' || !json) throw new Error(`${path}: expected a name`);
	return json;
}

function boolean(json: unknown, path: string): boolean {
	if (typeof json !== 'boolean') throw new Error(`${path}: expected true or false`);
	return json;
}

function number(json: unknown, path: string): number {
	if (typeof json !== 'number' || !Number.isFinite(json))
		throw new Error(`${path}: expected a number`);
	return json;
}

function nonNegative(json: unknown, path: string): number {
	const value = number(json, path);
	if (value < 0) throw new Error(`${path}: must not be negative`);
	return value;
}

function positive(json: unknown, path: string): number {
	const value = number(json, path);
	if (value <= 0) throw new Error(`${path}: must be more than zero`);
	return value;
}
