/**
 * Procedural grooms: guide strands grown over the scalp from a few style parameters, and the many
 * render strands that are interpolated between them on the GPU.
 *
 * Everything here is in the head bone's frame (x to the avatar's left, y up, z forward, metres),
 * the frame the scalp is measured in. Guides are grown the way a stylist combs: from the root
 * along a flow over the scalp (away from the parting, or back off the forehead), lifted by the
 * hair's volume, bending under gravity, kept out of the head and body, then waved.
 */

import { pushOutOfBody as pushOutOfBindBody, transform, type Body } from '#lib/hair/colliders.ts';
import { add, cross, dot, normalize, scale, sub, type Scalp } from '#lib/hair/scalp.ts';
import type { Mat4, Vec3 } from '#lib/math.ts';

export type Style = 'loose' | 'ponytail';

/** What a groom's geometry depends on; changing any of these grows the hair anew. */
export interface GroomShape {
	style: Style;
	/** Strand length in metres (the ponytail's tail, for a ponytail). */
	length: number;
	/** Spread of strand lengths, 0 (all equal) to 1. */
	lengthVariation: number;
	/** Rendered strands. */
	strands: number;
	/** Hairline height in degrees over the default: positive recedes, negative lowers it. */
	hairline: number;
	/** Depth of a widow's peak, 0 to 1. */
	widowsPeak: number;
	/** Parting position across the top, -1 (avatar's right) to 1 (its left). */
	parting: number;
	/** How much the top is combed straight back instead of away from the parting, 0 to 1. */
	combBack: number;
	/** How far the hair stands off the scalp, in metres. */
	volume: number;
	/** Share of the hair at the sides that falls in front of the shoulders, 0 to 1. */
	front: number;
	/** Wave amplitude in metres, and wavelength. */
	wave: number;
	waveLength: number;
	/** Curl radius in metres (0: none), and the length of hair per turn. */
	curl: number;
	curlPitch: number;
	/** How tightly strands gather into clumps towards their tips, 0 to 1. */
	clumping: number;
	/** Small random kinks along each strand, 0 to 1. */
	frizz: number;
	/** Share of flyaway strands, 0 to 1. */
	flyaways: number;
	/** Resistance to bending, 0 (silky) to 1 (stiff). */
	stiffness: number;
}

/** How the hair looks; changes apply at once. */
export interface HairLook {
	/** Melanin, 0 (platinum) to 1 (black). */
	melanin: number;
	/** Share of red pheomelanin in it, 0 to 1. */
	redness: number;
	/** Strand-to-strand colour variation, 0 to 1. */
	variation: number;
	/** Fibre diameter in micrometres. */
	thickness: number;
	/** Cuticle roughness, 0 (glossy) to 1 (matte). */
	roughness: number;
	/** Strength of the shadows hair casts on the skin, 0 to 1. */
	shadow: number;
}

export interface Preset {
	shape: GroomShape;
	look: HairLook;
}

const BASE: GroomShape = {
	style: 'loose',
	length: 0.5,
	lengthVariation: 0.25,
	strands: 70000,
	hairline: 0,
	widowsPeak: 0,
	parting: 0,
	combBack: 0,
	volume: 0.01,
	front: 0.4,
	wave: 0,
	waveLength: 0.1,
	curl: 0,
	curlPitch: 0.04,
	clumping: 0.4,
	frizz: 0.2,
	flyaways: 0.04,
	stiffness: 0.3
};

const LOOK: HairLook = {
	melanin: 0.6,
	redness: 0.3,
	variation: 0.3,
	thickness: 80,
	roughness: 0.35,
	shadow: 0.45
};

export const PRESETS: Record<string, Preset> = {
	// After the user's reference photo: near-black, long and straight, combed back off the
	// forehead, over the ears and down in front of the shoulders, wispy ends, many flyaways.
	Default: {
		shape: {
			...BASE,
			length: 0.5,
			lengthVariation: 0.3,
			hairline: 0,
			widowsPeak: 0.5,
			combBack: 0.85,
			volume: 0.016,
			front: 0.85,
			wave: 0.004,
			waveLength: 0.16,
			clumping: 0.45,
			frizz: 0.3,
			flyaways: 0.06
		},
		look: { ...LOOK, melanin: 0.88, redness: 0.25, roughness: 0.32 }
	},
	'Long straight': {
		shape: { ...BASE, length: 0.6, parting: 0.35, front: 0.6, clumping: 0.3, frizz: 0.12 },
		look: { ...LOOK, melanin: 0.55, redness: 0.3 }
	},
	'Wavy blonde': {
		shape: {
			...BASE,
			length: 0.36,
			parting: -0.3,
			volume: 0.016,
			front: 0.5,
			wave: 0.018,
			waveLength: 0.09,
			curl: 0.004,
			curlPitch: 0.09,
			clumping: 0.55,
			frizz: 0.3,
			flyaways: 0.05
		},
		look: { ...LOOK, melanin: 0.22, redness: 0.25, thickness: 70, roughness: 0.4 }
	},
	'Red curls': {
		shape: {
			...BASE,
			length: 0.32,
			strands: 80000,
			parting: 0.1,
			volume: 0.03,
			front: 0.5,
			wave: 0.01,
			waveLength: 0.12,
			curl: 0.009,
			curlPitch: 0.03,
			clumping: 0.8,
			frizz: 0.25,
			flyaways: 0.05,
			stiffness: 0.5
		},
		look: { ...LOOK, melanin: 0.45, redness: 1, thickness: 75, roughness: 0.45 }
	},
	Short: {
		shape: {
			...BASE,
			length: 0.09,
			lengthVariation: 0.2,
			strands: 60000,
			parting: 0.5,
			combBack: 0.3,
			volume: 0.012,
			front: 0,
			clumping: 0.3,
			frizz: 0.15,
			flyaways: 0.02,
			stiffness: 0.6
		},
		look: { ...LOOK, melanin: 0.75, redness: 0.2, thickness: 85 }
	},
	Ponytail: {
		shape: {
			...BASE,
			style: 'ponytail',
			length: 0.4,
			lengthVariation: 0.2,
			combBack: 1,
			volume: 0.003,
			front: 0,
			wave: 0.006,
			waveLength: 0.15,
			clumping: 0.4,
			frizz: 0.2,
			flyaways: 0.04
		},
		look: { ...LOOK, melanin: 0.45, redness: 0.45 }
	}
};

/** Points per guide strand. */
export const GUIDE_POINTS = 32;
/** Points of a ponytail's guides that run over the scalp to the tie, the rest are its tail. */
const PONYTAIL_SCALP_POINTS = 10;
/** Values per render strand in `Groom.strandData`: four texels. */
export const STRAND_TEXELS = 4;

export interface Groom {
	shape: GroomShape;
	guides: number;
	/** `(guides * GUIDE_POINTS * 4)` rest positions in the head's frame and pinning, 0 to 1. */
	rest: Float32Array;
	strands: number;
	/**
	 * `(strands * STRAND_TEXELS * 4)` per strand: four guide indices; their weights; tip
	 * parameter, random seed, width scale and flyaway flag; root offset from the guides' blend
	 * (head frame) and depth in the hair, 0 (under) to 1 (on top).
	 */
	strandData: Float32Array;
	/** Points per render strand. */
	renderPoints: number;
	/** Longest guide, in metres. */
	reach: number;
}

/**
 * Grow a groom over `scalp`, avoiding `body` as it stands in the bind pose, with bone transforms
 * `bind` and head transform `head` (row-major, world).
 */
export function growGroom(
	shape: GroomShape,
	scalp: Scalp,
	body: Body,
	bind: Mat4[],
	head: Mat4
): Groom {
	const random = rng(7);
	const around = surroundings(body, bind, head);
	const guideCount = Math.round(Math.min(2400, Math.max(800, shape.strands / 40)));

	// Guide roots: a jittered Fibonacci lattice over the sphere, kept where hair grows.
	const roots: Vec3[] = [];
	const total = Math.round(guideCount * 2.6);
	for (let i = 0; i < total && roots.length < guideCount * 1.5; i++) {
		const y = 1 - (2 * (i + 0.5)) / total;
		const ring = Math.sqrt(1 - y * y);
		const angle = i * Math.PI * (3 - Math.sqrt(5));
		const d = normalize([ring * Math.sin(angle), y, ring * Math.cos(angle)]);
		if (hairDensity(d, shape) > 0.5) roots.push(d);
	}
	const guides = roots.length;
	const rest = new Float32Array(guides * GUIDE_POINTS * 4);
	const guideRoots: Vec3[] = [];
	const forward = roots.map((d) => falls(d, shape));
	let reach = 0;
	roots.forEach((d, g) => {
		const points =
			shape.style === 'ponytail'
				? growPonytail(d, shape, scalp, around, random)
				: growLoose(d, shape, scalp, around, forward[g], random);
		let length = 0;
		points.forEach((p, k) => {
			if (k > 0) length += Math.hypot(...sub(p, points[k - 1]));
			// Hair lying on the head is held there by the hair over and under it: only what
			// hangs free swings. (Follow-the-leader alone would let it slide off, as the hair
			// hanging further on cannot pull on it.)
			const lying = 0.02 + 1.5 * shape.volume;
			const onScalp = 0.8 * (1 - smoothstep(lying, lying + 0.025, clearance(scalp, p)));
			rest.set(p, 4 * (g * GUIDE_POINTS + k));
			rest[4 * (g * GUIDE_POINTS + k) + 3] = Math.max(pinning(k, length, shape), onScalp);
		});
		reach = Math.max(reach, length);
		guideRoots.push(points[0]);
	});

	// Render strands: roots spread at random, each following the guides around it.
	const grid = new GuideGrid(guideRoots, 0.01);
	const tipStart = shape.style === 'ponytail' ? PONYTAIL_SCALP_POINTS / (GUIDE_POINTS - 1) : 0;
	const strandData = new Float32Array(shape.strands * STRAND_TEXELS * 4);
	let made = 0;
	while (made < shape.strands) {
		const d = normalize([gauss(random), gauss(random), gauss(random)]);
		const density = hairDensity(d, shape);
		if (density <= 0 || random() > density) continue;
		const root = scalp.surface(d, 0.0004);
		const side = falls(d, shape);
		const near = grid.nearest(root, 5, (g) => forward[g] === side && sameSide(roots[g], d, shape));
		if (near.length === 0) continue;
		// A smooth kernel over the nearest guides, so neighbouring strands blend alike.
		const h = near[near.length - 1].distance + 1e-4;
		const used = near.slice(0, 4);
		const weights = used.map(({ distance }) => (1 - (distance / h) ** 2) ** 2 + 1e-6);
		const sum = weights.reduce((a, b) => a + b);
		const o = made * STRAND_TEXELS * 4;
		let blended: Vec3 = [0, 0, 0];
		for (let k = 0; k < 4; k++) {
			const pick = used[Math.min(k, used.length - 1)];
			const w = k < used.length ? weights[k] / sum : 0;
			strandData[o + k] = pick.index;
			strandData[o + 4 + k] = w;
			blended = add(blended, scale(guideRoots[pick.index], w));
		}
		const edge = density < 1;
		const flyaway = random() < shape.flyaways;
		// Shorter strands end early; fine baby hairs grow along the hairline.
		const tip = edge
			? tipStart + (1 - tipStart) * (0.05 + 0.2 * random())
			: tipStart + (1 - tipStart) * (1 - shape.lengthVariation * random() ** 1.5);
		strandData[o + 8] = tip;
		strandData[o + 9] = random();
		strandData[o + 10] = (edge ? 0.55 : 1) * (0.75 + 0.5 * random());
		strandData[o + 11] = flyaway ? 1 : 0;
		strandData.set(sub(root, blended), o + 12);
		// Crown hair lies on top of the hair that grows below it.
		strandData[o + 15] = Math.min(1, Math.max(0, 0.5 + 0.6 * (d[1] - 0.2)));
		made++;
	}

	const curlSegment = shape.curl > 0 ? shape.curlPitch / 8 : 0.012;
	const renderPoints = Math.round(
		Math.min(128, Math.max(16, (reach * 1.05) / Math.min(0.012, curlSegment)))
	);
	return { shape, guides, rest, strands: made, strandData, renderPoints, reach };
}

/**
 * Hair density at scalp direction `d` (head frame, from the cranium centre): 1 inside the
 * hairline, fading to 0 across it.
 */
export function hairDensity(d: Vec3, shape: GroomShape): number {
	const elevation = (Math.asin(d[1]) * 180) / Math.PI;
	const azimuth = Math.abs((Math.atan2(d[0], d[2]) * 180) / Math.PI);
	let line = interpolate(HAIRLINE, azimuth) + shape.hairline * Math.max(0, 1 - azimuth / 120);
	line -= shape.widowsPeak * 7 * Math.max(0, 1 - azimuth / 9);
	// A natural hairline wanders a little.
	line += 1.5 * Math.sin(azimuth * 0.31) + Math.sin(azimuth * 0.73 + 1);
	return smoothstep(line - 1, line + 2.5, elevation);
}

/** Hairline elevation in degrees by azimuth from the face (0) round to the back (180). */
const HAIRLINE: [number, number][] = [
	[0, 27],
	[25, 26],
	[45, 21],
	[62, 11],
	[74, 0],
	[80, -16],
	[86, -14],
	[92, -2],
	[104, 2],
	[114, -4],
	[122, -26],
	[136, -42],
	[155, -52],
	[180, -54]
];

/**
 * Whether hair from scalp direction `d` falls in front of the shoulders (1) or behind (-1): by
 * where it grows, so neighbours fall together. Strands blend only guides falling the same way, or
 * they would hang between them, across the shoulder.
 */
function falls(d: Vec3, shape: GroomShape): number {
	const azimuth = Math.abs((Math.atan2(d[0], d[2]) * 180) / Math.PI);
	return azimuth < 140 * shape.front ? 1 : -1;
}

/** Hair at the top doesn't blend across the parting, so the parting stays a line. */
function sameSide(guide: Vec3, d: Vec3, shape: GroomShape): boolean {
	if (shape.style === 'ponytail' || shape.combBack > 0.7) return true;
	if (d[1] < 0.45 || d[2] < -0.55) return true;
	const part = partingX(shape);
	return Math.sign(guide[0] - part) === Math.sign(d[0] - part);
}

/** Height of head-frame point `p` over the scalp, or infinity below the jaw. */
function clearance(scalp: Scalp, p: Vec3): number {
	const q = sub(p, scalp.centre);
	const r = Math.hypot(...q);
	const surface = scalp.radius(scale(q, 1 / r));
	return surface > 0 ? r - surface : Infinity;
}

function partingX(shape: GroomShape): number {
	return shape.parting * 0.4;
}

/** A loose strand from scalp direction `d`: combed along the scalp, then falling. */
function growLoose(
	d: Vec3,
	shape: GroomShape,
	scalp: Scalp,
	body: Surroundings,
	forward: number,
	random: () => number
): Vec3[] {
	const n = scalp.normal(d);
	const root = scalp.surface(d, 0.0004);
	const top = smoothstep(0.2, 0.75, d[1]);
	const away = Math.sign(d[0] - partingX(shape)) || 1;
	// The comb: down, away from the parting at the top, back off the forehead.
	// Hair at the front sweeps back past the temples, whatever the parting, so it frames the face
	// instead of falling over it.
	const front = smoothstep(-0.2, 0.5, d[2]);
	let flow: Vec3 = [
		away * 1.1 * top * (1 - shape.combBack) + 0.3 * d[0],
		// Up over the curve of the forehead at the front, down elsewhere.
		-1 + 1.6 * front,
		-0.15 - 1.4 * front * (1 - shape.combBack) - shape.combBack * (0.6 + 2.2 * front) * (0.4 + top)
	];
	flow = normalize(sub(flow, scale(n, dot(flow, n))));
	const lift = 0.06 + 4 * shape.volume + 0.1 * shape.stiffness;
	let direction = normalize(add(flow, scale(n, lift)));
	const segment = (shape.length * (1 - 0.15 * shape.lengthVariation)) / (GUIDE_POINTS - 1);
	const sag = (0.32 - 0.22 * shape.stiffness) * Math.min(1, Math.max(0.6, segment / 0.016));

	const points: Vec3[] = [root];
	let travelled = 0;
	let onHead = false;
	for (let k = 1; k < GUIDE_POINTS; k++) {
		const previous = points[k - 1];
		// Near the root, hair lying on the head follows the comb; past it, gravity bends it.
		if (!onHead || travelled > 0.06) direction = normalize(add(direction, [0, -sag, 0]));
		const height = previous[1] - scalp.centre[1];
		// Past the ear and down to the chest, steer in front of the shoulder or behind it, and in
		// towards the neck, where hair falls rather than resting on the shoulder.
		// Only until it is clear of the shoulder: in front of the collarbone, or behind the back.
		const clear = forward > 0 ? previous[2] > 0.07 : previous[2] < -0.12;
		if (height < -0.04 && height > -0.3 && Math.abs(previous[0]) > 0.04 && !clear) {
			const inward = Math.abs(previous[0]) > 0.075 ? -0.1 * Math.sign(previous[0]) : 0;
			direction = normalize(add(direction, [inward, 0, 0.24 * forward]));
		}
		let p = add(previous, scale(direction, segment));
		travelled += segment;
		// Strands from higher up pile on top of those that grew below them.
		const lying = 0.0015 + shape.volume * smoothstep(0, 0.14, travelled);
		p = keepOut(p, previous, segment, lying, scalp, body);
		// Over the scalp, above its widest part (all over, if short), combed hair lies on the head
		// rather than leaving it
		// along a tangent.
		const q = sub(p, scalp.centre);
		const over = normalize(q);
		onHead = (height > -0.015 || shape.length < 0.2) && hairDensity(over, shape) > 0;
		if (onHead && clearance(scalp, p) > lying) {
			const surface = scalp.surface(over, lying);
			p = add(previous, scale(normalize(sub(surface, previous)), segment));
		}
		direction = normalize(sub(p, previous));
		points.push(p);
	}
	return wave(points, d, shape, scalp, body, random);
}

/** A ponytail strand: pulled over the scalp to the tie, then hanging from it. */
function growPonytail(
	d: Vec3,
	shape: GroomShape,
	scalp: Scalp,
	body: Surroundings,
	random: () => number
): Vec3[] {
	const tieDirection = normalize([0, 0.2, -1]);
	const tie = scalp.surface(tieDirection, 0.012);
	const points: Vec3[] = [];
	const angle = Math.acos(Math.min(1, Math.max(-1, dot(d, tieDirection))));
	// Where on the tie's ring this strand arrives, keeping neighbours together.
	const across = normalize(sub(d, scale(tieDirection, dot(d, tieDirection))));
	const ring = add(tie, scale(across, 0.011 * Math.min(1, angle / 1.2)));
	const last = PONYTAIL_SCALP_POINTS;
	for (let k = 0; k <= last; k++) {
		const t = k / last;
		if (k === last) {
			points.push(ring);
			continue;
		}
		const dir = slerp(d, tieDirection, t);
		const clearance = 0.0006 + shape.volume * t + 0.006 * smoothstep(0.7, 1, t);
		points.push(scalp.surface(dir, clearance));
	}
	let direction = normalize([0, 0.35, -1]);
	const segment = shape.length / (GUIDE_POINTS - 1 - last);
	const sag = 0.2 - 0.12 * shape.stiffness;
	for (let k = last + 1; k < GUIDE_POINTS; k++) {
		const previous = points[k - 1];
		direction = normalize(add(direction, [0, -sag, 0]));
		const spread = sub(previous, tie);
		direction = normalize(add(direction, scale(spread, 0.6)));
		let p = add(previous, scale(direction, segment));
		p = keepOut(p, previous, segment, 0.004, scalp, body);
		direction = normalize(sub(p, previous));
		points.push(p);
	}
	return wave(points, d, shape, scalp, body, random, last);
}

/** Lateral waves along a strand, coherent between neighbours, kept clear of the body. */
function wave(
	points: Vec3[],
	d: Vec3,
	shape: GroomShape,
	scalp: Scalp,
	body: Surroundings,
	random: () => number,
	from = 0
): Vec3[] {
	if (shape.wave <= 0) return points;
	const phase = 9 * d[0] + 5 * d[2] + 4 * d[1] + 0.6 * random();
	let arc = 0;
	const out = points.map((p) => [...p] as Vec3);
	for (let k = from + 1; k < points.length; k++) {
		arc += Math.hypot(...sub(points[k], points[k - 1]));
		const tangent = normalize(sub(points[k], points[k - 1]));
		const outward = normalize(sub(points[k], scalp.centre));
		const lateral = normalize(cross(tangent, outward));
		const amount = shape.wave * smoothstep(0.02, 0.1, arc);
		const offset = amount * Math.sin((2 * Math.PI * arc) / shape.waveLength + phase);
		const p = add(points[k], scale(lateral, offset));
		scalp.pushOut(p, 0.0015 + shape.volume * 0.5);
		pushOutOfBody(p, body, 0.004);
		out[k] = p;
	}
	return out;
}

/** Keep a new point out of the head and body, at `segment` from the previous one. */
function keepOut(
	p: Vec3,
	previous: Vec3,
	segment: number,
	clearance: number,
	scalp: Scalp,
	body: Surroundings
): Vec3 {
	for (let i = 0; i < 3; i++) {
		const moved = scalp.pushOut(p, clearance) || pushOutOfBody(p, body, 0.005);
		p = add(previous, scale(normalize(sub(p, previous)), segment));
		if (!moved) break;
	}
	return p;
}

/** Pinning of point `k`, `arc` metres from the root: how firmly it keeps its groomed place. */
function pinning(k: number, arc: number, shape: GroomShape): number {
	if (k <= 1) return 1;
	if (shape.style === 'ponytail') {
		// The tie holds the tail's first few points.
		return k <= PONYTAIL_SCALP_POINTS ? 1 : Math.max(0, 1 - 0.4 * (k - PONYTAIL_SCALP_POINTS)) ** 2;
	}
	const hold = 0.03 + 0.08 * shape.stiffness;
	return Math.max(0, 1 - arc / hold) ** 2;
}

/** The body in the bind pose, with the transforms between the head's frame and the world. */
interface Surroundings {
	capsules: { a: Vec3; b: Vec3; radius: number }[];
	torso: Body['torso'];
	toWorld: Mat4;
	toHead: Mat4;
}

function surroundings(body: Body, bind: Mat4[], head: Mat4): Surroundings {
	return {
		capsules: body.capsules.map((c) => ({
			a: transform(bind[c.boneA], c.a),
			b: transform(bind[c.boneB], c.b),
			radius: c.radius
		})),
		torso: body.torso,
		toWorld: head,
		toHead: invert(head)
	};
}

/** Push head-frame point `p` out of the body; returns whether it moved. */
function pushOutOfBody(p: Vec3, body: Surroundings, margin: number): boolean {
	const world = transform(body.toWorld, p);
	if (!pushOutOfBindBody(world, body.capsules, body.torso, margin)) return false;
	const back = transform(body.toHead, world);
	p[0] = back[0];
	p[1] = back[1];
	p[2] = back[2];
	return true;
}

/** Guide roots bucketed in a grid, for nearest-guide queries. */
class GuideGrid {
	private readonly cells = new Map<number, number[]>();

	constructor(
		private readonly points: Vec3[],
		private readonly size: number
	) {
		points.forEach((p, i) => {
			const key = this.key(p);
			const cell = this.cells.get(key);
			if (cell) cell.push(i);
			else this.cells.set(key, [i]);
		});
	}

	/** Up to `count` nearest points accepted by `accept`, nearest first. */
	nearest(
		p: Vec3,
		count: number,
		accept: (index: number) => boolean
	): { index: number; distance: number }[] {
		const indices: number[] = [];
		const squared: number[] = [];
		const i = Math.floor(p[0] / this.size);
		const j = Math.floor(p[1] / this.size);
		const k = Math.floor(p[2] / this.size);
		for (let ring = 1; ring <= 4 && indices.length < count; ring++) {
			indices.length = squared.length = 0;
			for (let a = i - ring; a <= i + ring; a++) {
				for (let b = j - ring; b <= j + ring; b++) {
					for (let c = k - ring; c <= k + ring; c++) {
						const cell = this.cells.get(cellKey(a, b, c));
						if (!cell) continue;
						for (const index of cell) {
							const q = this.points[index];
							const d = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2;
							if (indices.length === count && d >= squared[count - 1]) continue;
							if (!accept(index)) continue;
							// Insert in order, keeping only the nearest `count`.
							let at = Math.min(indices.length, count - 1);
							while (at > 0 && squared[at - 1] > d) {
								indices[at] = indices[at - 1];
								squared[at] = squared[at - 1];
								at--;
							}
							indices[at] = index;
							squared[at] = d;
						}
					}
				}
			}
		}
		return indices.map((index, n) => ({ index, distance: Math.sqrt(squared[n]) }));
	}

	private key(p: Vec3): number {
		const [a, b, c] = p.map((x) => Math.floor(x / this.size));
		return cellKey(a, b, c);
	}
}

function cellKey(a: number, b: number, c: number): number {
	return ((a + 512) * 1024 + (b + 512)) * 1024 + (c + 512);
}

function slerp(a: Vec3, b: Vec3, t: number): Vec3 {
	const angle = Math.acos(Math.min(1, Math.max(-1, dot(a, b))));
	if (angle < 1e-4) return a;
	const s = Math.sin(angle);
	return normalize(add(scale(a, Math.sin((1 - t) * angle) / s), scale(b, Math.sin(t * angle) / s)));
}

function invert(m: Mat4): Mat4 {
	const out = new Float64Array(16);
	for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) out[4 * i + j] = m[4 * j + i];
	for (let i = 0; i < 3; i++) {
		out[4 * i + 3] = -(out[4 * i] * m[3] + out[4 * i + 1] * m[7] + out[4 * i + 2] * m[11]);
	}
	out[15] = 1;
	return out;
}

function interpolate(table: [number, number][], x: number): number {
	for (let i = 1; i < table.length; i++) {
		if (x <= table[i][0]) {
			const [x0, y0] = table[i - 1];
			const [x1, y1] = table[i];
			return y0 + ((y1 - y0) * (x - x0)) / (x1 - x0);
		}
	}
	return table[table.length - 1][1];
}

export function smoothstep(edge0: number, edge1: number, x: number): number {
	const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
	return t * t * (3 - 2 * t);
}

/** A seeded random generator (mulberry32), so a groom grows the same each time. */
function rng(seed: number): () => number {
	let state = seed >>> 0;
	return () => {
		state = (state + 0x6d2b79f5) >>> 0;
		let t = state;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

function gauss(random: () => number): number {
	return Math.sqrt(-2 * Math.log(random() + 1e-12)) * Math.cos(2 * Math.PI * random());
}
