/**
 * Poses for trying the avatar out, as rotations in world axes (x to the avatar's left, y to its
 * back, z up), the convention of notebook 05.
 */

import type { Vec3 } from '#lib/math.ts';

export interface Joint {
	label: string;
	/** The bones that turn, with the share of the angle each takes. */
	bones: Record<string, number>;
	axis: Vec3;
	/** Range in degrees. */
	min: number;
	max: number;
}

/**
 * Raising an arm turns the shoulder girdle too, about a third of the way: turning the upper arm
 * alone crushes the shoulder under linear blend skinning.
 */
const SHOULDER_L = { 'clavicle.L': 0.2, 'shoulder01.L': 0.15, 'upperarm01.L': 0.65 };
const SHOULDER_R = { 'clavicle.R': 0.2, 'shoulder01.R': 0.15, 'upperarm01.R': 0.65 };

export const JOINTS: Joint[] = [
	{ label: 'Head turn', bones: { head: 1 }, axis: [0, 0, 1], min: -70, max: 70 },
	{ label: 'Head nod', bones: { head: 1 }, axis: [1, 0, 0], min: -40, max: 40 },
	{ label: 'Head tilt', bones: { head: 1 }, axis: [0, 1, 0], min: -35, max: 35 },
	{ label: 'Torso twist', bones: { spine02: 1 }, axis: [0, 0, 1], min: -40, max: 40 },
	{ label: 'Torso bend', bones: { spine02: 1 }, axis: [1, 0, 0], min: -30, max: 45 },
	{ label: 'Left arm raise', bones: SHOULDER_L, axis: [0, -1, 0], min: -45, max: 120 },
	{ label: 'Left arm forward', bones: { 'upperarm01.L': 1 }, axis: [-1, 0, 0], min: -60, max: 90 },
	{ label: 'Left elbow', bones: { 'lowerarm01.L': 1 }, axis: [0, 0, 1], min: 0, max: 130 },
	{ label: 'Right arm raise', bones: SHOULDER_R, axis: [0, 1, 0], min: -45, max: 120 },
	{ label: 'Right arm forward', bones: { 'upperarm01.R': 1 }, axis: [-1, 0, 0], min: -60, max: 90 },
	{ label: 'Right elbow', bones: { 'lowerarm01.R': 1 }, axis: [0, 0, -1], min: 0, max: 130 },
	{ label: 'Left hip', bones: { 'upperleg01.L': 1 }, axis: [1, 0, 0], min: -30, max: 100 },
	{ label: 'Left knee', bones: { 'lowerleg01.L': 1 }, axis: [-1, 0, 0], min: 0, max: 130 },
	{ label: 'Right hip', bones: { 'upperleg01.R': 1 }, axis: [1, 0, 0], min: -30, max: 100 },
	{ label: 'Right knee', bones: { 'lowerleg01.R': 1 }, axis: [-1, 0, 0], min: 0, max: 130 }
];

export const PRESETS: Record<string, Record<string, number>> = {
	Rest: {},
	'Arms up': { 'Left arm raise': 70, 'Right arm raise': 70 },
	Step: {
		'Left arm raise': 69,
		'Right arm raise': 69,
		'Left hip': 29,
		'Left knee': 34,
		'Head turn': 29
	},
	Sitting: { 'Left hip': 90, 'Left knee': 90, 'Right hip': 90, 'Right knee': 90 },
	Wave: { 'Right arm raise': 110, 'Right elbow': 60, 'Head turn': -20 }
};

/** Rotation vectors per bone from slider values in degrees by joint label. */
export function rotations(
	values: Record<string, number>,
	joints: Joint[] = JOINTS
): Record<string, Vec3> {
	const out: Record<string, Vec3> = {};
	for (const joint of joints) {
		const degrees = values[joint.label] ?? 0;
		if (degrees === 0) continue;
		const angle = (degrees * Math.PI) / 180;
		for (const [bone, share] of Object.entries(joint.bones)) {
			const current = out[bone] ?? [0, 0, 0];
			out[bone] = [
				current[0] + share * angle * joint.axis[0],
				current[1] + share * angle * joint.axis[1],
				current[2] + share * angle * joint.axis[2]
			];
		}
	}
	return out;
}
