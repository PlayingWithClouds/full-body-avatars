import { describe, expect, test } from 'bun:test';

import { gravityInBone, gravityWeights } from '#lib/avatar/rig.ts';
import type { Vec3 } from '#lib/math.ts';

/** Top three rows of a row-major skinning matrix turning by `angle` about x. */
function turnAboutX(angle: number): number[] {
	const c = Math.cos(angle);
	const s = Math.sin(angle);
	return [1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0];
}

const DOWN: Vec3 = [0, 0, -1];

describe('posture correctives', () => {
	test('upright weighs nothing', () => {
		const weights = gravityWeights(gravityInBone(turnAboutX(0), DOWN));
		for (const w of weights) expect(Math.abs(w)).toBeLessThan(1e-9);
	});

	test('leaning forward weighs the forward shape', () => {
		// her up turned towards her front (-y): gravity points out of her chest
		const [forward, back, left, right, lift] = gravityWeights(
			gravityInBone(turnAboutX(Math.PI / 2), DOWN)
		);
		expect(forward).toBeCloseTo(1);
		expect(back).toBeCloseTo(0);
		expect(left + right).toBeCloseTo(0);
		expect(lift).toBeCloseTo(1);
	});

	test('lying on her back weighs the back shape, upside down lifts twice', () => {
		const back = gravityWeights(gravityInBone(turnAboutX(-Math.PI / 2), DOWN));
		expect(back[1]).toBeCloseTo(1);
		expect(back[0]).toBeCloseTo(0);
		const inverted = gravityWeights(gravityInBone(turnAboutX(Math.PI), DOWN));
		expect(inverted[4]).toBeCloseTo(2);
	});

	test('her left side down weighs the left shape', () => {
		const weights = gravityWeights([1, 0, 0]);
		expect(weights[2]).toBeCloseTo(1);
		expect(weights[3]).toBeCloseTo(0);
	});
});
