/// <reference types="bun" />
import { describe, expect, test } from 'bun:test';
import { envelope, expressionIndices, parseSequence, schedule } from '#lib/animation/sequence.ts';

describe('parseSequence', () => {
	test('reads steps and cues with their defaults', () => {
		const sequence = parseSequence({
			name: 'demo',
			steps: [{ clip: 'walk', to: 2 }, { pose: 'Wave', hold: 1.5 }, { wait: 1 }],
			cues: [
				{ at: 0, for: 2, expression: { mouthSmile: 0.7 } },
				{ at: 1, for: 1, character: 1, look: [0, -2, 1.6] },
				{ at: 0, camera: 'video' }
			]
		});
		expect(sequence.loop).toBe(false);
		expect(sequence.steps[0]).toEqual({
			kind: 'clip', clip: 'walk', from: 0, to: 2, speed: 1, repeat: 1, fade: null
		}); // prettier-ignore
		expect(sequence.steps[1]).toEqual({ kind: 'pose', pose: 'Wave', hold: 1.5, fade: null });
		expect(sequence.cues[0]).toMatchObject({ kind: 'expression', character: 'all', fade: 0.3 });
		expect(sequence.cues[1]).toMatchObject({ kind: 'look', character: 1, target: [0, -2, 1.6] });
	});

	test('says where a sequence is wrong', () => {
		expect(() => parseSequence({ steps: [] })).toThrow('steps: a sequence needs at least one step');
		expect(() => parseSequence({ steps: [{ clip: 'a', to: 1, from: 2 }] })).toThrow(
			'steps[0].to: must be after from (2)'
		);
		expect(() => parseSequence({ steps: [{ clip: 'a', spead: 2 }] })).toThrow(
			'steps[0]: unknown key "spead"'
		);
		expect(() =>
			parseSequence({ steps: [{ wait: 1 }], cues: [{ at: 0, for: 1, look: 'moon' }] })
		).toThrow('cues[0].look: "camera", "partner" or a point [x, y, z]');
	});
});

test('schedule lays steps end to end', () => {
	const { steps } = parseSequence({
		steps: [{ clip: 'a', from: 1, speed: 2, repeat: 3 }, { pose: 'Wave', hold: 2 }, { wait: 0.5 }]
	});
	const plan = schedule(steps, () => 5);
	// (5 - 1) seconds at twice the speed, three times: 6 seconds.
	expect(plan.map(({ start, end }) => [start, end])).toEqual([
		[0, 6],
		[6, 8],
		[8, 8.5]
	]);
});

test('envelope rises, holds and falls', () => {
	expect(envelope(-0.1, 2, 0.5)).toBe(0);
	expect(envelope(0.25, 2, 0.5)).toBeCloseTo(0.5);
	expect(envelope(1, 2, 0.5)).toBe(1);
	expect(envelope(1.75, 2, 0.5)).toBeCloseTo(0.5);
	expect(envelope(2.1, 2, 0.5)).toBe(0);
});

test('expression names follow ARKit and the avatar', () => {
	const names = ['jawOpen', 'mouthSmile_L', 'mouthSmile_R', 'jawLeft'];
	expect(expressionIndices('jawOpen', names)).toEqual([0]);
	expect(expressionIndices('mouthSmileLeft', names)).toEqual([1]);
	expect(expressionIndices('mouthSmile', names)).toEqual([1, 2]);
	expect(expressionIndices('jawLeft', names)).toEqual([3]);
	expect(expressionIndices('tongueOut', names)).toEqual([]);
});
