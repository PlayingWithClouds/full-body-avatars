/// <reference types="bun" />
import { describe, expect, test } from 'bun:test';
import { gridShape, parseFaces, placeHead, tileUvs, turn } from '#lib/faces/faces.ts';

const frame = {
	index: 4,
	found: true,
	head: [0, 0, 0.1],
	eyes: [
		[0, 0, 0],
		[0, 0, 0]
	]
};

describe('parseFaces', () => {
	test('puts the named weights in the avatar order', () => {
		const faces = parseFaces(
			{
				format_version: 1,
				video: 'v',
				fps: 16,
				frames: [{ ...frame, expressions: { mouthSmile_L: 0.5, jawOpen: 0.25 } }]
			},
			['jawOpen', 'eyeBlink_L', 'mouthSmile_L']
		);
		expect(Array.from(faces.frames[0].expressions)).toEqual([0.25, 0, 0.5]);
		expect(faces.frames[0].index).toBe(4);
	});

	test('rejects what is not for this avatar', () => {
		expect(() => parseFaces({ frames: [] }, [])).toThrow('not a faces file');
		expect(() =>
			parseFaces({ format_version: 1, frames: [{ ...frame, expressions: { tail: 1 } }] }, [
				'jawOpen'
			])
		).toThrow('the avatar has no expression tail');
	});
});

test('gridShape is wider than tall and holds every cell', () => {
	const { columns, rows } = gridShape(81);
	expect(columns).toBe(12);
	expect(columns * rows).toBeGreaterThanOrEqual(81);
	expect(gridShape(1)).toEqual({ columns: 2, rows: 1 });
});

describe('placeHead', () => {
	// Two head vertices and one that is not (the third).
	const posed = Float32Array.of(1, 0, 0, 0, 1, 0, 5, 5, 5);
	const base = {
		vertices: Uint32Array.of(0, 1),
		pivot: [0, 0, 0] as [number, number, number],
		centre: [10, 0, 0] as [number, number, number],
		eyes: []
	};

	test('moves the pivot to the cell and collapses the rest there', () => {
		const out = new Float32Array(9);
		placeHead(posed, out, { ...base, rotation: turn([0, 0, 0]) });
		expect(Array.from(out)).toEqual([11, 0, 0, 10, 1, 0, 10, 0, 0]);
	});

	test('turns the head about its pivot', () => {
		const out = new Float32Array(9);
		placeHead(posed, out, { ...base, rotation: turn([0, 0, Math.PI / 2]) });
		expect(out[0]).toBeCloseTo(10);
		expect(out[1]).toBeCloseTo(1);
		expect(out[3]).toBeCloseTo(9);
		expect(out[4]).toBeCloseTo(0);
	});

	test('turns an eyeball about its own centre before the head', () => {
		const out = new Float32Array(9);
		const eye = { vertices: Uint32Array.of(0), centre: [0, 0, 0] as [number, number, number] };
		placeHead(posed, out, {
			...base,
			rotation: turn([0, 0, 0]),
			eyes: [{ ...eye, rotation: turn([0, 0, Math.PI / 2]) }]
		});
		expect(out[0]).toBeCloseTo(10);
		expect(out[1]).toBeCloseTo(1);
	});

	test('turn without rotation is the identity', () => {
		expect(Array.from(turn([1, 2, 3], false))).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
	});
});

test('tileUvs cut the atlas row by row from the top', () => {
	const crops = { file: 'a.jpg', columns: 3, rows: 2 };
	expect(tileUvs(0, crops)).toEqual([0, 0.5, 1 / 3, 1]);
	expect(tileUvs(4, crops)).toEqual([1 / 3, 0, 2 / 3, 0.5]);
});
