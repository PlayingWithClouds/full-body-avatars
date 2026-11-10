import { describe, expect, test } from 'bun:test';
import { cornerNormals, irradiance, shBasis, wrinkleExpressions } from '#lib/render/skin.ts';

describe('spherical harmonics', () => {
	test('basis at the zenith keeps only the zonal terms', () => {
		const basis = shBasis(0, 0, 1);
		const expected = [0.282095, 0, 0.488603, 0, 0, 0, 2 * 0.315392, 0, 0];
		basis.forEach((value, k) => expect(value).toBeCloseTo(expected[k], 6));
	});

	test('basis follows the file axes', () => {
		const [, y, z, x, xy, , , , xxyy] = shBasis(1, 0, 0);
		expect([y, z, x, xy]).toEqual([0, 0, 0.488603, 0]);
		expect(xxyy).toBeCloseTo(0.546274, 6);
		expect(shBasis(0, 1, 0)[1]).toBeCloseTo(0.488603, 6);
	});

	test('irradiance is the weighted sum per channel, never below zero', () => {
		const sh = Array.from({ length: 9 }, () => [0, 0, 0]);
		sh[0] = [1 / 0.282095, 2 / 0.282095, 0];
		// Light from above (+z) in the red channel only.
		sh[2] = [1 / 0.488603, 0, 0];
		const up = irradiance(sh, [0, 0, 1]);
		expect(up[0]).toBeCloseTo(2, 5);
		expect(up[1]).toBeCloseTo(2, 5);
		expect(up[2]).toBe(0);
		const down = irradiance(sh, [0, 0, -1]);
		expect(down[0]).toBeCloseTo(0, 5);
		sh[2] = [-3 / 0.488603, 0, 0];
		expect(irradiance(sh, [0, 0, 1])[0]).toBe(0);
	});
});

describe('cornerNormals', () => {
	test('smooths each vertex over its triangles, corner by corner', () => {
		// Two right triangles of equal area folded along the x axis: one facing -y, one +z.
		const posed = Float32Array.from([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0]);
		const corners = Uint32Array.from([0, 1, 2, 0, 1, 3]);
		const sums = new Float32Array(posed.length);
		const out = new Float32Array(3 * corners.length);
		cornerNormals(posed, corners, sums, out);

		const at = (i: number) => Array.from(out.subarray(3 * i, 3 * i + 3));
		const fold = [0, -Math.SQRT1_2, Math.SQRT1_2];
		for (const i of [0, 1, 3, 4])
			at(i).forEach((value, a) => expect(value).toBeCloseTo(fold[a], 6));
		at(2).forEach((value, a) => expect(value).toBeCloseTo([0, -1, 0][a], 6));
		at(5).forEach((value, a) => expect(value).toBeCloseTo([0, 0, 1][a], 6));
	});

	test('weights triangles by area', () => {
		// The shared vertex 0: a large triangle facing +z, a small one facing -y.
		const posed = Float32Array.from([0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 1, 0, 0]);
		const corners = Uint32Array.from([0, 1, 2, 0, 4, 3]);
		const out = new Float32Array(3 * corners.length);
		cornerNormals(posed, corners, new Float32Array(posed.length), out);
		const length = Math.hypot(1, 4);
		expect(out[1]).toBeCloseTo(-1 / length, 6);
		expect(out[2]).toBeCloseTo(4 / length, 6);
	});
});

describe('wrinkleExpressions', () => {
	test('finds each map expression by name', () => {
		const names = ['eyeBlinkLeft', 'browInnerUp', 'jawOpen'];
		expect(Array.from(wrinkleExpressions(['jawOpen', 'browInnerUp'], names))).toEqual([2, 1]);
		expect(() => wrinkleExpressions(['smile'], names)).toThrow('smile');
	});
});
