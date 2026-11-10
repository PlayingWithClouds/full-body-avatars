import { describe, expect, test } from 'bun:test';
import { readContainer } from '#lib/avatar/file.ts';

interface Entry {
	dtype: string;
	shape: number[];
	offset: number;
}

/**
 * A file in the avatar layout: magic, header length, a header padded to 16 bytes, then `body`;
 * the arrays' offsets are given from the body's start, as `write_avatar` lays them out.
 */
function container(arrays: Record<string, Entry>, body: Uint8Array): ArrayBuffer {
	let start = -1;
	let json = '';
	while (start !== 12 + json.length) {
		start = 12 + json.length;
		const absolute = Object.fromEntries(
			Object.entries(arrays).map(([name, entry]) => [
				name,
				{ ...entry, offset: entry.offset + start }
			])
		);
		json = JSON.stringify({ format_version: 4, arrays: absolute });
		json += ' '.repeat(-(12 + json.length) & 15);
	}
	const bytes = new Uint8Array(start + body.length);
	bytes.set(new TextEncoder().encode('AVATAR01'));
	new DataView(bytes.buffer).setUint32(8, json.length, true);
	bytes.set(new TextEncoder().encode(json), 12);
	bytes.set(body, start);
	return bytes.buffer;
}

describe('readContainer', () => {
	test('reads empty arrays whose offset lies past the end of the file', () => {
		const buffer = container(
			{
				faces: { dtype: '<u4', shape: [1, 3], offset: 0 },
				gaussian_faces: { dtype: '<u4', shape: [0], offset: 16 },
				gaussian_sh: { dtype: '<f4', shape: [0, 16, 3], offset: 16 }
			},
			new Uint8Array(Uint32Array.from([0, 1, 2]).buffer)
		);
		const { metadata, arrays } = readContainer(buffer, 'AVATAR01', 'avatar');
		expect(metadata.format_version).toBe(4);
		expect(Array.from(arrays.faces.data)).toEqual([0, 1, 2]);
		expect(arrays.gaussian_faces.data).toBeInstanceOf(Uint32Array);
		expect(arrays.gaussian_faces.data.length).toBe(0);
		expect(arrays.gaussian_sh.shape).toEqual([0, 16, 3]);
	});
});
