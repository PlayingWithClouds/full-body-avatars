/**
 * Reading `.avatar` files, as written by `avatar_splats.export.avatar_file`.
 *
 * Layout, little-endian: the magic `AVATAR01`, a uint32 header length, a UTF-8 JSON header
 * padded to 16 bytes, then C-ordered arrays at the byte offsets the header gives.
 */

export type ArrayData =
	Float32Array | Uint32Array | Int32Array | Uint16Array | Uint8Array | Int8Array;

export interface AvatarArray<T extends ArrayData = ArrayData> {
	data: T;
	shape: number[];
}

export interface AvatarMetadata {
	format_version: number;
	units: string;
	up_axis: string;
	forward_axis: string;
	bones: string[];
	expressions: string[];
	head_start: number;
	sh_degree: number;
	quaternion_order: string;
	textures: string[];
	occlusion_margin: number;
}

export interface AvatarFile {
	metadata: AvatarMetadata;
	arrays: Record<string, AvatarArray>;
}

const MAGIC = 'AVATAR01';
const SUPPORTED_VERSION = 2;

const DTYPES: Record<
	string,
	new (buffer: ArrayBuffer, offset: number, length: number) => ArrayData
> = {
	'<f4': Float32Array,
	'<u4': Uint32Array,
	'<i4': Int32Array,
	'<u2': Uint16Array,
	'|u1': Uint8Array,
	'|i1': Int8Array
};

export function readAvatar(buffer: ArrayBuffer): AvatarFile {
	const bytes = new Uint8Array(buffer);
	const magic = new TextDecoder().decode(bytes.subarray(0, MAGIC.length));
	if (magic !== MAGIC) throw new Error('not an avatar file');
	const length = new DataView(buffer).getUint32(MAGIC.length, true);
	const start = MAGIC.length + 4;
	const header = JSON.parse(new TextDecoder().decode(bytes.subarray(start, start + length)));
	const { arrays: layout, ...metadata } = header;
	if (metadata.format_version !== SUPPORTED_VERSION) {
		throw new Error(
			`avatar format ${metadata.format_version}, this viewer reads ${SUPPORTED_VERSION}`
		);
	}

	const arrays: Record<string, AvatarArray> = {};
	for (const [name, entry] of Object.entries(layout) as [string, ArrayEntry][]) {
		const type = DTYPES[entry.dtype];
		if (!type) throw new Error(`${name}: unsupported dtype ${entry.dtype}`);
		const count = entry.shape.reduce((a, b) => a * b, 1);
		// Offsets are 16-byte aligned, so typed arrays can view the buffer directly.
		arrays[name] = { data: new type(buffer, entry.offset, count), shape: entry.shape };
	}
	return { metadata, arrays };
}

interface ArrayEntry {
	dtype: string;
	shape: number[];
	offset: number;
}

export function array<T extends ArrayData>(file: AvatarFile, name: string): AvatarArray<T> {
	const found = file.arrays[name];
	if (!found) throw new Error(`avatar file has no ${name}`);
	return found as AvatarArray<T>;
}
