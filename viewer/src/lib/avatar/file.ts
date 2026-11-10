/**
 * Reading `.avatar` files, as written by `avatar_splats.export.avatar_file`.
 *
 * Layout, little-endian: an 8-byte magic (`AVATAR01`), a uint32 header length, a UTF-8 JSON
 * header padded to 16 bytes, then C-ordered arrays at the byte offsets the header gives.
 * `.motion` files share the layout with their own magic (`readContainer`).
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
	/** Since format 3: the parts' names, indexed by `face_parts`. */
	parts?: string[];
	/** Since format 3: what the eye material draws the eyeballs from. */
	eyes?: EyeMetadata;
	/** Since format 3: the inside of the mouth's regions (`mouth_regions`) and stock colours. */
	mouth?: MouthMetadata;
	/** Since format 3: the lashes and brows, strands bound to the face (`render/face_hair.ts`). */
	face_hair?: {
		points_per_strand: number;
		/** Indexed by `hair_groups`; colours RGB in [0, 1]. */
		groups: { name: string; colour: [number, number, number] }[];
	};
	/** Since format 4: the light the videos were lit by; the skin's textures then hold albedo. */
	lighting?: LightingMetadata;
	/** Since format 4: expression-dependent detail on the head's albedo (`wrinkle_textures`). */
	wrinkles?: WrinkleMetadata;
	/** Since format 5: shapes of the body's vertices (`body_shape_basis`), weighted by a pose. */
	body_shapes?: BodyShapeMetadata;
}

export interface BodyShapeMetadata {
	/** `(S)` names of `body_shape_basis`' shapes. */
	names: string[];
	/**
	 * What weights them: `gravity`, the breasts' posture correctives. Gravity (`down`, in the
	 * rest pose's axes) in the axes of `bone` (turned back by its skinning rotation) gives the
	 * weights (`gravityWeights` in `rig.ts`).
	 */
	driver: { kind: 'gravity'; bone: string; down: [number, number, number] };
}

export interface LightingMetadata {
	/** `(9, 3)` RGB coefficients of real spherical harmonics up to degree 2 (`render/skin.ts`). */
	sh: [number, number, number][];
}

export interface WrinkleMetadata {
	/** `(K)` the expression each wrinkle map belongs to, each one of `expressions`. */
	expressions: string[];
	/** The albedo change a map's byte of 255 makes at weight 1 (128 makes none, 1 the opposite). */
	scale: number;
	/** `[u0, v0, u1, v1]` the part of the head texture the maps cover. */
	region: [number, number, number, number];
	/** The maps' side, in texels. */
	size: number;
}

export interface MouthMetadata {
	/** Indexed by `mouth_regions`; the first, 0, is outside the mouth. */
	regions: string[];
	/** RGB in [0, 1] by region name: teeth, gums, tongue, socket, inner_lips. */
	colours: Record<string, [number, number, number]>;
}

export interface EyeMetadata {
	/** RGB in [0, 1], as the video shows them. */
	iris_colour: [number, number, number];
	sclera_colour: [number, number, number];
	/** How far the iris reaches from the eyeball's front, in radians seen from its centre. */
	iris_angle: number;
	eyeballs: Eyeball[];
}

export interface Eyeball {
	side: 'left' | 'right';
	/** In the rest pose, metres. */
	centre: [number, number, number];
	radius: number;
	/** The vertex at its front and the one at its top, which show how it is turned. */
	apex: number;
	top: number;
}

export interface AvatarFile {
	metadata: AvatarMetadata;
	arrays: Record<string, AvatarArray>;
}

const MAGIC = 'AVATAR01';
/**
 * Format 3 adds the parts and the eyes to format 2; without them the eyes show their texture.
 * Format 4 adds the learned light and wrinkle maps, and may have no Gaussians.
 * Format 5 adds body shapes: the breasts' posture correctives.
 */
const SUPPORTED_VERSIONS = [2, 3, 4, 5];

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
	const { metadata, arrays } = readContainer(buffer, MAGIC, 'avatar');
	if (!SUPPORTED_VERSIONS.includes(metadata.format_version as number)) {
		throw new Error(
			`avatar format ${metadata.format_version}, this viewer reads ${SUPPORTED_VERSIONS.join(', ')}`
		);
	}
	return { metadata: metadata as unknown as AvatarMetadata, arrays };
}

/** The metadata and arrays of a file in the avatar file's layout with the given magic. */
export function readContainer(
	buffer: ArrayBuffer,
	magic: string,
	kind: string
): { metadata: Record<string, unknown>; arrays: Record<string, AvatarArray> } {
	const bytes = new Uint8Array(buffer);
	if (new TextDecoder().decode(bytes.subarray(0, magic.length)) !== magic) {
		throw new Error(`not ${/^[aeiou]/.test(kind) ? 'an' : 'a'} ${kind} file`);
	}
	const length = new DataView(buffer).getUint32(magic.length, true);
	const start = magic.length + 4;
	const header = JSON.parse(new TextDecoder().decode(bytes.subarray(start, start + length)));
	const { arrays: layout, ...metadata } = header;

	const arrays: Record<string, AvatarArray> = {};
	for (const [name, entry] of Object.entries(layout) as [string, ArrayEntry][]) {
		const type = DTYPES[entry.dtype];
		if (!type) throw new Error(`${name}: unsupported dtype ${entry.dtype}`);
		const count = entry.shape.reduce((a, b) => a * b, 1);
		// Offsets are 16-byte aligned, so typed arrays can view the buffer directly. An empty
		// array's offset may lie past the end of the file.
		const data = count ? new type(buffer, entry.offset, count) : new type(new ArrayBuffer(0), 0, 0);
		arrays[name] = { data, shape: entry.shape };
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
