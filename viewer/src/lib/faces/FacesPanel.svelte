<!--
	The faces of a video (`task faces`), one head per frame in a grid beside the avatar: each head
	wears its frame's expression, gaze and head rotation. Open a `.faces.json` file, drop one on
	the viewer, or name one in `?faces=`.
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import * as THREE from 'three';
	import { FaceGridLayer } from '#lib/faces/layer.ts';
	import type { AvatarViewer } from '#lib/render/viewer.ts';

	let { viewer }: { viewer: AvatarViewer | undefined } = $props();

	let document: unknown = $state.raw(null);
	/** The video's crops, when the faces came from a URL with an atlas next to it. */
	let atlas: THREE.Texture | null = $state.raw(null);
	let name = $state('');
	let frames = $state(0);
	let rotate = $state(true);
	let labels = $state(true);
	let video = $state(true);
	let error = $state('');
	let layer: FaceGridLayer | undefined = $state.raw();

	/** Show the faces in a `.faces.json` file, a URL or a dropped file. */
	export async function loadFaces(source: Blob | string, title: string): Promise<void> {
		error = '';
		try {
			const text =
				typeof source === 'string'
					? await fetch(source).then((response) => {
							if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
							return response.text();
						})
					: await source.text();
			const parsed = JSON.parse(text);
			atlas = null;
			if (typeof source === 'string' && parsed.crops?.file) {
				const url = new URL(parsed.crops.file, new URL(source, location.href)).href;
				atlas = await new THREE.TextureLoader().loadAsync(url);
				atlas.colorSpace = THREE.NoColorSpace;
			}
			document = parsed;
			name = title.replace(/\.faces\.json$/, '');
		} catch (cause) {
			error = `Could not load ${title}: ${(cause as Error).message}`;
		}
	}

	// The grid lives with the viewer: a new avatar file brings a new viewer, so it is built again.
	$effect(() => {
		const current = viewer;
		const data = document;
		const image = atlas;
		if (!current || !data) return;
		let created: FaceGridLayer;
		try {
			created = new FaceGridLayer(
				current.file,
				data,
				image,
				untrack(() => ({ rotate, labels, video }))
			);
		} catch (cause) {
			error = `Could not show the faces: ${(cause as Error).message}`;
			return;
		}
		current.addLayer(created);
		current.frameAvatar();
		layer = created;
		frames = created.faces.frames.length;
		return () => {
			current.removeLayer(created);
			layer = undefined;
		};
	});
	$effect(() => {
		layer?.setOptions({ rotate, labels, video });
	});

	function openFile(event: Event) {
		const file = (event.currentTarget as HTMLInputElement).files?.[0];
		if (file) loadFaces(file, file.name);
	}
</script>

<section class="space-y-3">
	<div class="flex items-baseline justify-between">
		<h2 class="text-xs font-semibold tracking-wider text-zinc-400 uppercase">Faces</h2>
		<button
			class="text-xs text-sky-400 hover:text-sky-300 disabled:opacity-40"
			disabled={!document}
			onclick={() => {
				document = null;
				atlas = null;
				name = '';
			}}
		>
			Remove
		</button>
	</div>
	<label
		class="flex cursor-pointer items-center justify-center rounded-md border border-zinc-700 px-3 py-1.5 text-sm hover:bg-zinc-800"
	>
		Open .faces.json
		<input type="file" accept=".json" class="sr-only" onchange={openFile} />
	</label>
	{#if name && layer}
		<p class="truncate text-xs text-zinc-400" title={name}>{name} · {frames} frames</p>
		<div class="grid grid-cols-2 gap-2">
			<label class="flex items-center gap-2 text-sm" title="Turn each head as the video's did">
				<input type="checkbox" bind:checked={rotate} class="accent-sky-500" /> Head rotation
			</label>
			<label
				class="flex items-center gap-2 text-sm"
				title="The frame's index in the video; red where no face was found"
			>
				<input type="checkbox" bind:checked={labels} class="accent-sky-500" /> Labels
			</label>
			<label
				class="flex items-center gap-2 text-sm"
				title="The video's frame beside each head, when the faces came with crops"
			>
				<input type="checkbox" bind:checked={video} class="accent-sky-500" /> Video
			</label>
		</div>
	{/if}
	{#if error}
		<p class="text-xs text-red-400">{error}</p>
	{/if}
</section>
