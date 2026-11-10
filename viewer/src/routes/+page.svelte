<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import AnimationPanel from '#lib/animation/AnimationPanel.svelte';
	import { readAvatar } from '#lib/avatar/file.ts';
	import FacesPanel from '#lib/faces/FacesPanel.svelte';
	import HairPanel from '#lib/hair/HairPanel.svelte';
	import { JOINTS, PRESETS, rotations } from '#lib/avatar/poses.ts';
	import { AvatarViewer, type Stats } from '#lib/render/viewer.ts';

	const DEFAULT_SOURCE = '/performer.avatar';
	const BACKGROUNDS = ['#ffffff', '#9ca3af', '#18181b', '#000000'];

	let canvas: HTMLCanvasElement;
	let animation: AnimationPanel | undefined = $state();
	let faces: FacesPanel | undefined = $state();
	let viewer: AvatarViewer | undefined = $state();
	let status = $state('Drop an .avatar file here, or open one.');
	let loading = $state(false);
	let dragging = $state(false);
	let fileName = $state('');

	let showMesh = $state(true);
	let showGaussians = $state(true);
	let viewDependent = $state(true);
	let antialias = $state(false);
	let hasLighting = $state(false);
	let lit = $state(true);
	let background = $state(BACKGROUNDS[0]);
	let joints: Record<string, number> = $state({});
	let expressionNames: string[] = $state([]);
	let expressions: number[] = $state([]);
	let expressionFilter = $state('');
	let stats: Stats | undefined = $state();

	const shownExpressions = $derived(
		expressionNames
			.map((name, index) => ({ name, index }))
			.filter(({ name }) => name.toLowerCase().includes(expressionFilter.toLowerCase()))
	);

	async function load(source: Blob | string, name: string) {
		loading = true;
		status = `Loading ${name}…`;
		try {
			const buffer =
				typeof source === 'string'
					? await fetch(source).then((response) => {
							if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
							return response.arrayBuffer();
						})
					: await source.arrayBuffer();
			const file = readAvatar(buffer);
			viewer?.dispose();
			viewer = new AvatarViewer(canvas, file);
			fileName = name;
			hasLighting = file.metadata.lighting !== undefined;
			expressionNames = file.metadata.expressions;
			expressions = expressionNames.map(() => 0);
			joints = {};
			status = '';
		} catch (error) {
			status = `Could not load ${name}: ${(error as Error).message}`;
		} finally {
			loading = false;
		}
	}

	function openFile(event: Event) {
		const file = (event.currentTarget as HTMLInputElement).files?.[0];
		if (file) load(file, file.name);
	}

	function drop(event: DragEvent) {
		event.preventDefault();
		dragging = false;
		for (const file of event.dataTransfer?.files ?? []) {
			if (file.name.endsWith('.faces.json')) faces?.loadFaces(file, file.name);
			else if (file.name.endsWith('.motion')) animation?.loadMotion(file, file.name);
			else if (file.name.endsWith('.json')) animation?.loadSequence(file, file.name);
			else load(file, file.name);
		}
	}

	/** Back to posing the avatar with the sliders. */
	function poseByHand() {
		viewer?.pose(rotations(joints), Float32Array.from(expressions));
	}

	function applyPreset(name: string) {
		joints = { ...PRESETS[name] };
	}

	$effect(() => {
		viewer?.pose(rotations(joints), Float32Array.from(expressions));
	});
	$effect(() => {
		viewer?.show({ mesh: showMesh, gaussians: showGaussians });
	});
	$effect(() => {
		viewer?.setBackground(background);
	});
	$effect(() => {
		viewer?.setSplatOptions({ viewDependent, antialias });
	});
	$effect(() => {
		viewer?.setLit(lit);
	});

	let statsTimer: ReturnType<typeof setInterval>;
	onMount(() => {
		const params = new URLSearchParams(location.search);
		showMesh = params.get('mesh') !== '0';
		showGaussians = params.get('gaussians') !== '0';
		background = params.get('background') ?? background;
		viewDependent = params.get('view-colour') !== '0';
		antialias = params.get('antialias') === '1';
		const source = params.get('src') ?? DEFAULT_SOURCE;
		fetch(source, { method: 'HEAD' })
			.then((response) => {
				if (!response.ok) return;
				// Motion clips named in `?motion=a.motion,b.motion` play once the avatar is in, and a
				// sequence named in `?sequence=s.json` runs after them.
				return load(source, source.split('/').pop() ?? source).then(async () => {
					for (const motion of params.get('motion')?.split(',') ?? []) {
						await animation?.loadMotion(motion, motion.split('/').pop() ?? motion);
					}
					const sequence = params.get('sequence');
					if (sequence) await animation?.loadSequence(sequence, sequence.split('/').pop()!);
					// The faces of a video, `?faces=/faces/head_expressions.faces.json`, in a grid.
					const grid = params.get('faces');
					if (grid) await faces?.loadFaces(grid, grid.split('/').pop() ?? grid);
				});
			})
			.catch(() => {});
		statsTimer = setInterval(() => (stats = viewer?.stats()), 500);
	});
	onDestroy(() => {
		clearInterval(statsTimer);
		viewer?.dispose();
	});
</script>

<svelte:head>
	<title>{fileName ? `${fileName} · ` : ''}Avatar viewer</title>
</svelte:head>

<svelte:window
	ondragover={(event) => {
		event.preventDefault();
		dragging = true;
	}}
	ondragleave={() => (dragging = false)}
	ondrop={drop}
/>

<div class="flex h-dvh w-full overflow-hidden bg-zinc-950 text-zinc-200">
	<main class="relative min-w-0 flex-1">
		<canvas bind:this={canvas} class="block h-full w-full touch-none"></canvas>

		{#if status || dragging}
			<div
				class={[
					'pointer-events-none absolute inset-0 grid place-items-center p-8 text-center text-sm',
					dragging ? 'bg-sky-500/20 ring-4 ring-sky-400 ring-inset' : 'bg-zinc-950/80'
				]}
			>
				<p class={['max-w-sm', loading && 'animate-pulse']}>
					{dragging ? 'Drop to open' : status}
				</p>
			</div>
		{/if}

		{#if stats}
			<div
				class="pointer-events-none absolute bottom-3 left-3 rounded-md bg-black/50 px-2 py-1 font-mono text-xs text-zinc-300"
			>
				{stats.gaussians.toLocaleString()} Gaussians · {stats.triangles.toLocaleString()} triangles
			</div>
		{/if}
	</main>

	<aside class="flex w-80 shrink-0 flex-col border-l border-zinc-800 bg-zinc-900">
		<header class="border-b border-zinc-800 p-4">
			<h1 class="text-sm font-semibold tracking-wide text-zinc-100">Avatar viewer</h1>
			<p class="mt-0.5 truncate text-xs text-zinc-400">{fileName || 'No avatar loaded'}</p>
			<label
				class="mt-3 flex cursor-pointer items-center justify-center rounded-md bg-sky-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-sky-500"
			>
				Open .avatar file
				<input type="file" accept=".avatar" class="sr-only" onchange={openFile} />
			</label>
		</header>

		<div class="flex-1 space-y-6 overflow-y-auto p-4">
			<section class="space-y-3">
				<h2 class="text-xs font-semibold tracking-wider text-zinc-400 uppercase">View</h2>
				<div class="grid grid-cols-2 gap-2">
					<label class="flex items-center gap-2 text-sm">
						<input type="checkbox" bind:checked={showMesh} class="accent-sky-500" /> Mesh
					</label>
					<label class="flex items-center gap-2 text-sm">
						<input type="checkbox" bind:checked={showGaussians} class="accent-sky-500" /> Gaussians
					</label>
					<label
						class="flex items-center gap-2 text-sm"
						title="Colours change with the viewing direction"
					>
						<input type="checkbox" bind:checked={viewDependent} class="accent-sky-500" /> View colour
					</label>
					<label
						class="flex items-center gap-2 text-sm"
						title="Gaussians smaller than a pixel fade instead of twinkling"
					>
						<input type="checkbox" bind:checked={antialias} class="accent-sky-500" /> Antialias
					</label>
					{#if hasLighting}
						<label
							class="flex items-center gap-2 text-sm"
							title="Light the skin's albedo with the light learned from the videos, or show it unlit"
						>
							<input type="checkbox" bind:checked={lit} class="accent-sky-500" /> Lighting
						</label>
					{/if}
				</div>
				<div class="flex items-center gap-2">
					<span class="text-sm text-zinc-400">Background</span>
					{#each BACKGROUNDS as colour (colour)}
						<button
							aria-label={`Background ${colour}`}
							class={[
								'size-6 rounded-full border border-zinc-600',
								background === colour && 'ring-2 ring-sky-400 ring-offset-2 ring-offset-zinc-900'
							]}
							style:background-color={colour}
							onclick={() => (background = colour)}
						></button>
					{/each}
				</div>
				<button
					class="w-full rounded-md border border-zinc-700 px-3 py-1.5 text-sm hover:bg-zinc-800 disabled:opacity-40"
					disabled={!viewer}
					onclick={() => viewer?.frameAvatar()}
				>
					Reset camera
				</button>
			</section>

			<AnimationPanel bind:this={animation} {viewer} onstop={poseByHand} />

			<FacesPanel bind:this={faces} {viewer} />

			<HairPanel {viewer} />

			<section class="space-y-3">
				<div class="flex items-baseline justify-between">
					<h2 class="text-xs font-semibold tracking-wider text-zinc-400 uppercase">Pose</h2>
				</div>
				<div class="flex flex-wrap gap-1.5">
					{#each Object.keys(PRESETS) as preset (preset)}
						<button
							class="rounded-full border border-zinc-700 px-2.5 py-0.5 text-xs hover:bg-zinc-800"
							onclick={() => applyPreset(preset)}
						>
							{preset}
						</button>
					{/each}
				</div>
				{#each JOINTS as joint (joint.label)}
					<label class="block">
						<span class="flex justify-between text-xs text-zinc-400">
							{joint.label}
							<span class="font-mono tabular-nums">{joints[joint.label] ?? 0}°</span>
						</span>
						<input
							type="range"
							min={joint.min}
							max={joint.max}
							value={joints[joint.label] ?? 0}
							oninput={(event) =>
								(joints = { ...joints, [joint.label]: Number(event.currentTarget.value) })}
							class="w-full accent-sky-500"
						/>
					</label>
				{/each}
			</section>

			<section class="space-y-3">
				<div class="flex items-baseline justify-between">
					<h2 class="text-xs font-semibold tracking-wider text-zinc-400 uppercase">Expressions</h2>
					<button
						class="text-xs text-sky-400 hover:text-sky-300"
						onclick={() => (expressions = expressions.map(() => 0))}
					>
						Reset
					</button>
				</div>
				<input
					type="search"
					placeholder="Filter, e.g. jaw, smile, blink"
					bind:value={expressionFilter}
					class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm placeholder:text-zinc-500 focus:border-sky-500 focus:outline-none"
				/>
				{#each shownExpressions as { name, index } (name)}
					<label class="block">
						<span class="flex justify-between text-xs text-zinc-400">
							{name}
							<span class="font-mono tabular-nums">{expressions[index].toFixed(2)}</span>
						</span>
						<input
							type="range"
							min="0"
							max="1"
							step="0.01"
							bind:value={expressions[index]}
							class="w-full accent-sky-500"
						/>
					</label>
				{/each}
			</section>
		</div>
	</aside>
</div>
