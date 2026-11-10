<!--
	Motion clips: open `.motion` files, play them on copies of the avatar (one per person in the
	clip), and switch between them with blended transitions; or run a scripted sequence of them.
-->
<script lang="ts">
	import { onDestroy, untrack } from 'svelte';
	import { readMotion, type MotionClip } from '#lib/animation/clip.ts';
	import { Player, type EndMode } from '#lib/animation/player.ts';
	import SequencePanel from '#lib/animation/SequencePanel.svelte';
	import type { AvatarViewer } from '#lib/render/viewer.ts';

	interface Props {
		viewer: AvatarViewer | undefined;
		/** Called when playback stops and the avatar is back to being posed by hand. */
		onstop?: () => void;
	}

	let { viewer, onstop }: Props = $props();

	const SPEEDS = [0.25, 0.5, 1, 1.5, 2];
	const END_MODES: { value: EndMode; label: string }[] = [
		{ value: 'loop', label: 'Loop' },
		{ value: 'once', label: 'Play once' },
		{ value: 'next', label: 'Next clip' }
	];

	let player: Player | undefined = $state();
	let clips: { name: string; people: number; seconds: number }[] = $state([]);
	let current = $state(-1);
	/** Whether anything plays or holds: a clip from the list, or a sequence's pose. */
	let active = $state(false);
	let playing = $state(false);
	let sequences: SequencePanel | undefined = $state();
	let time = $state(0);
	let length = $state(0);
	let speed = $state(1);
	let mode: EndMode = $state('loop');
	let fade = $state(0.4);
	let align = $state(true);
	let error = $state('');
	let frame = 0;

	// A new avatar means a new viewer: start over with its own player.
	$effect(() => {
		const next = viewer ? new Player(viewer) : undefined;
		untrack(() => player?.stop());
		player = next;
		clips = [];
		current = -1;
	});

	$effect(() => {
		if (!player) return;
		player.speed = speed;
		player.mode = mode;
		player.fadeSeconds = fade;
		player.align = align;
	});

	/** Keep the transport in step with the player. */
	function tick() {
		frame = requestAnimationFrame(tick);
		if (!player) return;
		time = player.time;
		length = player.duration;
		playing = player.playing;
		current = player.current;
		active = player.clip !== null;
	}
	frame = requestAnimationFrame(tick);
	onDestroy(() => {
		cancelAnimationFrame(frame);
		player?.stop();
	});

	/**
	 * Add a `.motion` file (a URL or a dropped file) to the clips, and play it unless `play` is
	 * false; returns the clip, or null when it could not be read.
	 */
	export async function loadMotion(
		source: Blob | string,
		name: string,
		play = true
	): Promise<MotionClip | null> {
		if (!player) return null;
		error = '';
		try {
			const buffer =
				typeof source === 'string'
					? await fetch(source).then((response) => {
							if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
							return response.arrayBuffer();
						})
					: await source.arrayBuffer();
			const clip = readMotion(buffer, name.replace(/\.motion$/, ''));
			const index = player.add(clip);
			clips = [
				...clips,
				{ name: clip.name, people: clip.people, seconds: (clip.frames - 1) / clip.fps }
			];
			if (play) player.play(index);
			return clip;
		} catch (cause) {
			error = `Could not load ${name}: ${(cause as Error).message}`;
			return null;
		}
	}

	/** Load a sequence (`.json`, a URL or a dropped file) and run it. */
	export function loadSequence(source: Blob | string, name: string) {
		return sequences?.loadSequence(source, name);
	}

	function openFiles(event: Event) {
		for (const file of (event.currentTarget as HTMLInputElement).files ?? []) {
			loadMotion(file, file.name);
		}
	}

	function togglePlaying() {
		if (!player?.clip) return;
		if (player.playing) player.playing = false;
		else if (player.time >= player.duration) player.play(player.clip);
		else player.playing = true;
	}

	function stop() {
		sequences?.stopSequence();
		player?.stop();
		onstop?.();
	}

	function format(seconds: number): string {
		const minutes = Math.floor(seconds / 60);
		return `${minutes}:${(seconds - 60 * minutes).toFixed(1).padStart(4, '0')}`;
	}
</script>

<section class="space-y-3">
	<div class="flex items-baseline justify-between">
		<h2 class="text-xs font-semibold tracking-wider text-zinc-400 uppercase">Animation</h2>
		{#if active}
			<button class="text-xs text-sky-400 hover:text-sky-300" onclick={stop}>Stop</button>
		{/if}
	</div>

	<label
		class={[
			'flex cursor-pointer items-center justify-center rounded-md border border-zinc-700 px-3 py-1.5 text-sm hover:bg-zinc-800',
			!viewer && 'pointer-events-none opacity-40'
		]}
	>
		Open .motion files
		<input type="file" accept=".motion" multiple class="sr-only" onchange={openFiles} />
	</label>
	{#if error}
		<p class="text-xs text-red-400">{error}</p>
	{/if}

	{#if clips.length}
		<ul class="space-y-1">
			{#each clips as clip, index (index)}
				<li>
					<button
						class={[
							'flex w-full items-center justify-between rounded-md px-2.5 py-1.5 text-left text-sm',
							index === current ? 'bg-sky-600/20 text-sky-200' : 'hover:bg-zinc-800'
						]}
						title="Play, blending in from what is showing"
						onclick={() => player?.play(index)}
					>
						<span class="truncate">{clip.name}</span>
						<span class="ml-2 shrink-0 font-mono text-xs text-zinc-400">
							{clip.people}× · {format(clip.seconds)}
						</span>
					</button>
				</li>
			{/each}
		</ul>

		<div class="flex items-center gap-2">
			<button
				class="w-16 rounded-md bg-sky-600 px-2 py-1 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-40"
				disabled={!active}
				onclick={togglePlaying}
			>
				{playing ? 'Pause' : 'Play'}
			</button>
			<input
				type="range"
				min="0"
				max={length}
				step="0.001"
				value={time}
				disabled={!active}
				oninput={(event) => {
					if (!player) return;
					player.playing = false;
					player.seek(Number(event.currentTarget.value));
				}}
				class="min-w-0 flex-1 accent-sky-500"
			/>
			<span class="w-20 text-right font-mono text-xs text-zinc-400 tabular-nums">
				{format(time)}
			</span>
		</div>

		<div class="grid grid-cols-2 gap-2 text-sm">
			<label class="flex flex-col gap-1">
				<span class="text-xs text-zinc-400">Speed</span>
				<select
					bind:value={speed}
					class="rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 focus:border-sky-500 focus:outline-none"
				>
					{#each SPEEDS as value (value)}
						<option {value}>{value}×</option>
					{/each}
				</select>
			</label>
			<label class="flex flex-col gap-1">
				<span class="text-xs text-zinc-400">At the end</span>
				<select
					bind:value={mode}
					class="rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 focus:border-sky-500 focus:outline-none"
				>
					{#each END_MODES as option (option.value)}
						<option value={option.value}>{option.label}</option>
					{/each}
				</select>
			</label>
		</div>

		<label class="block">
			<span class="flex justify-between text-xs text-zinc-400">
				Transition
				<span class="font-mono tabular-nums">{fade.toFixed(2)} s</span>
			</span>
			<input
				type="range"
				min="0"
				max="1.5"
				step="0.05"
				bind:value={fade}
				class="w-full accent-sky-500"
			/>
		</label>
		<label
			class="flex items-center gap-2 text-sm"
			title="Each clip starts where the people are and facing the way they face"
		>
			<input type="checkbox" bind:checked={align} class="accent-sky-500" /> Continue from where they are
		</label>
		<button
			class="w-full rounded-md border border-zinc-700 px-3 py-1.5 text-sm hover:bg-zinc-800 disabled:opacity-40"
			disabled={!active}
			onclick={() => player?.videoView()}
		>
			Video's view
		</button>
	{/if}

	<SequencePanel bind:this={sequences} {player} {viewer} {loadMotion} />
</section>
