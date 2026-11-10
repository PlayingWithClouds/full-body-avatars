<!--
	Scripted sequences: open a `.json` sequence or write one, run it on the player, and follow
	which step plays. The format is documented in `sequence.ts`.
-->
<script lang="ts">
	import { onDestroy } from 'svelte';
	import type { MotionClip } from '#lib/animation/clip.ts';
	import type { Player } from '#lib/animation/player.ts';
	import { parseSequence, type Sequence, type Step } from '#lib/animation/sequence.ts';
	import { Sequencer } from '#lib/animation/sequencer.ts';
	import { PRESETS } from '#lib/avatar/poses.ts';
	import type { AvatarViewer } from '#lib/render/viewer.ts';

	interface Props {
		player: Player | undefined;
		viewer: AvatarViewer | undefined;
		loadMotion: (source: string, name: string, play: boolean) => Promise<MotionClip | null>;
	}

	let { player, viewer, loadMotion }: Props = $props();

	let text = $state('');
	let editing = $state(false);
	let name = $state('');
	let error = $state('');
	let steps: { label: string; start: number; end: number }[] = $state([]);
	let step = $state(-1);
	let time = $state(0);
	let total = $state(0);
	let running = $state(false);
	let sequencer: Sequencer | null = null;
	let frame = 0;

	function tick() {
		frame = requestAnimationFrame(tick);
		if (!sequencer) return;
		time = sequencer.time;
		step = sequencer.step;
		running = !sequencer.finished;
	}
	frame = requestAnimationFrame(tick);
	onDestroy(() => cancelAnimationFrame(frame));

	/** Read a sequence from a `.json` URL or file into the editor, and run it. */
	export async function loadSequence(source: Blob | string, fileName: string) {
		try {
			text =
				typeof source === 'string'
					? await fetch(source).then((response) => {
							if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
							return response.text();
						})
					: await source.text();
			await run();
		} catch (cause) {
			error = `Could not load ${fileName}: ${(cause as Error).message}`;
		}
	}

	export function stopSequence() {
		sequencer?.stop();
		sequencer = null;
		running = false;
		step = -1;
	}

	/** Parse the editor's sequence, load the clips it names by URL, and start it. */
	async function run() {
		if (!player || !viewer) return;
		error = '';
		let sequence: Sequence;
		try {
			sequence = parseSequence(JSON.parse(text));
		} catch (cause) {
			error = (cause as Error).message;
			editing = true;
			return;
		}
		const clips = new Map(player.clips.map((clip) => [clip.name, clip]));
		for (const s of sequence.steps) {
			if (s.kind !== 'clip' || clips.has(s.clip)) continue;
			if (!s.clip.endsWith('.motion')) {
				error = `no clip "${s.clip}" is loaded: open it, or name its .motion URL`;
				return;
			}
			const clip = await loadMotion(s.clip, s.clip.split('/').pop()!, false);
			if (!clip) {
				error = `could not load ${s.clip}`;
				return;
			}
			clips.set(s.clip, clip);
		}
		for (const s of sequence.steps) {
			if (s.kind === 'pose' && !(s.pose in PRESETS)) {
				error = `no pose "${s.pose}"; there are ${Object.keys(PRESETS).join(', ')}`;
				return;
			}
		}
		stopSequence();
		sequencer = new Sequencer(sequence, clips, player, viewer);
		name = sequence.name;
		steps = sequencer.schedule.map(({ step: s, start, end }) => ({
			label: describe(s),
			start,
			end
		}));
		total = sequencer.total;
		sequencer.start();
		running = true;
		editing = false;
	}

	function openFile(event: Event) {
		const file = (event.currentTarget as HTMLInputElement).files?.[0];
		if (file) loadSequence(file, file.name);
	}

	/** Start the editor from an example using the clips that are loaded. */
	function example() {
		const names = player?.clips.map((clip) => clip.name) ?? [];
		const first = names[0] ?? 'my_clip';
		const second = names[1] ?? first;
		text = JSON.stringify(
			{
				name: 'example',
				loop: true,
				steps: [
					{ clip: first, to: 3, fade: 0.5 },
					{ pose: 'Wave', hold: 2 },
					{ clip: second, speed: 0.75, repeat: 2 },
					{ wait: 1 }
				],
				cues: [
					{ at: 0, camera: 'video' },
					{ at: 1, for: 4, character: 'all', look: 'camera' },
					{ at: 3, for: 2.5, character: 0, expression: { mouthSmile: 0.8, eyeSquint: 0.3 } }
				]
			},
			null,
			2
		);
		editing = true;
	}

	function describe(s: Step): string {
		if (s.kind === 'clip') {
			const trim = s.from > 0 || s.to !== null ? ` ${s.from}–${s.to ?? 'end'} s` : '';
			const speed = s.speed !== 1 ? ` at ${s.speed}×` : '';
			return `${s.clip}${trim}${speed}${s.repeat !== 1 ? ` ×${s.repeat}` : ''}`;
		}
		return s.kind === 'pose' ? `${s.pose} pose, ${s.hold} s` : `wait ${s.seconds} s`;
	}
</script>

<div class="space-y-2 border-t border-zinc-800 pt-3">
	<div class="flex items-baseline justify-between">
		<h3 class="text-xs font-semibold tracking-wider text-zinc-400 uppercase">Sequence</h3>
		{#if running}
			<span class="font-mono text-xs text-zinc-400 tabular-nums">
				{time.toFixed(1)} / {total.toFixed(1)} s
			</span>
		{/if}
	</div>
	<div class="grid grid-cols-3 gap-2">
		<label
			class={[
				'flex cursor-pointer items-center justify-center rounded-md border border-zinc-700 px-2 py-1 text-sm hover:bg-zinc-800',
				!viewer && 'pointer-events-none opacity-40'
			]}
		>
			Open
			<input type="file" accept=".json,application/json" class="sr-only" onchange={openFile} />
		</label>
		<button
			class="rounded-md border border-zinc-700 px-2 py-1 text-sm hover:bg-zinc-800 disabled:opacity-40"
			disabled={!viewer}
			onclick={() => (editing ? (editing = false) : text ? (editing = true) : example())}
		>
			{editing ? 'Close' : 'Edit'}
		</button>
		{#if running}
			<button
				class="rounded-md bg-zinc-700 px-2 py-1 text-sm font-medium text-white hover:bg-zinc-600"
				onclick={stopSequence}
			>
				End
			</button>
		{:else}
			<button
				class="rounded-md bg-sky-600 px-2 py-1 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-40"
				disabled={!viewer || !text}
				onclick={run}
			>
				Run
			</button>
		{/if}
	</div>

	{#if editing}
		<textarea
			bind:value={text}
			spellcheck="false"
			rows="14"
			class="w-full rounded-md border border-zinc-700 bg-zinc-950 p-2 font-mono text-xs leading-relaxed focus:border-sky-500 focus:outline-none"
		></textarea>
		<button class="text-xs text-sky-400 hover:text-sky-300" onclick={example}>
			Replace with an example
		</button>
	{/if}
	{#if error}
		<p class="text-xs text-red-400">{error}</p>
	{/if}

	{#if steps.length && !editing}
		<p class="truncate text-xs text-zinc-400">{name}</p>
		<ol class="space-y-1">
			{#each steps as s, index (index)}
				<li
					class={[
						'flex justify-between rounded px-2 py-1 text-xs',
						running && index === step ? 'bg-sky-600/20 text-sky-200' : 'text-zinc-400'
					]}
				>
					<span class="truncate">{s.label}</span>
					<span class="ml-2 shrink-0 font-mono tabular-nums">{s.start.toFixed(1)} s</span>
				</li>
			{/each}
		</ol>
	{/if}
</div>
