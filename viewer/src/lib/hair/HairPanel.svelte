<script lang="ts">
	import { untrack } from 'svelte';
	import { PRESETS, type GroomShape, type HairLook, type Style } from '#lib/hair/groom.ts';
	import { HairLayer, type HairStats } from '#lib/hair/hair.ts';
	import type { AvatarViewer } from '#lib/render/viewer.ts';

	let { viewer }: { viewer: AvatarViewer | undefined } = $props();

	interface Slider<T> {
		key: keyof T;
		label: string;
		min: number;
		max: number;
		step: number;
		/** Display unit and the factor from the stored value to it. */
		unit?: string;
		factor?: number;
	}

	const SHAPE_SLIDERS: Slider<GroomShape>[] = [
		{ key: 'length', label: 'Length', min: 0.04, max: 0.8, step: 0.01, unit: 'cm', factor: 100 },
		{ key: 'strands', label: 'Strands', min: 10000, max: 120000, step: 5000 },
		{ key: 'lengthVariation', label: 'Length variation', min: 0, max: 1, step: 0.05 },
		{ key: 'volume', label: 'Volume', min: 0, max: 0.04, step: 0.001, unit: 'mm', factor: 1000 },
		{ key: 'wave', label: 'Wave', min: 0, max: 0.03, step: 0.001, unit: 'mm', factor: 1000 },
		{
			key: 'waveLength',
			label: 'Wavelength',
			min: 0.04,
			max: 0.25,
			step: 0.005,
			unit: 'cm',
			factor: 100
		},
		{
			key: 'curl',
			label: 'Curl radius',
			min: 0,
			max: 0.02,
			step: 0.0005,
			unit: 'mm',
			factor: 1000
		},
		{
			key: 'curlPitch',
			label: 'Curl length per turn',
			min: 0.015,
			max: 0.12,
			step: 0.005,
			unit: 'cm',
			factor: 100
		},
		{ key: 'clumping', label: 'Clumping', min: 0, max: 1, step: 0.05 },
		{ key: 'frizz', label: 'Frizz', min: 0, max: 1, step: 0.05 },
		{ key: 'flyaways', label: 'Flyaways', min: 0, max: 0.2, step: 0.01 },
		{ key: 'parting', label: 'Parting (right to left)', min: -1, max: 1, step: 0.05 },
		{ key: 'combBack', label: 'Combed back', min: 0, max: 1, step: 0.05 },
		{ key: 'front', label: 'In front of the shoulders', min: 0, max: 1, step: 0.05 },
		{ key: 'hairline', label: 'Hairline height', min: -8, max: 15, step: 0.5, unit: '°' },
		{ key: 'widowsPeak', label: "Widow's peak", min: 0, max: 1, step: 0.05 },
		{ key: 'stiffness', label: 'Stiffness', min: 0, max: 1, step: 0.05 }
	];

	const LOOK_SLIDERS: Slider<HairLook>[] = [
		{ key: 'melanin', label: 'Melanin (blonde to black)', min: 0, max: 1, step: 0.01 },
		{ key: 'redness', label: 'Redness', min: 0, max: 1, step: 0.01 },
		{ key: 'variation', label: 'Strand variation', min: 0, max: 1, step: 0.05 },
		{ key: 'thickness', label: 'Fibre diameter', min: 40, max: 160, step: 5, unit: 'µm' },
		{ key: 'roughness', label: 'Roughness', min: 0, max: 1, step: 0.05 },
		{ key: 'shadow', label: 'Shadow on skin', min: 0, max: 1, step: 0.05 }
	];

	const STYLES: Style[] = ['loose', 'ponytail'];

	let enabled = $state(true);
	let preset = $state('Default');
	let shape: GroomShape = $state({ ...PRESETS.Default.shape });
	let look: HairLook = $state({ ...PRESETS.Default.look });
	let simulate = $state(true);
	let wind = $state(0);
	let showGroom = $state(false);
	let layer: HairLayer | undefined = $state.raw();
	let stats: HairStats | undefined = $state();
	/** The shape the layer's hair was last grown with, as JSON. */
	let grown = '';

	function setShape(key: keyof GroomShape, value: number) {
		shape = { ...shape, [key]: value };
	}

	function applyPreset(name: string) {
		preset = name;
		shape = { ...PRESETS[name].shape };
		look = { ...PRESETS[name].look };
	}

	function format<T>(slider: Slider<T>, value: number): string {
		const shown = value * (slider.factor ?? 1);
		if (slider.key === 'strands') return `${Math.round(shown / 1000)}k`;
		const digits = slider.step * (slider.factor ?? 1) < 1 ? 2 : 0;
		return `${shown.toFixed(digits)}${slider.unit ? ` ${slider.unit}` : ''}`;
	}

	// The layer lives with the viewer: a new avatar file brings a new viewer and new hair.
	$effect(() => {
		const current = viewer;
		if (!current || !enabled) return;
		const created = untrack(
			() => new HairLayer(current, $state.snapshot(shape), $state.snapshot(look))
		);
		created.setSimulation(
			untrack(() => simulate),
			untrack(() => wind)
		);
		current.addLayer(created);
		layer = created;
		stats = created.stats();
		grown = JSON.stringify(untrack(() => $state.snapshot(shape)));
		return () => {
			current.removeLayer(created);
			layer = undefined;
		};
	});

	// Growing takes a moment: wait for the slider to rest.
	$effect(() => {
		const next = JSON.stringify($state.snapshot(shape));
		const target = layer;
		if (!target || next === grown) return;
		const timer = setTimeout(() => {
			grown = next;
			target.setShape(JSON.parse(next));
			stats = target.stats();
		}, 250);
		return () => clearTimeout(timer);
	});
	$effect(() => {
		layer?.setLook($state.snapshot(look));
	});
	$effect(() => {
		layer?.setSimulation(simulate, wind);
	});
</script>

<section class="space-y-3">
	<div class="flex items-baseline justify-between">
		<h2 class="text-xs font-semibold tracking-wider text-zinc-400 uppercase">Hair</h2>
		<button
			class="text-xs text-sky-400 hover:text-sky-300 disabled:opacity-40"
			disabled={!layer}
			onclick={() => layer?.reset()}
		>
			Reset hair
		</button>
	</div>
	<div class="grid grid-cols-2 gap-2">
		<label class="flex items-center gap-2 text-sm">
			<input type="checkbox" bind:checked={enabled} class="accent-sky-500" /> Hair
		</label>
		<label class="flex items-center gap-2 text-sm" title="Simulate the hair as the avatar moves">
			<input type="checkbox" bind:checked={simulate} class="accent-sky-500" /> Simulate
		</label>
	</div>
	<div class="flex flex-wrap gap-1.5">
		{#each Object.keys(PRESETS) as name (name)}
			<button
				class={[
					'rounded-full border border-zinc-700 px-2.5 py-0.5 text-xs hover:bg-zinc-800',
					preset === name && 'border-sky-500 text-sky-300'
				]}
				onclick={() => applyPreset(name)}
			>
				{name}
			</button>
		{/each}
	</div>
	{#if stats}
		<p class="font-mono text-xs text-zinc-500">
			{stats.strands.toLocaleString()} strands · {stats.guides.toLocaleString()} guides · grown in
			{Math.round(stats.growMs)} ms
		</p>
	{/if}

	<label class="block">
		<span class="flex justify-between text-xs text-zinc-400">
			Wind
			<span class="font-mono tabular-nums">{wind.toFixed(1)} m/s</span>
		</span>
		<input
			type="range"
			min="0"
			max="4"
			step="0.1"
			bind:value={wind}
			class="w-full accent-sky-500"
		/>
	</label>

	{#each LOOK_SLIDERS as slider (slider.key)}
		<label class="block">
			<span class="flex justify-between text-xs text-zinc-400">
				{slider.label}
				<span class="font-mono tabular-nums">{format(slider, look[slider.key])}</span>
			</span>
			<input
				type="range"
				min={slider.min}
				max={slider.max}
				step={slider.step}
				bind:value={look[slider.key]}
				class="w-full accent-sky-500"
			/>
		</label>
	{/each}

	<div class="flex items-baseline justify-between">
		<span class="text-xs text-zinc-400">Groom</span>
		<button
			class="text-xs text-sky-400 hover:text-sky-300"
			onclick={() => (showGroom = !showGroom)}
		>
			{showGroom ? 'Hide' : 'Show'}
		</button>
	</div>
	{#if showGroom}
		<div class="flex gap-1.5">
			{#each STYLES as style (style)}
				<button
					class={[
						'rounded-full border border-zinc-700 px-2.5 py-0.5 text-xs capitalize hover:bg-zinc-800',
						shape.style === style && 'border-sky-500 text-sky-300'
					]}
					onclick={() => (shape = { ...shape, style })}
				>
					{style}
				</button>
			{/each}
		</div>
		{#each SHAPE_SLIDERS as slider (slider.key)}
			<label class="block">
				<span class="flex justify-between text-xs text-zinc-400">
					{slider.label}
					<span class="font-mono tabular-nums">{format(slider, shape[slider.key] as number)}</span>
				</span>
				<input
					type="range"
					min={slider.min}
					max={slider.max}
					step={slider.step}
					value={shape[slider.key]}
					oninput={(event) => setShape(slider.key, Number(event.currentTarget.value))}
					class="w-full accent-sky-500"
				/>
			</label>
		{/each}
	{/if}
</section>
