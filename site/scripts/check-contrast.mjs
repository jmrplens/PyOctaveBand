// Contrast audit for the experimental colour directions.
//
// Parses src/styles/theme-directions.css, resolves the six palettes
// (3 directions x light/dark) and reports the WCAG 2.1 contrast ratio of
// every pair the site actually renders, plus the matplotlib ink used by the
// committed dark figures (transparent background, so it lands straight on
// the page ground).
//
// Pairs are either enforced (text 4.5:1, meaningful non-text 3:1, per WCAG
// 1.4.3 / 1.4.11) or informational (purely decorative hairlines and the
// figures' own gridlines, which carry no information on their own). Only
// enforced pairs can fail the run.
//
// Run: node scripts/check-contrast.mjs [--json]
import { readFileSync } from 'node:fs';

const css = readFileSync(new URL('../src/styles/theme-directions.css', import.meta.url), 'utf8');

/** All `--token: #hex;` declarations of the rule whose selector list is exactly `selector`. */
function block(selector) {
	const out = {};
	const re = /([^{}]+)\{([^{}]*)\}/g;
	let m;
	const want = selector.replace(/\s+/g, ' ').trim();
	while ((m = re.exec(css))) {
		const found = m[1].replace(/\/\*[\s\S]*?\*\//g, '').replace(/\s+/g, ' ').trim();
		if (found !== want) continue;
		for (const [, name, value] of m[2].matchAll(
			/(--[\w-]+)\s*:\s*(#[0-9a-fA-F]{6}|var\(--[\w-]+\))\s*;/g,
		)) {
			out[name] = value.toLowerCase();
		}
	}
	return out;
}

/** Resolve `var(--other)` aliases to their hex, in place. */
function resolve(palette) {
	for (let pass = 0; pass < 5; pass++) {
		for (const [name, value] of Object.entries(palette)) {
			const alias = /^var\((--[\w-]+)\)$/.exec(value);
			if (alias && palette[alias[1]]) palette[name] = palette[alias[1]];
		}
	}
	for (const [name, value] of Object.entries(palette)) {
		if (!value.startsWith('#')) delete palette[name];
	}
	return palette;
}

const instrumentDark = block(":root, :root[data-accent='instrument']");
const instrumentLight = block(":root[data-theme='light'], :root[data-accent='instrument'][data-theme='light']");
const PALETTES = {
	'instrument / dark': resolve({ ...instrumentDark }),
	'instrument / light': resolve({ ...instrumentDark, ...instrumentLight }),
};
for (const name of ['blueprint', 'graphite']) {
	const dark = block(`:root[data-accent='${name}']`);
	const light = block(`:root[data-accent='${name}'][data-theme='light']`);
	PALETTES[`${name} / dark`] = resolve({ ...dark });
	PALETTES[`${name} / light`] = resolve({ ...dark, ...light });
}

for (const [name, p] of Object.entries(PALETTES)) {
	if (!p['--sl-color-black']) throw new Error(`palette "${name}" did not parse`);
}

const srgb = (hex) => {
	const n = parseInt(hex.slice(1), 16);
	return [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((c) => {
		const s = c / 255;
		return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
	});
};
const luminance = (hex) => {
	const [r, g, b] = srgb(hex);
	return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const ratio = (a, b) => {
	const [l1, l2] = [luminance(a), luminance(b)].sort((x, y) => y - x);
	return (l1 + 0.05) / (l2 + 0.05);
};

// matplotlib tab10 ink used by the committed dark figures, which are
// transparent and therefore composited straight onto the page ground.
const FIGURE_INK = [
	['figure white ink (axes, labels)', '#ffffff', 3],
	['figure tab10 blue #1f77b4', '#1f77b4', 3],
	['figure tab10 red #d62728', '#d62728', 3],
	['figure tab10 green #2ca02c', '#2ca02c', 3],
	['figure gridline #555555 (decorative)', '#555555', 0],
];

const rows = [];
let failures = 0;

const check = (palette, label, fg, bg, min) => {
	if (!fg || !bg) throw new Error(`missing colour for "${label}" in ${palette}`);
	const r = ratio(fg, bg);
	const enforced = min > 0;
	const pass = !enforced || r >= min;
	if (!pass) failures++;
	rows.push({ palette, label, fg, bg, ratio: Math.round(r * 100) / 100, min, enforced, pass });
};

for (const [name, p] of Object.entries(PALETTES)) {
	const isLight = name.endsWith('light');
	const bg = p['--sl-color-black']; // --sl-color-bg resolves to this
	const textAccent = isLight ? p['--sl-color-accent'] : p['--sl-color-accent-high'];

	check(name, 'body text (gray-2) on page', p['--sl-color-gray-2'], bg, 4.5);
	check(name, 'muted text (gray-3) on page', p['--sl-color-gray-3'], bg, 4.5);
	check(name, 'headings on page', p['--sl-color-white'], bg, 4.5);
	check(name, 'body text on nav / sidebar', p['--sl-color-gray-2'], p['--sl-color-bg-nav'], 4.5);
	check(name, 'link colour on page', textAccent, bg, 4.5);
	check(name, 'link colour on card surface', textAccent, p['--ph-surface'], 4.5);
	check(name, 'muted text on card surface', p['--sl-color-gray-3'], p['--ph-surface'], 4.5);
	// Primary button: --sl-color-bg-accent ground with --sl-color-text-invert label.
	check(
		name,
		'primary button label',
		isLight ? p['--sl-color-black'] : p['--sl-color-accent-low'],
		isLight ? p['--sl-color-accent'] : p['--sl-color-accent-high'],
		4.5,
	);
	check(name, 'focus ring / accent mark on page', p['--ph-mark'], bg, 3);
	check(name, 'inline code text on its ground', p['--sl-color-gray-2'], p['--sl-color-bg-inline-code'], isLight ? 4.5 : 0);
	check(name, 'decorative hairline on page', p['--ph-border-strong'], bg, 0);

	if (isLight) {
		// Light figures carry their own opaque #ffffff ground: it has to be
		// invisible against the page, i.e. the page ground must stay white.
		const seam = ratio('#ffffff', bg);
		rows.push({
			palette: name,
			label: 'light figure ground seam (want 1.00)',
			fg: '#ffffff',
			bg,
			ratio: Math.round(seam * 100) / 100,
			min: 0,
			enforced: false,
			pass: true,
		});
		if (seam > 1.02) failures++;
	} else {
		for (const [label, ink, min] of FIGURE_INK) check(name, label, ink, bg, min);
		// 102 of the 506 dark figures are hand-authored diagrams that carry an
		// opaque #0d1117 ground instead of a transparent one, so they show as a
		// plate on the page. The closer this ratio is to 1.00 the less the
		// plate edge reads; it is informational, never a failure.
		check(name, 'dark diagram plate #0d1117 seam (want 1.00)', '#0d1117', bg, 0);
	}
}

if (process.argv.includes('--json')) {
	console.log(JSON.stringify(rows, null, 2));
} else {
	let current = '';
	for (const r of rows) {
		if (r.palette !== current) {
			current = r.palette;
			console.log(`\n### ${current}`);
		}
		const flag = r.enforced ? (r.pass ? 'ok  ' : 'FAIL') : 'info';
		const min = r.enforced ? `min ${r.min}` : 'decorative';
		console.log(
			`  ${flag} ${r.ratio.toFixed(2).padStart(6)}:1  (${min})  ${r.label}  ${r.fg} on ${r.bg}`,
		);
	}
	console.log(`\n${rows.length} pairs measured, ${failures} enforced pair(s) below threshold.`);
}

process.exit(failures ? 1 : 0);
