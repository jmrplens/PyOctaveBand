// Contrast audit for the site palette.
//
// Parses src/styles/theme.css, resolves the light and the dark theme and
// reports the WCAG 2.1 contrast ratio of every pair the site actually
// renders, plus the seam between the page ground and the opaque plate every
// committed figure carries.
//
// Pairs are either enforced (text 4.5:1, meaningful non-text 3:1, per WCAG
// 1.4.3 / 1.4.11) or informational (purely decorative hairlines, and the
// figure plate seams, where the goal is 1.00 rather than a floor). Only
// enforced pairs can fail the run.
//
// Run: node scripts/check-contrast.mjs [--json]
import { readFileSync } from 'node:fs';

const css = readFileSync(new URL('../src/styles/theme.css', import.meta.url), 'utf8');

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

// The dark theme is the root scope; the light theme overrides it, so it is
// resolved on top of the dark one exactly as the cascade does it.
const dark = block(':root');
const light = block(":root[data-theme='light']");
const PALETTES = {
	'instrument / dark': resolve({ ...dark }),
	'instrument / light': resolve({ ...dark, ...light }),
};

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

// Every committed figure is an opaque plate, so the page ground never shows
// through one: what it has to do is keep the plate EDGE from reading. The two
// dark plate colours are #000000 (404 matplotlib figures, whose figure patch
// is a <path> with no fill, which SVG paints black) and #0d1117 (the 102
// hand-authored diagrams). Both seams are informational: an invisible edge is
// the goal, and a contrast threshold cannot express that.
const DARK_PLATES = [
	['matplotlib plate #000000 seam (want 1.00)', '#000000'],
	['diagram plate #0d1117 seam (want 1.00)', '#0d1117'],
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

	// Page header chips. Each pill is a link, so its text is enforced at 4.5:1
	// and its border at 3:1 (the border is what carries the category, so it is
	// meaningful non-text). The hover state fills the pill, so the text is
	// checked against that ground too.
	for (const [label, kind] of [
		['standards chip', 'standard'],
		['named-work chip', 'named'],
		['"+N more" chip', 'muted'],
	]) {
		const ink = p[`--ph-chip-${kind}-ink`];
		check(name, `${label} text on page`, ink, bg, 4.5);
		check(name, `${label} text on its hover fill`, ink, p[`--ph-chip-${kind}-fill`], 4.5);
		check(name, `${label} border on page`, p[`--ph-chip-${kind}-line`], bg, 3);
	}

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
		for (const [label, plate] of DARK_PLATES) check(name, label, plate, bg, 0);
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
