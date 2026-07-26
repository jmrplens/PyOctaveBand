// Screenshot matrix for the integrated branch (page chips + instrument palette).
//
// Same approach as scripts/redesign-shots.mjs: it drives its own headless
// Chrome out of the pnpm store, so every capture states its viewport, locale
// and theme up front and nothing depends on an interactive browser session.
//
// Usage: node scripts/integration-shots.mjs [--base http://localhost:4321] [filter...]
import { readdirSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const siteDir = join(here, '..');
const OUT = join(siteDir, '..', 'integration-shots');

const require = createRequire(import.meta.url);
const store = join(siteDir, 'node_modules', '.pnpm');
const pkg = readdirSync(store).find((d) => /^puppeteer@/.test(d));
if (!pkg) throw new Error('puppeteer not found in the pnpm store; run pnpm install');
const puppeteer = require(join(store, pkg, 'node_modules', 'puppeteer'));

const args = process.argv.slice(2);
const baseIdx = args.indexOf('--base');
const BASE = baseIdx === -1 ? 'http://localhost:4321' : args[baseIdx + 1];
const filters = args.filter((a, i) => !a.startsWith('--') && !(baseIdx !== -1 && i === baseIdx + 1));

const DESKTOP = { width: 1440, height: 900 };
const PHONE = { width: 390, height: 844 };

/** name, path, viewport, theme, scroll offset, full page. */
const SHOTS = [
	// The chip run under the H1 on a standards-heavy guide, both themes: the
	// case where the chips have to coexist with the accent used by links.
	['01-guide-chips-light', '/guides/insulation-lab/', DESKTOP, 'light', 0],
	['02-guide-chips-dark', '/guides/insulation-lab/', DESKTOP, 'dark', 0],
	// A bibliography long enough to overflow the caps, so the "+N more" chip
	// shows next to both categories.
	['03-guide-chips-overflow-light', '/guides/underwater-propagation/', DESKTOP, 'light', 0],
	['04-guide-chips-overflow-dark', '/guides/underwater-propagation/', DESKTOP, 'dark', 0],
	// Chips against body links further down the same page, where the two have
	// to stay distinguishable.
	['05-guide-body-links-light', '/guides/insulation-lab/', DESKTOP, 'light', 1400],
	['06-guide-body-links-dark', '/guides/insulation-lab/', DESKTOP, 'dark', 1400],
	// The References section the chips point into.
	['07-references-light', '/guides/insulation-lab/', DESKTOP, 'light', 0, false, '#references'],
	// API reference page: no `references` frontmatter, so no chip run.
	['08-api-light', '/reference/api/building/lab-insulation/', DESKTOP, 'light', 0],
	['09-api-dark', '/reference/api/building/lab-insulation/', DESKTOP, 'dark', 0],
	// Front page: clause-numbered sections, no chips by design.
	['10-home-light', '/', DESKTOP, 'light', 0, true],
	['11-home-dark', '/', DESKTOP, 'dark', 0, true],
	// Spanish guide: chips carry the Spanish labels and the "y" conjunction.
	['12-guide-es-light', '/es/guides/insulation-lab/', DESKTOP, 'light', 0],
	['13-guide-es-dark', '/es/guides/insulation-lab/', DESKTOP, 'dark', 0],
	// Narrow width: the chip run wraps under the H1 and the mobile theme
	// toggle sits in the header. 14 and 15 are the same page in the two
	// themes, so they are also the two states of the toggle: the button shows
	// the mode a tap would give you, so it is a moon on the light shot and a
	// sun on the dark one.
	['14-guide-phone-light', '/guides/underwater-propagation/', PHONE, 'light', 0],
	['15-guide-phone-dark', '/guides/underwater-propagation/', PHONE, 'dark', 0],
	['16-home-phone-dark', '/', PHONE, 'dark', 0, true],
	// A page past the raised cap of nine, where the run wraps to several rows
	// and ends on the "+N more" chip.
	['17-cap-overflow-phone-light', '/guides/porous-absorbers/', PHONE, 'light', 0],
];

/**
 * The sidebar, which is stock Starlight with every group `collapsed: true`.
 * Some of these states exist on arrival and some only after a reader clicks,
 * so each shot names the group labels to press first. They are matched by
 * their text on the `<summary>` row rather than by a class, because there is
 * no markup of ours left in the tree to match.
 *
 * `anchor` names the group to bring to the top of the sidebar's own scroller
 * before the capture (null keeps the tree at its top); `phone` swaps the
 * desktop pane for the mobile drawer, which is opened first and captured
 * whole, because there the tree covers the page.
 */
const SIDEBAR = [
	// The whole tree as it arrives on a guide page: twelve groups, folded,
	// except the chain holding the current page.
	['38-tree-en-light', '/guides/loudness/', 'light', [], { anchor: null }],
	['39-tree-en-dark', '/guides/loudness/', 'dark', [], { anchor: null }],
	['40-tree-es-light', '/es/guides/loudness/', 'light', [], { anchor: null }],
	['41-tree-es-dark', '/es/guides/loudness/', 'dark', [], { anchor: null }],
	// A section landing page: the Overview row is the marked entry, which is
	// where the Overview-first convention is now visible.
	['42-section-landing-light', '/guides/sections/psychoacoustics/', 'light', [], { anchor: null }],
	[
		'43-section-landing-es-light',
		'/es/guides/sections/psychoacoustics/',
		'light',
		[],
		{ anchor: null },
	],
	// Reference opened from a guide page: its own Overview row above Theory,
	// Conformance report and Bibliography.
	['44-reference-open-light', '/guides/levels/', 'light', ['Reference'], { anchor: 'Reference' }],
	['45-reference-open-dark', '/guides/levels/', 'dark', ['Reference'], { anchor: 'Reference' }],
	// The Reference overview page itself, arriving with its row marked.
	['46-reference-page-light', '/reference/', 'light', [], { anchor: 'Reference' }],
	// The phone drawer: the same tree in a 788 px drawer, both languages.
	['47-drawer-phone-en-light', '/guides/loudness/', 'light', [], { phone: true, anchor: null }],
	['48-drawer-phone-en-dark', '/guides/loudness/', 'dark', [], { phone: true, anchor: null }],
	['49-drawer-phone-es-dark', '/es/guides/loudness/', 'dark', [], { phone: true, anchor: null }],
	// The mixed case, which is the one that showed the type mismatch: a group
	// whose children are nested groups (Rooms and buildings) directly above a
	// group whose children are pages (Materials and surfaces, open on arrival
	// because the current page is in it). Every row at the second level reads
	// the same; the caret and the indent rule are what separate them.
	[
		'50-drawer-mixed-levels-dark',
		'/guides/porous-absorbers/',
		'dark',
		['Rooms and buildings'],
		{ phone: true, anchor: null },
	],
	// Arriving on a guide page: the API group is closed.
	['30-api-closed-light', '/guides/levels/', 'light', []],
	['31-api-closed-dark', '/guides/levels/', 'dark', []],
	// One click: the sections appear and nothing below them unfolds.
	['32-api-open-light', '/guides/levels/', 'light', ['API reference']],
	['33-api-open-dark', '/guides/levels/', 'dark', ['API reference']],
	// A second click on one section: its pages, and only its pages.
	['34-api-nested-light', '/guides/levels/', 'light', ['API reference', 'Psychoacoustics']],
	['35-api-nested-dark', '/guides/levels/', 'dark', ['API reference', 'Psychoacoustics']],
	// On an API page the chain down to the current page is already open and
	// the page is marked, with no click at all.
	['36-api-page-light', '/reference/api/psychoacoustics/sharpness/', 'light', []],
	['37-api-page-dark', '/reference/api/psychoacoustics/sharpness/', 'dark', []],
];

/** Shots that also need a faked Accept-Language list, for the banner. */
const EXTRA = [
	{
		name: '20-lang-banner-over-chips',
		path: '/guides/insulation-lab/',
		viewport: DESKTOP,
		theme: 'light',
		languages: ['es-ES', 'es'],
	},
	{
		name: '21-lang-banner-phone',
		path: '/es/guides/insulation-lab/',
		viewport: PHONE,
		theme: 'dark',
		languages: ['en-US', 'en'],
	},
];

await mkdir(OUT, { recursive: true });

const browser = await puppeteer.launch({
	headless: true,
	args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--font-render-hinting=none'],
});

for (const [name, path, viewport, theme, scrollY = 0, fullPage = false, hash] of SHOTS) {
	if (filters.length && !filters.some((f) => name.includes(f))) continue;
	const page = await browser.newPage();
	await page.setViewport({ ...viewport, deviceScaleFactor: 1 });
	await page.evaluateOnNewDocument((t) => {
		try {
			localStorage.setItem('starlight-theme', t);
		} catch {}
	}, theme);
	await page.goto(`${BASE}/phonometry${path}${hash ?? ''}`, {
		waitUntil: 'networkidle0',
		timeout: 90000,
	});
	await page.evaluate((t) => {
		document.documentElement.dataset.theme = t;
	}, theme);
	// The dev server injects the Astro toolbar; it is not part of the design.
	await page.addStyleTag({ content: 'astro-dev-toolbar{display:none !important}' });
	if (fullPage) {
		await page.evaluate(async () => {
			const step = window.innerHeight * 0.8;
			for (let y = 0; y < document.body.scrollHeight; y += step) {
				window.scrollTo(0, y);
				await new Promise((r) => setTimeout(r, 120));
			}
			window.scrollTo(0, 0);
		});
		await new Promise((r) => setTimeout(r, 1200));
	}
	if (hash) {
		await page.evaluate((h) => document.querySelector(h)?.scrollIntoView(), hash);
		await new Promise((r) => setTimeout(r, 400));
	}
	if (scrollY) {
		await page.evaluate((y) => window.scrollTo(0, y), scrollY);
		await new Promise((r) => setTimeout(r, 400));
	}
	await page.screenshot({ path: join(OUT, `${name}.png`), fullPage });
	await page.close();
	console.log(`captured ${name}`);
}

/** Group labels, per locale, that a shot can be anchored on. */
const ANCHORS = {
	'API reference': /^(API reference|Referencia de la API)/,
	Reference: /^(Reference|Referencia)$/,
};

for (const [name, path, theme, clicks, opts = {}] of SIDEBAR) {
	if (filters.length && !filters.some((f) => name.includes(f))) continue;
	const { phone = false, anchor = 'API reference' } = opts;
	const page = await browser.newPage();
	await page.setViewport({ ...(phone ? PHONE : DESKTOP), deviceScaleFactor: 1 });
	await page.evaluateOnNewDocument((t) => {
		try {
			localStorage.setItem('starlight-theme', t);
		} catch {}
	}, theme);
	await page.goto(`${BASE}/phonometry${path}`, { waitUntil: 'networkidle0', timeout: 90000 });
	await page.evaluate((t) => {
		document.documentElement.dataset.theme = t;
	}, theme);
	await page.addStyleTag({ content: 'astro-dev-toolbar{display:none !important}' });
	// Below 50rem the sidebar lives behind the menu button, so open it first.
	if (phone) {
		await page.evaluate(() => {
			document.querySelector('starlight-menu-button button')?.click();
		});
		await new Promise((r) => setTimeout(r, 400));
	}
	for (const label of clicks) {
		await page.evaluate((text) => {
			[...document.querySelectorAll('#starlight__sidebar summary')]
				.find((s) => s.textContent.trim().startsWith(text))
				?.click();
		}, label);
		await new Promise((r) => setTimeout(r, 250));
	}
	// The drawer covers the page, so it is the whole viewport that is the
	// subject there; on desktop the sidebar pane is framed on its own.
	let clip;
	if (!phone) {
		clip = await page.evaluate((pattern) => {
			const scroller = document.getElementById('starlight__sidebar');
			if (pattern) {
				const group = [...document.querySelectorAll('#starlight__sidebar summary')]
					.find((s) => new RegExp(pattern).test(s.textContent.trim()))
					.closest('li');
				scroller.scrollTop +=
					group.getBoundingClientRect().top - scroller.getBoundingClientRect().top - 16;
			} else {
				scroller.scrollTop = 0;
			}
			const r = scroller.getBoundingClientRect();
			return {
				x: 0,
				y: Math.max(0, Math.floor(r.y)),
				width: Math.ceil(r.right + 8),
				height: Math.ceil(Math.min(r.height, window.innerHeight - Math.max(0, r.y))),
			};
		}, anchor ? ANCHORS[anchor].source : null);
	}
	await new Promise((r) => setTimeout(r, 300));
	// `captureBeyondViewport` would resize the viewport to fit the clip, which
	// changes how far the sidebar can scroll and frames the wrong rows.
	await page.screenshot({ path: join(OUT, `${name}.png`), clip, captureBeyondViewport: false });
	await page.close();
	console.log(`captured ${name}`);
}

for (const shot of EXTRA) {
	if (filters.length && !filters.some((f) => shot.name.includes(f))) continue;
	// A fresh context per shot, so a previous scenario's localStorage cannot
	// change what this one shows.
	const context = await browser.createBrowserContext();
	const page = await context.newPage();
	await page.setViewport({ ...shot.viewport, deviceScaleFactor: 1 });
	await page.evaluateOnNewDocument(
		(t, langs) => {
			try {
				localStorage.setItem('starlight-theme', t);
			} catch {}
			if (langs) {
				Object.defineProperty(navigator, 'languages', { get: () => langs });
				Object.defineProperty(navigator, 'language', { get: () => langs[0] });
			}
		},
		shot.theme,
		shot.languages ?? null,
	);
	await page.goto(`${BASE}/phonometry${shot.path}`, { waitUntil: 'networkidle0', timeout: 90000 });
	await page.evaluate((t) => {
		document.documentElement.dataset.theme = t;
	}, shot.theme);
	await page.addStyleTag({ content: 'astro-dev-toolbar{display:none !important}' });
	await new Promise((r) => setTimeout(r, 500));
	await page.screenshot({ path: join(OUT, `${shot.name}.png`) });
	await context.close();
	console.log(`captured ${shot.name}`);
}

await browser.close();
