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
 * The API sidebar disclosures, which only exist as states a reader clicks
 * into, so each shot names the controls to press before the capture. The
 * viewport is cropped to the sidebar, because that is the whole subject.
 */
const SIDEBAR = [
	// Arriving on a guide page: the API group is closed.
	['30-api-closed-light', '/guides/levels/', 'light', []],
	['31-api-closed-dark', '/guides/levels/', 'dark', []],
	// One click: the branches appear and nothing below them unfolds.
	[
		'32-api-open-light',
		'/guides/levels/',
		'light',
		['.api-group > .group-label-row > .api-caret'],
	],
	['33-api-open-dark', '/guides/levels/', 'dark', ['.api-group > .group-label-row > .api-caret']],
	// A second click on one branch: its pages, and only its pages.
	[
		'34-api-nested-light',
		'/guides/levels/',
		'light',
		[
			'.api-group > .group-label-row > .api-caret',
			'#sidebar-api-items > li:nth-child(3) > button[data-api-toggle]',
		],
	],
	[
		'35-api-nested-dark',
		'/guides/levels/',
		'dark',
		[
			'.api-group > .group-label-row > .api-caret',
			'#sidebar-api-items > li:nth-child(3) > button[data-api-toggle]',
		],
	],
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

for (const [name, path, theme, clicks] of SIDEBAR) {
	if (filters.length && !filters.some((f) => name.includes(f))) continue;
	const page = await browser.newPage();
	await page.setViewport({ ...DESKTOP, deviceScaleFactor: 1 });
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
	for (const selector of clicks) {
		await page.click(selector);
		await new Promise((r) => setTimeout(r, 250));
	}
	// Bring the API group to the top of the sidebar's own scroller, then frame
	// the sidebar pane rather than the whole page.
	const clip = await page.evaluate(() => {
		const scroller = document.getElementById('starlight__sidebar');
		const group = document.querySelector('.api-group');
		scroller.scrollTop +=
			group.getBoundingClientRect().top - scroller.getBoundingClientRect().top - 16;
		const r = scroller.getBoundingClientRect();
		return {
			x: 0,
			y: Math.max(0, Math.floor(r.y)),
			width: Math.ceil(r.right + 8),
			height: Math.ceil(Math.min(r.height, window.innerHeight - Math.max(0, r.y))),
		};
	});
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
