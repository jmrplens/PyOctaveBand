// Screenshot matrix for the redesign.
//
// Drives its own headless Chrome so the captures are reproducible and do not
// depend on an interactive browser session: every shot states its viewport,
// locale and theme up front.
//
// Puppeteer is not a declared dependency of the site; it is already in the
// tree because pa11y-ci pulls it in, so the script resolves it out of the
// pnpm store rather than adding a heavyweight devDependency for a branch that
// is only here to be looked at.
//
// Usage: node scripts/redesign-shots.mjs [--base http://localhost:4322] [filter...]
import { readdirSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const siteDir = join(here, '..');
const OUT = join(siteDir, '..', 'redesign-shots');

const require = createRequire(import.meta.url);
const store = join(siteDir, 'node_modules', '.pnpm');
const pkg = readdirSync(store).find((d) => /^puppeteer@/.test(d));
if (!pkg) throw new Error('puppeteer not found in the pnpm store; run pnpm install');
const puppeteer = require(join(store, pkg, 'node_modules', 'puppeteer'));

const args = process.argv.slice(2);
const baseIdx = args.indexOf('--base');
const BASE = baseIdx === -1 ? 'http://localhost:4322' : args[baseIdx + 1];
const filters = args.filter((a, i) => !a.startsWith('--') && !(baseIdx !== -1 && i === baseIdx + 1));

const DESKTOP = { width: 1440, height: 900 };
const PHONE = { width: 390, height: 844 };

/** name, path, viewport, theme, and how far down the page to sit. */
const SHOTS = [
	// The palette on a figure-heavy guide: the hand-authored diagram case,
	// whose plate is #0d1117.
	['10-guide-dark', '/guides/filter-banks/', DESKTOP, 'dark', 900],
	['11-guide-light', '/guides/filter-banks/', DESKTOP, 'light', 900],
	// A matplotlib plot rather than a hand-drawn diagram: the plate is black.
	['16-plot-dark', '/getting-started/', DESKTOP, 'dark', 1750],
	['19-plot-light', '/getting-started/', DESKTOP, 'light', 1750],
	// Front page, both locales, both themes, full page.
	['20-home-dark', '/', DESKTOP, 'dark', 0, true],
	['21-home-light', '/', DESKTOP, 'light', 0, true],
	['24-home-es-light', '/es/', DESKTOP, 'light', 0, true],
	['25-home-es-dark', '/es/', DESKTOP, 'dark', 0, true],
	// Narrow widths.
	['30-home-phone-dark', '/', PHONE, 'dark', 0, true],
	['31-home-phone-light', '/', PHONE, 'light', 0, true],
	['32-guide-phone-dark', '/guides/filter-banks/', PHONE, 'dark', 700],
	['33-header-phone-light', '/getting-started/', PHONE, 'light', 0],
	['34-header-phone-dark', '/es/getting-started/', PHONE, 'dark', 0],
];

/**
 * Shots that need something more than a viewport and a theme: a faked
 * Accept-Language list.
 */
const EXTRA = [
	{
		name: '40-lang-banner-on-en-page',
		path: '/guides/levels/',
		viewport: DESKTOP,
		theme: 'light',
		languages: ['es-ES', 'es'],
	},
	{
		name: '41-lang-banner-on-es-page-phone',
		path: '/es/guides/levels/',
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

for (const [name, path, viewport, theme, scrollY = 0, fullPage = false] of SHOTS) {
	if (filters.length && !filters.some((f) => name.includes(f))) continue;
	const page = await browser.newPage();
	await page.setViewport({ ...viewport, deviceScaleFactor: 1 });
	// Seed the theme before the document runs its before-paint script.
	await page.evaluateOnNewDocument((t) => {
		try {
			localStorage.setItem('starlight-theme', t);
		} catch {}
	}, theme);
	await page.goto(`${BASE}/phonometry${path}`, { waitUntil: 'networkidle0', timeout: 90000 });
	await page.evaluate((t) => {
		document.documentElement.dataset.theme = t;
	}, theme);
	// The dev server injects the Astro toolbar; it is not part of the design.
	await page.addStyleTag({ content: 'astro-dev-toolbar{display:none !important}' });
	if (fullPage) {
		// Walk the page so lazily-loaded figures and fiche previews are decoded
		// before the full-page capture, then go back to the top.
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
	if (scrollY) {
		await page.evaluate((y) => window.scrollTo(0, y), scrollY);
		await new Promise((r) => setTimeout(r, 400));
	}
	await page.screenshot({ path: join(OUT, `${name}.png`), fullPage });
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
