// Screenshot matrix for the experimental redesign.
//
// Drives its own headless Chrome so the captures are reproducible and do not
// depend on an interactive browser session: every shot states its viewport,
// locale, colour direction and theme up front.
//
// Puppeteer is not a declared dependency of the site; it is already in the
// tree because pa11y-ci pulls it in, so the script resolves it out of the
// pnpm store rather than adding a heavyweight devDependency for a throwaway
// branch.
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

/** name, path, viewport, accent, theme, and how far down the page to sit. */
const SHOTS = [
	// Colour directions on a figure-heavy guide.
	['10-guide-instrument-dark', '/guides/filter-banks/', DESKTOP, 'instrument', 'dark', 900],
	['11-guide-instrument-light', '/guides/filter-banks/', DESKTOP, 'instrument', 'light', 900],
	['12-guide-blueprint-dark', '/guides/filter-banks/', DESKTOP, 'blueprint', 'dark', 900],
	['13-guide-blueprint-light', '/guides/filter-banks/', DESKTOP, 'blueprint', 'light', 900],
	['14-guide-graphite-dark', '/guides/filter-banks/', DESKTOP, 'graphite', 'dark', 900],
	['15-guide-graphite-light', '/guides/filter-banks/', DESKTOP, 'graphite', 'light', 900],
	// A matplotlib plot rather than a hand-drawn diagram.
	['16-plot-instrument-dark', '/guides/weighting/', DESKTOP, 'instrument', 'dark', 1500],
	['17-plot-blueprint-dark', '/guides/weighting/', DESKTOP, 'blueprint', 'dark', 1500],
	['18-plot-graphite-dark', '/guides/weighting/', DESKTOP, 'graphite', 'dark', 1500],
	['19-plot-instrument-light', '/guides/weighting/', DESKTOP, 'instrument', 'light', 1500],
	// Front page, both locales, both themes, full page.
	['20-home-instrument-dark', '/', DESKTOP, 'instrument', 'dark', 0, true],
	['21-home-instrument-light', '/', DESKTOP, 'instrument', 'light', 0, true],
	['22-home-blueprint-dark', '/', DESKTOP, 'blueprint', 'dark', 0, true],
	['23-home-graphite-light', '/', DESKTOP, 'graphite', 'light', 0, true],
	['24-home-es-instrument-light', '/es/', DESKTOP, 'instrument', 'light', 0, true],
	['25-home-es-blueprint-dark', '/es/', DESKTOP, 'blueprint', 'dark', 0, true],
	// Narrow widths.
	['30-home-phone-instrument-dark', '/', PHONE, 'instrument', 'dark', 0, true],
	['31-home-phone-graphite-light', '/', PHONE, 'graphite', 'light', 0, true],
	['32-guide-phone-instrument-dark', '/guides/filter-banks/', PHONE, 'instrument', 'dark', 700],
	['33-header-phone-light', '/getting-started/', PHONE, 'instrument', 'light', 0],
	['34-header-phone-dark', '/es/getting-started/', PHONE, 'blueprint', 'dark', 0],
];

await mkdir(OUT, { recursive: true });

const browser = await puppeteer.launch({
	headless: true,
	args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--font-render-hinting=none'],
});

for (const [name, path, viewport, accent, theme, scrollY = 0, fullPage = false] of SHOTS) {
	if (filters.length && !filters.some((f) => name.includes(f))) continue;
	const page = await browser.newPage();
	await page.setViewport({ ...viewport, deviceScaleFactor: 1 });
	// Seed the preferences before the document runs its before-paint script.
	await page.evaluateOnNewDocument(
		(a, t) => {
			try {
				localStorage.setItem('phonometry:accent', a);
				localStorage.setItem('starlight-theme', t);
			} catch {}
		},
		accent,
		theme,
	);
	await page.goto(`${BASE}/phonometry${path}`, { waitUntil: 'networkidle0', timeout: 90000 });
	await page.evaluate((t) => {
		document.documentElement.dataset.theme = t;
	}, theme);
	if (scrollY) {
		await page.evaluate((y) => window.scrollTo(0, y), scrollY);
		await new Promise((r) => setTimeout(r, 400));
	}
	await page.screenshot({ path: join(OUT, `${name}.png`), fullPage });
	await page.close();
	console.log(`captured ${name}`);
}

await browser.close();
