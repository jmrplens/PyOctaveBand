// Behaviour checks for the page header chips.
//
// Two things that only a browser can answer:
//
//   - The cap. Nine chips is the budget for the whole run, and a tenth slot
//     for the "+N more" link when the page has more than nine. A page under
//     the cap must not show the more chip at all.
//   - The landing. Every chip is a link into the page bibliography, and the
//     sticky chrome above the content is taller on a phone than on a desktop
//     (the header plus the mobile table of contents). An entry chip and the
//     more chip point at different elements, so this is where they can drift:
//     both have to come to rest at the same offset, clear of that chrome.
//
// Usage: node scripts/check-page-chips.mjs [--base http://localhost:4321]
import { readdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const siteDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(import.meta.url);
const store = join(siteDir, 'node_modules', '.pnpm');
const pkg = readdirSync(store).find((d) => /^puppeteer@/.test(d));
const puppeteer = require(join(store, pkg, 'node_modules', 'puppeteer'));

const baseIdx = process.argv.indexOf('--base');
const BASE = baseIdx === -1 ? 'http://localhost:4321' : process.argv[baseIdx + 1];

/** Kept in step with MAX_CHIPS in src/lib/reference-chips.ts. */
const MAX_CHIPS = 9;

const browser = await puppeteer.launch({
	headless: true,
	args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'],
});

let failures = 0;
const expect = (name, actual, wanted) => {
	const ok = JSON.stringify(actual) === JSON.stringify(wanted);
	if (!ok) failures++;
	console.log(
		`${ok ? 'ok  ' : 'FAIL'} ${name}${ok ? '' : `\n       got ${JSON.stringify(actual)}, want ${JSON.stringify(wanted)}`}`,
	);
};

// Pages picked for their bibliographies: past the cap, exactly at it, and
// comfortably under it, in both locales.
const CAP_PAGES = [
	'/guides/underwater-propagation/',
	'/guides/porous-absorbers/',
	'/guides/reverberation-prediction/',
	'/guides/insulation-lab/',
	'/guides/room-acoustics/',
	'/es/guides/underwater-propagation/',
	'/es/guides/insulation-lab/',
];

{
	const page = await browser.newPage();
	await page.setViewport({ width: 1440, height: 900 });
	for (const path of CAP_PAGES) {
		await page.goto(`${BASE}/phonometry${path}`, { waitUntil: 'domcontentloaded', timeout: 60000 });
		const run = await page.evaluate(() => {
			const chips = [...document.querySelectorAll('.page-chips .page-chip')];
			const more = chips.filter((c) => c.classList.contains('chip-more'));
			return { real: chips.length - more.length, more: more.length };
		});
		const wanted = run.more
			? { real: MAX_CHIPS, more: 1, withinCap: true }
			: { real: run.real, more: 0, withinCap: run.real <= MAX_CHIPS };
		expect(`cap: ${path} shows ${run.real} chips and ${run.more} more link`, {
			...run,
			withinCap: run.real <= MAX_CHIPS,
		}, wanted);
	}
	await page.close();
}

/**
 * Click the first chip and the "+N more" chip on a page that overflows the
 * cap, and report where each target came to rest relative to the sticky
 * chrome.
 */
async function landing({ path, viewport, theme, banner }) {
	const context = await browser.createBrowserContext();
	const page = await context.newPage();
	await page.setViewport(viewport);
	await page.evaluateOnNewDocument(
		(t, langs) => {
			try {
				localStorage.setItem('starlight-theme', t);
				// Without a banner scenario the bar has to stay down, and this
				// browser's own language list would otherwise raise it.
				if (!langs) localStorage.setItem('phonometry:lang-dismissed', '1');
			} catch {}
			if (langs) {
				Object.defineProperty(navigator, 'languages', { get: () => langs });
				Object.defineProperty(navigator, 'language', { get: () => langs[0] });
			}
		},
		theme,
		// A language the page is not written in brings the bar up.
		banner ? (path.startsWith('/es/') ? ['en-US', 'en'] : ['es-ES', 'es']) : null,
	);
	await page.goto(`${BASE}/phonometry${path}`, { waitUntil: 'networkidle0', timeout: 90000 });
	await page.addStyleTag({ content: 'astro-dev-toolbar{display:none !important}' });
	const out = await page.evaluate(async (wantBanner) => {
		const wait = (ms) => new Promise((r) => setTimeout(r, ms));
		const bar = document.getElementById('lang-suggest');
		const barShown = !!bar && !bar.hidden;
		const header = document.querySelector('header.header');
		const mobileToc = document.querySelector('mobile-starlight-toc');
		const chrome = Math.max(
			header ? header.getBoundingClientRect().bottom : 0,
			mobileToc ? mobileToc.getBoundingClientRect().bottom : 0,
		);
		const chips = [...document.querySelectorAll('.page-chips .page-chip')];
		const more = chips.find((c) => c.classList.contains('chip-more'));
		const tops = [];
		for (const chip of [chips[0], more]) {
			window.scrollTo(0, 0);
			await wait(120);
			chip.click();
			await wait(600);
			const target = document.getElementById(decodeURIComponent(chip.getAttribute('href').slice(1)));
			const r = target.getBoundingClientRect();
			tops.push({ top: Math.round(r.top), clears: r.top >= chrome - 1, hidden: r.bottom < chrome });
		}
		const [entry, moreTop] = tops;
		return {
			banner: wantBanner ? barShown : !barShown,
			sameLanding: entry.top === moreTop.top,
			entryClears: entry.clears,
			moreClears: moreTop.clears,
		};
	}, !!banner);
	await page.close();
	await context.close();
	return out;
}

const LANDINGS = [
	{ path: '/guides/underwater-propagation/', viewport: { width: 1440, height: 900 }, theme: 'light' },
	{ path: '/guides/underwater-propagation/', viewport: { width: 1440, height: 1400 }, theme: 'dark' },
	{ path: '/guides/underwater-propagation/', viewport: { width: 390, height: 844 }, theme: 'dark' },
	{ path: '/guides/underwater-propagation/', viewport: { width: 390, height: 640 }, theme: 'light' },
	{ path: '/es/guides/underwater-propagation/', viewport: { width: 390, height: 844 }, theme: 'dark' },
	{ path: '/es/guides/underwater-propagation/', viewport: { width: 1440, height: 900 }, theme: 'light' },
	// The language bar is on screen, which is the case that changes how much
	// of the viewport the reader actually has.
	{
		path: '/guides/underwater-propagation/',
		viewport: { width: 390, height: 844 },
		theme: 'dark',
		banner: true,
	},
	{
		path: '/es/guides/underwater-propagation/',
		viewport: { width: 1440, height: 900 },
		theme: 'light',
		banner: true,
	},
];

for (const c of LANDINGS) {
	const label = `landing: ${c.path} ${c.viewport.width}x${c.viewport.height} ${c.theme}${c.banner ? ' with the language bar' : ''}`;
	expect(label, await landing(c), {
		banner: true,
		sameLanding: true,
		entryClears: true,
		moreClears: true,
	});
}

await browser.close();
console.log(`\n${failures} failing check(s).`);
process.exit(failures ? 1 : 0);
