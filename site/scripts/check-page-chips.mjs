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
import { readFileSync } from 'node:fs';

import { BASE_PATH, baseFrom, createExpect, launchBrowser } from './shared/audit.mjs';

const BASE = baseFrom();

/**
 * The cap, read out of the module that enforces it rather than restated here:
 * a copy would let the two drift and the check would then be measuring its own
 * stale idea of the budget.
 */
const chipSource = readFileSync(new URL('../src/lib/reference-chips.ts', import.meta.url), 'utf8');
const capDeclaration = /export const MAX_CHIPS\s*=\s*(\d+)\s*;/.exec(chipSource);
if (!capDeclaration) {
	throw new Error('no `export const MAX_CHIPS = <n>;` in src/lib/reference-chips.ts');
}
const MAX_CHIPS = Number(capDeclaration[1]);

const browser = await launchBrowser();

const { expect, failures } = createExpect({ quiet: true });

// Pages picked for their bibliographies: past the cap, exactly at it, and
// comfortably under it, in both locales.
const CAP_PAGES = [
	'/underwater/underwater-propagation/',
	'/materials/absorbers/porous-absorbers/',
	'/buildings/rooms/reverberation-prediction/',
	'/buildings/insulation/insulation-lab/',
	'/buildings/rooms/room-acoustics/',
	'/es/underwater/underwater-propagation/',
	'/es/buildings/insulation/insulation-lab/',
];

{
	const page = await browser.newPage();
	await page.setViewport({ width: 1440, height: 900 });
	// How many of the pages under test landed on each side of the cap.
	const sides = { overflowing: 0, under: 0 };
	for (const path of CAP_PAGES) {
		await page.goto(`${BASE}${BASE_PATH}${path}`, { waitUntil: 'domcontentloaded', timeout: 60000 });
		const run = await page.evaluate(() => {
			const chips = [...document.querySelectorAll('.page-chips .page-chip')];
			const more = chips.filter((c) => c.classList.contains('chip-more'));
			return { real: chips.length - more.length, more: more.length };
		});
		// Three promises, none of them derived from the run itself:
		//   - the budget is never exceeded,
		//   - there is at most one "+N more" link,
		//   - and that link only appears on a page whose run is full, so a page
		//     under the cap cannot show one and a page over it cannot hide one.
		expect(
			`cap: ${path} shows ${run.real} chips and ${run.more} more link`,
			{
				withinCap: run.real <= MAX_CHIPS,
				atMostOneMore: run.more <= 1,
				moreOnlyWhenFull: run.more === 0 || run.real === MAX_CHIPS,
			},
			{ withinCap: true, atMostOneMore: true, moreOnlyWhenFull: true },
		);
		if (run.more) sides.overflowing++;
		else sides.under++;
	}
	// The pages above were picked to sit on both sides of the cap. If they ever
	// stop doing that the three assertions still pass and mean nothing, so the
	// run has to prove it saw both cases.
	expect(
		`cap: the ${CAP_PAGES.length} pages under test cover both sides of the cap`,
		{ overflowing: sides.overflowing > 0, under: sides.under > 0 },
		{ overflowing: true, under: true },
	);
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
	await page.goto(`${BASE}${BASE_PATH}${path}`, { waitUntil: 'networkidle0', timeout: 90000 });
	// Only the dev server renders this toolbar, so against a preview of `dist`
	// the rule matches nothing. The call itself is not free of consequence: the
	// execution context can be replaced between the navigation and it, and the
	// rejection failed a whole run once for a style rule that had nothing to
	// hide. Losing it costs nothing, so it is allowed to fail.
	await page
		.addStyleTag({ content: 'astro-dev-toolbar{display:none !important}' })
		.catch(() => {});
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
		// This measurement needs a page that overflows the cap. Say so instead
		// of throwing on `undefined.click()` and losing the whole run.
		if (!chips[0] || !more) return { error: 'no entry chip and "+N more" chip to click' };
		const tops = [];
		for (const chip of [chips[0], more]) {
			window.scrollTo(0, 0);
			await wait(120);
			chip.click();
			await wait(600);
			const href = chip.getAttribute('href') || '';
			const target = document.getElementById(decodeURIComponent(href.slice(1)));
			if (!target) return { error: `the chip pointing at ${href} has no target on the page` };
			const r = target.getBoundingClientRect();
			// Unrounded: a target at innerHeight - 0.4 is on screen, and rounding
			// it to innerHeight would fail an audit the reader's eye passes.
			tops.push({ top: r.top, clears: r.top >= chrome - 1, hidden: r.bottom < chrome });
		}
		const [entry, moreTop] = tops;
		return {
			banner: wantBanner ? barShown : !barShown,
			// Not equality, and not a fixed band either. The two chips target
			// different element kinds (a bibliography <li> and the References
			// <h2>), which the engine lands a stable 12 px apart even with
			// identical scroll-margin and scroll-padding; and on a tall
			// viewport the references heading sits so close to the end of the
			// document that the scroll clamps before the alignment position,
			// which is correct behaviour, not a defect. What a reader needs is
			// the target on screen below the sticky chrome, which `clears`
			// asserts, and inside the viewport, asserted here.
			entryOnScreen: tops[0].top < innerHeight,
			moreOnScreen: tops[1].top < innerHeight,
			entryClears: entry.clears,
			moreClears: moreTop.clears,
		};
	}, !!banner);
	await page.close();
	await context.close();
	return out;
}

const LANDINGS = [
	{ path: '/underwater/underwater-propagation/', viewport: { width: 1440, height: 900 }, theme: 'light' },
	{ path: '/underwater/underwater-propagation/', viewport: { width: 1440, height: 1400 }, theme: 'dark' },
	{ path: '/underwater/underwater-propagation/', viewport: { width: 390, height: 844 }, theme: 'dark' },
	{ path: '/underwater/underwater-propagation/', viewport: { width: 390, height: 640 }, theme: 'light' },
	{ path: '/es/underwater/underwater-propagation/', viewport: { width: 390, height: 844 }, theme: 'dark' },
	{ path: '/es/underwater/underwater-propagation/', viewport: { width: 1440, height: 900 }, theme: 'light' },
	// The language bar is on screen, which is the case that changes how much
	// of the viewport the reader actually has.
	{
		path: '/underwater/underwater-propagation/',
		viewport: { width: 390, height: 844 },
		theme: 'dark',
		banner: true,
	},
	{
		path: '/es/underwater/underwater-propagation/',
		viewport: { width: 1440, height: 900 },
		theme: 'light',
		banner: true,
	},
];

for (const c of LANDINGS) {
	const label = `landing: ${c.path} ${c.viewport.width}x${c.viewport.height} ${c.theme}${c.banner ? ' with the language bar' : ''}`;
	expect(label, await landing(c), {
		banner: true,
		entryOnScreen: true,
		moreOnScreen: true,
		entryClears: true,
		moreClears: true,
	});
}

await browser.close();
console.log(`\n${failures()} failing check(s).`);
process.exit(failures() ? 1 : 0);
