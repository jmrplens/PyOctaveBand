// Layout check for the landing page's numbered and ruled headings.
//
// The clause number beside each landing block used to be an absolutely
// positioned `::before` whose gutter was reserved by a padding inside a
// `min-width: 62rem` media query. Below that width, at 200 % browser zoom and
// at any width once the reader's default text size was enlarged, no space was
// reserved and the number was painted on top of the first letter of the
// heading ("01" over the Q of "Qué es"). The number is now a real box in a
// wrapping flex line, so the layout itself reserves its space or drops it onto
// its own line above the title.
//
// This script proves that at every width the site is expected to be read at,
// at 200 % and 400 % zoom (WCAG 1.4.4 / 1.4.10) and with the default text size
// enlarged to 24 px, in both locales. It asserts:
//
//   1. no clause number overlaps any text of its own block,
//   2. no step number overlaps its step heading,
//   3. the list markers fit inside the space their list item reserves,
//   4. the document never scrolls sideways.
//
// It drives its own headless Chrome (puppeteer, already in the tree because
// pa11y-ci depends on it) against a running dev server or preview.
//
// Usage:
//   node scripts/check-home-headings.mjs [--base http://localhost:4321]
//                                        [--shots <dir>] [filter...]
//
// A filter is matched against "<locale> <case>", so `--shots dir 390 1440`
// writes crops for those two widths only. Filters never narrow the audit
// itself: every case is always measured.
import { readdirSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const siteDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(import.meta.url);
const store = join(siteDir, 'node_modules', '.pnpm');
const pkg = readdirSync(store).find((d) => /^puppeteer@/.test(d));
if (!pkg) throw new Error('puppeteer not found in the pnpm store; run pnpm install');
const puppeteer = require(join(store, pkg, 'node_modules', 'puppeteer'));

const argv = process.argv.slice(2);
const arg = (name, fallback) => {
	const i = argv.indexOf(name);
	return i === -1 ? fallback : argv[i + 1];
};
const BASE = arg('--base', 'http://localhost:4321');
const SHOTS = arg('--shots', null);
const values = new Set([BASE, SHOTS].filter(Boolean));
const FILTERS = argv.filter((a) => !a.startsWith('--') && !values.has(a));

// Width sweep, plus the two zoom levels and the enlarged-text cases. A zoom
// level is a viewport: 200 % zoom of a 1440x900 window is a 720x450 CSS
// viewport, and 400 % is 360x225. Enlarged text is the browser's default font
// size, which is what `rem` and every rem-based media query resolve against.
const CASES = [];
for (const [lang, path] of [
	['en', '/'],
	['es', '/es/'],
]) {
	for (const w of [320, 360, 390, 430, 500, 600, 768, 900, 991, 992, 1100, 1280, 1440, 1920]) {
		CASES.push({ lang, path, tag: `${w}px`, width: w, height: 900, font: 16 });
	}
	CASES.push({ lang, path, tag: 'zoom 200 %', width: 720, height: 450, font: 16 });
	CASES.push({ lang, path, tag: 'zoom 400 %', width: 360, height: 225, font: 16 });
	CASES.push({ lang, path, tag: 'text 24px @1440', width: 1440, height: 900, font: 24 });
	CASES.push({ lang, path, tag: 'text 24px @900', width: 900, height: 900, font: 24 });
	CASES.push({ lang, path, tag: 'text 24px @390', width: 390, height: 844, font: 24 });
	CASES.push({ lang, path, tag: 'text 32px @1440', width: 1440, height: 900, font: 32 });
}

/** Runs in the page: measures every marker against the text it sits beside. */
function audit() {
	const problems = [];
	const hit = (a, b) =>
		a.right > b.left + 0.5 && a.left < b.right - 0.5 && a.bottom > b.top + 0.5 && a.top < b.bottom - 0.5;
	const rectsOf = (el) => {
		const range = document.createRange();
		range.selectNodeContents(el);
		return [...range.getClientRects()].filter((r) => r.width > 0 && r.height > 0);
	};

	// 1. Clause number vs everything in its block.
	for (const block of document.querySelectorAll('.home .block')) {
		const marker = block.querySelector('.block-n');
		const body = block.querySelector('.block-body');
		const title = (block.querySelector('h2')?.textContent || '?').trim().slice(0, 24);
		if (!marker || !body) {
			problems.push(`block "${title}": no .block-n / .block-body box`);
			continue;
		}
		const m = marker.getBoundingClientRect();
		if (m.width < 1 || m.height < 1) problems.push(`block "${title}": clause number has no box`);
		for (const r of rectsOf(body)) {
			if (hit(m, r)) {
				problems.push(
					`block "${title}": clause number [${Math.round(m.left)},${Math.round(m.top)},${Math.round(m.right)},${Math.round(m.bottom)}]` +
						` overlaps content [${Math.round(r.left)},${Math.round(r.top)},${Math.round(r.right)},${Math.round(r.bottom)}]`,
				);
				break;
			}
		}
		// The number must also stay on screen, whichever state it is in.
		if (m.left < -0.5 || m.right > document.documentElement.clientWidth + 0.5) {
			problems.push(`block "${title}": clause number is outside the viewport`);
		}
	}

	// 2. Step number vs its step heading and body.
	for (const li of document.querySelectorAll('.home ol.steps > li')) {
		const marker = li.querySelector('.step-n');
		if (!marker) continue;
		const m = marker.getBoundingClientRect();
		for (const el of li.querySelectorAll('h3, p, .step-link')) {
			for (const r of rectsOf(el)) {
				if (hit(m, r)) {
					problems.push(`step "${marker.textContent.trim()}": number overlaps ${el.tagName.toLowerCase()}`);
					break;
				}
			}
		}
	}

	// 3. List markers: the '+' / '-' of the two panels is still absolutely
	//    placed, so the item's reserved indent has to be wider than the glyph
	//    at whatever text size is in force.
	for (const li of document.querySelectorAll('.home ul.tick > li')) {
		const cs = getComputedStyle(li, '::before');
		const probe = document.createElement('span');
		probe.textContent = (cs.content || '').replace(/^"|"$/g, '') || '+';
		probe.style.cssText = `position:absolute;visibility:hidden;white-space:pre;font:${cs.font};letter-spacing:${cs.letterSpacing};`;
		document.body.appendChild(probe);
		const glyph = probe.getBoundingClientRect().width;
		probe.remove();
		const indent = parseFloat(getComputedStyle(li).paddingInlineStart);
		if (glyph >= indent) {
			problems.push(`tick marker ${glyph.toFixed(1)}px wide in a ${indent.toFixed(1)}px indent`);
			break;
		}
	}

	// 4. Nothing may push the document sideways.
	const doc = document.documentElement;
	if (doc.scrollWidth > doc.clientWidth + 1) {
		problems.push(`document scrolls sideways: ${doc.scrollWidth}px in ${doc.clientWidth}px`);
	}

	return {
		problems,
		root: parseFloat(getComputedStyle(doc).fontSize),
		// Which of the two states the layout is in, for the report.
		wrapped: (() => {
			const block = document.querySelector('.home .block');
			const m = block?.querySelector('.block-n')?.getBoundingClientRect();
			const h = block?.querySelector('h2')?.getBoundingClientRect();
			return m && h ? m.bottom <= h.top + 0.5 : null;
		})(),
	};
}

if (SHOTS) await mkdir(SHOTS, { recursive: true });

const browser = await puppeteer.launch({
	headless: true,
	args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--font-render-hinting=none'],
});

let failures = 0;
for (const c of CASES) {
	const page = await browser.newPage();
	if (c.font !== 16) {
		const cdp = await page.createCDPSession();
		await cdp.send('Page.setFontSizes', { fontSizes: { standard: c.font, fixed: c.font } });
	}
	await page.setViewport({ width: c.width, height: c.height, deviceScaleFactor: SHOTS ? 2 : 1 });
	await page.goto(`${BASE}/phonometry${c.path}`, { waitUntil: 'domcontentloaded', timeout: 90000 });
	await page.addStyleTag({ content: 'astro-dev-toolbar{display:none !important}' });
	await new Promise((r) => setTimeout(r, 500));

	const { problems, root, wrapped } = await page.evaluate(audit);
	if (problems.length) failures++;
	const state = wrapped === null ? '?' : wrapped ? 'number above title' : 'number in gutter';
	console.log(
		`${problems.length ? 'FAIL' : 'ok  '} ${c.lang}  ${c.tag.padEnd(16)} ${String(c.width).padStart(4)}x${String(c.height).padEnd(4)} root ${String(root).padStart(2)}px  ${state}`,
	);
	for (const p of problems) console.log(`       ${p}`);

	const wanted =
		SHOTS && (!FILTERS.length || FILTERS.some((f) => `${c.lang} ${c.tag}`.includes(f)));
	if (wanted) {
		const clip = await page.evaluate(() => {
			const b = document.querySelector('.home .block');
			if (!b) return null;
			window.scrollTo(0, 0);
			const r = b.getBoundingClientRect();
			return {
				x: Math.max(0, r.left - 4),
				y: Math.max(0, r.top + window.scrollY - 10),
				width: Math.min(r.width + 8, document.documentElement.clientWidth),
				height: 180,
			};
		});
		if (clip) {
			const name = `${c.lang}-${c.tag.replace(/[^a-z0-9]+/gi, '-').replace(/-$/, '')}`;
			await page.screenshot({ path: join(SHOTS, `${name}.png`), clip });
		}
	}
	await page.close();
}

await browser.close();
console.log(`\n${CASES.length} layout case(s) measured, ${failures} failing.`);
process.exit(failures ? 1 : 0);
