// Behaviour checks for the sidebar tree.
//
// The tree is stock Starlight with every group `collapsed: true`, so what is
// worth asserting is not markup we wrote but the guarantees a reader depends
// on: the branch holding the current page is open on arrival with no
// client-side correction, everything else is closed, the folded entries are
// still findable by the browser's own find, the disclosure is keyboard
// operable, and the open state survives a navigation on desktop. Each check
// below is either server-rendered state or the result of a real click, key
// press or navigation.
//
// Usage: node scripts/check-sidebar.mjs [--base http://localhost:4327]
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
const BASE = baseIdx === -1 ? 'http://localhost:4327' : process.argv[baseIdx + 1];

const GUIDE = '/guides/levels/';
const DEEP_GUIDE = '/guides/loudness/';
const SECTION = '/guides/sections/psychoacoustics/';
const API_PAGE = '/reference/api/psychoacoustics/sharpness/';
const API_PAGE_ES = '/es/reference/api/psychoacoustics/sharpness/';

const browser = await puppeteer.launch({
	headless: true,
	args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'],
});

let failures = 0;
const expect = (name, actual, wanted) => {
	const ok = JSON.stringify(actual) === JSON.stringify(wanted);
	if (!ok) failures++;
	console.log(
		`${ok ? 'ok  ' : 'FAIL'} ${name}\n       got ${JSON.stringify(actual)}, want ${JSON.stringify(wanted)}`,
	);
};

/** Desktop viewport: the sidebar is a drawer below 50rem and is not the target. */
async function open(path) {
	const context = await browser.createBrowserContext();
	const page = await context.newPage();
	await page.setViewport({ width: 1440, height: 900 });
	await page.goto(`${BASE}/phonometry${path}`, { waitUntil: 'networkidle0', timeout: 90000 });
	await page.addStyleTag({ content: 'astro-dev-toolbar{display:none !important}' });
	return { context, page };
}

/** The state of the tree, as a reader would experience it. */
const readState = () => {
	const sidebar = document.getElementById('starlight__sidebar');
	const visible = (el) =>
		el.checkVisibility({ contentVisibilityAuto: true, visibilityProperty: true });
	const groups = [...sidebar.querySelectorAll('details')];
	const topLevel = [...sidebar.querySelectorAll('ul.top-level > li > details')];
	const label = (d) => d.querySelector('summary .large')?.textContent.trim() ?? '';
	const links = [...sidebar.querySelectorAll('a[href]')];
	const current = sidebar.querySelector('[aria-current="page"]');
	return {
		groups: groups.length,
		topLevelGroups: topLevel.length,
		openGroups: groups.filter((d) => d.open).map(label),
		// Nothing here is a custom widget: every disclosure is a summary inside
		// a details, which makes the keyboard and screen reader behaviour the
		// user agent's problem rather than ours.
		customToggles: sidebar.querySelectorAll('button, [role="button"], [aria-expanded]').length,
		linksInDom: links.length,
		linksVisible: links.filter(visible).length,
		apiLinksInDom: links.filter((a) => a.href.includes('/reference/api/')).length,
		apiLinksVisible: links.filter((a) => a.href.includes('/reference/api/') && visible(a)).length,
		current: current ? current.textContent.trim() : null,
		currentVisible: current ? visible(current) : false,
		// Every group with a landing page carries it as its first entry.
		overviewRows: links.filter((a) => /^(Overview|Resumen)$/.test(a.textContent.trim())).length,
	};
};

// --- A guide page: only the reader's own branch is open --------------------
{
	const { context, page } = await open(DEEP_GUIDE);
	const state = await page.evaluate(readState);
	expect('every group is a native details', state.groups, 46);
	expect('twelve top-level groups', state.topLevelGroups, 12);
	expect('no hand-built disclosure anywhere', state.customToggles, 0);
	expect('only the branch holding the page is open', state.openGroups, [
		'Hearing and perception',
		'Psychoacoustics',
	]);
	expect('the current page is marked', state.current, 'Loudness');
	expect('and it is on screen without scrolling', state.currentVisible, true);
	expect('every group with a landing page offers an Overview row', state.overviewRows, 24);
	expect('no API page link is visible', state.apiLinksVisible, 0);
	expect('while every one of them stays in the document', state.apiLinksInDom > 100, true);

	// A folded entry sits inside a closed `<details>`, which the browser's own
	// text finder still matches. The previous hand-built disclosure had to be
	// taught this; here it is the user agent's default.
	const found = await page.evaluate(() => {
		getSelection()?.removeAllRanges();
		return window.find('parametric_filters', false, false, true);
	});
	expect('find-in-page still reaches a folded API entry', found, true);

	// Keyboard: a summary is focusable and Enter operates it.
	await page.evaluate(() => {
		[...document.querySelectorAll('#starlight__sidebar summary')]
			.find((s) => s.textContent.includes('Underwater'))
			.focus();
	});
	await page.keyboard.press('Enter');
	await new Promise((r) => setTimeout(r, 200));
	const afterEnter = await page.evaluate(readState);
	expect(
		'Enter on a focused group label opens it',
		afterEnter.openGroups.includes('Underwater acoustics'),
		true,
	);

	// The open state is remembered for the session on desktop, by Starlight's
	// own SidebarPersister, so the group the reader opened is still open on the
	// next page.
	await page.goto(`${BASE}/phonometry/guides/sound-quality/`, {
		waitUntil: 'networkidle0',
		timeout: 90000,
	});
	const nextGuide = await page.evaluate(readState);
	expect(
		'and it is still open after a navigation',
		nextGuide.openGroups.includes('Underwater acoustics'),
		true,
	);
	await page.close();
	await context.close();
}

// --- A section landing page: its own group is open and the row is marked ---
{
	const { context, page } = await open(SECTION);
	const state = await page.evaluate(readState);
	expect('landing page: its group and its parent are open', state.openGroups, [
		'Hearing and perception',
		'Psychoacoustics',
	]);
	expect('landing page: the Overview row carries the marker', state.current, 'Overview');
	expect('landing page: and it is on screen', state.currentVisible, true);
	await page.close();
	await context.close();
}

// --- An API page: the chain down to it is open, and nothing else -----------
{
	const { context, page } = await open(API_PAGE);
	const state = await page.evaluate(readState);
	expect('API page: only the API chain is open', state.openGroups, [
		'API reference',
		'Psychoacoustics',
	]);
	expect('API page: the current page is marked', state.current, 'sharpness');
	expect('API page: no guide branch opened with it', state.linksVisible < 30, true);
	await page.close();
	await context.close();
}

// --- A guide page far from the API: the API group stays shut ---------------
{
	const { context, page } = await open(GUIDE);
	const state = await page.evaluate(readState);
	expect(
		'a guide page leaves the API group closed',
		state.openGroups.includes('API reference'),
		false,
	);
	await page.close();
	await context.close();
}

// --- The Spanish build ----------------------------------------------------
{
	const { context, page } = await open(API_PAGE_ES);
	const state = await page.evaluate(readState);
	expect('ES API page: the chain is open, in Spanish', state.openGroups, [
		'Referencia de la API',
		'Psicoacústica',
	]);
	expect('ES API page: the current page is marked', state.current, 'sharpness');
	await page.close();
	await context.close();
}

await browser.close();
console.log(`\n${failures} failing check(s).`);
process.exit(failures ? 1 : 0);
