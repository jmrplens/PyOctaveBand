// Behaviour checks for the API reference disclosures in the sidebar.
//
// The API subtree is progressive disclosure: closed on a guide page, opened
// one level at a time by the reader, and opened down to the current page when
// the reader is inside the API. Everything the checks below assert is either
// server-rendered state or the result of a real click or key press, because
// the whole point of the design is that the browser does not have to correct
// anything after first paint.
//
// Usage: node scripts/check-api-sidebar.mjs [--base http://localhost:4322]
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
const BASE = baseIdx === -1 ? 'http://localhost:4322' : process.argv[baseIdx + 1];

const GUIDE = '/guides/levels/';
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

/** Desktop viewport: the sidebar is a menu below 50rem and is not the target. */
async function open(path) {
	const context = await browser.createBrowserContext();
	const page = await context.newPage();
	await page.setViewport({ width: 1440, height: 900 });
	await page.goto(`${BASE}/phonometry${path}`, { waitUntil: 'networkidle0', timeout: 60000 });
	await page.addStyleTag({ content: 'astro-dev-toolbar{display:none !important}' });
	return { context, page };
}

/** The state of the whole API subtree, as a reader would experience it. */
const readState = () => {
	const sidebar = document.getElementById('starlight__sidebar');
	const group = sidebar.querySelector('.api-group');
	const caret = group.querySelector(':scope > .group-label-row > .api-caret');
	const list = document.getElementById(caret.getAttribute('aria-controls'));
	const branches = [...list.querySelectorAll(':scope > li > button[data-api-toggle]')].map((b) => ({
		label: b.querySelector('.large').textContent.trim(),
		expanded: b.getAttribute('aria-expanded'),
		buttonVisible: !!b.offsetParent,
		listVisible: !!document.getElementById(b.getAttribute('aria-controls'))?.offsetParent,
	}));
	const links = [...group.querySelectorAll('a[href*="/reference/api/"]')];
	return {
		caretExpanded: caret.getAttribute('aria-expanded'),
		caretName: caret.getAttribute('aria-label'),
		listVisible: !!list.offsetParent,
		// Collapsed or not, every API link stays in the document.
		linksInDom: links.length,
		linksVisible: links.filter((a) => a.offsetParent).length,
		openBranches: branches.filter((b) => b.expanded === 'true').map((b) => b.label),
		branchButtonsVisible: branches.filter((b) => b.buttonVisible).length,
		branchListsVisible: branches.filter((b) => b.listVisible).length,
		branchCount: branches.length,
		current: sidebar.querySelector('[aria-current="page"]')?.textContent.trim() ?? null,
		// Every control must name what it controls and be a real button.
		wiredUp: [...sidebar.querySelectorAll('[data-api-toggle]')].every(
			(b) =>
				b.tagName === 'BUTTON' &&
				b.type === 'button' &&
				!!document.getElementById(b.getAttribute('aria-controls')) &&
				(b.getAttribute('aria-label') || b.textContent).trim().length > 0,
		),
		// The guide side of the tree carries no disclosure at all.
		togglesOutsideApi: [...sidebar.querySelectorAll('[data-api-toggle]')].filter(
			(b) => !b.closest('.api-group'),
		).length,
		guideLinksVisible: [...sidebar.querySelectorAll('a[href*="/guides/"]')].filter(
			(a) => a.offsetParent,
		).length,
	};
};

// --- A guide page: closed, and the guides side unchanged ------------------
{
	const { context, page } = await open(GUIDE);
	const closed = await page.evaluate(readState);
	expect('guide page: the API group is collapsed', closed.caretExpanded, 'false');
	expect('guide page: its list is not on screen', closed.listVisible, false);
	// The group's own label link to the API overview is the one thing that
	// stays visible: it is the group, not one of the pages under it.
	expect('guide page: no API page link is visible', closed.linksVisible, 1);
	expect('guide page: no branch button is visible either', closed.branchButtonsVisible, 0);
	expect('guide page: every API link is still in the document', closed.linksInDom > 100, true);
	expect('guide page: no branch is open', closed.openBranches, []);
	expect('every toggle is a button naming the list it controls', closed.wiredUp, true);
	expect('the guides tree carries no toggle', closed.togglesOutsideApi, 0);
	expect('the guides tree is fully expanded', closed.guideLinksVisible > 30, true);

	// Nothing above or beside the group may move when it opens. Positions are
	// read as layout offsets rather than viewport rectangles, because clicking
	// scrolls the sidebar's own scroll container to reach the control.
	const geometry = () =>
		page.evaluate(() => {
			const r = document.querySelector('.main-frame, main').getBoundingClientRect();
			const group = document.querySelector('.api-group');
			return {
				main: [r.x, r.y, r.width, r.height],
				groupTop: group.offsetTop,
				labelHeight: group.querySelector('.group-label-row').offsetHeight,
			};
		});
	const before = await geometry();
	await page.click('.api-group > .group-label-row > .api-caret');
	await new Promise((r) => setTimeout(r, 250));
	expect('opening the group moves nothing else', await geometry(), before);

	const level1 = await page.evaluate(readState);
	expect('one click opens the API group', level1.caretExpanded, 'true');
	expect('its branches are on screen', level1.branchButtonsVisible, level1.branchCount);
	expect('and there are more than ten of them', level1.branchCount > 10, true);
	expect('none of them has opened its own list', level1.branchListsVisible, 0);
	expect('the branches stay closed: no level unfolds by itself', level1.openBranches, []);
	expect('so no API page link is visible yet', level1.linksVisible, 1);

	// Drill one level further.
	await page.click('#sidebar-api-items > li:nth-child(3) > button[data-api-toggle]');
	await new Promise((r) => setTimeout(r, 250));
	const level2 = await page.evaluate(readState);
	expect('a branch opens on click', level2.openBranches, ['Psychoacoustics']);
	expect('and only that branch reveals its pages', level2.linksVisible > 10, true);

	// Keyboard: the disclosure is a button, so Enter and Space operate it.
	await page.focus('.api-group > .group-label-row > .api-caret');
	await page.keyboard.press('Enter');
	await new Promise((r) => setTimeout(r, 200));
	const afterEnter = await page.evaluate(readState);
	expect('Enter on the focused control closes it', afterEnter.caretExpanded, 'false');
	await page.keyboard.press('Space');
	await new Promise((r) => setTimeout(r, 200));
	const afterSpace = await page.evaluate(readState);
	expect('Space opens it again', afterSpace.caretExpanded, 'true');
	expect('and the branch the reader had opened is still open', afterSpace.openBranches, [
		'Psychoacoustics',
	]);

	// The open state is derived from the route, so it does not survive a
	// navigation out of the API. This is the deliberate choice, asserted.
	await page.goto(`${BASE}/phonometry/guides/insulation-lab/`, {
		waitUntil: 'networkidle0',
		timeout: 60000,
	});
	const nextGuide = await page.evaluate(readState);
	expect('another guide page starts closed again', nextGuide.caretExpanded, 'false');
	expect('with no branch remembered', nextGuide.openBranches, []);
	await page.close();
	await context.close();
}

// --- An API page: the chain down to it is open ----------------------------
{
	const { context, page } = await open(API_PAGE);
	const state = await page.evaluate(readState);
	expect('API page: the API group is open', state.caretExpanded, 'true');
	expect('API page: only the branch holding the page is open', state.openBranches, [
		'Psychoacoustics',
	]);
	expect('API page: the current page is marked', state.current, 'sharpness');
	expect('API page: the guides tree is still fully expanded', state.guideLinksVisible > 30, true);
	await page.close();
	await context.close();
}

// --- The Spanish build ----------------------------------------------------
{
	const { context, page } = await open(API_PAGE_ES);
	const state = await page.evaluate(readState);
	expect('ES API page: the chain is open', state.caretExpanded, 'true');
	expect('ES API page: the control is named in Spanish', state.caretName, 'Submenú de Referencia de la API');
	expect('ES API page: only the branch holding the page is open', state.openBranches, [
		'Psicoacústica',
	]);
	await page.close();
	await context.close();
}

await browser.close();
console.log(`\n${failures} failing check(s).`);
process.exit(failures ? 1 : 0);
