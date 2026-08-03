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
// Nothing here counts pages. The expectations are built from
// src/data/sidebar.mjs, the same array astro.config.mjs hands to Starlight, so
// adding a guide or a whole group never fails this check, while a configured
// entry that does not render, renders at the wrong depth, arrives open when it
// was collapsed, or shows through a closed parent still does.
//
// Usage: node scripts/check-sidebar.mjs [--base http://localhost:4321]
import { sidebar } from '../src/data/sidebar.mjs';
import { BASE_PATH, baseFrom, createExpect, launchBrowser } from './shared/audit.mjs';

const BASE = baseFrom();

const GUIDE = '/signal/levels/levels/';
const DEEP_GUIDE = '/perception/psychoacoustics/loudness/';
const SECTION = '/perception/psychoacoustics/';
const API_PAGE = '/reference/api/psychoacoustics/sharpness/';
const API_PAGE_ES = '/es/reference/api/psychoacoustics/sharpness/';

const browser = await launchBrowser();

const { expect, failures } = createExpect();

// --- The configured tree ---------------------------------------------------

/** The label Starlight prints for `item` in `locale`. */
const labelOf = (item, locale) =>
	locale === 'es' ? (item.translations?.es ?? item.label) : item.label;

/** The URL Starlight gives a sidebar `slug` in `locale`. */
const hrefOf = (slug, locale) => `${BASE_PATH}/${locale === 'es' ? 'es/' : ''}${slug}/`;

/** src/data/sidebar.mjs, as the shape the rendered tree has to match. */
function configuredTree(items, locale) {
	return items.map((item) => {
		if (typeof item === 'string') return { kind: 'link', href: hrefOf(item, locale) };
		if (item.items) {
			return {
				kind: 'group',
				label: labelOf(item, locale),
				collapsed: item.collapsed !== false,
				items: configuredTree(item.items, locale),
			};
		}
		// An entry with its own label: the Overview row of a group that has a
		// landing page. The label is part of the expectation, the others are
		// page titles we do not restate here.
		return { kind: 'link', href: hrefOf(item.slug, locale), label: labelOf(item, locale) };
	});
}

/**
 * Fold the reader's position into the tree: a group is open when it holds the
 * current page (Starlight forces the chain open) or when it was configured
 * open, and a row is on screen only when every group above it is open.
 */
function foldFor(nodes, currentHref, visible = true) {
	let holdsCurrent = false;
	for (const node of nodes) {
		node.visible = visible;
		if (node.kind === 'link') {
			node.current = node.href === currentHref;
			holdsCurrent ||= node.current;
			continue;
		}
		const inside = foldFor(node.items, currentHref, false);
		node.open = inside || !node.collapsed;
		holdsCurrent ||= inside;
		if (node.open) foldFor(node.items, currentHref, visible);
	}
	return holdsCurrent;
}

function expectedTree(locale, currentHref) {
	const tree = configuredTree(sidebar, locale);
	foldFor(tree, currentHref);
	return tree;
}

/** Every configured group label, in tree order. */
function groupLabels(nodes, out = []) {
	for (const node of nodes) {
		if (node.kind !== 'group') continue;
		out.push(node.label);
		groupLabels(node.items, out);
	}
	return out;
}

/** Every configured page href, in tree order. */
function linkHrefs(nodes, out = []) {
	for (const node of nodes) {
		if (node.kind === 'link') out.push(node.href);
		else linkHrefs(node.items, out);
	}
	return out;
}

/** The differences between the configured tree and the rendered one. */
function diffTree(wanted, got, trail = 'the sidebar', problems = []) {
	if (wanted.length !== got.length) {
		problems.push(
			`${trail}: ${got.length} rows rendered, ${wanted.length} configured` +
				` (rendered ${JSON.stringify(got.map((n) => n.label ?? n.href))})`,
		);
		return problems;
	}
	for (const [i, want] of wanted.entries()) {
		const has = got[i];
		const where = `${trail} > ${want.label ?? want.href}`;
		if (want.kind !== has.kind) {
			problems.push(`${where}: rendered as a ${has.kind}, configured as a ${want.kind}`);
			continue;
		}
		if (want.visible !== has.visible) {
			problems.push(`${where}: ${has.visible ? 'on screen' : 'hidden'} where it should be the other`);
		}
		if (want.kind === 'link') {
			if (want.href !== has.href) problems.push(`${where}: links to ${has.href}`);
			if (want.label !== undefined && want.label !== has.label) {
				problems.push(`${where}: reads "${has.label}"`);
			}
			if (want.current !== has.current) {
				problems.push(`${where}: ${has.current ? 'carries' : 'does not carry'} the current-page marker`);
			}
			continue;
		}
		if (want.label !== has.label) problems.push(`${where}: the label reads "${has.label}"`);
		if (want.open !== has.open) problems.push(`${where}: arrived ${has.open ? 'open' : 'closed'}`);
		diffTree(want.items, has.items, where, problems);
	}
	return problems;
}

// --- The rendered tree -----------------------------------------------------

/** Runs in the page: the tree as a reader meets it. */
const readTree = () => {
	const sidebarEl = document.getElementById('starlight__sidebar');
	if (!sidebarEl) return { error: 'no #starlight__sidebar in the document' };
	const seen = (el) => el.checkVisibility({ contentVisibilityAuto: true, visibilityProperty: true });
	const walk = (ul) => {
		const out = [];
		for (const li of ul.children) {
			const link = li.querySelector(':scope > a');
			if (link) {
				out.push({
					kind: 'link',
					href: new URL(link.href).pathname,
					label: link.textContent.trim(),
					visible: seen(link),
					current: link.getAttribute('aria-current') === 'page',
				});
				continue;
			}
			const details = li.querySelector(':scope > details');
			if (!details) continue;
			const summary = details.querySelector(':scope > summary');
			out.push({
				kind: 'group',
				label: (summary?.querySelector('.large') ?? summary)?.textContent.trim() ?? '',
				open: details.open,
				visible: summary ? seen(summary) : false,
				items: walk(details.querySelector(':scope > ul')),
			});
		}
		return out;
	};
	const top = sidebarEl.querySelector('ul.top-level');
	if (!top) return { error: 'no ul.top-level inside #starlight__sidebar' };
	return {
		tree: walk(top),
		// Nothing here is a custom widget: every disclosure is a summary inside
		// a details, which makes the keyboard and screen reader behaviour the
		// user agent's problem rather than ours.
		customToggles: sidebarEl.querySelectorAll('button, [role="button"], [aria-expanded]').length,
		currentMarkers: sidebarEl.querySelectorAll('[aria-current="page"]').length,
	};
};

/** The labels of the open groups, for the checks that only care about those. */
function openLabels(nodes, out = []) {
	for (const node of nodes) {
		if (node.kind !== 'group') continue;
		if (node.open) out.push(node.label);
		openLabels(node.items, out);
	}
	return out;
}

/** Desktop by default; `phone` swaps in the drawer, which is opened first. */
async function open(path, { phone = false } = {}) {
	const context = await browser.createBrowserContext();
	const page = await context.newPage();
	await page.setViewport(phone ? { width: 390, height: 844 } : { width: 1440, height: 900 });
	await page.goto(`${BASE}${BASE_PATH}${path}`, { waitUntil: 'networkidle0', timeout: 90000 });
	await page.addStyleTag({ content: 'astro-dev-toolbar{display:none !important}' });
	if (phone) {
		await page.evaluate(() => document.querySelector('starlight-menu-button button')?.click());
		await new Promise((r) => setTimeout(r, 300));
	}
	return { context, page };
}

/**
 * The whole tree assertion for one page: what src/data/sidebar.mjs configures
 * is what renders, at the configured depth, with only the reader's own branch
 * unfolded.
 */
async function checkTree(page, path, where) {
	const locale = path.startsWith('/es/') ? 'es' : 'en';
	const rendered = await page.evaluate(readTree);
	if (rendered.error) {
		expect(`${where}: the sidebar renders`, rendered.error, 'a tree');
		return null;
	}
	const wanted = expectedTree(locale, `${BASE_PATH}${path}`);
	expect(`${where}: the rendered tree is the configured tree`, diffTree(wanted, rendered.tree), []);
	expect(`${where}: no hand-built disclosure anywhere`, rendered.customToggles, 0);
	expect(`${where}: exactly one row carries the current-page marker`, rendered.currentMarkers, 1);
	return rendered.tree;
}

/** The top-level groups the reader's own branch does not contain. */
function foldedGroups(tree) {
	return tree.filter((node) => node.kind === 'group' && !node.open);
}

// --- One type treatment per depth, on the phone and on the desktop ---------

/**
 * The type treatment of every visible row, reduced per depth.
 *
 * A group label is a `<summary>` and a page is an `<a>`, and stock Starlight
 * gives them different weight, ink and (below 50rem) size, which made a nested
 * group look like it was set in another typeface than the page beside it. What
 * a reader depends on is that rows at the same depth read as one list, so this
 * counts the distinct treatments seen at each depth: one is the whole point.
 * The current page is excluded, because it is deliberately louder (a filled
 * accent pill), and closed branches are excluded because they are not on
 * screen.
 */
const readTypography = () => {
	const sidebar = document.getElementById('starlight__sidebar');
	const visible = (el) =>
		el.checkVisibility({ contentVisibilityAuto: true, visibilityProperty: true });
	const byDepth = {};
	const walk = (ul, depth) => {
		for (const li of ul.children) {
			const link = li.querySelector(':scope > a');
			const details = li.querySelector(':scope > details');
			const row = link ?? details?.querySelector(':scope > summary > .group-label > .large');
			if (!row) continue;
			if (visible(row) && row.getAttribute('aria-current') !== 'page') {
				const s = getComputedStyle(row);
				const seen = (byDepth[depth] ??= { treatments: new Set(), kinds: new Set() });
				seen.treatments.add(
					[s.fontWeight, s.fontSize, s.fontFamily, s.letterSpacing, s.textTransform, s.color].join(
						' | ',
					),
				);
				seen.kinds.add(link ? 'page' : 'group');
			}
			const sub = details?.querySelector(':scope > ul');
			if (sub) walk(sub, depth + 1);
		}
	};
	walk(sidebar.querySelector('ul.top-level'), 1);
	return Object.fromEntries(
		Object.entries(byDepth).map(([depth, v]) => [
			depth,
			{ treatments: v.treatments.size, kinds: [...v.kinds].sort().join('+') },
		]),
	);
};

for (const phone of [true, false]) {
	const where = phone ? 'drawer' : 'column';
	// The open branch here is Hearing and perception, which holds an Overview
	// page and three nested groups, so the second level has both kinds of row.
	const { context, page } = await open(DEEP_GUIDE, { phone });
	const type = await page.evaluate(readTypography);
	expect(`${where}: the top-level headings are one treatment`, type[1], {
		treatments: 1,
		kinds: 'group',
	});
	expect(`${where}: second-level groups and pages read the same`, type[2], {
		treatments: 1,
		kinds: 'group+page',
	});
	expect(`${where}: third-level pages read the same`, type[3], {
		treatments: 1,
		kinds: 'page',
	});
	await page.close();
	await context.close();
}

// --- The same inside the API reference, which is the deepest branch --------
{
	const { context, page } = await open(API_PAGE);
	const type = await page.evaluate(readTypography);
	// Second level here is the API's own Overview row next to its category
	// groups; third level is the module pages of the open category.
	expect('API branch: its Overview row reads like its category groups', type[2], {
		treatments: 1,
		kinds: 'group+page',
	});
	expect('API branch: the module rows read the same', type[3], {
		treatments: 1,
		kinds: 'page',
	});
	await page.close();
	await context.close();
}

// --- A guide page: only the reader's own branch is open --------------------
{
	const { context, page } = await open(DEEP_GUIDE);
	await checkTree(page, DEEP_GUIDE, 'a guide page');

	// Two groups this page's own branch does not contain: one to operate from
	// the keyboard, and a page inside the other to navigate to afterwards, so
	// the persistence check cannot pass just because the group it looks at was
	// reopened by the navigation itself.
	const folded = foldedGroups(expectedTree('en', `${BASE_PATH}${DEEP_GUIDE}`));
	const target = folded[0]?.label;
	const elsewhere = linkHrefs(folded[1]?.items ?? [])[0];
	// Both come out of the configured tree, so a restructuring that leaves this
	// page with fewer than two folded siblings has to say so here rather than
	// navigating to ".../undefined" and failing with something unreadable.
	const havePair = expect(
		`${DEEP_GUIDE} has two folded sibling groups to exercise`,
		{ keyboardGroup: typeof target === 'string', pageElsewhere: typeof elsewhere === 'string' },
		{ keyboardGroup: true, pageElsewhere: true },
	);

	// A folded entry sits inside a closed `<details>`, which the browser's own
	// text finder still matches. The previous hand-built disclosure had to be
	// taught this; here it is the user agent's default. The term is taken from
	// the tree itself: a row that is in the document and not on screen, and
	// whose text the page body does not also carry.
	const found = await page.evaluate(() => {
		const sidebar = document.getElementById('starlight__sidebar');
		const whole = document.body.textContent ?? '';
		const occurrences = (text) => whole.split(text).length - 1;
		const hidden = [...sidebar.querySelectorAll('a[href]')].filter(
			(a) => !a.checkVisibility({ contentVisibilityAuto: true, visibilityProperty: true }),
		);
		// The term has to occur exactly once in the whole document, so a match
		// can only have come from inside the closed group.
		const row = hidden.find((a) => {
			const text = a.textContent.trim();
			return text.length >= 8 && occurrences(text) === 1;
		});
		if (!row) return { error: 'no folded row whose text is unique to the document' };
		const term = row.textContent.trim();
		getSelection()?.removeAllRanges();
		return { term, found: window.find(term, false, false, true) };
	});
	expect(`find-in-page still reaches a folded entry ("${found.term ?? found.error}")`, found.found, true);

	if (havePair) {
		// Keyboard: a summary is focusable and Enter operates it.
		const opened = await page.evaluate((label) => {
			const summary = [...document.querySelectorAll('#starlight__sidebar summary')].find((s) =>
				s.textContent.includes(label),
			);
			if (!summary) return false;
			summary.focus();
			return document.activeElement === summary;
		}, target);
		expect(`a closed group label ("${target}") takes focus`, opened, true);
		await page.keyboard.press('Enter');
		await new Promise((r) => setTimeout(r, 200));
		const afterEnter = await page.evaluate(readTree);
		expect(
			'Enter on a focused group label opens it',
			openLabels(afterEnter.tree).includes(target),
			true,
		);

		// The open state is remembered for the session on desktop, by Starlight's
		// own SidebarPersister, so the group the reader opened is still open on
		// the next page.
		await page.goto(`${BASE}${elsewhere}`, { waitUntil: 'networkidle0', timeout: 90000 });
		const nextGuide = await page.evaluate(readTree);
		expect(
			'and it is still open after a navigation',
			openLabels(nextGuide.tree).includes(target),
			true,
		);
	}
	await page.close();
	await context.close();
}

// --- A section landing page: its own group is open and the row is marked ---
{
	const { context, page } = await open(SECTION);
	await checkTree(page, SECTION, 'a section landing page');
	await page.close();
	await context.close();
}

// --- An API page: the chain down to it is open, and nothing else -----------
{
	const { context, page } = await open(API_PAGE);
	await checkTree(page, API_PAGE, 'an API page');
	await page.close();
	await context.close();
}

// --- A guide page far from the API: the API group stays shut ---------------
{
	const { context, page } = await open(GUIDE);
	await checkTree(page, GUIDE, 'a guide page far from the API');
	await page.close();
	await context.close();
}

// --- The Spanish build ----------------------------------------------------
{
	const { context, page } = await open(API_PAGE_ES);
	await checkTree(page, API_PAGE_ES, 'the Spanish API page');
	await page.close();
	await context.close();
}

// Every configured label and href had to match something above; if the tree
// ever renders nothing at all the diff would say so, but state the size of
// what was checked so a silent shrink is visible in the log.
console.log(
	`\nChecked ${groupLabels(configuredTree(sidebar, 'en')).length} configured groups and ` +
		`${linkHrefs(configuredTree(sidebar, 'en')).length} configured pages per locale.`,
);
await browser.close();
console.log(`${failures()} failing check(s).`);
process.exit(failures() ? 1 : 0);
