// Where every address this site has ever published now lives.
//
// The site has been reorganised repeatedly: guides were split into focused
// pages, then filed into topic folders, then given subgroups. Each of those
// moves changed the URL of every page it touched, and a moved page leaves its
// old address behind as a 404 unless something is done about it. Almost
// nothing was: `astro.config.mjs` carried eight redirect entries against 386
// addresses that had gone dead, so `/guides/room-acoustics/`,
// `/guides/sound-power/`, `/reference/why-phonometry/` and hundreds more
// answered 404 to anyone arriving from a search result, a bookmark or a
// citation. This project runs IndexNow and asks to be cited, so those are the
// arrivals that matter most.
//
// This module derives the answer instead of asking anyone to remember it. It
// replays the rename history of `site/src/content/docs/` commit by commit,
// carrying every page's identity across each move, so a page that has been
// renamed four times still knows all four of its former addresses. The map it
// produces is written to `src/data/redirects.mjs` by `generate-redirects.mjs`
// and checked against history again by `check-redirects.mjs`, so the next move
// cannot land without its redirects.
//
// Two things the replay cannot know are stated by hand below: the deployment
// cutoff, and where the content of a page that was split or deleted went.
import { execFileSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const siteDir = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const repoDir = join(siteDir, '..');

/** The content root, relative to the repository, as git spells it. */
const CONTENT = 'site/src/content/docs/';

/** The generated module both the generator writes and the check reads. */
export const TARGET = join(siteDir, 'src', 'data', 'redirects.mjs');

/**
 * The first commit that deployed the site.
 *
 * `1114b84b1` added `site/` and `.github/workflows/docs.yml` together, and the
 * Deploy Docs run for it succeeded at 2026-07-05T22:39:10Z. Nothing before it
 * was ever published, so nothing before it can have a dead address; every
 * commit after it that touches `site/**` deploys, which is why the replay can
 * treat each step of the main line as a published state of the site.
 */
export const FIRST_DEPLOY = '1114b84b17ad7607dd20e8222867b4855ac5b109';

/**
 * Pages whose content was split across several new pages, or dropped.
 *
 * A rename has one answer and the replay finds it. A split has several, and
 * git picks whichever fragment kept the most text, which is rarely the one a
 * reader wants: `/guides/psychoacoustics/` scored highest against the loudness
 * page, but a reader who bookmarked the psychoacoustics guide wants the
 * psychoacoustics section, not one page inside it. So a split lands on the
 * overview of the subsection its content became, which is a better answer than
 * any one of the pages it was split into. The four entries the site already
 * carried are kept verbatim; the two below them are the ones it lacked.
 *
 * Keyed by the locale-independent route, so one entry covers `/x/` and
 * `/es/x/` alike.
 */
export const SPLIT_DESTINATIONS = new Map([
	// Split into focused pages, 2026-07-15 (#179).
	['/guides/building-acoustics/', '/buildings/insulation/'],
	['/guides/psychoacoustics/', '/perception/psychoacoustics/'],
	// Split into absorbers, diffusers and the rest, 2026-07-16 and 2026-07-28.
	['/guides/materials/', '/materials/absorbers/'],
	['/guides/surface-scattering/', '/materials/diffusers/'],
	// Merged into the impulsive-sound page, 2026-08-03 (#481).
	[
		'/reference/api/environment/impulse-prominence/',
		'/reference/api/environment/impulsive-sound/',
	],
]);

function git(...args) {
	try {
		return execFileSync('git', ['-C', repoDir, ...args], {
			encoding: 'utf8',
			maxBuffer: 1 << 28,
			// stderr captured rather than inherited, so probing for a ref that may
			// not be in this checkout does not print an error that is not one.
			stdio: ['ignore', 'pipe', 'pipe'],
		});
	} catch (error) {
		error.message = `${error.message}\n${error.stderr ?? ''}`.trim();
		throw error;
	}
}

/**
 * What to tell whoever is looking at an address the replay cannot place.
 *
 * Shared by the generator and the check so the two say the same thing, because
 * whichever of them the reader hits first is the one that has to explain it.
 */
export function unresolvedMessage(unresolved) {
	return (
		`${unresolved.length} published address(es) are no longer served and cannot be traced\n` +
		`     to a page that is. They were split or dropped rather than renamed, so their\n` +
		`     destination is a judgement call: add each to SPLIT_DESTINATIONS in\n` +
		`     scripts/shared/redirects.mjs, pointing at the overview of the subsection its\n` +
		`     content became.\n` +
		unresolved.map((route) => `       ${route}`).join('\n')
	);
}

/** True for a content page (the tree also holds images and data files). */
const isPage = (path) => path.startsWith(CONTENT) && /\.mdx?$/.test(path);

/**
 * The route a content file is published at, without the locale prefix for
 * English and with it for Spanish, exactly as it appears in a URL: leading and
 * trailing slash, `index` collapsed into its directory.
 */
export function routeOf(path) {
	const rest = path
		.slice(CONTENT.length)
		.replace(/\.mdx?$/, '')
		.replace(/(^|\/)index$/, '');
	return `/${rest ? `${rest}/` : ''}`;
}

/** The Spanish address of an English route. */
const inSpanish = (route) => `/es${route}`;

/** Splits a route into its locale and the part both locales share. */
function unlocalise(route) {
	return route.startsWith('/es/')
		? { locale: 'es', shared: route.slice(3) }
		: { locale: 'en', shared: route };
}

/**
 * The commit the main line has reached.
 *
 * The replay has to see the same steps whether it runs on a branch, on the
 * merge commit CI builds for a pull request, or on `main` after the squash
 * merge, or the map would be regenerated differently in each and the freshness
 * check would fail the moment a change landed. So the main line is replayed
 * commit by commit only up to the point this branch left it, and everything
 * after that is applied as the single step the squash merge will make it.
 */
function mainLineTip() {
	for (const ref of ['origin/main', 'main']) {
		try {
			return git('merge-base', 'HEAD', ref).trim();
		} catch {
			/* the ref is not in this checkout: try the next one */
		}
	}
	return git('rev-parse', 'HEAD').trim();
}

/** Every page in a commit's tree, as routes. */
function pagesAt(commit) {
	const out = git('ls-tree', '-r', '--name-only', commit, '--', CONTENT).trim();
	return out ? out.split('\n').filter(isPage) : [];
}

/** The renames, additions and deletions of pages between two commits. */
function changesBetween(from, to) {
	// `-l0` lifts the rename-detection limit: one of these steps moved 272 pages
	// at once, and a git whose diff.renameLimit is set low enough would quietly
	// report those as deletions and additions instead. The map has to come out
	// the same on every machine that derives it, including the CI worker that
	// fails the build when it disagrees with the committed one.
	const out = git('diff', '--name-status', '--find-renames', '-l0', from, to, '--', CONTENT).trim();
	const renames = [];
	const added = [];
	const deleted = [];
	for (const line of out ? out.split('\n') : []) {
		const [status, first, second] = line.split('\t');
		if (status.startsWith('R')) {
			if (isPage(first) && isPage(second)) renames.push([routeOf(first), routeOf(second)]);
		} else if (status === 'A') {
			if (isPage(first)) added.push(routeOf(first));
		} else if (status === 'D') {
			if (isPage(first)) deleted.push(routeOf(first));
		}
	}
	return { renames, added, deleted };
}

/**
 * Replays the history of one locale's tree and returns, for every page, the
 * addresses it has held and the address it holds now (or `null` if it is gone).
 */
function replay(steps) {
	const pages = [];
	/** route -> page, for the pages that exist at the step being applied */
	let live = new Map();

	const create = (route) => {
		const page = { routes: new Set([route]), current: route };
		pages.push(page);
		return page;
	};

	for (const file of pagesAt(steps[0])) {
		const route = routeOf(file);
		live.set(route, create(route));
	}

	for (let i = 1; i < steps.length; i += 1) {
		const { renames, added, deleted } = changesBetween(steps[i - 1], steps[i]);
		const next = new Map(live);
		// Renames first: a page that moves keeps its identity and its history.
		for (const [from, to] of renames) {
			const page = live.get(from);
			if (!page) {
				next.set(to, create(to));
				continue;
			}
			next.delete(from);
			page.routes.add(to);
			page.current = to;
			next.set(to, page);
		}
		for (const route of deleted) {
			const page = live.get(route);
			if (page && page.current === route) {
				page.current = null;
				next.delete(route);
			}
		}
		// Additions last, so an address freed by a rename can be taken by a new
		// page in the same step without the two being confused for one.
		for (const route of added) if (!next.has(route)) next.set(route, create(route));
		live = next;
	}
	return pages;
}

/**
 * Every route the site publishes for a given set of source pages.
 *
 * Starlight serves the default locale at the root and falls back to it for a
 * page the other locale has not translated, so an English-only page such as an
 * API reference page is published at its Spanish address too. Counting source
 * files alone would have missed 47 dead Spanish addresses that never had a
 * Spanish file, every one of which answers 404 today, and would have claimed
 * one address was dead that the fallback still serves.
 */
export function routesFor(sourceRoutes) {
	const routes = new Set();
	for (const route of sourceRoutes) {
		const { locale, shared } = unlocalise(route);
		routes.add(route);
		if (locale === 'en') routes.add(inSpanish(shared));
	}
	return routes;
}

/**
 * The redirect map: every address the site has published and no longer serves,
 * pointing at where its content is now.
 *
 * Returns `{ redirects, unresolved }`. `unresolved` lists dead addresses the
 * replay could not place, which is the signal that a page was split or dropped
 * and `SPLIT_DESTINATIONS` needs an entry: the callers turn it into an error
 * rather than publishing a map with a hole in it.
 */
export function deriveRedirects() {
	const tip = mainLineTip();
	const head = git('rev-parse', 'HEAD').trim();
	const listed = git(
		'rev-list',
		'--first-parent',
		'--reverse',
		`${FIRST_DEPLOY}^..${tip}`,
		'--',
		CONTENT,
	)
		.trim()
		.split('\n')
		.filter(Boolean);
	const steps = head === tip ? listed : [...listed, head];

	const pages = replay(steps);

	// Where each page's former addresses lead, in the shared (locale-stripped)
	// route space, so the English and Spanish trees answer the same way even
	// where only one of them has a source file.
	const movedTo = new Map();
	const everPublished = new Set();
	for (const page of pages) {
		for (const route of page.routes) {
			everPublished.add(route);
			const from = unlocalise(route);
			const to = page.current ? unlocalise(page.current).shared : null;
			if (to && to !== from.shared) movedTo.set(from.shared, to);
		}
	}

	const served = routesFor(pagesAt(head).map(routeOf));
	const published = routesFor(everPublished);
	const dead = [...published].filter((route) => !served.has(route)).sort();

	const redirects = new Map();
	const unresolved = [];
	for (const route of dead) {
		const { locale, shared } = unlocalise(route);
		const destination = SPLIT_DESTINATIONS.has(shared)
			? SPLIT_DESTINATIONS.get(shared)
			: movedTo.get(shared);
		const target = locale === 'es' ? inSpanish(destination) : destination;
		if (destination === undefined || !served.has(target)) {
			unresolved.push(route);
			continue;
		}
		redirects.set(route, target);
	}
	return { redirects, unresolved, served, dead };
}

/**
 * The text of `src/data/redirects.mjs` for a derived map.
 *
 * Built here rather than in the generator so the freshness check compares the
 * committed file against exactly the bytes the generator would write, instead
 * of against a second opinion about how to format them.
 */
export function redirectModuleSource(redirects) {
	const entries = [...redirects.keys()]
		.sort()
		.map((from) => `\t'${from}': '${redirects.get(from)}',`)
		.join('\n');
	return `// Every address this site has published and no longer serves.
//
// GENERATED by scripts/generate-redirects.mjs from the rename history of
// site/src/content/docs/. Do not edit by hand: run \`pnpm run redirects\`.
// \`pnpm run check:redirects\` derives the map again and fails if this file has
// fallen behind, so a page cannot be moved without its old address following
// it. Pages that were split rather than renamed have no mechanical answer and
// are listed in SPLIT_DESTINATIONS in scripts/shared/redirects.mjs, which is
// where a deliberate destination belongs.
//
// The keys are addresses as a visitor typed or bookmarked them, without the
// site's base path, which is what Astro matches on; the values carry the base
// path, which is what Astro emits.
import { basePath } from './site.mjs';

/** Old address -> where its content is published today. ${redirects.size} entries. */
const moved = {
${entries}
};

export const redirects = Object.fromEntries(
\tObject.entries(moved).map(([from, to]) => [from, \`\${basePath}\${to}\`]),
);
`;
}
