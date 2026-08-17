// The redirect map is complete, current, and lands on real pages.
//
// Two page renames went out with no redirect behind them and nobody noticed
// for a month: /reference/why-phonometry/ was live and indexable from
// 2026-07-06 to 2026-08-03 and has answered 404 ever since, and
// /guides/room-acoustics/ has answered 404 since the taxonomy move. They were
// not special. 432 published addresses were in the same state, against eight
// redirect entries in the configuration. Nothing in the build looked: Astro
// does not know an address used to exist, and starlight-links-validator only
// reads the links the pages carry today.
//
// So this reads the history instead. It derives the map again from the renames
// git recorded and compares it with the committed file, which fails the moment
// a page moves without its old address following it; and, when a build is
// present, it checks the map against what was actually written to dist, so a
// redirect cannot point at a page that is not there or at another redirect.
//
// Usage: node scripts/check-redirects.mjs   (or `pnpm run check:redirects`)
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { basePath } from '../src/data/site.mjs';
import { redirects as committed } from '../src/data/redirects.mjs';
import { deriveRedirects, redirectModuleSource, TARGET } from './shared/redirects.mjs';

const siteDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const dist = join(siteDir, 'dist');

let failures = 0;
const fail = (message) => {
	console.error(`FAIL ${message}`);
	failures += 1;
};
const ok = (message) => console.log(`ok   ${message}`);

// --- The committed map still says what the history says ------------------
const { redirects, unresolved } = deriveRedirects();

if (unresolved.length > 0) {
	fail(
		`${unresolved.length} published address(es) are no longer served and cannot be traced\n` +
			`     to a page that is. They were split or dropped rather than renamed, so their\n` +
			`     destination is a judgement call: add each to SPLIT_DESTINATIONS in\n` +
			`     scripts/shared/redirects.mjs, pointing at the overview of the subsection its\n` +
			`     content became.\n` +
			unresolved.map((route) => `       ${route}`).join('\n'),
	);
} else {
	ok(`every address the site has published is either served or redirected`);
}

if (readFileSync(TARGET, 'utf8') === redirectModuleSource(redirects)) {
	ok(`src/data/redirects.mjs matches the history: ${redirects.size} redirects`);
} else {
	const derived = new Set(redirects.keys());
	const present = new Set(Object.keys(committed));
	const missing = [...derived].filter((route) => !present.has(route));
	const stale = [...present].filter((route) => !derived.has(route));
	const detail = [
		missing.length > 0 ? `${missing.length} missing: ${missing.slice(0, 8).join(' ')}` : '',
		stale.length > 0 ? `${stale.length} no longer derived: ${stale.slice(0, 8).join(' ')}` : '',
	]
		.filter(Boolean)
		.join('; ');
	fail(
		`src/data/redirects.mjs has fallen behind the rename history` +
			`${detail ? ` (${detail})` : ''}. Run \`pnpm run redirects\`.`,
	);
}

// --- Every redirect the build emitted lands on a real page ---------------
if (!existsSync(dist)) {
	console.log(
		'note the dist/ checks were skipped: no build to read. Run `pnpm build` first to\n' +
			'     check the redirects against the pages that were actually written.',
	);
} else {
	/** Every route the build wrote, and whether it is a redirect stub. */
	const built = new Map();
	(function walk(dir, route = '/') {
		for (const entry of readdirSync(dir, { withFileTypes: true })) {
			if (entry.isDirectory()) walk(join(dir, entry.name), `${route}${entry.name}/`);
			else if (entry.name === 'index.html') {
				const html = readFileSync(join(dir, entry.name), 'utf8');
				built.set(route, html.includes('http-equiv="refresh"'));
			}
		}
	})(dist);

	const emitted = [...built].filter(([, isStub]) => isStub).map(([route]) => route);
	const expected = new Set(Object.keys(committed));
	const unemitted = [...expected].filter((route) => !built.has(route));
	if (unemitted.length > 0) {
		fail(`${unemitted.length} redirect(s) the build did not write: ${unemitted.slice(0, 8).join(' ')}`);
	} else {
		ok(`the build wrote all ${expected.size} redirects (${emitted.length} stub pages in dist)`);
	}

	// A redirect that lands on a 404, or on another redirect, is the defect
	// this map exists to fix, one hop further along.
	const broken = [];
	const chained = [];
	for (const [from, to] of Object.entries(committed)) {
		const route = to.startsWith(basePath) ? to.slice(basePath.length) : to;
		if (!built.has(route)) broken.push(`${from} -> ${to}`);
		else if (built.get(route)) chained.push(`${from} -> ${to}`);
	}
	if (broken.length > 0) fail(`${broken.length} redirect(s) point at a 404: ${broken.slice(0, 8).join(', ')}`);
	else ok(`every redirect target is a page the build wrote`);
	if (chained.length > 0) {
		fail(`${chained.length} redirect(s) point at another redirect: ${chained.slice(0, 8).join(', ')}`);
	} else {
		ok(`no redirect points at another redirect`);
	}

	// An address cannot be both a page and a redirect. Astro resolves the
	// collision silently in favour of one of them, and which one is not
	// something the map should be relying on.
	const shadowed = [...expected].filter((route) => built.has(route) && !built.get(route));
	if (shadowed.length > 0) {
		fail(`${shadowed.length} redirect(s) shadow a real page: ${shadowed.slice(0, 8).join(' ')}`);
	} else {
		ok(`no redirect stands where a page is published`);
	}
}

console.log(`\n${failures} failing check(s).`);
process.exit(failures ? 1 : 0);
