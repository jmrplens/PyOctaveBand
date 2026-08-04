// Shared plumbing for the browser-driven audit scripts in this directory.
//
// The four of them (check-contrast excepted, which needs no browser) each used
// to hand-roll the same three things: finding puppeteer, parsing `--base` and
// printing an ok/FAIL line. Finding puppeteer is the one that mattered, since
// three of the four copies indexed into `readdirSync(...).find(...)` without
// checking the result, so a tree without a pnpm virtual store failed with
// `join(undefined)` instead of saying what was wrong.
import { readdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { basePath } from '../../src/data/site.mjs';

const siteDir = join(dirname(fileURLToPath(import.meta.url)), '..', '..');

/** Origin the preview serves on with no arguments (`astro preview`). */
export const DEFAULT_BASE = 'http://localhost:4321';

/** The site's own path prefix, so a check never spells it out itself. */
export const BASE_PATH = basePath;

/**
 * The puppeteer package.
 *
 * It is a declared dev dependency now. It used to be in the tree only because
 * pa11y-ci depends on it, which under pnpm's default layout does not link it
 * into site/node_modules at all: a plain import did not find it, and five
 * checks rested on a package that a pa11y-ci upgrade could have taken away
 * without a word. The store lookup below is kept because it costs nothing and
 * still answers the case where the declared copy is not linked.
 */
export function loadPuppeteer() {
	const require = createRequire(import.meta.url);
	try {
		return require('puppeteer');
	} catch {
		/* not linked at the top level: fall through to the pnpm store */
	}
	const store = join(siteDir, 'node_modules', '.pnpm');
	let entries = [];
	try {
		entries = readdirSync(store);
	} catch {
		/* no virtual store: report it below rather than throwing ENOENT here */
	}
	// Newest last, by version rather than by string, so a tree that still
	// carries an old copy uses the current one (a plain sort would put
	// puppeteer@9 after puppeteer@24).
	const packages = entries
		.filter((d) => /^puppeteer@/.test(d))
		.sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
	const pkg = packages[packages.length - 1];
	if (pkg) {
		try {
			return require(join(store, pkg, 'node_modules', 'puppeteer'));
		} catch (error) {
			throw new Error(`puppeteer is in ${store}/${pkg} but did not load: ${error.message}`);
		}
	}
	throw new Error(
		'puppeteer is not resolvable from site/. It arrives with pa11y-ci: run `pnpm install` in site/ and try again.',
	);
}

const CHROME_ARGS = ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'];

/** A headless browser with the arguments every check here wants. */
export async function launchBrowser({ args = [] } = {}) {
	return loadPuppeteer().launch({ headless: true, args: [...CHROME_ARGS, ...args] });
}

/** `--base <origin>`, defaulting to the preview's own default. */
export function baseFrom(argv = process.argv, fallback = DEFAULT_BASE) {
	const i = argv.indexOf('--base');
	return i === -1 ? fallback : (argv[i + 1] ?? fallback);
}

/**
 * An `expect(name, actual, wanted)` that counts what it rejects.
 *
 * Returns the assertion together with a `failures()` reader, so a script can
 * report a total and exit on it without a module-level mutable.
 */
export function createExpect({ quiet = false } = {}) {
	let failures = 0;
	const expect = (name, actual, wanted) => {
		const ok = JSON.stringify(actual) === JSON.stringify(wanted);
		if (!ok) failures++;
		const detail =
			ok && quiet
				? ''
				: `\n       got ${JSON.stringify(actual)}, want ${JSON.stringify(wanted)}`;
		console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail}`);
		return ok;
	};
	return { expect, failures: () => failures };
}
