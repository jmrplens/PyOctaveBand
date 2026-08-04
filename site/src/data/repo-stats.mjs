/**
 * The landing-page repository numbers, counted from the tree at build time.
 *
 * The guide, API-page, figure and fiche counts and the release version used to
 * be typed into `home.ts` by hand, and every one of them had fallen behind the
 * repository (the landing said 3.2.0 while VERSION said 3.3.0, and claimed 32
 * fiches while `.github/reports` held 66). Counting them here at build time
 * makes the tree the only place that can be wrong, exactly as
 * `conformance-stats.mjs` already does for the conformance headline.
 *
 * Definitions, so the labels these numbers sit under stay honest:
 *   version   first line of `VERSION`, the released version the tag points at
 *   guides    topic guides under the topic folders of `content/docs/`,
 *             excluding the topic and subgroup overviews (English side; the
 *             translation-parity check keeps Spanish identical)
 *   apiPages  module pages under `content/docs/reference/api/`, excluding the
 *             hub index
 *   figures   distinct figures in `.github/images`: SVG basenames with the
 *             `_dark` / `_es` theme and language suffixes folded away, so one
 *             figure means one drawing shipped in all four variants
 *   fiches    PDF fiches in `.github/reports`, each rendered by `.report()`
 */
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * The repository root, found by walking up until `VERSION` appears. The same
 * walk `conformance-stats.mjs` does, for the same reason: a relative URL from
 * this module breaks once Astro bundles it into `dist/.prerender/chunks/`.
 */
function findRoot() {
  const starts = [dirname(fileURLToPath(import.meta.url)), process.cwd()];
  for (const start of starts) {
    let dir = start;
    for (;;) {
      if (existsSync(join(dir, 'VERSION')) && existsSync(join(dir, '.github', 'images'))) {
        return dir;
      }
      const parent = dirname(dir);
      if (parent === dir) break;
      dir = parent;
    }
  }
  throw new Error('repository root (VERSION) not found above ' + starts.join(' or '));
}

const ROOT = findRoot();

/**
 * `readdirSync`, with the reason it was being read in the failure.
 *
 * A bare ENOENT from here names the path and nothing else, which in a build
 * log looks like a broken build rather than what it is: a directory this
 * module counts has been moved or renamed, and the landing page number that
 * comes out of it has to move with it.
 */
function listing(dir, what) {
  try {
    return readdirSync(dir, { withFileTypes: true });
  } catch (cause) {
    throw new Error(
      `cannot count ${what}: ${dir} is not readable. If the directory moved, ` +
        'update site/src/data/repo-stats.mjs, which counts it for the ' +
        'landing page.',
      { cause },
    );
  }
}

function pagesUnder(dir, what, { skipDirs = [] } = {}) {
  let count = 0;
  for (const entry of listing(dir, what)) {
    if (entry.isDirectory()) {
      if (!skipDirs.includes(entry.name)) {
        count += pagesUnder(join(dir, entry.name), what, { skipDirs });
      }
    } else if (/\.mdx?$/.test(entry.name) && !/^index\.mdx?$/.test(entry.name)) {
      count += 1;
    }
  }
  return count;
}

/** The released version, e.g. "3.3.0". */
export const version = (() => {
  const file = join(ROOT, 'VERSION');
  try {
    return readFileSync(file, 'utf8').trim();
  } catch (cause) {
    throw new Error(
      `cannot read the released version: ${file} is not readable. The landing ` +
        'page states it through site/src/data/repo-stats.mjs.',
      { cause },
    );
  }
})();

//: The topics the guides live under. Reference and the API reference are not
//: guides, and neither is Start, which is the front door to them.
const GUIDE_TOPICS = [
  'aircraft',
  'buildings',
  'devices',
  'environment',
  'materials',
  'perception',
  'signals',
  'simulation',
  'underwater',
  'vibration',
];

/** Topic guides, without the topic and subgroup overviews. */
export const guides = GUIDE_TOPICS.reduce(
  (total, topic) =>
    total +
    pagesUnder(join(ROOT, 'site', 'src', 'content', 'docs', topic), `the ${topic} guides`),
  0,
);

/** API reference pages, one per module. */
export const apiPages = pagesUnder(
  join(ROOT, 'site', 'src', 'content', 'docs', 'reference', 'api'),
  'the API reference pages',
);

/** Distinct committed figures, each shipped light/dark and English/Spanish. */
export const figures = new Set(
  listing(join(ROOT, '.github', 'images'), 'the figures')
    .map((entry) => entry.name)
    .filter((name) => name.endsWith('.svg'))
    .map((name) => name.replace(/\.svg$/, '').replace(/(?:_dark|_es)+$/, '')),
).size;

/** Normative PDF fiches rendered by `.report()`. */
export const fiches = listing(join(ROOT, '.github', 'reports'), 'the PDF fiches').filter(
  (entry) => entry.name.endsWith('.pdf'),
).length;
