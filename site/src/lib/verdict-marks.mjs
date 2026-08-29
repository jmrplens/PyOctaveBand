/**
 * The conformance verdict marks, inlined into the page as an SVG sprite.
 *
 * The same four files GitHub reads over `raw.githubusercontent.com`
 * (`.github/badges/verdict-*.svg`, written by `scripts/conformance_badges.py`),
 * read off disk at build time and turned into `<symbol>` definitions.
 *
 * **Inlined, not fetched, and the reason is arithmetic.** The report draws a
 * mark on every one of 566 rows. Fetched, that is 566 `<img>` elements against
 * one cached file: cheap on the network, but every one of them is a layout box
 * that arrives late, and each carries an `alt` a screen reader announces
 * beside a word the row already states. Copying the file into every row is
 * worse: at 319 bytes for the smallest of the four that is 176 kB of markup.
 * A sprite is neither. Measured on the built page, the geometry costs 1002
 * bytes once and a row spends 149 on the `<svg><use>` that points at it, so
 * 566 rows come to 82 kB with no request and no flash, and the marks are in
 * the HTML that Pagefind indexes and that a reader without JavaScript sees.
 *
 * The fills come across unchanged. They were chosen against five grounds at
 * once - GitHub light, dark and dim, GitHub's table stripe, and this site's
 * own `#17181c` - precisely so that one file is right in every theme, and
 * `tests/test_conformance_badges.py` measures that rather than asserting it.
 * So there is nothing here to re-theme, and re-theming would fork a palette
 * that is currently proved correct in one place.
 *
 * Read from the repository rather than copied into `public/`: a copy is a
 * second thing to keep in step, and `scripts/stage-media.mjs` exists for the
 * figures because those are megabytes, which these are not.
 */
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Locate the committed badge directory by walking up from a starting point.
 *
 * Same shape as `src/data/conformance-stats.mjs`, and for the same reason: a
 * literal relative URL breaks once Astro bundles this module into
 * `dist/.prerender/chunks/`, where the hops count from the chunk instead of
 * from `src/lib/`.
 *
 * @returns {string} Absolute path of `.github/badges`.
 */
function findBadgeDir() {
  const starts = [dirname(fileURLToPath(import.meta.url)), process.cwd()];
  for (const start of starts) {
    let dir = start;
    for (;;) {
      const candidate = join(dir, '.github', 'badges');
      if (existsSync(join(candidate, 'verdict-pass.svg'))) return candidate;
      const parent = dirname(dir);
      if (parent === dir) break;
      dir = parent;
    }
  }
  throw new Error(
    '.github/badges/verdict-pass.svg not found above ' +
      starts.join(' or ') +
      '. Regenerate the indicators with `make conformance`.',
  );
}

/**
 * The verdicts, in the order a legend should read them, paired with the file
 * each one is drawn in. Mirrors `MARKS` in `scripts/conformance/marks.py`,
 * which is the one thing here that cannot be derived: the order is editorial,
 * and the site does not run Python.
 *
 * What the mirror is held to is checked below rather than assumed. A rename
 * would already fail loudly, because the read throws; a *fifth* verdict added
 * on the Python side would not, and the page would quietly draw the
 * not-applicable ring for it for ever.
 */
const FILES = [
  ['pass', 'verdict-pass.svg'],
  ['fail', 'verdict-fail.svg'],
  ['by-design', 'verdict-by-design.svg'],
  ['not-applicable', 'verdict-not-applicable.svg'],
];

const BADGE_DIR = findBadgeDir();

// The directory is the authority on how many marks there are. Failing the
// build with this sentence is the whole point: the alternative is a page that
// renders, looks right, and is wrong about one verdict.
const onDisk = readdirSync(BADGE_DIR)
  .filter((name) => name.startsWith('verdict-') && name.endsWith('.svg'))
  .sort();
const named = FILES.map(([, file]) => file).sort();
if (onDisk.join() !== named.join()) {
  throw new Error(
    `.github/badges holds [${onDisk}] but this module names [${named}]. ` +
      'Add the new verdict to FILES in src/lib/verdict-marks.mjs, and its word ' +
      'to both `verdicts` tables in src/components/Conformance.astro.',
  );
}

/**
 * The drawn geometry of one mark, without its root element or its title.
 *
 * The `<title>` is dropped on purpose. It is the text alternative for a reader
 * who opens the file on its own; inside this page the word beside the mark is
 * the alternative, and a `<title>` in a `<symbol>` would be a second, English,
 * announcement on the Spanish page.
 *
 * @param {string} file  Filename under `.github/badges`.
 * @returns {string} The inner markup.
 */
function geometry(file) {
  const source = readFileSync(join(BADGE_DIR, file), 'utf8');
  const inner = source.match(/<svg\b[^>]*>([\s\S]*)<\/svg>/);
  if (!inner) {
    throw new Error(`${file} is not a single <svg> element; regenerate the badges.`);
  }
  return inner[1].replace(/<title>[\s\S]*?<\/title>/, '').trim();
}

/**
 * Every verdict mark as a `<symbol>`, ready to be dropped into the page once.
 *
 * @type {{verdict: string, id: string, symbol: string}[]}
 */
export const marks = FILES.map(([verdict, file]) => ({
  verdict,
  id: `cv-${verdict}`,
  symbol: `<symbol id="cv-${verdict}" viewBox="0 0 16 16">${geometry(file)}</symbol>`,
}));

/** The whole sprite: the four symbols, concatenated in legend order. */
export const sprite = marks.map((mark) => mark.symbol).join('');

/**
 * The `<symbol>` id a verdict draws with.
 *
 * The fallback is the hollow ring, which is the only honest one: it is the
 * mark that means "nothing was decided here". It cannot be reached by a mark
 * this module has drifted away from, because the check above makes that a
 * build failure; it is there for a verdict the artefact carries and the badge
 * vocabulary does not.
 */
export function markId(verdict) {
  return marks.some((mark) => mark.verdict === verdict)
    ? `cv-${verdict}`
    : 'cv-not-applicable';
}
