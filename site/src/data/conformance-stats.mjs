/**
 * The conformance headline numbers, parsed from the generated report.
 *
 * `docs/CONFORMANCE.md` is written by `make conformance` and CI fails the
 * build if it drifts from a fresh run, so its header line is the only place
 * in the repository where these three integers are authoritative. Every page
 * that quotes them imports from here instead of writing them out, which is
 * what stops a guide from still claiming last quarter's count after the
 * report has moved on.
 *
 * Parsing rather than importing a JSON sidecar keeps the report a single
 * artefact: one generator, one file, no second thing to regenerate.
 */
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Locate `docs/CONFORMANCE.md` by walking up from a starting directory.
 *
 * A literal `new URL('../../../docs/...', import.meta.url)` breaks once Astro
 * bundles this module into `dist/.prerender/chunks/`, because the relative
 * hops are then counted from the chunk, not from `src/data/`. Walking up until
 * the report appears works from the source tree, from the prerender bundle and
 * whichever directory the build was launched in.
 */
function findReport() {
  const starts = [dirname(fileURLToPath(import.meta.url)), process.cwd()];
  for (const start of starts) {
    let dir = start;
    for (;;) {
      const candidate = join(dir, 'docs', 'CONFORMANCE.md');
      if (existsSync(candidate)) return candidate;
      const parent = dirname(dir);
      if (parent === dir) break;
      dir = parent;
    }
  }
  throw new Error(
    'docs/CONFORMANCE.md not found above ' +
      starts.join(' or ') +
      '. Regenerate it with `make conformance`.',
  );
}

const REPORT = findReport();

// "**427/427 conformance checks pass** across 53 domains and 278 standards"
const HEADLINE =
  /\*\*(\d+)\s*\/\s*(\d+)\s+conformance checks pass\*\*\s+across\s+(\d+)\s+domains\s+and\s+(\d+)\s+standards/;

function parse() {
  const source = readFileSync(REPORT, 'utf8');
  const match = source.match(HEADLINE);
  if (!match) {
    throw new Error(
      'docs/CONFORMANCE.md: could not find the headline "N/N conformance checks ' +
        'pass across N domains and N standards". The report format changed; update ' +
        'the HEADLINE pattern in site/src/data/conformance-stats.mjs to match.',
    );
  }
  const [, passing, total, domains, standards] = match.map(Number);
  return { passing, total, domains, standards };
}

const stats = parse();

/** Checks that pass. Equal to `total` while the report is green. */
export const passingChecks = stats.passing;
/** Total numerical conformance checks in the report. */
export const totalChecks = stats.total;
/** Distinct domains the checks are grouped into. */
export const domains = stats.domains;
/** Distinct standards the checks reference. */
export const standards = stats.standards;

/** "427 / 427", for the landing-page stat tile. */
export const checksRatio = `${passingChecks} / ${totalChecks}`;

export default { passingChecks, totalChecks, domains, standards, checksRatio };
