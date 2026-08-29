/**
 * The conformance headline numbers, read from the generated artefact.
 *
 * `docs/conformance.json` is written by `make conformance` and CI fails the
 * build if it drifts from a fresh run, so its `counts` object is the only
 * place in the repository where these integers are authoritative. Every page
 * that quotes them imports from here instead of writing them out, which is
 * what stops a guide from still claiming last quarter's count after the
 * report has moved on.
 *
 * Page *bodies*, that is. Frontmatter is static YAML, parsed before MDX
 * compiles, so the meta descriptions and the JSON-LD blocks declared there
 * cannot import this module however the page is authored. Those, and the
 * plain-markdown mirror under `docs/`, and the `.zenodo.json` description, are
 * rewritten instead by `scripts/check_conformance_claims.py --write`, which
 * `make conformance` runs right after regenerating the report. Between the two
 * mechanisms nothing states the counts by hand.
 *
 * This file used to match a regular expression against the report's headline
 * sentence, and carried an error message naming two other files that parsed
 * the same sentence and had to be kept in step with it. Reading a field is
 * what replaced that.
 */
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Locate `docs/conformance.json` by walking up from a starting directory.
 *
 * A literal `new URL('../../../docs/...', import.meta.url)` breaks once Astro
 * bundles this module into `dist/.prerender/chunks/`, because the relative
 * hops are then counted from the chunk, not from `src/data/`. Walking up until
 * the artefact appears works from the source tree, from the prerender bundle and
 * whichever directory the build was launched in.
 */
function findUp(name) {
  const starts = [dirname(fileURLToPath(import.meta.url)), process.cwd()];
  for (const start of starts) {
    let dir = start;
    for (;;) {
      const candidate = join(dir, 'docs', name);
      if (existsSync(candidate)) return candidate;
      const parent = dirname(dir);
      if (parent === dir) break;
      dir = parent;
    }
  }
  throw new Error(
    `docs/${name} not found above ` +
      starts.join(' or ') +
      '. Regenerate it with `make conformance`.',
  );
}

const ARTIFACT = findUp('conformance.json');

/** The whole conformance document, parsed once per build. */
export const report = JSON.parse(readFileSync(ARTIFACT, 'utf8'));

const counts = report.counts;

/** Checks that pass. Equal to `totalChecks` while the report is green. */
export const passingChecks = counts.passing;
/** Total numerical conformance checks in the report. */
export const totalChecks = counts.checks;
/** Distinct domains the checks are grouped into. */
export const domains = counts.domains;
/**
 * Distinct standards the checks reference, counted as the report has always
 * counted them: by citation group, the citation string up to its first colon
 * or " Annex". This is the number the landing page, both READMEs, `llms.txt`
 * and `.zenodo.json` publish.
 */
export const standards = counts.standards;
/**
 * Distinct normative *documents*, from the split citation. Smaller than
 * `standards`, because seven clauses of one standard are one document here and
 * seven "standards" there.
 */
export const designations = counts.designations;
/** Distinct further cited works: books, articles, reports, datasets. */
export const sources = counts.sources;

/** "427 / 427", for the landing-page stat tile. */
export const checksRatio = `${passingChecks} / ${totalChecks}`;

/**
 * Confirmed defects in published sources, counted from `docs/ERRATA.md`.
 *
 * The registry lives beside the conformance report and is transplanted into
 * the site by `scripts/generate_site_reports.py`, so the source document is
 * the one authority for the count. One `##` heading is one entry; the single
 * `#` is the document title. Counted for the same reason the check counts are
 * read rather than typed: "dozens" was the only vague number left on the
 * page that argues nothing should be taken on trust.
 */
function countErrata() {
  const path = join(dirname(ARTIFACT), 'ERRATA.md');
  if (!existsSync(path)) {
    throw new Error(
      `${path} not found. It is the source of the errata count; regenerate ` +
        'the reports or fix the path.',
    );
  }
  const entries = readFileSync(path, 'utf8').match(/^## .+$/gm);
  return entries ? entries.length : 0;
}

/** Confirmed entries in the errata registry. */
export const errataEntries = countErrata();

export default {
  passingChecks,
  totalChecks,
  domains,
  standards,
  designations,
  sources,
  checksRatio,
  errataEntries,
  report,
};
