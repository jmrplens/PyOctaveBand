/**
 * What the library covers, in the one shape the structured data needs.
 *
 * The `SoftwareApplication` node used to carry a hand-written description and
 * an eight-item `featureList`, both written when the project was a fractional
 * octave filter bank and neither updated as it grew into a toolkit spanning
 * building acoustics, psychoacoustics, underwater and aircraft noise. The
 * visible `<meta name="description">` said one thing and the machine-readable
 * node said another, and a model reading the page takes the machine-readable
 * one.
 *
 * So the feature list is derived here from `home.ts`, which already holds the
 * nine areas and the standards actually implemented in each and is what the
 * landing page renders. Adding an area to the landing page now adds it to the
 * structured data, with nothing to keep in step by hand.
 */
import { en } from './home.ts';
import { domains, standards, totalChecks } from './conformance-stats.mjs';

/**
 * One `featureList` entry per documented area: the area, what it does, and the
 * designations implemented there. Schema.org wants short human-readable
 * strings, so this is the area summary plus its standards rather than a dump
 * of every function.
 */
export const featureList = en.coverage.areas.map(
  (area) => `${area.name}: ${area.summary} Standards implemented: ${area.standards.join(', ')}.`,
);

/**
 * The scope sentence, with the conformance counts interpolated so it cannot
 * fall behind the report.
 *
 * `src/content/docs/index.mdx` carries the same sentence as its frontmatter
 * `description`, because Starlight frontmatter is static and cannot import.
 * `scripts/check_conformance_claims.py` fails the build if the counts in that
 * copy drift from the report, which is the part that actually went wrong
 * before.
 */
export const siteDescription =
  'Acoustic measurement toolkit for Python covering sound level metrology, ' +
  'psychoacoustics, rooms and buildings, vibration, environment and transport, ' +
  'underwater acoustics, electroacoustics and wave simulation, with ' +
  `${totalChecks} conformance checks against ${standards} IEC, ISO, EN, ANSI, ` +
  'ECMA, DIN, ITU, EBU, SAE, ICAO and ECAC standards.';

/** The shorter form, for the `SoftwareApplication` node. */
export const softwareDescription =
  'Acoustic measurement, analysis and prediction for Python: sound level ' +
  'metrology, psychoacoustics, room and building acoustics, materials, ' +
  'vibration, environmental and transport noise, underwater acoustics, ' +
  'electroacoustics and wave simulation, with ' +
  `${totalChecks} CI-enforced conformance checks across ${domains} domains ` +
  `and ${standards} standards.`;

/**
 * Search keywords: the nine area names plus every designation implemented,
 * deduplicated. Built from the same source as `featureList` for the same
 * reason.
 */
export const keywords = [
  'acoustics',
  'Python',
  ...en.coverage.areas.map((area) => area.name.toLowerCase()),
  ...new Set(en.coverage.areas.flatMap((area) => area.standards)),
].join(', ');

/** The alt text for the social image, kept in step with the current scope. */
export const socialImageAlt =
  'phonometry: standards-conformant acoustic measurement, analysis and prediction for Python';
