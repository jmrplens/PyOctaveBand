/**
 * The project's own citation, built once at build time from the files that
 * already hold the answer.
 *
 * `CITATION.cff` in the repository root is the authoritative metadata: it is
 * what GitHub's "Cite this repository" widget renders and what Zenodo reads
 * when a release is archived. So the author, the title, the DOI, the URL and
 * the licence are read from it rather than restated here, and the version
 * cited on the site is read from the root `VERSION` file, which is the single
 * source astro.config already uses and the file whose change drives a
 * release: `CITATION.cff` also carries `version` and `date-released` now (a
 * JOSS review checklist reads them directly from the file), but this module
 * still prefers `VERSION` so a forgotten CFF bump on release day cannot stamp
 * the site with a stale version.
 *
 * That leaves the year, which `VERSION` does not carry. The release date of
 * the version in `VERSION` is already written down once, in the
 * `CHANGELOG.md` heading for that version, so that is where the year comes
 * from rather than from `CITATION.cff`'s `date-released`.
 */
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

/**
 * Paths are resolved by walking up from the working directory rather than from
 * `import.meta.url`. This module is imported by a component, so Vite bundles it
 * and re-runs it from `dist/` while the static routes are generated, where a
 * URL relative to the source file points at nothing. Looking for the file that
 * marks the repository root works from `site/` and from the root alike, and
 * says so plainly if it ever runs somewhere else.
 */
const repositoryRoot = (() => {
  let dir = resolve(process.cwd());
  while (!existsSync(join(dir, 'CITATION.cff'))) {
    const parent = dirname(dir);
    if (parent === dir) {
      throw new Error(`No CITATION.cff in any directory above ${process.cwd()}.`);
    }
    dir = parent;
  }
  return dir;
})();

const root = (name) => join(repositoryRoot, name);

/**
 * A reader for this repository's `CITATION.cff`, not a YAML parser.
 *
 * The site has no YAML dependency and this does not justify adding one: the
 * file is ours, it is forty lines, and its shape is fixed by the CFF schema.
 * What matters is that an unexpected shape stops the build rather than
 * silently yielding `undefined` into a citation someone then pastes into a
 * paper, so every line has to match one of the three forms below and every
 * field this module needs is asserted present afterwards.
 */
function readCitationFile(text) {
  const doc = {};
  let list = null;
  let item = null;

  for (const line of text.split('\n')) {
    if (!line.trim() || /^\s*#/.test(line)) continue;

    // Top level: either `key: value` or `key:` opening a block.
    if (!/^\s/.test(line)) {
      const pair = line.match(/^([\w-]+):\s*(.*)$/);
      if (!pair) throw new Error(`CITATION.cff: cannot read line: ${line}`);
      const [, key, value] = pair;
      list = null;
      item = null;
      if (value === '') {
        list = [];
        doc[key] = list;
      } else {
        doc[key] = unquote(value);
      }
      continue;
    }

    if (!list) throw new Error(`CITATION.cff: indented line outside a block: ${line}`);

    // `- value` or `- key: value`, opening a new list entry.
    const entry = line.match(/^\s*-\s+(.*)$/);
    if (entry) {
      const pair = entry[1].match(/^([\w-]+):\s*(.*)$/);
      if (pair) {
        item = { [pair[1]]: unquote(pair[2]) };
        list.push(item);
      } else {
        item = null;
        list.push(unquote(entry[1]));
      }
      continue;
    }

    // `key: value` continuing the mapping the last `-` opened.
    const pair = line.match(/^\s*([\w-]+):\s*(.*)$/);
    if (!pair || !item) throw new Error(`CITATION.cff: cannot read line: ${line}`);
    item[pair[1]] = unquote(pair[2]);
  }

  return doc;
}

const unquote = (value) => value.trim().replace(/^(["'])(.*)\1$/, '$2');

const cff = readCitationFile(readFileSync(root('CITATION.cff'), 'utf8'));

for (const field of ['title', 'doi', 'url', 'license', 'authors']) {
  if (!cff[field]?.length) {
    throw new Error(`CITATION.cff: missing "${field}", which the citation needs.`);
  }
}

export const version = readFileSync(root('VERSION'), 'utf8').trim();

/**
 * The Zenodo *concept* DOI, which resolves to the newest archived release.
 *
 * The BibTeX entry below is not the only thing that cites it: the site's
 * structured data publishes it as the software's identifier and among its
 * `sameAs` addresses. Those used to spell it out, which made three copies of
 * one string in the repository; they read it from here now, so `CITATION.cff`
 * is the only place it is written down.
 */
export const doi = cff.doi;

/** The same DOI as a resolvable address. */
export const doiUrl = `https://doi.org/${doi}`;

/**
 * The year of the release named by `VERSION`, taken from its changelog heading
 * (`## [3.3.0] - 2026-07-27`). Between releases those always agree, because the
 * commit that bumps `VERSION` is the one that dates the heading. If they ever
 * do not, the newest dated heading is used and the build says so, rather than
 * stamping the citation with whatever day the site happened to be rebuilt.
 */
const releaseYear = (() => {
  const changelog = readFileSync(root('CHANGELOG.md'), 'utf8');
  const headings = [...changelog.matchAll(/^## \[([^\]]+)\] - (\d{4})-\d{2}-\d{2}$/gm)];
  const exact = headings.find(([, released]) => released === version);
  if (exact) return exact[2];
  const [newest] = headings;
  if (!newest) throw new Error('CHANGELOG.md: no dated release heading to take the year from.');
  console.warn(
    `\n⚠ [citation] CHANGELOG.md has no dated heading for version ${version};` +
      ` citing the year of ${newest[1]} (${newest[2]}) instead.\n`,
  );
  return newest[2];
})();

/**
 * Latin-1 letters carrying an accent, written as BibTeX writes them.
 *
 * The author's name is "José", and a citation is pasted into a `.bib` file that
 * some readers still run through classic BibTeX and 8-bit LaTeX, where a raw
 * UTF-8 byte is not a letter. Decomposing the string and turning each combining
 * mark into its LaTeX control sequence keeps the accent instead of dropping it,
 * and `{\'e}` reads back as "é" under every engine.
 */
const LATEX_ACCENTS = {
  '̀': '`',
  '́': "'",
  '̂': '^',
  '̃': '~',
  '̄': '=',
  '̆': 'u ',
  '̇': '.',
  '̈': '"',
  '̊': 'r ',
  '̋': 'H ',
  '̌': 'v ',
  '̧': 'c ',
  '̨': 'k ',
};

/**
 * The characters LaTeX reads as syntax rather than as text, each with the text
 * that spells it.
 *
 * A table walked character by character below, rather than a pattern with a
 * replacement template. What every substitution produces is then written next
 * to the character it stands in for, and the set of characters that get spelled
 * out is written down once instead of once here and once in a character class
 * that has to agree with it.
 *
 * The backslash is deliberately absent and could not be added: `\\` is a line
 * break in LaTeX, not a literal backslash, so no entry here would be correct
 * for one. `toBibtex` refuses it outright instead.
 */
const LATEX_SPECIALS = {
  '&': '\\&',
  '%': '\\%',
  $: '\\$',
  '#': '\\#',
  _: '\\_',
};

function toBibtex(value) {
  // Refused rather than escaped. A backslash opens a BibTeX command, and the
  // accent pass below writes backslashes of its own, so escaping the input
  // would have to happen first and would then be indistinguishable from them.
  // These values come from CITATION.cff, VERSION and the changelog heading,
  // where a backslash is a mistake worth stopping the build for, not input to
  // be sanitised into something plausible.
  if (value.includes('\\')) {
    throw new Error(`A backslash cannot be spelled in BibTeX metadata: ${value}`);
  }
  const accented = value.normalize('NFD').replace(/([A-Za-z])([̀-ͯ])/g, (whole, letter, mark) => {
    const accent = LATEX_ACCENTS[mark];
    if (!accent) throw new Error(`No BibTeX spelling for the accent in: ${whole}`);
    return `{\\${accent}${letter}}`;
  });
  // Every character either has a spelling in the table above or stands for
  // itself, including the backslashes the accents just wrote, which are already
  // the control sequences they were meant to be.
  const escaped = [...accented].map((character) => LATEX_SPECIALS[character] ?? character).join('');
  // Anything left outside printable ASCII would reach a .bib file as a raw byte.
  if (/[^\x20-\x7e]/.test(escaped)) {
    throw new Error(`Value does not survive the trip to BibTeX: ${value}`);
  }
  return escaped;
}

/** "Requena-Plens, Jos{\'e} M." — the family name first, as BibTeX wants it. */
const authors = cff.authors
  .map((author) => toBibtex(`${author['family-names']}, ${author['given-names']}`))
  .join(' and ');

/** "phonometry", the part of the title before its subtitle. */
const projectName = cff.title.split(':')[0].trim();

/** `requenaplens_phonometry`, the key the About page has always published. */
const surname = cff.authors[0]['family-names'].toLowerCase().replace(/[^a-z0-9]/g, '');
const key = `${surname}_${projectName}`;

/**
 * A `@software` entry. That is the type for a citable program, it is what
 * Zenodo's own export produces for this record, and biblatex renders it
 * directly while classic BibTeX styles fall back to `@misc` without losing a
 * field.
 *
 * The DOI is the one on the About page and in the site's structured data: the
 * Zenodo *concept* DOI, which always resolves to the newest archived release.
 * The `version` field is what pins the entry to the release actually run.
 */
export const bibtex = `@software{${key},
  author  = {${authors}},
  title   = {${toBibtex(cff.title)}},
  year    = {${releaseYear}},
  version = {${version}},
  doi     = {${cff.doi}},
  url     = {${cff.url}},
  license = {${toBibtex(cff.license)}}
}`;

/**
 * The About page publishes this citation three times over, in both locales: as
 * a bold DOI link, as an APA reference, and as the BibTeX entry above. A reader
 * who copies one and reads another has to find them saying the same thing, and
 * the page is markdown and cannot import any of it, so the build checks all
 * three rather than only the block that is easiest to compare. A reflowed or
 * edited citation fails here instead of shipping three that disagree about the
 * version, the year or the DOI.
 *
 * What is checked is what this file generates. The APA reference also carries
 * prose that nothing here produces (the bracketed medium, and the label in
 * front of the version, both written in the page's own language), and those are
 * left to the translation-parity check and to review: inventing an APA
 * renderer here to compare against would be a second citation format to keep
 * correct, and it would still not be the one on the page.
 */
const collapse = (text) => text.replace(/\s+/g, ' ').trim();

/** "Requena-Plens, J. M.", one per author, as an APA reference names them. */
const apaNames = cff.authors.map((author) => {
  const initials = author['given-names']
    .split(/\s+/)
    .map((part) => `${part[0]}.`)
    .join(' ');
  return `${author['family-names']}, ${initials}`;
});

for (const page of ['start/about.md', 'es/start/about.md']) {
  const path = `site/src/content/docs/${page}`;
  const source = readFileSync(root(path), 'utf8');
  const fail = (problem) => {
    throw new Error(`${path}: ${problem}`);
  };

  const block = source.match(/```bibtex\n([\s\S]*?)```/);
  if (!block) fail('no bibtex block to check.');
  if (collapse(block[1]) !== collapse(bibtex)) {
    fail(
      'the BibTeX block has drifted from the one built from CITATION.cff. ' +
        `Expected, ignoring line breaks and padding:\n\n${bibtex}\n`,
    );
  }

  // The bold link to the archived record. Both halves of it are the DOI, so
  // one edited in CITATION.cff has to reach both or neither.
  const doiLink = `**[doi.org/${doi}](${doiUrl})**`;
  if (!source.includes(doiLink)) fail(`the link to the archived record should read ${doiLink}`);

  // The APA reference, quoted under its own label.
  const quoted = source.match(/\nAPA:\n+((?:>.*\n)+)/);
  if (!quoted) fail('no APA reference, quoted under an "APA:" label, to check.');
  const apa = collapse(quoted[1].replace(/^>\s?/gm, ''));

  for (const name of apaNames) {
    if (!apa.includes(name)) fail(`the APA reference should name ${name}: ${apa}`);
  }
  if (!apa.includes(`(${releaseYear}).`)) {
    fail(`the APA reference should carry the year ${releaseYear}: ${apa}`);
  }
  if (!apa.includes(doiUrl)) fail(`the APA reference should end at ${doiUrl}: ${apa}`);

  // The italicised title. The English page gives the full title, parenthetical
  // former name included, so that a reader who copies the reference cites the
  // same string as the Zenodo record; the Spanish page still gives the short
  // form. Both are accepted here by requiring the quoted title to *open* the
  // one in `CITATION.cff` rather than equal it.
  const italic = apa.match(/\*([^*]+)\*/);
  if (!italic) fail(`the APA reference should italicise the title: ${apa}`);
  if (!cff.title.startsWith(italic[1])) {
    fail(`the APA title "${italic[1]}" does not open the one in CITATION.cff: ${cff.title}`);
  }

  // The version, behind the label APA puts in front of it in either language.
  const cited = apa.match(/\(Versi[oó]n\s+([^)]+)\)/);
  if (!cited) fail(`the APA reference should give the version in parentheses: ${apa}`);
  if (cited[1] !== version) {
    fail(`the APA reference cites version ${cited[1]}, but VERSION says ${version}.`);
  }
}
