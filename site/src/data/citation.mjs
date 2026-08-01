/**
 * The project's own citation, built once at build time from the files that
 * already hold the answer.
 *
 * `CITATION.cff` in the repository root is the authoritative metadata: it is
 * what GitHub's "Cite this repository" widget renders and what Zenodo reads
 * when a release is archived. So the author, the title, the DOI, the URL and
 * the licence are read from it rather than restated here, and the version is
 * read from the root `VERSION` file, which is the single source astro.config
 * already uses and the file whose change drives a release.
 *
 * That leaves the year, which neither file carries: `CITATION.cff` deliberately
 * omits `version` and `date-released` because Zenodo stamps those per release
 * from the tag, and adding them here would be a second version of the truth to
 * keep in step by hand. The release date of the version in `VERSION` is already
 * written down once, in the `CHANGELOG.md` heading for that version, so that is
 * where the year comes from.
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

function toBibtex(value) {
  const escaped = value
    .normalize('NFD')
    // Before anything else, because the accent pass below writes backslashes
    // of its own and they must not be escaped in turn. A backslash reaching a
    // .bib file unescaped would open a command and swallow what follows.
    .replace(/\\/g, '\\textbackslash{}')
    .replace(/([A-Za-z])([̀-ͯ])/g, (whole, letter, mark) => {
      const accent = LATEX_ACCENTS[mark];
      if (!accent) throw new Error(`No BibTeX spelling for the accent in: ${whole}`);
      return `{\\${accent}${letter}}`;
    })
    .replace(/([&%$#_])/g, '\\$1');
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
 * The About page publishes this same entry in prose, in both locales, and a
 * reader who copies one and reads the other has to find them identical. The
 * page is markdown and cannot import, so instead the build checks it: a
 * reflowed or edited block there fails here rather than shipping two citations
 * that disagree about the version, the year or the DOI.
 */
const collapse = (text) => text.replace(/\s+/g, ' ').trim();

for (const page of ['about.md', 'es/about.md']) {
  const source = readFileSync(root(`site/src/content/docs/${page}`), 'utf8');
  const block = source.match(/```bibtex\n([\s\S]*?)```/);
  if (!block) throw new Error(`site/src/content/docs/${page}: no bibtex block to check.`);
  if (collapse(block[1]) !== collapse(bibtex)) {
    throw new Error(
      `site/src/content/docs/${page}: the BibTeX block has drifted from the one built ` +
        `from CITATION.cff. Expected, ignoring line breaks and padding:\n\n${bibtex}\n`,
    );
  }
}
