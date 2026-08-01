/**
 * Which pages have a published markdown copy, and where it sits.
 *
 * `scripts/emit-page-markdown.mjs` writes one `<route>/index.md` per file in
 * the docs content collection. Two places on the page then point readers at
 * it: the `<link rel="alternate" type="text/markdown">` in Head.astro and the
 * page actions menu. Neither may offer a copy the generator did not write, so
 * the rule that decides is stated once here and imported by all three.
 *
 * The route comes from the content entry's own id, never from the address of
 * the page being viewed. On a multilingual site those differ: Starlight serves
 * the whole English-only API reference under `/es/` as fallback content, and
 * those pages have a route but no file of their own, so no copy is written for
 * them. The copy that belongs to such a page is the one for the entry it is
 * actually rendering, which is the English text the reader has in front of
 * them.
 */

/**
 * The route a page is served at, from either a content file's path under
 * `src/content/docs/` or a content entry's id.
 *
 * Both sides need this and each used to spell it out for itself, with the same
 * regular expression Astro's own glob loader uses (`/\/index$/`). That
 * expression cannot match at the collection root, where the path is `index.mdx`
 * with no directory in front of it, so the two sides disagreed there: Starlight
 * hands the English splash page an entry id of `''` while the generator called
 * the same file `index` and wrote a copy of it at `/index/`, a route no page
 * occupies. Anchoring the segment at the start as well fixes both readings, and
 * the function is idempotent, so a route or an id that has already been through
 * it comes back unchanged.
 */
export const routeOf = (idOrPath) =>
  idOrPath
    .replace(/\\/g, '/')
    .replace(/\.mdx?$/, '')
    .replace(/(?:^|\/)index$/, '');

/**
 * The two splash pages are a component layout rather than prose, so there is
 * nothing meaningful to serve as markdown for them and the generator skips
 * them. Their routes are the roots of the two locales.
 */
export const hasMarkdownCopy = (route) => route !== '' && route !== 'es';

/** The address of the copy for a route, under the site's base path. */
export const markdownCopyPath = (route, base = '') => `${base}/${route}/index.md`;
