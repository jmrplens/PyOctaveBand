/**
 * Which pages have a published markdown copy, and where it sits.
 *
 * `scripts/emit-page-markdown.mjs` writes one `<route>/index.md` per file in
 * the docs content collection. Two places on the page then point readers at
 * it: the `<link rel="alternate" type="text/markdown">` in Head.astro and the
 * page actions menu. Neither may offer a copy the generator did not write, so
 * the rule that decides is stated once here and imported by all three.
 *
 * The route is the content entry's own id, never the address of the page being
 * viewed. On a multilingual site those differ: Starlight serves the whole
 * English-only API reference under `/es/` as fallback content, and those pages
 * have a route but no file of their own, so no copy is written for them. The
 * copy that belongs to such a page is the one for the entry it is actually
 * rendering, which is the English text the reader has in front of them.
 */

/**
 * The two splash pages are a component layout rather than prose, so there is
 * nothing meaningful to serve as markdown for them and the generator skips
 * them. Their routes are the roots of the two locales.
 */
export const hasMarkdownCopy = (route) => route !== '' && route !== 'es';

/** The address of the copy for a route, under the site's base path. */
export const markdownCopyPath = (route, base = '') => `${base}/${route}/index.md`;
