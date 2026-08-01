import { defineRouteMiddleware } from '@astrojs/starlight/route-data';
import { getCollection } from 'astro:content';
import { hasMarkdownCopy, markdownCopyPath, routeOf } from './lib/page-markdown.mjs';

const BASE = (import.meta.env.BASE_URL ?? '/').replace(/\/$/, '');

/**
 * The routes `scripts/emit-page-markdown.mjs` publishes a markdown copy for:
 * one per file in the docs collection, minus the splash pages.
 *
 * Membership is what decides, rather than the address of the page, because the
 * two do not agree everywhere. Starlight gives every English page a route in
 * the other locale and serves the English entry there when no translation
 * exists, which is how the whole generated API reference is reachable under
 * `/es/`; those routes have no file, so no copy is written for them. Astro's
 * own synthesised 404 entry is not in the collection either. Reading the copy
 * off the entry that is actually being rendered therefore hands a reader on a
 * fallback page the markdown of the very text in front of them, and offers
 * nothing at all where nothing was written.
 */
const published = new Set(
  (await getCollection('docs'))
    // Astro reports the collection root's index as `index` while Starlight
    // hands the same page the empty id, so both sides go through the one rule
    // the generator uses (src/lib/page-markdown.mjs) before they are compared.
    .map((entry) => routeOf(entry.id))
    .filter(hasMarkdownCopy),
);

export const onRequest = defineRouteMiddleware(({ locals }) => {
  const { starlightRoute } = locals;
  if (!starlightRoute) return;

  // Where this page's markdown copy is, or undefined if it has none. Read by
  // Head.astro for the `rel="alternate"` link and by PageActions.astro, which
  // renders nothing without it.
  const route = routeOf(starlightRoute.entry?.id ?? '');
  starlightRoute.markdownCopy = published.has(route)
    ? markdownCopyPath(route, BASE)
    : undefined;

  // The API reference is generated from the package docstrings, so its pages
  // have no hand-editable source: an "Edit page" link there would invite web
  // edits that the next regeneration silently discards. The prose that a
  // reader could usefully fix lives in the docstrings, not in these files.
  if (/^(es\/)?reference\/api(\/|$)/.test(starlightRoute.entry?.id ?? '')) {
    starlightRoute.editUrl = undefined;
  }

  // Splash (landing) pages don't render a sidebar, which also removes the
  // mobile menu button. Force the sidebar data on so the hamburger menu is
  // available on mobile; desktop hides the sidebar pane via CSS
  // (src/styles/splash-menu.css) to keep the splash layout clean.
  if (starlightRoute.entry?.data?.template === 'splash') {
    starlightRoute.hasSidebar = true;
  }

  // The unified bibliography (frontmatter `references`, rendered by the
  // MarkdownContent override) is injected after the markdown pipeline, so its
  // heading never reaches the generated table of contents. Append the entry
  // here to keep "On this page" complete; slug and heading id must match
  // src/components/References.astro.
  if (starlightRoute.entry?.data?.references?.length && starlightRoute.toc) {
    starlightRoute.toc.items.push({
      depth: 2,
      slug: 'references',
      text: locals.t('phonometry.references.title'),
      children: [],
    });
  }
});
