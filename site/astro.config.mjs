import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLinksValidator from 'starlight-links-validator';
import starlightImageZoom from 'starlight-image-zoom';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import starlightSidebarTopics from 'starlight-sidebar-topics';

import { topics } from './src/data/topics.mjs';
import { doi, doiUrl } from './src/data/citation.mjs';
import { redirects } from './src/data/redirects.mjs';
import { basePath, siteUrl } from './src/data/site.mjs';
import { isOurMedia, mediaUrl, REMOTE_PREFIXES } from './src/lib/media.mjs';
import { rehypeWrappableMath } from './src/lib/wrappable-math.mjs';
import {
  featureList,
  keywords,
  siteDescription,
  socialImageAlt,
  softwareDescription,
} from './src/data/scope.mjs';

// Rewrites documentation media to this site's own copy.
//
// The components (ThemeImage, Video, ReportPreview) call mediaUrl themselves,
// and every figure on the site now comes from one of them. This is the net
// under anything authored another way: an <img>, <video src> or <a href>
// written straight into the markdown, so nothing can be left hotlinking
// raw.githubusercontent.com however it got there.
//
// The markdown mirror under docs/ is deliberately untouched: GitHub renders it
// with no build step, so its URLs have to stay absolute.
function rehypeLocalMedia() {
  const attrs = { img: 'src', video: 'src', source: 'src', a: 'href' };
  // Literal HTML written inside markdown (the theory pages author their
  // light/dark <img> pairs by hand) never becomes hast elements: remark keeps
  // it as an unparsed `raw` node. Those are rewritten as text, which is why
  // the URL scheme has to be a plain deterministic string and not a hash.
  const rewriteRaw = (value) =>
    REMOTE_PREFIXES.reduce(
      (acc, prefix) => acc.split(prefix).join(`${basePath}/media/`),
      value,
    );
  return (tree) => {
    (function visit(node) {
      if ((node.type === 'raw' || node.type === 'html') && typeof node.value === 'string') {
        node.value = rewriteRaw(node.value);
      }
      if (node.type === 'element') {
        const attr = attrs[node.tagName];
        const value = attr && node.properties?.[attr];
        if (typeof value === 'string' && isOurMedia(value)) {
          node.properties[attr] = mediaUrl(value, basePath);
        }
        // Poster stills sit on their own attribute.
        const poster = node.properties?.poster;
        if (typeof poster === 'string' && isOurMedia(poster)) {
          node.properties.poster = mediaUrl(poster, basePath);
        }
      }
      node.children?.forEach(visit);
    })(tree);
  };
}

// KaTeX renders each formula three ways in one element: the styled HTML a
// reader sees, a MathML tree for assistive technology, and, inside that MathML,
// an <annotation encoding="application/x-tex"> holding the original TeX. Only
// the styled HTML is aria-hidden, so a text extractor that walks the DOM used
// to emit the formula two or three times over, concatenated: "L_eq" came out
// as "LeqL_{eq}Leq". On these pages the affected passages are the standards
// formulas, which are exactly the ones worth quoting. An earlier pass dropped
// just the annotation, but that still left MathML text plus visual text, and
// the doubled serialization is what AI crawlers actually quoted.
//
// So each formula is reduced to ONE textual representation: the MathML block
// is removed and the TeX it carried becomes an aria-label on the .katex root
// (with role="math"), the same pattern GitHub uses for rendered math. Text
// extractors now see the visual layer once; screen readers announce the TeX
// source, which trades MathML navigation for an unambiguous single reading.
function rehypeSingleTextMath() {
  const hasClass = (node, name) =>
    node.type === 'element' &&
    Array.isArray(node.properties?.className) &&
    node.properties.className.includes(name);

  const findTex = (node) => {
    if (node.type === 'element' && node.tagName === 'annotation') {
      return node.children?.map((c) => c.value ?? '').join('') || undefined;
    }
    for (const child of node.children ?? []) {
      const found = findTex(child);
      if (found) return found;
    }
    return undefined;
  };

  return (tree) => {
    (function visit(node) {
      if (hasClass(node, 'katex')) {
        const mathml = node.children?.find((c) => hasClass(c, 'katex-mathml'));
        if (mathml) {
          const tex = findTex(mathml);
          node.children = node.children.filter((c) => c !== mathml);
          node.properties.role = 'math';
          if (tex) node.properties.ariaLabel = tex.trim();
        }
        return; // Nothing below a formula root needs visiting.
      }
      node.children?.forEach(visit);
    })(tree);
  };
}

// Converts deprecated HTML align attributes (emitted by markdown table
// alignment) to CSS text-align, for WCAG2AA compliance (pa11y), and makes
// wide tables keyboard-scrollable: Starlight renders markdown tables as
// scrollable blocks (display:block + overflow:auto) and theme-tables.css
// gives 4+ column tables readable minimum cell widths on phones, so they
// scroll horizontally there. Chrome (127+) and Firefox focus scrollable
// regions without focusable children by default, but Safari/WebKit does
// not, so the scroll container needs an explicit tabindex for WCAG 2.1.1
// (theme-tables.css adds the matching :focus-visible outline).
function rehypeTableAlign() {
  const firstRow = (node) => {
    if (node.type === 'element' && node.tagName === 'tr') return node;
    for (const child of node.children ?? []) {
      const found = firstRow(child);
      if (found) return found;
    }
    return null;
  };
  return (tree) => {
    (function visit(node) {
      if (node.type === 'element' && node.tagName === 'table') {
        const row = firstRow(node);
        const cells = (row?.children ?? []).filter(
          (c) => c.type === 'element' && (c.tagName === 'th' || c.tagName === 'td'),
        ).length;
        if (cells >= 4) {
          node.properties = { ...node.properties, tabIndex: 0 };
        }
      }
      if (
        node.type === 'element' &&
        (node.tagName === 'td' || node.tagName === 'th') &&
        node.properties?.align
      ) {
        const val = node.properties.align;
        const existing = node.properties.style ? `${node.properties.style};` : '';
        node.properties.style = `${existing}text-align:${val}`;
        delete node.properties.align;
      }
      if (node.children) node.children.forEach(visit);
    })(tree);
  };
}

const fullUrl = `${siteUrl}${basePath}`;
const repositoryUrl = 'https://github.com/jmrplens/phonometry';
const authorUrl = 'https://jmrp.io';
const socialImageUrl = `${fullUrl}/og-image.png`;
const authorId = `${authorUrl}/#person`;
const websiteId = `${fullUrl}/#website`;
const softwareId = `${repositoryUrl}#software`;
const sourceCodeId = `${repositoryUrl}#source-code`;
const socialImage = {
  '@type': 'ImageObject',
  url: socialImageUrl,
  width: 1200,
  height: 630,
};

// Single-sourced version: read from the package, never duplicated here.
const version = readFileSync(new URL('../VERSION', import.meta.url), 'utf8').trim();

// Freshness signals for the SoftwareApplication node. `datePublished` is the
// first public release (v1.0.0) and is intentionally fixed. To avoid stamping
// a false "modified today" on every rebuild, `dateModified` tracks the last
// repository change (HEAD commit date), falling back to build time only when
// git history is unavailable.
const datePublished = '2026-01-06';
const dateModified = (() => {
  try {
    return execFileSync('git', ['log', '-1', '--format=%cI'], { encoding: 'utf8' })
      .trim()
      .slice(0, 10);
  } catch {
    return new Date().toISOString().slice(0, 10);
  }
})();

const softwareRequirements = 'Python >= 3.13 with NumPy and SciPy (matplotlib and numba optional).';

// --- Canonical identity --------------------------------------------------
// The `#person` entity is no longer restated here; it is fetched from its
// single source of truth on jmrp.io and spliced into the graph verbatim. Every
// property this site used to hand-copy (jobTitle, description, image, sameAs,
// alternateName) had to be kept in sync by hand across five project sites, and
// it drifted: two ended up publishing values that contradicted the canonical
// node, one an avatar URL that had started returning 404.
//
// Fetched from raw.githubusercontent.com rather than https://jmrp.io on
// purpose: this build runs on a CI runner, and jmrp.io sits behind Cloudflare,
// CrowdSec and a MikroTik bouncer, where a blocked runner IP would silently
// degrade this site to a stale snapshot. GitHub serves the same bytes and is
// already a hard dependency of the build — the checkout comes from it.
const CANONICAL_IDENTITY_URL =
  'https://raw.githubusercontent.com/jmrplens/jmrp.io/main/public/identity/person.jsonld';

// Committed fallback, only reached if the fetch fails — which, given the URL is
// on the same host as the checkout, effectively means GitHub is down and there
// is no build anyway. The warning is deliberately loud so a stale identity
// never ships unnoticed. Refresh with `pnpm run identity:sync`.
const identitySnapshot = JSON.parse(
  readFileSync(new URL('./identity/person.snapshot.json', import.meta.url), 'utf8'),
);

// `@context` is stripped: the document is standalone, but here it becomes one
// node of a graph that already declares the context once.
const { '@context': _identityContext, ...canonicalPerson } = await fetch(
  CANONICAL_IDENTITY_URL,
  { signal: AbortSignal.timeout(10_000) },
)
  .then((response) =>
    response.ok
      ? response.json()
      : Promise.reject(new Error(`HTTP ${response.status}`)),
  )
  .catch((error) => {
    console.warn(
      `\n⚠ [identity] Could not fetch the canonical Person entity (${error.message}).\n` +
        `  Falling back to the committed snapshot — this build may ship a stale identity.\n`,
    );
    return identitySnapshot;
  });

/**
 * Topics specific to this project, kept alongside the canonical expertise list
 * rather than replacing it. `knowsAbout` is multi-valued, so the two sets merge
 * instead of conflicting — which is why these may stay local while the
 * single-valued properties above may not. The full original set is listed, not
 * just the entries the canonical lacks, so this site keeps asserting its own
 * topics even if the canonical list changes.
 */
const projectTopics = [
  {
    '@type': 'Thing',
    name: 'Acoustics',
    '@id': 'http://www.wikidata.org/entity/Q82811',
  },
  {
    '@type': 'Thing',
    name: 'Signal processing',
    '@id': 'http://www.wikidata.org/entity/Q208163',
  },
  {
    '@type': 'Thing',
    name: 'Psychoacoustics',
    '@id': 'http://www.wikidata.org/entity/Q557399',
  },
  {
    '@type': 'Thing',
    name: 'Architectural acoustics',
    '@id': 'http://www.wikidata.org/entity/Q2305951',
  },
  {
    '@type': 'Thing',
    name: 'Noise control',
    '@id': 'http://www.wikidata.org/entity/Q1879301',
  },
  {
    '@type': 'Thing',
    name: 'Vibration',
    '@id': 'http://www.wikidata.org/entity/Q3695508',
  },
  {
    '@type': 'Thing',
    name: 'Underwater acoustics',
    '@id': 'http://www.wikidata.org/entity/Q1292425',
  },
  {
    '@type': 'Thing',
    name: 'Python',
    '@id': 'http://www.wikidata.org/entity/Q28865',
  },
  {
    '@type': 'Thing',
    name: 'Embedded system',
    '@id': 'http://www.wikidata.org/entity/Q193040',
  },
];

/** Reads a `knowsAbout` entry's label, which may be a string or a Thing. */
const topicName = (topic) =>
  typeof topic === 'string' ? topic : (topic.name ?? '');

/**
 * Canonical topics first, then this project's own, de-duplicated by label
 * (case-insensitively). Canonical entries win a collision because they carry a
 * verified Wikidata `@id`.
 */
const personNode = (() => {
  const canonical = canonicalPerson.knowsAbout ?? [];
  const seenNames = new Set(canonical.map((t) => topicName(t).toLowerCase()));
  // De-duplicating on the label alone is not enough. This project lists
  // "Embedded system" where the canonical says "Embedded Systems", and both
  // carry Wikidata Q193040 — so a name comparison let both through and emitted
  // two Thing nodes under one @id with conflicting names. That is the exact
  // contradiction this change exists to remove, reintroduced one level down.
  // The @id is the identity; the label is only how a document spells it.
  const seenIds = new Set(
    canonical
      .map((t) => (typeof t === 'object' ? t['@id'] : undefined))
      .filter(Boolean),
  );
  return {
    ...canonicalPerson,
    knowsAbout: [
      ...canonical,
      ...projectTopics.filter(
        (t) => !seenNames.has(topicName(t).toLowerCase()) && !seenIds.has(t['@id']),
      ),
    ],
  };
})();

const jsonLd = JSON.stringify({
  '@context': 'https://schema.org',
  '@graph': [
    // The canonical Person entity, fetched at build time — see the
    // "Canonical identity" block above. Spliced verbatim so this document
    // and jmrp.io describe the same node with the same values, instead of
    // two hand-maintained copies that drift.
    personNode,
    {
      '@type': 'WebSite',
      '@id': websiteId,
      name: 'phonometry',
      url: `${fullUrl}/`,
      description: siteDescription,
      inLanguage: ['en', 'es'],
      image: socialImage,
      publisher: { '@id': authorId },
      about: { '@id': softwareId },
    },
    {
      '@type': 'SoftwareApplication',
      '@id': softwareId,
      name: 'phonometry',
      softwareVersion: version,
      applicationCategory: 'DeveloperApplication',
      applicationSubCategory: 'Scientific/Engineering',
      operatingSystem: 'Windows, Linux, macOS',
      programmingLanguage: 'Python',
      url: repositoryUrl,
      downloadUrl: 'https://pypi.org/project/phonometry/',
      codeRepository: repositoryUrl,
      image: socialImage,
      screenshot: socialImage,
      license: 'https://opensource.org/licenses/MIT',
      isAccessibleForFree: true,
      identifier: {
        '@type': 'PropertyValue',
        propertyID: 'DOI',
        value: doi,
      },
      datePublished,
      dateModified,
      softwareRequirements,
      // Derived from the nine documented areas in src/data/scope.mjs. The
      // hand-written versions of these three fields described the filter bank
      // this used to be, while the visible meta description on the same page
      // described the toolkit it is now.
      featureList,
      keywords,
      description: softwareDescription,
      // The project shipped as PyOctaveBand until 3.0.0 and most of the
      // third-party references still say so, so the old name is declared
      // rather than left for a model to reconcile on its own.
      alternateName: 'PyOctaveBand',
      offers: {
        '@type': 'Offer',
        price: '0',
        priceCurrency: 'USD',
      },
      author: { '@id': authorId },
      // Same three agents jmrp.io/projects/ emits for this @id, so the merged
      // node states one consistent set instead of a partial one per document.
      creator: { '@id': authorId },
      maintainer: { '@id': authorId },
      sameAs: ['https://pypi.org/project/phonometry/', doiUrl, `${fullUrl}/`],
    },
    {
      '@type': 'SoftwareSourceCode',
      '@id': sourceCodeId,
      name: 'phonometry source code',
      codeRepository: repositoryUrl,
      programmingLanguage: 'Python',
      runtimePlatform: 'Windows, Linux, macOS',
      license: 'https://opensource.org/licenses/MIT',
      isPartOf: { '@id': softwareId },
      author: { '@id': authorId },
      // Same three agents jmrp.io/projects/ emits for this @id, so the merged
      // node states one consistent set instead of a partial one per document.
      creator: { '@id': authorId },
      maintainer: { '@id': authorId },
    },
  ],
});

export default defineConfig({
  site: siteUrl,
  base: basePath,
  // Every address this site has published and no longer serves. The eight
  // entries that used to be written out here covered the four guides that were
  // split, and nothing else: the reorganisations that followed moved 216 pages
  // in each language and left every one of their old addresses answering 404,
  // to search results, bookmarks and citations alike. The map is derived from
  // the rename history of the content tree and committed, so it is reviewable
  // in a diff; src/data/redirects.mjs says how it is kept honest.
  redirects,
  build: {
    // Render prerendered pages in parallel batches (default is 1 at a time).
    // At ~340 pages the static-generation phase is a small slice of the
    // build today; the win grows with the page count and it never hurts.
    // The dominant phase (content sync, ~18 s) cannot be cached while
    // starlight-links-validator is enabled: the plugin clears the content
    // layer cache on every build so its remark pass sees every page and the
    // link map stays complete. That is the validation gate working as
    // designed, not a regression.
    concurrency: 4,
  },
  markdown: {
    remarkPlugins: [remarkMath],
    // rehypeSingleTextMath must run after rehypeKatex: it rewrites the
    // elements KaTeX has just produced.
    rehypePlugins: [
      rehypeKatex,
      rehypeSingleTextMath,
      rehypeWrappableMath,
      rehypeLocalMedia,
      rehypeTableAlign,
    ],
  },
  integrations: [
    starlight({
      title: 'phonometry',
      routeMiddleware: './src/routeData.ts',
      plugins: [
        starlightLinksValidator({
          errorOnRelativeLinks: false,
          errorOnFallbackPages: false,
        }),
        // Full-viewport zoom for every figure. The technical plots carry ten
        // curves and axis labels at the width of the reading column, which is
        // 720 px at most, and the source SVGs are 1200 px and wider: the
        // detail is in the file and the page was the only thing hiding it.
        // Captions are off because this site's alt texts are deliberately
        // long, sentence-length descriptions of what a plot shows, written
        // for a reader who cannot see it; painted over the zoomed figure they
        // would cover the very thing the reader zoomed in to look at.
        starlightImageZoom({ showCaptions: false }),
        // One sidebar per topic. The tree a reader sees is the domain they
        // are in, with its own API branch, instead of every page in the
        // library folded three deep: on a guide the sidebar drops from 222
        // links to about a hundred, and an API page finally shows its own
        // entry on screen. The plugin forbids Starlight's `sidebar` key and
        // owns the `Sidebar` component, which is why there is no override of
        // it here: its own renders the topic list and then defers to the
        // default tree.
        starlightSidebarTopics(topics, {
          // The two splash pages are the site's front doors, not a topic.
          exclude: ['/', '/es/'],
        }),
      ],
      description: siteDescription,
      lastUpdated: true,
      // "Edit page" link in each page's footer. The docs live under site/ in
      // the repository, so the base includes that segment; Starlight appends
      // the page's own path (src/content/docs/...). The generated API tree
      // opts out in src/routeData.ts: those files are docstring-derived and a
      // web edit would be overwritten on the next regeneration.
      editLink: {
        baseUrl: `${repositoryUrl}/edit/main/site/`,
      },
      components: {
        // Per-page structured data (TechArticle / BreadcrumbList) and
        // per-page Twitter card tags, layered on the default head.
        Head: './src/components/Head.astro',
        // Human-visible maintainer block corroborating the Person node.
        Footer: './src/components/Footer.astro',
        // Default frame plus the first-visit language bar, mounted outside
        // the panes so the sidebar cannot paint over it.
        PageFrame: './src/components/PageFrame.astro',
        // Default header plus a mobile-visible language selector.
        Header: './src/components/Header.astro',
        // Default right column plus the page actions, which hand this page's
        // published markdown copy to a clipboard, a tab or a chat.
        PageSidebar: './src/components/PageSidebar.astro',
        // Default banner, shown on every page rather than only where a page
        // asks for one, carrying the pre-release notice in both languages.
        // Starlight's config-level banner is a single string with no locale
        // form, and half this site is Spanish.
        Banner: './src/components/Banner.astro',
        // Default left column with the topic switcher above it. The topics
        // plugin overrides `Sidebar` too, and its own hook spreads our
        // components last, so this wins and its override never runs: the
        // switcher renders the topic list itself, from the route data the
        // plugin publishes for exactly that.
        Sidebar: './src/components/Sidebar.astro',
        // Default article body plus the unified APA-7 references section
        // rendered from the typed frontmatter bibliography.
        MarkdownContent: './src/components/MarkdownContent.astro',
        // Default H1 plus the page header chips run derived from the page's
        // own `references` frontmatter (see TOC-REDESIGN-NOTES.md).
        PageTitle: './src/components/PageTitle.astro',
        LastUpdated: './src/components/LastUpdated.astro',
        // The generated brand lockup (mark plus wordmark) in place of the
        // plain text site title.
        SiteTitle: './src/components/SiteTitle.astro',
        // Previous/next links that name the section each neighbour belongs to.
        Pagination: './src/components/Pagination.astro',
        // Default splash hero plus the mark in its image slot.
        Hero: './src/components/Hero.astro',
      },
      customCss: [
        // The site palette. Unlayered, so its `:root` blocks win over
        // Starlight's `@layer starlight.base` defaults; loaded first so the
        // later sheets can still reference the tokens it defines.
        './src/styles/theme.css',
        './src/styles/typography.css',
        './src/styles/brand.css',
        './src/styles/see-also.css',
        './src/styles/scope.css',
        './src/styles/katex.css',
        './src/styles/containment.css',
        './src/styles/theme-images.css',
        './src/styles/image-zoom.css',
        './src/styles/theme-tables.css',
        './src/styles/splash-menu.css',
        './src/styles/home.css',
        './src/styles/sidebar.css',
        './src/styles/page-chips.css',
      ],
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/jmrplens/phonometry' },
        { icon: 'mastodon', label: 'Mastodon', href: 'https://mstdn.jmrp.io/@jmrplens' },
        { icon: 'linkedin', label: 'LinkedIn', href: 'https://linkedin.com/in/jmrplens' },
      ],
      defaultLocale: 'root',
      locales: {
        root: { label: 'English', lang: 'en' },
        es: { label: 'Español', lang: 'es' },
      },
      head: [
        // GitHub Pages cannot send a Content-Security-Policy header, so the
        // policy travels as a meta tag: everything the site loads is
        // first-party, and this pins that fact. Meta delivery cannot carry
        // frame-ancestors or report-uri (ignored there by spec), and inline
        // scripts/styles stay allowed because a static build cannot mint
        // per-response nonces: Starlight's theme bootstrap and Astro's island
        // hydration are inline by design. 'wasm-unsafe-eval' is for Pagefind,
        // whose search index is compiled WebAssembly fetched from this origin.
        {
          tag: 'meta',
          attrs: {
            'http-equiv': 'Content-Security-Policy',
            content: [
              "default-src 'self'",
              "script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'",
              "style-src 'self' 'unsafe-inline'",
              "img-src 'self' data:",
              "font-src 'self' data:",
              "connect-src 'self'",
              "media-src 'self'",
              "object-src 'none'",
              "base-uri 'self'",
              "form-action 'self'",
            ].join('; '),
          },
        },
        // No preconnect to raw.githubusercontent.com: every figure, animation
        // and fiche is served from this origin now (scripts/stage-media.mjs),
        // so there is no third-party handshake left to warm.
        // GitHub Pages serves no custom response headers, so the referrer
        // policy has to travel as a document-level tag.
        { tag: 'meta', attrs: { name: 'referrer', content: 'strict-origin-when-cross-origin' } },
        // Machine-readable pointer to the LLM-oriented summary. It was only
        // mentioned in a robots.txt comment, which nothing parses.
        {
          tag: 'link',
          attrs: {
            rel: 'alternate',
            type: 'text/plain',
            href: `${basePath}/llms.txt`,
            title: 'llms.txt',
          },
        },
        // Open Graph image. Only the invariant parts are set here: the image
        // itself is per page and comes from src/components/Head.astro, which
        // has the route. Emitting a site-wide og:image here too would leave two
        // in the document, and an Open Graph parser takes the first, so the
        // per-page card would silently lose to this one. No og:image:type
        // either: the per-page cards are JPEG and the fallback is PNG, and one
        // site-wide tag cannot describe both.
        { tag: 'meta', attrs: { property: 'og:image:alt', content: socialImageAlt } },
        { tag: 'meta', attrs: { property: 'og:image:width', content: '1200' } },
        { tag: 'meta', attrs: { property: 'og:image:height', content: '630' } },
        // Twitter card
        { tag: 'meta', attrs: { name: 'twitter:card', content: 'summary_large_image' } },
        { tag: 'meta', attrs: { name: 'twitter:image:alt', content: socialImageAlt } },
        // Author
        { tag: 'meta', attrs: { name: 'author', content: 'José Manuel Requena Plens' } },
        // Theme color
        { tag: 'meta', attrs: { name: 'theme-color', content: '#0a6f8c' } },
        // rel="me" identity links (canonical list from jmrp.io)
        { tag: 'link', attrs: { rel: 'me', href: 'https://github.com/jmrplens' } },
        { tag: 'link', attrs: { rel: 'me', href: 'https://www.linkedin.com/in/jmrplens' } },
        { tag: 'link', attrs: { rel: 'me', href: 'https://mstdn.jmrp.io/@jmrplens' } },
        { tag: 'link', attrs: { rel: 'me', href: 'https://scholar.google.com/citations?user=9b0kPaUAAAAJ' } },
        { tag: 'link', attrs: { rel: 'me', href: 'https://orcid.org/0000-0003-1250-6212' } },
        { tag: 'link', attrs: { rel: 'me', href: 'https://matrix.to/#/@jmrplens:matrix.jmrp.io' } },
        { tag: 'link', attrs: { rel: 'me', href: 'https://keyoxide.org/0A993B268654DBBA52B7E8D3FCF653391E2C91FC' } },
        { tag: 'link', attrs: { rel: 'me', href: 'https://jmrp.io' } },
        // PGP public key
        {
          tag: 'link',
          attrs: {
            rel: 'pgpkey',
            type: 'application/pgp-keys',
            href: 'https://keys.openpgp.org/vks/v1/by-fingerprint/0A993B268654DBBA52B7E8D3FCF653391E2C91FC',
          },
        },
        // Web app manifest
        { tag: 'link', attrs: { rel: 'manifest', href: `${basePath}/manifest.json` } },
        // Icons. The SVG favicon is Starlight's own default (`favicon:
        // '/favicon.svg'`), which emits its link tag itself; repeating it here
        // put two links to the same file in the head. These are the variants
        // the option cannot express, added exactly as the configuration
        // reference recommends: the .ico for clients that still request
        // /favicon.ico by name, and the touch icon for an iOS home screen,
        // which ignores both. The SVG carries its own prefers-color-scheme
        // rule, so the mark lightens itself against a dark browser chrome.
        { tag: 'link', attrs: { rel: 'icon', sizes: '48x48', href: `${basePath}/favicon.ico` } },
        {
          tag: 'link',
          attrs: { rel: 'apple-touch-icon', href: `${basePath}/apple-touch-icon.png` },
        },
        // Bing Webmaster Tools site verification (Google Search Console is
        // already verified for this property; no meta needed).
        { tag: 'meta', attrs: { name: 'msvalidate.01', content: '7574EB3B44624C239F14920DBC34EE25' } },
        // JSON-LD structured data
        { tag: 'script', attrs: { type: 'application/ld+json' }, content: jsonLd },
      ],
    }),
  ],
});
