import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLinksValidator from 'starlight-links-validator';
import mermaid from 'astro-mermaid';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { sidebar } from './src/data/sidebar.mjs';
import { basePath, siteUrl } from './src/data/site.mjs';
import { isOurMedia, mediaUrl, REMOTE_PREFIXES } from './src/lib/media.mjs';
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
// but around a hundred references are hand-written <img> pairs inside the
// markdown, mostly on the theory pages, and those never pass through a
// component. This catches them, plus any <a href> or <video src> pointing at
// the same files, so nothing is left hotlinking raw.githubusercontent.com
// whichever way it was authored.
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
// an <annotation encoding="application/x-tex"> holding the original TeX. The
// annotation is not aria-hidden, so a text extractor that walks the DOM emits
// the formula two or three times over: "L_eq" comes out as "LeqL_{eq}Leq", and
// on these pages the affected passages are the standards formulas, which are
// exactly the ones worth quoting. Dropping the annotation keeps the MathML
// (and the accessibility it provides) while leaving one readable copy.
function rehypeDropTexAnnotation() {
  return (tree) => {
    (function visit(node) {
      if (!node.children) return;
      node.children = node.children.filter(
        (child) => !(child.type === 'element' && child.tagName === 'annotation'),
      );
      node.children.forEach(visit);
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

const jsonLd = JSON.stringify({
  '@context': 'https://schema.org',
  '@graph': [
    {
      '@type': 'Person',
      '@id': authorId,
      name: 'José Manuel Requena Plens',
      alternateName: 'jmrplens',
      // Matches the canonical node published at jmrp.io. Both sites claim the
      // same `@id`, so where they disagreed a consumer merging them got two
      // conflicting descriptions of one person; this side used to be the
      // weaker of the two, missing the ORCID entirely.
      jobTitle: 'R&D · Firmware & Software Engineer',
      description:
        'Firmware and software engineer in Valencia, Spain: industrial embedded systems, open-source tooling, and self-hosted infrastructure.',
      url: authorUrl,
      image: 'https://github.com/jmrplens.png',
      identifier: {
        '@type': 'PropertyValue',
        propertyID: 'ORCID',
        value: '0000-0003-1250-6212',
        url: 'https://orcid.org/0000-0003-1250-6212',
      },
      // Wikidata-grounded rather than free text: a bare string leaves a model
      // to guess which "Acoustics" is meant, an entity URI does not. Every QID
      // below was checked against the Wikidata API.
      knowsAbout: [
        { '@type': 'Thing', name: 'Acoustics', '@id': 'http://www.wikidata.org/entity/Q82811' },
        { '@type': 'Thing', name: 'Signal processing', '@id': 'http://www.wikidata.org/entity/Q208163' },
        { '@type': 'Thing', name: 'Psychoacoustics', '@id': 'http://www.wikidata.org/entity/Q557399' },
        { '@type': 'Thing', name: 'Architectural acoustics', '@id': 'http://www.wikidata.org/entity/Q2305951' },
        { '@type': 'Thing', name: 'Noise control', '@id': 'http://www.wikidata.org/entity/Q1879301' },
        { '@type': 'Thing', name: 'Vibration', '@id': 'http://www.wikidata.org/entity/Q3695508' },
        { '@type': 'Thing', name: 'Underwater acoustics', '@id': 'http://www.wikidata.org/entity/Q1292425' },
        { '@type': 'Thing', name: 'Python', '@id': 'http://www.wikidata.org/entity/Q28865' },
        { '@type': 'Thing', name: 'Embedded system', '@id': 'http://www.wikidata.org/entity/Q193040' },
      ],
      sameAs: [
        'https://github.com/jmrplens',
        'https://www.linkedin.com/in/jmrplens',
        'https://mstdn.jmrp.io/@jmrplens',
        'https://bsky.app/profile/jmrp.io',
        'https://matrix.to/#/@jmrplens:matrix.jmrp.io',
        'https://keyoxide.org/0A993B268654DBBA52B7E8D3FCF653391E2C91FC',
        'https://scholar.google.com/citations?user=9b0kPaUAAAAJ',
        'https://orcid.org/0000-0003-1250-6212',
        'https://www.researchgate.net/profile/Jose-Requena-Plens-2',
        'https://www.mathworks.com/matlabcentral/profile/authors/5890853',
      ],
    },
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
        value: '10.5281/zenodo.21215280',
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
      sameAs: [
        'https://pypi.org/project/phonometry/',
        'https://doi.org/10.5281/zenodo.21215280',
        `${fullUrl}/`,
      ],
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
    },
  ],
});

export default defineConfig({
  site: siteUrl,
  base: basePath,
  redirects: {
    '/guides/building-acoustics/': `${basePath}/guides/insulation-field/`,
    '/es/guides/building-acoustics/': `${basePath}/es/guides/insulation-field/`,
    '/guides/psychoacoustics/': `${basePath}/guides/loudness/`,
    '/es/guides/psychoacoustics/': `${basePath}/es/guides/loudness/`,
  },
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
    // rehypeDropTexAnnotation must run after rehypeKatex: it prunes a node
    // KaTeX has just produced.
    rehypePlugins: [rehypeKatex, rehypeDropTexAnnotation, rehypeLocalMedia, rehypeTableAlign],
  },
  integrations: [
    mermaid({
      theme: 'default',
      autoTheme: true,
    }),
    starlight({
      title: 'phonometry',
      routeMiddleware: './src/routeData.ts',
      plugins: [
        starlightLinksValidator({
          errorOnRelativeLinks: false,
          errorOnFallbackPages: false,
        }),
      ],
      description: siteDescription,
      lastUpdated: true,
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
        // Default article body plus the unified APA-7 references section
        // rendered from the typed frontmatter bibliography.
        MarkdownContent: './src/components/MarkdownContent.astro',
        // Default H1 plus the page header chips run derived from the page's
        // own `references` frontmatter (see TOC-REDESIGN-NOTES.md).
        PageTitle: './src/components/PageTitle.astro',
      },
      customCss: [
        // The site palette. Unlayered, so its `:root` blocks win over
        // Starlight's `@layer starlight.base` defaults; loaded first so the
        // later sheets can still reference the tokens it defines.
        './src/styles/theme.css',
        './src/styles/katex.css',
        './src/styles/theme-images.css',
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
        // Open Graph image
        { tag: 'meta', attrs: { property: 'og:image', content: socialImageUrl } },
        { tag: 'meta', attrs: { property: 'og:image:alt', content: socialImageAlt } },
        { tag: 'meta', attrs: { property: 'og:image:type', content: 'image/png' } },
        { tag: 'meta', attrs: { property: 'og:image:width', content: '1200' } },
        { tag: 'meta', attrs: { property: 'og:image:height', content: '630' } },
        // Twitter card
        { tag: 'meta', attrs: { name: 'twitter:card', content: 'summary_large_image' } },
        { tag: 'meta', attrs: { name: 'twitter:image', content: socialImageUrl } },
        { tag: 'meta', attrs: { name: 'twitter:image:alt', content: socialImageAlt } },
        // Author
        { tag: 'meta', attrs: { name: 'author', content: 'José Manuel Requena Plens' } },
        // Theme color
        { tag: 'meta', attrs: { name: 'theme-color', content: '#1f77b4' } },
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
        // Bing Webmaster Tools site verification (Google Search Console is
        // already verified for this property; no meta needed).
        { tag: 'meta', attrs: { name: 'msvalidate.01', content: '7574EB3B44624C239F14920DBC34EE25' } },
        // JSON-LD structured data
        { tag: 'script', attrs: { type: 'application/ld+json' }, content: jsonLd },
      ],
      sidebar,
    }),
  ],
});
