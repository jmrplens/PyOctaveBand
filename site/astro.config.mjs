import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLinksValidator from 'starlight-links-validator';
import mermaid from 'astro-mermaid';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { apiSidebar } from './src/generated/api-sidebar.mjs';

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

const siteUrl = 'https://jmrplens.github.io';
const basePath = '/phonometry';
const fullUrl = `${siteUrl}${basePath}`;
const repositoryUrl = 'https://github.com/jmrplens/phonometry';
const authorUrl = 'https://jmrp.io';
const socialImageUrl = `${fullUrl}/og-image.png`;
const authorId = `${authorUrl}/#person`;
const websiteId = `${fullUrl}/#website`;
const softwareId = `${repositoryUrl}#software`;
const sourceCodeId = `${repositoryUrl}#source-code`;
const siteDescription =
  'Octave-band and fractional octave-band filter bank for Python. ANSI S1.11 / IEC 61260-1 compliant filters, IEC 61672-1 A/C/Z and Fast/Slow/Impulse weighting, Leq and statistical levels.';
const socialImageAlt =
  'phonometry: standards-compliant fractional octave analysis for Python';
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

const featureList = [
  '1/1, 1/3 and arbitrary fractional octave filter banks as stable SOS cascades with multirate decimation',
  'Five architectures: Butterworth, Chebyshev I/II, Elliptic and Bessel, all with -3 dB points on the ANSI band edges',
  'A/C/Z frequency weighting within IEC 61672-1 class 1 tolerances (verified against Table 3 in CI)',
  'Fast/Slow/Impulse time ballistics verified against the IEC 61672-1 Table 4 toneburst responses',
  'Leq, LAeq and L10/L50/L90 statistical levels, octave spectrograms and zero-phase offline filtering',
  'IEC 61260-1:2014 filter class verifier with per-band margins',
  'Physical SPL calibration (IEC 60942 calibrators) and dBFS mode',
  'Vectorized multichannel processing and stateful block (streaming) workflows',
];
const softwareRequirements = 'Python >= 3.13 with NumPy and SciPy (matplotlib and numba optional).';

const jsonLd = JSON.stringify({
  '@context': 'https://schema.org',
  '@graph': [
    {
      '@type': 'Person',
      '@id': authorId,
      name: 'José Manuel Requena Plens',
      alternateName: 'jmrplens',
      jobTitle: ['R&D Engineer', 'Firmware & Software Engineer'],
      url: authorUrl,
      image: 'https://github.com/jmrplens.png',
      knowsAbout: [
        'Acoustics',
        'Signal processing',
        'Octave-band filtering',
        'Python',
        'Embedded firmware',
      ],
      sameAs: [
        'https://github.com/jmrplens',
        'https://www.linkedin.com/in/jmrplens',
        'https://mstdn.jmrp.io/@jmrplens',
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
      featureList,
      keywords:
        'acoustics, octave band, fractional octave, sound level, signal processing, ANSI S1.11, IEC 61260, IEC 61672, Python',
      description:
        'Fractional octave filter banks, IEC 61672-1 weighting, Leq/LN levels and octave spectrograms for Python signals in the time domain.',
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
    rehypePlugins: [rehypeKatex, rehypeTableAlign],
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
        // Linked group labels + non-collapsible sidebar.
        Sidebar: './src/components/Sidebar.astro',
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
        './src/styles/ux-variants.css',
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
      // Overview-first convention: when a group's first item is a link with
      // `attrs: { 'data-group-link': true }`, the custom Sidebar override
      // consumes it and renders the group label itself as a link to that page
      // instead of a separate Overview row. Groups are never collapsible, so
      // `collapsed` has no effect.
      sidebar: [
        {
          label: 'Start',
          translations: { es: 'Inicio' },
          items: [
            'getting-started',
            'reference/why-phonometry',
          ],
        },
        {
          label: 'Core signal analysis',
          translations: { es: 'Análisis de señal' },
          items: [
            { slug: 'guides/sections/core-signal-analysis', attrs: { 'data-group-link': true } },
            'guides/sound-level-meter',
            {
              label: 'Octave filtering',
              translations: { es: 'Filtrado en octavas' },
              items: [
                { slug: 'guides/sections/octave-filtering', attrs: { 'data-group-link': true } },
                'guides/filter-banks',
                'guides/block-processing',
                'guides/multichannel',
              ],
            },
            {
              label: 'Levels and weighting',
              translations: { es: 'Niveles y ponderación' },
              items: [
                { slug: 'guides/sections/levels-weighting', attrs: { 'data-group-link': true } },
                'guides/weighting',
                'guides/time-weighting',
                'guides/levels',
              ],
            },
            {
              label: 'Signals and spectra',
              translations: { es: 'Señales y espectros' },
              items: [
                { slug: 'guides/sections/signals-spectra', attrs: { 'data-group-link': true } },
                'guides/spectral-analysis',
                'guides/miso-coherence',
                'guides/time-frequency',
                'guides/cepstrum-echoes',
                'guides/synchronous-averaging',
                'guides/correlation-delay',
                'guides/test-signals',
                'guides/system-measurement',
              ],
            },
            {
              label: 'Calibration and uncertainty',
              translations: { es: 'Calibración e incertidumbre' },
              items: [
                { slug: 'guides/sections/calibration-uncertainty', attrs: { 'data-group-link': true } },
                'guides/calibration',
                'guides/gum-uncertainty',
                'guides/data-qualification',
              ],
            },
          ],
        },
        {
          label: 'Hearing and perception',
          translations: { es: 'Audición y percepción' },
          items: [
            { slug: 'guides/sections/hearing-perception', attrs: { 'data-group-link': true } },
            {
              label: 'Psychoacoustics',
              translations: { es: 'Psicoacústica' },
              items: [
                { slug: 'guides/sections/psychoacoustics', attrs: { 'data-group-link': true } },
                'guides/loudness',
                'guides/sound-quality',
                'guides/tone-prominence',
                'guides/tone-audibility',
                'guides/psychoacoustic-annoyance',
              ],
            },
            {
              label: 'Speech',
              translations: { es: 'Habla' },
              items: [
                { slug: 'guides/sections/speech', attrs: { 'data-group-link': true } },
                'guides/speech-transmission',
                'guides/speech-intelligibility',
                'guides/objective-intelligibility',
              ],
            },
            {
              label: 'Hearing and exposure',
              translations: { es: 'Audición y exposición' },
              items: [
                { slug: 'guides/sections/hearing-exposure', attrs: { 'data-group-link': true } },
                'guides/hearing-threshold',
                'guides/noise-induced-hearing-loss',
                'guides/occupational-exposure',
              ],
            },
          ],
        },
        {
          label: 'Rooms and buildings',
          translations: { es: 'Salas y edificación' },
          items: [
            { slug: 'guides/sections/rooms-buildings', attrs: { 'data-group-link': true } },
            {
              label: 'Room acoustics',
              translations: { es: 'Acústica de salas' },
              items: [
                { slug: 'guides/sections/room-acoustics', attrs: { 'data-group-link': true } },
                'guides/room-acoustics',
                'guides/room-image-sources',
                'guides/room-noise',
                'guides/reverberation-prediction',
                'guides/enclosed-space-absorption',
              ],
            },
            {
              label: 'Sound insulation',
              translations: { es: 'Aislamiento acústico' },
              items: [
                { slug: 'guides/sections/sound-insulation', attrs: { 'data-group-link': true } },
                'guides/insulation-field',
                'guides/insulation-lab',
                'guides/insulation-prediction',
                'guides/panel-sound-insulation',
                'guides/dynamic-stiffness',
              ],
            },
          ],
        },
        {
          label: 'Materials and surfaces',
          translations: { es: 'Materiales y superficies' },
          items: [
            { slug: 'guides/sections/materials-surfaces', attrs: { 'data-group-link': true } },
            'guides/materials',
            'guides/porous-absorbers',
            'guides/surface-scattering',
          ],
        },
        {
          label: 'Vibration and structure-borne sound',
          translations: { es: 'Vibración y ruido estructural' },
          items: [
            { slug: 'guides/sections/vibration', attrs: { 'data-group-link': true } },
            {
              label: 'Structure-borne sources',
              translations: { es: 'Fuentes de ruido estructural' },
              items: [
                { slug: 'guides/sections/structure-borne', attrs: { 'data-group-link': true } },
                'guides/mechanical-mobility',
                'guides/junction-transmission',
                'guides/transfer-stiffness',
                'guides/vibration-sound-power',
                'guides/structure-borne-power',
                'guides/installed-structure-borne',
              ],
            },
            {
              label: 'Human vibration',
              translations: { es: 'Vibración en humanos' },
              items: [
                { slug: 'guides/sections/human-vibration', attrs: { 'data-group-link': true } },
                'guides/human-vibration',
                'guides/multiple-shock-vibration',
              ],
            },
          ],
        },
        {
          label: 'Environment and transport',
          translations: { es: 'Medio ambiente y transporte' },
          items: [
            { slug: 'guides/sections/environment-transport', attrs: { 'data-group-link': true } },
            {
              label: 'Outdoor sound',
              translations: { es: 'Sonido en exteriores' },
              items: [
                { slug: 'guides/sections/outdoor-sound', attrs: { 'data-group-link': true } },
                'guides/outdoor-propagation',
                'guides/ground-barriers',
                'guides/atmospheric-refraction',
                'guides/impulse-prominence',
              ],
            },
            {
              label: 'Aircraft and wind energy',
              translations: { es: 'Aeronaves y energía eólica' },
              items: [
                { slug: 'guides/sections/aircraft-wind', attrs: { 'data-group-link': true } },
                'guides/aircraft-noise',
                'guides/rotorcraft-noise',
                'guides/wind-turbine-noise',
              ],
            },
          ],
        },
        {
          label: 'Underwater acoustics',
          translations: { es: 'Acústica submarina' },
          items: [
            { slug: 'guides/sections/underwater', attrs: { 'data-group-link': true } },
            'guides/underwater-acoustics',
            'guides/underwater-propagation',
          ],
        },
        {
          label: 'Sources and devices',
          translations: { es: 'Fuentes y dispositivos' },
          items: [
            { slug: 'guides/sections/sources-devices', attrs: { 'data-group-link': true } },
            'guides/intensity',
            'guides/sound-power',
            'guides/electroacoustics',
            'guides/swept-sine-distortion',
            'guides/noise-control',
            'guides/program-loudness',
          ],
        },
        {
          label: 'Wave simulation',
          translations: { es: 'Simulación de ondas' },
          items: [
            { slug: 'guides/sections/simulation', attrs: { 'data-group-link': true } },
            'guides/fdtd-simulation',
          ],
        },
        {
          label: 'Reference',
          translations: { es: 'Referencia' },
          items: [
            apiSidebar,
            {
              label: 'Theory',
              translations: { es: 'Teoría' },
              items: [
                { slug: 'reference/theory', attrs: { 'data-group-link': true } },
                'reference/theory/signal-analysis',
                'reference/theory/perception',
                'reference/theory/rooms-buildings',
                'reference/theory/materials-surfaces',
                'reference/theory/environment-transport',
                'reference/theory/vibration',
              ],
            },
            'reference/conformance',
            'reference/bibliography',
          ],
        },
      ],
    }),
  ],
});
