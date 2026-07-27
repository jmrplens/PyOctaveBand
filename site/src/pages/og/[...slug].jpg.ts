/**
 * Per-page Open Graph card endpoint.
 *
 * One 1200x630 image per documentation page, in both languages, generated at
 * build time. The path mirrors the page's own, so `Head.astro` can point at
 * `/og/<slug>.jpg` without a lookup table:
 *
 *   /                      -> og/index.jpg
 *   /guides/filter-banks/  -> og/guides/filter-banks.jpg
 *   /es/guides/filter-banks/ -> og/es/guides/filter-banks.jpg
 *
 * The API reference is deliberately excluded. Those 120 pages are generated
 * from docstrings, nobody shares one on social media, and a card each would
 * add several minutes of build for no reader. They fall back to the site-wide
 * card, which `Head.astro` still emits when a page has no image of its own.
 */

import { areaLabelFor, artKeyFor } from '../../lib/og-areas.mjs';
import { generateOgImage } from '../../utils/og-image';
import type { APIContext } from 'astro';
import { getCollection } from 'astro:content';

interface Props {
  title: string;
  kicker?: string;
  art: string;
  locale: 'en' | 'es';
}

/** Docstring-generated API pages, which get the site-wide card instead. */
function isApiReference(id: string): boolean {
  return /^(es\/)?reference\/api\//.test(id);
}

/**
 * Enumerates every documentation page that gets a card of its own.
 * Both locales are generated: the site is fully bilingual, so each Spanish
 * card carries the Spanish title and the Spanish area name.
 */
export async function getStaticPaths() {
  const pages = await getCollection('docs');
  return pages
    .filter((page) => !isApiReference(page.id))
    .map((page) => {
      const locale = page.id.startsWith('es/') ? 'es' : 'en';
      const slug = page.id.replace(/\/index$/, '') || 'index';
      return {
        params: { slug: slug === 'index' ? 'index' : slug },
        props: {
          title: page.data.title,
          kicker: areaLabelFor(page.id, locale),
          art: artKeyFor(page.id),
          locale,
        } satisfies Props,
      };
    });
}

/**
 * Renders one card at build time.
 * @param context - Astro's API context, carrying the props from getStaticPaths.
 */
export async function GET({ props }: APIContext<Props>): Promise<Response> {
  const jpeg = await generateOgImage(props);
  return new Response(new Uint8Array(jpeg), {
    headers: {
      'Content-Type': 'image/jpeg',
      'Cache-Control': 'public, max-age=31536000, immutable',
    },
  });
}
