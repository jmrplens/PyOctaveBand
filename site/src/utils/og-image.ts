/**
 * Open Graph card generator.
 *
 * Builds the 1200x630 image a link preview shows, using satori to lay the card
 * out as SVG with the text converted to outlines, then sharp to rasterise. Both
 * are already how the rest of the build handles images, and neither needs a
 * browser.
 *
 * The artwork, the mark and the fonts are the committed brand assets under
 * `.github/brand`, read from disk rather than copied into the site tree, so
 * there is one master for the README, the repository card and these.
 *
 * Every page carries the artwork of its documentation area, so a link to a
 * room-acoustics guide previews differently from a link to an underwater one
 * while staying recognisably the same family.
 */

import { readFile } from 'node:fs/promises';
import path from 'node:path';

import satori from 'satori';
import sharp from 'sharp';

/** Repository root: the site builds from `site/`, the brand assets live above it. */
const REPO = path.join(process.cwd(), '..');
const BRAND = path.join(REPO, '.github', 'brand');

export const OG_WIDTH = 1200;
export const OG_HEIGHT = 630;

const PAD_X = 72;
const PAD_Y = 60;

// Straight from scripts/generate_brand.py, so a card cannot drift from the mark.
const HEADING = '#f2f8fa';
const SUPPORT = '#a8d8e6';
const SUPPORT_DIM = '#84bccf';
const ACCENT = '#4ec4e0';

type Cache<T> = { value?: T };
const fontDisplay: Cache<Buffer> = {};
const fontRegular: Cache<Buffer> = {};
const markUri: Cache<string> = {};
const artUris = new Map<string, string>();

/**
 * Reads a file once and keeps it for the rest of the build.
 * @param cache - Slot holding the value after the first read.
 * @param file - Absolute path to read.
 */
async function once(cache: Cache<Buffer>, file: string): Promise<Buffer> {
  cache.value ??= await readFile(file);
  return cache.value;
}

/**
 * The mark, reversed out for a dark ground, as a data URI.
 *
 * A PNG rather than the SVG: satori rasterises `img` sources itself and its
 * SVG support does not cover the whole file, so the raster is the safe input.
 */
async function getMark(): Promise<string> {
  if (!markUri.value) {
    const buf = await readFile(path.join(BRAND, 'logo-on-dark-512.png'));
    markUri.value = `data:image/png;base64,${buf.toString('base64')}`;
  }
  return markUri.value;
}

/**
 * The artwork for one documentation area, as a data URI.
 *
 * Transcoded to JPEG on the way in. The committed masters are WebP, which is
 * the right format to store them in, but satori silently drops an image whose
 * header it cannot read and WebP is not among the ones it parses: the card
 * came out as a flat background with no artwork at all.
 * @param key - Area slug, or "generic" for everything outside the nine areas.
 */
async function getArt(key: string): Promise<string> {
  const cached = artUris.get(key);
  if (cached) return cached;
  const file = key === 'generic' ? 'generic.webp' : `area-${key}.webp`;
  const buf = await readFile(path.join(BRAND, 'art', file));
  const jpeg = await sharp(buf).jpeg({ quality: 92 }).toBuffer();
  const uri = `data:image/jpeg;base64,${jpeg.toString('base64')}`;
  artUris.set(key, uri);
  return uri;
}

/**
 * Picks a title size that keeps the longest documentation titles to two lines.
 * @param title - The page title.
 */
function titleSize(title: string): number {
  const n = title.length;
  if (n <= 22) return 68;
  if (n <= 38) return 58;
  if (n <= 56) return 48;
  if (n <= 78) return 40;
  return 34;
}

type Style = Record<string, string | number | undefined>;
interface Node {
  type: string;
  key: null;
  props: { style?: Style; children?: unknown; [k: string]: unknown };
}

/**
 * Creates one satori node. Satori takes a React-shaped tree without React.
 * @param type - HTML tag name.
 * @param style - Inline style, CSS properties in camelCase.
 * @param children - Child nodes or text.
 */
function el(type: string, style: Style, ...children: (Node | string | null)[]): Node {
  const kept = children.filter((c): c is Node | string => c !== null);
  return {
    type,
    key: null,
    props: {
      style,
      children: kept.length === 0 ? undefined : kept.length === 1 ? kept[0] : kept,
    },
  };
}

/** The strapline under the wordmark, in the card's own language. */
const TAGLINE: Record<'en' | 'es', string> = {
  en: 'Acoustic measurement toolkit for Python',
  es: 'Herramientas de medida acústica para Python',
};

export interface OgCard {
  /** Large heading: the page's own title. */
  title: string;
  /** Muted kicker above it: the documentation area, when the page has one. */
  kicker?: string;
  /** Artwork key, from `artKeyFor`. */
  art: string;
  /** Which language the card's own furniture is written in. */
  locale: 'en' | 'es';
}

/**
 * Renders one card to PNG bytes.
 * @param card - Title, kicker and artwork key for this page.
 */
export async function generateOgImage(card: OgCard): Promise<Buffer> {
  const [display, regular, mark, art] = await Promise.all([
    once(fontDisplay, path.join(BRAND, 'fonts', 'FamiljenGrotesk-Bold.ttf')),
    once(fontRegular, path.join(BRAND, 'fonts', 'IBMPlexSans-Regular.ttf')),
    getMark(),
    getArt(card.art),
  ]);

  const tree = el(
    'div',
    {
      display: 'flex',
      width: '100%',
      height: '100%',
      position: 'relative',
      backgroundColor: '#04222b',
    },
    // The artwork, full bleed. It is already dark on the left, which is where
    // the type sits, so it needs no scrim over it.
    {
      type: 'img',
      key: null,
      props: {
        src: art,
        width: OG_WIDTH,
        height: OG_HEIGHT,
        style: { position: 'absolute', top: 0, left: 0, objectFit: 'cover' },
      },
    } as Node,
    el(
      'div',
      {
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        flex: '1',
        padding: `${PAD_Y}px ${PAD_X}px`,
      },
      // Top: the mark, at the size it keeps its grid legible.
      {
        type: 'img',
        key: null,
        props: { src: mark, width: 84, height: 84, style: { display: 'flex' } },
      } as Node,
      // Middle: the area, then the page title.
      el(
        'div',
        { display: 'flex', flexDirection: 'column', gap: '14px', maxWidth: '860px' },
        card.kicker
          ? el(
              'div',
              {
                fontFamily: 'IBM Plex Sans',
                fontSize: '21px',
                color: ACCENT,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
              },
              card.kicker,
            )
          : null,
        el(
          'div',
          {
            fontFamily: 'Familjen Grotesk',
            fontSize: `${titleSize(card.title)}px`,
            color: HEADING,
            lineHeight: '1.12',
            letterSpacing: '-0.015em',
          },
          card.title,
        ),
      ),
      // Bottom: the wordmark, with a rule keeping it off the artwork.
      el(
        'div',
        {
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-end',
          borderTop: '1px solid rgba(168,216,230,0.25)',
          paddingTop: '16px',
        },
        el(
          'div',
          { fontFamily: 'Familjen Grotesk', fontSize: '28px', color: SUPPORT },
          'phonometry',
        ),
        el(
          'div',
          { fontFamily: 'IBM Plex Sans', fontSize: '19px', color: SUPPORT_DIM },
          TAGLINE[card.locale],
        ),
      ),
    ),
  );

  const svg = await satori(tree as never, {
    width: OG_WIDTH,
    height: OG_HEIGHT,
    fonts: [
      { name: 'Familjen Grotesk', data: display, weight: 700, style: 'normal' },
      { name: 'IBM Plex Sans', data: regular, weight: 400, style: 'normal' },
    ],
  });

  // JPEG, not PNG: the cards carry photographic artwork, so a lossless encode
  // costs several times the bytes for a difference no scraper will ever show.
  return sharp(Buffer.from(svg)).jpeg({ quality: 88, progressive: true }).toBuffer();
}
