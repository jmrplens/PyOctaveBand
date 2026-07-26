/**
 * Intrinsic pixel size of a documentation figure, read at build time.
 *
 * The figures are authored as `<img style="width:NN%">` with no `width` or
 * `height` attributes, so the browser learns a figure's aspect ratio only once
 * the bytes arrive. With almost every figure lazy-loaded that means each one
 * pops in and shoves the text below it down the page: on a sampled guide, 0 of
 * 12 images carried dimensions.
 *
 * Every SVG already declares its own size, so nothing has to be measured or
 * guessed: `diagram_*.svg` in pixels, the matplotlib figures in points. This
 * reads that declaration and hands back numbers the component can emit, which
 * gives the browser an aspect ratio to reserve. `theme-images.css` already sets
 * `height: auto`, so the attributes constrain the ratio without fighting the
 * authored percentage width.
 *
 * The figures are SERVED from raw.githubusercontent.com but they live in this
 * repository, so the local copy is what gets measured.
 */
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Locate `.github/images/` by walking up from a starting directory.
 *
 * A fixed number of `..` hops from `import.meta.url` breaks once Astro bundles
 * this module into `dist/.prerender/chunks/`, because the hops are then counted
 * from the chunk rather than from `src/lib/`. Walking up until the directory
 * appears works from the source tree, from the prerender bundle, and from
 * whichever directory the build was launched in.
 */
function findImagesDir() {
  const starts = [dirname(fileURLToPath(import.meta.url)), process.cwd()];
  for (const start of starts) {
    let dir = start;
    for (;;) {
      const candidate = join(dir, '.github', 'images');
      if (existsSync(candidate)) return candidate;
      const parent = dirname(dir);
      if (parent === dir) break;
      dir = parent;
    }
  }
  return undefined;
}

const IMAGES_DIR = findImagesDir();

/** CSS points to pixels: 1pt = 4/3 px. Matplotlib writes SVG sizes in points. */
const PT_TO_PX = 4 / 3;

/** url -> {width, height} | null. Built once per build. */
const cache = new Map();

/** Local path for a figure URL, whether it is remote or site-relative. */
function localPath(src) {
  if (!IMAGES_DIR) return undefined;
  const name = src.split('/').pop();
  if (!name || !name.endsWith('.svg')) return undefined;
  const candidate = join(IMAGES_DIR, name);
  return existsSync(candidate) ? candidate : undefined;
}

function parseLength(value) {
  if (!value) return undefined;
  const match = /^([\d.]+)\s*(pt|px)?$/.exec(value.trim());
  if (!match) return undefined;
  const n = Number(match[1]);
  if (!Number.isFinite(n) || n <= 0) return undefined;
  return match[2] === 'pt' ? n * PT_TO_PX : n;
}

/**
 * Intrinsic size of a figure.
 *
 * @param {string} src Figure URL as authored.
 * @returns {{width: number, height: number}|undefined} Rounded pixel size, or
 *   undefined when the file is not a local SVG or declares no usable size. The
 *   caller then emits no attributes, which is exactly the old behaviour.
 */
export function figureSize(src) {
  if (cache.has(src)) return cache.get(src) ?? undefined;

  let size;
  const path = localPath(src);
  if (path) {
    // The declaration is in the opening <svg> tag; no need to read further.
    const head = readFileSync(path, 'utf8').slice(0, 2000);
    const open = /<svg\b[^>]*>/.exec(head)?.[0];
    if (open) {
      const width = parseLength(/\bwidth="([^"]+)"/.exec(open)?.[1]);
      const height = parseLength(/\bheight="([^"]+)"/.exec(open)?.[1]);
      if (width && height) {
        size = { width: Math.round(width), height: Math.round(height) };
      } else {
        // Some generators give only a viewBox. Its extent is the intrinsic
        // ratio, which is all the browser needs.
        const box = /\bviewBox="([^"]+)"/.exec(open)?.[1]?.trim().split(/[\s,]+/);
        if (box?.length === 4) {
          const w = Number(box[2]);
          const h = Number(box[3]);
          if (w > 0 && h > 0) size = { width: Math.round(w), height: Math.round(h) };
        }
      }
    }
  }

  cache.set(src, size ?? null);
  return size;
}
