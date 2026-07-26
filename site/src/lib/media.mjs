/**
 * Maps a documentation media reference to the copy this site serves.
 *
 * The figures, animations and report fiches are authored with the absolute
 * raw.githubusercontent.com URL, because the same markdown is mirrored under
 * docs/ and rendered by GitHub, which has no build step and needs a URL that
 * resolves on its own. The site does have a build step, so it serves its own
 * copy instead of hotlinking a third origin that is not a CDN, caches for five
 * minutes, is rate limited, and breaks silently when a file is renamed on
 * `main`.
 *
 * scripts/stage-media.mjs copies the media into public/media before the build.
 * The mapping is a plain string rewrite rather than a Vite asset import
 * because the same rewrite has to work in two places: here, for the
 * components, and in the rehype pass that fixes the hand-written <img> tags
 * inside markdown, which cannot resolve a content hash.
 */

/** Absolute prefixes that identify a reference to this repository's media. */
export const REMOTE_PREFIXES = [
  'https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/',
  'https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/',
];

/** Where stage-media.mjs puts them, relative to the site root. */
export const MEDIA_PATH = 'media';

/**
 * The URL this site should serve for a media reference.
 *
 * @param {string} src  Reference as authored, remote or already local.
 * @param {string} [base]  Site base path. Defaults to the build's BASE_URL,
 *   which is what the components want; the rehype pass passes it explicitly.
 * @returns {string} This site's URL for that file, or `src` unchanged when the
 *   reference is not one of ours. Leaving an unknown URL alone keeps an
 *   external illustration working instead of rewriting it into a 404.
 */
export function mediaUrl(src, base) {
  if (!src) return src;
  const prefix = REMOTE_PREFIXES.find((p) => src.startsWith(p));
  if (!prefix) return src;
  const root = (base ?? import.meta.env?.BASE_URL ?? '/').replace(/\/$/, '');
  return `${root}/${MEDIA_PATH}/${src.slice(prefix.length)}`;
}

/** True when `src` points at media this site is expected to serve itself. */
export function isOurMedia(src) {
  return Boolean(src) && REMOTE_PREFIXES.some((p) => src.startsWith(p));
}
