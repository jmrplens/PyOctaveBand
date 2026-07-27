/**
 * Which documentation area a page belongs to, derived from the sidebar.
 *
 * The nine areas are already declared once, as the sidebar's top-level groups,
 * and every page is placed in exactly one of them there. Reading that tree back
 * means the Open Graph cards inherit the same taxonomy the navigation uses, with
 * nothing to keep in step by hand: moving a guide into another group moves its
 * card artwork with it on the next build.
 */
import { sidebar } from '../data/sidebar.mjs';

/**
 * Sidebar group label to area slug.
 *
 * The slugs are the canonical ones from `AREAS` in scripts/generate_llms.py,
 * which already name the per-area `llms-*.txt` files, so the artwork lands on
 * the same identifiers rather than on a second set slugified from the labels.
 * A group missing here is simply not an area: Start, Reference and the API
 * tree carry the generic card.
 */
const AREA_SLUG = new Map([
  ['Core signal analysis', 'core-signal-analysis'],
  ['Hearing and perception', 'hearing-perception'],
  ['Rooms and buildings', 'rooms-buildings'],
  ['Materials and surfaces', 'materials-surfaces'],
  ['Vibration and structure-borne sound', 'vibration'],
  ['Environment and transport', 'environment-transport'],
  ['Underwater acoustics', 'underwater'],
  ['Sources and devices', 'sources-devices'],
  ['Wave simulation', 'simulation'],
]);

/**
 * Collects every slug reachable under a sidebar node, at any depth.
 * @param {unknown} node - A sidebar entry: a slug string, a link object or a group.
 * @param {string[]} out - Accumulator.
 */
function collectSlugs(node, out) {
  if (typeof node === 'string') {
    out.push(node);
    return;
  }
  if (!node || typeof node !== 'object') return;
  if (typeof node.slug === 'string') out.push(node.slug);
  if (Array.isArray(node.items)) {
    for (const child of node.items) collectSlugs(child, out);
  }
}

/** Slug (locale-free, no leading or trailing slash) to area key. */
export const areaBySlug = (() => {
  /** @type {Map<string, string>} */
  const map = new Map();
  for (const group of sidebar) {
    const key = AREA_SLUG.get(group?.label);
    if (!key) continue;
    /** @type {string[]} */
    const slugs = [];
    collectSlugs(group, slugs);
    for (const slug of slugs) {
      if (!map.has(slug)) map.set(slug, key);
    }
  }
  return map;
})();

/** Human-readable area label, keyed the same way, for the card's kicker. */
export const areaLabel = (() => {
  /** @type {Map<string, {en: string, es: string}>} */
  const map = new Map();
  for (const group of sidebar) {
    const key = AREA_SLUG.get(group?.label);
    if (!key) continue;
    map.set(key, { en: group.label, es: group.translations?.es ?? group.label });
  }
  return map;
})();

/**
 * The artwork key for a page, falling back to the generic card.
 * @param {string} id - The page's collection id, e.g. "guides/filter-banks" or
 *   "es/guides/filter-banks".
 */
export function artKeyFor(id) {
  const slug = id.replace(/^es\//, '').replace(/\.mdx?$/, '').replace(/\/index$/, '');
  return areaBySlug.get(slug) ?? 'generic';
}

/**
 * The area's display label for a page, or undefined outside the nine areas.
 * @param {string} id - The page's collection id.
 * @param {'en'|'es'} locale - Which language to label it in.
 */
export function areaLabelFor(id, locale) {
  const key = artKeyFor(id);
  return key === 'generic' ? undefined : areaLabel.get(key)?.[locale];
}
