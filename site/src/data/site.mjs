/**
 * The two site-wide strings that more than one place needs.
 *
 * astro.config.mjs builds every absolute URL from them, and the audit scripts
 * in site/scripts/ address the running preview with the same base, so a move
 * to another path or another host is one edit here rather than a hunt through
 * the checks.
 */
export const siteUrl = 'https://jmrplens.github.io';
export const basePath = '/phonometry';
