/**
 * The generated brand SVGs, ready to inline.
 *
 * The files are imported as raw text so the bundler resolves them at build
 * time: reading them from disk at render time would depend on the process's
 * working directory and on where the server bundle ends up, and both differ
 * between `astro dev` and `astro build`.
 *
 * The wordmark in the lockup is type converted to outlines, which no screen
 * reader can read, so the accessible name never comes from the artwork: the
 * generator's own `role="img"` and `<title>` are stripped here and naming is
 * left to the element the SVG is placed in.
 */
import lockup from '../../../.github/brand/lockup.svg?raw';
import mark from '../../../.github/brand/logo.svg?raw';

export type BrandAsset = 'lockup' | 'mark';

/** Exact strings from scripts/generate_brand.py, asserted rather than assumed. */
const NAMED = ' role="img" aria-label="phonometry"';
const TITLE = '  <title>phonometry</title>\n';

export function brandSvg(asset: BrandAsset): string {
	const source = asset === 'lockup' ? lockup : mark;
	if (!source.includes(NAMED) || !source.includes(TITLE)) {
		throw new Error(
			`brand asset "${asset}" is missing the generator's role/title markup; ` +
				'run `make brand` (scripts/generate_brand.py)',
		);
	}
	return source.replace(NAMED, ' aria-hidden="true" focusable="false"').replace(TITLE, '');
}
