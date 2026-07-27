// Recompresses the raster images the site serves (dist/ only; sources in
// public/ and src/ are never touched). PNGs are palette-quantized and JPEGs
// re-encoded with mozjpeg, each rewritten only when the result is smaller.
// SVG (all the figure pipeline output) passes through untouched, and WebP
// siblings are intentionally not emitted: every raster the site references
// today is a social-card or manifest image whose consumers (Open Graph and
// Twitter crawlers, web-app manifests) expect PNG/JPEG, and the content
// figures are remote SVG/WebP/WebM, so there is no <img> that could take a
// <picture> WebP source with fallback.
import { readdirSync, statSync, readFileSync, writeFileSync } from 'node:fs';
import { join, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';

const dist = fileURLToPath(new URL('../dist', import.meta.url));

function* walk(dir) {
	for (const entry of readdirSync(dir)) {
		const p = join(dir, entry);
		if (statSync(p).isDirectory()) yield* walk(p);
		else yield p;
	}
}

// Distinct colours above which an image is treated as photographic and kept
// off the palette path. The app icons are two flat colours plus antialiasing
// and land around 500; the Open Graph fallback card is artwork and lands near
// 15000. Quantising that one to 256 took the flat dark area from 249 grey
// levels to 22, which bands visibly, for 295 KiB nobody was asking for.
//
// `palette: false` has to be explicit on the photographic branch: passing
// `effort` (or `quality`, or `colours`) to sharp's png encoder turns the
// palette on by itself, so leaving the flag out quantises the image anyway.
const PALETTE_MAX_COLOURS = 1024;

/**
 * Counts distinct colours, to decide whether a palette will be near-lossless.
 * @param {import('sharp').Sharp} image - The decoded image.
 */
async function tooManyColours(image) {
	const { data, info } = await image
		.clone()
		.raw()
		.toBuffer({ resolveWithObject: true });
	const seen = new Set();
	for (let i = 0; i < data.length; i += info.channels) {
		seen.add((data[i] << 16) | (data[i + 1] << 8) | data[i + 2]);
		if (seen.size > PALETTE_MAX_COLOURS) return true;
	}
	return false;
}

const encoders = {
	'.png': async (image) =>
		(await tooManyColours(image))
			? image.png({ palette: false, compressionLevel: 9, effort: 10 })
			: image.png({ palette: true, quality: 90, compressionLevel: 9, effort: 10 }),
	'.jpg': (image) => image.jpeg({ mozjpeg: true, quality: 82 }),
	// The staged figure previews and animation stills (see
	// scripts/stage-media.mjs). Re-encoded at the same effort the generators
	// do not spend, since they optimise for regeneration speed.
	'.webp': (image) => image.webp({ quality: 82, effort: 6 }),
	'.jpeg': (image) => image.jpeg({ mozjpeg: true, quality: 82 }),
};

// PNG IHDR colour type (byte 25): 3 = indexed/palette. An indexed PNG has
// already been through the palette quantization; skip it so re-running the
// postbuild by hand cannot re-quantize (and degrade) an optimized image.
function isIndexedPng(file, buffer) {
	return extname(file).toLowerCase() === '.png' && buffer.length > 25 && buffer[25] === 3;
}

let totalBefore = 0;
let totalAfter = 0;
for (const file of walk(dist)) {
	const encode = encoders[extname(file).toLowerCase()];
	if (!encode) continue;
	const original = readFileSync(file);
	if (isIndexedPng(file, original)) continue;
	let optimized;
	try {
		optimized = await (await encode(sharp(original))).toBuffer();
	} catch (error) {
		// A broken or mislabelled image must not kill the deploy: keep the
		// original byte-for-byte and report which file failed.
		console.warn(
			`[optimize-images] skipping ${file.slice(dist.length + 1)}: ${error.message}`,
		);
		continue;
	}
	const kept = optimized.length < original.length ? optimized : original;
	if (kept !== original) writeFileSync(file, kept);
	totalBefore += original.length;
	totalAfter += kept.length;
	console.log(
		`[optimize-images] ${file.slice(dist.length + 1)}: ` +
			`${(original.length / 1024).toFixed(0)} KiB -> ${(kept.length / 1024).toFixed(0)} KiB`,
	);
}
if (totalBefore === 0) {
	console.log('[optimize-images] nothing to recompress in dist/');
} else {
	const saved = ((1 - totalAfter / totalBefore) * 100).toFixed(0);
	console.log(
		`[optimize-images] total ${(totalBefore / 1024).toFixed(0)} KiB -> ` +
			`${(totalAfter / 1024).toFixed(0)} KiB (${saved}% smaller)`,
	);
}
