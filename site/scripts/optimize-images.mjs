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

const encoders = {
	'.png': (image) =>
		image.png({ palette: true, quality: 90, compressionLevel: 9, effort: 10 }),
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
		optimized = await encode(sharp(original)).toBuffer();
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
