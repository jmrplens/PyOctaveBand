// Stages the documentation media into the site's own asset tree, so the built
// pages serve every figure, animation and report preview from this origin
// instead of hotlinking raw.githubusercontent.com.
//
// Why not hotlink: raw.githubusercontent.com is not a CDN, it caches for five
// minutes against the Pages origin's ten, it is rate limited, GitHub asks
// people not to use it as an asset host, and the URLs were pinned to `main`,
// so renaming a figure silently broke every published page that showed it.
// Serving them here also means the figures finally belong to the pages they
// illustrate rather than to a third origin.
//
// What is staged, and what is not:
//   *.svg    copied verbatim. These are the figure pipeline's output and are
//            already minimal; re-encoding them is not wanted.
//   *.webp   copied, then recompressed in dist by optimize-images.mjs.
//   *.jpg    animation poster stills, same treatment.
//   *.webm   copied verbatim. Video, nothing here can improve it.
//   *.gif    NOT staged. The GIFs exist only for the markdown mirror on
//            GitHub, which cannot play WebM; the site uses the WebM.
//
// The markdown under docs/ keeps its absolute raw.githubusercontent URLs on
// purpose: GitHub renders those files directly and has no build step.
//
// Staged into public/media rather than through Vite's hashed asset pipeline:
// a hundred of these references are hand-written <img> tags inside markdown,
// which a rehype pass rewrites at build time, and that pass cannot know a
// content hash. A stable path is the one scheme both the components and the
// markdown can resolve. GitHub Pages serves everything with the same
// max-age either way, so the hash would buy little here.
//
// Runs as a prebuild step. Output is generated, gitignored, and refreshed only
// when a source file actually changes so repeat builds stay fast.
import { copyFileSync, existsSync, mkdirSync, readdirSync, rmSync, statSync } from 'node:fs';
import { dirname, extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, '..', '..');
const target = join(here, '..', 'public', 'media');

/** Source directories, and the extensions taken from each. */
const SOURCES = [
  { dir: join(repoRoot, '.github', 'images'), exts: new Set(['.svg', '.webp', '.jpg', '.webm']) },
  // The example fiches: the WebP preview each page shows, and the PDF it links
  // to. Together they are under 6 MB, which is worth paying to leave nothing
  // pointing off-origin.
  { dir: join(repoRoot, '.github', 'reports'), exts: new Set(['.webp', '.pdf']) },
];

mkdirSync(target, { recursive: true });

const staged = new Set();
const totals = new Map();
let copied = 0;
let skipped = 0;

for (const { dir, exts } of SOURCES) {
  if (!existsSync(dir)) {
    console.warn(`[stage-media] ${dir} not found, skipping`);
    continue;
  }
  for (const name of readdirSync(dir)) {
    const ext = extname(name).toLowerCase();
    if (!exts.has(ext)) continue;

    const src = join(dir, name);
    const dest = join(target, name);
    if (staged.has(name)) {
      throw new Error(
        `[stage-media] two source files are both named "${name}"; the staged tree is flat, so the names have to be unique`,
      );
    }
    staged.add(name);

    const from = statSync(src);
    totals.set(ext, (totals.get(ext) ?? 0) + from.size);

    // Copy only when the destination is missing or out of date: this runs on
    // every build and the tree is around a thousand files.
    if (existsSync(dest)) {
      const to = statSync(dest);
      if (to.size === from.size && to.mtimeMs >= from.mtimeMs) {
        skipped += 1;
        continue;
      }
    }
    copyFileSync(src, dest);
    copied += 1;
  }
}

// Drop anything a previous run staged that no longer has a source, so a
// renamed or deleted figure does not linger in the published assets.
let removed = 0;
for (const name of readdirSync(target)) {
  if (!staged.has(name)) {
    rmSync(join(target, name), { recursive: true, force: true });
    removed += 1;
  }
}

const megabytes = (n) => (n / 1024 / 1024).toFixed(1);
const breakdown = [...totals.entries()]
  .sort((a, b) => b[1] - a[1])
  .map(([ext, size]) => `${ext.slice(1)} ${megabytes(size)} MB`)
  .join(', ');
const total = [...totals.values()].reduce((a, b) => a + b, 0);

console.log(
  `[stage-media] ${staged.size} files, ${megabytes(total)} MB (${breakdown})` +
    ` | ${copied} copied, ${skipped} unchanged, ${removed} removed`,
);
