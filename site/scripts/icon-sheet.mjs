// Renders the index of the site's own icons (src/data/icons.mjs) to
// src/assets/icon-sheet.svg, which is committed.
//
// The sheet is the template: every icon at the size it is drawn at and at the
// size it renders at, in one file, so the next one is drawn next to the twelve
// it has to look like rather than on its own. It is also what a pull request
// shows when a drawing changes, which a table of path data does not.
//
// The 16 px column is the one that matters. Every icon in this set was drawn
// twice because the first version was legible large and a smudge small, and
// three were drawn four times because at 16 px they were read as something
// else entirely: a smiling face, a pair of horns, a map pin.
//
// Usage: node scripts/icon-sheet.mjs [--check]
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { topicIcons } from '../src/data/icons.mjs';
import { topics } from '../src/data/topics.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const OUT = join(here, '..', 'src', 'assets', 'icon-sheet.svg');

const COLUMNS = 5;
const CELL_W = 150;
const CELL_H = 108;
const MARGIN = 24;
const HEADER = 46;

/** One icon drawn at `size`, centred on (`cx`, `y`). */
const glyph = (name, cx, y, size) =>
  `<g transform="translate(${(cx - size / 2).toFixed(1)} ${y.toFixed(1)}) scale(${(size / 24).toFixed(4)})" ` +
  `fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">` +
  `${topicIcons[name].trim().replace(/\n\s*/g, '')}</g>`;

/** The label under a cell, in the same grey the sheet's own headings use. */
const caption = (text, cx, y) =>
  `<text x="${cx.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="middle" font-size="11" ` +
  `font-family="sans-serif" fill="#6b7280">${text}</text>`;

const entries = topics.map((topic) => ({ id: topic.id, label: topic.label.en }));
const missing = entries.filter((entry) => !topicIcons[entry.id]);
if (missing.length > 0) {
  console.error(`icon-sheet: no drawing for ${missing.map((entry) => entry.id).join(', ')}`);
  process.exit(1);
}
const extra = Object.keys(topicIcons).filter((id) => !entries.some((entry) => entry.id === id));
if (extra.length > 0) {
  console.error(`icon-sheet: ${extra.join(', ')} is drawn but is no topic`);
  process.exit(1);
}

const rows = Math.ceil(entries.length / COLUMNS);
const width = MARGIN * 2 + COLUMNS * CELL_W;
const height = MARGIN * 2 + HEADER + rows * CELL_H;

const cells = entries.map((entry, index) => {
  const cx = MARGIN + (index % COLUMNS) * CELL_W + CELL_W / 2;
  const top = MARGIN + HEADER + Math.floor(index / COLUMNS) * CELL_H;
  return [
    glyph(entry.id, cx - 26, top, 48),
    // Drawn beside the large one rather than in a row of its own: the pair is
    // the judgement, and it is only a judgement when both are in one glance.
    glyph(entry.id, cx + 32, top + 16, 16),
    caption(entry.label, cx, top + 66),
    caption(entry.id, cx, top + 82),
  ].join('');
});

const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="The site's own icons, at 48 and 16 pixels">
  <title>phonometry icon sheet</title>
  <style>
    :root { color-scheme: light dark; }
    .sheet { color: #17181c; }
    @media (prefers-color-scheme: dark) { .sheet { color: #f6f7f9; } }
  </style>
  <g class="sheet">
    <text x="${MARGIN}" y="${MARGIN + 16}" font-size="15" font-family="sans-serif" font-weight="600" fill="currentColor">The site's own icons</text>
    <text x="${MARGIN}" y="${MARGIN + 34}" font-size="11" font-family="sans-serif" fill="#6b7280">Each drawn at 48 px and at the 16 px it renders at. Regenerate with pnpm run icons:sheet.</text>
    ${cells.join('\n    ')}
  </g>
</svg>
`;

if (process.argv.includes('--check')) {
  const committed = readFileSync(OUT, 'utf8');
  if (committed !== svg) {
    console.error('icon-sheet: src/assets/icon-sheet.svg is stale. Run pnpm run icons:sheet.');
    process.exit(1);
  }
  console.log(`icon sheet up to date: ${entries.length} icons.`);
} else {
  writeFileSync(OUT, svg);
  console.log(`icon sheet written: ${entries.length} icons -> src/assets/icon-sheet.svg`);
}
