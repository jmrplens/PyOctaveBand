// Fail the build when a formula did not render.
//
// KaTeX does not stop a build when it cannot parse an expression: with
// `throwOnError` off, rehype-katex emits `<span class="katex-error">` carrying
// the parse message as its title and the page ships. Worse, a display block
// whose `$$` delimiters sit on a content line is not merely mis-rendered: the
// parser consumes the rest of the document as math, so `spanish-building-code`
// silently lost **six** `## ` sections and everything after them, in both
// locales, for as long as it took someone to read that page to the end.
//
// Nothing in the pipeline noticed. `html-validate` sees well-formed markup and
// pa11y sees an accessible page; both are right, and both are answering a
// different question. This script asks the one that matters: did every formula
// on the site actually render?
//
// It also flags the delimiter shape that caused it, in the sources rather than
// in the output, because that failure is silent by construction: a display
// block must open and close on its own line. Every one of the site's other
// display blocks already does.
//
// Usage: node scripts/check-math-render.mjs [--dist dist] [--sources ../docs ../site]

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const SITE = resolve(HERE, '..');
const REPO = resolve(SITE, '..');

/** Every file under `root` whose name ends in one of `exts`. */
function walk(root, exts, out = []) {
  let entries;
  try {
    entries = readdirSync(root, { withFileTypes: true });
  } catch {
    return out;
  }
  for (const entry of entries) {
    const path = join(root, entry.name);
    if (entry.isDirectory()) {
      // `superpowers/` is gitignored working notes, not published prose.
      if (entry.name === 'node_modules' || entry.name === '.astro') continue;
      if (entry.name === 'superpowers') continue;
      walk(path, exts, out);
    } else if (exts.some((ext) => entry.name.endsWith(ext))) {
      out.push(path);
    }
  }
  return out;
}

/** Parse errors KaTeX left in the built HTML, with the message it recorded. */
function renderFailures(distDir) {
  const failures = [];
  for (const file of walk(distDir, ['.html'])) {
    const html = readFileSync(file, 'utf8');
    if (!html.includes('katex-error')) continue;
    // The parse message is the title attribute of the offending span.
    const messages = [...html.matchAll(/class="katex-error"[^>]*title="([^"]*)"/g)].map(
      (m) => m[1],
    );
    failures.push({
      file: relative(REPO, file),
      count: Math.max(messages.length, 1),
      messages: [...new Set(messages)].slice(0, 3),
    });
  }
  return failures;
}

/**
 * Display blocks whose delimiters share a line with content.
 *
 * Inline `$$...$$` complete on one line is fine; what breaks the parser is a
 * block that opens or closes on a line carrying other content.
 */
function delimiterProblems(sourceDirs) {
  const problems = [];
  for (const dir of sourceDirs) {
    for (const file of walk(dir, ['.md', '.mdx'])) {
      const lines = readFileSync(file, 'utf8').split('\n');
      let inFence = false;
      lines.forEach((line, index) => {
        if (/^\s*(```|~~~)/.test(line)) inFence = !inFence;
        if (inFence) return;
        // Prose that talks *about* `$$` writes it inside a code span.
        const trimmed = line.replace(/`[^`]*`/g, '').trim();
        if (!trimmed.includes('$$')) return;
        const only = trimmed === '$$';
        const complete = /^\$\$.*\$\$$/.test(trimmed) && trimmed.length > 4;
        if (only || complete) return;
        problems.push({
          file: relative(REPO, file),
          line: index + 1,
          text: trimmed.slice(0, 72),
        });
      });
    }
  }
  return problems;
}

const args = process.argv.slice(2);
const distArg = args.indexOf('--dist');
const distDir = resolve(SITE, distArg === -1 ? 'dist' : args[distArg + 1]);
const sourceDirs = [join(REPO, 'docs'), join(SITE, 'src', 'content')];

const failures = renderFailures(distDir);
const problems = delimiterProblems(sourceDirs);

if (failures.length === 0 && problems.length === 0) {
  console.log('Math renders: no KaTeX parse errors in the build, no stray display delimiters.');
  process.exit(0);
}

if (failures.length > 0) {
  console.error(
    `\n${failures.length} built page(s) contain a formula KaTeX could not parse.\n` +
      'The page still ships, and everything after the bad block may be swallowed:\n',
  );
  for (const failure of failures) {
    console.error(`  ${failure.file}: ${failure.count} error(s)`);
    for (const message of failure.messages) console.error(`      ${message}`);
  }
}

if (problems.length > 0) {
  console.error(
    `\n${problems.length} display-math delimiter(s) share a line with content.\n` +
      'A `$$` block must open and close on its own line, or the parser runs on\n' +
      'past the end of the formula and eats the rest of the document:\n',
  );
  for (const problem of problems) {
    console.error(`  ${problem.file}:${problem.line}: ${problem.text}`);
  }
}

process.exit(1);
