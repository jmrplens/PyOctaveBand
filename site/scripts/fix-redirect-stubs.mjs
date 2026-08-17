// Brings Astro's redirect stub pages up to the standard the rest of the site
// is held to. Runs after every build.
//
// Two things about the stub Astro writes fail `html-validate`, which the site
// runs over every built page. The doctype is lowercase where the rest of the
// site's is upper, and the title is the destination spelled out in full,
// `Redirecting to: /phonometry/es/perception/psychoacoustics/...`, which runs
// past the 70-character limit of the `long-title` rule on any address more
// than four segments deep. That cost nothing while there were eight stubs and
// all of them were shallow; the redirect map covers every address the site has
// ever published, and 66 of those titles were over the limit.
//
// The title becomes plain "Redirecting". Nothing is lost by it: the stub is
// `noindex`, it is on screen for one frame, and the destination is still in
// the meta refresh, the canonical link and the visible link in the body, which
// is what a reader with scripting or refresh disabled follows.
//
// Both rewrites are guarded on the exact shape Astro emits, so they are
// idempotent and cannot touch a real page.
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const dist = fileURLToPath(new URL('../dist/', import.meta.url));

const LOWER_DOCTYPE = '<!doctype html>';
const UPPER_DOCTYPE = '<!DOCTYPE html>';
const STUB_TITLE = /^(<!DOCTYPE html>)<title>Redirecting to: [^<]*<\/title>/;

function* htmlFiles(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) yield* htmlFiles(path);
    else if (entry.name.endsWith('.html')) yield path;
  }
}

let doctypes = 0;
let titles = 0;
for (const file of htmlFiles(dist)) {
  let html = readFileSync(file, 'utf8');
  const before = html;
  if (html.startsWith(LOWER_DOCTYPE)) {
    html = UPPER_DOCTYPE + html.slice(LOWER_DOCTYPE.length);
    doctypes += 1;
  }
  if (STUB_TITLE.test(html)) {
    html = html.replace(STUB_TITLE, '$1<title>Redirecting</title>');
    titles += 1;
  }
  if (html !== before) writeFileSync(file, html);
}
console.log(
  `[fix-redirect-stubs] uppercased DOCTYPE in ${doctypes} file(s), shortened ${titles} redirect title(s)`,
);
