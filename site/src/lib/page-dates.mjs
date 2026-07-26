/**
 * First-publication dates, read from git history.
 *
 * Starlight's `lastUpdated` gives the last commit that touched a page, and the
 * structured data used to serve that same value as both `datePublished` and
 * `dateModified`. That claims every page was first published on the day it was
 * last edited: a typo fix moved a two-year-old guide's publication date to
 * today, and a corpus with real history looked freshly minted. It also
 * contradicted the `SoftwareApplication` node on the same page, which pins a
 * fixed `datePublished`.
 *
 * So `datePublished` comes from the commit that added the file. `--follow`
 * tracks it across the renames the content tree has been through, which a
 * plain `--diff-filter=A` on the current path would miss.
 */
import { execFileSync } from 'node:child_process';

/** path -> ISO date string, or null when git cannot answer. */
const cache = new Map();

/**
 * The date a file first appeared in history.
 *
 * @param {string} filePath Absolute or repo-relative path to the content file.
 * @returns {string|undefined} ISO-8601 date (`YYYY-MM-DD`), or undefined when
 *   the file is untracked, git is unavailable, or the checkout is shallow.
 */
export function firstPublished(filePath) {
  if (!filePath) return undefined;
  if (cache.has(filePath)) return cache.get(filePath) ?? undefined;

  let date;
  try {
    // --follow needs exactly one pathspec, so this is per file. The result is
    // memoised and only pages that actually render pay for it.
    const out = execFileSync(
      'git',
      ['log', '--follow', '--diff-filter=A', '--format=%cI', '--', filePath],
      { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] },
    ).trim();
    // A file resurrected after a delete has several adds; the oldest is the
    // publication date, and --format output is newest first.
    const lines = out.split('\n').filter(Boolean);
    const oldest = lines.at(-1);
    date = oldest ? oldest.slice(0, 10) : undefined;
  } catch {
    // No git, shallow clone, or untracked new page. The caller falls back to
    // the last-updated date, which is the best available answer for a page
    // that has no history yet.
    date = undefined;
  }

  cache.set(filePath, date ?? null);
  return date;
}
