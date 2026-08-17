// Writes src/data/redirects.mjs from the rename history of the content tree.
//
// The map is committed rather than derived at build time on purpose. A build
// time generator cannot go stale, but it also cannot be reviewed: a
// reorganisation would silently rewrite hundreds of published addresses with
// nothing in the diff to look at, and a mis-detected rename would ship as a
// redirect nobody ever read. Committing the map puts every one of those
// addresses in the diff, and check-redirects.mjs derives it again in CI and
// fails if the two disagree, so the file cannot fall behind either.
//
// Usage: node scripts/generate-redirects.mjs   (or `pnpm run redirects`)
import { writeFileSync } from 'node:fs';

import { deriveRedirects, redirectModuleSource, TARGET } from './shared/redirects.mjs';

const { redirects, unresolved } = deriveRedirects();

if (unresolved.length > 0) {
	console.error(
		`FAIL ${unresolved.length} published address(es) no longer served and not traceable to a\n` +
			`     page that is: they were split or dropped rather than renamed, so their\n` +
			`     destination is a judgement call. Add each to SPLIT_DESTINATIONS in\n` +
			`     scripts/shared/redirects.mjs, pointing at the overview of the subsection\n` +
			`     its content became:\n` +
			unresolved.map((route) => `       ${route}`).join('\n'),
	);
	process.exit(1);
}

writeFileSync(TARGET, redirectModuleSource(redirects));
console.log(`[redirects] wrote ${redirects.size} redirects to src/data/redirects.mjs`);
