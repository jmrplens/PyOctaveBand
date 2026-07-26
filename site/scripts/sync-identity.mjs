#!/usr/bin/env node
/**
 * Refreshes the committed snapshot of the canonical `#person` entity.
 *
 * The build fetches the live document (see `astro.config.mjs`); this snapshot
 * is only the offline fallback. Refresh it deliberately — via this script, in
 * its own commit — so the identity this site would ship without a network is
 * visible in review rather than silently frozen at whatever it was on the day
 * the file was first added.
 *
 * Usage:
 *   node scripts/sync-identity.mjs           # write the snapshot
 *   node scripts/sync-identity.mjs --check   # fail if it is stale
 */
import { readFileSync, writeFileSync } from "node:fs";
import process from "node:process";

const SOURCE =
	"https://raw.githubusercontent.com/jmrplens/jmrp.io/main/public/identity/person.jsonld";
const TARGET = new URL("../identity/person.snapshot.json", import.meta.url);

const response = await fetch(SOURCE, { signal: AbortSignal.timeout(15_000) });
if (!response.ok) {
	console.error(`✗ Could not fetch ${SOURCE} — HTTP ${response.status}`);
	process.exit(1);
}
const latest = `${JSON.stringify(await response.json(), null, 2)}\n`;

if (process.argv.includes("--check")) {
	const committed = readFileSync(TARGET, "utf8");
	if (committed !== latest) {
		console.error(
			"✗ identity/person.snapshot.json is stale.\n" +
				"  Refresh it with: pnpm run identity:sync",
		);
		process.exit(1);
	}
	console.log("✓ Identity snapshot matches the canonical document.");
} else {
	writeFileSync(TARGET, latest);
	console.log("✓ Refreshed identity/person.snapshot.json");
}
