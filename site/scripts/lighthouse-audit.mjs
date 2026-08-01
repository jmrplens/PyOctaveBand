// Local Lighthouse audit over the built site, in the spirit of the runner
// jmrp.io uses in its CI: a fixed URL sample, the lighthouse CLI against a
// local preview server, JSON reports on disk and a compact console summary
// with the scores and every remaining opportunity/diagnostic.
//
// Usage:
//   pnpm build                 # the audit runs against dist/
//   pnpm run lighthouse        # mobile (default)
//   pnpm run lighthouse -- --desktop
//   BASE_URL=https://jmrplens.github.io node scripts/lighthouse-audit.mjs
//
// Without BASE_URL the script starts `astro preview` itself on PORT (default
// 4323) and stops it when done. Reports land in lighthouse-results/
// (gitignored); the console table is the deliverable.
import { execFileSync, spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const siteRoot = join(here, "..");
const RESULTS_DIR = join(siteRoot, "lighthouse-results");
const PORT = process.env.PORT ?? "4323";
const BASE = "/phonometry";

// A deliberate sample, not the whole site: the two locale homes, the hub and
// content shapes that exist (splash, guide with math and figures, the
// heaviest guide, an API page, the conformance table) so a regression in any
// page template shows up without auditing 450 URLs.
const AUDIT_PATHS = [
  `${BASE}/`,
  `${BASE}/es/`,
  `${BASE}/getting-started/`,
  `${BASE}/guides/`,
  `${BASE}/guides/calibration/`,
  `${BASE}/guides/insulation-field/`,
  `${BASE}/reference/api/signal/levels/`,
  `${BASE}/reference/conformance/`,
];

const isDesktop = process.argv.includes("--desktop");
const externalBase = process.env.BASE_URL;
const baseUrl = externalBase ?? `http://localhost:${PORT}`;

/** Start `astro preview` and resolve once it serves the base path. */
async function startPreview() {
  const child = spawn("pnpm", ["exec", "astro", "preview", "--port", PORT], {
    cwd: siteRoot,
    stdio: "ignore",
    detached: true,
  });
  for (let i = 0; i < 40; i++) {
    try {
      const res = await fetch(`${baseUrl}${BASE}/`);
      if (res.ok) return child;
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  child.kill();
  throw new Error(`astro preview did not come up on port ${PORT}`);
}

function runLighthouse(url, outPath) {
  execFileSync(
    "pnpm",
    [
      "exec",
      "lighthouse",
      url,
      `--output-path=${outPath}`,
      "--output=json",
      "--quiet",
      `--chrome-flags=--headless=new --no-sandbox --disable-gpu`,
      ...(isDesktop ? ["--preset=desktop"] : []),
    ],
    { cwd: siteRoot, stdio: ["ignore", "ignore", "inherit"] },
  );
}

const scoreCell = (v) => (v === null ? "  --" : String(Math.round(v * 100)).padStart(4));

if (!existsSync(RESULTS_DIR)) mkdirSync(RESULTS_DIR);
if (!externalBase && !existsSync(join(siteRoot, "dist"))) {
  console.error("No dist/ found. Run `pnpm build` first.");
  process.exit(1);
}

const preview = externalBase ? null : await startPreview();
const rows = [];
// audit id -> { savingsMs, urls } for everything Lighthouse still flags.
const flagged = new Map();

try {
  for (const path of AUDIT_PATHS) {
    const url = `${baseUrl}${path}`;
    const slug = path.replaceAll("/", "-").replace(/^-|-$/g, "") || "home";
    const outPath = join(RESULTS_DIR, `${slug}${isDesktop ? "-desktop" : "-mobile"}.json`);
    process.stdout.write(`auditing ${path} ... `);
    runLighthouse(url, outPath);
    const report = JSON.parse(readFileSync(outPath, "utf8"));

    const cat = report.categories;
    rows.push({
      path,
      perf: cat.performance?.score ?? null,
      a11y: cat.accessibility?.score ?? null,
      bp: cat["best-practices"]?.score ?? null,
      seo: cat.seo?.score ?? null,
    });
    console.log("done");

    for (const audit of Object.values(report.audits)) {
      const failing =
        (audit.scoreDisplayMode === "binary" && audit.score === 0) ||
        (audit.scoreDisplayMode === "numeric" && audit.score !== null && audit.score < 0.9) ||
        (audit.details?.overallSavingsMs ?? 0) > 100;
      if (!failing) continue;
      const entry = flagged.get(audit.id) ?? {
        title: audit.title,
        savingsMs: 0,
        urls: new Set(),
      };
      entry.savingsMs = Math.max(entry.savingsMs, audit.details?.overallSavingsMs ?? 0);
      entry.urls.add(path);
      flagged.set(audit.id, entry);
    }
  }
} finally {
  if (preview) process.kill(-preview.pid, "SIGTERM");
}

console.log(`\n${isDesktop ? "Desktop" : "Mobile"} scores (${baseUrl}):`);
console.log("Perf A11y BP   SEO   Path");
for (const r of rows) {
  console.log(
    `${scoreCell(r.perf)} ${scoreCell(r.a11y)} ${scoreCell(r.bp)} ${scoreCell(r.seo)}  ${r.path}`,
  );
}

if (flagged.size) {
  console.log("\nRemaining flags (audit, worst savings, affected paths):");
  const sorted = [...flagged.entries()].sort((a, b) => b[1].savingsMs - a[1].savingsMs);
  for (const [id, f] of sorted) {
    const ms = f.savingsMs ? ` ~${Math.round(f.savingsMs)} ms` : "";
    console.log(`- ${id}${ms} :: ${f.title} [${[...f.urls].join(", ")}]`);
  }
} else {
  console.log("\nNo audits flagged.");
}

const worst = Math.min(...rows.flatMap((r) => [r.perf, r.a11y, r.bp, r.seo].filter((v) => v !== null)));
process.exitCode = worst < 0.9 ? 1 : 0;
