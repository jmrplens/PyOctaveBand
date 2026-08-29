// Publishes a clean markdown copy of every page next to its HTML, so an agent
// can fetch one page as text instead of choosing between parsing the rendered
// HTML and downloading the whole corpus.
//
// Why this exists: a page's HTML is roughly 90% chrome. The sidebar alone is
// ~76 kB on every one of the 442 pages, syntax highlighting adds hundreds of
// kilobytes of inline attributes to the deepest guides, and each KaTeX formula
// is still markup-heavy even now that it serialises as text once. The only clean
// alternative used to be llms-full.txt, which is far too large for a single
// fetch. Now `/signals/levels/levels/index.md` sits beside `/signals/levels/levels/`, and
// Head.astro advertises it with <link rel="alternate" type="text/markdown">.
//
// Source preference: docs/<page>.md where it exists (already plain markdown,
// the same text the llms artifacts carry), otherwise the site source with its
// frontmatter and MDX machinery stripped.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { glob } from "node:fs/promises";
import { hasMarkdownCopy, routeOf } from "../src/lib/page-markdown.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const siteRoot = join(here, "..");
const repoRoot = join(siteRoot, "..");
const contentDir = join(siteRoot, "src", "content", "docs");
const docsDir = join(repoRoot, "docs");
const outDir = process.env.ASTRO_OUT_DIR ?? join(siteRoot, "dist");
const SITE_URL = "https://jmrplens.github.io/phonometry";

/** Strip YAML frontmatter and return the body. */
function stripFrontmatter(text) {
  if (!text.startsWith("---")) return text;
  const end = text.indexOf("\n---", 3);
  return end === -1 ? text : text.slice(text.indexOf("\n", end + 1) + 1);
}

/**
 * Remove the MDX-only constructs that are noise as text: import statements,
 * expression containers, and self-closing component tags. Component tags that
 * wrap content keep their children, since that is real prose.
 */
function stripMdx(text) {
  return text
    .replace(/^import\s.+?$/gm, "")
    .replace(/^export\s.+?$/gm, "")
    // Self-closing components: <ThemeImage ... />, <Video ... />
    .replace(/<[A-Z][\w.]*\b[^>]*\/>/g, "")
    // Opening and closing tags of wrapper components, keeping the children.
    .replace(/<\/?[A-Z][\w.]*\b[^>]*>/g, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/**
 * Route for a content file: `guides/levels.mdx` -> `guides/levels`. The rule
 * itself is in src/lib/page-markdown.mjs, because the page has to reach the
 * same answer about itself and the two used to be written out separately.
 */
const routeOfFile = (file) => routeOf(relative(contentDir, file));

// The two generated evidence documents live at the repository root of the
// mirror and are published at these routes. Both the link rewriter and the
// source lookup need the mapping: the conformance page is now MDX that renders
// its rows from `docs/conformance.json` through a component, so stripping the
// MDX out of the page source would leave an agent with the introduction and no
// evidence. `docs/CONFORMANCE.md` is the text copy of the same rows.
const TRANSPLANTED = {
  CONFORMANCE: "reference/conformance",
  ERRATA: "reference/errata",
};

/**
 * Rewrite links that only work inside the docs/ folder.
 *
 * The mirrored files link each other by relative path, as
 * `[x](../filters/filter-banks.md)`, which resolves when reading the
 * repository and resolves to nothing when the same text is served as a
 * standalone page copy. The mirror is laid out like the site, so the link
 * target resolved against the page's own folder is the route.
 */
function absolutize(text, fromRoute) {
  const here = fromRoute.includes("/") ? fromRoute.replace(/\/[^/]*$/, "") : "";
  return text
    .split("\n")
    .filter((line) => !line.startsWith("← [Documentation index]"))
    .join("\n")
    .replace(/\]\(((?:\.\.\/)*[\w./-]+)\.md(#[\w-]+)?\)/g, (whole, target, anchor = "") => {
      const base = target.replace(/^.*\//, "");
      if (base === "README") return `](${SITE_URL}/${anchor})`;
      if (base === "CONTRIBUTING") {
        return `](https://github.com/jmrplens/phonometry/blob/main/CONTRIBUTING.md${anchor})`;
      }
      const parts = here ? here.split("/") : [];
      for (const step of target.split("/")) {
        if (step === "..") parts.pop();
        else if (step !== ".") parts.push(step);
      }
      const resolved = parts.join("/").replace(/\/index$/, "");
      // The two generated documents live at the repository root in the mirror
      // and are transplanted into these routes on the site.
      const route = TRANSPLANTED[resolved] ?? resolved;
      return `](${SITE_URL}/${route}/${anchor})`;
    });
}

// The mirror is laid out like the site, so a page's copy is at the same route:
// `docs/signals/filters/filter-banks.md` for `/signals/filters/filter-banks/`.
const docsMirror = new Map();
if (existsSync(docsDir)) {
  for (const name of await Array.fromAsync(glob("**/*.md", { cwd: docsDir }))) {
    const bare = name.replace(/\\/g, "/").replace(/\.md$/, "").replace(/\/index$/, "");
    docsMirror.set(TRANSPLANTED[bare] ?? bare, join(docsDir, name));
  }
}

let written = 0;
for await (const file of glob("**/*.{md,mdx}", { cwd: contentDir })) {
  const abs = join(contentDir, file);
  const route = routeOfFile(abs);
  // The two splash pages are a component layout, not prose; there is no
  // meaningful markdown to serve for them. The same test decides what the page
  // itself advertises (src/lib/page-markdown.mjs).
  if (!hasMarkdownCopy(route)) continue;

  const isSpanish = route === "es" || route.startsWith("es/");
  const mirror = isSpanish ? undefined : docsMirror.get(route);

  const raw = mirror
    ? readFileSync(mirror, "utf8")
    : stripMdx(stripFrontmatter(readFileSync(abs, "utf8")));
  const body = absolutize(raw, route);

  const canonical = `${SITE_URL}/${route}/`;
  const text = `<!-- canonical: ${canonical} -->\nSource: ${canonical}\n\n${body.trim()}\n`;

  const target = join(outDir, route, "index.md");
  mkdirSync(dirname(target), { recursive: true });
  writeFileSync(target, text, "utf8");
  written += 1;
}

console.log(`[emit-page-markdown] wrote ${written} .md page copies into ${relative(siteRoot, outDir) || "dist"}`);
