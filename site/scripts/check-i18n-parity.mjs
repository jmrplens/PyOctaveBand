// Validates EN/ES translation parity: every English page under
// src/content/docs/ (outside es/) must have a Spanish counterpart at the same
// relative path under src/content/docs/es/, and vice-versa. Exits non-zero on
// any mismatch so CI catches a missing or orphaned translation.
//
// It also holds the one authoring rule that only bites in Spanish: figures come
// from ThemeImage, because a figure written as markdown gets a zoom control the
// plugin names in English and nothing translates it. See below.
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";

const docsDir = fileURLToPath(new URL("../src/content/docs", import.meta.url));
const esDir = join(docsDir, "es");

function walk(dir) {
	const out = [];
	for (const entry of readdirSync(dir)) {
		const p = join(dir, entry);
		if (statSync(p).isDirectory()) {
			if (p === esDir) continue; // skip the ES subtree when collecting EN
			out.push(...walk(p));
		} else if (/\.(md|mdx)$/.test(entry)) {
			out.push(p);
		}
	}
	return out;
}

// The API reference (reference/api/) is published in English in both
// languages, by decision: it is generated from the docstrings, which are the
// code's own text, and translating it would mean keeping a second copy of
// 146 pages in step with every signature change. Starlight's locale fallback
// serves it on /es/, where the untranslated notice says so (i18n/es.json).
// Everything else is bilingual and this check is what enforces that, so the
// exemption is this subtree and nothing else.
const apiRef = join("reference", "api");
const isApiRef = (p) => p === apiRef || p.startsWith(apiRef + sep);

const enPages = walk(docsDir)
	.map((p) => relative(docsDir, p))
	.filter((p) => !isApiRef(p));
const esPages = walk(esDir).map((p) => relative(esDir, p));

const enSet = new Set(enPages);
const esSet = new Set(esPages);

const missingEs = enPages.filter((p) => !esSet.has(p)).sort();
const orphanEs = esPages.filter((p) => !enSet.has(p)).sort();

if (missingEs.length || orphanEs.length) {
	console.error("i18n parity check FAILED:");
	for (const p of missingEs)
		console.error(`  EN page missing ES translation: ${p}`);
	for (const p of orphanEs)
		console.error(`  ES page with no EN original:    es/${p}`);
	process.exit(1);
}

// Figures on a Spanish page have to come from ThemeImage.astro.
//
// starlight-image-zoom finds images by walking the markdown tree and gives
// each one a zoom control named "Zoom image: <alt>", hard-coded in English
// with no translation hook. A figure written as markdown or as raw HTML on a
// Spanish page therefore ships a control whose accessible name puts an English
// verb in front of Spanish alt text, which is a WCAG 3.1.2 defect no automated
// audit can see: the name is there and it is not empty. ThemeImage wraps its
// own pair and hands it to ZoomI18n.astro, which translates the name and fails
// the build if the plugin ever renames it, so every figure goes through it.
const figures = [];
for (const page of esPages) {
	const source = readFileSync(join(esDir, page), "utf8")
		// Fenced blocks and inline code are samples of markup, not markup.
		.replace(/^```[\s\S]*?^```/gm, "")
		.replace(/`[^`\n]*`/g, "");
	// Every markdown image opens with the same two characters, whatever comes
	// after them: the inline `![alt](src)`, the two reference forms
	// `![alt][label]` and `![alt][]`, and the shortcut `![label]` whose target
	// is a link definition somewhere else in the file. Only the inline form
	// used to be looked for, so a figure written as a reference walked past
	// this check and shipped the English zoom control anyway. Matching what
	// opens all four is both shorter and complete, and Spanish prose has no
	// other use for the sequence once code has been taken out above.
	if (source.includes("![") || /<img\b/.test(source)) {
		figures.push(page);
	}
}

if (figures.length) {
	console.error("i18n parity check FAILED:");
	for (const p of figures)
		console.error(
			`  ES page authors a figure outside ThemeImage: es/${p}` +
				" (its zoom control would be named in English)",
		);
	process.exit(1);
}

console.log(
	`i18n parity OK: ${enPages.length} EN pages each have an ES translation,` +
		` and all ${esPages.length} ES pages take their figures from ThemeImage.`,
);
