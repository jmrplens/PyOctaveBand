// The guides index lists every guide, and says how many there are.
//
// start/guides.md is what the Start topic offers as the map of the library, and
// it is written by hand. Nothing checked it, so it drifted: it linked 100 of the
// 104 guides while stating 104 in the prose and again in the frontmatter
// description, in both languages. Two substantial features were reachable only
// from the sidebar, and the page contradicted itself in the text a search engine
// quotes.
//
// Both halves are checked here. The links, because a guide that is not on the
// map is as good as unwritten for a reader who starts there; and the count,
// because a number in prose is a claim like any other. The spelled-out form is
// checked too, since it is the half nobody remembers to update.
//
// Usage: node scripts/check-guides-index.mjs
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

import { topics } from '../src/data/topics.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const CONTENT = join(here, '..', 'src', 'content', 'docs');

/**
 * The topics whose pages are guides: every one that teaches, which is every
 * topic except the three carrying no teaching prose. Derived rather than
 * listed, for the same reason the byline derives it (src/components/PageTitle).
 */
const NOT_A_DOMAIN = new Set(['start', 'reference', 'api']);
const DOMAINS = topics.map((topic) => topic.id).filter((id) => id && !NOT_A_DOMAIN.has(id));

/** Every markdown file under `dir`, at any depth. */
function walk(dir, out = []) {
	for (const name of readdirSync(dir)) {
		const full = join(dir, name);
		if (statSync(full).isDirectory()) walk(full, out);
		else if (/\.mdx?$/.test(name)) out.push(full);
	}
	return out;
}

/** The guide routes of one locale, in tree order, without the index pages. */
function guideRoutes(root) {
	const routes = [];
	for (const domain of DOMAINS) {
		for (const file of walk(join(root, domain)).sort()) {
			const route = relative(root, file).replace(/\.mdx?$/, '');
			if (!/(^|\/)index$/.test(route)) routes.push(route);
		}
	}
	return routes;
}

/** A count in words, in the two languages the site publishes. */
function spell(n, lang) {
	const table = {
		en: {
			units: ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
				'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen',
				'eighteen', 'nineteen'],
			tens: ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'],
		},
		es: {
			units: ['', 'uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve',
				'diez', 'once', 'doce', 'trece', 'catorce', 'quince', 'dieciséis', 'diecisiete',
				'dieciocho', 'diecinueve'],
			tens: ['', '', 'veinte', 'treinta', 'cuarenta', 'cincuenta', 'sesenta', 'setenta',
				'ochenta', 'noventa'],
		},
	}[lang];
	const below100 = (v) =>
		v < 20
			? table.units[v]
			: table.tens[Math.floor(v / 10)] +
				(v % 10 ? (lang === 'es' ? ' y ' : '-') + table.units[v % 10] : '');
	if (n < 100) return below100(n);
	const rest = n % 100;
	const hundreds = Math.floor(n / 100);
	if (lang === 'es') {
		const head = hundreds === 1 ? (rest ? 'ciento' : 'cien') : `${table.units[hundreds]}cientos`;
		return rest ? `${head} ${below100(rest)}` : head;
	}
	// "a hundred and four" and "one hundred and four" are both the number; the
	// page uses the first, and a rewrite to the second should not fail a check
	// about arithmetic.
	const head = hundreds === 1 ? '(?:a|one) hundred' : `${table.units[hundreds]} hundred`;
	return rest ? `${head} and ${below100(rest)}` : head;
}

const LOCALES = [
	{ lang: 'en', root: CONTENT, word: 'guides' },
	{ lang: 'es', root: join(CONTENT, 'es'), word: 'guías' },
];

let failures = 0;
const fail = (message) => {
	console.error(`FAIL ${message}`);
	failures += 1;
};
const ok = (message) => console.log(`ok   ${message}`);

for (const { lang, root, word } of LOCALES) {
	const routes = guideRoutes(root);
	const index = readFileSync(join(root, 'start', 'guides.md'), 'utf8');

	// The route has to appear as a link destination, not merely as text: a slug
	// in a code block or a comment would satisfy a plain substring while giving
	// the reader nothing to select.
	const prefix = lang === 'es' ? '/phonometry/es/' : '/phonometry/';
	const missing = routes.filter((route) => !index.includes(`](${prefix}${route}/)`));
	if (missing.length > 0) {
		fail(`${lang}: ${missing.length} guide(s) the index does not link: ${missing.join(', ')}`);
	} else {
		ok(`${lang}: the index links all ${routes.length} guides`);
	}

	// Every number that introduces the word, in the prose and in the frontmatter
	// description, has to be the number of guides there are.
	const digits = [...index.matchAll(new RegExp(String.raw`(\d{2,4})\s+${word}`, 'g'))].map((m) =>
		Number(m[1]),
	);
	const wrong = digits.filter((n) => n !== routes.length);
	if (digits.length === 0) fail(`${lang}: the index states no count`);
	else if (wrong.length > 0) {
		fail(`${lang}: the index says ${[...new Set(wrong)].join(', ')} where there are ${routes.length}`);
	} else {
		ok(`${lang}: every count it states in figures says ${routes.length}`);
	}

	// The structured data has to agree with itself and with the page: the list
	// declared ten items over nine entries once, while the description said
	// nine and the frontmatter said ten, all in one file.
	const ld = index.match(/\{\s*\n\s*"@context": "https:\/\/schema\.org"[\s\S]*?\n {6}\}/);
	if (ld) {
		const data = JSON.parse(ld[0]);
		const declared = data.numberOfItems;
		const listed = (data.itemListElement ?? []).length;
		const headings = (index.match(/^## \[/gm) ?? []).length;
		if (declared === listed && listed === headings) {
			ok(`${lang}: the JSON-LD declares ${declared} areas, lists ${listed}, and the page has ${headings} sections`);
		} else {
			fail(`${lang}: numberOfItems ${declared}, itemListElement ${listed}, ## headings ${headings}`);
		}
	} else {
		fail(`${lang}: no ItemList JSON-LD found in the index`);
	}

	// And the same number in words, which is the half nobody updates.
	// `spell` returns a pattern rather than one spelling, since a number has more
	// than one correct wording and the check is about the arithmetic.
	const written = spell(routes.length, lang);
	if (new RegExp(String.raw`\b${written}\s+${word}`, 'i').test(index)) {
		ok(`${lang}: the count in words agrees with ${routes.length}`);
	} else {
		fail(`${lang}: the index does not say "${written} ${word}" anywhere, so its count in words has drifted from ${routes.length}`);
	}
}

console.log(`\n${failures} failing check(s).`);
process.exit(failures ? 1 : 0);
