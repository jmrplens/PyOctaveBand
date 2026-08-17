// The structured data the site actually publishes.
//
// Every page carries a JSON-LD graph: the site-wide Person / WebSite /
// SoftwareApplication / SoftwareSourceCode nodes from astro.config.mjs, and a
// per-page article and breadcrumb trail from src/components/Head.astro. It is
// the only part of a page written for machines alone, and nothing in this
// repository read it. html-validate does not look inside a <script>, pa11y has
// no reason to, and the link checker sees no links there: a graph could stop
// being JSON, lose a required field or point at nothing at all and every check
// in the pipeline would stay green.
//
// It did. `Head.astro` resolved the section name from a map written out by
// hand beside the topic list, the signal chapter was renamed `signals` and
// only the list was swept, so all 54 pages of the largest topic on the site
// shipped with no `articleSection` for as long as nobody looked. The
// breadcrumb trail on those same pages named the section correctly throughout,
// because it reads src/data/topics.mjs.
//
// So this check reads topics.mjs too, and the built HTML, and nothing else: it
// derives what each page should say from its own URL and asserts it against
// what the page emits. A check that read Head.astro's idea of the answer would
// have agreed with the defect.
//
// Usage: node scripts/check-structured-data.mjs [dist-directory]
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

import { basePath, siteUrl } from '../src/data/site.mjs';
import { topicForSlug } from '../src/data/topics.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const DIST = process.argv[2] ? join(process.cwd(), process.argv[2]) : join(here, '..', 'dist');
const ORIGIN = `${siteUrl}${basePath}`;

let failures = 0;
const fail = (message) => {
	console.error(`FAIL ${message}`);
	failures += 1;
};
const ok = (message) => console.log(`ok   ${message}`);

/** Every .html file under `dir`, at any depth, as paths relative to DIST. */
function walk(dir, out = []) {
	for (const name of readdirSync(dir)) {
		const full = join(dir, name);
		if (statSync(full).isDirectory()) walk(full, out);
		else if (name.endsWith('.html')) out.push(relative(DIST, full).split(sep).join('/'));
	}
	return out;
}

/**
 * The route a built file is served at: `signals/levels/index.html` is
 * `/signals/levels/`, and the one page that is not an index, `404.html`, is
 * `/404/`. This is deliberately computed from the file tree rather than read
 * out of the page, so a page whose own `url` is wrong cannot hide it.
 */
function routeOf(file) {
	const withoutIndex = file.replace(/(^|\/)index\.html$/, '$1').replace(/\.html$/, '/');
	return `/${withoutIndex}`.replace(/\/{2,}/g, '/');
}

/** `@type` as an array, whichever of the two shapes a node uses. */
const typesOf = (node) => (Array.isArray(node['@type']) ? node['@type'] : [node['@type']]);

/** The one node of a page carrying `type`, or undefined. */
const nodeOfType = (nodes, type) => nodes.find((node) => typesOf(node).includes(type));

/** Fields every article node carries, whatever its type. */
const ARTICLE_FIELDS = [
	'@id',
	'about',
	'author',
	'dateModified',
	'datePublished',
	'description',
	'headline',
	'image',
	'inLanguage',
	'isPartOf',
	'mainEntityOfPage',
	'publisher',
	'speakable',
	'url',
	'wordCount',
];

/** The site-wide nodes, by the `@id` the per-page nodes point at. */
const SITE_NODES = ['Person', 'WebSite', 'SoftwareApplication', 'SoftwareSourceCode'];

const files = walk(DIST).sort();
if (files.length === 0) throw new Error(`no HTML under ${DIST}: build the site first`);

// Tallies, so a check that silently stopped looking at anything says so.
const seen = { pages: 0, redirects: 0, blocks: 0, sections: 0, guides: 0, api: 0 };
const missingSection = [];
const wrongSection = [];

for (const file of files) {
	const html = readFileSync(join(DIST, file), 'utf8');

	// The legacy redirect stubs are a <meta refresh> and nothing else. They
	// carry no graph on purpose and are not pages a reader lands on.
	if (/http-equiv="refresh"/.test(html)) {
		seen.redirects += 1;
		const stray = /<script[^>]*type="application\/ld\+json"/.test(html);
		if (stray) fail(`${file}: a redirect stub publishes structured data`);
		continue;
	}
	seen.pages += 1;

	const route = routeOf(file);
	const lang = route === '/es/' || route.startsWith('/es/') ? 'es' : 'en';
	const slug = route.replace(/^\/(es\/)?/, '').replace(/\/$/, '');
	const canonical = `${ORIGIN}${route}`;
	const is404 = slug === '404';
	const isHome = slug === '';

	// Parsing is the whole point: a JSON-LD consumer reads each <script> as one
	// document, and a block that does not parse is not "mostly fine", it is a
	// page with no structured data at all.
	const blocks = [...html.matchAll(/<script[^>]*type="application\/ld\+json"[^>]*>([\s\S]*?)<\/script>/g)];
	if (blocks.length === 0) {
		fail(`${route}: no application/ld+json block`);
		continue;
	}
	const nodes = [];
	let broken = false;
	for (const [, body] of blocks) {
		seen.blocks += 1;
		let doc;
		try {
			doc = JSON.parse(body);
		} catch (error) {
			fail(`${route}: an ld+json block is not valid JSON (${error.message})`);
			broken = true;
			continue;
		}
		if (doc['@context'] !== 'https://schema.org') {
			fail(`${route}: an ld+json block declares @context ${JSON.stringify(doc['@context'])}`);
		}
		const graph = Array.isArray(doc['@graph']) ? doc['@graph'] : [doc];
		for (const node of graph) {
			if (!node || typeof node !== 'object' || !node['@type']) {
				fail(`${route}: a node in the graph has no @type`);
				continue;
			}
			nodes.push(node);
		}
	}
	if (broken) continue;

	// An `@id` names one thing. Two nodes sharing one on a page is a merge
	// conflict a consumer resolves by picking whichever it read last.
	const ids = nodes.map((node) => node['@id']).filter(Boolean);
	const duplicated = ids.filter((id, i) => ids.indexOf(id) !== i);
	if (duplicated.length > 0) {
		fail(`${route}: ${[...new Set(duplicated)].join(', ')} defined more than once`);
	}

	for (const type of SITE_NODES) {
		if (!nodeOfType(nodes, type)) fail(`${route}: no ${type} node`);
	}

	// The error page is not an article and must not be indexed as one, which is
	// the whole reason Head.astro special-cases it.
	if (is404) {
		for (const type of ['TechArticle', 'APIReference', 'BreadcrumbList']) {
			if (nodeOfType(nodes, type)) fail(`${route}: the error page publishes a ${type} node`);
		}
		continue;
	}

	if (isHome) {
		const page = nodeOfType(nodes, 'WebPage');
		if (!page) fail(`${route}: the home page has no WebPage node`);
		else {
			if (page['@id'] !== `${canonical}#webpage`) {
				fail(`${route}: WebPage @id is ${page['@id']}, not ${canonical}#webpage`);
			}
			if (page.inLanguage !== lang) {
				fail(`${route}: WebPage inLanguage is ${JSON.stringify(page.inLanguage)}, not ${lang}`);
			}
			if (!page.speakable) fail(`${route}: the home page names no speakable selector`);
		}
		continue;
	}

	// Every other page is an article of some kind.
	const article = nodes.find((node) =>
		typesOf(node).some((type) => ['TechArticle', 'APIReference', 'CollectionPage'].includes(type)),
	);
	if (!article) {
		fail(`${route}: no TechArticle, APIReference or CollectionPage node`);
		continue;
	}
	for (const field of ARTICLE_FIELDS) {
		if (article[field] === undefined) fail(`${route}: the article node has no ${field}`);
	}
	if (article['@id'] !== `${canonical}#article`) {
		fail(`${route}: article @id is ${article['@id']}, not ${canonical}#article`);
	}
	if (article.url !== canonical) fail(`${route}: article url is ${article.url}, not ${canonical}`);
	if (article.inLanguage !== lang) {
		fail(`${route}: article inLanguage is ${JSON.stringify(article.inLanguage)}, not ${lang}`);
	}

	// The two locales are one work. English pages point forward to their
	// translation, Spanish pages point back at the original; a page carrying
	// both, or neither, is claiming something else.
	const forward = article.workTranslation;
	const backward = article.translationOfWork;
	const expected = lang === 'en' ? 'workTranslation' : 'translationOfWork';
	if ((forward ? 1 : 0) + (backward ? 1 : 0) !== 1) {
		fail(`${route}: the article declares ${forward && backward ? 'both' : 'neither'} of workTranslation and translationOfWork`);
	} else if (!article[expected]) {
		fail(`${route}: a ${lang} page declares ${expected === 'workTranslation' ? 'translationOfWork' : 'workTranslation'}`);
	}

	// The generated API tree is documentation of modules, not a guide.
	if (slug.startsWith('reference/api/')) {
		seen.api += 1;
		if (!typesOf(article).includes('APIReference')) {
			fail(`${route}: an API page is typed ${JSON.stringify(article['@type'])}`);
		}
	}

	// The trail: contiguous positions from one, and the last crumb is the page
	// itself. A trail that ends anywhere else is describing another page.
	const crumbs = nodeOfType(nodes, 'BreadcrumbList');
	if (!crumbs) fail(`${route}: no BreadcrumbList node`);
	else {
		const items = crumbs.itemListElement ?? [];
		const positions = items.map((item) => item.position);
		const contiguous = positions.every((position, i) => position === i + 1);
		if (items.length < 2 || !contiguous) {
			fail(`${route}: breadcrumb positions are ${JSON.stringify(positions)}`);
		}
		const last = items[items.length - 1];
		if (last && last.item !== canonical) {
			fail(`${route}: the last crumb points at ${last.item}, not ${canonical}`);
		}
		const dangling = items.filter((item) => !String(item.item ?? '').startsWith(ORIGIN));
		if (dangling.length > 0) {
			fail(`${route}: ${dangling.length} crumb(s) point outside the site`);
		}
	}

	// The section, which is what this check exists for. The name comes from
	// src/data/topics.mjs, in the page's own language, and every page a topic
	// owns has to carry it.
	const topic = topicForSlug(slug);
	if (topic) {
		const wanted = topic.label[lang];
		if (article.articleSection === undefined) missingSection.push(route);
		else if (article.articleSection !== wanted) {
			wrongSection.push(`${route}: ${JSON.stringify(article.articleSection)} for ${JSON.stringify(wanted)}`);
		} else seen.sections += 1;

		// A topic that teaches publishes guides, and a guide is a learning
		// resource as well as an article.
		if (topic.isDomain) {
			seen.guides += 1;
			if (!typesOf(article).includes('LearningResource')) {
				fail(`${route}: a guide is typed ${JSON.stringify(article['@type'])} with no LearningResource`);
			}
			if (article.learningResourceType !== 'Guide') {
				fail(`${route}: a guide declares learningResourceType ${JSON.stringify(article.learningResourceType)}`);
			}
		}
	} else if (article.articleSection !== undefined) {
		fail(`${route}: belongs to no topic but claims the section ${JSON.stringify(article.articleSection)}`);
	}
}

// Reported per topic rather than per page: the defect that prompted this check
// took out an entire topic at once, and 54 identical lines would have buried
// the one fact worth reading.
const byTopic = (routes) => {
	const groups = new Map();
	for (const route of routes) {
		const slug = route.replace(/^\/(es\/)?/, '').replace(/\/$/, '');
		const id = topicForSlug(slug)?.id ?? '(none)';
		groups.set(id, (groups.get(id) ?? 0) + 1);
	}
	return [...groups].map(([id, n]) => `${id} (${n} pages)`).join(', ');
};

if (missingSection.length > 0) {
	fail(`${missingSection.length} page(s) under a topic publish no articleSection: ${byTopic(missingSection)}`);
	console.error(`     first: ${missingSection.slice(0, 3).join(', ')}`);
}
if (wrongSection.length > 0) {
	fail(`${wrongSection.length} page(s) name a section topics.mjs does not: ${wrongSection.slice(0, 3).join('; ')}`);
}
if (missingSection.length === 0 && wrongSection.length === 0) {
	ok(`every one of the ${seen.sections} pages under a topic names its section as topics.mjs does`);
}

ok(`parsed ${seen.blocks} ld+json blocks across ${seen.pages} pages (${seen.redirects} redirect stubs skipped)`);
ok(`${seen.guides} guide pages typed as learning resources, ${seen.api} API pages as APIReference`);

console.log(`\n${failures} failing check(s).`);
process.exit(failures ? 1 : 0);
