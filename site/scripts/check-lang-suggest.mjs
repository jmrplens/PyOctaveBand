// Behaviour checks for the first-visit language handling.
//
// Runs the scenarios that decide whether the feature is safe: a first visit
// with a Spanish browser, a visitor who deliberately opened an English URL
// after choosing Spanish, an explicit ?lang=, a dismissal, a crawler, the
// English-only API subtree, and both redirect guards (once per session, never
// after a stored choice).
//
// Usage: node scripts/check-lang-suggest.mjs [--base http://localhost:4322]
import { readdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const siteDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(import.meta.url);
const store = join(siteDir, 'node_modules', '.pnpm');
const pkg = readdirSync(store).find((d) => /^puppeteer@/.test(d));
const puppeteer = require(join(store, pkg, 'node_modules', 'puppeteer'));

const baseIdx = process.argv.indexOf('--base');
const BASE = baseIdx === -1 ? 'http://localhost:4322' : process.argv[baseIdx + 1];

const browser = await puppeteer.launch({
	headless: true,
	args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'],
});

let failures = 0;
const expect = (name, actual, wanted) => {
	const ok = JSON.stringify(actual) === JSON.stringify(wanted);
	if (!ok) failures++;
	console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}\n       got ${JSON.stringify(actual)}, want ${JSON.stringify(wanted)}`);
};

/**
 * Load `path` with a fake language list, an optional pre-seeded localStorage
 * and an optional user agent, and report what the page decided.
 */
async function visit(path, { languages = ['es-ES', 'es'], seed = {}, ua, mode } = {}) {
	// Every scenario gets its own browser context, otherwise localStorage
	// written by one of them leaks into the next and the results are garbage.
	const context = await browser.createBrowserContext();
	const page = await context.newPage();
	if (ua) await page.setUserAgent(ua);
	await page.evaluateOnNewDocument(
		(langs, seeded, m) => {
			Object.defineProperty(navigator, 'languages', { get: () => langs });
			Object.defineProperty(navigator, 'language', { get: () => langs[0] });
			try {
				for (const [k, v] of Object.entries(seeded)) localStorage.setItem(k, v);
				if (m) localStorage.setItem('phonometry:lang-mode', m);
			} catch {}
		},
		languages,
		seed,
		mode,
	);
	await page.goto(`${BASE}/phonometry${path}`, { waitUntil: 'networkidle0', timeout: 60000 });
	await new Promise((r) => setTimeout(r, 400));
	const result = await page.evaluate(() => {
		const b = document.getElementById('lang-suggest');
		return {
			url: location.pathname,
			banner: b ? !b.hidden : 'absent',
			stored: localStorage.getItem('phonometry:lang'),
		};
	});
	await page.close();
	await context.close();
	return result;
}

// 1. First visit, Spanish browser, English page: the banner offers Spanish
//    and nothing navigates.
expect('banner: first visit, es browser on an EN page', await visit('/guides/levels/'), {
	url: '/phonometry/guides/levels/',
	banner: true,
	stored: null,
});

// 2. Same page, English browser: silent.
expect(
	'banner: en browser on an EN page stays silent',
	await visit('/guides/levels/', { languages: ['en-GB', 'en'] }),
	{ url: '/phonometry/guides/levels/', banner: false, stored: null },
);

// 3. Spanish page, English browser: the banner offers English.
expect(
	'banner: en browser on an ES page',
	await visit('/es/guides/levels/', { languages: ['en-US', 'en'] }),
	{ url: '/phonometry/es/guides/levels/', banner: true, stored: null },
);

// 4. A stored Spanish choice does not fire on an English URL the visitor
//    opened deliberately: it offers, it never navigates.
expect(
	'no trap: stored es choice on an EN url only offers',
	await visit('/guides/levels/', { seed: { 'phonometry:lang': 'es' } }),
	{ url: '/phonometry/guides/levels/', banner: true, stored: 'es' },
);

// 5. An explicit ?lang=en is a decision: recorded, and silent.
expect(
	'explicit ?lang=en wins over the browser list',
	await visit('/guides/levels/?lang=en'),
	{ url: '/phonometry/guides/levels/', banner: false, stored: 'en' },
);

// 6. A previous dismissal keeps it quiet.
expect(
	'dismissed stays dismissed',
	await visit('/guides/levels/', { seed: { 'phonometry:lang-dismissed': '1' } }),
	{ url: '/phonometry/guides/levels/', banner: false, stored: null },
);

// 7. Crawlers see nothing.
expect(
	'crawler user agent is skipped',
	await visit('/guides/levels/', {
		ua: 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
	}),
	{ url: '/phonometry/guides/levels/', banner: false, stored: null },
);

// 8. The English-only API subtree opts out entirely.
expect(
	'API reference renders no banner at all',
	await visit('/reference/api/levels/levels/'),
	{ url: '/phonometry/reference/api/levels/levels/', banner: 'absent', stored: null },
);

// 9. Redirect mode, first visit with nothing stored: one hop to /es/.
expect(
	'redirect: first visit hops once',
	await visit('/guides/levels/', { mode: 'redirect' }),
	{ url: '/phonometry/es/guides/levels/', banner: false, stored: null },
);

// 10. Redirect mode with a stored choice: degrades to the banner, no hop.
expect(
	'redirect: a stored choice never navigates',
	await visit('/guides/levels/', { mode: 'redirect', seed: { 'phonometry:lang': 'es' } }),
	{ url: '/phonometry/guides/levels/', banner: true, stored: 'es' },
);

// 11. Redirect mode on the page it would redirect to: nothing to do, and the
//     session guard means a second load cannot bounce back.
expect(
	'redirect: no loop on the target page',
	await visit('/es/guides/levels/', { mode: 'redirect' }),
	{ url: '/phonometry/es/guides/levels/', banner: false, stored: null },
);

await browser.close();
console.log(`\n${failures} failing scenario(s).`);
process.exit(failures ? 1 : 0);
