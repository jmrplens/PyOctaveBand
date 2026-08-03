// Behaviour checks for the first-visit language handling.
//
// Runs the scenarios that decide whether the feature is safe: a first visit
// with a Spanish browser, a visitor who deliberately opened an English URL
// after choosing Spanish, an explicit ?lang=, a dismissal, a crawler and the
// English-only API subtree. The bar offers and never navigates, so every
// scenario also asserts that the URL is the one that was asked for.
//
// Usage: node scripts/check-lang-suggest.mjs [--base http://localhost:4321]
import { BASE_PATH, baseFrom, createExpect, launchBrowser } from './shared/audit.mjs';

const BASE = baseFrom();

const browser = await launchBrowser();

const { expect, failures } = createExpect();

/**
 * Load `path` with a fake language list, an optional pre-seeded localStorage
 * and an optional user agent, and report what the page decided.
 */
async function visit(path, { languages = ['es-ES', 'es'], seed = {}, ua } = {}) {
	// Every scenario gets its own browser context, otherwise localStorage
	// written by one of them leaks into the next and the results are garbage.
	const context = await browser.createBrowserContext();
	const page = await context.newPage();
	if (ua) await page.setUserAgent(ua);
	await page.evaluateOnNewDocument(
		(langs, seeded) => {
			Object.defineProperty(navigator, 'languages', { get: () => langs });
			Object.defineProperty(navigator, 'language', { get: () => langs[0] });
			try {
				for (const [k, v] of Object.entries(seeded)) localStorage.setItem(k, v);
			} catch {}
		},
		languages,
		seed,
	);
	await page.goto(`${BASE}${BASE_PATH}${path}`, { waitUntil: 'networkidle0', timeout: 60000 });
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
expect('banner: first visit, es browser on an EN page', await visit('/signal/levels/levels/'), {
	url: `${BASE_PATH}/signal/levels/levels/`,
	banner: true,
	stored: null,
});

// 2. Same page, English browser: silent.
expect(
	'banner: en browser on an EN page stays silent',
	await visit('/signal/levels/levels/', { languages: ['en-GB', 'en'] }),
	{ url: `${BASE_PATH}/signal/levels/levels/`, banner: false, stored: null },
);

// 3. Spanish page, English browser: the banner offers English.
expect(
	'banner: en browser on an ES page',
	await visit('/es/signal/levels/levels/', { languages: ['en-US', 'en'] }),
	{ url: `${BASE_PATH}/es/signal/levels/levels/`, banner: true, stored: null },
);

// 4. A stored Spanish choice does not fire on an English URL the visitor
//    opened deliberately: it offers, it never navigates.
expect(
	'no trap: stored es choice on an EN url only offers',
	await visit('/signal/levels/levels/', { seed: { 'phonometry:lang': 'es' } }),
	{ url: `${BASE_PATH}/signal/levels/levels/`, banner: true, stored: 'es' },
);

// 5. An explicit ?lang=en is a decision: recorded, and silent.
expect(
	'explicit ?lang=en wins over the browser list',
	await visit('/signal/levels/levels/?lang=en'),
	{ url: `${BASE_PATH}/signal/levels/levels/`, banner: false, stored: 'en' },
);

// 6. A previous dismissal keeps it quiet.
expect(
	'dismissed stays dismissed',
	await visit('/signal/levels/levels/', { seed: { 'phonometry:lang-dismissed': '1' } }),
	{ url: `${BASE_PATH}/signal/levels/levels/`, banner: false, stored: null },
);

// 7. Crawlers see nothing.
expect(
	'crawler user agent is skipped',
	await visit('/signal/levels/levels/', {
		ua: 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
	}),
	{ url: `${BASE_PATH}/signal/levels/levels/`, banner: false, stored: null },
);

// 8. The English-only API subtree opts out entirely.
expect(
	'API reference renders no banner at all',
	await visit('/reference/api/signals/levels/'),
	{ url: `${BASE_PATH}/reference/api/signals/levels/`, banner: 'absent', stored: null },
);

// 9. Nothing ever navigates on its own: the same first visit that a redirect
//    would have moved stays exactly where it was asked to be.
expect(
	'nothing navigates: the ES-preferring first visit stays on the EN url',
	await visit('/start/getting-started/'),
	{ url: `${BASE_PATH}/start/getting-started/`, banner: true, stored: null },
);

// 10. The bar is on top of everything it overlaps. It is fixed across the
//     full width, so on a desktop layout it crosses the sidebar pane and the
//     table of contents. Hit-testing the three points, plus the dismiss
//     button itself, is what catches a stacking regression; the markup alone
//     cannot.
{
	const context = await browser.createBrowserContext();
	const page = await context.newPage();
	await page.setViewport({ width: 1440, height: 900 });
	await page.evaluateOnNewDocument(() => {
		Object.defineProperty(navigator, 'languages', { get: () => ['es-ES', 'es'] });
		Object.defineProperty(navigator, 'language', { get: () => 'es-ES' });
	});
	await page.goto(`${BASE}${BASE_PATH}/signal/levels/levels/`, {
		waitUntil: 'networkidle0',
		timeout: 60000,
	});
	await new Promise((r) => setTimeout(r, 400));
	// The dev server parks its toolbar in the middle of the bottom edge, and
	// it hit-tests. It is not part of the site.
	await page.addStyleTag({ content: 'astro-dev-toolbar{display:none !important}' });
	const hits = await page.evaluate(() => {
		// A regression that keeps the bar off screen is exactly what this check
		// is for, so report it as a failed hit test rather than throwing a raw
		// TypeError that would abort the run and print nothing.
		const bar = document.getElementById('lang-suggest');
		if (!bar || bar.hidden) return { error: 'the bar did not come up' };
		const dismiss = bar.querySelector('[data-lang-dismiss]');
		if (!dismiss) return { error: 'the bar has no [data-lang-dismiss] control' };
		const box = bar.getBoundingClientRect();
		const mid = box.top + box.height / 2;
		const owns = (x, y) => {
			const el = document.elementFromPoint(x, y);
			return !!el && (el === bar || bar.contains(el));
		};
		const close = dismiss.getBoundingClientRect();
		return {
			// Over the sidebar pane, over the content, over the table of contents.
			// Taken off the bar's own box rather than assuming it starts at x=0.
			overSidebar: owns(box.left + 40, mid),
			overContent: owns(box.left + box.width / 2, mid),
			overToc: owns(box.right - 40, mid),
			// The dismiss button itself.
			dismissClickable: owns(close.left + close.width / 2, close.top + close.height / 2),
		};
	});
	expect('the bar is on top of the sidebar, the content and the toc', hits, {
		overSidebar: true,
		overContent: true,
		overToc: true,
		dismissClickable: true,
	});
	await page.close();
	await context.close();
}

await browser.close();
console.log(`\n${failures()} failing scenario(s).`);
process.exit(failures() ? 1 : 0);
