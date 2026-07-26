# Site redesign: palette, front page, mobile header, language suggestion

Experimental branch `feat/site-redesign`. It does not merge and has no PR. It
exists so I can look at the site as a whole: what colour it should be, what
the landing page should do, and two pieces of behaviour the site was missing
on phones and for Spanish-speaking visitors.

The branch started with three colour directions and three language modes
behind a floating prototype switcher. Both questions are now decided, so the
code carries one answer: the palette is `instrument` and the language handling
is the banner. The switcher, the two discarded palettes and the two discarded
language modes are gone from the working tree and stay in the branch history;
the last commit that still contains them is **`1113e19c`**, so any of them can
be read back with, for example:

```
git show 1113e19c:site/src/styles/theme-directions.css   # all three palettes
git show 1113e19c:site/src/components/PrototypeSwitcher.astro
git show 1113e19c:site/src/components/LangSuggest.astro  # banner/redirect/off
```

Nothing about the 1220 committed figures changes: the palette was chosen
*around* the figures, not the other way round, and section 1 records what the
figures turned out to actually be.

## 1. Palette

The site used to ride Starlight's defaults: hue 224 greys and a 100 %
saturated blue-violet accent that belongs to a startup landing page, not to a
library whose output is calibration numbers. It now uses one palette,
`instrument`, in `site/src/styles/theme.css`, covering the light and the dark
theme:

| | Light | Dark |
| :-- | :-- | :-- |
| Page ground | `#ffffff` | `#0d1114` |
| Surfaces (nav, sidebar, cards, code) | `#f3f7f9` / `#f4f8fa` | `#141b20` |
| Accent | `#0a6f8c` | `#35b8d8` (text `#b3e6f4`) |

Cool steel greys and one saturated deep-cyan indicator, the way the front
panel of a measuring instrument is put together: neutral metal, a single
colour that means something.

### What the figures dictate

The palette was not a free choice. I measured what the committed figures
actually contain before picking anything, and the first measurement corrected
an assumption I had made:

- **Every committed figure is an opaque plate.** The light ones carry a
  `#ffffff` rectangle, which is expected. The dark matplotlib ones (404 of
  506) emit their figure patch as `<path d="..."/>` with *no fill attribute
  at all*, and an SVG path with no fill is painted **black**. They are not
  transparent, whatever the intent was when they were generated. I confirmed
  it by sampling the rendered page: inside a dark figure the pixels are
  `#000000`, not the page ground.
- The remaining 102 dark figures are the hand-authored diagrams
  (`scripts/generate_diagrams.py`), whose plate is `#0d1117`.
- So the page ground cannot show through a figure, and the only thing it can
  do is make the plate **edge** disappear. That fixes the dark ground to the
  narrow band between `#000000` and `#0d1117`:

  | Dark ground | vs the `#000000` plate | vs the `#0d1117` plate |
  | :-- | :-- | :-- |
  | Starlight's default (`#17181c`) | 1.22:1 | 1.10:1 |
  | `instrument` (`#0d1114`) | **1.11:1** | **1.00:1** |

  That roughly halves the visible edge compared with the old theme, and makes
  the diagram plate disappear outright.
- **The light page ground has to stay `#ffffff`.** That single fact rules out
  the tinted "paper" light theme I would otherwise have picked: any off-white
  page would frame all 1220 figures in a white box. The palette therefore
  keeps the light ground white and puts its tint into the *surfaces*: nav,
  sidebar, cards, code blocks, the landing panels. Measured seam: 1.00:1.
- **The accent is nowhere near `#1f77b4` or `#d62728`.** The chrome must never
  read as part of a plot, which is exactly what Starlight's blue-violet accent
  did next to a blue trace.

Worth a separate change on `main`, outside this branch: if the figure
generator saved with `transparent=True` (or an explicit `facecolor`), the
dark plates would go away entirely and the page ground would show through
every plot, which is both better looking and would free the dark theme from
this constraint completely. It costs a regeneration of all 1220 files, so it
is a decision, not a tweak.

### Why this one, in one paragraph

Of the three directions I built, `instrument` is the one that behaves best
next to the thing the site is mostly made of. All three sat within 0.02 of
each other on both plate seams, so the figures did not decide it; what decides
it is what sits *around* the plate. A near-neutral cool grey lets a
black-grounded plot read as an inset panel rather than as a hole in a coloured
page, cyan is far enough from `#1f77b4` and `#d62728` that a link never looks
like a trace, and near-neutral greys let the pages that are mostly tables and
numbers read as paper rather than as a themed surface. `graphite` (warm greys,
amber) was the runner-up and the more distinctive choice; `blueprint` was the
handsomest in dark mode but committed the whole page to the same hue family as
the plots.

### Contrast

`site/scripts/check-contrast.mjs` parses the stylesheet, resolves the two
themes and measures every pair the site renders. Enforced pairs are text at
4.5:1 (WCAG 1.4.3 AA) and meaningful non-text at 3:1 (1.4.11); decorative
hairlines and the figures' own plate seams are measured and reported but
cannot fail the run.

```
$ node scripts/check-contrast.mjs
25 pairs measured, 0 enforced pair(s) below threshold.
```

| Enforced pair | Dark | Light |
| :-- | --: | --: |
| body text on page | 11.62 | 10.95 |
| muted text on page | 6.36 | 6.36 |
| headings on page | 18.96 | 17.90 |
| body text on nav / sidebar | 10.66 | 10.16 |
| link colour on page | 14.03 | 5.73 |
| link colour on a card | 12.87 | 5.36 |
| muted text on a card | 5.83 | 5.95 |
| primary button label | 9.11 | 5.73 |
| accent mark, non-text (min 3) | 8.13 | 5.73 |

The full table, including the informational rows, is the script's own output.

### How it is applied

`src/styles/theme.css` is loaded first in `customCss` and is unlayered, so it
wins over Starlight's `@layer starlight.base` defaults without any
`!important`. The dark theme is the plain `:root` rule and the light theme is
`:root[data-theme='light']`, which is Starlight's own switch. There is no
JavaScript involved in the palette at all, so there is nothing to flash and
nothing to regress with scripting disabled.

## 2. Front page

The old landing was the feature list from the README rendered as ten cards of
60 to 90 words each. It states everything and shows nothing: a first-time
visitor cannot tell what the library does with a signal, who it is for, or
where to start, and the standards are buried inside prose.

The new one, in `src/components/home/Home.astro` with all its text in
`src/data/home.ts` (one object per locale, one shared shape so EN and ES
cannot drift), answers one question per block:

1. **The numbers, first, without adjectives.** 427/427 conformance checks,
   278 standards over 53 domains, 506 figures, 32 PDF fiches. Every one is
   taken from the repository (`docs/CONFORMANCE.md`, `.github/images`,
   `.github/reports`, `VERSION`), and the header comment in `home.ts` records
   where each came from. The old page said 371 checks and 235 standards, which
   were the numbers of an earlier release; the conformance report says 427 and
   278.
2. **What this is**, in three paragraphs, plus **who it is for** and **what it
   is not**. The second list is the part I would not cut: it says in plain
   words that the conformance report is the library checking itself against
   published values and not an accredited calibration, that there is no
   hardware I/O, and that no standard is claimed to be implemented in full.
   A page that admits its limits is more credible about its claims.
3. **What it looks like in use**, showing rather than telling, with the
   artefacts the repository already has: eight lines of code beside the
   one-third-octave figure they produce (the committed figure, in the reader's
   language: the Spanish page shows the `_es` variant), then the ISO 717-1
   rating with the accredited-style PDF fiche it renders.
4. **What it covers**: nine areas as a table, each row linking to its area
   overview, with the standards as monospaced chips. Breadth becomes scannable
   instead of a wall of prose, and the chips are the visual signature of the
   page.
5. **Starting from zero**: three numbered steps, install, run one analysis end
   to end, go to your own domain.

The blocks are numbered and ruled like the clauses of the documents the
library implements, which is where the page gets its character from rather
than from illustration.

Layout notes: the splash column goes from 45rem to 62rem (the sidebar is
hidden there anyway), the sidebar gutter that Starlight reserves is zeroed so
the page is actually centred, the hero grid is collapsed to one column because
there is no hero image, and Starlight's prose flow margin is switched off
inside the landing, which is a layout and sets its own spacing.

### The clause number used to be painted on the heading

The first version of that numbering was a defect, and a bad one. The number
was an absolutely positioned `.block::before` at `inset-inline-start: 0`,
while the gutter it needs was reserved by `padding-inline-start: 3.5rem`
inside a `@media (min-width: 62rem)` block **only**. Everywhere else no space
existed and the number was painted straight onto the first letter of the
title: the teal `01` sat on the Q of "Qué es".

That is not only a phone problem. A `rem` media query resolves against the
reader's default text size, so the failure covers three separate axes:

- every viewport narrower than 992 px, which includes every phone and every
  tablet in portrait,
- 200 % and 400 % browser zoom, which are exactly the conditions WCAG 1.4.4
  and 1.4.10 are about (200 % zoom of a 1440 px window is a 720 px viewport),
- **any width at all**, including 1440 px, once the reader's default text size
  goes past about 23 px, because 62rem then stops matching.

Measured on the old component, both locales: 20 of the 26 cases I checked had
all four clause numbers overlapping their heading. At 390 px the number box
was `x 16..33` and the heading text started at `x 16`; at 992 px and above,
where the padding existed, the heading started at `x 72` and there was no
overlap.

The fix does not move the threshold, because moving a threshold only moves the
failure. The number is now a real box in the layout: each block is a wrapping
flex line of two items, the clause number (`flex: 0 0 3.5rem`) and the block
body (`flex: 1 1 26rem`). The space is therefore reserved by the layout
itself, and when the line cannot hold both the body wraps and the number ends
up on its own line above the title. Both states come out of the same rule, at
whatever width, zoom level or text size the reader has, and there is no state
in which the two can occupy the same pixels. The number sits on the heading's
baseline, which also holds at any text size.

`site/scripts/check-home-headings.mjs` is the proof and the regression guard:

```
$ node scripts/check-home-headings.mjs
40 layout case(s) measured, 0 failing.
```

It drives its own headless Chrome over both locales at 320, 360, 390, 430,
500, 600, 768, 900, 991, 992, 1100, 1280, 1440 and 1920 px, at 200 % and
400 % zoom, and with the browser's default text size set to 24 px (at 1440,
900 and 390 px wide) and to 32 px. In each case it asserts that no clause
number overlaps any text of its own block, that no step number overlaps its
step heading, that the `+` and `-` list markers still fit the indent their
item reserves, and that the document never scrolls sideways. The same 40 cases
pass against the built site served by `astro preview`, not just against the
dev server. The layout crosses over at about 500 px (and at 900 px with 24 px
text): below that the number is above the title, above it the number is in the
gutter.

Screenshots `50`..`53` are the before and after pair, at 390 px in Spanish and
at 1440 px with 24 px default text.

## 3. Mobile theme toggle

Below Starlight's `md` breakpoint the whole right-hand group of the header
(social links, theme, language) is hidden and only reachable through the
hamburger. The site already lifted the language picker out of it;
`Header.astro` now lifts the theme control out too, immediately before the
language picker, in the same desktop order.

Both are rendered as icon-and-caret stubs: the same `width` trick the language
picker already used (the visible label text is transparent, the icon and caret
remain), so the bar still fits `logo · search · theme · language · menu` at
360 px. Nothing about the desktop header changes, and nothing about
Starlight's own `ThemeSelect` behaviour changes either: `updatePickers()`
already walks *every* `starlight-theme-select` on the page, so the mobile stub
and the desktop select stay in sync by construction, and each instance wires
its own `change` listener. The accessible name comes from the visually hidden
`<span>` inside the label, so hiding the visible text costs nothing, and the
icon itself still reports the current state (sun, moon, laptop).

## 4. Language suggestion

`src/components/LangSuggest.astro`. One behaviour: a quiet, dismissible
one-line bar pinned to the bottom edge, written in the language it offers.
**Nothing navigates on its own, ever.**

I prototyped a redirect mode alongside it and rejected it, for reasons
specific to this site rather than general ones:

1. **English URLs here are deliberate, not default.** English is the root
   locale, so `/phonometry/guides/levels/` is a real canonical URL. It is what
   the README, PyPI, Zenodo, an answer on a forum and every English search
   result point at. A visitor who followed one of those made a choice about
   the page, not just about a language; silently replacing it overrides them.
2. **A redirect is worst exactly where the content is thinnest.** The 120
   generated API pages exist only in English. Sending a Spanish-preferring
   visitor into `/es/` there hands them an English page with a fallback notice
   instead of the canonical one. The banner has no such failure mode, and it
   opts out of that subtree anyway.
3. **It leaves the search engines alone.** The site already declares
   `hreflang` for both locales and `x-default` for English, which is the
   supported mechanism, and Google explicitly prefers server-side or
   link-based handling over client-side redirection.
4. **It teaches instead of switching.** A redirect never tells the reader that
   the other language exists or how to get back; the bar names both the
   language and the way back, and one dismissal ends it forever.
5. **The cost is one click**, on a two-language site with a visible picker in
   the header, and now on phones too.

### Rules

- **An explicit locale in the URL wins.** `?lang=en` or `?lang=es` is recorded
  as the visitor's choice and silences the suggestion.
- **Using the language picker records the choice.** A delegated capture-phase
  `change` listener on `starlight-lang-select` stores `en` or `es` before
  Starlight navigates, so a manual decision beats `Accept-Language` from then
  on. Dismissing the banner records the current page's language for the same
  reason: it is also a decision.
- **Once anything is stored, nothing is ever imposed.** Someone who chose
  Spanish and then deliberately opens an English URL stays on the English URL
  and is at most offered a link.
- **Crawlers are skipped** by user agent, and nothing here touches
  `rel=canonical` or the `hreflang` alternates, which is what search engines
  actually use to pick a locale (the site already emits `en`, `es` and
  `x-default` on every page).
- **The API reference is excluded entirely.** It is generated in English only
  and served under `/es/` by Starlight's locale fallback, so the banner is not
  even rendered in that subtree: offering a Spanish speaker a page that is
  still English, with a fallback notice on it, would be worse than silence.
- Nothing is inferred from a language the visitor does not have: if neither
  `es` nor `en` appears in `navigator.languages`, the feature does nothing.
- **There is a kill switch, and it is a constant.** `ENABLED` at the top of
  the component renders no markup and no script when it is false. It is
  deliberately not a control: nothing on the page and nothing in storage can
  turn the feature on or off.

`site/scripts/check-lang-suggest.mjs` drives every scenario in an isolated
browser context and asserts the resulting URL, banner state and stored value:

```
$ node scripts/check-lang-suggest.mjs
0 failing scenario(s).
```

## Screenshots

All captured by `site/scripts/redesign-shots.mjs`, which drives its own
headless Chrome so each shot states its viewport, locale and theme rather than
depending on a live browser session. Desktop is 1440x900, narrow is 390x844.
The heading shots (`50`..`53`) come from `check-home-headings.mjs`.

```
pnpm --dir site dev --host 0.0.0.0 --port 4322
node site/scripts/redesign-shots.mjs            # all of them
node site/scripts/redesign-shots.mjs 20-home    # or a subset, by name
```

| File | What it shows |
| :-- | :-- |
| `00-baseline-home-desktop-light.png` | The landing page before any of this: Starlight defaults, ten description cards. |
| `10-guide-dark.png`, `11-guide-light.png` | A guide with a hand-drawn diagram (the `#0d1117` plate case), both themes. |
| `16-plot-dark.png`, `19-plot-light.png` | A page whose figure is a matplotlib plot (the black plate case). |
| `20-home-dark.png`, `21-home-light.png` | The new landing, both themes, full page. |
| `24-home-es-light.png`, `25-home-es-dark.png` | The Spanish landing, including the Spanish variant of the figure. |
| `30-home-phone-dark.png`, `31-home-phone-light.png` | The landing at 390 px. |
| `32-guide-phone-dark.png` | A guide at 390 px. |
| `33-header-phone-light.png`, `34-header-phone-dark.png` | The mobile header with the new theme control, EN light and ES dark. |
| `40-lang-banner-on-en-page.png` | The language bar on an English page for a Spanish browser, desktop. |
| `41-lang-banner-on-es-page-phone.png` | The language bar on a Spanish page for an English browser, phone. |
| `50-heading-before-es-phone.png`, `51-heading-after-es-phone.png` | The clause number on the Q of "Qué es" at 390 px, and the same block fixed. |
| `52-heading-before-desktop-text24.png`, `53-heading-after-desktop-text24.png` | The same defect at 1440 px with the default text size at 24 px, and the fix. |

## Gates

Run against this branch, all green:

| Gate | Command | Result |
| :-- | :-- | :-- |
| Build and link check | `pnpm --dir site build` | 441 pages, all internal links valid |
| HTML validation | `pnpm --dir site html-validate` | clean |
| Accessibility | `pa11y-ci` (WCAG2AA, 46 URLs) | 46/46 passed |
| EN/ES parity | `node site/scripts/check-i18n-parity.mjs` | 100 EN pages each have an ES translation |
| Contrast | `node site/scripts/check-contrast.mjs` | 25 pairs, 0 enforced pair below threshold |
| Language handling | `node site/scripts/check-lang-suggest.mjs` | 9 scenarios, 0 failing |
| Landing headings | `node site/scripts/check-home-headings.mjs` | 40 layout cases, 0 failing |

The pa11y run used a copy of `.pa11yci.json` on port 4324, because 4321 to
4323 were busy with other dev servers; the URL list is otherwise identical.
The two browser-driven layout checks were also run against `astro preview` on
that port, so the results hold for the built output and not only for the dev
server.

## What is not done here

- The landing quotes 427/278/53 from `docs/CONFORMANCE.md`. The README still
  says 371/235/46 in its own prose; that is a separate fix, on main.
- The ISO 717 fiche preview on the landing is the English one in both
  locales, exactly as the guide embeds it today. Rendering a Spanish fiche
  would mean regenerating the reports, which is a `make reports` job and not a
  site change.
- `check-home-headings.mjs` and `check-lang-suggest.mjs` need a running server
  and are not wired into CI on this branch. If the landing stays, the heading
  check is the one I would add to the site workflow, next to pa11y, since it
  is the only guard against this class of defect coming back.
