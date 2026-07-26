# Site redesign: colour directions, front page, mobile header, language detection

Experimental branch `feat/site-redesign`. It does not merge and has no PR. It
exists so I can look at the site as a whole: what colour it should be, what
the landing page should do, and two pieces of behaviour the site is missing on
phones and for Spanish-speaking visitors.

Everything here is switchable and reversible. Nothing about the 1220 committed
figures changes: the palettes are chosen *around* the figures, not the other
way round, and section 1 records what the figures turned out to actually be.

## 1. Colour directions

The site rides Starlight's defaults today: hue 224 greys and a 100 % saturated
blue-violet accent that belongs to a startup landing page, not to a library
whose output is calibration numbers. Three complete directions now live in
`site/src/styles/theme-directions.css`, each covering light and dark.

| Direction | Idea | Light accent | Dark accent | Dark ground |
| :-- | :-- | :-- | :-- | :-- |
| `instrument` (default) | Cool steel greys, deep cyan. The front panel of a measuring instrument: neutral metal, one saturated indicator colour. | `#0a6f8c` | `#35b8d8` (text `#b3e6f4`) | `#0d1114` |
| `blueprint` | Drafting blue-greys, prussian ink. Reads like a drawing sheet; the dark theme is a navy ground rather than a neutral one. | `#23479a` | `#78a2f0` (text `#cbdcfb`) | `#0b0f1a` |
| `graphite` | Warm neutral greys, amber. Lab hardware and amber CRT: the only direction that is not blue, so the page never competes with the plots. | `#8a4b06` | `#e0a33e` (text `#f7dcab`) | `#100e0a` |

### What the figures dictate

The palettes are not free choices. I measured what the committed figures
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
  do is make the plate **edge** disappear. That fixes the dark grounds to the
  narrow band between `#000000` and `#0d1117`:

  | Direction | vs the `#000000` plate | vs the `#0d1117` plate |
  | :-- | :-- | :-- |
  | Starlight today (`#17181c`) | 1.22:1 | 1.10:1 |
  | `instrument` (`#0d1114`) | **1.11:1** | **1.00:1** |
  | `blueprint` (`#0b0f1a`) | **1.10:1** | **1.01:1** |
  | `graphite` (`#100e0a`) | **1.09:1** | **1.02:1** |

  All three roughly halve the visible edge compared with the current theme,
  and make the diagram plate disappear outright.
- **The light page ground has to stay `#ffffff`.** That single fact rules out
  the tinted "paper" light theme I would otherwise have picked: any off-white
  page would frame all 1220 figures in a white box. All three directions
  therefore keep the light ground white and put their tint into the
  *surfaces*: nav, sidebar, cards, code blocks, the landing panels. Measured
  seam: 1.00:1 in all three.
- **No direction uses a hue near `#1f77b4` or `#d62728`.** The chrome must
  never read as part of a plot, which is exactly what Starlight's blue-violet
  accent does today next to a blue trace.

Worth a separate change on `main`, outside this branch: if the figure
generator saved with `transparent=True` (or an explicit `facecolor`), the
dark plates would go away entirely and the page ground would show through
every plot, which is both better looking and would free the dark theme from
this constraint completely. It costs a regeneration of all 1220 files, so it
is a decision, not a tweak.

### Contrast

`site/scripts/check-contrast.mjs` parses the stylesheet, resolves the six
palettes and measures every pair the site renders. Enforced pairs are text at
4.5:1 (WCAG 1.4.3 AA) and meaningful non-text at 3:1 (1.4.11); decorative
hairlines and the figures' own gridlines are measured and reported but cannot
fail the run.

```
$ node scripts/check-contrast.mjs
75 pairs measured, 0 enforced pair(s) below threshold.
```

Worst enforced pair per direction (all comfortably over AA):

| Pair | instrument | blueprint | graphite |
| :-- | :-- | :-- | :-- |
| body text on page (dark / light) | 11.62 / 10.95 | 11.30 / 11.00 | 11.34 / 11.66 |
| muted text on page (dark / light) | 6.36 / 6.36 | 6.07 / 6.47 | 6.09 / 7.07 |
| link colour on page (dark / light) | 14.03 / 5.73 | 13.82 / 8.61 | 14.49 / 6.80 |
| link colour on a card (dark / light) | 12.87 / 5.36 | 12.79 / 7.88 | 13.55 / 6.30 |
| primary button label (dark / light) | 9.11 / 5.73 | 10.02 / 8.61 | 9.16 / 6.80 |
| accent mark, non-text (dark / light) | 8.13 / 5.73 | 7.49 / 8.61 | 8.70 / 6.80 |

The full table, including the informational rows, is the script's own output.

### Switch mechanism

Same spirit as the other experimental branch: an attribute on `<html>`.

- `data-accent` on `<html>`, one of `instrument` / `blueprint` / `graphite`.
- Stamped **before first paint** by an inline script in
  `src/components/Head.astro`, read from `localStorage['phonometry:accent']`,
  so switching never flashes the previous palette.
- Written by `src/components/PrototypeSwitcher.astro`, a small fixed control
  mounted from `Footer.astro` (the one custom component that renders on every
  page, splash included). Real `<button>`s in a labelled group with
  `aria-pressed`, collapsed by default.
- With JavaScript disabled, or with an unknown stored value, the site renders
  `instrument`: that direction is also the unconditional `:root` block, so
  there is no no-JS regression.
- The stylesheet is unlayered, so it wins over Starlight's
  `@layer starlight.base` defaults without any `!important`.

Shipping means deleting: the winning block becomes the plain `:root` rule and
the switcher and the other two blocks go away.

### Recommendation

**`instrument`.** It is the one that behaves best next to the thing the site is
mostly made of, which is 1220 figure files I am not going to redraw:

- The three are within 0.02 of each other on both plate seams, so the figures
  do not decide it. What decides it is what sits *around* the plate: a
  near-neutral cool grey lets a black-grounded plot read as an inset panel
  rather than as a hole in a coloured page. `blueprint`'s navy and
  `graphite`'s warm brown-black are both stronger opinions than a page full
  of plots wants.
- Cyan is far enough from `#1f77b4` and `#d62728` that a link never looks like
  a trace, which is exactly what the current Starlight blue-violet gets wrong.
- Its greys are close to neutral, so the pages that are mostly tables and
  numbers read as paper rather than as a themed surface.

`graphite` is the runner-up, and the one to pick if the site should look
unlike every other Starlight site at a glance: amber is the only accent here
that is not blue, so nothing on the page competes with a plot at all, and it
measures marginally best on the `#000000` seam. Its cost is that a warm page
around a cold plate is a slightly odder pairing than a cool page around one,
and that every screenshot of the site then reads as a deliberate style
statement, which is a thing to want on purpose.

`blueprint` is the weakest of the three for this site, not because it looks
bad (it is arguably the handsomest in dark mode) but because it commits the
whole page to the same hue family as the plots.

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

## 4. Language auto-detection

`src/components/LangSuggest.astro`. Three modes, chosen at runtime from
`localStorage['phonometry:lang-mode']` (the prototype switcher writes it) and
defaulting to the component's `DEFAULT_MODE`:

| Mode | Behaviour |
| :-- | :-- |
| `banner` (default) | A one-line dismissible bar pinned to the bottom edge, written in the language it offers. Nothing navigates on its own. |
| `redirect` | One `location.replace` to the counterpart page, on a genuine first visit only. |
| `off` | Nothing runs. |

### Rules that hold in every mode

- **An explicit locale in the URL wins.** `?lang=en` or `?lang=es` is recorded
  as the visitor's choice and silences the suggestion.
- **Using the language picker records the choice.** A delegated capture-phase
  `change` listener on `starlight-lang-select` stores `en` or `es` before
  Starlight navigates, so a manual decision beats `Accept-Language` from then
  on. Dismissing the banner records the current page's language for the same
  reason: it is also a decision.
- **Once anything is stored, nothing ever navigates on its own again.** This is
  the rule that keeps the feature from trapping anyone: someone who chose
  Spanish and then deliberately opens an English URL stays on the English URL
  and is at most offered a link.
- **Redirection is guarded three ways**: only with nothing stored, only once
  per session (`sessionStorage`), and only when the target differs from the
  current path. It cannot loop, and it uses `replace` so the back button
  returns to wherever the visitor came from instead of bouncing.
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

`site/scripts/check-lang-suggest.mjs` drives all eleven scenarios in isolated
browser contexts and asserts the resulting URL, banner state and stored value:

```
$ node scripts/check-lang-suggest.mjs
0 failing scenario(s).
```

### Redirect or banner: the recommendation

**The banner, and I would ship it that way.** The arguments are specific to
this site, not general:

1. **English URLs here are deliberate, not default.** English is the root
   locale, so `/phonometry/guides/levels/` is a real canonical URL. It is what
   the README, PyPI, Zenodo, an answer on a forum and every English search
   result point at. A visitor who followed one of those made a choice about
   the page, not just about a language; silently replacing it overrides them.
2. **A redirect is worst exactly where the content is thinnest.** The 120
   generated API pages exist only in English. Sending a Spanish-preferring
   visitor into `/es/` there hands them an English page with a fallback notice
   instead of the canonical one. The banner has no such failure mode, and I do
   not have to special-case a subtree for it (although it opts out of that
   subtree anyway).
3. **It leaves the search engines alone.** The site already declares
   `hreflang` for both locales and `x-default` for English, which is the
   supported mechanism, and Google explicitly prefers server-side or link-based
   handling over client-side redirection. A client-side hop on a static host
   is the one variant that risks being read as a redirect for some crawlers
   and not others, for no gain the `hreflang` links do not already give.
4. **It teaches instead of switching.** A redirect never tells the reader that
   the other language exists or how to get back; the bar names both the
   language and the way back, and one dismissal ends it forever.
5. **The cost is one click**, on a two-language site with a visible picker in
   the header, and now on phones too.

The case for `redirect` is that it is fewer clicks for the Spanish-speaking
audience that the Spanish mirror exists for, and it is implemented and
switchable here so it can be tried. But it is the mode that can be wrong, and
the banner is the mode that cannot.

## Screenshots

All captured by `site/scripts/redesign-shots.mjs`, which drives its own
headless Chrome so each shot states its viewport, locale, colour direction and
theme rather than depending on a live browser session. Desktop is 1440x900,
narrow is 390x844.

```
pnpm --dir site dev --host 0.0.0.0 --port 4322
node site/scripts/redesign-shots.mjs            # all of them
node site/scripts/redesign-shots.mjs 20-home    # or a subset, by name
```

| File | What it shows |
| :-- | :-- |
| `00-baseline-home-desktop-light.png` | The landing page before any of this: Starlight defaults, ten description cards. |
| `10-guide-instrument-dark.png` … `15-guide-graphite-light.png` | The three directions, light and dark, on a guide with a hand-drawn diagram (the `#0d1117` plate case). |
| `16-plot-instrument-dark.png` … `19-plot-instrument-light.png` | The same three directions on a page whose figure is a matplotlib plot (the transparent case). |
| `20-home-instrument-dark.png`, `21-home-instrument-light.png` | The new landing, recommended direction, both themes, full page. |
| `22-home-blueprint-dark.png`, `23-home-graphite-light.png` | The new landing in the other two directions. |
| `24-home-es-instrument-light.png`, `25-home-es-blueprint-dark.png` | The Spanish landing, including the Spanish variant of the figure. |
| `30-home-phone-instrument-dark.png`, `31-home-phone-graphite-light.png` | The landing at 390 px. |
| `32-guide-phone-instrument-dark.png` | A guide at 390 px. |
| `33-header-phone-light.png`, `34-header-phone-dark.png` | The mobile header with the new theme control, EN light and ES dark. |
| `40-lang-banner-on-en-page.png` | The language bar on an English page for a Spanish browser, desktop. |
| `41-lang-banner-on-es-page-phone.png` | The language bar on a Spanish page for an English browser, phone. |
| `42-prototype-switcher-open.png` | The prototype switcher open, with both groups. |

## Gates

Run against this branch, all green:

| Gate | Command | Result |
| :-- | :-- | :-- |
| Build and link check | `pnpm --dir site build` | 441 pages, all internal links valid |
| HTML validation | `pnpm --dir site html-validate` | clean |
| Accessibility | `pa11y-ci` (WCAG2AA, 46 URLs) | 46/46 passed |
| EN/ES parity | `node site/scripts/check-i18n-parity.mjs` | 100 EN pages each have an ES translation |
| Contrast | `node site/scripts/check-contrast.mjs` | 75 pairs, 0 enforced pair below threshold |
| Language handling | `node site/scripts/check-lang-suggest.mjs` | 11 scenarios, 0 failing |

The pa11y run used a copy of `.pa11yci.json` on port 4323, because 4321 was
busy with the other experimental branch's dev server; the URL list is
otherwise identical.

## What is not done here

- The three directions are all still present. Shipping means keeping one,
  promoting it to the plain `:root` block and deleting `PrototypeSwitcher`,
  the other two blocks and the `data-accent` script in `Head.astro`.
- The landing quotes 427/278/53 from `docs/CONFORMANCE.md`. The README still
  says 371/235/46 in its own prose; that is a separate fix, on main.
- The ISO 717 fiche preview on the landing is the English one in both
  locales, exactly as the guide embeds it today. Rendering a Spanish fiche
  would mean regenerating the reports, which is a `make reports` job and not a
  site change.
