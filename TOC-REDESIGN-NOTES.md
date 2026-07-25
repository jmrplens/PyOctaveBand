# Sidebar standards info and API topics: variant prototypes

Experimental branch `feat/toc-sidebar-info`. It does not merge and has no PR:
it exists so I can look at five ways of surfacing "what standard governs this
page" in the docs, plus three ways of dealing with the API reference tree,
switch between them live, and then keep one.

The predecessor of this branch, `exp/sidebar-standard-chips`
(`SIDEBAR-CHIPS-NOTES.md` there), put a coloured badge on every sidebar item.
The classification work was right; the presentation was too loud. Every
variant here reuses that classification untouched and only changes how, when
and where it is shown.

## Mechanism

The per-item classification travels as a JSON string in Starlight's sanctioned
`attrs` passthrough, under `data-chips`:

- `site/astro.config.mjs` builds it for the guides with the helpers `S(...)`
  (governing standard, teal), `T(...)` (named reference, amber) and
  `chips(...)`.
- `scripts/generate_api_docs.py` emits the same shape for the API modules into
  `site/src/generated/api-sidebar.mjs`, from the module docstrings.
- `site/src/components/SidebarSublist.astro` parses it, renders one `<Badge>`
  per chip inside a `.sidebar-chips` run in the link, and strips the raw
  attribute so it never reaches the DOM.
- `site/src/styles/toc-info.css` decides what that run looks like, or whether
  it is shown at all. All of its selectors are unlayered, so they win over
  Starlight's layered styles without specificity tricks.

Two more components exist for the page-level variant:
`PageTitle.astro` (the subtitle under the H1) and `SectionStandards.astro`
(the "standards by page" table on section landings). Both always render and
are only displayed by CSS when the relevant variant is active, so switching
never needs a rebuild.

The active variant is an attribute on `<html>`, applied before first paint by
an inline script in `Head.astro` and persisted in `localStorage`. The same
script stamps `data-area` (`guides` or `api`) from the URL, mounts the
floating switcher, wires the API caret, and mirrors the section table heading
into "On this page" when the clean variant is active.

Chip text is language neutral (designations and author names), so EN and ES
share one classification and there is nothing extra to translate. Only the
table headings, the visually hidden prefix and the back link are localized.

## Variant matrix

TOC info, attribute `data-toc-style` on `<html>`, localStorage key `tocStyle`:

| Value | What the reader sees |
| --- | --- |
| `focus` (default) | Annotation line under the label, only for the items of the section the reader is currently in. Rest of the tree untouched. |
| `annotate` | The same line on every item of the whole tree. |
| `hover` | Nothing at rest. A hairline dotted marker appears on annotated items while the pointer or the keyboard focus is inside the sidebar, and the info opens in a small floating card on hover or focus. On touch devices it falls back to the always-visible line. |
| `clean` | Nothing in the sidebar. The info moves to a subtitle under the H1 and to a table on each section landing page. |
| `none` | Nothing anywhere. The pre-experiment look. |

API sidebar, attribute `data-api-style`, localStorage key `apiStyle`:

| Value | What the reader sees |
| --- | --- |
| `split` (default) | Topic style. On guide pages the API group shrinks to its label with a forward arrow; on API pages the guide groups disappear and a "Guides" back link takes their place. |
| `collapsed` | The API group gets a caret and folds. Folded on guide pages, open on API pages, and the reader's choice is kept for the session. |
| `inline` | Status quo: the whole API tree always expanded in place. |

### How to switch

- The round "UX" button at the bottom right of every page opens the switcher.
  The choice is remembered across pages and reloads.
- Without the widget, from the console:
  `document.documentElement.dataset.tocStyle = 'annotate'` and
  `document.documentElement.dataset.apiStyle = 'collapsed'`. To make it stick,
  `localStorage.setItem('tocStyle', 'annotate')`.
- To go back to the shipped look, pick `none` and `inline`.

## Screenshots

In `ux-variants2/`. Desktop is 1440 x 1000, mobile is 390 x 844 with the menu
open. Sidebar-only crops for the sidebar variants, full viewport where the
page itself carries the information.

| File | What it shows |
| --- | --- |
| `focus-dark-desktop-en.png` | Focus, dark. Only the current section is annotated. |
| `focus-light-desktop-en.png` | Focus, light. |
| `focus-dark-desktop-es.png` | Focus on the Spanish tree: same chips, no translation needed. |
| `focus-dark-mobile-en.png` | Focus inside the mobile menu. |
| `annotate-dark-desktop-en.png` | Every item annotated, dark. |
| `annotate-light-desktop-en.png` | Every item annotated, light. |
| `annotate-dark-mobile-en.png` | Every item annotated, phone. |
| `hover-resting-dark-desktop-en.png` | Hover variant at rest: no marks at all. |
| `hover-card-dark-desktop-en.png` | Hover variant engaged: markers visible, card open. |
| `hover-card-light-desktop-en.png` | The same in light. |
| `hover-card-dark-mobile-en.png` | The card at 390 px in a hover-capable browser. |
| `hover-touchfallback-simulated-dark-mobile-en.png` | What a real phone gets instead: the annotation line. Simulated, see the caveat below. |
| `clean-subtitle-dark-desktop-en.png` | Clean: subtitle under the H1, sidebar empty. |
| `clean-subtitle-light-desktop-en.png` | The same in light. |
| `clean-subtitle-light-desktop-es.png` | The same in Spanish. |
| `clean-subtitle-dark-mobile-en.png` | Subtitle on a phone. |
| `clean-sectiontable-light-desktop-en.png` | Section landing table, light. |
| `clean-sectiontable-dark-desktop-en.png` | Section landing table, dark, with its entry in "On this page". |
| `clean-sectiontable-light-desktop-es.png` | Section landing table in Spanish. |
| `clean-sectiontable-dark-mobile-en.png` | The table stacked for a phone. |
| `none-dark-desktop-en.png`, `none-dark-mobile-en.png` | Reference: today's sidebar. |
| `api-inline-guidepage-dark-desktop-en.png` | API tree expanded in place on a guide page. |
| `api-collapsed-guidepage-dark-desktop-en.png` | API group folded behind a caret. |
| `api-split-guidepage-dark-desktop-en.png` | API group reduced to a label with an arrow. |
| `api-inline-apipage-dark-desktop-en.png` | Status quo on an API page. |
| `api-collapsed-apipage-dark-desktop-en.png` | Collapsed mode on an API page (opens itself). |
| `api-split-apipage-dark-desktop-en.png` | Split mode on an API page: back link plus reference only. |
| `api-split-apipage-dark-mobile-en.png` | The same on a phone. |
| `switcher-widget-dark-desktop-en.png` | The switcher panel. |

## Design critique

**focus.** The strongest of the five. The tree stays exactly as quiet as it is
today except for the five or six rows the reader is actually working in, and
those rows answer the only question the annotation is there to answer: which
of these neighbouring pages do I want. The annotation sits at 11 px, indented
under the label, one step below the label colour, so the eye reads label
first and metadata second without effort. Weaknesses: the annotated block is
a visible island in an otherwise plain tree, which looks slightly arbitrary
until you notice it tracks the current section; multi-standard rows such as
"Laboratory Insulation Measurement" wrap to two annotation lines and the run
dots then fall mid-line, which reads ragged; and if a reader wants to compare
two distant sections it shows nothing useful.

**annotate.** Honest and complete, and the least clever. It also roughly
doubles the height of the tree, which on this site means a very long scroll,
and it puts an almost equal density of text on every row, so the labels stop
standing out as the thing you click. It is the right baseline to compare
against and the wrong thing to ship. If it were shipped, the annotation
should probably be limited to one designation per item.

**hover.** The cleanest resting state by a wide margin and the only variant
that costs zero vertical space. Its structural problem is that nearly every
page in this library is governed by something, so the affordance would mark
almost every row and therefore distinguish nothing. That is why the marker is
now gone at rest and only fades in while the pointer or the keyboard focus is
inside the sidebar; even then it reads as texture across the whole tree. The
card itself is good: it is anchored to the trailing edge of the link so it can
never push the sidebar into horizontal scroll, it has a 250 ms intent delay,
it opens on keyboard focus, and Escape dismisses it without moving the focus.
It has two real costs. It hides the row underneath while open, because the
sidebar is a scroll container and a card placed outside it would be clipped.
And it is a desktop interaction: on touch there is no hover, so the fallback
turns it into `annotate`, which means a phone reader never gets the clean
tree the variant is for.

**clean.** The best looking of the five, and my second choice. The subtitle
under the H1 reads like the dateline of a standard, it is the first thing you
see on the page, and the sidebar keeps its current silence. The section
landing table is the weakest part: on the section pages it duplicates the
bullet list that already sits above it, where each bullet names its standards
in prose, so it repeats rather than adds. It also introduces a heading that
the markdown pipeline never sees, so its entry in "On this page" has to be
injected client side, and on a phone the three columns only work because they
now collapse into a stacked list. The deeper objection: the information
arrives only after you have already chosen the page, so it never helps you
choose one. That is exactly the job the sidebar annotation does.

**none.** The control. Worth flipping to now and then to check how much any
of the others actually cost.

**API modes.** `inline` is what makes the sidebar unmanageable today: the API
tree is far longer than the guide tree and it is always open. `collapsed` is
the cheap fix and behaves well, though a caret is a single small target for a
very large piece of navigation. `split` is the most comfortable to read on an
API page, since the guide groups disappear entirely, but it is also the most
custom, and it exposes a real flaw of the current information architecture:
the API items still live three levels deep, under "Reference" and then "API
reference", so module names like `loudness_moore_glasberg_time` wrap even when
they are the only thing on screen.

## Recommendation

Ship `focus` for the standards info and `split` for the API sidebar, with two
caveats.

`focus` because it is the only variant that puts the information where the
decision is made, at the moment it is made, without paying for it everywhere
else. `clean` is prettier but arrives too late to influence a choice, `hover`
hides the information behind an interaction that half the readers cannot
perform, and `annotate` makes the reader pay on every row for something they
need on five.

The two caveats:

1. `focus` and `clean` are not exclusive. The subtitle under the H1 from the
   `clean` variant costs nothing in the sidebar and confirms on arrival what
   the annotation promised. I would ship the subtitle together with `focus`
   and drop the section landing table, which duplicates prose that is already
   there.
2. For the API split, prefer the plugin over this prototype. See below.

Before shipping `focus` I would also trim the classification: cap the
annotation at two standards plus one reference per item, so the wrapped
three-designation rows stop happening, and keep the full list for the page
subtitle where there is room.

## `split` versus `starlight-sidebar-topics`

My `split` prototype and the `starlight-sidebar-topics` plugin solve the same
problem, and the plugin solves it better in the ways that matter long term.

What they share: one sidebar for the guides, another for the API reference,
and a way back. What differs:

- The plugin makes each topic a root sidebar, so the API items lose the two
  wrapper levels they currently sit under, and the topic switcher is a real
  navigation control at the top of the sidebar rather than a link that only
  appears on some pages. My prototype only hides things with CSS, so the
  nesting, and the wrapping module names that come with it, stay.
- The plugin's topic labels and badges take per-locale objects, so EN and ES
  are handled by the config. Mine hardcodes the back link label in
  `Sidebar.astro`.
- The plugin is maintained and documented. Mine is roughly 30 lines of CSS
  plus a hand-written back link, and every future Starlight release is my
  problem.

What I would want to check before adopting it: this repo already overrides
`Sidebar.astro` and `SidebarSublist.astro`, and the whole "overview first"
group-label-as-link convention lives in that override. If the plugin also
overrides `Sidebar`, one of the two has to give, and reconciling them is the
real cost of the migration. The plugin docs do not say. That is a
one-afternoon experiment, not a blocker.

So: keep `split` in this branch as the visual argument for splitting, and
implement the split with the plugin rather than with this CSS.

## Verification

Checked in Chrome at 1440 x 1000 and at 390 x 844, in both themes, on EN and
ES pages, on a guide page, a section landing and an API page:

- All five `data-toc-style` values and all three `data-api-style` values
  render as intended, including the API caret toggle and its session memory.
- No horizontal overflow of the sidebar in any variant at either width. The
  hover card used to cause it and no longer does.
- Keyboard: the hover card opens on `:focus-visible`, Escape dismisses it
  without moving the focus, any other key brings it back, and leaving the link
  resets it.
- The "Guides" back link used to point at a page that does not exist
  (`/guides/getting-started/`); it now points at `/getting-started/` and its
  Spanish counterpart, and both answer 200.
- `pnpm --dir site build` succeeds and the Starlight link validator reports
  all internal links valid.
- `pnpm run html-validate` passes. It did not at first: the section table
  needed `scope="col"` on its headers, 36 errors across 12 pages.
- `pnpm run pa11y` passes 46 of 46 URLs at WCAG2AA.
- `node scripts/check-i18n-parity.mjs` passes.

What I could not verify locally:

- pa11y only exercises the default variant, since it starts with an empty
  `localStorage`. The other four TOC variants and two API modes are not
  covered by the audit. Colour choices were checked by hand instead: the
  annotation ink is about 5.3:1 on the dark sidebar and 5.6:1 on the light
  one, both above the 4.5:1 that 11 px text needs.
- The touch fallback of the `hover` variant. The automation browser reports
  `hover: hover`, so `@media (hover: none)` never applies there and the
  screenshot of it is a simulation: the same declarations injected without the
  media query. It needs one look on the actual phone.
- Whether the injected "On this page" entry for the clean variant participates
  in scroll spy. It does not, since Starlight collects its links at connect
  time. It is a prototype-only wart that disappears with the client-side
  injection if the variant is chosen.

## When a variant is chosen

Everything below is prototype scaffolding and comes out:

- both inline scripts at the bottom of `Head.astro`, and the
  `.toc-switcher` rules in `toc-info.css`;
- every branch of `toc-info.css` for the variants not chosen;
- `SectionStandards.astro` and its import in `MarkdownContent.astro`, unless
  the section table survives;
- `PageTitle.astro` and its `components` entry in `astro.config.mjs`, unless
  the subtitle survives;
- the `api-caret` button in `SidebarSublist.astro` if the split is done with
  the plugin instead.

The classification itself, the `chips()` helpers in `astro.config.mjs`, the
`_API_CHIPS` table in `scripts/generate_api_docs.py` and the parsing in
`SidebarSublist.astro` stay whatever the outcome.
