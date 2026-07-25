# Standards info in the docs: variant prototypes

Experimental branch `feat/toc-sidebar-info`. It does not merge and has no PR:
it exists so I can look at several ways of surfacing "what standard governs
this page", plus three ways of dealing with the API reference tree, switch
between them live, and then keep one.

The predecessor of this branch, `exp/sidebar-standard-chips`
(`SIDEBAR-CHIPS-NOTES.md` there), put a coloured badge on every sidebar item.
The classification work was right; the presentation was too loud.

There are now two independent families:

- **Page chips** (`data-page-chips`), the newest and the one I would ship: a
  quiet run under the H1, derived from the page's own frontmatter
  bibliography, every item linking to its full entry at the bottom.
- **Sidebar info** (`data-toc-style`), five ways of putting the same
  information into the navigation tree instead.

The two can be viewed together or separately, and the target is the desktop
reader: this is a reference library that gets read next to an editor, so a
design is judged on the desktop layout first and only has to survive a phone.

## Mechanism

### Page chips: derived from the bibliography

The page chips take nothing from a hand-maintained list. Each page already
carries a Zod-typed `references` block in its frontmatter (schema in
`src/content.config.ts`), rendered as the APA-7 References section at the
bottom by `References.astro`. `src/lib/reference-chips.ts` turns that same
list into the header run, so the two views cannot disagree:

- `type: standard`, and `type: report` when it carries a document number,
  become the teal run. The designation collapses to its family, so
  `ISO 10140-2:2010`, `-3` and `-4` become one `ISO 10140` chip, and
  `ANSI S3.5-1997 (R2017)` becomes `ANSI S3.5`.
- `type: article` and `type: book` become the amber run, as author and year:
  `Schroeder 1965`, `Francois & Garrison 1982`, `Foret et al. 2011`. The
  Spanish build joins two authors with "y", like the bibliography does.
- `type: web` and numberless reports have neither a designation nor a citable
  author-date pair, so they stay in the bibliography only.
- Five standards and three works fit on one line at a desktop reading width.
  Anything past that folds into one `+N more` link to the References section.
- Pages with no `references` render nothing at all, which is most of the API
  reference and the getting-started page.

Every chip is a link to its entry, which now carries a stable id
(`referenceAnchors` derives the ids from the frontmatter order, so the
alphabetical sort of the bibliography does not move them). The target entry
clears the sticky header and is highlighted on arrival, so a click on
`ISO 10140` lands the reader on the right line of a fourteen-entry list.

### Sidebar info: the earlier classification

The per-item sidebar classification travels as a JSON string in Starlight's
sanctioned `attrs` passthrough, under `data-chips`:

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

Two more components exist for the `clean` variant: `PageTitle.astro` (the
subtitle under the H1) and `SectionStandards.astro` (the "standards by page"
table on section landings). Every one of these components always renders and
is only displayed by CSS when its variant is active, so switching never needs
a rebuild.

The active variant is an attribute on `<html>`, applied before first paint by
an inline script in `Head.astro` and persisted in `localStorage`. The same
script stamps `data-area` (`guides` or `api`) from the URL, mounts the
floating switcher, wires the API caret, and mirrors the section table heading
into "On this page" when the clean variant is active.

Chip text is language neutral (designations and author names), so EN and ES
share one classification and there is nothing extra to translate. Only the
table headings, the visually hidden prefix and the back link are localized.

## Variant matrix

Page chips, attribute `data-page-chips` on `<html>`, localStorage key
`pageChips`:

| Value | What the reader sees |
| --- | --- |
| `header` (default) | The run under the H1, standards then named works, each chip a link into the References section. |
| `off` | Nothing; the page header is untouched. |

Sidebar info, attribute `data-toc-style` on `<html>`, localStorage key
`tocStyle`:

| Value | What the reader sees |
| --- | --- |
| `focus` (default) | Annotation line under the label, only for the items of the section the reader is currently in. Rest of the tree untouched. |
| `annotate` | The same line on every item of the whole tree. |
| `hover` | Nothing at rest. A hairline dotted marker appears on annotated items while the pointer or the keyboard focus is inside the sidebar, and the info opens in a small floating card on hover or focus. On touch devices it falls back to the always-visible line. |
| `clean` | Nothing in the sidebar. The info moves to a subtitle under the H1 and to a table on each section landing page. The subtitle yields whenever the page chips run is on, since both sit under the H1 and say the same thing. |
| `none` | Nothing in the sidebar and no section tables. The pre-experiment tree. |

API sidebar, attribute `data-api-style`, localStorage key `apiStyle`:

| Value | What the reader sees |
| --- | --- |
| `split` (default) | Topic style. On guide pages the API group shrinks to its label with a forward arrow; on API pages the guide groups disappear and a "Guides" back link takes their place. |
| `collapsed` | The API group gets a caret and folds. Folded on guide pages, open on API pages, and the reader's choice is kept for the session. |
| `inline` | Status quo: the whole API tree always expanded in place. |

### How to switch

- The round "UX" button at the bottom right of every page opens the switcher,
  now with three groups: page chips, sidebar info, API sidebar. The choice is
  remembered across pages and reloads.
- Without the widget, from the console:
  `document.documentElement.dataset.pageChips = 'off'`,
  `document.documentElement.dataset.tocStyle = 'annotate'`,
  `document.documentElement.dataset.apiStyle = 'collapsed'`. To make it stick,
  `localStorage.setItem('tocStyle', 'annotate')` and so on.
- To go back to the shipped look, pick `off`, `none` and `inline`.
- The interesting comparison is `header` + `none` (page chips carrying
  everything, sidebar untouched) against `off` + `focus` (sidebar carrying
  everything) and `header` + `focus` (both).

## Screenshots

In `ux-variants2/`. Desktop is 1440 x 1000, mobile is 390 x 844 with the menu
open. Sidebar-only crops for the sidebar variants, full viewport where the
page itself carries the information.

Page chips:

| File | What it shows |
| --- | --- |
| `pagechips-many-dark-desktop-en.png` | The dense case, dark: Laboratory Insulation Measurement, five ISO families and three books on one line. |
| `pagechips-many-light-desktop-en.png` | The same in light. |
| `pagechips-many-dark-desktop-es.png` | The same page in Spanish, under a two-line H1. |
| `pagechips-overflow-light-desktop-en.png` | Underwater sound propagation, sixteen references: one standard, three works and `+12 more`. |
| `pagechips-single-dark-desktop-en.png` | A page with one reference and no standard. |
| `pagechips-none-light-desktop-en.png` | A page with no `references` block: nothing rendered. |
| `pagechips-anchorjump-light-desktop-en.png` | After clicking the `ISO 10140` chip: the matching bibliography entry, highlighted and clear of the header. |
| `pagechips-anchorjump-dark-desktop-es.png` | The same jump on the Spanish build. |

Sidebar info:

| File | What it shows |
| --- | --- |
| `focus-dark-desktop-en.png` | Focus, dark. Only the current section is annotated. |
| `focus-light-desktop-en.png` | Focus, light. |
| `focus-dark-desktop-es.png` | Focus on the Spanish tree: same chips, no translation needed. |
| `focus-dark-mobile-en.png` | Focus inside the mobile menu. |
| `annotate-dark-desktop-en.png` | Every item annotated, dark. |
| `annotate-light-desktop-en.png` | Every item annotated, light. |
| `annotate-dark-mobile-en.png` | Every item annotated, phone. |
| `hover-resting-dark-desktop-en.png` | Hover variant at rest: no marks at all. It is byte for byte the same image as `none-dark-desktop-en.png`, which is the whole point of the variant. |
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
| `api-inline-apipage-dark-desktop-en.png` | Status quo on an API page, scrolled to the reference group. |
| `api-collapsed-apipage-dark-desktop-en.png` | Collapsed mode on an API page: it opens the group by itself and adds the caret next to the label. |
| `api-split-apipage-dark-desktop-en.png` | Split mode on an API page: back link plus reference only. |
| `api-split-apipage-dark-mobile-en.png` | The same on a phone. |
| `switcher-widget-dark-desktop-en.png` | The switcher panel with its three groups. |

## Design critique

**page chips.** The best of everything here, and the first version that does
not have to fight for space. Under the H1 there is a full content column of
horizontal room, so five standards and three named works sit on one line
without wrapping and without displacing anything; the run reads as the
dateline of a standard, which is exactly the register this library wants. Two
coloured dots carry the whole legend, so the text can stay tinted rather than
boxed and never approaches the pill look that was rejected before.

Three things make it better than every sidebar variant rather than just
prettier. It is derived, so it cannot rot: the header and the bibliography are
the same list, and a new reference in the frontmatter shows up in the header
with no second edit. It is linked, so the chip is not a label but a way in:
clicking `ISO 10140` scrolls to the exact entry, highlighted, in a
fourteen-entry list. And it costs the navigation nothing at all, which means
it composes with any sidebar variant instead of competing with it.

Its honest weaknesses. It only helps once you are on the page, so it cannot
answer "which of these neighbouring pages do I want", which is the one thing
the sidebar annotation does. On literature-heavy pages the caps bite:
Underwater sound propagation shows one standard, three works and `+12 more`,
which is truthful but close to useless as a summary, and the choice of which
three works survive is just frontmatter order. The family collapse is a
judgement call: `ISO 10140` is what a reader recognises, but the chip then
links to Part 2 specifically, and someone who wanted Part 4 has to look one
line down. Finally it inherits whatever the frontmatter says, so a sloppy
`references` block shows up in the header rather than staying quietly at the
bottom, which is a feature for me and a risk for a careless page.

**focus.** The strongest of the five sidebar variants. The tree stays exactly as quiet as it is
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
Its remaining cost is that it hides the row underneath while open, because the
sidebar is a scroll container and a card placed outside it would be clipped.
It is a desktop interaction, and on touch it falls back to the always-visible
line; with the desktop weighting that is a footnote rather than a
disqualification, and it makes `hover` the only sidebar variant that can be
combined with the page chips without adding a single pixel to the resting
layout.

**clean.** Superseded. Its subtitle under the H1 was the right instinct, and
the page chips are that instinct done properly: derived from the bibliography
instead of from a parallel hand-written list, linked instead of inert, and
capped instead of unbounded. The two now exclude each other by CSS, since
both sit under the H1 and say the same thing, and the only reason the older
subtitle is still in the branch is so the two can be compared side by side by
toggling `pageChips` off. What is left of `clean` on its own merits is the
section landing table, and that is the weakest part: it duplicates the bullet
list already above it, where each bullet names its standards in prose, it
introduces a heading the markdown pipeline never sees so its "On this page"
entry has to be injected client side, and on a phone the three columns only
work because they collapse into a stacked list.

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

Ship `header` page chips, `hover` for the sidebar, `split` for the API tree,
done with the plugin rather than with my CSS.

**Page chips carry the feature.** They are the only version of this idea that
is derived rather than duplicated, that is a link rather than a label, and
that costs the navigation nothing. If only one thing ships, it is this, and it
should ship whatever happens to the sidebar.

**The sidebar then only has to answer the one question the header cannot:
which of these neighbouring pages do I want.** With the page chips in place,
that is a much smaller job, and it changes which sidebar variant wins.

My earlier recommendation was `focus`, chosen partly because `hover` was
penalised for being unusable on touch. Weighting the desktop the way the
library is actually read, that penalty mostly disappears and the ranking
flips:

- `hover` gives a resting tree that is pixel for pixel today's tree, with the
  standards a pointer-hover or a Tab away, and it adds nothing to the height
  of a sidebar that is already long. Next to a header run that is always
  visible, "quiet until asked" is the right register for the tree, and the two
  never repeat themselves on screen at the same time.
- `focus` remains the best variant if the sidebar has to work on its own, and
  it is the safer choice if the page chips do not ship. It is redundant with
  them on the current page and useful on the neighbours, which is a defensible
  reason to keep it too.
- `annotate` and `clean` are out. `annotate` doubles the height of the tree
  for information the header now gives for free, and `clean` is superseded by
  construction.

So: `header` + `hover` as the shipping pair, with `header` + `focus` as the
conservative alternative if the reveal-on-hover marker turns out to be too
subtle in practice. `header` + `none` is also a perfectly respectable answer
and the cheapest to maintain: everything in the page header, nothing in the
sidebar at all.

Two follow-ups before shipping the page chips:

1. Sort the amber run rather than taking frontmatter order, so the three works
   that survive the cap are the three the page actually leans on. Sorting by
   year, or an explicit `primary: true` flag in the frontmatter, would both
   work; the flag is more honest and costs one schema field.
2. Decide whether the family collapse should link to the first part or to the
   References heading. Landing on Part 2 when the chip says `ISO 10140` is
   slightly arbitrary, and jumping to the section start would be defensible.

If `focus` or `hover` ships as well, trim the sidebar classification to two
standards plus one reference per item so the wrapped three-designation rows
stop happening; the full list is in the header now.

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

- All five `data-toc-style` values, both `data-page-chips` values and all
  three `data-api-style` values render as intended, including the API caret
  toggle and its session memory.
- Page chips: every chip link resolves to an element on the page, on EN and
  on ES, on a page with eight chips and on a page with one. The caps produce
  `+12 more` on the heaviest page and it links to the References heading.
  Pages without a `references` block render no markup at all. With the chips
  on, the older `clean` subtitle is never displayed, in any combination.
- The Spanish build localizes the visually hidden run prefixes, the "+N más"
  label and the two-author join ("Francois y Garrison 1982"), while
  designations and surnames stay language neutral.
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

- pa11y only exercises the default combination, since it starts with an empty
  `localStorage`. That does cover the page chips, which are on by default, but
  the other four sidebar variants and two API modes are not covered by the
  audit. Colour choices were checked by hand instead: the annotation ink is
  about 5.3:1 on the dark sidebar and 5.6:1 on the light one, both above the
  4.5:1 that 11 px text needs.
- One content oddity the chips exposed: the ICAO Annex 16 entry on the
  aircraft-noise page carries `designation: "8th ed."`, which is an edition
  and not a document number, so no chip is derived for it. The helper skips
  designations of that shape rather than printing "8th ed." in the header. The
  real fix is in the frontmatter, on both language versions, and I left it
  alone here to keep this branch to presentation.
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
- every branch of `toc-info.css` for the variants not chosen, and the
  `data-page-chips` gate around `.page-chips` if the run ships unconditionally;
- `SectionStandards.astro` and its import in `MarkdownContent.astro`, unless
  the section table survives;
- the hand-derived subtitle in `PageTitle.astro` (the `.page-standards`
  paragraph and its `findCurrent` helper), which the page chips replace; the
  override itself stays to host `<PageChips />`;
- the `api-caret` button in `SidebarSublist.astro` if the split is done with
  the plugin instead.

What stays whatever the outcome: `src/lib/reference-chips.ts`,
`PageChips.astro` and the bibliography anchors in `References.astro` if the
page chips ship, and the sidebar classification (the `chips()` helpers in
`astro.config.mjs`, the `_API_CHIPS` table in `scripts/generate_api_docs.py`
and the parsing in `SidebarSublist.astro`) if any sidebar variant does. If
only the page chips ship, the whole `data-chips` classification can go: it is
a hand-maintained duplicate of what the frontmatter already says, which is the
strongest argument for the page chips of all.
