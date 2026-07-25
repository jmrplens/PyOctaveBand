# Standards in the docs: page header chips and the API sidebar

Experimental branch `feat/toc-sidebar-info`. It does not merge and has no PR:
it exists so I can compare presentations live on the running site and then
keep one. The target is the desktop reader; a design only has to survive a
phone, it is not judged there.

## The sidebar decision is final

The sidebar carries no standards information. None. Not a chip, not an
annotation line, not a hover card.

Two rounds of prototypes went the other way and both failed for the same
reason. `exp/sidebar-standard-chips` put a coloured badge on every item and
was too loud. This branch then tried five quieter presentations (`focus`,
`annotate`, `hover`, `clean`, `none`) and the best of them was still paying
for information on every row of a very tall tree that a single row in the
page header gives for free. The whole mechanism is now deleted rather than
switched off:

- gone: the `data-toc-style` attribute and its five variants, the annotation
  line, the hover card and its keyboard handling, the page subtitle, the
  section landing tables;
- gone: `SectionStandards.astro`, `src/styles/toc-info.css`, the
  `attrs['data-chips']` classification in `site/astro.config.mjs` and the
  `_API_CHIPS` table in `scripts/generate_api_docs.py`, together with the
  parsing that consumed them;
- gone with them: the shortened sidebar labels the chips had justified, so
  entries like "Reverberation-time prediction (Sabine, Eyring, Arau)" read in
  full again.

`SidebarSublist.astro` and `Sidebar.astro` are back to the smallest diff
against upstream Starlight 0.41.3 that still implements this repo's
conventions. What differs from upstream, and why, is documented in each file
header: groups are never collapsible, a group's marked landing entry is
consumed so the group label itself becomes the link (the overview-first
convention), and the API reference group carries a class plus a fold toggle
for the API treatments below. That last one is the only piece of this
experiment left in the sidebar, and it comes out if the API split is ever
done with a plugin.

## Page chips

One run under the H1: the standards that govern the page, then the works its
methods are attributed to. Nothing in the navigation.

### Derived from the bibliography, not from a second list

Each page already carries a Zod-typed `references` block in its frontmatter
(schema in `src/content.config.ts`), rendered as the APA-7 References section
at the bottom by `References.astro`. `src/lib/reference-chips.ts` turns that
same list into the header run, so the two are views of one source and cannot
drift apart:

- `type: standard`, and `type: report` when it carries a document number,
  become the standards run. The designation collapses to its family, so
  `ISO 10140-2:2010`, `-3` and `-4` become one `ISO 10140`, and
  `ANSI S3.5-1997 (R2017)` becomes `ANSI S3.5`.
- `type: article` and `type: book` become the references run, as author and
  year: `Schroeder 1965`, `Francois & Garrison 1982`, `Foret et al. 2011`.
  The Spanish build joins two authors with "y", like the bibliography does.
- `type: web` and numberless reports have neither a designation nor a citable
  author-date pair, so they stay in the bibliography only.
- Five standards and three works fit on one line at a desktop reading width.
  Anything past that folds into one `+N more` link to the References section.
- Pages with no `references` render nothing at all.

Every chip links to its entry, which carries a stable id (`referenceAnchors`
derives the ids from frontmatter order, so the alphabetical sort of the
bibliography does not move them). The target clears the sticky header and is
highlighted on arrival, so a click on `ISO 10140` lands the reader on the
right line of a fourteen-entry list.

### Presentations

`data-page-chips` on `<html>`, localStorage key `pageChips`:

| Value | What the reader sees |
| --- | --- |
| `header` | Tinted text, one small colour dot leading each category run. No boxes. |
| `pills` (my recommendation) | The same content as outlined pills: 12 px text, 1 px category-coloured border at 38 % alpha, fully rounded, transparent inside. |
| `filled` | The same pills with a faint category tint inside (18 % dark, 12 % light). |
| `off` | Nothing. |

`data-api-style`, localStorage key `apiStyle`:

| Value | What the reader sees |
| --- | --- |
| `split` (default) | Topic style. On guide pages the API group shrinks to its label with a forward arrow; on API pages the guide groups disappear and a "Guides" back link takes their place. |
| `collapsed` | The API group gets a caret and folds. Folded on guide pages, open on API pages, and the reader's choice is kept for the session. |
| `inline` | Status quo: the whole API tree always expanded in place. |

### How to switch

- The round "UX" button at the bottom right opens the switcher, now with two
  groups: page chips and API sidebar. The choice is remembered across pages
  and reloads.
- From the console: `document.documentElement.dataset.pageChips = 'pills'`,
  `document.documentElement.dataset.apiStyle = 'collapsed'`. To make it stick,
  `localStorage.setItem('pageChips', 'pills')`.
- Defaults for a fresh browser are `header` and `split`.

## Screenshots

In `ux-variants2/`, all desktop at 1440 x 1000 unless the name says otherwise.
The screenshots of the five rejected sidebar variants were removed along with
the code; they are in the branch history if they are ever wanted again.

Page chips, on a page with eight chips (Laboratory Insulation Measurement), a
page with a single chip, a page that overflows the caps and a page with no
references at all:

| File | What it shows |
| --- | --- |
| `chips-header-many-dark-desktop-en.png` | Text run, dark. |
| `chips-header-many-light-desktop-en.png` | Text run, light. |
| `chips-header-many-dark-desktop-es.png` | Text run in Spanish, under a two-line H1. |
| `chips-header-single-dark-desktop-en.png` | Text run, one reference and no standard. |
| `chips-header-overflow-light-desktop-en.png` | Text run with `+12 more`. |
| `chips-pills-many-dark-desktop-en.png` | Outlined pills, dark. |
| `chips-pills-many-light-desktop-en.png` | Outlined pills, light. |
| `chips-pills-many-dark-desktop-es.png` | Outlined pills in Spanish. |
| `chips-pills-single-light-desktop-en.png` | Outlined pills, single chip. |
| `chips-pills-overflow-dark-desktop-en.png` | Outlined pills with the `+12 more` pill. |
| `chips-filled-many-dark-desktop-en.png` | Filled pills, dark. |
| `chips-filled-many-light-desktop-en.png` | Filled pills, light. |
| `chips-filled-many-light-desktop-es.png` | Filled pills in Spanish. |
| `chips-filled-single-dark-desktop-en.png` | Filled pills, single chip. |
| `chips-none-light-desktop-en.png` | A page with no `references` block: nothing rendered. |
| `chips-anchorjump-light-desktop-en.png` | After clicking `ISO 10140`: the matching entry, highlighted and clear of the header. |
| `chips-anchorjump-dark-desktop-es.png` | The same jump on the Spanish build. |

Sidebar and API:

| File | What it shows |
| --- | --- |
| `sidebar-final-dark-desktop-en.png` | The tree as it stays: no chips, no annotations. |
| `api-inline-guidepage-dark-desktop-en.png` | API tree expanded in place on a guide page. |
| `api-collapsed-guidepage-dark-desktop-en.png` | API group folded behind a caret. |
| `api-split-guidepage-dark-desktop-en.png` | API group reduced to a label with an arrow. |
| `api-inline-apipage-dark-desktop-en.png` | Status quo on an API page, scrolled to the reference group. |
| `api-collapsed-apipage-dark-desktop-en.png` | Collapsed treatment on an API page: it opens the group by itself and adds the caret. |
| `api-split-apipage-dark-desktop-en.png` | Split on an API page: back link plus reference only. |
| `api-split-apipage-dark-mobile-en.png` | The same at 390 px. |
| `switcher-widget-dark-desktop-en.png` | The switcher, now two groups. |

## Design critique of the three chip presentations

**header (text run).** The quietest, and the one that reads most like the
dateline of a standard: no boxes, two coloured dots carrying the entire
legend, the run sitting under the H1 as if it were part of the title block.
Its flaw is the one that prompted this round: the chips are links and they
look exactly like plain text, so nothing tells the reader they can click
through to the bibliography. The affordance appears only on hover, which is
the definition of undiscoverable.

**pills (outlined).** My recommendation. It buys the affordance with the
least possible ink: a 1 px border at 38 % alpha and a full radius, 12 px text
instead of 14. The shape says "interactive", the border colour says which
category, and nothing is filled, so the run stays a hairline drawing rather
than a block of colour. The noise objection that killed the sidebar chips
genuinely does not transfer: that was a decoration repeated down 150 rows of
a tall tree, this is one row per page in a band that is otherwise empty.
Eight pills still fit on one line at a desktop width, and the gap between the
two category runs is wider than the gap inside a run, so the standards and
the named works still read as two groups without needing the dots.

**filled.** The same pills with a faint category tint. It groups each chip
slightly better and makes the category legible without reading the border
colour, but it adds a third visual channel (fill) that repeats what the
border and the text colour already say, and at eight chips the row starts to
read as a strip of status badges rather than as a citation line. The fill
also has to stay very low (18 % dark, 12 % light) to avoid that, and at that
alpha it is nearly invisible on the dark theme anyway, which makes it a lot
of machinery for very little.

Ranking: `pills`, then `header`, then `filled`. If discoverability turns out
not to be worth any ink at all, `header` is the fallback and nothing else
changes; the three styles share all their markup.

One caveat before shipping pills: the border sits at 1.2:1 against the light
background and 2.3:1 against the dark one. That is fine as decoration, since
the link is identified by its text, which clears 5.9:1 to 8.1:1, but it would
not pass as a UI component boundary under WCAG 1.4.11 if anyone argued that
it is one. Raising the border alpha fixes it and costs some quiet; I left it
quiet.

## `split` versus `starlight-sidebar-topics`

My `split` prototype and the plugin solve the same problem, and the plugin
solves it better in the ways that matter long term.

- The plugin makes each topic a root sidebar, so the API items lose the two
  wrapper levels they currently sit under, and the topic switcher is a real
  navigation control rather than a link that only appears on some pages. My
  prototype only hides things with CSS, so the nesting, and the wrapping
  module names that come with it, stay.
- The plugin's topic labels and badges take per-locale objects. Mine
  hardcodes the back link label in `Sidebar.astro`.
- The plugin is maintained. Mine is about 30 lines of CSS plus a hand-written
  back link, and every future Starlight release is my problem.

What to check before adopting it: this repo overrides `Sidebar.astro` and
`SidebarSublist.astro`, and the overview-first convention lives in that
override. If the plugin also overrides `Sidebar`, one of the two has to give.
The plugin docs do not say. That is an afternoon, not a blocker.

## Verification

Checked in Chrome at 1440 x 1000, both themes, EN and ES, on a guide page and
an API page:

- All four `data-page-chips` values and all three `data-api-style` values
  render as intended, including the API caret toggle and its session memory.
- The sidebar renders no chips, no annotation line and no hover card in any
  state, and no `data-toc-style`, `.sidebar-chips`, `.page-standards` or
  `.section-standards` remains anywhere in `site/src` or `scripts/`.
- Every chip link resolves to an element on the page, on EN and on ES, with
  eight chips and with one. The caps produce `+12 more` on the heaviest page.
  Pages without a `references` block render no markup at all.
- All three chip presentations keep the run on one line at 1440 px on the
  densest page.
- Text contrast measured against the composited chip background: 8.08:1 and
  7.24:1 dark, 7.18:1 and 5.94:1 light, all above the 4.5:1 that 12 px text
  needs.
- `pnpm --dir site build` succeeds and the Starlight link validator reports
  all internal links valid; `pnpm run html-validate` passes; `pnpm run pa11y`
  passes 46 of 46 URLs at WCAG2AA; `node scripts/check-i18n-parity.mjs`
  passes.

Known limits:

- pa11y only exercises the default combination, since it starts from an empty
  `localStorage`. That covers the `header` style; `pills` and `filled` were
  checked by hand with the contrast figures above.
- One content oddity the chips exposed: the ICAO Annex 16 entry on the
  aircraft-noise page carries `designation: "8th ed."`, an edition rather
  than a document number, so no chip is derived for it. The helper skips
  designations of that shape rather than printing "8th ed." in the header.
  The real fix is in the frontmatter, on both language versions; I left it
  alone to keep this branch to presentation.

## What is prototype scaffolding

When a presentation is chosen, these come out:

- both inline scripts at the bottom of `Head.astro` and the `.toc-switcher`
  rules in `src/styles/ux-variants.css`;
- the `data-page-chips` gate and the two presentations not chosen;
- the `api-caret` button in `SidebarSublist.astro` and the `topic-back` link
  in `Sidebar.astro`, if the API split is done with the plugin instead.

What stays: `src/lib/reference-chips.ts`, `PageChips.astro`, the minimal
`PageTitle.astro` override that mounts it, and the bibliography anchors and
`:target` highlight in `References.astro`.
