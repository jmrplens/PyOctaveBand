# Standards in the docs: page header chips and the API sidebar

Experimental branch `feat/toc-sidebar-info`. It does not merge and has no PR.
Both design questions it was opened for are now settled: the sidebar carries
no standards information, and the page header carries it as one outlined pill
per source. Neither is switchable any more. What is still a prototype is the
API sidebar treatment, which keeps its switcher.

The target is the desktop reader; a design only has to survive a phone, it is
not judged there.

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

### Presentation: one outlined pill per source

Settled. Each chip is a 12 px label in a fully rounded 1 px outline, tinted by
category (standards teal, named works amber), with a faint category fill on
hover and focus. Nothing is filled at rest.

The reasoning: the chips are links, and as plain text they looked exactly like
prose, so their clickability was undiscoverable. An outline is the smallest
mark that says "interactive" without filling anything in, and the border
colour carries the category so nothing else has to. The objection that killed
the sidebar chips does not transfer, because that was a decoration repeated
down 150 rows of a tall tree and this is one row per page in a band that is
otherwise empty. The two presentations that lost, a plain tinted text run and
the same pills with a permanent fill, are deleted along with their CSS and the
`data-page-chips` attribute that switched between them.

The only remaining API prototype dimension is `data-api-style`, localStorage
key `apiStyle`:

| Value | What the reader sees |
| --- | --- |
| `split` (default) | Topic style. On guide pages the API group shrinks to its label with a forward arrow; on API pages the guide groups disappear and a "Guides" back link takes their place. |
| `collapsed` | The API group gets a caret and folds. Folded on guide pages, open on API pages, and the reader's choice is kept for the session. |
| `inline` | Status quo: the whole API tree always expanded in place. |

### Border contrast

The pill outline is a decoration in the sense that the link is identified by
its text, but it is the only thing that draws the pill, so it is held to the
3:1 of WCAG 1.4.11 rather than left as a hairline. The alphas are the lowest
that clear it, measured on the composited colour against the page background:

| | dark | light |
| --- | --- | --- |
| standards border | 50 % alpha, 3.60:1 | 80 % alpha, 3.33:1 |
| works border | 50 % alpha, 3.32:1 | 90 % alpha, 3.40:1 |
| standards text | 12.16:1 | 7.87:1 |
| works text | 10.56:1 | 6.61:1 |

Dark was already close and cost nothing. Light had to go from 35 % to 80 and
90 %, which is a visible change: the outlines go from a whisper to a defined
thin stroke. I judged that as the pills gaining an edge rather than gaining
weight, since the text size, the padding, the radius and the spacing are all
untouched from the version that was approved, and the light amber cannot
reach 3:1 at all below 85 % because solid `#a8760a` on white is only 3.99:1.
If it reads too crisp on the real screen, the two `--chip-*-line` values in
`src/styles/ux-variants.css` are the single place to soften it, at the cost
of the criterion.

### Which sources survive the cap

Five standards and three named works fit on one line at a desktop reading
width; the rest folds into one `+N more` link. Which ones survive is now
deterministic and under the page author's control:

- add `primary: true` to a `references` entry to promote it. Flagged entries
  come first within their category run, and everything else keeps frontmatter
  order. The sort is stable, so a page that flags nothing behaves exactly as
  it did before.
- Flag the standard the page actually implements, or the work its method is
  named after. A page whose bibliography is shorter than the caps does not
  need the flag at all, which is most pages.
- The field is optional on every reference type and documented in the schema
  (`src/content.config.ts`).

### How to switch

- The round "API" button at the bottom right opens the switcher, which now
  holds the API treatment and nothing else. The choice is remembered across
  pages and reloads.
- From the console: `document.documentElement.dataset.apiStyle = 'collapsed'`.
  To make it stick, `localStorage.setItem('apiStyle', 'collapsed')`.
- The default for a fresh browser is `split`. The page chips have no switch:
  they render whenever the page has a `references` block.

## Screenshots

In `ux-variants2/`, all desktop at 1440 x 1000 unless the name says otherwise.
The screenshots of the five rejected sidebar variants, and of the two rejected
chip presentations, were removed along with their code; they are in the branch
history if they are ever wanted again.

| File | What it shows |
| --- | --- |
| `chips-pills-many-dark-desktop-en.png` | The dense case, dark: five ISO families and three books on one line. |
| `chips-pills-many-light-desktop-en.png` | The same in light, with the 3:1 outlines. |
| `chips-pills-many-dark-desktop-es.png` | The same page in Spanish, under a two-line H1. |
| `chips-pills-single-light-desktop-en.png` | A page with one reference and no standard. |
| `chips-pills-overflow-dark-desktop-en.png` | Sixteen references: one standard, three works and the `+12 more` pill. |
| `chips-none-light-desktop-en.png` | A page with no `references` block: nothing rendered. |
| `chips-anchorjump-light-desktop-en.png` | After clicking the `ISO 10140` pill: the matching bibliography entry, highlighted and clear of the header. |
| `chips-anchorjump-dark-desktop-es.png` | The same jump on the Spanish build. |
| `sidebar-final-dark-desktop-en.png` | The tree as it stays: no chips, no annotations. |
| `api-inline-guidepage-dark-desktop-en.png` | API tree expanded in place on a guide page. |
| `api-collapsed-guidepage-dark-desktop-en.png` | API group folded behind a caret. |
| `api-split-guidepage-dark-desktop-en.png` | API group reduced to a label with an arrow. |
| `api-inline-apipage-dark-desktop-en.png` | Status quo on an API page, scrolled to the reference group. |
| `api-collapsed-apipage-dark-desktop-en.png` | Collapsed treatment on an API page: it opens the group by itself and adds the caret. |
| `api-split-apipage-dark-desktop-en.png` | Split on an API page: back link plus reference only. |
| `api-split-apipage-dark-mobile-en.png` | The same at 390 px. |
| `switcher-widget-dark-desktop-en.png` | The switcher, down to the API treatment alone. |

## Why the pill, and what it cost

The three presentations that were compared live were a tinted text run with a
category dot per run, the outlined pill, and the outlined pill with a
permanent fill. The pill won and the other two are deleted.

The text run was the quietest and read most like the dateline of a standard,
but it hid the fact that the chips are links: the only affordance was a hover
underline, which nobody hovers to discover. The filled pill added a third
visual channel that repeated what the border and the text colour already said,
and at eight chips the row started to read as a strip of status badges rather
than a citation line; the fill also had to stay so faint to avoid that, that
on the dark theme it was nearly invisible anyway.

The outlined pill buys the affordance for one hairline per chip. What it cost
is the light-theme outline going to 80 and 90 % alpha to clear 3:1, which is
the one visible change from the version that was approved at 35 %. Numbers and
the single place to soften it are in the border contrast section above.

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

- The chips render unconditionally wherever a page has references, in the one
  settled style, and all three `data-api-style` values still work including
  the caret toggle and its session memory.
- The sidebar renders no chips, no annotation line and no hover card in any
  state, and no `data-toc-style`, `data-page-chips`, `.sidebar-chips`,
  `.page-standards` or `.section-standards` remains anywhere in `site/src` or
  `scripts/`.
- Every chip link resolves to an element on the page, on EN and on ES, with
  eight chips and with one. The caps produce `+12 more` on the heaviest page.
  Pages without a `references` block render no markup at all.
- The run stays on one line at 1440 px on the densest page.
- Contrast, measured on the composited colours (see the table above): borders
  3.60:1 and 3.32:1 dark, 3.33:1 and 3.40:1 light, all clearing 3:1; text
  12.16:1 and 10.56:1 dark, 7.87:1 and 6.61:1 light, all clearing the 4.5:1
  that 12 px text needs. Measure these after the theme transition settles: the
  0.12 s border transition makes a synchronous read return the previous
  theme's colour, which is how I first mis-measured them.
- The `primary: true` flag was exercised end to end by flagging one entry on
  the aircraft-noise page, confirming it jumped to the front of its run, and
  reverting the flag.
- The ICAO Annex 16 entry now carries `designation: "ICAO Annex 16, Vol. I,
  8th ed."` (and the `8.ª ed.` mirror in Spanish) instead of a bare edition
  string, so that page finally derives an `ICAO Annex 16` chip.
- `pnpm --dir site build` succeeds and the Starlight link validator reports
  all internal links valid; `pnpm run html-validate` passes; `pnpm run pa11y`
  passes 46 of 46 URLs at WCAG2AA; `node scripts/check-i18n-parity.mjs`
  passes.

Known limits:

- pa11y exercises the chips, since they are no longer behind a switch, but
  only the default API treatment.
- The same aircraft-noise page has a second cryptic designation,
  `Doc 9501, 3rd ed.`, which yields a `Doc 9501` chip with no issuing body in
  it. Prefixing it `ICAO Doc 9501` would read better; I left it alone because
  the brief was the Annex 16 block only.

## What is prototype scaffolding

The page chips are not a prototype any more: `src/lib/reference-chips.ts`,
`PageChips.astro`, the minimal `PageTitle.astro` override that mounts it, the
`.page-chips` rules and the bibliography anchors and `:target` highlight in
`References.astro` are all keepers, as is the `primary` field in the content
schema.

What still comes out when the API treatment is chosen:

- both inline scripts at the bottom of `Head.astro` and the `.toc-switcher`
  rules in `src/styles/ux-variants.css`;
- the two treatments not chosen;
- the `api-caret` button in `SidebarSublist.astro` and the `topic-back` link
  in `Sidebar.astro`, if the split is done with `starlight-sidebar-topics`
  instead.
