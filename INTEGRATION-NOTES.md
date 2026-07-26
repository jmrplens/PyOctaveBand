# Integration: the site palette and the page chips on one branch

`feat/site-and-toc` is the two design branches merged and made to work
together. They were designed in isolation against the same starting point and
they touch the same page header, so the merge being clean said nothing about
the result being coherent. This is what I checked, what I changed, and what is
still open.

The two source branches stay where they are; this branch is what ships.

## What came from each branch

From `feat/toc-sidebar-info`:

- The sidebar back to a clean navigation tree. Every standards annotation,
  hover card and count that the experiment put in it is gone, and no
  `data-toc-style` or `.sidebar-chips` remains.
- The page header chips: one run of outlined pills under the H1 naming the
  standards that govern the page and the works its methods are attributed to,
  derived from the page's own Zod-typed `references` frontmatter by
  `site/src/lib/reference-chips.ts` and rendered by `PageChips.astro` through
  a `PageTitle` override. Each pill links to its entry in the page's
  References section, which is generated from the same list, so the header and
  the bibliography cannot drift apart.
- The `primary: true` frontmatter flag that decides which sources survive the
  caps of five standards and three named works.
- The API sidebar treatment, then still switchable from a small floating
  panel. It is decided now and the panel is gone; see "The API sidebar"
  below.

From `feat/site-redesign`:

- The `instrument` palette as the one palette the site has, promoted to a
  plain `:root` in `site/src/styles/theme.css`: cool steel greys and a single
  saturated deep cyan, constrained by the committed figures, which are opaque
  plates and are not touched by any of it.
- The reworked front page (`src/components/home/Home.astro` and
  `src/data/home.ts`) with clause-numbered sections and the coverage table.
- The mobile theme toggle in the `Header.astro` override, so the theme and the
  language controls are not both buried behind the hamburger on a phone.
- The first-visit language bar (`LangSuggest.astro`), which offers the other
  locale and never navigates on its own.

## The gitignore trap

The root `.gitignore` carries `lib/`, a distutils build-artifact rule from the
Python side of the repo. A bare `lib/` matches at any depth, so it also
matched `site/src/lib`, and `reference-chips.ts` (the module every chip on the
site is derived from) was never committed. Everything worked in the branch
worktree, where the file exists on disk, and the site could not build from a
fresh clone.

The fix is the negation now sitting under that rule:

```gitignore
# The Python build-artifact rule above also matches the site's source
# helpers directory, which is real source and must stay tracked.
!site/src/lib/
```

Do not delete it while tidying the ignore file. Anything else the site grows
under a `lib/` name will hit the same rule, and the failure mode is silent:
`git status` stays clean and only a clean checkout shows the breakage.

## Colour: the chips against the instrument palette

This was the real risk of the merge. The chips were tuned against the old
Starlight default palette, where their teal and amber were the only saturated
colours on the page. Against `instrument` they were doing two things I do not
want, both visible in the first captures:

- The standards teal sat at OKLCH hue 181 with the palette accent at 216
  (dark) and 225 (light). Forty degrees apart is the worst distance there is:
  far enough to be a different colour, close enough to read as a near-miss of
  the accent rather than as a category.
- Both chip inks carried more chroma than the accent itself: 0.103 for the
  teal and 0.141 for the amber, against 0.056 for the accent on the dark
  theme. The header run was the loudest thing on the page, and the premise of
  this palette is neutral metal plus one colour that means something.

So the standards chip now takes the palette accent itself. That is honest:
the standards are what the page is about, the chip is a link, and the accent
is the site's one meaningful colour, which the front page coverage table
already uses for exactly the same content. The named works take its warm
mirror, the same OKLCH lightness and chroma at hue 74 (dark) and 70 (light),
which keeps the two categories 140 degrees apart without adding a second loud
colour.

The tokens moved into `theme.css` with the rest of the palette, as
`--ph-chip-*`, and `ux-variants.css` only references them. The borders are
opaque colours rather than alphas over the ground, which is what let them
live in the palette and be audited with everything else.

| | dark | light |
| --- | --- | --- |
| standards text | `var(--ph-accent-ink)` `#b3e6f4`, 14.03:1 | `var(--sl-color-accent)` `#0a6f8c`, 5.73:1 |
| standards border | `#2f7688`, 3.67:1 | `#4994b0`, 3.42:1 |
| standards text on hover fill | 10.06:1 | 5.00:1 |
| named-work text | `#f8d6a9`, 13.71:1 | `#865a1f`, 6.01:1 |
| named-work border | `#856537`, 3.53:1 | `#aa804e`, 3.56:1 |
| named-work text on hover fill | 9.96:1 | 5.24:1 |
| "+N more" text | `var(--sl-color-gray-3)`, 6.36:1 | `var(--sl-color-gray-3)`, 6.36:1 |
| "+N more" border | `var(--sl-color-gray-4)`, 3.04:1 | `var(--sl-color-gray-4)`, 3.58:1 |

`scripts/check-contrast.mjs` now measures all of it, nine pairs per theme,
text at 4.5:1 and border at 3:1 (the border is what draws the pill, so it is
meaningful non-text under WCAG 1.4.11), each label on the page ground and on
its own hover fill. 43 pairs measured across the two themes, none below
threshold. The tightest pair on the whole site is the "+N more" border at
3.04:1 on the dark theme, which is the neutral `--sl-color-gray-4` and was
not chosen for the chips at all.

One measuring note, the same one the chip branch recorded: the pill has a
0.12 s border transition, so reading a computed border colour immediately
after a theme switch returns the previous theme's value. Let it settle.

## The cap: nine chips for the run as a whole

The run used to cap at five standards plus three named works. It is nine chips
now, whatever they are, with a tenth slot for the "+N more" link when the page
has more than nine.

The budget is the run as a whole rather than one cap per category, with one
guard: a whole-run budget spent in order would let a standards-heavy page push
every named work out, and the second category is exactly what the run exists
to state. So the named works hold a reservation of up to three slots when the
page has any, the standards take the rest, and whatever the standards do not
use goes back to the works. Nine standards on a standards-only page, nine
papers on a paper-only page, six and three on a page with fourteen standards
and three papers, two and seven on a page with two standards and eight papers.

`primary: true` still decides who survives: the flagged entries are ordered
first and the cap is applied after. Flagging Jensen et al. 2011 on the
underwater page, which sat past the old cap, brought it to the front of the
works run and into the visible nine, and pushed the last unflagged work out.
The flag was reverted after the check; no page carries one today.

What this costs: nine pills do not fit the one line the run was designed
around. At 1440 px the densest page (`underwater-propagation`, one standard
and eight papers with long author-date labels) is four rows, and at 390 px it
is six. The categories still read as groups, because the wider gap between the
two runs survives the wrap, and the more chip ends the run on its own row. It
is a block under the H1 now rather than a dateline, which is the trade the
higher cap buys.

## Where the chips land

Every chip is a link into the page bibliography, and the two kinds of chip
point at different elements: an entry chip at its `<li>`, the "+N more" chip
at the References heading. The entries carried a landing offset and the
heading did not, so the same click put an entry a comfortable way down the
viewport and left the heading hard against the sticky bar. Neither was
actually hidden, because Starlight's own `scroll-padding-top` on the scroll
container already clears the sticky chrome (the header, plus the mobile table
of contents bar, which is why a phone needs more of it than a desktop), but
the two landings did not match and on a phone the difference is the whole gap.

There is one custom property in one rule now, covering the entries, the
heading and its wrapper, so the two targets cannot drift apart again. Both
come to rest at 168 px from the top of a 1440 px desktop viewport and 200 px
on a 390 px phone, in both themes, EN and ES, with and without the language
bar up.

## Where the chips are, and where they are not

- Guide pages: the run sits under the H1 and above the content divider, one
  line at 1440 px even on the densest page, wrapping to two on a phone.
- The front page carries no chips and should not. It is a splash template
  with no `references` block, and it already answers the same question better
  at its own scale: the "What it covers" table lists the designations actually
  implemented per area, and those tags are accent-coloured, which is now the
  same rule the header chips follow rather than a second convention.
- API reference pages carry no chips: generated pages have no `references`
  frontmatter, so `PageChips.astro` renders no markup at all.
- Spanish pages render the run with Spanish labels and the "y" conjunction in
  two-author works, under a title that wraps to two lines without colliding.

## The language bar

Two defects, one of them created by the merge.

The bar was mounted in the `Footer` override, which renders inside
Starlight's `.main-pane`. That pane sets `isolation: isolate`, so the bar got
its own stacking context and painted under the sidebar pane whatever its
z-index said. On a desktop layout the sentence was cut mid-word by the
sidebar and the dismiss button was cut by the table of contents. It is
mounted in a `PageFrame` override now, a sibling of the page frame in the
root stacking context, where its z-index means what it reads. The frame
renders on every route, splash included, so nothing lost the bar and the
English-only API subtree still opts out.

The second one is the merge's own: both branches pinned something to the
bottom edge, so the prototype treatment switcher landed on the bar's dismiss
button. The bar answered by publishing its height on `<html>` so the switcher
could step over it. The switcher is gone now and nothing else is pinned to
that corner, so the measurement went with it: no `--ph-lang-banner`, no
resize listener, one fewer layout read per dismissal.

It never collides with the chips or the H1, on any layout: it is pinned to the
bottom edge and the chips are in the page header band.

`check-lang-suggest.mjs` gained the scenario that would have caught the
stacking: hit test the bar over the sidebar, over the content and over the
table of contents, plus the dismiss button itself. The markup was always
right, so only a hit test can see it.

## The mobile theme control is a toggle now

The header was lifting Starlight's theme select out of the hamburger, so the
mobile control was still a menu: open it, read three options, pick one. It is
a button (`src/components/ThemeToggle.astro`), and one tap flips light and
dark.

Auto stays out of the cycle and stays implicit. With nothing stored the site
follows the system preference and keeps following it live; the first tap is
what turns an inferred theme into a stored choice. Anyone who wants to hand
the decision back to the system still has Starlight's own select, with its
explicit Auto option, in the mobile menu.

No theme state is forked. The button sets the value on Starlight's `<select>`
and fires its `change` event, so the site lands in exactly the state it would
have been in had the reader picked that option from the menu:
`StarlightThemeProvider.updatePickers()`, `data-theme` on `<html>` and the
`starlight-theme` key are all written by Starlight's own handler. The button
then reads its state back from `data-theme`, which is the output of that same
path, so it cannot disagree with the desktop control or with the before-paint
script. The one line that does not go through the select is a fallback for the
moment before Starlight's custom element upgrades, and it writes the same key
and calls the same picker sync.

The icon is the mode the tap would GIVE you, not the mode you are in (a moon
on a light page, a sun on a dark one), and the accessible name says the same
in words, "Switch to dark theme" or "Cambiar al tema oscuro", updated from the
same attribute by a `MutationObserver`. Icon and name are two statements of
one thing and cannot drift.

Driven in headless Chrome from both system preferences: the initial state
follows the system, a live system flip is followed while nothing is stored,
Enter on the focused button flips it, a tap flips it back, a reload keeps it,
a system flip after a choice is ignored, and clearing storage returns to
following the system. Both Starlight selects agree with the button after every
step. pa11y stays at 46 of 46.

Left alone on purpose: the desktop control is still Starlight's select and
still shows the CURRENT mode rather than the one a click would give. The two
are never on screen at once (the mobile group is `md:sl-hidden`, the desktop
group `sl-hidden md:sl-flex`), so nothing reads as inconsistent side by side,
and a reader who resizes past the breakpoint gets a different control anyway.
If you want them to match, the cheaper direction is to use the toggle in both
places and keep the select only in the mobile menu, where Auto lives; that
would cost the desktop reader the visible Auto option, which is why I did not
do it unasked.

## The API sidebar: one behaviour, and it is progressive disclosure

Decided: the API reference sits collapsed in the sidebar, and clicking it
opens the tree. `split` and `inline` are deleted, with the `data-api-style`
and `data-area` dimensions, the floating switcher and both of its inline
scripts in `Head.astro`, the `.toc-switcher` rules, and the "Guides" back
link in `Sidebar.astro` that only `split` ever displayed. The last commit
that still carries all three treatments and the switcher is `8eb915a7`; that
is where to look if any of it is ever wanted again.

### What the prototype did, and what ships instead

The prototype was one fold of one group, bolted on from a script. What ships
is a disclosure per level, rendered by the tree itself. The differences are
not cosmetic:

| | prototype | now |
| --- | --- | --- |
| How many things fold | the API group only; every category under it was always fully expanded, so one click dumped 150 rows | every group from the API reference down, so the reader drills in one level at a time |
| Where the state comes from | a `DOMContentLoaded` script that read `sessionStorage` and then folded what the server had already painted open | the server, from the current route: `holdsCurrent()` opens exactly the chain down to the current page |
| On an API page | the whole API tree unfolded at once | the chain down to the page is open, its siblings are closed, the page is marked as Starlight marks it |
| Memory | one `apiFolded` flag for the whole site, applied on guide pages too, so a fold made on one page silently governed another | none, deliberately; see below |
| Control | a caret with `aria-expanded`, `aria-controls` assigned by script after load | a button per group with both attributes server-rendered, the whole label row being the control where the group has no landing page |
| First paint | the group painted open and folded shut a moment later | the tree paints in its final state; there is nothing for the browser to correct |

The one thing the prototype did that survives unchanged is the caret next to
the API reference label: that group has a landing page, the overview-first
convention says the label is a link to it, and a link cannot also be a
disclosure. So the API row has two targets, and both of them end up showing
the tree: the caret opens it in place, the label navigates to the API
overview, which is an API page, which arrives with the tree open. The
category groups below have no landing page, so there the whole row is the
button and the thing the reader clicks is the label itself.

### Whether an opened state survives navigation: no, and why

The open state is a function of the route and nothing else. No
`sessionStorage`, no cookie, nothing carried across a page load.

The case for persisting is the reader who opens the API on a guide page and
wants it still open on the next guide page. The case against is that the
state worth restoring, the path to where the reader is, is already
reconstructed on every render, and the rest is guesswork about intent:

- Within the API, persistence buys almost nothing. Arriving on any API page
  already opens the chain down to it. A remembered branch could only add
  groups the reader is not in.
- Outside the API, a remembered open state is the prototype's actual bug in
  a new coat: a fold made on one page governing a different page, with no
  visible cause and no way to reason about it.
- Persistence costs either a flash (restore after paint, which is what the
  prototype did) or a blocking inline script that runs before the sidebar
  markup exists. The first is the layout shift this design set out to remove
  and the second is a script in the critical path for a convenience.

If it is ever wanted, the scope is fixed by the same argument: an API-only
key, written only while the reader is on an API page, read only into the API
subtree, never allowed to close the chain the route asked for.

### Accessibility

Each control is a real `<button type="button">` with `aria-expanded` and
`aria-controls` naming the `<ul>` it governs, so it is in the tab order, it
answers Enter and Space by being a button rather than by a key handler, and
its state is announced. The control with no visible text of its own (the
caret beside the API reference link) carries an `aria-label`, in Spanish on
the Spanish build. A collapsed list keeps its `<li>`s and `<a>`s in the
document under a `hidden` attribute, so the links are there for crawlers and
for the browser's own find-in-page, and the reader reaches them with one
activation rather than never. Opening moves nothing outside the sidebar,
which is its own scroll container; the check below asserts that against the
main frame's box rather than trusting it.

The guides side of the tree is untouched. It carries no disclosure at all:
those groups are non-collapsible by the overview-first convention, which is
the whole reason this override exists.

### The upstream delta

`Sidebar.astro` is down to 44 lines against upstream's 15, and what is left
is the two import paths, the local `SidebarSublist`, and the six-line click
handler. Dropping `split` took the back link, its label translation and its
base-URL arithmetic out, which is the shrink the decision was expected to
produce.

`SidebarSublist.astro` moved the other way, from 286 lines to 384, because
the behaviour that used to live in a prototype script now lives in the
component that renders the tree. That is the trade I want: upstream's own
collapsible groups are `<details open>` plus `SidebarPersister`, so a
disclosure in this file is a variation on upstream behaviour rather than a
foreign mechanism, and the site as a whole carries 327 lines less than it
did.

Choosing `collapsed` over `split` also settles the plugin question the chip
branch raised: `starlight-sidebar-topics` was the better way to do a topic
split, and there is no topic split. The site keeps that dependency off, and
keeps `Sidebar.astro` free for the overview-first convention that would have
collided with the plugin's own override.

## Gates

All run from this worktree against the integrated branch.

| Gate | Result |
| --- | --- |
| `pnpm --dir site build` (with the Starlight link validator) | pass: 441 pages, all internal links valid |
| `pnpm --dir site html-validate` | pass: no findings on 445 files |
| pa11y-ci, WCAG2AA | pass: 46 of 46 URLs, 0 errors |
| `node site/scripts/check-i18n-parity.mjs` | pass: 100 EN pages each have an ES translation |
| `node site/scripts/check-contrast.mjs` | pass: 43 pairs, 0 below threshold |
| `node site/scripts/check-lang-suggest.mjs` | pass: 10 scenarios, 0 failing |
| `node site/scripts/check-home-headings.mjs` | pass: 40 layout cases, 0 failing |
| `node site/scripts/check-page-chips.mjs` | pass: 7 cap checks and 8 landing checks, 0 failing |
| `node site/scripts/check-api-sidebar.mjs` | pass: 30 checks, 0 failing |

pa11y was run against `astro preview` on port 4390, with a copy of
`.pa11yci.json` rewritten to that port, because the branch's dev server holds
4321 and another branch holds 4322. The npm script's own default port is
4321, so it cannot be used while the dev server is up. The whole table above
was run against that same built preview rather than against the dev server,
so the numbers are the built site's.

The puppeteer checks default to `--base http://localhost:4322`, which is
another branch's dev server on this machine. Pass `--base` explicitly or they
will report on the wrong build, which they did to me once.

`check-api-sidebar.mjs` drives the disclosures rather than reading the
markup: it clicks the control, clicks a branch inside it, presses Enter and
Space on the focused control, navigates to an API page and back to a guide,
and asserts what the reader would see at each step, in both locales. It is
also where the two decisions above are pinned down: that a second guide page
starts closed again (no memory), and that the guides tree carries no
disclosure at all.

One dev-server note, for whoever repeats this: the managed `astro dev` process
did not pick up frontmatter edits (a changed title did not appear either), so
a chip cap or a `primary: true` flag looked like it had no effect. Restarting
it with `--force`, which clears the content layer cache, fixed it. Component
and stylesheet edits were hot-reloading normally throughout.

## Screenshots

`integration-shots/`, captured by `site/scripts/integration-shots.mjs`, which
drives its own headless Chrome out of the pnpm store the way
`redesign-shots.mjs` does.

| Shot | What it shows |
| --- | --- |
| `01`, `02` | Guide header, light and dark: the retuned run under the H1. |
| `03`, `04` | A bibliography past the cap: nine chips and `+7 more`, four rows at 1440 px. |
| `05`, `06` | Body links and code further down the same page, unchanged by any of it. |
| `07` | The References section the chips point into. |
| `08`, `09` | An API page: no chips, and the full guides tree still in the sidebar. |
| `10`, `11` | The front page, full height, both themes. |
| `12`, `13` | A Spanish guide, both themes. |
| `14`, `15` | The same guide at 390 px in the two themes, which is also the theme toggle in its two states: a moon on the light shot, a sun on the dark one, no caret on either. |
| `16` | The front page at 390 px. |
| `17` | A page past the raised cap at 390 px, where the run wraps to five rows and ends on the more chip. |
| `20`, `21` | The language bar on desktop and on a phone, after the stacking fix. |
| `30`, `31` | Arriving on a guide page, both themes: the API group closed, its caret pointing at the rows it is holding back, everything around it unchanged. |
| `32`, `33` | After one click: the twenty category groups, each closed. This is the shot that says the tree does not unfold all at once. |
| `34`, `35` | After a second click, on Psychoacoustics: its thirteen pages, and only its pages. |
| `36`, `37` | An API page with no click at all: the chain down to `sharpness` open, the page marked, the sibling categories closed. |

Shots `30` to `37` are cropped to the sidebar pane, which is the whole
subject; `08` and `09` are the same tree in the context of a full page.

Nothing in `integration-shots/` shows a treatment that does not ship any
more, and the `api-inline-*`, `api-split-*`, `api-collapsed-*` and switcher
captures in `ux-variants2/` are deleted. They are in the history at
`8eb915a7` with the code they documented.
