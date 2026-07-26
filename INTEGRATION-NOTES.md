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
- The API sidebar treatment, still switchable from the small floating panel
  (`split`, `collapsed`, `inline`). Undecided, see the end of this file.

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
button. The bar now publishes its height on `<html>` while it is on screen
and the switcher steps over it.

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

pa11y was run against `astro preview` on port 4390, with a copy of
`.pa11yci.json` rewritten to that port, because the branch's dev server holds
4321 and another branch holds 4322. The npm script's own default port is
4321, so it cannot be used while the dev server is up.

The puppeteer checks default to `--base http://localhost:4322`, which is
another branch's dev server on this machine. Pass `--base` explicitly or they
will report on the wrong build, which they did to me once.

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
| `08`, `09` | An API page: no chips, and the current API sidebar treatment. |
| `10`, `11` | The front page, full height, both themes. |
| `12`, `13` | A Spanish guide, both themes. |
| `14`, `15` | The same guide at 390 px in the two themes, which is also the theme toggle in its two states: a moon on the light shot, a sun on the dark one, no caret on either. |
| `16` | The front page at 390 px. |
| `17` | A page past the raised cap at 390 px, where the run wraps to five rows and ends on the more chip. |
| `20`, `21` | The language bar on desktop and on a phone, after the stacking fix. |

## Still undecided

The API sidebar treatment. `data-api-style` still defaults to `split` and the
floating panel still offers `collapsed` and `inline`; the switcher and its two
inline scripts in `Head.astro` come out when a treatment is chosen, along with
the `.toc-switcher` rules in `ux-variants.css` and the `--ph-lang-banner`
offset that exists only to keep the switcher off the language bar.

Until then, every screenshot in `integration-shots/` and every pa11y run
exercises `split`, the default, and only that one.
