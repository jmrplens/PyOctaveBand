# Site redesign: colour directions, front page, mobile header, language detection

Experimental branch `feat/site-redesign`. It does not merge and has no PR. It
exists so I can look at the site as a whole: what colour it should be, what
the landing page should do, and two pieces of behaviour the site is missing on
phones and for Spanish-speaking visitors.

Everything here is switchable and reversible. Nothing about the 1220 committed
figures changes: the palettes are chosen *around* the figures, not the other
way round.

## 1. Colour directions

The site rides Starlight's defaults today: hue 224 greys and a 100 % saturated
blue-violet accent that belongs to a startup landing page, not to a library
whose output is calibration numbers. Three complete directions now live in
`site/src/styles/theme-directions.css`, each covering light and dark.

| Direction | Idea | Light accent | Dark accent | Dark ground |
| :-- | :-- | :-- | :-- | :-- |
| `instrument` (default) | Cool steel greys, deep cyan. The front panel of a measuring instrument: neutral metal, one saturated indicator colour. | `#0a6f8c` | `#35b8d8` (text `#b3e6f4`) | `#12181c` |
| `blueprint` | Drafting blue-greys, prussian ink. Reads like a drawing sheet; the dark theme is a navy ground rather than a neutral one. | `#23479a` | `#78a2f0` (text `#cbdcfb`) | `#0f1421` |
| `graphite` | Warm neutral greys, amber. Lab hardware and amber CRT: the only direction that is not blue, so the page never competes with the plots. | `#8a4b06` | `#e0a33e` (text `#f7dcab`) | `#17150f` |

### What the figures dictate

The palettes are not free choices. I measured what the committed figures
actually contain before picking anything:

- **404 of the 506 dark figures are matplotlib plots with a transparent
  background**, drawing in `#ffffff` plus the tab10 palette, whose darkest
  regular ink is `#1f77b4`. The page ground has to stay dark enough for that
  ink to clear 3:1. All three dark grounds do: 3.71:1, 3.81:1 and 3.79:1
  respectively. A mid-slate dark theme would have failed here.
- **The other 102 dark figures are the hand-authored diagrams**
  (`scripts/generate_diagrams.py`), which carry an *opaque* `#0d1117` ground
  and therefore sit on the page as a plate. The closer the page ground is to
  `#0d1117`, the less the plate edge reads. Starlight's current dark ground
  `#17181c` gives 1.10:1; the three directions give **1.06, 1.03 and 1.04**,
  so every one of them is an improvement, `blueprint` most of all.
- **Every light figure carries an opaque `#ffffff` ground.** That single fact
  rules out the tinted "paper" light theme I would otherwise have picked: any
  off-white page would frame all 1220 figures in a white box. So all three
  directions keep the light page ground at `#ffffff` and put their tint into
  the *surfaces* instead: nav, sidebar, cards, code blocks, the landing-page
  panels. Measured seam: 1.00:1 in all three.
- **No direction uses a hue near `#1f77b4` or `#d62728`.** The chrome must
  never read as part of a plot, which is exactly what Starlight's blue accent
  does today next to a blue trace.

### Contrast

`site/scripts/check-contrast.mjs` parses the stylesheet, resolves the six
palettes and measures every pair the site renders. Enforced pairs are text at
4.5:1 (WCAG 1.4.3 AA) and meaningful non-text at 3:1 (1.4.11); decorative
hairlines and the figures' own gridlines are measured and reported but cannot
fail the run.

```
$ node scripts/check-contrast.mjs
87 pairs measured, 0 enforced pair(s) below threshold.
```

Worst enforced pair per direction (all comfortably over AA):

| Pair | instrument | blueprint | graphite |
| :-- | :-- | :-- | :-- |
| body text on page (dark / light) | 10.97 / 10.95 | 10.85 / 11.00 | 10.74 / 11.66 |
| muted text on page (dark / light) | 6.00 / 6.36 | 5.83 / 6.47 | 5.76 / 7.07 |
| link colour on page (dark / light) | 13.25 / 5.73 | 13.28 / 8.61 | 13.71 / 6.80 |
| link colour on a card (dark / light) | 12.47 / 5.36 | 12.40 / 7.88 | 13.03 / 6.30 |
| primary button label (dark / light) | 9.11 / 5.73 | 10.02 / 8.61 | 9.16 / 6.80 |
| accent mark, non-text (dark / light) | 7.67 / 5.73 | 7.20 / 8.61 | 8.24 / 6.80 |

The full table, including the informational rows, is the script's own output.

### Switch mechanism

Same spirit as the other experimental branch: an attribute on `<html>`.

- `data-accent` on `<html>`, one of `instrument` / `blueprint` / `graphite`.
- Stamped **before first paint** by an inline script in
  `src/components/Head.astro`, read from `localStorage['phonometry:accent']`,
  so switching never flashes the previous palette.
- Written by `src/components/AccentSwitcher.astro`, a small fixed control
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
