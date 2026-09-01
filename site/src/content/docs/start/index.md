---
title: "Start"
description: "Where to begin with phonometry: the first analysis end to end, the task index, the map of all the guides, why the library exists and who maintains it."
---

phonometry computes acoustic quantities from the text of the standards that
define them — ISO, IEC, ANSI and ASTM, the CNOSSOS-EU annex to Directive
2002/49/EC, and the ICAO and ECAC aircraft documents — and every metric names
the clause it implements. What that buys you, and how it is checked, is
[Why phonometry](/phonometry/start/why-phonometry/).

Six short pages, meant to be read once before anything else. Each answers one
question, and they are in the order the questions arrive.

**Can I install it and get a number out?**
[Getting Started](/phonometry/start/getting-started/) installs the library,
runs a first one-third-octave analysis on a synthetic signal, then anchors that
analysis to a calibrator tone so the levels are decibels re 20 µPa rather than
decibels re nothing, reduces them to one A-weighted level, and states what a
recording must satisfy for any of it to hold. It stops one stage short of a
meter: the Fast and Slow ballistics, $L_{\mathrm{A}E}$, $L_\mathrm{Cpeak}$ and the percentile
levels are in [Build a sound level
meter](/phonometry/signals/sound-level-meter/), which runs the whole chain end
to end on one page, and [Calibration and
dBFS](/phonometry/signals/metrology/calibration/) is the deep guide behind the
one step that matters most.

**I have a job, not a subject. Which page is it?**
[What do you need to measure?](/phonometry/start/tasks/) indexes the library by
the task instead of the topic: measure a reverberation time, check a wall
against a building code, rate a machine's sound power, decide whether a worker
is over the exposure limit.

**Where is the thing I came for?**
[All guides](/phonometry/start/guides/) is the map: every guide in the library,
grouped by the topic it belongs to, with a line on each.

**Should I trust the number?**
[Why phonometry](/phonometry/start/why-phonometry/) sets out what the library
is for and how it is validated against the standards it implements, with the
tone-burst check worked through against the acceptance limits.

**Who is answerable for it, and how do I cite it?**
[About](/phonometry/start/about/) states who maintains it, how to cite it and
under what licence.

[Support the project](/phonometry/start/support/) lists the ways to help,
starting with the primary sources the library still lacks.

## Two things to settle before any guide works

**Which reference frame a level is in.** A level is either *physical*, in
dB SPL, anchored by a recorded calibrator tone or by a known microphone
sensitivity, or *digital*, in dBFS relative to full scale. The two are not
interchangeable, and most guides assume the first: a level function handed raw
soundcard samples returns a number whose reference is arbitrary, which looks
exactly like a valid answer. [Calibration and
dBFS](/phonometry/signals/metrology/calibration/) settles it.

**That almost everything downstream consumes bands.** Below the raw signal
there is one decomposition: fractional-octave bands whose −3 dB edges sit on
the ANSI S1.11 / IEC 61260-1 nominal frequencies. A loudness model, a room
parameter and an environmental rating all start from it, which is why the
band-filtering page is the one prerequisite that turns up everywhere: [Filter
Banks](/phonometry/signals/filters/filter-banks/).

## What a guide looks like

Worth knowing before opening one, because it is what lets you decide in thirty
seconds whether a page answers your question. Every guide opens with the
standard it implements, the quantities that standard defines and the
assumptions the implementation makes; then comes runnable code and the figure
it draws; and it closes with a "What this guide covers" block that states
plainly which clauses, annexes and methods are implemented and which are not.
The last of those is the part a reviewer asks about, and it is deliberately the
part written most bluntly.

## If you already know what you need

- A first measurement carried end to end: [Build a sound level meter](/phonometry/signals/sound-level-meter/).
- The whole inventory, by topic: [All guides](/phonometry/start/guides/).
- A symbol you have but cannot name: the [glossary](/phonometry/reference/glossary/), with its unit, its defining clause and the guide that computes it.
- Evidence that a number is defensible: the [conformance report](/phonometry/reference/conformance/), which prints each standard's own expected value beside the computed one.
- A printed expected value that disagrees with the library: the [errata registry](/phonometry/reference/errata/), which says which of the two is wrong and why.

The assumed starting point is Python 3.13 or newer with working NumPy and
SciPy, and enough acoustics to know what a one-third-octave band and a sound
pressure level are.

## What Start is not

This is not a tutorial series, and it is not the API. Function signatures and
argument types are in the generated API reference. The derivations, the
numerical conformance report, the errata register for defects found in the
published standards themselves, the glossary of symbols and the bibliography
are all in [Reference](/phonometry/reference/). And the acoustics itself is
assumed rather than taught: the guides explain the method a standard
prescribes and why it is written that way, not what a decibel is.
