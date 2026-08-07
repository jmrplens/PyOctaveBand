---
title: "Start"
description: "Where to begin with phonometry: the first analysis end to end, the map of all the guides, why the library exists and who maintains it."
---

Four short pages, meant to be read once before anything else. Each answers one
question, and they are in the order the questions arrive.

**Can I install it and get a number out?**
[Getting Started](/phonometry/start/getting-started/) installs the library and
runs a first one-third-octave analysis, on a synthetic signal and then on a WAV
file, and states what a recording must satisfy before those numbers mean
anything physical. It stops short of a calibrated measurement on purpose:
[Calibration and dBFS](/phonometry/signals/metrology/calibration/) is the next
step, the one that turns band levels into pascals, and
[Build a sound level meter](/phonometry/signals/sound-level-meter/) runs the
whole chain end to end.

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

The assumed starting point is Python 3.13 or newer with working NumPy and
SciPy, and enough acoustics to know what a one-third-octave band and a sound
pressure level are. Any symbol the guides use without introducing is in the
[glossary](/phonometry/reference/glossary/), with its unit, its defining clause
and the guide that computes it.
