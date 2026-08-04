---
title: "Calibration and uncertainty"
description: "The two disciplines that make a computed level a defensible measurement: physical SPL calibration versus digital dBFS analysis, and the GUM and Monte Carlo propagation of measurement uncertainty."
---

A level printed by software is not yet a measurement. Two things separate the
one from the other: knowing what the digital samples mean **physically**, and
knowing how much the result could reasonably be **wrong**. This section covers
both, and they apply transversally to every other page of the documentation.

[Calibration and dBFS](/phonometry/signals/metrology/calibration/) handles the first.
phonometry works in two reference frames: physical **dB SPL**, established
either from a recorded calibrator tone (the IEC 60942 field ritual) or from a
known microphone sensitivity, and digital **dBFS**, levels relative to full
scale, appropriate when no physical reference exists or when characterising
the digital chain itself. The page explains how each mode is set up and, just
as important, which quantities are meaningful in which frame.

[Measurement uncertainty (GUM and Monte Carlo)](/phonometry/signals/metrology/gum-uncertainty/)
handles the second, implementing the *Guide to the Expression of Uncertainty
in Measurement* (**ISO/IEC Guide 98-3:2008**) and its Monte Carlo
**Supplement 1**. The GUM route propagates standard uncertainties analytically
through sensitivity coefficients into a combined and expanded uncertainty,
with Welch-Satterthwaite effective degrees of freedom; the Monte Carlo route
propagates whole probability distributions numerically and yields coverage
intervals that stay honest when the model is non-linear or the inputs are far
from Gaussian. The page shows both on the same models, including where they
diverge and why.

[Data qualification](/phonometry/signals/metrology/data-qualification/) guards the gate
in front of both: every average - a Leq, a Welch PSD, every averaged input
to an uncertainty budget - assumes the record is stationary, and the Bendat & Piersol reverse
arrangement and runs tests decide that objectively from segment mean squares,
with the book's own acceptance regions. The same page carries the Rice
statistics of level crossings and peaks - apparent frequency, peak rates,
the irregularity factor - that characterise a qualified Gaussian record and
screen for one that is not.

The same discipline extends into the frequency domain: the
[Signals and spectra](/phonometry/signals/spectra/) pages
apply the Bendat & Piersol error analysis to Welch spectral estimates, so
every PSD carries its effective number of averages, its normalized random
error and a chi-square confidence interval.

The pages meet in practice: an uncertainty budget for an acoustic
measurement almost always contains a calibration term, and several standards
implemented elsewhere in the library (ISO 9612, ISO 12999-1) ship uncertainty
budgets that are specialisations of the GUM machinery described here.

## Pages in this section

- [Calibration and dBFS](/phonometry/signals/metrology/calibration/): physical SPL
  calibration from a calibrator tone or a known sensitivity, and the digital
  full-scale mode.
- [Measurement uncertainty (GUM and Monte Carlo)](/phonometry/signals/metrology/gum-uncertainty/):
  the law of propagation of uncertainty and the Monte Carlo method, expanded
  uncertainty and coverage intervals.
- [Data qualification](/phonometry/signals/metrology/data-qualification/): the reverse
  arrangement and runs stationarity tests on segment statistics, and the
  Rice level-crossing and peak statistics with the irregularity factor.
