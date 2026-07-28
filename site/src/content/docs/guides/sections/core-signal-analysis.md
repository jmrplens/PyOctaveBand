---
title: "Core signal analysis"
description: "The measurement core of phonometry: fractional octave filter banks, frequency and time weighting, integrated and statistical levels, calibrated spectral and correlation analysis, physical calibration and measurement uncertainty, and how those pieces chain into a sound level meter in code."
---

Everything in phonometry starts here. This section covers the chain that turns
a raw digital signal into standards-compliant acoustic numbers: split it into
**fractional octave bands** (ANSI S1.11 / IEC 61260-1), shape it with the
**frequency weightings** of IEC 61672-1, smooth it with the **Fast/Slow/Impulse
time ballistics**, and integrate it into **Leq and statistical levels**. It is,
in effect, a sound level meter decomposed into composable functions, and every
other section of the documentation builds on it: a loudness model consumes
calibrated band levels, a room parameter starts from a filtered impulse
response, an environmental rating is an adjusted Leq.
[Build a sound level meter](/phonometry/guides/sound-level-meter/) assembles
that chain end to end on a single runnable page; it is the best starting point
if you want to see the whole area at work before opening the deep guides.

Around the level chain sit the general signal-analysis tools:
**calibrated spectral estimates** (Welch PSD and cross-spectral density with
confidence intervals), **correlation and time-delay estimation** and the
**Hilbert envelope**, all stated with the Bendat & Piersol error analysis.
And two transversal concerns complete the core. **Calibration** decides what
the digital samples mean physically: results can be referenced to a measured
calibrator tone or a known sensitivity (dB SPL), or stay in digital full
scale (dBFS). **Measurement uncertainty** (the GUM and its Monte Carlo
supplement) qualifies any result computed from uncertain inputs, which is
what makes a number defensible in a report.

If you are new to the library, read
[Filter Banks](/phonometry/guides/filter-banks/) first: it introduces the band
decomposition every other page assumes. Then
[Integrated and Statistical Levels](/phonometry/guides/levels/) shows the
metrics most measurements end in, and
[Calibration and dBFS](/phonometry/guides/calibration/) anchors them to
physical units.

## [Octave filtering](/phonometry/guides/sections/octave-filtering/)

Fractional octave band decomposition and the two ways to scale it: streaming
blocks and multichannel arrays.

- [Filter Banks](/phonometry/guides/filter-banks/): the five filter
  architectures, their frequency responses, band decomposition and zero-phase
  offline filtering.
- [Filter Class Verification (IEC 61260-1)](/phonometry/guides/filter-compliance/):
  the Table 1 acceptance mask band by band, the class 0 of the withdrawn 1995
  edition and the compliance fiche.
- [Block Processing](/phonometry/guides/block-processing/): stateful streaming
  analysis that carries filter state across buffers, for signals that never
  fit in memory.
- [Multichannel and Performance](/phonometry/guides/multichannel/): vectorized
  analysis of many channels at once, with performance notes.

## [Levels and weighting](/phonometry/guides/sections/levels-weighting/)

From weighted signal to reported level: the frequency weightings, the time
ballistics and the integrated, statistical and rating levels.

- [Frequency Weighting (A, C, G, Z)](/phonometry/guides/weighting/): the
  IEC 61672-1 ear-response curves and the ISO 7196 infrasound G-weighting.
- [Time Weighting](/phonometry/guides/time-weighting/): Fast, Slow and Impulse
  exponential ballistics per IEC 61672-1.
- [Integrated and Statistical Levels](/phonometry/guides/levels/): Leq and
  LAeq, percentile levels L10/L50/L90, LCpeak and SEL, noise dose (IEC 61252),
  Lden and rating levels (ISO 1996-1), and octave spectrograms.

## [Signals and spectra](/phonometry/guides/sections/signals-spectra/)

Fine-grained frequency- and time-domain analysis, every estimate calibrated
and carrying its statistical quality.

- [Calibrated spectral analysis](/phonometry/guides/spectral-analysis/): the
  Bendat & Piersol Welch estimators with their statistical quality: PSD and
  cross-spectral density with chi-square confidence intervals, the coherent
  output spectrum with the spectral SNR, 1/n-octave smoothing and
  exact-slope colored-noise generators.
- [Multiple and partial coherence](/phonometry/guides/miso-coherence/): the
  multiple-input/output coherence functions for multiple correlated
  sources and one output, with the conditioning that tells a genuine cause
  from a source that merely correlates with it, and the partial coherent
  output spectra that say which source dominates each band.
- [Time-frequency analysis](/phonometry/guides/time-frequency/): the
  calibrated STFT spectrogram in absolute units (dB SPL for pascals) and
  the zoom FFT that resolves tones closer than a practical FFT bin.
- [Cepstrum, echoes and the envelope spectrum](/phonometry/guides/cepstrum-echoes/):
  the power, real and complex cepstrum with quefrency analysis, echo
  detection with the reflection coefficient read off the cepstral peak,
  lowpass/highpass liftering of a log spectrum, and the envelope spectrum
  that turns amplitude modulations into discrete lines.
- [Time synchronous averaging](/phonometry/guides/synchronous-averaging/):
  extraction of a periodic waveform of known period by time domain averaging,
  the comb filter that describes it in the frequency domain, the square-root
  noise-reduction law, and the choice of the number of averages that places a
  comb node on an interfering order (McFadden 1987).
- [Correlation, time delay and envelope](/phonometry/guides/correlation-delay/):
  correlation estimates with the Bendat & Piersol random errors, time-delay
  estimation by direct correlation, cross-spectrum phase slope and the
  Knapp & Carter GCC weightings, sub-sample impulse-response delay and
  alignment, and the Hilbert envelope.
- [Test signals and sample-rate tools](/phonometry/guides/test-signals/):
  IEC 60268-1 tone bursts with exact gating, resampling with a stated
  anti-alias specification, and band-limited fractional delay.
- [System measurement](/phonometry/guides/system-measurement/):
  complementary Golay pairs, sweeps with an arbitrary target magnitude
  spectrum by group-delay shaping, and the Kirkeby-regularized inversion of
  a measured response.

## [Calibration and uncertainty](/phonometry/guides/sections/calibration-uncertainty/)

What the numbers mean and how much to trust them.

- [Calibration and dBFS](/phonometry/guides/calibration/): physical SPL
  calibration from a calibrator tone (IEC 60942) or a known sensitivity, and
  the digital dBFS mode.
- [Measurement uncertainty (GUM and Monte Carlo)](/phonometry/guides/gum-uncertainty/):
  the law of propagation of uncertainty and the Monte Carlo method of
  ISO/IEC Guide 98-3, with expanded uncertainty and coverage intervals.
- [Data qualification](/phonometry/guides/data-qualification/): the reverse
  arrangement and runs stationarity tests on segment statistics, and the Rice
  level-crossing and peak statistics with the irregularity factor.
