---
title: "Signal analysis"
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
[Build a sound level meter](/phonometry/signals/sound-level-meter/) assembles
that chain end to end on a single runnable page; it is the best starting point
if you want to see the whole area at work before opening the deep guides.

Around the level chain sit the general signal-analysis tools:
**calibrated spectral estimates** (Welch PSD and cross-spectral density with
confidence intervals), **correlation and time-delay estimation** and the
**Hilbert envelope**, all stated with the Bendat & Piersol error analysis.
And two transversal concerns complete the core. **Calibration** decides what
the digital samples mean physically: results can be referenced to a measured
calibrator tone (dB SPL), or stay in digital full scale (dBFS). **Measurement uncertainty** (the GUM and its Monte Carlo
supplement) qualifies any result computed from uncertain inputs, which is
what makes a number defensible in a report.

If you are new to the library, read
[Filter Banks](/phonometry/signals/filters/filter-banks/) first: it introduces the band
decomposition every other page assumes. Then
[Integrated and Statistical Levels](/phonometry/signals/levels/levels/) shows the
metrics most measurements end in, and
[Calibration and dBFS](/phonometry/signals/metrology/calibration/) anchors them to
physical units.

## [Octave filtering](/phonometry/signals/filters/)

Fractional octave band decomposition and the two ways to scale it: streaming
blocks and multichannel arrays.

- [Filter Banks](/phonometry/signals/filters/filter-banks/): the fractional-octave
  band mathematics, the bank parameters, the parametric EQ, band
  decomposition and zero-phase offline filtering.
- [Filter Architecture Gallery](/phonometry/signals/filters/filter-gallery/): the five
  filter architectures compared, the full response gallery and
  per-architecture usage, with the Linkwitz-Riley crossover.
- [Filter Class Verification (IEC 61260-1)](/phonometry/signals/filters/filter-compliance/):
  the Table 1 acceptance mask band by band, the class 0 of the withdrawn 1995
  edition and the compliance fiche.
- [Block Processing](/phonometry/signals/filters/block-processing/): stateful streaming
  analysis that carries filter state across buffers, for signals that never
  fit in memory.
- [Multichannel and Performance](/phonometry/signals/filters/multichannel/): vectorized
  analysis of many channels at once, with performance notes.

## [Levels and weighting](/phonometry/signals/levels/)

From weighted signal to reported level: the frequency weightings, the time
ballistics and the integrated, statistical and rating levels.

- [Frequency Weighting (A, C, Z)](/phonometry/signals/levels/weighting/): the
  IEC 61672-1 ear-response curves, the high-frequency accuracy mode and the
  Table 3 class verification.
- [Special Weightings (G, B, D, AU)](/phonometry/signals/levels/special-weightings/):
  the ISO 7196 infrasound G-weighting, the historical B and D curves and AU
  per IEC 61012.
- [Time Weighting](/phonometry/signals/levels/time-weighting/): Fast, Slow and Impulse
  exponential ballistics per IEC 61672-1.
- [Integrated and Statistical Levels](/phonometry/signals/levels/levels/): Leq and
  LAeq, percentile levels L10/L50/L90, LCpeak and SEL, noise dose (IEC 61252),
  and octave spectrograms.
- [Environmental Levels (ISO 1996-1/-2)](/phonometry/environment/assessment/environmental-levels/):
  Lden, Ldn and the composite rating levels, the tonal adjustment, the
  residual-noise correction and the uncertainty budget.
- [Spanish Noise Regulation (RD 1367/2007)](/phonometry/environment/assessment/spanish-noise-regulation/):
  the corrected level LKeq, the Kt/Kf/Ki corrections, the evaluation periods
  and noise phases, and the limit tables.

## [Signals and spectra](/phonometry/signals/spectra/)

Fine-grained frequency- and time-domain analysis, every estimate calibrated
and carrying its statistical quality.

- [Calibrated spectral analysis](/phonometry/signals/spectra/spectral-analysis/): the
  Bendat & Piersol Welch estimators with their statistical quality: PSD and
  cross-spectral density with chi-square confidence intervals, the coherent
  output spectrum with the spectral SNR, 1/n-octave smoothing and
  exact-slope colored-noise generators.
- [Multiple and partial coherence](/phonometry/signals/spectra/miso-coherence/): the
  multiple-input/output coherence functions for multiple correlated
  sources and one output, with the conditioning that tells a genuine cause
  from a source that merely correlates with it, and the partial coherent
  output spectra that say which source dominates each band.
- [Time-frequency analysis](/phonometry/signals/spectra/time-frequency/): the
  calibrated STFT spectrogram in absolute units (dB SPL for pascals) and
  the zoom FFT that resolves tones closer than a practical FFT bin.
- [Cepstrum, echoes and the envelope spectrum](/phonometry/signals/spectra/cepstrum-echoes/):
  the power, real and complex cepstrum with quefrency analysis, echo
  detection with the reflection coefficient read off the cepstral peak,
  lowpass/highpass liftering of a log spectrum, and the envelope spectrum
  that turns amplitude modulations into discrete lines.
- [Time synchronous averaging](/phonometry/signals/spectra/synchronous-averaging/):
  extraction of a periodic waveform of known period by time domain averaging,
  the comb filter that describes it in the frequency domain, the square-root
  noise-reduction law, and the choice of the number of averages that places a
  comb node on an interfering order (McFadden 1987).
- [Machine fault frequencies](/phonometry/vibration/machinery/machine-diagnostics/):
  the kinematic fault-frequency families of rotating machinery (Norton &
  Karczub Section 8.4) drawn on top of a measured envelope spectrum: bearing
  BPFO, BPFI, BSF and cage frequencies, gear-mesh sidebands, induction-motor
  slip, pole-pass and rotor-slot harmonics, and blade-passing tones.
- [Correlation, time delay and envelope](/phonometry/signals/spectra/correlation-delay/):
  correlation estimates with the Bendat & Piersol random errors, time-delay
  estimation by direct correlation, cross-spectrum phase slope and the
  Knapp & Carter GCC weightings, sub-sample impulse-response delay and
  alignment, and the Hilbert envelope.
- [Test signals and sample-rate tools](/phonometry/signals/spectra/test-signals/):
  IEC 60268-1 tone bursts with exact gating, resampling with a stated
  anti-alias specification, and band-limited fractional delay.
- [System measurement](/phonometry/signals/spectra/system-measurement/):
  complementary Golay pairs, sweeps with an arbitrary target magnitude
  spectrum by group-delay shaping, and the Kirkeby-regularized inversion of
  a measured response.

## [Calibration and uncertainty](/phonometry/signals/metrology/)

What the numbers mean and how much to trust them.

- [Calibration and dBFS](/phonometry/signals/metrology/calibration/): physical SPL
  calibration from a calibrator tone (IEC 60942), the stability check it applies
  to that recording, and the digital dBFS mode.
- [Measurement uncertainty (GUM and Monte Carlo)](/phonometry/signals/metrology/gum-uncertainty/):
  the law of propagation of uncertainty and the Monte Carlo method of
  ISO/IEC Guide 98-3, with expanded uncertainty and coverage intervals.
- [Data qualification](/phonometry/signals/metrology/data-qualification/): the reverse
  arrangement and runs stationarity tests on segment statistics, and the Rice
  level-crossing and peak statistics with the irregularity factor.
