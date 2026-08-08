← [Documentation index](../README.md)

# Signal analysis

Everything in phonometry starts here. This section covers the chain that turns
a raw digital signal into standards-compliant acoustic numbers: split it into
**fractional octave bands** (ANSI S1.11 / IEC 61260-1), shape it with the
**frequency weightings** of IEC 61672-1, smooth it with the **Fast/Slow/Impulse
time ballistics**, and integrate it into **Leq and statistical levels**. It is,
in effect, a sound level meter decomposed into composable functions, and every
other section of the documentation builds on it: a loudness model consumes
calibrated band levels, a room parameter starts from a filtered impulse
response, an environmental rating is an adjusted Leq.

Around the level chain sit the general signal-analysis tools:
**calibrated spectral estimates** (Welch PSD and cross-spectral density with
confidence intervals), **correlation and time-delay estimation** and the
**Hilbert envelope**, all stated with the Bendat & Piersol error analysis.
And two transversal concerns complete the core. **Calibration** decides what
the digital samples mean physically: results can be referenced to a measured
calibrator tone (dB SPL), or stay in digital full scale (dBFS). **Measurement uncertainty** (the GUM and its Monte Carlo
supplement) qualifies any result computed from uncertain inputs, which is
what makes a number defensible in a report.

Three conventions run through every page below, and every snippet on the site
assumes them. A signal is a NumPy array of sound pressure with **time on the
last axis**, so one channel is `(n,)` and several parallel channels are
`(channels, samples)`. The sample rate always travels as an explicit `fs`
argument: nothing is read from a file header, because the library never opens
the file. And the array is expected to hold **pascals**, which is why a level
function applied to raw soundcard samples returns a number whose reference is
arbitrary, and why every level function also accepts a `calibration_factor` in
pascals per digital unit or the `dbfs=True` escape hatch. Simple metrics come
back as floats and arrays; the richer ones come back as frozen result objects
that expose `.plot()`. [Calibration and
dBFS](metrology/calibration.md) resolves the third convention
in full, and [Multichannel and
Performance](filters/multichannel.md) the first.

Two ways in. To see the whole chain working at once, run [Build a sound level
meter](sound-level-meter.md): it calibrates against a
calibrator tone, applies the frequency and time weightings, integrates into
Leq, SEL and percentile levels, splits the signal into octave bands and checks
the class of every stage, on one runnable page. To learn the pieces in
dependency order, start at [Filter
Banks](filters/filter-banks.md), which introduces the band
decomposition every other page assumes, then [Integrated and Statistical
Levels](levels/levels.md) for the metrics most measurements
end in, and [Calibration and dBFS](metrology/calibration.md)
to anchor them to pascals.

## [Build a sound level meter](sound-level-meter.md)

- [Build a sound level meter](sound-level-meter.md): the whole
  chain assembled on one runnable page — calibration, frequency and time
  weighting, the integrated and statistical levels, the band decomposition and
  the class verdict of each stage — as the worked introduction to the four
  subsections below.

## [Octave filtering](filters/index.md)

Fractional octave band decomposition and the two ways to scale it: streaming
blocks and multichannel arrays.

- [Filter Banks](filters/filter-banks.md): the fractional-octave
  band mathematics, the bank parameters, the parametric EQ, band
  decomposition and zero-phase offline filtering.
- [Filter Architecture Gallery](filters/filter-gallery.md): the five
  filter architectures compared, the full response gallery and
  per-architecture usage, with the Linkwitz-Riley crossover.
- [Filter Class Verification (IEC 61260-1)](filters/filter-compliance.md):
  the Table 1 acceptance mask band by band, the class 0 of the withdrawn 1995
  edition and the compliance fiche.
- [Block Processing](filters/block-processing.md): stateful streaming
  analysis that carries filter state across buffers, for signals that never
  fit in memory.
- [Multichannel and Performance](filters/multichannel.md): vectorized
  analysis of many channels at once, with performance notes.

## [Levels and weighting](levels/index.md)

From weighted signal to reported level: the frequency weightings, the time
ballistics and the integrated, statistical and rating levels.

- [Frequency Weighting (A, C, Z)](levels/weighting.md): the
  IEC 61672-1 ear-response curves, the high-frequency accuracy mode and the
  Table 3 class verification.
- [Special Weightings (G, B, D, AU)](levels/special-weightings.md):
  the ISO 7196 infrasound G-weighting, the historical B and D curves and AU
  per IEC 61012.
- [Time Weighting](levels/time-weighting.md): Fast, Slow and Impulse
  exponential ballistics per IEC 61672-1.
- [Integrated and Statistical Levels](levels/levels.md): Leq and
  LAeq, percentile levels L10/L50/L90, LCpeak and SEL, noise dose (IEC 61252),
  and octave spectrograms.
- [Environmental Levels (ISO 1996-1/-2)](../environment/assessment/environmental-levels.md):
  Lden, Ldn and the composite rating levels, the tonal adjustment, the
  residual-noise correction and the uncertainty budget.
- [Spanish Noise Regulation (RD 1367/2007)](../environment/assessment/spanish-noise-regulation.md):
  the corrected level LKeq, the Kt/Kf/Ki corrections, the evaluation periods
  and noise phases, and the limit tables.

## [Signals and spectra](spectra/index.md)

Fine-grained frequency- and time-domain analysis, every estimate calibrated
and carrying its statistical quality.

- [Calibrated spectral analysis](spectra/spectral-analysis.md): the
  Bendat & Piersol Welch estimators with their statistical quality: PSD and
  cross-spectral density with chi-square confidence intervals, the coherent
  output spectrum with the spectral SNR, 1/n-octave smoothing and
  exact-slope colored-noise generators.
- [Multiple and partial coherence](spectra/miso-coherence.md): the
  multiple-input/output coherence functions for multiple correlated
  sources and one output, with the conditioning that tells a genuine cause
  from a source that merely correlates with it, and the partial coherent
  output spectra that say which source dominates each band.
- [Time-frequency analysis](spectra/time-frequency.md): the
  calibrated STFT spectrogram in absolute units (dB SPL for pascals) and
  the zoom FFT that resolves tones closer than a practical FFT bin.
- [Cepstrum, echoes and the envelope spectrum](spectra/cepstrum-echoes.md):
  the power, real and complex cepstrum with quefrency analysis, echo
  detection with the reflection coefficient read off the cepstral peak,
  lowpass/highpass liftering of a log spectrum, and the envelope spectrum
  that turns amplitude modulations into discrete lines.
- [Time synchronous averaging](spectra/synchronous-averaging.md):
  extraction of a periodic waveform of known period by time domain averaging,
  the comb filter that describes it in the frequency domain, the square-root
  noise-reduction law, and the choice of the number of averages that places a
  comb node on an interfering order (McFadden 1987).
- [Machine fault frequencies](../vibration/machinery/machine-diagnostics.md)
  (in the vibration section): the kinematic fault-frequency families of
  rotating machinery (Norton &
  Karczub Section 8.4) drawn on top of a measured envelope spectrum: bearing
  BPFO, BPFI, BSF and cage frequencies, gear-mesh sidebands, induction-motor
  slip, pole-pass and rotor-slot harmonics, and blade-passing tones.
- [Correlation, time delay and envelope](spectra/correlation-delay.md):
  correlation estimates with the Bendat & Piersol random errors, time-delay
  estimation by direct correlation, cross-spectrum phase slope and the
  Knapp & Carter GCC weightings, sub-sample impulse-response delay and
  alignment, and the Hilbert envelope.
- [Test signals and sample-rate tools](spectra/test-signals.md):
  IEC 60268-1 tone bursts with exact gating, resampling with a stated
  anti-alias specification, and band-limited fractional delay.
- [System measurement](spectra/system-measurement.md):
  complementary Golay pairs, sweeps with an arbitrary target magnitude
  spectrum by group-delay shaping, and the Kirkeby-regularized inversion of
  a measured response.

## [Calibration and uncertainty](metrology/index.md)

What the numbers mean and how much to trust them.

- [Calibration and dBFS](metrology/calibration.md): physical SPL
  calibration from a calibrator tone (IEC 60942), the stability check it applies
  to that recording, and the digital dBFS mode.
- [Measurement uncertainty (GUM and Monte Carlo)](metrology/gum-uncertainty.md):
  the law of propagation of uncertainty and the Monte Carlo method of
  ISO/IEC Guide 98-3, with expanded uncertainty and coverage intervals.
- [Data qualification](metrology/data-qualification.md): the reverse
  arrangement and runs stationarity tests on segment statistics, and the Rice
  level-crossing and peak statistics with the irregularity factor.

## What this section does not cover

Four things a reader reasonably expects here are absent, and each guide says so
in its own "Not covered" block. **No instrument is verified.**
`verify_filter_class` and `verify_weighting_class` check a designed digital
response against the tolerance tables of IEC 61260-1 and IEC 61672-1; the
IEC 61672-3 pattern-evaluation tests a physical meter needs for type approval,
and the IEC 60942 conformance tests of the calibrator itself, are not run, so a
class verdict here describes the algorithm and not a built device. **No file is
opened.** Nothing in the library decodes WAV, FLAC or any other container:
every function takes an array you have already read, which is why `fs` is
always an argument. **No array processing.** Correlation and time delay model
one common path between exactly two sensors and report the single largest peak;
there is no multi-sensor TDOA solver, no beamformer and no source localisation.
**No perceptual features.** The cepstrum here is the plain linear-frequency
one, with no mel warping or MFCC variant, and loudness as a sensation belongs
to [Psychoacoustics](../perception/psychoacoustics/index.md), not to the
energy metrics of this section.

## Before and after these pages

The derivations behind these pages are in [Signal analysis
theory](../reference/theory/signal-analysis.md): the band grid, the weighting curves,
the time integration, the intensity approximation and the uncertainty
framework. If you have not run anything yet, [Getting
Started](../start/getting-started.md) installs the library and
calibrates a first analysis.

If you arrived here from a search and want the shape of the whole library,
[What do you need to measure?](https://jmrplens.github.io/phonometry/start/tasks/) indexes it by the job
and [All guides](https://jmrplens.github.io/phonometry/start/guides/) lists every page with a line on
each.
