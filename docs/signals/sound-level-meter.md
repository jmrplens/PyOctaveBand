← [Documentation index](../README.md)

# Build a sound level meter

A sound level meter is not one algorithm but a short pipeline of them, and
IEC 61672-1 specifies every stage. phonometry implements each stage as an
independent, composable function; this page assembles them, in order, into a
working meter. Every snippet runs as written (the signals are synthesized so
the page is self-contained), and each stage links to the deep guide that
explains it fully.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_slm_pipeline_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_slm_pipeline.svg" alt="The sound level meter pipeline of this page: the calibrator tone feeding sensitivity() to obtain the factor in pascals per digital unit, the measurement recording joining it as the calibrated pressure, and three readout branches from there, weighting_filter with time_weighting for the displayed LAF(t) and the percentile levels, laeq, sel and lc_peak for the integrated levels, and octave_filter for the one-third-octave spectrum, all closed by verify_weighting_class and verify_filter_class against the IEC 61672-1 Table 3 and IEC 61260-1 Table 1 acceptance limits" width="92%"></picture>

This is the same chain IEC 61672-1 draws for the physical instrument: the
class 1 calibrator anchors the microphone to 94 dB at 1 kHz, and every stage
that follows is one function of this page.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_slm_chain_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_slm_chain.svg" alt="IEC 61672-1 sound level meter chain: a class 1 sound calibrator at 94 dB and 1 kHz coupled onto the measurement microphone with its windscreen, followed by the four instrument stages of microphone plus preamplifier, A, C or Z frequency weighting, squaring with the exponential F or S time weighting of 125 ms or 1 s, and the display of LAF(t) in dB re 20 µPa" width="92%"></picture>

The snippets on this page build on each other: run them top to bottom in one
session (or paste the whole page into a script).

## 1. The scenario

A meter needs two recordings from the *same* input chain: the calibrator tone
that anchors the digital numbers to pascals, and the measurement itself. Here
both are synthesized so you can run the page anywhere; in a real measurement
they come from your microphone.

```python
import numpy as np
from phonometry import filters, metrology, signals

fs = 48000

# Calibrator tone: 94 dB SPL = 1 Pa RMS at 1 kHz (IEC 60942).
#   Synthesized here; in the field, record a few seconds of your calibrator.
calibrator = np.sqrt(2) * np.sin(2 * np.pi * 1000 * np.arange(3 * fs) / fs)

# "Street" measurement: 10 s of pink background noise plus a 1 s horn-like
#   1 kHz event, so the statistical levels have something to separate.
recording = signals.noise_signal(fs, 10.0, color="pink", rms=0.02, seed=7)
recording[4 * fs : 5 * fs] += 0.2 * np.sqrt(2) * np.sin(
    2 * np.pi * 1000 * np.arange(fs) / fs
)
```

## 2. Calibrate: give the samples physical meaning

Digital samples are dimensionless; the **sensitivity factor** converts them
to pascals. `sensitivity()` computes it from the calibrator recording and, at
the same time, validates the recording's short-term stability the way
IEC 60942 qualifies the calibrator itself, so a badly coupled microphone is
caught here instead of corrupting every level downstream.

```python
cal = metrology.sensitivity(calibrator, target_spl=94.0, fs=fs)
# cal is in Pa per digital unit; every level function accepts it as
# calibration_factor. For this synthetic tone it is ~1.0.
```

Deep guide: [Calibration and dBFS](metrology/calibration.md), which
also covers calibrating from a known microphone sensitivity and the digital
dBFS mode used when no physical reference exists.

## 3. Weight: frequency and time (IEC 61672-1)

The meter never shows raw pressure. The signal first passes the **A
frequency weighting** (the ear-response curve of IEC 61672-1), is squared,
and is then smoothed by the **Fast exponential detector** (time constant
125 ms). The result is the moving level a meter's display follows,
$L_{AF}(t)$:

```python
pressure = cal * recording                                # digital units -> Pa
weighted = filters.weighting_filter(pressure, fs, curve="A")
envelope = filters.time_weighting(weighted, fs, mode="fast")  # mean-square Pa^2
laf_t = 10 * np.log10(np.maximum(envelope, 1e-12) / (2e-5) ** 2)
# laf_t peaks near 80 dB during the event and settles near 55 dB between.
```

You rarely write this chain yourself: every level function of the next step
applies the frequency weighting internally, and the percentile levels rebuild
this Fast envelope for you. The energy metrics ($L_{eq}$, SEL) integrate the
squared weighted signal directly, with no ballistics, exactly as a meter
does. The chain is shown here because it *is* the meter's display.

Deep guides: [Frequency Weighting (A, C, Z)](levels/weighting.md)
and [Time Weighting](levels/time-weighting.md).

## 4. Integrate: the numbers a meter reports

One pass over the calibrated recording yields the standard readouts: the
energy-equivalent **$L_{Aeq}$**, the **percentile levels** that describe how
the level fluctuated ($L_{90}$ is the background, $L_{10}$ the events), the
**sound exposure level** that normalizes the event to one second, and the
C-weighted **peak** for impulsive content.

```python
la_eq = signals.laeq(recording, fs, calibration_factor=cal)     # ~70.2 dB
ln = signals.ln_levels(
    recording, fs, n=(10, 50, 90), weighting="A", calibration_factor=cal
)                                                # L10 ~78.0, L50 ~55.1, L90 ~54.9
lae = signals.sel(recording, fs, weighting="A", calibration_factor=cal)  # ~80.2
lc_pk = signals.lc_peak(recording, fs, calibration_factor=cal)           # ~84.4

print(f"LAeq {la_eq:.1f} dB | L10 {ln[10]:.1f} | L90 {ln[90]:.1f} "
      f"| LAE {lae:.1f} | LCpeak {lc_pk:.1f}")
```

Note the arithmetic the numbers encode: the 1 s event dominates $L_{Aeq}$ (it
sits 25 dB above the background, far more than the 10 dB the
nine-times-longer background gets back in duration), $L_{AE}$ is $L_{Aeq}$
plus $10\log_{10}$ of the 10 s duration, and $L_{90}$ barely notices the event at
all.

Deep guide: [Integrated and Statistical Levels](levels/levels.md),
which adds noise dose and octave spectrograms; the $L_{den}$ and rating levels
continue in [Environmental levels](../environment/assessment/environmental-levels.md).

## 5. Band-filter: the spectrum view (IEC 61260-1)

A class 1 meter with a filter set reports band levels. `octave_filter`
decomposes the calibrated signal into fractional-octave bands whose design is
anchored to the IEC 61260-1 band edges; `nominal=True` labels them with the
preferred frequencies you would read on an instrument.

```python
spl, bands = filters.octave_filter(
    recording, fs, fraction=3, nominal=True,
    calibration=filters.LevelCalibration(factor=cal),
)
# 33 one-third-octave band levels in dB SPL, labeled '12.5' ... '20k'.
# The '1k' band holds the event: ~70 dB, while its neighbors stay ~25 dB below.
print(dict(zip(bands, np.round(spl, 1))))
```

Deep guides: [Filter Banks](filters/filter-banks.md) for the filter
architectures and zero-phase mode,
[Block Processing](filters/block-processing.md) for streaming, and
[Multichannel and Performance](filters/multichannel.md) for arrays.

## 6. Verify: is this meter class 1?

A real instrument is only a "class 1 sound level meter" after its weightings
and filters pass the acceptance limits of the standards. The library ships
the same verifiers it applies to itself in CI: `verify_weighting_class`
sweeps a `WeightingFilter` against the IEC 61672-1 Table 3 limits, and
`verify_filter_class` sweeps an `OctaveFilterBank` against the IEC 61260-1
Table 1 limits.

```python
wf = filters.WeightingFilter(fs, curve="A")
print(filters.verify_weighting_class(wf)["overall_class"])   # 1

bank = filters.OctaveFilterBank(fs, fraction=3)
print(filters.verify_filter_class(bank)["overall_class"])    # 1
```

The verdicts also come per band, so you can see exactly where a design would
leave its class corridor. Deep guides:
[Frequency Weighting](levels/weighting.md) (section on class
verification) and [Filter class verification](filters/filter-compliance.md) (the
Table 1 mask, class 0 and the compliance fiche).

## Where to go next

The meter built here is the trunk; the rest of the core grows from it.

- [Measurement uncertainty (GUM and Monte Carlo)](metrology/gum-uncertainty.md):
  attach an uncertainty to the $L_{Aeq}$ you just computed, calibration term
  included.
- [Calibrated spectral analysis](spectra/spectral-analysis.md): when
  bands are too coarse, the Welch PSD with confidence intervals.
- [Correlation, time delay and envelope](spectra/correlation-delay.md):
  two microphones instead of one, and the delay between them.
- [Block Processing](filters/block-processing.md): turn this page's
  offline meter into a streaming one with carried filter state.

## See also

- API reference: [`metrology.calibration`](https://jmrplens.github.io/phonometry/reference/api/metrology/calibration/),
  [`filters.weighting`](https://jmrplens.github.io/phonometry/reference/api/filters/weighting/),
  [`signals.levels`](https://jmrplens.github.io/phonometry/reference/api/signals/levels/),
  [`phonometry`](https://jmrplens.github.io/phonometry/reference/api/filters/phonometry/) and
  [`filters.compliance`](https://jmrplens.github.io/phonometry/reference/api/filters/compliance/).

## References

- International Electrotechnical Commission. (2013). *Electroacoustics —
  Sound level meters — Part 1: Specifications* (IEC 61672-1:2013).
  [IEC webstore](https://webstore.iec.ch/en/publication/5708).
  The blueprint of the instrument assembled on this page: the A frequency
  weighting and the Fast exponential time weighting of the level chain, the
  C-weighted peak and the sound exposure level, and the Table 3 class
  acceptance limits checked by `verify_weighting_class`.
- International Electrotechnical Commission. (2014). *Electroacoustics —
  Octave-band and fractional-octave-band filters — Part 1: Specifications*
  (IEC 61260-1:2014). [IEC webstore](https://webstore.iec.ch/en/publication/5063).
  The fractional-octave-band filters behind the spectrum stage, and the
  Table 1 class acceptance limits checked by `verify_filter_class`.
- International Electrotechnical Commission. (2017). *Electroacoustics —
  Sound calibrators* (IEC 60942:2017).
  [IEC webstore](https://webstore.iec.ch/en/publication/30045).
  The acoustic calibrator assumed by the sensitivity stage: the 94 dB
  principal level and the short-term stability check applied to the reference
  recording.
