← [Documentation index](../../README.md)

# Microphone Characterisation (IEC 60268-4)

Every acoustic measurement in this documentation starts at a microphone, and
IEC 60268-4 is the standard that fixes how the device itself is described:
what its sensitivity refers to, how its directional pattern is stated, and
how its self-noise and overload point bound the levels it can read. This
guide covers the conventions where datasheet comparisons go wrong (the
sensitivity references and the noise weightings), the directional patterns
with their directivity index, and the **rated-characteristics report** that
renders the standard's data sheet from a measured free-field response, with
every panel drawn to the IEC 60263 scale conventions. The distortion and
frequency-response measurements themselves live in
[Electroacoustics](electroacoustics.md); the loudspeaker counterpart has
[its own guide](loudspeakers.md).

## 1. Sensitivity and its references (IEC 60268-4 clause 11)

**Microphones (IEC 60268-4 clause 11).** The sensitivity $M$ is the output
voltage per unit sound pressure, quoted in mV/Pa; the sensitivity *level* is
$L_M = 20\log_{10}(M / 1\,\text{V/Pa})$ dB, which is negative for every real
microphone (a
50 mV/Pa studio condenser sits at −26 dB re 1 V/Pa). The classic pitfall is
the reference: −26 dB re 1 V/Pa and +34 dB re 1 mV/Pa are the *same*
microphone, 60 dB apart on paper, so a sensitivity level without its reference
is meaningless. The standard also distinguishes free-field, diffuse-field and
pressure sensitivity; for the same capsule they diverge at high frequency,
so the stated type matters as much as the number.

Behind the number sits the transducer. A **condenser** capsule (externally
polarised or electret) reads the sound pressure as a capacitance change and
needs powering, but converts with high sensitivity, typically some tens of
mV/Pa for a studio diaphragm and a few mV/Pa for a small measurement
capsule; virtually every measurement microphone is a condenser because the
stretched-membrane capsule is stable enough to calibrate. A **dynamic**
(moving-coil) microphone generates its voltage by induction, needs no
power and survives abuse, but converts far less efficiently, around
1-3 mV/Pa, which is why a dynamic on a quiet source runs out of
preamplifier gain before a condenser does. The rated sensitivity of the
standard is quoted at the 1 kHz reference frequency (clause 11.3),
into the rated load, and the stated type (free-field, diffuse-field or
pressure) is part of the rating, not a detail.

## 2. Directional pattern and the directivity index (clause 13)

The directional response is stated as the pattern of sensitivity against
the angle of incidence, normalized to the reference axis. The classic
first-order family mixes an omnidirectional (pressure) term and a cosine
(pressure-gradient) term, $M(\theta) \propto (1 - b) + b\cos\theta$: $b = 0$
is the omnidirectional capsule, $b = 1/2$ the cardioid with its null at 180°,
and $b = 1$ the figure-of-eight with nulls at ±90°. One number condenses the
pattern: the **directivity index** $D = 20\log_{10}(M_0/M_\mathrm{diff})$ compares the
reference-axis sensitivity with the diffuse-field sensitivity obtained from
the clause 11.2.2 a) integral over a rotationally symmetric pattern. An
omni scores 0 dB, the ideal cardioid and figure-of-eight both
$10\log_{10} 3 = 4.8$ dB: in a diffuse room field, a cardioid picks up three
times less reverberant power than an omni of equal axial sensitivity, which
is exactly the ratio a talker-to-microphone distance calculation wants.
Real patterns hold their textbook shape only over the middle of the band;
at high frequency the capsule's own size makes any microphone directive,
which is also why the free-field and diffuse-field sensitivities of one
capsule diverge there.

## 3. Inherent noise: dB(A) and dB(468) (clause 17)

A microphone's electronics and the air load on its diaphragm set a noise
floor, expressed as the **equivalent noise level**: the sound pressure
level whose output would equal the weighted inherent-noise voltage,
$L_\mathrm{N} = 20\log_{10}\bigl((U_\mathrm{N}/M)/20\,\mu\text{Pa}\bigr)$. The standard states it
with two weightings,
and the numbers are far apart on paper. The **A-weighted** figure (RMS
detector) is the one most datasheets quote, e.g. 14 dB(A) for a good studio
condenser. The **ITU-R BS.468-4** figure weights the same noise with the
curve that peaks +12.2 dB near 6.3 kHz and reads it with a quasi-peak
detector, so it penalises the hiss and crackle the ear actually notices;
for typical microphone noise it comes out roughly 10 dB above the
A-weighted number for the same capsule. Neither is wrong: they are
different weightings of the same voltage, and comparing a dB(A) figure from
one datasheet with a dB(468) figure from another silently flatters the
first by that margin. The signal-to-noise ratio re 1 Pa (94 dB SPL) is
derived from the same equivalent noise level, and the overload sound
pressure level (clause 15.2) bounds the usable range from above. The
BS.468 weighting curve itself is exposed as `itu_r_468_weighting` in the
[electroacoustics distortion set](electroacoustics.md), where it also
weights THD. The quasi-peak detector itself is not implemented: an
inherent-noise level computed from `itu_r_468_weighting` and an r.m.s. sum
is not a dBqps figure.

## 4. Microphone characteristics report (IEC 60268-4)

The microphone companion of the
[loudspeaker report](loudspeakers.md): the rated characteristics IEC 60268-4
defines around a measured free-field frequency response are gathered into a single
**microphone characteristics** result that renders the standard's
rated-characteristics data sheet. Four of the numbers are computed from the
standard's own definitions rather than merely repeated:

- **Sensitivity level** (11.1). The rated free-field sensitivity $M$ (mV/Pa,
  at the 1 kHz reference frequency of 11.3) as a level,
  $L_M = 20\log_{10}(M / 1\,\text{V/Pa})$: 12.5 mV/Pa is $-38.1$ dB re 1 V/Pa.
- **Effective frequency range** (12.2). The band over which the response,
  normalized to 0 dB at the reference frequency, stays within the stated
  $\pm$ tolerance; the edges are the interpolated tolerance crossings.
- **Directivity index** (13.2.2). $D = 20\log_{10}(M_0/M_\mathrm{diff})$ with the
  diffuse-field sensitivity from the 11.2.2 a) integral over a rotationally
  symmetric pattern; the ideal cardioid returns $10\log_{10} 3 = 4.8$ dB.
- **Equivalent noise level** (17.2). The weighted inherent-noise voltage over
  the rated sensitivity as a sound pressure level,
  $L_\mathrm{N} = 20\log_{10}\bigl((U_\mathrm{N}/M)/20\,\mu\text{Pa}\bigr)$, with the
  signal-to-noise ratio re 1 Pa
  (94 dB SPL) derived from it. The overload sound pressure level (15.2) is
  read from a distortion-against-level curve at the stated THD limit.

```python
import numpy as np
from phonometry import (
    MicrophoneDirectivity, MicrophoneElectrical, MicrophoneNoise,
    MicrophoneOverload, ReportMetadata, microphone_characteristics,
)

freqs = np.geomspace(20, 20000, 400)
response = -10 * np.log10(1 + (30.0 / freqs) ** 4)      # low-frequency roll-off
response -= 10 * np.log10(1 + (freqs / 19000.0) ** 8)   # high-frequency roll-off
response += 2.0 * np.exp(-(np.log2(freqs / 9000.0) ** 2) / 0.3)  # presence region

angles = np.linspace(0, 179, 359)
cardioid = 20 * np.log10((1 + np.cos(np.radians(angles))) / 2)
noise_f = np.geomspace(20, 20000, 31)

result = microphone_characteristics(
    freqs, response, 12.5, tolerance_db=3.0,          # 12.5 mV/Pa at 1 kHz
    directivity=MicrophoneDirectivity(polar=(angles, cardioid), frequency=1000.0),
    noise=MicrophoneNoise(                             # A-weighted, V
        voltage=1.25e-6,
        spectrum=(noise_f, 6.0 + 12.0 * np.log10(1000.0 / noise_f)),
    ),
    overload=MicrophoneOverload(
        distortion=(np.linspace(100, 140, 81),
                    0.5 * 10 ** ((np.linspace(100, 140, 81) - 130.0) * 0.08)),
        thd_percent=0.5,
    ),
    electrical=MicrophoneElectrical(
        rated_impedance=150.0, minimum_load_impedance=1000.0,
        powering="Phantom P48 (IEC 61938)", supply_current_ma=3.1,
    ),
)
print(round(result.sensitivity_level_db, 1))            # -38.1 dB re 1 V/Pa
print(tuple(round(x) for x in result.effective_range))  # Hz
print(round(result.directivity_index_db, 1))            # 4.8 dB (cardioid)
print(round(result.equivalent_noise_level_db, 1))       # dB(A)

result.report("microphone.pdf", metadata=ReportMetadata(measurement_standard="IEC 60268-4"))
```

`microphone_characteristics` returns a `MicrophoneCharacteristics` with the
computed `sensitivity_level_db`, `effective_range`, `directivity_index_db`,
`equivalent_noise_level_db`, `max_spl_db`, `signal_to_noise_ratio_db` and
`diffuse_field_sensitivity_level_db`, and a `.report()` that writes the fiche.
The free-field response is drawn with its tolerance band, the
reference-frequency marker and the effective-range markers to the IEC 60263
proportion (one frequency decade equal to 25 dB), and the directional pattern
on the IEC 60263 25 dB reference circle; the inherent-noise spectrum and the
distortion-against-level curve (with the THD limit and the overload level
marked) feed the secondary panels. A `requirement` in the metadata is checked
as a maximum permitted equivalent noise level. What the fiche cannot supply
is the laboratory behind the numbers: the anechoic room and its
qualification, the reference microphone and its calibration certificate, the
high-level source that produces a 140 dB field and the turntable are what
the reported numbers mean, and the acquisition itself is not implemented
here — the functions reduce the curves they are handed.

The rated characteristics are also available interactively through `.plot()`,
which draws **one concept per figure** with the same panel code the report
composes, selected by `quantity`. Passing an axes draws on it:

```python
result.plot()                        # free-field response (default)
result.plot(quantity="directivity")  # polar pattern on the 25 dB circle
result.plot(quantity="noise")        # inherent-noise band spectrum
result.plot(quantity="distortion")   # THD vs sound pressure level
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/microphone_response_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/microphone_response.svg" alt="Microphone free-field relative response with its shaded tolerance band, the reference-frequency marker and the effective-range markers on a nominal-frequency axis" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import microphone_characteristics

freqs = np.geomspace(20, 20000, 400)
response = -10 * np.log10(1 + (30.0 / freqs) ** 4)      # low-frequency roll-off
response -= 10 * np.log10(1 + (freqs / 19000.0) ** 8)   # high-frequency roll-off
response += 2.0 * np.exp(-(np.log2(freqs / 9000.0) ** 2) / 0.3)  # presence region

result = microphone_characteristics(freqs, response, 12.5, tolerance_db=3.0)
result.plot()   # quantity="response" (the default)
plt.show()
```

</details>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/microphone_directivity_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/microphone_directivity.svg" alt="Microphone cardioid directional pattern at 1000 Hz on the IEC 60263 25 dB reference circle" width="72%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import MicrophoneDirectivity, microphone_characteristics

freqs = np.geomspace(20, 20000, 400)
response = -10 * np.log10(1 + (30.0 / freqs) ** 4)
angles = np.linspace(0, 179, 359)
cardioid = 20 * np.log10((1 + np.cos(np.radians(angles))) / 2)

result = microphone_characteristics(
    freqs, response, 12.5, tolerance_db=3.0,
    directivity=MicrophoneDirectivity(polar=(angles, cardioid), frequency=1000.0),
)
result.plot(quantity="directivity")
plt.show()
```

</details>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/microphone_noise_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/microphone_noise.svg" alt="Microphone inherent-noise equivalent band-level spectrum against frequency" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import MicrophoneNoise, microphone_characteristics

freqs = np.geomspace(20, 20000, 400)
response = -10 * np.log10(1 + (30.0 / freqs) ** 4)
noise_f = np.geomspace(20, 20000, 31)

result = microphone_characteristics(
    freqs, response, 12.5, tolerance_db=3.0,
    noise=MicrophoneNoise(
        voltage=1.25e-6,
        spectrum=(noise_f, 6.0 + 12.0 * np.log10(1000.0 / noise_f)),
    ),
)
result.plot(quantity="noise")
plt.show()
```

</details>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/microphone_distortion_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/microphone_distortion.svg" alt="Microphone total harmonic distortion in percent against sound pressure level, with the THD limit and the overload sound pressure level marked" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import MicrophoneOverload, microphone_characteristics

freqs = np.geomspace(20, 20000, 400)
response = -10 * np.log10(1 + (30.0 / freqs) ** 4)
spl_axis = np.linspace(100, 140, 81)

result = microphone_characteristics(
    freqs, response, 12.5, tolerance_db=3.0,
    overload=MicrophoneOverload(
        distortion=(spl_axis, 0.5 * 10 ** ((spl_axis - 130.0) * 0.08)),
        thd_percent=0.5,
    ),
)
result.plot(quantity="distortion")
plt.show()
```

</details>

The example fiche, regenerated with `make reports`, is kept rendered in the
repository. Click the preview to open the PDF:

[![IEC 60268-4 microphone characteristics example report: a header with the manufacturer and model, the rated-characteristics table (free-field sensitivity in mV/Pa and its level re 1 V/Pa, effective frequency range, rated and minimum load impedances, equivalent noise level, signal-to-noise ratio, maximum SPL at the stated THD limit, directivity index and phantom powering) beside the free-field frequency response with its tolerance band and effective-range markers, and the directional-pattern, inherent-noise-spectrum and distortion panels, all drawn to the IEC 60263 scale conventions](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iec60268_4_microphone_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iec60268_4_microphone_example.pdf)

*Microphone characteristics fiche (`MicrophoneCharacteristics.report`), the
IEC 60268-4 rated-characteristics table beside the free-field response, with
the cardioid directional pattern, inherent-noise and THD-against-level panels
drawn to the IEC 60263 25 dB-per-decade and 25 dB reference-circle
conventions.*


## See also

- [Electroacoustics](electroacoustics.md): the IEC 60268-3 distortion set,
  THD+N and SINAD, the ITU-R 468 weighted THD and the H1/H2
  frequency-response estimators that produce the measured curves.
- [Loudspeaker Characterisation (IEC 60268-5)](loudspeakers.md): the
  companion rated-characteristics report for the reproducing side of the
  chain.
- [Calibration and dBFS](../../signals/metrology/calibration.md): turning a microphone's rated
  sensitivity into calibrated sound pressure levels.
- [Build a sound level meter](../../signals/sound-level-meter.md): the measurement chain a
  characterised microphone feeds.
- API reference: [`electroacoustics.microphone`](https://jmrplens.github.io/phonometry/reference/api/electroacoustics/microphone/).

## References

- Beranek, L. L., & Mellow, T. J. (2012). *Acoustics: Sound fields and
  transducers*. Academic Press. ISBN 978-0-12-391421-7.
  [doi:10.1016/C2011-0-05897-0](https://doi.org/10.1016/C2011-0-05897-0).
  The transducer physics behind sections 1 and 2: condenser and moving-coil
  transduction and first-order directional patterns.
- International Telecommunication Union. (1986). *Measurement of
  audio-frequency noise voltage level in sound broadcasting*
  (Recommendation ITU-R BS.468-4).
  [itu.int](https://www.itu.int/rec/R-REC-BS.468/en).
  The weighting network and quasi-peak detection behind the dB(468)
  inherent-noise figure of section 3.

## Standards

IEC 60268-4:2014, *Sound system equipment – Part 4: Microphones*: the rated
microphone characteristics of this guide, namely the free-field sensitivity and
its level re 1 V/Pa (11.1/11.3), the frequency response and effective
frequency range (12.1/12.2), the directional pattern and the directivity
index through the 11.2.2 a) diffuse-field integral (13.1/13.2), the
overload sound pressure level (15.2), the equivalent sound pressure level
due to inherent noise (17) and the rated impedances and power supply
(9/10); since revised as IEC 60268-4:2018, the 2014 edition is the
implemented one. IEC 60263:1982, *Scales and sizes for plotting frequency
characteristics and polar diagrams*: the scale proportions of the
characteristic graphs, one frequency decade equal to 25 dB on the ordinate
(clause 2) and the polar diagram on a 25 dB reference-circle radius
(clause 3). ITU-R BS.468-4: the weighting-network nominal response behind
the dB(468) noise figure.
