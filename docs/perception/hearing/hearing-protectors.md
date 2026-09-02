← [Documentation index](../../README.md)

# Hearing Protectors (ISO 4869-2)

A hearing protector is not measured on a coupler. ISO 4869-1 seats it on at
least sixteen people and records the threshold shift each of them gets, so what
comes back from the laboratory is a **distribution**: one attenuation per
subject per octave band, with a spread that is often a third of the mean.
ISO 4869-2 is the standard that turns that distribution into a number someone
can act on, and the first thing it does is refuse to use the mean.

## The distribution first (Clause 5)

Every method starts from the assumed protection value, the mean attenuation
reduced by a multiple of its own spread:

$$
APV_{fx} = m_f - \alpha\, s_f
$$

$\alpha$ is the inverse standard normal cumulative distribution at the
protection performance $x$ (Table 1), so $APV_{f84}$ with $\alpha = 1$ is the
attenuation 84 % of wearers reach or beat, and $APV_{f98}$ with $\alpha = 2$ is
what all but one in fifty reach.

```python
import numpy as np
from phonometry import hearing

# ISO 4869-1 attenuation: 16 subjects, eight octave bands from 63 Hz to 8 kHz.
attenuation = np.array([
    [4, 8, 13, 18, 20, 30, 35, 30],   [6, 12, 16, 21, 29, 35, 47, 35],
    [10, 16, 17, 23, 25, 32, 48, 37], [3, 7, 12, 18, 20, 25, 33, 30],
    [8, 10, 16, 16, 25, 27, 43, 32],  [4, 7, 10, 15, 19, 32, 35, 31],
    [5, 5, 9, 16, 20, 25, 30, 28],    [15, 15, 21, 26, 25, 38, 46, 38],
    [5, 6, 10, 13, 19, 22, 29, 28],   [9, 9, 10, 19, 20, 27, 37, 31],
    [9, 16, 18, 24, 25, 35, 44, 39],  [5, 6, 11, 12, 17, 20, 28, 28],
    [7, 10, 17, 22, 25, 35, 41, 44],  [6, 8, 16, 18, 19, 19, 30, 33],
    [10, 12, 17, 25, 28, 33, 45, 40], [12, 13, 17, 27, 29, 38, 49, 41],
], dtype=float)

apv = hearing.assumed_protection_value(attenuation)          # x = 84 % by default
print(np.round(apv.apv, 1))    # [ 4.1  6.4 10.7 14.9 18.8 23.4 31.3 28.9]
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/hearing_protector_methods_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/hearing_protector_methods.svg" alt="Left: the mean sound attenuation of a hearing protector across the eight octave bands from 63 Hz to 8 kHz, with its standard deviation shaded either side and the assumed protection value for 84 % of wearers drawn a full standard deviation below the mean. Right: the predicted noise level reduction as a function of the difference between the C-weighted and A-weighted levels of the noise, drawn as two straight segments through the H, M and L anchors, with the eight reference noises scattered at their own differences and the three methods' answers for one noise boxed" width="92%"></picture>

*Left, the protector: the assumed protection value sits a full standard
deviation below the mean, and the gap is widest where the spread is, at 4 kHz.
Right, the method: the HML line and the eight reference noises it was fitted
on, with the three methods' answers for the same noise.*

## Three methods, in decreasing order of what they need

**The octave-band method (Clause 6)** is the most faithful and the only one
that sees the shape of the noise: subtract the assumed protection value band by
band from the A-weighted spectrum and sum what is left (Formula (2)).

**The HML method (Clause 7)** collapses the protector to three numbers, the
predicted noise level reduction it gives for reference noises whose
$(L_{p,C} - L_{p,A})$ is $-2$, $+2$ and $+10$ dB, fitted across the eight
reference spectra of Table 2. Applying them needs only the C- and A-weighted
levels of the real noise, through two straight segments that meet at $+2$ dB.

**The SNR method (Clause 8)** collapses it to one number against a pink noise
and subtracts it from the C-weighted level.

```python
noise = [75.0, 84.0, 86.0, 88.0, 97.0, 99.0, 97.0, 96.0]   # LpA = 104, LpC = 103 dB
octave = hearing.octave_band_protected_level(noise, apv)
hml = hearing.hml_rating(attenuation)
snr = hearing.snr_rating(attenuation)

print(octave.reported_level)                                          # 81
print(hml.reported)                                                   # (24, 18, 13)
print(hearing.hml_protected_level(104.0, 103.0, hml).reported_level)  # 82
print(snr.reported)                                                   # 21
print(hearing.snr_protected_level(snr, l_p_c=103.0).reported_level)   # 82
```

The three answer the same question and rarely agree exactly: 81 dB, 82 dB and
82 dB for one protector in one noise. Clause 1's own NOTE puts differences of
3 dB or less between comparable protectors below the resolution of the
exercise. The ordering is not a ranking: the octave-band method uses more
information and is the one to prefer when the spectrum is available, while HML
and SNR exist precisely for when it is not.

The values that enter the HML and SNR applications are the **rounded** ones:
Clauses 7.2 and 8.2 round the ratings to the nearest integer, which is what a
protector is published with. All three computations begin at 125 Hz; Formula
(2) may start at 63 Hz when both the noise and the protector have data there.

One caution about the reference spectra: Annex C reprints Table 2 as its
Table C.1 and the reprint disagrees with the original in two cells. Table 2 is
the one that reproduces the annex's own worked results, and it is the one this
library carries; the discrepancy is registered in [ERRATA](../../ERRATA.md).

## References

- International Organization for Standardization (2018). *Acoustics — Hearing
  protectors — Part 2: Estimation of effective A-weighted sound pressure levels
  when hearing protectors are worn* (ISO 4869-2:2018).
  [iso.org catalogue](https://www.iso.org/standard/70090.html).
  The implemented standard: Clauses 5 to 8 and the worked examples of
  Annexes A to D.
- International Organization for Standardization (2018). *Acoustics — Hearing
  protectors — Part 1: Subjective method for the measurement of sound
  attenuation* (ISO 4869-1:2018).
  [iso.org catalogue](https://www.iso.org/standard/69233.html).
  Where the per-subject attenuation values come from.

## Standards

ISO 4869-2:2018, which defines the assumed protection value $APV_{fx}$
(Clause 5), the octave-band method (Clause 6), the $H$, $M$ and $L$ values
(Clause 7) and the single number rating $SNR$ (Clause 8). The attenuation they
all start from is measured to ISO 4869-1:2018.

## See also

- [Occupational Noise Exposure (ISO 9612)](occupational-exposure.md): the daily
  exposure level the protected level feeds into.
- [Noise-induced hearing loss (ISO 1999)](noise-induced-hearing-loss.md): what
  the exposure the protector did not stop does over a working life.
- [Hearing threshold (age and reference zero)](hearing-threshold.md): the
  baseline any protected exposure is judged against.
- API reference: [`hearing.hearing_protectors`](https://jmrplens.github.io/phonometry/reference/api/hearing/hearing-protectors/).
