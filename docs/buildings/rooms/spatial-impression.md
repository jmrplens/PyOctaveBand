← [Documentation index](../../README.md)

# Spatial impression (ISO 3382-1)

Two halls can have the same reverberation time, the same clarity and the
same [sound strength](sound-strength.md) and still
sound nothing alike, because those quantities do not care **where** the
sound comes from. A narrow hall throws early reflections at the listener's
ears from the side; a wide one throws them from above and in front. The
first sounds broad and enveloping, the second sounds flat, and no
omnidirectional microphone can tell them apart.

ISO 3382-1 measures the difference with a second microphone. Annex A.2.4
and A.2.5 use a figure-of-eight pattern alongside the omnidirectional one,
and Annex B uses a head with a microphone at each ear canal.

## 1. Two weightings for one reflection

The figure-of-eight microphone is aimed with its **null** at the source, so
the direct sound weighs nothing and its output follows the cosine of the
angle each reflection arrives at. Squaring the pressure, as Equation (A.14)
does, weights that reflection by $\cos^2 \theta$:

$$
J_\mathrm{LF} = \frac{\int_{0,005}^{0,080} p_L^2(t)\ \mathrm{d}t}
                     {\int_{0}^{0,080} p^2(t)\ \mathrm{d}t}.
$$

Multiplying it by the omnidirectional response instead, as Equation (A.15)
does, weights it by the cosine itself, which A.2.4 calls subjectively more
accurate:

$$
J_\mathrm{LFC} = \frac{\int_{0,005}^{0,080}
                       \left| p_L(t) \cdot p(t) \right|\ \mathrm{d}t}
                      {\int_{0}^{0,080} p^2(t)\ \mathrm{d}t}.
$$

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/lateral_energy_measures_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/lateral_energy_measures.svg" alt="Three panels. Left: the weight one reflection carries against its angle of incidence from the axis of maximum sensitivity of the figure-of-eight microphone, as the square of the cosine for the lateral fraction and as the cosine itself for the cosine-weighted variant, both peaking at one on that axis and falling to zero on the two nulls at plus and minus ninety degrees, where the source sits. Middle: both fractions band by band for a hall with two seconds of decay, over the shaded 0.05 to 0.35 typical range of Table A.1. Right: the interaural correlation function of the 500 Hz octave band across the plus and minus one millisecond search window, for one signal fed to both ears and for two independent ones, with the maximum of each marked" width="100%"></picture>

*The two weightings, both fractions band by band, and the window the
interaural coefficient is the maximum over.*

```python
import numpy as np
from phonometry import room

fs = 48000
# One reflection of half the direct amplitude, 20 ms late, arriving at 45
# degrees from the microphone axis. The figure-of-eight microphone sees it at
# 0.5 cos(45); the omnidirectional one at 0.5, on top of the unit direct.
n = int(0.3 * fs)
omni, lateral = np.zeros(n), np.zeros(n)
omni[100] = 1.0
omni[100 + int(0.020 * fs)] = 0.5
lateral[100 + int(0.020 * fs)] = 0.5 * np.cos(np.deg2rad(45.0))

squared = room.early_lateral_energy_fraction(omni, lateral, fs, limits=None)
cosine = room.early_lateral_energy_fraction(
    omni, lateral, fs, weighting="cosine", limits=None
)
print(squared.energy_fraction.round(4))     # [0.1]     = 0.25 cos^2(45) / 1.25
print(cosine.energy_fraction.round(4))      # [0.1414]  = 0.25 cos(45) / 1.25
```

**The two lower limits differ, and that is printed.** The numerator starts
at 5 ms and the denominator at 0: the direct sound belongs to the total
early energy but not to the lateral share, and the 5 ms keeps whatever
leaks through the microphone's null out of the numerator.

**Time zero comes from the omnidirectional response.** A figure-of-eight
microphone aimed as A.2.4 asks has no direct sound to trigger on, so its own
onset detector lands on the first strong reflection and shifts both
integration limits by that much. Both functions here take their time zero
from the response you pass first.

## 2. The modulus that a text layer deletes

Equation (A.15) prints a modulus around $p_L(t) \cdot p(t)$. It is easy to
lose, because `pdftotext` renders the same numerator without it, and losing
it inverts the meaning of the quantity: a figure-of-eight microphone
responds with opposite sign to the two sides, so two mirror-image
reflections cancel to **exactly zero** instead of adding.

```python
import numpy as np
from phonometry import room

fs = 48000
n = int(0.3 * fs)
omni, lateral = np.zeros(n), np.zeros(n)
omni[100] = 1.0
for time, side in ((0.020, +1.0), (0.030, -1.0)):
    omni[100 + int(time * fs)] = 0.5
    lateral[100 + int(time * fs)] = side * 0.5 * np.cos(np.deg2rad(45.0))

result = room.early_lateral_energy_fraction(
    omni, lateral, fs, weighting="cosine", limits=None
)
print(result.energy_fraction.round(4))      # [0.2357], twice one reflection

signed = (0.25 * np.cos(np.deg2rad(45.0)) - 0.25 * np.cos(np.deg2rad(45.0))) / 1.5
print(round(signed, 12))                    # 0.0, the misreading
```

Two rooms, one with mirror-image reflections from both sides and one with
no lateral reflections at all, would come out identical and zero. The
library reads the page.

## 3. Envelopment: the level of what arrives late

$J_\mathrm{LF}$ is a fraction, so it cancels its own calibration. The late
lateral sound level of Equation (A.16) is a level and does not: it is the
lateral energy after the early window against the free-field reference at
10 m, the same reference the sound strength uses.

$$
L_J = 10 \lg \frac{\int_{0,080}^{\infty} p_L^2(t)\ \mathrm{d}t}
                  {\int_{0}^{\infty} p_{10}^2(t)\ \mathrm{d}t}\ \text{dB}.
$$

Equation (A.17) averages it over the 125 Hz, 250 Hz, 500 Hz and 1 kHz
octave bands with a factor of 0,25, which is one quarter, so it is an
energy **mean**:

$$
L_{J,\mathrm{avg}} = 10 \lg \left[ 0{,}25
                     \sum_{i=1}^{4} 10^{L_{J_i}/10} \right]\ \text{dB}.
$$

Footnote a of Table A.1 makes this the one exception in the whole table:
every other quantity in it is averaged arithmetically over its bands, and
only $L_J$ is averaged over energy. The two are not close when the bands
disagree.

```python
import numpy as np
from phonometry import room

print(round(room.late_lateral_average([-8.0] * 4), 4))     # -8.0, unchanged
print(round(room.late_lateral_average([0.0, 0.0, 0.0, 6.0206]), 4))   # 2.4304

bands = [-14.0, -8.0, -5.0, 1.0]
print(round(room.late_lateral_average(bands), 4))          # -3.5324, energy
print(round(float(np.mean(bands)), 4))                     # -6.5, arithmetic
```

Nearly 3 dB apart, and Table A.1 does not print a just-noticeable
difference for $L_J$ at all: it says "Not known".

## 4. Two ears, one coefficient

Annex B measures the same aspect with a dummy head. Equation (B.1) is the
normalised cross correlation of the two ear responses,

$$
\mathrm{IACF}_{t_1,t_2}(\tau) =
    \frac{\int_{t_1}^{t_2} p_l(t)\, p_r(t + \tau)\ \mathrm{d}t}
         {\sqrt{\int_{t_1}^{t_2} p_l^2(t)\ \mathrm{d}t
                \int_{t_1}^{t_2} p_r^2(t)\ \mathrm{d}t}},
$$

and Equation (B.2) takes its largest magnitude within a millisecond of
coincidence, which is about the interaural delay of a head:

$$
\mathrm{IACC}_{t_1,t_2} = \max \left| \mathrm{IACF}_{t_1,t_2} \right|
    \quad \text{for } -1\ \text{ms} < \tau < +1\ \text{ms}.
$$

**Both the square root and the modulus are printed, and both are lost by a
text layer.** The root is what bounds the function by one; without it the
result is not a correlation at all and scales with the gain of either ear.
The modulus is what makes two anti-phase ears score 1: they are as
dissimilar as two signals can be in sign alone, and 0 would be the wrong
answer for them.

```python
import numpy as np
from phonometry import room

fs = 48000
t = np.arange(int(1.5 * fs)) / fs
rng = np.random.default_rng(3382)
ear = rng.standard_normal(t.size) * np.exp(-3.0 * np.log(10.0) * t / 2.0)

print(room.interaural_cross_correlation(ear, ear, fs, limits=None).coefficient)
# [1.]  the square root is what makes this exactly one
print(room.interaural_cross_correlation(-ear, ear, fs, limits=None).coefficient)
# [1.]  the modulus is what makes this one and not zero

delayed = np.concatenate([np.zeros(int(0.0005 * fs)), ear])[: ear.size]
found = room.interaural_cross_correlation(ear, delayed, fs, limits=None)
print(found.delay * 1000.0)                 # [0.5]  ms, the right ear lags
```

B.4 prints three windows: the general one from the direct sound to a time of
the order of the reverberation time, which is the default; the early one,
`IACC_EARLY_WINDOW_S`, from 0 to 80 ms; and the reverberant one from
`IACC_LATE_START_S` onwards. It puts the range at the 125 Hz to 4 kHz
octave bands and assumes a just-noticeable difference of 0,075.

A broadband correlation is a spike and says very little. Measure in bands,
which is what the default does and what the right-hand panel above draws.

## 5. What the second microphone has to be

Both lateral measures need the two microphones to record one event at one
point, so the library refuses two responses of different lengths. $L_J$
additionally needs their relative sensitivity to have been calibrated in a
free field (A.3.2), because it is a level and the figure-of-eight response
carries its own gain into it. $J_\mathrm{LF}$ does not: a common factor on
both cancels in the ratio, and a factor on the figure-of-eight alone shows
up as a straight scaling of the fraction.

The 80 ms windows are the printed ones, and a response that stops before
them raises rather than shortening itself: an 80 ms integral taken over the
48 ms that were recorded is not the printed quantity, because the 32 ms that
were never recorded are missing from the numerator and from the denominator
alike, and which way the fraction then moves depends on how lateral the part
that was lost was.

## See also

- [Sound strength G (ISO 3382-1)](sound-strength.md): the other quantity of
  Annex A that needs a calibrated source and the same free-field reference.
- [Room Acoustic Parameters](room-acoustics.md): the decay times and energy
  ratios one microphone is enough for.
- [Measuring the Room Impulse Response](room-impulse-response.md): the
  ISO 18233 acquisition of the responses this page pairs up.
- [Image sources and the steady-state room field](room-image-sources.md):
  where the early reflections this page weighs come from.
- API reference: [`room.auditorium`](https://jmrplens.github.io/phonometry/reference/api/rooms/auditorium/).

## Standards

ISO 3382-1:2009, Annex A (informative) A.2.4 and A.2.5 and Annex B
(informative): the early lateral energy fraction of Equations (A.14) and
(A.15), the late lateral sound level of Equation (A.16) with the energy
average of Equation (A.17), and the interaural cross correlation of
Equations (B.1) and (B.2). Validated against closed-form reflection
geometries and against the printed characters a text layer deletes, in the
[conformance report](../../CONFORMANCE.md).
