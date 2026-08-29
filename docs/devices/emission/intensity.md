← [Documentation index](../../README.md)

# Sound Intensity (p-p method)

Sound *pressure* tells you how loud a point is; sound **intensity** tells
you where the energy is *going*. It is the acoustic power flux (W/m²), a
signed vector quantity, which is why intensity probes can localize sources,
separate them from background noise and measure sound power in situ
(ISO 9614) where a pressure measurement alone cannot.

## The two-microphone principle (IEC 61043)

A p-p probe holds two matched microphones a small distance $\Delta r$ apart. The
pressure at the probe center is their mean, and the particle velocity comes
from the pressure *gradient* (Euler's equation, finite-difference form):

$$
p = \frac{p_1 + p_2}{2}, \qquad
u = -\frac{1}{\rho_0\ \Delta r}\int (p_2 - p_1)\ dt, \qquad
I = \overline{p\ u}
$$

In practice the estimator works in the frequency domain through the
cross-spectrum of the two channels (the standard's equivalent form):

$$
I(f) = -\ \frac{\mathrm{Im}\lbrace G_{12}(f)\rbrace}{2\pi f\ \rho_0\ \Delta r}
$$

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_pp_probe_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_pp_probe.svg" alt="Two-microphone p-p intensity probe with the spacer distance and the measurement axis" width="92%"></picture>

The probe itself is small enough to draw at true scale.
`plot_pp_probe_geometry` puts the classic 12 mm solid spacer between the two
face-to-face capsules, and a computed `IntensityResult` that retained its
`spacing` redraws its own probe with `res.plot_geometry()`.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/pp_probe_geometry_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/pp_probe_geometry.svg" alt="To-scale side view of the face-to-face p-p intensity probe: two blue half-inch microphone capsules on their grey cylindrical bodies facing each other across the light 12 mm solid spacer, the spacing dimensioned below and the intensity axis Ir drawn as a blue arrow to the right" width="92%"></picture>

*The finite difference at true scale: two half-inch capsules face to face
across 12 mm of solid spacer, and that $\Delta r$ is both the sensitivity of
the gradient estimate and the origin of the high-frequency error.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import emission

# The classic 12 mm solid spacer between two half-inch microphones.
emission.plot_pp_probe_geometry()
plt.show()

# A computed IntensityResult retains its spacing:
#   res = emission.sound_intensity(p1, p2, fs, spacing=0.012)
#   res.plot_geometry()
```

</details>

```python
import numpy as np
from phonometry import emission

fs = 48000
rng = np.random.default_rng(0)
# The two probe-microphone pressures in Pa, p1 closest to the source.
#   In a real measurement these are your two calibrated probe recordings;
#   synthesized here (p2 = p1 delayed one sample) so the guide runs.
p1 = 0.02 * rng.standard_normal(fs)
p2 = np.concatenate(([0.0], p1[:-1]))   # p2 = p1 delayed one sample

res = emission.sound_intensity(p1, p2, fs, spacing=0.012, fraction=3,
                      limits=[100, 2500])
print(res.total_intensity_level, res.total_direction)      # LI [dB], ±1
print(res.frequency, res.intensity_level)                  # per band
res.plot()   # Lp vs LI per band + the pressure-intensity index (needs matplotlib)
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/intensity_demo_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/intensity_demo.svg" alt="Third-octave pressure and intensity levels for a plane progressive wave versus a standing wave" width="92%"></picture>

*Left: in a plane progressive wave all pressure is transported, so
$L_I \approx L_p$. Right: a standing wave carries (almost) no net energy, so
the pressure is high but the intensity collapses. The gap $L_p - L_I$ is the
**pressure-intensity index**, the fundamental quality indicator of every
intensity measurement.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import emission

fs = 48000
rng = np.random.default_rng(0)
# The two probe-microphone pressures in Pa, p1 closest to the source.
#   In a real measurement these are your two calibrated probe recordings;
#   synthesized here (p2 = p1 delayed one sample) so the guide runs.
p1 = 0.02 * rng.standard_normal(fs)
p2 = np.concatenate(([0.0], p1[:-1]))   # p2 = p1 delayed one sample
res = emission.sound_intensity(p1, p2, fs, spacing=0.012, fraction=3,
                      limits=[100, 2500])

# res is the IntensityResult computed in the example above.
# One line — Lp vs LI per band with the pressure-intensity index on a twin axis:
res.plot()
plt.show()

# By hand, from the per-band fields the result carries — mirroring what
# IntensityResult.plot() draws (bar label, merged twin-axis legend, δpI title):
fig, ax = plt.subplots()
ax.semilogx(res.frequency, res.pressure_level, "o-", label="Pressure level Lp")
ax.semilogx(res.frequency, res.intensity_level, "s--", label="Intensity level LI")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Level [dB]")
twin = ax.twinx()
twin.bar(res.frequency, res.pressure_intensity_index,
         width=res.frequency * 0.2, color="#2ca02c", alpha=0.25,
         label="δpI = Lp − LI")
twin.set_ylabel("Pressure-intensity index δpI [dB]")
# Merge both axes' handles into a single legend, exactly as .plot() does:
lines, labels = ax.get_legend_handles_labels()
tlines, tlabels = twin.get_legend_handles_labels()
ax.legend(lines + tlines, labels + tlabels)
ax.set_title(f"Lp vs LI  (total δpI = {res.total_pressure_intensity_index:.1f} dB)")
plt.show()
```

</details>

The same contrast plays out dynamically below: the pressure and velocity
phasors of a progressive and a standing wave, with the instantaneous
intensity averaging to a net flow in one case and to zero in the other.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_instantaneous_intensity_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_instantaneous_intensity.gif" alt="Animation: a two-microphone p-p probe with rotating pressure and velocity phasors; the instantaneous intensity arrow flips while its running average settles to a net flow for the progressive wave and to zero for the standing wave" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_instantaneous_intensity.webm)

## Knowing when to trust the number

Two physical limits bound every p-p measurement, and the result object
carries both:

- **High frequency**: the finite-difference gradient underestimates $I$ by
  $\sin(k\Delta r)/(k\Delta r)$, verified in CI against IEC 61043 Table 3.
  `IntensityResult.bias_correction` provides the factor and
  `max_valid_frequency` ($\approx 0.1\,c/\Delta r$; 2.9 kHz for a 12 mm
  spacer) the
  practical ceiling. Larger spacers reach lower frequencies, smaller ones
  higher.
- **Reactive fields**: when `pressure_intensity_index` ($F_2$ in ISO 9614-1)
  approaches the probe's residual index $\delta_{pI0}$, phase errors dominate.

Before any of that, ISO 9614-1 asks a question about the *field* rather than
the surface: is it steady enough to be scanned at all? In the initial test
(clause 8.2) one typical position is picked on an initial measurement surface
and the normal intensity is sampled there $M$ times with a short averaging
time (Note 9 suggests $M = 10$, and, for periodic signals, 8 s to 12 s per
sample or any whole number of cycles). **$F_1$**, the
temporal variability indicator, is the coefficient of variation of those
samples (equations (A.1)–(A.2)), so it is zero for a perfectly steady field
and grows as the extraneous intensity wanders. Table B.3 asks for action code
(e) above $F_1 > 0.6$: reduce the variability, measure during quieter periods,
or lengthen the averaging time at each position. Annex B also has it evaluated
immediately before and after the measurement on any one surface (B.1.4).

```python
from phonometry import emission

# The M short-time samples of the normal intensity at one fixed position (W/m²).
samples = [1.20e-5, 0.94e-5, 1.51e-5, 1.08e-5, 1.33e-5,
           1.02e-5, 1.44e-5, 1.17e-5, 0.88e-5, 1.29e-5]

f1 = emission.temporal_variability_indicator(samples)
print(round(f1, 3))                          # 0.177, a steady field

# Or carried on the surface result alongside F2/F3/F4, by handing the same
# samples to field_indicators together with the per-position scan below:
fi = emission.field_indicators([74.1, 73.8, 74.5, 73.2],
                               [1.2e-5, 1.0e-5, 1.4e-5, 0.9e-5],
                               temporal_intensity=samples)
print(fi.field_is_stationary())              # True (Table B.3 limit 0.6)
```

Over a measurement surface, the remaining ISO 9614-1 Annex A field indicators
grade the scan itself. **$F_2$**, the surface pressure-intensity indicator, is the surface
pressure level minus the level of the mean *magnitude* of the normal
intensity: the larger it is, the closer the measurement sits to the probe's
phase-error floor. **$F_3$**, the negative partial power indicator, is the same
difference taken with the *signed* mean intensity: $F_3 - F_2 > 0$ reveals power
flowing inward through parts of the surface. **$F_4$**, the field non-uniformity
indicator, is the normalised spread of the per-position intensities: the
larger it is, the more measurement positions the surface needs. Together with
the dynamic-capability criterion they are available directly:

```python
import numpy as np
from phonometry import emission

# Per-position measurements over the ISO 9614-1 measurement surface
pressure_levels = np.array([74.1, 73.8, 74.5, 73.2])       # Lp per position (dB)
normal_intensity = np.array([1.2e-5, 1.0e-5, 1.4e-5, 0.9e-5])  # signed In per position (W/m²)

fi = emission.field_indicators(pressure_levels, normal_intensity)
print(round(fi.f2, 2), round(fi.f3, 2), round(fi.f4, 3))   # 3.41 3.41 0.197
ld = emission.dynamic_capability_index(18.0)   # δpI0 = 18 dB → Ld = δpI0 − K
print(ld, ld > fi.f2)                                      # 8.0 True (criterion 1)
```

With 2D `(positions, bands)` arrays and the band centres the indicators come
back **per band**, and the result is plottable in one line, the form in which
the criteria are actually checked (each band passes or fails on its own):

```python
fi = emission.field_indicators(lp_bands, in_bands, freqs)   # (positions, bands)
fi.plot(dynamic_capability=ld)   # F2/F3 per band vs Ld, F4 on a twin axis (needs matplotlib)
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/field_indicators_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/field_indicators.svg" alt="ISO 9614-1 field indicators of a ten-position scan over six octave bands: F2 and F3 climb towards low frequency as the field turns reactive, crossing the dashed dynamic-capability line Ld at 125 Hz where F3 also rises above F2, with the field non-uniformity F4 drawn as bars on a twin axis" width="88%"></picture>

*$F_2$ climbs towards low frequency as the field turns reactive, and at 125 Hz
it crosses the instrument's dynamic capability $L_\mathrm{d} = \delta_{pI0} - K$: that
band fails criterion 1, and no averaging will fix it — it calls for a larger
spacer, a different surface or a quieter room. $F_3$ rising above $F_2$ in the
same band reveals inward-flowing (negative) partial intensity, and the $F_4$
bars set the number of positions the surface needs (criterion 2,
$N > C \cdot F_4^2$).*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import emission

# A 10-position discrete-point scan over six octave bands: the surface
# pressure is nearly uniform, and the normal intensity per band is set so the
# field turns reactive towards low frequency, with two inward-flowing
# positions in the 125 Hz band (rescaled so the band mean keeps its target).
freqs = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
delta_pi = np.array([10.5, 8.5, 6.0, 4.5, 3.5, 3.0])   # target Lp − L|In|
rng = np.random.default_rng(9614)
lp_bands = 78.0 + rng.normal(0.0, 0.4, (10, freqs.size))
i_mean = 10.0 ** ((78.0 - delta_pi) / 10.0) * 1.0e-12
in_bands = i_mean[None, :] * (1.0 + rng.normal(0.0, 0.18, (10, freqs.size)))
in_bands[:2, 0] = -0.35 * i_mean[0]
in_bands[2:, 0] *= (10.0 * i_mean[0] - in_bands[:2, 0].sum()) / in_bands[2:, 0].sum()

fi = emission.field_indicators(lp_bands, in_bands, freqs)
ld = emission.dynamic_capability_index(18.0)   # δpI0 = 18 dB, K = 10 dB

# One line — F2/F3 per band against Ld, with F4 on a twin axis:
fi.plot(dynamic_capability=ld)
plt.show()

# By hand, from the per-band fields the result carries — mirroring what
# FieldIndicators.plot() draws (Ld step line, merged twin-axis legend):
fig, ax = plt.subplots()
ax.plot(fi.frequency, fi.f2, "o-", label="F2 (surface pressure-intensity)")
ax.plot(fi.frequency, fi.f3, "s--", label="F3 (negative partial power)")
ax.plot(fi.frequency, np.full(fi.frequency.size, ld), ":",
        drawstyle="steps-mid", label="Dynamic capability Ld")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Indicator [dB]")
twin = ax.twinx()
twin.bar(fi.frequency, fi.f4, width=fi.frequency * 0.2, alpha=0.25,
         color="#2ca02c", label="F4 (non-uniformity)")
twin.set_ylabel("Field non-uniformity F4")
lines, labels = ax.get_legend_handles_labels()
tlines, tlabels = twin.get_legend_handles_labels()
ax.legend(lines + tlines, labels + tlabels)
plt.show()
```

</details>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_intensity_scan_power_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_intensity_scan_power.gif" alt="Animation: a p-p probe traces the serpentine scan over the top face of the measurement box while the normal-intensity arrows appear behind it, and the partial powers of the five faces accumulate into the sound power level L_W" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_intensity_scan_power.webm)

### The margin over the residual index

The two channels of any real probe and analyzer are never perfectly phase
matched. Feed both channels the *same* signal (the residual-intensity test
of IEC 61043): the true intensity is exactly zero, yet the mismatch reports
a small false intensity. The gap between the pressure level and that false
intensity level is the **residual pressure-intensity index** $\delta_{pI0}$, the
instrument's phase-error floor expressed as an index; IEC 61043 grades
probes and processors (class 1 / class 2) chiefly by it.

In the field, the measured index $\delta_{pI} = L_p - L_I$ says how far the
pressure
level stands above the level of the net flow, and the systematic error of
the intensity estimate is bounded by the margin between the two indices:

$$
\varepsilon = 10 \log_{10}\!\left( 1 \pm 10^{(\delta_{pI} - \delta_{pI0})/10} \right)
$$

A 10 dB margin keeps the bias within about 0.5 dB and a 7 dB margin within
about 1 dB; these are precisely the bias factors $K$ of ISO 9614, and the
**dynamic capability** $L_\mathrm{d} = \delta_{pI0} - K$ is the largest field index the
instrument can afford at a given grade. Read it as a budget: every decibel
the field's $\delta_{pI}$ rises spends a decibel of margin, and when
$\delta_{pI}$ reaches $\delta_{pI0}$ the reading is pure phase error, of
either sign. This is why the
pressure-intensity index, not the microphone quality, gates the achievable
accuracy of every intensity measurement.

### Grading the instrument: IEC 61043 Table 2

$\delta_{pI0}$ is not just a number to subtract $K$ from; it is what IEC 61043
grades the hardware by. Table 2 of the standard sets a **minimum**
$\delta_{pI0}$ in every
one-third-octave band from 50 Hz to 6.3 kHz, separately for a **probe**, a
**processor** and a **complete instrument**, in class 1 and class 2, printed
for the nominal 25 mm microphone separation. Note 1 rescales the whole table
for any other spacer by $+10\log_{10}(x/25)$ with $x$ in millimetres, which is the
same 3 dB per doubling the previous section arrived at from the physics.

`intensity_class_compliance` compares a measured $\delta_{pI0}$ spectrum
against both
masks band by band and returns the class the chain actually meets:
the loosest class every band clears, or `None` when some band clears neither:

```python
from phonometry import metrology

# The band centres Table 2 is defined on, and the measured residual index of
# the chain: one value per band, taken with the spacer that will be fitted in
# the field. Replace the placeholder with your own residual-intensity test.
freqs, _, _ = metrology.residual_index_limits("instrument", spacing=0.012)
measured_delta_pi0 = [11.0, 11.9, 11.2, 10.0, 13.2, 15.9, 17.0, 18.0,
                      19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 24.0, 24.0,
                      24.0, 24.0, 24.0, 24.0, 24.0, 24.0]   # dB, 22 bands

res = metrology.intensity_class_compliance(measured_delta_pi0, freqs,
                                           device="instrument", spacing=0.012)
print(res.overall_class)          # 2: one band misses the class 1 minimum
print(res.binding_margin())       # smallest per-band margin to that class [dB]
print(res.failing_bands(1))       # the band centres that cost it class 1 [Hz]
res.plot()                        # measured δpI0 over the two Table 2 masks
res.report("verification.pdf")    # one-page verification fiche (PDF)
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/intensity_class_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/intensity_class.svg" alt="Measured pressure-residual intensity index of a complete intensity instrument with a 12 mm spacer, drawn as a step curve over the IEC 61043 Table 2 class 1 and class 2 minima; the shaded class 2 pass region lies above the dashed class 2 mask, and the 100 Hz band is ringed where the measured index dips below the solid class 1 mask" width="92%"></picture>

*Both Table 2 masks come down by $10\log_{10}(12/25) = -3.2\ \text{dB}$ for the 12 mm
spacer. The measured index climbs 10 dB per decade at low frequency, parallel
to the requirement, because a channel phase mismatch that is constant in
degrees buys exactly that slope; it flattens above 1 kHz where the mismatch of
a real chain starts growing with frequency instead. A vent resonance around
100 Hz costs 4 dB and drops that one band below the class 1 minimum, so the
whole chain is graded class 2.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import metrology

# A complete instrument with the common 12 mm spacer. The measured index is
# modelled from the physics behind Table 2: a residual phase mismatch φs reads
# as δpI0 = 10 lg(kd/φs), so a mismatch that is constant in degrees already
# climbs 10 dB per decade, and above 1 kHz the mismatch of a real chain grows
# with frequency, so the index levels off.
spacing = 0.012
freqs, _, _ = metrology.residual_index_limits("instrument", spacing=spacing)
phase_mismatch = 0.05 * np.maximum(1.0, freqs / 1000.0)          # degrees
measured = metrology.residual_index_from_phase_mismatch(phase_mismatch, freqs,
                                                        spacing)
measured = measured - 4.0 * np.exp(-((np.log(freqs / 100.0) / 0.25) ** 2))

res = metrology.intensity_class_compliance(measured, freqs, spacing=spacing)
res.plot()
plt.show()
```

</details>

Two companion rules of the standard come with it. Clause 6.1 fixes the
frequency range a class attests: 45 Hz to 7.1 kHz in one-third octaves, which
class 1 requires and class 2 may also use, or 45 Hz to 5.6 kHz in octaves,
which is offered to class 2 as an alternative. A verdict computed over fewer
bands is flagged `range_limited` so it cannot be read as a full-range claim.
The Spanish translation UNE-EN 61043:1999 records only the octave alternative
for class 2 and drops the one-third-octave one; the library follows the
EN/IEC text (see the [errata registry](../../ERRATA.md)). Clause 8 combines separately supplied components:
`instrument_class_from_components(probe_class, processor_class)` returns 1
only when both are class 1, and 2 for every other pairing.

The example fiche, regenerated with `make reports`, is kept rendered in the
repository. Click the preview to open the PDF:

[![One-page instrument-class-verification fiche: a metadata header, a per-band table listing the class 1 and class 2 minima, the measured residual index, the margin and the class achieved in each one-third-octave band from 50 Hz to 6.3 kHz, the measured index drawn as a step curve over the two Table 2 masks with the 100 Hz band ringed below the class 1 minimum, the boxed Class 2 - COMPLIES (binding margin +4.20 dB) result, the microphone separation and equivalent phase mismatch, and a FAIL verdict against the required class 1](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iec61043_intensity_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iec61043_intensity_example.pdf)

### Reading `δpI0` as a phase error

The requirement is really a phase-matching requirement in disguise. In an
axially propagating plane wave the *true* phase difference across the spacer
is $kd$, so a residual intensity produced by a channel mismatch $\varphi_s$
gives

$$
\delta_{pI0} = 10 \log_{10}\!\left( \frac{k d}{\varphi_s} \right)
$$

and the two conversions run both ways:

```python
from phonometry import metrology

# 20 dB of residual index at 1 kHz over a 25 mm spacer:
phi = metrology.phase_mismatch_from_residual_index(20.0, 1000.0, 0.025)
print(round(float(phi), 2))     # 0.26 degrees, a hundredth of kd

# And back, for a chain whose channels are matched to 0.05°:
print(round(float(metrology.residual_index_from_phase_mismatch(
    0.05, 1000.0, 0.012)), 1))  # 24.0 dB
```

That is why the low-frequency end of Table 2 rises 1 dB per third-octave band
and then flattens: below the knee the standard is asking for a *constant*
phase match, and above it for a constant index. The knee sits at 250 Hz for
the three class 1 columns and for the class 2 processor, and at 630 Hz for the
class 2 probe and the class 2 complete instrument, which is also where the
table stops stepping in whole decibels (the instrument goes 14 dB at 315 Hz,
14,5 dB at 400 Hz, 15 dB at 500 Hz). It is also why a
tenth of a degree of channel mismatch is a demanding specification, and why a
probe must be verified with the spacer it will actually be used with.

## Sound power at discrete points (ISO 9614-1)

Everything above qualifies a measurement. ISO 9614-1 is the part that turns it
into a sound power level, and it is the ordinary way of using a p-p probe:
hold it still at each of $N$ points, one per segment of a surface enclosing
the source, and read the signed normal intensity there. The partial power of a
segment and the sound power of the source are then (Formulae (11) and (12)):

$$
P_i = I_{\mathrm{n}i} \, S_i, \qquad
L_W = 10 \log_{10} \frac{\sum_{i=1}^{N} P_i}{P_0},
\qquad P_0 = 10^{-12}~\mathrm{W}
$$

Formula (12) prints without the absolute-value bars that the general
definition (Formula (8)) carries, because clause 9.2 disposes of the negative
case instead: **the method is not applicable to a band in which the sum is
negative**. One segment carrying inward flow is a different matter, and is
normal; it is what $F_3$ measures.

A.2.3 makes a second refusal, and on a different quantity: where
$\sum_i I_{\mathrm{n}i}$ is negative, "the test conditions do not satisfy the
requirements of this part of ISO 9614 in that frequency band". That sum is
unweighted over the $N$ positions, while clause 9.2 weights each segment by
its area, so equal segments make the two refusals agree and unequal ones let
them part company: a band clause 9.2 keeps, with a positive total power and a
finite level, can still be one A.2.3 refuses. Its Annex A indicators come back
`NaN` and the determination warns, because nothing else in the result flags
it.

Clause 8.2 sizes the position set: at least one position per square metre and
at least ten in all, distributed as evenly as the segment areas allow. Both
relaxations the clause offers (one position per 2 m² where extraneous noise is
significant, fifty positions over a surface larger than 50 m²) end at fifty
positions, so a sparser set than that is worth saying so about, and the
determination warns.

```python
import numpy as np
from phonometry import emission

# Measurement surface: a 2.0 m x 1.8 m x 1.4 m box over a reflecting floor,
# five faces and 14.24 m2 in all, sampled at 16 positions.
segment_areas = np.array([0.9] * 4 + [0.9333] * 6 + [0.84] * 6)
frequencies = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])

# The field: nearly uniform pressure, a reactivity that grows towards low
# frequency, and two positions of the 125 Hz band through which power flows
# back in.
rng = np.random.default_rng(9614)
reactivity = np.array([9.5, 7.0, 5.0, 4.0, 3.2, 3.0])
pressure_levels = 82.0 + rng.normal(0.0, 0.6, (16, 6))
band_mean = 10.0 ** ((82.0 - reactivity) / 10.0) * 1e-12
normal_intensity = band_mean * (1.0 + rng.normal(0.0, 0.22, (16, 6)))
normal_intensity[:2, 0] = -0.4 * band_mean[0]

power = emission.sound_power_intensity_points(
    normal_intensity,
    segment_areas,
    pressure_levels=pressure_levels,
    pressure_residual_index=19.0,
    frequencies=frequencies,
    band_type="octave",
)
print(np.round(power.sound_power_level, 1))  # [83.3 86.8 88.1 89.6 90.4 90.4]
print(round(power.sound_power_level_a, 1))   # 96.1 dB(A)
print(round(power.surface_area, 2), power.positions)   # 14.24 16
```

### The sign is in the print, not in the number

ISO 9614-1 never prints a signed level. A normal intensity level is written
`XX dB` when the flow through the segment is outward and `(-) XX dB` when it is
inward, with `XX` a positive number in both cases (clause 3.5, and the two
unnumbered equations of clauses 9.1 and A.2.3):

$$
I_{\mathrm{n}i} = I_0 \times 10^{XX/10}, \qquad
I_{\mathrm{n}i} = -I_0 \times 10^{XX/10}, \qquad
I_0 = 10^{-12}~\mathrm{W/m^2}
$$

So a caller reading a printed table has to carry the sign separately, which is
what the `negative` argument is for. It broadcasts, so one position of a
surface can flow inward while the rest flow outward:

```python
inward = emission.normal_intensity_from_levels(66.0, negative=True)
print(f"{inward:.3e}")                       # -3.981e-06 W/m2, flowing in
```

### Which grade the determination reaches

Annex B numbers **two** criteria, not three. Figure B.1 gates the
determination on four questions, in this order, and Table B.3 says what to
change when each one fails:

| Gate | Where | Action when it fails |
| --- | --- | --- |
| $F_1 \le 0.6$ | Table B.3 | (e) |
| $L_\mathrm{d} > F_2$ (criterion 1) | Formula (B.1) | (a) or (b) |
| $F_3 - F_2 \le 3$ dB | Figure B.1 | (a) or (b) |
| $N > C F_4^2$ (criterion 2) | Formula (B.2) | (c) or (d) |

Only the first failing gate is acted on: every action box in Figure B.1
returns to the next measurement rather than to the gate below it, so
`required_actions()` reports one action set per band and stops there. Two
codes mean the standard offers a choice, which is how Table B.3 prints them
("a **or** b").

```python
print(power.dynamic_capability_index[0])     # 9.0 dB (Ld = 19 - 10)
print(np.round(power.f2, 1))                 # [9.6 6.8 5.3 3.9 3.1 3.3]
print(power.criterion_1)      # [False  True  True  True  True  True]
print(list(power.achieved_grade))
# ['none', 'precision', 'precision', 'precision', 'precision', 'precision']
codes = ["".join(a.value for a in band) for band in power.required_actions()]
print(codes)                                 # ['ab', '', '', '', '', '']
print(power.required_actions()[0][0].criterion)  # F2 > Ld or (F3 - F2) > 3 dB
```

The 125 Hz band is the interesting one. Its $F_2$ of 9.6 dB sits just above the
instrument's dynamic capability of 9 dB, so criterion 1 fails there and no
amount of averaging will fix it: Table B.3 offers moving the surface (a) or
shielding it (b). Clause 10.5 b) then omits that band from the A-weighted
determination and asks for the omission to be stated, which
`a_weighting_omitted_bands` is:

```python
print(power.a_weighting_omitted_bands)  # [ True False False False False False]
```

### Grade 3 is an A-weighted determination

Table B.2 tabulates the criterion-2 factor $C$ band by band for grades 1 and 2
and gives grade 3 one A-weighted value, 8, and no band column at all. Table
B.1 does the same with the error factor $\Delta$ (0.20 and 0.29 for all bands,
0.60 A-weighted) and Table 2 with the standard deviation $s$ of the
determination. Three tables agree, so this is the standard's design rather
than a gap in it: **a per-band determination reaches grade 1 or grade 2, and
grade 3 is reached, if at all, by the A-weighted sum**. Asking for a per-band
grade-3 figure raises rather than returning a plausible number.

```python
print(emission.position_count_factor("engineering", 1000.0,
                                     band_type="octave"))   # 29.0
print(emission.position_count_factor("survey"))             # 8.0 (A-weighted)
print(power.achieved_grade_a, round(power.field_nonuniformity_a, 2))
# precision 0.13
```

The A-weighted determination has a field non-uniformity of its own: B.1.2 sums
the A-weighted band intensities of each position into one intensity per
position and applies Formulae (A.8) and (A.9) to those, which is
`field_nonuniformity_a`.

The uncertainty of the result is Table 2's $s$, with footnote 1 placing the
true level within $\pm 2s$ of the measured one at 95 % confidence. Clause 10.6
says which row of the table to read it in: **the grade achieved** in the final
test, not the grade the determination set out for, so `expanded_uncertainty`
follows `achieved_grade` band by band. Five bands here reached grade 1 and
carry its figures; 125 Hz reached no grade, Table 2 prints no $s$ for that, and
what clause 10.5 c) offers such a band instead is the 95 % confidence interval
of Formula (B.3), which is also what accompanies a band recorded after failing
criterion 2:

```python
print(power.expanded_uncertainty)          # [nan  3.  3.  2.  2.  2.] dB
print(np.round(power.confidence_interval[0], 1))     # [-1.7  1.2] dB
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/discrete_point_qualification_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/discrete_point_qualification.svg" alt="Two stacked panels over six octave bands from 125 Hz to 4 kHz for a sixteen-position survey. Above, the positions criterion 2 requires, C times F4 squared, as a pair of bars per band for grade 1 and grade 2, against a dashed line at the sixteen positions measured: the requirement grows with frequency, the grade-1 bar passes the line at 2 kHz and reaches 30 at 4 kHz, the grade-2 bar stays under it in every band, and the 125 Hz pair is greyed out because criterion 1 already failed there. A chip over each band names the grade reached: no grade, grade 1, grade 1, grade 1, grade 2, grade 2. Below, the 95 per cent confidence interval of Formula B.3 band by band, widening from about plus or minus half a decibel at 125 Hz to plus 1.3 and minus 1.9 decibels at 4 kHz" width="92%"></picture>

*The grade is a position budget, and it is spent band by band. The survey
drawn here is a second one, of a machine that radiates unevenly, and one
quantity sets both of its panels: the field non-uniformity $F_4$ decides how
many positions criterion 2 asks for, and how wide Formula (B.3) opens the
interval. Its field grows less uniform with frequency, as a source whose
radiation turns directional will, so the top two bands ask more than the
sixteen positions measured at grade 1 and settle for grade 2, while 125 Hz
never reaches criterion 2 at all: its $F_2$ is already above the instrument's
$L_\mathrm{d}$, and Figure B.1 sends it to an action box instead. Grade 3 is on
neither panel, because Table B.2 gives it no per-band $C$ to draw.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import emission

# The figure's own survey, which is not the example above: sixteen positions
# over a 1.2 m x 0.8 m x 1.0 m machine on a reflecting floor, measured at the
# 0.5 m the standard asks for, so the box is 2.2 m x 1.8 m x 1.5 m and
# 15.96 m2. The field is written from its two indicators rather than drawn at
# random. F2 widens towards low frequency as the field turns reactive; F4
# widens towards high frequency as the radiation grows directional and the
# flow concentrates on part of the surface, which is the growth criterion 2 is
# about.
freqs = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
areas = np.concatenate([np.full(4, 3.96 / 4),    # top, 2.2 m x 1.8 m
                        np.full(6, 3.3 / 3),     # the two long sides
                        np.full(6, 2.7 / 3)])    # the two ends
surface_pressure = 84.0
nonuniformity = np.array([0.25, 0.30, 0.38, 0.45, 0.58, 0.72])   # target F4
pressure_intensity = np.array([11.5, 8.0, 6.0, 4.5, 3.5, 3.0])   # target F2
rng = np.random.default_rng(9614)
spread = rng.normal(0.0, 1.0, (areas.size, freqs.size))
spread = (spread - spread.mean(axis=0)) / spread.std(axis=0, ddof=1)
mean_intensity = 10.0 ** ((surface_pressure - pressure_intensity) / 10.0) * 1e-12
normal_intensity = mean_intensity * (1.0 + nonuniformity * spread)

survey = emission.sound_power_intensity_points(
    normal_intensity, areas,
    pressure_levels=np.full((areas.size, freqs.size), surface_pressure),
    pressure_residual_index=20.0,    # Ld = 20 - 10 = 10 dB (Table 1, octaves)
    frequencies=freqs, band_type="octave",
)

# One line — the band sound power spectrum, with the bands outside the method
# hatched and the A-weighted total in the title:
survey.plot()
plt.show()

# The qualification figure above, by hand, from the fields the result carries
# and the Table B.2 factor the module exposes.
bands = np.arange(freqs.size)
required = {
    grade: np.array([emission.position_count_factor(grade, float(f),
                                                    band_type="octave")
                     for f in freqs]) * survey.f4 ** 2
    for grade in ("precision", "engineering")
}
fig, (axt, axb) = plt.subplots(2, 1, sharex=True)
for offset, grade, color in ((-0.18, "precision", "#1f77b4"),
                             (+0.18, "engineering", "#2ca02c")):
    axt.bar(bands + offset, required[grade], width=0.36, color=color)
axt.axhline(survey.positions, color="#d62728", linestyle="--")
axt.set_ylabel("Positions required, $C F_4^2$")
interval = survey.confidence_interval
axb.errorbar(bands, np.zeros_like(bands, dtype=float),
             yerr=np.vstack([-interval[:, 0], interval[:, 1]]), fmt="o", capsize=6)
axb.set_ylabel("95 % interval [dB]")
axb.set_xticks(bands)
axb.set_xticklabels([f"{f:g}" for f in freqs])
axb.set_xlabel("Frequency [Hz]")
plt.show()
```

</details>

### Spending fewer new positions when the power is concentrated

Where criterion 1 holds, criterion 2 does not and $F_3 - F_2 \le 1$ dB, little
power is flowing inward, so most of it may be leaving through a minority of
the segments. Clause 8.3.2 and B.1.3 make that testable: rank the positive
partial powers, take the top segments until more than half the total power is
accounted for, and require that subset to be fewer than half the segments.
Then evaluate the field non-uniformity separately over the subset and over the
remainder and size the new positions with Formula (B.4):

$$
N^* \ge 4 \left[ \frac{F_4(\alpha)}{\Delta_\alpha} \right]^2, \qquad
\Delta_\alpha = \frac{1}{\alpha} \left[ \Delta - (1 - \alpha)
\frac{2}{\sqrt{N_{1-\alpha}}} F_4(1 - \alpha) \right]
$$

$\Delta_\alpha$ is the share of the Table B.1 error budget left for the subset
once the remainder, measured at its existing density, has taken its own, so a
remainder too non-uniform to leave anything over exhausts the budget and the
procedure cannot help. That case is refused, and so are three others: no
subset satisfying B.1.3's two conditions, a subset of a single segment (over
one position Formula (A.8) has no spread, so Formula (B.4) has no $F_4(\alpha)$
to square), and a remainder whose own algebraic mean is not positive. Each
refusal says which of the four it is. None of them names a row of Table B.3:
where the subset cannot be found, B.1.3 asks for "alternative appropriate
actions to increase the accuracy of sound power determination according to
table B.3", and where the selective modification cannot be implemented, clause
8.3.2 asks for "alternative appropriate action according to B.2 and table
B.3". Which row applies also turns on criterion 2 and on $F_3 - F_2$, and
`partial_power_concentration` is given neither.

```python
# One band, twelve segments, most of the power through the first four.
concentrated = np.array([3.0e-5, 2.4e-5, 1.8e-5, 1.4e-5] + [8.0e-6] * 8)
subset = emission.partial_power_concentration(concentrated, np.full(12, 1.0))
print(subset.subset_positions, round(subset.power_fraction, 2))   # 4 0.57
print(subset.additional_positions)                                # 2
```

### Reactive fields near sources

Close to a source the field turns **reactive**: pressure and particle
velocity drift toward quadrature, so a large pressure carries little net
flow. For a small source the quadrature component grows as $1/(kr)$; at
100 Hz and 0.25 m from the source it is already about twice the active one,
and $\delta_{pI}$ climbs just as it does in the standing wave of the figure
above.
The same happens between a machine and a hard reflecting surface, and in
reverberant rooms where the diffuse field raises pressure without
transporting energy outward. This is why ISO 9614-1 keeps the measurement
surface on average more than 0.5 m away from the source, and why, when a
scan fails the dynamic-capability criterion, moving the surface outward or
adding absorption to the room usually lowers $F_2$ below $L_\mathrm{d}$ more cheaply
than better hardware.

### Choosing the spacer

The spacer sets both ends of the usable band, in opposite directions:

- **The top end is geometry.** The finite difference underestimates the
  gradient by $\sin(k\Delta r)/(k\Delta r)$, so the ceiling scales as
  $1/\Delta r$: `max_valid_frequency` $\approx 0.1\,c/\Delta r$ keeps the
  bias within about 0.3 dB,
  giving roughly 5.7 kHz for a 6 mm spacer, 2.9 kHz for 12 mm and 690 Hz
  for 50 mm (`bias_correct=True` undoes the known bias somewhat beyond
  that).
- **The bottom end is phase.** A progressive wave puts only
  $360\,f\,\Delta r/c$ degrees of true phase across the spacer, 0.8° at 63 Hz
  over 12 mm, while the channel mismatch stays fixed. Lowering the frequency
  shrinks the signal, not the error, so the margin over $\delta_{pI0}$
  collapses at low frequency. A larger spacer buys back that margin: the
  IEC 61043 residual-index requirements scale as
  $10\log_{10}(\Delta r/25\ \text{mm})$, so doubling the spacer is worth 3 dB of
  low-frequency margin.

No single spacer covers the full audio range: 6 mm suits high-frequency
work, 50 mm low-frequency work, and the common 12 mm covers the mid band.
Wide-band surveys are measured twice with two spacers and the band results
merged; whichever spacer is fitted, verify $\delta_{pI0}$ with that spacer in
place, since the index belongs to the probe-spacer-analyzer chain, not to
the microphones alone.

### `sound_intensity()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `p1`, `p2` | 1D arrays | Pa | equal length | Microphone closer to the source first; reversing them flips the sign |
| `fs` | int | Hz | > 0 | |
| `spacing` | float | m | > 0 | Microphone separation $\Delta r$ (typ. 6/12/50 mm) |
| `rho` | float | kg/m³ | default `1.204` | Air density |
| `c` | float | m/s | default `343.0` | Speed of sound (bias/validity estimates) |
| `fraction` | int, optional | — | `1`, `3` or `None` (default) | Octave/third-octave band integration |
| `limits` | list, optional | Hz | default library band range | Band analysis limits |
| `bias_correct` | bool | — | default `False` | Apply the per-bin $(k\Delta r)/\sin(k\Delta r)$ correction (IEC 61043 §7.3) before summing, so band/broadband totals stop under-reading as $f \to$ `max_valid_frequency`; bins past the first null are left uncorrected. The per-band `bias_correction` factor is reported either way |

See [Theory](../../reference/theory/signal-analysis.md) for the derivations and [Calibration](../../signals/metrology/calibration.md)
for absolute scaling of the two channels.

## See also

- [Sound Power by Intensity Scanning](sound-power-intensity.md) — the ISO 9614-2/-3 routes that consume $\delta_{pI}$, the field indicators and the dynamic capability.
- [Sound Power](sound-power.md) — choosing between the intensity, pressure and reverberation-room routes.
- [Sound power from surface vibration](vibration-sound-power.md) — the ISO/TS 7849-2 radiation factor needs an intensity-measured power.
- [Calibration](../../signals/metrology/calibration.md) — the absolute scaling of the two channels the estimator assumes.
- [Theory: signal analysis](../../reference/theory/signal-analysis.md) — the cross-spectral derivation behind $I(f) = -\mathrm{Im}\{G_{12}\}/(2\pi f \rho_0 \Delta r)$.
- API reference: [`emission.intensity`](https://jmrplens.github.io/phonometry/reference/api/power/intensity/).
- Theory: [Sound intensity (IEC 61043)](../../reference/theory/signal-analysis.md#sound-intensity-iec-61043): the finite-difference approximation behind a p-p probe and the errors it commits.

## References

- Fahy, F. J. (1995). *Sound intensity* (2nd ed.). E&FN Spon.
  ISBN 978-0-419-19810-9.
  [doi:10.4324/9780203475386](https://doi.org/10.4324/9780203475386).
  The monograph on the subject: active and reactive intensity, the p-p
  estimator and the phase-mismatch error budget behind this page.
- International Electrotechnical Commission. (1993). *Electroacoustics —
  Instruments for the measurement of sound intensity — Measurements with
  pairs of pressure sensing microphones* (IEC 61043:1993; adopted in Europe
  as EN 61043:1994).
  [IEC webstore](https://webstore.iec.ch/en/publication/4353).
  The instrument standard: the cross-spectral estimator, the
  residual-intensity test behind $\delta_{pI0}$, the Table 2 minima per band
  for probes, processors and instruments with the $+10\log_{10}(x/25)$ separation
  rule
  (Note 1), the frequency ranges of clause 6.1 and the component-combination
  rule of clause 8.
- International Organization for Standardization. (1993). *Acoustics —
  Determination of sound power levels of noise sources using sound
  intensity — Part 1: Measurement at discrete points* (ISO 9614-1:1993).
  [iso.org catalogue](https://www.iso.org/standard/17427.html).
  The field indicators $F_1$ to $F_4$, the dynamic-capability criterion and
  the 0.5 m surface-distance rule.

## Standards

IEC 61043:1993 (EN 61043:1994), *Electroacoustics —
Instruments for the measurement of sound intensity — Measurements with pairs
of pressure sensing microphones*: the two-microphone cross-spectral
intensity estimator, the finite-difference bias correction and the
usable-bandwidth bound (clause 7.3, Table 3), the minimum pressure-residual
intensity index per band for probes, processors and instruments in class 1
and class 2 with its separation rule (Table 2 and its Note 1), the processor
frequency ranges (clause 6.1) and the class of an instrument assembled from
separate components (clause 8). ISO 9614-1:1993, *Acoustics — Determination
of sound power levels of noise sources using sound intensity — Part 1:
Measurement at discrete points*: the pressure-intensity index, the Annex A
field indicators $F_1$ (equations (A.1)–(A.2), evaluated in the initial test
of clause 8.2 and again per Annex B, B.1.4), $F_2$, $F_3$ and $F_4$, the
Table B.3
temporal-variability limit and the dynamic-capability criterion (Annex B);
and the discrete-point sound-power determination of clauses 8 and 9, which
sums the partial powers $I_{\mathrm{n}i} S_i$ into $L_W$ (equations (11) and
(12)) and qualifies it against Annex B: criterion 1, the inward-flow gate of
Figure B.1 and criterion 2 with the Table B.2 factor $C$, the Table B.3
actions, the equation (B.3) confidence interval, the Table 2 uncertainty and
the optional concentration procedure of clause 8.3.2.

**Not covered.** The residual-intensity *test* of IEC 61043 is not performed
here: $\delta_{pI0}$ is a
value the caller measures on their own probe-and-analyser chain, with the
spacer that will be fitted, and supplies — what the library does with the
number is grade it against Table 2. The before-use check of clause 14 and the
ISO 9614-2 probe-reversal test are procedures, not functions.

