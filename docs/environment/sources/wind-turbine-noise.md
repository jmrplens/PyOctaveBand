← [Documentation index](../../README.md)

# Wind-turbine noise: apparent sound power and tonal audibility (IEC 61400-11)

IEC 61400-11 measures the acoustic emission of a wind turbine. This page covers
its two closed-form quantities: the **apparent sound power level** referred to
an equivalent point source at the rotor centre, and the **tonal audibility**
that decides whether a discrete tone (blade-passing, gearbox, generator) is
audible above the masking noise. Each is validated against the standard's
formulas and a hand-derived synthetic-tone oracle.

## 1. Apparent sound power level

With the reference microphone on a ground board at the horizontal distance
$R_0 = H + D/2$ (hub height $H$, rotor diameter $D$), the slant distance to
the rotor centre is $R_1 = \sqrt{H^2 + R_0^2}$ and the per-band apparent sound
power level is

$$
L_{WA,i} = L_{p,i} - 6 + 10\log_{10}\frac{4\pi R_1^2}{S_0}, \qquad S_0 = 1\ \mathrm{m}^2,
$$

energy-summed over bands (Formula 27). The −6 dB accounts for the ground-board
pressure doubling.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_wind_turbine_iec61400_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_wind_turbine_iec61400.svg" alt="Side view of a horizontal-axis wind turbine with hub height H and rotor diameter D, a microphone lying on a flat ground board downwind at the horizontal distance R0 = H + D/2 from the tower centreline, the slant distance R1 from the rotor centre to the microphone with the board inclination angle phi between 25 and 40 degrees, and a met mast measuring wind speed and direction; a plan-view inset shows the Figure 3 pattern with the reference position downwind and three optional positions at plus and minus 60 degrees and upwind, and the annotations give R1 equals the square root of H squared plus R0 squared and the apparent sound power formula LWA,i = Lp,i minus 6 plus 10 lg(4 pi R1 squared over S0)" width="94%"></picture>

```python
from phonometry import environment

# Background-corrected A-weighted one-third-octave band levels L_p,i (dB).
band_levels = [55.0, 58.0, 60.0, 57.0, 54.0]
r1 = environment.slant_distance(hub_height=80.0, rotor_diameter=100.0)
lwa = environment.apparent_sound_power_level(band_levels, r1)   # dB re 1 pW
```

### Why "apparent"

$L_{WA}$ is written like a sound power level, but it is not one in the
ISO 3744 sense of sampling the pressure field over an enveloping
surface. The
standard collapses the whole machine into an equivalent point source at the
rotor centre and asks what power that source would need, radiating
spherically, to reproduce the measured level at one downwind ground-board
position: by definition it is the power "giving the same sound emission in
the downwind direction as the wind turbine". Everything a 150 m rotor does
that a point source does not, the vertical and lateral directivity and the
blade-passing swish, is folded into the number and evaluated in a
single direction; the optional positions 2 to 4 of the plan-view pattern
exist precisely to document how the emission varies around the machine.
Apparent sound powers of different turbines are comparable because the
geometry scales with the machine ($R_0 = H + D/2$, so every rotor is seen
under a similar angle), which is the point of the definition, but an
$L_{WA}$ fed into an ISO 9613-2 prediction carries its built-in downwind
bias with it. The ground board, in turn, is why the formula subtracts 6 dB:
a capsule lying on a hard plate receives a perfectly coherent reflection
(pressure doubling, $+6$ dB) instead of the uncontrolled height-dependent
interference pattern a tripod microphone would sample (see
[the image source behind the ground effect](../propagation/outdoor-propagation.md)).

### Wind-speed bins and standardized conditions

A turbine's noise emission rises with wind speed toward rated power, so a
single number would be meaningless without its operating point: IEC 61400-11
reports $L_{WA}$ *as a function of wind speed*. Sound and wind are logged in
synchronized 10 s averages, and every period is sorted into a wind-speed
**bin 0.5 m/s wide centred on integer and half-integer hub-height wind
speeds**, with at least 10 periods of total noise and 10 of background
(turbine parked) per bin. The hub-height wind speed itself is preferably not
an anemometer reading at all: it is derived from the measured electric power
through the turbine's power curve (Clause 8.2.1), the most repeatable proxy
for the wind the rotor actually sees, with the nacelle anemometer and a met
mast as fallbacks. The measured range must at least cover 0.8 to 1.3 times
the wind speed at 85 % of maximum power (roughly 6 to 10 m/s at 10 m height
for a large machine). Within each bin the spectra are averaged, interpolated
to the bin centre and background-corrected; a total-minus-background margin
of 3 dB or less voids the bin, between 3 and 6 dB flags it with an asterisk. For
comparability with consent conditions and older editions, Formula (29) also
maps each result to the wind speed at 10 m height over a **reference
roughness length** $z_{0ref} = 0.05$ m (a logarithmic wind profile), giving
$L_{WA,10m}$ at integer 10 m wind speeds regardless of the site's actual
terrain. The library implements the closed-form quantities of this pipeline
(slant distance, per-band apparent power, tonal audibility); the binning,
averaging and uncertainty machinery operates on whole measurement campaigns
and stays out of scope. So does the IEC TS 61400-14 declaration route, which
turns a batch of measured machines into the declared value a planning authority
receives. And two things a wind-turbine reader often arrives looking for sit
outside IEC 61400-11 altogether: **amplitude modulation** (the swish, folded
into $L_{WA}$ and rated nowhere in this standard) and **infrasound** — optional
measurements under 7.2.1 with no rating method attached, so their assessment
falls to national guidance.

## 2. Tonal audibility

From a narrowband spectrum (1–2 Hz resolution), the lines in the **critical
band** about the tone,

$$
\mathrm{CBW} = 25 + 75\,[\,1 + 1.4\,(f_c/1000)^2\,]^{0.69}\ \mathrm{Hz},
$$

are classified into masking noise and tone lines (the 70 %-lowest energy mean,
the +6 dB criteria; tone lines must additionally lie within 10 dB of the
*highest* line above the threshold, and that highest line is the frequency of
the tone, subclauses 9.5.3/9.5.4). The candidate itself must first pass the
9.5.2 *possible tone* screening: a local maximum more than 6 dB above the
band energy average excluding the maximum and its adjacent lines. The
masking-noise level $L_{pn}$ follows Formula 31, the tonality is
$\Delta L_{tn} = L_{pt} - L_{pn}$, and the tonal audibility is

$$
\Delta L_a = \Delta L_{tn} - L_a, \qquad L_a = -2 - \log_{10}\!\big[1 + (f/502)^{2.5}\big],
$$

reported when $\Delta L_a \ge -3\ \text{dB}$; a tone is audible when
$\Delta L_a > 0$.

For a candidate between 20 and 70 Hz the Zwicker band above is not what the
standard uses: subclause 9.5.3 substitutes the fixed absolute 20–120 Hz band,
so `critical_bandwidth` comes back as 100 Hz for every such candidate and the
band is *not centred on the tone* — a low-frequency result cannot be reconciled
with the CBW formula printed above. Blade-passing harmonics and low-speed
gearbox tones live exactly there, so for a wind turbine this is no corner case.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/wind_turbine_tonality_dark.svg">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/wind_turbine_tonality.svg" alt="A wind-turbine narrowband spectrum with a discrete tone near 200 Hz standing above a shaped broadband floor, the critical band about the tone shaded, the masking-noise level drawn as a horizontal line, and the tonal audibility annotated" width="82%">
</picture>

<details>
<summary>Show the code for this figure</summary>

```python
import numpy as np
from phonometry import environment

df = 2.0
freqs = np.arange(50.0, 400.0 + df, df)
levels = 42.0 - 6.0 * np.log10(freqs / 100.0)
levels[int(np.argmin(np.abs(freqs - 200.0)))] += 22.0   # blade-passing-style tone
environment.wind_turbine_tonality(levels, freqs, tone_frequency=200.0).plot()
```

</details>

```python
import numpy as np
from phonometry import environment

# A uniformly-spaced narrowband spectrum (2 Hz resolution): a flat 30 dB floor
# with a discrete 60 dB tone at 500 Hz.
frequencies = np.arange(440.0, 562.0, 2.0)
levels = np.full(frequencies.size, 30.0)
levels[np.argmin(np.abs(frequencies - 500.0))] = 60.0

res = environment.wind_turbine_tonality(levels, frequencies)
print(res.tone_frequency, res.tonality, res.tonal_audibility, res.is_audible)
res.plot()   # spectrum + critical band + masking level (needs matplotlib)
```

`wind_turbine_tonality` returns a `WindTurbineTonalityResult` with the
`critical_bandwidth`, `tone_level`, `masking_level`, `tonality`,
`audibility_criterion`, `tonal_audibility`, `is_audible` and
`has_identified_tone`. When the candidate fails the 9.5.2 screening or no
line classifies as "tone", `has_identified_tone` is `False`: the numeric
fields are non-standard fallbacks and such spectra must be **excluded** from
the 9.5.1 energy averaging of $\Delta L_a$ over the spectra of a wind-speed bin
(`is_audible` also requires an identified tone). The tone frequency and the
$L_a$ criterion anchor to the highest classified tone line, not the probed
candidate. The audibility formula is the ISO 1996-2 Annex C one; what is
specific to IEC 61400-11 is the determination of the tone and masking levels
and the Zwicker critical band from the spectrum. IEC 61400-11 itself stops at
$\Delta L_a$ and prescribes no rating adjustment. The 9.5.1 energy average over
the spectra of a wind-speed bin is a *mean* audibility, the quantity
ISO 1996-2:2017 Table J.1 was written for
(`tonal_adjustment_from_mean_audibility`, integer 0–6 dB); the piecewise
(C.4)–(C.6) law of `tonal_adjustment` applies to a single spectrum's
$\Delta L_{ta}$, and the two differ by 1 to 2 dB on the same input.

### Assessment report (`.report()`)

Tonal audibility is the part of a wind-turbine assessment that ends in a
document handed to a regulator, so the result renders one.
`WindTurbineTonalityResult.report(path)` writes a one-page fiche following
IEC 61400-11:2012+A1:2018 subclauses 9.5.2 to 9.5.8: a standard-basis line, an
optional metadata header (source/situation, client, measurement position,
instrumentation, date), the critical-band analysis table — tone frequency,
critical bandwidth, tone level $L_{pt}$, masking-noise level $L_{pn}$, tonality
$\Delta L_{tn}$, audibility criterion $L_a$ and tonal audibility $\Delta L_a$ —
beside the narrowband spectrum with the critical band, the masking level and
the tone marked, then the boxed $\Delta L_a$ with the tone frequency and the
audibility decision, an optional PASS/FAIL row, a note on how $\Delta L_a$ is
built, and the fixed disclaimer in the footer.

A supplied `requirement` is the **maximum acceptable** tonal audibility in dB,
so a *less* audible tone passes. Rendering needs reportlab and, for the
embedded figure, matplotlib (`pip install "phonometry[report,plot]"`); only
`engine="reportlab"` is supported, and `language="es"` renders a Spanish fiche.

```python
from phonometry import ReportMetadata

# The 500 Hz gearbox tone over the flat 30 dB floor of the snippet above.
res.report(
    "tonality_fiche.pdf",
    metadata=ReportMetadata(
        specimen="Horizontal-axis wind turbine, gearbox tone",
        measurement_standard="IEC 61400-11",
        laboratory="Phonometry Reference Laboratory",
        requirement=6.0,            # maximum acceptable tonal audibility (dB)
    ),
)                                   # tonal audibility (dB) and the decision
```

[![IEC 61400-11 wind-turbine tonal audibility example report: a metadata header, a critical-band analysis table (tone frequency 500.0 Hz, critical bandwidth 117.3 Hz, tone level Lpt = 60.0 dB, masking-noise level Lpn = 45.9 dB, tonality dLtn = 14.1 dB, audibility criterion La = -2.3 dB) beside the narrowband-spectrum plot with the critical band shaded and the masking level drawn, the boxed tonal audibility dLa = 16.4 dB at the 500.0 Hz tone with the decision that the tone is audible, and a FAIL verdict against a maximum acceptable audibility of 6.0 dB](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iec61400_wind_turbine_tonality_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iec61400_wind_turbine_tonality_example.pdf)

*Wind-turbine tonal-audibility fiche (`WindTurbineTonalityResult.report`): a
30 dB masking floor under a 60 dB tone leaves $\Delta L_a$ = 16.4 dB, far past
the 0 dB audibility line and past a 6 dB acceptance requirement.*

## See also

- [Environmental noise levels](../assessment/environmental-levels.md): the ISO 1996-2 rating adjustment this page's $\Delta L_a$ feeds.
- [Tone audibility](../../perception/psychoacoustics/tone-audibility.md): the same tonal-audibility idea outside the wind-turbine context.
- [Outdoor Sound Propagation](../propagation/outdoor-propagation.md): the chain that carries $L_{WA}$ from the rotor to a dwelling.
- API reference: [`environment.sources.wind_turbine`](https://jmrplens.github.io/phonometry/reference/api/environment/wind-turbine/).

## References

- International Electrotechnical Commission. (2018). *Wind turbines —
  Part 11: Acoustic noise measurement techniques* (IEC 61400-11:2012+AMD1:2018
  CSV). [IEC webstore](https://webstore.iec.ch/en/publication/63367).
  The implemented edition: the measurement geometry of section 1, the
  wind-speed binning and the tonal-audibility chain of section 2.
- International Electrotechnical Commission. (2005). *Wind turbines —
  Part 14: Declaration of apparent sound power level and tonality values*
  (IEC TS 61400-14:2005).
  [IEC webstore](https://webstore.iec.ch/en/publication/5432).
  How a manufacturer turns IEC 61400-11 measurements of a batch of turbines
  into declared values with a stated uncertainty, the number a planning
  authority actually receives.
- International Organization for Standardization. (2017). *Acoustics —
  Description, measurement and assessment of environmental noise — Part 2:
  Determination of sound pressure levels* (ISO 1996-2:2017).
  [iso.org catalogue](https://www.iso.org/standard/59766.html).
  The tonal-audibility criterion the IEC method reuses comes from the
  Annex C of its 2007 edition (the 2017 edition carries the tonal methods
  in Annexes J and K), and its Table J.1 mapping,
  `tonal_adjustment_from_mean_audibility`, consumes the bin-mean audibility.

## Standards

IEC 61400-11:2012+A1:2018, *Wind turbines — Part 11: Acoustic
noise measurement techniques*: the apparent sound power level (Formula 26), the
critical bandwidth (Formula 30), the masking-noise level (Formula 31) and the
tonality/audibility (Formulae 32–34). The tonal-audibility formula coincides
with ISO 1996-2 Annex C. The full measurement pipeline (wind-speed binning,
regression to standardised speeds and the uncertainty budgets) is out of scope
here; these are the underlying closed forms.
