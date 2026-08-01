← [Documentation index](README.md)

# Sound Power in the Reverberation Room (ISO 3741)

A reverberation room turns its main defect as a listening space into a
measuring instrument: because the field is diffuse, a handful of microphone
positions sample the whole radiated energy, and the sound power follows from
the mean room level and the room's absorption with no enveloping surface to
build. That is why ISO 3741 is a grade-1 (precision) method, and why it is
the laboratory route of choice for steady, broadband sources small enough to
travel to a qualified room. This guide covers the direct method through the
Sabine absorption area, the Waterhouse and meteorological corrections, the
comparison method against a reference sound source, the qualification
warnings and the accredited-style test fiche. Which route fits which job,
and the pressure and intensity alternatives, are weighed in
[Sound Power](sound-power.md).

## 1. Reverberation room, precision grade (ISO 3741)

In a qualified hard-walled **reverberation room** the field is diffuse, so a
handful of microphones sample the whole radiated energy and the method
reaches grade 1. The sound power comes from the mean room level
$L_p(\text{ST})$, the Sabine absorption area
$A = (55.26/c)\cdot(V/T_{60})$ and a chain of small corrections
(ISO 3741 Eq. 20):

$$
L_W = \bar{L}_p + 10 \log_{10}\frac{A}{A_0} + 4.34\ \frac{A}{S}
      + 10 \log_{10}\left( 1 + \frac{S c}{8 V f} \right) + C_1 + C_2 - 6 .
$$

The bracketed term is the **Waterhouse correction**: near the room
boundaries the sound energy density is higher than in the interior, and
this term (which vanishes as frequency grows) restores the energy the
interior microphones miss. $C_1$ (reference-quantity) and $C_2$
(radiation-impedance) carry the result to the reference meteorological
conditions of 23 °C and 101.325 kPa,

$$
C_1 = -10 \log_{10}\frac{p_s}{p_{s0}} + 5 \log_{10}\frac{273.15 + \theta}{314}, \qquad
C_2 = -10 \log_{10}\frac{p_s}{p_{s0}} + 15 \log_{10}\frac{273.15 + \theta}{296},
$$

with the speed of sound $c = 20.05\sqrt{273 + \theta}$. The
**comparison method** replaces the absorption-area, Waterhouse and $C_1$
terms by a reference sound source of known power $L_W(\text{RSS})$ measured
in the same room, so the room need not be characterised:
$L_W = L_W(\text{RSS}) + (L_p(\text{ST}) - L_p(\text{RSS}) + C_2)$.

```python
import numpy as np
from phonometry import emission

# One-third-octave mean room SPL (dB), 100 Hz - 10 kHz, and the room's T60.
freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
                  1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000],
                 dtype=float)
lp = np.linspace(80.0, 70.0, freqs.size)
t60 = np.full(freqs.size, 2.0)

rev = emission.sound_power_reverberation(
    lp, t60, volume=200.0, surface_area=220.0, frequencies=freqs,
    temperature=20.0, static_pressure=101.0,
)
print(round(rev.speed_of_sound, 1))                     # c = 343.2 m/s
print(round(float(rev.absorption_area[0]), 1))          # A = 16.1 m^2 at 100 Hz
print(round(float(rev.waterhouse_correction[0]), 2))    # 1.68 dB at 100 Hz
print(round(float(rev.sound_power_level[0]), 1))        # LW = 87.9 dB
print(round(rev.sound_power_level_a, 1))                # LWA = 92.1 dB

# Comparison method: a reference source of known LW measured at the same spots.
lw_rss = np.full(freqs.size, 85.0)
lp_rss = np.linspace(78.0, 69.0, freqs.size)
cmp = emission.sound_power_comparison(lp, lp_rss, lw_rss, frequencies=freqs, temperature=20.0)
print(round(float(cmp.sound_power_level[0]), 1), cmp.method)   # 86.9 comparison

rev.plot()   # reverberation-room LW spectrum, LWA in the title (needs matplotlib)
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/sound_power_reverberation_result_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/sound_power_reverberation_result.svg" alt="The reverberation-room sound power level spectrum of the ISO 3741 example, one bar per one-third-octave band from 100 Hz to 10 kHz falling gently with frequency, with the A-weighted total of 92.1 dB(A) in the title" width="88%"></picture>

*The mean room level carried through the absorption-area, Waterhouse and
meteorological terms of Eq. 20 gives the one-third-octave $L_W(f)$, and the
A-weighted energy sum across the 21 bands gives the $L_{WA}$ in the title.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import emission

# One-third-octave mean room SPL (dB), 100 Hz - 10 kHz, and the room's T60.
freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
                  1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000],
                 dtype=float)
lp = np.linspace(80.0, 70.0, freqs.size)
t60 = np.full(freqs.size, 2.0)
rev = emission.sound_power_reverberation(
    lp, t60, volume=200.0, surface_area=220.0, frequencies=freqs,
    temperature=20.0, static_pressure=101.0,
)

# rev is the ReverberationSoundPowerResult computed above. One line:
rev.plot()
plt.show()

# By hand: a bar spectrum of LW with the A-weighted total in the title.
freqs = rev.frequencies
positions = np.arange(freqs.size)
fig, ax = plt.subplots()
ax.bar(positions, rev.sound_power_level, width=0.7, color="#1f77b4")
ax.set_xticks(positions)
ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Sound power level LW [dB]")
ax.set_title(
    f"Reverberation-room sound power (ISO 3741)  "
    f"LWA = {rev.sound_power_level_a:.1f} dB(A)")
plt.show()
```

</details>

`levels` may be a 1D mean spectrum or a 2D `(NM, NB)` array averaged over
positions. When the room volume, its reverberation time or the microphone
count fail an ISO 3741 qualification criterion (Table 1 minimum volume, the
$V/S$ reverberation floor, fewer than 6 positions, or an inter-position
spread above 1.5 dB), an advisory `SoundPowerWarning` is emitted and the
result still returns.

### `sound_power_reverberation()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `levels` | 1D or 2D array | dB | per band, or `(NM, NB)` | Mean room SPL; 2D is energy-averaged over positions |
| `t60` | float or 1D array | s | > 0 | Room reverberation time (scalar broadcasts) |
| `volume` | float | m³ | > 0 | Room volume $V$ |
| `surface_area` | float | m² | > 0 | Total room surface $S$ (Waterhouse, $A/S$) |
| `frequencies` | 1D array | Hz | one per band | Required (Waterhouse needs $f$); enables $L_{WA}$ |
| `background_levels` | 1D or 2D array | dB | matches `levels` | $K_{1i}$ per microphone position (Eq. 14/15, before the Eq. 16 average; frequency-dependent criterion) |
| `temperature` | float | °C | default `23.0` | Sets $c$, $C_1$, $C_2$ |
| `static_pressure` | float | kPa | default `101.325` | Sets $C_1$, $C_2$ |

`sound_power_comparison(levels, levels_ref, lw_ref, *, frequencies=None,
background_levels=…, background_levels_ref=…, temperature=23.0,
static_pressure=101.325)` takes the same room levels plus the reference
source's levels and known power.

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `levels_ref` | 1D or 2D array | dB | matches `levels` | Mean room SPL with the reference source (RSS) running |
| `lw_ref` | 1D array | dB | per band | Known sound power $L_W(\text{RSS})$ of the reference source |
| `background_levels` | 1D or 2D array | dB | matches `levels` | Background for the test source; per-band $K_1$ on $L_p(\text{ST})$ |
| `background_levels_ref` | 1D or 2D array | dB | matches `levels_ref` | Background for the **reference** source; per-band $K_1$ on $L_p(\text{RSS})$ |

`background_levels_ref` background-corrects the reference-source room level
$L_p(\text{RSS})$ exactly as `background_levels` does for the test source;
both need `frequencies` (the ISO 3741 criterion is frequency-dependent). Both
return a
`ReverberationSoundPowerResult`
(`sound_power_level`, `mean_pressure_level`, `absorption_area`,
`waterhouse_correction`, `background_correction`, `c1`, `c2`,
`speed_of_sound`, `sound_power_level_a`, `method`; the absorption/Waterhouse/`c1`
fields are `NaN` for the comparison method).

## 2. The measurement report (`.report()`)

The reverberation-room result (`ReverberationSoundPowerResult`, ISO 3741)
writes a one-page PDF fiche laid out like a sound-power test sheet through
its own `.report()`, sharing the layout and the `ReportMetadata` container of
the [pressure-method fiche](sound-power-pressure.md#3-the-measurement-report-report).
The standard-basis line
names ISO 3741:2010 and the precision accuracy grade (grade 1) and states which
method was used, the direct method using the room equivalent absorption area
(Eq. 20) or the comparison method using a reference sound source (Eq. 21). The
per-band table lists the mean room sound-pressure level $L_p$ and the band
sound-power level $L_W$, and the boxed $L_{WA}$ carries the total $L_W$ and the
determination method (the reverberation result has no expanded uncertainty $U$).
`verbose=True` adds the background correction $K_1$ and, for the direct method,
the equivalent absorption area $A$ and the Waterhouse boundary correction $C_w$;
the basis strip states the correction model (Eq. 20 or Eq. 21), the applied
meteorological corrections $C_1$/$C_2$ and the speed of sound, and cites the
Annex F A-weighting.

```python
import numpy as np
from phonometry import ReportMetadata, emission

freqs = np.array([125, 250, 500, 1000, 2000, 4000, 8000], float)
# Octave-band mean room sound-pressure levels in a qualified
# reverberation room of V = 200 m3, S = 240 m2, with a uniform T60 = 2.0 s.
lp = np.array([80.0, 83.0, 85.0, 84.0, 80.0, 75.0, 68.0])
res = emission.sound_power_reverberation(
    lp, 2.0, volume=200.0, surface_area=240.0, frequencies=freqs,
    temperature=20.0, static_pressure=101.325,
)

res.report(
    "sound_power_reverberation.pdf",
    metadata=ReportMetadata(
        client="Example manufacturing plant",
        specimen="Hydraulic power pack (floor-standing)",
        test_room="Qualified reverberation room, V = 200 m3, T60 = 2.0 s",
        instrumentation="Class 1 sound level meter (IEC 61672-1), s/n 0042",
        laboratory="Phonometry reference example",
        report_id="EXAMPLE-3741",
        requirement=96.0,
    ),
)   # LWA = 94.3 dB(A) re 1 pW -> declared limit 96 dB(A): PASS
```

The example fiche, regenerated with `make reports`, is kept rendered in the
repository. Click the preview to open the PDF:

[![ISO 3741 reverberation-room sound power example report: a header with the client, the noise source, the qualified reverberation test room and the instrumentation and climate, the octave-band table (125 Hz to 8 kHz) of mean room sound-pressure levels Lp and band sound-power levels LW, the sound-power spectrum LW(f) with a nominal band axis, the boxed A-weighted sound power level LWA = 94.3 dB(A) re 1 pW with the total LW = 96.7 dB and the direct determination method, and a PASS verdict against the declared 96 dB(A) limit, closed by a basis strip stating the Eq. 20 correction model with the Sabine absorption area, the Waterhouse boundary term and the meteorological corrections C1 and C2, and the Annex F A-weighting](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso3741_reverberation_power_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso3741_reverberation_power_example.pdf)

*Reverberation-room sound power fiche (`ReverberationSoundPowerResult.report`),
an ISO 3741 precision-grade direct-method determination with the Waterhouse and
$C_1$/$C_2$ corrections and the boxed $L_{WA}$.*


## See also

- [Sound Power](sound-power.md): choosing among the five determination
  routes, what the accuracy grades promise, and the ISO 4871 noise-emission
  declaration a measured $L_{WA}$ feeds.
- [Sound Power by Pressure Methods (ISO 3744 / ISO 3746 / ISO 3745)](sound-power-pressure.md):
  the in-situ enveloping surface and the precision anechoic array.
- [Sound Power by Intensity Scanning (ISO 9614)](sound-power-intensity.md):
  the routes that tolerate steady background noise.
- [Room Acoustics](room-acoustics.md): measuring the reverberation time
  $T_{60}$ that sets the Sabine absorption area.
- [Levels](levels.md): energy averaging and the A-weighting behind $L_{WA}$.
- [Theory](theory-environment-transport.md): the Waterhouse and $C_1$/$C_2$
  derivations.
- API reference: [`emission.sound_power_reverberation`](https://jmrplens.github.io/phonometry/reference/api/power/sound-power-reverberation/).

## References

- Beranek, L. L., & Mellow, T. J. (2012). *Acoustics: Sound fields and
  transducers*. Academic Press. ISBN 978-0-12-391421-7.
  [doi:10.1016/C2011-0-05897-0](https://doi.org/10.1016/C2011-0-05897-0).
  Radiation and sound fields: the diffuse-field relation between pressure
  and power that the reverberation-room method rests on.
- International Organization for Standardization. (2010). *Acoustics —
  Determination of sound power levels and sound energy levels of noise
  sources using sound pressure — Precision methods for reverberation test
  rooms* (ISO 3741:2010).
  [iso.org catalogue](https://www.iso.org/standard/52053.html).
  The direct (Eq. 20) and comparison (Eq. 21) methods of this guide, with
  the Waterhouse and meteorological corrections and the Table 1
  qualification criteria.

## Standards

ISO 3741:2010, *Acoustics — Determination of sound power levels and sound
energy levels of noise sources using sound pressure — Precision methods for
reverberation test rooms*: the direct (Eq. 20) and comparison methods with
the Waterhouse and meteorological corrections, the Table 1 qualification
criteria and the Annex F A-weighting.
