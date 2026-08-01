← [Documentation index](README.md)

# Aircraft noise: Effective Perceived Noise Level (ICAO Annex 16 / IEC 61265)

The **Effective Perceived Noise Level (EPNL)** is the noise-certification metric
for transport-category aircraft. It condenses a half-second one-third-octave
spectral time history of a flyover into a single number, in EPNdB, through five
steps of **ICAO Annex 16, Vol. I, Appendix 2**. This page covers the four
primitives that build the metric and the IEC 61265 measurement-system verifier.
Each quantity is validated against the worked examples of the ICAO Doc 9501
Environmental Technical Manual (ETM) Vol. I.

Every certification level on this page is asked for at one of the three
reference points Annex 16 fixes around the runway, so it helps to see them
before the mathematics.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_aircraft_certification_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_aircraft_certification.svg" alt="The three aircraft noise certification points of ICAO Annex 16 Chapter 3, in plan and side view sharing the same distance scale: a lateral reference line parallel to the runway at 450 m where take-off noise is greatest, with a symmetric point on the other side; the flyover reference point on the extended centre line 6 500 m from the start of roll, under the climb-out; and the approach reference point 2 000 m from the threshold, 120 m below the 3 degree glide path that meets the ground 300 m beyond the threshold; microphones sit 1.2 m above the ground and the metric at all three points is EPNL in EPNdB" width="92%"></picture>

## 1. Perceived noisiness and PNL

Each of the 24 one-third-octave-band levels (50 Hz–10 kHz) is converted to a
perceived **noisiness** in noys by the analytic piecewise law of Table A2-3,
then combined into the total noisiness and the perceived noise level:

$$
N = 0.85\,n_{\max} + 0.15\sum_i n_i, \qquad
\mathrm{PNL} = 40 + \frac{10}{\lg 2}\,\lg N .
$$

```python
from phonometry import aircraft

noys = aircraft.perceived_noisiness(spl)      # per-band noys (spl = 24 band levels, dB)
pnl = aircraft.perceived_noise_level(spl)      # PNdB
```

## 2. Tone correction

Spectral irregularities (fan/turbine tones) are penalised by a **tone
correction** $C$, found with the slope ("encircling") method: slopes are
smoothed to a background spectrum $\mathrm{SPL}''$, the tone excess
$F = \mathrm{SPL} - \mathrm{SPL}''$ above 1.5 dB is mapped to a correction
factor (frequency-split at 500 Hz / 5000 Hz, capped at 6⅔ dB), and the maximum
over bands is taken.

```python
from phonometry import aircraft

c = aircraft.tone_correction(spl)              # dB; added to PNL to give PNLT
```

The implementation reproduces the ICAO Doc 9501 ETM Vol. I **Table 3-7**
turbofan example exactly, including its $\mathrm{SPL}''$ background column and
the resulting $C = 2.0\ \text{dB}$ at 2500 Hz.

## 3. EPNL

Over the flyover, the tone-corrected level is
$\mathrm{PNLT} = \mathrm{PNL} + C$; its maximum is $\mathrm{PNLTM}$. The metric
integrates $\mathrm{PNLT}$ over the **10 dB-down** window (the records nearest
to $\mathrm{PNLTM} - 10$ on each side) and normalises to 10 s:

$$
\mathrm{EPNL} = 10\lg\!\Big(\sum_{k=k_F}^{k_L} 10^{\mathrm{PNLT}(k)/10}\,\Delta t(k)\Big)
- 10\lg T_0, \qquad T_0 = 10\ \mathrm{s},
$$

so $\mathrm{EPNL} = \mathrm{PNLTM} + D$ with the duration correction $D$.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/epnl_dark.svg">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/epnl.svg" alt="Aircraft-flyover perceived-noise-level time history: PNL and the tone-corrected PNLT versus time, with the maximum PNLTM marked and the 10 dB-down integration window shaded, annotated with the resulting EPNL and duration correction" width="82%">
</picture>

```python
import numpy as np
from phonometry import aircraft

# spectra: a (K, 24) array of one-third-octave band levels sampled every dt s
res = aircraft.effective_perceived_noise_level(spectra, dt=0.5)
print(res.epnl, res.pnltm, res.duration_correction, res.band_limits)
res.plot()   # PNL/PNLT time history (needs matplotlib)
```

`effective_perceived_noise_level` returns an `EPNLResult` bundling the per-record
`pnl`, `tone_correction`, `pnlt`, the peak `pnltm`, the `duration_correction`,
the `epnl` and the 10 dB-down `band_limits`. The reference-condition
integrated-method example of ETM Vol. I **Table 4-4** (a 31-record
$\mathrm{PNLT}$ history with non-uniform durations) is reproduced as
$\mathrm{EPNL} = 92.6\ \text{EPNdB}$. `epnl_from_pnlt` exposes the
duration/limit machinery directly from a $\mathrm{PNLT}$ series.

<details>
<summary>Show the code for this figure</summary>

```python
import numpy as np
from phonometry import aircraft

k, dt = 41, 0.5
idx = np.arange(k)
shape = 15.0 * np.exp(-((np.log10(aircraft.NOY_BANDS) - np.log10(400.0)) ** 2) / 0.5)
gain = 30.0 * np.exp(-((idx - 20.0) ** 2) / (2 * 5.0**2)) - 5.0
spectra = (55.0 + shape)[None, :] + gain[:, None]
spectra[:, 17] += 12.0 * np.exp(-((idx - 20.0) ** 2) / (2 * 6.0**2))  # 2500 Hz fan tone
aircraft.effective_perceived_noise_level(spectra, dt).plot()
```

</details>

### ICAO EPNL report (`.report()`)

`EPNLResult.report(path)` renders a one-page PDF fiche laid out like an
aircraft-noise-certification data sheet: a standard-basis line (ICAO Annex 16
Vol. I Appendix 2), an optional TCDSN-style metadata header (aircraft,
manufacturer / type-certificate holder, applicant, measurement point), a
metrics table of the informational intermediate quantities (the peak
$\mathrm{PNLTM}$, the duration correction $D$, the 10 dB-down record window
and, when non-zero, the bandsharing adjustment) above the full-width landscape
$\mathrm{PNLT}$-versus-time plot (the result's own `.plot()`), the boxed
$\mathrm{EPNL} = X\ \text{EPNdB}$ single number, a
Level | Limit | Margin verdict row when a certification limit is supplied, a
static reference-conditions strip (25 °C, 70 % RH, sea level, zero wind, ISA)
and a footer with the fixed disclaimer. It uses the same `ReportMetadata`
container (documented under [Insulation
ratings](insulation-ratings.md#report-metadata-reportmetadata)) and rendering
engine as the ISO 717 insulation fiche; a supplied `requirement` is read as the
certification EPNL limit in EPNdB (the EPNL passes at or below it), and
`metadata=None` produces a lightweight prediction fiche with no verdict row.
Rendering needs reportlab and, for the figure the fiche embeds, matplotlib (`pip
install "phonometry[report,plot]"`); only `engine="reportlab"` is supported. The
fiche renders in English by default; pass `language="es"` for a Spanish fiche
(translated fixed strings and a comma decimal separator), e.g.
`res.report("epnl_fiche_es.pdf", language="es")`. The fiche is a computational
EPNL result and is not an official State noise certificate; it does not
reproduce any TCDSN.

```python
from phonometry import effective_perceived_noise_level, ReportMetadata

# spectra: a (K, 24) array of one-third-octave band levels sampled every dt s
res = effective_perceived_noise_level(spectra, dt=0.5)
res.report(
    "epnl_fiche.pdf",
    metadata=ReportMetadata(
        specimen="Example twin-turbofan transport",
        manufacturer="Example Aircraft Company",
        measurement_standard="ICAO Annex 16 Vol I Amendment 14 Chapter 4",
        laboratory="Phonometry Reference Laboratory",
        requirement=101.0,          # certification EPNL limit (EPNdB)
    ),
)                                   # EPNL (EPNdB) with PNLTM and D
```

The example fiche is regenerated with `make reports` and kept rendered in the
repository:

<p align="center">
  <a href="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/icao_epnl_example.pdf"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/icao_epnl_example.webp" alt="One-page ICAO Annex 16 EPNL certification fiche: a metadata header, a reference-conditions strip, an intermediate-quantities table (PNLTM, duration correction D, 10 dB-down window), the landscape PNL/PNLT time-history plot, the boxed EPNL = 98.3 EPNdB single-number result and a PASS verdict against a 101 EPNdB limit" width="70%"></a>
</p>

## 4. Measurement-system verification (IEC 61265)

`verify_aircraft_noise_system` checks a supplied set of measured performance
values against the IEC 61265:1995 tolerances for aircraft-noise measurement
systems: the microphone directional-response limits (Table 1, with the
"intermediate angle uses the greater angle's limit" rule) and the scalar
frequency-response, linearity and resolution limits. The one-third-octave
filtering itself is covered by the library's IEC 61260 class-2 filter
verification (`verify_filter_class`).

```python
from phonometry import metrology

report = metrology.verify_aircraft_noise_system(
    directional={4000.0: {30: 0.4, 60: 0.9, 90: 1.9, 120: 2.4, 150: 2.4}},
    frequency_response={1000.0: 1.2},
)
print(report["passed"], report["checks"])
```

## 5. Atmospheric absorption (SAE ARP 5534)

Correcting a measured flyover spectrum to reference atmospheric conditions
needs the one-third-octave-band attenuation over the path. The pure-tone
coefficient is the ISO 9613-1 one (identical, per ARP 5534 §3.1) already
provided by `air_attenuation`; `sae_band_attenuation` adds the **SAE Method**
(ARP 5534 §3.2.2), a regression that maps the pure-tone mid-band path-length
attenuation $\delta_t = \alpha\,s$ to the band attenuation $\delta_B$ and stays
consistent with the ISO/ANSI Exact Method well beyond the 50 dB limit of the
older Approximate Method.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/aircraft_atmospheric_absorption_dark.svg">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/aircraft_atmospheric_absorption.svg" alt="Aircraft atmospheric absorption versus frequency for two path lengths: the SAE-Method one-third-octave-band attenuation rises with frequency and stays below the pure-tone mid-band value at high absorption" width="82%">
</picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import aircraft

freqs = 1000.0 * 10.0 ** (np.arange(-13, 11) / 10.0)   # 50 Hz-10 kHz thirds
fig, ax = plt.subplots()
# solid: SAE band attenuation, dashed: pure-tone mid-band
for s in (1000.0, 7620.0):
    att = aircraft.sae_band_attenuation(freqs, s, temperature=25.0, relative_humidity=70.0)
    line, = ax.semilogx(att.frequency, att.band_attenuation, marker="o",
                        markersize=3, label=f"SAE band ({s:.0f} m)")
    ax.semilogx(att.frequency, att.midband_attenuation, "--", alpha=0.6,
                color=line.get_color())
ax.set(xlabel="Frequency [Hz]", ylabel="Attenuation [dB]",
       title="Aircraft atmospheric absorption at 25 °C, 70% RH")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.show()
```

</details>

```python
import numpy as np
from phonometry import aircraft

freqs = 1000.0 * 10.0 ** (np.arange(-13, 11) / 10.0)   # 50 Hz–10 kHz thirds
att = aircraft.sae_band_attenuation(freqs, path_length=7620.0,
                              temperature=25.0, relative_humidity=70.0)
print(att.band_attenuation)   # δ_B per band, dB
att.plot()                    # band vs pure-tone mid-band (needs matplotlib)
```

`sae_band_attenuation` returns an `AircraftBandAttenuation` with `band_attenuation`
($\delta_B$), `midband_attenuation` ($\delta_t = \alpha\,s$) and the pure-tone
`coefficient` ($\alpha$, dB/m). The SAE Method is valid roughly 6–32 °C and 20–95 % RH (the 14 CFR
Part 36 test window), over path lengths to 7620 m, and is reciprocal
(source↔receiver).

The certification chain ends here. Turning these aeroplanes into noise
around an airport, the noise-power-distance tables, the per-segment
corrections of a flight path and the ground contour of a single event, is
the ECAC Doc 29 method of [Airport noise](airport-noise.md).

## References

- International Civil Aviation Organization. (2017). *Annex 16 to the
  Convention on International Civil Aviation: Environmental protection —
  Volume I: Aircraft noise* (8th ed.).
  [ICAO store](https://store.icao.int/en/annex-16-environmental-protection-volume-i-aircraft-noise).
  The normative Appendix 2 EPNL procedure implemented in sections 1-3.
- International Civil Aviation Organization. (2018). *Environmental technical
  manual — Volume I: Procedures for the noise certification of aircraft*
  (Doc 9501, 3rd ed.).
  [ICAO store](https://store.icao.int/en/environmental-technical-manual-volume-1-procedures-for-the-noise-certification-of-aircraft-doc-9501-1).
  The worked examples (Table 3-7 tone correction, Table 4-4 integrated-method
  EPNL) used as the numeric oracles of sections 2-3.
- International Electrotechnical Commission. (1995). *Electroacoustics —
  Instruments for measurement of aircraft noise — Performance requirements for
  systems to measure one-third-octave-band sound pressure levels in noise
  certification of transport-category aeroplanes* (IEC 61265:1995; since
  revised as [IEC 61265:2018](https://webstore.iec.ch/en/publication/32635),
  the 1995 edition is the implemented one).
  [IEC webstore](https://webstore.iec.ch/en/publication/5076).
  The measurement-system tolerances checked by the section 4 verifier.
- SAE International. (2013). *Application of pure-tone atmospheric absorption
  losses to one-third octave-band data* (SAE ARP 5534, reaffirmed 2021).
  [sae.org](https://www.sae.org/standards/content/arp5534/).
  The SAE-Method band attenuation of section 5.
- SAE International. (2012). *Standard values of atmospheric absorption as a
  function of temperature and humidity* (SAE ARP 866B, stabilized 2012).
  [sae.org](https://www.sae.org/standards/content/arp866b/).
  The predecessor SAE atmospheric-absorption practice, source of the 50
  dB-limited Approximate Method that section 5 contrasts with the SAE Method.

## Standards

ICAO Annex 16, *Environmental Protection*, Vol. I, *Aircraft
Noise*, Appendix 2: the analytic EPNL procedure (perceived noisiness Table A2-3,
tone correction Table A2-2, duration correction). ICAO Doc 9501, *Environmental
Technical Manual*, Vol. I: the worked examples (Table 3-7 tone correction,
Table 4-4 integrated-method EPNL) used as numeric oracles. IEC 61265:1995,
*Instruments for the measurement of aircraft noise*: the measurement-system
performance tolerances. SAE ARP 5534:2021, *Application of Pure-Tone
Atmospheric Absorption Losses to One-Third-Octave-Band Data*: the SAE-Method
band attenuation (Eqs. 7–10), with the pure-tone coefficient from ISO 9613-1.
The ECAC Doc 29 airport-noise method is documented in
[Airport noise](airport-noise.md).
