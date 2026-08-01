---
title: "emission.sound_power_reverberation"
description: "Sound power level of a noise source measured in a reverberation test room: ISO 3741:2010 (precision method, accuracy grade 1)."
sidebar:
  label: "sound_power_reverberation"
---

Sound power level of a noise source measured in a reverberation test room:
ISO 3741:2010 (precision method, accuracy grade 1).

The source is placed in a hard-walled reverberation room whose reverberant
field is sampled by microphones. Two methods are provided.

The **direct method** derives the sound power from the mean corrected room
sound pressure level `Lp(ST)` and the equivalent absorption area `A` of the
room, with the Sabine absorption area and the speed of sound `c` in m/s
(ISO 3741:2010 clause 9.1.4, Eq. 20):

$$
L_p(\text{ST}) = 10 \log_{10}\!\left[ \frac{1}{N_M} \sum_i 10^{0.1 L_{pi}} \right] \tag{Eq. 16}
$$

$$
A = \frac{55.26}{c} \, \frac{V}{T_{60}}
$$

$$
c = 20.05 \sqrt{273 + \theta}
$$

$$
L_W = L_p(\text{ST}) + 10 \log_{10}\frac{A}{A_0} + 4.34 \frac{A}{S} + 10 \log_{10}\!\left( 1 + \frac{S c}{8 V f} \right) + C_1 + C_2 - 6 \tag{Eq. 20}
$$

$10 \log_{10}(1 + Sc/(8Vf))$ is the Waterhouse boundary correction (energy
stored near the room boundaries); it vanishes as the frequency grows. `C1`
(Eq. 20, reference-quantity correction) and `C2` (radiation-impedance
correction) carry the result to the reference meteorological conditions of
clause 4 (23.0 C, 101.325 kPa, 50 %), per clause 9.1.4:

$$
C_1 = -10 \log_{10}\frac{p_s}{p_{s0}} + 5 \log_{10}\frac{273.15 + \theta}{314}
$$

$$
C_2 = -10 \log_{10}\frac{p_s}{p_{s0}} + 15 \log_{10}\frac{273.15 + \theta}{296}
$$

The **comparison method** replaces the absorption-area terms by a reference
sound source (RSS) of known sound power `LW(RSS)` measured at the same
positions (ISO 3741:2010 clause 9.1.5, Eq. 21):

$$
L_W = L_W(\text{RSS}) + \left( L_p(\text{ST}) - L_p(\text{RSS}) + C_2 \right) \tag{Eq. 21}
$$

Both methods cover the one-third-octave bands from 100 Hz to 10 kHz (clause
8.1). Octave-band, A-weighted and total levels follow ISO 3741 Annex F, which
reuses the ISO 3744 Annex E A-weighting band corrections.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## ReverberationSoundPowerResult

```python
ReverberationSoundPowerResult(
    frequencies: np.ndarray | None,
    sound_power_level: np.ndarray,
    mean_pressure_level: np.ndarray,
    absorption_area: np.ndarray,
    waterhouse_correction: np.ndarray,
    background_correction: np.ndarray,
    c1: float,
    c2: float,
    speed_of_sound: float,
    sound_power_level_a: float,
    method: str,
)
```

Result of an ISO 3741:2010 reverberation-room sound power determination.

`sound_power_level` is the per-band `LW` (Eq. 20 direct method, Eq. 21
comparison method). `mean_pressure_level` is the mean corrected room level
`Lp(ST)` (Eq. 16). For the direct method `absorption_area` is the Sabine
equivalent absorption area `A` per band and `waterhouse_correction` the
boundary term $10 \log_{10}(1 + Sc/(8Vf))$; both are `NaN` for the
comparison method. `background_correction` is the effective per-band
background correction `K1`: with per-position input each position is
corrected by its own `K1i` (Eq. 14/15) before the energy average
(Eq. 16), and the reported value is the resulting per-band shift of the
mean level (zero when no background is supplied).
`c1` and `c2` are the reference-quantity and radiation-impedance
corrections (`c1` is `NaN` for the comparison method, which uses only
`c2`). `speed_of_sound` is `c` at the test temperature.
`sound_power_level_a` is the A-weighted total `LWA` (Annex F Eq. F.2),
computed only when `frequencies` are supplied (`NaN` for several bands
without them; equal to `LW` for a single band). `method` is `'direct'`
or `'comparison'`.

### ReverberationSoundPowerResult.plot()

```python
ReverberationSoundPowerResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the LW spectrum with the A-weighted total annotated.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### ReverberationSoundPowerResult.report()

```python
ReverberationSoundPowerResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ISO 3741 reverberation-room sound-power determination fiche.

Writes a one-page sound-power test sheet: the standard-basis line naming
the reverberation-room method (the direct method using the room
equivalent absorption area, or the comparison method using a reference
sound source) and the precision accuracy grade (ISO 3741:2010, grade 1),
an optional metadata header (client, noise source, test environment,
instrumentation, climate, date), a per-band table (nominal octave/
one-third-octave frequency, the mean room sound-pressure level `Lp`
and the band sound-power level `LW`), the sound-power spectrum
`LW(f)` with a nominal band axis, the boxed A-weighted sound power
level `LWA` (dB re 1 pW) with the total `LW` and the determination
method, an optional verdict row against a declared limit, and a
measurement-basis strip stating the correction model (Eq. 20 direct or
Eq. 21 comparison), the applied meteorological corrections `C1`/`C2`
and the speed of sound.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header (`client`, `specimen` the noise source, `test_room` the reverberation test room, `instrumentation`, `temperature`, `relative_humidity`, `pressure`, `test_date`), the footer identity (`laboratory`, `operator`, `report_id`, `notes`) and, via `requirement`, a declared A-weighted sound-power limit the fiche checks the result against (lower is better). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True` the per-band table adds the background correction `K1` and, for the direct method, the equivalent absorption area `A` and the Waterhouse boundary correction `Cw`. |
| `language` | Fiche language: `"en"` (default) or `"es"`. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is unknown. |
| ImportError | If reportlab (or, for the figure, matplotlib) is not installed (`pip install phonometry[report]`). |

## sound_power_comparison

```python
sound_power_comparison(
    levels: np.ndarray,
    levels_ref: np.ndarray,
    lw_ref: np.ndarray,
    *,
    frequencies: np.ndarray | None = None,
    background_levels: np.ndarray | None = None,
    background_levels_ref: np.ndarray | None = None,
    temperature: float = 23.0,
    static_pressure: float = 101.325,
) -> ReverberationSoundPowerResult
```

Sound power level in a reverberation room, comparison method (ISO 3741).

A reference sound source of known per-band sound power `lw_ref` is
measured at the same microphone positions as the source under test. The
sound power level in each band follows Eq. (21):

$$
L_W = L_W(\text{RSS}) + \left( L_p(\text{ST}) - L_p(\text{RSS}) + C_2 \right)
$$

where `Lp(ST)` and `Lp(RSS)` are the mean room levels (Eq. 16/17) of the
test source and the reference source and `C2` is the radiation-impedance
correction. The absorption-area, Waterhouse and `C1` terms cancel between
the two sources, so the room absorption need not be known.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | Mean room SPL per band (1D) or `(NM, NB)` per-position levels of the source under test, in decibels. |
| `levels_ref` | Same, for the reference sound source, in decibels. |
| `lw_ref` | Known sound power level `LW(RSS)` per band, in decibels. |
| `frequencies` | Band mid-frequencies (Hz) for the A-weighted total. |
| `background_levels` | Background levels for the `K1` correction of `levels` (per position, or a single spectrum; applied per position per Eq. 14/15 before the Eq. 16 average when `levels` is 2D). |
| `background_levels_ref` | Background levels matching `levels_ref`. |
| `temperature` | Air temperature `theta` in the room, in degrees Celsius. |
| `static_pressure` | Static pressure `ps` in the room, in kilopascals. |

**Returns:** [`ReverberationSoundPowerResult`](/phonometry/reference/api/power/sound-power-reverberation/#reverberationsoundpowerresult) (`method='comparison'`).

## sound_power_reverberation

```python
sound_power_reverberation(
    levels: np.ndarray,
    t60: np.ndarray,
    volume: float,
    surface_area: float,
    frequencies: np.ndarray,
    *,
    background_levels: np.ndarray | None = None,
    temperature: float = 23.0,
    static_pressure: float = 101.325,
) -> ReverberationSoundPowerResult
```

Sound power level in a reverberation room, direct method (ISO 3741:2010).

`levels` is either a 1D per-band spectrum of the mean room sound pressure
level or a 2D `(NM, NB)` array (one row per microphone position, one
column per band) that is energy-averaged over positions (Eq. 16). The sound
power level in each band follows Eq. (20):

$$
L_W = L_p(\text{ST}) + 10 \log_{10}\frac{A}{A_0} + 4.34 \frac{A}{S} + 10 \log_{10}\!\left( 1 + \frac{S c}{8 V f} \right) + C_1 + C_2 - 6
$$

with the Sabine equivalent absorption area
$A = (55.26/c)(V/T_{60})$ and the speed of sound
$c = 20.05 \sqrt{273 + \theta}$. The Waterhouse term
$10 \log_{10}(1 + Sc/(8Vf))$ needs the band mid-frequencies, so
`frequencies` is required. `C1` and `C2` carry the result to the
reference meteorological conditions (clause 4).

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | Mean room SPL per band (1D) or `(NM, NB)` per-position levels, in decibels. |
| `t60` | Reverberation time `T60` per band, in seconds (scalar or one value per band). |
| `volume` | Reverberation-room volume `V`, in cubic metres. |
| `surface_area` | Total room surface area `S`, in square metres. |
| `frequencies` | One-third-octave (or octave) band mid-frequencies, Hz. |
| `background_levels` | Background levels for the `K1` correction: per-position `(NM, NB)` (or a single `(NB,)` spectrum used at every position) with per-position `levels`, applied per position (Eq. 14/15) before the energy average (Eq. 16). With 1D pre-averaged `levels` a single `K1` from the averaged spectra approximates the per-position procedure of clause 9.1.2. |
| `temperature` | Air temperature `theta` in the room, in degrees Celsius. |
| `static_pressure` | Static pressure `ps` in the room, in kilopascals. |

**Returns:** [`ReverberationSoundPowerResult`](/phonometry/reference/api/power/sound-power-reverberation/#reverberationsoundpowerresult).
