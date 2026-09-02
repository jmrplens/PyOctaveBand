---
title: "emission.sound_power_in_duct"
description: "Sound power radiated into a duct by a fan, in-duct method: ISO 5136:2003."
sidebar:
  label: "sound_power_in_duct"
---

Sound power radiated into a duct by a fan, in-duct method: ISO 5136:2003.

A ducted fan does not radiate into a room: what leaves it travels down the
duct, so ISO 5136 measures it *inside* the duct. The fan is connected to an
anechoically terminated test duct on its inlet and/or outlet side, and a
microphone in that duct samples the one-third-octave sound pressure level at
three circumferential positions (clause 6.2.2), by multiplexing, or over one
continuous revolution (clauses 7.2.3 and 7.2.4). Three things stand between
that reading and the sound power. The microphone sits in a mean flow that
adds turbulent pressure fluctuations, so it is shielded by a sampling tube,
a nose cone or a foam ball; the shield has a frequency response of its own;
and above the first cut-on the duct carries higher-order modes to which a
sampling tube does not respond as it does to a plane wave. Clause 8 gathers
the three into one combined correction $C$ on the averaged level
(equations (9), (10) and (11)):

$$
\overline{L_p} = 10 \lg\!\left[\frac{1}{n}\sum_{i=1}^{n} 10^{0.1 L_{pi}}\right] \mathrm{dB} + C, \qquad C = C_1 + C_2 + C_{3,4} \tag{Eqs 9, 10}
$$

where $C_1$ is the microphone free-field correction from the
manufacturer's data, $C_2$ the frequency response correction of the
shield measured per clause 5.3.3.2 c) or 5.3.4.2, and $C_{3,4}$ the
combined mean flow velocity and modal correction. For a level already averaged
by multiplexing or a traverse, $\overline{L_p} = \overline{L_{pm}} + C$
(equation (11)). The sound power then follows from the plane-wave relation of
clause 8.2:

$$
L_W = \overline{L_p} + \left(10 \lg\frac{S}{S_0} - 10 \lg\frac{\rho c}{(\rho c)_0}\right) \mathrm{dB}, \qquad S = \frac{\pi d^2}{4}, \quad S_0 = 1~\mathrm{m^2}, \quad (\rho c)_0 = 400~\mathrm{N \cdot s/m^3} \tag{Eq. 12}
$$

$C_{3,4}$ is the part of the standard that is not arithmetic. For the
sampling tube it is a polynomial in the mean flow velocity $U$, in
metres per second, negative on the inlet side and positive on the outlet side
(clause 5.3.3.4, equation (7)):

$$
C_{3,4} = \sum_{i=0}^{10} a_i U^i \tag{Eq. 7}
$$

whose coefficients $a_i$ Annex A tabulates per one-third-octave band
and per range of test-duct diameter (Tables A.1 to A.6, 0,15 m to 2 m), an
empty cell being zero. The coefficients are normative for 50 Hz to 10 kHz and
$|U| \le 40$ m/s, and given for information only for
$40 < |U| \le 60$ m/s and for 12,5 kHz to 20 kHz (the footnote of every
table). For the omni-directional nose cone and foam ball no modal data exist,
and clause 5.3.4.3 replaces the polynomial by the frequency-independent
convective term (equation (8)):

$$
C_{3,4} = 10 \lg \frac{1}{(1 - U/c)^2}~\mathrm{dB} \tag{Eq. 8}
$$

with $c$ the speed of sound, 340 m/s under normal conditions.

The A-weighted sound power level is the energy sum of the band levels with the
$C_j$ of Table C.1 (Annex C, equation (C.1)), and the uncertainty to
be recorded is the reproducibility of Table 2, $\sigma_R$ per band for
the sampling tube, expanded to $2\sigma_R$ at 95 % coverage (clause 9.2).
Above 10 kHz the standard suggests the extrapolated values of Table 3 without
making them part of itself.

What is *not* a term of $L_W$, and is therefore not computed here, is
the qualification of the facility and the instrument: the reflection
coefficient of the anechoic termination (Table 5, Annex F), the directivity
of the sampling tube (equation (6), Table 6), the signal-to-noise ratio
against turbulence (Annex B) and the duct geometry (clause 5.2). The
informative Annexes H and I extend the coefficient tables below 0,15 m and
above 2 m; the standard's own scope stops at those diameters and so does this
module.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## flow_modal_correction

```python
flow_modal_correction(
    frequencies: ArrayLike,
    flow_velocity: float,
    duct_diameter: float,
    *,
    shield: MicrophoneShield = 'sampling-tube',
    speed_of_sound: float = 340.0,
) -> np.ndarray
```

Combined mean flow velocity and modal correction $C_{3,4}$.

For the sampling tube, the polynomial of clause 5.3.3.4 with the
coefficients of Annex A for the band and the test-duct diameter, an empty
cell of the print counting as zero (Eq. (7)):

$$
C_{3,4} = \sum_{i=0}^{10} a_i U^i
$$

For the omni-directional nose cone and foam ball, the frequency-independent
convective term of clause 5.3.4.3 (Eq. (8)):

$$
C_{3,4} = 10 \lg \frac{1}{(1 - U/c)^2}~\mathrm{dB}
$$

$U$ is signed: negative for an inlet-side measurement, positive on
the outlet side (Table 1 NOTE 2), so the same speed reads as a different
correction on the two sides of the fan.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Nominal one-third-octave centre frequencies, in hertz, 50 Hz to 20 kHz. |
| `flow_velocity` | Mean flow velocity $U$ at the microphone position, in metres per second, negative on the inlet side. |
| `duct_diameter` | Test-duct diameter $d$, in metres, 0,15 m to 2 m; it selects the Annex A table for the sampling tube and is checked against the scope for the other shields. |
| `shield` | `"sampling-tube"` (default), `"nose-cone"` or `"foam-ball"`. |
| `speed_of_sound` | The $c$ of Eq. (8), in metres per second; the 340 m/s the standard states for normal conditions by default. Unused by the sampling tube. |

**Returns:** $C_{3,4}$ per band, in decibels.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a band that is not a nominal centre, a diameter outside 0,15 m to 2 m, a velocity beyond the shield's limit (60 m/s for the sampling tube, 20 m/s for the nose cone, 15 m/s for the foam ball) or a non-positive speed of sound. |

## in_duct_reproducibility

```python
in_duct_reproducibility(frequencies: ArrayLike) -> np.ndarray
```

Standard deviation of reproducibility $\sigma_R$ per band.

Table 2 of ISO 5136:2003 for the sampling tube, 50 Hz to 10 kHz, which is
what clause 9.2 doubles into the expanded uncertainty to be recorded at
95 % coverage. For 12,5 kHz, 16 kHz and 20 kHz the extrapolated values of
Table 3 are returned; clause 4 suggests them while saying that
measurements above 10 kHz are not part of the standard, so a result that
carries those bands says so in `information_only_band`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Nominal one-third-octave centre frequencies, in hertz, 50 Hz to 20 kHz. |

**Returns:** $\sigma_R$ per band, in decibels.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a band that is not a nominal centre. |

## InDuctSoundPowerResult

```python
InDuctSoundPowerResult(
    frequencies: np.ndarray,
    sound_power_level: np.ndarray,
    mean_pressure_level: np.ndarray,
    corrected_pressure_level: np.ndarray,
    microphone_correction: np.ndarray,
    shield_correction: np.ndarray,
    flow_modal_correction: np.ndarray,
    combined_correction: np.ndarray,
    reproducibility_standard_deviation: np.ndarray,
    expanded_uncertainty: np.ndarray,
    information_only_band: np.ndarray,
    duct_diameter: float,
    duct_area: float,
    characteristic_impedance: float,
    speed_of_sound: float,
    flow_velocity: float,
    shield: str,
    sound_power_level_a: float,
)
```

Result of an ISO 5136:2003 in-duct sound power determination.

`sound_power_level` is the per-band $L_W$ of Eq. (12), and
`sound_power_level_a` the A-weighted total of Annex C, Eq. (C.1), over
the bands supplied. `mean_pressure_level` is the spatially averaged
level before any correction, the bracket of Eq. (9) or the
$\overline{L_{pm}}$ of Eq. (11); `corrected_pressure_level` is
$\overline{L_p}$ after the combined correction. The three corrections
are kept apart so the record clause 9.1 f) asks for can be made:
`microphone_correction` ($C_1$), `shield_correction`
($C_2$) and `flow_modal_correction` ($C_{3,4}$), with
`combined_correction` their sum (Eq. 10).

`reproducibility_standard_deviation` is $\sigma_R$ of Table 2 per
band (Table 3 above 10 kHz) and `expanded_uncertainty` is twice it, the
95 % figure clause 9.2 says to record. `information_only_band` marks the
bands the standard gives for information rather than as part of itself:
those above 10 kHz, and every band when the sampling tube is used between
40 m/s and 60 m/s (5.3.3.4 NOTE, clause 4). Table 2 is stated for the
sampling tube; clause 4 NOTE 5 expects the figures to grow for the other
shields and gives no others, so the same values are reported for them.

`duct_diameter` and `duct_area` are $d$ and $S$,
`characteristic_impedance` is the $\rho c$ of the duct air and
`speed_of_sound` its $c$, `flow_velocity` is the signed $U$
(negative on the inlet side) and `shield` names the microphone shield.

### InDuctSoundPowerResult.plot()

```python
InDuctSoundPowerResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the in-duct sound power spectrum with the A-weighted total.

One bar per one-third-octave band of `sound_power_level`, the
$L_{W\\mathrm{A}}$ of Annex C in the title. Requires matplotlib
(`pip install phonometry[plot]`); returns the
`Axes`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the band `bar`. |

**Returns:** The axes.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `language` is unknown. |

## sound_power_in_duct

```python
sound_power_in_duct(
    levels: ArrayLike,
    frequencies: ArrayLike,
    duct_diameter: float,
    flow_velocity: float,
    *,
    shield: MicrophoneShield = 'sampling-tube',
    microphone_correction: ArrayLike = 0.0,
    shield_correction: ArrayLike = 0.0,
    temperature: float = 20.0,
    static_pressure: float = 101.325,
) -> InDuctSoundPowerResult
```

Sound power radiated into a test duct, in-duct method (ISO 5136:2003).

`levels` is either a `(positions, bands)` array of the time-averaged
sound pressure level at each circumferential microphone position, energy-
averaged by Eq. (9), or a `(bands,)` spectrum already averaged by
multiplexing or a continuous traverse (Eq. (11)). The combined correction
$C = C_1 + C_2 + C_{3,4}$ of Eq. (10) is added to the average, with
$C_1$ and $C_2$ supplied per band or as a scalar and
$C_{3,4}$ from [`flow_modal_correction`](/phonometry/reference/api/power/sound-power-in-duct/#flow_modal_correction), and the plane-wave
relation of Eq. (12) gives the sound power in each band:

$$
L_W = \overline{L_p} + 10 \lg\frac{\pi d^2 / 4}{S_0} - 10 \lg\frac{\rho c}{(\rho c)_0}
$$

The A-weighted total follows Annex C over the bands supplied, and the
uncertainty statement of clause 9.2, twice the reproducibility of
Table 2, is carried per band.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | Sound pressure levels, in decibels, `(positions, bands)` or an already averaged `(bands,)` spectrum. |
| `frequencies` | Nominal one-third-octave centre frequencies of the bands, in hertz, 50 Hz to 20 kHz. |
| `duct_diameter` | Test-duct diameter $d$, in metres, 0,15 m to 2 m. |
| `flow_velocity` | Mean flow velocity $U$ at the microphone position, in metres per second; negative on the inlet side, positive on the outlet side. |
| `shield` | Microphone shield, `"sampling-tube"` (default), `"nose-cone"` or `"foam-ball"`. |
| `microphone_correction` | $C_1$, the manufacturer's free-field correction of the microphone, in decibels, per band or scalar. |
| `shield_correction` | $C_2$, the frequency response correction of the shield determined per clause 5.3.3.2 c) or 5.3.4.2, in decibels, per band or scalar. |
| `temperature` | Air temperature in the duct, in degrees Celsius, -50 degC to 70 degC; sets $c$ and $\rho$. |
| `static_pressure` | Static pressure in the duct, in kilopascals; sets $\rho$. |

**Returns:** [`InDuctSoundPowerResult`](/phonometry/reference/api/power/sound-power-in-duct/#inductsoundpowerresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for levels of the wrong shape or not finite, a band that is not a nominal centre, a diameter, velocity or temperature outside the scope of the standard, a correction that is neither a scalar nor one value per band, or a non-positive static pressure. |
