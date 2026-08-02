---
title: "vibration.structural.mechanical_mobility"
description: "Mechanical mobility and the frequency-response-function family (ISO 7626-1:2011)."
sidebar:
  label: "mechanical_mobility"
---

Mechanical mobility and the frequency-response-function family (ISO 7626-1:2011).

Mechanical **mobility** is the complex ratio of a velocity response to the
excitation force that produces it, $Y_{ij} = v_i / F_j$ (ISO 7626-1,
3.1.2). It is
one of a family of motion-per-force frequency-response functions (FRFs); which
member is used depends only on whether the motion is expressed as displacement,
velocity or acceleration, and each has a force-per-motion reciprocal
(ISO 7626-1, Table 1):

===============  =====================  ===========  =========================
Motion           FRF (motion / force)   Unit          Reciprocal (force / motion)
===============  =====================  ===========  =========================
displacement     dynamic compliance /   m/N           dynamic stiffness  (N/m)
                 receptance `H`
velocity         mobility `Y`         m/(N.s)        impedance          (N.s/m)
acceleration     accelerance `A`      1/kg           apparent mass      (kg)
===============  =====================  ===========  =========================

For a harmonic motion $x e^{j \omega t}$ the velocity is
$j \omega x$ and the acceleration $-\omega^2 x$, so every FRF
follows from the receptance `H`:

$$
Y = j \omega H \qquad A = -\omega^2 H
$$

$$
Z~\text{(impedance)} = \frac{1}{Y}
$$

$$
M~\text{(apparent mass)} = \frac{1}{A} \quad \text{(Table 1 name: "effective mass")}
$$

$$
K~\text{(dyn. stiffness)} = \frac{1}{H}
$$

These element-wise reciprocals are the **free** quantities of ISO 7626-1,
3.1.4; the *blocked* matrix quantities of Table 1 do not invert element-wise
($Z_{ij} \ne 1/Y_{ij}$ for multi-coordinate systems); see
[`convert_frf`](/phonometry/reference/api/vibration/mechanical-mobility/#convert_frf).

[`convert_frf`](/phonometry/reference/api/vibration/mechanical-mobility/#convert_frf) moves between any two of the six FRFs through the receptance
pivot. A **driving-point** FRF has the response and force at the same point
($i = j$); a **transfer** FRF has them at different points.

The canonical closed-form reference is the single-degree-of-freedom resonator
of mass `m`, viscous damping `c` and stiffness `k`, whose receptance is
$H(\omega) = 1 / (k - \omega^2 m + j \omega c)$ (consistent with the
ISO 7626-1
Table 1 / 3.1.2 definitions). At its resonance $\omega_0 = \sqrt{k/m}$
the driving-point mobility is purely real and equal to $1/c$ -- the
mobility peak
measures the damping. This module is the FRF backbone for the structure-borne
source and transmission standards (ISO 9611, ISO 10846, EN 15657, EN 12354-5).

**Measured FRFs (ISO 7626-2:2015).** The single-point measurement side is
covered by the library's spectral estimators: processing of random-excitation
records per ISO 7626-2, 8.1.3 -- the H1 estimator
$H = G(\text{response},\text{force}) / G(\text{force},\text{force})$
-- and the ordinary coherence
$\gamma^2 = |G_{xy}|^2 / (G_{xx} G_{yy})$
used for its data-quality checks are
[`phonometry.electroacoustics.frequency_response.transfer_function`](/phonometry/reference/api/electroacoustics/frequency-response/#transfer_function) (with
`estimator="H1"`, the default) and
[`phonometry.electroacoustics.frequency_response.coherence`](/phonometry/reference/api/electroacoustics/frequency-response/#coherence); both are
exported at the package top level. This module adds the ISO 7626-2 acceptance
criteria on top of them: the operational rigid-mass calibration of 7.5.2
([`rigid_mass_calibration_check`](/phonometry/reference/api/vibration/mechanical-mobility/#rigid_mass_calibration_check), +/- 5 %) and the Annex A normalized
random error with its \< 5 % averaging criterion ([`random_error_percent`](/phonometry/reference/api/vibration/mechanical-mobility/#random_error_percent)).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## convert_frf

```python
convert_frf(
    value: ArrayLike,
    frequency: ArrayLike,
    source: str,
    target: str,
) -> np.ndarray
```

Convert a frequency-response function between any two kinds (Table 1).

:::note
The force-per-motion kinds returned here are arithmetic reciprocals of
the motion-per-force FRFs: the **free** quantities of ISO 7626-1,
3.1.4 (all other response coordinates unconstrained). They coincide
with the **blocked** matrix quantities of Table 1 only for a scalar
(single-coordinate) system: blocked matrices do not invert
element-wise, $Z_{ij} \ne 1/Y_{ij}$ in general. For
driving-point or
single-path use (e.g. the ISO 10846-1 Table A.2 relations) the free
forms are exactly what is needed; for multi-coordinate blocked
quantities invert the full FRF matrix instead. ISO 7626-1 Table 1
names `F/a` the **effective mass** (also known as apparent mass,
the name used here).
:::

**Parameters**

| Name | Description |
| :--- | :--- |
| `value` | The (complex) FRF value(s) of kind *source*. |
| `frequency` | Frequency `f`, in hertz (scalar or array, broadcast with *value*). |
| `source` | The FRF kind of *value* -- one of `"receptance"`, `"mobility"`, `"accelerance"`, `"dynamic_stiffness"`, `"impedance"` or `"apparent_mass"`. |
| `target` | The FRF kind to convert to (same set). |

**Returns:** The FRF value(s) of kind *target*, as a complex array.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for an unknown FRF name, a non-positive frequency, or a zero *value* (dead channel) when the conversion involves a force-per-motion reciprocal. |

## FRF_UNITS

*Constant* (`dict`).

```python
FRF_UNITS = {'receptance': 'm/N', 'mobility': 'm/(N·s)', 'accelerance': '1/kg', 'dynamic_stiffness': 'N/m', 'impedance': 'N·s/m', 'apparent_mass': 'kg'}
```

## MobilityResult

```python
MobilityResult(
    frequencies: np.ndarray,
    mobility: np.ndarray,
    driving_point: bool = True,
)
```

A measured or modelled mobility FRF over frequency.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies, in hertz. |
| `mobility` | Complex mobility `Y` per frequency, in m/(N.s). |
| `driving_point` | `True` if response and force are co-located (i = j). |

### MobilityResult.magnitude

*property*

Mobility magnitude `|Y|`, in m/(N.s).

### MobilityResult.phase

*property*

Mobility phase, in radians.

### MobilityResult.plot()

```python
MobilityResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the mobility magnitude `|Y(f)|`.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### MobilityResult.report()

```python
MobilityResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render a mechanical-mobility measurement fiche to a PDF (ISO 7626).

Writes a one-page mobility measurement report: the standard-basis line
naming whether the mobility is a driving-point or a transfer FRF
(ISO 7626-1:2011 definitions; measurement per ISO 7626-2:2015), an
optional metadata header, a two-panel body with a compact table of the
FRF's characteristic points (the FRF type, the frequency range, the peak
frequency, the peak mobility magnitude and the phase there) beside the
mobility magnitude spectrum `|Y(f)|`, a boxed peak mobility `|Y|`
with the frequency it occurs at, and a footer identity/disclaimer block.

Mechanical mobility is a continuous frequency-response function, so the
fiche presents it as a spectrum plus a table of characteristic points; a
mobility measurement is a characterisation, so there is no pass/fail
verdict.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header identity (`specimen` is the tested structure) and the footer identity; the `requirement` field is ignored. |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | Accepted for a uniform `.report()` signature; the mobility fiche has a single body layout, so it has no effect. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is unknown. |
| ImportError | If reportlab or matplotlib is not installed. The fiche always embeds the mobility spectrum, so both are required (`pip install "phonometry[report,plot]"`). |

### MobilityResult.to()

```python
MobilityResult.to(target: str) -> np.ndarray
```

Convert the mobility to another FRF kind (see [`convert_frf`](/phonometry/reference/api/vibration/mechanical-mobility/#convert_frf)).

The force-per-motion kinds are element-wise reciprocals, i.e. the
*free* quantities of ISO 7626-1, 3.1.4: on a transfer FRF
(`driving_point=False`), `to("impedance")` returns the free
impedance $1/Y_{ij}$, not the blocked matrix impedance of Table 1
(which does not invert element-wise).

## random_error_percent

```python
random_error_percent(coherence: ArrayLike, n_averages: int) -> np.ndarray
```

Normalized random error of an averaged FRF magnitude (ISO 7626-2 Annex A).

$\epsilon = \sqrt{(1 - \gamma^2) / (2 n \gamma^2)}$, the normalized
random
error of the frequency-response-function magnitude estimated from `n`
averaged spectra with ordinary coherence $\gamma^2$ (the relation
behind
Figure A.2, from Bendat & Piersol). ISO 7626-2, 8.1.3 requires enough
averages that the error at each resonance of a driving-point mobility is
below 5 %: e.g. $\gamma^2 = 0.8$ needs about $n = 75$ spectra
($\epsilon = 4.08$ %), the Annex A example.

**Parameters**

| Name | Description |
| :--- | :--- |
| `coherence` | Ordinary coherence $\gamma^2$ per frequency, in (0, 1]. |
| `n_averages` | Number of averaged spectra `n` (>= 1). |

**Returns:** The normalized random error, in percent (same shape as input).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a coherence outside (0, 1] or `n_averages < 1`. |

## resonance_frequency

```python
resonance_frequency(mass: float, stiffness: float) -> float
```

Undamped natural frequency $f_0 = (1/2\pi) \sqrt{k/m}$ of the
SDOF, in Hz.

**Parameters**

| Name | Description |
| :--- | :--- |
| `mass` | Mass `m`, in kg. |
| `stiffness` | Stiffness `k`, in N/m. |

**Returns:** The natural frequency `f0`, in hertz.

## rigid_mass_calibration_check

```python
rigid_mass_calibration_check(
    frf: ArrayLike,
    frequencies: ArrayLike,
    mass: float,
    *,
    quantity: str = 'accelerance',
    tolerance: float = 0.05,
) -> RigidMassCalibrationResult
```

Check an operational calibration on a rigid mass (ISO 7626-2, 7.5.2).

The measured frequency response of a freely suspended rigid calibration
block of known mass `m` shall agree within +/- 5 % with its known correct
value: the accelerance magnitude $\lvert A \rvert = 1/m$ or the
mobility magnitude
$\lvert Y \rvert = 1/(2 \pi f m)$. All components of the
measurement chain (including
the attachment hardware) are connected as in the test series, so a failure
flags transducer, chain or attachment-compliance errors.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frf` | Measured calibration FRF (complex or magnitude, scalar or array), in 1/kg (accelerance) or m/(N.s) (mobility). |
| `frequencies` | Frequencies of *frf*, in hertz (> 0, same shape). |
| `mass` | Known mass `m` of the calibration block, in kg (> 0). |
| `quantity` | `"accelerance"` ($\lvert A \rvert = 1/m$) or `"mobility"` ($\lvert Y \rvert = 1/(\omega m)$). (Default: `"accelerance"`.) |
| `tolerance` | Relative tolerance (Default: 0.05, the +/- 5 % of 7.5.2). |

**Returns:** A [`RigidMassCalibrationResult`](/phonometry/reference/api/vibration/mechanical-mobility/#rigidmasscalibrationresult) with per-band pass flags.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for an unknown quantity, non-positive mass, tolerance or frequency, or mismatched shapes. |

## RigidMassCalibrationResult

```python
RigidMassCalibrationResult(
    frequencies: np.ndarray,
    measured: np.ndarray,
    expected: np.ndarray,
    deviation: np.ndarray,
    within_tolerance: np.ndarray,
    passed: bool,
    mass: float,
    quantity: str,
    tolerance: float,
)
```

Operational rigid-mass calibration check (ISO 7626-2:2015, 7.5.2).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies of the calibration FRF, in hertz. |
| `measured` | Measured FRF magnitude per frequency (`1/kg` for accelerance, `m/(N.s)` for mobility). |
| `expected` | Known correct magnitude of the rigid calibration block per frequency: $1/m$ (accelerance) or $1/(2 \pi f m)$ (mobility). |
| `deviation` | Relative deviation `measured/expected - 1` per frequency. |
| `within_tolerance` | Per-frequency pass flag `\|deviation\| <= tolerance`. |
| `passed` | `True` if every frequency is within the tolerance. |
| `mass` | Mass `m` of the calibration block, in kg. |
| `quantity` | FRF kind checked (`"accelerance"` or `"mobility"`). |
| `tolerance` | Relative tolerance applied (the standard's is 0.05). |

### RigidMassCalibrationResult.plot()

```python
RigidMassCalibrationResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes | np.ndarray
```

Plot the rigid-mass calibration check (ISO 7626-2, 7.5.2).

The measured FRF magnitude against the known rigid-mass line with its
tolerance band, and the relative deviation against the same band. With
no `ax` a two-panel figure is drawn and its axes array returned; with
`ax` the deviation diagnostic is drawn on it and that axes returned.

Requires matplotlib (`pip install phonometry[plot]`).

## sdof_accelerance

```python
sdof_accelerance(
    frequency: ArrayLike,
    mass: float,
    stiffness: float,
    damping: float,
) -> np.ndarray
```

Accelerance of a viscously damped SDOF resonator: $A = -\omega^2 H$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency `f`, in hertz. |
| `mass` | Mass `m`, in kg. |
| `stiffness` | Stiffness `k`, N/m. |
| `damping` | Viscous damping `c`, N.s/m. |

**Returns:** The complex accelerance `A`, in 1/kg.

## sdof_mobility

```python
sdof_mobility(
    frequency: ArrayLike,
    mass: float,
    stiffness: float,
    damping: float,
) -> np.ndarray
```

Mobility of a viscously damped SDOF resonator: $Y = j \omega H$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency `f`, in hertz. |
| `mass` | Mass `m`, in kg. |
| `stiffness` | Stiffness `k`, N/m. |
| `damping` | Viscous damping `c`, N.s/m. |

**Returns:** The complex mobility `Y`, in m/(N.s).

## sdof_mobility_result

```python
sdof_mobility_result(
    frequency: ArrayLike,
    mass: float,
    stiffness: float,
    damping: float,
) -> MobilityResult
```

SDOF driving-point mobility bundled as a [`MobilityResult`](/phonometry/reference/api/vibration/mechanical-mobility/#mobilityresult).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequencies `f`, in hertz (array). |
| `mass` | Mass `m`, in kg. |
| `stiffness` | Stiffness `k`, N/m. |
| `damping` | Viscous damping `c`, N.s/m. |

**Returns:** The [`MobilityResult`](/phonometry/reference/api/vibration/mechanical-mobility/#mobilityresult) (driving point).

## sdof_receptance

```python
sdof_receptance(
    frequency: ArrayLike,
    mass: float,
    stiffness: float,
    damping: float,
) -> np.ndarray
```

Receptance of a viscously damped SDOF resonator (closed form).

$H(\omega) = 1 / (k - \omega^2 m + j \omega c)$, the textbook
single-degree-of-freedom reference, expressed in the FRF taxonomy of
ISO 7626-1 (Table 1 / 3.1.2 definitions).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency `f`, in hertz (scalar or array). |
| `mass` | Mass `m`, in kg. |
| `stiffness` | Stiffness `k`, in N/m. |
| `damping` | Viscous damping coefficient `c`, in N.s/m (>= 0). |

**Returns:** The complex receptance `H`, in m/N.
