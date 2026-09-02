---
title: "hearing.hearing_protectors"
description: "What a hearing protector leaves at the ear (ISO 4869-2:2018)."
sidebar:
  label: "hearing_protectors"
---

What a hearing protector leaves at the ear (ISO 4869-2:2018).

A protector is measured on people, not on a coupler: ISO 4869-1 seats it on at
least 16 subjects and records the threshold shift it produces in each octave
band. What comes out is a **distribution**, one attenuation per subject per
band, and ISO 4869-2 is the standard that turns that distribution into a level
someone can act on.

**The distribution first (Clause 5).** Every method here starts from the
assumed protection value, the mean attenuation reduced by a multiple of its own
spread (Formula (1)):

$$
APV_{fx} = m_f - \alpha\, s_f
$$

The constant $\alpha$ is the inverse standard normal cumulative
distribution at the protection performance $x$ (Table 1), so
$APV_{f84}$ with $\alpha = 1$ is the attenuation 84 % of wearers
reach or beat, and $APV_{f98}$ with $\alpha = 2$ is what all but
one in fifty reach. A protector is never quoted at its mean.

**Then one of three methods**, in decreasing order of what they need to know
about the noise:

- The **octave-band method** (Clause 6) subtracts the assumed protection value
  band by band from the A-weighted noise spectrum, Formula (2). It needs the
  spectrum and is the most faithful.
- The **HML method** (Clause 7) collapses the protector to three numbers, its
  high-, medium- and low-frequency attenuation values, each the predicted noise
  level reduction for a reference noise of a stated $(L_{p,C} - L_{p,A})$.
  It needs only the C- and A-weighted levels of the noise.
- The **SNR method** (Clause 8) collapses it to one number against a pink noise
  and subtracts it from the C-weighted level. It needs only that level.

The three answer the same question and rarely agree exactly: on the worked
example of Annexes B, C and D the same protector in the same noise gives 81 dB,
82 dB and 82 dB. Clause 1's own NOTE puts differences of 3 dB or less between
comparable protectors below the resolution of the exercise.

All three computations begin at 125 Hz. Formula (2) may start at 63 Hz when
both the noise and the protector have data there, but the `HML` and `SNR`
computations always start at 125 Hz regardless (Clause 6), which is why the
reference spectra of Tables 2 and 3 begin there.

Clause, formula and table numbers refer to ISO 4869-2:2018(E).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## assumed_protection_value

```python
assumed_protection_value(
    attenuation: Sequence[Sequence[float]] | np.ndarray,
    *,
    performance: int = 84,
    frequencies: Sequence[float] | np.ndarray | None = None,
) -> AssumedProtectionResult
```

Assumed protection values of a hearing protector (Formula (1)).

The attenuation of a protector is a distribution over people, and Clause 5
reduces it to the level a stated share of wearers reaches or beats:

$$
APV_{fx} = m_f - \alpha\, s_f \tag{1}
$$

with $m_f$ and $s_f$ the mean and standard deviation of the
per-subject attenuations of ISO 4869-1 and $\alpha$ the inverse
standard normal cumulative distribution at the protection performance
(Table 1). The standard deviation is the sample one, over `N - 1`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `attenuation` | A `(subjects, bands)` grid of sound attenuation values, in dB, one row per subject, measured to ISO 4869-1. |
| `performance` | The protection performance `x`, in per cent, from Table 1: 50, 75, 80, 84 (the default), 90, 95 or 98. |
| `frequencies` | Octave-band mid-frequencies, in hertz, or `None` for the eight bands of Formula (2) when the grid has eight columns. |

**Returns:** [`AssumedProtectionResult`](/phonometry/reference/api/hearing/hearing-protectors/#assumedprotectionresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if the grid is not two-dimensional or holds fewer than two subjects, if any value is not finite, if `performance` is not one Table 1 tabulates, or if `frequencies` does not match the grid. |

:::note
Annex A prints its `APV` row as the difference of the *rounded*
`m_f` and `s_f` it displays above it, which differs from Formula (1)
applied to the underlying data by 0,1 dB in three of its eight bands.
This returns Formula (1) applied to the data; round afterwards if you
need the annex's table back.
:::

## AssumedProtectionResult

```python
AssumedProtectionResult(
    apv: np.ndarray,
    mean_attenuation: np.ndarray,
    standard_deviation: np.ndarray,
    performance: int,
    alpha: float,
    frequencies: np.ndarray,
    subjects: int,
)
```

Assumed protection values of a hearing protector (Clause 5).

**Attributes**

| Name | Description |
| :--- | :--- |
| `apv` | $APV_{fx} = m_f - \alpha s_f$ per octave band, in dB. |
| `mean_attenuation` | The mean attenuation $m_f$ per band, in dB. |
| `standard_deviation` | The standard deviation $s_f$ per band, in dB, over the test subjects. |
| `performance` | The protection performance `x`, in per cent. |
| `alpha` | The Table 1 constant that `performance` selected. |
| `frequencies` | Octave-band mid-frequencies, in hertz. |
| `subjects` | Number of test subjects the distribution came from. |

### AssumedProtectionResult.plot()

```python
AssumedProtectionResult.plot(
    ax: Axes | None = None,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the mean attenuation, its spread and the assumed protection.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the `APV` curve. |

**Returns:** The axes.

## hml_protected_level

```python
hml_protected_level(
    l_p_a: float,
    l_p_c: float,
    rating: HMLRatingResult,
) -> ProtectedLevelResult
```

Effective A-weighted level by the `HML` method (Formulas (16) to (18)).

Two straight segments through the three anchors, in
$(L_{p,C} - L_{p,A})$:

$$
PNR_x = M_x - \frac{H_x - M_x}{4}(L_{p,C} - L_{p,A} - 2\ \mathrm{dB}) \quad\text{for } (L_{p,C} - L_{p,A}) \leq 2\ \mathrm{dB} \tag{16}
$$

$$
PNR_x = M_x - \frac{M_x - L_x}{8}(L_{p,C} - L_{p,A} - 2\ \mathrm{dB}) \quad\text{for } (L_{p,C} - L_{p,A}) > 2\ \mathrm{dB} \tag{17}
$$

$$
L'_{p,Ax} = L_{p,A} - PNR_x \tag{18}
$$

Both branches pass through $M_x$ at $+2$ dB, which is where the
medium-frequency value is defined (Clause 3.6). The three values that enter
them are the **rounded** ones: Clause 7.2 rounds $H_x$, $M_x$
and $L_x$ to the nearest integer, so that is what a protector is
published with and what this consumes, whatever the unrounded fit behind
them was. Clause 7.3 allows the unweighted level in place of the
C-weighted one, which for very low-frequency noise returns a higher, safer
$L'_{p,Ax}$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `l_p_a` | A-weighted sound pressure level of the noise, in dB. |
| `l_p_c` | C-weighted sound pressure level of the noise, in dB. |
| `rating` | The protector's [`HMLRatingResult`](/phonometry/reference/api/hearing/hearing-protectors/#hmlratingresult). |

**Returns:** [`ProtectedLevelResult`](/phonometry/reference/api/hearing/hearing-protectors/#protectedlevelresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if either level is not finite. |

## hml_rating

```python
hml_rating(
    attenuation: Sequence[Sequence[float]] | np.ndarray,
    *,
    performance: int = 84,
) -> HMLRatingResult
```

The `H`, `M` and `L` values of a protector (Formulas (3) to (15)).

Clause 7.2 fits the protector against the eight reference noises of
Table 2. For each subject and each noise it forms the predicted noise level
reduction

$$
PNR_{ji} = 100\ \mathrm{dB} - 10 \lg \sum_{k=2}^{8} 10^{0,1\left(L_{p,\mathrm{A}f(k)i} - a_{jf(k)}\right)} \mathrm{dB} \tag{15}
$$

and collapses the eight into three, weighted by the empirical constants
$d_i$ of Table 2:

$$
H_j = 0{,}25 \sum_{i=1}^{4} PNR_{ji} - 0{,}48 \sum_{i=1}^{4} d_i PNR_{ji} \tag{12}
$$

$$
M_j = 0{,}25 \sum_{i=5}^{8} PNR_{ji} - 0{,}16 \sum_{i=5}^{8} d_i PNR_{ji} \tag{13}
$$

$$
L_j = 0{,}25 \sum_{i=5}^{8} PNR_{ji} + 0{,}23 \sum_{i=5}^{8} d_i PNR_{ji} \tag{14}
$$

The four quiet-spectrum noises carry `H` and the four loud-spectrum ones
carry `M` and `L`, which is the split the formulas print. Each index is
then reduced by its own spread across subjects exactly as Formula (1)
reduces the attenuation (Formulas (3) to (11)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `attenuation` | A `(subjects, bands)` grid of sound attenuation values, in dB. Formula (15) reads 125 Hz to 8000 Hz, so a grid of eight bands is taken to start at 63 Hz and its first column is dropped; a grid of seven is taken to start at 125 Hz. |
| `performance` | The protection performance `x`, in per cent, from Table 1. |

**Returns:** [`HMLRatingResult`](/phonometry/reference/api/hearing/hearing-protectors/#hmlratingresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if the grid is not two-dimensional, holds fewer than two subjects, carries neither seven nor eight bands, or if `performance` is not one Table 1 tabulates. |

## HML_REFERENCE_C_MINUS_A

*Constant* (`tuple`).

```python
HML_REFERENCE_C_MINUS_A = (-1.2, -0.5, 0.1, 1.6, 2.3, 4.3, 6.1, 8.4)
```

## HML_REFERENCE_D

*Constant* (`tuple`).

```python
HML_REFERENCE_D = (-1.2, -0.49, 0.14, 1.56, -2.98, -1.01, 0.85, 3.14)
```

## HML_REFERENCE_NOISES

*Constant* (`tuple`).

```python
HML_REFERENCE_NOISES = ((62.6, 70.8, 81.0, 90.4, 96.2, 94.7, 92.3), (68.9, 78.3, 84.3, 92.8, 96.3, 94.0, 90.0), (71.1, 80.8, 88.0, 95.0, 94.4, 94.1, 89.0), (77.2, 84.5, 89.8, 95.5, 94.3, 92.5, 88.8), (77.4, 86.5, 92.5, 96.4, 93.0, 90.4, 83.7), (82.0, 89.3, 93.3, 95.6, 93.0, 90.1, 83.0), (84.2, 90.1, 93.6, 96.2, 91.3, 87.9, 81.9), (88.0, 93.4, 93.8, 94.2, 91.4, 87.9, 79.9))
```

## HMLRatingResult

```python
HMLRatingResult(
    high: float,
    medium: float,
    low: float,
    subject_h: np.ndarray,
    subject_m: np.ndarray,
    subject_l: np.ndarray,
    predicted_reduction: np.ndarray,
    performance: int,
    alpha: float,
)
```

The three `HML` attenuation values of a protector (Clause 7.2).

**Attributes**

| Name | Description |
| :--- | :--- |
| `high` | $H_x$, the high-frequency value, in dB, unrounded. |
| `medium` | $M_x$, the medium-frequency value, in dB, unrounded. |
| `low` | $L_x$, the low-frequency value, in dB, unrounded. |
| `subject_h` | $H_j$ per test subject, in dB (Formula (12)). |
| `subject_m` | $M_j$ per test subject, in dB (Formula (13)). |
| `subject_l` | $L_j$ per test subject, in dB (Formula (14)). |
| `predicted_reduction` | $PNR_{ji}$ per subject and reference noise, in dB, a `(subjects, 8)` grid (Formula (15)). |
| `performance` | The protection performance `x`, in per cent. |
| `alpha` | The Table 1 constant that `performance` selected. |

### HMLRatingResult.plot()

```python
HMLRatingResult.plot(
    ax: Axes | None = None,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the predicted noise level reduction against `LpC - LpA`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the `PNR` curve. |

**Returns:** The axes.

### HMLRatingResult.reported

*property*

`(H, M, L)` rounded the way Clause 7.2 reports them.

**Returns:** The three values as integers, halves away from zero.

## octave_band_protected_level

```python
octave_band_protected_level(
    noise_levels: Sequence[float] | np.ndarray,
    apv: Sequence[float] | np.ndarray | AssumedProtectionResult,
    *,
    frequencies: Sequence[float] | np.ndarray | None = None,
    a_weighting: Sequence[float] | np.ndarray | None = None,
) -> ProtectedLevelResult
```

Effective A-weighted level by the octave-band method (Formula (2)).

The most faithful of the three methods, and the only one that sees the
shape of the noise:

$$
L'_{p,Ax} = 10 \lg \sum_{k=1}^{8} 10^{0,1\left(L_{p,f(k)} + A_{f(k)} - APV_{f(k)x}\right)} \mathrm{dB} \tag{2}
$$

The summation runs over the eight octaves from 63 Hz, or over seven from
125 Hz when 63 Hz data is missing for either the noise or the protector
(Clause 6). Pass seven values to both arguments for that case.

**Parameters**

| Name | Description |
| :--- | :--- |
| `noise_levels` | Octave-band sound pressure levels of the noise, $L_{p,f(k)}$, in dB. Unweighted: the A weighting is added here. |
| `apv` | Assumed protection values, in dB, or the [`AssumedProtectionResult`](/phonometry/reference/api/hearing/hearing-protectors/#assumedprotectionresult) that carries them. |
| `frequencies` | Octave-band mid-frequencies, in hertz, or `None` for the eight bands of Formula (2), or those seven without 63 Hz. |
| `a_weighting` | Frequency weighting A at those bands, in dB, or `None` for [`PROTECTOR_A_WEIGHTING`](/phonometry/reference/api/hearing/hearing-protectors/#protector_a_weighting), which is IEC 61672-1:2013 Table 3. |

**Returns:** [`ProtectedLevelResult`](/phonometry/reference/api/hearing/hearing-protectors/#protectedlevelresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if the band counts disagree, if any value is not finite, or if the bands are neither the eight of Formula (2) nor those seven without 63 Hz and `frequencies` was not given. |

## PINK_NOISE_A_WEIGHTED

*Constant* (`tuple`).

```python
PINK_NOISE_A_WEIGHTED = (75.9, 83.4, 88.8, 92.0, 93.2, 93.0, 90.9)
```

## ProtectedLevelResult

```python
ProtectedLevelResult(
    effective_level: float,
    noise_reduction: float,
    performance: int | None,
    method: str,
    band_levels: np.ndarray | None = None,
    frequencies: np.ndarray | None = None,
)
```

The A-weighted level left at the ear behind a protector.

**Attributes**

| Name | Description |
| :--- | :--- |
| `effective_level` | $L'_{p,Ax}$, in dB, unrounded. Clauses 6, 7.3 and 8.3 all report it to the nearest integer, which `reported_level` does. |
| `noise_reduction` | $PNR_x = L_{p,A} - L'_{p,Ax}$, in dB. |
| `performance` | The protection performance `x`, in per cent, or `None` when the rating that produced it did not carry one. |
| `method` | `"octave-band"`, `"HML"` or `"SNR"`. |
| `band_levels` | The A-weighted band levels behind the protector, in dB, for the octave-band method, and `None` for the other two, which never see a spectrum. |
| `frequencies` | Octave-band mid-frequencies, in hertz, or `None`. |

### ProtectedLevelResult.plot()

```python
ProtectedLevelResult.plot(
    ax: Axes | None = None,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the band levels the protector leaves, where there are any.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the protected-level bars. |

**Returns:** The axes.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for an `HML` or `SNR` result, which carries no spectrum to draw. |

### ProtectedLevelResult.reported_level

*property*

`effective_level` rounded the way the standard reports it.

**Returns:** The nearest integer, halves away from zero.

## PROTECTION_PERFORMANCES

*Constant* (`dict`).

```python
PROTECTION_PERFORMANCES = {50: 0.0, 75: 0.67, 80: 0.84, 84: 1.0, 90: 1.28, 95: 1.64, 98: 2.0}
```

## PROTECTOR_A_WEIGHTING

*Constant* (`tuple`).

```python
PROTECTOR_A_WEIGHTING = (-26.2, -16.1, -8.6, -3.2, 0.0, 1.2, 1.0, -1.1)
```

## PROTECTOR_OCTAVE_BANDS

*Constant* (`tuple`).

```python
PROTECTOR_OCTAVE_BANDS = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
```

## snr_protected_level

```python
snr_protected_level(
    rating: SNRRatingResult,
    *,
    l_p_c: float | None = None,
    l_p_a: float | None = None,
    c_minus_a: float | None = None,
) -> ProtectedLevelResult
```

Effective A-weighted level by the `SNR` method (Formulas (23) and (24)).

$$
L'_{p,Ax} = L_{p,C} - SNR_x \tag{23}
$$

$$
L'_{p,Ax} = L_{p,A} + (L_{p,C} - L_{p,A}) - SNR_x \tag{24}
$$

Formula (24) is Formula (23) with the C-weighted level reassembled from an
A-weighted measurement and an estimate of the difference, for the common
case where only the A-weighted level was recorded. Pass `l_p_c`, or pass
`l_p_a` together with `c_minus_a`. Clause 8.3 allows the unweighted
level in place of the C-weighted one, which for very low-frequency noise
returns a higher, safer $L'_{p,Ax}$.

The rating is used as Clause 8.2 reports it, rounded to the nearest
integer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `rating` | The protector's [`SNRRatingResult`](/phonometry/reference/api/hearing/hearing-protectors/#snrratingresult). |
| `l_p_c` | C-weighted sound pressure level of the noise, in dB, for Formula (23). |
| `l_p_a` | A-weighted sound pressure level of the noise, in dB, for Formula (24). |
| `c_minus_a` | The difference $(L_{p,C} - L_{p,A})$, in dB, for Formula (24). |

**Returns:** [`ProtectedLevelResult`](/phonometry/reference/api/hearing/hearing-protectors/#protectedlevelresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if neither pairing is complete, if both are given, or if any level is not finite. |

## snr_rating

```python
snr_rating(
    attenuation: Sequence[Sequence[float]] | np.ndarray,
    *,
    performance: int = 84,
) -> SNRRatingResult
```

The single number rating of a protector (Formulas (19) to (22)).

One reference noise instead of eight, and one number instead of three:

$$
SNR_j = 100\ \mathrm{dB} - 10 \lg \sum_{k=2}^{8} 10^{0,1\left(L_{p,\mathrm{A}f(k)} - a_{jf(k)}\right)} \tag{22}
$$

$$
SNR_x = SNR_m - \alpha\, SNR_s \tag{19}
$$

where $L_{p,\mathrm{A}f(k)}$ is the pink noise of Table 3, whose
C-weighted level is 100 dB. Because the reference noise is fixed, the
rating says nothing about the shape of the noise it will meet, which is
what the `HML` method's three values recover.

**Parameters**

| Name | Description |
| :--- | :--- |
| `attenuation` | A `(subjects, bands)` grid of sound attenuation values, in dB, over the eight octaves from 63 Hz or the seven from 125 Hz. |
| `performance` | The protection performance `x`, in per cent, from Table 1. |

**Returns:** [`SNRRatingResult`](/phonometry/reference/api/hearing/hearing-protectors/#snrratingresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if the grid is not two-dimensional, holds fewer than two subjects, carries neither seven nor eight bands, or if `performance` is not one Table 1 tabulates. |

## SNRRatingResult

```python
SNRRatingResult(
    snr: float,
    subject_snr: np.ndarray,
    mean: float,
    standard_deviation: float,
    performance: int,
    alpha: float,
)
```

The single number rating of a protector (Clause 8.2).

**Attributes**

| Name | Description |
| :--- | :--- |
| `snr` | $SNR_x$, in dB, unrounded. |
| `subject_snr` | $SNR_j$ per test subject, in dB (Formula (22)). |
| `mean` | $SNR_m$, in dB (Formula (20)). |
| `standard_deviation` | $SNR_s$, in dB (Formula (21)). |
| `performance` | The protection performance `x`, in per cent. |
| `alpha` | The Table 1 constant that `performance` selected. |

### SNRRatingResult.plot()

```python
SNRRatingResult.plot(
    ax: Axes | None = None,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the per-subject ratings the single number was reduced from.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the per-subject bars. |

**Returns:** The axes.

### SNRRatingResult.reported

*property*

`snr` rounded the way Clause 8.2 reports it.

**Returns:** The nearest integer, halves away from zero.
