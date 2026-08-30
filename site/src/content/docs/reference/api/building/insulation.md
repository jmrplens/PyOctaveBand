---
title: "building.measurement.insulation"
description: "Field measurement of sound insulation: airborne (ISO 16283-1:2014), impact (ISO 16283-2:2020) and facade (ISO 16283-3:2016)."
sidebar:
  label: "insulation"
---

Field measurement of sound insulation: airborne (ISO 16283-1:2014), impact
(ISO 16283-2:2020) and facade (ISO 16283-3:2016).

**Field quantities (ISO 16283-1:2014).** From the energy-average sound
pressure levels in the source and receiving rooms this module forms the
level difference $D = L_1 - L_2$ (Clause 3.12, Formula (1)), the
standardized level difference $D_\mathrm{nT} = D + 10 \log_{10}(T/T_0)$ with the
reference reverberation time $T_0 = 0.5$ s (Clause 3.13,
Formula (2)), and the apparent sound reduction index
$R' = D + 10 \log_{10}(S/A)$ with the Sabine equivalent absorption area
$A = 0.16 V / T$ (Clause 3.14/3.15, Formula (4) and (5)). Source and
receiving levels may be supplied already averaged (one value per band) or
as several microphone positions, which are then energy-averaged with
$10 \log_{10}\left( \frac{1}{n} \sum 10^{L_j/10} \right)$ (Clause 7.8.1,
Formula (9)). The quantities are evaluated per one-third-octave band, the
caller having already applied any background-noise correction (Clause 9.2).

**Field impact quantities (ISO 16283-2:2020).** With the tapping machine as
the impact source this module forms, from the energy-average impact sound
pressure level `Li` in the receiving room, the standardized impact sound
pressure level $L'_\mathrm{nT} = L_\mathrm{i} - 10 \log_{10}(T/T_0)$ with
$T_0 = 0.5$ s (Clause
3.13, Formula (1)) and the normalized impact sound pressure level
$L'_\mathrm{n} = L_\mathrm{i} + 10 \log_{10}(A/A_0)$ with the Sabine absorption area
$A = 0.16 V/T$ (Clause 3.14, Formula (2))
and the reference area $A_0 = 10$ m² (Clause 3.15, Formula (3)).
Levels
may be supplied already averaged or as several microphone positions, then
energy-averaged (Clause 7.8.1, Formula (11)).

**Field façade quantities (ISO 16283-3:2016).** With an outdoor sound
source this module forms, from the level 2 m in front of the façade
`L1,2m` and the receiving-room level `L2`, the level difference
$D_{2\mathrm{m}} = L_{1,2\mathrm{m}} - L_2$ (Clause 3.14), its standardized form
$D_{2\mathrm{m,nT}} = D_{2\mathrm{m}} + 10 \log_{10}(T/T_0)$ with $T_0 = 0.5$ s
(Clause 3.15) and
normalized form $D_{2\mathrm{m,n}} = D_{2\mathrm{m}} - 10 \log_{10}(A/A_0)$ with the Sabine
absorption
area $A = 0.16 V/T$ (Clause 3.17) and reference $A_0 = 10$ m²
(Clause 3.16): the global loudspeaker / traffic quantities
`Dls,2m,*` / `Dtr,2m,*`. When a surface level `L1,s` (microphone on
the test element) with the element area `S` and volume are given it
forms the apparent sound reduction index
$R'_{45^\circ} = L_{1,\mathrm{s}} - L_2 + 10 \log_{10}(S/A) - 1.5$ for the
loudspeaker element method
(Clause 3.12) or $R'_\mathrm{tr,s} = L_{1,\mathrm{s}} - L_2 + 10 \log_{10}(S/A) - 3$ for
the
road-traffic element method (Clause 3.13). These quantities are defined by
unnumbered formulas inline in the Clause 3 terms; positions are
energy-averaged with the surface-level formula (Clause 9.5.1, Formula (7)).
The façade quantity is airborne, so its single-number rating uses the
**ISO 717-1 airborne** reference curve and method (Clause 2, Annex F) via
[`weighted_rating`](/phonometry/reference/api/building/ratings/#weighted_rating) unchanged.

**Frequency range.** The three parts require the same 16 core one-third-octave
bands, 100 Hz to 3150 Hz (Part 1 and Part 3 Clause 5, Part 2 Clause 5.1), and
make the low range 50 Hz to 80 Hz and the high range 4000 Hz to 5000 Hz
optional additions. The functions below evaluate whatever bands they are given
and impose no range of their own; it is
[`ratings`](/phonometry/reference/api/building/ratings/) that fixes a range, taking
either those 16 one-third-octave bands or the 5 octave bands from 125 Hz to
2000 Hz, and the `report()` fiche that needs exactly the 16.

When the optional low range **is** measured, the low-frequency procedure of
[`phonometry.building.measurement.low_frequency`](/phonometry/reference/api/building/low-frequency/) becomes mandatory for the
50 Hz, 63 Hz and 80 Hz bands in any room whose volume, to the nearest cubic
metre, is under 25 m³. Each function below takes it through a
[`LowFrequencyProcedure`](/phonometry/reference/api/building/low-frequency/#lowfrequencyprocedure): `source_low_frequency` and
`receiver_low_frequency` for the airborne pair of rooms, `low_frequency`
for the single receiving room of the other two. Computing those three bands
from the default procedure alone in a room that small is not the ISO 16283
quantity, so a function handed the volume and the band centres without a
procedure raises a [`LowFrequencyWarning`](/phonometry/reference/api/building/low-frequency/#lowfrequencywarning) rather than
answering silently. The one exception is the road-traffic façade, which
ISO 16283-3 Clause 6 gives the default procedure and nothing else.

The single-number rating of any of these curves is the subject of
[`phonometry.building.measurement.ratings`](/phonometry/reference/api/building/ratings/), which implements ISO 717-1
(airborne) and ISO 717-2 (impact); the `report()` methods below call it to
box the rating on the field report form.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## airborne_insulation

```python
airborne_insulation(
    l1: Sequence[float] | np.ndarray,
    l2: Sequence[float] | np.ndarray,
    t2: Sequence[float] | np.ndarray,
    *,
    area: float,
    volume: float,
    t0: float = ...,
    frequencies: Sequence[float] | np.ndarray | None = ...,
    source_low_frequency: LowFrequencyProcedure | None = ...,
    receiver_low_frequency: LowFrequencyProcedure | None = ...,
) -> AirborneInsulationResult

airborne_insulation(
    l1: Sequence[float] | np.ndarray,
    l2: Sequence[float] | np.ndarray,
    t2: Sequence[float] | np.ndarray,
    *,
    t0: float = ...,
    frequencies: Sequence[float] | np.ndarray | None = ...,
    source_low_frequency: LowFrequencyProcedure | None = ...,
    receiver_low_frequency: LowFrequencyProcedure | None = ...,
) -> AirborneInsulationResult
```

Field airborne sound insulation per ISO 16283-1:2014.

Computes, per frequency band, the level difference
$D = L_1 - L_2$
(Formula (1)), the standardized level difference
$D_\mathrm{nT} = D + 10 \log_{10}(T/T_0)$ (Formula (2)) and, when the partition
area
and receiving-room volume are given, the apparent sound reduction
index $R' = D + 10 \log_{10}(S/A)$ with $A = 0.16\,V/T$
(Formula (4) and (5)).

`l1` and `l2` may be one value per band (already energy-averaged)
or a two-dimensional `(positions, bands)` array, in which case the
positions are energy-averaged with Formula (9). The band levels are
assumed already corrected for background noise (Clause 9.2).

**Rooms under 25 m³.** Clause 8.1 makes the low-frequency procedure
mandatory for the 50 Hz, 63 Hz and 80 Hz bands "in the source and/or
receiving room when *its* volume is smaller than 25 m³ (calculated to the
nearest cubic metre)", so the two rooms are tested independently and either
may carry a [`LowFrequencyProcedure`](/phonometry/reference/api/building/low-frequency/#lowfrequencyprocedure). Whichever room
carries one has its level at those three bands replaced by
$L_\mathrm{LF}$ (Formula (13)) before `D` is formed. Clause 10.4 is
a receiving-room clause alone, so the 63 Hz octave reverberation time is
taken from `receiver_low_frequency` and required there: a small source
room beside a large receiving room changes `L1` and leaves `t2` as
measured. Note 4 to entry of Clause 3.14 records that `R'` determined
this way no longer maps exactly onto the sound-power ratio of Formula (3).

Clause 8.1 is a *shall*, so a receiving room that rounds below 25 m³ and a
`frequencies` vector naming the three bands are together enough to know
the procedure was owed. When `receiver_low_frequency` is then absent this
function computes `D` from the default procedure alone and raises a
[`LowFrequencyWarning`](/phonometry/reference/api/building/low-frequency/#lowfrequencywarning) saying that those three bands
are not the ISO 16283 quantity. Only the receiving room, because only its
volume is an argument here.

**Parameters**

| Name | Description |
| :--- | :--- |
| `l1` | Source-room sound pressure levels, in dB. |
| `l2` | Receiving-room sound pressure levels, in dB. |
| `t2` | Receiving-room reverberation time per band, in seconds. |
| `area` | Area `S` of the common partition, in m² (optional; required together with `volume` for `R'`). |
| `volume` | Receiving-room volume `V`, in m³ (optional; required together with `area` for `R'`). |
| `t0` | Reference reverberation time `T0`, in seconds (default 0,5 s for dwellings, Clause 3.13). |
| `frequencies` | Band centre frequencies, in Hz; required with either low-frequency argument, ignored otherwise (the result carries no band axis of its own). |
| `source_low_frequency` | Source-room corner measurements and volume (Clause 8). Must not carry a 63 Hz octave reverberation time. |
| `receiver_low_frequency` | Receiving-room corner measurements and volume (Clause 8), which must also carry the 63 Hz octave reverberation time of Clause 10.4. |

**Returns:** [`AirborneInsulationResult`](/phonometry/reference/api/building/insulation/#airborneinsulationresult) with `d`, `dnt` and `r_prime` (the latter `None` unless `area` and `volume` are both given), plus whichever low-frequency records were produced.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the band counts of `l1`, `l2` and `t2` differ, if only one of `area`/`volume` is supplied, if `t2`/`t0` are not positive, if inputs are non-finite, if a low-frequency argument is given without `frequencies`, if `volume` disagrees with the receiving-room procedure, or if the 63 Hz octave reverberation time is missing from the receiving room or present on the source room. A room that does not round below 25 m³ is refused where the corner measurements are described, by [`LowFrequencyProcedure`](/phonometry/reference/api/building/low-frequency/#lowfrequencyprocedure) itself. |

**Warns**

| Warning | When |
| :--- | :--- |
| LowFrequencyWarning | when the receiving room rounds below 25 m³ and `frequencies` names the three bands but no `receiver_low_frequency` answers Clause 8.1. |

## AirborneInsulationResult

```python
AirborneInsulationResult(
    d: np.ndarray,
    dnt: np.ndarray,
    r_prime: np.ndarray | None,
    l1: np.ndarray | None = None,
    l2: np.ndarray | None = None,
    t2: np.ndarray | None = None,
    t0: float | None = None,
    source_low_frequency: LowFrequencyResult | None = None,
    receiver_low_frequency: LowFrequencyResult | None = None,
)
```

Per-band field airborne sound insulation (ISO 16283-1:2014).

**Attributes**

| Name | Description |
| :--- | :--- |
| `d` | Level difference $D = L_1 - L_2$ per band, in dB (Clause 3.12, Formula (1)). |
| `dnt` | Standardized level difference `DnT` per band, in dB (Clause 3.13, Formula (2)). |
| `r_prime` | Apparent sound reduction index `R'` per band, in dB (Clause 3.14, Formula (4)), or `None` when the partition area and receiving-room volume were not supplied. |
| `l1` | Energy-average source-room levels the quantities were formed from, in dB (after any position averaging, Formula (9)). Defaults to `None` for backward-compatible construction. |
| `l2` | Energy-average receiving-room levels, in dB. Defaults to `None`. |
| `t2` | Receiving-room reverberation time per band, in seconds, after any 63 Hz octave substitution. Defaults to `None`. |
| `t0` | Reference reverberation time `T0` used for `DnT`, in seconds. Defaults to `None`. |
| `source_low_frequency` | What the ISO 16283-1 Clause 8 low-frequency procedure did to `l1`, or `None` when the source room was not under 25 m³ or carried no corner measurements. |
| `receiver_low_frequency` | The same for `l2` and, through Clause 10.4, for `t2`. |

### AirborneInsulationResult.plot()

```python
AirborneInsulationResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the per-band insulation quantities (`DnT`, `D`, `R'`).

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### AirborneInsulationResult.report()

```python
AirborneInsulationResult.report(
    path: str,
    *,
    quantity: str = 'dnt',
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ISO 16283-1 field airborne sound-insulation report to a PDF.

Writes the one-page field test report of ISO 16283-1:2014 Clause 14
in the layout of the recommended Annex B form: the standard-basis
line, an optional metadata header block (client, construction, room
volumes, partition area ...), the one-third-octave table beside the
measured-versus-shifted-reference curve, the boxed field rating
`DnT,w (C; Ctr)` or `R'w (C; Ctr)` (evaluated per ISO 717-1 over
the 16 core bands), the mandatory field-method statement, an optional
verdict row and a footer with the identity block and disclaimer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `quantity` | The reported field quantity: `"dnt"` (default, the standardized level difference of Annex B Figure B.1) or `"r_prime"` (the apparent sound reduction index of Figure B.2; requires the result to carry `r_prime`). |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a lightweight fiche (body, rating and disclaimer only). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the table shows the measurement chain per band (energy-average `L1` and `L2`, reverberation time `T` and the quantity) instead of the two-column `f \| value` form; it requires the result to carry `l1`, `l2` and `t2` (populated by [`airborne_insulation`](/phonometry/reference/api/building/insulation/#airborne_insulation)). |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` or `quantity` is unknown, the selected quantity is not available, the result does not hold the 16 core one-third-octave bands (100 Hz to 3150 Hz) the ISO 717-1 rating needs, or `verbose=True` without the per-band chain. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |

## energy_average_level

```python
energy_average_level(
    levels: Sequence[float] | np.ndarray,
    axis: int = -1,
) -> np.ndarray | float
```

Energy-average sound pressure level (ISO 16283-1:2014, Formula (9)).

Combines sound pressure levels measured at several microphone
positions into
$L = 10 \log_{10}\left( \frac{1}{n} \sum_j 10^{L_j/10} \right)$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | Sound pressure levels, in dB, at the `n` positions to be averaged along `axis`. |
| `axis` | Axis over which to average (default the last axis). |

**Returns:** The energy-average level, in dB; a scalar `float` when the result is zero-dimensional, otherwise an array.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `levels` is empty or contains non-finite values. |

## facade_insulation

```python
facade_insulation(
    l1_2m: Sequence[float] | np.ndarray,
    l2: Sequence[float] | np.ndarray,
    t2: Sequence[float] | np.ndarray,
    *,
    area: float | None = None,
    volume: float | None = None,
    surface_level: Sequence[float] | np.ndarray | None = None,
    method: str = 'loudspeaker',
    t0: float = 0.5,
    frequencies: Sequence[float] | np.ndarray | None = None,
    low_frequency: LowFrequencyProcedure | None = None,
) -> FacadeInsulationResult
```

Field façade sound insulation per ISO 16283-3:2016.

Computes, per frequency band, the global-method level difference
$D_{2\mathrm{m}} = L_{1,2\mathrm{m}} - L_2$ (Clause 3.14), its standardized form
$D_{2\mathrm{m,nT}} = D_{2\mathrm{m}} + 10 \log_{10}(T/T_0)$ (Clause 3.15) and, when the
receiving-room volume is given, its normalized form
$D_{2\mathrm{m,n}} = D_{2\mathrm{m}} - 10 \log_{10}(A/A_0)$ with the Sabine equivalent
absorption
area $A = 0.16\,V/T$ (Clause 3.17) and $A_0 = 10$ m²
(Clause 3.16).
When a surface level `L1,s` (microphone on the test element),
together with the element area `S` and the volume, is supplied it
also computes the apparent sound reduction index of the element
method: $R'_{45^\circ} = L_{1,\mathrm{s}} - L_2 + 10 \log_{10}(S/A) - 1.5$ for a
loudspeaker
source (Clause 3.12) or
$R'_\mathrm{tr,s} = L_{1,\mathrm{s}} - L_2 + 10 \log_{10}(S/A) - 3$ for a
road-traffic source (Clause 3.13). The defining formulas are unnumbered
inline in the Clause 3 terms.

`l1_2m`, `l2` and `surface_level` may be one value per band
(already energy-averaged) or a two-dimensional `(positions, bands)`
array, in which case the positions are energy-averaged with the
surface-level formula (Clause 9.5.1, Formula (7)). Band levels are
assumed already corrected for background
noise. The single-number rating uses the ISO 717-1 airborne reference
curve (Annex F); pass the desired 16-band quantity to
[`weighted_rating`](/phonometry/reference/api/building/ratings/#weighted_rating).

**Rooms under 25 m³.** Clause 7.3.1 makes the low-frequency procedure
mandatory for the 50 Hz, 63 Hz and 80 Hz bands in the receiving room once
its volume, to the nearest cubic metre, is under 25 m³, and Clause 8.4 puts
one 63 Hz octave reverberation time in place of the three one-third-octave
ones. Pass both through `low_frequency`. A loudspeaker-method call whose
`volume` rounds below the line and whose `frequencies` names the three
bands, with no `low_frequency` to run the procedure, answers from the
default procedure alone and raises a
[`LowFrequencyWarning`](/phonometry/reference/api/building/low-frequency/#lowfrequencywarning) saying that those three bands
are not the ISO 16283 quantity.

**Parameters**

| Name | Description |
| :--- | :--- |
| `l1_2m` | Outdoor sound pressure levels 2 m in front of the façade, in dB. |
| `l2` | Receiving-room sound pressure levels, in dB. |
| `t2` | Receiving-room reverberation time per band, in seconds. |
| `area` | Area `S` of the test element, in m² (optional; required with `volume` and `surface_level` for `R'`). |
| `volume` | Receiving-room volume `V`, in m³ (optional; required for `D2m,n` and for `R'`). |
| `surface_level` | Outdoor surface level `L1,s` on the test element, in dB (optional; required with `area` and `volume` for `R'`). |
| `method` | `"loudspeaker"` (45° incidence, -1,5 dB) or `"road_traffic"` (all-angle incidence, -3 dB); selects the `R'` correction (Clause 3.12 / 3.13). |
| `t0` | Reference reverberation time `T0`, in seconds (default 0,5 s for dwellings, Clause 3.15). |
| `frequencies` | Optional band centre frequencies, in Hz, carried on the result for plotting; required with `low_frequency`. |
| `low_frequency` | Receiving-room corner measurements, volume and 63 Hz octave reverberation time (Clause 7.3 and Clause 8.4). Loudspeaker methods only: Clause 6 says that for the element and global road traffic methods "only the default procedure shall be used", so a small room measured against traffic is conforming without corners and is neither refused nor warned about. |

**Returns:** [`FacadeInsulationResult`](/phonometry/reference/api/building/insulation/#facadeinsulationresult) with `d_2m`, `d_2m_nt`, `d_2m_n` (`None` unless `volume` is given) and `r_prime` (`None` unless `surface_level`, `area` and `volume` are all given), plus the low-frequency record when one was produced.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If band counts differ, if `method` is unknown, if `t2`/`t0`/`area`/`volume` are not positive, if `area` is given without `surface_level`, if `surface_level` and `area` are given without `volume`, if `frequencies` is given with a shape that differs from the band axis, if inputs are non-finite, if `low_frequency` is given without `frequencies` or with `method="road_traffic"`, if `volume` disagrees with the procedure, or if the 63 Hz octave reverberation time is missing. A room that does not round below 25 m³ is refused where the corner measurements are described, by [`LowFrequencyProcedure`](/phonometry/reference/api/building/low-frequency/#lowfrequencyprocedure) itself. Supplying `surface_level` alone is not an error: `r_prime` simply stays `None`. |

**Warns**

| Warning | When |
| :--- | :--- |
| LowFrequencyWarning | when a loudspeaker-method `volume` rounds below 25 m³ and `frequencies` names the three bands but no `low_frequency` answers Clause 7.3.1. |

## FacadeInsulationResult

```python
FacadeInsulationResult(
    d_2m: np.ndarray,
    d_2m_nt: np.ndarray,
    d_2m_n: np.ndarray | None,
    r_prime: np.ndarray | None,
    frequencies: np.ndarray | None = None,
    method: str = 'loudspeaker',
    low_frequency: LowFrequencyResult | None = None,
)
```

Per-band field façade sound insulation (ISO 16283-3).

**Attributes**

| Name | Description |
| :--- | :--- |
| `d_2m` | Level difference $D_{2\mathrm{m}} = L_{1,2\mathrm{m}} - L_2$ per band, in dB (Clause 3.14; `Dls,2m` loudspeaker, `Dtr,2m` traffic). |
| `d_2m_nt` | Standardized level difference $D_{2\mathrm{m,nT}} = D_{2\mathrm{m}} + 10 \log_{10}(T/T_0)$ per band, in dB (Clause 3.15). |
| `d_2m_n` | Normalized level difference $D_{2\mathrm{m,n}} = D_{2\mathrm{m}} - 10 \log_{10}(A/A_0)$ per band, in dB (Clause 3.16), or `None` when the receiving-room volume was not supplied. |
| `r_prime` | Apparent sound reduction index `R'45°` (loudspeaker, Clause 3.12) or `R'tr,s` (road traffic, Clause 3.13) per band, in dB, or `None` unless a surface level together with the element area and receiving-room volume were supplied. |
| `frequencies` | Band centre frequencies, in Hz, or `None`. |
| `method` | The sound source of the measurement: `"loudspeaker"` (45° incidence) or `"road_traffic"`. It selects which apparent index `r_prime` is (`R'45` or `R'tr,s`) and how the report labels it. |
| `low_frequency` | What the ISO 16283-3 Clause 7.3 low-frequency procedure did to `L2` and, through Clause 8.4, to the reverberation time; `None` when the receiving room was not under 25 m³ or carried no corner measurements. |

### FacadeInsulationResult.plot()

```python
FacadeInsulationResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the per-band façade insulation profile (ISO 16283-3).

Draws the standardized level difference and any other available
quantities (`D2m`, `D2m,n`, `R'`) against frequency. Requires
matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### FacadeInsulationResult.report()

```python
FacadeInsulationResult.report(
    path: str,
    *,
    quantity: str = 'd_2m_nt',
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ISO 16283-3 field facade sound-insulation report to a PDF.

Writes the one-page field facade test report of ISO 16283-3:2016: the
standard-basis line, an optional metadata header block, the
one-third-octave table beside the measured-versus-shifted-reference
curve, the boxed ISO 717-1 field rating of the reported `quantity`
(`D2m,nT,w` for `d_2m_nt`, `D2m,n,w` for `d_2m_n` or, for
`r_prime`, `R'45,w` / `R'tr,s,w` following the result's
`method`, each with `C; Ctr` and evaluated over the 16 core
bands), the mandatory field-method statement, an optional verdict row
and a footer with the identity block and disclaimer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `quantity` | The reported facade quantity: `"d_2m_nt"` (default, the standardized facade level difference `D2m,nT`), `"d_2m_n"` (the normalized facade level difference `D2m,n`; requires the result to carry `d_2m_n`) or `"r_prime"` (the apparent sound reduction index; requires the result to carry `r_prime`). The `r_prime` fiche is labelled `R'45` (Clause 3.12, loudspeaker method) or `R'tr,s` (Clause 3.13, road traffic method) according to the result's `method`. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a lightweight fiche (body, rating and disclaimer only). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the table shows the ISO 717 evaluation per band (the reported quantity, the shifted reference and the unfavourable deviation) instead of the two-column `f \| value` form. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` or `quantity` is unknown, the selected quantity is not available, or the result does not hold the 16 core one-third-octave bands (100 Hz to 3150 Hz) the ISO 717-1 rating needs. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |

## impact_insulation

```python
impact_insulation(
    li: Sequence[float] | np.ndarray,
    t2: Sequence[float] | np.ndarray,
    *,
    volume: float | None = None,
    t0: float = 0.5,
    frequencies: Sequence[float] | np.ndarray | None = None,
    low_frequency: LowFrequencyProcedure | None = None,
) -> ImpactInsulationResult
```

Field impact sound insulation per ISO 16283-2:2020 (tapping machine).

Computes, per frequency band, the standardized impact sound pressure
level $L'_\mathrm{nT} = L_\mathrm{i} - 10 \log_{10}(T/T_0)$ (Formula (1)) and, when the
receiving-room volume is given, the normalized impact sound pressure
level $L'_\mathrm{n} = L_\mathrm{i} + 10 \log_{10}(A/A_0)$ (Formula (3)) with the
Sabine equivalent absorption
area $A = 0.16\,V/T$ (Formula (2)) and the reference absorption
area
$A_0 = 10$ m².

`li` may be one value per band (already energy-averaged) or a
two-dimensional `(positions, bands)` array, in which case the
positions are energy-averaged with Formula (11). The band levels are
assumed already corrected for background noise (Clause 9).

**Rooms under 25 m³.** Clause 8.1 makes the low-frequency procedure
mandatory for the 50 Hz, 63 Hz and 80 Hz bands in the receiving room once
its volume, to the nearest cubic metre, is under 25 m³, and Clause 10.4
puts one 63 Hz octave reverberation time in place of the three
one-third-octave ones. Pass both through `low_frequency`. Part 2 confines
the corner procedure to the tapping machine (the heading of its Clause 8),
which is the only impact source this function models; the rubber ball is
[`phonometry.building.measurement.heavy_impact`](/phonometry/reference/api/building/heavy-impact/).

Clause 8.1 is a *shall*, so a `volume` that rounds below 25 m³ beside a
`frequencies` vector naming the three bands is enough to know the
procedure was owed. With no `low_frequency` to run it, this function
answers from the default procedure alone and raises a
[`LowFrequencyWarning`](/phonometry/reference/api/building/low-frequency/#lowfrequencywarning) saying that those three bands
are not the ISO 16283 quantity.

**Parameters**

| Name | Description |
| :--- | :--- |
| `li` | Energy-average impact sound pressure levels, in dB. |
| `t2` | Receiving-room reverberation time per band, in seconds. |
| `volume` | Receiving-room volume `V`, in m³ (optional; required for `L'n`). |
| `t0` | Reference reverberation time `T0`, in seconds (default 0,5 s for dwellings, Clause 3.13). |
| `frequencies` | Band centre frequencies, in Hz; required with `low_frequency`, ignored otherwise (the result carries no band axis of its own). |
| `low_frequency` | Receiving-room corner measurements, volume and 63 Hz octave reverberation time (Clause 8 and Clause 10.4). |

**Returns:** [`ImpactInsulationResult`](/phonometry/reference/api/building/insulation/#impactinsulationresult) with `l_n_t` and `l_n` (the latter `None` unless `volume` is given), plus the low-frequency record when one was produced.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the band counts of `li` and `t2` differ, if `t2`/`t0`/`volume` are not positive, if inputs are non-finite, if `low_frequency` is given without `frequencies`, if `volume` disagrees with the procedure, or if the 63 Hz octave reverberation time is missing. A room that does not round below 25 m³ is refused where the corner measurements are described, by [`LowFrequencyProcedure`](/phonometry/reference/api/building/low-frequency/#lowfrequencyprocedure) itself. |

**Warns**

| Warning | When |
| :--- | :--- |
| LowFrequencyWarning | when `volume` rounds below 25 m³ and `frequencies` names the three bands but no `low_frequency` answers Clause 8.1. |

## ImpactInsulationResult

```python
ImpactInsulationResult(
    l_n_t: np.ndarray,
    l_n: np.ndarray | None,
    li: np.ndarray | None = None,
    t2: np.ndarray | None = None,
    t0: float | None = None,
    low_frequency: LowFrequencyResult | None = None,
)
```

Per-band field impact sound insulation (ISO 16283-2).

**Attributes**

| Name | Description |
| :--- | :--- |
| `l_n_t` | Standardized impact sound pressure level $L'_\mathrm{nT} = L_\mathrm{i} - 10 \log_{10}(T/T_0)$ per band, in dB (Clause 3.13, Formula (1)). |
| `l_n` | Normalized impact sound pressure level $L'_\mathrm{n} = L_\mathrm{i} + 10 \log_{10}(A/A_0)$ per band, in dB (Clause 3.14, Formula (2)), or `None` when the receiving-room volume was not supplied. |
| `li` | Energy-average impact sound pressure levels the quantities were formed from, in dB (after any position averaging, Formula (11)). Defaults to `None` for backward-compatible construction. |
| `t2` | Receiving-room reverberation time per band, in seconds, after any 63 Hz octave substitution. Defaults to `None`. |
| `t0` | Reference reverberation time `T0` used for `L'nT`, in seconds. Defaults to `None`. |
| `low_frequency` | What the ISO 16283-2 Clause 8 low-frequency procedure did to `li` and, through Clause 10.4, to `t2`; `None` when the receiving room was not under 25 m³ or carried no corner measurements. |

### ImpactInsulationResult.plot()

```python
ImpactInsulationResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the per-band impact levels (`L'nT` and, if present, `L'n`).

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### ImpactInsulationResult.report()

```python
ImpactInsulationResult.report(
    path: str,
    *,
    quantity: str = 'l_n_t',
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ISO 16283-2 field impact sound-insulation report to a PDF.

Writes the one-page field test report of ISO 16283-2:2020 Clause 14
in the layout of the recommended Annex C form: the standard-basis
line, an optional metadata header block, the one-third-octave table
beside the measured-versus-shifted-reference curve, the boxed field
rating `L'nT,w (CI)` or `L'n,w (CI)` (evaluated per ISO 717-2
over the 16 core bands), the mandatory field-method statement, an
optional verdict row and a footer with the identity block and
disclaimer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `quantity` | The reported field quantity: `"l_n_t"` (default, the standardized level of Annex C Figure C.1) or `"l_n"` (the normalized level of Figure C.2; requires the result to carry `l_n`). |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a lightweight fiche (body, rating and disclaimer only). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the table shows the measurement chain per band (energy-average `Li`, reverberation time `T` and the quantity) instead of the two-column `f \| value` form; it requires the result to carry `li` and `t2` (populated by [`impact_insulation`](/phonometry/reference/api/building/insulation/#impact_insulation)). |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` or `quantity` is unknown, the selected quantity is not available, the result does not hold the 16 core one-third-octave bands (100 Hz to 3150 Hz) the ISO 717-2 rating needs, or `verbose=True` without the per-band chain. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |

## ReportMetadata

```python
ReportMetadata(
    specimen: str | None = None,
    client: str | None = None,
    mounted_by: str | None = None,
    manufacturer: str | None = None,
    area: float | None = None,
    mass_per_area: float | None = None,
    source_volume: float | None = None,
    receiving_volume: float | None = None,
    room_volume: float | None = None,
    source_positions: int | None = None,
    receiver_positions: int | None = None,
    temperature: float | None = None,
    relative_humidity: float | None = None,
    source_temperature: float | None = None,
    source_relative_humidity: float | None = None,
    receiving_temperature: float | None = None,
    receiving_relative_humidity: float | None = None,
    pressure: float | None = None,
    tube_diameter: float | None = None,
    mic_spacing: float | None = None,
    thickness: float | None = None,
    test_room: str | None = None,
    instrumentation: str | None = None,
    calibration: str | None = None,
    mounting: str | None = None,
    measurement_standard: str | None = None,
    test_date: str | None = None,
    laboratory: str | None = None,
    operator: str | None = None,
    report_id: str | None = None,
    requirement: float | None = None,
    required_class: int | None = None,
    notes: str | None = None,
    tube_shape: str | None = None,
)
```

Descriptive metadata for the accredited ISO 717 report fiche.

All fields are optional (default `None`); the report renders only the
fields that are supplied, so a partially populated instance is valid. The
numeric fields are validated on construction by physical range: the
dimension, mass, volume and pressure fields must be finite and strictly
positive; the temperature and requirement fields need only be finite (0
degrees Celsius or below is a valid test condition, and a programme-loudness
target in LUFS is negative); and the relative-humidity fields must lie
within 0..100 %. A violation raises `ValueError`.

**Attributes**

| Name | Description |
| :--- | :--- |
| `specimen` | Specimen description printed in the header (the tested element, e.g. `"200 mm concrete wall"`). |
| `client` | Client the test was carried out for. |
| `mounted_by` | Who mounted the specimen in the test opening. |
| `manufacturer` | Manufacturer of the tested element. |
| `area` | Specimen area `S`, in m^2 (the free test opening area). |
| `mass_per_area` | Measured mass per unit area, in kg/m^2. |
| `source_volume` | Source-room volume, in m^3. |
| `receiving_volume` | Receiving-room volume, in m^3. |
| `room_volume` | Volume of the single room under test, in m^3. Room acoustics (ISO 3382-1/-2) characterises one enclosure rather than a source/receiving pair, and ISO 3382-2:2008 Clause 9 requires the room volume to be reported; the room-acoustics fiche prints it in the header. Distinct from the `source_volume`/`receiving_volume` pair, which describe a sound-transmission measurement. |
| `source_positions` | Number of source (loudspeaker/omnidirectional) positions used in the measurement, an integer (ISO 3382-1:2009 Table 1 and ISO 3382-2:2008 Clause 8 require reporting the number of source positions). Printed by the room-acoustics fiche. |
| `receiver_positions` | Number of microphone (receiver) positions used, an integer (ISO 3382-1:2009 Table 1 and ISO 3382-2:2008 Clause 8 require reporting the number of microphone positions). Printed by the room-acoustics fiche. |
| `temperature` | Air temperature during the test, in degrees Celsius (a single representative value; use the per-room fields below when the source and receiving rooms are reported separately). |
| `relative_humidity` | Relative humidity during the test, in %. |
| `source_temperature` | Source-room air temperature, in degrees Celsius. |
| `source_relative_humidity` | Source-room relative humidity, in %. |
| `receiving_temperature` | Receiving-room air temperature, in degrees Celsius. |
| `receiving_relative_humidity` | Receiving-room relative humidity, in %. |
| `pressure` | Ambient (static) air pressure during the test, in kPa. |
| `test_room` | Test-room / facility identification. |
| `instrumentation` | Identification and class of the instrumentation used (manufacturer, model, serial number), as free text. The occupational noise-exposure fiche prints it for ISO 9612:2009 Clause 15 c; when it is not supplied that fiche falls back to the result's own instrument class. |
| `calibration` | Calibration traceability, as free text (calibrator, date and result of the most recent verification, the before/after field checks). Printed by the occupational noise-exposure fiche (ISO 9612:2009 Clause 15 c). |
| `tube_diameter` | Impedance-tube inner diameter `d` (circular tube) or maximum lateral dimension (rectangular tube), in metres. Printed by the impedance-tube fiche (ISO 10534-2), where it fixes the upper plane-wave cut-on frequency. |
| `mic_spacing` | Microphone spacing `s` between the two measurement positions of the impedance tube, in metres. Printed by the impedance-tube fiche (ISO 10534-2), where it bounds the working frequency range. |
| `tube_shape` | Cross-section of the impedance tube: `"circular"`, `"rectangular"` or `"square"`. Printed by the impedance-tube fiche (ISO 10534-2) next to the tube diameter, which it qualifies (inner diameter for a circular tube, maximum lateral dimension otherwise). |
| `thickness` | Specimen thickness under the applied static load, in metres. Printed by the dynamic-stiffness fiche (EN 29052-1 / ISO 9052-1), where EN 29052-1:1992 Clause 9 b) requires reporting the thickness of the resilient layer under load; it is shown in millimetres. |
| `mounting` | Mounting condition of the specimen (e.g. the ISO 10140-1 mounting code or a short description). |
| `measurement_standard` | Measurement standard the spectrum was obtained under (e.g. `"ISO 10140-2"` or `"ISO 16283-1"`); it forms the report's standard-basis line together with the ISO 717 rating part. |
| `test_date` | Date of the test, as a free-form string. |
| `laboratory` | Testing laboratory / institute name (footer). |
| `operator` | Operator who carried out the test (footer signature line). |
| `report_id` | Report / test number (footer). |
| `requirement` | Target single-number value the verdict row compares the rating against, expressed in the rating's own unit (e.g. dB, a dimensionless absorption coefficient, sone, or a programme-loudness level in LUFS). It need only be finite (a loudness target in LUFS is negative), so its sign is not constrained. The pass direction is defined by each rating's `report` method: quantities where more is better (airborne insulation, absorption) pass at or above the requirement, and quantities where less is better (impact level, loudness, aircraft noise) pass at or below it; the programme-loudness fiche reads it as the target level and passes within a tolerance. |
| `required_class` | Target performance-class index for a class-compliance verdict (the IEC 61260-1 filter fiche): `0`, `1` or `2`, where class 0 is the strictest. When supplied, the fiche's verdict passes if the achieved overall class is at least as strict as this class (a smaller or equal class index). `None` (the default) prints no verdict row. |
| `notes` | Free-form remarks printed in the footer. |

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a supplied dimension/mass/volume/pressure is not finite and strictly positive, a temperature or requirement is not finite, a relative humidity is outside 0..100 %, a required class is not one of 0, 1, 2, a position count is not a finite, positive integer, or a tube shape is not one of `"circular"`, `"rectangular"`, `"square"`. |

### ReportMetadata.is_empty()

```python
ReportMetadata.is_empty() -> bool
```

Return `True` when no field is set (an all-`None` instance).
