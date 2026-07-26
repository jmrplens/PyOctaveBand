---
title: "noise_control.silencers"
description: "Reactive silencers by the four-pole (transmission-matrix) method."
sidebar:
  label: "silencers"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

Reactive silencers by the four-pole (transmission-matrix) method.

A reactive silencer controls noise by *reflecting* it back to the source with
impedance discontinuities -- sudden area changes and side branches -- rather
than by dissipating it in absorptive material. The one-dimensional plane-wave
theory represents each acoustic element by a 2x2 **transfer (four-pole)
matrix** relating the sound pressure `p` and volume velocity `S u` at its
two ends, and a compound silencer is the ordered matrix product of its
elements (Bies, Hansen & Howard, *Engineering Noise Control* 5th ed., §8.8-8.9;
Munjal, *Acoustics of Ducts and Mufflers*).

**Transfer matrix** (Bies Eq. (8.133)), state vector `[p, S u]` with the
characteristic acoustic impedance `Z = rho c / S`. The plane-wave element for
a straight duct of length `L` and area `S` is (Bies Eq. (8.143), no flow)

    [[ cos(kL),              j (rho c / S) sin(kL) ],
     [ j (S / rho c) sin(kL), cos(kL)              ]],    k = omega / c,

and a **side branch** of acoustic impedance `Z_b` is the shunt element
(Bies Eq. (8.144))

    [[ 1,       0 ],
     [ 1 / Z_b, 1 ]].

**Transmission loss** from the compound matrix `T` (Munjal, *Acoustics of
Ducts and Mufflers* 2nd ed., Eq. (3.27), no flow; reduces to Bies Eq. (8.148)
for equal inlet/outlet areas):

    TL = 10 log10[ (Zn / Z1) (1/4) | T11 + T12 / Zn + Z1 T21 + (Z1 / Zn) T22 |^2 ]

with `Z1 = rho c / S_in` and `Zn = rho c / S_out`. A zero-length element
between unequal areas then reproduces the classic sudden-expansion result
`TL = 10 log10[(1 + m)^2 / (4 m)]` with `m = S_out / S_in`, and the TL is
the same from either side, as reciprocity of a passive two-port requires.
Bies Eq. (8.141) prints this formula with impedance ratios on `T11` and
`T22` (`Z_A1/Z_An` and `Z_An/Z_A1`) instead of the overall `Zn/Z1`
prefactor; as printed it fails the sudden-expansion limit (see
`docs/ERRATA.md`). `TL` is the intrinsic attenuation for an anechoic
termination. The **insertion loss** for a source of internal impedance
`Z_s` radiating into a termination impedance `Z_r` is the extra
attenuation of inserting the silencer in place of a direct connection,

    IL = 20 log10 | (T11 Z_r + T12 + Z_s Z_r T21 + Z_s T22) / (Z_s + Z_r) |,

which is `0` when the silencer reduces to a through connection (`T = I`)
and, for equal inlet/outlet areas, equals the transmission loss for the
anechoic reference `Z_s = Z_r = rho c / S` (with unequal areas the direct
connection contains the same area jump, so its mismatch loss cancels from
the insertion loss but not from the transmission loss).

**Simple expansion chamber.** A chamber of area `S_exp` and length `L`
between pipes of area `S_duct` has the closed-form transmission loss (Bies
Eq. (8.111)) with area ratio `m = S_exp / S_duct`

    TL = 10 log10[ 1 + (1/4) (m - 1/m)^2 sin^2(kL) ],

peaking at `10 log10[1 + (1/4)(m - 1/m)^2]` when `kL = pi/2, 3pi/2, ...` and
dropping to `0` at `kL = n pi` (no dissipation). The four-pole product
reproduces this exactly, and the machinery extends to side-branch (Helmholtz,
quarter-wave) and extended-tube resonators that the closed form cannot cover.

## expansion_chamber

```python
expansion_chamber(
    frequencies: ArrayLike,
    length: float,
    chamber_area: float,
    pipe_area: float,
    *,
    speed_of_sound: float = 343.0,
    density: float = 1.206,
    source_impedance: ArrayLike | None = None,
    radiation_impedance: ArrayLike | None = None,
) -> ReactiveSilencerResult
```

Simple expansion-chamber silencer (Bies Eq. (8.111) / four-pole).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies `f`, Hz (1-D array). |
| `length` | Chamber length `L`, m. |
| `chamber_area` | Chamber cross-sectional area `S_exp`, m2. |
| `pipe_area` | Inlet/outlet pipe area `S_duct`, m2. |
| `speed_of_sound` | Speed of sound `c`, m/s. |
| `density` | Air density `rho`, kg/m3. |
| `source_impedance` | Optional source impedance `Z_s` for the insertion loss, Pa s/m3. |
| `radiation_impedance` | Optional radiation impedance `Z_r` for the insertion loss, Pa s/m3. |

**Returns:** A [`ReactiveSilencerResult`](/phonometry/reference/api/noise_control/silencers/#reactivesilencerresult) (its `transmission_loss` equals the closed form `10 log10[1 + (1/4)(m - 1/m)^2 sin^2(kL)]`).

## extended_tube_chamber

```python
extended_tube_chamber(
    frequencies: ArrayLike,
    length: float,
    chamber_area: float,
    pipe_area: float,
    *,
    inlet_extension: float = 0.0,
    outlet_extension: float = 0.0,
    speed_of_sound: float = 343.0,
    density: float = 1.206,
    source_impedance: ArrayLike | None = None,
    radiation_impedance: ArrayLike | None = None,
) -> ReactiveSilencerResult
```

Extended-inlet/outlet expansion chamber (Bies §8.9.7).

The inlet and outlet pipes extend a distance into the chamber, forming
annular quarter-wave side branches (of area `S_exp - S_duct` and lengths
equal to the extensions, Bies Eq. (8.156)) at the two junctions. Tuning the
extensions (classically `L/4` and `L/2`) places quarter-wave peaks that
fill the `kL = n pi` troughs of the plain expansion chamber. With both
extensions `0` the result reduces exactly to [`expansion_chamber`](/phonometry/reference/api/noise_control/silencers/#expansion_chamber).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies `f`, Hz (1-D array). |
| `length` | Chamber length `L`, m. |
| `chamber_area` | Chamber cross-sectional area `S_exp`, m2. |
| `pipe_area` | Inlet/outlet pipe area `S_duct`, m2. |
| `inlet_extension` | Inlet pipe extension into the chamber `L_a`, m. |
| `outlet_extension` | Outlet pipe extension into the chamber `L_b`, m. |
| `speed_of_sound` | Speed of sound `c`, m/s. |
| `density` | Air density `rho`, kg/m3. |
| `source_impedance` | Optional source impedance `Z_s`, Pa s/m3. |
| `radiation_impedance` | Optional radiation impedance `Z_r`, Pa s/m3. |

**Returns:** A [`ReactiveSilencerResult`](/phonometry/reference/api/noise_control/silencers/#reactivesilencerresult).

## helmholtz_resonator

```python
helmholtz_resonator(
    frequencies: ArrayLike,
    duct_area: float,
    neck_area: float,
    neck_length: float,
    cavity_volume: float,
    *,
    resistance: float = 0.0,
    speed_of_sound: float = 343.0,
    density: float = 1.206,
    source_impedance: ArrayLike | None = None,
    radiation_impedance: ArrayLike | None = None,
) -> ReactiveSilencerResult
```

Side-branch Helmholtz resonator on a duct (Bies Eqs. (8.144), (8.152)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies `f`, Hz (1-D array). |
| `duct_area` | Main-duct cross-sectional area `S_d`, m2. |
| `neck_area` | Resonator neck area `S_neck`, m2. |
| `neck_length` | Effective neck length `l_e`, m. |
| `cavity_volume` | Cavity volume `V`, m3. |
| `resistance` | Neck acoustic resistance `R`, Pa s/m3 (default 0). |
| `speed_of_sound` | Speed of sound `c`, m/s. |
| `density` | Air density `rho`, kg/m3. |
| `source_impedance` | Optional source impedance `Z_s`, Pa s/m3. |
| `radiation_impedance` | Optional radiation impedance `Z_r`, Pa s/m3. |

**Returns:** A [`ReactiveSilencerResult`](/phonometry/reference/api/noise_control/silencers/#reactivesilencerresult); `resonances` holds `f_0 = (c / 2 pi) sqrt(S_neck / (l_e V))`.

## quarter_wave_resonator

```python
quarter_wave_resonator(
    frequencies: ArrayLike,
    duct_area: float,
    length: float,
    branch_area: float,
    *,
    speed_of_sound: float = 343.0,
    density: float = 1.206,
    source_impedance: ArrayLike | None = None,
    radiation_impedance: ArrayLike | None = None,
) -> ReactiveSilencerResult
```

Closed quarter-wave side-branch tube on a duct (Bies Eqs. (8.144), (8.146)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies `f`, Hz (1-D array). |
| `duct_area` | Main-duct cross-sectional area `S_d`, m2. |
| `length` | Effective branch length `l_e`, m. |
| `branch_area` | Branch tube area `S`, m2. |
| `speed_of_sound` | Speed of sound `c`, m/s. |
| `density` | Air density `rho`, kg/m3. |
| `source_impedance` | Optional source impedance `Z_s`, Pa s/m3. |
| `radiation_impedance` | Optional radiation impedance `Z_r`, Pa s/m3. |

**Returns:** A [`ReactiveSilencerResult`](/phonometry/reference/api/noise_control/silencers/#reactivesilencerresult); `resonances` holds the odd multiples of `f = c / (4 l_e)` within the frequency range.

## ReactiveSilencerResult

```python
ReactiveSilencerResult(
    frequencies: np.ndarray,
    transmission_loss: np.ndarray,
    insertion_loss: np.ndarray | None,
    transfer_matrix: np.ndarray,
    kind: str,
    resonances: np.ndarray | None = None,
)
```

Transmission and insertion loss of a reactive silencer over frequency.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies `f`, Hz. |
| `transmission_loss` | Transmission loss per frequency, dB. |
| `insertion_loss` | Insertion loss per frequency, dB, or `None` when no source/radiation impedance was supplied. |
| `transfer_matrix` | The compound `(n_freq, 2, 2)` four-pole matrix. |
| `kind` | A short label of the device (e.g. `"expansion chamber"`). |
| `resonances` | Notable resonance frequencies, Hz (e.g. the resonator tuning frequency), or `None`. |

### ReactiveSilencerResult.plot()

```python
ReactiveSilencerResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the transmission (and insertion) loss against frequency.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### ReactiveSilencerResult.report()

```python
ReactiveSilencerResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render a reactive-silencer transmission-loss fiche to `path`.

Writes a one-page silencer-performance sheet: the method-basis line
naming the plane-wave four-pole (transfer-matrix) method (Munjal,
Acoustics of Ducts and Mufflers 2nd ed., Eq. (3.27); Bies, Hansen &
Howard, Engineering Noise Control 5th ed., sections 8.8-8.9), an
optional metadata header (client, device, test environment,
instrumentation, climate, date), a per-band table (nominal frequency,
the transmission loss `TL` and, when computed, the insertion loss
`IL`) beside the `TL` (and `IL`) curves, the boxed mean
transmission loss over the analysis bands with the peak transmission
loss and the device kind, an optional verdict row against a declared
minimum, and a method-basis strip stating the four-pole
transmission-loss relation.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header (`client`, `specimen` the device, `test_room` the test environment, `instrumentation`, `temperature`, `relative_humidity`, `pressure`, `test_date`), the footer identity (`laboratory`, `operator`, `report_id`, `notes`) and, via `requirement`, a declared minimum mean transmission loss (more transmission loss is better). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | Accepted for signature symmetry with the other fiches; the silencer table already shows the insertion loss when it was computed. |
| `language` | Fiche language: `"en"` (default) or `"es"`. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is unknown. |
| ImportError | If reportlab (or, for the figure, matplotlib) is not installed (`pip install phonometry[report]`). |
