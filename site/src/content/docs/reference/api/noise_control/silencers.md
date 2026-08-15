---
title: "noise_control.silencers"
description: "Reactive silencers by the four-pole (transmission-matrix) method."
sidebar:
  label: "silencers"
---

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
characteristic acoustic impedance $Z = \rho c / S$. The plane-wave
element for
a straight duct of length `L` and area `S` is (Bies Eq. (8.143), no flow)

$$
\begin{bmatrix} \cos(kL) & j (\rho c / S) \sin(kL) \\ j (S / \rho c) \sin(kL) & \cos(kL) \end{bmatrix}, \qquad k = \omega / c,
$$

and a **side branch** of acoustic impedance `Z_b` is the shunt element
(Bies Eq. (8.144))

$$
\begin{bmatrix} 1 & 0 \\ 1 / Z_b & 1 \end{bmatrix}.
$$

**Transmission loss** from the compound matrix `T` (Munjal, *Acoustics of
Ducts and Mufflers* 2nd ed., Eq. (3.27), no flow; reduces to Bies Eq. (8.148)
for equal inlet/outlet areas):

$$
\mathrm{TL} = 10 \log_{10}\!\left[\frac{Z_n}{Z_1} \cdot \frac{1}{4} \left\lvert T_{11} + \frac{T_{12}}{Z_n} + Z_1 T_{21} + \frac{Z_1}{Z_n} T_{22} \right\rvert^2\right]
$$

with $Z_1 = \rho c / S_{\mathrm{in}}$ and
$Z_n = \rho c / S_{\mathrm{out}}$. A zero-length element
between unequal areas then reproduces the classic sudden-expansion result
$\mathrm{TL} = 10 \log_{10}[(1 + m)^2 / (4 m)]$ with
$m = S_{\mathrm{out}} / S_{\mathrm{in}}$, and the TL is
the same from either side, as reciprocity of a passive two-port requires.
Bies Eq. (8.141) prints this formula with impedance ratios on `T11` and
`T22` ($Z_{A1}/Z_{An}$ and $Z_{An}/Z_{A1}$) instead of the
overall $Z_n/Z_1$
prefactor; as printed it fails the sudden-expansion limit (see
`docs/ERRATA.md`). `TL` is the intrinsic attenuation for an anechoic
termination. The **insertion loss** for a source of internal impedance
`Z_s` radiating into a termination impedance `Z_r` is the extra
attenuation of inserting the silencer in place of a direct connection,

$$
\mathrm{IL} = 20 \log_{10} \left\lvert \frac{T_{11} Z_r + T_{12} + Z_s Z_r T_{21} + Z_s T_{22}} {Z_s + Z_r} \right\rvert,
$$

which is `0` when the silencer reduces to a through connection
($T = I$)
and, for equal inlet/outlet areas, equals the transmission loss for the
anechoic reference $Z_s = Z_r = \rho c / S$ (with unequal areas the
direct
connection contains the same area jump, so its mismatch loss cancels from
the insertion loss but not from the transmission loss).

**Simple expansion chamber.** A chamber of area `S_exp` and length `L`
between pipes of area `S_duct` has the closed-form transmission loss (Bies
Eq. (8.111)) with area ratio $m = S_{\mathrm{exp}} / S_{\mathrm{duct}}$

$$
\mathrm{TL} = 10 \log_{10}\!\left[1 + \frac{1}{4} \left(m - \frac{1}{m}\right)^2 \sin^2(kL)\right],
$$

peaking at $10 \log_{10}[1 + (1/4)(m - 1/m)^2]$ when
$kL = \pi/2, 3\pi/2, \ldots$ and
dropping to `0` at $kL = n \pi$ (no dissipation). The four-pole product
reproduces this exactly, and the machinery extends to side-branch (Helmholtz,
quarter-wave) and extended-tube resonators that the closed form cannot cover.

**Validity.** All of this is one-dimensional: it holds while the duct and the
chamber carry plane waves only, that is below the first higher-order-mode
cut-on frequency of the widest cross section
([`phonometry.noise_control.duct_modes`](/phonometry/reference/api/noise_control/duct-modes/)). Every result reports that
frequency as [`ReactiveSilencerResult.plane_wave_limit`](/phonometry/reference/api/noise_control/duct-modes/#plane_wave_limit) and raises a
[`PlaneWaveWarning`](/phonometry/reference/api/noise_control/duct-modes/#planewavewarning) when the
analysis grid reaches past it: the numbers are still returned, but above cut-on
they describe the plane-wave mode alone and a measurement will show the rest.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

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

**Returns:** A [`ReactiveSilencerResult`](/phonometry/reference/api/noise_control/silencers/#reactivesilencerresult) (its `transmission_loss` equals the closed form $10 \log_{10}[1 + (1/4)(m - 1/m)^2 \sin^2(kL)]$).

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
annular quarter-wave side branches (of area
$S_{\mathrm{exp}} - S_{\mathrm{duct}}$ and lengths
equal to the extensions, Bies Eq. (8.156)) at the two junctions. Tuning the
extensions (classically $L/4$ and $L/2$) places quarter-wave
peaks that
fill the $kL = n \pi$ troughs of the plain expansion chamber. With
both extensions `0` the result reduces exactly to
[`expansion_chamber`](/phonometry/reference/api/noise_control/silencers/#expansion_chamber).

The junction where each extended pipe ends is where its three ducts meet,
so the straight chamber element cascaded between the two side branches is
the length left over,
$L_c = L - L_a - L_b$ (Bies Figure 8.19(a) and Example 8.2, where
$L = L_a + L_b + L_c$), and not the full chamber length. When the two
extensions meet ($L_a + L_b = L$) the straight element vanishes and
the two annular branches shunt the same plane, which is the well-defined
limit of the cascade; extensions that would overlap are rejected.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies `f`, Hz (1-D array). |
| `length` | Overall chamber length `L`, m, extensions included. |
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

**Returns:** A [`ReactiveSilencerResult`](/phonometry/reference/api/noise_control/silencers/#reactivesilencerresult); `resonances` holds $f_0 = (c / 2 \pi) \sqrt{S_{\mathrm{neck}} / (l_e V)}$.

## plot_silencer_geometry

```python
plot_silencer_geometry(
    kind: str,
    ax: Axes | None = None,
    *,
    length: float | None = None,
    chamber_area: float | None = None,
    pipe_area: float | None = None,
    inlet_extension: float = 0.0,
    outlet_extension: float = 0.0,
    duct_area: float | None = None,
    neck_area: float | None = None,
    neck_length: float | None = None,
    cavity_volume: float | None = None,
    branch_area: float | None = None,
    language: str = 'en',
) -> Axes
```

Draw a reactive silencer cross-section to scale.

Side cut through the duct axis with equivalent circular diameters
(`d = 2 sqrt(S / pi)`) for every cross-section area, matching the
parameters of the four `noise_control` silencer
constructors. A Helmholtz cavity is drawn as the cube of equal volume
with its volume annotated.

**Parameters**

| Name | Description |
| :--- | :--- |
| `kind` | One of `"expansion chamber"`, `"extended-tube chamber"`, `"Helmholtz resonator"`, `"quarter-wave resonator"` (the `ReactiveSilencerResult.kind` strings). |
| `ax` | Existing axes, or `None` to create a figure. |
| `length` | Chamber length or quarter-wave tube length, in metres. |
| `chamber_area` | Chamber cross-section, in m2 (chambers). |
| `pipe_area` | Inlet/outlet pipe cross-section, in m2 (chambers). |
| `inlet_extension` | Inlet tube extension into the chamber, in metres. |
| `outlet_extension` | Outlet tube extension, in metres. |
| `duct_area` | Main duct cross-section, in m2 (side branches). |
| `neck_area` | Neck cross-section, in m2 (Helmholtz). |
| `neck_length` | Neck length, in metres (Helmholtz). |
| `cavity_volume` | Cavity volume, in m3 (Helmholtz). |
| `branch_area` | Branch tube cross-section, in m2 (quarter-wave). |
| `language` | Label language, `"en"` (default) or `"es"`. |

**Returns:** The axes.

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

**Returns:** A [`ReactiveSilencerResult`](/phonometry/reference/api/noise_control/silencers/#reactivesilencerresult); `resonances` holds the odd multiples of $f = c / (4 l_e)$ within the frequency range.

## ReactiveSilencerResult

```python
ReactiveSilencerResult(
    frequencies: np.ndarray,
    transmission_loss: np.ndarray,
    insertion_loss: np.ndarray | None,
    transfer_matrix: np.ndarray,
    kind: str,
    resonances: np.ndarray | None = None,
    geometry: dict[str, float] | None = None,
    plane_wave_limit: float | None = None,
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
| `geometry` | The defining geometry the constructor was called with (keys matching its keyword names, e.g. `length`/`chamber_area`/ `pipe_area` for a chamber), retained so `plot_geometry` can draw the device; appended after the original fields and `None` for hand-built results. |
| `plane_wave_limit` | The first higher-order-mode cut-on frequency of the widest cross section of the device, Hz (Norton & Karczub Eq. 7.6, [`phonometry.noise_control.duct_modes.plane_wave_limit`](/phonometry/reference/api/noise_control/duct-modes/#plane_wave_limit)). The four-pole algebra of this module is one-dimensional and is valid below it; above it several modes propagate at once and the result describes the plane-wave mode only, which is why a [`PlaneWaveWarning`](/phonometry/reference/api/noise_control/duct-modes/#planewavewarning) is raised when the analysis reaches past it. `None` for hand-built results that do not retain their geometry. |

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

### ReactiveSilencerResult.plot_geometry()

```python
ReactiveSilencerResult.plot_geometry(
    ax: Axes | None = None,
    *,
    language: str = 'en',
) -> Axes
```

Draw the silencer cross-section to scale (dimensioned side cut).

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the result does not retain its `geometry`. |

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
