---
title: "materials.absorbers.four_microphone"
description: "Four-microphone transfer-matrix method for the transmission of a specimen."
sidebar:
  label: "four_microphone"
---

Four-microphone transfer-matrix method for the transmission of a specimen.

**ASTM E2611-19**, the two-tube method: the specimen sits between an upstream
and a downstream tube section with two microphones on each side, and the wave
field is decomposed into forward/backward amplitudes on each side
(Eqs. (17)-(20)). The face pressures and particle velocities are formed
(Eq. (21)) and the transfer matrix `[[T11, T12], [T21, T22]]` is solved from
a two-load (Eq. (22)) or a symmetric one-load (Eq. (24)) measurement.
Transmission loss (Eq. (26)), hard-backed reflection/absorption
(Eqs. (27)/(28)) and the material wavenumber/characteristic impedance
(Eqs. (29)/(30)) all read out of those four poles, which is what makes the
standard one subject: everything here exists to fill the matrix or to
interpret it.

Time convention $e^{+j\omega t}$ with the forward wave carried by
$e^{-jkx}$ (Eq. (21)); air properties from Clause 8.2/8.3, Eqs. (4)/(5),
use temperature in **degrees Celsius**. Both differ from the ISO 10534-2
ansatz of [`impedance_tube`](/phonometry/reference/api/materials/impedance-tube/), whose
wavenumber is $k_0 = k_0' - jk_0''$ and whose air properties take
kelvin; the two are **not** interchangeable, so the air-property and
working-range helpers are named per standard and each stays with the method
that prescribes it. What the two transfer methods genuinely share - the tube
cross-section and the plane-wave working-range arithmetic - is imported from
that module.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## air_density_astm

```python
air_density_astm(
    temperature: ArrayLike,
    atmospheric_pressure: ArrayLike = 101.325,
) -> Real
```

Air density (ASTM E2611-19, Eq. (5)).

$\rho = 1.290 \, \frac{P}{101.325} \, \frac{273.15}{273.15 + T}$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `temperature` | Room temperature `T`, in **degrees Celsius**. |
| `atmospheric_pressure` | Atmospheric pressure `P`, in kilopascals (default 101,325 kPa). |

**Returns:** Air density `rho`, in kilograms per cubic metre.

## air_layer_transfer_matrix

```python
air_layer_transfer_matrix(
    wavenumber: ArrayLike,
    thickness: float,
    characteristic_impedance: float,
) -> TransferMatrix
```

Analytic transfer matrix of a pure air layer of thickness `d`.

$$
T = [[\cos(k d),\; j \rho c \sin(k d)],\; [j \sin(k d) / (\rho c),\; \cos(k d)]]
$$

the classical loss-free layer used to validate the ASTM E2611-19 reduction
(it is reciprocal, $\operatorname{det}(T) = 1$, and symmetric,
$T_{11} = T_{22}$).

**Parameters**

| Name | Description |
| :--- | :--- |
| `wavenumber` | Air wavenumber `k`. |
| `thickness` | Layer thickness `d`, in metres. |
| `characteristic_impedance` | Characteristic impedance `rho c`, in rayls. |

**Returns:** The air-layer [`TransferMatrix`](/phonometry/reference/api/materials/four-microphone/#transfermatrix).

## face_quantities

```python
face_quantities(
    a: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    d: ArrayLike,
    *,
    wavenumber: ArrayLike,
    thickness: float,
    characteristic_impedance: float,
) -> tuple[Complex, Complex, Complex, Complex]
```

Face pressures and particle velocities (ASTM E2611-19, Eq. (21)).

$$
p_0 = A + B, \qquad p_d = C e^{-jkd} + D e^{+jkd}
$$

$$
u_0 = \frac{A - B}{\rho c}, \qquad u_d = \frac{C e^{-jkd} - D e^{+jkd}}{\rho c}
$$

**Parameters**

| Name | Description |
| :--- | :--- |
| `a` | Upstream forward amplitude `A`. |
| `b` | Upstream backward amplitude `B`. |
| `c` | Downstream forward amplitude `C`. |
| `d` | Downstream backward amplitude `D`. |
| `wavenumber` | Air wavenumber `k`. |
| `thickness` | Specimen thickness `d`, in metres. |
| `characteristic_impedance` | Characteristic impedance `rho c`, in rayls. |

**Returns:** Tuple `(p0, pd, u0, ud)` of face pressures and velocities.

## plane_wave_frequency_range_astm

```python
plane_wave_frequency_range_astm(
    spacing: float,
    speed_of_sound: float,
    *,
    diameter: float | None = None,
    shape: str = 'circular',
) -> tuple[float, float]
```

Working plane-wave frequency range `(f_l, f_u)` (ASTM E2611-19).

The upper limit is the smaller of the microphone-spacing bound
$s \le 0.8 c / (2 f_\mathrm{u})$, i.e. $f_\mathrm{u} s < 0.40 c$ (6.5.4), and,
when the tube `diameter` is given, the cut-on bound
$f_\mathrm{u} < K c / d$ with $K = 0.586$ for a circular tube
(6.2.4.1, Eq. (2)) or $K = 0.500$ for a rectangular tube with `d`
the largest section dimension (6.2.5). The lower limit follows 6.2.3: the
spacing shall be greater than 1 % of the wavelength, i.e.
$f_\mathrm{l} = c / (100 s)$.

With two different spacings `s1`/`s2`, call with the larger one for
the upper bound and the smaller one for the lower bound (each bound is
binding for every microphone pair).

**Parameters**

| Name | Description |
| :--- | :--- |
| `spacing` | Microphone spacing `s`, in metres. |
| `speed_of_sound` | Speed of sound `c`, in metres per second. |
| `diameter` | Tube diameter (circular) or largest section dimension (rectangular/square) `d`, in metres; `None` applies only the spacing bound. |
| `shape` | `"circular"`, `"rectangular"` or `"square"`. |

**Returns:** Tuple `(f_l, f_u)` of the lower and upper frequency limits, in Hz.

## speed_of_sound_astm

```python
speed_of_sound_astm(temperature: ArrayLike) -> Real
```

Speed of sound in air (ASTM E2611-19, Eq. (4)).

$c = 20.047 \sqrt{273.15 + T}$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `temperature` | Room temperature `T`, in **degrees Celsius**. |

**Returns:** Speed of sound `c`, in metres per second.

## transfer_matrix_one_load

```python
transfer_matrix_one_load(
    load: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
    *,
    l1: float,
    s1: float,
    l2: float,
    s2: float,
    thickness: float,
    wavenumber: ArrayLike,
    characteristic_impedance: float,
    frequency: ArrayLike | None = None,
    diameter: float | None = None,
    shape: str = 'circular',
) -> TransferMatrix
```

One-load transfer matrix, symmetric specimen (ASTM E2611-19, Eqs. (23)-(24)).

Valid only for a reciprocal **and** symmetric specimen
($T_{11} = T_{22}$ and $T_{11} T_{22} - T_{12} T_{21} = 1$,
Eq. (23)). A single termination suffices:

$$
\begin{aligned} \mathrm{DEN} &= p_0 u_d + p_d u_0 \\ T_{11} = T_{22} &= (p_d u_d + p_0 u_0) / \mathrm{DEN} \\ T_{12} &= (p_0^{2} - p_d^{2}) / \mathrm{DEN} \\ T_{21} &= (u_0^{2} - u_d^{2}) / \mathrm{DEN} \end{aligned}
$$

**Parameters**

| Name | Description |
| :--- | :--- |
| `load` | Microphone transfer functions `(H1, H2, H3, H4)`. |
| `l1` | Upstream reference distance `l1`, in metres. |
| `s1` | Upstream microphone spacing `s1`, in metres. |
| `l2` | Downstream reference distance `l2`, in metres. |
| `s2` | Downstream microphone spacing `s2`, in metres. |
| `thickness` | Specimen thickness `d`, in metres. |
| `wavenumber` | Air wavenumber `k`. |
| `characteristic_impedance` | Characteristic impedance `rho c`. |
| `frequency` | Optional frequency vector `f`, in hertz, retained on the result so [`TransferMatrix.plot`](/phonometry/reference/api/materials/four-microphone/#transfermatrixplot) needs no arguments. |
| `diameter` | Optional tube diameter (circular) or largest section dimension (rectangular/square), in metres, that activates the plane-wave working-range check (6.2.3-6.2.5, 6.5.4). |
| `shape` | Tube cross-section, `"circular"`, `"rectangular"` or `"square"`. |

**Returns:** The specimen [`TransferMatrix`](/phonometry/reference/api/materials/four-microphone/#transfermatrix) (measurement context retained on the result).

## transfer_matrix_two_load

```python
transfer_matrix_two_load(
    load_a: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
    load_b: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
    *,
    l1: float,
    s1: float,
    l2: float,
    s2: float,
    thickness: float,
    wavenumber: ArrayLike,
    characteristic_impedance: float,
    frequency: ArrayLike | None = None,
    diameter: float | None = None,
    shape: str = 'circular',
) -> TransferMatrix
```

Two-load transfer matrix (ASTM E2611-19, Eqs. (17)-(22)).

Each load is the tuple `(H1, H2, H3, H4)` of the four microphone transfer
functions measured with a different downstream termination. The two loads
give four equations for the four unknowns (Eq. (22)):

$$
\begin{aligned} \mathrm{DEN} &= p_{da} u_{db} - p_{db} u_{da} \\ T_{11} &= (p_{0a} u_{db} - p_{0b} u_{da}) / \mathrm{DEN} \\ T_{12} &= (p_{0b} p_{da} - p_{0a} p_{db}) / \mathrm{DEN} \\ T_{21} &= (u_{0a} u_{db} - u_{0b} u_{da}) / \mathrm{DEN} \\ T_{22} &= (p_{da} u_{0b} - p_{db} u_{0a}) / \mathrm{DEN} \end{aligned}
$$

**Parameters**

| Name | Description |
| :--- | :--- |
| `load_a` | Microphone transfer functions `(H1, H2, H3, H4)` for load a. |
| `load_b` | Microphone transfer functions `(H1, H2, H3, H4)` for load b. |
| `l1` | Upstream reference distance `l1`, in metres. |
| `s1` | Upstream microphone spacing `s1`, in metres. |
| `l2` | Downstream reference distance `l2`, in metres. |
| `s2` | Downstream microphone spacing `s2`, in metres. |
| `thickness` | Specimen thickness `d`, in metres. |
| `wavenumber` | Air wavenumber `k`. |
| `characteristic_impedance` | Characteristic impedance `rho c`. |
| `frequency` | Optional frequency vector `f`, in hertz, retained on the result so [`TransferMatrix.plot`](/phonometry/reference/api/materials/four-microphone/#transfermatrixplot) needs no arguments. |
| `diameter` | Optional tube diameter (circular) or largest section dimension (rectangular/square), in metres, that activates the plane-wave working-range check (6.2.3-6.2.5, 6.5.4). |
| `shape` | Tube cross-section, `"circular"`, `"rectangular"` or `"square"`. |

**Returns:** The specimen [`TransferMatrix`](/phonometry/reference/api/materials/four-microphone/#transfermatrix) (measurement context retained on the result).

## TransferMatrix

```python
TransferMatrix(
    t11: Complex,
    t12: Complex,
    t21: Complex,
    t22: Complex,
    l1: float | None = None,
    s1: float | None = None,
    l2: float | None = None,
    s2: float | None = None,
    thickness: float | None = None,
    diameter: float | None = None,
    shape: str | None = None,
    frequency: Real | None = None,
    air_characteristic_impedance: float | None = None,
)
```

Acoustic transfer matrix `[[T11, T12], [T21, T22]]` (ASTM E2611-19).

Relates the pressure and normal particle velocity across a specimen,
$[p; u]_{x=0} = T \, [p; u]_{x=d}$ (Eq. (16)). Each entry is complex and
may be scalar or a per-frequency array of matching shape.

The trailing fields retain the measurement context when the matrix comes
out of [`transfer_matrix_two_load`](/phonometry/reference/api/materials/four-microphone/#transfer_matrix_two_load) / [`transfer_matrix_one_load`](/phonometry/reference/api/materials/four-microphone/#transfer_matrix_one_load)
(tube geometry `l1`/`s1`/`l2`/`s2`, specimen `thickness`, tube
`diameter` and canonical cross-section `shape`, the `frequency`
vector when supplied to the solver, and the air
`air_characteristic_impedance` `rho c`); all default to `None` so a
hand-built matrix (for example [`air_layer_transfer_matrix`](/phonometry/reference/api/materials/four-microphone/#air_layer_transfer_matrix)) is
unchanged.

### TransferMatrix.absorption_hard_backed()

```python
TransferMatrix.absorption_hard_backed(
    characteristic_impedance: float,
) -> Real
```

Hard-backed absorption coefficient (ASTM E2611-19, Eq. (28)).

$\alpha = 1 - \lvert R \rvert^2$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `characteristic_impedance` | Characteristic impedance `rho c`. |

**Returns:** Absorption coefficient `alpha`.

### TransferMatrix.characteristic_impedance_material()

```python
TransferMatrix.characteristic_impedance_material() -> Complex
```

Characteristic impedance of the material (ASTM E2611-19, Eq. (30)).

$Z = \sqrt{T_{12} / T_{21}}$.

**Returns:** Complex characteristic impedance `Z`, in rayls.

### TransferMatrix.determinant()

```python
TransferMatrix.determinant() -> Complex
```

Determinant $T_{11} T_{22} - T_{12} T_{21}$ (unity for a reciprocal specimen).

### TransferMatrix.material_wavenumber()

```python
TransferMatrix.material_wavenumber(thickness: float) -> Complex
```

Propagation wavenumber inside the material (ASTM E2611-19, Eq. (29)).

$k' = \arccos(T_{11}) / d$ (complex `arccos`).

**Parameters**

| Name | Description |
| :--- | :--- |
| `thickness` | Specimen thickness `d`, in metres. |

**Returns:** Complex material wavenumber `k'`, in reciprocal metres.

### TransferMatrix.plot()

```python
TransferMatrix.plot(
    frequency: ArrayLike | None = None,
    characteristic_impedance: float | None = None,
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the transmission loss with the hard-backed absorption overlaid.

Reads the four-pole entries out as the two ASTM E2611-19 spectra a
laboratory quotes: the normal-incidence transmission loss `TLn(f)`
(Eq. (26), the primary curve, left axis) and the hard-backed
absorption coefficient `alpha(f)` (Eq. (28), a muted companion on a
0..1 right axis). The four-pole entries carry no frequency axis of
their own, so the plot needs the measurement's `frequency` vector
(matching the shape of the entries) and the air characteristic
impedance `rho c`. A matrix built by the solvers retains both
(`self.frequency` / `self.air_characteristic_impedance`), so
`plot()` takes no arguments there; only a hand-built matrix (for
example [`air_layer_transfer_matrix`](/phonometry/reference/api/materials/four-microphone/#air_layer_transfer_matrix)) must supply them.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes` of the transmission-loss curve.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz, matching the shape of the matrix entries; `None` uses the stored `frequency`. |
| `characteristic_impedance` | Characteristic impedance `rho c` of the air in the tube, in rayls; `None` uses the stored `air_characteristic_impedance`. |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Plot language: `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the transmission-loss `plot` call. |

**Returns:** The axes.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `frequency` or `characteristic_impedance` is neither supplied nor stored on the matrix. |

### TransferMatrix.plot_geometry()

```python
TransferMatrix.plot_geometry(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the four-microphone tube to scale (dimensioned side view).

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the matrix does not retain its tube geometry (`l1`/`s1`/`l2`/`s2`/`thickness`). |

### TransferMatrix.reflection_hard_backed()

```python
TransferMatrix.reflection_hard_backed(
    characteristic_impedance: float,
) -> Complex
```

Hard-backed reflection coefficient (ASTM E2611-19, Eq. (27)).

$R = (T_{11} - \rho c T_{21}) / (T_{11} + \rho c T_{21})$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `characteristic_impedance` | Characteristic impedance `rho c`. |

**Returns:** Complex reflection coefficient `R`.

### TransferMatrix.transmission_loss()

```python
TransferMatrix.transmission_loss(characteristic_impedance: float) -> Real
```

Normal-incidence transmission loss in dB (ASTM E2611-19, Eq. (26)).

With

$$
t = \frac{2 e^{jkd}} {T_{11} + T_{12}/(\rho c) + \rho c \, T_{21} + T_{22}} \tag{Eq. 25}
$$

$$
TL = 20 \log_{10} \left| \frac{1}{t} \right| = 20 \log_{10} \frac{\lvert T_{11} + T_{12}/(\rho c) + \rho c \, T_{21} + T_{22} \rvert}{2} \tag{Eq. 26}
$$

(the $e^{jkd}$ factor has unit magnitude for
a real wavenumber).

**Parameters**

| Name | Description |
| :--- | :--- |
| `characteristic_impedance` | Characteristic impedance `rho c`. |

**Returns:** Transmission loss `TLn`, in decibels.

## wave_decomposition

```python
wave_decomposition(
    h1: ArrayLike,
    h2: ArrayLike,
    h3: ArrayLike,
    h4: ArrayLike,
    *,
    l1: float,
    s1: float,
    l2: float,
    s2: float,
    wavenumber: ArrayLike,
    diameter: float | None = None,
    shape: str = 'circular',
) -> tuple[Complex, Complex, Complex, Complex]
```

Decompose the wave field into `(A, B, C, D)` (ASTM E2611-19, Eqs. (17)-(20)).

The exponents are implemented exactly as printed:

$$
A = \frac{j \left( H_1 e^{-jkl_1} - H_2 e^{-jk(l_1+s_1)} \right)} {2 \sin(k s_1)}
$$

$$
B = \frac{j \left( H_2 e^{+jk(l_1+s_1)} - H_1 e^{+jkl_1} \right)} {2 \sin(k s_1)}
$$

$$
C = \frac{j \left( H_3 e^{+jk(l_2+s_2)} - H_4 e^{+jkl_2} \right)} {2 \sin(k s_2)}
$$

$$
D = \frac{j \left( H_4 e^{-jkl_2} - H_3 e^{-jk(l_2+s_2)} \right)} {2 \sin(k s_2)}
$$

`A`/`B` are the forward/backward complex amplitudes on the upstream
(source) side and `C`/`D` those on the downstream side, all referenced
to the front face $x = 0$. With the $e^{+j\omega t}$ /
forward-$e^{-jkx}$
convention these exponents correspond to the microphone whose transfer
function is `H2` sitting nearest the front face at distance `l1` (and
`H1` at $l_1 + s_1$), and to `H3` nearest the downstream side at
`l2` (and `H4` at $l_2 + s_2$), with `l1`, `l2` measured
from the front reference plane. The convention was locked down against the analytic
air-layer transfer matrix (see [`air_layer_transfer_matrix`](/phonometry/reference/api/materials/four-microphone/#air_layer_transfer_matrix)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `h1` | Transfer function `H1,ref` (upstream, farther microphone). |
| `h2` | Transfer function `H2,ref` (upstream, nearer microphone). |
| `h3` | Transfer function `H3,ref` (downstream, nearer microphone). |
| `h4` | Transfer function `H4,ref` (downstream, farther microphone). |
| `l1` | Distance `l1` from the front reference plane, in metres. |
| `s1` | Upstream microphone spacing `s1`, in metres. |
| `l2` | Distance `l2` from the front reference plane, in metres. |
| `s2` | Downstream microphone spacing `s2`, in metres. |
| `wavenumber` | Air wavenumber `k` (real or complex), scalar or per band. |
| `diameter` | Optional tube diameter (circular) or largest section dimension (rectangular/square), in metres, that activates the plane-wave working-range check (6.2.3-6.2.5, 6.5.4). |
| `shape` | Tube cross-section, `"circular"`, `"rectangular"` or `"square"`. |

**Returns:** Tuple `(A, B, C, D)` of complex amplitudes.
