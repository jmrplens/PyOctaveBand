---
title: "vibration.transfer_stiffness"
description: "Dynamic transfer stiffness of resilient elements (ISO 10846-1/-2/-3)."
sidebar:
  label: "transfer_stiffness"
---

Dynamic transfer stiffness of resilient elements (ISO 10846-1/-2/-3).

The vibro-acoustic transfer property of a resilient element (a vibration
isolator, mount, bellows or hose) is its **dynamic transfer stiffness**: the
frequency-dependent ratio of the *blocking force* phasor `F2,b` on the output
(receiver) side to the displacement phasor `u1` on the input (source) side,
with the output blocked (ISO 10846-1, 3.7), in N/m:

$$
k_{2,1} = \frac{F_{2,b}}{u_1}
$$

For an isolator between two structures of large driving-point stiffness, the
force delivered to the receiver approximates this blocking force (ISO 10846-1,
Equation 7), so $k_{2,1}$ characterises the isolator's transmission.
Results are reported as a **level**, in dB, re the reference stiffness
$k_0 = 1$ N/m (ISO 10846-2 and -3, 3.17):

$$
L_k = 10 \lg\!\left( \frac{|k_{2,1}|^2}{k_0^2} \right) = 20 \lg\!\left( \frac{|k_{2,1}|}{k_0} \right)
$$

and, in the low-frequency range where inertial forces in the element are
negligible, the **loss factor** is the tangent of the phase angle of
$k_{2,1}$ (ISO 10846-1, 3.8):
$\eta = \operatorname{Im}(k_{2,1}) / \operatorname{Re}(k_{2,1})$.

Two laboratory methods determine $k_{2,1}$:

* **Direct method** (ISO 10846-2): measure the blocked output force `F2,b`
  and the input displacement `u1` directly:
  $k_{2,1} = F_{2,b} / u_1$.
* **Indirect method** (ISO 10846-3): load the output with a compact blocking
  mass `m2` and measure the vibration transmissibility
  $T = u_2/u_1$; the blocking force is the mass's inertia force
  (ISO 10846-3, Equation 1):
  $k_{2,1} = -(2\pi f)^2 (m_2 + m_f) T$ for $T \ll 1$,
  where `mf` is the mass of the output flange of the test element. The
  approximation is valid only where $|T| \le 0.1$ (Inequality (2):
  $\Delta L_{1,2} \ge 20$ dB) and while the blocking mass still
  behaves rigidly, $10 \lg(m_{2,\mathrm{eff}}^2/m_2^2) \le 1$ dB
  (Inequality (3)); see [`transfer_stiffness_indirect`](/phonometry/reference/api/vibration/transfer-stiffness/#transfer_stiffness_indirect).

The dynamic transfer stiffness is a member of the frequency-response-function
family (ISO 10846-1, Annex A / Table A.2):
$k = j\omega Z = -\omega^2 m_{\mathrm{eff}}$,
so it converts to mechanical impedance and effective mass through
[`phonometry.convert_frf`](/phonometry/reference/api/vibration/mechanical-mobility/#convert_frf) (`"dynamic_stiffness"` \<-> `"impedance"` \<->
`"apparent_mass"`). This module feeds the structure-borne source and building
prediction standards (ISO 9611, EN 15657, EN 12354-5).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## base_transmissibility

```python
base_transmissibility(
    frequency: ArrayLike,
    mass: float,
    stiffness: float,
    damping: float = 0.0,
) -> np.ndarray
```

Transmissibility of a mass on an ideal resilient element (model).

The output mass `m` on a massless Kelvin-Voigt element (spring `k` in
parallel with a viscous damper `c`) driven at the input has the
base-excitation transmissibility

$$
T = \frac{u_2}{u_1} = \frac{k + j\omega c}{k - \omega^2 m + j\omega c}
$$

This ideal-element model is the counterpart of the indirect-method test
arrangement (ISO 10846-3): feeding `T` into
[`transfer_stiffness_indirect`](/phonometry/reference/api/vibration/transfer-stiffness/#transfer_stiffness_indirect) with the same mass recovers the
element's transfer stiffness $k + j\omega c$ in the high-frequency
limit $T \ll 1$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency `f`, in hertz (scalar or array). |
| `mass` | Output mass `m`, in kg. |
| `stiffness` | Element stiffness `k`, in N/m. |
| `damping` | Viscous damping `c`, in N.s/m (Default: 0.0). |

**Returns:** The complex transmissibility `T`.

## blocking_force_ratio

```python
blocking_force_ratio(
    driving_point_stiffness: ArrayLike,
    termination_stiffness: ArrayLike,
) -> np.ndarray
```

Ratio of the delivered force to the blocking force (ISO 10846-1, Eq. 6).

For an isolator driving a receiving structure, the output force for a
given source displacement `u1` is
$F_2 = k_{2,1} u_1 / (1 + k_{2,2}/k_t)$
(Equation (6)), where `k2,2` is the isolator's output driving-point
stiffness (output blocked at the input) and `kt` the dynamic
driving-point stiffness of the termination. This function returns

$$
\frac{F_2}{F_{2,b}} = \frac{1}{1 + k_{2,2}/k_t}
$$

the factor by which the delivered force deviates from the blocking force
$F_{2,b} = k_{2,1} u_1$ of Equation (7). For
$|k_{2,2}| < 0.1 |k_t|$ the ratio is within 10 % of unity
($1/1.1 = 0.909$ at the limit), which is the
stiffness mismatch that justifies characterising an isolator by its
blocked transfer stiffness alone.

**Parameters**

| Name | Description |
| :--- | :--- |
| `driving_point_stiffness` | Output driving-point stiffness `k2,2` of the isolator (complex, scalar or array), in N/m. |
| `termination_stiffness` | Driving-point stiffness `kt` of the receiving structure (complex, scalar or array, non-zero), in N/m. |

**Returns:** The complex ratio `F2/F2,b`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a zero termination stiffness. |

## indirect_transfer_stiffness_result

```python
indirect_transfer_stiffness_result(
    frequency: ArrayLike,
    transmissibility: ArrayLike,
    blocking_mass: float,
    *,
    flange_mass: float = 0.0,
) -> TransferStiffnessResult
```

Indirect-method transfer stiffness bundled as a [`TransferStiffnessResult`](/phonometry/reference/api/vibration/transfer-stiffness/#transferstiffnessresult).

See [`transfer_stiffness_indirect`](/phonometry/reference/api/vibration/transfer-stiffness/#transfer_stiffness_indirect) for the ISO 10846-3 validity
conditions (Inequalities (2) and (3)); bands with $|T| > 0.1$
trigger a [`PhonometryWarning`](/phonometry/reference/api/filters/phonometry/#phonometrywarning).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequencies `f`, in hertz (array). |
| `transmissibility` | Vibration transmissibility $T = u_2/u_1$ (complex). |
| `blocking_mass` | Blocking mass `m2`, in kg (> 0). |
| `flange_mass` | Output-flange mass `mf`, in kg (Default: 0.0). |

**Returns:** The [`TransferStiffnessResult`](/phonometry/reference/api/vibration/transfer-stiffness/#transferstiffnessresult) (indirect method).

**Warns**

| Warning | When |
| :--- | :--- |
| PhonometryWarning | where any $\lvert T\rvert > 0.1$ (Inequality (2) violated). |

## loss_factor

```python
loss_factor(stiffness: ArrayLike) -> np.ndarray
```

Loss factor
$\eta = \operatorname{Im}(k_{2,1}) / \operatorname{Re}(k_{2,1})$
(ISO 10846-1, 3.8).

Valid in the low-frequency range where inertial forces in the element are
negligible; it is the tangent of the phase angle of the transfer stiffness.

**Parameters**

| Name | Description |
| :--- | :--- |
| `stiffness` | Dynamic transfer stiffness $k_{2,1}$ (complex, scalar or array, with a non-zero real part), in N/m. |

**Returns:** The loss factor `eta` (dimensionless).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a purely imaginary stiffness ($\operatorname{Re}(k_{2,1}) = 0$), for which the loss factor is undefined. |

## REFERENCE_STIFFNESS

*Constant* (`float`).

```python
REFERENCE_STIFFNESS = 1.0
```

## transfer_stiffness_direct

```python
transfer_stiffness_direct(
    blocking_force: ArrayLike,
    input_displacement: ArrayLike,
) -> np.ndarray
```

Dynamic transfer stiffness by the direct method (ISO 10846-2).

$k_{2,1} = F_{2,b} / u_1$, the blocked output force phasor over
the input displacement phasor.

**Parameters**

| Name | Description |
| :--- | :--- |
| `blocking_force` | Blocked output force phasor `F2,b` (complex), in N. |
| `input_displacement` | Input displacement phasor `u1` (complex, non-zero), in m. |

**Returns:** The dynamic transfer stiffness $k_{2,1}$, in N/m.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a zero input displacement (dead input channel). |

## transfer_stiffness_indirect

```python
transfer_stiffness_indirect(
    frequency: ArrayLike,
    transmissibility: ArrayLike,
    blocking_mass: float,
    *,
    flange_mass: float = 0.0,
) -> np.ndarray
```

Dynamic transfer stiffness by the indirect method (ISO 10846-3, Eq. 1).

$k_{2,1} = -(2\pi f)^2 (m_2 + m_f) T$: the blocking force is the
inertia force of a compact blocking mass `m2` (plus the output flange
mass `mf`), derived from the measured vibration transmissibility
$T = u_2/u_1$. Valid for $T \ll 1$ (i.e. well above the
mass/spring resonance).

**Validity (ISO 10846-3, clause 6).** The $T \ll 1$ approximation
of Formula (1) is required accurate within 1 dB, i.e. within 12 % of the
calculated stiffness magnitude. This holds only where Inequality (2) is
met: $\Delta L_{1,2} = L_{a1} - L_{a2} \ge 20$ dB, i.e.
$|T| \le 0.1$ ([`TRANSMISSIBILITY_LIMIT`](/phonometry/reference/api/vibration/transfer-stiffness/#transmissibility_limit)). Bands with
`|T|` above that limit
(routine near or below the mass/spring resonance) trigger a
[`PhonometryWarning`](/phonometry/reference/api/filters/phonometry/#phonometrywarning); treat those bands as outside the
valid frequency range of the test arrangement. The upper frequency limit
`f3` additionally requires the blocking mass to vibrate as a rigid
body: results are valid only while its effective mass `m2,eff`,
measured per Formula (4) as
$m_{2,\mathrm{eff}} = 2 F_2 / (a'_1 + a''_1)$ (two accelerometers
spaced $D = \sqrt{S}$ across the contact area), stays within 1 dB
of the rigid mass, $10 \lg(m_{2,\mathrm{eff}}^2 / m_2^2) \le 1$ dB
(Inequality (3), 6.2.3).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency `f`, in hertz (scalar or array). |
| `transmissibility` | Vibration transmissibility $T = u_2/u_1$ (complex, scalar or array; velocity and acceleration ratios have the same value). |
| `blocking_mass` | Blocking mass `m2`, in kg (> 0). |
| `flange_mass` | Output-flange mass `mf`, in kg (Default: 0.0). |

**Returns:** The dynamic transfer stiffness $k_{2,1}$, in N/m.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive frequency or blocking mass. |

**Warns**

| Warning | When |
| :--- | :--- |
| PhonometryWarning | where any $\lvert T\rvert > 0.1$ (Inequality (2) violated). |

## transfer_stiffness_level

```python
transfer_stiffness_level(
    stiffness: ArrayLike,
    *,
    reference: float = 1.0,
) -> np.ndarray
```

Level of the dynamic transfer stiffness (ISO 10846-2/-3, 3.17).

$L_k = 20 \lg(|k_{2,1}| / k_0)$ dB, with `k0` the reference
stiffness.

**Parameters**

| Name | Description |
| :--- | :--- |
| `stiffness` | Dynamic transfer stiffness $k_{2,1}$ (complex or real, scalar or array, non-zero), in N/m. |
| `reference` | Reference stiffness `k0` (Default: 1 N/m), in N/m. |

**Returns:** The level `L_k`, in dB re `k0`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive reference or a zero stiffness magnitude (a dead channel has no level). |

## TransferStiffnessResult

```python
TransferStiffnessResult(
    frequencies: np.ndarray,
    transfer_stiffness: np.ndarray,
    blocking_mass: float | None = None,
)
```

A dynamic transfer stiffness over frequency (ISO 10846).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies, in hertz. |
| `transfer_stiffness` | Complex $k_{2,1}$ per frequency, in N/m. |
| `blocking_mass` | Blocking mass `m2` used (indirect method), in kg, or `None` for the direct method. |

### TransferStiffnessResult.level

*property*

Transfer-stiffness level `L_k` re 1 N/m, in dB (3.17).

### TransferStiffnessResult.loss_factor

*property*

Loss factor $\eta = \operatorname{Im}/\operatorname{Re}$
per frequency (3.8).

### TransferStiffnessResult.magnitude

*property*

Transfer-stiffness magnitude $|k_{2,1}|$, in N/m.

### TransferStiffnessResult.plot()

```python
TransferStiffnessResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the transfer-stiffness level `L_k(f)`.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### TransferStiffnessResult.report()

```python
TransferStiffnessResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render a dynamic-transfer-stiffness fiche to a PDF (ISO 10846).

Writes a one-page transfer-stiffness characterisation report for a
resilient element: the standard-basis line naming the determination
method (direct, ISO 10846-2:2008, or indirect blocking-mass,
ISO 10846-3:2002; definition per ISO 10846-1:2008), an optional metadata
header, a two-panel body with a compact table of the FRF's
characteristic points (the method, the blocking mass for the indirect
method, the frequency range, and the low-frequency stiffness plateau
$|k_{2,1}|$, its level `L_k` and the loss factor `eta`
there) beside
the transfer-stiffness level spectrum `L_k(f)`, a boxed low-frequency
`L_k` with the stiffness magnitude and method alongside, and a footer
identity/disclaimer block.

Dynamic transfer stiffness is a continuous frequency-response function,
so the fiche presents it as a spectrum plus a table of characteristic
points; a transfer-stiffness determination is a characterisation, so
there is no pass/fail verdict.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header identity (`specimen` is the tested resilient element) and the footer identity; the `requirement` field is ignored. |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | Accepted for a uniform `.report()` signature; the transfer-stiffness fiche has a single body layout, so it has no effect. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is unknown. |
| ImportError | If reportlab or matplotlib is not installed. The fiche always embeds the `L_k(f)` spectrum, so both are required (`pip install "phonometry[report,plot]"`). |

### TransferStiffnessResult.to()

```python
TransferStiffnessResult.to(target: str) -> np.ndarray
```

Convert $k_{2,1}$ to a related FRF (ISO 10846-1 Annex A).

`target` is `"impedance"` ($Z = k/(j\omega)$) or
`"apparent_mass"` ($m_{\mathrm{eff}} = -k/\omega^2$); see
[`phonometry.convert_frf`](/phonometry/reference/api/vibration/mechanical-mobility/#convert_frf).

## TRANSMISSIBILITY_LIMIT

*Constant* (`float`).

```python
TRANSMISSIBILITY_LIMIT = 0.1
```
