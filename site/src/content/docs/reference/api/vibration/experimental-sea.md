---
title: "vibration.structural.experimental_sea"
description: "Experimental statistical energy analysis: coupling loss factors from measured energies (Norton & Karczub Ch. 6)."
sidebar:
  label: "experimental_sea"
---

Experimental statistical energy analysis: coupling loss factors from measured
energies (Norton & Karczub Ch. 6).

Statistical energy analysis (SEA) has two routes to the coupling loss factor
`eta_ij`. The **predictive** route derives it from a wave transmission
coefficient at the junction, which is what
[`coupling_loss_factor`](/phonometry/reference/api/vibration/junction-transmission/#coupling_loss_factor) does
with the closed-form coefficients of
[`junction_transmission`](/phonometry/reference/api/vibration/junction-transmission/#junction_transmission). The
**experimental** route, implemented here, inverts the steady-state power
balance from *measured* subsystem energies: it needs no model of the junction
at all, and it is the only route open for real joints (welds, bolt rows,
spot welds, adhesives) whose wave behaviour is not tractable.

For two subsystems the steady-state power balance is (Norton Eqs. 6.10 and
6.11, generalised to a drive on either subsystem):

$$
\Pi_1 = \omega \left[ (\eta_1 + \eta_{12}) E_1 - \eta_{21} E_2 \right] \tag{6.10}
$$

$$
\Pi_2 = \omega \left[ (\eta_2 + \eta_{21}) E_2 - \eta_{12} E_1 \right] \tag{6.11}
$$

with $E_i = M_i \langle v_i^2 \rangle$ the band energy of subsystem
`i` (mass times the space- and time-averaged mean-square velocity),
`eta_i` its internal loss factor and `omega` the band centre frequency
in rad/s. Two inversions follow.

**Single drive plus reciprocity** ([`power_injection_clf`](/phonometry/reference/api/vibration/experimental-sea/#power_injection_clf)). Drive
subsystem 1 only. The second equation with $\Pi_2 = 0$ gives one
relation between `eta_12` and `eta_21`; the SEA consistency
(reciprocity) relationship $\eta_{12} n_1 = \eta_{21} n_2$ (Eq. 6.8)
supplies the second, so with the modal densities `n_1`, `n_2` known
(Eq. 6.15):

$$
\eta_{12} = \frac{\eta_2 E_2}{E_1 - E_2\, n_1/n_2}
$$

$$
\eta_{21} = \eta_{12} \frac{n_1}{n_2}
$$

$$
\Pi_{\text{in}} = \omega (\eta_1 E_1 + \eta_2 E_2)
$$

The input power reduces to the total dissipated power, as it must in the
steady state: substituting the balance of subsystem 2 into Eq. (6.10)
cancels the two coupling terms exactly. The bracket
$E_1 - E_2\, n_1/n_2$ is positive exactly when the *modal* energy of
the driven subsystem exceeds that of the receiver,
$E_1/n_1 > E_2/n_2$; a measurement that violates it is not a
two-subsystem SEA system and is rejected.

**Two drives, no reciprocity assumed** ([`power_injection_matrix`](/phonometry/reference/api/vibration/experimental-sea/#power_injection_matrix)). The
classical power-injection method drives each subsystem in turn and measures
both energies each time, giving four equations for the four unknowns
`eta_1`, `eta_2`, `eta_12`, `eta_21` with no prior assumption at all.
Reciprocity then becomes a *check* on the measurement rather than an input:
[`PowerInjectionResult.modal_density_ratio`](/phonometry/reference/api/vibration/experimental-sea/#powerinjectionresultmodal_density_ratio) compares the measured
$\eta_{12}/\eta_{21}$ with the expected $n_2/n_1$.

Modal densities for the usual subsystems are provided as well
([`flat_plate_modal_density`](/phonometry/reference/api/vibration/experimental-sea/#flat_plate_modal_density), [`cylindrical_shell_modal_density`](/phonometry/reference/api/vibration/experimental-sea/#cylindrical_shell_modal_density),
[`bar_modal_density`](/phonometry/reference/api/vibration/experimental-sea/#bar_modal_density), [`beam_modal_density`](/phonometry/reference/api/vibration/experimental-sea/#beam_modal_density)), following Norton
Eqs. 6.23-6.29. The flat-plate expression
$n(f) = S \sqrt{12} / (2 c_\mathrm{L} t)$ is the same quantity as EN 12354-4's
$n = \pi S f_\mathrm{c} / c_0^2$
([`modal_density`](/phonometry/reference/api/building/flanking-transmission/#modal_density)), only
parametrised by the plate itself rather than by its critical frequency; the
two agree identically and a regression test pins that.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## bar_modal_density

```python
bar_modal_density(length: float, longitudinal_wave_speed: float) -> float
```

Modal density of a uniform bar in longitudinal vibration (Eq. 6.23).

$n(f) = 2 L / c_\mathrm{L}$, independent of frequency.

**Parameters**

| Name | Description |
| :--- | :--- |
| `length` | Bar length `L`, in m (> 0). |
| `longitudinal_wave_speed` | Bar wave speed $c_\mathrm{L} = \sqrt{E/\rho}$, in m/s (> 0). |

**Returns:** The modal density `n(f)`, in modes per hertz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input. |

## beam_modal_density

```python
beam_modal_density(
    frequency: ArrayLike,
    length: float,
    mass_per_length: float,
    bending_stiffness: float,
) -> NDArray[np.float64]
```

Modal density of a uniform beam in flexure (Norton Eq. 6.24).

$n(f) = L (\rho A / E I)^{1/4} / \sqrt{2\pi f}$: unlike every
other subsystem here, it *decreases* with frequency.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Band centre frequency `f`, in hertz (scalar or array, > 0). |
| `length` | Beam length `L`, in m (> 0). |
| `mass_per_length` | Mass per unit length `rho A`, in kg/m (> 0). |
| `bending_stiffness` | Flexural stiffness `E I`, in N.m^2 (> 0). |

**Returns:** The modal density `n(f)`, in modes per hertz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input. |

## cylindrical_shell_modal_density

```python
cylindrical_shell_modal_density(
    frequency: ArrayLike,
    area: float,
    thickness: float,
    mean_radius: float,
    longitudinal_wave_speed: float,
    *,
    band: str = 'octave',
) -> NDArray[np.float64]
```

Average modal density of a thin-walled cylinder (Norton 6.27-6.29).

The semi-empirical approximations of Szechenyi, as collected by Clarkson &
Pope, in three regimes of $x = f / f_\mathrm{r}$ around the ring frequency
[`ring_frequency`](/phonometry/reference/api/vibration/experimental-sea/#ring_frequency):

$$
n = \frac{5 S}{\pi c_\mathrm{L} t} \sqrt{x}, \qquad x \le 0.48 \tag{6.27}
$$

$$
n = \frac{7.2 S}{\pi c_\mathrm{L} t}\, x, \qquad 0.48 < x \le 0.83 \tag{6.28}
$$

$$
n = \frac{2 S}{\pi c_\mathrm{L} t} \left[ 2 + \frac{0.596}{F - 1/F} \left( F \arccos\frac{1.745}{F^2 x^2} - \frac{1}{F} \arccos\frac{1.745 F^2}{x^2} \right) \right], \qquad x > 0.83 \tag{6.29}
$$

with the bandwidth factor
$F = \sqrt{f_{\text{upper}}/f_{\text{lower}}}$. These are
*average* values: they do not resolve the large fluctuations that the
cut-on of successive circumferential orders produces below the ring
frequency, which for long thin shells can be substantial.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Band centre frequency `f`, in hertz (scalar or array, > 0). |
| `area` | Shell surface area `S`, in m^2 (> 0). |
| `thickness` | Wall thickness `t`, in m (> 0). |
| `mean_radius` | Mean shell radius `a_m`, in m (> 0). |
| `longitudinal_wave_speed` | Plate wave speed `cL`, in m/s (> 0). |
| `band` | Analysis bandwidth, `"octave"` (Default, $F = 1.414$) or `"third"` ($F = 1.122$). |

**Returns:** The modal density `n(f)`, in modes per hertz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input or an unknown band. |

## flat_plate_modal_density

```python
flat_plate_modal_density(
    area: float,
    thickness: float,
    longitudinal_wave_speed: float,
) -> float
```

Modal density of a flat plate in flexure (Norton Eq. 6.25).

$n(f) = S \sqrt{12} / (2 c_\mathrm{L} t)$, independent of frequency, with
the plate (quasi-longitudinal) wave speed
$c_\mathrm{L} = \sqrt{E / (\rho (1 - \nu^2))}$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `area` | Plate surface area `S`, in m^2 (> 0). |
| `thickness` | Plate thickness `t`, in m (> 0). |
| `longitudinal_wave_speed` | Plate wave speed `cL`, in m/s (> 0). |

**Returns:** The modal density `n(f)`, in modes per hertz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input. |

## power_injection_clf

```python
power_injection_clf(
    frequency: ArrayLike,
    energy1: ArrayLike,
    energy2: ArrayLike,
    internal_loss_factor1: ArrayLike,
    internal_loss_factor2: ArrayLike,
    modal_density1: ArrayLike,
    modal_density2: ArrayLike,
) -> PowerInjectionResult
```

Coupling loss factors from a single-drive energy measurement.

Subsystem 1 is driven and subsystem 2 receives power only through the
junction. Inverting the steady-state balance of subsystem 2 together with
the reciprocity relationship $\eta_{12} n_1 = \eta_{21} n_2$ gives

$$
\eta_{12} = \frac{\eta_2 E_2}{E_1 - E_2\, n_1/n_2}
$$

$$
\eta_{21} = \eta_{12} \frac{n_1}{n_2}
$$

$$
\Pi_{\text{in}} = \omega (\eta_1 E_1 + \eta_2 E_2)
$$

Energies are $E_i = M_i \langle v_i^2 \rangle$ with `M_i` the
subsystem mass and $\langle v_i^2 \rangle$ the space- and
time-averaged mean-square velocity in the band.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Band centre frequencies `f`, in hertz (> 0). |
| `energy1` | Band energy `E_1` of the driven subsystem, in J (> 0). |
| `energy2` | Band energy `E_2` of the receiving subsystem, in J (> 0). |
| `internal_loss_factor1` | `eta_1` (> 0), scalar or per band. |
| `internal_loss_factor2` | `eta_2` (> 0), scalar or per band. |
| `modal_density1` | `n_1`, in modes per hertz (> 0). |
| `modal_density2` | `n_2`, in modes per hertz (> 0). |

**Returns:** A [`PowerInjectionResult`](/phonometry/reference/api/vibration/experimental-sea/#powerinjectionresult) (method `"single-drive"`).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input, mismatched band lengths, or a measurement with $E_1/n_1 \le E_2/n_2$ (the receiving subsystem holds at least as much modal energy as the driven one, so no two-subsystem SEA model fits it). |

## power_injection_matrix

```python
power_injection_matrix(
    frequency: ArrayLike,
    energies: ArrayLike,
    input_powers: ArrayLike,
) -> PowerInjectionResult
```

Full two-drive power-injection inversion (no reciprocity assumed).

Each subsystem is driven in turn with a known injected power while both
band energies are measured, giving four equations for the four unknowns
`eta_1`, `eta_2`, `eta_12`, `eta_21`. Because reciprocity is not
used, [`PowerInjectionResult.modal_density_ratio`](/phonometry/reference/api/vibration/experimental-sea/#powerinjectionresultmodal_density_ratio) becomes an
independent check on the measurement.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Band centre frequencies `f`, in hertz (> 0), shape `(nb,)`. |
| `energies` | Measured band energies, in joules, shape `(2, 2, nb)`: `energies[i][j]` is the energy of subsystem `i` while subsystem `j` is driven (all > 0). |
| `input_powers` | Injected powers, in watts, shape `(2, nb)`: `input_powers[j]` is the power injected into subsystem `j` during test `j` (all > 0). |

**Returns:** A [`PowerInjectionResult`](/phonometry/reference/api/vibration/experimental-sea/#powerinjectionresult) (method `"two-drive"`); its `energy1`/`energy2` and `input_power1`/`input_power2` report the first test (subsystem 1 driven).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a bad shape, a non-positive value, or a singular energy matrix in some band. |

## PowerInjectionResult

```python
PowerInjectionResult(
    frequencies: NDArray[np.float64],
    coupling_loss_factor12: NDArray[np.float64],
    coupling_loss_factor21: NDArray[np.float64],
    internal_loss_factor1: NDArray[np.float64],
    internal_loss_factor2: NDArray[np.float64],
    energy1: NDArray[np.float64],
    energy2: NDArray[np.float64],
    input_power1: NDArray[np.float64],
    input_power2: NDArray[np.float64],
    modal_density1: NDArray[np.float64] | None,
    modal_density2: NDArray[np.float64] | None,
    method: str,
)
```

Loss-factor budget of a two-subsystem SEA model, per band.

All arrays share the band axis `frequencies`.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies `f`, in hertz. |
| `coupling_loss_factor12` | `eta_12`, subsystem 1 to 2. |
| `coupling_loss_factor21` | `eta_21`, subsystem 2 to 1. |
| `internal_loss_factor1` | `eta_1` of subsystem 1. |
| `internal_loss_factor2` | `eta_2` of subsystem 2. |
| `energy1` | Band energy `E_1`, in joules. |
| `energy2` | Band energy `E_2`, in joules. |
| `input_power1` | Power injected into subsystem 1, in watts. |
| `input_power2` | Power injected into subsystem 2, in watts. |
| `modal_density1` | Modal density `n_1`, in modes per hertz, or `None` when the inversion did not need it. |
| `modal_density2` | Modal density `n_2`, or `None`. |
| `method` | `"single-drive"` (reciprocity assumed) or `"two-drive"` (full power-injection matrix). |

### PowerInjectionResult.coupling_strength

*property*

Coupling ratio $\eta_{12} / \eta_1$.

SEA subsystems should be *weakly* coupled: values well below 1 mean
the junction leaks far less power than the subsystem dissipates, the
regime where the two-subsystem model is trustworthy.

### PowerInjectionResult.dissipated_power

*property*

Total power dissipated internally, in watts.

$\omega (\eta_1 E_1 + \eta_2 E_2)$, which equals
`input_power` in the steady state and is the round-trip check
of the inversion.

### PowerInjectionResult.input_power

*property*

Total power injected into the system, in watts.

### PowerInjectionResult.modal_density_ratio

*property*

Ratio $n_1 / n_2$ implied by the measured coupling loss
factors.

From the reciprocity relationship
$\eta_{12} n_1 = \eta_{21} n_2$, so it equals
$\eta_{21} / \eta_{12}$. For a `"two-drive"` inversion, where
reciprocity was never assumed, comparing this with the modal densities
computed from the geometry is the consistency check on the
measurement.

### PowerInjectionResult.plot()

```python
PowerInjectionResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the loss-factor budget against frequency.

The two coupling loss factors and the two internal loss factors on one
log axis: their ordering is the whole diagnosis, because SEA is only
valid where the coupling stays below the internal damping.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the `eta_12` curve. |

### PowerInjectionResult.transmitted_power

*property*

Net power flowing from subsystem 1 to 2, in watts.

$\Pi_{12} = \omega (\eta_{12} E_1 - \eta_{21} E_2)$; negative
when the net flow runs the other way.

## ring_frequency

```python
ring_frequency(mean_radius: float, longitudinal_wave_speed: float) -> float
```

Ring frequency of a cylindrical shell (Norton Eq. 6.26).

$f_\mathrm{r} = c_\mathrm{L} / (2\pi a_\mathrm{m})$: the frequency at which the shell vibrates
uniformly in the breathing mode. Above it a cylinder behaves like a flat
plate; below it the modes group by circumferential order and the modal
density is no longer a simple function of frequency.

**Parameters**

| Name | Description |
| :--- | :--- |
| `mean_radius` | Mean shell radius `a_m`, in m (> 0). |
| `longitudinal_wave_speed` | Plate wave speed `cL`, in m/s (> 0). |

**Returns:** The ring frequency `fr`, in hertz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input. |
