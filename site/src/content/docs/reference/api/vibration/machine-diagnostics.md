---
title: "vibration.machine_diagnostics"
description: "Kinematic fault frequencies of rotating machinery (Norton & Karczub Ch. 8)."
sidebar:
  label: "machine_diagnostics"
---

Kinematic fault frequencies of rotating machinery (Norton & Karczub Ch. 8).

Condition monitoring starts with arithmetic, not with signal processing: every
rolling-contact bearing, gear pair, induction motor and bladed rotor excites a
family of **discrete frequencies fixed by its geometry and its shaft speed**.
Knowing where those lines fall turns a featureless envelope spectrum into a
diagnosis, because a peak is only evidence when it sits on a named line.

This module computes the families set out in M. P. Norton and D. G. Karczub,
*Fundamentals of Noise and Vibration Analysis for Engineers* (2nd ed., CUP
2003), Section 8.4 (8.4.1 gears, 8.4.3 bearings, 8.4.4 fans and blowers,
8.4.7 pumps, 8.4.8 electrical equipment), and hands them to the signal chain
that already exists in `phonometry.metrology`: band-pass the structural
resonance the defect impacts ring, detect its envelope and transform it
([`envelope_spectrum`](/phonometry/reference/api/correlation/envelope/#envelope_spectrum)), average
synchronously with the shaft
([`time_synchronous_average`](/phonometry/reference/api/spectra/synchronous-average/#time_synchronous_average))
or collapse the harmonic families in the cepstrum
([`cepstrum`](/phonometry/reference/api/spectra/cepstrum/#cepstrum)). The result object's
[`FaultFrequencyResult.plot`](/phonometry/reference/api/vibration/machine-diagnostics/#faultfrequencyresultplot) draws the predicted lines **on top of a
measured envelope spectrum**, which is the working view.

**Rolling-contact bearings** (Eqs. 8.4 to 8.14, after Shahan & Kamperman).
With shaft speed `N` in r/min, `Z` rolling elements of diameter `d` on
a pitch diameter `D` and a contact angle `phi`, writing
$g = (d/D) \cos\phi$ and $f_s = N/60$:

$$
\mathrm{FTF} = \frac{f_s}{2}(1 - g) \qquad \text{cage, stationary outer race} \tag{8.5}
$$

$$
\mathrm{FTF_{rel}} = f_s - \mathrm{FTF} \qquad \text{cage seen from the shaft} \tag{8.11}
$$

$$
\mathrm{BSF} = \frac{f_s}{2} \frac{D}{d} (1 - g^2) \qquad \text{element rotation} \tag{8.7}
$$

$$
\mathrm{BDF} = 2\, \mathrm{BSF} \qquad \text{element spin (both races)} \tag{8.10}
$$

$$
\mathrm{BPFO} = Z \frac{f_s}{2}(1 - g) \qquad \text{element pass, outer race} \tag{8.8}
$$

$$
\mathrm{BPFI} = Z \frac{f_s}{2}(1 + g) \qquad \text{element pass, inner race} \tag{8.9}
$$

so $\mathrm{BPFO} + \mathrm{BPFI} = Z f_s$ exactly, and
$\mathrm{BPFO} = Z\, \mathrm{FTF}$. Norton notes that
Eqs. (8.8) and (8.14), and (8.9) and (8.13), are identical: `BPFO` and
`BPFI` do not depend on which race turns. Only the cage does, and with the
outer race rotating it becomes $(f_s/2)(1 + g)$ (Eq. 8.6).

**Gears** (Eq. 8.3). The gear-meshing (tooth-passing) frequency of a wheel
with `n_teeth` teeth is $\mathrm{GMF} = n_{\text{teeth}} f_s$, with
integer harmonics. A discrete tooth fault adds shaft-rate lines and low,
flat sidebands around every mesh harmonic; distributed wear raises tall
sideband groups at $k\, \mathrm{GMF} \pm m f_s$.

**Induction motors** (Eqs. 8.19, 8.20). The three lines always present in a
motor bearing signal are `fs` (mechanical unbalance), $2 f_s$
(misalignment with the driven load) and $2 f_e$ (non-uniform air gap,
torque pulses and the electrical faults). Norton's Eq. (8.19) writes the
supply frequency as $f_e = f_s p / 2$ for `p` magnetic poles, which
is the synchronous, zero-slip form; this module uses the slip-consistent
$f_e = f_s p / (2 (1 - s))$, identical at $s = 0$ and the only
version that makes the rotor-slot harmonics of Eq. (8.20),

$$
f_{sh} = f_e \left[ \frac{2 R}{p}(1 - s) \pm 2 (n - 1) \right] \tag{8.20}
$$

collapse to the physical rotor-bar passing rate $R f_s$ at
$n = 1$ for `R` rotor bars. Dynamic eccentricity dresses that line
with sidebands at $\pm$ the shaft rate and $\pm$ the slip
frequency.

**Fans, blowers and pumps** (Eqs. 8.15 to 8.18). The blade-passing frequency
of an impeller with `N` blades is $f_b = n N f_s$; a pump's
hydraulic pulsations follow the same form with `N` pumping events per
revolution (Eq. 8.18), and a rotary positive-displacement blower repeats
four times per revolution. In a ducted axial fan the blade-vane interaction
sets up rotating pressure patterns with $m_L = n N \pm k V$ lobes for
`V` vanes (Eq. 8.16), turning at $n N f_s / m_L$ (Eq. 8.17): a
pattern that spins faster than the blades themselves, and the reason a
careful choice of `N` and `V` matters for radiated power.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## bearing_fault_frequencies

```python
bearing_fault_frequencies(
    speed_rpm: float,
    n_elements: int,
    element_diameter: float,
    pitch_diameter: float,
    *,
    contact_angle_deg: float = 0.0,
    rotating_race: str = 'inner',
) -> FaultFrequencyResult
```

Kinematic frequencies of a rolling-contact bearing (Norton 8.4-8.14).

Returns the seven lines a bearing generates, named with the acronyms used
in condition monitoring:

============ ================================================= ==========
Name         Meaning                                           Norton eq.
============ ================================================= ==========
`shaft`    shaft rotational frequency `fs`                  (8.4)
`FTF`      cage (fundamental train) frequency                 (8.5)/(8.6)
`FTF_rel`  cage rotation relative to the rotating race        (8.11)/(8.12)
`BSF`      rolling-element rotational frequency               (8.7)
`BDF`      rolling-element spin frequency, `2 BSF`          (8.10)
`BPFO`     element pass frequency on the outer race           (8.8)/(8.14)
`BPFI`     element pass frequency on the inner race           (8.9)/(8.13)
============ ================================================= ==========

`BPFO` and `BPFI` are the outer- and inner-race defect lines and
`BDF` the rolling-element/cage defect line; all three are exact
kinematics of a pure-rolling contact, so a real bearing's lines wander by
1 % to 2 % with load-dependent slip.
$\mathrm{BPFO} + \mathrm{BPFI} = Z f_s$ always, and
both are independent of which race turns (Norton's Eqs. 8.8 and 8.14, and
8.9 and 8.13, are identical); *rotating_race* only moves the cage.

**Parameters**

| Name | Description |
| :--- | :--- |
| `speed_rpm` | Shaft speed `N`, in r/min (> 0). |
| `n_elements` | Number of rolling elements `Z` (integer >= 1). |
| `element_diameter` | Rolling-element diameter `d`, in the same unit as *pitch_diameter* (> 0); only the ratio `d/D` enters. |
| `pitch_diameter` | Bearing pitch diameter `D` (> `d`). |
| `contact_angle_deg` | Contact angle `phi` between element and raceway, in degrees (Default: 0, a radial ball bearing); `0 <= phi < 90`. |
| `rotating_race` | Which race turns with the shaft, `"inner"` (Default) or `"outer"`. |

**Returns:** A [`FaultFrequencyResult`](/phonometry/reference/api/vibration/machine-diagnostics/#faultfrequencyresult) (source `"rolling-contact bearing"`).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive or inconsistent geometry. |

## blade_pass_frequencies

```python
blade_pass_frequencies(
    speed_rpm: float,
    n_blades: int,
    *,
    harmonics: int = 3,
    n_vanes: int | None = None,
    lobe_orders: int = 1,
) -> FaultFrequencyResult
```

Blade-passing frequency and interaction patterns (Norton 8.15-8.17).

$f_b = n\, N_{\text{blades}} \times N/60$ for `n = 1 .. harmonics`,
the discrete tone
family of any fan, blower or pump impeller; a rotary positive-displacement
blower repeats four times per revolution, so pass `4 x` its speed.

In a ducted axial fan the blades also interact with the stator vanes and
set up rotating pressure patterns with
$m_L = n\, N_{\text{blades}} \pm k\, N_{\text{vanes}}$
lobes (Eq. 8.16) turning at
$M_L = n\, N_{\text{blades}} f_s / m_L$ (Eq. 8.17). Those
patterns radiate strongly when they can drive a higher-order duct mode, so
they are the lines to check against the duct cut-on frequencies. Give
*n_vanes* to include them, named `"lobe n=1 m=2"`, `"lobe n=1 m=10"`,
... The blade harmonic is part of the name because Eq. (8.17) carries
`n`: the same lobe count reached from a different harmonic is a distinct
pattern turning at a different speed.

**Parameters**

| Name | Description |
| :--- | :--- |
| `speed_rpm` | Shaft speed `N`, in r/min (> 0). |
| `n_blades` | Number of blades (integer >= 1). |
| `harmonics` | Number of blade-pass harmonics (integer >= 1, Default: 3). |
| `n_vanes` | Number of stator vanes `V` (integer >= 1) to include the lobed interaction patterns (Default: `None`, blade tones only). |
| `lobe_orders` | Highest `\|k\|` in $m_L = n\, N_{\text{blades}} \pm k\, N_{\text{vanes}}$ (integer >= 1, Default: 1). |

**Returns:** A [`FaultFrequencyResult`](/phonometry/reference/api/vibration/machine-diagnostics/#faultfrequencyresult) (source `"bladed rotor"`).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive or non-integer input. |

## combine_fault_lines

```python
combine_fault_lines(
    *results: FaultFrequencyResult,
    source: str | None = None,
) -> FaultFrequencyResult
```

Merge several fault-line families into one overlay.

A gearbox bearing carries its own bearing lines, the mesh family of the
gear it supports and the shaft harmonics; this puts them on one axes.
Duplicate names are disambiguated by appending the source of the family
they came from.

**Parameters**

| Name | Description |
| :--- | :--- |
| `results` | Two or more [`FaultFrequencyResult`](/phonometry/reference/api/vibration/machine-diagnostics/#faultfrequencyresult) objects. |
| `source` | Label for the merged family (Default: the sources joined by `" + "`). |

**Returns:** A [`FaultFrequencyResult`](/phonometry/reference/api/vibration/machine-diagnostics/#faultfrequencyresult) holding every line.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If no result is given, or the shaft rates disagree. |

## FaultFrequencyResult

```python
FaultFrequencyResult(
    lines: tuple[FaultLine, ...],
    shaft_rate: float,
    source: str,
)
```

A family of predicted fault lines for one machine element.

**Attributes**

| Name | Description |
| :--- | :--- |
| `lines` | The predicted [`FaultLine`](/phonometry/reference/api/vibration/machine-diagnostics/#faultline) entries, in the order the generating function produced them. |
| `shaft_rate` | Shaft rotational frequency `fs`, in hertz. |
| `source` | Description of the element (`"rolling-contact bearing"`, `"gear pair"`, ...), used as the plot title. |

### FaultFrequencyResult.as_dict()

```python
FaultFrequencyResult.as_dict() -> dict[str, float]
```

The lines as a `{name: frequency}` mapping.

### FaultFrequencyResult.frequencies

*property*

Predicted frequencies, in hertz, in order.

### FaultFrequencyResult.harmonics()

```python
FaultFrequencyResult.harmonics(name: str, count: int) -> np.ndarray
```

The first *count* integer harmonics of the line called *name*.

**Parameters**

| Name | Description |
| :--- | :--- |
| `name` | Line name (see `names`). |
| `count` | Number of harmonics, $\ge 1$ (the fundamental first). |

**Returns:** Frequencies $n f$ for `n = 1 .. count`, in hertz.

**Raises**

| Exception | When |
| :--- | :--- |
| KeyError | If no line carries that name. |
| ValueError | If *count* is not a positive integer. |

### FaultFrequencyResult.names

*property*

Names of the predicted lines, in order.

### FaultFrequencyResult.orders

*property*

Predicted frequencies in shaft orders, in order.

### FaultFrequencyResult.plot()

```python
FaultFrequencyResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Overlay the predicted lines on a measured envelope spectrum.

Pass the measurement as `spectrum=` (an
[`EnvelopeSpectrumResult`](/phonometry/reference/api/correlation/envelope/#envelopespectrumresult), or any
object exposing `frequencies` and `amplitude`); without it the
predicted lines are drawn alone as a labelled stem plot.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | `spectrum`, `max_frequency` and anything forwarded to the spectrum curve; see `phonometry._plot.vibration.plot_fault_frequencies`. |

### FaultFrequencyResult.within()

```python
FaultFrequencyResult.within(low: float, high: float) -> FaultFrequencyResult
```

The lines falling in `[low, high]` hertz, as a new result.

Handy before plotting on an envelope spectrum whose useful span is
much narrower than the highest predicted harmonic.

**Parameters**

| Name | Description |
| :--- | :--- |
| `low` | Lower edge, in hertz (>= 0). |
| `high` | Upper edge, in hertz (> *low*). |

**Returns:** A [`FaultFrequencyResult`](/phonometry/reference/api/vibration/machine-diagnostics/#faultfrequencyresult) with the surviving lines.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the edges are invalid. |

## FaultLine

```python
FaultLine(
    name: str,
    frequency: float,
    order: float,
    family: str,
    description: str,
)
```

One predicted discrete line of a machine's kinematic signature.

**Attributes**

| Name | Description |
| :--- | :--- |
| `name` | Short label, unique within a result (`"BPFO"`, `"2xGMF"`, `"GMF-1x"`, ...). Acronyms are language neutral and are what the [`FaultFrequencyResult.plot`](/phonometry/reference/api/vibration/machine-diagnostics/#faultfrequencyresultplot) overlay annotates. |
| `frequency` | Predicted frequency, in hertz. |
| `order` | Frequency expressed in shaft orders, `frequency / fs`. |
| `family` | One of `"shaft"`, `"bearing"`, `"gear"`, `"motor"` or `"blade"`. |
| `description` | One-line English description of the mechanism. |

## gear_mesh_frequencies

```python
gear_mesh_frequencies(
    speed_rpm: float,
    n_teeth: int,
    *,
    harmonics: int = 3,
    sidebands: int = 0,
    sideband_rate: float | None = None,
) -> FaultFrequencyResult
```

Gear-meshing frequency and its sideband family (Norton Eq. 8.3).

$\mathrm{GMF} = n_{\text{teeth}} \times N/60$, with integer
harmonics `k GMF` (`k = 1 .. harmonics`) named `"GMF"`,
`"2xGMF"`, ... Each harmonic can carry a
modulation family at `k GMF +/- m f_mod` (`m = 1 .. sidebands`), named
`"GMF-1x"`, `"2xGMF+2x"`, and so on. The default modulation rate is
the shaft rate of the wheel: a chipped tooth or an eccentric wheel
modulates the mesh once per revolution, which is what produces those
sidebands (Norton Figs. 8.23 and 8.24). Pass *sideband_rate* to modulate
at the mating wheel's shaft rate instead.

Only positive sideband frequencies are returned.

**Parameters**

| Name | Description |
| :--- | :--- |
| `speed_rpm` | Shaft speed `N` of the wheel, in r/min (> 0). |
| `n_teeth` | Number of teeth on that wheel (integer >= 1). |
| `harmonics` | Number of mesh harmonics (integer >= 1, Default: 3). |
| `sidebands` | Sideband order per harmonic (integer >= 0, Default: 0, no sidebands). |
| `sideband_rate` | Modulation rate `f_mod`, in hertz (> 0, Default: the wheel's own shaft rate). |

**Returns:** A [`FaultFrequencyResult`](/phonometry/reference/api/vibration/machine-diagnostics/#faultfrequencyresult) (source `"gear pair"`).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive or non-integer input. |

## induction_motor_frequencies

```python
induction_motor_frequencies(
    speed_rpm: float,
    poles: int,
    rotor_bars: int,
    *,
    slip: float = 0.0,
    supply_frequency: float | None = None,
    slot_harmonics: int = 1,
    sidebands: int = 0,
) -> FaultFrequencyResult
```

Electrical and slot lines of an induction motor (Norton 8.19, 8.20).

The three lines always present in a motor bearing vibration signal are
`1x` (mechanical unbalance), `2x` (misalignment with the driven load)
and `2fe` (a non-uniform air gap, torque pulses and the winding/rotor-bar
electrical faults). Rotor defects that produce static or dynamic air-gap
eccentricity are read on the **slot harmonics** of the stator core,

$f_{sh} = f_e\left[\left(\dfrac{2R}{p}\right)(1-s) \pm 2(n-1)\right] = R f_s \pm 2(n-1) f_e$

for `R` rotor bars, `p` magnetic poles (not pole pairs), unit slip
`s` and `n = 1, 2, ...`. Dynamic eccentricity modulates the dominant
slot harmonic at +/- the shaft rate and +/- the slip frequency, which is
what *sidebands* adds around `fsh`.

Give the slip directly, or give the mains *supply_frequency* and let it be
derived from `fe` and the measured shaft speed. The supply frequency is
taken as $f_e = f_s p / (2(1-s))$: Norton's Eq. (8.19) writes
$f_e = f_s p / 2$, which is the same expression at zero slip but
does not reduce Eq. (8.20) to the physical rotor-bar passing rate
`R fs` when the machine is loaded.

With a non-zero slip the pole-pass line $F_P = p\,f_{\text{slip}}$
is included as well. That name is standard condition-monitoring practice
rather than Norton's: he gives the slip frequency itself as the sideband
spacing of a broken rotor bar (his Section 8.4.8) and does not multiply
it by the pole count.

**Parameters**

| Name | Description |
| :--- | :--- |
| `speed_rpm` | Shaft speed `N`, in r/min (> 0). |
| `poles` | Number of magnetic poles `p` (even integer >= 2). |
| `rotor_bars` | Number of rotor bars/slots `R` (integer >= 1). |
| `slip` | Unit slip `s` (`0 <= s < 1`, Default: 0); typically 0,02 to 0,05 under load. Ignored when *supply_frequency* is given. |
| `supply_frequency` | Mains frequency `fe`, in hertz (> 0), from which the slip is derived (Default: `None`, use *slip*). |
| `slot_harmonics` | Number of slot-harmonic orders `n` (integer >= 1, Default: 1, the fundamental `R fs` alone). |
| `sidebands` | Sideband order around the fundamental slot harmonic at +/- the shaft rate and +/- the slip frequency (integer >= 0, Default: 0). |

**Returns:** A [`FaultFrequencyResult`](/phonometry/reference/api/vibration/machine-diagnostics/#faultfrequencyresult) (source `"induction motor"`).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive, non-integer or inconsistent input. |

## shaft_rate

```python
shaft_rate(speed_rpm: float) -> float
```

Shaft rotational frequency $f_s = N/60$ (Norton Eq. 8.4).

**Parameters**

| Name | Description |
| :--- | :--- |
| `speed_rpm` | Shaft speed `N`, in r/min (> 0). |

**Returns:** The shaft rotational frequency $f_s$, in hertz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive speed. |
