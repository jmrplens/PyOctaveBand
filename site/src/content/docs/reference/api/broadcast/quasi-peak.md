---
title: "broadcast.quasi_peak"
description: "Psophometric quasi-peak programme-level meter (ITU-R BS.468-4 clause 2)."
sidebar:
  label: "quasi_peak"
---

Psophometric quasi-peak programme-level meter (ITU-R BS.468-4 clause 2).

ITU-R BS.468-4 specifies the instrument that measures audio-frequency noise
voltage in sound broadcasting as two parts. Clause 1 is the weighting
network, realised elsewhere in this library
([`phonometry.filters.weighting_filter`](/phonometry/reference/api/filters/weighting/#weighting_filter) with `curve="468"`, from the
Fig. 1a ladder). Clause 2 is the detector this module implements: a
**quasi-peak** value method, whose reading is reported in **dBqps**
(clause 3).

**Clause 2 prints no dynamics.** It gives no time constant, no rise time, no
decay law, no transfer function and no differential equation. The words
"time constant" occur once in the whole Recommendation, in an informative
Note that offers "two peak rectifier circuits of different time-constants
connected in tandem" as *a possible arrangement* after full-wave
rectification, and the preamble says the performance "may be realized in a
variety of ways". What clause 2 does give is eleven acceptance windows:
Table 2 reads a single 5 kHz tone burst at eight durations from 1 ms to
200 ms, Table 3 reads a train of 5 ms bursts at 2, 10 and 100 per second,
and each cell carries a reference reading with a lower and an upper limit,
expressed as a percentage of the reading the same tone gives steadily. Those
eleven windows are the entire specification of the dynamics, and they are
what [`verify_quasi_peak_dynamics`](/phonometry/reference/api/broadcast/quasi-peak/#verify_quasi_peak_dynamics) checks.

The one absolute statement clause 2 does make is clause 2.6: a steady 1 kHz
sine at 0.775 V r.m.s. shall read 0.775 V, that is 0 dBqps. So the detector
reads the **r.m.s.** of a steady sine, not its peak, and that fixes the scale
of the whole instrument (NOTE 2 to clause 1: "The whole instrument is
calibrated at 1 kHz"). Here that scale is not a stored number but a
measurement: [`quasi_peak_meter`](/phonometry/reference/api/broadcast/quasi-peak/#quasi_peak_meter) runs its own chain on a 1 kHz sine of
1 V r.m.s. at the caller's sample rate and divides by what it reads, so
clause 2.6 holds exactly at every rate rather than at the one rate a frozen
constant would have been fitted at.

What this module chooses, because the Recommendation does not
--------------------------------------------------------------

Six decisions, none of them in BS.468-4, all of them inseparable: change any
one and [`BS468_BALLISTICS`](/phonometry/reference/api/broadcast/quasi-peak/#bs468_ballistics) has to be refitted.

* **Full-wave rectification** as $|x|$, following the Note's own
  wording. An analytic-signal envelope also meets all eleven windows when it
  is refitted from scratch, with a three times larger worst-case residual.
* **A peak rectifier** as a one-pole recursion with different rise and fall
  rates, the same asymmetric kernel the impulse time weighting uses
  ([`phonometry.filters.weighting.time_weighting`](/phonometry/reference/api/filters/weighting/#time_weighting)), discharging toward
  the input rather than toward zero. Refitting toward zero instead moves the
  worst residual by 0.002 dB, so the discharge law is not load-bearing.
* **One symmetric reading device** after it, first order. Clause 2.5 treats
  "the reading device" as a separate object, and a peak rectifier alone
  cannot meet the tables: its best fit is 4.3 dB out at the worst point and
  leaves only 4 of the 11 readings inside their windows.
* **The reading is the maximum of the reading device's output over the
  record**, which is what an operator writes down from a pointer. The record
  therefore has to include the decay: a burst that ends at the last sample
  is read before the needle has finished rising.
* **The reference for Tables 2 and 3** is the same chain fed a steady 5 kHz
  tone through the same network, so both tables are ratios and every
  absolute factor, the network's +11.7 dB at 5 kHz included, cancels.
* **The ballistics run at the caller's rate** on the record
  [`phonometry.filters.weighting_filter`](/phonometry/reference/api/filters/weighting/#weighting_filter) returns. Measured across 32,
  44.1, 48, 96 and 192 kHz, no reading moves by more than 0.080 dB, or 4.2 %
  of its own window.

What no oracle can check
------------------------

The tables use one carrier, one waveform, one amplitude, eight durations and
three repetition rates, and that is the whole specification. There is no
reference implementation, no tabulated impulse response and no worked
example in BS.468-4, in IEC 60268-1:1985 Appendix A or in either 1988
amendment. So the **response to noise**, which is the instrument's entire
purpose, is unverifiable: two conforming detectors will disagree on noise,
on speech and on programme material by an amount nothing in the document
bounds. So is the relationship between a dB(468) quasi-peak figure and a
dB(A) r.m.s. one, and so is everything outside the tabulated stimuli
(bursts shorter than 0.6 ms or longer than 200 ms, carriers other than
5 kHz, anything that is not a gated sine).

Four clauses are computable and pass **by construction** rather than by
measurement, because rectification, first-order recursions and a maximum are
all positively homogeneous and cannot overshoot a monotone step: clause 2.1's
two attenuator variants, clause 2.3's +-1 dB over 20 dB of 0.6 ms bursts,
clause 2.4's 0.5 dB reversibility ($|-x| = |x|$, so the difference is
exactly zero) and clause 2.5's 0.3 dB overswing cap. They are properties of
the arithmetic, not evidence about the design. The rest of clauses 2.3, 2.6
and 2.7 has nothing to compute at all: overload capacity above full scale,
the law of a logarithmic stage, a calibrated scale range of at least 20 dB,
an input impedance of at least 20 kOhm and a 600 Ohm termination are
instrument hardware, and "80 % of full scale" constrains where a real needle
sits during the test rather than the quantity, which is why this module does
not model a full scale at all.

**There is no** [`report`](/phonometry/reference/api/broadcast/quasi-peak/#quasipeakresult) **fiche.** BS.468-4
prescribes no report format, no declaration form and no worked example; the
conformity evidence is the eleven windows, and its home is
`docs/CONFORMANCE.md` through [`verify_quasi_peak_dynamics`](/phonometry/reference/api/broadcast/quasi-peak/#verify_quasi_peak_dynamics).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## BS468_BALLISTICS

*Constant* (`phonometry.broadcast.quasi_peak.QuasiPeakBallistics`).

## DBQPS_REFERENCE

*Constant* (`float`).

```python
DBQPS_REFERENCE = 0.775
```

## quasi_peak_meter

```python
quasi_peak_meter(
    x: SignalInput,
    fs: float | None = None,
    *,
    weighted: bool = True,
    reference: float | None = None,
) -> QuasiPeakResult
```

Read a record with the ITU-R BS.468-4 quasi-peak meter.

Runs clause 2's detector over the whole record and returns the largest
excursion of the reading device, on the scale clause 2.6 fixes. The
reading rule is the maximum over the record, so the record must include
the decay of whatever it is measuring: a burst that ends at the last
sample is read before the needle has finished rising.

The chain and everything it chooses that the Recommendation does not are
described in the module docstring; the time constants are
[`BS468_BALLISTICS`](/phonometry/reference/api/broadcast/quasi-peak/#bs468_ballistics), a fit to Tables 2 and 3 rather than a quoted
value, and [`verify_quasi_peak_dynamics`](/phonometry/reference/api/broadcast/quasi-peak/#verify_quasi_peak_dynamics) is what checks them.

**Units.** The detector itself is unit-agnostic: rectification,
first-order recursions and a maximum carry whatever unit the record
arrived in, and the eleven acceptance windows are ratios in which every
unit cancels. Only the *level* needs a unit, and clause 2.6 gives
BS.468-4 exactly one, the volt. A bare array or an uncalibrated
[`Signal`](/phonometry/reference/api/io/io/#signal) is therefore read against 0.775 V and its
`level_db` is dBqps. A **calibrated** Signal is analysed in pascals,
as everywhere else in this library, and then has to be given a
*reference* of its own: 0.775 V is not a pressure, and BS.468-4 offers
no pressure to put in its place. `result.level_unit` names whichever
scale came out.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | The record: a 1-D array, or a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal). Multichannel input is refused rather than mixed, because a quasi-peak reading is one needle over one channel. |
| `fs` | Sample rate, Hz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `weighted` | Whether to run the record through the clause 1 weighting network first (default `True`, which is what clause 2's preamble requires of every test but the clause 2.4 reversibility check). |
| `reference` | The reference of `level_db`, in the record's own unit. `None` (the default) is the 0.775 V of clause 2.6, and is refused for a calibrated Signal, whose samples are pascals. |

**Returns:** A [`QuasiPeakResult`](/phonometry/reference/api/broadcast/quasi-peak/#quasipeakresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the record is empty or not 1-D, if *fs* is not positive, if *reference* is not positive, or if a calibrated Signal arrives without one. |

## QuasiPeakBallistics

```python
QuasiPeakBallistics(charge: float, discharge: float, reading_device: float)
```

Time constants of the quasi-peak chain, in seconds.

**Attributes**

| Name | Description |
| :--- | :--- |
| `charge` | Rise time constant of the peak rectifier. |
| `discharge` | Fall time constant of the peak rectifier. |
| `reading_device` | Time constant of the symmetric first-order reading device that follows it (clause 2.5's "reading device"). |

## QuasiPeakResult

```python
QuasiPeakResult(
    reading: float,
    level_db: float,
    reference: float,
    trace: np.ndarray,
    fs: float,
    weighted: bool,
)
```

A quasi-peak reading of one record (ITU-R BS.468-4 clause 2).

**Attributes**

| Name | Description |
| :--- | :--- |
| `reading` | The quasi-peak reading: the largest value `trace` reaches, on the scale clause 2.6 fixes, in the record's own unit. |
| `level_db` | `20 lg(reading / reference)`. This is **dBqps** (clause 3) when `reference` is the 0.775 V clause 2.6 names, and a level against whatever else the caller referred it to otherwise; `level_unit` is the one place that says which. |
| `reference` | The reference `level_db` is taken against, in the record's own unit. |
| `trace` | The reading device's output over the record, sample for sample with it, in the record's own unit. This is the needle, and the shape of its rise and decay is the whole content of clause 2. |
| `fs` | Sample rate of the record, Hz. |
| `weighted` | Whether the ITU-R BS.468-4 weighting network was in the path. Clause 2's preamble runs every test but the clause 2.4 reversibility check through it. |

### QuasiPeakResult.level_unit

*property*

The name of the scale `level_db` is on.

The one place that answers whether the level may be called
**dBqps**, which clause 3 defines only against the 0.775 V of
clause 2.6. Anything else is named by its own reference instead,
because BS.468-4 gives no other scale a name -- a quasi-peak sound
pressure level read against 20 uPa is a perfectly good number and is
not dBqps.

The name is read off the *reference*, which is the only unit-bearing
thing the result holds: the samples arrive in whatever unit they were
already in and the library never learns which. A caller who refers a
voltage record to 20 uV rather than to 20 uPa gets the pascal name;
nothing here can tell those two apart, and no other reference is
close enough to either to collide.

### QuasiPeakResult.plot()

```python
QuasiPeakResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the quasi-peak trace over time, with the reading marked.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the trace `plot` call. |

**Returns:** The axes.

### QuasiPeakResult.time

*property*

Time of each sample of `trace`, in seconds.

## verify_quasi_peak_dynamics

```python
verify_quasi_peak_dynamics(
    fs: float = 48000.0,
    ballistics: QuasiPeakBallistics = ...,
) -> dict[str, Any]
```

Check the detector against the eleven acceptance windows of clause 2.

Runs the clause 2.1 and 2.2 stimuli exactly as they are specified: a
5 kHz sine starting at a zero crossing and lasting an integral number of
full periods (at 5 kHz one period is 0.2 ms, so the eight tabulated
durations are 5, 10, 25, 50, 100, 250, 500 and 1000 cycles), through the
weighting network, and reads each one against the same chain's steady
reading for the same tone. Table 3 repeats the 5 ms burst at 2, 10 and
100 per second.

This is the whole conformance statement BS.468-4 authorises for a
detector, and it is an acceptance region rather than a set of values:
every reading inside a window conforms, and the Recommendation tolerates
about 2.5 dB of disagreement between two conforming instruments on a
single burst. `deviation_db` is reported against the printed reference
reading as well, but that column is a *self-imposed* regression bound,
not conformance: the reference is printed to two significant figures on
nine of the eleven cells, so its own quantum is 0.027 to 0.108 dB.

The stimulus amplitude does not appear anywhere: both tables are ratios
to the steady reading at the same amplitude, and a chain built from
rectification, first-order recursions and a maximum is exactly
homogeneous, so clause 2.1's two attenuator variants give the same
answer by construction.

**Parameters**

| Name | Description |
| :--- | :--- |
| `fs` | Sample rate to run the stimuli at, in Hz. At 44.1 kHz the 25-cycle burst is not sample-exact (it spans 220.5 samples) and [`tone_burst`](/phonometry/reference/api/signals/test-signals/#tone_burst) warns; the consequence measures 0.006 dB against a 2.626 dB window. |
| `ballistics` | The chain to check, defaulting to the fitted [`BS468_BALLISTICS`](/phonometry/reference/api/broadcast/quasi-peak/#bs468_ballistics). Passing another set is how the published statement about how far each constant can move is reproduced, and the reason the chain takes them as an argument at all. |

**Returns:** Dict with `fs`, `passed` (every reading inside its window), `worst_margin_db` (the smallest margin over the eleven, negative when one is outside), `worst_deviation_db` (the largest departure from a printed reference reading) and `stimuli`: eleven rows of `{"stimulus", "table", "reading_percent", "lower_percent", "reference_percent", "upper_percent", "deviation_db", "margin_db"}`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If *fs* is not positive. |
