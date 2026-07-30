#  Copyright (c) 2026. Jose M. Requena-Plens
"""
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
that already exists in :mod:`phonometry.metrology`: band-pass the structural
resonance the defect impacts ring, detect its envelope and transform it
(:func:`~phonometry.metrology.envelope.envelope_spectrum`), average
synchronously with the shaft
(:func:`~phonometry.metrology.synchronous_average.time_synchronous_average`)
or collapse the harmonic families in the cepstrum
(:func:`~phonometry.metrology.cepstrum.cepstrum`). The result object's
:meth:`FaultFrequencyResult.plot` draws the predicted lines **on top of a
measured envelope spectrum**, which is the working view.

**Rolling-contact bearings** (Eqs. 8.4 to 8.14, after Shahan & Kamperman).
With shaft speed ``N`` in r/min, ``Z`` rolling elements of diameter ``d`` on a
pitch diameter ``D`` and a contact angle ``phi``, writing ``g = (d/D) cos phi``
and ``fs = N/60``::

    FTF     = (fs/2)(1 - g)             cage, stationary outer race  (8.5)
    FTF_rel = fs - FTF                  cage seen from the shaft     (8.11)
    BSF     = (fs/2)(D/d)(1 - g**2)     element rotation             (8.7)
    BDF     = 2 BSF                     element spin (both races)    (8.10)
    BPFO    = Z (fs/2)(1 - g)           element pass, outer race     (8.8)
    BPFI    = Z (fs/2)(1 + g)           element pass, inner race     (8.9)

so ``BPFO + BPFI = Z fs`` exactly, and ``BPFO = Z FTF``. Norton notes that
Eqs. (8.8) and (8.14), and (8.9) and (8.13), are identical: ``BPFO`` and
``BPFI`` do not depend on which race turns. Only the cage does, and with the
outer race rotating it becomes ``(fs/2)(1 + g)`` (Eq. 8.6).

**Gears** (Eq. 8.3). The gear-meshing (tooth-passing) frequency of a wheel with
``n_teeth`` teeth is ``GMF = n_teeth fs``, with integer harmonics. A discrete
tooth fault adds shaft-rate lines and low, flat sidebands around every mesh
harmonic; distributed wear raises tall sideband groups at ``k GMF +/- m fs``.

**Induction motors** (Eqs. 8.19, 8.20). The three lines always present in a
motor bearing signal are ``fs`` (mechanical unbalance), ``2 fs``
(misalignment with the driven load) and ``2 fe`` (non-uniform air gap, torque
pulses and the electrical faults). Norton's Eq. (8.19) writes the supply
frequency as ``fe = fs p / 2`` for ``p`` magnetic poles, which is the
synchronous, zero-slip form; this module uses the slip-consistent
``fe = fs p / (2 (1 - s))``, identical at ``s = 0`` and the only version that
makes the rotor-slot harmonics of Eq. (8.20),

``fsh = fe [(2 R / p)(1 - s) +/- 2 (n - 1)]``,

collapse to the physical rotor-bar passing rate ``R fs`` at ``n = 1`` for
``R`` rotor bars. Dynamic eccentricity dresses that line with sidebands at
+/- the shaft rate and +/- the slip frequency.

**Fans, blowers and pumps** (Eqs. 8.15 to 8.18). The blade-passing frequency
of an impeller with ``N`` blades is ``fb = n N fs``; a pump's hydraulic
pulsations follow the same form with ``N`` pumping events per revolution
(Eq. 8.18), and a rotary positive-displacement blower repeats four times per
revolution. In a ducted axial fan the blade-vane interaction sets up rotating
pressure patterns with ``mL = n N +/- k V`` lobes for ``V`` vanes (Eq. 8.16),
turning at ``n N fs / mL`` (Eq. 8.17): a pattern that spins faster than the
blades themselves, and the reason a careful choice of ``N`` and ``V`` matters
for radiated power.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.validation import (
    require_choice,
    require_non_negative,
    require_positive,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: Description of the shaft-rate line, shared by every family.
_SHAFT_DESCRIPTION = "shaft rotational frequency"


def _require_count(value: int, name: str, minimum: int = 1) -> int:
    """An integer parameter that must be at least *minimum*."""
    n = int(value)
    if n < minimum:
        raise ValueError(
            f"'{name}' must be a non-negative integer."
            if minimum == 0
            else f"'{name}' must be a positive integer."
        )
    return n


__all__ = [
    "FaultFrequencyResult",
    "FaultLine",
    "bearing_fault_frequencies",
    "blade_pass_frequencies",
    "combine_fault_lines",
    "gear_mesh_frequencies",
    "induction_motor_frequencies",
    "shaft_rate",
]


def shaft_rate(speed_rpm: float) -> float:
    """Shaft rotational frequency ``fs = N/60`` (Norton Eq. 8.4).

    :param speed_rpm: Shaft speed ``N``, in r/min (> 0).
    :return: The shaft rotational frequency ``fs``, in hertz.
    :raises ValueError: for a non-positive speed.
    """
    return require_positive(speed_rpm, "speed_rpm") / 60.0


@dataclass(frozen=True)
class FaultLine:
    """One predicted discrete line of a machine's kinematic signature.

    :ivar name: Short label, unique within a result (``"BPFO"``, ``"2xGMF"``,
        ``"GMF-1x"``, ...). Acronyms are language neutral and are what the
        :meth:`FaultFrequencyResult.plot` overlay annotates.
    :ivar frequency: Predicted frequency, in hertz.
    :ivar order: Frequency expressed in shaft orders, ``frequency / fs``.
    :ivar family: One of ``"shaft"``, ``"bearing"``, ``"gear"``, ``"motor"``
        or ``"blade"``.
    :ivar description: One-line English description of the mechanism.
    """

    name: str
    frequency: float
    order: float
    family: str
    description: str


def _line(
    name: str, frequency: float, fs: float, family: str, description: str
) -> FaultLine:
    """Build a :class:`FaultLine`, deriving its shaft order from ``fs``."""
    return FaultLine(
        name=name,
        frequency=float(frequency),
        order=float(frequency) / fs,
        family=family,
        description=description,
    )


@dataclass(frozen=True)
class FaultFrequencyResult:
    """A family of predicted fault lines for one machine element.

    :ivar lines: The predicted :class:`FaultLine` entries, in the order the
        generating function produced them.
    :ivar shaft_rate: Shaft rotational frequency ``fs``, in hertz.
    :ivar source: Description of the element (``"rolling-contact bearing"``,
        ``"gear pair"``, ...), used as the plot title.
    """

    lines: tuple[FaultLine, ...]
    shaft_rate: float
    source: str

    def __getitem__(self, name: str) -> float:
        """Frequency of the line called *name*, in hertz.

        :raises KeyError: If no line carries that name.
        """
        for line in self.lines:
            if line.name == name:
                return line.frequency
        raise KeyError(
            f"no fault line named {name!r}; available: "
            f"{', '.join(line.name for line in self.lines)}."
        )

    def __contains__(self, name: object) -> bool:
        """Whether a line called *name* is present."""
        return any(line.name == name for line in self.lines)

    @property
    def names(self) -> tuple[str, ...]:
        """Names of the predicted lines, in order."""
        return tuple(line.name for line in self.lines)

    @property
    def frequencies(self) -> np.ndarray:
        """Predicted frequencies, in hertz, in order."""
        return np.array([line.frequency for line in self.lines], dtype=np.float64)

    @property
    def orders(self) -> np.ndarray:
        """Predicted frequencies in shaft orders, in order."""
        return np.array([line.order for line in self.lines], dtype=np.float64)

    def as_dict(self) -> dict[str, float]:
        """The lines as a ``{name: frequency}`` mapping."""
        return {line.name: line.frequency for line in self.lines}

    def harmonics(self, name: str, count: int) -> np.ndarray:
        """The first *count* integer harmonics of the line called *name*.

        :param name: Line name (see :attr:`names`).
        :param count: Number of harmonics, ``>= 1`` (the fundamental first).
        :return: Frequencies ``n * f`` for ``n = 1 .. count``, in hertz.
        :raises KeyError: If no line carries that name.
        :raises ValueError: If *count* is not a positive integer.
        """
        n = _require_count(count, "count")
        return self[name] * np.arange(1, n + 1, dtype=np.float64)

    def within(self, low: float, high: float) -> FaultFrequencyResult:
        """The lines falling in ``[low, high]`` hertz, as a new result.

        Handy before plotting on an envelope spectrum whose useful span is
        much narrower than the highest predicted harmonic.

        :param low: Lower edge, in hertz (>= 0).
        :param high: Upper edge, in hertz (> *low*).
        :return: A :class:`FaultFrequencyResult` with the surviving lines.
        :raises ValueError: If the edges are invalid.
        """
        lo = require_non_negative(low, "low")
        hi = require_positive(high, "high")
        if hi <= lo:
            raise ValueError("'high' must be greater than 'low'.")
        kept = tuple(ln for ln in self.lines if lo <= ln.frequency <= hi)
        return FaultFrequencyResult(
            lines=kept, shaft_rate=self.shaft_rate, source=self.source
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Overlay the predicted lines on a measured envelope spectrum.

        Pass the measurement as ``spectrum=`` (an
        :class:`~phonometry.metrology.envelope.EnvelopeSpectrumResult`, or any
        object exposing ``frequencies`` and ``amplitude``); without it the
        predicted lines are drawn alone as a labelled stem plot.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: ``spectrum``, ``max_frequency`` and anything forwarded
            to the spectrum curve; see
            :func:`phonometry._plot.vibration.plot_fault_frequencies`.
        """
        from .._i18n import check_language
        from .._plot.vibration import plot_fault_frequencies

        check_language(language)
        return plot_fault_frequencies(self, ax=ax, language=language, **kwargs)


def combine_fault_lines(
    *results: FaultFrequencyResult, source: str | None = None
) -> FaultFrequencyResult:
    """Merge several fault-line families into one overlay.

    A gearbox bearing carries its own bearing lines, the mesh family of the
    gear it supports and the shaft harmonics; this puts them on one axes.
    Duplicate names are disambiguated by appending the source of the family
    they came from.

    :param results: Two or more :class:`FaultFrequencyResult` objects.
    :param source: Label for the merged family (Default: the sources joined
        by ``" + "``).
    :return: A :class:`FaultFrequencyResult` holding every line.
    :raises ValueError: If no result is given, or the shaft rates disagree.
    """
    if not results:
        raise ValueError("'combine_fault_lines' needs at least one result.")
    rates = {round(res.shaft_rate, 9) for res in results}
    if len(rates) > 1:
        raise ValueError(
            "all results must share the same shaft rate; got "
            f"{sorted(res.shaft_rate for res in results)} Hz."
        )
    seen: set[str] = set()
    merged: list[FaultLine] = []
    for res in results:
        for line in res.lines:
            name = line.name
            if name in seen:
                name = f"{line.name} ({res.source})"
            seen.add(name)
            merged.append(replace(line, name=name))
    label = source if source is not None else " + ".join(r.source for r in results)
    return FaultFrequencyResult(
        lines=tuple(merged), shaft_rate=results[0].shaft_rate, source=label
    )


def bearing_fault_frequencies(
    speed_rpm: float,
    n_elements: int,
    element_diameter: float,
    pitch_diameter: float,
    *,
    contact_angle_deg: float = 0.0,
    rotating_race: str = "inner",
) -> FaultFrequencyResult:
    """Kinematic frequencies of a rolling-contact bearing (Norton 8.4-8.14).

    Returns the seven lines a bearing generates, named with the acronyms used
    in condition monitoring:

    ============ ================================================= ==========
    Name         Meaning                                           Norton eq.
    ============ ================================================= ==========
    ``shaft``    shaft rotational frequency ``fs``                  (8.4)
    ``FTF``      cage (fundamental train) frequency                 (8.5)/(8.6)
    ``FTF_rel``  cage rotation relative to the rotating race        (8.11)/(8.12)
    ``BSF``      rolling-element rotational frequency               (8.7)
    ``BDF``      rolling-element spin frequency, ``2 BSF``          (8.10)
    ``BPFO``     element pass frequency on the outer race           (8.8)/(8.14)
    ``BPFI``     element pass frequency on the inner race           (8.9)/(8.13)
    ============ ================================================= ==========

    ``BPFO`` and ``BPFI`` are the outer- and inner-race defect lines and
    ``BDF`` the rolling-element/cage defect line; all three are exact
    kinematics of a pure-rolling contact, so a real bearing's lines wander by
    1 % to 2 % with load-dependent slip. ``BPFO + BPFI = Z fs`` always, and
    both are independent of which race turns (Norton's Eqs. 8.8 and 8.14, and
    8.9 and 8.13, are identical); *rotating_race* only moves the cage.

    :param speed_rpm: Shaft speed ``N``, in r/min (> 0).
    :param n_elements: Number of rolling elements ``Z`` (integer >= 1).
    :param element_diameter: Rolling-element diameter ``d``, in the same unit
        as *pitch_diameter* (> 0); only the ratio ``d/D`` enters.
    :param pitch_diameter: Bearing pitch diameter ``D`` (> ``d``).
    :param contact_angle_deg: Contact angle ``phi`` between element and
        raceway, in degrees (Default: 0, a radial ball bearing); ``0 <= phi
        < 90``.
    :param rotating_race: Which race turns with the shaft, ``"inner"``
        (Default) or ``"outer"``.
    :return: A :class:`FaultFrequencyResult` (source
        ``"rolling-contact bearing"``).
    :raises ValueError: for a non-positive or inconsistent geometry.
    """
    fs = shaft_rate(speed_rpm)
    z = _require_count(n_elements, "n_elements")
    d = require_positive(element_diameter, "element_diameter")
    dm = require_positive(pitch_diameter, "pitch_diameter")
    if d >= dm:
        raise ValueError("'element_diameter' must be smaller than 'pitch_diameter'.")
    phi = require_non_negative(contact_angle_deg, "contact_angle_deg")
    if phi >= 90.0:
        raise ValueError("'contact_angle_deg' must be below 90 degrees.")
    race = require_choice(rotating_race, "rotating_race", ("inner", "outer"))

    # g = (d/D) cos(phi). The cage rate carries (1 - g) with a stationary
    # outer race (Eq. 8.5) and (1 + g) with a stationary inner one (Eq. 8.6);
    # the element pass frequencies do not swap, because Eqs. (8.8)/(8.14) and
    # (8.9)/(8.13) are pairwise identical.
    g = (d / dm) * math.cos(math.radians(phi))
    sign = -1.0 if race == "inner" else 1.0
    cage = 0.5 * fs * (1.0 + sign * g)  # (8.5) / (8.6)
    cage_rel = fs - cage  # (8.11) / (8.12)
    spin = 0.5 * fs * (dm / d) * (1.0 - g**2)  # (8.7)
    bpfo = 0.5 * z * fs * (1.0 - g)  # (8.8) == (8.14)
    bpfi = 0.5 * z * fs * (1.0 + g)  # (8.9) == (8.13)

    lines = (
        _line("shaft", fs, fs, "shaft", _SHAFT_DESCRIPTION),
        _line("FTF", cage, fs, "bearing", "cage (fundamental train) frequency"),
        _line(
            "FTF_rel",
            cage_rel,
            fs,
            "bearing",
            "cage rotation relative to the rotating race",
        ),
        _line("BSF", spin, fs, "bearing", "rolling-element rotational frequency"),
        _line(
            "BDF",
            2.0 * spin,
            fs,
            "bearing",
            "rolling-element spin frequency (contact with both races)",
        ),
        _line("BPFO", bpfo, fs, "bearing", "element pass frequency, outer race"),
        _line("BPFI", bpfi, fs, "bearing", "element pass frequency, inner race"),
    )
    return FaultFrequencyResult(
        lines=lines, shaft_rate=fs, source="rolling-contact bearing"
    )


def _sideband_pairs(
    centre: float, spacing: float, orders: int
) -> list[tuple[int, str, float]]:
    """``(order, sign symbol, frequency)`` of a symmetric sideband family.

    Only sidebands that land above 0 Hz are returned.
    """
    out: list[tuple[int, str, float]] = []
    for m in range(1, orders + 1):
        for sign, symbol in ((-1.0, "-"), (1.0, "+")):
            freq = centre + sign * m * spacing
            if freq > 0.0:
                out.append((m, symbol, freq))
    return out


def _gear_sidebands(
    stem: str, centre: float, fs: float, spacing: float, orders: int
) -> list[FaultLine]:
    """The modulation sidebands around one gear-mesh harmonic."""
    return [
        _line(
            f"{stem}{symbol}{m}x",
            freq,
            fs,
            "gear",
            f"modulation sideband at {symbol}{m} x the modulating rate",
        )
        for m, symbol, freq in _sideband_pairs(centre, spacing, orders)
    ]


def gear_mesh_frequencies(
    speed_rpm: float,
    n_teeth: int,
    *,
    harmonics: int = 3,
    sidebands: int = 0,
    sideband_rate: float | None = None,
) -> FaultFrequencyResult:
    """Gear-meshing frequency and its sideband family (Norton Eq. 8.3).

    ``GMF = n_teeth x N/60``, with integer harmonics ``k GMF`` (``k = 1 ..
    harmonics``) named ``"GMF"``, ``"2xGMF"``, ... Each harmonic can carry a
    modulation family at ``k GMF +/- m f_mod`` (``m = 1 .. sidebands``), named
    ``"GMF-1x"``, ``"2xGMF+2x"``, and so on. The default modulation rate is
    the shaft rate of the wheel: a chipped tooth or an eccentric wheel
    modulates the mesh once per revolution, which is what produces those
    sidebands (Norton Figs. 8.23 and 8.24). Pass *sideband_rate* to modulate
    at the mating wheel's shaft rate instead.

    Only positive sideband frequencies are returned.

    :param speed_rpm: Shaft speed ``N`` of the wheel, in r/min (> 0).
    :param n_teeth: Number of teeth on that wheel (integer >= 1).
    :param harmonics: Number of mesh harmonics (integer >= 1, Default: 3).
    :param sidebands: Sideband order per harmonic (integer >= 0, Default: 0,
        no sidebands).
    :param sideband_rate: Modulation rate ``f_mod``, in hertz (> 0, Default:
        the wheel's own shaft rate).
    :return: A :class:`FaultFrequencyResult` (source ``"gear pair"``).
    :raises ValueError: for a non-positive or non-integer input.
    """
    fs = shaft_rate(speed_rpm)
    teeth = _require_count(n_teeth, "n_teeth")
    n_harm = _require_count(harmonics, "harmonics")
    n_side = _require_count(sidebands, "sidebands", minimum=0)
    f_mod = (
        fs
        if sideband_rate is None
        else require_positive(sideband_rate, "sideband_rate")
    )

    gmf = teeth * fs
    lines: list[FaultLine] = [_line("shaft", fs, fs, "shaft", _SHAFT_DESCRIPTION)]
    for k in range(1, n_harm + 1):
        stem = "GMF" if k == 1 else f"{k}xGMF"
        lines.append(
            _line(
                stem,
                k * gmf,
                fs,
                "gear",
                "gear-meshing (tooth-passing) frequency"
                if k == 1
                else f"harmonic {k} of the gear-meshing frequency",
            )
        )
        lines.extend(_gear_sidebands(stem, k * gmf, fs, f_mod, n_side))
    return FaultFrequencyResult(lines=tuple(lines), shaft_rate=fs, source="gear pair")


def _slot_harmonics(
    f_slot: float, fs: float, fe: float, orders: int
) -> list[FaultLine]:
    """Slot harmonics of order 2 and above, ``f_slot +/- 2(n - 1) fe``."""
    out: list[FaultLine] = []
    for n in range(2, orders + 1):
        for _m, symbol, freq in _sideband_pairs(f_slot, 2.0 * (n - 1) * fe, 1):
            out.append(
                _line(
                    f"fsh{symbol}{n - 1}",
                    freq,
                    fs,
                    "motor",
                    f"slot harmonic order {n}, {symbol}2({n} - 1) fe",
                )
            )
    return out


def _slot_sidebands(
    f_slot: float, fs: float, spacing: float, tag: str, orders: int
) -> list[FaultLine]:
    """Eccentricity sidebands around the dominant slot harmonic."""
    if spacing <= 0.0:
        return []
    return [
        _line(
            f"fsh{symbol}{m}{tag}",
            freq,
            fs,
            "motor",
            "eccentricity sideband around the slot harmonic",
        )
        for m, symbol, freq in _sideband_pairs(f_slot, spacing, orders)
    ]


def induction_motor_frequencies(
    speed_rpm: float,
    poles: int,
    rotor_bars: int,
    *,
    slip: float = 0.0,
    supply_frequency: float | None = None,
    slot_harmonics: int = 1,
    sidebands: int = 0,
) -> FaultFrequencyResult:
    """Electrical and slot lines of an induction motor (Norton 8.19, 8.20).

    The three lines always present in a motor bearing vibration signal are
    ``1x`` (mechanical unbalance), ``2x`` (misalignment with the driven load)
    and ``2fe`` (a non-uniform air gap, torque pulses and the winding/rotor-bar
    electrical faults). Rotor defects that produce static or dynamic air-gap
    eccentricity are read on the **slot harmonics** of the stator core,

    ``fsh = fe [(2 R / p)(1 - s) +/- 2 (n - 1)] = R fs +/- 2 (n - 1) fe``

    for ``R`` rotor bars, ``p`` magnetic poles (not pole pairs), unit slip
    ``s`` and ``n = 1, 2, ...``. Dynamic eccentricity modulates the dominant
    slot harmonic at +/- the shaft rate and +/- the slip frequency, which is
    what *sidebands* adds around ``fsh``.

    Give the slip directly, or give the mains *supply_frequency* and let it be
    derived from ``fe`` and the measured shaft speed. The supply frequency is
    taken as ``fe = fs p / (2 (1 - s))``: Norton's Eq. (8.19) writes
    ``fe = fs p / 2``, which is the same expression at zero slip but does not
    reduce Eq. (8.20) to the physical rotor-bar passing rate ``R fs`` when the
    machine is loaded.

    With a non-zero slip the pole-pass line ``FP = p f_slip`` is included as
    well. That name is standard condition-monitoring practice rather than
    Norton's: he gives the slip frequency itself as the sideband spacing of a
    broken rotor bar (his Section 8.4.8) and does not multiply it by the pole
    count.

    :param speed_rpm: Shaft speed ``N``, in r/min (> 0).
    :param poles: Number of magnetic poles ``p`` (even integer >= 2).
    :param rotor_bars: Number of rotor bars/slots ``R`` (integer >= 1).
    :param slip: Unit slip ``s`` (``0 <= s < 1``, Default: 0); typically 0,02
        to 0,05 under load. Ignored when *supply_frequency* is given.
    :param supply_frequency: Mains frequency ``fe``, in hertz (> 0), from
        which the slip is derived (Default: ``None``, use *slip*).
    :param slot_harmonics: Number of slot-harmonic orders ``n`` (integer >= 1,
        Default: 1, the fundamental ``R fs`` alone).
    :param sidebands: Sideband order around the fundamental slot harmonic at
        +/- the shaft rate and +/- the slip frequency (integer >= 0,
        Default: 0).
    :return: A :class:`FaultFrequencyResult` (source ``"induction motor"``).
    :raises ValueError: for a non-positive, non-integer or inconsistent input.
    """
    fs = shaft_rate(speed_rpm)
    p = int(poles)
    if p < 2 or p % 2:
        raise ValueError("'poles' must be an even integer of at least 2.")
    bars = _require_count(rotor_bars, "rotor_bars")
    n_slot = _require_count(slot_harmonics, "slot_harmonics")
    n_side = _require_count(sidebands, "sidebands", minimum=0)

    if supply_frequency is not None:
        fe = require_positive(supply_frequency, "supply_frequency")
        synchronous = 2.0 * fe / p
        if fs >= synchronous:
            raise ValueError(
                "the shaft rate must stay below the synchronous speed "
                f"{synchronous:g} Hz implied by 'supply_frequency' and "
                "'poles'; check the pole count."
            )
        s = 1.0 - fs / synchronous
    else:
        s = require_non_negative(slip, "slip")
        if s >= 1.0:
            raise ValueError("'slip' must be below 1.")
        # Eq. (8.19), fe = fs p / 2, made slip-consistent: the field turns at
        # fs/(1 - s), so fe = fs p / (2 (1 - s)). Identical at s = 0.
        fe = fs * p / (2.0 * (1.0 - s))
        synchronous = 2.0 * fe / p
    f_slip = synchronous - fs
    # Eq. (8.20) at n = 1: fe (2 R / p)(1 - s), which is exactly R fs.
    f_slot = fe * (2.0 * bars / p) * (1.0 - s)

    lines: list[FaultLine] = [
        _line("1x", fs, fs, "shaft", f"{_SHAFT_DESCRIPTION} (unbalance)"),
        _line("2x", 2.0 * fs, fs, "shaft", "twice shaft rate (misalignment)"),
        _line("fe", fe, fs, "motor", "electrical supply frequency"),
        _line(
            "2fe",
            2.0 * fe,
            fs,
            "motor",
            "twice supply frequency (non-uniform air gap, electrical faults)",
        ),
    ]
    if f_slip > 0.0:
        lines.append(
            _line("f_slip", f_slip, fs, "motor", "slip frequency"),
        )
        lines.append(
            _line(
                "FP",
                p * f_slip,
                fs,
                "motor",
                "pole-pass frequency (poles x slip frequency)",
            )
        )
    lines.append(
        _line(
            "fsh", f_slot, fs, "motor", "fundamental rotor-slot harmonic frequency"
        )
    )
    lines.extend(_slot_harmonics(f_slot, fs, fe, n_slot))
    for rate, tag in ((fs, "x"), (f_slip, "s")):
        lines.extend(_slot_sidebands(f_slot, fs, rate, tag, n_side))
    return FaultFrequencyResult(
        lines=tuple(lines), shaft_rate=fs, source="induction motor"
    )


def _lobe_patterns(
    blades: int, vanes: int, fs: float, harmonics: int, k_max: int
) -> list[FaultLine]:
    """Rotating blade-vane interaction patterns (Norton Eqs. 8.16 and 8.17).

    ``mL = n N +/- k V`` lobes turning at ``n N fs / mL``. The pattern speed of
    Eq. (8.17) carries ``n`` in its numerator, so the same lobe count reached
    from two different blade harmonics rotates at two different speeds and both
    are real lines: the de-duplication is on the ``(n, mL)`` pair, not on the
    lobe count alone. A zero lobe count (no pattern) is skipped.
    """
    out: list[FaultLine] = []
    seen: set[tuple[int, int]] = set()
    for n in range(1, harmonics + 1):
        for k in range(1, k_max + 1):
            for sign in (-1, 1):
                lobes = n * blades + sign * k * vanes
                count = abs(lobes)
                if lobes == 0 or (n, count) in seen:
                    continue
                seen.add((n, count))
                out.append(
                    _line(
                        f"lobe n={n} m={count}",
                        abs(n * blades * fs / lobes),
                        fs,
                        "blade",
                        f"rotating interaction pattern with {count} pressure "
                        f"lobes, from blade harmonic {n}",
                    )
                )
    return out


def blade_pass_frequencies(
    speed_rpm: float,
    n_blades: int,
    *,
    harmonics: int = 3,
    n_vanes: int | None = None,
    lobe_orders: int = 1,
) -> FaultFrequencyResult:
    """Blade-passing frequency and interaction patterns (Norton 8.15-8.17).

    ``fb = n N_blades x N/60`` for ``n = 1 .. harmonics``, the discrete tone
    family of any fan, blower or pump impeller; a rotary positive-displacement
    blower repeats four times per revolution, so pass ``4 x`` its speed.

    In a ducted axial fan the blades also interact with the stator vanes and
    set up rotating pressure patterns with ``mL = n N_blades +/- k N_vanes``
    lobes (Eq. 8.16) turning at ``ML = n N_blades fs / mL`` (Eq. 8.17). Those
    patterns radiate strongly when they can drive a higher-order duct mode, so
    they are the lines to check against the duct cut-on frequencies. Give
    *n_vanes* to include them, named ``"lobe n=1 m=2"``, ``"lobe n=1 m=10"``,
    ... The blade harmonic is part of the name because Eq. (8.17) carries
    ``n``: the same lobe count reached from a different harmonic is a distinct
    pattern turning at a different speed.

    :param speed_rpm: Shaft speed ``N``, in r/min (> 0).
    :param n_blades: Number of blades (integer >= 1).
    :param harmonics: Number of blade-pass harmonics (integer >= 1,
        Default: 3).
    :param n_vanes: Number of stator vanes ``V`` (integer >= 1) to include the
        lobed interaction patterns (Default: ``None``, blade tones only).
    :param lobe_orders: Highest ``|k|`` in ``mL = n N_blades +/- k N_vanes``
        (integer >= 1, Default: 1).
    :return: A :class:`FaultFrequencyResult` (source ``"bladed rotor"``).
    :raises ValueError: for a non-positive or non-integer input.
    """
    fs = shaft_rate(speed_rpm)
    blades = _require_count(n_blades, "n_blades")
    n_harm = _require_count(harmonics, "harmonics")
    k_max = _require_count(lobe_orders, "lobe_orders")
    vanes = None if n_vanes is None else _require_count(n_vanes, "n_vanes")

    lines: list[FaultLine] = [
        _line("shaft", fs, fs, "shaft", _SHAFT_DESCRIPTION)
    ]
    for n in range(1, n_harm + 1):
        name = "BPF" if n == 1 else f"{n}xBPF"
        lines.append(
            _line(
                name,
                n * blades * fs,
                fs,
                "blade",
                "blade-passing frequency"
                if n == 1
                else f"harmonic {n} of the blade-passing frequency",
            )
        )
    if vanes is not None:
        lines.extend(_lobe_patterns(blades, vanes, fs, n_harm, k_max))
    return FaultFrequencyResult(
        lines=tuple(lines), shaft_rate=fs, source="bladed rotor"
    )
