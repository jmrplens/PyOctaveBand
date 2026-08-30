#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Low-frequency procedure of ISO 16283, shared by all three parts.

Below 100 Hz a dwelling-sized room has too few modes for the central
microphone positions of the default procedure to stand for the whole volume,
so ISO 16283 adds a second measurement in the room corners and combines the
two. The procedure is **not optional**: Part 1 Clause 8.1, Part 2 Clause 8.1
and Part 3 Clause 7.3.1 all say it *shall* be used for the 50 Hz, 63 Hz and
80 Hz one-third-octave bands once the room volume, calculated to the nearest
cubic metre, is smaller than 25 m³. Most bedrooms and bathrooms are under that
line, which is why this sits under the field-measurement entry points rather
than beside them.

**The corner level.** With the source running, the highest level of the set of
measured corners is taken, band by band, and the values for the three bands may
come from three different corners (the NOTE under Formula (12)). Where a single
loudspeaker or tapping machine is moved between q positions those q maxima are
energy-averaged, Part 1 Formula (12) and Part 2 Formula (15):

.. math::

    L_\mathrm{Corner} = 10 \lg \frac{p^2_\mathrm{Corner,1} + \cdots +
    p^2_\mathrm{Corner,q}}{q\,p_0^2}

Part 3 numbers no such formula: Clause 7.3.4 defines :math:`L_\mathrm{2,Corner}`
as the maximum over corners and averages the *level difference* over
loudspeaker positions later (Clause 9.6.3, Formula (8)). The maximum is the
q = 1 case of the formula above, so the same code answers all three.

**The combination.** The low-frequency energy-average level weighs the corner
level one third against two thirds of the default-procedure level. Part 1
Formula (13), Part 2 Formula (16) and Part 3 Formula (5) print it identically,
only the subscripts of the level symbols changing:

.. math::

    L_\mathrm{LF} = 10 \lg \left[ \frac{10^{0,1 L_\mathrm{Corner}} +
    (2 \cdot 10^{0,1 L})}{3} \right]

**The reverberation time.** Under the same 25 m³ trigger, Part 1 Clause 10.4,
Part 2 Clause 10.4 and Part 3 Clause 8.4 stop the 50 Hz, 63 Hz and 80 Hz
one-third-octave reverberation times being measured at all and put one 63 Hz
*octave* band value in their place, used for all three bands. It is a
prescription about what to measure, not a claim that the octave value equals
the three one-third-octave ones: in a small room there are too few modes for a
one-third-octave decay to be single-sloped, and in timber or steel frame
construction the decay can be shorter than the analyser's own one-third-octave
filter (NOTE 1 and NOTE 2 under each of those clauses). Below the trigger there
is no default value to fall back on either, because Clause 10.3 / 8.3 confines
the default reverberation-time procedure to 100 Hz and above once the room is
under 25 m³.

**Which room.** Part 1 applies the corner procedure to "the source and/or
receiving room when *its* volume" is under the line, so a 18 m³ source room
next to a 40 m³ receiving room gets the corner treatment on :math:`L_1` alone.
Parts 2 and 3 have only a receiving room to treat. The 63 Hz octave
substitution is keyed to the **receiving** room in all three parts, Part 1
included: its Clause 10 is headed "Reverberation time in the receiving room",
its Clause 10.1 scopes the whole clause to that room and its Clause 10.4 names
it again. So that asymmetry is real and this module encodes it: a source-room
procedure that carries a 63 Hz octave reverberation time is refused. Part 1
Clause 6 does contradict its own Clause 10 on this and asks for the
reverberation-time procedure "in the source and/or receiving room"; that is a
defect of the printed text, registered in ``docs/ERRATA.md``.

**Not optional, and not silent.** Clause 8.1 (Part 3: Clause 7.3.1) says the
procedure *shall* be used, so the three entry points that consume this module
do not simply wait to be asked. When the room volume and the band centres are
both in hand, a room that rounds below the trigger and names the three bands
without bringing a :class:`LowFrequencyProcedure` gets a
:class:`LowFrequencyWarning` rather than a quiet answer several decibels away
from the ISO 16283 one. They warn rather than refuse: the corner measurements
may genuinely not exist, and the default-procedure spectrum is what a reader
compares the ISO 16283 one against.

**Which methods.** Part 3 restricts the whole procedure to the element and
global *loudspeaker* methods; Clause 6 NOTE 1 records that there is no
experience of running it with traffic as the source, and the heading of
Clause 7.3 carries the restriction. Part 2 restricts it to the tapping machine
(the heading of its Clause 8), while the 63 Hz octave reverberation time of its
Clause 10.4 also feeds the rubber-ball quantity :math:`L'_\mathrm{i,Fmax,V,T}`.

**No numeric oracle.** Neither part publishes a worked example of this
procedure: Annexes B and C of Parts 1 and 2 are blank recording forms and the
"Examples" of Annexes D and E are loudspeaker-position drawings. The
conformance of this module therefore rests on closed forms and on printed
numbers rather than on a tabulated result; see
:mod:`scripts.conformance.domains.building` and ``docs/CONFORMANCE.md`` for the
checks that stand in for one.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from ..._internal.levels_math import energy_mean
from ..._internal.warnings import PhonometryWarning

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

__all__ = [
    "LOW_FREQUENCY_BANDS",
    "LOW_FREQUENCY_VOLUME_LIMIT",
    "LowFrequencyProcedure",
    "LowFrequencyResult",
    "LowFrequencyWarning",
    "apply_low_frequency_procedure",
    "corner_level",
    "low_frequency_level",
    "low_frequency_procedure_applies",
]


class LowFrequencyWarning(PhonometryWarning):
    """An ISO 16283 low-frequency requirement the measurement does not meet.

    Two conditions raise it, and the message says which.

    The corner count of Part 1 Clause 8.3, Part 2 Clause 8.3 and Part 3
    Clause 7.3.2, which the arithmetic of Formula (12) does not depend on: four
    corners and three give the same maximum-then-average, and it is the report
    that has to say the room was undersampled.

    A room under the 25 m³ trigger whose 50 Hz, 63 Hz and 80 Hz bands are about
    to be answered from the default procedure alone, which Clause 8.1 (Part 3:
    Clause 7.3.1) says *shall* not happen. That one does change the number, by
    several decibels, so it is worth filtering the two apart by message rather
    than silencing the class.
    """


#: The three one-third-octave bands the low-frequency procedure covers, in Hz.
#: Part 1 Clause 8.1, Part 2 Clause 8.1 and Part 3 Clause 7.3.1 name the same
#: three and no others; 100 Hz upwards always keeps the default procedure.
LOW_FREQUENCY_BANDS: Final[tuple[float, float, float]] = (50.0, 63.0, 80.0)

#: Room volume, in m³, at and above which the low-frequency procedure is not
#: used. Every clause prints the trigger as "smaller than 25 m³ (calculated to
#: the nearest cubic metre)", so the comparison is strict and 25 m³ exactly
#: does not trigger.
LOW_FREQUENCY_VOLUME_LIMIT: Final = 25.0

#: Corners the standards require: "a minimum of four corners" in Part 1
#: Clause 8.3, Part 2 Clause 8.3 and Part 3 Clause 7.3.2 alike. Parts 1 and 2
#: ask for them again at each source position ("for each loudspeaker position",
#: "for each impact source position"); Part 3 prints the sentence without that
#: qualifier, and has no second position to repeat them at.
_MIN_CORNERS: Final = 4

#: How far a supplied band centre may sit from the nominal 50 / 63 / 80 Hz and
#: still be that band. The exact one-third-octave centres are 49,6 Hz, 62,5 Hz
#: and 79,4 Hz, all within 0,9 % of their nominal names, so 3 % separates each
#: band from its neighbours (the next centres out are 40 Hz and 100 Hz) while
#: accepting either spelling. Same tolerance the ISO 717 rating range uses to
#: find its bands in :mod:`~phonometry.building.prediction.detailed_model`.
_BAND_MATCH_RTOL: Final = 0.03

#: Positions axis of a three-dimensional corner-level array, and the corners
#: axis under it: ``(positions, corners, bands)``.
_POSITION_AXIS: Final = 0
_CORNER_AXIS: Final = 1

#: Rank of a ``(corners, bands)`` corner-level array, which is the single
#: source position of Part 3 and of simultaneously operated loudspeakers.
_SINGLE_POSITION_RANK: Final = 2

#: The two rooms the procedure can be applied to. Only ISO 16283-1 has both;
#: what separates them here is Clause 10.4, which is receiving-room only.
_SOURCE_ROOM: Final = "source"
_ROOMS: Final = ("receiving", _SOURCE_ROOM)


def low_frequency_procedure_applies(volume: float) -> bool:
    """Whether a room of this volume triggers the low-frequency procedure.

    The printed condition is the same in all three parts: the volume,
    "calculated to the nearest cubic metre", is "smaller than 25 m³"
    (Part 1 Clause 8.1 and 10.4, Part 2 Clause 8.1 and 10.4, Part 3
    Clause 7.3.1 and 8.4). The comparison is strict, so a room that rounds to
    25 m³ exactly does not trigger, and neither does anything larger.

    The rounding is half away from zero, ``floor(V + 0,5)``, which is the rule
    the rest of this tree rounds printed quantities with; the standards give no
    tie rule of their own. It matters on the boundary: ``V = 24,5`` m³ rounds
    to 25 m³ here and does **not** trigger, where Python's built-in
    :func:`round`, which is half-to-even, would answer 24 and trigger.

    :param volume: Room volume ``V``, in m³.
    :return: ``True`` when the low-frequency procedure is required.
    :raises ValueError: If ``volume`` is not a positive, finite number.
    """
    value = float(volume)
    # Ordered, not merged: `math.isfinite` is what catches NaN and both
    # infinities, and it has to run first because `value <= 0.0` is False for
    # NaN and would let one through to `math.floor`, which raises on it. The
    # sign test is written `value <= 0.0` rather than `not value > 0.0`
    # precisely because NaN is already gone by the time it runs, so the two
    # spellings can no longer disagree and the reading one is kept.
    if not math.isfinite(value) or value <= 0.0:
        msg = f"'volume' must be a positive, finite room volume in m³; got {volume!r}."
        raise ValueError(msg)
    return math.floor(value + 0.5) < LOW_FREQUENCY_VOLUME_LIMIT


def _require_volume_triggers(volume: float, name: str) -> None:
    r"""Check that the room is small enough for the procedure to be defined.

    The standards say what to do below 25 m³ and say nothing at all about
    corner measurements above it, so a corner set from a room that does not
    trigger is a question none of the three parts answers, and answering it
    anyway would silently move the reported level by up to
    :math:`10 \lg(2/3)`.

    :param volume: Room volume ``V``, in m³.
    :param name: Argument name to quote in the message.
    :raises ValueError: If the volume is not positive and finite, or if it does
        not trigger the procedure.
    """
    if not low_frequency_procedure_applies(volume):
        rounded = math.floor(float(volume) + 0.5)
        msg = (
            f"The ISO 16283 low-frequency procedure is defined for a room "
            f"volume smaller than 25 m³ (calculated to the nearest cubic "
            f"metre); '{name}' is {float(volume):g} m³, which rounds to "
            f"{rounded} m³. Use the default procedure alone above the trigger."
        )
        raise ValueError(msg)


def _low_frequency_bands_are_present(frequencies: ArrayLike) -> bool:
    """Whether a band-centre vector names all three low-frequency bands.

    The non-raising half of :func:`_low_frequency_indices`, and the question
    :func:`_warn_when_the_procedure_is_required` has to answer before it can
    say anything. Every vector the raising version would refuse comes back
    ``False`` here, an ambiguous duplicate centre included: a warning about a
    missing procedure is not the place to report a malformed band axis, and the
    caller who does supply a procedure gets the proper message from the other
    one.

    :param frequencies: Band centre frequencies, in Hz.
    :return: ``True`` when 50 Hz, 63 Hz and 80 Hz are each named exactly once.
    """
    centres = np.asarray(frequencies, dtype=np.float64)
    if centres.ndim != 1:
        return False
    return all(
        np.count_nonzero(np.isclose(centres, target, rtol=_BAND_MATCH_RTOL)) == 1
        for target in LOW_FREQUENCY_BANDS
    )


def _warn_when_the_procedure_is_required(
    volume: float | None,
    frequencies: ArrayLike | None,
    supplied: LowFrequencyProcedure | None,
    *,
    owner: str,
    argument: str,
) -> None:
    """Say so when a room under the trigger is answered without the procedure.

    Clause 8.1 (Part 3: Clause 7.3.1) is a *shall*, not an option, so the
    50 Hz, 63 Hz and 80 Hz bands of a room that rounds below 25 m³ are not the
    ISO 16283 quantity unless the corner measurements went into them. The
    measurement functions cannot supply the corners themselves and refusing
    would take away the default-procedure spectrum the reader needs to compare
    against, so they say it and answer.

    Everything the test needs is already in hand for other reasons: the volume
    sizes the Sabine absorption area and the band centres label the spectrum.
    That is also the limit of it. The warning stays silent when the caller ran
    the procedure, when no volume was given, when the room does not trigger,
    when the band centres were not given, and when the optional low range of
    Clause 5 was not measured, because in none of those cases is there anything
    the library can be sure went wrong.

    :param volume: Receiving-room volume ``V``, in m³, or ``None``.
    :param frequencies: Band centre frequencies, in Hz, or ``None``.
    :param supplied: The procedure the caller passed for that room, if any.
    :param owner: Function name, for the message.
    :param argument: Name of the argument the procedure would arrive through.
    """
    if supplied is not None or volume is None or frequencies is None:
        return
    value = float(volume)
    # `low_frequency_procedure_applies` raises on a volume that is not one, and
    # a warning is the wrong voice for that: every entry point has its own
    # complaint about a non-positive volume, so this check stands aside and
    # lets it be made. Ordered as in that function, and for the same reason.
    if not math.isfinite(value) or value <= 0.0:
        return
    if not low_frequency_procedure_applies(value):
        return
    if not _low_frequency_bands_are_present(frequencies):
        return
    rounded = math.floor(value + 0.5)
    warnings.warn(
        f"{owner}: the receiving room is {value:g} m³, which rounds to "
        f"{rounded} m³, and 'frequencies' names the 50 Hz, 63 Hz and 80 Hz "
        "bands, so ISO 16283 requires the low-frequency procedure there "
        "(ISO 16283-1 and -2 Clause 8.1, ISO 16283-3 Clause 7.3.1, all "
        f"'shall'). Without '{argument}' those three bands carry the default "
        "procedure alone, which is not the ISO 16283 quantity: Formula (13) "
        "weighs a corner measurement into the level and Clause 10.4 (Clause "
        "8.4 in ISO 16283-3) puts the 63 Hz octave reverberation time in place "
        "of the three one-third-octave ones. Pass a LowFrequencyProcedure as "
        f"'{argument}', or leave the optional low range of Clause 5 out and "
        "report 100 Hz to 3150 Hz alone.",
        LowFrequencyWarning,
        # Three frames out: this helper, the entry point that called it, and
        # the caller's own line, which is the one worth pointing at.
        stacklevel=3,
    )


def corner_level(corner_levels: ArrayLike) -> np.ndarray:
    r"""Corner sound pressure level :math:`L_\mathrm{Corner}` per band.

    Two shapes are accepted, and they are the two the standards describe.

    A ``(corners, bands)`` array is one source position: the level is the
    highest of the measured corners in each band, taken independently per band
    because "the values for :math:`L_\mathrm{Corner}` may be associated with
    different corners in the room" (NOTE under Part 1 Formula (12)). This is
    also the case of loudspeakers operated simultaneously (Part 1 Clause 8.5,
    first paragraph) and the whole of Part 3, whose Clause 7.3.4 defines
    :math:`L_\mathrm{2,Corner}` as that maximum and numbers no formula.

    A ``(positions, corners, bands)`` array is a single source moved between
    ``q`` positions: the maximum is taken per position and the ``q`` results
    are energy-averaged, which is Part 1 Formula (12) and Part 2 Formula (15)
    written in levels rather than in mean-square pressures. The two forms agree
    because :math:`p^2/p_0^2 = 10^{L/10}`, and the ``(corners, bands)`` shape
    is the same formula at ``q = 1``.

    Corner levels are assumed already corrected for background noise. All
    three parts require a background measurement in **every** corner used, in
    their background-noise clause (Part 1 Clause 9.1, Part 2 Clause 9.1,
    Part 3 Clause 7.4.1), and Part 2 says it a second time in Formula (15)'s
    own where-list.

    :param corner_levels: Corner sound pressure levels, in dB, as
        ``(corners, bands)`` or ``(positions, corners, bands)``.
    :return: :math:`L_\mathrm{Corner}`, one value per band, in dB.
    :raises ValueError: If the array is not two- or three-dimensional, is
        empty, or holds a non-finite value.
    """
    data = np.asarray(corner_levels, dtype=np.float64)
    if data.ndim == _SINGLE_POSITION_RANK:
        data = data[np.newaxis, ...]
    elif data.ndim != _SINGLE_POSITION_RANK + 1:
        msg = (
            "'corner_levels' must be 2-D (corners x bands) for one source "
            "position or 3-D (positions x corners x bands) for several; got "
            f"{data.ndim} dimension(s)."
        )
        raise ValueError(msg)
    if data.size == 0:
        msg = "'corner_levels' must not be empty."
        raise ValueError(msg)
    if not np.all(np.isfinite(data)):
        msg = "'corner_levels' must contain only finite values."
        raise ValueError(msg)
    corners = data.shape[_CORNER_AXIS]
    if corners < _MIN_CORNERS:
        warnings.warn(
            f"Only {corners} corner(s) per source position; ISO 16283 requires "
            f"a minimum of {_MIN_CORNERS} (and recommends two of them at "
            "ground level and two at ceiling level).",
            LowFrequencyWarning,
            stacklevel=2,
        )
    highest = np.max(data, axis=_CORNER_AXIS)
    return np.asarray(energy_mean(highest, axis=_POSITION_AXIS), dtype=np.float64)


def low_frequency_level(level: ArrayLike, corner: ArrayLike) -> np.ndarray:
    r"""Combine the default and corner levels into :math:`L_\mathrm{LF}`.

    Part 1 Formula (13), Part 2 Formula (16) and Part 3 Formula (5), which are
    the same expression under three sets of subscripts:

    .. math::

        L_\mathrm{LF} = 10 \lg \left[ \frac{10^{0,1 L_\mathrm{Corner}} +
        (2 \cdot 10^{0,1 L})}{3} \right]

    The two levels are weighted one third to two thirds, so the result
    degenerates to ``L`` when the corner level equals it, rises with the corner
    level and can never fall further than :math:`10 \lg(2/3) = -1,76` dB below
    ``L``, however quiet the corners are.

    :param level: Energy-average level ``L`` from the default procedure, in dB,
        one value per band.
    :param corner: Corner level :math:`L_\mathrm{Corner}` from
        :func:`corner_level`, in dB, same bands.
    :return: :math:`L_\mathrm{LF}`, in dB, one value per band.
    :raises ValueError: If the two shapes differ, either is empty, or either
        holds a non-finite value.
    """
    default = np.asarray(level, dtype=np.float64)
    highest = np.asarray(corner, dtype=np.float64)
    if default.shape != highest.shape:
        msg = (
            "'level' and 'corner' must cover the same bands; got shapes "
            f"{default.shape} and {highest.shape}."
        )
        raise ValueError(msg)
    if default.size == 0:
        msg = "'level' and 'corner' must not be empty."
        raise ValueError(msg)
    for name, values in (("level", default), ("corner", highest)):
        if not np.all(np.isfinite(values)):
            msg = f"'{name}' must contain only finite values."
            raise ValueError(msg)
    combined = (10.0 ** (0.1 * highest) + 2.0 * 10.0 ** (0.1 * default)) / 3.0
    return np.asarray(10.0 * np.log10(combined), dtype=np.float64)


@dataclass(frozen=True)
class LowFrequencyProcedure:
    """The extra measurements ISO 16283 asks for in a room under 25 m³.

    One of these describes one room. Part 1 tests the source and the receiving
    room independently, so an airborne measurement may carry two, one, or
    neither; Parts 2 and 3 have only a receiving room.

    :ivar volume: Volume ``V`` of the room the corners were measured in, in m³.
        It is this room's own volume that decides the trigger (Part 1
        Clause 8.1, "in the source and/or receiving room when **its** volume").
    :ivar corner_levels: Corner sound pressure levels, in dB, already corrected
        for background noise, over the **three low-frequency bands only** and
        in 50 / 63 / 80 Hz order: ``(corners, 3)`` for one source position, or
        ``(positions, corners, 3)`` for a source moved between positions. Only
        those three bands are measured in the corners at all, so the corner
        sheet is three columns wide whatever range the default procedure
        covered.
    :ivar reverberation_63_octave: Reverberation time measured in the 63 Hz
        **octave** band, in seconds, which replaces the 50 Hz, 63 Hz and 80 Hz
        one-third-octave values (Part 1 and Part 2 Clause 10.4, Part 3
        Clause 8.4). Required for the receiving room, which is the only room
        those clauses speak about; must be ``None`` for a source room.
    """

    volume: float
    corner_levels: Sequence[float] | np.ndarray
    reverberation_63_octave: float | None = None

    def __post_init__(self) -> None:
        """Check the volume, the band axis and the 63 Hz octave time.

        The arithmetic of the corner levels is left to :func:`corner_level`;
        what is checked here is the band axis, which is fixed at three by the
        procedure itself, and the two scalars, because a non-positive volume or
        reverberation time would otherwise reach :func:`math.floor` and
        :func:`numpy.log10` and come back as an exception naming neither.

        :raises ValueError: If ``volume`` does not trigger the procedure, if
            ``corner_levels`` does not carry exactly the three low-frequency
            bands, or if ``reverberation_63_octave`` is given and is not
            positive and finite.
        """
        _require_volume_triggers(self.volume, "volume")
        corners = np.asarray(self.corner_levels, dtype=np.float64)
        if corners.ndim < 1 or corners.shape[-1] != len(LOW_FREQUENCY_BANDS):
            msg = (
                "'corner_levels' must carry exactly the three low-frequency "
                "bands (50 Hz, 63 Hz, 80 Hz) on its last axis; got shape "
                f"{corners.shape}."
            )
            raise ValueError(msg)
        t63 = self.reverberation_63_octave
        if t63 is None:
            return
        value = float(t63)
        # Same ordering, and for the same reason, as in
        # `low_frequency_procedure_applies`: NaN is not caught by `<= 0.0`.
        if not math.isfinite(value) or value <= 0.0:
            msg = (
                "'reverberation_63_octave' must be a positive, finite "
                f"reverberation time in seconds; got {t63!r}."
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class LowFrequencyResult:
    r"""What the low-frequency procedure did to one room's band values.

    :ivar frequencies: Band centre frequencies of the whole measurement, in Hz,
        as supplied.
    :ivar level: The energy-average levels of the whole measurement, in dB,
        with the 50 Hz, 63 Hz and 80 Hz bands replaced by
        :math:`L_\mathrm{LF}` and every other band untouched.
    :ivar reverberation_time: The reverberation times of the whole
        measurement, in seconds, with those same three bands replaced by the
        63 Hz octave value; ``None`` for a source room, which Clause 10.4 does
        not speak about.
    :ivar low_frequency_bands: The three band centres the procedure was applied
        at, in Hz, as they were spelled in ``frequencies``.
    :ivar l_default: The default-procedure levels at those three bands, in dB,
        before the combination.
    :ivar l_corner: :math:`L_\mathrm{Corner}` at those three bands, in dB
        (Part 1 Formula (12), Part 2 Formula (15), Part 3 Clause 7.3.4).
    :ivar l_lf: :math:`L_\mathrm{LF}` at those three bands, in dB (Part 1
        Formula (13), Part 2 Formula (16), Part 3 Formula (5)).
    :ivar volume: Volume of the room, in m³, that put the procedure in force.
    :ivar reverberation_63_octave: The 63 Hz octave reverberation time, in
        seconds, or ``None`` when none was substituted.
    """

    frequencies: np.ndarray
    level: np.ndarray
    reverberation_time: np.ndarray | None
    low_frequency_bands: np.ndarray
    l_default: np.ndarray
    l_corner: np.ndarray
    l_lf: np.ndarray
    volume: float
    reverberation_63_octave: float | None

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the three low-frequency bands, default against corner and LF.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_low_frequency_procedure

        check_language(language)
        return plot_low_frequency_procedure(self, ax=ax, language=language, **kwargs)


def _low_frequency_indices(frequencies: np.ndarray) -> np.ndarray:
    """Locate the 50 Hz, 63 Hz and 80 Hz bands in a band-centre vector.

    Matched by nominal centre within :data:`_BAND_MATCH_RTOL`, so a caller who
    labels the bands 49,6 / 62,5 / 79,4 Hz is understood as well as one who
    writes 50 / 63 / 80. All three must be present: the procedure is stated for
    the three together and there is no reading of it that treats two.

    :param frequencies: Band centre frequencies, in Hz.
    :return: The three indices, in 50 / 63 / 80 Hz order.
    :raises ValueError: If a band is missing, or if more than one band centre
        answers to the same nominal frequency.
    """
    indices: list[int] = []
    missing: list[float] = []
    for target in LOW_FREQUENCY_BANDS:
        match = np.flatnonzero(np.isclose(frequencies, target, rtol=_BAND_MATCH_RTOL))
        if match.size == 0:
            missing.append(target)
            continue
        if match.size > 1:
            msg = (
                f"'frequencies' carries {match.size} band centres within 3 % of "
                f"{target:g} Hz, so the low-frequency band cannot be identified."
            )
            raise ValueError(msg)
        indices.append(int(match[0]))
    if missing:
        names = ", ".join(f"{f:g} Hz" for f in missing)
        msg = (
            "The ISO 16283 low-frequency procedure covers the 50 Hz, 63 Hz and "
            f"80 Hz one-third-octave bands together; 'frequencies' is missing "
            f"{names}."
        )
        raise ValueError(msg)
    return np.asarray(indices, dtype=np.intp)


def apply_low_frequency_procedure(
    level: ArrayLike,
    frequencies: ArrayLike,
    procedure: LowFrequencyProcedure,
    *,
    reverberation_time: ArrayLike | None = None,
    room: str = "receiving",
) -> LowFrequencyResult:
    r"""Run the low-frequency procedure over one room's band values.

    The single implementation behind
    :func:`~phonometry.building.measurement.insulation.airborne_insulation`,
    :func:`~phonometry.building.measurement.insulation.impact_insulation` and
    :func:`~phonometry.building.measurement.insulation.facade_insulation`: it
    takes :math:`L_\mathrm{Corner}` from the corner levels, combines it with
    the default-procedure level into :math:`L_\mathrm{LF}`, writes the result
    back over the 50 Hz, 63 Hz and 80 Hz bands, and puts the 63 Hz octave
    reverberation time over the same three bands.

    ``room`` decides which half of the procedure is in force. Clause 10.4
    (Part 3: Clause 8.4) is a receiving-room clause in all three parts, even in
    Part 1 where the corner procedure itself also admits the source room, so a
    ``"receiving"`` call carries both halves and a ``"source"`` call carries
    neither: it takes no reverberation times and refuses a procedure that
    brings a 63 Hz octave value.

    :param level: Energy-average levels of the whole measurement, in dB, one
        value per band.
    :param frequencies: Band centre frequencies, in Hz, same length.
    :param procedure: The room's corner measurements and volume.
    :param reverberation_time: Reverberation times of the whole measurement, in
        seconds. Required for ``room="receiving"``; refused for
        ``room="source"``.
    :param room: ``"receiving"`` (default) or ``"source"``. It selects whether
        Clause 10.4 applies, and it is quoted in the messages so a two-room
        airborne measurement says which side failed.
    :return: A :class:`LowFrequencyResult`.
    :raises ValueError: If ``room`` is neither name, if the band counts
        disagree, if the 50 Hz, 63 Hz or 80 Hz band is absent from
        ``frequencies``, if a receiving room is missing either half of
        Clause 10.4, or if a source room brings either half of it.
    """
    if room not in _ROOMS:
        expected = " or ".join(repr(name) for name in _ROOMS)
        msg = f"'room' must be {expected}, got {room!r}."
        raise ValueError(msg)
    levels = np.asarray(level, dtype=np.float64)
    freqs = np.asarray(frequencies, dtype=np.float64)
    if levels.ndim != 1 or freqs.ndim != 1:
        msg = (
            "'level' and 'frequencies' must each be one-dimensional (one "
            f"value per band); got {levels.ndim} and {freqs.ndim} dimension(s)."
        )
        raise ValueError(msg)
    if levels.size != freqs.size:
        msg = (
            f"'frequencies' has {freqs.size} band(s) and the {room}-room "
            f"levels have {levels.size}; they must match for the "
            "low-frequency procedure to know which bands are 50, 63 and 80 Hz."
        )
        raise ValueError(msg)
    indices = _low_frequency_indices(freqs)

    # The corner sheet is already three columns wide, in 50 / 63 / 80 Hz order:
    # `LowFrequencyProcedure` fixes that on the way in, so no band lookup is
    # needed on this side and the indices are only used against the default
    # procedure's own, wider, band axis.
    l_corner = corner_level(procedure.corner_levels)
    l_default = levels[indices]
    l_lf = low_frequency_level(l_default, l_corner)

    corrected = levels.copy()
    corrected[indices] = l_lf

    t63 = procedure.reverberation_63_octave
    substituted: np.ndarray | None = None
    if room == _SOURCE_ROOM:
        if t63 is not None or reverberation_time is not None:
            msg = (
                "Clause 10.4 (Clause 8.4 in ISO 16283-3) substitutes the 63 Hz "
                "octave reverberation time in the receiving room and in no "
                "other, so a source-room call takes neither "
                "'reverberation_63_octave' nor 'reverberation_time'."
            )
            raise ValueError(msg)
    else:
        if t63 is None:
            msg = (
                "A receiving room under 25 m³ needs the 63 Hz octave "
                "reverberation time: ISO 16283-1 and -2 Clause 10.4 and "
                "ISO 16283-3 Clause 8.4 require it in place of the 50 Hz, "
                "63 Hz and 80 Hz one-third-octave values, and below the "
                "trigger Clause 10.3 (8.3) leaves no default value there to "
                "fall back on. Set 'reverberation_63_octave' on the procedure."
            )
            raise ValueError(msg)
        if reverberation_time is None:
            msg = (
                "A receiving-room call needs 'reverberation_time', the "
                "measured one-third-octave values, because Clause 10.4 "
                "replaces three of them and leaves the rest as they are."
            )
            raise ValueError(msg)
        times = np.asarray(reverberation_time, dtype=np.float64)
        if times.shape != levels.shape:
            msg = (
                f"The {room}-room reverberation times cover {times.shape} and "
                f"the levels cover {levels.shape}; they must match."
            )
            raise ValueError(msg)
        substituted = times.copy()
        substituted[indices] = float(t63)

    return LowFrequencyResult(
        frequencies=freqs,
        level=corrected,
        reverberation_time=substituted,
        low_frequency_bands=freqs[indices],
        l_default=l_default,
        l_corner=l_corner,
        l_lf=l_lf,
        volume=float(procedure.volume),
        reverberation_63_octave=None if t63 is None else float(t63),
    )
