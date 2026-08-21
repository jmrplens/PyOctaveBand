#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Prediction of wall and ceiling linings: the weighted improvement an additional
layer gives the element behind it (ISO 12354-1:2017 Annex D).

A lining is not a floor covering. It is added to a wall or a ceiling, not laid
under a tapping machine, and what it changes is the airborne sound reduction
index of the basic element it is fixed to. Annex D therefore rates it as a
shift of a single number rather than as a spectrum, and it reads that shift off
one quantity: the mass-spring resonance of the lining against the element. That
single frequency is what makes the annex one subject and this module one file.

A lining improves or *degrades* the sound insulation depending on where its
resonance falls, so Annex D predicts the weighted improvement from ``fo``
alone: Formula (D.1) for a layer bonded directly to the wall, Formula (D.2) for
one on studs over a filled cavity, then Table D.1 for interior linings
(:func:`weighted_lining_improvement`), Formulae (D.3) to (D.6) for exterior
thermal systems and (D.7) for stud systems (:func:`lining_improvement`), and
Formula (D.8) to carry a laboratory rating to the field
(:func:`lining_improvement_in_situ`).

Citations are to ISO 12354-1:2017. One printed defect is relevant here and is
recorded in ``docs/ERRATA.md``: the overlap of the last two rows of Table D.1
at 1 600 Hz.

Several relations used here carry no published worked example, so they are
implemented as printed and checked only for self-consistency: the cavity
stiffness :math:`0.111/d` of Formula (D.2) and the exterior-system and stud
fits of Formulae (D.3) to (D.8). The guide "Predicting Resilient-Layer
Performance" says which pieces have an oracle and which do not.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

from ..._internal.validation import require_choice, require_positive

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "LiningImprovementResult",
    "lining_improvement",
    "lining_improvement_in_situ",
    "lining_resonance_frequency",
    "weighted_lining_improvement",
]

#: Cavity stiffness per unit area of ISO 12354-1:2017 Formula (D.2), in N/m³·m:
#: ``s' = 0,111/d`` with ``s'`` in MN/m³, i.e. ``0,111e6/d`` in SI.
_CAVITY_STIFFNESS: float = 0.111e6

#: ISO 12354-1:2017 Table D.1, the branch for ``30 ≤ fo ≤ 160``:
#: ``ΔRw = 74,4 − 20 lg(fo) − Rw/2``, floored at 0 dB by NOTE 1.
_TABLE_D1_LOW = (74.4, 20.0, 2.0)

#: Highest nominal one-third-octave centre served by the ``_TABLE_D1_LOW``
#: branch of Table D.1, in Hz; above it the fixed ``_TABLE_D1_HIGH`` rows
#: apply.
_TABLE_D1_LOW_MAX = 160.0

#: ISO 12354-1:2017 Table D.1, the fixed rows above 160 Hz, as
#: ``(upper bound of fo in Hz, ΔRw in dB)`` evaluated in order. The 1 600 Hz
#: row is printed twice with different values; see ``docs/ERRATA.md``.
_TABLE_D1_HIGH: tuple[tuple[float, float], ...] = (
    (200.0, -1.0),
    (250.0, -3.0),
    (315.0, -5.0),
    (400.0, -7.0),
    (500.0, -9.0),
    (1600.0, -10.0),
    (5000.0, -5.0),
)

#: Validity range of ``fo`` covered by ISO 12354-1:2017 Table D.1, in Hz. It
#: is the only range of lining resonances Annex D puts numbers on, so the
#: Formula (D.3) to (D.7) fits are warned about outside it too.
_TABLE_D1_RANGE = (30.0, 5000.0)

#: Validity range of ``Rw`` Clause D.2.2 states Table D.1 for, in dB:
#: "For basic structural elements with a weighted sound reduction index in the
#: range of 20 dB <= Rw <= 60 dB".
_TABLE_D1_RW_RANGE = (20.0, 60.0)

#: ISO 12354-1:2017 Formulae (D.3), (D.4) and (D.7): the reference-situation
#: single-number ratings as ``(slope, intercept, floor)`` triples for
#: ``ΔRw``, ``ΔRA`` and ``ΔRA,tr``, keyed by the interlayer type.
_ANNEX_D_SYSTEMS: dict[str, tuple[tuple[float, float, float], ...]] = {
    # Formula (D.3): exterior system glued to the wall, mineral wool.
    "mineral_wool": (
        (-36.0, 82.5, -4.0),
        (-42.0, 92.0, -4.0),
        (-39.0, 87.7, -4.0),
    ),
    # Formula (D.4): the same, foams (PS, EPS, EEPS).
    "foam": (
        (-33.0, 76.0, -3.0),
        (-33.0, 74.0, -3.0),
        (-36.0, 77.0, -3.0),
    ),
    # Formula (D.7): system on studs, not directly fixed to the basic wall.
    "studs": (
        (-20.0, 48.0, -4.0),
        (-22.0, 51.0, -4.0),
        (-24.0, 54.0, -4.0),
    ),
}

#: ISO 12354-1:2017 Formula (D.5): correction for 4 to 10 anchors or battens
#: per m², as ``(factor, offset)`` per rating.
_ANNEX_D_ANCHORS = ((0.66, -1.2), (0.62, -1.3), (0.54, -1.6))

#: ISO 12354-1:2017 Formula (D.6): correction for a glued area other than the
#: 40 % reference, ``ΔR − 0,05 %So + 2,0``.
_ANNEX_D_GLUE = (-0.05, 2.0)

#: ISO 12354-1:2017 Formula (D.8): laboratory-to-field transfer of a weighted
#: improvement, ``a = 1,35 lg(fo) − 3,5 ≤ 0`` and ``X = Rw,situ − 53`` clamped
#: to ``[−10, +7]``.
_ANNEX_D8_A = (1.35, -3.5)
_ANNEX_D8_X = (53.0, -10.0, 7.0)

#: Nominal one-third-octave centre frequencies (ISO 266) used to round ``fo``
#: before reading Table D.1 (Clause D.2.2), in ascending band order.
_THIRD_OCTAVE_CENTRES: tuple[float, ...] = (
    12.5,
    16.0,
    20.0,
    25.0,
    31.5,
    40.0,
    50.0,
    63.0,
    80.0,
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
    1600.0,
    2000.0,
    2500.0,
    3150.0,
    4000.0,
    5000.0,
    6300.0,
    8000.0,
    10000.0,
)

#: Base-ten band index ``n`` of the first entry of ``_THIRD_OCTAVE_CENTRES``:
#: the nominal 12,5 Hz band has the exact midband frequency ``10^(11/10)``.
_THIRD_OCTAVE_FIRST_INDEX: int = 11

#: Additional-layer system of ISO 12354-1:2017 Annex D.
LiningSystem = Literal["mineral_wool", "foam", "studs"]


@overload
def lining_resonance_frequency(
    base_mass_per_area: float,
    lining_mass_per_area: float,
    *,
    dynamic_stiffness: float,
) -> float: ...


@overload
def lining_resonance_frequency(
    base_mass_per_area: float,
    lining_mass_per_area: float,
    *,
    cavity_depth: float,
) -> float: ...


def lining_resonance_frequency(
    base_mass_per_area: float,
    lining_mass_per_area: float,
    *,
    dynamic_stiffness: float | None = None,
    cavity_depth: float | None = None,
) -> float:
    r"""Resonance ``fo`` of a lining on a basic element (Formulae D.1/D.2).

    Exactly one of the two branches applies:

    * ``dynamic_stiffness`` (Formula D.1), for an insulation layer fixed
      **directly** to the basic construction, without studs or battens:
      :math:`f_\mathrm{o} = \sqrt{s' (1/m'_1 + 1/m'_2)}/(2 \pi)`.
    * ``cavity_depth`` (Formula D.2), for a layer built on metal or wooden
      studs **not** connected to the basic element, with the cavity filled by a
      porous layer of airflow resistivity :math:`r \ge 5` kPa·s/m²:
      :math:`f_\mathrm{o} = \sqrt{(0.111/d)(1/m'_1 + 1/m'_2)}/(2 \pi)`, i.e. the
      near-isothermal stiffness of the filled cavity replaces ``s'``.

    :param base_mass_per_area: Mass per unit area ``m'1`` of the basic
        structural element, in kg/m².
    :param lining_mass_per_area: Mass per unit area ``m'2`` of the additional
        layer, in kg/m².
    :param dynamic_stiffness: Dynamic stiffness per unit area ``s'`` of the
        insulation layer (EN 29052-1), in N/m³.
    :param cavity_depth: Depth ``d`` of the stud cavity, in m.
    :return: The resonance frequency ``fo``, in Hz.
    :raises ValueError: If an input is not positive and finite, or if the two
        branches are both given or both omitted.
    """
    m1 = require_positive(base_mass_per_area, "base_mass_per_area")
    m2 = require_positive(lining_mass_per_area, "lining_mass_per_area")
    if (dynamic_stiffness is None) == (cavity_depth is None):
        msg = (
            "Give exactly one of 'dynamic_stiffness' (Formula D.1) or "
            "'cavity_depth' (Formula D.2)."
        )
        raise ValueError(msg)
    if dynamic_stiffness is not None:
        stiffness = require_positive(dynamic_stiffness, "dynamic_stiffness")
    else:
        stiffness = _CAVITY_STIFFNESS / require_positive(
            cavity_depth or 0.0, "cavity_depth"
        )
    return float(np.sqrt(stiffness * (1.0 / m1 + 1.0 / m2)) / (2.0 * np.pi))


def _round_to_third_octave(frequency: float) -> float:
    r"""Nominal centre of the one-third-octave band containing ``frequency``.

    Clause D.2.2 rounds ``fo`` to "the centre frequency of the one-third-octave
    band in which fo falls", which is band membership, not proximity to a
    nominal label. The band is therefore found from the exact midband
    frequencies of the base-ten system, :math:`10^{n/10}`, whose edges are
    :math:`10^{(n \pm 0.5)/10}`; the nominal label of that band is returned.

    The distinction is not cosmetic. Nominal labels are rounded, so the
    midpoint between two of them is not the band edge: the 63 Hz band ends at
    :math:`10^{1.85} = 70.79` Hz while the geometric mean of the labels 63
    and 80 is 70.99 Hz, and any ``fo`` between the two would be read off the
    wrong row of Table D.1, by 2.1 dB at that boundary and by 8.8 dB at the
    160 Hz to 200 Hz one.
    """
    index = int(np.floor(10.0 * np.log10(frequency) + 0.5))
    band = index - _THIRD_OCTAVE_FIRST_INDEX
    if band < 0:
        return _THIRD_OCTAVE_CENTRES[0]
    if band >= len(_THIRD_OCTAVE_CENTRES):
        return _THIRD_OCTAVE_CENTRES[-1]
    return _THIRD_OCTAVE_CENTRES[band]


def weighted_lining_improvement(
    resonance_frequency: float, base_rating: float
) -> float:
    r"""Weighted improvement ``ΔRw`` of an interior lining (Table D.1).

    ISO 12354-1:2017 Table D.1 reads ``ΔRw`` off the lining's resonance
    frequency, rounded to the centre of the one-third-octave band in which it
    falls. Below 200 Hz the improvement also depends on the bare element:
    :math:`\Delta R_\mathrm{w} = 74.4 - 20 \log_{10}(f_\mathrm{o}) - R_\mathrm{w}/2`, never below 0 dB
    (NOTE 1). At and above 200 Hz the lining *degrades* the insulation, by
    1 dB at 200 Hz down to 10 dB from 630 Hz to 1 600 Hz, recovering to 5 dB
    from 1 600 Hz to 5 000 Hz.

    Table D.1 is stated for basic elements with :math:`20 \le R_\mathrm{w} \le 60` dB.
    Its last two rows both cover 1 600 Hz with different values; this function
    takes the more conservative −10 dB there (see ``docs/ERRATA.md``).

    :param resonance_frequency: Resonance frequency ``fo`` of the lining, in
        Hz (:func:`lining_resonance_frequency`); must fall in the 30 Hz to
        5 000 Hz range Table D.1 covers.
    :param base_rating: Weighted sound reduction index ``Rw`` of the bare wall
        or floor, in dB.
    :return: The weighted improvement ``ΔRw``, in dB.
    :raises ValueError: If ``fo`` is outside the tabulated range or an input is
        not finite.
    """
    f0 = require_positive(resonance_frequency, "resonance_frequency")
    rw = float(base_rating)
    if not np.isfinite(rw):
        msg = "'base_rating' must be finite."
        raise ValueError(msg)
    low, high = _TABLE_D1_RANGE
    if not low <= f0 <= high:
        msg = (
            f"'resonance_frequency' must lie in [{low:g}, {high:g}] Hz; "
            "ISO 12354-1 Table D.1 is not tabulated outside it."
        )
        raise ValueError(msg)
    rw_low, rw_high = _TABLE_D1_RW_RANGE
    if not rw_low <= rw <= rw_high:
        warnings.warn(
            f"base_rating = {rw:g} dB lies outside the "
            f"{rw_low:g} dB <= Rw <= {rw_high:g} dB range Clause D.2.2 states "
            "Table D.1 for; the result is an extrapolation of "
            "74,4 - 20 lg(fo) - Rw/2 and is not covered by the standard.",
            UserWarning,
            stacklevel=2,
        )
    nominal = _round_to_third_octave(f0)
    if nominal <= _TABLE_D1_LOW_MAX:
        a, b, c = _TABLE_D1_LOW
        return float(max(0.0, a - b * np.log10(nominal) - rw / c))
    for upper, value in _TABLE_D1_HIGH:
        if nominal <= upper:
            return float(value)
    return float(_TABLE_D1_HIGH[-1][1])


@dataclass(frozen=True)
class LiningImprovementResult:
    """Single-number ratings of an additional layer (ISO 12354-1 Annex D).

    :ivar resonance_frequency: Resonance frequency ``fo`` of the system, in Hz.
    :ivar system: ``"mineral_wool"``, ``"foam"`` (exterior systems glued to the
        wall, Formulae D.3/D.4) or ``"studs"`` (Formula D.7).
    :ivar delta_rw: Improvement of the weighted sound reduction index
        ``ΔRw``, in dB.
    :ivar delta_ra: Improvement of the A-weighted rating ``ΔRA``, in dB.
    :ivar delta_ratr: Improvement of the traffic-weighted rating ``ΔRA,tr``,
        in dB.
    :ivar anchors: ``True`` when the Formula (D.5) anchor correction was
        applied.
    :ivar glued_area: Glued area as a percentage of the element area, or
        ``None`` when the 40 % reference was kept.
    """

    resonance_frequency: float
    system: LiningSystem
    delta_rw: float
    delta_ra: float
    delta_ratr: float
    anchors: bool = False
    glued_area: float | None = None

    @property
    def ratings(self) -> tuple[float, float, float]:
        """``(ΔRw, ΔRA, ΔRA,tr)`` as a tuple, in dB."""
        return self.delta_rw, self.delta_ra, self.delta_ratr

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the Annex D ratings against the resonance frequency.

        Draws the three Annex D curves over the tabulated range with this
        system's own resonance marked, the analogue of Figures D.2 and D.3.
        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_lining_improvement

        check_language(language)
        return plot_lining_improvement(self, ax=ax, language=language, **kwargs)


def lining_improvement(
    resonance_frequency: float,
    *,
    system: LiningSystem = "mineral_wool",
    anchors: bool = False,
    glued_area: float | None = None,
) -> LiningImprovementResult:
    r"""Single-number ratings of an additional layer (Formulae D.3 to D.7).

    For the reference situation of ISO 12354-1:2017 Annex D, a system applied
    to a heavy basic wall of about 350 kg/m²:

    * ``system="mineral_wool"`` (Formula D.3), an exterior thermal system on
      mineral wool with 40 % glued area and no anchors:
      :math:`\Delta R_\mathrm{w} = -36 \log_{10}(f_\mathrm{o}) + 82.5`,
      :math:`\Delta R_\mathrm{A} = -42 \log_{10}(f_\mathrm{o}) + 92.0`,
      :math:`\Delta R_\mathrm{A,tr} = -39 \log_{10}(f_\mathrm{o}) + 87.7`, each floored at −4 dB.
    * ``system="foam"`` (Formula D.4), the same on PS, EPS or EEPS foams:
      :math:`-33 \log_{10}(f_\mathrm{o}) + 76.0`, :math:`-33 \log_{10}(f_\mathrm{o}) + 74.0`,
      :math:`-36 \log_{10}(f_\mathrm{o}) + 77.0`, floored at −3 dB.
    * ``system="studs"`` (Formula D.7), a layer on studs not directly fixed to
      the basic wall: :math:`-20 \log_{10}(f_\mathrm{o}) + 48`, :math:`-22 \log_{10}(f_\mathrm{o}) + 51`,
      :math:`-24 \log_{10}(f_\mathrm{o}) + 54`, floored at −4 dB.

    ``anchors=True`` applies Formula (D.5) for 4 to 10 anchors or battens per
    m² (:math:`0.66 \Delta R_\mathrm{w,ref} - 1.2` and its two companions), and
    ``glued_area`` applies Formula (D.6),
    :math:`\Delta R - 0.05\,\%S_o + 2.0`, for a glued area other than the
    40 % reference. Both corrections are applied after the floor of the
    reference formula, in the order the annex states them.

    The annex places the :math:`\ge -4` dB (or :math:`\ge -3` dB) floor
    inside Formulae (D.3) and (D.4) and says nothing about re-applying it
    after (D.5) and (D.6), so this function does not: a fully glued system on
    anchors can
    return about −6.8 dB, below the reference floor. That is the annex read
    literally, and the reason the two corrections are exposed as flags rather
    than folded into the fit.

    :param resonance_frequency: Resonance frequency ``fo``, in Hz
        (:func:`lining_resonance_frequency`).
    :param system: ``"mineral_wool"``, ``"foam"`` or ``"studs"``.
    :param anchors: Apply the Formula (D.5) anchor/batten correction.
    :param glued_area: Glued area ``%So`` as a percentage of the element area,
        greater than 0 and at most 100, or ``None`` to keep the 40 % reference.
        Formula (D.6) divides by the glued area, so a wholly unglued system is
        not a case of it: use ``anchors`` for a mechanically fixed lining. It
        corrects the glued exterior systems only, so it is rejected for
        ``system="studs"``.
    :return: A :class:`LiningImprovementResult`.
    :raises ValueError: If an input is not positive and finite, ``system`` is
        unknown, or ``glued_area`` is out of range or combined with
        ``system="studs"``.
    """
    f0 = require_positive(resonance_frequency, "resonance_frequency")
    require_choice(system, "system", tuple(_ANNEX_D_SYSTEMS))
    low, high = _TABLE_D1_RANGE
    if not low <= f0 <= high:
        warnings.warn(
            f"resonance_frequency = {f0:g} Hz lies outside the "
            f"{low:g} Hz to {high:g} Hz range Annex D puts lining resonances "
            "on; the fits are monotonic in lg(fo) and unbounded below it, so "
            "the result is an extrapolation.",
            UserWarning,
            stacklevel=2,
        )
    ratings = [
        max(floor, slope * float(np.log10(f0)) + intercept)
        for slope, intercept, floor in _ANNEX_D_SYSTEMS[system]
    ]
    if anchors:
        ratings = [
            factor * value + offset
            for value, (factor, offset) in zip(ratings, _ANNEX_D_ANCHORS, strict=True)
        ]
    if glued_area is not None:
        if system == "studs":
            msg = (
                "'glued_area' applies to the glued exterior systems of "
                "Formulae (D.3)/(D.4); Formula (D.7) has no glued area."
            )
            raise ValueError(msg)
        area = require_positive(glued_area, "glued_area")
        if area > 100.0:  # noqa: PLR2004
            msg = "'glued_area' is a percentage and cannot exceed 100."
            raise ValueError(msg)
        slope, offset = _ANNEX_D_GLUE
        ratings = [value + slope * area + offset for value in ratings]
    return LiningImprovementResult(
        resonance_frequency=f0,
        system=system,
        delta_rw=float(ratings[0]),
        delta_ra=float(ratings[1]),
        delta_ratr=float(ratings[2]),
        anchors=anchors,
        glued_area=None if glued_area is None else float(glued_area),
    )


def lining_improvement_in_situ(
    laboratory_improvement: float,
    resonance_frequency: float,
    base_rating_in_situ: float,
) -> float:
    r"""Transfer a weighted lining improvement to the field (Formula D.8).

    Even when the per-band improvement is invariant, its single-number rating
    still depends on the basic element it sits on, so ISO 12354-1:2017
    Formula (D.8) shifts the laboratory rating by :math:`a X` with

    :math:`a = 1.35 \log_{10}(f_\mathrm{o}) - 3.5`, capped at 0, and
    :math:`X = R_\mathrm{w,situ} - 53`, clamped to ``[−10, +7]``.

    The same formula applies to ``ΔRw``, ``ΔRA`` and ``ΔRA,tr``.

    :param laboratory_improvement: Laboratory rating ``ΔRlab`` measured to
        ISO 10140-1:2016 Annex G for the heavy basic element, in dB.
    :param resonance_frequency: Resonance frequency ``fo`` of the system, in
        Hz.
    :param base_rating_in_situ: Weighted sound reduction index ``Rw,situ`` of
        the basic element in the field situation, in dB.
    :return: The field rating ``ΔRsitu``, in dB.
    :raises ValueError: If an input is not finite, or ``fo`` is not positive.
    """
    delta_lab = float(laboratory_improvement)
    rw = float(base_rating_in_situ)
    if not np.isfinite(delta_lab) or not np.isfinite(rw):
        msg = "'laboratory_improvement' and 'base_rating_in_situ' must be finite."
        raise ValueError(msg)
    f0 = require_positive(resonance_frequency, "resonance_frequency")
    slope, offset = _ANNEX_D8_A
    a = min(0.0, slope * float(np.log10(f0)) + offset)
    reference, low, high = _ANNEX_D8_X
    x = min(high, max(low, rw - reference))
    return float(delta_lab + a * x)
