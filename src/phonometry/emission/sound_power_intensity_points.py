#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Sound power level of a noise source from sound intensity measured at
**discrete points**: ISO 9614-1:1993.

A p-p probe is held still at each of ``N`` points, one per segment of a
hypothetical surface enclosing the source, and reports the signed normal
intensity :math:`I_{\mathrm{n}i}` there. Each point stands for its segment, so
the partial power of the segment and the sound power of the source are
(clause 9.1 equation (11), clause 9.2 equation (12)):

.. math::

   P_i = I_{\mathrm{n}i} \, S_i \tag{Eq. 11}

   L_W = 10 \log_{10} \frac{\sum_{i=1}^{N} P_i}{P_0}, \qquad
   P_0 = 10^{-12}~\text{W} \tag{Eq. 12}

Equation (12) prints without the absolute-value bars of the general definition
(equation (8)), because clause 9.2 disposes of the negative case instead: **the
method is not applicable to a band in which** :math:`\sum_i P_i` **is
negative**. A single segment may still carry negative power, and normally
does; that is energy flowing inward through part of the surface, which is what
:math:`F_3` exists to quantify, not an error to reject.

A.2.3 makes a second refusal, on a different quantity: where
:math:`\sum_i I_{\mathrm{n}i}` is not positive, "the test conditions do not
satisfy the requirements of this part of ISO 9614 in that frequency band". That sum is unweighted over the ``N`` positions, so equal
segments make the two refusals agree and unequal ones let them part company: a
band clause 9.2 keeps, with a positive total power and a finite level, can
still be one A.2.3 refuses. Both are reported, each in its own terms.

The sign lives in the print, not in the number. ISO 9614-1 writes a normal
intensity level as ``XX dB`` when the flow is outward and as ``(-) XX dB`` when
it is inward, ``XX`` being a positive number in both cases (clause 3.5, and the
two unnumbered equations of clauses 9.1 and A.2.3):

.. math::

   I_{\mathrm{n}i} = I_0 \times 10^{XX/10}, \qquad
   I_{\mathrm{n}i} = -I_0 \times 10^{XX/10}

with :math:`I_0 = 10^{-12}` W/m^2. :func:`normal_intensity_from_levels` takes
both forms; its ``negative`` argument is the ``(-)`` of the print.

The four Annex A field indicators come from
:func:`phonometry.emission.field_indicators`, which is written for this part of
ISO 9614 and averages over positions without area weighting, exactly as
equations (A.4), (A.5), (A.7) and (A.9) do. What this module adds is Annex B:
the qualification of the surface and of the position set, and what to change
when they do not qualify.

Annex B numbers **two** criteria, not three (B.1.1 and B.1.2). Figure B.1
gates the determination on four questions, in this order, and Table B.3 gives
the action to take when each fails:

.. math::

   F_1 \le 0.6 \quad\text{(Table B.3, action e)}

   \text{criterion 1:} \quad L_\mathrm{d} > F_2, \qquad
   L_\mathrm{d} = \delta_{pI0} - K \quad\text{(Eq. (B.1), actions a or b)}

   F_3 - F_2 \le 3~\text{dB} \quad\text{(Figure B.1, actions a or b)}

   \text{criterion 2:} \quad N > C F_4^2 \quad\text{(Eq. (B.2), actions c or d)}

The third gate is unnumbered in the print: it appears in Figure B.1 and shares
the second row of Table B.3 with criterion 1, and it is *not* a "criterion 3".
Only Figure B.1's first failing gate is acted on, because every action box in
the figure returns to the next measurement rather than to the gate below it, so
:meth:`DiscretePointIntensityResult.required_actions` reports one action set per
band and stops there.

**Grade 3 is an A-weighted determination only.** Table B.2 tabulates the
criterion-2 factor ``C`` for grades 1 and 2 band by band and gives grade 3 a
single A-weighted value (8) and no per-band column at all; Table B.1 does the
same with the error factor :math:`\Delta` (0,20 and 0,29 for all bands at
grades 1 and 2, 0,60 A-weighted at grade 3); and Table 2 does the same with the
standard deviation ``s`` of the determination. Three tables agree, so the
asymmetry is the standard's design and not a gap in it: a per-band
determination can reach grade 1 or grade 2, and grade 3 is reached, if at all,
by the A-weighted sum. Asking any of the three lookups here for a per-band
grade-3 figure raises rather than returning a plausible number.

The uncertainty of the determination is Table 2's ``s``, with footnote 1
placing the true value within :math:`\pm 2s` of the measured one at 95 %
confidence. Clause 10.6 says which row of the table to read it from: "the
grade of accuracy attained in the final test, according to table 2, shall
be stated", the grade **attained** and not the grade set out for, so a band
that only reached grade 2 carries the grade-2 figure even where grade 1 was asked
for. A band that fails criterion 2 may still be recorded, provided the 95 %
confidence interval of equation (B.3) accompanies it (B.1.2, clause 10.5 c)
and clause 10.6):

.. math::

   10 \log_{10}\!\left( 1 \pm \frac{2 F_4}{\sqrt{N}} \right)~\text{dB}
   \tag{Eq. B.3}

:func:`partial_power_concentration` is the optional procedure of clause 8.3.2
and B.1.3: where criterion 1 holds, criterion 2 does not and
:math:`F_3 - F_2 \le 1` dB, most of the power may pass through a minority of
the segments, and adding positions only there is cheaper than densifying the
whole surface. It is implemented because it is the only consumer of Table B.1
and because equation (B.4) is the standard's own answer to the commonest way a
discrete-point survey fails.

Two things this module does not do. It does not scan: the continuous sweep of
ISO 9614-2:1996 and ISO 9614-3:2002, whose indicators are area weighted and
whose criteria are numbered differently, is
:func:`phonometry.emission.sound_power_intensity`. And it does not grade the
instrument: :math:`\delta_{pI0}` is a property of the probe-spacer-analyser
chain, classified against IEC 61043:1993 Table 2 by
:func:`phonometry.emission.intensity_class_compliance`.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

from .._internal.validation import (
    require_finite_fields,
    require_ranks,
    require_same_length,
)
from ._shared import SoundPowerWarning, _a_weighting_corrections
from .intensity import (
    _F4_NON_POSITIVE,
    _F4_REMAINDER_NON_POSITIVE,
    TEMPORAL_VARIABILITY_LIMIT,
    _coefficient_of_variation,
    dynamic_capability_index,
    field_indicators,
    temporal_variability_indicator,
)

__all__ = [
    "ActionCode",
    "DiscretePointIntensityResult",
    "PartialPowerConcentration",
    "determination_standard_deviation",
    "error_factor",
    "normal_intensity_from_levels",
    "partial_power_concentration",
    "position_count_factor",
    "sound_power_intensity_points",
]

_P0 = 1.0e-12  #: Reference sound power, in watts (clause 3.6.3).
_I0 = 1.0e-12  #: Reference sound intensity, in W/m^2 (clause 3.5).

#: The three accuracy grades of ISO 9614-1, in the library's shared spelling.
#: ``"precision"`` is grade 1, ``"engineering"`` grade 2 and ``"survey"``
#: grade 3 (Table 1 prints them as precision, "peritaje" and "control"). The
#: two-grade vocabulary of :mod:`phonometry.emission._shared` cannot be reused
#: here because this is the only determination in the library that recognises
#: all three.
DeterminationGrade = Literal["precision", "engineering", "survey"]

#: Octave or one-third-octave band centres, selecting the column of Tables B.2
#: and 2 a frequency is looked up in.
BandType = Literal["octave", "third"]

_GRADES: tuple[str, ...] = ("precision", "engineering", "survey")

#: Deviation error factor K, in dB (Table 1, printed p. 11). Grades 1 and 2
#: share 10 dB; grade 3 is allowed 7 dB, so its criterion 1 is the looser one.
_K: dict[str, float] = {"precision": 10.0, "engineering": 10.0, "survey": 7.0}

#: Figure B.1's unnumbered third gate: the excess of the signed over the
#: unsigned surface indicator, in dB. Above it the surface carries too much
#: inward flow for the determination to stand (Table B.3, second row).
_NEGATIVE_POWER_LIMIT = 3.0

#: Below this excess (in dB) the inward flow is small enough that the power may
#: be concentrated in a minority of segments, which is the entry condition of
#: the optional procedure of clause 8.3.2 and Table B.3's fourth row.
_CONCENTRATION_LIMIT = 1.0

#: Fewest measurement positions clause 8.2 asks for on any surface. Fewer only
#: warns: the count is a property of the plan, not of the arithmetic.
_MIN_POSITIONS = 10

#: Position count above which clause 8.2's two relaxations both apply (one
#: position per 2 m^2 where extraneous noise is significant, or 50 positions
#: spread over a surface larger than 50 m^2), so a density below one position
#: per square metre is no longer worth a warning.
_RELAXED_POSITIONS = 50

#: Fewest positions the Bessel-corrected (N - 1) spread of equations (A.1) and
#: (A.8) is defined from.
_MIN_VARIATION_OBSERVATIONS = 2

#: One printed row of Table B.2 or Table 2: the octave centre range it covers
#: (``None`` where the row has no octave counterpart), the one-third-octave
#: range, and the grade 1 and grade 2 figures. There is no third figure,
#: because there is no grade 3 column in either table.
_BandRow = tuple[tuple[float, float] | None, tuple[float, float], float, float]

#: ISO 9614-1:1993 Table B.2, "Valores para el factor C" (printed p. 25), as
#: the four per-band rows are printed. The 6 300 Hz row has no octave
#: counterpart, and the whole of grade 3 in this table is the A-weighted 8 of
#: :data:`_C_A_WEIGHTED`.
_TABLE_B2: tuple[_BandRow, ...] = (
    ((63.0, 125.0), (50.0, 160.0), 19.0, 11.0),
    ((250.0, 500.0), (200.0, 630.0), 29.0, 19.0),
    ((1000.0, 4000.0), (800.0, 5000.0), 57.0, 29.0),
    (None, (6300.0, 6300.0), 19.0, 14.0),
)

#: The A-weighted row of Table B.2, whose footnote fixes the summed range at
#: 63 Hz to 4 kHz (octave) or 50 Hz to 6,3 kHz (one-third octave). Grade 3 is
#: the only grade in it.
_C_A_WEIGHTED: dict[str, float] = {"survey": 8.0}

#: ISO 9614-1:1993 Table 2, "Incertidumbre en la determinación de los niveles
#: de potencia sonora" (printed p. 13), in the same shape as
#: :data:`_TABLE_B2`: the standard deviation s in dB for grades 1 and 2, with
#: grade 3 again carrying only the A-weighted figure of
#: :data:`_S_A_WEIGHTED`. Footnote 1 places the true level within +/- 2s of the
#: measured one with 95 % confidence.
_TABLE_2_S: tuple[_BandRow, ...] = (
    ((63.0, 125.0), (50.0, 160.0), 2.0, 3.0),
    ((250.0, 500.0), (200.0, 630.0), 1.5, 2.0),
    ((1000.0, 4000.0), (800.0, 5000.0), 1.0, 1.5),
    (None, (6300.0, 6300.0), 2.0, 2.5),
)

#: The A-weighted row of Table 2. Footnote 3 calls the figure tentative, "in
#: view of the wide variety of equipment the standards may be applied to".
_S_A_WEIGHTED: dict[str, float] = {"survey": 4.0}

#: ISO 9614-1:1993 Table B.1, "Factor de error Delta" (printed p. 25). One row
#: covers all bands and holds grades 1 and 2; the other is A-weighted and holds
#: grade 3. Delta feeds equation (B.4) and nothing else.
_TABLE_B1_ALL_BANDS: dict[str, float] = {"precision": 0.20, "engineering": 0.29}
_TABLE_B1_A_WEIGHTED: dict[str, float] = {"survey": 0.60}

#: Nominal octave mid-band centres Tables B.2 and 2 tabulate, in Hz.
_OCTAVE_BANDS: tuple[float, ...] = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)

#: Nominal one-third-octave mid-band centres Tables B.2 and 2 tabulate, in Hz.
_THIRD_BANDS: tuple[float, ...] = (
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
)

#: Largest relative deviation accepted between a supplied band centre and the
#: nominal label it designates, as in
#: :mod:`phonometry.emission.intensity_compliance`: the exact base-ten centres
#: of IEC 61260-1 Annex A sit within 0,8 % of their nominal labels, so 3 %
#: takes both conventions and still refuses a frequency that is no band at all.
_BAND_MATCH_TOLERANCE = 0.03


class ActionCode(Enum):
    """An action of ISO 9614-1:1993 Table B.3, by its printed code letter.

    Table B.3 answers the question the criteria leave open: the surface or the
    position set does not qualify, so what is to be *changed*. Its five actions
    are lettered a to e, and Figure B.1 routes each failing gate to one or two
    of them. The letter is the member value, because that is what a test report
    cites; :attr:`criterion` and :attr:`action` carry the row it was read from
    and what it asks for.
    """

    ADJUST_MEASUREMENT_DISTANCE = "a"  #: Move the surface, in or out.
    SHIELD_OR_REDUCE_REFLECTIONS = "b"  #: Shield the surface, damp reflections.
    INCREASE_POSITION_DENSITY = "c"  #: More positions, uniformly.
    INCREASE_DISTANCE_OR_POSITIONS = "d"  #: Move the surface out, or add points.
    REDUCE_TEMPORAL_VARIABILITY = "e"  #: Steady the field, or average longer.

    @property
    def criterion(self) -> str:
        """The Table B.3 criterion row that calls for this action."""
        return _ACTION_CRITERION[self]

    @property
    def action(self) -> str:
        """What Table B.3 asks the operator to change, in one sentence."""
        return _ACTION_TEXT[self]


#: The criterion column of Table B.3, one entry per action code.
_ACTION_CRITERION: dict[ActionCode, str] = {
    ActionCode.REDUCE_TEMPORAL_VARIABILITY: "F1 > 0,6",
    ActionCode.ADJUST_MEASUREMENT_DISTANCE: "F2 > Ld or (F3 - F2) > 3 dB",
    ActionCode.SHIELD_OR_REDUCE_REFLECTIONS: "F2 > Ld or (F3 - F2) > 3 dB",
    ActionCode.INCREASE_POSITION_DENSITY: (
        "criterion 2 not satisfied and 1 dB <= (F3 - F2) <= 3 dB"
    ),
    ActionCode.INCREASE_DISTANCE_OR_POSITIONS: (
        "criterion 2 not satisfied, (F3 - F2) <= 1 dB and the procedure of "
        "8.3.2 either fails or is not selected"
    ),
}

#: The action column of Table B.3, one entry per action code.
_ACTION_TEXT: dict[ActionCode, str] = {
    ActionCode.REDUCE_TEMPORAL_VARIABILITY: (
        "Take action to reduce the temporal variability of the extraneous "
        "intensity, or measure during periods of lower variability, or "
        "increase the measurement period at each position (where applicable)."
    ),
    ActionCode.ADJUST_MEASUREMENT_DISTANCE: (
        "In the presence of significant extraneous noise and/or strong "
        "reverberation, reduce the mean distance from the measurement surface "
        "to the source to a minimum mean value of 0,25 m. In the absence of "
        "significant extraneous noise and/or strong reverberation, increase "
        "the measurement distance to 1 m."
    ),
    ActionCode.SHIELD_OR_REDUCE_REFLECTIONS: (
        "Shield the measurement surface from extraneous sources, or take "
        "action to reduce reflections back towards the source."
    ),
    ActionCode.INCREASE_POSITION_DENSITY: (
        "Increase the density of measurement positions uniformly, so as to "
        "satisfy criterion 2."
    ),
    ActionCode.INCREASE_DISTANCE_OR_POSITIONS: (
        "Increase the mean distance from the measurement surface to the "
        "source, keeping the same number of measurement positions, or "
        "increase the number of measurement positions over the same surface."
    ),
}


def _check_grade(grade: str) -> DeterminationGrade:
    """Validate one of the three accuracy grades of ISO 9614-1 Table 1."""
    if grade not in _GRADES:
        msg = (
            "'grade' must be 'precision' (grade 1), 'engineering' (grade 2) or "
            f"'survey' (grade 3), the three grades of ISO 9614-1 Table 1; got "
            f"{grade!r}."
        )
        raise ValueError(msg)
    return cast(DeterminationGrade, grade)


def _check_band_type(band_type: str) -> BandType:
    """Validate the column group of Tables B.2 and 2 a frequency is read in."""
    if band_type not in ("octave", "third"):
        msg = f"'band_type' must be 'octave' or 'third'; got {band_type!r}."
        raise ValueError(msg)
    return cast(BandType, band_type)


def _dynamic_capability(
    residual_index: np.ndarray, grade: DeterminationGrade
) -> np.ndarray:
    r"""Per-band :math:`L_\mathrm{d} = \delta_{pI0} - K` at one grade (Eq. (10)).

    Goes through :func:`~phonometry.emission.dynamic_capability_index` rather
    than subtracting ``K`` here, so the one definition of the dynamic capability
    in the library is the one every criterion 1 in it is judged against.
    """
    return np.asarray(
        [dynamic_capability_index(float(value), _K[grade]) for value in residual_index],
        dtype=np.float64,
    )


def _nominal_band(frequency: float, band_type: BandType) -> float:
    """The tabulated nominal centre a supplied frequency designates, in Hz.

    :raises ValueError: If ``frequency`` is not finite and positive, or no
        tabulated centre of ``band_type`` lies within
        :data:`_BAND_MATCH_TOLERANCE` of it.
    """
    if not math.isfinite(frequency) or frequency <= 0.0:
        msg = "Band centre frequencies must be finite and positive."
        raise ValueError(msg)
    bands = _OCTAVE_BANDS if band_type == "octave" else _THIRD_BANDS
    nearest = min(bands, key=lambda f: abs(math.log(f / frequency)))
    if abs(nearest / frequency - 1.0) > _BAND_MATCH_TOLERANCE:
        low, high = bands[0], bands[-1]
        msg = (
            f"{frequency:g} Hz is not a band of ISO 9614-1 Tables B.2 and 2; "
            f"the {band_type} column covers {low:g} Hz to {high:g} Hz."
        )
        raise ValueError(msg)
    return nearest


def _row_for(
    table: tuple[_BandRow, ...], frequency: float, band_type: BandType
) -> tuple[float, float]:
    """The (grade 1, grade 2) pair of the table row covering one band centre."""
    nominal = _nominal_band(frequency, band_type)
    for octave_range, third_range, grade1, grade2 in table:
        span = octave_range if band_type == "octave" else third_range
        if span is None:
            continue
        if span[0] <= nominal <= span[1]:
            return grade1, grade2
    msg = (
        f"ISO 9614-1 tabulates no value at {nominal:g} Hz ({band_type} bands); "
        "the 6 300 Hz row of Tables B.2 and 2 is printed for one-third-octave "
        "bands only."
    )
    raise ValueError(msg)


def _per_band_or_raise(
    grade: DeterminationGrade, table_name: str, a_weighted_value: float
) -> None:
    """Refuse a per-band lookup at grade 3, where the table has no such column.

    Grades 1 and 2 fill the per-band columns of Tables B.2, B.1 and 2 and leave
    the A-weighted row blank; grade 3 does the reverse. Returning the
    A-weighted figure for a band would be inventing a cell the standard did not
    print, and the number is plausible enough that nothing downstream would
    notice.
    """
    if grade == "survey":
        msg = (
            f"ISO 9614-1 {table_name} has no per-band value for the survey "
            f"grade (grade 3): the whole of grade 3 in that table is the "
            f"A-weighted {a_weighted_value:g}. A grade-3 determination under "
            "this part of ISO 9614 is an A-weighted one."
        )
        raise ValueError(msg)


def position_count_factor(
    grade: DeterminationGrade,
    frequency: float | None = None,
    *,
    band_type: BandType = "third",
) -> float:
    r"""Criterion-2 factor ``C`` of ISO 9614-1:1993 Table B.2.

    Criterion 2 (equation (B.2)) asks for :math:`N > C F_4^2` measurement
    positions, so ``C`` converts the field non-uniformity into a position
    count. It depends on the band and on the grade claimed: a surface needs
    roughly twice as many positions for precision as for engineering grade.

    The table's two halves do not overlap. Grades 1 and 2 have a value in every
    band and none for the A-weighted sum; grade 3 has the single A-weighted 8
    and no band column at all. So ``frequency`` selects which half is being
    asked for, and asking the wrong half raises.

    :param grade: ``'precision'`` (grade 1), ``'engineering'`` (grade 2) or
        ``'survey'`` (grade 3).
    :param frequency: Nominal mid-band centre in Hz, as a Table B.2 label or
        the exact base-ten centre behind it. ``None`` asks for the A-weighted
        row instead, whose footnote fixes the summed range at 63 Hz to 4 kHz
        (octave) or 50 Hz to 6,3 kHz (one-third octave).
    :param band_type: ``'octave'`` or ``'third'``, selecting the frequency
        column; ignored when ``frequency`` is ``None``.
    :return: ``C``, dimensionless.
    :raises ValueError: If ``grade`` or ``band_type`` is unknown; if a per-band
        value is asked for at the survey grade, or an A-weighted one at the
        precision or engineering grade; or if ``frequency`` is not a tabulated
        band.
    """
    grade = _check_grade(grade)
    if frequency is None:
        value = _C_A_WEIGHTED.get(grade)
        if value is None:
            msg = (
                "ISO 9614-1 Table B.2 prints an A-weighted C only for the "
                f"survey grade (grade 3, C = 8); the {grade} grade is "
                "tabulated band by band, so pass a 'frequency'."
            )
            raise ValueError(msg)
        return value
    _check_band_type(band_type)
    _per_band_or_raise(grade, "Table B.2", _C_A_WEIGHTED["survey"])
    grade1, grade2 = _row_for(_TABLE_B2, float(frequency), band_type)
    return grade1 if grade == "precision" else grade2


def determination_standard_deviation(
    grade: DeterminationGrade,
    frequency: float | None = None,
    *,
    band_type: BandType = "third",
) -> float:
    r"""Standard deviation ``s`` of the determination, ISO 9614-1:1993 Table 2.

    Footnote 1 of the table states what ``s`` is for: the true sound power
    level is expected to lie within :math:`\pm 2s` of the measured one with
    95 % confidence, so twice this figure is the expanded uncertainty of a
    qualified determination.

    The table has the same shape as Table B.2 and the same asymmetry, and this
    is where the asymmetry is load bearing rather than merely odd: the standard
    defines no per-band uncertainty for grade 3 at all, only an A-weighted one,
    which is why grade 3 under this part of ISO 9614 is an A-weighted
    determination. Footnote 3 calls the grade-3 figure tentative.

    :param grade: ``'precision'`` (grade 1), ``'engineering'`` (grade 2) or
        ``'survey'`` (grade 3).
    :param frequency: Nominal mid-band centre in Hz, or ``None`` for the
        A-weighted row.
    :param band_type: ``'octave'`` or ``'third'``; ignored when ``frequency``
        is ``None``.
    :return: ``s`` in decibels.
    :raises ValueError: On the same mismatches
        :func:`position_count_factor` refuses.
    """
    grade = _check_grade(grade)
    if frequency is None:
        value = _S_A_WEIGHTED.get(grade)
        if value is None:
            msg = (
                "ISO 9614-1 Table 2 prints an A-weighted standard deviation "
                f"only for the survey grade (grade 3, s = 4 dB); the {grade} "
                "grade is tabulated band by band, so pass a 'frequency'."
            )
            raise ValueError(msg)
        return value
    _check_band_type(band_type)
    _per_band_or_raise(grade, "Table 2", _S_A_WEIGHTED["survey"])
    grade1, grade2 = _row_for(_TABLE_2_S, float(frequency), band_type)
    return grade1 if grade == "precision" else grade2


def error_factor(grade: DeterminationGrade, *, a_weighted: bool = False) -> float:
    r"""Error factor :math:`\Delta` of ISO 9614-1:1993 Table B.1.

    :math:`\Delta` is the sampling error the optional procedure of B.1.3 is
    allowed to leave on the determination, and equation (B.4) is its only
    consumer. Table B.1 prints one row for all bands, holding 0,20 at grade 1
    and 0,29 at grade 2, and one A-weighted row holding 0,60 at grade 3; the
    remaining four cells are blank in the print.

    :param grade: ``'precision'`` (grade 1), ``'engineering'`` (grade 2) or
        ``'survey'`` (grade 3).
    :param a_weighted: ``True`` reads the A-weighted row (grade 3 only),
        ``False`` (default) the all-bands row (grades 1 and 2 only).
    :return: :math:`\Delta`, dimensionless.
    :raises ValueError: If ``grade`` is unknown, or the row asked for is blank
        at that grade.
    """
    grade = _check_grade(grade)
    table = _TABLE_B1_A_WEIGHTED if a_weighted else _TABLE_B1_ALL_BANDS
    value = table.get(grade)
    if value is None:
        row = "A-weighted" if a_weighted else "all bands"
        other = "all bands" if a_weighted else "A-weighted"
        msg = (
            f"ISO 9614-1 Table B.1 leaves the {row!r} cell of the {grade} "
            f"grade blank; that grade carries its error factor on the "
            f"{other!r} row instead."
        )
        raise ValueError(msg)
    return value


def normal_intensity_from_levels(
    levels: ArrayLike, *, negative: ArrayLike = False
) -> np.ndarray:
    r"""Signed normal intensity from printed intensity levels (clause 9.1).

    ISO 9614-1 does not print a signed level. A normal intensity level is
    written ``XX dB`` when the flow through the segment is outward and
    ``(-) XX dB`` when it is inward, with ``XX`` a positive number in both
    cases (clause 3.5, and the two unnumbered equations of clauses 9.1 and
    A.2.3):

    .. math::

       I_{\mathrm{n}i} = I_0 \times 10^{XX/10}, \qquad
       I_{\mathrm{n}i} = -I_0 \times 10^{XX/10}, \qquad
       I_0 = 10^{-12}~\text{W/m}^2

    So the sign is not in the number, and a caller reading a printed table has
    to carry it separately: ``negative`` is the ``(-)`` of the print, and it
    broadcasts against ``levels``, which is what lets one position of a
    measurement surface flow inward while the rest flow outward. Negative
    partial power is normal and is what :math:`F_3` measures; it is not an
    error, and only the *sum* going negative puts a band outside the method
    (clause 9.2).

    :param levels: Normal intensity levels ``XX`` in decibels, of any shape.
        Each is the level of the magnitude, so it is positive for a level above
        the reference intensity and negative for one below it; the direction of
        flow is ``negative``, never the sign of this number.
    :param negative: ``True`` where the printed level carried the ``(-)``
        prefix, i.e. where the flow is inward. A single bool applies to every
        level; an array broadcasts against ``levels``.
    :return: The signed normal intensity in W/m^2, of the shape of ``levels``.
        ``negative`` is broadcast onto that shape and never widens it: one flag
        per level, or one flag for all of them, and a mask of any other shape
        is refused rather than returning more intensities than levels went in.
    :raises ValueError: If a level is not finite, or ``negative`` cannot be
        broadcast against ``levels``.
    """
    values = np.asarray(levels, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        msg = "'levels' must contain only finite intensity levels in decibels."
        raise ValueError(msg)
    inward = np.asarray(negative, dtype=bool)
    try:
        sign = np.where(np.broadcast_to(inward, values.shape), -1.0, 1.0)
    except ValueError as exc:
        msg = (
            f"'negative' of shape {inward.shape} does not broadcast against "
            f"'levels' of shape {values.shape}; pass one flag, or one per level."
        )
        raise ValueError(msg) from exc
    return np.asarray(sign * _I0 * 10.0 ** (values / 10.0), dtype=np.float64)


@dataclass(frozen=True)
class PartialPowerConcentration:
    r"""Outcome of the optional procedure of ISO 9614-1:1993 clause 8.3.2/B.1.3.

    The segments carrying most of the sound power, and how many new positions
    equation (B.4) asks to be spread over them.

    :ivar positions: ``N``, the number of positions on the whole surface.
    :ivar subset_positions: :math:`N_\alpha`, the segments in the selected
        subset. B.1.3 requires this to be fewer than half of ``N``.
    :ivar subset_area: :math:`S_\alpha`, the total area of the subset, m^2; the
        new positions are distributed over it in proportion to segment area.
    :ivar power_fraction: :math:`\alpha`, the fraction of the total sound power
        passing through the subset, always above 0,5.
    :ivar subset_nonuniformity: :math:`F_4(\alpha)`, the field non-uniformity
        of the subset alone (equations (A.8), (A.9)).
    :ivar remainder_nonuniformity: :math:`F_4(1-\alpha)`, that of the remaining
        segments.
    :ivar error_factor: :math:`\Delta` for the requested grade (Table B.1).
    :ivar subset_error_factor: :math:`\Delta_\alpha`, the share of that error
        budget the subset may spend, after the remainder has taken its own.
    :ivar additional_positions: :math:`N^*`, the smallest whole number of new
        positions satisfying equation (B.4).
    """

    positions: int
    subset_positions: int
    subset_area: float
    power_fraction: float
    subset_nonuniformity: float
    remainder_nonuniformity: float
    error_factor: float
    subset_error_factor: float
    additional_positions: int

    def __post_init__(self) -> None:
        r"""Reject a concentration whose figures are not finite.

        Every field is read against the others by whoever plans the extra
        measurements: :math:`N^*` positions are spread over :math:`S_\alpha`
        in proportion to segment area, and the two non-uniformities are
        reported beside them. A NaN reaching that plan is a measurement
        campaign sized from a number that is not one.

        :raises ValueError: if any of the real-valued fields is not finite.
        """
        require_finite_fields(
            self,
            "subset_area",
            "power_fraction",
            "subset_nonuniformity",
            "remainder_nonuniformity",
            "error_factor",
            "subset_error_factor",
        )


def _positive_partial_powers(
    normal_intensity: ArrayLike, areas: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """One band of signed normal intensities and its segment areas, validated."""
    i_n = np.atleast_1d(np.asarray(normal_intensity, dtype=np.float64))
    seg = np.atleast_1d(np.asarray(areas, dtype=np.float64))
    if i_n.ndim != 1 or seg.ndim != 1:
        msg = (
            "partial_power_concentration works on one frequency band: pass 1D "
            "'normal_intensity' and 'areas', one entry per position."
        )
        raise ValueError(msg)
    _validate_positions(i_n, seg, intensity_name="normal_intensity")
    return i_n, seg


def partial_power_concentration(
    normal_intensity: ArrayLike,
    areas: ArrayLike,
    *,
    grade: DeterminationGrade = "engineering",
) -> PartialPowerConcentration:
    r"""Positive partial power concentration and the new positions it needs.

    The optional procedure of clause 8.3.2, computed as B.1.3 specifies. It
    applies to a band in which criterion 1 holds, criterion 2 does not and
    :math:`F_3 - F_2 \le 1` dB: little power flows inward, so most of it may be
    leaving through a minority of the segments, and densifying only those
    segments qualifies the surface for far less work than densifying all of it.

    The positive partial powers are ranked in decreasing order and the top
    segments are taken until more than half the total sound power has been
    accounted for. That subset is :math:`N_\alpha` segments of total area
    :math:`S_\alpha` carrying the fraction :math:`\alpha > 0,5`, and B.1.3
    requires :math:`N_\alpha` to be fewer than half of ``N``. The field
    non-uniformity is then evaluated separately over the subset and over the
    remainder (equations (A.8) and (A.9)), and:

    .. math::

       N^* \ge 4 \left[ \frac{F_4(\alpha)}{\Delta_\alpha} \right]^2 \tag{Eq. B.4}

       \Delta_\alpha = \frac{1}{\alpha} \left[ \Delta - (1 - \alpha)
       \frac{2}{\sqrt{N_{1-\alpha}}} F_4(1 - \alpha) \right], \qquad
       N_{1-\alpha} = N - N_\alpha

    :math:`\Delta_\alpha` is the share of the Table B.1 error budget left for
    the subset once the remainder, measured at its existing density, has taken
    its own; a remainder too non-uniform to leave anything over exhausts the
    budget, and the procedure cannot help.

    :param normal_intensity: Signed normal intensity :math:`I_{\mathrm{n}i}` at
        each position of one frequency band, in W/m^2 (1D).
    :param areas: Segment areas :math:`S_i` in m^2, one per position (1D).
    :param grade: The grade whose Table B.1 :math:`\Delta` is spent:
        ``'precision'`` (0,20), ``'engineering'`` (0,29) or ``'survey'``
        (0,60, A-weighted).
    :return: A :class:`PartialPowerConcentration`.
    :raises ValueError: If the positions and areas disagree in length, an area
        is not positive and finite, an intensity is not finite, the total sound
        power is not positive (clause 9.2 puts the band outside the method), no
        subset satisfies the two conditions of B.1.3, the subset is a single
        segment (equation (A.8) has no spread over one position, so equation
        (B.4) is undefined), the algebraic mean normal intensity over the
        remainder is not positive (A.2.3, which happens when the subset takes
        all the outward flow), or the remainder leaves no error budget for the
        subset. In the last four cases the selective modification cannot be
        carried out, and clause 8.3.2 then asks for the appropriate alternative
        actions in accordance with clause B.2 and Table B.3. Which row of that
        table applies is not settled here: its two lower rows are conditioned
        on criterion 2 and on :math:`F_3 - F_2`, neither of which this function
        is given.
    """
    grade = _check_grade(grade)
    i_n, seg = _positive_partial_powers(normal_intensity, areas)
    n_positions = int(i_n.size)
    if n_positions < _MIN_VARIATION_OBSERVATIONS:
        msg = (
            "At least two measurement positions are required to split a "
            "surface into a subset and a remainder."
        )
        raise ValueError(msg)

    partial = i_n * seg
    total = float(np.sum(partial))
    if not total > 0.0:
        msg = (
            "The total sound power of this band is not positive, so ISO "
            "9614-1 is not applicable to it (clause 9.2) and the optional "
            "procedure of 8.3.2 has nothing to concentrate."
        )
        raise ValueError(msg)

    # B.1.3: rank the positive partial powers in decreasing order and take the
    # top ones until more than half the total sound power has passed through
    # them. The ranking is over the positive powers alone, so a segment with
    # inward flow can never join the subset.
    positive = np.flatnonzero(partial > 0.0)
    order = positive[np.argsort(-partial[positive], kind="stable")]
    cumulative = np.cumsum(partial[order])
    reached = np.flatnonzero(cumulative > 0.5 * total)
    half = 0.5 * n_positions
    subset_size = int(reached[0]) + 1 if reached.size else n_positions + 1
    if subset_size >= half:
        # This is the one case B.1.3 answers itself: "si no existe un
        # subconjunto de elementos que satisfaga las anteriores condiciones,
        # tomar las acciones alternativas apropiadas ... de acuerdo a la tabla
        # B.3". It names the table and no row of it.
        msg = (
            f"No subset of fewer than half the {n_positions} segments carries "
            "more than half the sound power, so the concentration test of ISO "
            "9614-1 clause 8.3.2 fails; take the appropriate alternative "
            "actions of Table B.3 (B.1.3)."
        )
        raise ValueError(msg)
    # B.1.3 bounds N_alpha from above and never from below, so one segment
    # carrying more than half the power is legal for any N of three or more,
    # and it is the archetypal concentrated source. Equation (B.4) is still
    # undefined there: F4(alpha) is the Bessel-corrected spread of (A.8), which
    # divides by N_alpha - 1. Refusing it by name beats dividing by zero and
    # rounding the NaN up, which happens inside the constructor call below,
    # where __post_init__ never gets to speak.
    if subset_size < _MIN_VARIATION_OBSERVATIONS:
        msg = (
            "The sound power is concentrated in a single segment, so the top "
            "subset taken here is one segment and equation (A.8) has no "
            "spread to measure over it, leaving equation (B.4) without an "
            "F4(alpha). ISO 9614-1 B.1.3 asks only for a top subset carrying "
            "more than half the total sound power and numbering fewer than "
            "half of the N segments, and does not require the smallest such "
            "subset, so a larger one may still be admissible; where this "
            "selective modification cannot be carried out, clause 8.3.2 asks "
            "for the appropriate alternative actions in accordance with "
            "clause B.2 and Table B.3."
        )
        raise ValueError(msg)

    subset = order[:subset_size]
    remainder = np.setdiff1d(np.arange(n_positions), subset, assume_unique=False)
    alpha = float(cumulative[subset_size - 1] / total)
    f4_subset = _coefficient_of_variation(
        i_n[subset], non_positive_message=_F4_NON_POSITIVE
    )
    f4_remainder = _coefficient_of_variation(
        i_n[remainder], non_positive_message=_F4_REMAINDER_NON_POSITIVE
    )

    # Table B.1 keeps the survey grade's Delta on its A-weighted row and the
    # other two on the all-bands row, so the row follows from the grade.
    delta = error_factor(grade, a_weighted=grade == "survey")
    n_remainder = int(remainder.size)
    delta_alpha = (
        delta - (1.0 - alpha) * (2.0 / math.sqrt(n_remainder)) * f4_remainder
    ) / alpha
    if not delta_alpha > 0.0:
        msg = (
            f"The remaining {n_remainder} segments exhaust the ISO 9614-1 "
            f"Table B.1 error factor on their own (Delta_alpha = "
            f"{delta_alpha:.3g}), so no number of new positions on the subset "
            "can qualify the surface; this selective modification cannot be "
            "carried out, and clause 8.3.2 asks for the appropriate "
            "alternative actions in accordance with clause B.2 and Table B.3."
        )
        raise ValueError(msg)

    return PartialPowerConcentration(
        positions=n_positions,
        subset_positions=subset_size,
        subset_area=float(np.sum(seg[subset])),
        power_fraction=alpha,
        subset_nonuniformity=float(f4_subset),
        remainder_nonuniformity=float(f4_remainder),
        error_factor=delta,
        subset_error_factor=float(delta_alpha),
        additional_positions=int(math.ceil(4.0 * (f4_subset / delta_alpha) ** 2)),
    )


@dataclass(frozen=True)
class DiscretePointIntensityResult:
    r"""Result of an ISO 9614-1:1993 discrete-point sound-power determination.

    ``partial_power`` is the signed :math:`P_i = I_{\mathrm{n}i} S_i` per
    position and band (equation (11)), ``sound_power`` its signed band total
    and ``sound_power_level`` the level of that total (equation (12)), ``NaN``
    in a band whose total is not positive: ``not_applicable_band`` is ``True``
    there and clause 9.2 puts the band outside the method.

    ``f1`` to ``f4`` are the Annex A field indicators per band, ``None`` when
    the inputs they need were not supplied. ``f2``, ``f3`` and ``f4`` are
    ``NaN`` in a band whose algebraic mean normal intensity is not positive,
    which A.2.3 makes a failure of the test conditions in that band. ``f1``
    is not: equation (A.1) is the spread of the M short-time samples at one
    position over time, A.2.1 puts no positivity condition on it, and a band
    A.2.3 refuses still has a perfectly good temporal variability. Measured on
    such a band, ``f1`` comes back finite while the other three are ``NaN``. A.2.3's refusal is not
    ``not_applicable_band``, whose quantity is clause 9.2's area-weighted sum,
    and the determination warns about it separately, since a band can fail
    A.2.3 and still carry a finite level here. ``criterion_1``
    (:math:`L_\mathrm{d} > F_2`, equation (B.1)), ``negative_power_within_limit``
    (Figure B.1's unnumbered :math:`F_3 - F_2 \le 3` dB gate) and
    ``criterion_2`` (:math:`N > C F_4^2`, equation (B.2)) are the per-band
    verdicts, ``None`` where they could not be evaluated;
    ``minimum_positions`` is the :math:`C F_4^2` that criterion 2 compares
    ``positions`` against.

    ``achieved_grade`` is the per-band grade, one of ``'precision'``,
    ``'engineering'`` and ``'none'``. Grade 3 is never among them: Table B.2
    gives it no per-band ``C``, so criterion 2 has no per-band form there and
    grade 3 is reached, if at all, by ``achieved_grade_a`` on the A-weighted
    sum, whose own field non-uniformity is ``field_nonuniformity_a``
    (B.1.2, computed from the A-weighted band intensities of each position).

    ``confidence_interval`` is the pair
    :math:`10 \lg (1 \pm 2 F_4 / \sqrt{N})` of equation (B.3) per band, which
    clause 10.5 c) requires beside the level of any band that failed criterion
    2, and ``expanded_uncertainty`` the :math:`2s` of Table 2 footnote 1 read
    at the grade the band *achieved*, that being the grade clause 10.6 has a
    report state: ``NaN`` in a band that reached no grade, for which Table 2
    prints no ``s``, and ``None`` for the whole determination where
    ``achieved_grade`` could not be established either.
    ``sound_power_level_a`` omits the bands outside the method and, per clause
    10.5 b), those failing criteria 1 and/or 2, which
    ``a_weighting_omitted_bands`` flags.
    """

    frequencies: np.ndarray | None
    partial_power: np.ndarray
    sound_power: np.ndarray
    sound_power_level: np.ndarray
    not_applicable_band: np.ndarray
    f1: np.ndarray | None
    f2: np.ndarray | None
    f3: np.ndarray | None
    f4: np.ndarray | None
    dynamic_capability_index: np.ndarray | None
    criterion_1: np.ndarray | None
    negative_power_within_limit: np.ndarray | None
    criterion_2: np.ndarray | None
    minimum_positions: np.ndarray | None
    achieved_grade: np.ndarray | None
    confidence_interval: np.ndarray | None
    expanded_uncertainty: np.ndarray | None
    surface_area: float
    positions: int
    sound_power_level_a: float
    a_weighting_omitted_bands: np.ndarray | None
    field_nonuniformity_a: float
    achieved_grade_a: str | None
    grade: str

    def __post_init__(self) -> None:
        """Reject a determination whose per-band quantities disagree.

        Every column here is read against another: the figure draws the level
        spectrum against ``frequencies`` and hatches it with
        ``not_applicable_band``, :meth:`required_actions` walks the four gates
        of Figure B.1 band by band, and a reader tabulating the result puts the
        indicators, the verdicts and the confidence interval on one row with
        the level. A column of the wrong length either raises somewhere inside
        numpy about two shapes, naming no field, or is silently broadcast: a
        single-band verdict array decides every band at once, and nothing in
        the result then says that one band decided the rest.

        ``partial_power`` carries the positions on its first axis and the bands
        on its second, so its band axis is index 1, and ``confidence_interval``
        carries the two interval ends on its second axis, so it is checked on
        the first.

        ``surface_area`` must be finite: it is the sum of segment areas the
        determination already refused unless positive and finite, and it is
        what a report prints beside the boxed level. The band levels stay
        unpinned, ``NaN`` being clause 9.2's reading of a band the method does
        not apply to.

        :raises ValueError: if any per-band quantity disagrees with the rest,
            or ``surface_area`` is not finite.
        """
        require_ranks(
            self,
            frequencies=1,
            partial_power=2,
            sound_power=1,
            sound_power_level=1,
            not_applicable_band=1,
            f1=1,
            f2=1,
            f3=1,
            f4=1,
            dynamic_capability_index=1,
            criterion_1=1,
            negative_power_within_limit=1,
            criterion_2=1,
            minimum_positions=1,
            achieved_grade=1,
            confidence_interval=2,
            expanded_uncertainty=1,
            a_weighting_omitted_bands=1,
        )
        require_same_length(
            self,
            "frequencies",
            ("partial_power", 1),
            "sound_power",
            "sound_power_level",
            "not_applicable_band",
            "f1",
            "f2",
            "f3",
            "f4",
            "dynamic_capability_index",
            "criterion_1",
            "negative_power_within_limit",
            "criterion_2",
            "minimum_positions",
            "achieved_grade",
            "confidence_interval",
            "expanded_uncertainty",
            "a_weighting_omitted_bands",
        )
        require_finite_fields(self, "surface_area")

    def required_actions(self) -> tuple[tuple[ActionCode, ...], ...]:
        r"""The Table B.3 actions each band calls for, in Figure B.1's order.

        Figure B.1 gates the determination on four questions and sends the
        first failing one to an action box, from which the flow returns to the
        next measurement rather than to the gate below. So a band gets one
        action set, not a list of everything that went wrong: the tuple is
        empty for a band that passed every gate it could be judged on, and
        holds one or two codes otherwise. Two codes mean the standard offers a
        choice, which is how Table B.3's second row prints them ("a **o** b").

        A gate whose inputs are absent is skipped rather than failed. ``F1``
        comes from the initial test and is legitimately missing; criterion 2
        needs ``frequencies`` for the Table B.2 lookup, and has no per-band
        form at all at the survey grade. Where criterion 2 was not evaluated,
        the fourth gate and its actions (c) and (d) cannot be reached, and a
        band that clears the first three is reported as clear.

        Action (d) is the one Table B.3 conditions on the operator: it applies
        when criterion 2 fails with :math:`F_3 - F_2 \le 1` dB and the optional
        procedure of clause 8.3.2 "either fails or is not selected". Not
        selecting it is the default, so it is reported here; see
        :func:`partial_power_concentration` for the alternative.

        :return: One tuple of :class:`ActionCode` per band.
        :raises ValueError: If the determination was never qualified, i.e. it
            carries no ``criterion_1`` because ``pressure_levels`` and
            ``pressure_residual_index`` were not supplied.
        """
        if self.criterion_1 is None:
            msg = (
                "This determination was not qualified, so ISO 9614-1 Table B.3 "
                "has nothing to act on; call sound_power_intensity_points with "
                "'pressure_levels' and 'pressure_residual_index'."
            )
            raise ValueError(msg)
        return tuple(
            _band_actions(
                f1=None if self.f1 is None else float(self.f1[band]),
                criterion_1=bool(self.criterion_1[band]),
                inward_flow_ok=self.negative_power_within_limit is None
                or bool(self.negative_power_within_limit[band]),
                criterion_2=(
                    None if self.criterion_2 is None else bool(self.criterion_2[band])
                ),
                low_inward_flow=(
                    self.f2 is not None
                    and self.f3 is not None
                    and float(self.f3[band] - self.f2[band]) <= _CONCENTRATION_LIMIT
                ),
            )
            for band in range(int(np.asarray(self.sound_power_level).size))
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the LW spectrum; bands outside the method are hatched.

        Draws the same figure as the scanning determinations of ISO 9614-2 and
        ISO 9614-3, because the quantity is the same one: the band sound power
        level, with the bands of non-positive net power (clause 9.2) hatched as
        unusable and the A-weighted total in the title. The field indicators
        behind the qualification have their own figure, on the
        :class:`~phonometry.emission.intensity.FieldIndicators` that
        :func:`~phonometry.emission.field_indicators` returns.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the band :meth:`~matplotlib.axes.Axes.bar`.
        """
        from .._i18n import check_language
        from .._plot.emission import plot_sound_power

        check_language(language)
        return plot_sound_power(self, ax=ax, language=language, **kwargs)


def _band_actions(
    *,
    f1: float | None,
    criterion_1: bool,
    inward_flow_ok: bool,
    criterion_2: bool | None,
    low_inward_flow: bool,
) -> tuple[ActionCode, ...]:
    """One band's Table B.3 actions, walking Figure B.1 top to bottom.

    The gates are exclusive, because each action box of the figure returns to
    the next measurement rather than to the gate below it: the band leaves with
    the actions of the first gate it failed, and with none at all where it
    passed every gate that could be put to it.
    """
    actions: list[ActionCode] = []
    # ``not (f1 <= limit)`` rather than ``f1 > limit`` so a NaN F1, which every
    # comparison answers False, is treated as a field that failed to qualify
    # rather than as one that passed.
    if f1 is not None and not f1 <= TEMPORAL_VARIABILITY_LIMIT:
        actions.append(ActionCode.REDUCE_TEMPORAL_VARIABILITY)
    elif not criterion_1 or not inward_flow_ok:
        actions.extend(
            (
                ActionCode.ADJUST_MEASUREMENT_DISTANCE,
                ActionCode.SHIELD_OR_REDUCE_REFLECTIONS,
            )
        )
    elif criterion_2 is not None and not criterion_2:
        # Table B.3's last two rows split on the same 1 dB: above it the inward
        # flow is what stops criterion 2 being met by density alone (action c),
        # below it the power may be concentrated, and action d is what remains
        # once the optional procedure of 8.3.2 is not taken.
        actions.append(
            ActionCode.INCREASE_DISTANCE_OR_POSITIONS
            if low_inward_flow
            else ActionCode.INCREASE_POSITION_DENSITY
        )
    return tuple(actions)


def _validate_positions(
    intensity: np.ndarray, areas: np.ndarray, *, intensity_name: str
) -> None:
    """Reject a position set whose intensities or segment areas are unusable."""
    if areas.size == 0:
        msg = "At least one measurement position is required."
        raise ValueError(msg)
    if intensity.shape[0] != areas.shape[0]:
        msg = (
            f"'{intensity_name}' first axis ({intensity.shape[0]}) must match "
            f"the number of segment 'areas' ({areas.shape[0]}): ISO 9614-1 "
            "associates one segment with each measurement position (3.8)."
        )
        raise ValueError(msg)
    # NaN beside the bound rather than folded into it: NaN compares False
    # against every bound, so the positivity test alone would pass it through
    # to a measurement surface whose area is not a number.
    if not np.all(np.isfinite(areas)):
        msg = "All segment 'areas' must be finite."
        raise ValueError(msg)
    if np.any(areas <= 0.0):
        msg = "All segment 'areas' must be positive."
        raise ValueError(msg)
    if not np.all(np.isfinite(intensity)):
        msg = f"'{intensity_name}' must contain only finite values."
        raise ValueError(msg)


def _position_grid(
    normal_intensity: ArrayLike, areas: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """The measured intensities as ``(positions, bands)``, with their areas.

    The rank of the input as supplied is what tells one band at ``N`` positions
    from ``N`` bands at one position, so it is read before the array is widened:
    a 1-D input is unambiguously the first, and reading the rank afterwards
    would take a genuine one-position row for a column of positions whenever
    the two counts happen to agree. A pair that cannot be a survey at all is
    refused here, before any of it is measured against the standard.
    """
    raw = np.asarray(normal_intensity, dtype=np.float64)
    seg = np.atleast_1d(np.asarray(areas, dtype=np.float64))
    if raw.ndim not in (1, 2):
        msg = (
            "'normal_intensity' must be 1D (positions,) for a single band or "
            "2D (positions, bands)."
        )
        raise ValueError(msg)
    if seg.ndim != 1:
        msg = "'areas' must be a 1D array of segment areas."
        raise ValueError(msg)
    intensity = raw.reshape(-1, 1) if raw.ndim == 1 else raw
    _validate_positions(intensity, seg, intensity_name="normal_intensity")
    return intensity, seg


def _checked_position_band_grid(
    values: ArrayLike | None, name: str, n_positions: int, n_bands: int
) -> np.ndarray | None:
    """A per-position (per-band) input on the ``(positions, bands)`` grid.

    ``None`` for an input that was not supplied, which is how every optional
    array of this determination is carried through to the end. Anything
    supplied has to span the same grid as the intensities and be finite
    throughout, since it is a measured quantity of the same survey.
    """
    if values is None:
        return None
    raw = np.asarray(values, dtype=np.float64)
    grid = raw.reshape(-1, 1) if raw.ndim == 1 else np.atleast_2d(raw)
    if grid.shape != (n_positions, n_bands):
        msg = (
            f"'{name}' must have shape ({n_positions}, {n_bands}) matching "
            f"'normal_intensity'; got {grid.shape}."
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(grid)):
        msg = f"'{name}' must contain only finite values."
        raise ValueError(msg)
    return grid


def _checked_frequencies(
    frequencies: ArrayLike | None, n_bands: int, band_type: BandType
) -> np.ndarray | None:
    """The supplied band centres, one per band and each one a tabulated band.

    Read here rather than at the first lookup so that a band count that does
    not match the intensities, or a frequency that labels no row of Tables B.2
    and 2, is refused before any indicator has been computed against it.
    """
    if frequencies is None:
        return None
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if freqs.shape != (n_bands,):
        msg = (
            f"'frequencies' must carry one value per band ({n_bands}); got "
            f"shape {freqs.shape}."
        )
        raise ValueError(msg)
    for value in freqs:
        _nominal_band(float(value), band_type)
    return freqs


def _checked_residual_index(
    pressure_residual_index: float | ArrayLike | None, n_bands: int
) -> np.ndarray | None:
    r"""The instrument's :math:`\delta_{pI0}` as one finite value per band.

    A scalar is the usual form and stands for the whole spectrum, so it is
    broadcast rather than required per band.
    """
    if pressure_residual_index is None:
        return None
    residual = np.broadcast_to(
        np.asarray(pressure_residual_index, dtype=np.float64), (n_bands,)
    ).astype(np.float64)
    if not np.all(np.isfinite(residual)):
        msg = "'pressure_residual_index' must be finite."
        raise ValueError(msg)
    return residual


def _sampling_warnings(n_positions: int, area: float) -> None:
    """Warn where the position set falls short of clause 8.2.

    Clause 8.2 asks for at least one position per square metre and at least ten
    positions in all, and offers two relaxations that both end at fifty
    positions: one position per 2 m^2 where extraneous noise is significant and
    more than fifty would otherwise be needed, and fifty spread over a surface
    larger than 50 m^2 where it is not. Neither the extraneous noise nor the
    plan behind the surface is visible here, so a set of fifty or more is left
    alone and anything sparser is only warned about: the arithmetic of clause 9
    is unaffected, and it is the report that has to say the surface was
    undersampled.
    """
    if n_positions < _MIN_POSITIONS:
        warnings.warn(
            f"Only {n_positions} measurement position(s); ISO 9614-1:1993 "
            "clause 8.2 requires at least 10.",
            SoundPowerWarning,
            stacklevel=3,
        )
    if n_positions < area and n_positions < _RELAXED_POSITIONS:
        warnings.warn(
            f"{n_positions} position(s) over {area:g} m^2 is below the one "
            "position per square metre of ISO 9614-1:1993 clause 8.2, and the "
            "relaxations of that clause need at least 50 positions.",
            SoundPowerWarning,
            stacklevel=3,
        )


def _test_conditions_met(intensity: np.ndarray) -> np.ndarray:
    r"""A.2.3's per-band verdict on the test conditions, ``True`` where met.

    "If :math:`\sum I_{\mathrm{n}i}/I_0` is negative in any frequency band,
    the test conditions do not satisfy the requirements of this part of
    ISO 9614 in that frequency band." The refusal here is on a sum that is not
    *positive*, which takes in the exact zero the clause does not mention: a
    zero mean leaves equations (A.7) to (A.9) dividing by it, so there is no
    indicator to report either way. The sum is
    unweighted over the N positions, so for N > 0 its sign is the sign of the
    arithmetic mean of equation (A.9), the very mean :math:`F_3` and
    :math:`F_4` divide by.

    That is not the area-weighted sum clause 9.2 puts a band outside the method
    for, and the two part company as soon as the segments differ in area: a
    band can be one the method still applies to, with a positive total power
    and a finite level, and have been measured under conditions this part of
    ISO 9614 does not accept.

    ``mean > 0`` rather than ``not mean <= 0``, as at the other sites here: a
    mean that is not a number answers every comparison False, and it is a band
    whose test conditions were not met, not one that met them.
    """
    return np.asarray(np.mean(intensity, axis=0) > 0.0, dtype=bool)


def _band_indicators(
    intensity: np.ndarray,
    pressure_levels: np.ndarray | None,
    conditions_met: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    r"""Annex A ``F2``, ``F3`` and ``F4`` per band, ``NaN`` where undefined.

    All three are ``None``, and not merely ``NaN``, below the two positions the
    Bessel-corrected spread of equation (A.8) is defined from: one position
    measures no spread ACROSS THE SURFACE, so there is no F2, F3 or F4 to
    report rather than a spectrum of blanks.

    Not "no Annex A indicators": F1 is one, and it survives. Equation (A.1) is
    the spread of the M short-time samples at ONE position over time, so a
    one-position determination has a perfectly good F1 beside three absent
    spatial indicators. That confusion has now been written into this module
    three times and corrected three times, which is why it is spelled out here
    rather than left to the reader: F1 is temporal, F2 to F4 are spatial, and
    every statement about "the indicators" has to say which it means.

    A.2.3 makes a non-positive algebraic mean normal intensity a failure of the
    test conditions in that band, and F3 and F4 both divide by that mean, so
    the three indicators are ``NaN`` there rather than raising: the refusal is
    scoped to the band ("en esa banda de frecuencia") and the other bands of
    the same determination stand.

    Being ``NaN`` is the whole of what these indicators can say about such a
    band, and it is not enough on its own, because the band need not be flagged
    anywhere else: guarding on the area-weighted sum of clause 9.2 instead
    would hand a column of negative mean to the coefficient of variation, which
    raises, but a band A.2.3 refuses can equally be one clause 9.2 keeps, with
    ``not_applicable_band`` false and a finite level beside it. So the
    determination announces the refusal itself; see
    :func:`sound_power_intensity_points`.

    F2 and F3 need the pressure levels and F4 does not, so a caller who
    measured only the intensity still gets the indicator criterion 2 is built
    on. Where the levels are present the three come from
    :func:`~phonometry.emission.field_indicators`, so this module and the
    library's own Annex A implementation cannot drift apart.
    """
    n_positions, n_bands = intensity.shape
    if n_positions < _MIN_VARIATION_OBSERVATIONS:
        return None, None, None
    f2 = np.full(n_bands, np.nan)
    f3 = np.full(n_bands, np.nan)
    f4 = np.full(n_bands, np.nan)
    for band in range(n_bands):
        column = intensity[:, band]
        if not conditions_met[band]:
            continue
        if pressure_levels is None:
            f4[band] = _coefficient_of_variation(
                column, non_positive_message=_F4_NON_POSITIVE
            )
            continue
        indicators = field_indicators(pressure_levels[:, band], column)
        f2[band] = float(cast(float, indicators.f2))
        f3[band] = float(cast(float, indicators.f3))
        f4[band] = float(cast(float, indicators.f4))
    if pressure_levels is None:
        return None, None, f4
    return f2, f3, f4


def _temporal_indicator(
    temporal_intensity: ArrayLike | None, n_bands: int
) -> np.ndarray | None:
    """F1 per band from the short-time samples of the initial test (A.2.1)."""
    if temporal_intensity is None:
        return None
    samples = np.asarray(temporal_intensity, dtype=np.float64)
    f1 = np.atleast_1d(
        np.asarray(temporal_variability_indicator(samples), dtype=np.float64)
    )
    if f1.size != n_bands:
        msg = (
            f"'temporal_intensity' must carry one column per band, got "
            f"{f1.size} for {n_bands} band(s)."
        )
        raise ValueError(msg)
    return f1


def _criterion_2(
    f4: np.ndarray,
    positions: int,
    frequencies: np.ndarray | None,
    band_type: BandType,
    grade: DeterminationGrade,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    r"""Criterion 2 per band, and the :math:`C F_4^2` it compares ``N`` against.

    ``None`` when the criterion has no per-band form: without ``frequencies``
    there is no Table B.2 row to read, and at the survey grade there is no
    per-band column to read it from.
    """
    if frequencies is None or grade == "survey":
        return None, None
    factors = np.asarray(
        [
            position_count_factor(grade, float(f), band_type=band_type)
            for f in frequencies
        ],
        dtype=np.float64,
    )
    minimum = factors * f4**2
    # ``positions > minimum`` and not ``not (positions <= minimum)``: a NaN F4
    # marks a band whose test conditions already failed A.2.3, and it must not
    # come back qualified.
    return np.asarray(positions > minimum, dtype=bool), minimum


def _per_band_grade(
    applicable: np.ndarray,
    f1: np.ndarray | None,
    f2: np.ndarray | None,
    f3: np.ndarray | None,
    f4: np.ndarray,
    residual_index: np.ndarray | None,
    positions: int,
    frequencies: np.ndarray | None,
    band_type: BandType,
) -> np.ndarray | None:
    """The grade each band reaches, over the gates of Figure B.1.

    Grades 1 and 2 share the bias error factor K = 10 dB of Table 1, so
    criterion 1 and the inward-flow gate decide them together and only
    criterion 2 tells them apart. The survey grade is absent by construction:
    Table B.2 gives it no per-band C.
    """
    if f2 is None or f3 is None or residual_index is None or frequencies is None:
        return None
    ld = _dynamic_capability(residual_index, "engineering")
    stationary = (
        np.ones(f4.shape, dtype=bool)
        if f1 is None
        else f1 <= TEMPORAL_VARIABILITY_LIMIT
    )
    gates = applicable & stationary & (ld > f2) & ((f3 - f2) <= _NEGATIVE_POWER_LIMIT)
    verdict = np.empty(f4.shape, dtype=object)
    for band in range(f4.size):
        verdict[band] = "none"
        if not gates[band]:
            continue
        for grade in ("precision", "engineering"):
            factor = position_count_factor(
                grade, float(frequencies[band]), band_type=band_type
            )
            if positions > factor * f4[band] ** 2:
                verdict[band] = grade
                break
    return verdict


def _confidence_interval(f4: np.ndarray, positions: int) -> np.ndarray:
    r"""The 95 % interval :math:`10 \lg(1 \pm 2 F_4/\sqrt{N})` of equation (B.3).

    Returned as ``(bands, 2)``, the lower end first. The lower end is ``NaN``
    where :math:`2 F_4 / \sqrt{N} \ge 1` leaves nothing to take the logarithm
    of; that happens only well past the point where criterion 2 has failed,
    since every tabulated ``C`` is at least 8 and a satisfied criterion 2 puts
    the ratio below :math:`2/\sqrt{8}`.
    """
    spread = 2.0 * f4 / math.sqrt(positions)
    with np.errstate(divide="ignore", invalid="ignore"):
        lower = np.where(1.0 - spread > 0.0, 10.0 * np.log10(1.0 - spread), np.nan)
        upper = 10.0 * np.log10(1.0 + spread)
    return np.column_stack([lower, upper])


def _expanded_uncertainty(
    achieved: np.ndarray | None, frequencies: np.ndarray | None, band_type: BandType
) -> np.ndarray | None:
    r"""Table 2 footnote 1's :math:`\pm 2s` per band, at the grade *achieved*.

    Clause 10.6 is explicit about which grade a report states: "debe
    especificarse el grado de precision logrado en el ensayo final de acuerdo a
    la tabla 2". The grade achieved, not the grade asked for, so the row of
    Table 2 a band is read in is the row of the grade that band reached. A band
    that only reached grade 2 carries the wider grade-2 figure even where grade
    1 was requested, which is the direction that matters: the alternative
    understates the uncertainty of every band that fell short.

    A band that reached no grade carries no figure at all. Table 2 tabulates
    ``s`` against the three grades and prints no row for a determination that
    qualified as none of them, and what clause 10.5 c) offers such a band
    instead is the confidence interval of equation (B.3), which the result
    carries beside this. ``None`` for the whole determination where no per-band
    grade could be established, since there is then nothing achieved to state.
    """
    if achieved is None or frequencies is None:
        return None
    values = np.full(frequencies.size, np.nan)
    for band, reached in enumerate(achieved):
        if reached == "none":
            continue
        values[band] = 2.0 * determination_standard_deviation(
            cast(DeterminationGrade, reached),
            float(frequencies[band]),
            band_type=band_type,
        )
    return values


def _a_weighting_omission(
    criterion_1: np.ndarray | None,
    criterion_2: np.ndarray | None,
    applicable: np.ndarray,
) -> np.ndarray | None:
    """The bands clause 10.5 b) keeps out of the A-weighted sum.

    The clause omits "the bands failing criteria 1 and/or 2" and asks for the
    omission to be stated, which is what this is. A criterion that was never
    put is not one that failed, so criterion 2 narrows the set only where it
    was evaluated, and a determination carrying no criterion 1 at all has no
    omission to state rather than an empty one: ``None``, and its A-weighted
    sum goes unscreened.

    Bands outside the method are not marked here. Clause 9.2 has already taken
    them out of the sum, and marking them again would report one exclusion as
    two.
    """
    if criterion_1 is None:
        return None
    fails = ~criterion_1
    if criterion_2 is not None:
        fails = fails | ~criterion_2
    return np.asarray(fails & applicable, dtype=bool)


def _a_weighted_factor(
    contributions: np.ndarray,
    frequencies: np.ndarray,
    band_type: BandType,
    grade: DeterminationGrade,
) -> float:
    """The ``C`` criterion 2 uses on an A-weighted sum (B.1.2 and Note 11).

    B.1.2 asks for "the largest value of C in the frequency range comprised by
    this sum for the required grade", and Note 11 qualifies it: where the
    800 Hz to 5 kHz one-third-octave bands contribute less than half of the
    A-weighted sound *power*, the C values of the 200 Hz to 630 Hz bands are
    used instead. The comparison is of powers, not of levels: the Spanish print
    says "menos de la mitad del nivel total", which is not a defined operation,
    and the ISO text it translates says "less than half the total power" (see
    ``docs/ERRATA.md``). The Note is written for one-third octaves; the octave
    row covering the same decade (1 kHz to 4 kHz) is treated the same way,
    since it is the same row of Table B.2 read in the other column.
    """
    if grade == "survey":
        return position_count_factor(grade)
    mid_row, high_row = _TABLE_B2[1], _TABLE_B2[2]
    high_span = high_row[0] if band_type == "octave" else high_row[1]
    total = float(np.sum(contributions))
    nominal = np.asarray(
        [_nominal_band(float(f), band_type) for f in frequencies], dtype=np.float64
    )
    if high_span is not None and total > 0.0:
        in_high = (nominal >= high_span[0]) & (nominal <= high_span[1])
        if float(np.sum(contributions[in_high])) < 0.5 * total:
            return mid_row[2] if grade == "precision" else mid_row[3]
    return max(
        position_count_factor(grade, float(f), band_type=band_type) for f in nominal
    )


def _a_weighted_determination(
    intensity: np.ndarray,
    sound_power_level: np.ndarray,
    applicable: np.ndarray,
    omitted: np.ndarray | None,
    frequencies: np.ndarray | None,
    band_type: BandType,
    positions: int,
    residual_index: np.ndarray | None,
    f1: np.ndarray | None,
    f2: np.ndarray | None,
    f3: np.ndarray | None,
) -> tuple[float, float, str | None]:
    r"""The A-weighted level, its field non-uniformity and the grade it reaches.

    ISO 9614-1 tabulates no A-weighting corrections of its own, so the
    :math:`C_k` of ISO 3744:2010 Annex E are used, as everywhere else in this
    package. Clause 10.5 b) omits from the sum the bands failing criteria 1
    and/or 2, and clause 9.2 the bands outside the method.

    B.1.2 makes the A-weighted determination a determination in its own right,
    with its own :math:`F_4`: the A-weighted band intensities of each position
    are summed into one intensity per position, and equations (A.8) and (A.9)
    are applied to those. That is the only route to the survey grade, whose
    criterion 2 uses the single A-weighted ``C`` of Table B.2.
    """
    if frequencies is None:
        single = applicable.size == 1 and bool(applicable[0])
        return (
            float(sound_power_level[0]) if single else float("nan"),
            float("nan"),
            None,
        )
    corrections = _a_weighting_corrections(frequencies)
    summed = applicable if omitted is None else applicable & ~omitted
    contributions = np.where(
        summed, 10.0 ** (0.1 * (np.nan_to_num(sound_power_level) + corrections)), 0.0
    )
    total = float(np.sum(contributions))
    level = 10.0 * math.log10(total) if total > 0.0 else float("nan")
    if not np.any(summed) or positions < _MIN_VARIATION_OBSERVATIONS:
        return level, float("nan"), None

    # B.1.2: one A-weighted intensity per position, from the bands the sum
    # actually covers, then (A.8)/(A.9) over the positions.
    per_position = np.sum(
        intensity[:, summed] * 10.0 ** (0.1 * corrections[summed]), axis=1
    )
    if not float(np.mean(per_position)) > 0.0:
        return level, float("nan"), None
    f4_a = _coefficient_of_variation(
        per_position, non_positive_message=_F4_NON_POSITIVE
    )

    grade_a = _a_weighted_grade(
        f4_a=f4_a,
        contributions=contributions[summed],
        frequencies=frequencies[summed],
        band_type=band_type,
        positions=positions,
        residual_index=residual_index,
        f1=f1,
        f2=f2,
        f3=f3,
        summed=summed,
    )
    return level, float(f4_a), grade_a


def _a_weighted_grade(
    *,
    f4_a: float,
    contributions: np.ndarray,
    frequencies: np.ndarray,
    band_type: BandType,
    positions: int,
    residual_index: np.ndarray | None,
    f1: np.ndarray | None,
    f2: np.ndarray | None,
    f3: np.ndarray | None,
    summed: np.ndarray,
) -> str | None:
    """The grade the A-weighted sum reaches, best first.

    The gates before criterion 2 are per band and must hold in every band the
    sum covers, because an A-weighted level built on a band whose instrument
    was inadequate is no better than that band. Criterion 2 is then applied
    once, to the A-weighted ``F4``, with the ``C`` of :func:`_a_weighted_factor`.
    """
    if f2 is None or f3 is None or residual_index is None:
        return None
    stationary = (
        True if f1 is None else bool(np.all(f1[summed] <= TEMPORAL_VARIABILITY_LIMIT))
    )
    inward_ok = bool(np.all((f3[summed] - f2[summed]) <= _NEGATIVE_POWER_LIMIT))
    for grade in _GRADES:
        typed = cast(DeterminationGrade, grade)
        ld = _dynamic_capability(residual_index, typed)
        if not (stationary and inward_ok and bool(np.all(ld[summed] > f2[summed]))):
            continue
        factor = _a_weighted_factor(contributions, frequencies, band_type, typed)
        if positions > factor * f4_a**2:
            return grade
    return "none"


def sound_power_intensity_points(
    normal_intensity: ArrayLike,
    areas: ArrayLike,
    *,
    pressure_levels: ArrayLike | None = None,
    pressure_residual_index: float | ArrayLike | None = None,
    temporal_intensity: ArrayLike | None = None,
    frequencies: ArrayLike | None = None,
    band_type: BandType = "third",
    grade: DeterminationGrade = "engineering",
) -> DiscretePointIntensityResult:
    r"""Sound power by sound intensity at discrete points (ISO 9614-1:1993).

    ``normal_intensity`` is an ``(N, bands)`` array (or ``(N,)`` for a single
    band) of the signed normal intensity :math:`I_{\mathrm{n}i}` measured with
    the probe held still at each of the ``N`` points, and ``areas`` the
    ``(N,)`` areas :math:`S_i` of the segments those points stand for. The
    partial powers :math:`P_i = I_{\mathrm{n}i} S_i` (equation (11)) sum to the
    band sound power and its level :math:`L_W = 10 \lg(\sum_i P_i / P_0)`
    (equation (12)); a band whose sum is not positive is flagged
    ``not_applicable_band`` and reported as ``NaN``, the method not applying to
    it (clause 9.2).

    A single position may carry inward flow and usually does. Levels printed as
    ``(-) XX dB`` are converted by :func:`normal_intensity_from_levels`, whose
    ``negative`` argument is that ``(-)``.

    A.2.3 conditions the Annex A indicators on a different quantity, the
    unweighted mean of the ``N`` normal intensities, and makes a band whose
    mean is not positive a band in which the test conditions do not satisfy
    this part of ISO 9614. Its indicators come back ``NaN`` and the
    determination warns, because that band need not be flagged anywhere else:
    where the segments differ in area it can be a band clause 9.2 keeps, with a
    finite level and ``not_applicable_band`` false.

    Supplying ``pressure_levels`` evaluates the spatial Annex A indicators
    ``F2`` and ``F3``; ``F4`` is evaluated from the intensities alone. All
    three are spatial, so all three need at least two positions, which is the
    fewest equation (A.8) has a spread over, and all three are absent below
    that however much else was supplied. ``F1`` is the temporal one and does
    not take part in that: see ``temporal_intensity`` below. Supplying ``pressure_residual_index`` gives the dynamic
    capability :math:`L_\mathrm{d} = \delta_{pI0} - K` and criterion 1; supplying
    ``frequencies`` gives criterion 2 through the Table B.2 factor ``C`` and
    the A-weighted total. ``temporal_intensity`` carries the ``M`` short-time
    samples of the initial test into ``F1``.

    The requested ``grade`` selects ``K``, the omission rule of clause 10.5 b)
    and the tabulated factors; the grade each band actually reaches is reported
    per band in ``achieved_grade``, and for the A-weighted sum in
    ``achieved_grade_a``. The survey grade appears only in the latter, Table
    B.2 giving it no per-band ``C``. ``expanded_uncertainty`` follows the
    achieved grade rather than the requested one, which is the grade clause
    10.6 has a report state, so it needs everything ``achieved_grade`` needs.

    :param normal_intensity: ``(N, bands)`` or ``(N,)`` signed normal
        intensity, W/m^2.
    :param areas: ``(N,)`` segment areas :math:`S_i`, m^2.
    :param pressure_levels: Optional ``(N, bands)`` or ``(N,)`` sound pressure
        levels :math:`L_{pi}` at the same positions, dB.
    :param pressure_residual_index: Optional :math:`\delta_{pI0}` of the
        instrument, dB, as a scalar or one value per band.
    :param temporal_intensity: Optional ``(M, bands)`` or ``(M,)`` short-time
        samples of the normal intensity at one typical position (clause 8.2),
        W/m^2, for ``F1``.
    :param frequencies: Optional ``(bands,)`` nominal mid-band centres, Hz.
    :param band_type: ``'octave'`` or ``'third'``, the column of Tables B.2 and
        2 the frequencies are read in.
    :param grade: ``'precision'`` (grade 1), ``'engineering'`` (grade 2,
        default) or ``'survey'`` (grade 3).
    :return: A :class:`DiscretePointIntensityResult`.
    :raises ValueError: If ``grade`` is none of the three grades of Table 1, or
        ``band_type`` is neither ``'octave'`` nor ``'third'``; if
        ``normal_intensity`` is neither 1D nor 2D, or ``areas`` is not 1D; if
        the position set is empty, the positions and areas disagree in length,
        or an area is not positive and finite; if an input is not finite; if
        the optional arrays do not span the same positions or bands; if a
        frequency is not a band of Tables B.2 and 2; or if
        ``temporal_intensity`` is neither 1D nor 2D, carries fewer than two
        samples, or holds a band whose mean is not positive and so leaves
        ``F1`` nothing to normalize by.
    """
    grade = _check_grade(grade)
    band_type = _check_band_type(band_type)

    intensity, seg = _position_grid(normal_intensity, areas)
    n_positions, n_bands = intensity.shape
    freqs = _checked_frequencies(frequencies, n_bands, band_type)
    levels = _checked_position_band_grid(
        pressure_levels, "pressure_levels", n_positions, n_bands
    )
    residual = _checked_residual_index(pressure_residual_index, n_bands)

    partial_power = intensity * seg[:, None]  # Eq. (11)
    sound_power = np.sum(partial_power, axis=0)  # the sum of Eq. (12)
    # ``~(P > 0)`` and not ``P <= 0``: a NaN total answers every comparison
    # False, so the second form would call an unusable band applicable.
    applicable = sound_power > 0.0
    not_applicable = ~applicable
    with np.errstate(divide="ignore", invalid="ignore"):
        sound_power_level = np.where(
            applicable,
            10.0 * np.log10(np.maximum(sound_power, np.finfo(float).tiny) / _P0),
            np.nan,
        )
    surface_area = float(np.sum(seg))

    if np.any(not_applicable):
        warnings.warn(
            "The total sound power is not positive in one or more bands; ISO "
            "9614-1:1993 is not applicable to those bands (clause 9.2).",
            SoundPowerWarning,
            stacklevel=2,
        )
    conditions_met = _test_conditions_met(intensity)
    if not np.all(conditions_met):
        warnings.warn(
            "The algebraic mean normal intensity is not positive in one or "
            "more bands; the test conditions do not satisfy ISO 9614-1:1993 in "
            "those bands (A.2.3) and their Annex A field indicators are "
            "undefined. This is not the refusal of clause 9.2: a band whose "
            "area-weighted sum of partial powers is still positive keeps a "
            "finite sound power level and is not flagged inapplicable.",
            SoundPowerWarning,
            stacklevel=2,
        )
    _sampling_warnings(n_positions, surface_area)

    f1 = _temporal_indicator(temporal_intensity, n_bands)
    f2, f3, f4 = _band_indicators(intensity, levels, conditions_met)

    ld = None if residual is None else _dynamic_capability(residual, grade)
    criterion_1 = None if ld is None or f2 is None else np.asarray(ld > f2, dtype=bool)
    inward_ok = (
        None
        if f2 is None or f3 is None
        else np.asarray((f3 - f2) <= _NEGATIVE_POWER_LIMIT, dtype=bool)
    )
    criterion_2, minimum_positions = (
        (None, None)
        if f4 is None
        else _criterion_2(f4, n_positions, freqs, band_type, grade)
    )
    achieved = (
        None
        if f4 is None
        else _per_band_grade(
            applicable, f1, f2, f3, f4, residual, n_positions, freqs, band_type
        )
    )
    interval = None if f4 is None else _confidence_interval(f4, n_positions)

    omitted = _a_weighting_omission(criterion_1, criterion_2, applicable)
    if omitted is None and freqs is not None and n_bands > 1:
        warnings.warn(
            "The A-weighted total sums every applicable band without the ISO "
            "9614-1:1993 clause 10.5 b) screening (the bands failing criteria "
            "1 and/or 2 must be omitted and the omission stated); supply "
            "'pressure_levels' and 'pressure_residual_index' to evaluate the "
            "criteria.",
            SoundPowerWarning,
            stacklevel=2,
        )
    level_a, f4_a, grade_a = _a_weighted_determination(
        intensity,
        sound_power_level,
        applicable,
        omitted,
        freqs,
        band_type,
        n_positions,
        residual,
        f1,
        f2,
        f3,
    )

    return DiscretePointIntensityResult(
        frequencies=freqs,
        partial_power=partial_power,
        sound_power=np.asarray(sound_power, dtype=np.float64),
        sound_power_level=np.asarray(sound_power_level, dtype=np.float64),
        not_applicable_band=np.asarray(not_applicable, dtype=bool),
        f1=f1,
        f2=f2,
        f3=f3,
        f4=f4,
        dynamic_capability_index=ld,
        criterion_1=criterion_1,
        negative_power_within_limit=inward_ok,
        criterion_2=criterion_2,
        minimum_positions=minimum_positions,
        achieved_grade=achieved,
        confidence_interval=interval,
        expanded_uncertainty=_expanded_uncertainty(achieved, freqs, band_type),
        surface_area=surface_area,
        positions=int(n_positions),
        sound_power_level_a=level_a,
        a_weighting_omitted_bands=omitted,
        field_nonuniformity_a=f4_a,
        achieved_grade_a=grade_a,
        grade=grade,
    )
