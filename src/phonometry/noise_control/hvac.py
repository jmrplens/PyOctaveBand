#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""HVAC duct acoustics: fan power, duct losses, plenums and flow-generated noise.

A ventilation duct network attenuates fan noise through several mechanisms
that add up along the path, and it *regenerates* noise wherever the airflow is
disturbed. This module gathers the element models that a duct-borne noise
calculation needs, from two engineering references that are kept side by side
rather than merged:

* **Bies, Hansen & Howard**, *Engineering Noise Control* 5th ed., Chapter 8,
  for the **duct end reflection** (§8.13, Table 8.14), the **bends/elbows**
  (§8.11, Table 8.11), the **plenum chambers** (§8.17, Wells' method) and the
  **flow-generated (self) noise** of straight ducts and bends (§8.15).
* **Long**, *Architectural Acoustics* 2nd ed., Chapters 13 and 14, for the
  **fan sound power** from the operating point (Eq. 13.1 with the ASHRAE
  Tables 13.5-13.7), the **straight-duct attenuation** of unlined and lined
  rectangular and circular ducts (Eqs. 14.9-14.13 with Tables 14.1-14.3, the
  Reynolds regressions), the **lined flexible duct** insertion loss
  (Table 14.4), the **branch split loss** (Eq. 14.17), the closed-form **end
  reflection** (Eqs. 14.14-14.16), the **silencer self-noise** (Eq. 14.31 with
  Table 14.8) and the **room effect** that turns the sound power arriving at
  the terminal device into a sound pressure level in the room.

Both references trace back to the same ASHRAE data for the elbows: Bies
Table 8.11 is indexed by :math:`W / \lambda` and Long Tables 14.5-14.7 by the
frequency-width product ``f w`` (kHz times inches), and the two indexings agree
band by band (:math:`W / \lambda = 0.074 f w`), so :func:`elbow_insertion_loss`
serves both. Where they genuinely differ -- the end reflection, tabulated by
Bies and given in closed form by Long -- both are selectable
(``method="bies"`` or ``method="long"``) and neither replaces the other.

:mod:`phonometry.noise_control.duct_path` chains these elements into the
end-to-end fan-to-room calculation.

.. note::
   Bies 5th ed. gives the duct end reflection only as the ASHRAE Table 8.14
   look-up (there is no closed form in this edition); this module reproduces
   that table and interpolates it. Rectangular ducts use the equivalent
   diameter :math:`D = \sqrt{4S/\pi}`.

.. warning::
   Long's worked duct-borne sheet (Table 14.9) was produced by a commercial
   computer program, not by hand from the tables printed alongside it, and
   several of its element rows do **not** follow from the book's own data.
   The functions here implement the *printed* equations and tables, so they
   reproduce some rows of that sheet and not others. Verified band by band:

   * ``split_loss`` reproduces the 25 per cent split row (-6 dB) exactly, and
     :func:`elbow_insertion_loss` reproduces the unlined-elbow row exactly
     when the elbow is read as round (Table 14.7) at :math:`w = 24` in;
   * :func:`lined_rectangular_duct_attenuation` with ``include_unlined=True``
     reproduces the 18 x 12 in run from 500 Hz up (11/25/22/16/13 dB) and the
     36 x 24 in run at 500 Hz and 8 kHz, but is 1-2 dB low at 63-250 Hz on
     one run and 2 dB high on the other;
   * the fan row (90/86/82/79/77/75/71/61 dB) is **not** reproducible from
     Eq. 13.1 with the Table 13.5 forward-curved constants, which give
     99/99/89/84/82/77/72/67 dB at the same duty; the printed spectrum is not
     a level shift of the tabulated one, so it comes from other data;
   * the flexible-duct row (14/14/16/15/17/22/16/13 dB) is **not** the
     Table 14.4 entry for 12 in by 6 ft (3/5/10/15/17/16/9 dB);
   * :func:`diffuser_sound_power` reproduces the supply diffuser row
     (33/32/29/23/15/4/0/0 dB) to better than 1 dB in five of the six bands
     that carry it (+0.4/+0.4/+0.2/+0.7/+0.9 dB from 63 Hz to 1 kHz) and to
     1.9 dB in the sixth (2 kHz), reading the device as a 24 x 24 in
     rectangular diffuser;
   * the silencer and grille rows are manufacturer data, which is what a
     real sheet uses and what :class:`~.duct_path.DuctElement` accepts.

   The cascade arithmetic of that sheet is reproduced exactly (see the
   duct-path tests, which feed it its own printed element rows), and the
   sheet's own internal rounding is 1 dB.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

from .._internal.validation import (
    check_engine,
    require_choice,
    require_non_negative,
    require_positive,
    require_ranks,
    require_same_length,
)
from .._internal.warnings import PhonometryWarning
from ..room.steady_field import room_constant
from .duct_modes import plane_wave_limit

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray

    from .._report.metadata import ReportMetadata

_C_AIR = 343.0

#: The eight octave bands of the ASHRAE / Long duct-borne noise calculation, Hz.
OCTAVE_BANDS: NDArray[np.float64] = np.array(
    [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
)

# Half an octave, in log2 (octave) units: the half-spacing of adjacent
# octave-band centres, used as the "same band" criterion so the blade frequency
# increment C_BFI (Long Table 13.7) lands in exactly the one band of
# OCTAVE_BANDS that matches the target band.
_HALF_OCTAVE = 0.5

# Imperial-to-SI conversions. The Reynolds regressions and the ASHRAE fan
# equation are unit-sensitive empirical fits stated in foot-pound units, so the
# SI arguments of this module are converted before the published constants are
# applied and never the other way round.
_M_PER_FT = 0.3048
_M_PER_IN = 0.0254
_M3S_PER_CFM = 0.0004719474432  # 1 ft3/min in m3/s
_PA_PER_IN_WG = 249.0  # Long Eq. 13.1 reference pressure P_REF

# The perimeter-to-area ratio P/S, in ft^-1, at which the Reynolds (1990)
# low-frequency (63-250 Hz) regression for unlined rectangular ducts switches
# between its two fitted branches (Long Eqs. 14.9-14.11).
_REYNOLDS_PS_SPLIT_PER_FT = 3.0

# ---------------------------------------------------------------------------
# Bies Table 8.14 -- duct end reflection loss (dB), ASHRAE.
# Rows: internal diameter (mm). Columns: octave band centre (Hz).
# Two termination conditions: "flush" (duct flush with a wall/ceiling) and
# "free" (free space / suspended in the room).
# ---------------------------------------------------------------------------
_END_REFLECTION_BANDS: NDArray[np.float64] = np.array(
    [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0]
)
_END_REFLECTION_DIAMETERS_MM: NDArray[np.float64] = np.array(
    [150, 200, 250, 300, 400, 510, 610, 710, 810, 910, 1220, 1830], dtype=float
)
_END_REFLECTION_FLUSH: NDArray[np.float64] = np.array(
    [
        [18, 12, 7, 3, 1, 0],
        [15, 10, 5, 2, 1, 0],
        [14, 8, 4, 1, 0, 0],
        [12, 7, 3, 1, 0, 0],
        [10, 5, 2, 1, 0, 0],
        [8, 4, 1, 0, 0, 0],
        [7, 3, 1, 0, 0, 0],
        [6, 2, 1, 0, 0, 0],
        [5, 2, 1, 0, 0, 0],
        [4, 2, 0, 0, 0, 0],
        [3, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
    ],
    dtype=float,
)
_END_REFLECTION_FREE: NDArray[np.float64] = np.array(
    [
        [20, 14, 9, 5, 2, 1],
        [18, 12, 7, 3, 1, 0],
        [16, 11, 6, 2, 1, 0],
        [14, 9, 5, 2, 1, 0],
        [12, 7, 3, 1, 0, 0],
        [10, 6, 2, 1, 0, 0],
        [9, 5, 2, 1, 0, 0],
        [8, 4, 1, 0, 0, 0],
        [7, 3, 1, 0, 0, 0],
        [6, 3, 1, 0, 0, 0],
        [5, 2, 0, 0, 0, 0],
        [3, 1, 0, 0, 0, 0],
    ],
    dtype=float,
)

# ---------------------------------------------------------------------------
# Bies Table 8.11 -- elbow/bend insertion loss (dB per bend) vs W / lambda.
# The five columns are the supported (bend_type, vanes, lined, round) cases;
# each row is the value for the W/lambda band whose upper edge is the key. The
# printed rows read "W/lambda < 0.14", "0.14 <= W/lambda < 0.28" and so on, so
# each edge opens its row rather than closing the one below (see
# :func:`elbow_insertion_loss`).
#
# The same five cases are Long Tables 14.5-14.7, indexed there by the
# frequency-width product f w (kHz times inches) rather than by W/lambda; the
# two indexings agree band by band because W/lambda = 0.074 f w, and the
# tabulated values are identical. One detail of Long's Table 14.7 (round
# elbows) is a printing slip in that book rather than a difference of data: its
# rows jump from "3.8 < f w < 7.5 -> 2 dB" to "f w > 15 -> 3 dB", leaving
# 7.5 < f w < 15 unlisted. Bies Table 8.11 covers that band with 3 dB, which is
# what the round column below carries.
# ---------------------------------------------------------------------------
_ELBOW_WL_UPPER: NDArray[np.float64] = np.array([0.14, 0.28, 0.55, 1.11, 2.22, np.inf])
_ELBOW_TABLE: dict[str, NDArray[np.float64]] = {
    # square, no vanes, unlined
    "square": np.array([0, 1, 5, 8, 4, 3], dtype=float),
    # square, no vanes, lined
    "square_lined": np.array([0, 1, 6, 11, 10, 10], dtype=float),
    # square, with vanes, unlined
    "square_vanes": np.array([0, 1, 4, 6, 4, 4], dtype=float),
    # square, with vanes, lined
    "square_vanes_lined": np.array([0, 1, 4, 7, 7, 7], dtype=float),
    # round, no vanes, unlined
    "round": np.array([0, 1, 2, 3, 3, 3], dtype=float),
}

# ---------------------------------------------------------------------------
# Long Table 13.5 -- level correction K_\mathrm{F} of the ASHRAE fan sound-power model
# (Eq. 13.1), dB, over the octave bands 63 Hz to 8 kHz, with the blade
# frequency increment of Table 13.7 (its octave band, Hz, and increment, dB).
# ---------------------------------------------------------------------------
_FAN_LEVEL_CORRECTION: dict[str, NDArray[np.float64]] = {
    "airfoil_large": np.array([40, 40, 39, 34, 30, 23, 19, 17], dtype=float),
    "airfoil_small": np.array([45, 45, 43, 39, 34, 28, 24, 19], dtype=float),
    "forward_curved": np.array([53, 53, 43, 36, 36, 31, 26, 21], dtype=float),
    "radial_low": np.array([56, 47, 43, 39, 37, 32, 29, 26], dtype=float),
    "radial_medium": np.array([58, 54, 45, 42, 38, 33, 29, 26], dtype=float),
    "radial_high": np.array([61, 58, 53, 48, 46, 44, 41, 38], dtype=float),
    "vaneaxial_hub_low": np.array([49, 43, 43, 48, 47, 45, 38, 34], dtype=float),
    "vaneaxial_hub_medium": np.array([49, 43, 46, 43, 41, 36, 30, 28], dtype=float),
    "vaneaxial_hub_high": np.array([53, 52, 51, 51, 49, 47, 43, 40], dtype=float),
    "tubeaxial_large": np.array([51, 46, 47, 49, 47, 46, 39, 37], dtype=float),
    "tubeaxial_small": np.array([48, 47, 49, 53, 52, 51, 43, 40], dtype=float),
    "propeller": np.array([48, 51, 58, 56, 55, 52, 46, 42], dtype=float),
}
_FAN_BLADE_INCREMENT: dict[str, tuple[float, float]] = {
    "airfoil_large": (250.0, 3.0),
    "airfoil_small": (250.0, 3.0),
    "forward_curved": (500.0, 2.0),
    "radial_low": (125.0, 8.0),
    "radial_medium": (125.0, 8.0),
    "radial_high": (125.0, 8.0),
    "vaneaxial_hub_low": (125.0, 6.0),
    "vaneaxial_hub_medium": (125.0, 6.0),
    "vaneaxial_hub_high": (125.0, 6.0),
    "tubeaxial_large": (63.0, 7.0),
    "tubeaxial_small": (63.0, 7.0),
    "propeller": (63.0, 5.0),
}
#: Long Table 13.6 -- off-peak efficiency correction ``C_EFF``: the lower edge
#: of each static-efficiency band (per cent of peak) and its correction, dB.
_EFFICIENCY_CORRECTION: tuple[tuple[float, float], ...] = (
    (90.0, 0.0),
    (85.0, 3.0),
    (75.0, 6.0),
    (65.0, 9.0),
    (55.0, 12.0),
    (50.0, 15.0),
    (0.0, 16.0),
)
# ---------------------------------------------------------------------------
# VDI 2081 Part 1:2001-07, Section 4.3 -- the German fan model, an alternative
# to the ASHRAE scaling law above. It splits ventilation fans into three
# assembly types after VDI 3731 Part 2 and gives each one a specific sound
# power level and a spectral parameter, so a fan is described by its assembly
# and its duty rather than by a per-type band table.
# ---------------------------------------------------------------------------
#: Per assembly type: the representative specific sound power level ``L_WSM``
#: in dB, from Section 4.3.3, and the spectral parameter ``c3`` of Equation
#: (15) with the blade-frequency allowance ``dL_Wd`` in dB, both from Section
#: 4.3.4. The three ``L_WSM`` values are the printed representative ones, not
#: ``71,6 - 10 lg psi + L_USM`` recomputed. Two of the three land inside the
#: tolerance the guideline states beside the relation (RR: 34,6 against 34,
#: within +/-1; AM: 41,6 against 42, within +/-2); assembly T does not, and
#: which column is read decides it. The German prints ``67,3 +/-0,5`` and the
#: English beside it ``67.3 +/-1``, so the representative 36 dB, 0,7 dB from
#: the 35,3 the relation gives, is inside the English tolerance and outside the
#: normative German one. The printed representative value is what is stored
#: either way: it is what the guideline tells a planner to use.
_VDI2081_ASSEMBLY: dict[str, tuple[float, float, float]] = {
    # radial fans with rearwards curved blades, psi = 0,63 to 1,0
    "rr": (34.0, 0.4, 0.0),
    # cylindrical rotors with forwards curved blades, psi = 2,4 to 3,0
    "t": (36.0, 0.15, 0.0),
    # axial fans with a downstream diffuser, psi = 0,25 to 0,63
    "am": (42.0, -0.6, 4.0),
}
#: Equation (15) -- ``dL_W_Okt = -c1 - c2 (lg St + c3)^2``, both constants 5 dB.
_VDI2081_SPECTRUM_C1 = 5.0
_VDI2081_SPECTRUM_C2 = 5.0
#: Figure 13 -- the level allowance for a duty away from the best efficiency
#: point, as a cubic in ``V/V_opt``, for assemblies RR and AM. Figure 14 gives
#: the same for assembly T. Constant first, ascending powers after.
_VDI2081_OFF_DUTY: dict[str, tuple[float, float, float, float]] = {
    "rr": (18.9, -46.6, 33.0, -5.2),
    "am": (18.9, -46.6, 33.0, -5.2),
    "t": (1.5, -0.453, -7.05, 6.11),
}

#: VDI 2081 Part 1 Table 5 -- level reduction of a straight duct of 1 mm steel
#: sheet, dB/m, by size band. The table's columns are 63, 125, 250, 500 and
#: "> 1000" Hz; the last is held across 1 kHz and above, and a dash (no value
#: printed) is taken as nought. The size band is keyed on the upper edge, in m.
_VDI2081_STRAIGHT_DUCT: dict[str, tuple[tuple[float, tuple[float, ...]], ...]] = {
    "rectangular": (
        (0.20, (0.6, 0.6, 0.45, 0.3, 0.3, 0.3, 0.3, 0.3)),
        (0.40, (0.6, 0.6, 0.45, 0.3, 0.2, 0.2, 0.2, 0.2)),
        (0.80, (0.6, 0.6, 0.3, 0.15, 0.15, 0.15, 0.15, 0.15)),
        (1.00, (0.45, 0.3, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05)),
    ),
    "circular": (
        (0.20, (0.1, 0.1, 0.15, 0.15, 0.3, 0.3, 0.3, 0.3)),
        (0.40, (0.05, 0.1, 0.1, 0.15, 0.2, 0.2, 0.2, 0.2)),
        (0.80, (0.0, 0.05, 0.05, 0.1, 0.15, 0.15, 0.15, 0.15)),
        (1.00, (0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05)),
    ),
}
#: VDI 2081 Part 1 Table 7 -- level reduction of a 90 degree bend, dB, over the
#: nine octaves 31,5 Hz to 8 kHz, for a side length of 1250 mm and a limit
#: frequency in the 125 Hz octave. Section 6.2 shifts the whole row so that its
#: 125 Hz column lands on the octave holding the duct's own limit frequency.
_VDI2081_BEND: dict[str, tuple[float, ...]] = {
    "sharp": (0.0, 3.0, 7.0, 6.0, 3.0, 3.0, 3.0, 3.0, 3.0),
    "sharp_vaned": (0.0, 1.0, 6.0, 6.0, 1.0, 1.0, 1.0, 1.0, 2.0),
    "sharp_lined_both": (0.0, 3.0, 10.0, 10.0, 14.0, 18.0, 18.0, 18.0, 18.0),
    "sharp_lined_both_vaned": (0.0, 1.0, 9.0, 10.0, 14.0, 14.0, 14.0, 14.0, 14.0),
    "sharp_lined_one": (0.0, 2.0, 8.0, 6.0, 8.0, 10.0, 10.0, 10.0, 10.0),
    "radiused": (0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0),
    "round_radiused": (0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0),
}
#: The octave the Table 7 rows are tabulated against, Hz: the row's 125 Hz
#: column is the one that carries the limit frequency.
_VDI2081_BEND_REFERENCE_BAND = 125.0
#: The nine octave centres of Table 7, Hz.
_VDI2081_BEND_BANDS: tuple[float, ...] = (
    31.5,
    63.0,
    125.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
    8000.0,
)
#: VDI 2081 Part 1 Figure 28 -- the solid angle a duct nozzle radiates into,
#: as a multiple of pi: into the room, into a wall, along an edge, into a
#: corner. The library's own two terminations are the first two of these.
_VDI2081_SOLID_ANGLE: dict[str, float] = {
    "room": 4.0,
    "wall": 2.0,
    "edge": 1.0,
    "corner": 0.5,
    # the names this function already takes, mapped onto the same geometry
    "free": 4.0,
    "flush": 2.0,
}
#: Section 6.6 -- the practical ceiling on the reduction, dB. The theoretical
#: value is not reached because the duct walls radiate what the nozzle
#: reflects; the guideline's worked example applies it as a flat 15 dB.
_VDI2081_END_REFLECTION_CAP = 15.0


#: VDI 2081 Part 2 Section 1.1 -- the spectral assessment curve ``K_A``, dB,
#: over the octaves 63 Hz to 8 kHz. It is the inverse A-weighting less the 5 dB
#: the guideline allows for summing eight octave bands, rounded to whole
#: decibels as printed.
VDI2081_SPECTRAL_CORRECTION: NDArray[np.float64] = np.array(
    [21, 11, 4, -2, -5, -6, -6, -4], dtype=float
)

#: Section 6.3 -- VDI 3733's recommendation that no more than 5 dB be taken
#: from a change of cross-section, since the printed reduction is only reached
#: when the duct is anechoically terminated at both ends.
_VDI2081_SECTION_CHANGE_CAP = 5.0

#: Long Table 13.8 -- approximate fan-housing (casing) attenuation, dB, over the
#: octave bands 63 Hz to 8 kHz (Miller, 1980).
_FAN_CASING_ATTENUATION: NDArray[np.float64] = np.array(
    [0, 0, 5, 10, 15, 20, 22, 25], dtype=float
)

# ---------------------------------------------------------------------------
# Long Table 14.1 -- losses in unlined circular ducts, dB/ft, at the octave
# bands 63 Hz to 4 kHz (the table stops at 4 kHz; the 4 kHz value is held above
# it, see :func:`unlined_circular_duct_attenuation`).
# ---------------------------------------------------------------------------
_UNLINED_CIRCULAR_DB_PER_FT: NDArray[np.float64] = np.array(
    [0.03, 0.03, 0.03, 0.05, 0.07, 0.07, 0.07, 0.07]
)

# ---------------------------------------------------------------------------
# Long Table 14.2 -- constants B, C, D of the Reynolds (1990) lined
# rectangular-duct regression, Eq. 14.12.
# ---------------------------------------------------------------------------
_LINED_RECT_B: NDArray[np.float64] = np.array(
    [0.0133, 0.0574, 0.2710, 1.0147, 1.7700, 1.3920, 1.5180, 1.5810]
)
_LINED_RECT_C: NDArray[np.float64] = np.array(
    [1.959, 1.410, 0.824, 0.500, 0.695, 0.802, 0.451, 0.219]
)
_LINED_RECT_D: NDArray[np.float64] = np.array(
    [0.917, 0.941, 1.079, 1.087, 0.000, 0.000, 0.000, 0.000]
)

# ---------------------------------------------------------------------------
# Long Table 14.3 -- constants A..F of the Reynolds (1990) lined circular-duct
# third-order regression, Eq. 14.13 (columns A, B, C, D, E, F; rows 63 Hz to
# 8 kHz).
# ---------------------------------------------------------------------------
_LINED_ROUND_COEFFS: NDArray[np.float64] = np.array(
    [
        [0.2825, 0.3447, -5.251e-2, -3.837e-2, 9.132e-4, -8.294e-6],
        [0.5237, 0.2234, -4.936e-3, -2.724e-2, 3.377e-4, -2.490e-6],
        [0.3652, 0.7900, -0.1157, -1.834e-2, -1.211e-4, 2.681e-6],
        [0.1333, 1.8450, -0.3735, -1.293e-2, 8.624e-5, -4.986e-6],
        [1.9330, 0.0000, 0.0000, 6.135e-2, -3.891e-3, 3.934e-5],
        [2.7300, 0.0000, 0.0000, -7.341e-2, 4.428e-4, 1.006e-6],
        [2.8000, 0.0000, 0.0000, -0.1467, 3.404e-3, -2.851e-5],
        [1.5450, 0.0000, 0.0000, -5.452e-2, 1.290e-3, -1.318e-5],
    ]
)

#: Long: flanking limits the attenuation of a lined duct run, dB.
_LINED_DUCT_LIMIT = 40.0

# ---------------------------------------------------------------------------
# Long Table 14.4 -- lined flexible duct insertion loss, dB (ASHRAE, 1995).
# Axis 0: internal diameter, in; axis 1: length, ft (ascending); axis 2: the
# octave bands 63 Hz to 4 kHz.
# ---------------------------------------------------------------------------
_FLEX_DIAMETERS_IN: NDArray[np.float64] = np.array(
    [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0]
)
_FLEX_LENGTHS_FT: NDArray[np.float64] = np.array([3.0, 6.0, 9.0, 12.0])
_FLEX_BANDS: NDArray[np.float64] = np.array(
    [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
)
_FLEX_INSERTION_LOSS: NDArray[np.float64] = np.array(
    [
        [
            [2, 3, 3, 8, 9, 11, 7],
            [3, 6, 6, 16, 19, 21, 14],
            [5, 8, 9, 23, 28, 32, 20],
            [6, 11, 12, 31, 37, 42, 27],
        ],
        [
            [2, 3, 4, 8, 10, 10, 7],
            [4, 6, 7, 16, 19, 21, 13],
            [5, 9, 11, 24, 29, 31, 20],
            [7, 12, 14, 32, 38, 41, 26],
        ],
        [
            [2, 3, 4, 8, 10, 10, 7],
            [4, 6, 9, 17, 19, 20, 13],
            [6, 9, 13, 25, 29, 30, 20],
            [8, 12, 17, 33, 38, 40, 26],
        ],
        [
            [2, 3, 5, 8, 9, 10, 6],
            [4, 6, 10, 17, 19, 19, 13],
            [6, 9, 14, 25, 28, 29, 19],
            [8, 12, 19, 33, 37, 38, 25],
        ],
        [
            [2, 3, 5, 8, 9, 9, 6],
            [4, 6, 11, 17, 19, 19, 12],
            [6, 8, 16, 25, 28, 28, 18],
            [8, 11, 21, 33, 37, 36, 22],
        ],
        [
            [2, 3, 6, 8, 9, 9, 6],
            [4, 6, 11, 17, 19, 18, 11],
            [6, 8, 17, 25, 28, 27, 17],
            [8, 11, 22, 33, 37, 36, 22],
        ],
        [
            [2, 3, 6, 8, 9, 9, 5],
            [4, 5, 11, 16, 18, 17, 11],
            [6, 8, 17, 24, 27, 26, 16],
            [8, 10, 22, 32, 36, 34, 21],
        ],
        [
            [2, 2, 5, 8, 9, 8, 5],
            [3, 5, 10, 15, 17, 16, 9],
            [5, 7, 15, 23, 26, 23, 14],
            [7, 9, 20, 30, 34, 31, 18],
        ],
        [
            [1, 2, 4, 7, 8, 7, 4],
            [3, 4, 8, 14, 16, 14, 7],
            [4, 5, 12, 20, 23, 20, 11],
            [5, 7, 16, 27, 31, 27, 14],
        ],
        [
            [1, 1, 2, 6, 7, 6, 2],
            [1, 2, 5, 12, 14, 12, 5],
            [2, 3, 7, 17, 21, 17, 7],
            [2, 4, 9, 23, 28, 23, 9],
        ],
    ],
    dtype=float,
)

#: Long Table 14.8 -- silencer self-noise octave-band corrections, dB, to be
#: subtracted from the overall level of Eq. 14.31 (63 Hz to 8 kHz).
_SILENCER_SELF_NOISE_CORRECTION: NDArray[np.float64] = np.array(
    [4, 4, 6, 8, 13, 18, 23, 28], dtype=float
)

# ---------------------------------------------------------------------------
# ASHRAE (2019) HVAC Applications Handbook, Chapter 49, Table 9 -- maximum
# recommended "free" opening airflow velocity, m/s, at the neck of a supply
# diffuser or return register to achieve a design RC(N), keyed by design RC.
# ---------------------------------------------------------------------------
_TERMINAL_VELOCITY_LIMIT: dict[str, dict[int, float]] = {
    "supply": {45: 3.2, 40: 2.8, 35: 2.5, 30: 2.2, 25: 1.8},
    "return": {45: 3.8, 40: 3.4, 35: 3.0, 30: 2.5, 25: 2.2},
}

# ---------------------------------------------------------------------------
# ASHRAE (2019) HVAC Applications Handbook, Chapter 49, Table 10 -- decibels to
# be added to the diffuser sound rating to allow for throttling of a volume
# damper, keyed by the damper pressure ratio and by where the damper sits.
# ---------------------------------------------------------------------------
_DAMPER_PRESSURE_RATIOS: NDArray[np.float64] = np.array([1.5, 2.0, 2.5, 3.0, 4.0, 6.0])
_DAMPER_CORRECTION: dict[str, NDArray[np.float64]] = {
    "diffuser_neck": np.array([5, 9, 12, 15, 18, 24], dtype=float),
    "plenum_inlet": np.array([2, 3, 4, 5, 6, 9], dtype=float),
    "supply_duct": np.array([0, 0, 0, 2, 3, 5], dtype=float),
}


def _frequencies(frequencies: ArrayLike) -> NDArray[np.float64]:
    f = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if f.ndim != 1 or f.size == 0:
        msg = "'frequencies' must be a non-empty 1-D array."
        raise ValueError(msg)
    if np.any(f <= 0.0) or not np.all(np.isfinite(f)):
        msg = "'frequencies' must be positive and finite."
        raise ValueError(msg)
    return f


def _octave_slots(
    frequencies: ArrayLike | None,
    bands: NDArray[np.float64] = OCTAVE_BANDS,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Map requested frequencies onto tabulated octave-band slots.

    The fan model and the Reynolds duct regressions are tabulated per octave
    band, so their frequencies must be (a subset of) the tabulated centres.
    ``None`` selects all of them. A centre is matched when it is within 5 per
    cent of a tabulated value, which accepts both the nominal 63/125/... and
    the exact base-two 62.5/125/... series.

    :return: ``(frequencies, indices)`` with one table index per frequency.
    """
    if frequencies is None:
        return bands.copy(), np.arange(bands.size, dtype=np.intp)
    f = _frequencies(frequencies)
    ratio = np.abs(np.log2(f[:, None] / bands[None, :]))
    idx = np.asarray(np.argmin(ratio, axis=1), dtype=np.intp)
    if np.any(ratio[np.arange(f.size), idx] > np.log2(1.05)):
        msg = (
            "'frequencies' must be octave-band centres of "
            f"{bands.tolist()} Hz for this tabulated method."
        )
        raise ValueError(msg)
    return f, idx


#: What an HVAC spectrum holds: "attenuation" (insertion loss/attenuation, dB)
#: or "sound_power_level" (regenerated noise, dB re 1e-12 W).
HvacQuantity = Literal["attenuation", "sound_power_level"]


@dataclass(frozen=True)
class HvacSpectrumResult:
    """A per-frequency HVAC quantity (attenuation or regenerated power level).

    :ivar frequencies: Frequencies ``f``, Hz.
    :ivar values: The quantity per frequency (dB, or dB re 1e-12 W for a
        sound power level).
    :ivar quantity: What ``values`` holds (``"attenuation"`` or
        ``"sound_power_level"``).
    :ivar label: A short human label of the element.
    """

    frequencies: np.ndarray
    values: np.ndarray
    quantity: HvacQuantity
    label: str

    def __post_init__(self) -> None:
        """Reject a spectrum whose values do not run over its own bands.

        The HVAC fiche takes the number of rows it prints from ``values`` and
        the nominal frequency labelling each one from ``frequencies``, and the
        figure it embeds is the one ``plot()`` draws, one array against the
        other. Between them no length disagreement gets through in either
        direction; what it buys is an exception a long way from the mistake,
        and no sheet at all. More values than frequencies runs the row loop
        past the last label and raises ``IndexError: list index out of range``
        from inside the table builder, naming no field. Fewer, and the table
        would print a short sheet, but the figure refuses first, with
        matplotlib's ``x and y must have same first dimension, but have
        shapes (6,) and (5,)``: two bare shapes, naming neither the result nor
        which array is which. A sound-power spectrum dies earlier still, on the
        A-weighting corrections read at the band centres, with ``operands could
        not be broadcast together with shapes (7,) (8,)``.

        The extra axis is the silent one. A ``values`` of shape ``(bands, 2)``
        counts one entry per band, so every length agrees, and ``plot()`` draws
        each column as an ordinary curve and hands back axes carrying two
        spectra and the same legend entry twice. Only the fiche notices, with
        ``TypeError: only 0-dimensional arrays can be converted to Python
        scalars`` raised while formatting the first row.

        The ``quantity`` tag is pinned here as well, because everything the
        fiche says hangs off it: the basis line, the caption, the boxed
        figure and -- decisively -- the verdict's direction. The tag is a
        ``Literal`` for the type checker only; at run time an unexpected
        string used to fall through every ``== "sound_power_level"`` test and
        render a regenerated-noise spectrum as an attenuation sheet whose
        higher-is-better verdict passed 85 dB of duct noise against a 40 dB
        maximum.

        :raises ValueError: if ``values`` does not carry one entry per band,
            either field carries an extra axis, or ``quantity`` is not one of
            the two tags.
        """
        require_ranks(self, frequencies=1, values=1)
        require_same_length(self, "frequencies", "values")
        require_choice(self.quantity, "quantity", ("attenuation", "sound_power_level"))

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the quantity against a continuous log-frequency axis.

        Requires matplotlib (``pip install phonometry[plot]``).
        """
        from .._i18n import check_language
        from .._plot.noise_control import plot_hvac_spectrum

        check_language(language)
        return plot_hvac_spectrum(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an HVAC duct-noise-spectrum fiche to ``path``.

        Writes a one-page HVAC-noise sheet: the method-basis line naming the
        reported quantity and the Bies, Hansen & Howard chapter (Engineering
        Noise Control 5th ed., Chapter 8), an optional metadata header (client,
        duct element, test environment, instrumentation, climate, date), a
        per-band table (nominal frequency and the reported quantity) beside the
        spectrum, the boxed single-number result (for a regenerated-noise
        spectrum the A-weighted sound power level ``L_WA`` re 1 pW with the
        overall unweighted total; for an attenuation spectrum the mean
        attenuation with its band range), an optional verdict row against a
        declared limit, and a method-basis strip stating the reported quantity's
        relation.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the header (``client``, ``specimen`` the duct element, ``test_room``
            the test environment, ``instrumentation``, ``temperature``,
            ``relative_humidity``, ``pressure``, ``test_date``), the footer
            identity (``laboratory``, ``operator``, ``report_id``, ``notes``)
            and, via ``requirement``, a declared maximum A-weighted sound power
            level for a regenerated-noise spectrum (lower is better) or a
            declared minimum mean attenuation for an attenuation spectrum (more
            is better).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True`` a regenerated-noise table adds the
            A-weighting correction and the A-weighted band level columns.
        :param language: Fiche language: ``"en"`` (default) or ``"es"``.
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"`` or ``language``
            is unknown.
        :raises ImportError: If reportlab (or, for the figure, matplotlib) is
            not installed (``pip install phonometry[report]``).
        """
        from .._i18n import check_language

        check_language(language)
        check_engine(engine)
        from .._report.hvac import render_hvac_report

        return render_hvac_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def end_reflection_loss(
    frequencies: ArrayLike,
    diameter: float,
    *,
    termination: str = "flush",
    method: str = "bies",
    speed_of_sound: float = _C_AIR,
    aspect_ratio: float = 1.0,
    maximum_reduction_db: float | None = _VDI2081_END_REFLECTION_CAP,
) -> HvacSpectrumResult:
    """Duct end reflection loss (Bies Table 8.14, ASHRAE; or Long's closed form).

    The low-frequency reflection of sound back up a duct at its open
    termination into a room. Two published methods are offered and neither
    replaces the other:

    * ``method="bies"`` (default) interpolates the ASHRAE look-up of Bies
      Table 8.14 over ``log`` diameter and ``log`` frequency, passing exactly
      through the tabulated ``(diameter, octave band)`` nodes. The table covers
      63 Hz to 2 kHz and 150 mm to 1830 mm.
    * ``method="long"`` evaluates Reynolds' closed form as given by Long
      (Eqs. 14.14-14.15), :func:`end_reflection_loss_closed_form`, which has no
      frequency or diameter range limit.

    The two agree within a couple of decibels over the bands both cover.

    :param frequencies: Frequencies ``f``, Hz (1-D array).
    :param diameter: Duct internal diameter ``D``, m (use
        :func:`equivalent_diameter` for a rectangular duct of area ``S``).
    :param termination: ``"flush"`` (duct flush with a wall/ceiling) or
        ``"free"`` (free space / suspended in the room).
    :param method: ``"bies"`` (Table 8.14 look-up), ``"long"`` (closed form)
        or ``"vdi2081"`` (VDI 2081 Part 1 Figure 28).
    :param aspect_ratio: **VDI 2081 only.** Nozzle length over height ``m``
        (default 1, a square opening). Figure 28 is drawn from 1 to 30.
    :param maximum_reduction_db: **VDI 2081 only.** Ceiling on the reduction,
        dB (default 15). Section 6.6 says the theoretical value is not reached
        because the duct walls radiate what the nozzle reflects, and the
        guideline's own worked example applies exactly this cap. ``None``
        returns the uncapped closed form.
    :param speed_of_sound: Speed of sound ``c``, m/s (used by the closed form;
        the table is indexed by frequency directly).
    :return: A :class:`HvacSpectrumResult` of the reflection loss, dB.
    """
    if method == "vdi2081":
        bands = _frequencies(frequencies)
        bore = require_positive(diameter, "diameter")
        angle = _VDI2081_SOLID_ANGLE.get(termination)
        if angle is None:
            options = sorted(set(_VDI2081_SOLID_ANGLE))
            msg = f"'termination' must be one of {options} for method='vdi2081'."
            raise ValueError(msg)
        values = _vdi2081_end_reflection(
            bands,
            area=math.pi * bore**2 / 4.0,
            solid_angle_over_pi=angle,
            aspect_ratio=require_positive(aspect_ratio, "aspect_ratio"),
            speed_of_sound=require_positive(speed_of_sound, "speed_of_sound"),
        )
        if maximum_reduction_db is not None:
            values = np.minimum(
                values, require_positive(maximum_reduction_db, "maximum_reduction_db")
            )
        return HvacSpectrumResult(
            frequencies=bands,
            values=values,
            quantity="attenuation",
            label=f"End reflection, VDI 2081 ({termination}, dia {bore * 1000:.0f} mm)",
        )
    if require_choice(method, "method", ("bies", "long")) == "long":
        return end_reflection_loss_closed_form(
            frequencies,
            diameter,
            termination=termination,
            speed_of_sound=speed_of_sound,
        )
    f = _frequencies(frequencies)
    d_mm = require_positive(diameter, "diameter") * 1000.0
    if termination == "flush":
        table = _END_REFLECTION_FLUSH
    elif termination == "free":
        table = _END_REFLECTION_FREE
    else:
        msg = "'termination' must be 'flush' or 'free'."
        raise ValueError(msg)
    require_positive(speed_of_sound, "speed_of_sound")

    log_d = np.log(_END_REFLECTION_DIAMETERS_MM)
    log_f_band = np.log(_END_REFLECTION_BANDS)
    # Interpolate the table in log-diameter for each band, then in log-freq.
    per_band = np.array(
        [np.interp(np.log(d_mm), log_d, table[:, j]) for j in range(table.shape[1])]
    )
    values = np.interp(np.log(f), log_f_band, per_band)
    return HvacSpectrumResult(
        frequencies=f,
        values=values,
        quantity="attenuation",
        label=f"End reflection ({termination}, D = {diameter * 1000:.0f} mm)",
    )


#: The Table 7 row a sharp-edged rectangular bend takes, keyed by whether it
#: carries turning vanes and by how it is lined ("bare", or which side of the
#: corner). ``None`` marks the one combination the table does not print.
_VDI2081_SQUARE_BEND_ROW: dict[tuple[bool, str], str | None] = {
    (False, "bare"): "sharp",
    (True, "bare"): "sharp_vaned",
    (False, "one"): "sharp_lined_one",
    (True, "one"): None,
    (False, "both"): "sharp_lined_both",
    (True, "both"): "sharp_lined_both_vaned",
}


def _vdi2081_bend_row(
    *, bend_type: str, vanes: bool, lined: bool, lined_side: str
) -> tuple[str, str]:
    """The duct shape and the Table 7 row the arguments name.

    :return: The shape ``_vdi2081_bend`` reads the limit frequency with, and
        the key of the printed row.
    :raises ValueError: If the bend type is not one of the two, a round bend
        is given vanes or lining, or the combination has no printed row.
    """
    if bend_type == "round":
        if vanes or lined:
            msg = "round bends take neither vanes nor lining."
            raise ValueError(msg)
        return "circular", "round_radiused"
    if bend_type != "square":
        msg = "'bend_type' must be 'square' or 'round'."
        raise ValueError(msg)
    key = _VDI2081_SQUARE_BEND_ROW[vanes, lined_side if lined else "bare"]
    if key is None:
        msg = (
            "VDI 2081 Table 7 prints no row for a bend lined on one "
            "side with a baffle plate; use lined_side='both'."
        )
        raise ValueError(msg)
    return "rectangular", key


def _vdi2081_bend_result(
    bands: NDArray[np.float64],
    *,
    width: float,
    bend_type: str,
    vanes: bool,
    lined: bool,
    lined_side: str,
    speed_of_sound: float,
) -> HvacSpectrumResult:
    """Pick the Table 7 row the arguments describe and shift it into place."""
    side = require_choice(lined_side, "lined_side", ("both", "one"))
    size = require_positive(width, "width")
    c = require_positive(speed_of_sound, "speed_of_sound")
    shape, key = _vdi2081_bend_row(
        bend_type=bend_type, vanes=vanes, lined=lined, lined_side=side
    )
    return HvacSpectrumResult(
        frequencies=bands,
        values=_vdi2081_bend(
            bands, bend_type=key, shape=shape, size=size, speed_of_sound=c
        ),
        quantity="attenuation",
        label=f"Bend, VDI 2081 ({key.replace('_', ' ')}, {size * 1000:.0f} mm)",
    )


def elbow_insertion_loss(
    frequencies: ArrayLike,
    width: float,
    *,
    bend_type: str = "square",
    vanes: bool = False,
    lined: bool = False,
    speed_of_sound: float = _C_AIR,
    model: str = "ashrae",
    lined_side: str = "both",
) -> HvacSpectrumResult:
    r"""Duct bend/elbow insertion loss per bend, by either method.

    Indexed by the frequency-to-width ratio :math:`W / \lambda`
    (:math:`\lambda = c / f`).
    Lined bends assume the lining extends at least three duct diameters up- and
    downstream. Round bends are treated as unlined with no vanes.

    ``model="vdi2081"`` reads Table 7 of VDI 2081 Part 1 Section 6.2, which is
    printed once, for a 1250 mm side, and carried along the frequency axis for
    every other size. The duct's limit frequency comes from Equation (33),
    ``c / (2 a)``, or Equation (34), ``0,586 c / d``; the octave holding it
    takes the place of the table's own 125 Hz column, and the whole row moves
    with it. Below the shifted row the loss is nought, which is the guideline's
    statement that a bend reflects nothing while only plane waves run.

    The two methods index the same physics on the same ratio, but they do not
    tabulate the same bends: Table 7 distinguishes lining before, after, or on
    both sides of the corner, which the ASHRAE table does not, and ``width`` is
    read there as the largest side of a rectangular duct or the bore of a round
    one rather than as the width in the plane of the bend.

    :param frequencies: Frequencies ``f``, Hz (1-D array).
    :param width: Duct width ``W`` in the plane of the bend, m. For
        ``model="vdi2081"`` it is the largest side of a rectangular duct, or
        the internal diameter of a round one.
    :param bend_type: ``"square"`` or ``"round"``. VDI 2081 reads a square bend
        as sharp-edged and a round one as radiused with ``r <= 2 D``.
    :param vanes: Turning vanes fitted (square bends only).
    :param lined: Acoustically lined bend (square bends only).
    :param lined_side: **VDI 2081 only.** ``"both"`` (default) for lining
        before and after the corner, or ``"one"`` for lining on one side of it,
        which Table 7 tabulates separately. Ignored unless ``lined``.
    :param speed_of_sound: Speed of sound ``c``, m/s.
    :param model: ``"ashrae"`` (default) or ``"vdi2081"``.
    :return: A :class:`HvacSpectrumResult` of the insertion loss, dB per bend.
    """
    scheme = require_choice(model, "model", ("ashrae", "vdi2081"))
    if scheme == "vdi2081":
        return _vdi2081_bend_result(
            _frequencies(frequencies),
            width=width,
            bend_type=bend_type,
            vanes=vanes,
            lined=lined,
            lined_side=lined_side,
            speed_of_sound=speed_of_sound,
        )
    f = _frequencies(frequencies)
    w = require_positive(width, "width")
    c = require_positive(speed_of_sound, "speed_of_sound")
    if bend_type == "round":
        if vanes or lined:
            msg = "round bends take neither vanes nor lining."
            raise ValueError(msg)
        key = "round"
    elif bend_type == "square":
        key = "square" + ("_vanes" if vanes else "") + ("_lined" if lined else "")
    else:
        msg = "'bend_type' must be 'square' or 'round'."
        raise ValueError(msg)
    col = _ELBOW_TABLE[key]
    wl = w * f / c
    # Table 8.11 bounds each row as "a <= W/lambda < b", so a ratio exactly on
    # a bound belongs to the row it opens: side="right" places W/lambda = 0.14
    # in the 0.14-0.28 row (1 dB), not in the row below it (0 dB).
    idx = np.searchsorted(_ELBOW_WL_UPPER, wl, side="right")
    idx = np.clip(idx, 0, col.size - 1)
    values = col[idx]
    return HvacSpectrumResult(
        frequencies=f,
        values=values,
        quantity="attenuation",
        label=f"Elbow ({key.replace('_', ', ')}, W = {w * 1000:.0f} mm)",
    )


def plenum_attenuation(
    exit_area: float,
    line_of_sight: float,
    wall_area: float,
    mean_absorption: ArrayLike,
    *,
    angle: float = 0.0,
) -> np.ndarray | float:
    r"""Plenum-chamber transmission loss by Wells' method (Bies Eq. (8.275)).

    .. math::

       \mathrm{TL} = -10 \log_{10}\!\left[S_{\mathrm{out}}
       \left(\frac{\cos(\theta)}{\pi r^2}
       + \frac{1 - \alpha}{S_\mathrm{w} \alpha}\right)\right],

    where the reverberant term uses the plenum room constant
    :math:`R = S_\mathrm{w} \alpha / (1 - \alpha)`
    (:func:`phonometry.room.room_constant`). The
    method holds above the inlet cut-on and when the plenum is large compared
    with the wavelength; it underpredicts the low-frequency loss by 5-10 dB.

    :param exit_area: Outlet-opening area ``S_out``, m2.
    :param line_of_sight: Straight-line inlet-to-outlet distance ``r``, m.
    :param wall_area: Total internal wall area ``S_\mathrm{w}``, m2.
    :param mean_absorption: Mean Sabine wall absorption ``alpha`` in ``(0, 1)``
        (scalar or per-band).
    :param angle: Angle ``theta`` between the inlet axis and the line to the
        outlet, in ``[0, pi/2]`` rad (default 0).
    :return: The transmission loss, dB (float for scalar absorption, else a
        per-band array).
    :raises ValueError: If a dimension is not positive, ``mean_absorption``
        leaves ``(0, 1)`` or ``angle`` leaves ``[0, pi/2]``.
    """
    s_out = require_positive(exit_area, "exit_area")
    r = require_positive(line_of_sight, "line_of_sight")
    s_w = require_positive(wall_area, "wall_area")
    # Past pi/2 the direct term of Eq. (8.275) turns negative, which the
    # method does not model; a NaN fails the same comparison and is refused.
    if not (math.isfinite(angle) and 0.0 <= angle <= math.pi / 2.0):
        msg = "'angle' must lie in [0, pi/2] radians."
        raise ValueError(msg)
    alpha = np.asarray(mean_absorption, dtype=np.float64)
    if alpha.ndim > 1 or alpha.size == 0:
        msg = "'mean_absorption' must be a scalar or a non-empty 1-D array."
        raise ValueError(msg)
    if np.any(alpha <= 0.0) or np.any(alpha >= 1.0) or not np.all(np.isfinite(alpha)):
        msg = "'mean_absorption' must lie strictly in (0, 1)."
        raise ValueError(msg)
    r_const = np.asarray(room_constant(s_w, alpha), dtype=np.float64)
    direct = np.cos(angle) / (np.pi * r**2)
    reverberant = 1.0 / r_const
    tl = -10.0 * np.log10(s_out * (direct + reverberant))
    return float(tl) if tl.ndim == 0 else tl


def flow_noise_straight_duct(
    frequencies: ArrayLike,
    flow_velocity: float,
    area: float,
) -> HvacSpectrumResult:
    r"""Flow-generated octave-band sound power of a straight duct (VDI 2081-1, 5.2.1).

    .. math::

       L_{W\mathrm{B}} = \underbrace{7 + 50 \log_{10}(v) + 10 \log_{10}(S)}_{(16)}
       \underbrace{- 2 - 26 \log_{10}(1.14 + 0.02 f_\mathrm{m} / v)}_{\text{Figure 16}}

    in dB re 1e-12 W, for airflow speed ``v`` in a duct of area ``S``. The two
    halves come from two places on the same page of VDI 2081 Part 1: Equation
    (16) is the overall level, and the level difference
    :math:`\Delta L_W = L_{W\mathrm{Okt}} - L_W` is printed inside Figure 16,
    whose abscissa is :math:`f_\mathrm{m}/v`. Bies, Hansen & Howard 5e
    Eq. (8.251) prints the two as one line and credits the same guideline.

    :param frequencies: Octave-band centre frequencies ``f_m``, Hz (1-D array).
    :param flow_velocity: Mean flow speed ``v`` in the duct, m/s.
    :param area: Duct cross-sectional area ``S``, m2.
    :return: A :class:`HvacSpectrumResult` of the band sound power level,
        dB re 1e-12 W.
    """
    f = _frequencies(frequencies)
    u = require_positive(flow_velocity, "flow_velocity")
    s = require_positive(area, "area")
    lw = (
        7.0
        + 50.0 * np.log10(u)
        + 10.0 * np.log10(s)
        - 2.0
        - 26.0 * np.log10(1.14 + 0.02 * f / u)
    )
    return HvacSpectrumResult(
        frequencies=f,
        values=lw,
        quantity="sound_power_level",
        label=f"Straight-duct flow noise (U = {u:.1f} m/s)",
    )


def flow_noise_straight_duct_overall(
    flow_velocity: float,
    area: float,
    *,
    weighting: str = "Z",
) -> float:
    r"""Overall flow-noise sound power of a straight duct (VDI 2081 Part 1, 5.2.1).

    .. math::

       L_W = 7 + 50 \log_{10}(v) + 10 \log_{10}(S)

       L_{WA} = -25 + 70 \log_{10}(v) + 10 \log_{10}(S)

    Equations (16) and (17): the unweighted overall level and the A-weighted
    one, in dB re 1e-12 W, for a mean flow speed ``v`` in a duct of section
    ``S``. Neither depends on how long the run is, only on how fast the air
    moves through how large a section, and the fifth power of the speed in the
    unweighted form is why halving the duct velocity is worth fifteen decibels.

    The two are not one number weighted: fifty times the logarithm is a fifth
    power of the speed, and seventy is a seventh, because raising the speed
    also moves the spectrum up into the part of the curve the weighting stops
    attenuating. Faster air is worse than the unweighted number says.

    :func:`flow_noise_straight_duct` is this level with the relative spectrum
    of Figure 16 on it. The band levels do not sum back here, and not by a
    fixed amount either: the difference the figure prints tends to
    :math:`-2 - 26 \lg 1{,}14 = -2{,}5` dB as :math:`f_\mathrm{m}/v` falls, so
    the energy sum grows with however many octaves are taken. Over the range
    the figure is drawn for it lands within half a decibel of this number, and
    outside it the sum means nothing.

    :param flow_velocity: Mean flow speed ``v`` in the duct, m/s.
    :param area: Duct cross-sectional area ``S``, m2.
    :param weighting: ``"Z"`` (default) for Equation (16) or ``"A"`` for
        Equation (17).
    :return: The overall sound power level, dB re 1e-12 W.
    :raises ValueError: If the speed or the area is not positive, or the
        weighting is neither of the two the guideline prints.
    """
    v = require_positive(flow_velocity, "flow_velocity")
    s = require_positive(area, "area")
    kind = require_choice(weighting, "weighting", ("Z", "A"))
    offset, speed_exponent = (7.0, 50.0) if kind == "Z" else (-25.0, 70.0)
    return offset + speed_exponent * math.log10(v) + 10.0 * math.log10(s)


def flow_noise_bend(
    frequencies: ArrayLike,
    flow_velocity: float,
    area: float,
    height: float,
    *,
    density: float = 1.206,
    model: str = "ashrae",
    branch_diameter: float | None = None,
    approach_velocity: float | None = None,
    rounding_ratio: float | None = None,
) -> HvacSpectrumResult:
    r"""Flow-generated octave-band sound power of a bend or junction, by either method.

    .. math::

       L_{W\mathrm{B}} = L_{W\mathrm{s}} - 10 \log_{10}(1 + 0.165 N_\mathrm{s}^2)
       + 30 \log_{10}(U) - 103

       L_{W\mathrm{s}} = 30 \log_{10}(U) + 10 \log_{10}(S)
       + 10 \log_{10}(\rho) + 117

    with the stream power level :math:`L_{W\mathrm{s}}` (Bies Eq. (8.252)) and the
    Strouhal number :math:`N_\mathrm{s} = f H / U` (``H`` the duct
    height in the plane of the bend). The radiated sound power grows as the
    sixth power of the stream speed at low ``N_\mathrm{s}`` (the inner-corner drag
    dipole) and the eighth power at high ``N_\mathrm{s}`` (the outer-corner shear
    quadrupole); equivalently, the *efficiency* referenced to the stream power
    grows as :math:`U^3` and :math:`U^5` respectively.

    ``model="vdi2081"`` is Equation (18) of VDI 2081 Part 1 Section 5.2.2,
    which covers a junction and a bend with one law:

    .. math::

       L_W = L_W^{*} + 10 \log_{10} \Delta f + 30 \log_{10} d_\mathrm{a}
       + 50 \log_{10} v_\mathrm{a} + K

    with the normalised level of Figure 17,
    :math:`L_W^{*} = 12 - 21{,}5 (\lg St)^{1{,}268} + (32 + 13 \lg St)
    \lg (v_\mathrm{h}/v_\mathrm{a})`, the rounding correction of Figure 18,
    :math:`K = 13{,}9 (3{,}43 - \lg St)(0{,}15 - r/d_\mathrm{a})`, and
    :math:`St = f d_\mathrm{a} / v_\mathrm{a}`. A bend is the same law with
    the two velocities equal, which sends the second term of :math:`L_W^{*}`
    and the velocity ratio to nought.

    Both figures state that they hold only for :math:`St > 1`, so a band below
    that returns negative infinity, the level of no contribution at all, rather
    than an extrapolation: the fit turns over there and its fractional power of
    :math:`\lg St` is not real below one.

    :param frequencies: Octave-band centre frequencies ``f``, Hz (1-D array).
    :param flow_velocity: Mean flow speed ``U``, m/s.
    :param area: Duct cross-sectional area ``S``, m2.
    :param height: Duct height ``H`` in the plane of the bend, m.
    :param density: Air density ``rho``, kg/m3.
    :param model: ``"ashrae"`` (default, Bies) or ``"vdi2081"``.
    :param branch_diameter: **VDI 2081 only.** Diameter of the branch duct
        ``d_a``, m; for a bend, the duct's own diameter.
    :param approach_velocity: **VDI 2081 only.** Flow speed in the main duct
        ahead of the junction ``v_h``, m/s. ``None`` (default) takes it equal
        to ``flow_velocity``, which is the bend case.
    :param rounding_ratio: **VDI 2081 only.** Rounding radius over branch
        diameter ``r / d_a``, which applies the correction of Figure 18.
        ``None`` (default) leaves it out altogether, which is how the
        guideline's own worked example treats a bend: Figure 18 is drawn for
        the rounding of a **junction**, and its curves all cross zero at
        ``r / d_a = 0,15``, so passing 0 asks for a sharp-cornered junction and
        is worth over 6 dB rather than nothing. Figure 18 is drawn from 0
        to 0,20.
    :return: A :class:`HvacSpectrumResult` of the band sound power level,
        dB re 1e-12 W.
    """
    f = _frequencies(frequencies)
    u = require_positive(flow_velocity, "flow_velocity")
    if require_choice(model, "model", ("ashrae", "vdi2081")) == "vdi2081":
        if branch_diameter is None:
            msg = (
                "model='vdi2081' needs 'branch_diameter': Equation (18) is "
                "written on the diameter of the branch duct, which the Bies "
                "form does not take."
            )
            raise ValueError(msg)
        approach = u if approach_velocity is None else approach_velocity
        return HvacSpectrumResult(
            frequencies=f,
            values=_vdi2081_branch_flow_noise(
                f,
                branch_diameter=require_positive(branch_diameter, "branch_diameter"),
                branch_velocity=u,
                approach_velocity=require_positive(approach, "approach_velocity"),
                rounding_ratio=rounding_ratio,
            ),
            quantity="sound_power_level",
            label=f"Junction flow noise, VDI 2081 (v_a = {u:.1f} m/s)",
        )
    s = require_positive(area, "area")
    h = require_positive(height, "height")
    rho = require_positive(density, "density")
    lws = 30.0 * np.log10(u) + 10.0 * np.log10(s) + 10.0 * np.log10(rho) + 117.0
    ns = f * h / u
    lw = lws - 10.0 * np.log10(1.0 + 0.165 * ns**2) + 30.0 * np.log10(u) - 103.0
    return HvacSpectrumResult(
        frequencies=f,
        values=lw,
        quantity="sound_power_level",
        label=f"Mitred-bend flow noise (U = {u:.1f} m/s)",
    )


# ---------------------------------------------------------------------------
# Fan sound power (Long Chapter 13)
# ---------------------------------------------------------------------------
def blade_passing_frequency(rotational_speed: float, blades: int) -> float:
    r"""Blade passing frequency (Long Eq. 13.4).

    :math:`f_{bp} = \mathrm{rpm} \times \mathrm{blades} / 60`.

    :param rotational_speed: Fan speed, revolutions per minute.
    :param blades: Number of impeller blades.
    :return: The blade passing frequency, Hz.
    :raises ValueError: If ``blades`` is not a positive integer.
    """
    rpm = require_positive(rotational_speed, "rotational_speed")
    if blades <= 0 or not float(blades).is_integer():
        msg = "'blades' must be a positive integer."
        raise ValueError(msg)
    return rpm * float(blades) / 60.0


class HvacWarning(PhonometryWarning):
    """An HVAC input outside the span the table it feeds was tabulated from."""


#: The lowest relative efficiency Long Table 13.6 tabulates, in per cent.
#: Below it the table has one catch-all row, which is also where a caller who
#: passed a fraction rather than a percentage lands. That landing is what
#: :class:`HvacWarning` announces, since the two cases are indistinguishable
#: from the value alone.
_TABLE_13_6_FLOOR_PERCENT = 50.0


def fan_efficiency_correction(*, relative_efficiency_percent: float) -> float:
    """Off-peak efficiency correction ``C_EFF`` (Long Table 13.6).

    A fan running away from its peak static efficiency is noisier at the same
    duty. The correction is a step function of the static efficiency expressed
    as a percentage of the peak (Long Eq. 13.3): 90 per cent of peak and above
    adds nothing, and anything below 50 per cent adds 16 dB. When the peak
    efficiency is unknown Long recommends assuming 80 per cent, which lands in
    the 6 dB step.

    :param relative_efficiency_percent: **ASHRAE only.** Static efficiency as
        a **percentage**
        of the peak, in ``(0, 100]``. A fraction is not accepted in disguise:
        the table is tabulated from 50 % up, so 0,8 would fall to its bottom
        row and return 16 dB where 80 % returns 6, ten decibels with nothing to
        say it happened. Below 50 % the caller is warned that the value is
        outside the span Table 13.6 tabulates.
    :return: The correction ``C_EFF``, dB.
    :raises ValueError: If the efficiency is not in ``(0, 100]``.
    """
    eta = require_positive(relative_efficiency_percent, "relative_efficiency_percent")
    if eta > 100.0:  # noqa: PLR2004
        msg = "'relative_efficiency_percent' must not exceed 100 per cent."
        raise ValueError(msg)
    if eta < _TABLE_13_6_FLOOR_PERCENT:
        warnings.warn(
            f"A relative efficiency of {eta:g} % is below the 50 % floor Long "
            f"Table 13.6 is tabulated from, so the table's worst-case "
            f"correction is returned. A fraction such as 0,8 lands here; this "
            f"argument is a percentage.",
            HvacWarning,
            stacklevel=2,
        )
    for lower, correction in _EFFICIENCY_CORRECTION:
        if eta >= lower:
            return correction
    return _EFFICIENCY_CORRECTION[-1][1]  # pragma: no cover - the last edge is 0


def _vdi2081_fan_spectrum(
    bands: NDArray[np.float64],
    *,
    assembly: str,
    volume_flow: float,
    fan_total_pressure_pa: float,
    fan_speed_rpm: float,
    specific_sound_power_level: float | None,
    blade_count: int | None,
    relative_flow: float,
) -> NDArray[np.float64]:
    """The VDI 2081 Part 1 Section 4.3 octave spectrum, dB re 1e-12 W.

    Equation (13) sets the level, Equation (15) the shape and Figure 13 or 14
    the allowance for running away from the best efficiency point.
    """
    printed, c3, blade_allowance = _VDI2081_ASSEMBLY[assembly]
    # Positive rather than merely finite: the guideline publishes 34, 36 and
    # 42 dB and says each can rise by up to 7 dB at the best duty point, so
    # nothing near nought is a specific level, and a NaN here would otherwise
    # travel the whole way to a spectrum.
    level = (
        printed
        if specific_sound_power_level is None
        else require_positive(specific_sound_power_level, "specific_sound_power_level")
    )
    speed = require_positive(fan_speed_rpm, "fan_speed_rpm")
    ratio = require_positive(relative_flow, "relative_flow")

    # Equation (13): the total-pressure form, with the Mach number exponent
    # taken as 5 for every ventilation fan (Section 4.3.2), which is what puts
    # the factor 20 on the pressure rather than the 5 (gamma - 1) of Eq. (11).
    overall = (
        level
        + 10.0 * math.log10(volume_flow)
        + 20.0 * math.log10(fan_total_pressure_pa)
    )

    # Equation (15). The Strouhal number carries no diameter: it cancels
    # between the tip speed and the impeller circumference.
    strouhal = bands * 60.0 / (math.pi * speed)
    shape = (
        -_VDI2081_SPECTRUM_C1 - _VDI2081_SPECTRUM_C2 * (np.log10(strouhal) + c3) ** 2
    )

    # Figure 13 (RR, AM) or Figure 14 (T), a cubic in the relative flow. It is
    # nought at the best efficiency point to within a tenth of a decibel, which
    # is the value the guideline's own worked example prints there.
    a0, a1, a2, a3 = _VDI2081_OFF_DUTY[assembly]
    off_duty = a0 + a1 * ratio + a2 * ratio**2 + a3 * ratio**3

    spectrum = overall + shape + off_duty
    if blade_count is not None:
        blades = require_positive(blade_count, "blade_count")
        passing = speed * blades / 60.0
        nearest = int(np.argmin(np.abs(np.log2(passing / bands))))
        spectrum[nearest] += blade_allowance
    return spectrum


@overload
def fan_sound_power(
    volume_flow: float,
    *,
    fan_static_pressure_pa: float,
    model: Literal["ashrae"] = ...,
    fan_type: str = ...,
    relative_efficiency_percent: float = ...,
    blade_frequency: float | None = ...,
    frequencies: ArrayLike | None = ...,
) -> HvacSpectrumResult: ...


@overload
def fan_sound_power(
    volume_flow: float,
    *,
    model: Literal["vdi2081"],
    fan_total_pressure_pa: float,
    assembly: str,
    fan_speed_rpm: float,
    specific_sound_power_level: float | None = ...,
    blade_count: int | None = ...,
    relative_flow: float = ...,
    frequencies: ArrayLike | None = ...,
) -> HvacSpectrumResult: ...


# Two models with two argument groups, and the overloads above fix which
# group a given call may use.
def fan_sound_power(  # noqa: PLR0913
    volume_flow: float,
    *,
    fan_static_pressure_pa: float | None = None,
    fan_type: str = "forward_curved",
    relative_efficiency_percent: float = 80.0,
    blade_frequency: float | None = None,
    model: str = "ashrae",
    fan_total_pressure_pa: float | None = None,
    assembly: str | None = None,
    fan_speed_rpm: float | None = None,
    specific_sound_power_level: float | None = None,
    blade_count: int | None = None,
    relative_flow: float = 1.0,
    frequencies: ArrayLike | None = None,
) -> HvacSpectrumResult:
    r"""Octave-band fan sound power from the operating point, by either method.

    Two schools of calculation answer the same question and do not agree on
    how. ``model="ashrae"`` (the default) is the scaling law below;
    ``model="vdi2081"`` is the German method, described after it. Each takes
    the arguments its own standard is written on, so neither can be handed the
    other's: the ASHRAE law scales the **static** pressure, VDI 2081 the
    **total** pressure rise, and confusing them is worth 20 log of the ratio.

    The ASHRAE (1987) scaling law, originally due to Beranek and published by
    Graham (1975):

    .. math::

       L_W = K_\mathrm{F} + 10 \log_{10}(Q_\mathrm{F} / Q_\mathrm{REF}) + 10 \log_{10}(P_\mathrm{F} / P_\mathrm{REF})
       + C_{EFF} + C_{BFI}

    with the spectral constant ``K_\mathrm{F}`` of Long Table 13.5 (one row per fan
    type), the off-peak efficiency correction ``C_EFF`` of Table 13.6
    (:func:`fan_efficiency_correction`) and the blade frequency increment
    ``C_BFI`` of Table 13.7, added to the single octave band that contains the
    blade passing frequency. In SI the reference volume flow is
    :math:`Q_\mathrm{REF} = 0.472` L/s and the reference pressure
    :math:`P_\mathrm{REF} = 249` Pa, so the two logarithmic terms take the same
    values as the foot-pound form in cfm and inches of water gauge.

    The law assumes ideal inlet and outlet flow conditions and gives the power
    radiated into the duct; the fan radiates the same power from its intake and
    from its discharge. Manufacturer data measured to AMCA 300 should be
    preferred wherever it exists: this model is the early-design fallback, and
    ASHRAE's own current guidance (2019 *HVAC Applications Handbook*, Ch. 49)
    is that a fan's sound power "is best obtained from manufacturers' test data"
    to AMCA Standard 300 or ASHRAE Standard 68. Long's worked sheet (Table 14.9)
    prints a forward-curved row that this equation does not reproduce; see the
    module warning.

    **VDI 2081 Part 1:2001-07, Section 4.3.** The German method describes a fan
    by its assembly type rather than by a per-type band table:

    .. math::

       L_{W4} = L_\mathrm{WSM} + 10 \log_{10} \dot{V} + 20 \log_{10} \Delta p_\mathrm{t}

       \Delta L_{W,\mathrm{oct}} = -5 - 5 \left( \log_{10} St + c_3 \right)^{2},
       \qquad St = f \, 60 / (\pi n)

    Equation (13) and Equation (15). The factor 20 on the pressure is the
    general :math:`5(\gamma - 1)` of Equation (11) with the Mach number
    exponent taken as 5, which Section 4.3.2 does for every ventilation fan.
    The Strouhal number carries no diameter: it cancels between the tip speed
    and the impeller circumference, so the impeller size a nomogram gives is
    not an input here. Section 4.3.3 sets the specific level and the spectral
    parameter for each of the three assemblies of VDI 3731 Part 2, and
    Figures 13 and 14 add a cubic allowance for running away from the best
    duty point, worth 0,1 dB at the optimum itself.

    :param volume_flow: Volume flow through the fan ``Q_\mathrm{F}``, m3/s.
    :param fan_static_pressure_pa: **ASHRAE only.** Fan static pressure
        ``P_\mathrm{F}``, in
        **pascals gauge**. This is the pressure rise the fan produces across
        itself, not an ambient pressure, and it shares neither the unit nor the
        datum of the ``static_pressure`` the ISO 3740 family takes in
        kilopascals absolute. No plausibility guard can separate the two:
        101,325 Pa is a legitimate duty for a panel or propeller fan, so the
        name is what keeps them apart.
    :param fan_type: **ASHRAE only.** One of ``"airfoil_large"`` / ``"airfoil_small"``
        (backward-curved or backward-inclined centrifugal wheels above and
        below 36 in diameter), ``"forward_curved"``, ``"radial_low"`` /
        ``"radial_medium"`` / ``"radial_high"`` (radial blades by total
        pressure), ``"vaneaxial_hub_low"`` / ``"vaneaxial_hub_medium"`` /
        ``"vaneaxial_hub_high"`` (hub ratios 0.3-0.4, 0.4-0.6 and 0.6-0.8),
        ``"tubeaxial_large"`` / ``"tubeaxial_small"`` (above and below 40 in
        wheel diameter) or ``"propeller"``.
    :param relative_efficiency_percent: Static efficiency as a **percentage**
        of the peak (default 80, Long's recommendation when the peak is
        unknown). Table 13.6 is tabulated from 50 % up, so a fraction such as
        0,8 falls through to the table's bottom row and returns its worst-case
        16 dB correction instead of the 6 dB that 80 % earns. That is what
        :class:`HvacWarning` says when it fires below the floor.
    :param blade_frequency: **ASHRAE only.** Blade passing frequency ``f_bp``,
        Hz (from
        :func:`blade_passing_frequency`). ``None`` (default) places the
        increment in the octave band Table 13.7 tabulates for the fan type.
    :param model: ``"ashrae"`` (default) or ``"vdi2081"``.
    :param fan_total_pressure_pa: **VDI 2081 only.** Total pressure rise
        ``\Delta p_\mathrm{t}`` across the fan, Pa. Not the static pressure
        of the ASHRAE law: the total pressure carries the dynamic head as
        well, and Equation (13) scales it by 20 rather than by 10.
    :param assembly: **VDI 2081 only.** ``"rr"`` (radial, rearwards curved
        blades), ``"t"`` (cylindrical rotor, forwards curved blades) or
        ``"am"`` (axial with a downstream diffuser).
    :param fan_speed_rpm: **VDI 2081 only.** Impeller speed ``n``, min^-1.
    :param specific_sound_power_level: **VDI 2081 only.** The specific sound
        power level ``L_\mathrm{WSM}``, dB. ``None`` (default) takes the
        representative value of the assembly, 34, 36 or 42 dB. Section 4.3.3
        says a fan can sit up to 7 dB above its assembly average at the
        optimum duty point, so a manufacturer's own value belongs here.
    :param blade_count: **VDI 2081 only.** Number of impeller blades ``z``,
        which places the blade-frequency allowance of Section 4.3.4 in the
        octave holding ``n z / 60``. ``None`` (default) omits it. The
        allowance is nought for assemblies RR and T built to the state of the
        art and 4 dB for AM.
    :param relative_flow: **VDI 2081 only.** Duty as a fraction of the best
        efficiency point, ``\dot{V} / \dot{V}_\mathrm{opt}`` (default 1).
    :param frequencies: Octave-band centres, Hz; ``None`` (default) uses the
        63 Hz to 8 kHz bands of :data:`OCTAVE_BANDS`.
    :return: An :class:`HvacSpectrumResult` of the band sound power level,
        dB re 1e-12 W.
    """
    scheme = require_choice(model, "model", ("ashrae", "vdi2081"))
    q = require_positive(volume_flow, "volume_flow")
    f, idx = _octave_slots(frequencies)

    if scheme == "vdi2081":
        if fan_total_pressure_pa is None or assembly is None or fan_speed_rpm is None:
            msg = (
                "model='vdi2081' needs 'fan_total_pressure_pa', 'assembly' and "
                "'fan_speed_rpm'. VDI 2081 Part 1 Equation (13) is written on "
                "the total pressure rise, which is not the static pressure the "
                "ASHRAE model takes."
            )
            raise ValueError(msg)
        group = require_choice(assembly, "assembly", tuple(_VDI2081_ASSEMBLY))
        dpt = require_positive(fan_total_pressure_pa, "fan_total_pressure_pa")
        return HvacSpectrumResult(
            frequencies=f,
            values=_vdi2081_fan_spectrum(
                f,
                assembly=group,
                volume_flow=q,
                fan_total_pressure_pa=dpt,
                fan_speed_rpm=fan_speed_rpm,
                specific_sound_power_level=specific_sound_power_level,
                blade_count=blade_count,
                relative_flow=relative_flow,
            ),
            quantity="sound_power_level",
            label=f"Fan (VDI 2081 {group.upper()}, {q * 3600:.0f} m3/h, {dpt:.0f} Pa)",
        )

    if fan_static_pressure_pa is None:
        msg = "model='ashrae' needs 'fan_static_pressure_pa'."
        raise ValueError(msg)
    kind = require_choice(fan_type, "fan_type", tuple(_FAN_LEVEL_CORRECTION))
    p = require_positive(fan_static_pressure_pa, "fan_static_pressure_pa")

    duty = 10.0 * np.log10(q * 1000.0 / 0.472) + 10.0 * np.log10(p / _PA_PER_IN_WG)
    c_eff = fan_efficiency_correction(
        relative_efficiency_percent=relative_efficiency_percent
    )
    band, c_bfi = _FAN_BLADE_INCREMENT[kind]
    if blade_frequency is not None:
        f_bp = require_positive(blade_frequency, "blade_frequency")
        nearest = int(np.argmin(np.abs(np.log2(f_bp / OCTAVE_BANDS))))
        band = float(OCTAVE_BANDS[nearest])
    increment = np.where(
        np.abs(np.log2(OCTAVE_BANDS / band)) < _HALF_OCTAVE, c_bfi, 0.0
    )

    lw = _FAN_LEVEL_CORRECTION[kind][idx] + duty + c_eff + increment[idx]
    return HvacSpectrumResult(
        frequencies=f,
        values=lw,
        quantity="sound_power_level",
        label=f"Fan ({kind.replace('_', ' ')}, {q * 3600:.0f} m3/h, {p:.0f} Pa)",
    )


def octave_band_limits(
    a_weighted_limit_db: float,
    frequencies: ArrayLike | None = None,
) -> HvacSpectrumResult:
    r"""Octave limits from an A-weighted room requirement (VDI 2081 Part 2 Eq. (1)).

    .. math::

       L_{\mathrm{Okt,max}} = L_A + K_A

    with the correction :data:`VDI2081_SPECTRAL_CORRECTION`, which is the
    inverse A-weighting less 5 dB. The 5 dB is what Section 1.1 allows for
    summing eight octave bands: a spectrum flat in A-weighted terms would earn
    9 dB, and the guideline takes 5 because the noise of an air-conditioning
    system does not follow the inverse A curve.

    The result is the **unweighted** octave level each band may reach. The same
    requirement can be read the other way round, which is what the guideline's
    own worked example does: add the A-weighting to the computed spectrum and
    compare every band against the flat ``L_A - 5``. The two are the same
    test, since ``K_A = -A - 5``.

    :param a_weighted_limit_db: The A-weighted level the room is required to
        meet ``L_A``, dB.
    :param frequencies: Octave-band centres, Hz; ``None`` (default) uses the
        63 Hz to 8 kHz bands of :data:`OCTAVE_BANDS`.
    :return: An :class:`HvacSpectrumResult` of the per-band limit, dB.
    """
    f, idx = _octave_slots(frequencies)
    limit = float(a_weighted_limit_db)
    if not math.isfinite(limit):
        msg = "'a_weighted_limit_db' must be finite."
        raise ValueError(msg)
    return HvacSpectrumResult(
        frequencies=f,
        values=limit + VDI2081_SPECTRAL_CORRECTION[idx],
        quantity="sound_power_level",
        label=f"VDI 2081 octave limits for {limit:.0f} dB(A)",
    )


def fan_casing_attenuation(
    frequencies: ArrayLike | None = None,
) -> HvacSpectrumResult:
    """Fan-housing (casing) attenuation of the radiated power (Long Table 13.8).

    Subtracted from the sound power level of :func:`fan_sound_power` to
    estimate what the fan radiates *through its housing* into the plant room
    rather than into the duct. The values assume no separate enclosure and no
    absorption inside the housing, but a silencer or a lining in the ductwork
    close to the fan; at low frequency the vibrating casing radiates as much as
    the unhoused fan would, hence the zeroes. Miller (1980) states them as
    approximate: real values depend strongly on the gauge and construction of
    the housing.

    :param frequencies: Octave-band centres, Hz; ``None`` (default) uses
        :data:`OCTAVE_BANDS`.
    :return: An :class:`HvacSpectrumResult` of the attenuation, dB.
    """
    f, idx = _octave_slots(frequencies)
    return HvacSpectrumResult(
        frequencies=f,
        values=_FAN_CASING_ATTENUATION[idx].copy(),
        quantity="attenuation",
        label="Fan casing",
    )


# ---------------------------------------------------------------------------
# Straight-duct attenuation (Long Chapter 14)
# ---------------------------------------------------------------------------
def _vdi2081_silencer_self_noise(
    bands: NDArray[np.float64],
    *,
    airway_velocity: float,
    pressure_drop_pa: float,
    approach_area: float,
    hydraulic_diameter: float,
) -> NDArray[np.float64]:
    """Equations (46), (49), (50) and (51) of VDI 2081 Part 1 Section 7.2.4.2.

    The A-weighted level comes from the speed in the clear section, the
    pressure drop across the silencer and the area it is approached over; the
    shape is a quartic in ``lg St`` and the offset a term in the same speed.
    """
    weighted = (
        56.6 * math.log10(airway_velocity)
        - 0.5 * math.log10(pressure_drop_pa)
        + 10.0 * math.log10(approach_area)
        - 12.7
    )
    offset = -13.0 * math.log10(airway_velocity) + 13.5
    lg_st = np.log10(bands * hydraulic_diameter / airway_velocity)
    shape = 11.4 - 14.9 * lg_st - 1.4 * lg_st**2 + 2.2 * lg_st**3 - 0.5 * lg_st**4
    return np.asarray(weighted + shape + offset, dtype=np.float64)


def _vdi2081_end_reflection(
    bands: NDArray[np.float64],
    *,
    area: float,
    solid_angle_over_pi: float,
    aspect_ratio: float,
    speed_of_sound: float,
) -> NDArray[np.float64]:
    """Figure 28 of VDI 2081 Part 1 Section 6.6, in its printed closed form.

    The first term is the reflection of a piston small against the wavelength;
    the second is the correction for a slot-shaped rather than square nozzle,
    which is why it carries the aspect ratio.
    """
    piston = 10.0 * np.log10(
        1.0
        + (speed_of_sound / (4.0 * math.pi * bands)) ** 2
        * (solid_angle_over_pi * math.pi)
        / area
    )
    slot = aspect_ratio * (0.04283 * np.log10(bands * math.sqrt(area)) - 0.0303)
    return np.asarray(piston + slot, dtype=np.float64)


def _vdi2081_octave_bandwidth(bands: NDArray[np.float64]) -> NDArray[np.float64]:
    """Table 1 of VDI 2081 Part 1: an octave is ``f_m / sqrt(2)`` wide.

    Its edges are ``f_m / sqrt(2)`` and ``f_m sqrt(2)``, so the width is the
    difference, which reduces to the centre over the root of two: 44,55 Hz at
    63 Hz, as the guideline's own tables print in their header row.
    """
    return np.asarray(bands / math.sqrt(2.0), dtype=np.float64)


def _vdi2081_straight_flow_noise(
    bands: NDArray[np.float64], *, velocity: float, area: float
) -> NDArray[np.float64]:
    """Equations (16) and Figure 16 of VDI 2081 Part 1 Section 5.2.1.

    The overall level does not depend on how long the run is, only on how fast
    the air moves through how large a section; the shape is a single closed
    curve in ``f / v``.
    """
    overall = 7.0 + 50.0 * math.log10(velocity) + 10.0 * math.log10(area)
    shape = -2.0 - 26.0 * np.log10(1.14 + 0.02 * bands / velocity)
    return np.asarray(overall + shape, dtype=np.float64)


def _vdi2081_branch_flow_noise(
    bands: NDArray[np.float64],
    *,
    branch_diameter: float,
    branch_velocity: float,
    approach_velocity: float,
    rounding_ratio: float | None,
) -> NDArray[np.float64]:
    """Equation (18) with Figures 17 and 18, Section 5.2.2.

    Both figures are printed with a closed form beside them and both are
    stated to hold only above a Strouhal number of one, so a band below that
    is returned as no contribution rather than extrapolated: the fit turns
    over there and ``(lg St)^1.268`` is not real for ``St < 1``.
    """
    strouhal = bands * branch_diameter / branch_velocity
    lg_st = np.log10(np.where(strouhal > 1.0, strouhal, 1.0))
    ratio = math.log10(approach_velocity / branch_velocity)
    normalised = 12.0 - 21.5 * lg_st**1.268 + (32.0 + 13.0 * lg_st) * ratio
    correction = (
        0.0
        if rounding_ratio is None
        else 13.9
        * (3.43 - lg_st)
        * (0.15 - require_non_negative(rounding_ratio, "rounding_ratio"))
    )
    level = (
        normalised
        + 10.0 * np.log10(_vdi2081_octave_bandwidth(bands))
        + 30.0 * math.log10(branch_diameter)
        + 50.0 * math.log10(branch_velocity)
        + correction
    )
    return np.asarray(np.where(strouhal > 1.0, level, -np.inf), dtype=np.float64)


def _vdi2081_limit_frequency(shape: str, size: float, speed_of_sound: float) -> float:
    """Equation (33) or (34): the frequency below which only plane waves run.

    The guideline prints ``c / (2 a)`` for a rectangular duct of largest side
    ``a`` and ``0,586 c / d`` for a round one of bore ``d``, and both are the
    first cut-on frequency of the duct, which :func:`plane_wave_limit` already
    computes from the mode theory: the rectangular one exactly, and the round
    one to the three figures the guideline rounds 1,8412 over pi to: 0,586
    against 0,58607, which is one part in eight thousand. So the law is called
    rather than written a second time, and the width the rectangular form
    needs is the one it is given, the other side being irrelevant to the first
    mode across the largest one.
    """
    if shape == "rectangular":
        return plane_wave_limit(width=size, height=size, speed_of_sound=speed_of_sound)
    return plane_wave_limit(diameter=size, speed_of_sound=speed_of_sound)


def _vdi2081_bend(
    bands: NDArray[np.float64],
    *,
    bend_type: str,
    shape: str,
    size: float,
    speed_of_sound: float,
) -> NDArray[np.float64]:
    """Table 7 of VDI 2081 Part 1, shifted onto the duct's own limit frequency.

    The table is printed for a 1250 mm side, whose limit frequency falls in the
    125 Hz octave. Section 6.2 carries the whole spectrum along the frequency
    axis so that its 125 Hz column lands on the octave holding the limit
    frequency of the duct at hand, which is what makes one printed row serve
    every duct size.
    """
    row = np.array(_VDI2081_BEND[bend_type])
    table_bands = np.array(_VDI2081_BEND_BANDS)
    limit = _vdi2081_limit_frequency(shape, size, speed_of_sound)
    # Step 2: the octave holding the limit frequency, whose edges Section 6.2
    # defines as f_m / sqrt(2) and f_m * sqrt(2).
    holder = float(table_bands[int(np.argmin(np.abs(np.log2(limit / table_bands))))])
    # Step 3: shift, in whole octaves, of the reference column onto it.
    shift = round(math.log2(holder / _VDI2081_BEND_REFERENCE_BAND))

    values = np.zeros_like(bands)
    for slot, band in enumerate(bands):
        source = band / 2.0**shift
        below = source < table_bands[0] / math.sqrt(2.0)
        if below:
            # Off the bottom of the shifted row: below the limit frequency a
            # bend reflects nothing the table accounts for.
            continue
        index = int(np.argmin(np.abs(np.log2(source / table_bands))))
        values[slot] = row[index]
    return values


def _vdi2081_straight_run(
    bands: NDArray[np.float64],
    *,
    shape: str,
    size: float,
    length: float,
) -> NDArray[np.float64]:
    """Table 5 of VDI 2081 Part 1, dB, for one straight run.

    ``size`` is the largest clear side length of a rectangular duct or the
    internal diameter of a round one, which is what selects the table's row.
    """
    rows = _VDI2081_STRAIGHT_DUCT[shape]
    upper = rows[-1][0]
    if size > upper:
        msg = (
            f"'{shape}' duct of {size:g} m is outside VDI 2081 Table 5, which "
            f"stops at {upper:g} m. Above it Section 6.1 says the attenuation "
            "of a rigid, massive duct is negligible rather than tabulated."
        )
        raise ValueError(msg)
    per_metre = next(values for edge, values in rows if size <= edge)
    # The table's five columns run 63, 125, 250, 500 and "> 1000" Hz, so a band
    # is placed by its own centre rather than by its index.
    slots = np.array(
        [min(int(round(math.log2(f / 63.0))), len(per_metre) - 1) for f in bands]
    )
    rates = np.take(np.array(per_metre), np.clip(slots, 0, len(per_metre) - 1))
    return np.asarray(rates * length, dtype=np.float64)


def _perimeter_over_area(width: float, height: float) -> float:
    """``P / S`` of a rectangular duct in ft^-1, from SI side lengths in m."""
    w = require_positive(width, "width") / _M_PER_FT
    h = require_positive(height, "height") / _M_PER_FT
    return 2.0 * (w + h) / (w * h)


def unlined_rectangular_duct_attenuation(
    frequencies: ArrayLike,
    width: float,
    height: float,
    length: float,
    *,
    wrapped: bool = False,
    model: str = "ashrae",
) -> HvacSpectrumResult:
    r"""Attenuation of an unlined rectangular sheet-metal duct (Long Eqs. 14.9-14.11).

    Sound running down an unlined duct loses energy into the induced motion of
    the duct walls, so the loss grows with the perimeter-to-area ratio ``P / S``
    (a wide, shallow duct has floppier side walls). Reynolds (1990) fits the
    63 Hz to 250 Hz bands with :math:`R = 17.0 (P/S)^{0.25} f^{-0.85} l` for
    :math:`P/S \ge 3` ft^-1 and :math:`R = 1.64 (P/S)^{0.73} f^{-0.58} l`
    below it, and
    everything above 250 Hz with :math:`R = 0.02 (P/S)^{0.8} l`. An external
    fibreglass blanket adds surface mass and doubles the low-frequency loss
    (``wrapped=True``).

    ``model="vdi2081"`` reads Table 5 of VDI 2081 Part 1 Section 6.1 instead: a
    step table in dB per metre, keyed on the **largest** clear side length and
    on five frequency columns, 63, 125, 250, 500 and above 1000 Hz. It is a
    different account of the same loss, not a restatement of this one: Reynolds
    fits a continuous power law of the perimeter-to-area ratio, VDI 2081
    tabulates four size bands of 1 mm steel sheet, and where the two overlap
    they differ by a decibel or two per metre. ``wrapped`` has no meaning there
    and is refused.

    :param frequencies: Octave-band centre frequencies ``f``, Hz (1-D array).
    :param width: Duct width, m.
    :param height: Duct height, m.
    :param length: Duct run length ``l``, m.
    :param wrapped: The duct is externally wrapped with a fibreglass blanket,
        which doubles the 63 Hz to 250 Hz attenuation.
    :param model: ``"ashrae"`` (default, Reynolds) or ``"vdi2081"``
        (Table 5, the largest side length selecting the row).
    :return: An :class:`HvacSpectrumResult` of the attenuation, dB.
    """
    scheme = require_choice(model, "model", ("ashrae", "vdi2081"))
    f = _frequencies(frequencies)
    if scheme == "vdi2081":
        if wrapped:
            msg = (
                "'wrapped' has no meaning for model='vdi2081': Table 5 is "
                "tabulated for bare 1 mm steel sheet and prints no lagged row."
            )
            raise ValueError(msg)
        side = max(require_positive(width, "width"), require_positive(height, "height"))
        return HvacSpectrumResult(
            frequencies=f,
            values=_vdi2081_straight_run(
                f,
                shape="rectangular",
                size=side,
                length=require_positive(length, "length"),
            ),
            quantity="attenuation",
            label=(
                f"Straight duct, VDI 2081 ({width * 1000:.0f} x "
                f"{height * 1000:.0f} mm, {length:.2f} m)"
            ),
        )
    ps = _perimeter_over_area(width, height)
    ell = require_positive(length, "length") / _M_PER_FT
    low = (
        17.0 * ps**0.25 * f**-0.85 * ell
        if ps >= _REYNOLDS_PS_SPLIT_PER_FT
        else 1.64 * ps**0.73 * f**-0.58 * ell
    )
    if wrapped:
        low = 2.0 * low
    high = np.full_like(f, 0.02 * ps**0.8 * ell)
    values = np.where(f <= 250.0 * 1.05, low, high)
    return HvacSpectrumResult(
        frequencies=f,
        values=values,
        quantity="attenuation",
        label=(
            f"Unlined rectangular duct ({width * 1000:.0f} x {height * 1000:.0f} mm, "
            f"{length:.2f} m)"
        ),
    )


def unlined_circular_duct_attenuation(
    frequencies: ArrayLike | None,
    length: float,
    *,
    diameter: float | None = None,
    model: str = "ashrae",
) -> HvacSpectrumResult:
    """Attenuation of an unlined circular sheet-metal duct (Long Table 14.1).

    A circular duct is far stiffer than a rectangular one in its breathing
    mode, so the sound field can hardly excite it: the loss is about a tenth of
    the rectangular value and is tabulated as a length rate alone, 0.03 dB/ft
    up to 250 Hz and 0.05 to 0.07 dB/ft above. The published table stops at
    4 kHz; the 4 kHz rate is held for the 8 kHz band.

    ``model="vdi2081"`` reads Table 5 of VDI 2081 Part 1 Section 6.1, which
    does depend on the diameter: a wide round duct is stiffer still and its
    tabulated loss falls to nothing at 63 Hz above 400 mm, where the table
    prints a dash. That is the substantive difference between the two accounts
    of this element, and it is why ``diameter`` is required there and not here.

    :param frequencies: Octave-band centres, Hz; ``None`` uses
        :data:`OCTAVE_BANDS`.
    :param length: Duct run length, m.
    :param diameter: **VDI 2081 only.** Internal diameter, m, which selects the
        Table 5 row. The table stops at 1,00 m.
    :param model: ``"ashrae"`` (default, Long Table 14.1) or ``"vdi2081"``.
    :return: An :class:`HvacSpectrumResult` of the attenuation, dB.
    """
    scheme = require_choice(model, "model", ("ashrae", "vdi2081"))
    if scheme == "vdi2081":
        if diameter is None:
            msg = (
                "model='vdi2081' needs 'diameter': Table 5 tabulates a round "
                "duct by its bore, where Long Table 14.1 does not."
            )
            raise ValueError(msg)
        bands = _frequencies(OCTAVE_BANDS if frequencies is None else frequencies)
        bore = require_positive(diameter, "diameter")
        return HvacSpectrumResult(
            frequencies=bands,
            values=_vdi2081_straight_run(
                bands,
                shape="circular",
                size=bore,
                length=require_positive(length, "length"),
            ),
            quantity="attenuation",
            label=(
                f"Straight duct, VDI 2081 (dia {bore * 1000:.0f} mm, {length:.2f} m)"
            ),
        )
    f, idx = _octave_slots(frequencies)
    ell = require_positive(length, "length") / _M_PER_FT
    return HvacSpectrumResult(
        frequencies=f,
        values=_UNLINED_CIRCULAR_DB_PER_FT[idx] * ell,
        quantity="attenuation",
        label=f"Unlined circular duct ({length:.2f} m)",
    )


def lined_rectangular_duct_attenuation(
    frequencies: ArrayLike | None,
    width: float,
    height: float,
    length: float,
    lining_thickness: float,
    *,
    include_unlined: bool = False,
) -> HvacSpectrumResult:
    r"""Insertion loss of a lined rectangular duct (Long Eq. 14.12, Table 14.2).

    The Reynolds (1990) regression :math:`R = B (P/S)^C t^D l`, with the duct
    perimeter ``P`` in feet, its area ``S`` in square feet, the lining
    thickness ``t`` in inches and the run length ``l`` in feet. It was fitted
    to 25 mm to 52 mm linings of 24 to 48 kg/m3 density over ``P / S`` from
    1.1667 to 6 ft^-1; linings thinner than 25 mm are generally ineffective.
    The insertion loss is measured by substituting the lined section for an
    unlined one of the same face size, so the unlined attenuation may be added
    on top (``include_unlined=True``, which Long recommends for rectangular
    ducts). Flanking limits the total to 40 dB.

    :param frequencies: Octave-band centres, Hz; ``None`` uses
        :data:`OCTAVE_BANDS`.
    :param width: Duct width, m.
    :param height: Duct height, m.
    :param length: Duct run length ``l``, m.
    :param lining_thickness: Lining thickness ``t``, m.
    :param include_unlined: Add the unlined-duct attenuation of
        :func:`unlined_rectangular_duct_attenuation`, the side-wall
        contribution the insertion-loss measurement subtracts out.
    :return: An :class:`HvacSpectrumResult` of the attenuation, dB.
    """
    f, idx = _octave_slots(frequencies)
    ps = _perimeter_over_area(width, height)
    ell = require_positive(length, "length") / _M_PER_FT
    t_in = require_positive(lining_thickness, "lining_thickness") / _M_PER_IN
    values = (
        _LINED_RECT_B[idx] * ps ** _LINED_RECT_C[idx] * t_in ** _LINED_RECT_D[idx] * ell
    )
    if include_unlined:
        values = (
            values
            + unlined_rectangular_duct_attenuation(f, width, height, length).values
        )
    return HvacSpectrumResult(
        frequencies=f,
        values=np.minimum(values, _LINED_DUCT_LIMIT),
        quantity="attenuation",
        label=(
            f"Lined rectangular duct ({width * 1000:.0f} x {height * 1000:.0f} mm, "
            f"{length:.2f} m, {lining_thickness * 1000:.0f} mm lining)"
        ),
    )


def lined_circular_duct_attenuation(
    frequencies: ArrayLike | None,
    diameter: float,
    length: float,
    lining_thickness: float,
) -> HvacSpectrumResult:
    r"""Insertion loss of a lined circular duct (Long Eq. 14.13, Table 14.3).

    The Reynolds (1990) third-order regression
    :math:`R = (A + B t + C t^2 + D d + E d^2 + F d^3) l`, with the lining
    thickness ``t`` and the internal diameter ``d`` in inches and the length
    ``l`` in feet. It was developed for spiral ducts with a 12 kg/m3 fibreglass
    lining 25 mm to 76 mm thick behind a 25 per cent open perforated facing,
    over internal diameters from 150 mm to 1.5 m. Negative regression values
    are clipped to zero and, as for rectangular ducts, flanking limits the run
    to 40 dB. The unlined contribution is so small for circular ducts that Long
    ignores it.

    :param frequencies: Octave-band centres, Hz; ``None`` uses
        :data:`OCTAVE_BANDS`.
    :param diameter: Internal diameter ``d``, m.
    :param length: Duct run length ``l``, m.
    :param lining_thickness: Lining thickness ``t``, m.
    :return: An :class:`HvacSpectrumResult` of the attenuation, dB.
    """
    f, idx = _octave_slots(frequencies)
    d_in = require_positive(diameter, "diameter") / _M_PER_IN
    ell = require_positive(length, "length") / _M_PER_FT
    t_in = require_positive(lining_thickness, "lining_thickness") / _M_PER_IN
    a, b, c, d, e, g = (_LINED_ROUND_COEFFS[idx, j] for j in range(6))
    rate = a + b * t_in + c * t_in**2 + d * d_in + e * d_in**2 + g * d_in**3
    values = np.clip(rate * ell, 0.0, _LINED_DUCT_LIMIT)
    return HvacSpectrumResult(
        frequencies=f,
        values=values,
        quantity="attenuation",
        label=(
            f"Lined circular duct (D = {diameter * 1000:.0f} mm, {length:.2f} m, "
            f"{lining_thickness * 1000:.0f} mm lining)"
        ),
    )


def flexible_duct_insertion_loss(
    frequencies: ArrayLike | None,
    diameter: float,
    length: float,
) -> HvacSpectrumResult:
    """Insertion loss of a lined round flexible duct (Long Table 14.4, ASHRAE 1995).

    The last run of a supply branch is usually flexible duct: a fabric liner
    inside a lightweight fibreglass fill inside a plastic membrane. Its
    published insertion loss is remarkably high, 2 to 3 dB per foot in the mid
    bands, partly because the test replaces a length of sheet-metal duct and so
    credits the flexible duct's breakout as well as its dissipation. That same
    property makes a serpentine run of flexible duct in an attic or a joist
    space work as an improvised breakout silencer. The table is interpolated
    linearly over length and over log diameter; it stops at 4 kHz, so no 8 kHz
    value is returned.

    :param frequencies: Octave-band centres, Hz, within 63 Hz to 4 kHz;
        ``None`` uses all seven tabulated bands.
    :param diameter: Internal diameter, m (100 mm to 406 mm tabulated).
    :param length: Duct run length, m (0.9 m to 3.7 m tabulated).
    :return: An :class:`HvacSpectrumResult` of the insertion loss, dB.
    """
    f, idx = _octave_slots(frequencies, _FLEX_BANDS)
    d_in = require_positive(diameter, "diameter") / _M_PER_IN
    ell_ft = require_positive(length, "length") / _M_PER_FT
    log_d = np.log(_FLEX_DIAMETERS_IN)
    # Interpolate over length first (linear), then over log diameter.
    per_diameter = np.array(
        [
            [
                np.interp(ell_ft, _FLEX_LENGTHS_FT, _FLEX_INSERTION_LOSS[i, :, j])
                for j in idx
            ]
            for i in range(_FLEX_DIAMETERS_IN.size)
        ]
    )
    values = np.array(
        [np.interp(np.log(d_in), log_d, per_diameter[:, j]) for j in range(idx.size)]
    )
    return HvacSpectrumResult(
        frequencies=f,
        values=values,
        quantity="attenuation",
        label=f"Flexible duct (D = {diameter * 1000:.0f} mm, {length:.2f} m)",
    )


def section_change_loss(
    frequencies: ArrayLike,
    upstream_area: float,
    downstream_area: float,
    *,
    shape: str = "rectangular",
    upstream_size: float | None = None,
    speed_of_sound: float = _C_AIR,
    cap: float = _VDI2081_SECTION_CHANGE_CAP,
) -> HvacSpectrumResult:
    r"""Reflection at a sudden change of duct section (VDI 2081 Part 1, 6.3).

    Figure 26 gives the reduction of the internal sound power level in closed
    form, for the area ratio :math:`r = S_1/S_2`:

    .. math::

       \Delta L_{Wi} = 10 \log_{10} \frac{(r + 1)^{2}}{4r}

    and it applies differently on the two sides of unity. A **sudden
    reduction** (:math:`r > 1`) reflects at every frequency; the figure's own
    column says the frequency has no effect. A **sudden increase**
    (:math:`r < 1`) reflects only below the limit frequency of the upstream
    duct, Equation (33) or (34), and above it the figure gives nought: past
    that frequency the duct carries more than plane waves and the mismatch
    stops behaving as one.

    A gradual change is not this: 6.3 says that where the transition is
    smooth, through a tapered adapter long compared with the wavelength, the
    reduction is negligibly small.

    The reduction is reached only with the duct anechoically terminated at
    both ends, which practice rarely is, so VDI 3733 recommends taking no more
    than 5 dB from it. That is the default ``cap``, and it binds from an area
    ratio of about 10,5 upwards, or 0,095 downwards.

    There is no ASHRAE counterpart to call through ``model=``: Long folds the
    reflection from a change of total section into the junction itself, which
    is what :func:`split_loss` implements for ``model="ashrae"``, while
    VDI 2081 treats the two as separate elements of the chain.

    :param frequencies: Frequencies ``f``, Hz (1-D array).
    :param upstream_area: Section ``S1`` the sound arrives through, m².
    :param downstream_area: Section ``S2`` it continues into, m².
    :param shape: ``"rectangular"`` (default) or ``"round"``, which decides
        which limit-frequency equation the upstream duct takes.
    :param upstream_size: The largest side of the upstream duct, m, for a
        rectangular one; its internal diameter for a round one. Needed only
        for a sudden increase, which is the only case a limit frequency enters
        the answer, and only for a rectangular duct, since a round one's
        diameter follows from its area. Giving a round duct both is allowed
        while the two agree.
    :param speed_of_sound: Speed of sound ``c``, m/s.
    :param cap: The largest reduction to take, dB (default 5).
    :return: An :class:`HvacSpectrumResult` of the reflection loss, dB.
    :raises ValueError: If an area, a size, the speed of sound or the cap is
        not positive, the shape is unknown, a round duct's size contradicts
        its area, or a rectangular duct meeting a sudden increase is given no
        size.
    """
    f = _frequencies(frequencies)
    s_1 = require_positive(upstream_area, "upstream_area")
    s_2 = require_positive(downstream_area, "downstream_area")
    c = require_positive(speed_of_sound, "speed_of_sound")
    ceiling = require_positive(cap, "cap")
    kind = require_choice(shape, "shape", ("rectangular", "round"))
    given = (
        None
        if upstream_size is None
        else require_positive(upstream_size, "upstream_size")
    )
    if kind == "round" and given is not None:
        # For a round duct the diameter and the area are one fact stated twice.
        implied = equivalent_diameter(s_1)
        if not math.isclose(given, implied, rel_tol=1e-6):
            msg = (
                f"'upstream_size' is {given:g} m and 'upstream_area' "
                f"{s_1:g} m2 implies a diameter of {implied:g} m; give one or "
                "the other, or a pair that agrees."
            )
            raise ValueError(msg)

    ratio = s_1 / s_2
    reduction = min(10.0 * math.log10((ratio + 1.0) ** 2 / (4.0 * ratio)), ceiling)
    values = np.full_like(f, reduction)
    if ratio < 1.0:
        # A sudden increase reflects only while the upstream duct carries
        # plane waves alone; Figure 26 prints "approximately 0" above that.
        # It is also the only case that needs a size at all: the frequency
        # column of a sudden reduction reads "no effect".
        if kind == "rectangular" and given is None:
            msg = (
                "'upstream_size' is the largest side of a rectangular duct and "
                "cannot be recovered from its area; a sudden increase needs it "
                "for the limit frequency, so give it, or pass shape='round'."
            )
            raise ValueError(msg)
        size = given if given is not None else equivalent_diameter(s_1)
        values = np.where(f <= _vdi2081_limit_frequency(kind, size, c), reduction, 0.0)
    direction = "reduction" if ratio > 1.0 else "increase"
    return HvacSpectrumResult(
        frequencies=f,
        values=values,
        quantity="attenuation",
        label=f"Section change, VDI 2081 (sudden {direction}, r = {ratio:.3g})",
    )


def split_loss(
    main_area: float,
    branch_areas: ArrayLike,
    *,
    branch: int = 0,
    model: str = "ashrae",
) -> float:
    r"""Power split loss into one branch of a duct division (Long Eq. 14.17).

    Where a duct divides, the sound power is shared between the branches in
    proportion to their areas, and a further reflection occurs when the total
    branch area does not match the feeder area:

    .. math::

       R = -10 \log_{10}\!\left[ 1
       - \left( \frac{\sum S_i - S_m}{\sum S_i + S_m} \right)^2 \right]
       - 10 \log_{10}\!\left( \frac{S_i}{\sum S_i} \right)

    Long prints this as a negative level change (a 25 per cent area split shows
    as -6 dB in his worked sheet); this function returns it as a positive
    attenuation, like every other loss in the module.

    ``model="vdi2081"`` keeps only the second term, which is Equation (35) of
    VDI 2081 Part 1 Section 6.4:

    .. math::

       \Delta L_W = \left| 10 \log_{10} \frac{S_1}{\sum_i S_i} \right|

    The two standards divide the same physics differently rather than
    disagreeing about it. Long folds the reflection from a change of total
    section into the junction; VDI 2081 treats a junction and a change of
    section as two elements of the chain, the second in Section 6.3, and its
    junction is the area split alone. Where the branches happen to sum to the
    feeder area the two agree; where they do not, the difference is exactly
    the reflection term, half a decibel in the guideline's own Table 1 for the
    junction whose branch areas sum to twice its feeder.

    The split is the same in every octave. The German text says so
    ("frequenzunabhängig") and Figure 27 has no frequency axis; the English
    column of the same page says the opposite, which ``docs/ERRATA.md``
    records.

    :param main_area: Cross-sectional area of the main feeder duct ``S_m``, m2.
    :param branch_areas: Areas ``S_i`` of the branches continuing on from the
        main duct, m2 (1-D array-like).
    :param branch: Index into ``branch_areas`` of the branch being followed.
    :param model: ``"ashrae"`` (default, Long Eq. 14.17, reflection included)
        or ``"vdi2081"`` (Equation (35), the area split alone).
    :return: The split loss, dB (positive).
    :raises ValueError: If the areas are not positive or ``branch`` is out of
        range.
    """
    s_m = require_positive(main_area, "main_area")
    areas = np.atleast_1d(np.asarray(branch_areas, dtype=np.float64))
    if areas.ndim != 1 or areas.size == 0:
        msg = "'branch_areas' must be a non-empty 1-D array."
        raise ValueError(msg)
    if np.any(areas <= 0.0) or not np.all(np.isfinite(areas)):
        msg = "'branch_areas' must be positive and finite."
        raise ValueError(msg)
    if not 0 <= branch < areas.size:
        msg = f"'branch' must index 'branch_areas' (0..{areas.size - 1})."
        raise ValueError(msg)
    scheme = require_choice(model, "model", ("ashrae", "vdi2081"))
    total = float(np.sum(areas))
    share = -10.0 * np.log10(areas[branch] / total)
    if scheme == "vdi2081":
        return float(abs(share))
    reflection = 1.0 - ((total - s_m) / (total + s_m)) ** 2
    return float(-10.0 * np.log10(reflection) + share)


def end_reflection_loss_closed_form(
    frequencies: ArrayLike,
    diameter: float,
    *,
    termination: str = "flush",
    speed_of_sound: float = _C_AIR,
) -> HvacSpectrumResult:
    r"""Duct end reflection loss in closed form (Long Eqs. 14.14-14.15, Reynolds).

    :math:`R = 10 \log_{10}[1 + (c / (\pi f d))^{1.88}]` for a duct terminated in
    free space and :math:`R = 10 \log_{10}[1 + (0.8 c / (\pi f d))^{1.88}]` for one
    terminated flush with
    a wall, ``d`` being the duct diameter (use the equivalent diameter
    :func:`equivalent_diameter` for a rectangular duct, Eq. 14.16). The
    exponent 1.88 is Reynolds' empirical fit: the plane-wave area-change result
    over-predicts at high frequency, where the sound leaves the duct as a beam
    and never sees the expansion. This is the closed-form alternative to the
    Bies/ASHRAE table look-up of :func:`end_reflection_loss`; the two agree
    within a couple of decibels over the bands where both are defined.

    End-reflection loss does not occur when the duct terminates in a diffuser,
    whose flare smooths the impedance transition into the room.

    :param frequencies: Frequencies ``f``, Hz (1-D array).
    :param diameter: Duct internal diameter ``d``, m.
    :param termination: ``"flush"`` (flush with a wall or ceiling) or
        ``"free"`` (free space).
    :param speed_of_sound: Speed of sound ``c``, m/s.
    :return: An :class:`HvacSpectrumResult` of the reflection loss, dB.
    """
    f = _frequencies(frequencies)
    d = require_positive(diameter, "diameter")
    c = require_positive(speed_of_sound, "speed_of_sound")
    kind = require_choice(termination, "termination", ("flush", "free"))
    factor = 0.8 if kind == "flush" else 1.0
    values = 10.0 * np.log10(1.0 + (factor * c / (np.pi * f * d)) ** 1.88)
    return HvacSpectrumResult(
        frequencies=f,
        values=values,
        quantity="attenuation",
        label=f"End reflection ({kind}, D = {d * 1000:.0f} mm)",
    )


def equivalent_diameter(area: float) -> float:
    r"""Equivalent duct diameter :math:`d = \sqrt{4S/\pi}` (Long Eq. 14.16).

    :param area: Duct cross-sectional area ``S``, m2.
    :return: The equivalent diameter, m.
    """
    return float(np.sqrt(4.0 * require_positive(area, "area") / np.pi))


# ---------------------------------------------------------------------------
# Silencers and terminal devices
# ---------------------------------------------------------------------------
def splitter_silencer_insertion_loss(
    frequencies: ArrayLike | None,
    height: float,
    length: float,
    airway_widths: ArrayLike,
    splitter_thickness: float,
) -> HvacSpectrumResult:
    r"""Insertion loss of a parallel-splitter (dissipative) silencer.

    A splitter silencer divides the duct into parallel airways separated by
    absorbent baffles. Bies, Hansen & Howard (§8.10.5) reduce it to a lined
    duct: each airway is calculated *as a lined duct whose liner thickness is
    half the splitter thickness*, because each face of a splitter lines the
    airway beside it, and the insertion losses of the airways combine as

    .. math::

       \mathrm{IL}_{tot} = -10 \log_{10}\!\left[ \frac{1}{N}
       \sum_i 10^{-\mathrm{IL}_i / 10} \right] \tag{8.241}

    which is the energy average over the airways: when they are identical the
    total equals the loss of a single passage, and when they differ the leakiest
    airway dominates, exactly as a real unit does. The airway loss itself comes
    from the Reynolds (1990) lined-rectangular-duct regression of
    :func:`lined_rectangular_duct_attenuation` (Long Eq. 14.12), so the same
    validity envelope applies: linings of 25 mm to 52 mm at 24 to 48 kg/m3 and
    a perimeter-to-area ratio of the airway between 1.1667 and 6 ft^-1.

    Published dynamic insertion loss (DIL) from the silencer manufacturer,
    measured with the design airflow and in the design direction, should be
    preferred wherever it exists; this estimate is the early-design fallback and
    ignores the entrance and exit losses of the unit. The unit's regenerated
    noise is a separate quantity, :func:`silencer_self_noise`.

    :param frequencies: Octave-band centres, Hz; ``None`` (default) uses
        :data:`OCTAVE_BANDS`.
    :param height: Height of the silencer face, m (the airway dimension the
        splitters do not divide).
    :param length: Length of the silencer in the flow direction, m.
    :param airway_widths: Free width of each airway between splitters, m
        (a scalar is taken as a single airway; give one value per airway when
        they differ).
    :param splitter_thickness: Full thickness of a splitter baffle, m; the
        equivalent liner thickness of an airway is half of it.
    :return: An :class:`HvacSpectrumResult` of the insertion loss, dB.
    :raises ValueError: If any dimension is not positive, or if
        ``airway_widths`` is not a non-empty 1-D array.
    """
    widths = np.atleast_1d(np.asarray(airway_widths, dtype=np.float64))
    if widths.ndim != 1 or widths.size == 0:
        msg = "'airway_widths' must be a non-empty 1-D array."
        raise ValueError(msg)
    if np.any(widths <= 0.0) or not np.all(np.isfinite(widths)):
        msg = "'airway_widths' must be positive and finite."
        raise ValueError(msg)
    thickness = require_positive(splitter_thickness, "splitter_thickness")
    liner = 0.5 * thickness
    per_airway = np.array(
        [
            lined_rectangular_duct_attenuation(
                frequencies, float(width), height, length, liner
            ).values
            for width in widths
        ]
    )
    values = -10.0 * np.log10(np.mean(10.0 ** (-per_airway / 10.0), axis=0))
    f, _ = _octave_slots(frequencies)
    return HvacSpectrumResult(
        frequencies=f,
        values=values,
        quantity="attenuation",
        label=(
            f"Splitter silencer ({widths.size} airways, {length:.2f} m, "
            f"{thickness * 1000:.0f} mm splitters)"
        ),
    )


def silencer_self_noise(
    frequencies: ArrayLike | None,
    airway_velocity: float,
    passages: int,
    height: float,
    *,
    model: str = "ashrae",
    pressure_drop_pa: float | None = None,
    approach_area: float | None = None,
    airway_width: float | None = None,
) -> HvacSpectrumResult:
    r"""Regenerated (self) noise of a splitter silencer (Long Eq. 14.31).

    Fry's (1988) estimate, for when manufacturer self-noise data is not
    available:

    .. math::

       L_W = 55 \log_{10}(V / V_0) + 10 \log_{10} N + 10 \log_{10}(H / H_0) - 45

    with ``V`` the velocity in the splitter airway (:math:`V_0 = 1` m/s),
    ``N`` the number of air passages and ``H`` the silencer height or, for a
    round unit, its circumference (:math:`H_0 = 1` mm). The octave-band
    spectrum follows by subtracting the corrections of Table 14.8, which
    fall steeply above 500 Hz.

    The fifth-and-a-half power of the airway velocity is the practical message:
    doubling the face velocity of a silencer adds about 17 dB, which is how a
    silencer ends up *making* the noise it was bought to remove.

    Manufacturer self-noise data is measured on a 600 x 600 mm face, so a
    published spectrum has to be corrected by :math:`10 \log_{10}(S / S_0)` for
    the actual face area before it is used; this estimate needs no such
    correction because the face size enters through ``N`` and ``H``.

    :param frequencies: Octave-band centres, Hz; ``None`` uses
        :data:`OCTAVE_BANDS`.
    :param airway_velocity: Velocity ``V`` in the splitter airway, m/s.
    :param passages: Number of air passages ``N`` between the splitters.
    :param height: Silencer height ``H`` (or circumference, if round), m.
    :return: An :class:`HvacSpectrumResult` of the band sound power level,
        dB re 1e-12 W.
    :raises ValueError: If ``passages`` is not a positive integer.
    :param model: ``"ashrae"`` (default, Long Eq. 14.31) or ``"vdi2081"``
        (Section 7.2.4.2).
    :param pressure_drop_pa: **VDI 2081 only.** Total pressure drop across the
        silencer, Pa, which Equation (49) takes.
    :param approach_area: **VDI 2081 only.** Frontal area the silencer is
        approached over ``S``, m2. That is the duct's whole section, not the
        clear area between the splitters.
    :param airway_width: **VDI 2081 only.** Clear gap between splitters ``s``,
        m. The Strouhal number is taken on the hydraulic diameter of that gap,
        which the guideline's own worked example computes as ``2 s``, the
        parallel-plate limit, rather than as ``4 A / P``; ``docs/ERRATA.md``
        records the difference.
    """
    if require_choice(model, "model", ("ashrae", "vdi2081")) == "vdi2081":
        missing = [
            name
            for name, value in (
                ("pressure_drop_pa", pressure_drop_pa),
                ("approach_area", approach_area),
                ("airway_width", airway_width),
            )
            if value is None
        ]
        if pressure_drop_pa is None or approach_area is None or airway_width is None:
            msg = (
                f"model='vdi2081' needs {missing}: Equation (49) is written on "
                "the pressure drop and the approach area, and its Strouhal "
                "number on the gap between splitters, none of which the Long "
                "form takes."
            )
            raise ValueError(msg)
        bands = _frequencies(OCTAVE_BANDS if frequencies is None else frequencies)
        gap = require_positive(airway_width, "airway_width")
        return HvacSpectrumResult(
            frequencies=bands,
            values=_vdi2081_silencer_self_noise(
                bands,
                airway_velocity=require_positive(airway_velocity, "airway_velocity"),
                pressure_drop_pa=require_positive(pressure_drop_pa, "pressure_drop_pa"),
                approach_area=require_positive(approach_area, "approach_area"),
                hydraulic_diameter=2.0 * gap,
            ),
            quantity="sound_power_level",
            label=f"Splitter silencer self-noise, VDI 2081 ({airway_velocity:.1f} m/s)",
        )
    f, idx = _octave_slots(frequencies)
    v = require_positive(airway_velocity, "airway_velocity")
    if passages <= 0 or not float(passages).is_integer():
        msg = "'passages' must be a positive integer."
        raise ValueError(msg)
    h_mm = require_positive(height, "height") * 1000.0
    overall = (
        55.0 * np.log10(v)
        + 10.0 * np.log10(float(passages))
        + 10.0 * np.log10(h_mm)
        - 45.0
    )
    return HvacSpectrumResult(
        frequencies=f,
        values=overall - _SILENCER_SELF_NOISE_CORRECTION[idx],
        quantity="sound_power_level",
        label=f"Silencer self-noise (V = {v:.1f} m/s, N = {passages})",
    )


def _octave_band_number(frequencies: NDArray[np.float64]) -> NDArray[np.float64]:
    """Long's octave band number ``N_B``: 0 at 32 Hz, 1 at 63 Hz, 2 at 125 Hz."""
    return np.round(np.log2(frequencies / 32.0))


def diffuser_sound_power(
    frequencies: ArrayLike | None,
    face_area: float,
    volume_flow: float,
    pressure_drop: float,
    *,
    shape: str = "rectangular",
    count: int = 1,
) -> HvacSpectrumResult:
    r"""Regenerated (self) noise of a grille, register or diffuser.

    Reynolds's estimate as Long Eqs. 13.27 to 13.33, for when the
    manufacturer's ASHRAE Standard 70 data is not to hand. The overall sound
    power level is Eq. 13.27:

    .. math::

       L_W = 10 \log_{10} S_\mathrm{G} + 30 \log_{10} \xi + 60 \log_{10} U_\mathrm{G} - 31.3

    with ``S_\mathrm{G}`` the face area of the device (ft2),
    :math:`U_\mathrm{G} = Q / (60 S_\mathrm{G})` the approach velocity (ft/s) and
    :math:`\xi = 334.9\, dP / (\rho_0 U_\mathrm{G}^2)` the normalised pressure-drop
    coefficient of Eq. 13.28 (``dP`` in inches of water gauge,
    :math:`\rho_0 = 0.075` lb/ft3); this function takes and returns SI and
    converts internally.

    The octave-band spectrum follows from Eq. 13.29,
    :math:`L_{W,\mathrm{oct}} = L_W + C_\mathrm{D}`, with the shape functions of Eqs. 13.30
    and 13.31:

    .. math::

       C_\mathrm{D} = -5.82 - 0.15 A - 1.13 A^2 \qquad \text{(round)}

       C_\mathrm{D} = -11.82 - 0.15 A - 1.13 A^2
       \qquad \text{(rectangular, including slot)}

    normalised to the peak frequency :math:`f_P = 48.8 U_\mathrm{G}` of Eq. 13.32,
    where :math:`A = N_B(f_P) - N_B(f)` is the distance in octaves from the
    peak band (Eq. 13.33) counted on Long's band numbering, 0 at 32 Hz.

    The sixth power of velocity in Eq. 13.27 is the design message: the level
    rises about 18 dB for every doubling of the approach velocity, and for a
    given air volume doubling the face area buys about 15 dB. Nothing
    downstream can take that noise back out, because there is no ductwork
    left, which is why the terminal device usually sets the room criterion in
    the mid and high bands.

    Several identical devices serving the same room add :math:`10 \log_{10} n`,
    which is what ``count`` applies.

    :param frequencies: Octave-band centres, Hz; ``None`` uses
        :data:`OCTAVE_BANDS`.
    :param face_area: Cross-sectional face area ``S_\mathrm{G}`` of one device, m2.
    :param volume_flow: Volume flow ``Q`` through one device, m3/s.
    :param pressure_drop: Static pressure drop ``dP`` across the device, Pa.
    :param shape: ``"rectangular"`` (Eq. 13.31, includes slot diffusers) or
        ``"round"`` (Eq. 13.30).
    :param count: Number of identical devices ``n`` in the room.
    :return: An :class:`HvacSpectrumResult` of the band sound power level,
        dB re 1e-12 W.
    :raises ValueError: If a dimension is not positive, ``count`` is not a
        positive integer or ``shape`` is unknown.
    """
    f = OCTAVE_BANDS.copy() if frequencies is None else _frequencies(frequencies)
    area_ft2 = require_positive(face_area, "face_area") / _M_PER_FT**2
    flow_cfm = require_positive(volume_flow, "volume_flow") / _M3S_PER_CFM
    drop_in_wg = require_positive(pressure_drop, "pressure_drop") / _PA_PER_IN_WG
    profile = require_choice(shape, "shape", ("rectangular", "round"))
    if count <= 0 or not float(count).is_integer():
        msg = "'count' must be a positive integer."
        raise ValueError(msg)
    # Eq. 13.28: Long prints "ft/min" under U_\mathrm{G} but defines it in the same
    # breath as Q / (60 S_\mathrm{G}), which is ft/s, the unit Eq. 13.27 declares and
    # the only one for which the 334.9 constant is the velocity-pressure
    # relation (see docs/ERRATA.md).
    velocity = flow_cfm / (60.0 * area_ft2)
    xi = 334.9 * drop_in_wg / (0.075 * velocity**2)
    overall = (
        10.0 * np.log10(area_ft2)
        + 30.0 * np.log10(xi)
        + 60.0 * np.log10(velocity)
        - 31.3
        + 10.0 * np.log10(float(count))
    )
    # Eqs. 13.32 and 13.33: the distance in octaves from the peak band.
    peak_band = float(np.round(np.log2(48.8 * velocity / 32.0)))
    a = peak_band - _octave_band_number(f)
    base = -5.82 if profile == "round" else -11.82
    return HvacSpectrumResult(
        frequencies=f,
        values=overall + base - 0.15 * a - 1.13 * a**2,
        quantity="sound_power_level",
        label=f"Diffuser self-noise (U = {velocity * _M_PER_FT:.2f} m/s)",
    )


def air_terminal_velocity_limit(
    design_criterion: float, *, opening: str = "supply"
) -> float:
    """Maximum recommended neck velocity of a diffuser or register.

    ASHRAE (2019) *HVAC Applications Handbook* Chapter 49, Table 9: the "free"
    opening airflow velocity not to be exceeded if the room is to reach a given
    design ``RC(N)``, for use when no sound data is available for the selected
    device. It is a screening check, not a spectrum: the sound power of a real
    grille, register or diffuser comes from manufacturer data measured to
    ASHRAE Standard 70, and :func:`diffuser_sound_power` estimates it when that
    data is not to hand. Several devices in the same room, or a damper
    throttled in the neck, raise the level further and the allowable velocity
    has to be reduced accordingly.

    :param design_criterion: Design ``RC(N)`` of the room; one of 25, 30, 35,
        40 or 45.
    :param opening: ``"supply"`` (supply air outlet) or ``"return"`` (return
        air opening).
    :return: The maximum recommended neck velocity, m/s.
    :raises ValueError: If the design criterion is not tabulated.
    """
    side = require_choice(opening, "opening", ("supply", "return"))
    table = _TERMINAL_VELOCITY_LIMIT[side]
    # The finiteness test keeps a NaN or infinite criterion out of round(),
    # whose own refusal names neither the parameter nor the tabulated values.
    if not math.isfinite(design_criterion) or round(design_criterion) not in table:
        msg = (
            f"'design_criterion' must be one of {sorted(table)}; "
            f"got {design_criterion!r}."
        )
        raise ValueError(msg)
    return table[round(design_criterion)]


def air_terminal_damper_correction(
    pressure_ratio: float, *, location: str = "diffuser_neck"
) -> float:
    """Level to add to a diffuser sound rating for a throttled volume damper.

    ASHRAE (2019) *HVAC Applications Handbook* Chapter 49, Table 10. A balancing
    damper throttled in the neck of a diffuser turns the pressure it drops into
    noise right at the outlet, where the room hears it: at a damper pressure
    ratio of 3 the published penalty is 15 dB in the neck, 5 dB in the inlet
    plenum and 2 dB when the damper sits at least 1.5 m back in the supply duct.
    That ordering is the whole design rule: throttle far from the outlet, or
    balance the system with duct sizing instead.

    The table is interpolated linearly between its tabulated pressure ratios
    (1.5 to 6) and held flat outside them.

    :param pressure_ratio: Damper pressure ratio, the total pressure drop across
        the damper divided by the pressure drop of the outlet itself.
    :param location: Where the damper sits: ``"diffuser_neck"`` (in the neck of
        a linear diffuser), ``"plenum_inlet"`` (in the inlet of the plenum of a
        linear diffuser) or ``"supply_duct"`` (in the supply duct at least 1.5 m
        from the inlet plenum).
    :return: The level to add to the diffuser's rated sound power, dB.
    :raises ValueError: If ``pressure_ratio`` is not positive or ``location`` is
        unknown.
    """
    ratio = require_positive(pressure_ratio, "pressure_ratio")
    key = require_choice(location, "location", tuple(_DAMPER_CORRECTION))
    return float(np.interp(ratio, _DAMPER_PRESSURE_RATIOS, _DAMPER_CORRECTION[key]))


# ---------------------------------------------------------------------------
# Room effect
# ---------------------------------------------------------------------------
def _room_measure(
    room_constant: ArrayLike | None, absorption_area: ArrayLike | None
) -> NDArray[np.float64]:
    """The reverberant-field denominator, from whichever measure was given.

    :raises ValueError: unless exactly one of the two was given, or if it is
        not a positive, finite scalar or 1-D spectrum.
    """
    if room_constant is not None and absorption_area is None:
        name, value = "room_constant", room_constant
    elif absorption_area is not None and room_constant is None:
        name, value = "absorption_area", absorption_area
    else:
        got = "both" if room_constant is not None else "neither"
        msg = (
            "Give exactly one of 'room_constant' (R = S alpha / (1 - alpha)) "
            f"and 'absorption_area' (A = S alpha); got {got}."
        )
        raise ValueError(msg)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim > 1 or arr.size == 0:
        msg = f"'{name}' must be a scalar or a non-empty 1-D array."
        raise ValueError(msg)
    if np.any(arr <= 0.0) or not np.all(np.isfinite(arr)):
        msg = f"'{name}' must be positive and finite."
        raise ValueError(msg)
    return arr


@overload
def room_effect(
    distance: float,
    room_constant: ArrayLike,
    *,
    directivity: ArrayLike = ...,
) -> np.ndarray | float: ...


@overload
def room_effect(
    distance: float,
    *,
    absorption_area: ArrayLike,
    directivity: ArrayLike = ...,
) -> np.ndarray | float: ...


def room_effect(
    distance: float,
    room_constant: ArrayLike | None = None,
    *,
    absorption_area: ArrayLike | None = None,
    directivity: ArrayLike = 2.0,
) -> np.ndarray | float:
    r"""Room effect: the drop from the terminal sound power to the room level.

    The last step of a duct-path calculation turns the sound power arriving at
    the terminal device into a sound pressure level at the listener, through
    the steady-state room relation
    :math:`L_p = L_W + 10 \log_{10}[Q / (4 \pi r^2) + 4 / R]` (Long Eq. 14.40; Bies
    Eq. (6.43), :func:`phonometry.room.steady_state_spl`). This function
    returns the *attenuation*, the positive number
    :math:`-10 \log_{10}[Q / (4 \pi r^2) + 4 / R]`, so it drops into a duct-path
    cascade beside every other loss; Long's worked sheets print it as the
    negative level change. A ceiling diffuser radiates into a half space,
    hence the default :math:`Q = 2`.

    VDI 2081 Blatt 1 Section 6.7.3 closes its own chain with the same
    expression, written in the equivalent absorption area ``A`` rather than
    the room constant ``R`` (Equation (36)), and prints the result as the
    *Raumdämpfung* :math:`L_W - L_p`, which is what this returns. The two
    measures are not interchangeable, so each has its own argument; and the
    guideline's directivity comes from a chart against frequency, which is why
    ``Q`` may be given one value per band.

    :param distance: Terminal-to-listener distance ``r``, m.
    :param room_constant: Room constant :math:`R = S \alpha / (1 - \alpha)`, m2
        (scalar or per-band; from :func:`phonometry.room.room_constant`). Give
        this or ``absorption_area``, not both.
    :param absorption_area: Equivalent absorption area :math:`A = S \alpha`, m2
        (scalar or per-band; from
        :func:`phonometry.room.equivalent_absorption_area`). Give this or
        ``room_constant``, not both.
    :param directivity: Directivity factor ``Q`` of the terminal device
        (``2`` flush in a ceiling or wall, ``4`` at an edge, ``8`` in a
        corner), scalar or per-band.
    :return: The room effect as a positive attenuation, dB (a float when every
        input is scalar, otherwise a per-band array).
    :raises ValueError: unless exactly one absorption measure is given, or if
        an argument is not positive and finite.
    """
    r = require_positive(distance, "distance")
    measure = _room_measure(room_constant, absorption_area)
    q = np.asarray(directivity, dtype=np.float64)
    if np.any(q <= 0.0) or not np.all(np.isfinite(q)):
        msg = "'directivity' must be positive and finite."
        raise ValueError(msg)
    values = -10.0 * np.log10(q / (4.0 * np.pi * r**2) + 4.0 / measure)
    return float(values) if values.ndim == 0 else values
