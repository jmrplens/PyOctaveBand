#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
HVAC duct acoustics: fan power, duct losses, plenums and flow-generated noise.

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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.validation import require_choice, require_positive
from ..room.steady_field import room_constant

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray

    from .._report.metadata import ReportMetadata

_C_AIR = 343.0

#: The eight octave bands of the ASHRAE / Long duct-borne noise calculation, Hz.
OCTAVE_BANDS: NDArray[np.float64] = np.array(
    [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
)

# Imperial-to-SI conversions. The Reynolds regressions and the ASHRAE fan
# equation are unit-sensitive empirical fits stated in foot-pound units, so the
# SI arguments of this module are converted before the published constants are
# applied and never the other way round.
_M_PER_FT = 0.3048
_M_PER_IN = 0.0254
_M3S_PER_CFM = 0.0004719474432  # 1 ft3/min in m3/s
_PA_PER_IN_WG = 249.0  # Long Eq. 13.1 reference pressure P_REF

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
        raise ValueError("'frequencies' must be a non-empty 1-D array.")
    if np.any(f <= 0.0) or not np.all(np.isfinite(f)):
        raise ValueError("'frequencies' must be positive and finite.")
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
        raise ValueError(
            "'frequencies' must be octave-band centres of "
            f"{bands.tolist()} Hz for this tabulated method."
        )
    return f, idx


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
    quantity: str
    label: str

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
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
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
    :param method: ``"bies"`` (Table 8.14 look-up) or ``"long"`` (closed form).
    :param speed_of_sound: Speed of sound ``c``, m/s (used by the closed form;
        the table is indexed by frequency directly).
    :return: A :class:`HvacSpectrumResult` of the reflection loss, dB.
    """
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
        raise ValueError("'termination' must be 'flush' or 'free'.")
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


def elbow_insertion_loss(
    frequencies: ArrayLike,
    width: float,
    *,
    bend_type: str = "square",
    vanes: bool = False,
    lined: bool = False,
    speed_of_sound: float = _C_AIR,
) -> HvacSpectrumResult:
    r"""Duct bend/elbow insertion loss per bend (Bies Table 8.11, ASHRAE).

    Indexed by the frequency-to-width ratio :math:`W / \lambda`
    (:math:`\lambda = c / f`).
    Lined bends assume the lining extends at least three duct diameters up- and
    downstream. Round bends are treated as unlined with no vanes.

    :param frequencies: Frequencies ``f``, Hz (1-D array).
    :param width: Duct width ``W`` in the plane of the bend, m.
    :param bend_type: ``"square"`` or ``"round"``.
    :param vanes: Turning vanes fitted (square bends only).
    :param lined: Acoustically lined bend (square bends only).
    :param speed_of_sound: Speed of sound ``c``, m/s.
    :return: A :class:`HvacSpectrumResult` of the insertion loss, dB per bend.
    """
    f = _frequencies(frequencies)
    w = require_positive(width, "width")
    c = require_positive(speed_of_sound, "speed_of_sound")
    if bend_type == "round":
        if vanes or lined:
            raise ValueError("round bends take neither vanes nor lining.")
        key = "round"
    elif bend_type == "square":
        key = "square" + ("_vanes" if vanes else "") + ("_lined" if lined else "")
    else:
        raise ValueError("'bend_type' must be 'square' or 'round'.")
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
        outlet, rad (default 0).
    :return: The transmission loss, dB (float for scalar absorption, else a
        per-band array).
    """
    s_out = require_positive(exit_area, "exit_area")
    r = require_positive(line_of_sight, "line_of_sight")
    s_w = require_positive(wall_area, "wall_area")
    alpha = np.asarray(mean_absorption, dtype=np.float64)
    if np.any(alpha <= 0.0) or np.any(alpha >= 1.0) or not np.all(np.isfinite(alpha)):
        raise ValueError("'mean_absorption' must lie strictly in (0, 1).")
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
    r"""Flow-generated octave-band sound power of a straight duct (Bies Eq. (8.251)).

    .. math::

       L_{W\mathrm{B}} = 7 + 50 \log_{10}(U) + 10 \log_{10}(S) - 2
       - 26 \log_{10}(1.14 + 0.02 f / U)

    in dB re 1e-12 W (VDI 2081-1), for airflow speed ``U`` in a duct of area
    ``S``.

    :param frequencies: Octave-band centre frequencies ``f``, Hz (1-D array).
    :param flow_velocity: Mean flow speed ``U``, m/s.
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


def flow_noise_bend(
    frequencies: ArrayLike,
    flow_velocity: float,
    area: float,
    height: float,
    *,
    density: float = 1.206,
) -> HvacSpectrumResult:
    r"""Flow-generated octave-band sound power of a mitred bend (Bies Eqs. (8.252), (8.254)).

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

    :param frequencies: Octave-band centre frequencies ``f``, Hz (1-D array).
    :param flow_velocity: Mean flow speed ``U``, m/s.
    :param area: Duct cross-sectional area ``S``, m2.
    :param height: Duct height ``H`` in the plane of the bend, m.
    :param density: Air density ``rho``, kg/m3.
    :return: A :class:`HvacSpectrumResult` of the band sound power level,
        dB re 1e-12 W.
    """
    f = _frequencies(frequencies)
    u = require_positive(flow_velocity, "flow_velocity")
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
    if blades <= 0:
        raise ValueError("'blades' must be a positive integer.")
    return rpm * float(blades) / 60.0


def fan_efficiency_correction(relative_efficiency: float) -> float:
    """Off-peak efficiency correction ``C_EFF`` (Long Table 13.6).

    A fan running away from its peak static efficiency is noisier at the same
    duty. The correction is a step function of the static efficiency expressed
    as a percentage of the peak (Long Eq. 13.3): 90 per cent of peak and above
    adds nothing, and anything below 50 per cent adds 16 dB. When the peak
    efficiency is unknown Long recommends assuming 80 per cent, which lands in
    the 6 dB step.

    :param relative_efficiency: Static efficiency as a percentage of the peak,
        in ``(0, 100]``.
    :return: The correction ``C_EFF``, dB.
    :raises ValueError: If the efficiency is not in ``(0, 100]``.
    """
    eta = require_positive(relative_efficiency, "relative_efficiency")
    if eta > 100.0:
        raise ValueError("'relative_efficiency' must not exceed 100 per cent.")
    for lower, correction in _EFFICIENCY_CORRECTION:
        if eta >= lower:
            return correction
    return _EFFICIENCY_CORRECTION[-1][1]  # pragma: no cover - the last edge is 0


def fan_sound_power(
    volume_flow: float,
    static_pressure: float,
    *,
    fan_type: str = "forward_curved",
    relative_efficiency: float = 80.0,
    blade_frequency: float | None = None,
    frequencies: ArrayLike | None = None,
) -> HvacSpectrumResult:
    r"""Octave-band fan sound power from the operating point (Long Eq. 13.1).

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

    :param volume_flow: Volume flow through the fan ``Q_\mathrm{F}``, m3/s.
    :param static_pressure: Fan static pressure ``P_\mathrm{F}``, Pa (gauge).
    :param fan_type: One of ``"airfoil_large"`` / ``"airfoil_small"``
        (backward-curved or backward-inclined centrifugal wheels above and
        below 36 in diameter), ``"forward_curved"``, ``"radial_low"`` /
        ``"radial_medium"`` / ``"radial_high"`` (radial blades by total
        pressure), ``"vaneaxial_hub_low"`` / ``"vaneaxial_hub_medium"`` /
        ``"vaneaxial_hub_high"`` (hub ratios 0.3-0.4, 0.4-0.6 and 0.6-0.8),
        ``"tubeaxial_large"`` / ``"tubeaxial_small"`` (above and below 40 in
        wheel diameter) or ``"propeller"``.
    :param relative_efficiency: Static efficiency as a percentage of the peak
        (default 80, Long's recommendation when the peak is unknown).
    :param blade_frequency: Blade passing frequency ``f_bp``, Hz (from
        :func:`blade_passing_frequency`). ``None`` (default) places the
        increment in the octave band Table 13.7 tabulates for the fan type.
    :param frequencies: Octave-band centres, Hz; ``None`` (default) uses the
        63 Hz to 8 kHz bands of :data:`OCTAVE_BANDS`.
    :return: An :class:`HvacSpectrumResult` of the band sound power level,
        dB re 1e-12 W.
    """
    kind = require_choice(fan_type, "fan_type", tuple(_FAN_LEVEL_CORRECTION))
    q = require_positive(volume_flow, "volume_flow")
    p = require_positive(static_pressure, "static_pressure")
    f, idx = _octave_slots(frequencies)

    duty = 10.0 * np.log10(q * 1000.0 / 0.472) + 10.0 * np.log10(p / _PA_PER_IN_WG)
    c_eff = fan_efficiency_correction(relative_efficiency)
    band, c_bfi = _FAN_BLADE_INCREMENT[kind]
    if blade_frequency is not None:
        f_bp = require_positive(blade_frequency, "blade_frequency")
        nearest = int(np.argmin(np.abs(np.log2(f_bp / OCTAVE_BANDS))))
        band = float(OCTAVE_BANDS[nearest])
    increment = np.where(np.abs(np.log2(OCTAVE_BANDS / band)) < 0.5, c_bfi, 0.0)

    lw = _FAN_LEVEL_CORRECTION[kind][idx] + duty + c_eff + increment[idx]
    return HvacSpectrumResult(
        frequencies=f,
        values=lw,
        quantity="sound_power_level",
        label=f"Fan ({kind.replace('_', ' ')}, {q * 3600:.0f} m3/h, {p:.0f} Pa)",
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

    :param frequencies: Octave-band centre frequencies ``f``, Hz (1-D array).
    :param width: Duct width, m.
    :param height: Duct height, m.
    :param length: Duct run length ``l``, m.
    :param wrapped: The duct is externally wrapped with a fibreglass blanket,
        which doubles the 63 Hz to 250 Hz attenuation.
    :return: An :class:`HvacSpectrumResult` of the attenuation, dB.
    """
    f = _frequencies(frequencies)
    ps = _perimeter_over_area(width, height)
    ell = require_positive(length, "length") / _M_PER_FT
    low = (
        17.0 * ps**0.25 * f**-0.85 * ell
        if ps >= 3.0
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
) -> HvacSpectrumResult:
    """Attenuation of an unlined circular sheet-metal duct (Long Table 14.1).

    A circular duct is far stiffer than a rectangular one in its breathing
    mode, so the sound field can hardly excite it: the loss is about a tenth of
    the rectangular value and is tabulated as a length rate alone, 0.03 dB/ft
    up to 250 Hz and 0.05 to 0.07 dB/ft above. The published table stops at
    4 kHz; the 4 kHz rate is held for the 8 kHz band.

    :param frequencies: Octave-band centres, Hz; ``None`` uses
        :data:`OCTAVE_BANDS`.
    :param length: Duct run length, m.
    :return: An :class:`HvacSpectrumResult` of the attenuation, dB.
    """
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


def split_loss(
    main_area: float,
    branch_areas: ArrayLike,
    *,
    branch: int = 0,
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

    :param main_area: Cross-sectional area of the main feeder duct ``S_m``, m2.
    :param branch_areas: Areas ``S_i`` of the branches continuing on from the
        main duct, m2 (1-D array-like).
    :param branch: Index into ``branch_areas`` of the branch being followed.
    :return: The split loss, dB (positive).
    :raises ValueError: If the areas are not positive or ``branch`` is out of
        range.
    """
    s_m = require_positive(main_area, "main_area")
    areas = np.atleast_1d(np.asarray(branch_areas, dtype=np.float64))
    if areas.ndim != 1 or areas.size == 0:
        raise ValueError("'branch_areas' must be a non-empty 1-D array.")
    if np.any(areas <= 0.0) or not np.all(np.isfinite(areas)):
        raise ValueError("'branch_areas' must be positive and finite.")
    if not 0 <= branch < areas.size:
        raise ValueError(f"'branch' must index 'branch_areas' (0..{areas.size - 1}).")
    total = float(np.sum(areas))
    reflection = 1.0 - ((total - s_m) / (total + s_m)) ** 2
    return float(-10.0 * np.log10(reflection) - 10.0 * np.log10(areas[branch] / total))


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
        raise ValueError("'airway_widths' must be a non-empty 1-D array.")
    if np.any(widths <= 0.0) or not np.all(np.isfinite(widths)):
        raise ValueError("'airway_widths' must be positive and finite.")
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
    """
    f, idx = _octave_slots(frequencies)
    v = require_positive(airway_velocity, "airway_velocity")
    if passages <= 0:
        raise ValueError("'passages' must be a positive integer.")
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
    if count <= 0:
        raise ValueError("'count' must be a positive integer.")
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
    key = round(design_criterion)
    if key not in table:
        raise ValueError(
            f"'design_criterion' must be one of {sorted(table)}; "
            f"got {design_criterion!r}."
        )
    return table[key]


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
def room_effect(
    distance: float,
    room_constant: ArrayLike,
    *,
    directivity: float = 2.0,
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

    :param distance: Terminal-to-listener distance ``r``, m.
    :param room_constant: Room constant :math:`R = S \alpha / (1 - \alpha)`, m2
        (scalar or per-band; from :func:`phonometry.room.room_constant`).
    :param directivity: Directivity factor ``Q`` of the terminal device
        (``2`` flush in a ceiling or wall, ``4`` at an edge, ``8`` in a corner).
    :return: The room effect as a positive attenuation, dB (a float for a
        scalar room constant, otherwise a per-band array).
    """
    r = require_positive(distance, "distance")
    q = require_positive(directivity, "directivity")
    r_const = np.asarray(room_constant, dtype=np.float64)
    if np.any(r_const <= 0.0) or not np.all(np.isfinite(r_const)):
        raise ValueError("'room_constant' must be positive and finite.")
    values = -10.0 * np.log10(q / (4.0 * np.pi * r**2) + 4.0 / r_const)
    return float(values) if values.ndim == 0 else values
