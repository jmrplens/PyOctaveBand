#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Sound power level of a noise source by sound-intensity **scanning**:
ISO 9614-2:1996 (engineering, grade 2; survey/control, grade 3) and
ISO 9614-3:2002 (precision, grade 1).

A probe is swept continuously over each segment of a hypothetical surface that
encloses the source, reporting the time-averaged signed normal intensity
:math:`\langle I_{\mathrm{n},i} \rangle` and mean-square pressure per segment. The
sound power follows from the partial powers
:math:`P_i = \langle I_{\mathrm{n},i} \rangle S_i` summed over the ``N`` segments
(clause 9, equations (5), (6), (12), (13)):

.. math::

   P_i = \langle I_{\mathrm{n},i} \rangle \, S_i \tag{Eq. 12}

   P = \sum_i P_i \tag{Eq. 6}

   L_W = 10 \log_{10}\frac{P}{P_0}, \qquad P_0 = 10^{-12}~\text{W} \tag{Eq. 13}

The method is **not applicable to any band in which** :math:`P < 0`
(clause 9.2): a strong parasitic source outside the surface makes the net
energy flow inward and the determination invalid for that band.

Two scanning-method field indicators qualify the determination, with
:math:`[L_p]` the area-weighted surface sound pressure level (Annex A,
normative):

.. math::

   F_{pI} = [L_p] - L_W + 10 \log_{10}\frac{S}{S_0} \tag{Eq. A.1}

   [L_p] = 10 \log_{10}\!\left[ \frac{1}{S} \sum_i S_i \, 10^{0.1 L_{pi}} \right]

   F_{+/-} = 10 \log_{10}\frac{\sum_i |P_i|}{\left| \sum_i P_i \right|}
   \tag{Eq. A.2}

``FpI`` is the surface pressure-intensity indicator (equivalent to ISO 9614-1
``F3`` for uniform-area segments, Note 14); ``F+/-`` the negative-partial-power
indicator (equivalent to ISO 9614-1 ``F3-F2``, Note 15). Because Part 2 weights
by segment area ``Si`` while :func:`phonometry.emission.field_indicators` (ISO
9614-1) assumes equal-area positions, the indicators are computed directly
here; only the dynamic-capability index :math:`L_\mathrm{d} = \delta_{pI0} - K`
is shared with :func:`phonometry.emission.dynamic_capability_index`.

Qualification criteria per band (Annex B), where ``K`` is 10 (engineering) or
7 (survey) per Table 1, criterion 2 is mandatory for grade 2 and optional for
grade 3, and the per-segment repeatability limit ``s`` comes from Table 2:

.. math::

   \text{criterion 1:} \quad L_\mathrm{d} > F_{pI}, \qquad L_\mathrm{d} = \delta_{pI0} - K

   \text{criterion 2:} \quad F_{+/-} \le 3~\text{dB}

   \text{criterion 3:} \quad |L_{Wi}(1) - L_{Wi}(2)| \le s
   \quad \text{per segment}

A band achieves the **engineering** grade when criteria 1, 2 and 3 hold, the
**survey** grade when criteria 1 and 3 hold (clause 8.4), otherwise none.
An A-weighted sound power level omits, besides the non-determinable
:math:`P \le 0` bands, the bands in which criteria 1 and/or 2 are not
satisfied (clause 10.6 b).

ISO 9614-3:2002 is the same method at precision grade, and that is why it is
filed here and not in a module of its own: the same probe swept over the same
enclosing surface, the same partial powers summed the same way (equations (5),
(8), (9)), only a stricter procedure around them. Part 3 recognises a single
grade, fixes the bias-error factor at :math:`K = 10` dB, takes as its input the
result of the two scans that its repeatability criterion compares, and refers
the level to the reference atmosphere:

.. math::

   L_{W0} = L_W - 15 \log_{10}\!\left( \frac{B}{101325} \cdot
   \frac{296.15}{273.15 + \theta} \right) \tag{Eq. 10}

Qualification is stricter in the same proportion: four field indicators
(Annex B) feed five acceptance criteria evaluated per band (Annex C), and a
band that satisfies the scan-density criterion 5 is qualified as a final
result even where the field non-uniformity of criterion 4 is not met
(C.1.6.2). The exclusion of the net-negative bands is the one rule both parts
state alike (clause 9.2).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

    from .._report.metadata import ReportMetadata

from .._internal.levels_math import energy_mean, weighted_energy_mean
from .._internal.validation import (
    check_engine,
    require_equal_shapes,
    require_finite_fields,
    require_per_band,
    require_positive_array,
    require_ranks,
    require_same_length,
)
from ._shared import SoundPowerWarning, _a_weighting_corrections, _check_grade
from .intensity import dynamic_capability_index

_P0 = 1.0e-12  #: Reference sound power, in watts (ISO 9614-2, 3.6.3).
_S0 = 1.0  #: Reference surface area, in square metres (ISO 9614-2, A.2.1).

#: Rejection message for a 'frequencies' vector that does not span the bands.
_FREQUENCIES_BAND_COUNT_MSG = "'frequencies' length must match the number of bands."

Grade = Literal["engineering", "survey"]
BandType = Literal["octave", "third"]

#: Deviation error factor K, in dB (ISO 9614-2:1996 Table 1).
_K: dict[str, float] = {"engineering": 10.0, "survey": 7.0}
#: Criterion 2 limit on the negative-partial-power indicator (Eq. B.2), in dB.
_F_PLUS_MINUS_LIMIT = 3.0
#: Grade-3 (control) per-band repeatability limit s, in dB. Table 2 tabulates
#: per-band s only for grade 2; grade 3 carries only the A-weighted value
#: (4 dB), reused here as a per-band survey limit (extrapolated -- the
#: per-band grade-3 use of the A-weighted 4 dB is non-normative).
_S_SURVEY = 4.0
#: Minimum measurement segments of a scan (ISO 9614-2:1996 clause 8.2); fewer
#: segments only warns, it does not reject.
_MIN_SEGMENTS = 4


def _table2_s(nominal: int, band_type: BandType) -> float:
    """Grade-2 reproducibility standard deviation s (ISO 9614-2 Table 2), in dB.

    Octave 63-125 Hz / third 50-160 Hz -> 3; octave 250 Hz / third 200-315 Hz
    -> 2; octave 500-4000 Hz / third 400-5000 Hz -> 1,5; third 6300 Hz -> 2,5.
    """
    if band_type == "octave":
        table: dict[int, float] = {
            63: 3.0,
            125: 3.0,
            250: 2.0,
            500: 1.5,
            1000: 1.5,
            2000: 1.5,
            4000: 1.5,
        }
    else:
        table = {
            50: 3.0,
            63: 3.0,
            80: 3.0,
            100: 3.0,
            125: 3.0,
            160: 3.0,
            200: 2.0,
            250: 2.0,
            315: 2.0,
            400: 1.5,
            500: 1.5,
            630: 1.5,
            800: 1.5,
            1000: 1.5,
            1250: 1.5,
            1600: 1.5,
            2000: 1.5,
            2500: 1.5,
            3150: 1.5,
            4000: 1.5,
            5000: 1.5,
            6300: 2.5,
        }
    if nominal not in table:
        msg = (
            f"No ISO 9614-2 Table 2 standard deviation for {nominal} Hz "
            f"({band_type}); expected a nominal band centre in the qualified "
            "range (Table 2)."
        )
        raise ValueError(msg)
    return table[nominal]


@dataclass(frozen=True)
class SoundPowerIntensityResult:
    r"""Result of an ISO 9614-2:1996 sound-power-by-scanning determination.

    ``partial_power`` is the signed :math:`P_i = \langle I_{\mathrm{n},i} \rangle S_i`
    per segment and band (Eq. 12); ``partial_power_level`` the magnitude level
    :math:`10 \log_{10}(|P_i|/P_0)` (Eq. 8), with the sign carried by
    ``partial_power``. ``sound_power`` is the signed band total
    :math:`P = \sum P_i` (Eq. 6) and ``sound_power_level`` its level
    :math:`10 \log_{10}(P/P_0)` (Eq. 13), ``NaN`` where :math:`P \le 0`
    (``negative_band`` True, method not applicable, clause 9.2).
    ``surface_pressure_intensity_index``
    (FpI, Eq. A.1) and ``negative_partial_power_index`` (F+/-, Eq. A.2) are
    per band, ``None`` when the inputs they need are absent. ``repeatability``
    is :math:`|L_{Wi}(1) - L_{Wi}(2)|` per segment and band (criterion 3),
    ``None`` without
    a second scan; it is :math:`+\infty` where the two sweeps reverse the flow
    direction on a segment (opposite-sign partial powers), a gross
    non-repeatability that criterion 3 must reject even when the magnitudes
    happen to match. ``dynamic_capability_index`` is ``Ld`` for the requested
    grade. ``achieved_grade`` is the per-band class ``'engineering'``/
    ``'survey'``/``'none'`` (clause 8.4), ``None`` when the qualifying inputs
    (``delta_pI0`` and a second scan) are absent. ``sound_power_level_a`` is the
    A-weighted total over determinable bands (``NaN`` without ``frequencies``
    and more than one band), which omits the bands failing criteria 1 and/or 2
    (clause 10.6 b) whenever those criteria are evaluable.
    ``a_weighting_omitted_bands`` flags the bands so omitted (per band,
    ``True`` = omitted); it is ``None`` when the criteria inputs
    (``pressure_levels`` and ``pressure_residual_index``) are absent, in which
    case every determinable band is summed and a warning is emitted.
    """

    frequencies: np.ndarray | None
    partial_power: np.ndarray
    partial_power_level: np.ndarray
    sound_power: np.ndarray
    sound_power_level: np.ndarray
    negative_band: np.ndarray
    surface_pressure_intensity_index: np.ndarray | None
    negative_partial_power_index: np.ndarray | None
    repeatability: np.ndarray | None
    dynamic_capability_index: np.ndarray | None
    achieved_grade: np.ndarray | None
    surface_area: float
    sound_power_level_a: float
    a_weighting_omitted_bands: np.ndarray | None
    grade: str

    def __post_init__(self) -> None:
        """Reject a determination whose per-band quantities disagree.

        The field indicators reach the fiche's measurement-basis strip as a
        range rather than as a column, so a short or long indicator array
        prints a range taken over the wrong set of bands, on a sheet whose
        next sentence turns that range into a qualification verdict. Nothing
        downstream can catch that, because a range is well formed whatever it
        was taken over.

        ``surface_area`` must also be finite. The scanned surface is the sum
        of the segment areas, every one of which the determination refuses
        unless it is positive and finite, so no scan can hand back a surface
        that is not a number; the per-band levels are another matter, and
        ``sound_power_level`` and ``sound_power_level_a`` stay unpinned here
        because clause 9.2 makes ``NaN`` their reading for a band of negative
        net power, which the fiche prints as an em dash and leaves out of its
        totals. An unpinned surface reached the boxed result of the sheet as
        the literal "Measurement surface S = nan m2".

        :raises ValueError: if any per-band quantity disagrees with the rest,
            or ``surface_area`` is not finite.
        """
        require_ranks(
            self,
            frequencies=1,
            partial_power=2,
            partial_power_level=2,
            sound_power=1,
            sound_power_level=1,
            negative_band=1,
            surface_pressure_intensity_index=1,
            negative_partial_power_index=1,
            repeatability=2,
            dynamic_capability_index=1,
            achieved_grade=1,
            a_weighting_omitted_bands=1,
        )
        require_same_length(
            self,
            "frequencies",
            ("partial_power", 1),
            ("partial_power_level", 1),
            "sound_power",
            "sound_power_level",
            "negative_band",
            "surface_pressure_intensity_index",
            "negative_partial_power_index",
            ("repeatability", 1),
            "dynamic_capability_index",
            "achieved_grade",
            "a_weighting_omitted_bands",
        )
        require_same_length(
            self,
            "partial_power",
            "partial_power_level",
            "repeatability",
            axis="measurement segment",
        )
        require_finite_fields(self, "surface_area")

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the LW spectrum; non-positive bands are hatched as unusable.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.emission import plot_sound_power

        check_language(language)
        return plot_sound_power(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an ISO 9614-2 sound-power-by-intensity determination fiche.

        Writes a one-page sound-power test sheet: the standard-basis line naming
        the intensity-scanning method and its measurement grade (ISO 9614-2:1996
        engineering grade 2 or survey grade 3), an optional metadata header
        (client, noise source, test environment, instrumentation, climate,
        date), a per-band table (nominal octave/one-third-octave frequency and
        the intensity-derived band sound-power level ``LW``), the sound-power
        spectrum ``LW(f)`` with net-negative bands hatched as unusable, the
        boxed A-weighted sound power level ``LWA`` (dB re 1 pW) with the total
        ``LW``, the measurement surface area ``S`` and the determination grade,
        an optional verdict row against a declared limit, and a
        measurement-basis strip stating the partial-power model, the field
        indicators (``FpI``, ``F+/-``) and the Annex B qualification criteria.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the header (``client``, ``specimen`` the noise source, ``test_room``
            the test environment, ``instrumentation``, ``temperature``,
            ``relative_humidity``, ``pressure``, ``test_date``), the footer
            identity (``laboratory``, ``operator``, ``report_id``, ``notes``)
            and, via ``requirement``, a declared A-weighted sound-power limit
            the fiche checks the result against (lower is better).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True`` the per-band table adds the field
            indicators ``FpI`` and ``F+/-`` and the per-band achieved grade.
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
        from .._report.iso9614 import render_intensity_power_report

        return render_intensity_power_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def _level_magnitude(values: np.ndarray) -> np.ndarray:
    r"""Magnitude level :math:`10 \log_{10}(|P_i|/P_0)` in dB, with a tiny-floor
    guard for zeros.
    """
    guarded = np.maximum(np.abs(values), np.finfo(float).tiny)
    return np.asarray(10.0 * np.log10(guarded / _P0), dtype=np.float64)


def _as_2d(name: str, arr: np.ndarray, n_seg: int, n_bands: int) -> np.ndarray:
    a = np.atleast_2d(np.asarray(arr, dtype=np.float64))
    if a.shape == (1, n_seg) and n_bands != n_seg:
        a = a.T  # a 1D (N_seg,) input arrives as (1, N_seg)
    if a.shape != (n_seg, n_bands):
        msg = (
            f"'{name}' must have shape ({n_seg}, {n_bands}) matching "
            f"'normal_intensity', got {a.shape}."
        )
        raise ValueError(msg)
    return a


def _validate_scan(
    normal_intensity: np.ndarray,
    areas: np.ndarray,
    frequencies: np.ndarray | None,
    band_type: BandType,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate an ISO 9614-2 scan and normalise its arrays.

    Returns the signed normal intensity as ``(N_seg, N_bands)`` and the segment
    areas as ``(N_seg,)``.
    """
    if band_type not in ("octave", "third"):
        msg = "'band_type' must be 'octave' or 'third'."
        raise ValueError(msg)

    intensity = np.atleast_2d(np.asarray(normal_intensity, dtype=np.float64))
    seg = np.asarray(areas, dtype=np.float64)
    if seg.ndim != 1:
        msg = "'areas' must be a 1D array of segment areas."
        raise ValueError(msg)
    n_seg = seg.shape[0]
    # A 1D (N_seg,) intensity arrives from atleast_2d as (1, N_seg): transpose.
    if intensity.shape == (1, n_seg) and n_seg != 1:
        intensity = intensity.T
    if intensity.shape[0] != n_seg:
        msg = (
            f"'normal_intensity' first axis ({intensity.shape[0]}) must match "
            f"the number of segment 'areas' ({n_seg})."
        )
        raise ValueError(msg)
    if frequencies is not None:
        # Validate up front so a mismatched length raises the public ValueError
        # rather than an IndexError from the Table 2 lookup during
        # classification, and a NaN or infinite band centre a named refusal
        # rather than round()'s conversion error there.
        if np.asarray(frequencies).shape != (intensity.shape[1],):
            raise ValueError(_FREQUENCIES_BAND_COUNT_MSG)
        require_positive_array(frequencies, "frequencies")
    # NaN beside the bound, not folded into it: a NaN compares False against
    # every bound, so the positivity test alone passes it through to a
    # measurement surface that is not a number.
    if not np.all(np.isfinite(seg)):
        msg = "All segment 'areas' must be finite."
        raise ValueError(msg)
    if np.any(seg <= 0.0):
        msg = "All segment 'areas' must be positive."
        raise ValueError(msg)
    if n_seg < _MIN_SEGMENTS:
        warnings.warn(
            f"Only {n_seg} segment(s); ISO 9614-2:1996 clause 8.2 requires at "
            "least 4 measurement segments.",
            SoundPowerWarning,
            stacklevel=3,
        )
    return intensity, seg


def sound_power_intensity(
    normal_intensity: np.ndarray,
    areas: np.ndarray,
    *,
    normal_intensity_2: np.ndarray | None = None,
    pressure_levels: np.ndarray | None = None,
    pressure_residual_index: float | np.ndarray | None = None,
    frequencies: np.ndarray | None = None,
    band_type: BandType = "third",
    grade: Grade = "engineering",
    repeatability_limit: float | np.ndarray | None = None,
) -> SoundPowerIntensityResult:
    r"""Sound power level by sound-intensity scanning (ISO 9614-2:1996).

    ``normal_intensity`` is an ``(N_seg, N_bands)`` array (or ``(N_seg,)`` for a
    single band) of the signed, segment-averaged normal sound intensity
    :math:`\langle I_{\mathrm{n},i} \rangle` (W/m^2), and ``areas`` the ``(N_seg,)``
    segment areas ``Si`` (m^2). The partial powers
    :math:`P_i = \langle I_{\mathrm{n},i} \rangle S_i` are summed to the band sound
    power ``P`` and level :math:`L_W = 10 \log_{10}(P/P_0)` (equations (12), (6),
    (13)). Bands with :math:`P < 0` are flagged (``negative_band``) and
    reported as ``NaN`` (clause 9.2).

    Supplying ``normal_intensity_2`` (the second grade-2 sweep) makes
    ``normal_intensity`` the first sweep, uses their mean for the partial powers
    (Eq. 12), and evaluates the repeatability criterion 3. Supplying
    ``pressure_levels`` (``Lpi``) evaluates ``FpI`` (Eq. A.1) and, with
    ``pressure_residual_index`` (``delta_pI0``), criterion 1. The per-band
    achieved grade (clause 8.4) is returned when both a second sweep and
    ``delta_pI0`` are available. When criteria 1 and 2 are evaluable
    (``pressure_levels`` and ``pressure_residual_index`` supplied), the bands
    failing them are omitted from the A-weighted total and flagged in
    ``a_weighting_omitted_bands`` (clause 10.6 b); otherwise every determinable
    band is summed and a :class:`SoundPowerWarning` notes the missing
    screening.

    :param normal_intensity: ``(N_seg, N_bands)`` signed normal intensity, W/m^2.
    :param areas: ``(N_seg,)`` segment areas ``Si``, m^2.
    :param normal_intensity_2: Optional second sweep, same shape (criterion 3).
    :param pressure_levels: Optional ``(N_seg, N_bands)`` ``Lpi`` (dB) for FpI.
    :param pressure_residual_index: ``delta_pI0`` (dB), scalar or per band, for
        the dynamic-capability index / criterion 1.
    :param frequencies: ``(N_bands,)`` nominal band centres (Hz), for the
        A-weighted total and the Table 2 repeatability limits.
    :param band_type: ``'octave'`` or ``'third'`` (Table 2 lookup).
    :param grade: ``'engineering'`` (grade 2) or ``'survey'`` (grade 3);
        selects ``K`` for the reported ``Ld`` and the criterion-2 warning.
    :param repeatability_limit: Override for the criterion-3 limit ``s`` (dB),
        scalar or per band; defaults to ISO 9614-2 Table 2 by ``frequencies``
        for ``'engineering'``. For ``'survey'`` the default is the A-weighted
        4 dB reused per band (extrapolated -- non-normative).
    :return: :class:`SoundPowerIntensityResult`.
    """
    grade = _check_grade(grade)
    intensity, seg = _validate_scan(normal_intensity, areas, frequencies, band_type)
    n_seg = seg.shape[0]
    n_bands = intensity.shape[1]

    # --- partial powers, using the mean of the two sweeps when available -----
    repeatability: np.ndarray | None = None
    if normal_intensity_2 is not None:
        scan2 = _as_2d(
            "normal_intensity_2", np.asarray(normal_intensity_2), n_seg, n_bands
        )
        pi1 = intensity * seg[:, None]
        pi2 = scan2 * seg[:, None]
        repeatability = np.abs(_level_magnitude(pi1) - _level_magnitude(pi2))
        # Criterion 3 (B.1.3) tests whether the partial power of a segment
        # repeats between the two sweeps. A complete flow reversal (pi1 and
        # pi2 of opposite sign) is grossly non-repeatable, yet |ΔL| of the
        # magnitudes alone can be ~0 when |pi1| ~ |pi2|. Force the criterion to
        # fail (repeatability = +inf) wherever the signs differ; exact zeros
        # carry no direction and are treated as matching either sign.
        reversed_flow = (np.sign(pi1) * np.sign(pi2)) < 0.0
        repeatability = np.where(reversed_flow, np.inf, repeatability)
        mean_intensity = 0.5 * (intensity + scan2)
    else:
        mean_intensity = intensity

    partial_power = mean_intensity * seg[:, None]  # Eq. 12
    partial_power_level = _level_magnitude(partial_power)  # Eq. 8 (magnitude)
    total_power = np.sum(partial_power, axis=0)  # Eq. 6
    negative_band = total_power <= 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        sound_power_level = np.where(
            total_power > 0.0,
            10.0 * np.log10(np.maximum(total_power, np.finfo(float).tiny) / _P0),
            np.nan,
        )
    s_total = float(np.sum(seg))

    # --- field indicators ----------------------------------------------------
    abs_total = np.abs(total_power)
    guarded_abs = np.maximum(abs_total, np.finfo(float).tiny)
    lw_magnitude = 10.0 * np.log10(guarded_abs / _P0)

    fpi: np.ndarray | None = None
    if pressure_levels is not None:
        lp = _as_2d("pressure_levels", np.asarray(pressure_levels), n_seg, n_bands)
        # Eq. A.1: area-weighted surface pressure level [Lp].
        lp_surface = weighted_energy_mean(lp, seg[:, None], axis=0)
        fpi = np.asarray(
            lp_surface - lw_magnitude + 10.0 * np.log10(s_total / _S0),
            dtype=np.float64,
        )

    # Eq. A.2: negative-partial-power indicator F+/-.
    sum_abs = np.sum(np.abs(partial_power), axis=0)
    f_plus_minus = np.asarray(
        10.0 * np.log10(np.maximum(sum_abs, np.finfo(float).tiny) / guarded_abs),
        dtype=np.float64,
    )

    # --- dynamic capability index Ld (reused from ISO 9614-1 machinery) -------
    ld: np.ndarray | None = None
    dpi0_arr: np.ndarray | None = None
    if pressure_residual_index is not None:
        dpi0_arr = np.broadcast_to(
            np.asarray(pressure_residual_index, dtype=np.float64), (n_bands,)
        ).astype(np.float64)
        ld = np.array(
            [dynamic_capability_index(float(d), _K[grade]) for d in dpi0_arr],
            dtype=np.float64,
        )

    # --- warnings ------------------------------------------------------------
    if np.any(negative_band):
        warnings.warn(
            "Total sound power is negative in one or more bands; ISO 9614-2:1996 "
            "is not applicable to those bands (clause 9.2).",
            SoundPowerWarning,
            stacklevel=2,
        )
    # Criterion 2 (F+/- <= 3 dB) is mandatory only for the engineering grade;
    # for a survey run it is optional (ISO 9614-2:1996, B.1.2), so exceeding
    # the limit does not by itself disqualify the survey grade and the warning
    # is suppressed there.
    if grade == "engineering" and np.any(
        f_plus_minus[~negative_band] > _F_PLUS_MINUS_LIMIT
    ):
        warnings.warn(
            f"Negative-partial-power indicator F+/- exceeds {_F_PLUS_MINUS_LIMIT:g} "
            "dB in one or more bands; criterion 2 is not satisfied and the "
            "engineering grade is not achieved there (ISO 9614-2:1996, B.1.2).",
            SoundPowerWarning,
            stacklevel=2,
        )

    # --- achieved grade per band (clause 8.4) --------------------------------
    achieved_grade: np.ndarray | None = None
    if fpi is not None and ld is not None and repeatability is not None:
        achieved_grade = _classify(
            fpi,
            f_plus_minus,
            repeatability,
            dpi0_arr,  # type: ignore[arg-type]
            negative_band,
            frequencies,
            band_type,
            repeatability_limit,
        )

    # --- A-weighted total over determinable, qualified bands (10.6 b) --------
    omitted = _a_weighting_omission(fpi, ld, f_plus_minus, grade, negative_band)
    lwa = _a_weighted_total(
        sound_power_level, negative_band, omitted, frequencies, n_bands
    )

    freqs = None if frequencies is None else np.asarray(frequencies, dtype=np.float64)
    return SoundPowerIntensityResult(
        frequencies=freqs,
        partial_power=partial_power,
        partial_power_level=partial_power_level,
        sound_power=np.asarray(total_power, dtype=np.float64),
        sound_power_level=np.asarray(sound_power_level, dtype=np.float64),
        negative_band=np.asarray(negative_band, dtype=bool),
        surface_pressure_intensity_index=fpi,
        negative_partial_power_index=f_plus_minus,
        repeatability=repeatability,
        dynamic_capability_index=ld,
        achieved_grade=achieved_grade,
        surface_area=s_total,
        sound_power_level_a=lwa,
        a_weighting_omitted_bands=omitted,
        grade=grade,
    )


def _classify(
    fpi: np.ndarray,
    f_plus_minus: np.ndarray,
    repeatability: np.ndarray,
    dpi0: np.ndarray,
    negative_band: np.ndarray,
    frequencies: np.ndarray | None,
    band_type: BandType,
    repeatability_limit: float | np.ndarray | None,
) -> np.ndarray:
    """Per-band class ('engineering'/'survey'/'none') from Annex B criteria."""
    n_bands = fpi.shape[0]
    if repeatability_limit is not None:
        s_eng = np.broadcast_to(
            np.asarray(repeatability_limit, dtype=np.float64), (n_bands,)
        )
        s_sur = s_eng
    else:
        if frequencies is None:
            msg = (
                "The achieved grade needs the criterion-3 limit s: provide "
                "'frequencies' (Table 2 lookup) or 'repeatability_limit'."
            )
            raise ValueError(msg)
        nominal = [round(float(f)) for f in np.asarray(frequencies)]
        s_eng = np.array([_table2_s(f, band_type) for f in nominal], dtype=np.float64)
        s_sur = np.full(n_bands, _S_SURVEY, dtype=np.float64)

    result = np.empty(n_bands, dtype=object)
    for b in range(n_bands):
        if negative_band[b]:
            result[b] = "none"
            continue
        ld_eng = dpi0[b] - _K["engineering"]
        ld_sur = dpi0[b] - _K["survey"]
        c1_eng = ld_eng > fpi[b]
        c1_sur = ld_sur > fpi[b]
        c2 = f_plus_minus[b] <= _F_PLUS_MINUS_LIMIT
        c3_eng = bool(np.all(repeatability[:, b] <= s_eng[b]))
        c3_sur = bool(np.all(repeatability[:, b] <= s_sur[b]))
        if c1_eng and c2 and c3_eng:
            result[b] = "engineering"
        elif c1_sur and c3_sur:
            result[b] = "survey"
        else:
            result[b] = "none"
    return result


def _a_weighting_omission(
    fpi: np.ndarray | None,
    ld: np.ndarray | None,
    f_plus_minus: np.ndarray,
    grade: str,
    negative_band: np.ndarray,
) -> np.ndarray | None:
    r"""Bands to omit from the A-weighted total (ISO 9614-2:1996 clause 10.6 b).

    Clause 10.6 b requires the contribution of the bands in which criteria 1
    and/or 2 are not satisfied to be omitted from an A-weighted sound power
    level determination (unless negligible per 4.3), and the omission to be
    stated. Criterion 1 (:math:`L_\mathrm{d} > F_{pI}`, B.1.1) needs the surface
    pressure-intensity indicator and the dynamic capability index; without them
    the screening cannot be evaluated and ``None`` is returned. Criterion 2
    (:math:`F_{+/-} \le 3` dB, B.1.2) binds engineering-grade determinations
    only
    (optional for grade 3, Note 16). Net-negative bands are excluded already
    as non-determinable (clause 9.2), so they are not flagged here.
    """
    if fpi is None or ld is None:
        return None
    fails = ~(ld > fpi)
    if grade == "engineering":
        fails = fails | (f_plus_minus > _F_PLUS_MINUS_LIMIT)
    return np.asarray(fails & ~negative_band, dtype=bool)


def _a_weighted_total(
    sound_power_level: np.ndarray,
    negative_band: np.ndarray,
    omitted: np.ndarray | None,
    frequencies: np.ndarray | None,
    n_bands: int,
) -> float:
    r"""A-weighted band sum over determinable, qualified bands (clause 10.6 b).

    Sums the determinable bands (net power :math:`P > 0`, clause 9.2) minus the
    bands omitted per clause 10.6 b (criteria 1 and/or 2 failed). When the
    screening could not be evaluated (``omitted`` is ``None``) every
    determinable band is summed and a :class:`SoundPowerWarning` is emitted.
    """
    determinable = ~negative_band
    if frequencies is not None:
        freqs = np.asarray(frequencies, dtype=np.float64)
        if freqs.shape[0] != n_bands:
            raise ValueError(_FREQUENCIES_BAND_COUNT_MSG)
        if omitted is None:
            if n_bands > 1:
                warnings.warn(
                    "The A-weighted total sums every determinable band without "
                    "the ISO 9614-2:1996 clause 10.6 b screening (bands failing "
                    "criteria 1 and/or 2 must be omitted); supply "
                    "'pressure_levels' and 'pressure_residual_index' to "
                    "evaluate the criteria.",
                    SoundPowerWarning,
                    stacklevel=3,
                )
        else:
            determinable = determinable & ~omitted
        ck = _a_weighting_corrections(freqs)
        contrib = 10.0 ** (0.1 * (sound_power_level + ck))
        total = float(np.sum(contrib[determinable]))
        return 10.0 * np.log10(total) if total > 0.0 else float("nan")
    if n_bands == 1:
        return float(sound_power_level[0])
    return float("nan")


# ===========================================================================
# ISO 9614-3:2002 - sound power by sound-intensity scanning (precision). The
# precision sibling of ISO 9614-2 (engineering). Single grade, bias-error
# factor K = 10 dB, five acceptance criteria, and a meteorologically
# normalized sound power level LW0 (Eq. 10).
# ===========================================================================

_P0_INTENSITY = 1.0e-12  #: Reference sound power, in watts (3.6.3).
_I0 = 1.0e-12  #: Reference sound intensity, in W/m^2 (3.5).
_K_9614_3 = 10.0  #: Bias-error factor K, in dB (def. 3.11).
_FS_LIMIT = 2.0  #: Criterion 4 field-non-uniformity limit (Eq. C.4).
_F_PI_DIFF_LIMIT = 3.0  #: Criterion 3 signed-minus-unsigned limit, dB (Eq. C.3).
_FS_RATIO_LOW = 0.83  #: Criterion 5 lower bound on FS(1)/FS(2) (Eq. C.5).
_FS_RATIO_HIGH = 1.2  #: Criterion 5 upper bound on FS(1)/FS(2) (Eq. C.5).
#: Fewest samples a Bessel-corrected (N-1) sample standard deviation needs:
#: segments for FS (Eq. B.8) and time windows for FT (Eq. B.1).
_MIN_STDDEV_SAMPLES = 2
_ABS_ZERO_C = -273.15  #: Absolute zero, deg C: validity floor for theta (Eq. 10).


@dataclass(frozen=True)
class PrecisionFieldIndicators:
    r"""ISO 9614-3:2002 Annex B field indicators (per band).

    ``ft`` is the temporal-variability indicator (= F1 of ISO 9614-1, Eq. B.1),
    ``None`` unless time-window intensities are supplied. ``f_pi_unsigned`` is
    the unsigned pressure-intensity indicator (= F2, Eq. B.3, using the mean
    magnitude of the segment intensities) and ``f_pi_signed`` the signed one
    (= F3, Eq. B.6, using the algebraic mean); by construction
    :math:`F_{pI_\mathrm{n}}^{\mathrm{signed}} \ge F_{pI_\mathrm{n}}^{\mathrm{unsigned}}`.
    ``fs`` is the field-non-uniformity indicator (= F4, Eq. B.8).
    """

    ft: np.ndarray | None
    f_pi_unsigned: np.ndarray
    f_pi_signed: np.ndarray
    fs: np.ndarray

    def __post_init__(self) -> None:
        """Reject indicators that do not all span the same bands.

        The four indicators are what clause 10 f) 1) has the report state,
        band by band in the verbose table and as a range in the
        measurement-basis strip, and they are read against each other before
        that: :func:`precision_qualification` forms criterion 3 from the
        difference of the two pressure-intensity indicators and criterion 4
        from ``fs``, taking its band count from ``f_pi_signed`` alone. An
        indicator carrying a single value is broadcast into the difference
        for every band, so one band's reading qualifies the whole spectrum;
        one of another length raises from inside numpy about two shapes,
        naming neither indicator. ``ft`` is absent unless time-window
        intensities were supplied, which is not a disagreement.

        :raises ValueError: if two of the indicators span different numbers
            of bands.
        """
        require_ranks(self, ft=1, f_pi_unsigned=1, f_pi_signed=1, fs=1)
        require_same_length(self, "ft", "f_pi_unsigned", "f_pi_signed", "fs")


@dataclass(frozen=True)
class PrecisionCriteria:
    r"""ISO 9614-3:2002 Annex C acceptance criteria (per band, pass/fail).

    Each attribute is a boolean array (True = satisfied) or ``None`` when its
    inputs are absent. ``criterion_1`` scan repeatability
    :math:`\lvert L_{I_\mathrm{n}}(1) - L_{I_\mathrm{n}}(2) \rvert \le s/2` (Eq. C.1);
    ``criterion_2`` dynamic-capability
    adequacy :math:`L_\mathrm{d} \ge F_{pI_\mathrm{n}}^{\mathrm{signed}}` (Eq. C.2);
    ``criterion_3``
    :math:`F_{pI_\mathrm{n}}^{\mathrm{signed}} - F_{pI_\mathrm{n}}^{\mathrm{unsigned}} \le 3` dB
    (Eq. C.3); ``criterion_4``
    :math:`F_\mathrm{S} \le 2` (Eq. C.4); ``criterion_5`` scan-density convergence
    :math:`0.83 \le F_\mathrm{S}(1)/F_\mathrm{S}(2) \le 1.2` (Eq. C.5). ``qualified`` is the
    conjunction of criteria 1-3 with the field non-uniformity accepted
    through criterion 4
    or, where evaluated, criterion 5 (C.1.6.2: a band satisfying criterion 5
    is qualified as a final result even if :math:`F_\mathrm{S}(2) \ge 2`); ``None``
    unless both criterion 1 and criterion 2 are evaluable.
    """

    criterion_1: np.ndarray | None
    criterion_2: np.ndarray | None
    criterion_3: np.ndarray
    criterion_4: np.ndarray
    criterion_5: np.ndarray | None
    qualified: np.ndarray | None

    def __post_init__(self) -> None:
        """Reject a verdict set whose criteria do not span the same bands.

        ``qualified`` is the conjunction of the others, and a conjunction is
        taken with numpy's broadcasting: a criterion carrying a single value
        decides every band at once. :func:`precision_qualification` reaches
        that state on its own: it forms criterion 1 from the scan intensity
        levels and the Table 1 limit read from ``frequencies``, and the rest
        from the indicators, without comparing the two sets, so a
        repeatability pair read on a single band leaves criterion 1 one band
        long and ``qualified`` as long as the indicators, every band of it
        decided by that one. The fiche is not where this surfaces:
        ``PrecisionIntensityResult.report`` refuses a criteria set that does
        not span the determination and names the criterion. What the
        broadcast verdict reaches is ``qualified`` itself, at full length,
        with nothing left in it to say that one band decided the rest. A
        criterion is ``None`` when its inputs were not supplied, which is
        not a disagreement.

        :raises ValueError: if two of the criteria span different numbers of
            bands.
        """
        require_ranks(
            self,
            criterion_1=1,
            criterion_2=1,
            criterion_3=1,
            criterion_4=1,
            criterion_5=1,
            qualified=1,
        )
        require_same_length(
            self,
            "criterion_1",
            "criterion_2",
            "criterion_3",
            "criterion_4",
            "criterion_5",
            "qualified",
        )


def _check_report_bands(
    indicators: PrecisionFieldIndicators | None,
    criteria: PrecisionCriteria | None,
    residual_index: float | Sequence[float] | np.ndarray | None,
    n_bands: int,
) -> None:
    """Reject fiche inputs that do not span the determination's bands.

    The indicators, the criteria and the residual index are measured beside
    the determination rather than derived from it, so nothing has yet forced
    them onto the same band set. A mismatch would print one band's indicator
    against another band's level, which is the one error an accredited sheet
    must not make, so it is refused here rather than rendered.
    """
    arrays: list[tuple[str, np.ndarray]] = []
    if indicators is not None:
        arrays += [
            (f"indicators.{name}", np.asarray(values))
            for name, values in (
                ("ft", indicators.ft),
                ("f_pi_unsigned", indicators.f_pi_unsigned),
                ("f_pi_signed", indicators.f_pi_signed),
                ("fs", indicators.fs),
            )
            if values is not None
        ]
    if criteria is not None:
        arrays += [
            (f"criteria.{name}", np.asarray(values))
            for name, values in (
                ("criterion_1", criteria.criterion_1),
                ("criterion_2", criteria.criterion_2),
                ("criterion_3", criteria.criterion_3),
                ("criterion_4", criteria.criterion_4),
                ("criterion_5", criteria.criterion_5),
                ("qualified", criteria.qualified),
            )
            if values is not None
        ]
    if residual_index is not None:
        index = np.atleast_1d(np.asarray(residual_index, dtype=np.float64))
        if index.size not in (1, n_bands):
            msg = (
                f"'residual_index' must be a scalar or span the {n_bands} "
                f"bands of the determination; got {index.size} values."
            )
            raise ValueError(msg)
    for name, values in arrays:
        if values.shape != (n_bands,):
            msg = (
                f"'{name}' must span the {n_bands} bands of the determination; "
                f"got shape {values.shape}."
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class PrecisionIntensityResult:
    r"""Result of an ISO 9614-3:2002 sound-power-by-scanning determination.

    ``partial_power`` is the signed :math:`P_i = I_{\mathrm{n},i} S_i` per partial
    surface and band (Eq. 5); ``sound_power`` the signed band total
    :math:`P = \sum P_i` (Eq. 8) and ``sound_power_level`` its level
    :math:`L_W = 10 \log_{10}(P/P_0)` (Eq. 9), ``NaN``
    where :math:`P \le 0` (``not_applicable_band`` True, clause 9.2).
    ``sound_power_level_normalized`` is ``LW0`` normalized to 23 deg C /
    101 325 Pa (Eq. 10). ``sound_power_level_a`` is the A-weighted total over
    applicable bands (``NaN`` without ``frequencies`` and more than one band).
    """

    frequencies: np.ndarray | None
    partial_power: np.ndarray
    sound_power: np.ndarray
    sound_power_level: np.ndarray
    sound_power_level_normalized: np.ndarray
    not_applicable_band: np.ndarray
    surface_area: float
    sound_power_level_a: float

    def __post_init__(self) -> None:
        """Reject a determination whose per-band quantities disagree.

        The fiche prints one row per band, taking the row count from
        ``sound_power_level`` and the labels from ``frequencies``, and it
        reads each of those columns against another, so a band axis of the
        wrong length raises part way through the sheet: from numpy about two
        shapes it could not broadcast, or from a strict ``zip`` about two
        argument lengths, naming neither field. ``not_applicable_band``
        marks the rows outside the method (clause 9.2) and joins the
        qualification mask that drops bands from the A-weighted total, and
        that join is an in-place ``|=``, which refuses a mask of another
        length as a non-broadcastable operand rather than stretching it over
        every band. ``sound_power`` is the silent one: the signed band total
        in watts is the only per-band quantity no reader in the library
        opens, the sheet printing its level instead, so a determination
        carrying one power too many renders a complete fiche and keeps the
        extra band for whoever reads that column against ``frequencies``.

        ``partial_power`` carries the partial surfaces on its first axis and
        the bands on its second, so its band axis is index 1; the partial
        surfaces are its own axis, shared with no other field.

        ``surface_area`` must also be finite, for the reason it must be in
        :class:`SoundPowerIntensityResult`: it is the sum of partial surfaces
        the determination has already refused unless positive and finite, and
        the clause 10 sheet prints it as "Measurement surface S = ... m2"
        beside the boxed level. The per-band levels stay unpinned, ``NaN``
        being clause 9.2's reading of a band the method does not apply to.

        :raises ValueError: if any per-band quantity disagrees with the rest,
            or ``surface_area`` is not finite.
        """
        require_ranks(
            self,
            frequencies=1,
            partial_power=2,
            sound_power=1,
            sound_power_level=1,
            sound_power_level_normalized=1,
            not_applicable_band=1,
        )
        require_same_length(
            self,
            "frequencies",
            ("partial_power", 1),
            "sound_power",
            "sound_power_level",
            "sound_power_level_normalized",
            "not_applicable_band",
        )
        require_finite_fields(self, "surface_area")

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the ``LW`` spectrum; non-applicable bands are hatched/greyed.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.emission import plot_sound_power

        check_language(language)
        return plot_sound_power(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
        indicators: PrecisionFieldIndicators | None = None,
        criteria: PrecisionCriteria | None = None,
        residual_index: float | Sequence[float] | np.ndarray | None = None,
    ) -> str:
        """Render an ISO 9614-3 precision sound-power determination fiche.

        Writes the one-page sound-power test sheet with what ISO 9614-3:2002
        clause 10 asks a report of this method to state: the standard-basis
        line naming the precision scanning method and its single accuracy
        grade, an optional metadata header (client, noise source, test
        environment, instrumentation, air temperature, relative humidity,
        barometric pressure and date, clause 10 a) to d)), a per-band table of
        the band sound-power level ``LW``, the normalized level ``LW0`` the
        standard reports (Eq. 10, clause 10 f) 2)) and the expanded uncertainty
        ``U`` of clause 4.3 (clause 10 f) 4)), the sound-power spectrum
        ``LW(f)`` with the non-applicable bands hatched, the boxed A-weighted
        sound power level ``LWA`` (dB re 1 pW) with the totals, the measurement
        surface area and the grade, an optional verdict row against a declared
        limit, and a measurement-basis strip carrying the partial-power model,
        the meteorological normalization, the Annex B field indicators and the
        Annex C criteria.

        Supplying ``criteria`` makes the fiche state what clause 10 f) 2)
        requires it to state: the bands whose criteria are not satisfied are
        dropped from the A-weighted determination and named on the sheet
        alongside the bands the method is not applicable to (clause 9.2). The
        boxed ``LWA`` is then the level of the qualified bands, which differs
        from the result's own ``sound_power_level_a`` whenever a band is
        rejected; without ``criteria`` the fiche boxes the result's value and
        says that no qualification was supplied.

        The items of clause 10 that are free description rather than computed
        quantities (the scan geometry and speed, the drawing of the scanning
        paths, the scanning time per partial surface, the calibration and
        field-check history, the windscreen, and the probe-reversal checks of
        clause 6.2.3) belong in the metadata ``notes`` and ``calibration``
        fields; the fiche prints them verbatim in its footer.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the header (``client``, ``specimen`` the noise source, ``test_room``
            the test environment, ``instrumentation``, ``temperature``,
            ``relative_humidity``, ``pressure``, ``test_date``), the footer
            identity (``laboratory``, ``operator``, ``report_id``, ``notes``)
            and, via ``requirement``, a declared A-weighted sound-power limit
            the fiche checks the result against (lower is better).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True`` the per-band table adds the four Annex B
            field indicators and the per-band grade cell.
        :param language: Fiche language: ``"en"`` (default) or ``"es"``.
        :param indicators: Optional :class:`PrecisionFieldIndicators` from
            :func:`precision_field_indicators`, tabulated per band with
            ``verbose`` and summarised in the basis strip (clause 10 f) 1)).
        :param criteria: Optional :class:`PrecisionCriteria` from
            :func:`precision_qualification`, which decides the per-band grade
            cell and the clause 10 f) 2) omission described above.
        :param residual_index: Optional pressure-residual intensity index
            ``delta_pI0`` of the probe and analyser (clause 10 d) 5)), a scalar
            or a per-band array; the strip states it and the dynamic capability
            ``Ld = delta_pI0 - K`` that criterion 2 tests.
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"``, ``language``
            is unknown, or a supplied ``indicators``, ``criteria`` or
            ``residual_index`` does not span the result's bands.
        :raises ImportError: If reportlab (or, for the figure, matplotlib) is
            not installed (``pip install phonometry[report]``).
        """
        from .._i18n import check_language

        check_language(language)
        check_engine(engine)
        n_bands = int(np.asarray(self.sound_power_level).size)
        _check_report_bands(indicators, criteria, residual_index, n_bands)

        from .._report.iso9614 import render_precision_intensity_report

        return render_precision_intensity_report(
            self,
            path,
            metadata=metadata,
            verbose=verbose,
            language=language,
            indicators=indicators,
            criteria=criteria,
            residual_index=residual_index,
        )


def precision_field_indicators(
    segment_intensity: np.ndarray,
    segment_pressure_levels: np.ndarray,
    *,
    time_window_intensity: np.ndarray | None = None,
) -> PrecisionFieldIndicators:
    r"""ISO 9614-3:2002 Annex B field indicators from segment data.

    Over the ``N`` segments of the whole measurement surface (per band):

    .. math::

       \overline{L_p} = 10 \log_{10}\!\left[ \frac{1}{N}
       \sum_j 10^{0.1 L_{pj}} \right] \tag{Eq. B.4}

       L_{|I_\mathrm{n}|} = 10 \log_{10}\!\left[ \frac{1}{N}
       \sum_j \frac{|I_{\mathrm{n}j}|}{I_0} \right] \tag{Eq. B.5}

       L_{I_\mathrm{n}} = 10 \log_{10}\!\left[ \frac{1}{I_0} \left| \frac{1}{N}
       \sum_j I_{\mathrm{n}j} \right| \right] \tag{Eq. B.7}

       F_{pI_\mathrm{n}}^{\mathrm{unsigned}} = \overline{L_p} - L_{|I_\mathrm{n}|}
       \tag{Eq. B.3}

       F_{pI_\mathrm{n}}^{\mathrm{signed}} = \overline{L_p} - L_{I_\mathrm{n}} \tag{Eq. B.6}

       F_\mathrm{S} = \frac{1}{\overline{I_\mathrm{n}}} \sqrt{ \frac{1}{N-1}
       \sum_j \left( I_{\mathrm{n}j} - \overline{I_\mathrm{n}} \right)^2 } \tag{Eq. B.8}

    With ``time_window_intensity`` (an ``(M, NB)`` array of window-averaged
    intensities) the temporal-variability indicator ``FT`` (Eq. B.1) is also
    returned.

    :param segment_intensity: ``(N, NB)`` signed segment normal intensity, W/m^2.
    :param segment_pressure_levels: ``(N, NB)`` segment pressure levels, dB.
    :param time_window_intensity: Optional ``(M, NB)`` window intensities for FT.
    :return: :class:`PrecisionFieldIndicators`.
    """
    i_n = np.atleast_2d(np.asarray(segment_intensity, dtype=np.float64))
    lp = np.atleast_2d(np.asarray(segment_pressure_levels, dtype=np.float64))
    require_equal_shapes(
        "precision_field_indicators",
        {"segment_intensity": i_n.shape, "segment_pressure_levels": lp.shape},
        "segment",
    )
    n_seg = i_n.shape[0]
    if n_seg < _MIN_STDDEV_SAMPLES:
        msg = "At least two segments are required for the indicators."
        raise ValueError(msg)

    lp_bar = energy_mean(lp, axis=0)  # Eq. B.4
    li_unsigned = 10.0 * np.log10(np.mean(np.abs(i_n), axis=0) / _I0)  # Eq. B.5
    mean_signed = np.mean(i_n, axis=0)
    li_signed = 10.0 * np.log10(
        np.maximum(np.abs(mean_signed), np.finfo(float).tiny) / _I0
    )  # Eq. B.7 (magnitude; sign carried separately by the P<0 rule)
    f_pi_unsigned = np.asarray(lp_bar - li_unsigned, dtype=np.float64)
    f_pi_signed = np.asarray(lp_bar - li_signed, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        fs = (
            np.sqrt(
                np.sum((i_n - mean_signed[np.newaxis, :]) ** 2, axis=0) / (n_seg - 1)
            )
            / mean_signed
        )  # Eq. B.8
    fs = np.asarray(fs, dtype=np.float64)

    ft: np.ndarray | None = None
    if time_window_intensity is not None:
        win = np.atleast_2d(np.asarray(time_window_intensity, dtype=np.float64))
        if win.shape[-1] != i_n.shape[-1]:
            msg = "'time_window_intensity' last axis must match the number of bands."
            raise ValueError(msg)
        m = win.shape[0]
        if m < _MIN_STDDEV_SAMPLES:
            msg = "At least two time windows are required for FT."
            raise ValueError(msg)
        mean_t = np.mean(win, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ft = np.asarray(
                np.sqrt(np.sum((win - mean_t[np.newaxis, :]) ** 2, axis=0) / (m - 1))
                / mean_t,
                dtype=np.float64,
            )  # Eq. B.1

    return PrecisionFieldIndicators(
        ft=ft, f_pi_unsigned=f_pi_unsigned, f_pi_signed=f_pi_signed, fs=fs
    )


def _checked_band_centres(
    frequencies: np.ndarray | None, f_pi_signed: np.ndarray, n_bands: int
) -> np.ndarray | None:
    """The band centres validated against the indicator's band axis.

    Split out of :func:`precision_qualification`, which was one branch past
    what one reader holds: the centre validation is one idea, and it reads
    the same whether the caller supplied centres or not.
    """
    if frequencies is None:
        return None
    freqs = require_positive_array(frequencies, "frequencies")
    if freqs.shape != f_pi_signed.shape:
        msg = (
            "'frequencies' must carry one value per band "
            f"({n_bands} in 'indicators.f_pi_signed'); got shape {freqs.shape}."
        )
        raise ValueError(msg)
    return freqs


def _spread_over_bands(value: ArrayLike, n_bands: int) -> np.ndarray:
    """Put a field non-uniformity on the band axis without inventing values.

    A bare number states one figure that holds for every band, so it is spread
    over them. A value that already carries an axis is returned exactly as it
    came, even when its length is wrong: stretching that one too would state a
    figure for bands it was never measured on, and would hide the disagreement
    behind numpy's broadcast message instead of letting it reach the per-band
    check on the result, which names the criterion that does not fit.
    """
    spectrum = np.asarray(value, dtype=np.float64)
    if spectrum.ndim == 0:
        return np.broadcast_to(spectrum, (n_bands,))
    return spectrum


def _sigma_r0_9614_3(nominal: int) -> float:
    """Per-band sigma_R0 (ISO 9614-3:2002 Table 1), in dB; also criterion-1 s."""
    if 50 <= nominal <= 160:  # noqa: PLR2004
        return 2.0
    if 200 <= nominal <= 315:  # noqa: PLR2004
        return 1.5
    if 400 <= nominal <= 5000:  # noqa: PLR2004
        return 1.0
    if nominal == 6300:  # noqa: PLR2004
        return 2.0
    msg = (
        f"No ISO 9614-3:2002 Table 1 sigma_R0 for {nominal} Hz; expected a "
        "nominal one-third-octave mid-band from 50 Hz to 6300 Hz."
    )
    raise ValueError(msg)


def precision_qualification(
    indicators: PrecisionFieldIndicators,
    *,
    scan_intensity_level_1: np.ndarray | None = None,
    scan_intensity_level_2: np.ndarray | None = None,
    pressure_residual_index: float | np.ndarray | None = None,
    field_nonuniformity_1: np.ndarray | None = None,
    field_nonuniformity_2: np.ndarray | None = None,
    frequencies: np.ndarray | None = None,
    repeatability_limit: float | np.ndarray | None = None,
) -> PrecisionCriteria:
    r"""Evaluate the five ISO 9614-3:2002 Annex C acceptance criteria per band.

    :param indicators: The :class:`PrecisionFieldIndicators` (gives criteria 3
        and 4 directly).
    :param scan_intensity_level_1: ``LIn(1)`` per band (dB), first scan.
    :param scan_intensity_level_2: ``LIn(2)`` per band (dB), second scan; with
        the first scan and ``s`` this gives criterion 1
        (:math:`\lvert \Delta L \rvert \le s/2`).
    :param pressure_residual_index: ``delta_pI0`` (dB), scalar or per band; with
        :math:`K = 10` gives ``Ld`` for criterion 2
        (:math:`L_\mathrm{d} \ge F_{pI_\mathrm{n}}^{\mathrm{signed}}`).
    :param field_nonuniformity_1: ``FS(1)`` per band (initial scan density).
    :param field_nonuniformity_2: ``FS(2)`` per band (doubled density); with
        ``FS(1)`` gives criterion 5.
    :param frequencies: ``(NB,)`` nominal mid-band frequencies (Hz), selecting
        the criterion-1 limit ``s`` from Table 1.
    :param repeatability_limit: Override for ``s`` (dB), scalar or per band.
    :return: :class:`PrecisionCriteria`.
    """
    f_pi_signed = indicators.f_pi_signed
    n_bands = f_pi_signed.shape[0]

    freqs = _checked_band_centres(frequencies, f_pi_signed, n_bands)

    # Criteria 3 and 4 are always available from the indicators.
    criterion_3 = np.asarray(
        (f_pi_signed - indicators.f_pi_unsigned) <= _F_PI_DIFF_LIMIT, dtype=bool
    )
    criterion_4 = np.asarray(indicators.fs <= _FS_LIMIT, dtype=bool)

    # Criterion 1: |LIn(1) - LIn(2)| <= s/2.
    criterion_1: np.ndarray | None = None
    if scan_intensity_level_1 is not None and scan_intensity_level_2 is not None:
        l1 = require_per_band(
            scan_intensity_level_1,
            "scan_intensity_level_1",
            f_pi_signed,
            "indicators.f_pi_signed",
        )
        l2 = require_per_band(
            scan_intensity_level_2,
            "scan_intensity_level_2",
            f_pi_signed,
            "indicators.f_pi_signed",
        )
        if repeatability_limit is not None:
            s = np.broadcast_to(
                np.asarray(repeatability_limit, dtype=np.float64), (n_bands,)
            ).astype(np.float64)
        elif freqs is not None:
            nominal = [round(float(f)) for f in freqs]
            s = np.array([_sigma_r0_9614_3(f) for f in nominal], dtype=np.float64)
        else:
            msg = (
                "Criterion 1 needs the limit s: provide 'frequencies' (Table 1) "
                "or 'repeatability_limit'."
            )
            raise ValueError(msg)
        criterion_1 = np.asarray(np.abs(l1 - l2) <= s / 2.0, dtype=bool)

    # Criterion 2: Ld >= F_pIn(signed), Ld = delta_pI0 - K.
    criterion_2: np.ndarray | None = None
    if pressure_residual_index is not None:
        dpi0 = np.broadcast_to(
            np.asarray(pressure_residual_index, dtype=np.float64), (n_bands,)
        ).astype(np.float64)
        ld = dpi0 - _K_9614_3
        criterion_2 = np.asarray(ld >= f_pi_signed, dtype=bool)

    # Criterion 5: 0,83 <= FS(1)/FS(2) <= 1,2.
    criterion_5: np.ndarray | None = None
    if field_nonuniformity_1 is not None and field_nonuniformity_2 is not None:
        fs1 = _spread_over_bands(field_nonuniformity_1, n_bands)
        fs2 = _spread_over_bands(field_nonuniformity_2, n_bands)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = fs1 / fs2
        criterion_5 = np.asarray(
            (ratio >= _FS_RATIO_LOW) & (ratio <= _FS_RATIO_HIGH), dtype=bool
        )

    qualified: np.ndarray | None = None
    if criterion_1 is not None and criterion_2 is not None:
        # C.1.6.2: where the doubled-density scan satisfies criterion 5, the
        # band qualifies as a final result even if FS(2) >= 2 fails criterion 4.
        non_uniformity_ok = (
            criterion_4 if criterion_5 is None else (criterion_4 | criterion_5)
        )
        qualified = criterion_1 & criterion_2 & criterion_3 & non_uniformity_ok

    return PrecisionCriteria(
        criterion_1=criterion_1,
        criterion_2=criterion_2,
        criterion_3=criterion_3,
        criterion_4=criterion_4,
        criterion_5=criterion_5,
        qualified=qualified,
    )


def _precision_a_weighted_total(
    lw: np.ndarray,
    not_applicable: np.ndarray,
    frequencies: np.ndarray | None,
    n_bands: int,
) -> float:
    """A-weighted total over the applicable bands (ISO 9614-3 clause 9.2 / 4.3).

    Without ``frequencies`` there is no weighting to apply: a single applicable
    band is its own A-weighted total, anything else is ``NaN``.
    """
    if frequencies is None:
        if n_bands == 1 and not bool(not_applicable[0]):
            return float(lw[0])
        return float("nan")
    ck = _a_weighting_corrections(frequencies)
    contrib = 10.0 ** (0.1 * (lw + ck))
    total = float(np.sum(contrib[~not_applicable]))
    return 10.0 * np.log10(total) if total > 0.0 else float("nan")


def sound_power_intensity_precision(
    partial_intensity: np.ndarray,
    areas: np.ndarray,
    *,
    frequencies: np.ndarray | None = None,
    temperature: float = 23.0,
    barometric_pressure: float = 101325.0,
) -> PrecisionIntensityResult:
    r"""Sound power by intensity scanning, precision (ISO 9614-3:2002).

    ``partial_intensity`` is an ``(N, NB)`` array (or ``(N,)`` for a single
    band) of the signed normal intensity :math:`I_{\mathrm{n}i}` on each of the ``N``
    partial surfaces (already the two-scan result), and ``areas`` the ``(N,)``
    partial surface areas :math:`S_i`. The partial powers
    :math:`P_i = I_{\mathrm{n}i} S_i` (Eq. 5) are summed to :math:`P` (Eq. 8) and
    :math:`L_W = 10 \log_{10}(P/P_0)` (Eq. 9); a band with net :math:`P \le 0` is
    flagged (``not_applicable_band``, clause 9.2) and reported as ``NaN``.
    :math:`L_{W0}` normalizes to reference meteorology:

    .. math::

       L_{W0} = L_W - 15 \log_{10}\!\left( \frac{B}{101325} \cdot
       \frac{296.15}{273.15 + \theta} \right) \tag{Eq. 10}

    :param partial_intensity: ``(N, NB)`` signed normal intensity, W/m^2.
    :param areas: ``(N,)`` partial surface areas ``Si``, m^2.
    :param frequencies: ``(NB,)`` nominal mid-band frequencies (Hz), for LWA.
    :param temperature: Air temperature ``theta`` (deg C), for LW0 (Eq. 10).
    :param barometric_pressure: Barometric pressure ``B`` (Pa), for LW0.
    :return: :class:`PrecisionIntensityResult`.
    """
    raw_intensity = np.asarray(partial_intensity, dtype=np.float64)
    seg = np.asarray(areas, dtype=np.float64)
    if seg.ndim != 1:
        msg = "'areas' must be a 1D array of partial surface areas."
        raise ValueError(msg)
    n_seg = seg.shape[0]
    # A 1-D input is unambiguously ``(N,)`` segments with one band -> ``(N, 1)``;
    # a 2-D input is taken as ``(segments, bands)`` as given. Keying off the
    # original ndim avoids misreading a genuine ``(1, N)`` single-segment,
    # N-band array as N segments when ``n_seg == N``.
    if raw_intensity.ndim == 1:
        intensity = raw_intensity.reshape(-1, 1)
    else:
        intensity = np.atleast_2d(raw_intensity)
    if intensity.shape[0] != n_seg:
        msg = (
            f"'partial_intensity' first axis ({intensity.shape[0]}) must match "
            f"the number of 'areas' ({n_seg})."
        )
        raise ValueError(msg)
    # As in the Part 2 scan: a NaN passes every bound, and the surface area
    # summed from it reaches the clause 10 sheet as a number that is not one.
    if not np.all(np.isfinite(seg)):
        msg = "All 'areas' must be finite."
        raise ValueError(msg)
    if np.any(seg <= 0.0):
        msg = "All 'areas' must be positive."
        raise ValueError(msg)
    if temperature <= _ABS_ZERO_C:
        msg = "'temperature' must be above -273.15 degrees Celsius."
        raise ValueError(msg)
    if barometric_pressure <= 0.0:
        msg = "'barometric_pressure' must be positive (Pa)."
        raise ValueError(msg)
    n_bands = intensity.shape[1]
    if frequencies is not None and np.asarray(frequencies).shape != (n_bands,):
        raise ValueError(_FREQUENCIES_BAND_COUNT_MSG)

    partial_power = intensity * seg[:, None]  # Eq. 5
    total_power = np.sum(partial_power, axis=0)  # Eq. 8
    not_applicable = total_power <= 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        lw = np.where(
            total_power > 0.0,
            10.0
            * np.log10(np.maximum(total_power, np.finfo(float).tiny) / _P0_INTENSITY),
            np.nan,
        )

    # Eq. 10: meteorological normalization to 23 deg C / 101 325 Pa.
    norm = 15.0 * np.log10(
        (barometric_pressure / 101325.0) * (296.15 / (273.15 + temperature))
    )
    lw0 = lw - norm

    if np.any(not_applicable):
        warnings.warn(
            "Net sound power is non-positive in one or more bands; ISO "
            "9614-3:2002 is not applicable to those bands (clause 9.2).",
            SoundPowerWarning,
            stacklevel=2,
        )

    # A-weighted total over applicable bands (clause 9.2 / 4.3).
    freqs = None if frequencies is None else np.asarray(frequencies, dtype=np.float64)
    lwa = _precision_a_weighted_total(lw, not_applicable, freqs, n_bands)

    return PrecisionIntensityResult(
        frequencies=freqs,
        partial_power=np.asarray(partial_power, dtype=np.float64),
        sound_power=np.asarray(total_power, dtype=np.float64),
        sound_power_level=np.asarray(lw, dtype=np.float64),
        sound_power_level_normalized=np.asarray(lw0, dtype=np.float64),
        not_applicable_band=np.asarray(not_applicable, dtype=bool),
        surface_area=float(np.sum(seg)),
        sound_power_level_a=lwa,
    )
