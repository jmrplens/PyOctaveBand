#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Sound power level of a noise source by sound-intensity **scanning**:
ISO 9614-2:1996 (engineering, grade 2; survey/control, grade 3).

A probe is swept continuously over each segment of a hypothetical surface that
encloses the source, reporting the time-averaged signed normal intensity
:math:`\langle I_{n,i} \rangle` and mean-square pressure per segment. The
sound power follows from the partial powers
:math:`P_i = \langle I_{n,i} \rangle S_i` summed over the ``N`` segments
(clause 9, equations (5), (6), (12), (13)):

.. math::

   P_i = \langle I_{n,i} \rangle \, S_i \tag{Eq. 12}

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
by segment area ``Si`` while :func:`phonometry.field_indicators` (ISO 9614-1)
assumes equal-area positions, the indicators are computed directly here; only
the dynamic-capability index :math:`L_d = \delta_{pI0} - K` is shared with
:func:`phonometry.dynamic_capability_index`.

Qualification criteria per band (Annex B), where ``K`` is 10 (engineering) or
7 (survey) per Table 1, criterion 2 is mandatory for grade 2 and optional for
grade 3, and the per-segment repeatability limit ``s`` comes from Table 2:

.. math::

   \text{criterion 1:} \quad L_d > F_{pI}, \qquad L_d = \delta_{pI0} - K

   \text{criterion 2:} \quad F_{+/-} \le 3~\text{dB}

   \text{criterion 3:} \quad |L_{Wi}(1) - L_{Wi}(2)| \le s
   \quad \text{per segment}

A band achieves the **engineering** grade when criteria 1, 2 and 3 hold, the
**survey** grade when criteria 1 and 3 hold (clause 8.4), otherwise none.
An A-weighted sound power level omits, besides the non-determinable
:math:`P \le 0` bands, the bands in which criteria 1 and/or 2 are not
satisfied (clause 10.6 b).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

from .._internal.levels_math import weighted_energy_mean
from .intensity import dynamic_capability_index
from .sound_power import SoundPowerWarning, _a_weighting_corrections, _check_grade

_P0 = 1.0e-12  #: Reference sound power, in watts (ISO 9614-2, 3.6.3).
_S0 = 1.0  #: Reference surface area, in square metres (ISO 9614-2, A.2.1).

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


def _table2_s(nominal: int, band_type: BandType) -> float:
    """Grade-2 reproducibility standard deviation s (ISO 9614-2 Table 2), in dB.

    Octave 63-125 Hz / third 50-160 Hz -> 3; octave 250 Hz / third 200-315 Hz
    -> 2; octave 500-4000 Hz / third 400-5000 Hz -> 1,5; third 6300 Hz -> 2,5.
    """
    if band_type == "octave":
        table: dict[int, float] = {
            63: 3.0, 125: 3.0, 250: 2.0, 500: 1.5, 1000: 1.5, 2000: 1.5, 4000: 1.5,
        }
    else:
        table = {
            50: 3.0, 63: 3.0, 80: 3.0, 100: 3.0, 125: 3.0, 160: 3.0,
            200: 2.0, 250: 2.0, 315: 2.0,
            400: 1.5, 500: 1.5, 630: 1.5, 800: 1.5, 1000: 1.5, 1250: 1.5,
            1600: 1.5, 2000: 1.5, 2500: 1.5, 3150: 1.5, 4000: 1.5, 5000: 1.5,
            6300: 2.5,
        }
    if nominal not in table:
        raise ValueError(
            f"No ISO 9614-2 Table 2 standard deviation for {nominal} Hz "
            f"({band_type}); expected a nominal band centre in the qualified "
            "range (Table 2)."
        )
    return table[nominal]


@dataclass(frozen=True)
class SoundPowerIntensityResult:
    r"""Result of an ISO 9614-2:1996 sound-power-by-scanning determination.

    ``partial_power`` is the signed :math:`P_i = \langle I_{n,i} \rangle S_i`
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

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
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
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from .._report.iso9614 import render_intensity_power_report

        return render_intensity_power_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def _level_magnitude(values: np.ndarray) -> np.ndarray:
    r"""Magnitude level :math:`10 \log_{10}(|P_i|/P_0)` in dB, with a tiny-floor
    guard for zeros."""
    guarded = np.maximum(np.abs(values), np.finfo(float).tiny)
    return np.asarray(10.0 * np.log10(guarded / _P0), dtype=np.float64)


def _as_2d(name: str, arr: np.ndarray, n_seg: int, n_bands: int) -> np.ndarray:
    a = np.atleast_2d(np.asarray(arr, dtype=np.float64))
    if a.shape == (1, n_seg) and n_bands != n_seg:
        a = a.T  # a 1D (N_seg,) input arrives as (1, N_seg)
    if a.shape != (n_seg, n_bands):
        raise ValueError(
            f"'{name}' must have shape ({n_seg}, {n_bands}) matching "
            f"'normal_intensity', got {a.shape}."
        )
    return a


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
    :math:`\langle I_{n,i} \rangle` (W/m^2), and ``areas`` the ``(N_seg,)``
    segment areas ``Si`` (m^2). The partial powers
    :math:`P_i = \langle I_{n,i} \rangle S_i` are summed to the band sound
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
    if band_type not in ("octave", "third"):
        raise ValueError("'band_type' must be 'octave' or 'third'.")

    intensity = np.atleast_2d(np.asarray(normal_intensity, dtype=np.float64))
    seg = np.asarray(areas, dtype=np.float64)
    if seg.ndim != 1:
        raise ValueError("'areas' must be a 1D array of segment areas.")
    n_seg = seg.shape[0]
    # A 1D (N_seg,) intensity arrives from atleast_2d as (1, N_seg): transpose.
    if intensity.shape == (1, n_seg) and n_seg != 1:
        intensity = intensity.T
    if intensity.shape[0] != n_seg:
        raise ValueError(
            f"'normal_intensity' first axis ({intensity.shape[0]}) must match "
            f"the number of segment 'areas' ({n_seg})."
        )
    n_bands = intensity.shape[1]
    if frequencies is not None and np.asarray(frequencies).shape != (n_bands,):
        # Validate up front so a mismatched length raises the public ValueError
        # rather than an IndexError from the Table 2 lookup during classification.
        raise ValueError("'frequencies' length must match the number of bands.")
    if np.any(seg <= 0.0):
        raise ValueError("All segment 'areas' must be positive.")
    if n_seg < 4:
        warnings.warn(
            f"Only {n_seg} segment(s); ISO 9614-2:1996 clause 8.2 requires at "
            "least 4 measurement segments.",
            SoundPowerWarning,
            stacklevel=2,
        )

    # --- partial powers, using the mean of the two sweeps when available -----
    repeatability: np.ndarray | None = None
    if normal_intensity_2 is not None:
        scan2 = _as_2d("normal_intensity_2", np.asarray(normal_intensity_2), n_seg, n_bands)
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
            raise ValueError(
                "The achieved grade needs the criterion-3 limit s: provide "
                "'frequencies' (Table 2 lookup) or 'repeatability_limit'."
            )
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
    stated. Criterion 1 (:math:`L_d > F_{pI}`, B.1.1) needs the surface
    pressure-intensity indicator and the dynamic capability index; without them
    the screening cannot be evaluated and ``None`` is returned. Criterion 2
    (:math:`F_{+/-} \le 3` dB, B.1.2) binds engineering-grade determinations
    only
    (optional for grade 3, Note 16). Net-negative bands are excluded already
    as non-determinable (clause 9.2), so they are not flagged here."""
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
    determinable band is summed and a :class:`SoundPowerWarning` is emitted."""
    determinable = ~negative_band
    if frequencies is not None:
        freqs = np.asarray(frequencies, dtype=np.float64)
        if freqs.shape[0] != n_bands:
            raise ValueError("'frequencies' length must match the number of bands.")
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
