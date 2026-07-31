#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Rated microphone characteristics (IEC 60268-4).

A microphone measurement/rating report gathers the *rated characteristics*
IEC 60268-4:2014 defines around a measured free-field frequency response: the
**free-field sensitivity** and its level re 1 V/Pa (clauses 11.1/11.2.1), the
**frequency response** with the manufacturer's tolerance (12.1) and the
**effective frequency range** read against that tolerance (12.2), the
**directional pattern** and the **directivity index** (13.1/13.2), the
**equivalent sound pressure level due to inherent noise** (17), the **overload
sound pressure level** at a stated total-harmonic-distortion limit (14.2/15.2),
the **rated impedance** and the **rated minimum permitted load impedance**
(10.2/10.3) and the **rated power supply** (9.1). This module bundles those
into a single :class:`MicrophoneCharacteristics` result whose ``report`` method
renders the IEC 60268-4 rated-characteristics fiche, with the characteristic
graphs laid out to the IEC 60263:1982 scale conventions.

Four of the characteristics are *computed* from the standard's own definitions
so the report never merely repeats a manufacturer number:

* **Sensitivity level** (11.1). The sensitivity ``M`` is the ratio of the
  output voltage to the sound pressure, in volts per pascal; its level is

      L_M = 20 lg(M / M_r),   M_r = 1 V/Pa,

  the rated sensitivity referring to the standard reference frequency of
  1 000 Hz (11.3). This is the first clean-room oracle: 12,5 mV/Pa returns
  ``20 lg 0,0125 = -38,06`` dB re 1 V/Pa exactly.

* **Effective frequency range** (12.2). The range of frequencies over which
  the response does not deviate by more than a specified amount from the ideal
  (flat) response through the reference-frequency level. The band edges are
  the interpolated frequencies where the relative response crosses the
  ``+/- tolerance`` limits on either side of the reference frequency, which is
  the second oracle: a response crossing a limit at chosen frequencies returns
  exactly those frequencies.

* **Directivity index** (13.2.2). ``D = 20 lg(M_0 / M_diff)`` where the
  diffuse-field sensitivity of a rotationally symmetric pattern follows
  11.2.2 a):

      M_diff^2 = (1/2) * integral over 0..pi of M^2(theta) sin(theta) dtheta.

  For the ideal cardioid ``M(theta) = M_0 (1 + cos theta) / 2`` the integral
  is ``M_0^2 / 3``, so ``D = 10 lg 3 = 4,77`` dB, the third oracle.

* **Equivalent sound pressure level due to inherent noise** (17.2 d/e). The
  equivalent sound pressure is the ratio of the weighted inherent-noise output
  voltage to the rated free-field sensitivity, ``p_N = U_N / M``, and its
  level is ``L_N = 20 lg(p_N / p_0)`` with ``p_0 = 20`` uPa, the fourth
  oracle. The overload sound pressure level (15.2.2) is read from a measured
  distortion-against-level curve as the interpolated sound pressure level
  where the distortion reaches the specified limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.validation import require_positive
from .loudspeaker import _as_curve, _threshold_crossing

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray

    from .._report.metadata import ReportMetadata

#: Standard reference sound pressure, 20 uPa (IEC 60268-4 17.2 e).
_P_REF = 20e-6
#: Reference sensitivity M_r = 1 V/Pa (IEC 60268-4 11.1).
_M_REF = 1.0
#: Standard reference frequency for the rated sensitivity, Hz (11.3).
_REFERENCE_FREQUENCY = 1000.0
#: Smallest polar-pattern angular span, in degrees, accepted for the 11.2.2 a)
#: diffuse-field integral (the pattern must essentially reach the rear).
_MIN_POLAR_SPAN_DEG = 150.0


def _sensitivity_level_db(sensitivity_v_per_pa: float) -> float:
    """Sensitivity level ``20 lg(M / 1 V/Pa)``, in dB re 1 V/Pa (11.1)."""
    return float(20.0 * np.log10(sensitivity_v_per_pa / _M_REF))


def _normalized_response(
    f: NDArray[np.float64], response_db: NDArray[np.float64], f_ref: float
) -> NDArray[np.float64]:
    """The response relative to its level at the reference frequency (12.1.1).

    The reference-frequency level is interpolated linearly in log-frequency, so
    the returned curve is 0 dB at ``f_ref`` even when it falls between samples.
    """
    ref_level = float(np.interp(np.log2(f_ref), np.log2(f), response_db))
    return np.asarray(response_db - ref_level, dtype=np.float64)


def _limit_edge(
    f: NDArray[np.float64],
    rel_db: NDArray[np.float64],
    tolerance: float,
    edge: int,
    neighbour: int,
) -> float:
    """Interpolated ``+/- tolerance`` crossing between ``edge`` and ``neighbour``.

    ``edge`` is inside the tolerance and ``neighbour`` outside it; the crossed
    limit is the one the neighbour exceeds. When the in-band region already
    reaches the measured band end (no outer neighbour), the measured edge
    frequency is returned.
    """
    if not 0 <= neighbour < f.size:
        return float(f[edge])
    threshold = tolerance if rel_db[neighbour] > tolerance else -tolerance
    return _threshold_crossing(
        float(f[edge]), float(rel_db[edge]),
        float(f[neighbour]), float(rel_db[neighbour]), threshold,
    )


def _effective_range(
    f: NDArray[np.float64],
    rel_db: NDArray[np.float64],
    tolerance: float,
    f_ref: float,
) -> tuple[float, float]:
    """Effective frequency range against the ``+/- tolerance`` limits (12.2).

    The contiguous band around the reference frequency over which the relative
    response stays within the tolerance; the edges are the interpolated limit
    crossings, or the measured band edge where the response never leaves it.
    """
    within = np.abs(rel_db) <= tolerance
    anchor = int(np.argmin(np.abs(np.log2(f / f_ref))))
    if not within[anchor]:
        raise ValueError(
            "the response deviates beyond 'tolerance_db' at the reference "
            "frequency, so no effective frequency range contains it."
        )
    lo_idx = anchor
    while lo_idx > 0 and within[lo_idx - 1]:
        lo_idx -= 1
    hi_idx = anchor
    while hi_idx < within.size - 1 and within[hi_idx + 1]:
        hi_idx += 1
    lower = _limit_edge(f, rel_db, tolerance, lo_idx, lo_idx - 1)
    upper = _limit_edge(f, rel_db, tolerance, hi_idx, hi_idx + 1)
    return lower, upper


def _fold_angles_deg(angles_deg: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fold full-circle angles onto the 0..180 half plane (rotational symmetry).

    The 11.2.2 a) integral runs over the polar angle 0..pi; for a rotationally
    symmetric pattern an angle measured beyond 180 degrees re-measures the
    polar angle ``360 - angle``, so the rear half of a full-circle measurement
    folds onto the front half.
    """
    return np.asarray(
        np.where(angles_deg > 180.0, 360.0 - angles_deg, angles_deg),
        dtype=np.float64,
    )


def _directivity_index_from_polar(
    angles_deg: NDArray[np.float64], rel_db: NDArray[np.float64]
) -> float:
    """Directivity index from a rotationally symmetric pattern (13.2.2).

    ``D = 20 lg(M_0 / M_diff)`` with the diffuse-field sensitivity from the
    11.2.2 a) integral, evaluated by the trapezoidal rule over the supplied
    angles. The polar levels are relative to the reference axis (13.1.2), so
    ``Gamma(theta) = 10 ** (G(theta) / 20)`` with ``Gamma(0) = 1``. Angles
    beyond 180 degrees are folded onto ``360 - angle`` first
    (:func:`_fold_angles_deg`), interleaving both halves of a full-circle
    measurement into the same 0..pi integral (duplicated folded angles form
    zero-width trapezoid segments and contribute nothing extra).
    """
    folded = _fold_angles_deg(angles_deg)
    order = np.argsort(folded, kind="stable")
    theta = np.radians(folded[order])
    gamma_sq = 10.0 ** (rel_db[order] / 10.0)
    ratio = 0.5 * float(np.trapezoid(gamma_sq * np.sin(theta), theta))
    return float(-10.0 * np.log10(ratio))


def _overload_spl(
    spl_db: NDArray[np.float64],
    thd_percent: NDArray[np.float64],
    limit_percent: float,
) -> float | None:
    """Sound pressure level where the distortion reaches the limit (15.2.2).

    Interpolated linearly between the bracketing samples of the measured
    distortion-against-level curve; ``None`` when the curve never reaches the
    limit, and the lowest measured level when it already starts above it.
    """
    if thd_percent.size == 0:  # pragma: no cover - _as_curve enforces two points
        return None
    if float(thd_percent[0]) >= limit_percent:
        return float(spl_db[0])
    above = thd_percent >= limit_percent
    if not bool(np.any(above)):
        return None
    j = int(np.argmax(above))
    i = j - 1
    frac = (limit_percent - float(thd_percent[i])) / (
        float(thd_percent[j]) - float(thd_percent[i])
    )
    return float(spl_db[i] + frac * (spl_db[j] - spl_db[i]))


@dataclass(frozen=True)
class MicrophoneCharacteristics:
    """Rated microphone characteristics for an IEC 60268-4 report.

    The free-field frequency response and the rated free-field sensitivity are
    the required inputs; the directional pattern, the inherent-noise spectrum
    and the distortion-against-level curve are optional panels rendered when
    supplied. The sensitivity level, the effective frequency range, the
    directivity index and the equivalent noise level are computed from the
    standard's definitions (see the module docstring).

    :ivar frequencies: Free-field response frequency axis, in Hz.
    :ivar response_db: Free-field frequency response relative to the level at
        :attr:`reference_frequency` (0 dB there), in dB (12.1.1).
    :ivar reference_frequency: Stated reference frequency of the rated
        sensitivity and the response normalization, in Hz (11.3).
    :ivar sensitivity_mv_per_pa: Rated free-field sensitivity ``M`` at the
        reference frequency, in mV/Pa (11.2.1/11.3).
    :ivar sensitivity_level_db: Sensitivity level ``20 lg(M / 1 V/Pa)``, in dB
        re 1 V/Pa (11.1).
    :ivar tolerance_db: Half-width of the response tolerance, in dB (12.1.1).
    :ivar effective_range: Computed effective frequency range ``(lo, hi)``
        against the tolerance limits, in Hz (12.2).
    :ivar rated_impedance: Rated (internal) impedance, in ohm (10.2), or
        ``None``.
    :ivar minimum_load_impedance: Rated minimum permitted load impedance, in
        ohm (10.3), or ``None``.
    :ivar equivalent_noise_level_db: Equivalent sound pressure level due to
        inherent noise, in dB SPL with :attr:`noise_weighting` weighting (17),
        or ``None``.
    :ivar noise_weighting: Weighting of the inherent-noise measurement
        (IEC 60268-1), ``"A"`` by default.
    :ivar max_spl_db: Overload sound pressure level at
        :attr:`max_spl_thd_percent` total harmonic distortion, in dB SPL
        (15.2), or ``None``.
    :ivar max_spl_thd_percent: Distortion limit defining :attr:`max_spl_db`,
        in % (15.2.1).
    :ivar distortion_spl_db: Distortion-curve sound-pressure-level axis, in
        dB SPL (14.2), or ``None``.
    :ivar distortion_thd_percent: Total harmonic distortion against level, in
        % (14.2), or ``None``.
    :ivar noise_frequencies: Inherent-noise spectrum frequency axis, in Hz
        (17.2 b), or ``None``.
    :ivar noise_band_levels_db: Inherent-noise equivalent band levels, in
        dB SPL (17.2 b), or ``None``.
    :ivar polar_angles_deg: Directional-pattern angles, in degrees (13.1), or
        ``None``.
    :ivar polar_db: Directional pattern ``G(theta)`` relative to the
        reference-axis response, in dB (13.1.2), or ``None``.
    :ivar polar_frequency: Stated frequency of the directional pattern, in Hz,
        or ``None``.
    :ivar directivity_index_db: Directivity index at :attr:`polar_frequency`,
        in dB (13.2), or ``None``.
    :ivar powering: Rated power supply description, e.g. the IEC 61938
        phantom-powering designation and voltage (9.1), or ``None``.
    :ivar supply_current_ma: Current drawn from the power supply, in mA (9.1),
        or ``None``.
    """

    frequencies: NDArray[np.float64]
    response_db: NDArray[np.float64]
    reference_frequency: float
    sensitivity_mv_per_pa: float
    sensitivity_level_db: float
    tolerance_db: float
    effective_range: tuple[float, float]
    rated_impedance: float | None
    minimum_load_impedance: float | None
    equivalent_noise_level_db: float | None
    noise_weighting: str
    max_spl_db: float | None
    max_spl_thd_percent: float
    distortion_spl_db: NDArray[np.float64] | None
    distortion_thd_percent: NDArray[np.float64] | None
    noise_frequencies: NDArray[np.float64] | None
    noise_band_levels_db: NDArray[np.float64] | None
    polar_angles_deg: NDArray[np.float64] | None
    polar_db: NDArray[np.float64] | None
    polar_frequency: float | None
    directivity_index_db: float | None
    powering: str | None
    supply_current_ma: float | None

    @property
    def sensitivity_v_per_pa(self) -> float:
        """Rated free-field sensitivity ``M``, in V/Pa (11.1)."""
        return float(self.sensitivity_mv_per_pa / 1000.0)

    @property
    def signal_to_noise_ratio_db(self) -> float | None:
        """Signal-to-noise ratio re 1 Pa (94 dB SPL), in dB, or ``None``.

        The datasheet companion of the equivalent noise level: the level of
        1 Pa (``20 lg(1 Pa / 20 uPa) = 93,98`` dB SPL) minus the equivalent
        sound pressure level due to inherent noise (17), carrying the same
        weighting.
        """
        if self.equivalent_noise_level_db is None:
            return None
        return float(20.0 * np.log10(1.0 / _P_REF) - self.equivalent_noise_level_db)

    @property
    def diffuse_field_sensitivity_level_db(self) -> float | None:
        """Diffuse-field sensitivity level, in dB re 1 V/Pa, or ``None``.

        Per 11.2.2.1 the diffuse-field sensitivity level equals the free-field
        plane-wave sensitivity level minus the directivity index (13.2).
        """
        if self.directivity_index_db is None:
            return None
        return float(self.sensitivity_level_db - self.directivity_index_db)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render the IEC 60268-4 microphone-characteristics fiche to a PDF.

        Writes a one-page rated-characteristics data sheet: the standard-basis
        line (IEC 60268-4:2014, graphs to IEC 60263:1982), an optional metadata
        header, the rated-characteristics table beside the free-field response
        with its tolerance band and effective-range markers, the directional,
        inherent-noise and distortion panels for the data supplied, a boxed
        sensitivity/range result, an optional equivalent-noise verdict when a
        requirement is given, and the footer.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the header identity and, through ``requirement``, a maximum
            permitted equivalent noise level (dB SPL) the verdict row compares
            against.
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for a uniform ``.report()`` signature; the
            fiche has one layout, so it has no effect.
        :param language: Fiche language: ``"en"`` (default) or ``"es"``.
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"`` or ``language``
            is unknown.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        del verbose  # uniform signature; the fiche has a single layout
        from .._report.iec60268_4 import render_iec60268_4_report

        return render_iec60268_4_report(
            self, path, metadata=metadata, language=language
        )

    def plot(
        self, quantity: str = "response", ax: Axes | None = None, *,
        language: str = "en", **kwargs: Any,
    ) -> Axes:
        """Plot one IEC 60268-4 microphone rated characteristic on one axes.

        One concept per figure, drawn by the same shared renderer the
        ``.report()`` fiche composes: ``"response"`` (the free-field response
        with its tolerance band, reference-frequency and effective-range
        markers, the default), ``"directivity"`` (the polar directional pattern
        on the 25 dB reference circle), ``"noise"`` (the inherent-noise
        band-level spectrum) and ``"distortion"`` (total harmonic distortion
        against sound pressure level).

        :param quantity: Which characteristic to plot (see above).
        :param ax: Existing axes to draw on, or ``None`` for a fresh figure (a
            polar axes is created for ``"directivity"``).
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :return: The axes the characteristic was drawn on.
        :raises ValueError: If ``quantity`` or ``language`` is unknown, or the
            result carries no data for ``quantity``.
        :raises ImportError: If matplotlib is not installed
            (``pip install phonometry[plot]``).
        """
        from .._i18n import check_language
        from .._plot.electroacoustics import plot_microphone_characteristics

        check_language(language)
        return plot_microphone_characteristics(
            self, quantity, ax=ax, language=language, **kwargs
        )


def _optional_positive(value: float | None, name: str) -> float | None:
    """Validate a supplied rated value as finite and positive, else keep ``None``."""
    return require_positive(value, name) if value is not None else None


def _resolve_noise_level(
    noise_voltage: float | None,
    equivalent_noise_level_db: float | None,
    sensitivity_v_per_pa: float,
) -> float | None:
    """Resolve the equivalent noise level, computing it from a voltage (17.2)."""
    if noise_voltage is not None and equivalent_noise_level_db is not None:
        raise ValueError(
            "give either 'noise_voltage' or 'equivalent_noise_level_db', not both."
        )
    if noise_voltage is not None:
        u_n = require_positive(noise_voltage, "noise_voltage")
        return float(20.0 * np.log10((u_n / sensitivity_v_per_pa) / _P_REF))
    if equivalent_noise_level_db is not None and not np.isfinite(
        equivalent_noise_level_db
    ):
        raise ValueError("'equivalent_noise_level_db' must be finite.")
    return equivalent_noise_level_db


def _resolve_distortion(
    distortion: tuple[ArrayLike, ArrayLike] | None,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Validate the optional THD-against-level curve, in (dB SPL, %) (14.2)."""
    if distortion is None:
        return None, None
    spl, thd = _as_curve(distortion[0], distortion[1], "distortion")
    if np.any(thd < 0.0):
        raise ValueError("'distortion' THD values must be non-negative.")
    return spl, thd


def _resolve_polar(
    polar: tuple[ArrayLike, ArrayLike] | None,
    directivity_index_db: float | None,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
    """Resolve the optional directional pattern and directivity index (13.1/13.2).

    A stated ``directivity_index_db`` is kept; otherwise it is computed from
    the pattern via the 11.2.2 a) integral when the supplied angles, folded
    onto the 0..180 half plane, span at least :data:`_MIN_POLAR_SPAN_DEG`
    towards the rear.
    """
    if directivity_index_db is not None and not np.isfinite(directivity_index_db):
        raise ValueError("'directivity_index_db' must be finite.")
    if polar is None:
        return None, None, directivity_index_db
    p_ang = np.atleast_1d(np.asarray(polar[0], dtype=np.float64))
    p_db = np.atleast_1d(np.asarray(polar[1], dtype=np.float64))
    if p_ang.ndim != 1 or p_ang.shape != p_db.shape:
        raise ValueError("'polar' angles and levels must be 1-D and equal length.")
    if p_ang.size < 2:
        raise ValueError("'polar' needs at least two angle points.")
    if not (np.all(np.isfinite(p_ang)) and np.all(np.isfinite(p_db))):
        raise ValueError("'polar' angles and levels must be finite.")
    if np.any(p_ang < 0.0) or np.any(p_ang > 360.0):
        raise ValueError("'polar' angles must lie within 0..360 degrees.")
    order = np.argsort(p_ang)
    p_ang, p_db = p_ang[order], p_db[order]
    di = directivity_index_db
    if di is None and float(np.max(_fold_angles_deg(p_ang))) >= _MIN_POLAR_SPAN_DEG:
        di = _directivity_index_from_polar(p_ang, p_db)
    return p_ang, p_db, di


def microphone_characteristics(
    frequencies: ArrayLike,
    response_db: ArrayLike,
    sensitivity_mv_per_pa: float,
    *,
    reference_frequency: float = _REFERENCE_FREQUENCY,
    tolerance_db: float = 2.0,
    rated_impedance: float | None = None,
    minimum_load_impedance: float | None = None,
    noise_voltage: float | None = None,
    equivalent_noise_level_db: float | None = None,
    noise_weighting: str = "A",
    max_spl_db: float | None = None,
    max_spl_thd_percent: float = 1.0,
    distortion: tuple[ArrayLike, ArrayLike] | None = None,
    noise_spectrum: tuple[ArrayLike, ArrayLike] | None = None,
    polar: tuple[ArrayLike, ArrayLike] | None = None,
    polar_frequency: float | None = None,
    directivity_index_db: float | None = None,
    powering: str | None = None,
    supply_current_ma: float | None = None,
) -> MicrophoneCharacteristics:
    """Assemble the rated microphone characteristics for an IEC 60268-4 report.

    The sensitivity level (11.1), the effective frequency range (12.2), the
    directivity index (13.2.2, when a rear-reaching pattern is supplied) and
    the equivalent noise level (17.2, when the noise voltage is supplied) are
    computed from the standard's definitions; the optional directional, noise
    and distortion data feed the corresponding report panels.

    :param frequencies: Free-field response frequency axis, in Hz (1-D, > 0).
    :param response_db: Free-field frequency response, in dB, relative to the
        output at a stated frequency (12.1.1); it is re-normalized to 0 dB at
        ``reference_frequency``.
    :param sensitivity_mv_per_pa: Rated free-field sensitivity ``M`` at the
        reference frequency, in mV/Pa (11.2.1/11.3).
    :param reference_frequency: Stated reference frequency, in Hz; the 11.3
        standard reference frequency of 1 000 Hz by default.
    :param tolerance_db: Half-width of the response tolerance, in dB
        (default 2), defining the effective frequency range (12.2).
    :param rated_impedance: Rated impedance, in ohm (10.2).
    :param minimum_load_impedance: Rated minimum permitted load impedance, in
        ohm (10.3).
    :param noise_voltage: Weighted r.m.s. output voltage due to inherent
        noise, in V (17.2 b); the equivalent noise level is computed from it.
    :param equivalent_noise_level_db: Stated equivalent sound pressure level
        due to inherent noise, in dB SPL (17.1), when not computed from
        ``noise_voltage``.
    :param noise_weighting: Weighting of the inherent-noise measurement
        (default ``"A"``, the IEC 60268-1 6.2.1 recommendation).
    :param max_spl_db: Stated overload sound pressure level, in dB SPL (15.2);
        when omitted it is read from ``distortion`` at ``max_spl_thd_percent``.
    :param max_spl_thd_percent: Distortion limit defining the overload level,
        in % (default 1, a common 15.2.1 note value).
    :param distortion: Total harmonic distortion against level as
        ``(spl_db, thd_percent)`` (14.2).
    :param noise_spectrum: Inherent-noise spectrum as
        ``(frequencies, band_levels_db)`` in (Hz, dB SPL) (17.2 b).
    :param polar: Directional pattern as ``(angles_deg, relative_db)`` at
        ``polar_frequency`` (13.1.2).
    :param polar_frequency: Stated frequency of the directional pattern, Hz.
    :param directivity_index_db: Stated directivity index, in dB (13.2); when
        omitted it is computed from ``polar`` if the pattern reaches the rear.
    :param powering: Rated power supply description (9.1), e.g.
        ``"Phantom P48, 48 V (IEC 61938)"``.
    :param supply_current_ma: Current drawn from the power supply, in mA (9.1).
    :return: A :class:`MicrophoneCharacteristics`.
    :raises ValueError: If the inputs are invalid.
    """
    f, resp = _as_curve(frequencies, response_db, "frequency response")
    m_mv = require_positive(sensitivity_mv_per_pa, "sensitivity_mv_per_pa")
    f_ref = require_positive(reference_frequency, "reference_frequency")
    tol = require_positive(tolerance_db, "tolerance_db")
    if not float(f[0]) <= f_ref <= float(f[-1]):
        raise ValueError(
            "'reference_frequency' must lie within the measured frequency band."
        )
    rel = _normalized_response(f, resp, f_ref)
    effective = _effective_range(f, rel, tol, f_ref)
    level = _sensitivity_level_db(m_mv / 1000.0)
    noise_level = _resolve_noise_level(
        noise_voltage, equivalent_noise_level_db, m_mv / 1000.0
    )
    thd_limit = require_positive(max_spl_thd_percent, "max_spl_thd_percent")
    d_spl, d_thd = _resolve_distortion(distortion)
    max_spl: float | None = None
    if max_spl_db is not None:
        if not np.isfinite(max_spl_db):
            raise ValueError("'max_spl_db' must be finite.")
        max_spl = float(max_spl_db)
    elif d_spl is not None and d_thd is not None:
        max_spl = _overload_spl(d_spl, d_thd, thd_limit)
    n_f: NDArray[np.float64] | None = None
    n_db: NDArray[np.float64] | None = None
    if noise_spectrum is not None:
        n_f, n_db = _as_curve(noise_spectrum[0], noise_spectrum[1], "noise_spectrum")
    p_ang, p_db, di = _resolve_polar(polar, directivity_index_db)

    return MicrophoneCharacteristics(
        frequencies=f,
        response_db=rel,
        reference_frequency=f_ref,
        sensitivity_mv_per_pa=m_mv,
        sensitivity_level_db=level,
        tolerance_db=tol,
        effective_range=effective,
        rated_impedance=_optional_positive(rated_impedance, "rated_impedance"),
        minimum_load_impedance=_optional_positive(
            minimum_load_impedance, "minimum_load_impedance"
        ),
        equivalent_noise_level_db=noise_level,
        noise_weighting=str(noise_weighting),
        max_spl_db=max_spl,
        max_spl_thd_percent=thd_limit,
        distortion_spl_db=d_spl,
        distortion_thd_percent=d_thd,
        noise_frequencies=n_f,
        noise_band_levels_db=n_db,
        polar_angles_deg=p_ang,
        polar_db=p_db,
        polar_frequency=(
            require_positive(polar_frequency, "polar_frequency")
            if polar_frequency is not None
            else None
        ),
        directivity_index_db=di,
        powering=powering,
        supply_current_ma=_optional_positive(supply_current_ma, "supply_current_ma"),
    )
