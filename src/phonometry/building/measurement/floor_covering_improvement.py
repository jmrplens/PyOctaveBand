#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Impact-sound improvement of floor coverings on a small mock-up
(ISO 16251-1:2014).

Laboratory method for the **improvement of impact sound insulation** ``ΔL`` of a
soft, locally-reacting floor covering (carpet, PVC, linoleum; ISO 10140-1
Annex H category I) laid on a small concrete mock-up plate and excited by a
standard tapping machine. The two rooms of the ISO 10140 series are removed and
the floor is replaced by a softly-supported concrete plate; instead of the
receiving-room sound pressure level, the **structure-borne acceleration level**
on the underside of the plate is measured with and without the covering. For
locally-reacting coverings this acceleration-level difference equals the ISO
10140 impact sound reduction (Clause 4).

**Acceleration level (Formula (1)).**
:math:`L_a = 10 \log_{10}[(1/T_m) \int a(t)^2/a_0^2\,dt]` dB,
with the reference acceleration :math:`a_0 = 10^{-6}` m/s², i.e.
:math:`L_a = 10 \log_{10}(\overline{a^2}/a_0^2)`.

**Background correction (Formula (2)).** Each measured level ``L'`` is
corrected against the background ``Lb`` per accelerometer position, by the
margin :math:`L' - L_\mathrm{b}`: unchanged for :math:`\ge 15` dB; energy
subtraction :math:`10 \log_{10}(10^{L'/10} - 10^{L_\mathrm{b}/10})` for
:math:`6 \le \mathrm{margin} < 15` dB; and the fixed :math:`L' - 1.3` dB limit
for :math:`< 6` dB. Bands hitting the 1.3 dB limit are flagged as the *limit of
measurement* (reported as :math:`> \Delta L`). This differs from the ISO
10140-4 correction (:func:`phonometry.building.background_correction`) only at
exactly :math:`\mathrm{margin} = 6` dB.

**Improvement (Formulae (3)/(4)).** The per-position difference is
:math:`\Delta L_{t,a} = L_{0,t,a} - L_{1,t,a}` (0 = bare plate, 1 = specimen)
and the improvement is
their arithmetic mean over all tapping-machine (t) and accelerometer (a)
positions, :math:`\Delta L = (1/(t \cdot a)) \sum_t \sum_a \Delta L_{t,a}`.

**Octave bands (Formula (5)).**
:math:`\Delta L_\mathrm{oct} = -10 \log_{10}[(1/3) \sum 10^{-\Delta L_j/10}]` dB from
the three one-third-octave values in each octave.

**Weighted improvement.** ``ΔLw`` is the ISO 717-2 weighted reduction of impact
sound pressure level (Clause 6.5), computed with the heavyweight reference
floor via :func:`phonometry.building.weighted_impact_improvement` on the 16
rating bands 100 Hz to 3150 Hz; a wider clause 6.3 spectrum (18 bands 100-5000
Hz, optionally extended to 50 Hz) is rated on that sub-range. The statement of
results (Clause 8 e)) also carries the spectrum adaptation term ``CI,Δ`` (ISO
717-2:2020 Formula (A.4)) via
:func:`phonometry.building.impact_improvement_adaptation_term`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .insulation import (
    impact_improvement_adaptation_term,
    weighted_impact_improvement,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

    from ..._report.metadata import ReportMetadata

__all__ = [
    "FloorCoveringImprovementResult",
    "acceleration_level",
    "background_corrected_level",
    "impact_improvement",
    "improvement_octave_bands",
]

#: Reference acceleration ``a0`` for the acceleration level (Formula (1)): 1e-6 m/s².
_ACCELERATION_REFERENCE = 1e-6

#: Background-correction thresholds (Formula (2)), in dB.
_MARGIN_NEGLIGIBLE = 15.0
_MARGIN_LIMIT = 6.0
_LIMIT_CORRECTION = 1.3

#: The 16 one-third-octave centre frequencies rated by ISO 717-2 (100-3150 Hz).
_RATING_FREQS = (
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
)


def _finite(values: float | Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    a = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if not np.all(np.isfinite(a)):
        raise ValueError(f"'{name}' must contain only finite values.")
    return a


def acceleration_level(
    acceleration: float | Sequence[float] | np.ndarray,
    *,
    reference: float = _ACCELERATION_REFERENCE,
) -> np.ndarray:
    r"""Vibratory acceleration level ``La`` (ISO 16251-1 Formula (1)).

    :math:`L_a = 10 \log_{10}(a_\mathrm{rms}^2 / a_0^2) = 20 \log_{10}(a_\mathrm{rms} / a_0)` dB.

    :param acceleration: RMS acceleration ``a_rms`` per band, in m/s² (> 0).
    :param reference: Reference acceleration ``a0``, in m/s² (default 1e-6).
    :return: The acceleration level ``La`` per band, in dB.
    :raises ValueError: Non-finite or non-positive acceleration, or a
        non-positive reference.
    """
    a = _finite(acceleration, "acceleration")
    if np.any(a <= 0.0):
        raise ValueError("'acceleration' must be positive.")
    if reference <= 0.0:
        raise ValueError("'reference' must be positive.")
    return 20.0 * np.log10(a / reference)


def background_corrected_level(
    signal_and_background: float | Sequence[float] | np.ndarray,
    background: float | Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Correct a measured level for background noise (ISO 16251-1 F. (2)).

    :param signal_and_background: Measured level ``L'`` (with background) per
        band, in dB.
    :param background: Background-noise level ``Lb`` per band, in dB.
    :return: ``(corrected, limited)``: the corrected levels ``L`` per band,
        in dB, and a boolean mask of bands at the 1.3 dB limit of measurement
        (report as :math:`> \Delta L`).
    :raises ValueError: Mismatched shapes or non-finite values.
    """
    lp = _finite(signal_and_background, "signal_and_background")
    lb = _finite(background, "background")
    if lp.shape != lb.shape:
        raise ValueError(
            "'signal_and_background' and 'background' must share their shape."
        )
    margin = lp - lb
    diff = 10.0 ** (lp / 10.0) - 10.0 ** (lb / 10.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        subtracted = 10.0 * np.log10(np.where(diff > 0.0, diff, 1.0))
    # >= 15: unchanged; 6 <= margin < 15: energy subtraction; < 6: 1,3 dB limit.
    corrected = np.where(margin >= _MARGIN_NEGLIGIBLE, lp, subtracted)
    limited = margin < _MARGIN_LIMIT
    corrected = np.where(limited, lp - _LIMIT_CORRECTION, corrected)
    return corrected, limited


@dataclass(frozen=True)
class FloorCoveringImprovementResult:
    r"""Impact-sound improvement of a floor covering (ISO 16251-1).

    :ivar frequencies: One-third-octave band centre frequencies, in Hz.
    :ivar improvement: Improvement of impact sound insulation ``ΔL`` per
        band, in dB.
    :ivar limited: Per-band boolean mask of bands at the 1.3 dB limit of
        measurement (reported as :math:`> \Delta L`); all ``False`` when no
        background
        correction was applied.
    :ivar delta_lw: Weighted improvement ``ΔLw`` (ISO 717-2), in dB, or ``None``
        when the spectrum does not contain the 16 one-third-octave rating
        bands 100-3150 Hz. A wider clause 6.3 spectrum (e.g. the 18 bands
        100-5000 Hz, optionally extended down to 50 Hz) is rated on its
        100-3150 Hz sub-range.
    :ivar ci_delta: Spectrum adaptation term ``CI,Δ`` (ISO 717-2:2020
        Formula (A.4); required in the ISO 16251-1 Clause 8 e) statement of
        results), in dB, or ``None`` when ``delta_lw`` is ``None``.
    """

    frequencies: np.ndarray
    improvement: np.ndarray
    limited: np.ndarray
    delta_lw: int | None
    ci_delta: int | None = None

    def octave_bands(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(octave_freqs, ΔLoct)`` via Formula (5) (needs 16 1/3-oct bands)."""
        return improvement_octave_bands(self.improvement, self.frequencies)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the improvement spectrum ``ΔL`` (with ``ΔLw`` when available).

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_floor_covering_improvement

        check_language(language)
        return plot_floor_covering_improvement(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        r"""Render an ISO 16251-1 impact-improvement test-report fiche to PDF.

        Writes a one-page accredited report (BS EN ISO 16251-1:2014, the
        small-mock-up laboratory method for the reduction of transmitted impact
        sound by soft floor coverings): the standard-basis line, an optional
        metadata header block, a two-panel body with the per-band table
        (frequency and the reduction of impact sound pressure level ``ΔL``,
        bands at the 1,3 dB limit of measurement prefixed ``>``) beside the
        ``ΔL(f)`` improvement curve, a boxed single-number weighted improvement
        ``ΔLw (CI,Δ)`` (ISO 717-2, the Clause 8 e) statement of results; a
        characterisation headline replaces it when the spectrum lacks the 16
        rating bands 100-3150 Hz), an optional verdict row and a footer.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`; ``None``
            produces a body-and-disclaimer fiche whose header shows only the
            measured frequency range. The applicable descriptive fields are
            ``client``, ``manufacturer``, ``specimen`` (the floor covering under
            test), ``mounting``, ``mass_per_area``, ``test_room``,
            ``test_date``, ``temperature``, ``pressure``,
            ``measurement_standard``, ``laboratory``, ``operator``,
            ``report_id`` and ``notes``. The bare reference floor is the
            standardised heavyweight floor of ISO 717-2:2020 Table 4, fixed by
            the standard. When ``requirement`` is set the fiche adds a verdict
            row (a higher weighted improvement is better, so the result passes
            at or above the requirement).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True``, the per-band table gains the
            reference-floor-with-covering column
            :math:`L_\mathrm{n,r} = L_\mathrm{n,r,0} - \Delta L`
            (ISO 717-2:2020 Formula (1)), the derivation basis of ``ΔLw``, when
            the spectrum is exactly the 16 rating bands 100-3150 Hz.
        :param language: Fiche language: ``"en"`` (default, English, decimal
            point) or ``"es"`` (Spanish, decimal comma).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"``.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        from ..._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from ..._report.iso16251 import render_iso16251_report

        return render_iso16251_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def _rating_slice(frequencies: np.ndarray) -> np.ndarray | None:
    """Indices of the 16 rating bands 100-3150 Hz within ``frequencies``.

    Clause 6.3 measures over 18 bands 100-5000 Hz (optionally extended down
    to 50 Hz); the ISO 717-2 rating of clause 6.5 uses the 100-3150 Hz
    sub-range. Returns ``None`` when the spectrum does not contain all 16
    rating bands (nominal centres matched within 6 %).
    """
    if frequencies.ndim != 1:
        return None
    indices: list[int] = []
    for target in _RATING_FREQS:
        candidates = np.nonzero(np.abs(frequencies - target) <= 0.06 * target)[0]
        if candidates.size != 1:
            return None
        indices.append(int(candidates[0]))
    return np.asarray(indices, dtype=np.intp)


def _rating(
    improvement: np.ndarray, frequencies: np.ndarray
) -> tuple[int | None, int | None]:
    """``(ΔLw, CI,Δ)`` over the 100-3150 Hz sub-range, or ``(None, None)``."""
    idx = _rating_slice(frequencies)
    if idx is None:
        return None, None
    rated = improvement[idx]
    return (
        weighted_impact_improvement(rated),
        impact_improvement_adaptation_term(rated),
    )


def impact_improvement(
    bare: float | Sequence[float] | np.ndarray,
    with_covering: float | Sequence[float] | np.ndarray,
    frequencies: float | Sequence[float] | np.ndarray,
    *,
    background: float | Sequence[float] | np.ndarray | None = None,
) -> FloorCoveringImprovementResult:
    r"""Improvement of impact sound insulation ``ΔL`` of a floor covering.

    ISO 16251-1. The acceleration levels without (``L0``) and with (``L1``)
    the specimen may be
    given either as one value per band (already averaged over the
    tapping-machine
    and accelerometer positions) or, for the raw measurement, as a
    ``(positions, bands)`` array. The standard's order of operations is
    followed:
    the background correction (Formula (2)) is applied **per position**, then
    the
    level difference :math:`\Delta L_{t,a} = L_0 - L_1` (Formula (3)), then
    the arithmetic mean
    over positions :math:`\Delta L = \mathrm{mean}(\Delta L_{t,a})`
    (Formula (4)), necessary because Formula
    (2) is non-linear, so correcting must precede averaging.

    :param bare: Bare-plate acceleration level ``L0``, in dB; ``(bands,)`` or
        ``(positions, bands)``.
    :param with_covering: Level with the specimen ``L1``, in dB; same shape as ``bare``.
    :param frequencies: One-third-octave band centre frequencies, in Hz (``(bands,)``).
    :param background: Optional background-noise level ``Lb``, in dB; ``(bands,)``
        (applied to every position) or ``(positions, bands)``. When given, ``L0``
        and ``L1`` are background-corrected (Formula (2)) per position before
        the difference, and bands where any position hits the 1.3 dB limit
        are flagged.
    :return: :class:`FloorCoveringImprovementResult` (``ΔL`` per band).
    :raises ValueError: Mismatched shapes or non-finite values.
    """
    l0 = _finite(bare, "bare")
    l1 = _finite(with_covering, "with_covering")
    freqs = _finite(frequencies, "frequencies")
    n_bands = freqs.shape[0]
    if l0.shape != l1.shape:
        raise ValueError("'bare' and 'with_covering' must share their shape.")
    if l0.ndim not in (1, 2) or l0.shape[-1] != n_bands:
        raise ValueError(
            "'bare'/'with_covering' must be (bands,) or (positions, bands) "
            "matching 'frequencies'."
        )

    limited_pos = np.zeros(l0.shape, dtype=bool)
    if background is not None:
        lb = _finite(background, "background")
        if lb.shape not in ((n_bands,), l0.shape):
            raise ValueError(
                "'background' must be (bands,) or match the (positions, bands) "
                "shape of 'bare'/'with_covering'."
            )
        lb = np.broadcast_to(lb, l0.shape)
        # Formula (2) per position, then Formula (3), then Formula (4).
        l0, lim0 = background_corrected_level(l0, lb)
        l1, lim1 = background_corrected_level(l1, lb)
        limited_pos = lim0 | lim1

    delta_per_position = l0 - l1  # Formula (3)
    if delta_per_position.ndim == 2:
        improvement = delta_per_position.mean(axis=0)  # Formula (4)
        limited = np.any(limited_pos, axis=0)
    else:
        improvement = delta_per_position
        limited = limited_pos
    delta_lw, ci_delta = _rating(improvement, freqs)
    return FloorCoveringImprovementResult(
        frequencies=freqs,
        improvement=improvement,
        limited=limited,
        delta_lw=delta_lw,
        ci_delta=ci_delta,
    )


def improvement_octave_bands(
    improvement: Sequence[float] | np.ndarray,
    frequencies: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Convert a one-third-octave improvement spectrum to octaves (F. (5)).

    :math:`\Delta L_\mathrm{oct} = -10 \log_{10}[(1/3) \sum_{j=1}^{3} 10^{-\Delta L_j/10}]`
    dB over the three thirds of
    each octave.

    :param improvement: Improvement ``ΔL`` per one-third-octave band, in dB.
    :param frequencies: The matching one-third-octave centre frequencies, in Hz;
        must contain whole octave triplets (centre / centre·2^(±1/3)).
    :return: ``(octave_freqs, ΔLoct)``.
    :raises ValueError: The thirds do not group into complete octave triplets.
    """
    dl = _finite(improvement, "improvement")
    freqs = _finite(frequencies, "frequencies")
    if dl.shape != freqs.shape:
        raise ValueError("'improvement' and 'frequencies' must share their shape.")
    octave_freqs: list[float] = []
    octave_values: list[float] = []
    by_freq = {round(float(f), 3): float(d) for f, d in zip(freqs, dl)}
    for centre in _octave_centres(freqs):
        thirds = [centre * 2.0 ** (k / 3.0) for k in (-1, 0, 1)]
        try:
            vals = [by_freq[round(t, 3)] for t in _nearest(thirds, by_freq)]
        except KeyError:
            raise ValueError(
                f"Octave {centre:g} Hz is missing one of its one-third-octave "
                "bands; provide complete triplets."
            ) from None
        octave_freqs.append(centre)
        octave_values.append(
            -10.0 * np.log10(np.mean([10.0 ** (-v / 10.0) for v in vals]))
        )
    return np.asarray(octave_freqs), np.asarray(octave_values)


def _octave_centres(freqs: np.ndarray) -> list[float]:
    """The octave centres (…125, 250, 500…) present among one-third-octave freqs."""
    octaves = [
        31.5,
        63.0,
        125.0,
        250.0,
        500.0,
        1000.0,
        2000.0,
        4000.0,
        8000.0,
    ]
    lo, hi = float(np.min(freqs)), float(np.max(freqs))
    return [c for c in octaves if lo <= c <= hi]


def _nearest(targets: list[float], available: dict[float, float]) -> list[float]:
    """Snap each nominal third-octave frequency to the nearest available key."""
    keys = np.asarray(sorted(available))
    out = []
    for t in targets:
        idx = int(np.argmin(np.abs(keys - t)))
        if abs(keys[idx] - t) > t * 0.06:  # within ~6 % of the nominal ratio
            raise KeyError(t)
        out.append(float(keys[idx]))
    return out
