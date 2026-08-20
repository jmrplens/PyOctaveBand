#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Laboratory sound insulation of building elements (ISO 10140).

This is the **laboratory** counterpart of the field ISO 16283 family in
:mod:`phonometry.building.measurement.insulation`. In a qualified test facility flanking
transmission is suppressed, so the *direct* airborne sound reduction index
``R`` (not the apparent ``R'``) is the primary quantity, and the receiving
room's equivalent absorption area ``A`` is a property of the known facility.

**Airborne sound reduction index (ISO 10140-2:2010).** From the
energy-average sound pressure levels in the source room ``L1`` and receiving
room ``L2`` this module forms, per one-third-octave band,
:math:`R = L_1 - L_2 + 10 \log_{10}(S/A)` (Clause 3.1, Formula (2)) with the free
test
opening area ``S`` and the Sabine equivalent absorption area
:math:`A = 0.16\,V/T` (ISO 10140-4:2010, Clause 4.6.3, Formula (5)). The
single-number weighted rating ``Rw`` and the adaptation terms ``C`` / ``Ctr``
follow ISO 717-1 (Clause 5.3) through the verified
:func:`phonometry.building.weighted_rating` engine, reused unchanged.

**Impact sound pressure level (ISO 10140-3:2010).** With the standard
tapping machine exciting the floor under test this module forms, from the
energy-average impact sound pressure level ``Li`` in the receiving room, the
normalized impact sound pressure level :math:`L_\mathrm{n} = L_\mathrm{i} + 10 \log_{10}(A/A_0)`
(Clause 3.2,
Formula (1)) with :math:`A = 0.16\,V/T` and the reference absorption area
:math:`A_0 = 10` m². The single-number weighted rating ``Ln,w`` and the term
``CI`` follow ISO 717-2 (Clause 5.3) through
:func:`phonometry.building.weighted_impact_rating`, reused unchanged.

**Background-noise correction (ISO 10140-4:2010, Clause 4.3, Formula (4)).**
The receiving-room levels must be corrected for background noise before the
insulation is formed. :func:`background_correction` implements the correction
:math:`L = 10 \log_{10}(10^{L_\mathrm{sb}/10} - 10^{L_\mathrm{b}/10})` for a signal-to-background
margin
between 6 dB and 15 dB, the fixed 1.3 dB correction (limit of measurement)
for a margin of 6 dB or less, and no correction for a margin of 15 dB or
more. The 6/15 dB criteria are the laboratory analogue of the 6/10 dB
criteria of ISO 16283-1 Clause 9.2; both cap the correction at 1.3 dB.

**Frequency range (ISO 10140-4:2010, Clause 4.1).** Quantities are measured
over the mandatory one-third-octave range 100 Hz to 5000 Hz (optionally down
to 50 Hz). The single-number rating uses the core 100 Hz to 3150 Hz (16
one-third-octave bands) / 125 Hz to 2000 Hz (5 octave bands) range of
ISO 717-1/2, so the automatic rating is formed only when exactly 16 or 5
per-band values are supplied.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._internal.warnings import PhonometryWarning
from .insulation import (
    ImpactRatingResult,
    WeightedRatingResult,
    _as_band_levels,
    weighted_impact_rating,
    weighted_rating,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..._report.metadata import ReportMetadata

#: Reference equivalent absorption area A0 for the normalized impact level
#: (ISO 10140-3:2010, Clause 3.2) and the element-normalized level difference
#: (ISO 10140-2:2010, Clause 3.3): 10 m² for the laboratory.
_A0_LAB = 10.0

#: Sabine constant in ``A = 0,16 V / T`` (ISO 10140-4:2010, Formula (5)).
_SABINE = 0.16

#: Fixed correction (dB) applied when the signal-to-background margin is at
#: most 6 dB, the limit of measurement (ISO 10140-4:2010, Clause 4.3).
_BACKGROUND_CAP = 1.3

#: Signal-to-background margins (dB) bounding Formula (4) of ISO 10140-4:2010
#: Clause 4.3: at or below the lower margin the fixed 1,3 dB cap applies; at
#: or above the upper margin no correction is applied.
_BACKGROUND_LOW = 6.0
_BACKGROUND_HIGH = 15.0


class LabInsulationWarning(PhonometryWarning):
    """Warning for laboratory-insulation limit-of-measurement conditions."""


@dataclass(frozen=True)
class LabAirborneInsulationResult:
    r"""Per-band laboratory airborne sound insulation (ISO 10140-2:2010).

    :ivar r: Sound reduction index :math:`R = L_1 - L_2 + 10 \log_{10}(S/A)` per
        band, in dB (Clause 3.1, Formula (2)).
    :ivar absorption: Equivalent sound absorption area :math:`A = 0.16\,V/T`
        per band, in m² (ISO 10140-4:2010, Formula (5)).
    :ivar rating: Single-number weighted rating ``Rw`` with ``C`` / ``Ctr``
        (ISO 717-1), or ``None`` when the number of bands is neither 16
        (one-third octave) nor 5 (octave) and no rating can be formed.
    """

    r: np.ndarray
    absorption: np.ndarray
    rating: WeightedRatingResult | None

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot ``R`` against the shifted ISO 717-1 reference curve.

        Delegates to the weighted-rating plot (measured ``R`` versus the
        shifted reference, unfavourable deviations shaded). Requires the
        automatic rating to be available (16 or 5 bands) and matplotlib
        (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        if self.rating is None:
            raise ValueError(
                "No single-number rating is available to plot (need 16 "
                "one-third-octave or 5 octave bands)."
            )
        return self.rating.plot(ax=ax, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an ISO 10140-2 laboratory airborne-insulation report to a PDF.

        Writes the one-page laboratory test report of ISO 10140-2:2010: the
        standard-basis line, an optional metadata header block (client,
        specimen, mounting, room volumes, climatic conditions ...), the band
        table (16 one-third-octave or 5 octave bands) beside the
        measured-versus-shifted-reference curve, the boxed laboratory rating
        ``Rw (C; Ctr)`` (evaluated per ISO 717-1), the laboratory-method
        statement, an optional verdict row and a footer with the identity
        block and disclaimer. The report requires the single-number ``rating``
        to be present on the result; it is formed automatically only for
        exactly 16 one-third-octave (100 Hz to 3150 Hz) or 5 octave (125 Hz to
        2000 Hz) bands, and a result carrying no rating (any other band count)
        is rejected.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`;
            ``None`` produces a lightweight fiche (body, rating, statement and
            disclaimer only).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True``, the table annexes the per-band
            equivalent sound absorption area ``A`` beside the reported ``R``.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is unknown, ``language`` is not one
            of the supported values, or the result carries no single-number
            rating (its band count is neither 16 one-third-octave nor 5
            octave, so the ISO 717-1 rating the fiche needs was not formed).
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded rating figure (``pip install phonometry[plot]``).
        """
        return _render_iso10140(
            self,
            path,
            quantity="r",
            metadata=metadata,
            engine=engine,
            verbose=verbose,
            language=language,
        )


@dataclass(frozen=True)
class LabImpactInsulationResult:
    r"""Per-band laboratory impact sound insulation (ISO 10140-3:2010).

    :ivar l_n: Normalized impact sound pressure level
        :math:`L_\mathrm{n} = L_\mathrm{i} + 10 \log_{10}(A/A_0)` per band, in dB (Clause 3.2,
        Formula (1)).
    :ivar absorption: Equivalent sound absorption area :math:`A = 0.16\,V/T`
        per band, in m² (ISO 10140-4:2010, Formula (5)).
    :ivar rating: Single-number weighted rating ``Ln,w`` with ``CI``
        (ISO 717-2), or ``None`` when the number of bands is neither 16
        (one-third octave) nor 5 (octave) and no rating can be formed.
    """

    l_n: np.ndarray
    absorption: np.ndarray
    rating: ImpactRatingResult | None

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot ``Ln`` against the shifted ISO 717-2 reference curve.

        Delegates to the weighted impact-rating plot. Requires the automatic
        rating to be available (16 or 5 bands) and matplotlib
        (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        if self.rating is None:
            raise ValueError(
                "No single-number rating is available to plot (need 16 "
                "one-third-octave or 5 octave bands)."
            )
        return self.rating.plot(ax=ax, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an ISO 10140-3 laboratory impact-insulation report to a PDF.

        Writes the one-page laboratory test report of ISO 10140-3:2010: the
        standard-basis line, an optional metadata header block, the band table
        (16 one-third-octave or 5 octave bands) beside the
        measured-versus-shifted-reference curve, the boxed laboratory rating
        ``Ln,w (CI)`` (evaluated per ISO 717-2), the laboratory-method
        statement, an optional verdict row and a footer with the identity
        block and disclaimer. The report requires the single-number ``rating``
        to be present on the result; it is formed automatically only for
        exactly 16 one-third-octave (100 Hz to 3150 Hz) or 5 octave (125 Hz to
        2000 Hz) bands, and a result carrying no rating (any other band count)
        is rejected.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`;
            ``None`` produces a lightweight fiche (body, rating, statement and
            disclaimer only).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True``, the table annexes the per-band
            equivalent sound absorption area ``A`` beside the reported ``Ln``.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is unknown, ``language`` is not one
            of the supported values, or the result carries no single-number
            rating (its band count is neither 16 one-third-octave nor 5
            octave, so the ISO 717-2 rating the fiche needs was not formed).
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded rating figure (``pip install phonometry[plot]``).
        """
        return _render_iso10140(
            self,
            path,
            quantity="l_n",
            metadata=metadata,
            engine=engine,
            verbose=verbose,
            language=language,
        )


def _render_iso10140(
    result: LabAirborneInsulationResult | LabImpactInsulationResult,
    path: str,
    *,
    quantity: str,
    metadata: ReportMetadata | None,
    engine: str,
    verbose: bool,
    language: str,
) -> str:
    """Validate the laboratory-report request and delegate to the renderer.

    Shared by :meth:`LabAirborneInsulationResult.report` and
    :meth:`LabImpactInsulationResult.report`: it rejects unknown engines,
    requires the single-number rating (16 one-third-octave or 5 octave bands,
    ISO 717-1/-2) so the fiche curve is defined, and calls the reportlab
    renderer (which raises a clear :class:`ImportError` when reportlab is
    absent).
    """
    from ..._i18n import check_language

    check_language(language)
    if engine != "reportlab":
        raise ValueError(
            f"Unknown report engine {engine!r}; only 'reportlab' is supported."
        )
    rating = result.rating
    if rating is None:
        raise ValueError(
            "The laboratory report needs the ISO 717 single-number rating; "
            "it is formed only for exactly 16 one-third-octave (100 Hz to "
            "3150 Hz) or 5 octave (125 Hz to 2000 Hz) bands. Build the result "
            "with that band count."
        )

    from ..._report.iso10140 import render_iso10140_report

    return render_iso10140_report(
        result,
        rating,
        path,
        quantity=quantity,
        metadata=metadata,
        verbose=verbose,
        language=language,
    )


def _absorption_area(
    t2: Sequence[float] | np.ndarray, volume: float, n_bands: int
) -> np.ndarray:
    r"""Sabine equivalent absorption area :math:`A = 0.16\,V/T` per band.

    (ISO 10140-4:2010, Clause 4.6.3, Formula (5).)
    """
    t = np.asarray(t2, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError("'t2' must be one-dimensional (one value per band).")
    if t.size != n_bands:
        raise ValueError("'t2' must share the band count of the level input.")
    if not np.all(np.isfinite(t)) or np.any(t <= 0.0):
        raise ValueError("'t2' must contain positive, finite values.")
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("'volume' must be positive.")
    return _SABINE * volume / t


def background_correction(
    signal_and_background: Sequence[float] | np.ndarray,
    background: Sequence[float] | np.ndarray,
) -> np.ndarray:
    r"""Correct receiving-room levels for background noise (ISO 10140-4:2010).

    Applies the correction of Clause 4.3 per band from the combined
    signal-plus-background level ``Lsb`` and the background level ``Lb``,
    using the margin ``Lsb - Lb``:

    - :math:`\mathrm{margin} \ge 15` dB: the background is negligible and
      the level is
      returned unchanged (Clause 4.3, quality requirement).
    - :math:`6\,\mathrm{dB} < \mathrm{margin} < 15\,\mathrm{dB}`: the level
      is corrected with Formula (4),
      :math:`L = 10 \log_{10}(10^{L_\mathrm{sb}/10} - 10^{L_\mathrm{b}/10})`.
    - :math:`\mathrm{margin} \le 6` dB: the fixed 1.3 dB correction is
      applied
      (:math:`L = L_\mathrm{sb} - 1.3`); such bands are the *limit of measurement*
      and a
      :class:`LabInsulationWarning` is emitted (Clause 4.3). A *negative*
      margin (:math:`L_\mathrm{b} > L_\mathrm{sb}`, i.e. background above the measured
      signal) falls
      in this branch and is likewise capped at :math:`L_\mathrm{sb} - 1.3`: the
      band is
      simply flagged as the limit of measurement rather than yielding a
      nonsensical (or ``NaN``) corrected level.

    This is the sound-insulation counterpart of
    :func:`phonometry.emission.background_noise_correction` (ISO 3744:2010):
    both apply the same energy subtraction
    :math:`10 \log_{10}(10^{L_\mathrm{sb}/10} - 10^{L_\mathrm{b}/10})`, but
    that routine returns the correction *offset* ``K1`` (to subtract from
    ``Lsb``), whereas this one returns the already-corrected levels ``L``
    directly.

    :param signal_and_background: Combined signal-plus-background levels
        ``Lsb`` per band, in dB.
    :param background: Background-noise levels ``Lb`` per band, in dB.
    :return: The background-corrected levels per band, in dB.
    :raises ValueError: If the shapes differ or contain non-finite values.
    """
    lsb = np.asarray(signal_and_background, dtype=np.float64)
    lb = np.asarray(background, dtype=np.float64)
    if lsb.shape != lb.shape:
        raise ValueError(
            "'signal_and_background' and 'background' must share their shape."
        )
    if not (np.all(np.isfinite(lsb)) and np.all(np.isfinite(lb))):
        raise ValueError("Levels must contain only finite values.")

    margin = lsb - lb
    # Formula (4) for the 6 < margin < 15 band; unchanged at margin >= 15.
    # The argument of the logarithm stays positive because Formula (4) is only
    # selected where margin > 6 dB (10^(Lsb/10) > 10^(Lb/10)).
    diff = 10.0 ** (lsb / 10.0) - 10.0 ** (lb / 10.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        formula = 10.0 * np.log10(np.where(diff > 0.0, diff, 1.0))
    corrected = np.where(margin >= _BACKGROUND_HIGH, lsb, formula)
    limited = margin <= _BACKGROUND_LOW
    corrected = np.where(limited, lsb - _BACKGROUND_CAP, corrected)
    if bool(np.any(limited)):
        warnings.warn(
            "Signal-to-background margin at or below 6 dB in one or more "
            "bands; the fixed 1,3 dB correction was applied and those levels "
            "are the limit of measurement (ISO 10140-4:2010, Clause 4.3).",
            LabInsulationWarning,
            stacklevel=2,
        )
    return np.asarray(corrected, dtype=np.float64)


def lab_airborne_insulation(
    l1: Sequence[float] | np.ndarray,
    l2: Sequence[float] | np.ndarray,
    t2: Sequence[float] | np.ndarray,
    *,
    area: float,
    volume: float,
) -> LabAirborneInsulationResult:
    r"""
    Laboratory airborne sound reduction index per ISO 10140-2:2010.

    Computes, per frequency band, the sound reduction index
    :math:`R = L_1 - L_2 + 10 \log_{10}(S/A)` (Clause 3.1, Formula (2)) with the
    free test
    opening area ``S`` and the Sabine equivalent absorption area
    :math:`A = 0.16\,V/T` (ISO 10140-4:2010, Formula (5)). When exactly 16
    one-third-octave (100-3150 Hz) or 5 octave (125-2000 Hz) values are
    supplied, the single-number weighted rating ``Rw`` with ``C`` / ``Ctr``
    is also formed via :func:`phonometry.building.weighted_rating` (ISO 717-1).

    ``l1`` and ``l2`` may be one value per band (already energy-averaged) or a
    two-dimensional ``(positions, bands)`` array, in which case the positions
    are energy-averaged (ISO 10140-4:2010, Formula (2)). The band levels are
    assumed already corrected for background noise (see
    :func:`background_correction`).

    :param l1: Source-room sound pressure levels, in dB.
    :param l2: Receiving-room sound pressure levels, in dB.
    :param t2: Receiving-room reverberation time per band, in seconds.
    :param area: Area ``S`` of the free test opening, in m².
    :param volume: Receiving-room volume ``V``, in m³.
    :return: :class:`LabAirborneInsulationResult` with ``r``, ``absorption``
        and ``rating``.
    :raises ValueError: If the band counts of ``l1``, ``l2`` and ``t2``
        differ, if ``area``/``volume``/``t2`` are not positive, or if inputs
        are non-finite.
    """
    l1_bands = _as_band_levels(l1, "l1")
    l2_bands = _as_band_levels(l2, "l2")
    if l1_bands.shape != l2_bands.shape:
        raise ValueError("'l1' and 'l2' must share the same band count.")
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("'area' must be positive.")

    absorption = _absorption_area(t2, volume, int(l1_bands.size))
    r = l1_bands - l2_bands + 10.0 * np.log10(area / absorption)

    rating: WeightedRatingResult | None = None
    if r.size in (16, 5):
        rating = weighted_rating(r)
    return LabAirborneInsulationResult(r=r, absorption=absorption, rating=rating)


def lab_impact_insulation(
    li: Sequence[float] | np.ndarray,
    t2: Sequence[float] | np.ndarray,
    *,
    volume: float,
) -> LabImpactInsulationResult:
    r"""
    Laboratory impact sound pressure level per ISO 10140-3:2010.

    Computes, per frequency band, the normalized impact sound pressure level
    :math:`L_\mathrm{n} = L_\mathrm{i} + 10 \log_{10}(A/A_0)` (Clause 3.2, Formula (1)) with the
    Sabine
    equivalent absorption area :math:`A = 0.16\,V/T` (ISO 10140-4:2010,
    Formula (5)) and the reference absorption area :math:`A_0 = 10` m².
    When exactly
    16 one-third-octave (100-3150 Hz) or 5 octave (125-2000 Hz) values are
    supplied, the single-number weighted rating ``Ln,w`` with ``CI`` is also
    formed via :func:`phonometry.building.weighted_impact_rating` (ISO 717-2).

    ``li`` may be one value per band (already energy-averaged) or a
    two-dimensional ``(positions, bands)`` array, in which case the positions
    are energy-averaged (ISO 10140-4:2010, Formula (2)). The band levels are
    assumed already corrected for background noise (see
    :func:`background_correction`).

    :param li: Energy-average impact sound pressure levels, in dB.
    :param t2: Receiving-room reverberation time per band, in seconds.
    :param volume: Receiving-room volume ``V``, in m³.
    :return: :class:`LabImpactInsulationResult` with ``l_n``, ``absorption``
        and ``rating``.
    :raises ValueError: If the band counts of ``li`` and ``t2`` differ, if
        ``volume``/``t2`` are not positive, or if inputs are non-finite.
    """
    li_bands = _as_band_levels(li, "li")
    absorption = _absorption_area(t2, volume, int(li_bands.size))
    l_n = li_bands + 10.0 * np.log10(absorption / _A0_LAB)

    rating: ImpactRatingResult | None = None
    if l_n.size in (16, 5):
        rating = weighted_impact_rating(l_n)
    return LabImpactInsulationResult(l_n=l_n, absorption=absorption, rating=rating)
