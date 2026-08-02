#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Field survey method for sound insulation and service-equipment noise
(ISO 10052:2021).

This is the **survey (control) method**: a fast, octave-band field procedure
for dwellings and rooms of comparable size (up to 150 m³). It trades the
resolution of the ISO 16283 engineering method (:mod:`phonometry.building.measurement.insulation`)
for speed: a single hand-held integrating sound level meter swept through the
room. It measures airborne and impact sound insulation between rooms, façade
sound insulation, and the sound pressure level from building service equipment.
Clause references and the reverberation-index table follow the 2021 edition;
the formulas and that table are identical in the harmonized
EN ISO 10052:2004+A1:2010.

**Reverberation index (Clause 3.3).** The correction for the receiving room is
carried by a single quantity, the reverberation index
:math:`k = 10 \log_{10}(T/T_0)` dB
with the reference reverberation time :math:`T_0 = 0.5` s. It may be
**measured** (pass
the reverberation time ``T`` per band to :func:`reverberation_index`) or,
in a control survey, **estimated** from the room type and volume with
:func:`estimate_reverberation_index` (Clause 6.5, Tables 3 and 4).

**Airborne between rooms (Clauses 3.2-3.6).** From the source- and
receiving-room levels ``L1`` and ``L2`` the level difference
:math:`D = L_1 - L_2`
(Clause 3.2) gives the standardized level difference :math:`D_{nT} = D + k`
(Clause 3.4), the normalized level difference
:math:`D_n = D + k + 10 \log_{10}(A_0 T_0 / (0.16\,V))` (Clause 3.5,
:math:`A_0 = 10` m²) and, when a common
partition area ``S`` is given, the apparent sound reduction index
:math:`R' = D + k + 10 \log_{10}(S T_0 / (0.16\,V))` (Clause 3.6). Where
:math:`V/7.5 > S` the
value :math:`V/7.5` is used for ``S``, with ``V`` the smaller room.

**Impact (Clauses 3.7-3.9).** From the impact level ``Li`` (Clause 3.7,
energy-averaged over the tapping-machine positions; the 2021 edition also
admits the heavy/soft impact source of Clause 3.10 with the maximum level
``Li,Fmax`` of Clause 3.11) the standardized impact level
:math:`L'_{nT} = L_i - k`
(Clause 3.8) and the normalized impact level
:math:`L'_n = L_i - k - 10 \log_{10}(A_0 T_0 / (0.16\,V))` (Clause 3.9).

**Façade (Clauses 3.13-3.15).** From the outdoor level 2 m in front of the
façade ``L1,2m`` and the receiving-room level ``L2`` the façade level
difference :math:`D_{2m} = L_{1,2m} - L_2` (Clause 3.13), the standardized
:math:`D_{2m,nT} = D_{2m} + k` (Clause 3.14) and the normalized
:math:`D_{2m,n} = D_{2m} + k + 10 \log_{10}(A_0 T_0 / (0.16\,V))` (Clause 3.15).

**Service equipment (Clauses 3.16-3.18).** From three A- or C-weighted sound
pressure levels (one near a room corner, two in the reverberant field) the
service-equipment level
:math:`L_{XY} = 10 \log_{10}[(1/3) \sum 10^{0.1 L_{XY,i}}]`
(Clause 3.16), its standardized form :math:`L_{XY,nT} = L_{XY} - k`
(Clause 3.17) and
normalized form :math:`L_{XY,n} = L_{XY} - k - 10 \log_{10}(A_0 T_0 / (0.16\,V))`
(Clause 3.18).

**Frequency range (Clause 6.4).** Airborne and tapping-machine impact
quantities are measured in octave bands 125 Hz to 2000 Hz (5 bands); the
heavy/soft impact source uses 63 Hz to 500 Hz. The single-number weighted
ratings reuse the verified :func:`phonometry.weighted_rating` (ISO 717-1) and
:func:`phonometry.weighted_impact_rating` (ISO 717-2) engines, formed only when
exactly 5 octave (or 16 one-third-octave) values are supplied. No background
correction is applied (Clause 6.2).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .insulation import (
    ImpactRatingResult,
    WeightedRatingResult,
    _as_band_levels,
    energy_average_level,
    weighted_impact_rating,
    weighted_rating,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..._report.metadata import ReportMetadata


def _validate_report_request(engine: str, language: str) -> None:
    """Reject an unknown engine or language before a survey fiche is rendered.

    Shared by the survey result ``report`` methods: only the ``"reportlab"``
    engine is supported, and the language must be one the renderers translate.
    """
    from ..._i18n import check_language

    check_language(language)
    if engine != "reportlab":
        raise ValueError(
            f"Unknown report engine {engine!r}; only 'reportlab' is supported."
        )

#: Reference reverberation time T0 (ISO 10052:2021, Clause 3.3): 0,5 s.
_T0 = 0.5

#: Reference equivalent absorption area A0 (Clause 3.5): 10 m².
_A0 = 10.0

#: Sabine constant in A = 0,16 V / T.
_SABINE = 0.16

#: Minimum-area factor of Clause 3.6: where V/7,5 > S, use V/7,5 for S.
_MIN_AREA_FACTOR = 7.5

#: Upper volume limits of the four Table 4 ranges (Clause 6.5); the table is
#: valid for rooms up to 150 m³.
_TABLE3_VOLUME_LIMITS = (15.0, 35.0, 60.0, 150.0)

#: Reverberation index data transcribed digit-for-digit from ISO 10052:2021,
#: Table 4 (identical to Table 3 of EN ISO 10052:2004+A1:2010). Keyed by the
#: volume-range index (0: V < 15, 1: 15 <= V < 35, 2: 35 <= V < 60,
#: 3: 60 <= V <= 150) then the room type; each value is the six-tuple
#: (125, 250, 500, 1000, 2000 Hz, A/C-weighted) in dB. Room types: the
#: furnished categories ``"kitchen"`` / ``"bathroom"`` (only tabulated for
#: V < 35) and ``"furnished"`` (the general furnished room), and the
#: unfurnished types ``"a"``-``"h"`` and the mixed ``"a+e"`` / ``"b+f"`` /
#: ``"c+g"`` / ``"d+h"`` (Table 2 classifies a room into these letters).
_TABLE3: dict[int, dict[str, tuple[float, float, float, float, float, float]]] = {
    0: {  # V < 15
        "kitchen": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "bathroom": (1.0, 1.0, 0.0, 0.0, -0.5, 0.0),
        "furnished": (0.0, 0.0, -0.5, -0.5, -1.0, -0.5),
        "a": (0.0, 1.0, 1.0, 1.0, 0.0, 0.5),
        "b": (1.0, 2.5, 3.0, 2.5, 2.0, 2.0),
        "c": (0.0, 2.5, 3.5, 4.0, 4.0, 4.0),
        "d": (0.0, 2.5, 3.0, 4.0, 4.0, 4.0),
        "e": (3.5, 3.5, 3.5, 3.5, 1.5, 3.5),
        "f": (4.5, 4.5, 4.5, 3.5, 2.5, 3.5),
        "g": (3.5, 4.0, 4.5, 5.0, 5.0, 5.0),
        "h": (4.0, 4.5, 5.0, 5.0, 4.5, 5.0),
        "a+e": (2.0, 2.5, 2.5, 2.5, 1.0, 2.0),
        "b+f": (3.0, 3.5, 4.0, 3.0, 2.5, 3.0),
        "c+g": (2.0, 3.5, 4.0, 4.5, 4.5, 4.5),
        "d+h": (2.0, 3.5, 4.0, 4.5, 4.5, 4.5),
    },
    1: {  # 15 <= V < 35
        "kitchen": (0.0, 0.5, 0.0, 0.0, 0.0, 0.0),
        "bathroom": (1.5, 1.5, 0.5, 0.5, 0.0, 0.5),
        "furnished": (0.0, 0.0, 0.0, 0.0, -0.5, 0.0),
        "a": (1.0, 1.5, 1.5, 1.0, 0.5, 1.0),
        "b": (1.0, 3.0, 3.5, 3.0, 2.5, 2.5),
        "c": (1.0, 3.0, 4.0, 4.5, 4.0, 4.5),
        "d": (1.0, 3.0, 3.5, 4.5, 4.0, 4.5),
        "e": (3.5, 4.0, 4.0, 4.0, 2.0, 4.0),
        "f": (4.5, 4.5, 4.5, 4.0, 3.0, 4.0),
        "g": (4.0, 5.0, 5.0, 5.0, 5.0, 5.5),
        "h": (4.5, 5.0, 5.5, 5.5, 5.0, 5.0),
        "a+e": (2.5, 3.0, 3.0, 2.5, 1.5, 2.5),
        "b+f": (3.0, 4.0, 4.0, 3.5, 3.0, 3.5),
        "c+g": (2.5, 4.0, 4.5, 5.0, 4.5, 5.0),
        "d+h": (3.0, 4.0, 4.5, 5.0, 4.5, 5.0),
    },
    2: {  # 35 <= V < 60
        "furnished": (0.5, 0.5, 0.5, 0.0, 0.0, 0.0),
        "a": (1.0, 2.0, 2.0, 1.5, 1.0, 1.5),
        "b": (2.0, 3.5, 4.0, 3.5, 2.5, 3.0),
        "c": (1.5, 3.5, 4.5, 5.0, 4.5, 5.0),
        "d": (1.5, 3.5, 4.0, 5.0, 5.0, 5.0),
        "e": (4.0, 4.0, 4.5, 4.0, 2.5, 4.0),
        "f": (4.5, 4.5, 4.5, 4.0, 3.0, 5.0),
        "g": (4.5, 5.0, 5.5, 5.5, 5.5, 5.5),
        "h": (5.0, 5.5, 6.0, 5.0, 5.5, 5.5),
        "a+e": (2.5, 3.0, 3.5, 3.0, 2.0, 3.0),
        "b+f": (3.5, 4.0, 4.5, 4.0, 3.0, 4.0),
        "c+g": (3.0, 4.5, 5.0, 5.5, 5.0, 5.5),
        "d+h": (3.5, 4.5, 5.0, 5.0, 5.5, 5.5),
    },
    3: {  # 60 <= V <= 150
        "furnished": (0.5, 0.5, 0.5, 0.5, 0.0, 0.5),
        "a": (1.0, 2.5, 2.5, 2.0, 1.5, 2.0),
        "b": (2.5, 4.0, 4.5, 3.5, 2.5, 3.5),
        "c": (2.0, 4.0, 5.0, 5.5, 5.0, 5.5),
        "d": (2.0, 4.0, 4.5, 5.5, 5.5, 5.5),
        "e": (4.0, 4.0, 5.0, 4.5, 3.0, 4.5),
        "f": (4.5, 5.0, 5.0, 4.0, 3.0, 5.0),
        "g": (5.0, 5.5, 6.0, 6.0, 6.0, 6.0),
        "h": (5.5, 6.0, 6.5, 5.5, 6.0, 6.0),
        "a+e": (2.5, 3.5, 4.0, 3.5, 2.5, 3.5),
        "b+f": (3.5, 4.5, 5.0, 4.0, 3.0, 4.5),
        "c+g": (3.5, 5.0, 5.5, 6.0, 5.5, 6.0),
        "d+h": (4.0, 5.0, 5.5, 5.5, 6.0, 6.0),
    },
}


def _finite_bands(
    values: float | Sequence[float] | np.ndarray, name: str
) -> np.ndarray:
    """Return ``values`` as a finite float array, or raise."""
    a = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"'{name}' must contain only finite values.")
    return a


def _positive(value: float, name: str) -> float:
    """Return ``value`` as a positive, finite float, or raise."""
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"'{name}' must be positive.")
    return v


def _normalization_term(volume: float) -> float:
    """The 10 lg(A0 T0 / (0,16 V)) normalization offset, in dB."""
    return float(
        10.0 * np.log10(_A0 * _T0 / (_SABINE * _positive(volume, "volume")))
    )


def _rate(values: np.ndarray) -> WeightedRatingResult | None:
    """ISO 717-1 rating when the band count is 5 (octave) or 16 (third)."""
    return weighted_rating(values) if values.size in (16, 5) else None


def _rate_impact(values: np.ndarray) -> ImpactRatingResult | None:
    """ISO 717-2 rating when the band count is 5 (octave) or 16 (third)."""
    return weighted_impact_rating(values) if values.size in (16, 5) else None


def reverberation_index(
    t: Sequence[float] | np.ndarray, *, t0: float = _T0
) -> np.ndarray:
    r"""Reverberation index :math:`k = 10 \log_{10}(T/T_0)` (ISO 10052, Cl. 3.3).

    :param t: Receiving-room reverberation time per band, in seconds.
    :param t0: Reference reverberation time ``T0``, in seconds (Default: 0,5 s).
    :return: The reverberation index ``k`` per band, in dB.
    :raises ValueError: If ``t``/``t0`` are not positive or ``t`` is non-finite.
    """
    tt = _finite_bands(t, "t")
    if np.any(tt <= 0.0):
        raise ValueError("'t' must contain positive values.")
    return 10.0 * np.log10(tt / _positive(t0, "t0"))


def estimate_reverberation_index(
    volume: float, room: str, *, weighted: bool = False
) -> np.ndarray | float:
    r"""Estimate the reverberation index from room type and volume (Clause 6.5).

    In a control survey the reverberation time need not be measured: the
    reverberation index ``k`` may be read from ISO 10052:2021 Table 4 (Table 3
    of EN ISO 10052:2004+A1:2010) by classifying the room. Furnished rooms use
    the ``room`` categories ``"kitchen"`` / ``"bathroom"`` (tabulated only for
    :math:`V < 35`) and ``"furnished"`` (a general furnished living/sleeping
    room);
    unfurnished rooms use the Table 3 (2004: Table 2) construction letters
    ``"a"``-``"h"`` and the area-averaged mixed classes ``"a+e"``, ``"b+f"``,
    ``"c+g"`` and ``"d+h"``. The table is valid for :math:`T_0 = 0.5` s and
    rooms up
    to 150 m³.

    :param volume: Receiving-room volume ``V``, in m³ (0 < V <= 150).
    :param room: Room-type key (see above).
    :param weighted: When ``True`` return the single A-/C-weighted index (the
        Table 4 ``A, C`` column) instead of the five octave-band values, for
        globally weighted service-equipment noise (Clause 3.17).
    :return: The reverberation index ``k`` per octave band (125-2000 Hz), in
        dB, or the scalar A-/C-weighted index when ``weighted`` is ``True``.
    :raises ValueError: If ``volume`` is not in ``(0, 150]``, or ``room`` is
        not tabulated for that volume range.
    """
    v = _positive(volume, "volume")
    if v > _TABLE3_VOLUME_LIMITS[-1]:
        raise ValueError(
            f"'volume' must be at most {_TABLE3_VOLUME_LIMITS[-1]:g} m³ "
            "(the Table 4 range)."
        )
    # V < 15, 15 <= V < 35, 35 <= V < 60, 60 <= V <= 150 (last is inclusive).
    idx = next(
        (i for i, hi in enumerate(_TABLE3_VOLUME_LIMITS) if v < hi),
        len(_TABLE3_VOLUME_LIMITS) - 1,
    )

    entries = _TABLE3[idx]
    room_norm = room.strip().lower()
    key = (
        "furnished"
        if room_norm in ("furnished", "other", "others")
        else room_norm
    )
    if key not in entries:
        available = ", ".join(sorted(entries))
        limits = _TABLE3_VOLUME_LIMITS
        if idx == 0:
            rng = f"V < {limits[0]:g}"
        elif idx == len(limits) - 1:
            rng = f"{limits[idx - 1]:g} <= V <= {limits[idx]:g}"
        else:
            rng = f"{limits[idx - 1]:g} <= V < {limits[idx]:g}"
        raise ValueError(
            f"'room' {room!r} is not tabulated for the volume range "
            f"{rng} (m³); available: {available}."
        )
    row = entries[key]
    return float(row[5]) if weighted else np.asarray(row[:5], dtype=np.float64)


@dataclass(frozen=True)
class SurveyAirborneResult:
    r"""Per-band airborne sound insulation, survey method (ISO 10052).

    :ivar d: Level difference :math:`D = L_1 - L_2` per band, dB (Clause 3.2).
    :ivar d_nt: Standardized level difference :math:`D_{nT} = D + k` (Cl. 3.4).
    :ivar d_n: Normalized level difference ``Dn`` (Clause 3.5), or ``None``
        when the receiving-room volume was not supplied.
    :ivar r_prime: Apparent sound reduction index ``R'`` (Clause 3.6), or
        ``None`` when the partition area and volume were not both supplied.
    :ivar rating: Weighted standardized level difference ``DnT,w`` with ``C`` /
        ``Ctr`` (ISO 717-1), or ``None`` off the 5/16-band count.
    :ivar r_prime_rating: Weighted apparent sound reduction index ``R'w``, or
        ``None`` when ``r_prime`` is unavailable or off the band count.
    """

    d: np.ndarray
    d_nt: np.ndarray
    d_n: np.ndarray | None
    r_prime: np.ndarray | None
    rating: WeightedRatingResult | None
    r_prime_rating: WeightedRatingResult | None

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot ``DnT`` against the shifted ISO 717-1 reference curve.

        Delegates to the weighted-rating plot. Requires the automatic rating
        (5 octave or 16 one-third-octave bands) and matplotlib
        (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        if self.rating is None:
            raise ValueError(
                "No single-number rating is available to plot (need 5 octave "
                "or 16 one-third-octave bands)."
            )
        return self.rating.plot(ax=ax, **kwargs)

    def report(
        self,
        path: str,
        *,
        quantity: str = "dnt",
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render a survey-method airborne field report to a PDF (ISO 10052).

        Writes the one-page survey (control) method field report: the
        standard-basis line naming ISO 10052 (octave bands), an optional
        metadata header block, the octave-band table beside the
        measured-versus-shifted-ISO 717-1-reference curve, the boxed field
        rating ``DnT,w (C; Ctr)`` or ``R'w (C; Ctr)``, the survey-method
        statement, an optional verdict row (airborne passes at or above the
        requirement) and a footer with the identity block and disclaimer.

        :param path: Destination path of the PDF file.
        :param quantity: The reported quantity: ``"dnt"`` (default, the
            standardized level difference ``DnT``) or ``"r_prime"`` (the
            apparent sound reduction index ``R'``; requires the result to
            carry ``r_prime``, built with the ``area`` and ``volume``
            arguments).
        :param metadata: Optional :class:`~phonometry.ReportMetadata`;
            ``None`` produces a lightweight fiche (body, rating and
            disclaimer only).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True``, the table shows the ISO 717 evaluation
            per band instead of the two-column ``f | value`` form.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` or ``quantity`` is unknown, or the
            selected quantity has no single-number rating (the survey needs
            5 octave or 16 one-third-octave bands, and ``R'`` needs ``area``
            and ``volume``).
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        _validate_report_request(engine, language)
        if quantity not in ("dnt", "r_prime"):
            raise ValueError(
                f"Unknown survey quantity {quantity!r}; expected 'dnt' or "
                "'r_prime'."
            )
        rating = self.rating if quantity == "dnt" else self.r_prime_rating
        if rating is None:
            raise ValueError(
                "The survey airborne report needs the ISO 717-1 single-number "
                "rating; supply 5 octave or 16 one-third-octave bands "
                "(and, for R', the 'area' and 'volume' arguments)."
            )
        from ..._report.iso10052 import render_survey_airborne_report

        return render_survey_airborne_report(
            self, rating, path, quantity=quantity, metadata=metadata,
            verbose=verbose, language=language,
        )


@dataclass(frozen=True)
class SurveyImpactResult:
    r"""Per-band impact sound insulation, survey method (ISO 10052).

    :ivar l_i: Energy-average impact sound pressure level ``Li`` per band, in
        dB (Clause 3.7).
    :ivar l_nt: Standardized impact level :math:`L'_{nT} = L_i - k` (Cl. 3.8).
    :ivar l_n: Normalized impact level ``L'n`` (Clause 3.9), or ``None``
        when the receiving-room volume was not supplied.
    :ivar rating: Weighted standardized impact level ``L'nT,w`` with ``CI``
        (ISO 717-2), or ``None`` off the 5/16-band count.
    """

    l_i: np.ndarray
    l_nt: np.ndarray
    l_n: np.ndarray | None
    rating: ImpactRatingResult | None

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot ``L'nT`` against the shifted ISO 717-2 reference curve."""
        if self.rating is None:
            raise ValueError(
                "No single-number rating is available to plot (need 5 octave "
                "or 16 one-third-octave bands)."
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
        """Render a survey-method impact field report to a PDF (ISO 10052).

        Writes the one-page survey (control) method field report: the
        standard-basis line naming ISO 10052 (octave bands, tapping machine),
        an optional metadata header block, the octave-band ``L'nT`` table
        beside the measured-versus-shifted-ISO 717-2-reference curve, the
        boxed field rating ``L'nT,w (CI)``, the survey-method statement, an
        optional verdict row (impact passes at or below the requirement) and
        a footer with the identity block and disclaimer.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`;
            ``None`` produces a lightweight fiche (body, rating and
            disclaimer only).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True``, the table shows the ISO 717 evaluation
            per band instead of the two-column ``f | value`` form.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is unknown or the result has no
            single-number rating (the survey needs 5 octave or 16
            one-third-octave bands).
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        _validate_report_request(engine, language)
        if self.rating is None:
            raise ValueError(
                "The survey impact report needs the ISO 717-2 single-number "
                "rating; supply 5 octave or 16 one-third-octave bands."
            )
        from ..._report.iso10052 import render_survey_impact_report

        return render_survey_impact_report(
            self, self.rating, path, metadata=metadata, verbose=verbose,
            language=language,
        )


@dataclass(frozen=True)
class SurveyFacadeResult:
    r"""Per-band façade sound insulation, survey method (ISO 10052).

    :ivar d_2m: Façade level difference :math:`D_{2m} = L_{1,2m} - L_2`
        (Clause 3.13).
    :ivar d_2m_nt: Standardized façade level difference ``D2m,nT``
        (Clause 3.14).
    :ivar d_2m_n: Normalized façade level difference ``D2m,n`` (Clause 3.15),
        or ``None`` when the receiving-room volume was not supplied.
    :ivar rating: Weighted standardized façade level difference ``D2m,nT,w``
        (ISO 717-1), or ``None`` off the 5/16-band count.
    """

    d_2m: np.ndarray
    d_2m_nt: np.ndarray
    d_2m_n: np.ndarray | None
    rating: WeightedRatingResult | None

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot ``D2m,nT`` against the shifted ISO 717-1 reference curve."""
        if self.rating is None:
            raise ValueError(
                "No single-number rating is available to plot (need 5 octave "
                "or 16 one-third-octave bands)."
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
        """Render a survey-method facade field report to a PDF (ISO 10052).

        Writes the one-page survey (control) method field facade report: the
        standard-basis line naming ISO 10052 (octave bands), an optional
        metadata header block, the octave-band ``D2m,nT`` table beside the
        measured-versus-shifted-ISO 717-1-reference curve, the boxed field
        rating ``D2m,nT,w (C; Ctr)``, the survey-method statement, an optional
        verdict row (a facade level difference passes at or above the
        requirement) and a footer with the identity block and disclaimer.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`;
            ``None`` produces a lightweight fiche (body, rating and
            disclaimer only).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True``, the table shows the ISO 717 evaluation
            per band instead of the two-column ``f | value`` form.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is unknown or the result has no
            single-number rating (the survey needs 5 octave or 16
            one-third-octave bands).
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        _validate_report_request(engine, language)
        if self.rating is None:
            raise ValueError(
                "The survey facade report needs the ISO 717-1 single-number "
                "rating; supply 5 octave or 16 one-third-octave bands."
            )
        from ..._report.iso10052 import render_survey_facade_report

        return render_survey_facade_report(
            self, self.rating, path, metadata=metadata, verbose=verbose,
            language=language,
        )


@dataclass(frozen=True)
class SurveyServiceEquipmentResult:
    r"""Service-equipment sound pressure level, survey method (ISO 10052).

    :ivar l_xy: Service-equipment level ``LXY`` (Clause 3.16), the energy
        average of the three measurement positions, in dB.
    :ivar l_xy_nt: Standardized level :math:`L_{XY,nT} = L_{XY} - k`
        (Clause 3.17).
    :ivar l_xy_n: Normalized level ``LXY,n`` (Clause 3.18), or ``None`` when
        the receiving-room volume was not supplied.
    """

    l_xy: np.ndarray
    l_xy_nt: np.ndarray
    l_xy_n: np.ndarray | None


def _validate_index(k: Sequence[float] | np.ndarray, n_bands: int) -> np.ndarray:
    """Coerce the reverberation index to a finite per-band array."""
    kk = _finite_bands(k, "reverberation_index")
    if kk.shape != (n_bands,):
        raise ValueError(
            "'reverberation_index' must give one value per band, matching the "
            "level inputs."
        )
    return kk


def survey_airborne_insulation(
    l1: Sequence[float] | np.ndarray,
    l2: Sequence[float] | np.ndarray,
    reverberation_index: Sequence[float] | np.ndarray,
    *,
    volume: float | None = None,
    area: float | None = None,
) -> SurveyAirborneResult:
    r"""
    Airborne sound insulation between rooms, survey method (ISO 10052:2021).

    Computes, per octave band, the level difference :math:`D = L_1 - L_2`
    (Clause 3.2), the standardized level difference :math:`D_{nT} = D + k`
    (Clause 3.4) and, when ``volume`` is given, the normalized level
    difference :math:`D_n = D + k + 10 \log_{10}(A_0 T_0 / (0.16\,V))` (Clause 3.5).
    When a
    common-partition ``area`` is also given, the apparent sound reduction index
    :math:`R' = D + k + 10 \log_{10}(S T_0 / (0.16\,V))` (Clause 3.6) is formed,
    using
    :math:`V/7.5` for ``S`` where that exceeds the given area (Clause 3.6).
    The
    reverberation index ``k`` comes from :func:`reverberation_index` (measured
    ``T``) or a Clause 6.5 estimate.

    ``l1`` and ``l2`` may be one value per band or a two-dimensional
    ``(positions, bands)`` array (energy-averaged over the positions).

    :param l1: Source-room sound pressure levels, in dB.
    :param l2: Receiving-room sound pressure levels, in dB.
    :param reverberation_index: Reverberation index ``k`` per band, in dB.
    :param volume: Receiving-room volume ``V``, in m³ (required for ``Dn`` and
        ``R'``).
    :param area: Common-partition area ``S``, in m² (with ``volume``, gives
        ``R'``).
    :return: :class:`SurveyAirborneResult`.
    :raises ValueError: If band counts differ, if ``area`` is given without
        ``volume``, or if ``area``/``volume`` are not positive.
    """
    l1_bands = _as_band_levels(l1, "l1")
    l2_bands = _as_band_levels(l2, "l2")
    if l1_bands.shape != l2_bands.shape:
        raise ValueError("'l1' and 'l2' must share the same band count.")
    k = _validate_index(reverberation_index, int(l1_bands.size))

    d = l1_bands - l2_bands
    d_nt = d + k

    if area is not None and volume is None:
        raise ValueError("'area' requires 'volume' to compute R'.")

    d_n: np.ndarray | None = None
    r_prime: np.ndarray | None = None
    if volume is not None:
        v = _positive(volume, "volume")
        d_n = d + k + _normalization_term(v)
        if area is not None:
            s_eff = max(_positive(area, "area"), v / _MIN_AREA_FACTOR)
            r_prime = d + k + 10.0 * np.log10(s_eff * _T0 / (_SABINE * v))

    return SurveyAirborneResult(
        d=d,
        d_nt=d_nt,
        d_n=d_n,
        r_prime=r_prime,
        rating=_rate(d_nt),
        r_prime_rating=_rate(r_prime) if r_prime is not None else None,
    )


def survey_impact_insulation(
    li: Sequence[float] | np.ndarray,
    reverberation_index: Sequence[float] | np.ndarray,
    *,
    volume: float | None = None,
) -> SurveyImpactResult:
    r"""
    Impact sound insulation between rooms, survey method (ISO 10052:2021).

    Computes, per octave band, the energy-average impact sound pressure level
    ``Li`` (Clause 3.7), the standardized impact level
    :math:`L'_{nT} = L_i - k`
    (Clause 3.8) and, when ``volume`` is given, the normalized impact level
    :math:`L'_n = L_i - k - 10 \log_{10}(A_0 T_0 / (0.16\,V))` (Clause 3.9).

    ``li`` may be one value per band or a two-dimensional
    ``(positions, bands)`` array (energy-averaged over the tapping-machine
    positions, Clause 3.7).

    :param li: Impact sound pressure levels, in dB.
    :param reverberation_index: Reverberation index ``k`` per band, in dB.
    :param volume: Receiving-room volume ``V``, in m³ (required for ``L'n``).
    :return: :class:`SurveyImpactResult`.
    :raises ValueError: If band counts differ or ``volume`` is not positive.
    """
    l_i = _as_band_levels(li, "li")
    k = _validate_index(reverberation_index, int(l_i.size))

    l_nt = l_i - k
    l_n: np.ndarray | None = None
    if volume is not None:
        l_n = l_i - k - _normalization_term(_positive(volume, "volume"))

    return SurveyImpactResult(l_i=l_i, l_nt=l_nt, l_n=l_n, rating=_rate_impact(l_nt))


def survey_facade_insulation(
    l1_2m: Sequence[float] | np.ndarray,
    l2: Sequence[float] | np.ndarray,
    reverberation_index: Sequence[float] | np.ndarray,
    *,
    volume: float | None = None,
) -> SurveyFacadeResult:
    r"""
    Façade sound insulation, survey method (ISO 10052:2021).

    Computes, per octave band, the façade level difference
    :math:`D_{2m} = L_{1,2m} - L_2` (Clause 3.13), the standardized façade
    level
    difference :math:`D_{2m,nT} = D_{2m} + k` (Clause 3.14) and, when
    ``volume`` is
    given, the normalized façade level difference
    :math:`D_{2m,n} = D_{2m} + k + 10 \log_{10}(A_0 T_0 / (0.16\,V))`
    (Clause 3.15).

    :param l1_2m: Outdoor sound pressure levels 2 m in front of the façade,
        in dB (one value per band or ``(positions, bands)``).
    :param l2: Receiving-room sound pressure levels, in dB.
    :param reverberation_index: Reverberation index ``k`` per band, in dB.
    :param volume: Receiving-room volume ``V``, in m³ (required for ``D2m,n``).
    :return: :class:`SurveyFacadeResult`.
    :raises ValueError: If band counts differ or ``volume`` is not positive.
    """
    out = _as_band_levels(l1_2m, "l1_2m")
    l2_bands = _as_band_levels(l2, "l2")
    if out.shape != l2_bands.shape:
        raise ValueError("'l1_2m' and 'l2' must share the same band count.")
    k = _validate_index(reverberation_index, int(out.size))

    d_2m = out - l2_bands
    d_2m_nt = d_2m + k
    d_2m_n: np.ndarray | None = None
    if volume is not None:
        d_2m_n = d_2m + k + _normalization_term(_positive(volume, "volume"))

    return SurveyFacadeResult(
        d_2m=d_2m, d_2m_nt=d_2m_nt, d_2m_n=d_2m_n, rating=_rate(d_2m_nt)
    )


def survey_service_equipment_level(
    measurements: Sequence[float] | np.ndarray,
    reverberation_index: float | Sequence[float] | np.ndarray,
    *,
    volume: float | None = None,
) -> SurveyServiceEquipmentResult:
    r"""
    Service-equipment sound pressure level, survey method (ISO 10052:2021).

    Computes the service-equipment level
    :math:`L_{XY} = 10 \log_{10}[(1/3) \sum 10^{0.1 L_{XY,i}}]`
    (Clause 3.16) as the energy average of the three measurement
    positions (one near a corner, two in the reverberant field, Clause 6.3.3),
    the standardized level :math:`L_{XY,nT} = L_{XY} - k` (Clause 3.17) and,
    when
    ``volume`` is given, the normalized level
    :math:`L_{XY,n} = L_{XY} - k - 10 \log_{10}(A_0 T_0 / (0.16\,V))`
    (Clause 3.18). ``X`` is the frequency weighting
    (A or C) and ``Y`` the time weighting (F, S or Leq).

    :param measurements: The three A- or C-weighted levels, in dB; either
        three scalars, or a ``(3, bands)`` array for a banded analysis.
    :param reverberation_index: Reverberation index ``k``, in dB; a scalar
        for a weighted level, or one value per band; for a global weighted
        level ``k`` is taken from the mean of the 500/1000/2000 Hz octave
        reverberation times (Clause 3.17).
    :param volume: Receiving-room volume ``V``, in m³ (required for ``LXY,n``).
    :return: :class:`SurveyServiceEquipmentResult`.
    :raises ValueError: If not exactly three measurements are given, if shapes
        are inconsistent, or if ``volume`` is not positive.
    """
    m = _finite_bands(measurements, "measurements")
    if m.ndim not in (1, 2) or m.shape[0] != 3:
        raise ValueError(
            "'measurements' must give exactly three measurement positions "
            "(three scalars, or a (3, bands) array)."
        )
    l_xy = np.asarray(energy_average_level(m, axis=0), dtype=np.float64)

    k = _finite_bands(reverberation_index, "reverberation_index")
    if k.shape != l_xy.shape:
        raise ValueError(
            "'reverberation_index' must match the shape of the per-position "
            "average (scalar, or one value per band)."
        )

    l_xy_nt = l_xy - k
    l_xy_n: np.ndarray | None = None
    if volume is not None:
        l_xy_n = l_xy - k - _normalization_term(_positive(volume, "volume"))

    return SurveyServiceEquipmentResult(l_xy=l_xy, l_xy_nt=l_xy_nt, l_xy_n=l_xy_n)
