#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Measurement uncertainty for sound absorption (ISO 12999-2:2020).

Companion of the sound-insulation uncertainty of :mod:`phonometry.building.measurement.uncertainty`
(ISO 12999-1). This part gives the standard uncertainty ``u`` of the quantities
produced by a reverberation-room absorption measurement and its ratings:

* the **sound absorption coefficient** ``αs`` and the **equivalent sound absorption
  area** ``AT`` measured according to ISO 354, in one-third-octave bands
  (Clause 5, Table 1);
* the **practical sound absorption coefficient** ``αp`` determined according to
  ISO 11654, in octave bands (Clause 6, Table 2);
* the single-number **weighted sound absorption coefficient** ``αw`` (ISO 11654)
  and **single-number rating** ``DLα,NRD`` (EN 1793-1) (Clause 7).

The standard uncertainty is estimated by the reproducibility standard deviation
``σR`` (different laboratories) or the repeatability standard deviation ``σr``
(same location, operator and equipment), both derived from inter-laboratory tests
to ISO 5725. Where specimen-specific uncertainty data exist they take precedence
(Clause 4).

**Band formulae (Clause 5-6).** For the absorption coefficient
:math:`\sigma_\mathrm{R} = m \alpha + n` (Formula (1)/(4)); for the equivalent area
:math:`\sigma_\mathrm{R} = m A_\mathrm{T} + n S` with :math:`S = 10~\text{m}^2` (Formula (2));
the repeatability value is :math:`\sigma_\mathrm{r} = 0.6\,\sigma_\mathrm{R}` (Formula (3)/(5)).
``m`` and ``n`` are the frequency-dependent
constants of Table 1 (one-third-octave, 63-5000 Hz) and Table 2 (octave,
250-4000 Hz).

**Single numbers (Clause 7).** ``αw``: :math:`\sigma_\mathrm{R} = 0.035`
(Formula (6)), :math:`\sigma_\mathrm{r} = 0.020` (Formula (7)). ``DLα,NRD``:
:math:`\sigma_\mathrm{R} = 0.10\,D_{L\alpha,\mathrm{NRD}}` (Formula (8)),
:math:`\sigma_\mathrm{r} = 0.02\,D_{L\alpha,\mathrm{NRD}}` (Formula (9)).

**Reporting (Clause 8).** The expanded uncertainty is
:math:`U = k\,u` (Formula (10))
with the coverage factor ``k`` of Table 3 (Gaussian assumption). The reported
``U`` is rounded to two decimal digits for absorption coefficients and one decimal
digit for the equivalent area and ``DLα,NRD``; the code keeps the exact ``U`` and
exposes the rounded view through :attr:`AbsorptionUncertaintyResult.reported_expanded_uncertainty`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

__all__ = [
    "AbsorptionUncertaintyResult",
    "absorption_coverage_factor",
    "equivalent_area_uncertainty",
    "practical_coefficient_uncertainty",
    "single_number_rating_uncertainty",
    "sound_absorption_coefficient_uncertainty",
    "weighted_coefficient_uncertainty",
]

#: Reference equivalent absorption area constant ``S`` in Formula (2): 10 m².
_EQUIV_AREA_REFERENCE_S = 10.0

#: Repeatability factor ``σr = 0,6·σR`` (Formulae (3)/(5)).
_REPEATABILITY_FACTOR = 0.6

#: Measurement conditions selecting σR (reproducibility) or σr (repeatability).
_CONDITIONS = ("reproducibility", "repeatability")

#: Table 1 (Clause 5): ``(m, n)`` for Formulae (1)/(2), keyed by one-third-octave
#: midband frequency in Hz (63-5000). Transcribed digit-for-digit from
#: ISO 12999-2:2020, Table 1.
_TABLE1: dict[float, tuple[float, float]] = {
    63.0: (0.450, 0.015),
    80.0: (0.330, 0.015),
    100.0: (0.240, 0.015),
    125.0: (0.180, 0.015),
    160.0: (0.140, 0.015),
    200.0: (0.110, 0.015),
    250.0: (0.090, 0.015),
    315.0: (0.075, 0.015),
    400.0: (0.060, 0.015),
    500.0: (0.050, 0.015),
    630.0: (0.045, 0.015),
    800.0: (0.040, 0.015),
    1000.0: (0.040, 0.015),
    1250.0: (0.040, 0.016),
    1600.0: (0.037, 0.018),
    2000.0: (0.035, 0.021),
    2500.0: (0.030, 0.026),
    3150.0: (0.030, 0.032),
    4000.0: (0.030, 0.040),
    5000.0: (0.026, 0.060),
}

#: Table 2 (Clause 6): ``(m, n)`` for Formula (4), keyed by octave midband
#: frequency in Hz (250-4000). Transcribed from ISO 12999-2:2020, Table 2.
_TABLE2: dict[float, tuple[float, float]] = {
    250.0: (0.059, 0.016),
    500.0: (0.000, 0.040),
    1000.0: (0.000, 0.040),
    2000.0: (0.000, 0.040),
    4000.0: (0.000, 0.050),
}

#: Weighted sound absorption coefficient ``αw`` (Clause 7, Formulae (6)/(7)).
_ALPHA_W_SIGMA: dict[str, float] = {"reproducibility": 0.035, "repeatability": 0.020}

#: Single-number rating ``DLα,NRD`` scale factors (Clause 7, Formulae (8)/(9)).
_DLALPHA_FACTOR: dict[str, float] = {"reproducibility": 0.10, "repeatability": 0.02}

#: Table 3 (Clause 8): coverage factors ``k`` for a Gaussian measurand, keyed by
#: confidence level as a fraction. Values are the standard's rounded factors
#: (1,0 / 1,3 / 1,6 / 2,0 / 2,6 / 3,3), which differ from the Gaussian-exact
#: factors of ISO 12999-1 Table 8.
_COVERAGE_FACTORS: dict[float, float] = {
    0.68: 1.0,
    0.80: 1.3,
    0.90: 1.6,
    0.95: 2.0,
    0.99: 2.6,
    0.999: 3.3,
}

#: Quantities whose reported expanded uncertainty rounds to two decimals
#: (absorption coefficients); everything else rounds to one decimal (Clause 8).
_COEFFICIENT_QUANTITIES = frozenset(
    {"absorption_coefficient", "practical_coefficient", "weighted_coefficient"}
)


def _condition(condition: str) -> str:
    if condition not in _CONDITIONS:
        msg = f"'condition' must be one of {_CONDITIONS}; got {condition!r}."
        raise ValueError(msg)
    return condition


def _finite_bands(
    values: float | Sequence[float] | np.ndarray, name: str
) -> np.ndarray:
    a = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if not np.all(np.isfinite(a)):
        msg = f"'{name}' must contain only finite values."
        raise ValueError(msg)
    return a


def _round_report(u: np.ndarray, decimals: int) -> np.ndarray:
    """Round-half-up to *decimals* (numpy rounds half-to-even, unlike ISO 80000)."""
    scale = 10.0**decimals
    return np.floor(np.asarray(u, dtype=np.float64) * scale + 0.5) / scale


def absorption_coverage_factor(confidence: float = 0.95) -> float:
    """Return the coverage factor ``k`` for a confidence level (Table 3).

    :param confidence: Confidence level as a fraction; one of
        ``0.68, 0.80, 0.90, 0.95, 0.99, 0.999`` (Table 3, Gaussian measurand).
    :raises ValueError: Confidence level not tabulated in Table 3.
    """
    for level, k in _COVERAGE_FACTORS.items():
        if abs(level - confidence) < 1e-9:
            return k
    valid = ", ".join(f"{level:g}" for level in _COVERAGE_FACTORS)
    msg = (
        f"Confidence level {confidence!r} is not tabulated in Table 3. Valid: {valid}."
    )
    raise ValueError(msg)


@dataclass(frozen=True)
class AbsorptionUncertaintyResult:
    r"""Standard and expanded uncertainty of an absorption quantity
    (ISO 12999-2).

    For the single-number quantities (``αw``, ``DLα,NRD``) the frequency and value
    arrays are empty and the uncertainty arrays hold a single element.

    :ivar quantity: ``"absorption_coefficient"``, ``"equivalent_area"``,
        ``"practical_coefficient"``, ``"weighted_coefficient"`` or ``"single_number_rating"``.
    :ivar condition: ``"reproducibility"`` (``σR``) or ``"repeatability"`` (``σr``).
    :ivar frequencies: Band centre frequencies, in Hz (empty for single numbers).
    :ivar values: The input quantity per band (empty for single numbers).
    :ivar standard_uncertainty: Standard uncertainty ``u`` (``σR``/``σr``), one per band.
    :ivar coverage_factor: Coverage factor ``k`` (Table 3).
    :ivar expanded_uncertainty: Exact :math:`U = k\,u` (Formula (10)), one
        per band.
    """

    quantity: str
    condition: str
    frequencies: np.ndarray
    values: np.ndarray
    standard_uncertainty: np.ndarray
    coverage_factor: float
    expanded_uncertainty: np.ndarray

    @property
    def reported_expanded_uncertainty(self) -> np.ndarray:
        """``U`` rounded for reporting (Clause 8): two decimals for absorption
        coefficients, one decimal for the equivalent area and ``DLα,NRD``.
        """
        decimals = 2 if self.quantity in _COEFFICIENT_QUANTITIES else 1
        return _round_report(self.expanded_uncertainty, decimals)

    @property
    def lower(self) -> np.ndarray:
        """Lower interval bound ``value − U`` (exact ``U``)."""
        return np.asarray(self.values - self.expanded_uncertainty, dtype=np.float64)

    @property
    def upper(self) -> np.ndarray:
        """Upper interval bound ``value + U`` (exact ``U``)."""
        return np.asarray(self.values + self.expanded_uncertainty, dtype=np.float64)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the quantity with its ``±U`` uncertainty ribbon (band quantities).

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.materials import plot_absorption_uncertainty

        check_language(language)
        return plot_absorption_uncertainty(self, ax=ax, language=language, **kwargs)


def _table_constants(
    frequencies: np.ndarray, table: dict[float, tuple[float, float]], band_kind: str
) -> tuple[np.ndarray, np.ndarray]:
    """Look up ``(m, n)`` per band, raising on an untabulated frequency."""
    try:
        mn = np.array([table[float(f)] for f in frequencies], dtype=np.float64)
    except KeyError as exc:
        valid = ", ".join(f"{f:g}" for f in table)
        msg = (
            f"Frequency {exc.args[0]:g} Hz is not a tabulated {band_kind} band. "
            f"Valid: {valid}."
        )
        raise ValueError(msg) from None
    return mn[:, 0], mn[:, 1]


def _band_uncertainty(
    values: np.ndarray,
    frequencies: np.ndarray,
    table: dict[float, tuple[float, float]],
    *,
    quantity: str,
    condition: str,
    confidence: float,
    band_kind: str,
) -> AbsorptionUncertaintyResult:
    r"""Shared engine for the coefficient formulae
    :math:`\sigma_\mathrm{R} = m \alpha + n` (values == α).
    """
    if values.shape != frequencies.shape:
        msg = (
            f"'{quantity}' values and frequencies must have the same shape; "
            f"got {values.shape} and {frequencies.shape}."
        )
        raise ValueError(msg)
    m, n = _table_constants(frequencies, table, band_kind)
    sigma_r = m * values + n
    u = sigma_r if condition == "reproducibility" else _REPEATABILITY_FACTOR * sigma_r
    k = absorption_coverage_factor(confidence)
    return AbsorptionUncertaintyResult(
        quantity=quantity,
        condition=condition,
        frequencies=frequencies,
        values=values,
        standard_uncertainty=u,
        coverage_factor=k,
        expanded_uncertainty=k * u,
    )


def sound_absorption_coefficient_uncertainty(
    alpha: float | Sequence[float] | np.ndarray,
    frequencies: float | Sequence[float] | np.ndarray,
    *,
    condition: str = "reproducibility",
    confidence: float = 0.95,
) -> AbsorptionUncertaintyResult:
    r"""Uncertainty of the ISO 354 sound absorption coefficient ``αs``
    (Clause 5).

    :math:`\sigma_\mathrm{R} = m \alpha_\mathrm{s} + n` (Formula (1)) per one-third-octave band;
    :math:`\sigma_\mathrm{r} = 0.6\,\sigma_\mathrm{R}` (Formula (3)).

    :param alpha: Sound absorption coefficient ``αs`` per band.
    :param frequencies: One-third-octave midband frequencies in Hz (Table 1,
        63-5000), aligned with ``alpha``.
    :param condition: ``"reproducibility"`` (default) or ``"repeatability"``.
    :param confidence: Confidence level for the coverage factor ``k`` (Table 3).
    :raises ValueError: Mismatched shapes, an untabulated frequency, an unknown
        ``condition`` or an untabulated ``confidence``.
    """
    return _band_uncertainty(
        _finite_bands(alpha, "alpha"),
        _finite_bands(frequencies, "frequencies"),
        _TABLE1,
        quantity="absorption_coefficient",
        condition=_condition(condition),
        confidence=confidence,
        band_kind="one-third-octave",
    )


def equivalent_area_uncertainty(
    area: float | Sequence[float] | np.ndarray,
    frequencies: float | Sequence[float] | np.ndarray,
    *,
    condition: str = "reproducibility",
    confidence: float = 0.95,
) -> AbsorptionUncertaintyResult:
    r"""Uncertainty of the ISO 354 equivalent sound absorption area ``AT``
    (Clause 5).

    :math:`\sigma_\mathrm{R} = m A_\mathrm{T} + n S` with :math:`S = 10~\text{m}^2` (Formula (2))
    per one-third-octave band; :math:`\sigma_\mathrm{r} = 0.6\,\sigma_\mathrm{R}` (Formula (3)).

    :param area: Equivalent sound absorption area ``AT`` per band, in m².
    :param frequencies: One-third-octave midband frequencies in Hz (Table 1),
        aligned with ``area``.
    :param condition: ``"reproducibility"`` (default) or ``"repeatability"``.
    :param confidence: Confidence level for the coverage factor ``k`` (Table 3).
    :raises ValueError: Mismatched shapes, an untabulated frequency, an unknown
        ``condition`` or an untabulated ``confidence``.
    """
    a = _finite_bands(area, "area")
    f = _finite_bands(frequencies, "frequencies")
    if a.shape != f.shape:
        msg = (
            f"'area' values and frequencies must have the same shape; "
            f"got {a.shape} and {f.shape}."
        )
        raise ValueError(msg)
    m, n = _table_constants(f, _TABLE1, "one-third-octave")
    cond = _condition(condition)
    sigma_r = m * a + n * _EQUIV_AREA_REFERENCE_S
    u = sigma_r if cond == "reproducibility" else _REPEATABILITY_FACTOR * sigma_r
    k = absorption_coverage_factor(confidence)
    return AbsorptionUncertaintyResult(
        quantity="equivalent_area",
        condition=cond,
        frequencies=f,
        values=a,
        standard_uncertainty=u,
        coverage_factor=k,
        expanded_uncertainty=k * u,
    )


def practical_coefficient_uncertainty(
    alpha_p: float | Sequence[float] | np.ndarray,
    frequencies: float | Sequence[float] | np.ndarray,
    *,
    condition: str = "reproducibility",
    confidence: float = 0.95,
) -> AbsorptionUncertaintyResult:
    r"""Uncertainty of the ISO 11654 practical absorption coefficient ``αp``
    (Clause 6).

    :math:`\sigma_\mathrm{R} = m \alpha_\mathrm{p} + n` (Formula (4)) per octave band;
    :math:`\sigma_\mathrm{r} = 0.6\,\sigma_\mathrm{R}` (Formula (5)).

    :param alpha_p: Practical sound absorption coefficient ``αp`` per band.
    :param frequencies: Octave midband frequencies in Hz (Table 2, 250-4000),
        aligned with ``alpha_p``.
    :param condition: ``"reproducibility"`` (default) or ``"repeatability"``.
    :param confidence: Confidence level for the coverage factor ``k`` (Table 3).
    :raises ValueError: Mismatched shapes, an untabulated frequency, an unknown
        ``condition`` or an untabulated ``confidence``.
    """
    return _band_uncertainty(
        _finite_bands(alpha_p, "alpha_p"),
        _finite_bands(frequencies, "frequencies"),
        _TABLE2,
        quantity="practical_coefficient",
        condition=_condition(condition),
        confidence=confidence,
        band_kind="octave",
    )


def _single_number(
    value: float,
    u: float,
    *,
    quantity: str,
    condition: str,
    confidence: float,
) -> AbsorptionUncertaintyResult:
    k = absorption_coverage_factor(confidence)
    empty = np.asarray([], dtype=np.float64)
    return AbsorptionUncertaintyResult(
        quantity=quantity,
        condition=condition,
        frequencies=empty,
        values=np.asarray([value], dtype=np.float64),
        standard_uncertainty=np.asarray([u], dtype=np.float64),
        coverage_factor=k,
        expanded_uncertainty=np.asarray([k * u], dtype=np.float64),
    )


def weighted_coefficient_uncertainty(
    alpha_w: float = 0.0,
    *,
    condition: str = "reproducibility",
    confidence: float = 0.95,
) -> AbsorptionUncertaintyResult:
    r"""Uncertainty of the ISO 11654 weighted absorption coefficient ``αw``
    (Clause 7).

    The standard uncertainty is a constant: :math:`\sigma_\mathrm{R} = 0.035`
    (Formula (6)), :math:`\sigma_\mathrm{r} = 0.020` (Formula (7)), independent of
    the value.

    :param alpha_w: Weighted sound absorption coefficient ``αw`` (carried through
        for the reported interval; does not affect ``u``).
    :param condition: ``"reproducibility"`` (default) or ``"repeatability"``.
    :param confidence: Confidence level for the coverage factor ``k`` (Table 3).
    :raises ValueError: Unknown ``condition`` or an untabulated ``confidence``.
    """
    cond = _condition(condition)
    return _single_number(
        float(alpha_w),
        _ALPHA_W_SIGMA[cond],
        quantity="weighted_coefficient",
        condition=cond,
        confidence=confidence,
    )


def single_number_rating_uncertainty(
    dl_alpha: float,
    *,
    condition: str = "reproducibility",
    confidence: float = 0.95,
) -> AbsorptionUncertaintyResult:
    r"""Uncertainty of the EN 1793-1 single-number rating ``DLα,NRD``
    (Clause 7).

    :math:`\sigma_\mathrm{R} = 0.10\,D_{L\alpha,\mathrm{NRD}}` (Formula (8)),
    :math:`\sigma_\mathrm{r} = 0.02\,D_{L\alpha,\mathrm{NRD}}` (Formula (9)).

    :param dl_alpha: Single-number rating ``DLα,NRD``, in dB (non-negative).
    :param condition: ``"reproducibility"`` (default) or ``"repeatability"``.
    :param confidence: Confidence level for the coverage factor ``k`` (Table 3).
    :raises ValueError: Negative ``dl_alpha``, unknown ``condition`` or an
        untabulated ``confidence``.
    """
    value = float(dl_alpha)
    if not np.isfinite(value) or value < 0.0:
        msg = "'dl_alpha' must be a non-negative, finite value."
        raise ValueError(msg)
    cond = _condition(condition)
    return _single_number(
        value,
        _DLALPHA_FACTOR[cond] * value,
        quantity="single_number_rating",
        condition=cond,
        confidence=confidence,
    )
