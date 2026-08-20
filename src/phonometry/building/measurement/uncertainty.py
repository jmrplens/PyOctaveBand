#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Measurement uncertainty in building acoustics (ISO 12999-1:2020).

This module supplies the **measurement uncertainty** of the sound-insulation
quantities produced by the field/lab/prediction modules
(:mod:`phonometry.building.measurement.insulation`, :mod:`phonometry.building.measurement.lab_insulation`,
:mod:`phonometry.building.prediction.simplified_model`). ISO 12999-1 does not re-measure anything;
it tabulates *standard uncertainties* ``u`` derived from inter-laboratory tests
(ISO 5725) and prescribes how to expand and combine them.

**Three measurement situations (Clause 5.2)** fix which standard deviation is the
standard uncertainty ``u``:

- **A**: laboratory characterisation (ISO 10140); ``u`` = reproducibility ``σR``.
- **B**: same location, different teams; ``u`` = in-situ ``σsitu``.
- **C**: same location, same operator/equipment repeated; ``u`` = repeatability ``σr``.

**Tabulated standard uncertainties** (one-third-octave and single-number):

- Airborne ``R``/``R'``/``Dn``/``DnT``: Table 2 (bands) and Table 3 (ratings).
- Impact ``Ln``/``L'n``/``L'nT``: Table 4 (bands, situations B/C only) and Table 5
  (ratings). ISO 12999-1:2020 Table 4 has **no 500 Hz band** (the 2014 edition did).
- Reduction of impact noise by floor coverings ``ΔL``/``ΔLw``: Table 6 (bands) and
  Table 7 (rating), situation A only.
- Upper 95 % limit of airborne reproducibility ``σR95``: Annex D Tables D.1/D.2
  (situation A; informative). In ISO 12999-1:2014 these were extra columns of
  Tables 2/3.
- Maximum repeatability standard deviation for lab self-verification: Table 1.

**Expansion (Clause 8).** :math:`U = k\,u` (Formula 2) with the coverage
factor ``k`` of
Table 8 (a minimum of :math:`k = 1` is enforced). Declaring conformity with a
requirement uses the **one-sided** factor (Formulae 4/5); reporting a two-sided
interval :math:`Y = y \pm U` (Formula 3) uses the two-sided factor.

**Combination.** Uncorrelated quadrature :math:`u_\mathrm{c} = \sqrt{\sum u_i^2}`
(Formula C.2);
prediction input uncertainty (Formula A.1); model/reality combination (Formula A.2);
reduction by ``m`` independent measurements :math:`u/\sqrt{m}` (Formula A.7);
and the
uncorrelated single-number combination of Annex B (Formula B.2).

Clause/table numbers refer to ISO 12999-1:2020(E).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes

Situation = Literal["A", "B", "C"]
Measurand = Literal["airborne", "impact", "impact_reduction"]

# --------------------------------------------------------------------------- #
# One-third-octave-band frequency axes (Hz).
# --------------------------------------------------------------------------- #
#: 21 bands 50-5000 Hz including 500 Hz (Tables 2, 6, D.1).
_FREQ_FULL: tuple[float, ...] = (
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
)
#: 20 bands 50-5000 Hz **without** 500 Hz (Table 4, ISO 12999-1:2020).
_FREQ_IMPACT: tuple[float, ...] = (
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
)

# --------------------------------------------------------------------------- #
# Table 1: Maximum standard deviation of repeatability (Clause 5.8).
# --------------------------------------------------------------------------- #
_TABLE1: tuple[float, ...] = (
    4.0,
    3.5,
    3.0,
    2.6,
    2.2,
    1.9,
    1.7,
    1.5,
    1.4,
    1.3,
    1.3,
    1.3,
    1.3,
    1.3,
    1.3,
    1.3,
    1.3,
    1.3,
    1.3,
    1.3,
    1.3,
)

# --------------------------------------------------------------------------- #
# Table 2: Airborne one-third-octave (Clause 7.2). Columns A/B/C = σR/σsitu/σr.
# --------------------------------------------------------------------------- #
_TABLE2_A: tuple[float, ...] = (
    6.8,
    4.6,
    3.8,
    3.0,
    2.7,
    2.4,
    2.1,
    1.8,
    1.8,
    1.8,
    1.8,
    1.8,
    1.8,
    1.8,
    1.8,
    1.8,
    1.8,
    1.9,
    2.0,
    2.4,
    2.8,
)
_TABLE2_B: tuple[float, ...] = (
    4.0,
    3.6,
    3.2,
    2.8,
    2.4,
    2.0,
    1.8,
    1.6,
    1.4,
    1.2,
    1.1,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.3,
    1.6,
    1.9,
    2.2,
)
_TABLE2_C: tuple[float, ...] = (
    2.0,
    1.8,
    1.6,
    1.4,
    1.2,
    1.0,
    0.9,
    0.8,
    0.7,
    0.6,
    0.6,
    0.6,
    0.6,
    0.6,
    0.6,
    0.6,
    0.6,
    0.6,
    0.6,
    0.6,
    0.6,
)

# --------------------------------------------------------------------------- #
# Table 4: Impact one-third-octave (Clause 7.3). Situations B/C only, no 500 Hz.
# --------------------------------------------------------------------------- #
_TABLE4_B: tuple[float, ...] = (
    3.2,
    2.8,
    2.4,
    2.0,
    1.6,
    1.4,
    1.3,
    1.2,
    1.2,
    1.2,
    1.2,
    1.2,
    1.2,
    1.3,
    1.4,
    1.5,
    1.7,
    1.9,
    2.1,
    2.3,
)
_TABLE4_C: tuple[float, ...] = (
    1.5,
    1.4,
    1.3,
    1.2,
    1.1,
    1.0,
    0.9,
    0.8,
    0.8,
    0.8,
    0.8,
    0.8,
    0.8,
    0.8,
    0.8,
    0.8,
    1.0,
    1.2,
    1.4,
    1.6,
)

# --------------------------------------------------------------------------- #
# Table 6: Reduction of impact noise by floor coverings ΔL (Clause 7.4).
# Situation A only.
# --------------------------------------------------------------------------- #
_TABLE6_A: tuple[float, ...] = (
    1.4,
    1.3,
    1.2,
    1.1,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.1,
    1.2,
    1.3,
    1.6,
    1.9,
    2.2,
    2.5,
    2.8,
    3.2,
    3.6,
    4.0,
    4.4,
)

# --------------------------------------------------------------------------- #
# Annex D Table D.1: σR95 airborne one-third-octave (situation A upper limit).
# --------------------------------------------------------------------------- #
_TABLED1: tuple[float, ...] = (
    11.7,
    6.7,
    5.9,
    5.0,
    5.0,
    3.8,
    3.3,
    3.3,
    3.3,
    3.3,
    3.3,
    3.3,
    3.3,
    3.3,
    3.4,
    3.4,
    3.4,
    3.5,
    3.6,
    4.0,
    4.7,
)

# Registry: (measurand, upper_limit) -> (frequencies, {situation: values}).
_BAND_TABLES: dict[
    tuple[str, bool], tuple[tuple[float, ...], dict[str, tuple[float, ...]]]
] = {
    ("airborne", False): (_FREQ_FULL, {"A": _TABLE2_A, "B": _TABLE2_B, "C": _TABLE2_C}),
    ("airborne", True): (_FREQ_FULL, {"A": _TABLED1}),
    ("impact", False): (_FREQ_IMPACT, {"B": _TABLE4_B, "C": _TABLE4_C}),
    ("impact_reduction", False): (_FREQ_FULL, {"A": _TABLE6_A}),
}

# --------------------------------------------------------------------------- #
# Single-number values. Each entry maps canonical descriptor -> per-situation
# (A, B, C) standard uncertainty and the situation-A σR95 upper limit (or None).
# Tables 3 (airborne), 5 (impact), 7 (reduction); Annex D Table D.2 (σR95).
# --------------------------------------------------------------------------- #
_SINGLE: dict[
    str, tuple[tuple[float | None, float | None, float | None], float | None]
] = {
    # Airborne: Table 3 (A/B/C) and Table D.2 (σR95, situation A).
    "r_w": ((1.2, 0.9, 0.4), 2.0),
    "r_w+c_100_3150": ((1.3, 0.9, 0.5), 2.1),
    "r_w+c_100_5000": ((1.3, 1.1, 0.5), 2.1),
    "r_w+c_50_3150": ((1.3, 1.0, 0.7), 2.1),
    "r_w+c_50_5000": ((1.3, 1.1, 0.7), 2.1),
    "r_w+ctr_100_3150": ((1.5, 1.1, 0.7), 2.4),
    "r_w+ctr_100_5000": ((1.5, 1.1, 0.7), 2.4),
    "r_w+ctr_50_3150": ((1.5, 1.3, 1.0), 2.4),
    # NOTE: σsitu(B)=1.0 here is anomalous: it is *lower* than the 50-3150 row
    # above (B=1.3) and equal to its own σr(C)=1.0, breaking the otherwise
    # monotonic pattern. Verified digit-by-digit against the ISO 12999-1:2020(E)
    # Table 3 (standard page 8): the standard normatively prints 1,5 / 1,0 / 1,0.
    "r_w+ctr_50_5000": ((1.5, 1.0, 1.0), 2.4),
    # Impact: Table 5 (situation A values are estimates, footnote a).
    "ln_w": ((1.5, 1.0, 0.5), None),
    "ln_w+ci": ((1.5, 1.0, 0.6), None),
    # Reduction: Table 7 (situation A only).
    "delta_lw": ((1.1, None, None), None),
}

#: Aliases so equivalent descriptors resolve to the same table row (Clause 7.2/7.3).
_ALIASES: dict[str, str] = {
    "rprime_w": "r_w",
    "r_prime_w": "r_w",
    "dn_w": "r_w",
    "dnw": "r_w",
    "dnt_w": "r_w",
    "dntw": "r_w",
    "lprime_n_w": "ln_w",
    "lnprime_w": "ln_w",
    "lnt_w": "ln_w",
    "lprime_nt_w": "ln_w",
    "delta_l_w": "delta_lw",
}

_SITUATION_INDEX: dict[str, int] = {"A": 0, "B": 1, "C": 2}

# --------------------------------------------------------------------------- #
# Table 8: Coverage factors (Clause 8). Keyed by confidence level (fraction).
# --------------------------------------------------------------------------- #
_COVERAGE_TWO_SIDED: dict[float, float] = {
    0.68: 1.00,
    0.80: 1.28,
    0.90: 1.65,
    0.95: 1.96,
    0.99: 2.58,
    0.999: 3.29,
}
_COVERAGE_ONE_SIDED: dict[float, float] = {
    0.84: 1.00,
    0.90: 1.28,
    0.95: 1.65,
    0.975: 1.96,
    0.995: 2.58,
    0.9995: 3.29,
}


# --------------------------------------------------------------------------- #
# Result containers.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BandUncertainty:
    """One-third-octave-band standard uncertainties (ISO 12999-1 Tables 2/4/6/D.1).

    :ivar measurand: ``"airborne"``, ``"impact"`` or ``"impact_reduction"``.
    :ivar situation: Measurement situation ``"A"``, ``"B"`` or ``"C"`` (Clause 5.2).
    :ivar frequencies: Band centre frequencies, in Hz.
    :ivar uncertainties: Standard uncertainty ``u`` per band, in dB.
    :ivar upper_limit: ``True`` for the ``σR95`` upper limit (Annex D Table D.1).
    """

    measurand: str
    situation: str
    frequencies: tuple[float, ...]
    uncertainties: tuple[float, ...]
    upper_limit: bool = False

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(frequencies, uncertainties)`` as float :class:`numpy.ndarray`."""
        return (
            np.asarray(self.frequencies, dtype=float),
            np.asarray(self.uncertainties, dtype=float),
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the per-band standard uncertainty spectrum.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_band_uncertainty

        check_language(language)
        return plot_band_uncertainty(self, ax=ax, language=language, **kwargs)


@dataclass(frozen=True)
class UncertainValue:
    r"""A best estimate with its ISO 12999-1 expanded uncertainty (Clause 8).

    :ivar value: Best estimate ``y`` (e.g. a weighted rating), in dB.
    :ivar standard_uncertainty: Standard uncertainty ``u``, in dB.
    :ivar coverage_factor: Coverage factor ``k`` (Table 8).
    :ivar expanded_uncertainty: :math:`U = k\,u`, in dB.
    :ivar confidence: Confidence level as a fraction (e.g. ``0.95``).
    :ivar one_sided: ``True`` for a one-sided interval (conformity checks).
    """

    value: float
    standard_uncertainty: float
    coverage_factor: float
    expanded_uncertainty: float
    confidence: float
    one_sided: bool

    @property
    def lower(self) -> float:
        r"""Lower interval bound :math:`y - U` (Formula 3/5)."""
        return self.value - self.expanded_uncertainty

    @property
    def upper(self) -> float:
        r"""Upper interval bound :math:`y + U` (Formula 3/4)."""
        return self.value + self.expanded_uncertainty


# --------------------------------------------------------------------------- #
# Lookups.
# --------------------------------------------------------------------------- #
def _canonical(quantity: str) -> str:
    key = quantity.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _SINGLE:
        valid = ", ".join(sorted(set(_SINGLE) | set(_ALIASES)))
        msg = f"Unknown single-number quantity {quantity!r}. Valid: {valid}."
        raise ValueError(msg)
    return key


def band_uncertainty(
    measurand: Measurand,
    situation: Situation,
    *,
    upper_limit: bool = False,
) -> BandUncertainty:
    """Return the one-third-octave standard uncertainties for a measurand.

    Airborne (Table 2) offers situations A/B/C; impact (Table 4) only B/C; the
    reduction ``ΔL`` (Table 6) only A. ``upper_limit=True`` selects the ``σR95``
    upper limit for airborne, situation A (Annex D Table D.1).

    :param measurand: ``"airborne"``, ``"impact"`` or ``"impact_reduction"``.
    :param situation: Measurement situation ``"A"``, ``"B"`` or ``"C"`` (Clause 5.2).
    :param upper_limit: Select the ``σR95`` upper limit (airborne, situation A).
    :raises ValueError: Unknown measurand, or a situation not tabulated for it.
    """
    try:
        frequencies, columns = _BAND_TABLES[(measurand, upper_limit)]
    except KeyError:
        if upper_limit:
            msg = (
                f"No σR95 upper limit tabulated for measurand {measurand!r} "
                "(only airborne, Annex D)."
            )
            raise ValueError(msg) from None
        valid = ", ".join(sorted({m for m, _ in _BAND_TABLES}))
        msg = f"Unknown measurand {measurand!r}. Valid: {valid}."
        raise ValueError(msg) from None
    if situation not in columns:
        msg = (
            f"Situation {situation!r} is not tabulated for measurand {measurand!r} "
            f"(available: {', '.join(sorted(columns))})."
        )
        raise ValueError(msg)
    return BandUncertainty(
        measurand=measurand,
        situation=situation,
        frequencies=frequencies,
        uncertainties=columns[situation],
        upper_limit=upper_limit,
    )


def single_number_uncertainty(
    quantity: str,
    situation: Situation,
    *,
    upper_limit: bool = False,
) -> float:
    """Return the tabulated single-number standard uncertainty ``u``, in dB.

    Descriptors (case-insensitive, with aliases) cover the ISO 717 ratings:
    ``"r_w"`` (also ``rprime_w``/``dn_w``/``dnt_w``) and its spectrum-adaptation
    variants ``"r_w+c_50_5000"`` etc. (Table 3); ``"ln_w"``/``"ln_w+ci"`` (Table 5);
    ``"delta_lw"`` (Table 7). ``upper_limit=True`` selects the situation-A ``σR95``
    (Annex D Table D.2), defined for airborne descriptors only.

    .. note::
        For the impact descriptors (``"ln_w"``/``"ln_w+ci"``, Table 5) the
        situation-A value is an *estimate*: no reproducibility results are
        available for impact sound insulation (Table 5, footnote a).

    :param quantity: Rating descriptor (see above).
    :param situation: Measurement situation ``"A"``, ``"B"`` or ``"C"`` (Clause 5.2).
    :param upper_limit: Select the ``σR95`` upper limit (airborne, situation A).
    :raises ValueError: Unknown descriptor, an untabulated situation, or an
        ``upper_limit`` request outside airborne/situation A.
    """
    key = _canonical(quantity)
    situations, sigma_r95 = _SINGLE[key]
    if upper_limit:
        if situation != "A":
            msg = "σR95 upper limit is defined for situation A only."
            raise ValueError(msg)
        if sigma_r95 is None:
            msg = f"No σR95 upper limit tabulated for {quantity!r}."
            raise ValueError(msg)
        return sigma_r95
    if situation not in _SITUATION_INDEX:
        msg = f"Unknown situation {situation!r}. Valid: A, B, C."
        raise ValueError(msg)
    value = situations[_SITUATION_INDEX[situation]]
    if value is None:
        msg = f"Situation {situation!r} is not tabulated for descriptor {quantity!r}."
        raise ValueError(msg)
    return value


def maximum_repeatability_standard_deviation() -> BandUncertainty:
    """Return Table 1, the maximum repeatability standard deviation per band (Clause 5.8).

    A laboratory verifies its own procedure when the repeatability standard
    deviation of ``nx`` repeated measurements stays below these values.
    """
    return BandUncertainty(
        measurand="airborne",
        situation="C",
        frequencies=_FREQ_FULL,
        uncertainties=_TABLE1,
    )


# --------------------------------------------------------------------------- #
# Coverage factors and expansion (Clause 8, Table 8).
# --------------------------------------------------------------------------- #
def insulation_coverage_factor(
    confidence: float = 0.95, one_sided: bool = False
) -> float:
    """Return the coverage factor ``k`` for a confidence level (Table 8).

    :param confidence: Confidence level as a fraction. Two-sided values are
        ``0.68, 0.80, 0.90, 0.95, 0.99, 0.999``; one-sided values are
        ``0.84, 0.90, 0.95, 0.975, 0.995, 0.9995``.
    :param one_sided: Use the one-sided column (conformity checks, Formulae 4/5).
    :raises ValueError: Confidence level not tabulated in Table 8.
    """
    table = _COVERAGE_ONE_SIDED if one_sided else _COVERAGE_TWO_SIDED
    for level, k in table.items():
        if abs(level - confidence) < 1e-9:
            return k
    kind = "one-sided" if one_sided else "two-sided"
    valid = ", ".join(f"{level:g}" for level in table)
    msg = (
        f"Confidence level {confidence!r} is not tabulated for the {kind} test. "
        f"Valid: {valid}."
    )
    raise ValueError(msg)


def insulation_expanded_uncertainty(
    u: float,
    coverage: float = 0.95,
    one_sided: bool = False,
) -> float:
    r"""Return the expanded uncertainty :math:`U = k\,u` (Formula 2, Clause 8).

    The coverage factor ``k`` is taken from Table 8 for the requested confidence
    level; a minimum of :math:`k = 1` is enforced (Clause 8).

    :param u: Standard uncertainty ``u``, in dB (must be non-negative).
    :param coverage: Confidence level as a fraction (see :func:`insulation_coverage_factor`).
    :param one_sided: Use the one-sided coverage factor (conformity checks).
    :raises ValueError: Negative ``u`` or an untabulated confidence level.
    """
    if u < 0:
        msg = "Standard uncertainty u must be non-negative."
        raise ValueError(msg)
    k = max(insulation_coverage_factor(coverage, one_sided), 1.0)
    return k * u


def uncertain_value(
    value: float,
    quantity: str,
    situation: Situation,
    *,
    coverage: float = 0.95,
    one_sided: bool = False,
    upper_limit: bool = False,
) -> UncertainValue:
    r"""Attach the ISO 12999-1 expanded uncertainty to a single-number rating.

    Convenience wrapper combining :func:`single_number_uncertainty`,
    :func:`insulation_coverage_factor` and :func:`insulation_expanded_uncertainty` into an
    :class:`UncertainValue` (:math:`y \pm U`) without modifying the rating
    dataclasses. For conformity checks pass ``one_sided=True`` and read
    :attr:`UncertainValue.lower` / :attr:`UncertainValue.upper` (Formulae 4/5).

    :param value: Best estimate ``y`` (e.g. ``Rw`` in dB).
    :param quantity: Rating descriptor (see :func:`single_number_uncertainty`).
    :param situation: Measurement situation ``"A"``, ``"B"`` or ``"C"`` (Clause 5.2).
    :param coverage: Confidence level as a fraction.
    :param one_sided: Use the one-sided coverage factor.
    :param upper_limit: Use the ``σR95`` upper limit (airborne, situation A).
    """
    u = single_number_uncertainty(quantity, situation, upper_limit=upper_limit)
    k = max(insulation_coverage_factor(coverage, one_sided), 1.0)
    return UncertainValue(
        value=value,
        standard_uncertainty=u,
        coverage_factor=k,
        expanded_uncertainty=k * u,
        confidence=coverage,
        one_sided=one_sided,
    )


# --------------------------------------------------------------------------- #
# Combination of uncertainties (Clause 6, Annexes A/B/C).
# --------------------------------------------------------------------------- #
def combine_uncertainties(*components: float) -> float:
    r"""Combine independent standard uncertainties in quadrature (Formula C.2).

    :math:`u_\mathrm{c} = \sqrt{\sum u_i^2}` for uncorrelated contributions with unit
    sensitivity
    coefficients, also the model/reality combination of Formula (A.2).

    :param components: Standard-uncertainty contributions, in dB (non-negative).
    :raises ValueError: No components, or a negative component.
    """
    if not components:
        msg = "At least one uncertainty component is required."
        raise ValueError(msg)
    if any(c < 0 for c in components):
        msg = "Uncertainty components must be non-negative."
        raise ValueError(msg)
    return sqrt(sum(c * c for c in components))


def prediction_input_uncertainty(
    sigma_reproducibility: float,
    sigma_product: float,
    n: int,
) -> float:
    r"""Return the prediction input uncertainty ``u_input`` (Formula A.1).

    .. math::

       u_{\mathrm{input}} =
       \sqrt{\frac{\sigma_\mathrm{R}^2 + \sigma_{\mathrm{product}}^2}{n}
       + \sigma_{\mathrm{product}}^2}

    combines the
    reproducibility standard deviation with the product-homogeneity scatter over
    ``n`` measurements of nominally identical specimens.

    :param sigma_reproducibility: Reproducibility standard deviation ``σR``, in dB.
    :param sigma_product: Product-homogeneity standard deviation ``σ_product``, in dB.
    :param n: Number of measurements of the product (:math:`n \ge 1`).
    :raises ValueError: Non-positive ``n`` or a negative standard deviation.
    """
    if n < 1:
        msg = "n must be a positive integer."
        raise ValueError(msg)
    if sigma_reproducibility < 0 or sigma_product < 0:
        msg = "Standard deviations must be non-negative."
        raise ValueError(msg)
    return sqrt((sigma_reproducibility**2 + sigma_product**2) / n + sigma_product**2)


def reduce_by_independent_measurements(u: float, m: int) -> float:
    r"""Reduce a standard uncertainty by ``m`` independent measurements (Formula A.7).

    :math:`u_{\mathrm{reduced}} = u / \sqrt{m}`: measurements by different
    persons with different
    equipment lower the in-situ uncertainty.

    :param u: Standard uncertainty of a single measurement, in dB (non-negative).
    :param m: Number of independent measurements (:math:`m \ge 1`).
    :raises ValueError: Non-positive ``m`` or negative ``u``.
    """
    if m < 1:
        msg = "m must be a positive integer."
        raise ValueError(msg)
    if u < 0:
        msg = "Standard uncertainty u must be non-negative."
        raise ValueError(msg)
    return u / sqrt(m)


def single_number_uncertainty_uncorrelated(
    band_uncertainties: Sequence[float] | np.ndarray,
    reference_differences: Sequence[float] | np.ndarray,
) -> float:
    r"""Uncorrelated single-number uncertainty from band uncertainties (Formula B.2).

    .. math::

       u(R_\mathrm{w}{+}C) = \sqrt{\sum_i (w_i \, u_i)^2}

    with energy weights

    .. math::

       w_i = \frac{10^{(L_i - R_i)/10}}{\sum_j 10^{(L_j - R_j)/10}}

    derived from the
    reference spectrum. This is the *no-correlation* estimate of Annex B; the
    fully correlated bound (Formulae B.3-B.6) instead re-runs the ISO 717 rating
    and is not reproduced here.

    :param band_uncertainties: Per-band standard uncertainties ``u_i``, in dB.
    :param reference_differences: Per-band :math:`L_i - R_i`
        (reference-spectrum level minus measured band value), in dB.
    :raises ValueError: Mismatched lengths, empty input, or negative ``u_i``.
    """
    u_arr = np.asarray(band_uncertainties, dtype=float)
    d_arr = np.asarray(reference_differences, dtype=float)
    if u_arr.ndim != 1 or d_arr.ndim != 1:
        msg = "Inputs must be one-dimensional sequences."
        raise ValueError(msg)
    if u_arr.size == 0:
        msg = "At least one band is required."
        raise ValueError(msg)
    if u_arr.shape != d_arr.shape:
        msg = "band_uncertainties and reference_differences differ in length."
        raise ValueError(msg)
    if np.any(u_arr < 0):
        msg = "Band uncertainties must be non-negative."
        raise ValueError(msg)
    if not np.all(np.isfinite(u_arr)) or not np.all(np.isfinite(d_arr)):
        msg = (
            "band_uncertainties and reference_differences must contain only "
            "finite values."
        )
        raise ValueError(msg)
    energies = np.power(10.0, d_arr / 10.0)
    weights = energies / energies.sum()
    return float(sqrt(np.sum((weights * u_arr) ** 2)))


# --------------------------------------------------------------------------- #
# Conformity with a requirement (Clause 8, Formulae 4/5).
# --------------------------------------------------------------------------- #
def satisfies_lower_requirement(
    value: float,
    expanded_uncertainty_value: float,
    requirement: float,
) -> bool:
    r"""Test a minimum requirement with one-sided uncertainty (Formula 5).

    Returns ``True`` when
    :math:`\text{value} - U > \text{requirement}`, e.g. an apparent sound
    reduction index ``R'w`` provably exceeds a minimum. ``U`` should be computed
    with the one-sided coverage factor.
    """
    return (value - expanded_uncertainty_value) > requirement


def satisfies_upper_requirement(
    value: float,
    expanded_uncertainty_value: float,
    requirement: float,
) -> bool:
    r"""Test a maximum requirement with one-sided uncertainty (Formula 4).

    Returns ``True`` when
    :math:`\text{value} + U < \text{requirement}`, e.g. a normalized impact
    level ``L'n,w`` provably stays below a maximum. ``U`` should be computed with
    the one-sided coverage factor.
    """
    return (value + expanded_uncertainty_value) < requirement


#: Coverage factors of Table 8 keyed by ``(confidence, one_sided)`` (read-only view).
COVERAGE_FACTORS: Mapping[tuple[float, bool], float] = MappingProxyType(
    {
        **{(level, False): k for level, k in _COVERAGE_TWO_SIDED.items()},
        **{(level, True): k for level, k in _COVERAGE_ONE_SIDED.items()},
    }
)
