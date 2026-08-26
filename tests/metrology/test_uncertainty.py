#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for :mod:`phonometry.metrology.uncertainty` (GUM Guide 98-3 and Supplement 1).

Validated against the Guides' own worked examples: the combined uncertainty of
the additive model (Supplement 1 clause 9.2), the coverage factor of the
end-gauge example (GUM Annex H.1 / Table G.2), the Welch-Satterthwaite
effective degrees of freedom (Annex G.4) and the Monte Carlo coverage interval
of the four-rectangular additive model (Supplement 1 Table 3, clause 9.2.3).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from reference_data import GUM_ADDITIVE_UC, GUM_COVERAGE_K99_16, GUM_WELCH_VEFF

from phonometry.metrology import uncertainty as u


def _add4(
    a: float | np.ndarray,
    b: float | np.ndarray,
    c: float | np.ndarray,
    d: float | np.ndarray,
) -> float | np.ndarray:
    return a + b + c + d


def _add2(a: float, b: float) -> float:
    return a + b


def test_additive_model_combined_uncertainty() -> None:
    # Supplement 1 clause 9.2: y = x1+x2+x3+x4, u(xi)=1 -> uc = 2.0.
    qs = [u.Quantity(0.0, 1.0) for _ in range(4)]
    result = u.combine_uncertainty(_add4, qs)
    assert result.value == pytest.approx(0.0)
    assert result.combined_uncertainty == pytest.approx(GUM_ADDITIVE_UC, abs=1e-6)
    np.testing.assert_allclose(result.sensitivities, 1.0, atol=1e-6)


def test_coverage_factor_matches_gum_table() -> None:
    # GUM Annex H.1 / Table G.2: t at p=0.99, v=16 -> k = 2.92.
    assert u.coverage_factor(0.99, 16) == pytest.approx(GUM_COVERAGE_K99_16, abs=5e-3)
    # Large dof approaches the normal quantile.
    assert u.coverage_factor(0.95) == pytest.approx(1.960, abs=1e-3)
    assert u.coverage_factor(0.9545) == pytest.approx(2.0, abs=1e-3)


def test_end_gauge_expanded_uncertainty() -> None:
    # GUM Annex H.1: uc = 32 nm, v_eff = 16 -> U99 = 2.92 * 32 = 93 nm.
    k = u.coverage_factor(0.99, 16)
    assert k * 32.0 == pytest.approx(93.0, abs=1.0)


def test_welch_satterthwaite_effective_dof() -> None:
    # Equal contributions each with dof v -> v_eff = N*v (Annex G.4).
    qs = [u.Quantity(0.0, 1.0, dof=10) for _ in range(4)]
    result = u.combine_uncertainty(_add4, qs)
    assert result.effective_dof == pytest.approx(GUM_WELCH_VEFF, abs=1e-6)


def test_all_type_b_gives_infinite_dof() -> None:
    qs = [u.rectangular(0.0, 1.0), u.triangular(0.0, 1.0)]
    result = u.combine_uncertainty(lambda a, b: a + b, qs)
    assert math.isinf(result.effective_dof)
    _, big = result.expanded(0.95)
    assert big == pytest.approx(1.960 * result.combined_uncertainty, rel=1e-3)


def test_type_b_standard_uncertainties() -> None:
    assert u.rectangular(5.0, 3.0).uncertainty == pytest.approx(3.0 / math.sqrt(3))
    assert u.triangular(5.0, 3.0).uncertainty == pytest.approx(3.0 / math.sqrt(6))
    assert u.u_shaped(5.0, 3.0).uncertainty == pytest.approx(3.0 / math.sqrt(2))


def test_correlation_matrix() -> None:
    # Fully correlated sum: uc = u1 + u2 (not the quadrature sum).
    qs = [u.Quantity(0.0, 1.0), u.Quantity(0.0, 1.0)]
    r = np.array([[1.0, 1.0], [1.0, 1.0]])
    result = u.combine_uncertainty(lambda a, b: a + b, qs, correlation=r)
    assert result.combined_uncertainty == pytest.approx(2.0, abs=1e-6)
    # Uncorrelated for comparison: sqrt(2).
    plain = u.combine_uncertainty(lambda a, b: a + b, qs)
    assert plain.combined_uncertainty == pytest.approx(math.sqrt(2.0), abs=1e-6)


def test_monte_carlo_matches_supplement1_table3() -> None:
    # Supplement 1 Table 3 (clause 9.2.3): four rectangular Xi, mean 0, sd 1
    # -> u(y) = 2.00 and a 95 % symmetric coverage interval ~ [-3.88, 3.88].
    qs = [u.Quantity(0.0, 1.0, "rectangular") for _ in range(4)]
    mc = u.monte_carlo(_add4, qs, trials=2_000_000, coverage=0.95, seed=42)
    assert mc.value == pytest.approx(0.0, abs=0.01)
    assert mc.standard_uncertainty == pytest.approx(2.0, abs=0.01)
    assert mc.interval[0] == pytest.approx(-3.88, abs=0.03)
    assert mc.interval[1] == pytest.approx(3.88, abs=0.03)


def test_monte_carlo_agrees_with_gum_for_linear_gaussian() -> None:
    qs = [u.Quantity(10.0, 0.5), u.Quantity(5.0, 0.3)]
    gum = u.combine_uncertainty(lambda a, b: a - b, qs)
    mc = u.monte_carlo(lambda a, b: a - b, qs, trials=1_000_000, coverage=0.95, seed=1)
    assert mc.value == pytest.approx(gum.value, abs=0.01)
    assert mc.standard_uncertainty == pytest.approx(gum.combined_uncertainty, rel=0.02)


def test_nonlinear_model_sensitivities() -> None:
    # y = a * b: dy/da = b, dy/db = a at the estimates.
    qs = [u.Quantity(3.0, 0.1), u.Quantity(4.0, 0.2)]
    result = u.combine_uncertainty(lambda a, b: a * b, qs)
    assert result.value == pytest.approx(12.0)
    np.testing.assert_allclose(result.sensitivities, [4.0, 3.0], atol=1e-4)
    # uc = sqrt((4*0.1)^2 + (3*0.2)^2) = sqrt(0.16+0.36) = sqrt(0.52).
    assert result.combined_uncertainty == pytest.approx(math.sqrt(0.52), abs=1e-4)


def test_zero_uncertainty_input() -> None:
    # A constant (zero-uncertainty) input contributes nothing but is handled.
    qs = [u.Quantity(2.0, 0.0), u.Quantity(3.0, 0.5)]
    result = u.combine_uncertainty(lambda a, b: a + b, qs)
    assert result.value == pytest.approx(5.0)
    assert result.contributions[0] == pytest.approx(0.0)
    assert result.combined_uncertainty == pytest.approx(0.5, abs=1e-6)


def test_expanded_uncertainty_function() -> None:
    qs = [u.Quantity(0.0, 1.0) for _ in range(4)]
    result = u.combine_uncertainty(_add4, qs)
    k, big = u.expanded_uncertainty(result, 0.95)
    assert k == pytest.approx(1.960, abs=1e-3)
    assert big == pytest.approx(k * 2.0, abs=1e-6)


@pytest.mark.parametrize("dist", ["triangular", "u-shaped", "gaussian"])
def test_monte_carlo_distributions_recover_std(dist: str) -> None:
    # Every PDF is parameterised so that its standard deviation is the given
    # standard uncertainty; the Monte Carlo std must recover it.
    qs = [u.Quantity(0.0, 1.0, dist) for _ in range(2)]
    mc = u.monte_carlo(lambda a, b: a + b, qs, trials=500_000, coverage=0.95, seed=7)
    assert mc.standard_uncertainty == pytest.approx(math.sqrt(2.0), rel=0.03)


@pytest.mark.parametrize("dist", ["gaussian", "rectangular", "triangular", "u-shaped"])
def test_monte_carlo_zero_uncertainty_input(dist: str) -> None:
    # A constant (zero-uncertainty) input must not break the sampler for any
    # PDF (rng.triangular rejects a zero-width support, so it is guarded).
    qs = [u.Quantity(5.0, 0.0, dist), u.Quantity(0.0, 1.0)]
    mc = u.monte_carlo(lambda a, b: a + b, qs, trials=10_000, seed=3)
    assert mc.value == pytest.approx(5.0, abs=0.05)
    assert mc.standard_uncertainty == pytest.approx(1.0, rel=0.05)


def test_monte_carlo_triangular_tiny_uncertainty() -> None:
    # A tiny but non-zero triangular u whose support underflows against mu must
    # not crash rng.triangular (bounds mu +/- a round to the same float).
    qs = [u.Quantity(1.0, 1e-17, "triangular"), u.Quantity(0.0, 1.0)]
    mc = u.monte_carlo(lambda a, b: a + b, qs, trials=5_000, seed=5)
    assert mc.value == pytest.approx(1.0, abs=0.05)
    assert mc.standard_uncertainty == pytest.approx(1.0, rel=0.05)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        u.Quantity(0.0, -1.0)
    with pytest.raises(ValueError, match="distribution"):
        u.Quantity(0.0, 1.0, "weird")
    with pytest.raises(ValueError, match="dof must be positive"):
        u.Quantity(0.0, 1.0, dof=0.0)
    with pytest.raises(ValueError, match="at least one"):
        u.combine_uncertainty(lambda: 0.0, [])
    with pytest.raises(ValueError, match="coverage"):
        u.coverage_factor(1.5)
    pair = [u.Quantity(0, 1)] * 2
    wrong_shape = np.eye(3)
    asymmetric = np.array([[1.0, 0.5], [0.2, 1.0]])
    non_unit_diagonal = np.array([[1.0, 0.0], [0.0, 0.9]])
    # Symmetric, unit diagonal, but indefinite (|r| > 1).
    indefinite = np.array([[1.0, 1.5], [1.5, 1.0]])
    with pytest.raises(ValueError, match="shape"):
        u.combine_uncertainty(_add4, pair, correlation=wrong_shape)
    with pytest.raises(ValueError, match="symmetric"):
        u.combine_uncertainty(_add2, pair, correlation=asymmetric)
    with pytest.raises(ValueError, match="diagonal"):
        u.combine_uncertainty(_add2, pair, correlation=non_unit_diagonal)
    with pytest.raises(ValueError, match="positive semi-definite"):
        u.combine_uncertainty(_add2, pair, correlation=indefinite)
    quantities = [u.Quantity(0, 1)]
    with pytest.raises(ValueError, match="trials"):
        u.monte_carlo(_add4, quantities, trials=0)
    with pytest.raises(ValueError, match="at least 2"):
        # trials=1 used to return NaN (ddof=1) with a raw numpy warning.
        u.monte_carlo(_add4, quantities, trials=1)
    with pytest.raises(ValueError, match="coverage"):
        u.monte_carlo(_add4, quantities, coverage=2.0)
    with pytest.raises(ValueError, match="at least one"):
        u.monte_carlo(_add4, [])


def test_result_fields_and_plot() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    qs = [u.Quantity(3.0, 0.1, name="a"), u.Quantity(4.0, 0.2, name="b")]
    result = u.combine_uncertainty(lambda a, b: a * b, qs)
    assert result.names == ("a", "b")
    assert result.contributions.shape == (2,)
    ax = result.plot()
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_correlated_budget_requires_explicit_coverage_factor() -> None:
    """GUM G.4.1 derives Welch-Satterthwaite for independent inputs only and
    the GUM defines no correlated form: with finite input dof the effective
    dof are undefined (NaN) and expanded() demands an explicit coverage
    factor, instead of a meaningless (possibly ~1e-5) effective dof that
    explodes the coverage factor or an invented normal fallback.
    """
    qs = [u.Quantity(10.0, 0.1, dof=5), u.Quantity(10.0, 0.1, dof=5)]
    r = np.array([[1.0, 0.999], [0.999, 1.0]])
    with pytest.warns(u.UncertaintyWarning, match="Welch-Satterthwaite"):
        result = u.combine_uncertainty(lambda a, b: a - b, qs, correlation=r)
    assert math.isnan(result.effective_dof)
    # uc = sqrt(2*(1-r)) * 0.1 ~ 0.00447; sane (was ~1e149 via the
    # Welch-Satterthwaite pathology).
    assert result.combined_uncertainty == pytest.approx(
        0.1 * math.sqrt(2.0 * (1.0 - 0.999)), rel=1e-6
    )
    with pytest.raises(ValueError, match="coverage_factor_override"):
        result.expanded(0.95)
    k, big = result.expanded(coverage_factor_override=2.0)
    assert k == 2.0
    assert big < 0.02


def test_correlated_budget_with_infinite_dof_stays_normal() -> None:
    """All-infinite input dof: the output is treated as normal regardless of
    correlation, so veff stays infinite and no warning fires.
    """
    qs = [u.Quantity(10.0, 0.1), u.Quantity(10.0, 0.1)]
    r = np.array([[1.0, 0.5], [0.5, 1.0]])
    result = u.combine_uncertainty(lambda a, b: a + b, qs, correlation=r)
    assert math.isinf(result.effective_dof)
    k, _ = result.expanded(0.95)
    assert k == pytest.approx(1.960, abs=1e-3)


def test_sensitivity_step_stays_local_on_nonlinear_model() -> None:
    """The finite-difference step is the input uncertainty with only a
    64-ULP floor: for a large-magnitude input with structure at the
    uncertainty scale, a sqrt(eps)*|x| step (~15 units at 1e9) would probe
    far outside the uncertainty region and bias the sensitivity.
    """
    x0, ux = 1.0e9, 1.0e-3

    def model(x: float) -> float:
        return math.sin((x - 1.0e9) / 1.0e-2)

    result = u.combine_uncertainty(model, [u.Quantity(x0, ux)])
    # d/dx sin((x-x0)/1e-2) at x0 = 100; sensitivity must be ~100, not the
    # aliased near-zero value a 15-unit step returns.
    assert abs(result.sensitivities[0]) == pytest.approx(100.0, rel=0.05)


def test_identity_correlation_keeps_welch_satterthwaite() -> None:
    """An explicit identity matrix is the uncorrelated case: the finite input
    dof still propagate (no fallback, no warning).
    """
    qs = [u.Quantity(0.0, 1.0, dof=10) for _ in range(4)]
    result = u.combine_uncertainty(_add4, qs, correlation=np.eye(4))
    assert result.effective_dof == pytest.approx(40.0, abs=1e-6)


def test_sensitivity_step_survives_tiny_relative_uncertainty() -> None:
    """M10 regression: xi = 1e9 with u = 1e-6 used to underflow the 1e-9
    perturbation below np.spacing(1e9) = 1.2e-7, zeroing both sensitivities
    and reporting uc = 0. The step max(u, sqrt(eps)|x|) recovers the exact
    uc = sqrt(2) * 1e-6.
    """
    qs = [u.Quantity(1e9, 1e-6), u.Quantity(1e9, 1e-6)]
    result = u.combine_uncertainty(lambda a, b: a + b, qs)
    np.testing.assert_allclose(result.sensitivities, 1.0, rtol=1e-6)
    assert result.combined_uncertainty == pytest.approx(math.sqrt(2.0) * 1e-6, rel=1e-6)


def test_flat_direction_warns_but_computes() -> None:
    """A model genuinely flat along one input (here b, multiplied by zero)
    warns that its uncertainty does not propagate, and the rest of the budget
    is unaffected.
    """
    qs = [u.Quantity(3.0, 0.1), u.Quantity(4.0, 0.2)]
    with pytest.warns(u.UncertaintyWarning, match="does not change"):
        result = u.combine_uncertainty(lambda a, b: a + 0.0 * b, qs)
    assert result.contributions[1] == pytest.approx(0.0)
    assert result.combined_uncertainty == pytest.approx(0.1, rel=1e-9)


# ---------------------------------------------------------------------------
# GUM Annex H.1 end-gauge calibration, end to end (published oracle)
# ---------------------------------------------------------------------------
def _h1_budget() -> u.UncertaintyResult:
    from reference_data import GUM_H1_INPUTS

    qs = [u.Quantity(v, unc, dof=dof) for v, unc, dof in GUM_H1_INPUTS]

    def model(
        ls: float,
        d: float,
        alpha_s: float,
        theta: float,
        dalpha: float,
        dtheta: float,
    ) -> float:
        return ls + d - ls * (dalpha * theta + alpha_s * dtheta)

    # alphaS and theta are genuinely flat directions at the H.1 estimates
    # (their sensitivities vanish because dtheta = dalpha = 0): the
    # zero-response warning is expected and documented.
    with pytest.warns(u.UncertaintyWarning, match="does not change"):
        return u.combine_uncertainty(model, qs)


def test_gum_h1_end_to_end_combined_uncertainty() -> None:
    """GUM H.1.4/H.1.6: l = 50 000 838 nm, uc = 32 nm (31.71 unrounded),
    contributions (25, 9.7, 0, 0, 2.9, 16.7) nm, veff = 16.7 (printed
    truncated to 16).
    """
    from reference_data import (
        GUM_H1_CONTRIBUTIONS,
        GUM_H1_UC,
        GUM_H1_VALUE,
        GUM_H1_VEFF,
    )

    result = _h1_budget()
    assert result.value == pytest.approx(GUM_H1_VALUE, abs=0.5)
    assert result.combined_uncertainty == pytest.approx(GUM_H1_UC, abs=0.01)
    np.testing.assert_allclose(result.contributions, GUM_H1_CONTRIBUTIONS, atol=0.05)
    assert result.effective_dof == pytest.approx(GUM_H1_VEFF, abs=0.01)


def test_gum_h1_expanded_uncertainty_99() -> None:
    """GUM H.1.6: U99 = 93 nm at k(0.99, veff=16) = 2.92; interpolating at
    the untruncated veff = 16.66 (G.4.2 NOTE 1) gives 92.1 nm.
    """
    from reference_data import GUM_H1_U99

    result = _h1_budget()
    _k, big = result.expanded(0.99)
    assert big == pytest.approx(GUM_H1_U99, abs=0.1)
    # The printed route: truncate veff to 16 and use Table G.2's k = 2.92.
    k16 = u.coverage_factor(0.99, 16.0)
    assert k16 == pytest.approx(2.9208, abs=5e-4)
    assert k16 * result.combined_uncertainty == pytest.approx(93.0, abs=0.7)


# ---------------------------------------------------------------------------
# GUM Annex H.2 correlated resistance/reactance measurement (the only
# published numeric oracle of the correlated Equation (16) path)
# ---------------------------------------------------------------------------
def test_gum_h2_correlated_measurement() -> None:
    from reference_data import GUM_H2_OBSERVATIONS, GUM_H2_RESULTS

    obs = np.array(GUM_H2_OBSERVATIONS)  # columns: V / V, I / mA, phi / rad
    obs[:, 1] *= 1e-3  # mA -> A
    means = obs.mean(axis=0)
    u_means = obs.std(axis=0, ddof=1) / math.sqrt(obs.shape[0])
    r = np.corrcoef(obs.T)
    # Table H.2 prints the correlations rounded to two decimals.
    assert r[0, 1] == pytest.approx(-0.36, abs=0.005)
    assert r[0, 2] == pytest.approx(0.86, abs=0.005)
    assert r[1, 2] == pytest.approx(-0.65, abs=0.005)

    qs = [u.Quantity(m, s) for m, s in zip(means, u_means, strict=True)]
    models = {
        "R": lambda v, i, p: v / i * math.cos(p),
        "X": lambda v, i, p: v / i * math.sin(p),
        "Z": lambda v, i, p: v / i,
    }
    for name, model in models.items():
        expected_value, expected_uc = GUM_H2_RESULTS[name]
        if name == "Z":
            # Z = V/I ignores phi: the zero-sensitivity warning is expected.
            with pytest.warns(u.UncertaintyWarning, match="does not change"):
                result = u.combine_uncertainty(model, qs, correlation=r)
        else:
            result = u.combine_uncertainty(model, qs, correlation=r)
        assert result.value == pytest.approx(expected_value, abs=5e-3), name
        assert result.combined_uncertainty == pytest.approx(expected_uc, abs=1e-3), name


def test_monte_carlo_matches_supplement1_table2_gaussian() -> None:
    """Supplement 1 Table 2 (clause 9.2.2): four standard Gaussian Xi ->
    u(y) = 2.00 and the 95 % symmetric interval [-3.92, 3.92].
    """
    from reference_data import GUMS1_TABLE2_INTERVAL_95

    qs = [u.Quantity(0.0, 1.0) for _ in range(4)]
    mc = u.monte_carlo(_add4, qs, trials=2_000_000, coverage=0.95, seed=7)
    assert mc.value == pytest.approx(0.0, abs=0.01)
    assert mc.standard_uncertainty == pytest.approx(2.0, abs=0.01)
    assert mc.interval[0] == pytest.approx(-GUMS1_TABLE2_INTERVAL_95, abs=0.03)
    assert mc.interval[1] == pytest.approx(GUMS1_TABLE2_INTERVAL_95, abs=0.03)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("names", ("a",), r"'names'.*one value per input quantity"),
        ("names", ("a", "b", "c"), r"'names'.*one value per input quantity"),
        (
            "sensitivities",
            np.ones(3),
            r"'sensitivities'.*one value per input quantity",
        ),
        ("sensitivities", np.ones((2, 2)), r"'sensitivities' must have one axis"),
        ("contributions", np.ones((2, 2)), r"'contributions' must have one axis"),
    ],
)
def test_budget_rows_must_line_up(field: str, value: object, match: str) -> None:
    """A budget whose rows disagree is refused when built, not when drawn.

    The plot pairs the contributions with the labels and refuses a mismatch
    only from inside matplotlib, and it never reads the sensitivity
    coefficients at all, so an extra or missing coefficient reaches the chart
    unremarked.
    """
    good = u.combine_uncertainty(_add2, [u.Quantity(1.0, 0.1), u.Quantity(2.0, 0.2)])
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(good, **{field: value})


def test_unlabelled_budget_is_not_a_disagreement() -> None:
    """An empty ``names`` is the budget the plot labels ``x1``, ``x2``, ... itself."""
    good = u.combine_uncertainty(_add2, [u.Quantity(1.0, 0.1), u.Quantity(2.0, 0.2)])
    assert dataclasses.replace(good, names=()).names == ()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("value", float("nan"), r"'value' must be finite"),
        (
            "combined_uncertainty",
            float("inf"),
            r"'combined_uncertainty' must be finite",
        ),
        (
            "contributions",
            np.array([0.1, float("nan")]),
            r"'contributions' must contain only finite values",
        ),
        (
            "sensitivities",
            np.array([float("nan"), 1.0]),
            r"'sensitivities' must contain only finite values",
        ),
    ],
)
def test_budget_numbers_must_be_finite(field: str, value: object, match: str) -> None:
    """A NaN number of the budget is drawn as nothing at all.

    The bar of a NaN contribution has width NaN, so matplotlib draws it
    nowhere and the input reads as one that contributes nothing; the
    combined-uncertainty line goes the same way, under a legend printing
    "u_c = nan".
    """
    good = u.combine_uncertainty(_add2, [u.Quantity(1.0, 0.1), u.Quantity(2.0, 0.2)])
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(good, **{field: value})


def test_undefined_effective_dof_is_not_a_non_finite_number() -> None:
    """NaN effective dof is the GUM's own value for a correlated budget.

    Annex G.4 defines no Welch-Satterthwaite form for correlated inputs, so
    the finiteness guard must leave that one field alone and ``expanded``
    must keep asking for an explicit coverage factor there.
    """
    pair = [u.Quantity(0.0, 1.0, dof=5), u.Quantity(0.0, 1.0, dof=5)]
    with pytest.warns(u.UncertaintyWarning, match="Welch-Satterthwaite"):
        correlated = u.combine_uncertainty(
            _add2, pair, correlation=np.array([[1.0, 0.5], [0.5, 1.0]])
        )
    assert math.isnan(correlated.effective_dof)
    k, big = correlated.expanded(coverage_factor_override=2.0)
    assert (k, big) == (2.0, 2.0 * correlated.combined_uncertainty)


def test_quantity_rejects_nan_uncertainty() -> None:
    # NaN passes a bare `< 0.0` and flows into a NaN contribution the budget
    # plot draws as an invisible bar under a legend reading "u_c = nan".
    with pytest.raises(ValueError, match="'uncertainty' must be non-negative"):
        u.Quantity(2.0, float("nan"), name="length")


def test_combine_uncertainty_rejects_model_undefined_at_the_estimates() -> None:
    """The other producer of the NaN contribution the budget bar hides.

    Pinning ``Quantity.uncertainty`` finite closes only one route: a model
    that is not defined at its own input estimates returns NaN there and
    leaves value, combined uncertainty and every contribution undefined.
    """

    def outside_domain(a: float, b: float) -> float:
        return math.sqrt(a - 1.0) + b if a >= 1.0 else math.nan

    inputs = [u.Quantity(0.0, 1.0, name="a"), u.Quantity(2.0, 0.1, name="b")]
    with pytest.raises(
        ValueError, match=r"'model' returned a non-finite output at the input estimates"
    ):
        u.combine_uncertainty(outside_domain, inputs)


def test_combine_uncertainty_rejects_model_undefined_one_step_away() -> None:
    """A model finite at the estimate but not where clause 5.1 probes it.

    ``sqrt`` at an input estimated at zero is finite at the estimate and NaN
    a step below it, so the central difference, that input's contribution and
    the whole budget come out NaN; the chart then draws the input as a bar of
    width NaN, indistinguishable from one contributing nothing. The refusal
    names the input whose probe left the domain.
    """

    def half_domain(a: float, b: float) -> float:
        return math.sqrt(a) + b if a >= 0.0 else math.nan

    inputs = [u.Quantity(0.0, 1.0, name="a"), u.Quantity(2.0, 0.1, name="b")]
    with pytest.raises(ValueError, match=r"non-finite output when probing a by"):
        u.combine_uncertainty(half_domain, inputs)


def test_monte_carlo_rejects_model_with_non_finite_output() -> None:
    """A model undefined over part of the sampled domain poisons every statistic.

    Without the guard the histogram shows only the finite trials under a
    title reading "u(y) = nan" and an invisible coverage band.
    """

    def partial_model(x: np.ndarray) -> np.ndarray:
        return np.where(x < 0.0, np.nan, 0.0)

    with pytest.raises(ValueError, match=r"'model' returned a non-finite output"):
        u.monte_carlo(
            partial_model, [u.Quantity(0.0, 1.0, name="x")], trials=2000, seed=1
        )
