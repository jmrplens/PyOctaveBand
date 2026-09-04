#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the spherical-wave ground effect and advanced barrier diffraction.

Clean-room anchors:
- Attenborough & Van Renterghem, *Predicting Outdoor Sound* 2e (2021):
  Eq. (2.40a-c) Weyl-Van der Pol Q = Rp + (1 - Rp) F(w); Ch. 9 barrier
  diffraction (Eqs. (9.19)-(9.20), MacDonald / Hadden & Pierce).
- Salomons, *Computational Atmospheric Acoustics* (2001): Eq. (3.2)/(3.4)
  two-ray field, Eqs. (D.57)-(D.60) numerical distance / Rp / F(w).
- Bies, Hansen & Howard, *Engineering Noise Control* 5e (2017): Eq. (5.134)
  Fresnel number, Eq. (5.138) Kurze-Anderson, Eq. (5.157) double diffraction.

Primary oracles (analytic limits + published statements):
- hard ground |Z| -> inf: |Q| = 1 exactly, dL -> +6 dB in phase (Salomons 3.4);
- Salomons Fig. D.3 (grassland sigma = 200 kPa s/m2, hs = hr = 2 m, r = 100 m):
  the ground-effect dip reaches about -12.7 dB near 395 Hz;
- sigma -> inf tends to the hard ground; grazing hs, hr -> 0 gives Rp -> -1;
- reciprocity (swap source/receiver heights);
- Kurze-Anderson N -> 0 -> 5 dB (Bies 5.138) and monotone growth in N;
- the exact rigid half-plane gives ~6 dB at the shadow boundary (field halved)
  and tracks Kurze-Anderson within ~1.5 dB in the shadow zone (Bies statement);
- a thick barrier attenuates more than the thin screen of the same height;
- the coherent ground model reduces to the single edge when the ground is hard
  and the geometry removes the ground bounce.
"""

from __future__ import annotations

import dataclasses
import inspect
import warnings

import numpy as np
import pytest

from phonometry import environment
from phonometry.environment.propagation import ground_barriers
from phonometry.fluids import Fluid
from phonometry.materials import delany_bazley

_BANDS = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])


# --------------------------------------------------------------------------- #
# B1. Spherical-wave ground effect
# --------------------------------------------------------------------------- #
def test_hard_ground_reflection_coefficient_is_unity() -> None:
    # |Z| -> inf: Rp -> 1 so (1 - Rp) -> 0 and Q -> 1 regardless of the boundary
    # loss F (the ground wave vanishes) (Attenborough Eq. 2.40c).
    q = environment.spherical_reflection_coefficient(_BANDS, 1e12, 1.0, 1.5, 50.0)
    assert np.allclose(np.abs(q), 1.0, atol=1e-6)


def test_hard_ground_low_frequency_enhancement_is_6_db() -> None:
    # Small path difference + |Q| = 1 -> constructive +6 dB (Salomons Sec. 3.4).
    res = environment.ground_effect([31.5, 63.0], 0.5, 0.5, 200.0, impedance=1e12)
    assert res.excess_attenuation[0] == pytest.approx(6.0, abs=0.2)


@pytest.mark.filterwarnings("ignore::phonometry.PhonometryWarning")
def test_excess_attenuation_stays_near_the_6_db_bound() -> None:
    # Physical anchor: the spherical-wave Q = Rp + (1 - Rp) F(w) is not an
    # energy coefficient; its ground-wave term lets |Q| slightly exceed 1 at
    # low frequency near grazing (the surface-wave contribution, Salomons
    # Sec. 3.4, Attenborough Ch. 2), so dL may top +6 dB by a fraction of a
    # dB. The pre-fix convention (conjugated impedance) suppressed this and
    # pinned |Q| <= 1, which is not the physical behaviour.
    res = environment.ground_effect(_BANDS, 1.0, 2.0, 100.0, flow_resistivity=2e5)
    assert np.all(res.excess_attenuation <= 6.5)
    assert np.all(np.abs(res.reflection_coefficient) <= 1.15)


@pytest.mark.filterwarnings("ignore::phonometry.PhonometryWarning")
def test_miki_ground_is_softer_than_delany_bazley() -> None:
    """The second ground model is reachable, and it is a different ground.

    Miki's revision of Delany-Bazley predicts a lower surface impedance at the
    same flow resistivity: measured across 50 Hz to 5 kHz over a 200 kPa.s/m2
    ground, between 0.58 and 0.96 of it, everywhere. A softer ground moves the
    interference dip down, from 481 Hz to 381 Hz here, which is the whole
    reason to offer the choice.
    """
    freqs = np.logspace(np.log10(50.0), np.log10(5000.0), 60)
    bazley = environment.ground_effect(
        freqs, 1.0, 2.0, 100.0, flow_resistivity=2e5, model="delany_bazley"
    )
    miki = environment.ground_effect(
        freqs, 1.0, 2.0, 100.0, flow_resistivity=2e5, model="miki"
    )
    assert np.all(
        np.abs(miki.normalized_impedance) < np.abs(bazley.normalized_impedance)
    )
    dip_bazley = freqs[np.argmin(bazley.excess_attenuation)]
    dip_miki = freqs[np.argmin(miki.excess_attenuation)]
    assert dip_miki < dip_bazley


def test_sigma_to_infinity_tends_to_hard_ground() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        soft = environment.ground_effect(_BANDS, 1.0, 1.0, 30.0, flow_resistivity=1e12)
    hard = environment.ground_effect(_BANDS, 1.0, 1.0, 30.0, impedance=1e12)
    assert np.allclose(soft.excess_attenuation, hard.excess_attenuation, atol=0.02)


def test_grazing_incidence_plane_coefficient_tends_to_minus_one() -> None:
    # hs, hr -> 0 => cos(theta) -> 0 => Rp -> -1 (Salomons Eq. (D.59)).
    # Im(Z) > 0: a passive ground in the e^{-i omega t} convention.
    q = environment.spherical_reflection_coefficient(
        _BANDS, 12.0 + 6.0j, 1e-4, 1e-4, 100.0
    )
    res = environment.ground_effect(_BANDS, 1e-4, 1e-4, 100.0, impedance=12.0 + 6.0j)
    assert np.all(np.real(res.plane_reflection_coefficient) < -0.9)
    assert q.shape == _BANDS.shape


@pytest.mark.filterwarnings("ignore::phonometry.PhonometryWarning")
def test_ground_effect_reciprocity_in_heights() -> None:
    ab = environment.ground_effect(_BANDS, 0.5, 3.0, 75.0, flow_resistivity=2e5)
    ba = environment.ground_effect(_BANDS, 3.0, 0.5, 75.0, flow_resistivity=2e5)
    assert np.allclose(ab.excess_attenuation, ba.excess_attenuation, atol=1e-9)


def test_grassland_shows_a_ground_dip() -> None:
    freqs = np.linspace(200.0, 2000.0, 400)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = environment.ground_effect(freqs, 1.0, 1.0, 10.0, flow_resistivity=2e5)
    assert res.excess_attenuation.min() < -2.0  # a genuine interference dip


def test_salomons_fig_d3_grassland_dip() -> None:
    # Physical anchor: Salomons, Computational Atmospheric Acoustics, Fig. D.3
    # (grassland sigma = 200 kPa s/m2, hs = hr = 2 m, r = 100 m). The two-ray
    # interference dip reaches about -12.7 dB near 395 Hz. Before the impedance
    # time-convention fix (materials e^{+j omega t} fed unconjugated into the
    # Salomons e^{-i omega t} formulas) the dip almost vanished (-0.55 dB).
    freqs = np.linspace(50.0, 1000.0, 1901)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = environment.ground_effect(freqs, 2.0, 2.0, 100.0, flow_resistivity=2e5)
    i = int(np.argmin(res.excess_attenuation))
    assert res.excess_attenuation[i] == pytest.approx(-12.7, abs=0.3)
    assert 380.0 <= freqs[i] <= 410.0


def test_impedance_from_porous_medium_result_matches_flow_resistivity() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        medium = delany_bazley(_BANDS, 2e5)
        by_result = environment.ground_effect(_BANDS, 1.0, 1.5, 40.0, impedance=medium)
        by_sigma = environment.ground_effect(
            _BANDS, 1.0, 1.5, 40.0, flow_resistivity=2e5
        )
    assert np.allclose(by_result.excess_attenuation, by_sigma.excess_attenuation)


def test_user_impedance_convention_matches_flow_resistivity_path() -> None:
    # A user-supplied array is e^{-i omega t} (Im(Z) > 0 for a passive ground):
    # conjugating the materials' e^{+j omega t} impedance by hand must equal
    # the internal flow_resistivity path exactly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = np.conj(delany_bazley(_BANDS, 2e5).normalized_impedance)
        by_array = environment.ground_effect(_BANDS, 1.0, 1.5, 40.0, impedance=z)
        by_sigma = environment.ground_effect(
            _BANDS, 1.0, 1.5, 40.0, flow_resistivity=2e5
        )
    assert np.all(np.imag(z) > 0.0)  # passive ground in e^{-i omega t}
    assert np.allclose(by_array.excess_attenuation, by_sigma.excess_attenuation)


def test_ground_effect_result_is_frozen_and_has_plot() -> None:
    res = environment.ground_effect(_BANDS, 1.0, 1.0, 20.0, impedance=10.0 + 5.0j)
    assert isinstance(res, environment.SphericalGroundResult)
    assert res.r_reflected > res.r_direct
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.excess_attenuation = np.zeros(1)  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # neither impedance nor flow_resistivity
        {"impedance": 10.0, "flow_resistivity": 2e5},  # both
    ],
)
def test_ground_effect_requires_exactly_one_impedance_source(kwargs: dict) -> None:
    with pytest.raises(
        ValueError, match=r"exactly one of 'impedance' or 'flow_resistivity'"
    ):
        environment.ground_effect(_BANDS, 1.0, 1.0, 20.0, **kwargs)


def test_ground_effect_rejects_bad_geometry() -> None:
    with pytest.raises(ValueError, match="heights must be non-negative"):
        environment.ground_effect(_BANDS, -1.0, 1.0, 20.0, impedance=10.0)
    with pytest.raises(ValueError, match="'distance' must be positive"):
        environment.ground_effect(_BANDS, 1.0, 1.0, 0.0, impedance=10.0)


def test_ground_effect_rejects_non_finite_impedance() -> None:
    # An infinite Z is not the hard-ground limit (that is a large finite Z); it
    # would give inf/inf = NaN in Rp, so it is rejected outright.
    with pytest.raises(ValueError, match=r"'impedance' must be finite"):
        environment.ground_effect(_BANDS, 1.0, 1.0, 20.0, impedance=np.inf)
    inf_z = complex(np.inf, 0.0)
    with pytest.raises(ValueError, match=r"'impedance' must be finite"):
        environment.spherical_reflection_coefficient(_BANDS, inf_z, 1.0, 1.5, 50.0)
    with pytest.raises(ValueError, match=r"'impedance' must be non-zero"):
        environment.ground_effect(_BANDS, 1.0, 1.0, 20.0, impedance=0.0)


def test_impedance_is_a_scalar_or_one_value_per_frequency() -> None:
    # Two admissible forms, and the message names both: a scalar is deliberately
    # broadcast over every frequency, an array must carry one value per band.
    broadcast = environment.ground_effect(_BANDS, 1.0, 1.0, 20.0, impedance=10.0 + 5.0j)
    assert broadcast.normalized_impedance.shape == _BANDS.shape
    with pytest.raises(ValueError, match=r"'impedance' must be a scalar or match"):
        environment.ground_effect(_BANDS, 1.0, 1.0, 20.0, impedance=_BANDS[:3] * 1j)
    with pytest.raises(ValueError, match=r"'impedance' must be a scalar or match"):
        environment.spherical_reflection_coefficient(
            _BANDS, _BANDS[:3] * 1j, 1.0, 1.5, 50.0
        )


def test_unknown_ground_model_raises() -> None:
    with pytest.raises(ValueError, match="unknown ground model"):
        environment.ground_effect(
            _BANDS, 1.0, 1.0, 20.0, flow_resistivity=2e5, model="jca"
        )  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# B4. Barriers: Fresnel number and Kurze-Anderson
# --------------------------------------------------------------------------- #
def test_fresnel_number_formula_and_sign() -> None:
    # N = (2/lambda)(A + B - d); at 343 Hz lambda = 1 m so N = 2 * path difference.
    n = environment.fresnel_number(30.0, 30.0, 59.0, [343.0], speed_of_sound=343.0)
    assert n[0] == pytest.approx(2.0 * (60.0 - 59.0), rel=1e-12)
    # Receiver on the sight line (A + B = d): N = 0.
    assert environment.fresnel_number(30.0, 30.0, 60.0, [500.0])[0] == pytest.approx(
        0.0
    )


def test_kurze_anderson_zero_fresnel_number_is_5_db() -> None:
    assert environment.kurze_anderson_attenuation(0.0) == pytest.approx(5.0, abs=1e-9)


def test_kurze_anderson_is_monotone_in_fresnel_number() -> None:
    n = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    att = environment.kurze_anderson_attenuation(n)
    assert np.all(np.diff(att) > 0.0)
    # Large-N growth ~ 10 lg(N): from N=1 to N=10 gains close to 10 dB.
    assert att[-1] - att[2] == pytest.approx(10.0, abs=1.0)


def test_kurze_anderson_rejects_non_numeric_and_nan() -> None:
    """The Fresnel number is guarded by name: a non-numeric input used to die
    with numpy's own coercion message (or a bare TypeError for a complex one)
    and a NaN came back as a NaN "attenuation" against the documented
    >= 0 dB contract.
    """
    with pytest.raises(ValueError, match="'fresnel_number' must be numeric"):
        environment.kurze_anderson_attenuation("loud")
    with pytest.raises(ValueError, match="'fresnel_number' must be numeric"):
        environment.kurze_anderson_attenuation(1.0 + 2.0j)
    with pytest.raises(ValueError, match="'fresnel_number' must contain only finite"):
        environment.kurze_anderson_attenuation(float("nan"))


def test_kurze_anderson_bright_zone_falls_below_5_db() -> None:
    # Negative Fresnel number (receiver in line of sight) -> less than 5 dB.
    assert environment.kurze_anderson_attenuation(-0.2) < 5.0
    assert environment.kurze_anderson_attenuation(-5.0) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# B4. Barrier insertion loss (exact half-plane, thick, ground-coherent)
# --------------------------------------------------------------------------- #
def test_exact_thin_screen_tracks_kurze_anderson_in_shadow() -> None:
    ka = environment.barrier_insertion_loss(
        _BANDS, 1.0, 50.0, 4.0, 100.0, 1.5, method="kurze_anderson"
    )
    exact = environment.barrier_insertion_loss(
        _BANDS, 1.0, 50.0, 4.0, 100.0, 1.5, method="exact"
    )
    shadow = ka.fresnel_number > 0.3
    assert np.all(
        np.abs(exact.insertion_loss[shadow] - ka.insertion_loss[shadow]) < 2.0
    )


def test_exact_barrier_at_shadow_boundary_is_about_6_db() -> None:
    # Edge just on the source-receiver sight line -> N ~ 0, exact half-plane
    # halves the field (6 dB). Symmetric geometry with a barely-blocking edge.
    freqs = np.array([500.0])
    # Source (0,1) and receiver (100,1): sight line at height 1 m at the barrier.
    il = environment.barrier_insertion_loss(
        freqs, 1.0, 50.0, 1.0 + 1e-3, 100.0, 1.0, method="exact"
    )
    assert il.insertion_loss[0] == pytest.approx(6.0, abs=0.6)


def test_thick_barrier_attenuates_more_than_thin_screen() -> None:
    thin = environment.barrier_insertion_loss(
        _BANDS, 1.0, 50.0, 4.0, 100.0, 1.5, method="exact"
    )
    thick = environment.barrier_insertion_loss(
        _BANDS, 1.0, 50.0, 4.0, 100.0, 1.5, method="exact", thickness=4.0
    )
    # A thick barrier adds a second diffraction; IL is >= the thin screen.
    assert np.all(thick.insertion_loss >= thin.insertion_loss - 1e-6)
    assert thick.insertion_loss.sum() > thin.insertion_loss.sum()


def test_thick_barrier_double_edge_fresnel_number_grows_with_width() -> None:
    thin = environment.barrier_insertion_loss(
        _BANDS, 1.0, 50.0, 4.0, 100.0, 1.5, method="kurze_anderson"
    )
    thick = environment.barrier_insertion_loss(
        _BANDS,
        1.0,
        50.0,
        4.0,
        100.0,
        1.5,
        method="kurze_anderson",
        thickness=6.0,
    )
    assert np.all(thick.fresnel_number > thin.fresnel_number)


@pytest.mark.filterwarnings("ignore::phonometry.PhonometryWarning")
def test_ground_coherent_barrier_differs_from_free_barrier() -> None:
    free = environment.barrier_insertion_loss(
        _BANDS, 1.0, 50.0, 4.0, 100.0, 1.5, method="exact"
    )
    with_ground = environment.barrier_insertion_loss(
        _BANDS,
        1.0,
        50.0,
        4.0,
        100.0,
        1.5,
        method="exact",
        ground_flow_resistivity=2e5,
    )
    assert with_ground.ground is True
    assert free.ground is False
    # The ground adds coherent image paths -> the spectra must differ somewhere.
    assert np.max(np.abs(with_ground.insertion_loss - free.insertion_loss)) > 1.0


def test_ground_coherent_barrier_impedance_paths_agree() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_z = environment.barrier_insertion_loss(
            _BANDS,
            1.0,
            50.0,
            4.0,
            100.0,
            1.5,
            method="exact",
            ground_impedance=delany_bazley(_BANDS, 2e5),
        )
        by_sigma = environment.barrier_insertion_loss(
            _BANDS,
            1.0,
            50.0,
            4.0,
            100.0,
            1.5,
            method="exact",
            ground_flow_resistivity=2e5,
        )
    assert np.allclose(by_z.insertion_loss, by_sigma.insertion_loss)


def test_barrier_reciprocity_source_receiver_swap() -> None:
    fwd = environment.barrier_insertion_loss(
        _BANDS, 1.0, 30.0, 4.0, 100.0, 1.5, method="exact"
    )
    # Swap the roles: source at the old receiver position (distance 100, h 1.5),
    # receiver at the old source (h 1.0). The barrier distance mirrors to 70 m
    # (an asymmetric placement, so the swap genuinely exercises the geometry).
    rev = environment.barrier_insertion_loss(
        _BANDS, 1.5, 70.0, 4.0, 100.0, 1.0, method="exact"
    )
    assert np.allclose(fwd.insertion_loss, rev.insertion_loss, atol=1e-9)


def test_barrier_result_type_and_plot() -> None:
    res = environment.barrier_insertion_loss(_BANDS, 1.0, 50.0, 4.0, 100.0, 1.5)
    assert isinstance(res, environment.BarrierInsertionLoss)
    assert res.method == "exact"
    assert res.insertion_loss.shape == _BANDS.shape


def test_a_barrier_method_the_library_does_not_implement_is_refused() -> None:
    """The fiche cites the method tag as the basis of the prediction.

    Both readers resolved it through a dict of model phrases with the tag
    itself as the silent default, so any string became an accredited claim:
    ``method="ISO 9613-2 Dz"`` printed "predicted with ISO 9613-2 Dz, a
    wave-acoustics complement to the tabulated ISO 9613-2:1996 screening
    term", naming as the applied model the very formula the sheet exists to
    distinguish itself from.
    """
    res = environment.barrier_insertion_loss(_BANDS, 1.0, 50.0, 4.0, 100.0, 1.5)
    with pytest.raises(ValueError, match="'method' must be one of"):
        dataclasses.replace(res, method="ISO 9613-2 Dz")


def test_the_method_docstring_names_every_model_and_counts_them_right() -> None:
    """The ``method`` prose is published verbatim as the API reference row.

    ``scripts/generate_api_docs.py`` copies this ivar into the attributes
    table of the ground-barriers page, so a stale count there is a published
    claim about the library's coverage. Pin the sentence to
    ``_BARRIER_METHODS``: a third model added to the tuple without a rewrite
    of the prose fails here rather than shipping "the two models the library
    implements" over three.
    """
    doc = inspect.getdoc(environment.BarrierInsertionLoss) or ""
    prose = " ".join(doc.split())
    assert len(ground_barriers._BARRIER_METHODS) == 2
    for tag in ground_barriers._BARRIER_METHODS:
        assert f'``"{tag}"``' in prose
    assert "the two models the library implements" in prose


@pytest.mark.parametrize("field_name", ["insertion_loss", "fresnel_number"])
@pytest.mark.parametrize("trim", [True, False], ids=["short", "long"])
def test_a_barrier_series_off_the_frequency_axis_is_refused(
    field_name: str, trim: bool
) -> None:
    """The fiche and the plot both pair these two with the frequency axis.

    One entry too many crashed in the plot as matplotlib's bare "x and y must
    have same first dimension, but have shapes (8,) and (9,)"; one too few
    crashed in the table as "index 7 is out of bounds for axis 0 with size 7".
    Neither names the class or a field.
    """
    res = environment.barrier_insertion_loss(_BANDS, 1.0, 50.0, 4.0, 100.0, 1.5)
    values = np.asarray(getattr(res, field_name))
    wrong = values[:-1] if trim else np.append(values, values[-1])
    count = 7 if trim else 9  # the eight-band fixture, one short or one long
    with pytest.raises(ValueError, match=rf"'{field_name}' \({count}\)"):
        dataclasses.replace(res, **{field_name: wrong})


@pytest.mark.parametrize(
    "field_name", ["frequencies", "insertion_loss", "fresnel_number"]
)
def test_a_non_finite_barrier_value_is_refused(field_name: str) -> None:
    """``np.mean`` computes through a NaN, so it becomes the boxed headline.

    One NaN band rendered a fiche with a ``nan`` table cell under a BOXED
    "Mean insertion loss (63 Hz to 8 kHz) IL = nan dB", and with a declared
    requirement the verdict then died on ``display_round(nan)``. The one
    producer, :func:`barrier_insertion_loss`, computes the loss from a
    geometry and frequency axis it has already pinned finite.
    """
    res = environment.barrier_insertion_loss(_BANDS, 1.0, 50.0, 4.0, 100.0, 1.5)
    values = np.asarray(getattr(res, field_name), dtype=float).copy()
    values[3] = float("nan")
    with pytest.raises(
        ValueError, match=f"BarrierInsertionLoss: '{field_name}' must contain"
    ):
        dataclasses.replace(res, **{field_name: values})


def test_a_barrier_loss_covering_no_frequency_is_refused() -> None:
    """Three empty axes agree, and the plot complains about values there are none of.

    Length-0 axes satisfy every rank and count, so the result was built and
    the fiche asked matplotlib to log-scale an axis with nothing on it, which
    came back as "Data cannot be log-scaled because all values are <= 0" --
    a complaint about values, from a result that has none.
    :func:`barrier_insertion_loss` refuses an empty frequency axis of its
    own, so an empty loss can only be hand-built.
    """
    empty = np.array([], dtype=float)
    with pytest.raises(
        ValueError, match="'frequencies' must carry at least one frequency"
    ):
        environment.BarrierInsertionLoss(
            frequencies=empty,
            insertion_loss=empty,
            fresnel_number=empty,
            method="kurze_anderson",
            ground=False,
        )


def test_barrier_rejects_bad_geometry() -> None:
    with pytest.raises(
        ValueError,
        match=r"'barrier_height' must exceed the source and receiver heights",
    ):
        # barrier not taller than source/receiver -> no shadow.
        environment.barrier_insertion_loss(_BANDS, 2.0, 50.0, 1.5, 100.0, 1.0)
    with pytest.raises(ValueError, match="exceed 'barrier_distance'"):
        environment.barrier_insertion_loss(_BANDS, 1.0, 80.0, 4.0, 50.0, 1.5)
    with pytest.raises(ValueError, match="ground model requires"):
        environment.barrier_insertion_loss(
            _BANDS,
            1.0,
            50.0,
            4.0,
            100.0,
            1.5,
            method="kurze_anderson",
            ground_flow_resistivity=2e5,
        )
    with pytest.raises(ValueError, match=r"unknown method 'utd'"):
        environment.barrier_insertion_loss(
            _BANDS, 1.0, 50.0, 4.0, 100.0, 1.5, method="utd"
        )  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match=r"'receiver_distance' must exceed 'barrier_distance \+ thickness'",
    ):
        # A thick barrier whose far edge lies at/beyond the receiver.
        environment.barrier_insertion_loss(
            _BANDS, 1.0, 50.0, 4.0, 100.0, 1.5, method="exact", thickness=60.0
        )
    # The speed of sound is a property of the fluid now, and the fluid refuses a
    # non-positive one on construction, so the refusal happens before the call.
    with pytest.raises(ValueError, match=r"'properties\['speed_of_sound'\]'"):
        Fluid(
            temperature_c=20.0,
            static_pressure_pa=101_325.0,
            composition={},
            model="an air that does not carry sound",
            validity="",
            properties={"speed_of_sound": 0.0, "density": 1.205},
        )
    for bad in (float("nan"), float("inf"), -1.0):
        # thickness NaN/inf/<=0 are all rejected (NaN slips past a bare <= 0).
        with pytest.raises(ValueError, match=r"'thickness' must be positive"):
            environment.barrier_insertion_loss(
                _BANDS,
                1.0,
                50.0,
                4.0,
                100.0,
                1.5,
                method="exact",
                thickness=bad,
            )


def test_barrier_rejects_non_finite_geometry() -> None:
    """A NaN distance or height used to sail through the ordering comparisons
    (every comparison against NaN is False) and come back from the diffraction
    maths as an all-NaN insertion loss; each of the five is refused by name.
    """
    with pytest.raises(ValueError, match="'barrier_distance' must be positive"):
        environment.barrier_insertion_loss(_BANDS, 2.0, float("nan"), 4.0, 50.0, 1.5)
    with pytest.raises(ValueError, match="'receiver_distance' must be positive"):
        environment.barrier_insertion_loss(_BANDS, 2.0, 1.0, 4.0, float("nan"), 1.5)
    with pytest.raises(ValueError, match="'barrier_height' must be finite"):
        environment.barrier_insertion_loss(_BANDS, 2.0, 1.0, float("nan"), 50.0, 1.5)
    with pytest.raises(ValueError, match="'source_height' must be finite"):
        environment.barrier_insertion_loss(_BANDS, float("nan"), 1.0, 4.0, 50.0, 1.5)
    with pytest.raises(ValueError, match="'receiver_height' must be finite"):
        environment.barrier_insertion_loss(_BANDS, 2.0, 1.0, 4.0, 50.0, float("nan"))
