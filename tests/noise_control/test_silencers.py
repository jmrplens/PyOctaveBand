#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for reactive silencers (Bies §8.8-8.9, four-pole method).

Oracles: the simple-expansion-chamber closed form ``TL = 10 log10[1 + (1/4)
(m - 1/m)^2 sin^2(kL)]`` (Bies Eq. (8.111)) with its published peak values
(m = 2 -> 1.94, 4 -> 6.55, 8 -> 12.18, 16 -> 18.10 dB), the four-pole TL
reducing to the side-branch closed form (Bies Eq. (8.73)), the quarter-wave
tube tuning ``f = c/(4 l_e)`` (Example 8.1: 56.6 Hz), the Helmholtz resonance
``f_0 = (c/2 pi) sqrt(S/(l_e V))``, matrix reciprocity, the sudden-expansion
limit ``TL = 10 log10[(1 + m)^2/(4 m)]`` of a zero-length element between
unequal port areas (Munjal Eq. (3.27)), and limiting cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from phonometry.noise_control import duct_modes as dm
from phonometry.noise_control import silencers as sl


def _k(f: np.ndarray, c: float = 343.0) -> np.ndarray:
    return 2.0 * np.pi * f / c


def test_expansion_chamber_matches_closed_form() -> None:
    f = np.linspace(20.0, 2000.0, 2000)
    for m in (2.0, 4.0, 8.0, 16.0):
        s_duct, length = 0.01, 0.3
        res = sl.expansion_chamber(f, length, m * s_duct, s_duct)
        cf = 10.0 * np.log10(
            1.0 + 0.25 * (m - 1.0 / m) ** 2 * np.sin(_k(f) * length) ** 2
        )
        assert np.allclose(res.transmission_loss, cf, atol=1e-9)


@pytest.mark.parametrize(
    ("m", "peak"), [(2.0, 1.94), (4.0, 6.55), (8.0, 12.18), (16.0, 18.10)]
)
def test_expansion_chamber_peaks(m: float, peak: float) -> None:
    f = np.linspace(20.0, 2000.0, 6000)
    res = sl.expansion_chamber(f, 0.3, m * 0.01, 0.01)
    assert res.transmission_loss.max() == pytest.approx(peak, abs=0.02)


def test_expansion_chamber_troughs_zero() -> None:
    # TL = 0 at kL = n pi (chamber transparent, no dissipation).
    c, length, s_duct = 343.0, 0.3, 0.01
    f_trough = c / (2.0 * length)  # kL = pi
    res = sl.expansion_chamber(np.array([f_trough]), length, 0.04, s_duct)
    assert res.transmission_loss[0] == pytest.approx(0.0, abs=1e-9)


def test_side_branch_reduces_to_closed_form() -> None:
    # A shunt branch Z_b between equal-area ducts: TL = 20 log10|1 + rho c/(2 Sd Zb)|.
    f = np.linspace(50.0, 500.0, 200)
    zb = sl.quarter_wave_impedance(f, 0.5, 0.002)
    t = sl.shunt_matrix(zb)
    tl = sl.transmission_loss(t, inlet_area=0.01, outlet_area=0.01)
    closed = 20.0 * np.log10(np.abs(1.0 + 1.206 * 343.0 / (2.0 * 0.01 * zb)))
    assert np.allclose(tl, closed, atol=1e-9)


def test_quarter_wave_tuning_frequency() -> None:
    # Bies Example 8.1: l_e = 1.516 m, c = 343.24 -> first peak at 56.6 Hz.
    # Restrict below the third harmonic (169.8 Hz) so the fundamental is the
    # only resonance in range (all odd harmonics peak equally for a QWT).
    f = np.linspace(10.0, 120.0, 8000)
    area = np.pi * 0.05**2 / 4.0
    res = sl.quarter_wave_resonator(f, area, 1.516, area, speed_of_sound=343.24)
    peak_f = f[np.argmax(res.transmission_loss)]
    assert peak_f == pytest.approx(56.6, abs=0.2)
    assert res.resonances[0] == pytest.approx(56.6, abs=0.1)


def test_helmholtz_resonance_frequency() -> None:
    c, s_neck, le, vol = 343.0, 1e-4, 0.02, 1e-3
    res = sl.helmholtz_resonator(np.linspace(50.0, 400.0, 50), 0.01, s_neck, le, vol)
    f0 = c / (2.0 * np.pi) * np.sqrt(s_neck / (le * vol))
    assert res.resonances[0] == pytest.approx(f0)


def test_helmholtz_peak_at_resonance() -> None:
    # Lossless resonator: TL peaks sharply at f_0.
    c, s_neck, le, vol = 343.0, 1e-4, 0.02, 1e-3
    f0 = c / (2.0 * np.pi) * np.sqrt(s_neck / (le * vol))
    f = np.linspace(0.6 * f0, 1.4 * f0, 4000)
    res = sl.helmholtz_resonator(f, 0.01, s_neck, le, vol)
    assert f[np.argmax(res.transmission_loss)] == pytest.approx(f0, rel=0.02)


def test_insertion_loss_identity_is_zero() -> None:
    f = np.linspace(50.0, 500.0, 50)
    ident = sl.duct_matrix(f, 0.0, 0.01)  # zero-length duct = identity
    il = sl.insertion_loss(ident, source_impedance=400.0, radiation_impedance=50.0)
    assert np.allclose(il, 0.0, atol=1e-12)


def test_insertion_loss_equals_tl_for_anechoic_reference() -> None:
    # The insertion loss with Z_s = rho c / S_in and Z_r = rho c / S_out equals
    # the (anechoic) transmission loss; a positive, not negative, quantity.
    c, rho, s = 343.0, 1.206, 0.01
    z = rho * c / s
    f = np.linspace(50.0, 1500.0, 400)
    t = sl.expansion_chamber(f, 0.3, 0.04, s).transfer_matrix
    tl = sl.transmission_loss(t, inlet_area=s, outlet_area=s)
    il = sl.insertion_loss(t, source_impedance=z, radiation_impedance=z)
    assert np.allclose(il, tl, atol=1e-9)
    assert il.max() > 3.0  # a real silencer gives a positive insertion loss


def test_identity_element_is_sudden_expansion() -> None:
    # A zero-length element (T = I) between unequal port areas is a sudden
    # area change, with the classic TL = 10 log10[(1 + m)^2 / (4 m)],
    # m = S_out / S_in: 0.512 dB for m = 2, and identical in both directions
    # (the formula is invariant under m -> 1/m). Pins the Munjal Eq. (3.27)
    # prefactor Zn/Z1; the misprinted Bies Eq. (8.141) gives 1.938 dB here
    # and an inverted Z1/Zn prefactor gives 6.532 dB (see docs/ERRATA.md).
    f = np.linspace(50.0, 500.0, 20)
    ident = sl.duct_matrix(f, 0.0, 0.01)  # zero-length duct = identity
    m = 2.0
    oracle = 10.0 * np.log10((1.0 + m) ** 2 / (4.0 * m))
    assert oracle == pytest.approx(0.512, abs=5e-4)
    expansion = sl.transmission_loss(ident, inlet_area=0.01, outlet_area=0.02)
    contraction = sl.transmission_loss(ident, inlet_area=0.02, outlet_area=0.01)
    assert np.allclose(expansion, oracle, atol=1e-12)
    assert np.allclose(contraction, oracle, atol=1e-12)


def test_transmission_loss_reciprocity_unequal_ports() -> None:
    # A lossless reciprocal two-port must show the same TL from either side.
    # Reversing a two-port with det T = 1 exchanges T11 and T22; the port
    # areas swap with it. Checks a chamber and an asymmetric two-duct cascade
    # between unequal pipes, and that the TL of a passive element never goes
    # negative.
    f = np.linspace(50.0, 1500.0, 300)
    chamber = sl.duct_matrix(f, 0.3, 0.04)
    cascaded = sl.cascade(sl.duct_matrix(f, 0.2, 0.03), sl.duct_matrix(f, 0.15, 0.05))
    for t in (chamber, cascaded):
        t_rev = t.copy()
        t_rev[:, 0, 0], t_rev[:, 1, 1] = t[:, 1, 1].copy(), t[:, 0, 0].copy()
        forward = sl.transmission_loss(t, inlet_area=0.01, outlet_area=0.02)
        reverse = sl.transmission_loss(t_rev, inlet_area=0.02, outlet_area=0.01)
        assert np.allclose(forward, reverse, atol=1e-9)
        assert forward.min() > -1e-9


def test_transfer_matrix_reciprocity() -> None:
    # Reciprocal passive elements have det(T) = 1.
    f = np.linspace(50.0, 500.0, 20)
    t = sl.expansion_chamber(f, 0.3, 0.04, 0.01).transfer_matrix
    dets = np.linalg.det(t)
    assert np.allclose(dets, 1.0, atol=1e-9)


def test_cascade_order() -> None:
    f = np.linspace(50.0, 500.0, 10)
    a = sl.duct_matrix(f, 0.1, 0.01)
    b = sl.duct_matrix(f, 0.2, 0.01)
    # Two concatenated ducts equal one duct of the summed length.
    both = sl.cascade(a, b)
    whole = sl.duct_matrix(f, 0.3, 0.01)
    assert np.allclose(both, whole, atol=1e-9)


def test_extended_tube_reduces_to_expansion_chamber() -> None:
    f = np.linspace(20.0, 2000.0, 500)
    ext = sl.extended_tube_chamber(f, 0.3, 0.04, 0.01)
    plain = sl.expansion_chamber(f, 0.3, 0.04, 0.01)
    assert np.allclose(ext.transmission_loss, plain.transmission_loss, atol=1e-9)


def test_extended_tube_fills_the_trough_its_length_tunes() -> None:
    # An extension is a quarter-wave stub of length L_ext, so it shorts the
    # duct at c / 4 L_ext and fills exactly the chamber trough that lands
    # there: L/2 covers the first (c/2L), L/4 the second (c/L). The extension
    # tuned to the *other* trough leaves this one essentially untouched.
    c, length = 343.0, 0.4
    first, second = c / (2.0 * length), c / length
    f = np.array([first, second])
    plain = sl.expansion_chamber(f, length, 0.04, 0.01)
    half = sl.extended_tube_chamber(f, length, 0.04, 0.01, inlet_extension=length / 2.0)
    quarter = sl.extended_tube_chamber(
        f, length, 0.04, 0.01, inlet_extension=length / 4.0
    )
    assert plain.transmission_loss == pytest.approx([0.0, 0.0], abs=1e-9)
    assert half.transmission_loss[0] > 100.0  # tuned to the first trough
    assert half.transmission_loss[1] < 0.1  # transparent at the second
    assert quarter.transmission_loss[1] > 100.0  # tuned to the second trough
    assert quarter.transmission_loss[0] < 1.0  # nearly nothing at the first


def test_extended_tube_straight_section_excludes_the_extensions() -> None:
    # The junction where each extended pipe ends is where its ducts meet, so
    # the straight chamber element is L - L_a - L_b (Bies Figure 8.19(a) and
    # Example 8.2, L = L_a + L_b + L_c), not the full chamber length. Building
    # the same cascade by hand from the documented building blocks reproduces
    # the result; using the full length does not.
    c, length, la, lb = 343.0, 0.5, 0.25, 0.1
    f = np.linspace(50.0, 900.0, 400)
    s_exp, s_duct = 0.04, 0.01
    result = sl.extended_tube_chamber(
        f,
        length,
        s_exp,
        s_duct,
        inlet_extension=la,
        outlet_extension=lb,
        speed_of_sound=c,
    )

    def by_hand(straight: float) -> np.ndarray:
        annulus = s_exp - s_duct
        return sl.transmission_loss(
            sl.cascade(
                sl.shunt_matrix(
                    sl.quarter_wave_impedance(f, la, annulus, speed_of_sound=c)
                ),
                sl.duct_matrix(f, straight, s_exp, speed_of_sound=c),
                sl.shunt_matrix(
                    sl.quarter_wave_impedance(f, lb, annulus, speed_of_sound=c)
                ),
            ),
            inlet_area=s_duct,
            outlet_area=s_duct,
            speed_of_sound=c,
        )

    assert result.transmission_loss == pytest.approx(
        by_hand(length - la - lb), abs=1e-9
    )
    assert not np.allclose(result.transmission_loss, by_hand(length), atol=1.0)


def test_extended_tube_extensions_may_meet_but_not_overlap() -> None:
    # Extensions that meet leave no straight section: the cascade degenerates
    # to the two annular branches shunting the same plane, which is finite and
    # well defined. Extensions that would overlap are rejected instead of
    # cascading a duct of negative length.
    f = np.array([200.0, 400.0])
    meeting = sl.extended_tube_chamber(
        f, 0.5, 0.04, 0.01, inlet_extension=0.3, outlet_extension=0.2
    )
    assert np.all(np.isfinite(meeting.transmission_loss))
    by_hand = sl.transmission_loss(
        sl.cascade(
            sl.shunt_matrix(sl.quarter_wave_impedance(f, 0.3, 0.03)),
            sl.shunt_matrix(sl.quarter_wave_impedance(f, 0.2, 0.03)),
        ),
        inlet_area=0.01,
        outlet_area=0.01,
    )
    assert meeting.transmission_loss == pytest.approx(by_hand, abs=1e-9)
    with pytest.raises(ValueError, match="negative length"):
        sl.extended_tube_chamber(
            f, 0.5, 0.04, 0.01, inlet_extension=0.3, outlet_extension=0.3
        )


def test_extensions_that_meet_only_in_decimal_are_still_accepted() -> None:
    # 0.1 + 0.2 exceeds 0.3 in binary, so a chamber described in round decimal
    # metres meets the case above without satisfying it in arithmetic. The
    # rejection is of a materially negative straight section, not of an ulp.
    f = np.array([200.0, 400.0])
    assert 0.1 + 0.2 > 0.3  # the premise, so this test cannot quietly go stale
    meeting = sl.extended_tube_chamber(
        f, 0.3, 0.04, 0.01, inlet_extension=0.1, outlet_extension=0.2
    )
    assert np.all(np.isfinite(meeting.transmission_loss))
    # An overlap far below any drawn dimension, and far above the arithmetic,
    # is still an overlap.
    with pytest.raises(ValueError, match="negative length"):
        sl.extended_tube_chamber(
            f, 0.3, 0.04, 0.01, inlet_extension=0.1, outlet_extension=0.2 + 1e-6
        )


def test_insertion_loss_present_when_impedances_given() -> None:
    f = np.linspace(50.0, 500.0, 20)
    res = sl.expansion_chamber(
        f, 0.3, 0.04, 0.01, source_impedance=4e4, radiation_impedance=5e3
    )
    assert res.insertion_loss is not None
    plain = sl.expansion_chamber(f, 0.3, 0.04, 0.01)
    assert plain.insertion_loss is None


def test_validation() -> None:
    with pytest.raises(ValueError, match="'frequencies' must be positive"):
        sl.expansion_chamber([0.0], 0.3, 0.04, 0.01)
    with pytest.raises(ValueError, match="'chamber_area' must be positive"):
        sl.expansion_chamber([100.0], 0.3, -0.04, 0.01)
    with pytest.raises(ValueError, match="must exceed"):
        sl.extended_tube_chamber([100.0], 0.3, 0.01, 0.02)


def test_chain_matches_the_hand_built_cascade() -> None:
    # The chain is the same three calls a hand-built layout makes, so its
    # compound matrix must equal the cascade of the bare matrices exactly.
    f = np.linspace(20.0, 400.0, 200)
    zb = sl.quarter_wave_impedance(f, 0.686, 7.85e-3)
    chain = sl.SilencerChain(f).duct(0.15, 0.0314).shunt(zb).duct(0.60, 0.1257)
    expected = sl.cascade(
        sl.duct_matrix(f, 0.15, 0.0314),
        sl.shunt_matrix(zb),
        sl.duct_matrix(f, 0.60, 0.1257),
    )
    assert np.array_equal(chain.transfer_matrix, expected)
    res = chain.result(inlet_area=0.0314, outlet_area=0.0314)
    assert np.array_equal(
        res.transmission_loss,
        sl.transmission_loss(expected, inlet_area=0.0314, outlet_area=0.0314),
    )
    # The widest section of the chain sets the plane-wave ceiling, exactly as
    # the widest declared area of a named device does.
    assert res.plane_wave_limit == pytest.approx(
        dm.plane_wave_limit(area=0.1257, speed_of_sound=343.0)
    )
    assert res.kind == "element chain"
    assert res.chain is not None


def test_chain_of_one_duct_is_the_expansion_chamber() -> None:
    # A chamber is one duct element between two pipes: the chain reproduces
    # the named constructor to the last bit.
    f = np.linspace(20.0, 800.0, 400)
    chain = sl.SilencerChain(f).duct(0.3, 0.04)
    res = chain.result(inlet_area=0.01, outlet_area=0.01)
    named = sl.expansion_chamber(f, 0.3, 0.04, 0.01)
    assert np.array_equal(res.transmission_loss, named.transmission_loss)


def test_chain_records_only_what_it_was_given() -> None:
    # The asymmetry the drawing rests on: a duct call carries a length and an
    # area, a shunt call carries neither.
    f = np.linspace(20.0, 400.0, 381)
    chain = (
        sl.SilencerChain(f)
        .duct(0.25, 0.0314)
        .shunt(sl.quarter_wave_impedance(f, 0.686, 7.85e-3), label="stub")
    )
    duct, shunt = chain.elements
    assert duct.is_duct
    assert (duct.length, duct.area) == (0.25, 0.0314)
    assert duct.label is None
    assert duct.shorting_frequency is None
    assert not shunt.is_duct
    assert shunt.length is None
    assert shunt.area is None
    assert shunt.label == "stub"
    # The one thing an impedance says about itself: where it is least, which
    # for a lossless quarter-wave tube is its tuning c/(4 l_e) = 125 Hz.
    assert shunt.shorting_frequency == pytest.approx(125.0, abs=1.0)


def test_chain_shunt_accepts_a_constant_and_rejects_a_mismatch() -> None:
    f = np.linspace(20.0, 400.0, 100)
    chain = sl.SilencerChain(f).duct(0.2, 0.01).shunt(1.0e4)
    assert chain.transfer_matrix.shape == (100, 2, 2)
    # A constant impedance has no least value inside the grid.
    assert chain.elements[1].shorting_frequency is None
    empty = sl.SilencerChain(f)
    with pytest.raises(ValueError, match="one value per analysis frequency"):
        empty.shunt(np.ones(7, dtype=np.complex128))


def test_chain_result_is_immune_to_later_additions() -> None:
    f = np.linspace(20.0, 400.0, 50)
    chain = sl.SilencerChain(f).duct(0.2, 0.01)
    res = chain.result(inlet_area=0.01, outlet_area=0.01)
    chain.duct(0.4, 0.02)
    assert res.chain is not None
    assert len(res.chain.elements) == 1
    assert len(chain.elements) == 2


def test_empty_chain_has_no_matrix() -> None:
    with pytest.raises(ValueError, match="at least one matrix"):
        _ = sl.SilencerChain([100.0, 200.0]).transfer_matrix


def test_plot_language_spanish_and_validation() -> None:
    # The reactive-silencer .plot() renderer localises to Spanish and rejects
    # an unknown language code; English remains the unchanged default.
    f = np.linspace(20.0, 2000.0, 500)
    res = sl.expansion_chamber(f, 0.3, 0.04, 0.01)
    ax = res.plot(language="es")
    assert "Frecuencia" in ax.get_xlabel()
    assert "Pérdida" in ax.get_ylabel()
    assert "Silenciador reactivo" in ax.get_title()
    ax_en = res.plot()
    assert ax_en.get_xlabel() == "Frequency [Hz]"
    assert ax_en.get_ylabel() == "Loss [dB]"
    with pytest.raises(ValueError, match="Unknown language"):
        res.plot(language="xx")
