#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the limp-frame equivalent fluid (Allard & Atalla 2e, Sect. 11.3.4).

**Honest statement of the oracle.** Allard & Atalla contains exactly one table
of computed numbers in the whole book (Table 10.2, printed p. 234, Rayleigh
waves in a transversely isotropic frame in vacuum); every prediction-versus-
measurement pair in the porous chapters, Figure 11.2 included, is a *figure*.
There is therefore **no published table of digits** for the limp effective
density, the limp surface impedance or the limp absorption coefficient, here or
anywhere else that has been checked. The model is consequently anchored on:

* the **printed closed forms** themselves, transcribed from Eqs. (11.53)-(11.55)
  (printed pp. 252-253, pdf pp. 258-259: the pdf-to-printed offset is +6 in
  chapter 11, not the +9 of chapter 6 or the +7 of chapter 10). **This is the
  anchor.** ``test_limp_effective_density_matches_the_printed_closed_form``
  restates Eq. (11.55) independently of the implementation, and a human reading
  of the printed page is what stands behind it;
* the two **exact limits the book states in prose** on printed p. 253: that the
  rigid-frame model is recovered "when the frame is heavy", and that
  ``lim_{w->0} rho_limp = rho_t``, the apparent total density, where the
  rigid-frame effective density instead diverges as ``sigma/(j w)``
  (Eq. (5.37)). These are **corroboration, not the anchor**, and they carry
  less weight than their names suggest: the two limits do not constrain the
  ``rho0**2`` and ``2 rho0`` terms at all. A sign-flipped variant of
  Eq. (11.55) satisfies the heavy-frame limit, the ``w -> 0`` limit *and* the
  ``1/rho1`` scaling of the heavy-frame residual, whose analytic value is
  ``-(rho_eq - rho0)**2 / (rho_t + rho_eq - 2 rho0)``. They are kept because
  they would catch a broken composition or a wrong ``rho_t``, not because they
  pin the form;
* the **decoupling frequency** ``Fd = sigma phi^2 / (2 pi rho1)`` printed on
  p. 251 (the same closed form as Eq. (6.90), printed p. 126), evaluated on the
  fully specified glass wool of Table 6.1 (printed p. 124: ``sigma = 40 000``
  N.s/m4, ``phi = 0,94``, ``rho1 = 130`` kg/m3) where pure arithmetic gives
  43,27 Hz;
* the **printed rule-of-thumb thresholds** of printed pp. 253-254: Beranek's
  ``|Kc/Kf| < 0,05``, Doutres et al.'s relaxed ``< 0,2``, and the "bulk modulus
  lower than 20 kPa" that the latter becomes for air.

The material used throughout is the soft fibrous layer of A&A Table 11.2
(printed p. 254), the input set behind Figure 11.2.
"""

from __future__ import annotations

import numpy as np
import pytest

from phonometry import materials

#: A&A Table 11.2 (printed p. 254): soft fibrous material, 50 mm thick.
TABLE_11_2: dict[str, float] = {
    "porosity": 0.98,
    "tortuosity": 1.02,
    "viscous_length": 90e-6,
    "thermal_length": 180e-6,
}
TABLE_11_2_RESISTIVITY = 25.0e3
TABLE_11_2_FRAME_DENSITY = 30.0
TABLE_11_2_THICKNESS = 0.050


def _rigid(frequency: np.ndarray) -> materials.PorousMediumResult:
    return materials.johnson_champoux_allard(
        frequency, TABLE_11_2_RESISTIVITY, **TABLE_11_2
    )


def test_decoupling_frequency_allard_table_6_1() -> None:
    """A&A printed p. 251: ``Fd = sigma phi^2/(2 pi rho1)``.

    On the fully specified glass wool of Table 6.1 (printed p. 124), pure
    arithmetic gives ``40 000 x 0,94^2 / (2 pi x 130) = 43,27`` Hz.
    """
    assert materials.decoupling_frequency(
        40.0e3, porosity=0.94, frame_density=130.0
    ) == pytest.approx(43.27, abs=0.005)


def test_decoupling_frequency_scaling() -> None:
    """The closed form is quadratic in porosity and inverse in frame density."""
    base = materials.decoupling_frequency(
        25.0e3, porosity=0.5, frame_density=30.0
    )
    assert materials.decoupling_frequency(
        25.0e3, porosity=1.0, frame_density=30.0
    ) == pytest.approx(4.0 * base)
    assert materials.decoupling_frequency(
        25.0e3, porosity=0.5, frame_density=60.0
    ) == pytest.approx(0.5 * base)


def test_decoupling_frequency_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        materials.decoupling_frequency(-1.0, porosity=0.98, frame_density=30.0)
    with pytest.raises(ValueError, match="must not exceed 1"):
        materials.decoupling_frequency(
            25.0e3, porosity=1.5, frame_density=30.0
        )


def test_limp_effective_density_matches_the_printed_closed_form() -> None:
    """A&A Eq. (11.55), transcribed here from the printed equation.

    ``rho_limp = (rho_t rho_eq - rho0^2)/(rho_t + rho_eq - 2 rho0)`` with
    ``rho_t = rho1 + phi rho0``.
    """
    f = np.array([50.0, 125.0, 500.0, 2000.0])
    rigid = _rigid(f)
    limp = materials.limp_frame(
        rigid, TABLE_11_2_FRAME_DENSITY, porosity=TABLE_11_2["porosity"]
    )
    rho0 = rigid.air_density
    rho_t = TABLE_11_2_FRAME_DENSITY + TABLE_11_2["porosity"] * rho0
    rho_eq = rigid.effective_density
    expected = (rho_t * rho_eq - rho0**2) / (rho_t + rho_eq - 2.0 * rho0)
    assert np.allclose(limp.effective_density, expected, rtol=1e-12)


def test_limp_frame_low_frequency_limit_is_the_apparent_total_density() -> None:
    """A&A printed p. 253: ``lim_{w->0} rho_limp = rho_t``.

    The rigid-frame effective density diverges as ``sigma/(j w)`` there
    (Eq. (5.37)); the limp one converges on a finite, real
    ``rho_t = rho1 + phi rho0``, which is what allows rigid-body motion of an
    unconstrained sample.
    """
    rigid = _rigid(np.array([1.0e-4]))
    limp = materials.limp_frame(
        rigid, TABLE_11_2_FRAME_DENSITY, porosity=TABLE_11_2["porosity"]
    )
    rho0 = rigid.air_density
    rho_t = TABLE_11_2_FRAME_DENSITY + TABLE_11_2["porosity"] * rho0
    assert float(np.real(limp.effective_density[0])) == pytest.approx(
        rho_t, rel=1e-6
    )
    assert float(np.imag(limp.effective_density[0])) == pytest.approx(0.0, abs=1e-3)
    # The rigid-frame model has no finite limit to compare with.
    assert abs(rigid.effective_density[0]) > 1.0e6 * rho_t


def test_limp_frame_recovers_the_rigid_frame_for_a_heavy_frame() -> None:
    """A&A printed p. 253: "when the frame is heavy, the rigid model is recovered".

    The already-oracled ``johnson_champoux_allard`` is the oracle here. The
    residual falls in proportion to ``1/rho1``, so a thousandfold heavier frame
    must give a thousandfold smaller error.
    """
    f = np.array([50.0, 125.0, 500.0, 2000.0])
    rigid = _rigid(f)
    errors = []
    for rho1 in (1.0e9, 1.0e12):
        limp = materials.limp_frame(
            rigid, rho1, porosity=TABLE_11_2["porosity"]
        )
        errors.append(
            float(np.max(np.abs(limp.effective_density / rigid.effective_density - 1.0)))
        )
    assert errors[1] < 1.0e-5
    assert errors[0] / errors[1] == pytest.approx(1000.0, rel=0.05)
    heavy = materials.limp_frame(
        rigid, 1.0e12, porosity=TABLE_11_2["porosity"]
    )
    assert np.allclose(
        heavy.characteristic_impedance, rigid.characteristic_impedance, rtol=1e-5
    )
    assert np.allclose(heavy.wavenumber, rigid.wavenumber, rtol=1e-5)


def test_limp_frame_keeps_the_bulk_modulus_and_the_equivalent_fluid_relations() -> None:
    """Only the density changes: ``Zc = sqrt(K_e rho_limp)``, ``k = w sqrt(rho_limp/K_e)``."""
    f = np.array([100.0, 400.0, 1600.0])
    rigid = _rigid(f)
    limp = materials.limp_frame(
        rigid, TABLE_11_2_FRAME_DENSITY, porosity=TABLE_11_2["porosity"]
    )
    assert np.array_equal(limp.bulk_modulus, rigid.bulk_modulus)
    omega = 2.0 * np.pi * f
    assert np.allclose(
        limp.characteristic_impedance,
        np.sqrt(limp.bulk_modulus * limp.effective_density),
        rtol=1e-12,
    )
    assert np.allclose(
        limp.wavenumber,
        omega * np.sqrt(limp.effective_density / limp.bulk_modulus),
        rtol=1e-12,
    )
    assert limp.model == "limp_frame(johnson_champoux_allard)"
    assert limp.flow_resistivity == TABLE_11_2_RESISTIVITY


def test_limp_frame_stays_passive() -> None:
    """A passive medium keeps ``Re(Zc) > 0`` and ``Im(k) < 0``."""
    f = np.logspace(0.0, 4.0, 60)
    limp = materials.limp_frame(
        _rigid(f), TABLE_11_2_FRAME_DENSITY, porosity=TABLE_11_2["porosity"]
    )
    assert np.all(np.real(limp.characteristic_impedance) > 0.0)
    assert np.all(np.imag(limp.wavenumber) < 0.0)


def test_limp_and_rigid_converge_well_above_the_decoupling_frequency() -> None:
    """A&A printed p. 253: the two models differ mainly at low frequency.

    For the Table 11.2 material ``Fd = 127`` Hz; a decade above it the two
    effective densities agree to a few per cent, and at ``Fd`` itself they do
    not agree at all.
    """
    fd = materials.decoupling_frequency(
        TABLE_11_2_RESISTIVITY,
        porosity=TABLE_11_2["porosity"],
        frame_density=TABLE_11_2_FRAME_DENSITY,
    )
    assert fd == pytest.approx(127.4, abs=0.1)
    f = np.array([fd, 10.0 * fd, 100.0 * fd])
    rigid = _rigid(f)
    limp = materials.limp_frame(
        rigid, TABLE_11_2_FRAME_DENSITY, porosity=TABLE_11_2["porosity"]
    )
    ratio = np.abs(limp.effective_density / rigid.effective_density - 1.0)
    assert ratio[0] > 0.3
    assert ratio[1] < 0.1
    assert ratio[2] < 0.02


def test_limp_layer_drops_the_low_frequency_absorption_of_the_table_11_2_layer() -> None:
    """The 50 mm Table 11.2 layer, rigidly backed, in the transfer-matrix stack.

    The limp medium is a drop-in ``PorousLayer`` medium. Below the decoupling
    frequency the frame moves with the fluid, so less energy is dissipated and
    ``alpha`` falls; well above it the two predictions coincide.
    """
    f = np.array([100.0, 2000.0, 5000.0])
    rigid = _rigid(f)
    limp = materials.limp_frame(
        rigid, TABLE_11_2_FRAME_DENSITY, porosity=TABLE_11_2["porosity"]
    )
    a_rigid = materials.layered_absorber(
        f, [materials.PorousLayer(TABLE_11_2_THICKNESS, rigid)]
    )
    a_limp = materials.layered_absorber(
        f, [materials.PorousLayer(TABLE_11_2_THICKNESS, limp)]
    )
    assert float(a_limp.absorption[0]) < float(a_rigid.absorption[0]) - 0.02
    assert float(a_limp.absorption[1]) == pytest.approx(
        float(a_rigid.absorption[1]), abs=0.01
    )
    assert float(a_limp.absorption[2]) == pytest.approx(
        float(a_rigid.absorption[2]), abs=0.01
    )


def test_limp_frame_plot_smoke() -> None:
    import matplotlib

    matplotlib.use("Agg")
    limp = materials.limp_frame(
        _rigid(np.array([100.0, 400.0, 1600.0])), TABLE_11_2_FRAME_DENSITY,
        porosity=TABLE_11_2["porosity"],
    )
    ax = limp.plot()
    assert ax.get_xlabel()
    ax_es = limp.plot(language="es")
    assert ax_es.get_xlabel()


def test_limp_frame_rejects_bad_input() -> None:
    rigid = _rigid(np.array([100.0]))
    with pytest.raises(ValueError, match="must be positive"):
        materials.limp_frame(rigid, -1.0)
    with pytest.raises(ValueError, match="must not exceed 1"):
        materials.limp_frame(rigid, 30.0, porosity=1.2)


def test_limp_frame_criteria_match_the_printed_thresholds() -> None:
    """A&A pp. 253-254: 0,05 (Beranek), 0,2 (Doutres), "lower than 20 kPa".

    With ``Kf`` approximated by the isothermal bulk modulus of air,
    ``P0 = 101 325`` Pa, the relaxed criterion admits up to 20,3 kPa, which is
    the book's rounded 20 kPa; Beranek's original admits 5,1 kPa.
    """
    assert materials.LIMP_FRAME_CRITERIA == {"beranek": 0.05, "doutres": 0.2}
    assert materials.limp_frame_applicable(20.0e3)
    assert not materials.limp_frame_applicable(20.5e3)
    assert materials.limp_frame_applicable(5.0e3, criterion="beranek")
    assert not materials.limp_frame_applicable(5.1e3, criterion="beranek")
    # Exactly at the boundary of each printed threshold.
    for name, ratio in materials.LIMP_FRAME_CRITERIA.items():
        assert materials.limp_frame_applicable(
            ratio * 101325.0, criterion=name
        )
        assert not materials.limp_frame_applicable(
            ratio * 101325.0 * (1.0 + 1e-9), criterion=name
        )


def test_limp_frame_applicable_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        materials.limp_frame_applicable(-1.0)
    with pytest.raises(ValueError):
        materials.limp_frame_applicable(1.0e3, criterion="panneton")
