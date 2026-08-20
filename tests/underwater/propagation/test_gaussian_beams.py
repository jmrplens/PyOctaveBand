#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Tests for Gaussian beam tracing.

Every oracle here is built in the test from a published closed form and shares
nothing with the solver.

* **Free space.** A point source in a homogeneous medium radiates
  :math:`e^{ikR}/R` at unit pressure at 1 m. That single comparison pins the
  three ways the calibration of Jensen Eq. (3.92) can go wrong at once: its
  normalisation is the :math:`1/s` of Eq. (3.80) rather than the
  :math:`1/(4\pi s)` of Eq. (3.51), so a spurious :math:`4\pi` would show as a
  flat 21.98 dB; the printed :math:`e^{+i\pi/4}` is a quarter turn out against
  the :math:`q(0) = i\omega W^2/2` of Eq. (3.91), so it would show as
  :math:`\pi/2` of phase; and Eq. (3.88) is written in the conjugate of the time
  convention the rest of the module uses, so a missing conjugation would show as
  the phase running backwards. None of the last two touches the loss, which is
  exactly why the phase is asserted here and not only the magnitude.

* **One reflection.** With the source far from the surface the exact field is
  the two-ray sum, source plus its negative image, which pins the reflection
  coefficient and the impulse the marcher applies to the spreading at a
  boundary.

* **Volume absorption.** Jensen Sect. 3.6.2 perturbs the eikonal with the
  complex sound speed a volume loss implies and attaches
  :math:`e^{-\int_0^s \alpha\,ds'}` to each ray (Eq. 3.116), so in a
  homogeneous medium the absorbed field is the free field times
  :math:`e^{-\alpha R}` with :math:`R` the slant distance. The
  :math:`\alpha` here is transcribed into this file from the published
  formulas (Thorp 1967 as printed by Etter 2003; Ainslie & McColm, JASA 103,
  1998), independently of the ``closed_form`` module the solver reuses, and a
  steep-receiver case separates arc length from range, which no on-axis
  comparison can.

* **The ideal waveguide.** A pressure-release surface and bottom over
  isovelocity water is an image lattice, summed here to convergence. It is the
  worst case for a beam fan, because nothing but :math:`1/R` attenuates the
  steep multiple bounces, so the same image sum truncated to the fan's own
  half-angle is used to separate the fan's truncation from the method's error.

* **The same guide over a lossy seabed.** The image expansion survives an
  absorbing bottom with one change (Jensen Eq. 2.138 for the geometry,
  Eq. 3.126 for what a boundary touch does to amplitude and phase): each
  image's term carries :math:`(-1)^{n_s}\,\mathcal{R}(\theta)^{n_b}` with the
  counts read off the unfolding and :math:`\theta` the one grazing angle the
  straight unfolded path makes with every bottom plane, closed form by
  geometry. The :math:`\mathcal{R}` is ``reflection_coefficient`` called
  directly on those angles, so the oracle shares nothing with the solver's
  wiring, which never sees an image angle: it reads one angle per beam off
  Snell's invariant. One geometry exercises both sides of the critical angle,
  where the physics changes character: above it :math:`|\mathcal{R}| < 1` and
  the seabed drains the steep multiples, below it :math:`|\mathcal{R}| = 1`
  and *only* the phase of :math:`\mathcal{R}` carries the seabed, which is
  what caught the field being assembled with that phase conjugated (15 dB
  wrong at the range the fringes moved most).

* **The same guide, expanded over modes instead of paths.** Jensen Eq. (5.13)
  is a second closed form of that same exact field, built from the boundary
  conditions rather than from a stack of ray paths, and normalised by Eq. (5.16)
  rather than by anything fitted. It reaches the absolute phase, which the image
  lattice is used for level only, and it puts the beams on the footing
  :func:`normal_modes` and :func:`parabolic_equation` are already held to.

* **A refracting guide.** Everything isovelocity leaves the refracting half of
  the marcher untested: :math:`c''` vanishes there, so the coupling coefficient
  of the dynamic ray equations is zero and the boundary impulse with it. The
  :math:`n^2`-linear profile of Eq. (3.77) has exactly linear :math:`k^2(z)`,
  which makes its modes Airy functions in closed form, eigenvalues included, so
  the bent case gets an oracle of the same standard as the straight one.

* **Beam width and wavefront curvature.** In free space Sect. 3.5.1 gives both
  in closed form, Eqs. (3.85)-(3.86), and the second of them is a real
  cross-check rather than a restatement: it is derived there from a complex
  source point, a different argument from the dynamic ray equations the solver
  integrates.

* **The caustic.** Located by driving the *geometric* spreading of Eq. (3.63) to
  zero by bisection on the launch angle, which is a construction the beam solver
  never performs, and which makes the classical amplitude of Eq. (3.65) diverge
  at a point of the field the beams then have to answer for.

* **The branch of the square root**, in two steps, because no single comparison
  reaches it. Eq. (3.58) is linear with real coefficients and Eq. (3.91) starts
  :math:`q` on the imaginary axis with :math:`p` real, so :math:`\mathrm{Re}[q]`
  *is* the geometric spreading and every caustic puts :math:`q` on that axis.
  The unwrapped argument then has to read :math:`-(2k+1)\pi/2` at the
  :math:`k`-th caustic, a closed-form ladder which says :math:`\sqrt{q}` turns
  by :math:`-\pi/2` there: Jensen Eq. (3.79)'s :math:`(-i)^m` recovered instead
  of imposed. That pins the branch; a hand-built single beam handed to
  :func:`_beam_influence` with its argument set one turn below the principal
  value then pins that the sum spends it, since one turn in :math:`\arg q` is a
  factor of -1 in the field.

  It is worth saying what is *not* here, because it was looked for. The branch
  changes the summed field by 1.7 to 3.9 dB, and only at a caustic. That is
  under the beam method's own reproducibility in the only configurations that
  cross one: against the parabolic equation on a caustic cut the two disagree by
  5 dB in the mean, against :func:`normal_modes` by 5.4 dB, and against a
  closed-form two-ray sum with the KMAH factor put in by hand (exact parabolic
  rays for this profile, checked to millimetres) by 4.4 dB, the fringe periods
  agreeing while their positions slip. Doubling the beam width moves the same
  cut by 12 dB. So the two tests above are white box on purpose: a field-value
  oracle sharp enough to see the branch does not exist here.

Two further candidates were tried and rejected, with the measurements that ruled
them out, so that nobody spends the afternoon on them again.

* **Fig. 3.17** (book p. 185) is the obvious target on paper: transmission loss
  for the :math:`n^2`-linear profile at F = 2000 Hz, SD = 1000 m, RD = 800 m
  over 3.000 to 3.100 km, the exact solution against Gaussian beam theory,
  straight through the caustic that Fig. 3.14 spikes on, and every parameter of
  it printed on the figure. Nothing is tabulated anywhere in the chapter,
  though, and the "exact solution" it is drawn against is a Chapter 4 spectral
  integral that would have to be built first. Digitising a printed curve is good
  to perhaps half a decibel and would make the oracle an image, which is not the
  standard the rest of this file holds itself to. The three falsifiable things
  that figure does say are asserted instead, and separately: the level stays
  finite through a caustic where the classical amplitude provably diverges, the
  caustic is located by a construction the solver never performs, and the shadow
  beyond decays gradually rather than off a cliff.

* **Agreement with :func:`normal_modes` and :func:`parabolic_equation`** on a
  refracting range-independent case is the criterion the guide already uses to
  set those two against each other, but as a test here it measures the wrong
  thing. On the 200 m, 50 Hz, 1500 to 1530 m/s case their own trend test uses,
  the beams sat 8.8 dB from the modes and 6.8 dB from the PE, energy-averaged,
  while the two siblings agreed with each other to 2.0 dB on the same cut.
  That gap was the quarter-depth width cap of the era putting :math:`W_0` at
  1.7 wavelengths and nothing else: the cap is retired now (the default there
  is the ten-wavelength floor, 300 m) and the gap goes with it, but the
  comparison still contains no oracle, only three solvers agreeing, and the
  Airy oracle below pins the beams against an exact closed form instead, so
  it is not worth having.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from phonometry import PhonometryWarning
from phonometry._internal.rays import DynamicRays, march_rays
from phonometry.underwater.propagation.numerical import (
    BeamFan,
    FluidSeabed,
    GaussianBeamResult,
    VolumeAbsorption,
    _ocean_ray_derivative,
    gaussian_beams,
    ray_trace,
)

_C = 1500.0


def _free_space_column(max_range: float, max_angle_deg: float = 80.0) -> float:
    """A column deep enough that no beam of the fan can reach a boundary.

    The steepest ray climbs ``max_range * tan(max_angle)``, so putting the
    source in the middle of four times that leaves the whole fan in free water
    for the length of the run and the two boundaries out of the problem.
    """
    return 4.0 * max_range * np.tan(np.radians(max_angle_deg)) + 2000.0


# --- The absolute calibration -----------------------------------------------


def test_the_free_field_is_the_unit_point_source_in_magnitude_and_in_phase() -> None:
    """The beam sum against ``exp(ikR)/R``, which is the whole calibration."""
    f, rmax = 100.0, 2000.0
    depth = _free_space_column(rmax)
    zs = depth / 2.0
    r = np.array([500.0, 1000.0, 1500.0, 2000.0])
    z = zs + np.array([-400.0, -100.0, 0.0, 100.0, 400.0])
    res = gaussian_beams(
        f,
        [0.0, depth],
        [_C, _C],
        source_depth=zs,
        max_range=rmax,
        ranges_m=r,
        receiver_depths_m=z,
        range_step=10.0,
    )
    assert isinstance(res, GaussianBeamResult)
    slant = np.hypot(r[None, :], z[:, None] - zs)
    exact = np.exp(1j * 2.0 * np.pi * f / _C * slant) / slant
    ratio = res.pressure / exact
    # 1.3e-4 in amplitude and 1.8e-3 rad in phase. A missing 4*pi would be a
    # factor of 12.6, the printed e^(+i pi/4) exactly pi/2, and a missing
    # conjugation twice the phase of the ratio, which grows with range.
    assert np.abs(np.abs(ratio) - 1.0).max() < 5e-4
    assert np.abs(np.angle(ratio)).max() < 5e-3
    assert np.abs(res.propagation_loss - 20.0 * np.log10(slant)).max() < 0.01


def test_the_near_field_survives_a_beam_whose_foot_lands_on_the_source() -> None:
    r"""The one place Eq. (3.88) still divides by something unprotected.

    ``r q`` is the Jacobian of Eq. (3.46), range times ray-tube width. Complex
    initial data keeps the second factor off zero, which is the method; nothing
    keeps the *first* off zero, and it vanishes on the axis every ray starts
    from. The geometry that finds it is not exotic: the foot of the
    perpendicular from a receiver lands exactly on the source for whichever beam
    is launched perpendicular to the source-receiver line, so it is built here
    on purpose. A fan of 321 beams over +-80 degrees is spaced 0.5 degrees and
    therefore contains -45.0 exactly, and the receivers sit exactly 45 degrees
    below the source, well inside the fan so its truncation is not in the
    answer.

    Unfloored, the surviving cell had ``r`` at 1.4e-14 m and the field came out
    119 dB too loud, which is a *negative* propagation loss: louder at 71 m than
    at the 1 m the result is normalised to. That is the assertion below, and it
    is the one worth making, because it is the failure that cannot be mistaken
    for physics.
    """
    f, rmax = 100.0, 2000.0
    depth = _free_space_column(rmax)
    zs = depth / 2.0
    offs = np.array([50.0, 100.0, 200.0, 400.0])
    res = gaussian_beams(
        f,
        [0.0, depth],
        [_C, _C],
        source_depth=zs,
        max_range=rmax,
        ranges_m=offs,
        receiver_depths_m=zs + offs,
        fan=BeamFan(max_angle_deg=80.0, n_beams=321),
    )
    # Not asserted as exact equality: the fan is built in radians and read back
    # in degrees, so -45 lands on a rung to within a rounding rather than by
    # construction. A rounding of a degree is close enough to be the same test.
    assert np.abs(res.launch_angles + 45.0).min() < 1e-9, (
        "the singular beam has to be in the fan"
    )
    slant = np.hypot(res.ranges[None, :], res.depths[:, None] - zs)
    err = res.propagation_loss - 20.0 * np.log10(slant)
    assert np.all(np.isfinite(res.propagation_loss))
    # Measured -82.09 dB before the floor, +37.52 after.
    assert res.propagation_loss.min() > 0.0, "no receiver is louder than the source"
    # On the 45-degree diagonal, worst 0.73 dB after and 119 dB before.
    diagonal = np.diag(err)
    assert np.abs(diagonal).max() < 1.5
    # By 400 m the floor is out of the answer entirely: 0.003 dB after, 2.7 before.
    assert abs(diagonal[-1]) < 0.05


def test_one_reflection_is_the_two_ray_field() -> None:
    """Source and its negative image, with the source many beams off the surface.

    The beam's own half-width has to be small against the source depth for this
    to be the two-ray field at all: the fan splits at the grazing angle into
    beams that reflect and beams that do not, so a beam as wide as the source is
    deep gets only half of the launch-angle integral each arrival needs. A
    kilometre of water over a 150 m beam is comfortably clear of that.
    """
    f, depth, zs, zr = 100.0, 20_000.0, 1000.0, 1050.0
    r = np.array([1000.0, 2000.0, 4000.0])
    k = 2.0 * np.pi * f / _C
    direct, image = np.hypot(r, zr - zs), np.hypot(r, zr + zs)
    exact = -20.0 * np.log10(
        np.abs(np.exp(1j * k * direct) / direct - np.exp(1j * k * image) / image)
    )
    res = gaussian_beams(
        f,
        [0.0, depth],
        [_C, _C],
        source_depth=zs,
        max_range=4200.0,
        ranges_m=r,
        receiver_depths_m=np.array([zr]),
        range_step=10.0,
    )
    assert np.abs(res.propagation_loss[0] - exact).max() < 0.05


# --- Volume absorption ------------------------------------------------------
#
# The published formulas, transcribed here from their sources and shared with
# nothing in src/. Both give alpha in dB/km for f in Hz.


def _thorp_1967(f_hz: float) -> float:
    """Thorp (1967) as printed by Etter (2003): dB/kyd in f kHz, per km."""
    f2 = (f_hz / 1000.0) ** 2
    return 1.0936 * (0.1 * f2 / (1.0 + f2) + 40.0 * f2 / (4100.0 + f2))


def _ainslie_mccolm_1998(
    f_hz: float,
    t_c: float,
    s_ppt: float,
    z_km: float,
    ph: float,
) -> float:
    """Ainslie & McColm, JASA 103 (1998), Eqs. (2)-(4): dB/km, f in kHz."""
    f = f_hz / 1000.0
    f1 = 0.78 * np.sqrt(s_ppt / 35.0) * np.exp(t_c / 26.0)
    f2 = 42.0 * np.exp(t_c / 17.0)
    boric = 0.106 * f1 * f**2 / (f**2 + f1**2) * np.exp((ph - 8.0) / 0.56)
    mgso4 = (
        0.52
        * (1.0 + t_c / 43.0)
        * (s_ppt / 35.0)
        * f2
        * f**2
        / (f**2 + f2**2)
        * np.exp(-z_km / 6.0)
    )
    water = 0.00049 * f**2 * np.exp(-(t_c / 27.0 + z_km / 17.0))
    return float(boric + mgso4 + water)


def test_absorption_multiplies_the_free_field_by_exp_minus_alpha_R() -> None:
    r"""The absorbed free field against :math:`20\lg R + \alpha R`, exactly.

    A 4000 m homogeneous column with the source in the middle and a 30 degree
    fan keeps every beam off both boundaries over the 3 km run, so the exact
    field is :math:`e^{ikR - \alpha_e R}/R` with :math:`R` the slant distance
    and :math:`\alpha_e` the loss in nepers/m (Jensen Eq. 3.116 with constant
    :math:`\alpha`): in loss form, spherical spreading plus
    :math:`\alpha R / 1000` with :math:`\alpha` in dB/km. The two alphas are
    the test's own transcriptions above; the environmental arguments handed to
    the solver are repeated in the transcription, the depth being the source
    depth the solver documents it evaluates the coefficient at. Measured
    residuals are at or below 2e-4 dB, so the 0.02 dB bound keeps two orders
    in hand while a wrong measure (range for arc length) or a wrong unit
    (a factor ln(10)/20 = 0.115) would blow through it at every range.
    """
    depth, zs = 4000.0, 2000.0
    r = np.array([1000.0, 2000.0, 3000.0])
    # Off-axis rows stay inside the 30 degree fan at the nearest range
    # (atan(300/1000) = 16.7 degrees); outside it the field is legitimately
    # not illuminated at all.
    dz = np.array([-300.0, 0.0, 300.0])
    cases = [
        ("thorp", 3000.0, _thorp_1967(3000.0)),
        ("thorp", 10_000.0, _thorp_1967(10_000.0)),
        (
            "ainslie-mccolm",
            20_000.0,
            _ainslie_mccolm_1998(20_000.0, 10.0, 35.0, zs / 1000.0, 8.0),
        ),
    ]
    for model, f, alpha in cases:
        res = gaussian_beams(
            f,
            [0.0, depth],
            [_C, _C],
            source_depth=zs,
            max_range=3000.0,
            ranges_m=r,
            receiver_depths_m=zs + dz,
            fan=BeamFan(max_angle_deg=30.0),
            absorption=model,
        )
        assert res.absorption_model == model
        # The recorded coefficient is the same published number, digit for
        # digit, which pins the solver's model wiring as well as its arithmetic.
        assert res.absorption_coefficient == pytest.approx(alpha, rel=1e-12)
        slant = np.hypot(r[None, :], dz[:, None])
        oracle = 20.0 * np.log10(slant) + alpha * slant / 1000.0
        worst = np.abs(res.propagation_loss - oracle).max()
        assert worst < 0.02, f"{model} at {f} Hz: worst {worst:.4f} dB"
        assert alpha * r[-1] / 1000.0 > 0.5, "the loss under test must be material"


def test_absorption_is_charged_along_the_arc_and_not_along_the_range() -> None:
    r"""A steep receiver separates the two measures by the obliquity.

    Jensen Eq. (3.116) integrates the loss along the ray path; the closing
    remark of Sect. 3.6.2 notes that "many ray models" charge
    :math:`\alpha r` over the horizontal range instead. On the axis the two
    coincide, which is why the test above cannot tell them apart. Here the
    receiver sits 1500 m below the source at 500 m of range, a slant of
    1581 m at 71.6 degrees from the horizontal, and Thorp at 10 kHz makes the
    two measures differ by :math:`\alpha (R - r) = 1.24` dB: the assertion
    band of 0.05 dB around the arc-length answer is twenty-five times
    narrower than that gap, so charging the range cannot pass. Thorp on
    purpose, since it ignores every environmental argument and leaves the
    geometry as the only thing under test.
    """
    f, rmax = 10_000.0, 600.0
    depth = _free_space_column(rmax)
    zs = depth / 2.0
    r, dz = 500.0, 1500.0
    alpha = _thorp_1967(f)
    res = gaussian_beams(
        f,
        [0.0, depth],
        [_C, _C],
        source_depth=zs,
        max_range=rmax,
        ranges_m=np.array([r]),
        receiver_depths_m=np.array([zs + dz]),
        range_step=10.0,
        absorption="thorp",
    )
    slant = float(np.hypot(r, dz))
    along_arc = 20.0 * np.log10(slant) + alpha * slant / 1000.0
    along_range = 20.0 * np.log10(slant) + alpha * r / 1000.0
    got = float(res.propagation_loss[0, 0])
    assert abs(got - along_arc) < 0.05
    assert got - along_range > 1.0, "the range measure must be far outside the band"


def test_absorption_is_off_by_default_and_bit_for_bit_identical() -> None:
    """OFF is the default and leaves every bit of the field where it was.

    The published validation numbers of this module were all measured without
    absorption, so the default has to reproduce them exactly, not merely
    closely; and the environmental arguments must be inert while the model is
    ``None``, which the deliberately absurd values below would betray at the
    first floating-point operation that touched them.
    """
    f, rmax = 100.0, 1500.0
    depth = _free_space_column(rmax, 30.0)
    zs = depth / 2.0
    kwargs = {
        "source_depth": zs,
        "max_range": rmax,
        "ranges_m": np.array([500.0, 1500.0]),
        "receiver_depths_m": np.array([zs - 200.0, zs + 350.0]),
        "fan": BeamFan(max_angle_deg=30.0),
    }
    default = gaussian_beams(f, [0.0, depth], [_C, _C], **kwargs)
    explicit = gaussian_beams(f, [0.0, depth], [_C, _C], **kwargs, absorption=None)
    assert default.absorption_model is None
    assert default.absorption_coefficient == 0.0
    assert np.array_equal(default.pressure, explicit.pressure)
    withit = gaussian_beams(
        f, [0.0, depth], [_C, _C], **kwargs, absorption="francois-garrison"
    )
    assert withit.absorption_coefficient > 0.0
    assert not np.array_equal(default.pressure, withit.pressure)


def test_absorption_defaults_route_to_francois_garrison_at_the_source() -> None:
    """The advertised seam: the same model names, defaults and machinery as
    ``seawater_absorption``, evaluated at the source depth.

    This is wiring, not an oracle: Francois-Garrison itself is validated
    digit for digit against its published tables in the closed-form tests,
    and the two transcription oracles above own the field-level assertion.
    What is pinned here is that the beams apply exactly the coefficient they
    record, and that the recorded one is ``seawater_absorption`` evaluated
    with the same arguments the docstring names.
    """
    from phonometry.underwater.propagation.closed_form import seawater_absorption

    depth, zs = 4000.0, 2000.0
    f = 10_000.0
    r = np.array([1000.0, 2500.0])
    res = gaussian_beams(
        f,
        [0.0, depth],
        [_C, _C],
        source_depth=zs,
        max_range=2500.0,
        ranges_m=r,
        receiver_depths_m=np.array([zs]),
        fan=BeamFan(max_angle_deg=30.0),
        absorption=VolumeAbsorption(
            "francois-garrison", temperature=4.0, salinity=34.0, ph=7.9
        ),
    )
    expected = float(
        seawater_absorption(
            f,
            temperature=4.0,
            salinity=34.0,
            depth=zs,
            ph=7.9,
            model="francois-garrison",
        )[0]
    )
    assert res.absorption_coefficient == expected
    off = gaussian_beams(
        f,
        [0.0, depth],
        [_C, _C],
        source_depth=zs,
        max_range=2500.0,
        ranges_m=r,
        receiver_depths_m=np.array([zs]),
        fan=BeamFan(max_angle_deg=30.0),
    )
    added = res.propagation_loss[0] - off.propagation_loss[0]
    assert np.abs(added - expected * r / 1000.0).max() < 1e-3


# --- The ideal waveguide ----------------------------------------------------
#
# Images of a point source between two pressure-release planes sit at
# 2 m D + z_s with strength +1 and at 2 m D - z_s with strength -1, for every
# integer m: the pair at m = 0 cancels at z = 0 and the whole lattice cancels at
# z = D. The sum converges slowly (each term falls off only as 1/R and it is the
# alternating signs that close it), hence the term count below.

_GUIDE = {
    "water_depth": 1000.0,
    "frequency": 300.0,
    "source_depth": 300.0,
    "receiver_depth": 600.0,
}


def _image_source_loss(
    ranges: np.ndarray,
    *,
    water_depth: float,
    frequency: float,
    source_depth: float,
    receiver_depth: float,
    max_angle_deg: float | None = None,
    n_images: int = 40_000,
) -> np.ndarray:
    """Propagation loss of the ideal guide, optionally cut to a fan half-angle."""
    k = 2.0 * np.pi * frequency / _C
    m = np.arange(-n_images, n_images + 1)[:, None]
    rr = np.asarray(ranges, dtype=np.float64)[None, :]
    up = receiver_depth - (2.0 * m * water_depth + source_depth)
    down = receiver_depth - (2.0 * m * water_depth - source_depth)
    r_up, r_down = np.hypot(rr, up), np.hypot(rr, down)
    take_up = take_down = 1.0
    if max_angle_deg is not None:
        slope = np.tan(np.radians(max_angle_deg))
        take_up = np.abs(up) < slope * rr
        take_down = np.abs(down) < slope * rr
    field = (
        np.exp(1j * k * r_up) / r_up * take_up
        - np.exp(1j * k * r_down) / r_down * take_down
    ).sum(axis=0)
    return np.asarray(-20.0 * np.log10(np.abs(field)))


def test_the_ideal_waveguide_matches_the_image_source_sum() -> None:
    """With the fan opened wide enough, to a hundredth of a decibel.

    Everything is in this one number: the reflection coefficients of both
    boundaries, the impulse the spreading takes at each of them, the branch of
    the square root through a path that has bounced dozens of times, and the
    ladder of folded receiver images without which the answer is several
    decibels light.

    The bound is the measured 0.00024 dB doubled, not a loose ceiling, and
    deliberately so: the perfect reflector is now the inert limit of the lossy
    seabed machinery, and this is the assertion that the wiring for it left
    the default exactly where it always was.
    """
    r = np.array([2000.0])
    exact = _image_source_loss(r, **_GUIDE)
    res = gaussian_beams(
        _GUIDE["frequency"],
        [0.0, _GUIDE["water_depth"]],
        [_C, _C],
        source_depth=_GUIDE["source_depth"],
        max_range=2200.0,
        ranges_m=r,
        receiver_depths_m=np.array([_GUIDE["receiver_depth"]]),
        fan=BeamFan(max_angle_deg=88.0, beam_width=100.0),
        range_step=2.0,
    )
    assert np.abs(res.propagation_loss[0] - exact).max() < 5e-4


def test_a_narrower_fan_costs_exactly_what_cutting_the_image_sum_costs() -> None:
    """The 80-degree error is the fan's truncation, not the method's.

    A guide with two perfect reflectors is the worst case a beam fan can be
    asked for, because nothing but 1/R attenuates the steep multiple bounces and
    the image lattice never dies out. Cutting the *oracle* to the same half
    angle shows where the difference comes from: the solver tracks the truncated
    lattice far more closely than either tracks the complete one.
    """
    r = np.array([2000.0])
    complete = _image_source_loss(r, **_GUIDE)
    truncated = _image_source_loss(r, max_angle_deg=80.0, **_GUIDE)
    res = gaussian_beams(
        _GUIDE["frequency"],
        [0.0, _GUIDE["water_depth"]],
        [_C, _C],
        source_depth=_GUIDE["source_depth"],
        max_range=2200.0,
        ranges_m=r,
        receiver_depths_m=np.array([_GUIDE["receiver_depth"]]),
        fan=BeamFan(max_angle_deg=80.0, beam_width=100.0),
        range_step=10.0,
    )
    missed = float(np.abs(truncated - complete).max())
    assert missed > 0.1, "the fan has to bite for this test to mean anything"
    assert np.abs(res.propagation_loss[0] - truncated).max() < 0.5 * missed


@pytest.mark.parametrize(
    ("bottom", "at_bottom"), [("pressure-release", 0.0), ("rigid", 2.0)]
)
def test_both_boundary_conditions_fall_out_of_the_folded_images(
    bottom: str,
    at_bottom: float,
) -> None:
    """Nothing imposes them; they are what the image ladder sums to.

    At the surface the two families of images coincide with opposite signs and
    cancel whatever the bottom is. At a rigid bottom they coincide with equal
    signs, so the field doubles against its mid-column scale and its depth
    derivative cancels, which is the Neumann condition; at a pressure-release
    bottom they cancel there too.
    """
    depth = _GUIDE["water_depth"]
    res = gaussian_beams(
        _GUIDE["frequency"],
        [0.0, depth],
        [_C, _C],
        source_depth=_GUIDE["source_depth"],
        max_range=2200.0,
        ranges_m=np.array([2000.0]),
        receiver_depths_m=np.array([0.0, 0.5 * depth, depth]),
        fan=BeamFan(max_angle_deg=85.0, beam_width=100.0),
        range_step=5.0,
        bottom=bottom,
    )
    scale = np.abs(res.pressure[1, 0])
    assert np.abs(res.pressure[0, 0]) / scale < 1e-3
    assert np.abs(res.pressure[2, 0]) / scale == pytest.approx(at_bottom, abs=0.2)


# --- A lossy seabed ---------------------------------------------------------
#
# The image lattice above survives an absorbing bottom with one change: each
# image's term carries the coefficients of the boundary planes standing
# between it and the receiver. Jensen Eq. (2.138) writes the lattice as four
# families per period m, and reading the counts off Fig. 2.17's unfolding
# gives, for the family offsets in depth,
#
#     2Dm - zs + z:      m surface touches,   m bottom touches   (direct at m=0)
#     2D(m+1) - zs - z:  m,                   m+1                (bottom first)
#     2Dm + zs + z:      m+1,                 m                  (surface first)
#     2D(m+1) + zs - z:  m+1,                 m+1
#
# which reproduces the book's signs for the ideal guide, (-1)^(touches), and
# its naming of the first four terms (direct, bottom, surface, bottom-surface).
# Checked against the two-sided lattice `_image_source_loss` sums, the four
# families with R = -1 agree to 6e-10 dB. A boundary touch multiplies the
# amplitude by |R| and adds arg R to the phase (Eq. 3.126), and in isovelocity
# water every image path is one straight line, so all n_b of its bottom
# touches share the single grazing angle that line makes with the horizontal:
# atan2(|offset|, r), closed form by geometry, and the term's coefficient is
# exactly (-1)^(n_s) R(theta)^(n_b).

_SEABED = {"rho1": 1000.0, "rho2": 1800.0, "c2": 1700.0}  # sand-like, phi_c = 28.07 deg


def _lossy_image_loss(
    ranges: np.ndarray,
    *,
    water_depth: float,
    frequency: float,
    source_depth: float,
    receiver_depth: float,
    rho1: float,
    rho2: float,
    c2: float,
    n_images: int = 400,
    magnitude_only: bool = False,
) -> np.ndarray:
    """The lossy guide's loss as the image sum described above.

    ``magnitude_only`` replaces every R by |R|, which below the critical angle
    is exactly the perfect reflector each such touch would be if the phase of
    R meant nothing: the degraded oracle the phase assertion needs.

    Convergence is far faster than the ideal lattice's: above the critical
    angle |R| < 1 caps every steep family geometrically (0.342 at normal
    incidence for this sand), so 400 periods is already indistinguishable
    from 1600 at the printed precision.
    """
    from phonometry.underwater.propagation.seabed_reflection import (
        reflection_coefficient,
    )

    k = 2.0 * np.pi * frequency / _C
    r = np.asarray(ranges, dtype=np.float64)
    field = np.zeros(r.size, dtype=np.complex128)
    for m in range(n_images):
        families = [
            (2.0 * water_depth * m - source_depth + receiver_depth, m, m),
            (2.0 * water_depth * (m + 1) - source_depth - receiver_depth, m, m + 1),
            (2.0 * water_depth * m + source_depth + receiver_depth, m + 1, m),
            (2.0 * water_depth * (m + 1) + source_depth - receiver_depth, m + 1, m + 1),
        ]
        for offset, n_surface, n_bottom in families:
            distance = np.hypot(r, offset)
            grazing = np.degrees(np.arctan2(abs(offset), r))
            refl = reflection_coefficient(grazing, rho1=rho1, c1=_C, rho2=rho2, c2=c2)
            if magnitude_only:
                refl = np.abs(refl)
            field = field + (
                (-1.0) ** n_surface
                * refl**n_bottom
                * np.exp(1j * k * distance)
                / distance
            )
    return np.asarray(-20.0 * np.log10(np.abs(field)))


def test_a_lossy_seabed_is_the_image_sum_with_R_at_each_images_own_angle() -> None:
    r"""Above and below the critical angle at once, phase included.

    The three ranges are chosen for what their dominant images do. At 2 km
    every bottom-touching image lies *above* the 28.07 degree critical angle
    (the shallowest touches the bottom at 28.8 degrees), so the comparison
    there pins the magnitude of R and nothing else; replacing R by |R| moves
    the oracle by exactly zero at that range. At 4 km the two dominant
    bottom-touching images have slid *below* critical (15.4 and 23.0 degrees),
    where |R| = 1 and the seabed survives only in the phase of R, and
    replacing R by |R| now moves the oracle by 5.7 dB: that gap is the
    fringes moving, it is asserted as material below, and the solver has to
    land on the true side of it. This is also the configuration that caught
    the phase being spent backwards: the beam sum is assembled in the
    conjugate time convention and conjugated once at the end, so an
    un-conjugated R survives to the exposed field as conj(R), which measured
    15.3 dB wrong at this 4 km point while leaving every real-R
    configuration, which is all the other tests, untouched.

    Measured agreement with the settings below: 0.073, 0.042 and 0.019 dB at
    2, 3 and 4 km. The 2 km error is the largest for a reason worth recording:
    its shallowest bottom image sits 0.74 degrees above the critical kink,
    where R varies fastest, and a beam reconstructs each arrival from a cone
    of launch angles about a wavelength/(pi W0) = 0.9 degrees wide, so the
    kink is smeared across the very arrival it matters most to; widening the
    beams to 150 m narrows the cone and halves that error (0.031 dB), which
    is what says it is angular smear and not a wrong coefficient. The 0.15 dB
    bound keeps a hand's margin over all three while sitting fifty times
    under the 8 dB the ideal guide differs by and far under either failure
    the test exists to catch.
    """
    from phonometry.underwater.propagation.seabed_reflection import (
        critical_angle,
        reflection_coefficient,
    )

    r = np.array([2000.0, 3000.0, 4000.0])
    exact = _lossy_image_loss(r, **_GUIDE, **_SEABED)
    res = gaussian_beams(
        _GUIDE["frequency"],
        [0.0, _GUIDE["water_depth"]],
        [_C, _C],
        source_depth=_GUIDE["source_depth"],
        max_range=4200.0,
        ranges_m=r,
        receiver_depths_m=np.array([_GUIDE["receiver_depth"]]),
        fan=BeamFan(max_angle_deg=85.0, beam_width=100.0),
        range_step=2.5,
        bottom=FluidSeabed(
            density=_SEABED["rho2"],
            sound_speed=_SEABED["c2"],
            water_density=_SEABED["rho1"],
        ),
    )
    assert np.abs(res.propagation_loss[0] - exact).max() < 0.15
    assert res.seabed_density == _SEABED["rho2"]
    assert res.seabed_sound_speed == _SEABED["c2"]

    # The geometry really does straddle the critical angle: the shallowest
    # bottom-touching image at 2 km is above it with |R| < 1, and at 4 km the
    # dominant one is below it with |R| = 1 and a phase that means something.
    phi_c = critical_angle(_C, _SEABED["c2"])
    depth, zs, zr = (
        _GUIDE["water_depth"],
        _GUIDE["source_depth"],
        _GUIDE["receiver_depth"],
    )
    shallowest_touch = np.degrees(np.arctan2(2.0 * depth - zs - zr, r))
    assert shallowest_touch[0] > phi_c > shallowest_touch[-1]
    r_above = reflection_coefficient(
        float(shallowest_touch[0]),
        rho1=_SEABED["rho1"],
        c1=_C,
        rho2=_SEABED["rho2"],
        c2=_SEABED["c2"],
    )
    r_below = reflection_coefficient(
        float(shallowest_touch[-1]),
        rho1=_SEABED["rho1"],
        c1=_C,
        rho2=_SEABED["rho2"],
        c2=_SEABED["c2"],
    )
    assert np.abs(r_above[0]) < 0.999
    assert np.abs(r_below[0]) == pytest.approx(1.0, abs=1e-12)
    assert abs(np.angle(r_below[0])) > 0.3

    # The phase of R is load-bearing, not decorative: strip it from the oracle
    # and the 4 km point moves by decibels, while the solver stays with the
    # phase-true sum. |R| = 1 below critical means the stripped oracle is what
    # a solver that dropped, or conjugated, the phase would be chasing.
    stripped = _lossy_image_loss(r, **_GUIDE, **_SEABED, magnitude_only=True)
    assert abs(stripped[-1] - exact[-1]) > 3.0
    assert abs(res.propagation_loss[0, -1] - exact[-1]) < 0.1

    # And the seabed is material full stop: the ideal guide is decibels away.
    ideal = _image_source_loss(r, **_GUIDE)
    assert np.abs(exact - ideal).max() > 2.0


def test_the_perfect_default_is_the_seabed_machinery_off_state() -> None:
    """The perfect reflector is the seabed machinery's off state, bit for bit.

    Every published validation number of this module was measured over a
    perfect bottom, so the default has to reproduce it exactly, not merely
    closely, and spelling the default out must change nothing.
    """
    r = np.array([2000.0])
    kwargs = {
        "source_depth": _GUIDE["source_depth"],
        "max_range": 2200.0,
        "ranges_m": r,
        "receiver_depths_m": np.array([_GUIDE["receiver_depth"]]),
        "fan": BeamFan(max_angle_deg=85.0, beam_width=100.0),
        "range_step": 10.0,
    }
    default = gaussian_beams(
        _GUIDE["frequency"], [0.0, _GUIDE["water_depth"]], [_C, _C], **kwargs
    )
    explicit = gaussian_beams(
        _GUIDE["frequency"],
        [0.0, _GUIDE["water_depth"]],
        [_C, _C],
        **kwargs,
        bottom="pressure-release",
    )
    assert default.seabed_density is None
    assert default.seabed_sound_speed is None
    assert np.array_equal(default.pressure, explicit.pressure)
    lossy = gaussian_beams(
        _GUIDE["frequency"],
        [0.0, _GUIDE["water_depth"]],
        [_C, _C],
        **kwargs,
        bottom=FluidSeabed(density=_SEABED["rho2"], sound_speed=_SEABED["c2"]),
    )
    assert not np.array_equal(default.pressure, lossy.pressure)


def test_a_malformed_seabed_is_rejected() -> None:
    """A ``FluidSeabed`` cannot half-arrive, so what is left to reject is a
    nonphysical one, each field with its own message."""
    iso = ([0.0, 1000.0], [_C, _C])
    airy = FluidSeabed(density=-1.0, sound_speed=1700.0)
    still = FluidSeabed(density=1800.0, sound_speed=0.0)
    dry = FluidSeabed(density=1800.0, sound_speed=1700.0, water_density=0.0)
    with pytest.raises(ValueError, match="density"):
        gaussian_beams(200.0, *iso, source_depth=500.0, bottom=airy)
    with pytest.raises(ValueError, match="sound_speed"):
        gaussian_beams(200.0, *iso, source_depth=500.0, bottom=still)
    with pytest.raises(ValueError, match="water_density"):
        gaussian_beams(200.0, *iso, source_depth=500.0, bottom=dry)


# --- The same field, expanded the other way ---------------------------------
#
# The image lattice above and the modal sum below are two expansions of one
# exact field, and they are wrong in different ways when they are wrong: the
# lattice is a sum over paths that closes only by cancellation, the modal sum a
# sum over the eigenfunctions of the depth operator that closes term by term.
# The second is worth building because it is the expansion the other two solvers
# of this module use, so anchoring the beams to it puts all three on one
# footing, and because it can be written down from the boundary conditions
# alone, absolute phase included.

_MODAL_GUIDE = {
    "water_depth": 1000.0,
    "frequency": 290.0,
    "source_depth": 300.0,
    "receiver_depth": 600.0,
}


def _modal_pressure(
    ranges: np.ndarray,
    *,
    water_depth: float,
    frequency: float,
    source_depth: float,
    receiver_depth: float,
    n_evanescent: int = 200,
) -> np.ndarray:
    r"""The ideal guide's field as its own eigenfunction expansion.

    A pressure-release plane at each end makes the depth operator
    :math:`Z'' + (k^2 - k_r^2)Z = 0` an eigenproblem with
    :math:`Z_m = \sqrt{2/D}\,\sin(m\pi z/D)`, orthonormal at constant density,
    and Jensen Eq. (5.13) then reads

    .. math::

        p = \frac{i}{4\rho(z_s)} \sum_m \Psi_m(z_s)\,\Psi_m(z)\,
            H_0^{(1)}(k_{rm} r), \qquad k_{rm} = \sqrt{k^2 - (m\pi/D)^2},

    whose free field is the :math:`e^{ikr}/(4\pi r)` of Eq. (5.16). Multiplying
    through by :math:`4\pi` puts it on this module's unit-pressure-at-1-m
    normalisation, and that is the entire content of the comparison: there is no
    fitted constant left anywhere in it, so the test below reaches the absolute
    level and the absolute phase at once.

    The modes past cutoff are carried rather than dropped. Theirs is
    :math:`k_r = i\gamma`, and :math:`H_0^{(1)}(i\gamma r) =
    -2i K_0(\gamma r)/\pi` turns the same term into
    :math:`2\Psi\Psi K_0(\gamma r)`: real, decaying, and evaluated through the
    exponentially scaled :math:`K_0` so that a deep tail neither overflows nor
    underflows to a NaN. That makes this the exact field at every range rather
    than the far-field one of Eq. (5.14), which is what lets the phase be
    asserted at all.

    One geometry has to be avoided rather than handled: a mode sitting exactly
    at cutoff has :math:`k_r = 0` and an infinite :math:`H_0^{(1)}(0)`, which
    happens when :math:`kD/\pi` is an integer. It is 386.7 here, and 400 at the
    300 Hz of ``_GUIDE`` above, which is why this oracle is anchored at 290 Hz.
    """
    from scipy.special import hankel1, k0e

    k = 2.0 * np.pi * frequency / _C
    n_modes = int(np.floor(k * water_depth / np.pi)) + n_evanescent
    kz = (np.arange(1, n_modes + 1) * np.pi / water_depth)[:, None]
    excitation = (2.0 / water_depth) * (
        np.sin(kz * source_depth) * np.sin(kz * receiver_depth)
    )
    r = np.asarray(ranges, dtype=np.float64)[None, :]
    below = (kz < k).ravel()
    field = (
        1j
        * np.pi
        * (excitation[below] * hankel1(0, np.sqrt(k**2 - kz[below] ** 2) * r)).sum(
            axis=0
        )
    )
    beyond = np.sqrt(kz[~below] ** 2 - k**2) * r
    return np.asarray(
        field + 2.0 * (excitation[~below] * k0e(beyond) * np.exp(-beyond)).sum(axis=0),
        dtype=np.complex128,
    )


def test_the_ideal_waveguide_matches_its_own_modal_expansion() -> None:
    """Level and phase together, against a sum over modes rather than paths.

    Nothing about this oracle resembles the beam sum: it never mentions a ray, a
    launch angle or an arc length, and its range dependence is a Hankel function
    rather than a stack of straight-line delays. What survives the comparison is
    therefore the whole calibration at once, and in a channel this one is worth
    something: at 290 Hz over a kilometre of water 386 modes propagate and the
    beams that carry them have bounced up to a few dozen times by 2 km.

    Measured: 0.034 and 0.018 dB in level and 2.0e-5 and 8.4e-4 rad in phase,
    at 1.5 and 2 km. The bounds below sit an order of magnitude above that and
    orders below anything the test exists to catch, all of which are gross: a
    missing 4 pi is 21.98 dB, the printed ``e^(+i pi/4)`` of Eq. (3.92) is a
    quarter turn, a dropped conjugation reverses the phase, and losing the
    folded receiver images costs several decibels.
    """
    r = np.array([1500.0, 2000.0])
    exact = _modal_pressure(r, **_MODAL_GUIDE)
    res = gaussian_beams(
        _MODAL_GUIDE["frequency"],
        [0.0, _MODAL_GUIDE["water_depth"]],
        [_C, _C],
        source_depth=_MODAL_GUIDE["source_depth"],
        max_range=2200.0,
        ranges_m=r,
        receiver_depths_m=np.array([_MODAL_GUIDE["receiver_depth"]]),
        fan=BeamFan(max_angle_deg=88.0, beam_width=100.0),
        range_step=2.0,
    )
    assert np.abs(res.propagation_loss[0] + 20.0 * np.log10(np.abs(exact))).max() < 0.1
    assert np.abs(np.angle(res.pressure[0] / exact)).max() < 5e-3


# --- A guide the beams have to bend through ---------------------------------
#
# Everything above is isovelocity, and an isovelocity channel says nothing about
# refraction: c'' vanishes, so the coupling coefficient of the dynamic ray
# equations is zero, p is constant, q is a straight line and the impulse the
# marcher applies at a flat boundary is identically zero. The oracle below is
# built to fill exactly that gap, and it is the only one here that does.

_AIRY_GUIDE = {
    "water_depth": 200.0,
    "frequency": 200.0,
    "bottom_speed": 1530.0,
    "source_depth": 30.5,
    "receiver_depth": 120.5,
}


def _n2_linear_speeds(
    depths: np.ndarray, *, bottom_speed: float, water_depth: float
) -> np.ndarray:
    r"""Jensen's :math:`n^2`-linear profile, Eq. (3.77), scaled to two speeds.

    :math:`c(z) = c_0/\sqrt{1 + 2az/c_0}` makes :math:`k^2(z) = \omega^2/c^2`
    exactly linear in depth, which is what the closed-form modes below need.
    The ``a`` returned by the algebra carries ``_C`` at the surface to
    ``bottom_speed`` at the bottom.
    """
    a = _C / (2.0 * water_depth) * ((_C / bottom_speed) ** 2 - 1.0)
    return np.asarray(_C / np.sqrt(1.0 + 2.0 * a * np.asarray(depths) / _C))


def _airy_shape(x: float, pair: tuple[float, float]) -> float:
    """``alpha Ai(x) + beta Bi(x)``, the mode read at one depth."""
    from scipy.special import airy

    ai, _, bi, _ = airy(x)
    return float(pair[0] * ai + pair[1] * bi)


def _airy_square_integral(x: float, pair: tuple[float, float]) -> float:
    r"""Antiderivative of that mode squared, in :math:`x`.

    :math:`\int\mathrm{Ai}^2 = x\mathrm{Ai}^2 - \mathrm{Ai}'^2`,
    :math:`\int\mathrm{Ai}\,\mathrm{Bi} = x\mathrm{Ai}\mathrm{Bi} -
    \mathrm{Ai}'\mathrm{Bi}'` and :math:`\int\mathrm{Bi}^2 = x\mathrm{Bi}^2 -
    \mathrm{Bi}'^2`, each of which differentiates straight back through
    :math:`\mathrm{Ai}'' = x\mathrm{Ai}`.
    """
    from scipy.special import airy

    alpha, beta = pair
    ai, aip, bi, bip = airy(x)
    return float(
        alpha**2 * (x * ai * ai - aip * aip)
        + 2.0 * alpha * beta * (x * ai * bi - aip * bip)
        + beta**2 * (x * bi * bi - bip * bip)
    )


def _airy_mode_pressure(
    ranges: np.ndarray,
    *,
    water_depth: float,
    frequency: float,
    bottom_speed: float,
    source_depth: float,
    receiver_depth: float,
) -> np.ndarray:
    r"""The refracting guide's exact field, mode by mode, in closed form.

    With :math:`k^2(z) = A + Bz` the depth equation is Airy's own. Substituting
    :math:`x = \mu(z - z_t)`, :math:`\mu = (-B)^{1/3}`,
    :math:`z_t = (k_r^2 - A)/B`, turns :math:`Z'' + (A + Bz - k_r^2)Z = 0` into
    :math:`y'' = xy` with no approximation whatever, so

    .. math::

        Z(z) = \mathrm{Bi}(x_0)\,\mathrm{Ai}(x) - \mathrm{Ai}(x_0)\,\mathrm{Bi}(x)

    already vanishes at the surface, and the eigenvalues are the roots of
    :math:`\mathrm{Ai}(x_0)\mathrm{Bi}(x_D) - \mathrm{Ai}(x_D)\mathrm{Bi}(x_0)`,
    hunted here by bracketing a scan and polishing with Brent. The normalisation
    is closed form too, since
    :math:`\int\mathrm{Ai}^2 = x\mathrm{Ai}^2 - \mathrm{Ai}'^2` and its two
    companions differentiate straight back through :math:`\mathrm{Ai}'' =
    x\mathrm{Ai}`; against a 40001-point quadrature it agrees to 3.9e-14.

    So this is a genuine closed form, not a fine numerical solution of the same
    problem: no eigenvalue matrix, no depth grid, no far-field approximation of
    the Hankel function. Checked in scratch against a DOP853 shooting
    integration of the same boundary-value problem at ``rtol=1e-12``, which is
    stable for the 45 of the 52 modes whose turning point lies below the bottom:
    the eigenvalues agree to 4.0e-13 1/m and the normalised modal products
    :math:`Z(z_s)Z(z_r)` to 6.1e-11 relative. The mode count, 52, is the 52.8 a
    WKB integral gives.

    Only the propagating modes are summed. The first one past cutoff decays as
    :math:`e^{-65}` by the shortest range used below, so it is not an
    approximation at these ranges, it is a rounding.
    """
    from scipy.optimize import brentq
    from scipy.special import airy, hankel1

    omega = 2.0 * np.pi * frequency
    a = _C / (2.0 * water_depth) * ((_C / bottom_speed) ** 2 - 1.0)
    coef_a = (omega / _C) ** 2
    coef_b = 2.0 * a * omega**2 / _C**3
    mu = (-coef_b) ** (1.0 / 3.0)

    def edges(kr: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        turning = (np.asarray(kr) ** 2 - coef_a) / coef_b
        return -mu * turning, mu * (water_depth - turning)

    def determinant(kr: float) -> float:
        x0, xd = edges(kr)
        ai0, _, bi0, _ = airy(x0)
        aid, _, bid, _ = airy(xd)
        return float(ai0 * bid - aid * bi0)

    scan = np.linspace(1e-6, np.sqrt(coef_a) * (1.0 - 1e-12), 100_001)
    scan_x0, scan_xd = edges(scan)
    ai0s, _, bi0s, _ = airy(scan_x0)
    aids, _, bids, _ = airy(scan_xd)
    values = ai0s * bids - aids * bi0s
    brackets = np.flatnonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0.0)

    r = np.asarray(ranges, dtype=np.float64)
    field = np.zeros(r.size, dtype=np.complex128)
    for lo in brackets:
        kr = brentq(determinant, scan[lo], scan[lo + 1], xtol=1e-14, rtol=8.9e-16)
        x0, xd = edges(kr)
        ai0, _, bi0, _ = airy(x0)
        pair = (float(bi0), -float(ai0))
        turning = (kr**2 - coef_a) / coef_b
        norm = (
            _airy_square_integral(float(xd), pair)
            - _airy_square_integral(float(x0), pair)
        ) / mu
        at_source = _airy_shape(mu * (source_depth - turning), pair)
        at_receiver = _airy_shape(mu * (receiver_depth - turning), pair)
        field += 1j * np.pi * at_source * at_receiver / norm * hankel1(0, kr * r)
    return np.asarray(field, dtype=np.complex128)


def _incoherent(loss: np.ndarray, window: int = 11) -> np.ndarray:
    """Energy-average a loss curve over neighbouring ranges.

    The two fields interfere on the same fringes but not with the same phase to
    the last degree, so a point-by-point comparison in a null measures the null
    and not the method. Averaging in intensity, which is what
    :func:`normal_modes`' own trend test does, leaves the quantity a
    transmission-loss curve is read for.
    """
    kernel = np.ones(window) / window
    return np.asarray(
        -10.0
        * np.log10(np.convolve(10.0 ** (-np.asarray(loss) / 10.0), kernel, mode="same"))
    )


def test_a_refracting_guide_matches_its_exact_airy_modes() -> None:
    r"""The one test here that puts a bend in the rays and an exact number on it.

    Measured over 0.5 to 4 km, energy-averaged, against the closed form above,
    with everything at its default: a mean of +0.19 dB with a 1.31 dB scatter
    and a 3.7 dB worst bin. The mean is the assertion that means something,
    since a mistake in the refracting half of the marcher shows as a bias and
    not as scatter; the scatter is fringes that have slipped by a fraction of
    a period and is bounded here at the same 2 dB the module's other
    cross-solver comparison uses.

    The default width is under test on purpose, because THIS configuration is
    the one that forced its overhaul and the numbers are worth keeping. An
    earlier :func:`_default_beam_width` clamped :math:`W_0` to a quarter of
    the water depth, 50 m here, half the
    :math:`\sqrt{\lambda r_\mathrm{max}/\pi}` free-space optimum the same
    function computed first: same configuration, same oracle, that width
    measured +3.08 dB mean and 5.86 dB worst, systematically too quiet, while
    explicit 100, 150 and 200 m gave +1.13, +0.26 and -0.22. It was never the
    refraction: the same profile in 1000 m of water, where the cap did not
    bite, came out at +0.72 dB mean with a 1.37 dB worst bin, better than
    :func:`normal_modes` manages against the same closed form. The per-angle
    default (100 to 255 m across the 80 degree fan, the guide's
    :math:`4D\cos\theta_0/\pi` on the flat beams and the free-space optimum
    on the steep ones) holds the same cut at +0.19 dB with no override and no
    warning, and the shallow-guide test further down pins the retired cap's
    cost against it on a second, independent configuration.
    """
    r = np.linspace(500.0, 4000.0, 100)
    exact = -20.0 * np.log10(np.abs(_airy_mode_pressure(r, **_AIRY_GUIDE)))
    depths = np.linspace(0.0, _AIRY_GUIDE["water_depth"], 21)
    speeds = _n2_linear_speeds(
        depths,
        bottom_speed=_AIRY_GUIDE["bottom_speed"],
        water_depth=_AIRY_GUIDE["water_depth"],
    )
    res = gaussian_beams(
        _AIRY_GUIDE["frequency"],
        depths,
        speeds,
        source_depth=_AIRY_GUIDE["source_depth"],
        max_range=4200.0,
        ranges_m=r,
        receiver_depths_m=np.array([_AIRY_GUIDE["receiver_depth"]]),
        fan=BeamFan(max_angle_deg=80.0),
        range_step=5.0,
    )
    # The convolution's own edges are not an average of anything, so they go.
    difference = (_incoherent(res.propagation_loss[0]) - _incoherent(exact))[10:-10]
    assert abs(float(difference.mean())) < 1.0
    assert float(difference.std()) < 2.0
    # The default is the per-angle rule, not a flat width: the flat beams
    # carry the modal-resolution width 4 D / pi and the steep ones the
    # free-space optimum, and the vertical footprint W_0 / cos(theta_0) of
    # every guide-ruled beam is the same 4 D / pi.
    four_d_over_pi = 4.0 * _AIRY_GUIDE["water_depth"] / np.pi
    assert float(res.initial_beam_widths.max()) == pytest.approx(
        four_d_over_pi, rel=1e-4
    )
    assert res.initial_beam_widths[0] < res.initial_beam_widths.max()


def test_a_shallow_guide_no_longer_pays_the_quarter_depth_caps_toll() -> None:
    r"""The cap's cost measured against the width that replaced it, both in dB.

    A second shallow refracting configuration, sharing nothing numerical with
    the one above (half the depth, a different frequency, gradient, source and
    receiver), against the same exact closed form. The quarter-depth cap this
    solver used to clamp :math:`W_0` with would put 25 m here, under half the
    ten-wavelength floor the book itself recommends; run explicitly at that
    width, the loss comes out +4.12 dB high in the mean (7.5 dB at the worst
    bin), energy-averaged over 0.3 to 2.5 km. The per-angle default (70.5 m
    free-space optimum on the steep beams, rising to the guide's
    :math:`4D/\pi = 127.3` m on the flat ones) measures +0.39 dB on the same
    cut: over ninety per cent of the cap's error, gone with the cap. Both
    numbers are asserted, the first from below and the second from above, so
    this stays a measurement of the cap's cost and not a story about it.
    """
    guide = {
        "water_depth": 100.0,
        "frequency": 250.0,
        "bottom_speed": 1520.0,
        "source_depth": 15.5,
        "receiver_depth": 70.5,
    }
    r = np.linspace(300.0, 2500.0, 100)
    exact = -20.0 * np.log10(np.abs(_airy_mode_pressure(r, **guide)))
    depths = np.linspace(0.0, guide["water_depth"], 21)
    speeds = _n2_linear_speeds(
        depths, bottom_speed=guide["bottom_speed"], water_depth=guide["water_depth"]
    )

    def bias(beam_width: float | None) -> float:
        res = gaussian_beams(
            guide["frequency"],
            depths,
            speeds,
            source_depth=guide["source_depth"],
            max_range=2600.0,
            ranges_m=r,
            receiver_depths_m=np.array([guide["receiver_depth"]]),
            fan=BeamFan(max_angle_deg=80.0, beam_width=beam_width),
            range_step=2.5,
        )
        return float(
            (_incoherent(res.propagation_loss[0]) - _incoherent(exact))[10:-10].mean()
        )

    capped = bias(guide["water_depth"] / 4.0)  # what the retired cap forced
    unclamped = bias(None)
    assert capped > 3.0, f"the cap has to cost decibels here, measured {capped:.2f}"
    assert abs(unclamped) < 0.8, (
        f"the default has to remove it, measured {unclamped:.2f}"
    )
    assert abs(unclamped) < 0.2 * capped


# --- What the beam itself does ----------------------------------------------

_FREE_BEAM_WIDTH = 120.0
_FREE_BEAM_FREQUENCY = 200.0


def _free_space_beam() -> tuple[GaussianBeamResult, np.ndarray, float]:
    """One free-space fan, with its arc lengths and its Rayleigh range.

    The run has to reach well past :math:`kW_0^2/2` for the hyperbola's scale to
    be measurable at all, so it is traced to twice that; the field itself is
    asked for at one point, since only the per-ray beam history is under test.
    """
    f, w0 = _FREE_BEAM_FREQUENCY, _FREE_BEAM_WIDTH
    rayleigh = (2.0 * np.pi * f / _C) * w0**2 / 2.0
    rmax = 2.0 * rayleigh * np.cos(np.radians(45.0))
    depth = _free_space_column(rmax, 45.0)
    res = gaussian_beams(
        f,
        [0.0, depth],
        [_C, _C],
        source_depth=depth / 2.0,
        max_range=rmax,
        fan=BeamFan(max_angle_deg=45.0, beam_width=w0, n_beams=9),
        range_step=rmax / 400.0,
        n_depth_points=2,
        ranges_m=np.array([rmax]),
    )
    arc = res.ray_ranges / np.cos(np.radians(res.launch_angles))[:, None]
    return res, arc, rayleigh


def test_the_beam_half_width_is_the_free_space_hyperbola() -> None:
    r"""Eq. (3.86) with the waist at the source, to machine precision.

    In a homogeneous medium ``p`` stays at 1 and ``q(s) = q(0) + c_0 s``, so
    Eq. (3.89) collapses to :math:`W(s) = W_0\sqrt{1 + (2s/(kW_0^2))^2}`, a
    hyperbola with its waist at the source and the Rayleigh range
    :math:`kW_0^2/2` for its scale. The arc length is ``r/cos(theta_0)`` because
    the rays are straight.
    """
    res, arc, rayleigh = _free_space_beam()
    # An explicit width is every beam's width, and the result records it so.
    assert np.all(res.initial_beam_widths == _FREE_BEAM_WIDTH)
    w0 = float(res.initial_beam_widths[0])
    exact = w0 * np.sqrt(1.0 + (arc / rayleigh) ** 2)
    assert res.beam_widths.shape == exact.shape
    assert np.allclose(res.beam_widths, exact, rtol=1e-12)
    # The waist is at the source, and one Rayleigh range out the beam has
    # widened by exactly sqrt(2), which is what fixes the scale of the
    # hyperbola rather than merely its shape.
    assert res.beam_widths[:, 0] == pytest.approx(w0, rel=1e-13)
    at_rayleigh = np.interp(rayleigh, arc[0], res.beam_widths[0])
    assert at_rayleigh == pytest.approx(w0 * np.sqrt(2.0), rel=1e-4)
    assert np.all(np.diff(res.beam_widths, axis=1) > 0.0)


def test_the_wavefront_curvature_is_the_complex_source_point_one() -> None:
    r"""Eq. (3.90) against Eq. (3.85), which is derived a different way.

    Sect. 3.5.1 gets :math:`K(x) = x/(x^2 + a^2)` by offsetting a point source
    into the complex plane, with :math:`a = kW_0^2/2` the same Rayleigh range;
    Sect. 3.5.2 gets its :math:`K` from the dynamic ray equations. They have to
    agree, and the sign is the one that belongs to the conjugated field this
    result exposes: positive for a beam spreading away from its waist.
    """
    res, arc, rayleigh = _free_space_beam()
    assert np.allclose(
        res.wavefront_curvatures, arc / (arc**2 + rayleigh**2), rtol=1e-12, atol=1e-15
    )
    # Flat at the waist, and the curvature peaks one Rayleigh range out.
    assert np.all(res.wavefront_curvatures[:, 0] == 0.0)
    assert arc[0, int(np.argmax(res.wavefront_curvatures[0]))] == pytest.approx(
        rayleigh, rel=0.02
    )


def test_the_conserved_wronskian_is_why_the_spreading_never_vanishes() -> None:
    r"""``q_R p_I - q_I p_R`` holds at ``-omega W_0^2/2`` through everything.

    This is the structural claim the whole method rests on, so it is worth
    measuring rather than asserting in prose. Eq. (3.58) is linear with real
    coefficients, so the real and imaginary parts of :math:`(q, p)` are two real
    solutions of it and their Wronskian is constant; the impulse at a profile
    node and the one at a reflection are both shears of unit determinant, so
    they cannot change it either. A constant that starts non-zero means ``q``
    can never vanish, hence no caustic singularity, and it also gives
    :math:`\mathrm{Im}[p/q] = -\omega W_0^2/(2|q|^2)`, so Eq. (3.89) becomes
    :math:`W = 2|q|/(\omega W_0)`, which is what the solver reports.

    The path below crosses two hard gradient kinks and works both boundaries,
    which is every kind of event the marcher knows how to apply.
    """
    f, w0, depth = 300.0, 60.0, 1000.0
    z_prof = np.array([0.0, 100.0, 300.0, depth])
    c_prof = np.array([1500.0, 1510.0, 1488.0, 1512.0])
    omega = 2.0 * np.pi * f
    angles = np.radians(np.array([-25.0, -8.0, 3.0, 17.0, 40.0]))
    c0 = float(np.interp(200.0, z_prof, c_prof))
    xi = np.cos(angles) / c0
    march = march_rays(
        _ocean_ray_derivative(z_prof, c_prof),
        xi=xi,
        z0=np.full(angles.size, 200.0),
        zeta0=np.sin(angles) / c0,
        range_step=10.0,
        n_steps=1201,
        lower=0.0,
        upper=depth,
        dynamic=DynamicRays(
            np.full(angles.size, 0.5j * omega * w0**2),
            np.full(angles.size, 1.0 + 0.0j),
            z_prof,
            c_prof,
        ),
    )
    assert march.reflections.sum() > 10  # the path really does work both walls
    q = np.asarray(march.spreadings)
    p = np.asarray(march.spreading_slopes)
    wronskian = q.real * p.imag - q.imag * p.real
    expected = -0.5 * omega * w0**2
    assert np.abs(wronskian / expected - 1.0).max() < 1e-12
    # Which is the same as saying q stays away from the origin, by a margin
    # that no reflection or kink can close.
    assert np.abs(q[:, 1:]).min() > 0.0
    ratio = np.imag(p / q) * np.abs(q) ** 2 / expected
    assert np.abs(ratio - 1.0).max() < 1e-11


# --- Caustics and shadow zones ----------------------------------------------
#
# The n^2-linear profile of Jensen Eq. (3.77), c(z) = c0/sqrt(1 + 2.4 z/c0) with
# c0 = 1550 m/s, which Sect. 3.4.1 uses for exactly these two illustrations: a
# deep source makes a caustic and a shallow one makes a shadow zone.

_N2_C0, _N2_DEPTH = 1550.0, 1000.0
_N2_DEPTHS = np.linspace(0.0, _N2_DEPTH, 201)
_N2_SPEEDS = _N2_C0 / np.sqrt(1.0 + 2.4 * _N2_DEPTHS / _N2_C0)


def _geometric_spreading(
    source_depth: float,
    angles_deg: np.ndarray | list[float],
    *,
    max_range: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``(q, z)`` for the *geometric* initial conditions of Eq. (3.63).

    With ``q(0) = 0`` and ``p(0) = 1/c(0)`` the pair is the ray-tube spreading
    itself, ``r q = J`` (Eq. 3.64), and the classical amplitude of Eq. (3.65)
    divides by it. That is the configuration with closed-form answers and the
    one that has caustics, so it is the oracle for where the caustics are; the
    beam solver never runs it.
    """
    th = np.radians(np.atleast_1d(np.asarray(angles_deg, dtype=np.float64)))
    c0 = float(np.interp(source_depth, _N2_DEPTHS, _N2_SPEEDS))
    xi = np.cos(th) / c0
    march = march_rays(
        _ocean_ray_derivative(_N2_DEPTHS, _N2_SPEEDS),
        xi=xi,
        z0=np.full(th.size, source_depth),
        zeta0=np.sin(th) / c0,
        range_step=max_range / (n_steps - 1),
        n_steps=n_steps,
        lower=0.0,
        upper=_N2_DEPTH,
        dynamic=DynamicRays(
            np.zeros(th.size), np.full(th.size, 1.0 / c0), _N2_DEPTHS, _N2_SPEEDS
        ),
    )
    return np.asarray(march.spreadings, dtype=np.float64), march.positions


def _beam_spreading(
    source_depth: float,
    angles_deg: np.ndarray | list[float],
    *,
    max_range: float,
    n_steps: int,
    omega: float,
    beam_width: float,
) -> np.ndarray:
    """``q`` for the *beam* initial conditions of Eq. (3.91), same marcher."""
    th = np.radians(np.atleast_1d(np.asarray(angles_deg, dtype=np.float64)))
    c0 = float(np.interp(source_depth, _N2_DEPTHS, _N2_SPEEDS))
    xi = np.cos(th) / c0
    march = march_rays(
        _ocean_ray_derivative(_N2_DEPTHS, _N2_SPEEDS),
        xi=xi,
        z0=np.full(th.size, source_depth),
        zeta0=np.sin(th) / c0,
        range_step=max_range / (n_steps - 1),
        n_steps=n_steps,
        lower=0.0,
        upper=_N2_DEPTH,
        dynamic=DynamicRays(
            np.full(th.size, 0.5j * omega * beam_width**2),
            np.full(th.size, 1.0 + 0.0j),
            _N2_DEPTHS,
            _N2_SPEEDS,
        ),
    )
    return np.asarray(march.spreadings, dtype=np.complex128)


def test_the_tracked_branch_of_the_square_root_is_the_kmah_index() -> None:
    r"""Eq. (3.79)'s :math:`(-i)^m`, recovered from the branch rather than imposed.

    Ray theory has to count caustics and multiply by :math:`(-i)^m` by hand;
    Sect. 3.5's claim is that a complex :math:`q` carries that phase on its own,
    which is why this module has no KMAH index in it. The claim is exactly
    checkable, because Eq. (3.58) is linear with real coefficients and
    Eq. (3.91) starts :math:`q` at :math:`i\omega W_0^2/2` with :math:`p` real:
    the real and imaginary parts evolve independently, and
    :math:`\mathrm{Re}[q]` is then the *geometric* spreading of Eq. (3.63) to
    the last bits. So a caustic, where the geometric spreading vanishes, is
    exactly where the beam's :math:`q` is purely imaginary, and the question of
    the branch is the question of which way it goes round.

    It goes round one way, always, by :math:`\pi` per caustic. The unwrapped
    argument starts at :math:`+\pi/2` and reads :math:`-(2k+1)\pi/2` at the
    :math:`k`-th caustic, a descending ladder with no jumps in it, so
    :math:`\sqrt{q}` picks up :math:`e^{-i\pi/2} = -i` at each one. That is
    Eq. (3.79), arrived at from the other end.

    A principal-value square root taken sample by sample sees the same ladder
    folded into :math:`\pm\pi/2` and is wrong from the first caustic on, by
    :math:`2\pi` in the argument on rays that cross one and by :math:`6\pi` on
    the shallow rays that cross six. That difference is asserted too, so this
    stays a test of something the solver actually depends on.
    """
    zs, rmax, n_steps = 992.5, 4000.0, 4001
    omega, w0 = 2.0 * np.pi * 600.0, 35.9
    angles = np.array([-25.0, -20.0, -13.3886, -8.0, -3.0, 3.0, 10.0, 20.0, 30.0])
    q = _beam_spreading(
        zs, angles, max_range=rmax, n_steps=n_steps, omega=omega, beam_width=w0
    )

    # Re[q] is the geometric spreading, which is what puts caustics on the axis.
    geometric, _z = _geometric_spreading(zs, angles, max_range=rmax, n_steps=n_steps)
    c0 = float(np.interp(zs, _N2_DEPTHS, _N2_SPEEDS))
    assert np.abs(q.real - c0 * geometric).max() / np.abs(q.real).max() < 1e-10

    crossed, deep = 0, 0
    for i in range(angles.size):
        real, phase = q[i].real, np.unwrap(np.angle(q[i]))
        assert phase[0] == pytest.approx(0.5 * np.pi)
        # Interpolate onto each zero of Re[q] rather than the sample beside it.
        zeros = np.flatnonzero(np.sign(real[1:-1]) != np.sign(real[2:])) + 1
        for k_th, k in enumerate(zeros):
            frac = real[k] / (real[k] - real[k + 1])
            at = phase[k] + frac * (phase[k + 1] - phase[k])
            assert at == pytest.approx(-(2 * k_th + 1) * 0.5 * np.pi, abs=5e-3)
        crossed += zeros.size
        # The whole point: past the second caustic the ladder has left the
        # principal strip, and a principal value is a different function.
        if zeros.size >= 2:
            assert np.abs(phase - np.angle(q[i])).max() > 6.0
            deep += 1
    assert crossed >= 20, "the fan has to cross plenty of caustics"
    assert deep >= 4, "and several rays have to get past the second one"

    # And the march resolves the winding rather than aliasing it: at a twentieth
    # of the sampling the ladder is the same one, so no rung was ever skipped.
    coarse = _beam_spreading(
        zs, angles, max_range=rmax, n_steps=201, omega=omega, beam_width=w0
    )
    fine_turn = np.unwrap(np.angle(q), axis=1)[:, -1]
    coarse_turn = np.unwrap(np.angle(coarse), axis=1)[:, -1]
    assert np.abs(fine_turn - coarse_turn).max() < 0.5 * np.pi


def test_the_influence_sum_uses_the_tracked_branch_and_not_the_principal_one() -> None:
    r"""One beam, by hand, with a winding the principal value cannot see.

    The test above pins the branch as a property of :math:`q`; this one pins
    that the sum actually spends it. A single horizontal beam is handed to
    :func:`_beam_influence` with the receiver exactly on its axis, so
    :math:`n = 0`, the extrapolation along the ray is nothing, and Eq. (3.88)
    collapses to :math:`A\sqrt{c/(r q)}\,e^{-i\omega\tau}` with only the branch
    left to get wrong. Its tracked argument is set a full turn below the
    principal value, which is what a ray that has passed two caustics carries,
    and a full turn in :math:`\arg q` is half a turn in :math:`\sqrt{q}`: the
    two readings differ by a factor of exactly -1, not by a little.

    The water column is made enormous and the beam's reach tiny so the folded
    images of the receiver are all thrown out and the one beam is the whole
    answer.
    """
    from phonometry.underwater.propagation.numerical import (
        _beam_influence,
        _BeamSamples,
    )

    c, r_col, omega, w0 = 1500.0, 3000.0, 2.0 * np.pi * 100.0, 40.0
    q = 1.0e6 * (0.3 + 0.7j)
    p = 2.0e-4 * (1.0 - 0.5j)
    tau, z_ray = 2.0, 5.0e5
    one = np.array([[1.0]])
    samples = _BeamSamples(
        xi=np.array([[1.0 / c]]),
        column_range=np.array([[r_col]]),
        range_offset=np.array([[0.0]]),
        depth=np.array([[z_ray]]),
        vertical=np.array([[0.0]]),
        speed=one * c,
        spreading=np.array([[q]]),
        slope=np.array([[p]]),
        time=one * tau,
        path=one * r_col,
        # A ray two caustics along: one full turn under the principal value.
        phase=np.array([[np.angle(q) - 2.0 * np.pi]]),
        weight=np.array([[1.0 + 0.0j]]),
        reach=np.array([50.0]),
    )
    got = _beam_influence(
        samples,
        np.array([z_ray]),
        water_depth=1.0e6,
        bottom_reflection=1.0,
        omega=omega,
        beam_width=w0,
        attenuation=0.0,
    )

    expected = (
        np.sqrt(c / r_col)
        / np.sqrt(abs(q))
        * np.exp(-0.5j * (np.angle(q) - 2.0 * np.pi) - 1j * omega * tau)
    )
    assert got.shape == (1, 1)
    assert got[0, 0] == pytest.approx(expected, rel=1e-12)
    # The principal value is the same number with the opposite sign, so this is
    # not a tolerance that a rounding change could drift across.
    principal = (
        np.sqrt(c / r_col)
        / np.sqrt(abs(q))
        * np.exp(-0.5j * np.angle(q) - 1j * omega * tau)
    )
    assert principal == pytest.approx(-expected, rel=1e-12)
    assert abs(got[0, 0] - principal) == pytest.approx(2.0 * abs(expected), rel=1e-12)


def test_the_field_is_finite_where_the_classical_amplitude_is_not() -> None:
    """A caustic pinned by bisection, then asked of the beam field.

    The launch angle is refined until the geometric spreading at a chosen range
    is zero to the last bits, which is the definition of a caustic: the ray tube
    has closed and Eq. (3.65)'s ``1/sqrt(r q)`` is unbounded there. Ray theory
    answers infinity, and the whole point of the beams is that they answer a
    number.
    """
    zs, rmax, n_steps = 992.5, 4000.0, 2001
    column = n_steps // 2
    ranges = np.linspace(0.0, rmax, n_steps)
    scan = np.linspace(-25.0, -5.0, 41)
    q_scan, _ = _geometric_spreading(zs, scan, max_range=rmax, n_steps=n_steps)
    crossings = np.flatnonzero(np.diff(np.sign(q_scan[:, column])) != 0)
    assert crossings.size, "the scan has to straddle a caustic"
    lo, hi = scan[crossings[0]], scan[crossings[0] + 1]
    sign_lo = np.sign(
        _geometric_spreading(zs, [lo], max_range=rmax, n_steps=n_steps)[0][0, column]
    )
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        q_mid, _z = _geometric_spreading(zs, [mid], max_range=rmax, n_steps=n_steps)
        lo, hi = (mid, hi) if np.sign(q_mid[0, column]) == sign_lo else (lo, mid)
    q_c, z_c = _geometric_spreading(
        zs, [0.5 * (lo + hi)], max_range=rmax, n_steps=n_steps
    )
    r_caustic, z_caustic = float(ranges[column]), float(z_c[0, column])
    tube = np.abs(ranges[1:] * q_c[0, 1:])
    # The classical amplitude is 1/sqrt(|r q|): seven orders of magnitude above
    # its own median along the very same ray, and unbounded under refinement.
    assert np.sqrt(np.median(tube) / abs(r_caustic * q_c[0, column])) > 1e6

    # A cut across the caustic, kept inside the water column: the solver has
    # nothing to say about a receiver under the seabed and refuses to pretend.
    reach = min(120.0, z_caustic - 1.0, _N2_DEPTH - z_caustic - 1.0)
    offsets = np.linspace(-reach, reach, 121)
    res = gaussian_beams(
        600.0,
        _N2_DEPTHS,
        _N2_SPEEDS,
        source_depth=zs,
        max_range=rmax,
        ranges_m=np.array([r_caustic]),
        receiver_depths_m=z_caustic + offsets,
        fan=BeamFan(max_angle_deg=45.0),
        range_step=2.0,
    )
    pl = res.propagation_loss[:, 0]
    assert np.all(np.isfinite(pl))
    assert pl.min() > 30.0  # finite, and not absurdly loud either
    # And it is a real focus rather than merely finite: the caustic stands well
    # above the interference either side of it.
    near = np.abs(offsets) < 0.2 * reach
    assert float(pl[near].min()) < float(np.median(pl)) - 5.0


def test_the_shadow_zone_decays_smoothly_without_going_silent() -> None:
    """Beyond the limiting ray, where ray theory has nothing at all to say.

    The shallow-source case of Sect. 3.4.1: with the source at 75 m in this
    profile no ray reaches that depth again beyond about 0.88 km, which
    :func:`ray_trace` is asked to confirm here rather than being taken on trust.
    Classical ray theory then returns silence; the exact solution decays
    gradually (Fig. 3.11) and so does this one.
    """
    zs, f = 77.5, 2000.0
    rays = ray_trace(
        _N2_DEPTHS,
        _N2_SPEEDS,
        source_depth=zs,
        launch_angles_deg=np.linspace(-19.0, 19.0, 761),
        max_range=1400.0,
        n_steps=1401,
    )
    closest = np.abs(rays.depths - zs).min(axis=0)
    limiting = float(rays.ranges[0][np.flatnonzero(closest > 2.0)[0]])
    assert 800.0 < limiting < 950.0
    # Well inside the shadow, no ray comes within a hundred metres of the
    # receiver, so there is no eigenray to be found and no classical answer.
    assert closest[np.argmin(np.abs(rays.ranges[0] - 1300.0))] > 100.0

    r = np.linspace(600.0, 1350.0, 76)
    res = gaussian_beams(
        f,
        _N2_DEPTHS,
        _N2_SPEEDS,
        source_depth=zs,
        max_range=1500.0,
        ranges_m=r,
        receiver_depths_m=np.array([zs]),
        fan=BeamFan(max_angle_deg=30.0),
        range_step=2.0,
    )
    pl = res.propagation_loss[0]
    assert np.all(np.isfinite(pl))
    # Nonzero everywhere: the beams' tails carry energy into the shadow, which
    # is the behaviour a wave solution has and a ray solution does not.
    assert pl.max() < 140.0
    # And decaying: incoherent averages over three successive windows, since the
    # shadow still carries the interference of whatever reached it.
    lit = r < limiting
    windows = [pl[lit], pl[(~lit) & (r < 1150.0)], pl[r >= 1150.0]]
    means = [-10.0 * np.log10(np.mean(10.0 ** (-w / 10.0))) for w in windows]
    assert means[0] < means[1] < means[2]
    assert means[2] - means[0] > 20.0


# --- Defaults, warnings and validation --------------------------------------


@pytest.mark.parametrize(
    ("frequency", "max_range", "water_depth", "axial", "edge"),
    [
        # sqrt(lambda r / pi) inside the book's 10-50 wavelength band, flat
        # across the fan: 5 km of water is beyond the guide term's reach.
        (100.0, 10_000.0, 5000.0, 218.5, 218.5),
        (1000.0, 10_000.0, 5000.0, 69.1, 69.1),
        # Too low a frequency for the optimum: the ten-wavelength floor lifts it.
        (20.0, 10_000.0, 5000.0, 750.0, 750.0),
        # A shallow channel: the modal-resolution width 4 D cos(theta_0)/pi
        # rules wherever it beats the optimum, so the fan is widest on the
        # axis (4 D/pi) and relaxes towards the edges. The retired
        # quarter-depth cap would have forced 50 m on all nine beams.
        (100.0, 10_000.0, 200.0, 254.6, 239.3),
    ],
)
def test_the_default_width_is_the_optimum_raised_to_the_guides_need(
    frequency: float,
    max_range: float,
    water_depth: float,
    axial: float,
    edge: float,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PhonometryWarning)
        res = gaussian_beams(
            frequency,
            [0.0, water_depth],
            [_C, _C],
            source_depth=0.5 * water_depth,
            max_range=max_range,
            ranges_m=np.array([max_range]),
            n_depth_points=2,
            fan=BeamFan(max_angle_deg=20.0, n_beams=9),
            range_step=max_range / 4.0,
        )
    assert res.initial_beam_widths.shape == (9,)
    assert res.initial_beam_widths[4] == pytest.approx(axial, rel=1e-3)
    assert res.initial_beam_widths[0] == pytest.approx(edge, rel=1e-3)
    assert res.initial_beam_widths[-1] == pytest.approx(edge, rel=1e-3)


def test_the_answer_does_not_care_which_width_inside_the_band_is_used() -> None:
    """Doubling and halving the beam width moves the loss by hundredths of a dB.

    Which is what makes a defensible default cost nothing: the choice of
    :math:`W_0` is "a matter of current research" only outside the band the book
    recommends.
    """
    r = np.array([2000.0])
    losses = []
    for w0 in (60.0, 120.0, 240.0):
        res = gaussian_beams(
            _GUIDE["frequency"],
            [0.0, _GUIDE["water_depth"]],
            [_C, _C],
            source_depth=_GUIDE["source_depth"],
            max_range=2200.0,
            ranges_m=r,
            receiver_depths_m=np.array([_GUIDE["receiver_depth"]]),
            fan=BeamFan(max_angle_deg=85.0, beam_width=w0),
            range_step=5.0,
        )
        losses.append(float(res.propagation_loss[0, 0]))
    assert max(losses) - min(losses) < 0.2  # measured 0.13 dB


def test_a_source_on_a_profile_kink_is_warned_about() -> None:
    z = np.array([0.0, 100.0, 300.0, 1000.0])
    c = np.array([1500.0, 1510.0, 1488.0, 1512.0])
    r = np.array([1000.0])
    fan = BeamFan(max_angle_deg=30.0, n_beams=9)
    with pytest.warns(PhonometryWarning, match="gradient discontinuity"):
        gaussian_beams(
            200.0,
            z,
            c,
            source_depth=100.0,
            max_range=1000.0,
            ranges_m=r,
            n_depth_points=2,
            fan=fan,
            range_step=100.0,
        )
    # A metre off the node is a different problem, and silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error", PhonometryWarning)
        gaussian_beams(
            200.0,
            z,
            c,
            source_depth=101.0,
            max_range=1000.0,
            ranges_m=np.array([1000.0]),
            n_depth_points=2,
            fan=BeamFan(max_angle_deg=30.0, n_beams=9),
            range_step=100.0,
        )


def test_a_beam_wider_than_a_quarter_of_the_channel_passes_in_silence() -> None:
    """The quarter-depth warning went with the quarter-depth cap.

    An explicit width above a quarter of the water depth used to warn that the
    folded field would drift from the true one; the receiver-image ladder is
    what makes that untrue, and the two shallow-guide oracles above measure the
    wide width as the *better* answer. A warning against the better answer
    trains callers to ignore warnings, so it is gone, and this pins that it
    stays gone.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", PhonometryWarning)
        gaussian_beams(
            200.0,
            [0.0, 400.0],
            [_C, _C],
            source_depth=200.0,
            max_range=1000.0,
            ranges_m=np.array([1000.0]),
            n_depth_points=2,
            range_step=20.0,
            fan=BeamFan(max_angle_deg=30.0, n_beams=9, beam_width=150.0),
        )


def test_a_step_that_cannot_follow_the_steepest_beam_is_warned_about() -> None:
    r = np.array([2000.0])
    steep_fan = BeamFan(max_angle_deg=85.0, n_beams=9)
    with pytest.warns(PhonometryWarning, match="steepest beam"):
        gaussian_beams(
            200.0,
            [0.0, 400.0],
            [_C, _C],
            source_depth=200.0,
            max_range=2000.0,
            ranges_m=r,
            n_depth_points=2,
            fan=steep_fan,
            range_step=100.0,
        )


def test_every_warning_is_reported_against_the_line_that_caused_it() -> None:
    """The call site, not the module the check happens to live in.

    One of the two checks runs inside a helper of its own and so stands a
    frame further from the caller than the other; a single ``stacklevel`` for
    both would report that one against ``numerical.py``, where the reader has
    nothing to change. The filename is the assertion because it is what a
    caller actually sees, and it is wrong in exactly the case a shared
    constant would make wrong.
    """
    z = np.array([0.0, 100.0, 300.0, 1000.0])
    c = np.array([1500.0, 1510.0, 1488.0, 1512.0])
    on_kink = np.array([1000.0])
    flat_fan = BeamFan(max_angle_deg=30.0, n_beams=9)
    far = np.array([2000.0])
    steep_fan = BeamFan(max_angle_deg=85.0, n_beams=9)
    with pytest.warns(PhonometryWarning) as kink:
        gaussian_beams(
            200.0,
            z,
            c,
            source_depth=100.0,
            max_range=1000.0,
            ranges_m=on_kink,
            n_depth_points=2,
            fan=flat_fan,
            range_step=100.0,
        )
    with pytest.warns(PhonometryWarning) as steep:
        gaussian_beams(
            200.0,
            [0.0, 400.0],
            [_C, _C],
            source_depth=200.0,
            max_range=2000.0,
            ranges_m=far,
            n_depth_points=2,
            fan=steep_fan,
            range_step=100.0,
        )
    for caught in (kink, steep):
        assert caught[0].filename == __file__


def test_invalid_inputs_rejected() -> None:
    iso = ([0.0, 1000.0], [_C, _C])
    with pytest.raises(ValueError, match="source_depth"):
        gaussian_beams(200.0, *iso, source_depth=1200.0)
    vertical = BeamFan(max_angle_deg=90.0)
    with pytest.raises(ValueError, match="max_angle_deg"):
        gaussian_beams(200.0, *iso, source_depth=500.0, fan=vertical)
    with pytest.raises(ValueError, match="bottom"):
        gaussian_beams(200.0, *iso, source_depth=500.0, bottom="sandy")
    with pytest.raises(ValueError, match="range_step"):
        gaussian_beams(
            200.0, *iso, source_depth=500.0, max_range=100.0, range_step=200.0
        )
    unpaired = BeamFan(n_beams=1)
    with pytest.raises(ValueError, match="n_beams"):
        gaussian_beams(200.0, *iso, source_depth=500.0, fan=unpaired)
    with pytest.raises(ValueError, match="ranges_m"):
        gaussian_beams(200.0, *iso, source_depth=500.0, ranges_m=[-1.0])
    with pytest.raises(ValueError, match="receiver_depths_m"):
        gaussian_beams(
            200.0, *iso, source_depth=500.0, ranges_m=[500.0], receiver_depths_m=[]
        )
    with pytest.raises(ValueError, match="receiver_depths_m"):
        gaussian_beams(
            200.0,
            *iso,
            source_depth=500.0,
            ranges_m=[500.0],
            receiver_depths_m=[1200.0],
        )  # below the seabed
    with pytest.raises(ValueError, match="ranges_m"):
        # Past the march there is no column to read a beam off, and answering
        # by extrapolating the last one would be a silent wrong answer.
        gaussian_beams(
            200.0, *iso, source_depth=500.0, max_range=1000.0, ranges_m=[2000.0]
        )
    with pytest.raises(ValueError, match="n_depth_points"):
        gaussian_beams(
            200.0, *iso, source_depth=500.0, ranges_m=[500.0], n_depth_points=1
        )
    with pytest.raises(ValueError, match="absorption"):
        gaussian_beams(200.0, *iso, source_depth=500.0, absorption="mud")


def test_the_result_lines_up_with_the_parabolic_equation_grid_and_plots() -> None:
    """Same shape, same depths, and a ``.plot()`` in the sibling's own frame."""
    from phonometry.underwater.propagation.numerical import parabolic_equation

    iso = ([0.0, 1000.0], [_C, _C])
    beams = gaussian_beams(
        200.0,
        *iso,
        source_depth=500.0,
        max_range=2000.0,
        range_step=100.0,
        n_depth_points=32,
        fan=BeamFan(max_angle_deg=45.0, n_beams=101),
    )
    pe = parabolic_equation(
        200.0,
        *iso,
        source_depth=500.0,
        max_range=2000.0,
        range_step=100.0,
        n_depth_points=32,
    )
    assert np.allclose(beams.depths, pe.depths)
    assert beams.propagation_loss.shape == (32, beams.ranges.size)
    assert beams.propagation_loss.shape[0] == pe.propagation_loss.shape[0]
    with np.errstate(divide="ignore"):
        assert np.allclose(
            beams.propagation_loss, -20.0 * np.log10(np.abs(beams.pressure))
        )
    assert beams.plot() is not None
    assert beams.plot(language="es") is not None


def test_the_colour_window_lands_on_the_field_and_not_beside_it() -> None:
    """That a raster came out is not evidence that anything is visible in it.

    A loss field is bounded below by its strongest arrival and unbounded above,
    so hanging a fixed 50 dB window on a *high* percentile lets the empty part
    of the picture decide where the window sits. This is the case that shows it:
    the source sits 7.5 m off the bottom of an n^2-linear profile, the up-going
    fan turns into a caustic ladder, and 18 per cent of the field is the wedge
    no beam reaches at all. Anchored at the 95th percentile the window was
    [110, 160] dB and 85 per cent of the finite field fell below it and clipped
    to one flat colour, caustics and all.

    The assertion is on the fraction of the field the window actually resolves,
    which is what "the picture shows something" means in numbers.
    """
    c0 = 1550.0
    z = np.linspace(0.0, 1000.0, 201)
    c = c0 / np.sqrt(1.0 + 2.4 * z / c0)
    beams = gaussian_beams(
        600.0,
        z,
        c,
        source_depth=992.5,
        max_range=2500.0,
        range_step=25.0,
        fan=BeamFan(max_angle_deg=45.0),
        n_depth_points=80,
    )
    pl = beams.propagation_loss
    finite = pl[np.isfinite(pl)]
    assert np.isinf(pl).mean() > 0.1, "the un-illuminated wedge has to be there"
    ax = beams.plot()
    vmin, vmax = ax.get_images()[0].get_clim()
    assert vmax - vmin == pytest.approx(50.0)
    resolved = float(((finite > vmin) & (finite < vmax)).mean())
    assert resolved > 0.75, f"only {resolved:.1%} of the field is inside the window"
    # The bright end is where the window is pinned, so only the tail saturates.
    assert float((finite < vmin).mean()) < 0.1
