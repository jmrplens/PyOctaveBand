#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Tests for the eigenray search and its arrival list.

Every oracle here is built in the test from a published closed form and shares
nothing with the solver.

* **The isovelocity waveguide, by the method of images.** Two flat boundaries
  over constant sound speed unfold every path into a straight line to an image
  of the receiver, so the *entire* arrival structure is closed form: travel
  time :math:`\sqrt{r^2 + z_n^2}/c`, launch and arrival angles by plane
  geometry, amplitude :math:`1/R_n` with the boundary factors
  :math:`(-1)^{n_\mathrm{s}}` and the bottom coefficient raised to the touch
  count. That is one oracle for every field the result carries at once, and
  the count of eigenrays inside the fan's aperture is the count of images
  inside it, which pins completeness as well as accuracy. The lattice below
  is the receiver-image ladder written directly from the unfolding (the same
  bookkeeping Jensen Eq. (2.138) sums for the pressure), with the image
  angles kept two degrees clear of the fan's edge so the census is not a
  coin toss at the aperture.

* **The refracting guide.** The :math:`n^2`-linear profile of Jensen
  Eq. (3.77) integrates in closed form: with :math:`n^2` linear in depth the
  exact ray is the parabola of Eq. (3.195), so the upward-refracted
  eigenray's path is written down whole and its travel time
  :math:`\int \sqrt{1 + z'^2}/c(z)\,\mathrm{d}r` is a one-dimensional
  quadrature ``scipy`` drives to machine precision. What limits the
  comparison is not the oracle but the medium handed to the solver: the
  profile is *sampled*, and a ray flying through the piecewise-linear
  interpolant of a curved profile is a ray in a slightly different ocean.
  The measured gap (2e-7 s in time, 1e-4 degrees in angle at a 2 m sampling)
  is that discretisation, not the search; the bounds keep an order in hand.

The amplitude convention under test is stated in :func:`eigenrays`: classical
ray amplitude (Eq. 3.65) over the 1 m reference of Eqs. (3.67)-(3.68), caustic
factor :math:`(-i)^m` (Eq. 3.79), boundary factors per touch
(Eqs. 3.125-3.126), all in the :math:`e^{-i\omega t}` convention, so the
Rayleigh coefficient of a lossy seabed enters *as printed*, unconjugated,
which the below-critical arrivals of the seabed test would catch inverted:
there :math:`|\mathcal{R}| = 1` and only its phase carries the seabed.
"""

from __future__ import annotations

import functools
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from phonometry import PhonometryWarning
from phonometry.underwater.propagation.numerical import (
    EigenrayResult,
    FluidSeabed,
    _caustic_crossings,
    eigenrays,
    ray_trace,
)

_C = 1500.0
#: The isovelocity guide every image-lattice test below runs in.
_GUIDE = {"depth": 100.0, "zs": 36.0, "zr": 46.0, "r": 500.0, "aperture": 48.0}


def _image_arrivals(
    *, depth: float, zs: float, zr: float, r: float, aperture: float,
) -> list[tuple[float, float, float, int, int]]:
    """Every arrival of the ideal waveguide inside the fan's aperture.

    The receiver's images stand at ``2 l D + zr`` (upright, ``|l|`` surface
    and ``|l|`` bottom planes between image and receiver) and ``2 l D - zr``
    (mirrored, one count up on whichever boundary did the extra fold), and
    each eigenray is the straight line from the source to one image: travel
    time by Pythagoras over the sound speed, launch angle by arctangent,
    arrival angle the launch angle with the sign flipped once per boundary
    touch (a specular reflection reverses the vertical direction and nothing
    else in isovelocity water). Sorted by travel time, as the solver sorts.
    Returns ``(time, launch_deg, arrival_deg, n_surface, n_bottom)`` tuples;
    amplitudes are formed in the tests from these counts so each test can
    apply its own bottom coefficient.
    """
    out = []
    for ell in range(-6, 7):
        for side, (n_s, n_b) in (
            (+1, (abs(ell), abs(ell))),
            (-1, ((ell - 1, ell) if ell >= 1 else (abs(ell) + 1, abs(ell)))),
        ):
            dz = 2.0 * ell * depth + side * zr - zs
            angle = float(np.degrees(np.arctan2(dz, r)))
            if abs(angle) >= aperture:
                continue
            slant = float(np.hypot(r, dz))
            arrival = angle if (n_s + n_b) % 2 == 0 else -angle
            out.append((slant / _C, angle, arrival, n_s, n_b))
    out.sort()
    return out


def _guide_trace(aperture: float = 48.0, step_deg: float = 0.5):
    return ray_trace(
        [0.0, _GUIDE["depth"]], [_C, _C], source_depth=_GUIDE["zs"],
        launch_angles_deg=np.arange(-aperture, aperture + 1e-9, step_deg),
        max_range=600.0, n_steps=201)


#: Refinement steps for the isovelocity searches. Straight rays are exact at
#: any step count (the stages reproduce a line digit for digit and every
#: bounce is polished onto its plane), so a coarse march buys the same digits
#: for a fraction of the marching; the main oracle's 1e-9/1e-6 bounds are the
#: proof it costs nothing.
_GUIDE_STEPS = 64


@functools.lru_cache(maxsize=1)
def _guide_arrivals() -> tuple[object, EigenrayResult]:
    """The shared trace and pressure-release arrival list, computed once."""
    trace = _guide_trace()
    return trace, eigenrays(trace, receiver_range=_GUIDE["r"],
                            receiver_depth=_GUIDE["zr"], n_steps=_GUIDE_STEPS)


# --- The image-method oracle ------------------------------------------------


def test_the_ideal_waveguide_arrivals_are_the_image_lattice_exactly() -> None:
    """Count, times, angles, amplitudes and bounce counts, all at once.

    The required standard (microseconds, millidegrees, a fraction of a
    percent) is met with three to six orders in hand, because in isovelocity
    water everything the search does is exact: the rays are straight, the
    Runge-Kutta stages reproduce them digit for digit, the boundary crossings
    are polished onto the reflection planes, and the dynamic pair integrates
    to ``q = r / cos(theta0)``, which turns Eq. (3.65) into ``1/R_n`` on the
    nose. Measured: 1.2e-13 s, 1.4e-11 degrees, 2.4e-13 relative. What is
    left is the bisection's convergence and float64 accumulation, so a
    failure at these bounds is a wrong formula, not a loose one.
    """
    exact = _image_arrivals(
        depth=_GUIDE["depth"], zs=_GUIDE["zs"], zr=_GUIDE["zr"],
        r=_GUIDE["r"], aperture=_GUIDE["aperture"])
    _trace, res = _guide_arrivals()
    assert isinstance(res, EigenrayResult)
    # The census: every image inside the aperture and nothing else. Eleven
    # arrivals, the nearest to the fan's edge two degrees inside it.
    assert res.travel_times.size == len(exact) == 11
    assert np.all(np.diff(res.travel_times) > 0.0)  # earliest first
    for i, (t, launch, arrival, n_s, n_b) in enumerate(exact):
        slant = t * _C
        assert res.travel_times[i] == pytest.approx(t, abs=1e-9)
        assert res.launch_angles[i] == pytest.approx(launch, abs=1e-6)
        assert res.arrival_angles[i] == pytest.approx(arrival, abs=1e-6)
        assert int(res.surface_reflections[i]) == n_s
        assert int(res.bottom_reflections[i]) == n_b
        amp = (-1.0) ** (n_s + n_b) / slant  # pressure-release both ways
        assert res.amplitudes[i] == pytest.approx(amp, rel=1e-9)
    # Straight rays cannot form caustics: the KMAH index must be zero for
    # every path, or the amplitude phase would be turned without cause.
    assert np.all(res.caustic_crossings == 0)
    assert res.receiver_range == _GUIDE["r"]
    assert res.receiver_depth == _GUIDE["zr"]
    assert res.water_depth == _GUIDE["depth"]


def test_a_rigid_bottom_flips_exactly_the_bottom_signs() -> None:
    """Same geometry, so same times bit for bit; only ``(+1)^n_b`` differs.

    The bottom's coefficient touches amplitudes alone: a specular reflection
    is the same fold whatever the boundary is made of, so every geometric
    field of the result must come out identical to the pressure-release run,
    to the last bit, and the amplitudes must differ by ``(-1)^n_b`` exactly
    (the rigid coefficient is +1 where the pressure-release one was -1).
    """
    trace, soft = _guide_arrivals()
    rigid = eigenrays(trace, receiver_range=_GUIDE["r"],
                      receiver_depth=_GUIDE["zr"], bottom="rigid",
                      n_steps=_GUIDE_STEPS)
    assert np.array_equal(rigid.travel_times, soft.travel_times)
    assert np.array_equal(rigid.launch_angles, soft.launch_angles)
    assert np.array_equal(rigid.arrival_angles, soft.arrival_angles)
    np.testing.assert_allclose(
        rigid.amplitudes,
        soft.amplitudes * (-1.0) ** soft.bottom_reflections, rtol=0.0, atol=0.0)


def test_a_lossy_seabed_charges_R_at_each_arrivals_own_angle_as_printed() -> None:
    r"""The Rayleigh coefficient per arrival, magnitude and phase, unconjugated.

    The oracle applies :func:`reflection_coefficient` directly to each image
    path's own grazing angle (in isovelocity water the straight unfolded line
    meets every bottom plane at its launch inclination), raised to the touch
    count: Jensen Eq. (3.126) at every touch, collapsed to a power. The
    sand-like seabed's critical angle is 28.07 degrees and the eleven
    arrivals straddle it, which is the point of the geometry: above it
    :math:`|\mathcal{R}| < 1` and the amplitude magnitudes must shrink below
    the perfect reflector's, below it :math:`|\mathcal{R}| = 1` and *only*
    the phase carries the seabed, so a conjugated coefficient would leave
    every magnitude in this test untouched and turn every below-critical
    phase backwards. The geometry, again, must not move at all.
    """
    from phonometry.underwater.propagation.seabed_reflection import (
        reflection_coefficient,
    )

    rho1, rho2, c2 = 1000.0, 1800.0, 1700.0
    trace, soft = _guide_arrivals()
    lossy = eigenrays(trace, receiver_range=_GUIDE["r"],
                      receiver_depth=_GUIDE["zr"],
                      bottom=FluidSeabed(density=rho2, sound_speed=c2,
                                         water_density=rho1),
                      n_steps=_GUIDE_STEPS)
    assert np.array_equal(lossy.travel_times, soft.travel_times)
    exact = _image_arrivals(
        depth=_GUIDE["depth"], zs=_GUIDE["zs"], zr=_GUIDE["zr"],
        r=_GUIDE["r"], aperture=_GUIDE["aperture"])
    below = above = 0
    for i, (t, launch, _arrival, n_s, n_b) in enumerate(exact):
        slant = t * _C
        r_b = complex(np.ravel(reflection_coefficient(
            abs(launch), rho1=rho1, c1=_C, rho2=rho2, c2=c2))[0])
        amp = (-1.0) ** n_s * r_b ** n_b / slant
        assert lossy.amplitudes[i] == pytest.approx(amp, rel=1e-9), launch
        if n_b and abs(launch) < 28.0:
            below += 1
            # Total reflection: the seabed lives in the phase alone here.
            assert abs(lossy.amplitudes[i]) == pytest.approx(
                abs(soft.amplitudes[i]), rel=1e-9)
            assert abs(np.angle(lossy.amplitudes[i] / soft.amplitudes[i])) > 0.1
        elif n_b:
            above += 1
            assert abs(lossy.amplitudes[i]) < abs(soft.amplitudes[i]) * 0.999
    assert below >= 2, "the total-reflection regime must actually be exercised"
    assert above >= 2, "the lossy above-critical regime must actually be exercised"


def test_the_cap_keeps_the_earliest_arrivals_and_says_so() -> None:
    """``max_arrivals`` truncates the sorted tail and warns, changing nothing
    about the arrivals it keeps."""
    trace, full = _guide_arrivals()
    with pytest.warns(PhonometryWarning, match="11 eigenrays.*6 earliest"):
        capped = eigenrays(trace, receiver_range=_GUIDE["r"],
                           receiver_depth=_GUIDE["zr"], max_arrivals=6,
                           n_steps=_GUIDE_STEPS)
    assert capped.travel_times.size == 6
    assert np.array_equal(capped.travel_times, full.travel_times[:6])
    assert np.array_equal(capped.amplitudes, full.amplitudes[:6])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        uncapped = eigenrays(trace, receiver_range=_GUIDE["r"],
                             receiver_depth=_GUIDE["zr"], max_arrivals=11,
                             n_steps=_GUIDE_STEPS)
    assert uncapped.travel_times.size == 11


def test_an_unreached_receiver_has_no_eigenrays_and_that_is_an_answer() -> None:
    """A receiver outside the fan's reach returns an empty list, not an error.

    A five-degree fan from 36 m spans 18 to 54 m of depth at 200 m of range,
    so a receiver at 80 m is simply not illuminated by the traced aperture:
    no bracket, no arrivals, and the empty result still knows its geometry
    and still plots.
    """
    trace = ray_trace([0.0, _GUIDE["depth"]], [_C, _C],
                      source_depth=_GUIDE["zs"],
                      launch_angles_deg=np.linspace(-5.0, 5.0, 21),
                      max_range=250.0, n_steps=101)
    res = eigenrays(trace, receiver_range=200.0, receiver_depth=80.0)
    assert res.travel_times.size == 0
    assert res.amplitudes.size == 0
    assert res.receiver_depth == 80.0
    assert res.plot() is not None
    plt.close("all")


# --- The refracting oracle --------------------------------------------------
#
# Jensen Eq. (3.77) scaled as printed: c(z) = c0 / sqrt(1 + 2.4 z / c0) with
# c0 = 1550 m/s, so n^2 = 1 + g z with g = 2.4 / c0 and the exact ray is the
# parabola of Eq. (3.195),
#
#     z(r) = zs + tan(theta0) r + (g / 4) (r / (n0 cos theta0))^2 ,
#
# every coefficient launch data. The travel time along it is
# int sqrt(1 + z'(r)^2) / c(z(r)) dr, which scipy integrates to its own
# roundoff (quad reports ~1e-14); none of it comes from the solver.
_N2_C0, _N2_DEPTH, _N2_SOURCE = 1550.0, 1000.0, 900.0
_N2_G = 2.4 / _N2_C0


def _n2_speed(z: np.ndarray) -> np.ndarray:
    return np.asarray(_N2_C0 / np.sqrt(1.0 + _N2_G * z))


def _n2_parabola(theta0_deg: float):
    """The exact ray as callables ``z(r)``, ``z'(r)`` (Eq. 3.195)."""
    th = np.radians(theta0_deg)
    n0 = np.sqrt(1.0 + _N2_G * _N2_SOURCE)
    a2 = _N2_G / 4.0 / (n0 * np.cos(th)) ** 2

    def z(r: np.ndarray | float) -> np.ndarray | float:
        return _N2_SOURCE + np.tan(th) * r + a2 * np.asarray(r) ** 2

    def zp(r: np.ndarray | float) -> np.ndarray | float:
        return np.tan(th) + 2.0 * a2 * np.asarray(r)

    return z, zp


def test_the_upward_refracted_eigenray_matches_the_closed_form_parabola() -> None:
    """Travel time against scipy on the exact path, launch and arrival angles
    against the parabola's own tangents.

    The eigenray is launched at -11.13 degrees (off the fan's half-degree
    rungs on purpose, so the bracket is an interior one and this test stands
    on the plain bisection), climbs 29 m to its apex at 842 m and refracts
    back down to the receiver without touching either boundary. Measured
    against the closed form: 2.0e-7 s in time, 1.1e-4 degrees in launch,
    4.2e-4 in arrival, all of it the piecewise-linear sampling of the smooth
    profile (2 m here) plus the marcher's node-crossing resolution rather
    than the search; the bounds keep an order in hand. The bounce counts and
    the KMAH index must be exactly zero: the path turns above the seabed,
    and a smooth turning point is not a caustic, so a nonzero count here
    would be the amplitude phase turning without cause.
    """
    from scipy.integrate import quad

    theta0 = -11.13
    r_rec = 1000.0
    z_path, zp = _n2_parabola(theta0)
    z_rec = float(z_path(r_rec))
    t_exact = quad(
        lambda r: float(np.sqrt(1.0 + zp(r) ** 2) / _n2_speed(np.asarray(z_path(r)))),
        0.0, r_rec, epsabs=1e-14, epsrel=1e-13, limit=400)[0]

    depths = np.arange(0.0, _N2_DEPTH + 1.0, 2.0)
    trace = ray_trace(depths, _n2_speed(depths), source_depth=_N2_SOURCE,
                      launch_angles_deg=np.arange(-14.0, -7.999, 0.5),
                      max_range=1100.0, n_steps=201)
    res = eigenrays(trace, receiver_range=r_rec, receiver_depth=z_rec,
                    n_steps=501)
    assert res.travel_times.size == 1
    assert res.travel_times[0] == pytest.approx(t_exact, abs=2e-6)
    assert res.launch_angles[0] == pytest.approx(theta0, abs=1e-3)
    arrival_exact = float(np.degrees(np.arctan(zp(r_rec))))
    assert arrival_exact > 0.0  # past the apex: arriving downward again
    assert res.arrival_angles[0] == pytest.approx(arrival_exact, abs=2e-3)
    assert int(res.surface_reflections[0]) == 0
    assert int(res.bottom_reflections[0]) == 0
    assert int(res.caustic_crossings[0]) == 0
    # No touch, no caustic: the amplitude is the bare Eq. (3.65), real and
    # positive in this convention.
    assert res.amplitudes[0].imag == 0.0
    assert res.amplitudes[0].real > 0.0


def test_a_root_on_a_fan_rung_is_recovered_by_the_widened_bracket() -> None:
    """The rescue path, exercised deterministically.

    With the receiver placed on the closed-form parabola of a launch angle
    that *is* a fan rung (-11.0 degrees), the true root in the sampled medium
    sits a few ten-thousandths of a degree to one side of the rung, and the
    fan's own reading (traced at its own step) places the bracket on the
    other: the fresh endpoint marches then agree in sign and the bracket
    would be disbelieved. Verified by instrumentation for exactly this
    configuration: the endpoint check fails, the bracket is widened by one
    rung each way, and bisection recovers the eigenray at -11.0004 degrees.
    Losing it silently is the failure mode this guards against.
    """
    z_path, _zp = _n2_parabola(-11.0)
    z_rec = float(z_path(1000.0))
    depths = np.arange(0.0, _N2_DEPTH + 2.0, 4.0)
    trace = ray_trace(depths, _n2_speed(depths), source_depth=_N2_SOURCE,
                      launch_angles_deg=np.arange(-14.0, -7.999, 0.5),
                      max_range=1100.0, n_steps=201)
    res = eigenrays(trace, receiver_range=1000.0, receiver_depth=z_rec,
                    n_steps=301)
    assert res.travel_times.size == 1
    assert res.launch_angles[0] == pytest.approx(-11.0, abs=5e-3)


# --- The pieces on their own ------------------------------------------------


def test_the_caustic_counter_reads_sign_changes_and_nothing_else() -> None:
    """White box on the KMAH bookkeeping of Eq. (3.79).

    The launch sample is q = 0 by the initial conditions of Eq. (3.63) and
    must not count; an interior zero is a caustic the march happened to land
    a sample on, and counts once, not twice. No field-level oracle reaches
    this cheaply (a caustic needs a refracting run long enough to fold the
    fan, tens of seconds of marching), so the counter is pinned directly.
    """
    rows = np.array([
        [0.0, 1.0, 2.0, 3.0, 4.0],    # monotone: no caustic
        [0.0, 1.0, 0.5, -0.5, -1.0],  # one crossing between samples
        [0.0, 1.0, 0.0, -1.0, 1.0],   # a crossing sampled exactly, then back
        [0.0, 2.0, -1.0, 1.0, -2.0],  # three crossings
    ])
    assert _caustic_crossings(rows).tolist() == [0, 1, 2, 3]


def test_the_trace_records_the_profile_it_flew_through() -> None:
    trace, _res = _guide_arrivals()
    np.testing.assert_array_equal(trace.profile_depths, [0.0, _GUIDE["depth"]])
    np.testing.assert_array_equal(trace.profile_speeds, [_C, _C])


def test_invalid_inputs_rejected() -> None:
    trace, _res = _guide_arrivals()
    with pytest.raises(ValueError, match="receiver_depth"):
        eigenrays(trace, receiver_range=500.0, receiver_depth=0.0)
    with pytest.raises(ValueError, match="receiver_range"):
        eigenrays(trace, receiver_range=700.0, receiver_depth=46.0)
    with pytest.raises(ValueError, match="receiver_range"):
        eigenrays(trace, receiver_range=0.0, receiver_depth=46.0)
    with pytest.raises(ValueError, match="max_arrivals"):
        eigenrays(trace, receiver_range=500.0, receiver_depth=46.0,
                  max_arrivals=0)
    with pytest.raises(ValueError, match="n_steps"):
        eigenrays(trace, receiver_range=500.0, receiver_depth=46.0, n_steps=1)
    slow = FluidSeabed(density=1800.0, sound_speed=-1700.0)
    with pytest.raises(ValueError, match="sound_speed"):
        eigenrays(trace, receiver_range=500.0, receiver_depth=46.0,
                  bottom=slow)
    with pytest.raises(ValueError, match="bottom"):
        eigenrays(trace, receiver_range=500.0, receiver_depth=46.0,
                  bottom="sandy")
    lone = ray_trace([0.0, 100.0], [_C, _C], source_depth=36.0,
                     launch_angles_deg=[10.0], max_range=600.0, n_steps=101)
    with pytest.raises(ValueError, match="at least two rays"):
        eigenrays(lone, receiver_range=500.0, receiver_depth=46.0)


def test_the_arrival_plot_is_the_impulse_response_picture() -> None:
    """Stems against delay, loss increasing downward, both languages."""
    _trace, res = _guide_arrivals()
    ax = res.plot()
    assert ax.get_title() == "Eigenray arrivals"
    assert ax.get_xlabel() == "Travel time [s]"
    assert ax.get_ylabel() == "Propagation loss [dB]"
    assert ax.yaxis_inverted()
    ax_es = res.plot(language="es")
    assert ax_es.get_title() == "Llegadas de eigenrayos"
    assert ax_es.get_xlabel() == "Tiempo de propagación [s]"
    plt.close("all")
