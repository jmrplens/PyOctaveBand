#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Independent FDTD cross-checks of the reactive silencers.

The four-pole expansion-chamber transmission loss (Bies Eq. (8.111)) predicts,
for an area ratio ``m = S_exp / S_duct``, a ``TL`` peak of
``10 log10[1 + (1/4)(m - 1/m)^2]`` at ``kL = pi/2`` (``f = c / 4L``) and a
``TL = 0`` trough (fully transparent) at ``kL = pi`` (``f = c / 2L``).

These tests drive an *independent* solver -- the 2D FDTD engine of
:mod:`phonometry.simulation` -- with the same geometry the four-pole model
describes, and compare the two.

The expansion chamber is driven with a steady tone: a plane-wave duct that
widens into a chamber and narrows back, sampled at the peak and trough
frequencies. The measured ``20 log10`` ratio of the transmitted amplitudes
(trough over peak) must reproduce the closed-form peak ``TL``. The run is
deterministic and small (a 16x200 grid, ~3000 steps, well under a second).

The Helmholtz side branch is driven with a broadband pulse instead, because
the quantity to compare is a *resonance frequency* and a tone sweep would need
one run per frequency. One pulse run of the bare duct and one of the duct with
the resonator give the whole transmission spectrum
``|p_resonator(f)| / |p_bare(f)|``, that is the insertion loss, which for equal
inlet and outlet areas and anechoic ends is the transmission loss. The point
sources are additive (soft), so the wave the resonator reflects back passes
through the source cell untouched and is swallowed by the upstream sponge: the
source really does present the anechoic internal impedance that identity needs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from phonometry.noise_control import silencers as sl
from phonometry.simulation.fdtd import CWSource, GaussianPulse, fdtd_simulation

_C = 343.0

# --- Simple expansion chamber, driven with a steady tone --------------------

_DX = 0.02
_NY, _NX = 16, 200
_DUCT_ROWS = (6, 7, 8, 9)      # 4-cell duct; chamber is the full 16 rows -> m = 4
_X1, _X2 = 80, 100             # chamber columns
_L = (_X2 - _X1) * _DX         # chamber length, 0.4 m
_M = _NY / len(_DUCT_ROWS)     # area (height) ratio = 4


def _chamber_mask() -> np.ndarray:
    mask = np.ones((_NY, _NX), dtype=bool)
    for r in _DUCT_ROWS:
        mask[r, :] = False     # open the narrow inlet/outlet duct
    mask[:, _X1:_X2] = False    # open the full-height chamber
    return mask


def _transmitted_rms(frequency: float) -> float:
    mask = _chamber_mask()
    boundaries = {
        "left": "absorbing", "right": "absorbing",
        "top": "rigid", "bottom": "rigid",
    }
    src = CWSource(ix=25, iy=7, frequency=frequency, amplitude=1.0, ramp_cycles=6.0)
    steps = 3000
    dt = 0.6 * _DX / (_C * math.sqrt(2.0))
    res = fdtd_simulation(
        _C, _DX, steps * dt, sources=[src], shape=(_NY, _NX),
        probes=[(170, 7)], boundaries=boundaries,
        absorbing_layer_cells=10, obstacle_mask=mask,
    )
    p = res.pressures[0]
    tail = p[int(p.size * 0.6):]        # steady-state window
    return float(np.sqrt(np.mean(tail**2)))


def test_fdtd_matches_expansion_chamber_peak_tl() -> None:
    f_peak = _C / (4.0 * _L)    # kL = pi/2, four-pole TL maximum
    f_trough = _C / (2.0 * _L)  # kL = pi, four-pole TL = 0 (transparent)

    down_peak = _transmitted_rms(f_peak)
    down_trough = _transmitted_rms(f_trough)

    # The chamber transmits more at the trough than at the peak.
    assert down_trough > down_peak

    # The measured attenuation difference reproduces the closed-form peak TL.
    measured_delta = 20.0 * math.log10(down_trough / down_peak)
    peak_tl = 10.0 * math.log10(1.0 + 0.25 * (_M - 1.0 / _M) ** 2)
    assert peak_tl == pytest.approx(6.55, abs=0.05)
    assert measured_delta == pytest.approx(peak_tl, abs=1.2)


def test_fdtd_run_is_deterministic() -> None:
    # Two independent runs of the (RNG-free) solver must be bit-identical.
    f = _C / (4.0 * _L)
    first = _transmitted_rms(f)
    second = _transmitted_rms(f)
    assert first == second


def test_four_pole_peak_frequency_and_trough() -> None:
    # The four-pole model itself: TL peak at f = c/4L, TL = 0 at f = c/2L.
    f = np.array([_C / (4.0 * _L), _C / (2.0 * _L)])
    res = sl.expansion_chamber(f, _L, _M * 0.01, 0.01)
    assert res.transmission_loss[0] == pytest.approx(
        10.0 * math.log10(1.0 + 0.25 * (_M - 1.0 / _M) ** 2), abs=1e-6
    )
    assert res.transmission_loss[1] == pytest.approx(0.0, abs=1e-9)


# --- Helmholtz side branch, driven with a broadband pulse -------------------
#
# Geometry, in cells of _HDX, stacked from the top of the grid downwards: the
# cavity, then the neck, then the main duct that runs the full grid length.
# Every plane of the lumped model falls on a cell face, so the mapping carries
# no rounding:
#
#   * neck area      S_neck = _HNECK_W * _HDX      (per metre of 2D depth)
#   * duct area      S_duct = _HDUCT * _HDX
#   * cavity volume  V      = _HCAV_D * width * _HDX^2
#   * neck geometric length l_g = _HNECK_L * _HDX, measured between the plane
#     flush with the duct wall and the plane flush with the cavity floor.
#
# The one quantity the drawing cannot fix is the *effective* neck length that
# `helmholtz_resonator` consumes: the evanescent field at each open end of the
# neck adds inertance, so l_e > l_g, and there is no closed form for a 2D slit
# opening into a duct on one side and a box on the other. The tests below
# therefore never assume a value for it. They bracket it, they check that it is
# a property of the neck alone (the assertion that carries the 1/sqrt(V) law of
# the Helmholtz formula), and only then do they hand the FDTD-measured
# resonance back as l_e to compare the whole transmission-loss curve, whose
# width and level are still predictions.

_HDX = 0.01
_HDUCT = 8          # main duct height, cells
_HNECK_W = 3        # neck width, cells
_HNECK_L = 8        # neck geometric length, cells
_HCAV_D = 14        # cavity depth along the neck axis, cells (both cavities)
_HCAV_NARROW = 14   # cavity width, cells
_HCAV_WIDE = 28     # the same cavity, twice the volume
_HNY = _HCAV_D + _HNECK_L + _HDUCT      # 30 rows
_HNX = 380
_HNECK_X = _HNX // 2 - 40               # first neck column
_HSTEPS = 5000
_HSPONGE = 20
_HSRC = (45, _HNY - 4)                  # on the duct axis, upstream
_HPROBE = (_HNX - 60, _HNY - 4)         # on the duct axis, downstream
_HPAD = 16                              # FFT zero-padding factor

_HS_DUCT = _HDUCT * _HDX
_HS_NECK = _HNECK_W * _HDX


def _h_volume(cavity_width: int) -> float:
    """Cavity volume per metre of 2D depth, m3/m."""
    return _HCAV_D * cavity_width * _HDX * _HDX


def _h_nominal_resonance(cavity_width: int) -> float:
    """Rough scale used to size the stimulus and the analysis band.

    Deliberately independent of the library, so that the excitation the two
    routes are compared over never depends on the model under test. The
    assertions use :func:`_h_lumped_resonance` instead.
    """
    volume = _h_volume(cavity_width)
    return (_C / (2.0 * math.pi)
            * math.sqrt(_HS_NECK / (_HNECK_L * _HDX * volume)))


def _h_lumped_resonance(neck_length: float, cavity_width: int) -> float:
    """The library's own ``f_0`` for this geometry and effective neck length."""
    result = sl.helmholtz_resonator(
        np.array([100.0]), _HS_DUCT, _HS_NECK, neck_length,
        _h_volume(cavity_width), speed_of_sound=_C,
    )
    assert result.resonances is not None
    return float(result.resonances[0])


def _helmholtz_mask(cavity_width: int | None) -> NDArray[np.bool_]:
    """Rigid-cell map; ``None`` gives the bare duct used as the reference."""
    mask = np.ones((_HNY, _HNX), dtype=bool)
    mask[_HCAV_D + _HNECK_L:, :] = False            # the straight main duct
    if cavity_width is None:
        return mask
    mask[_HCAV_D:_HCAV_D + _HNECK_L,
         _HNECK_X:_HNECK_X + _HNECK_W] = False      # the neck
    left = _HNECK_X + _HNECK_W // 2 - cavity_width // 2
    mask[:_HCAV_D, left:left + cavity_width] = False    # the cavity
    return mask


def _h_transmission(cavity_width: int | None) -> tuple[
    NDArray[np.float64], NDArray[np.complex128]
]:
    """Probe spectrum of one pulse run: ``(frequencies, P(f))``."""
    dt = 0.6 * _HDX / (_C * math.sqrt(2.0))
    # A Gaussian wide enough to cover DC up to about twice the resonance.
    width = 1.0 / (4.0 * _h_nominal_resonance(_HCAV_NARROW))
    res = fdtd_simulation(
        _C, _HDX, _HSTEPS * dt,
        sources=[GaussianPulse(ix=_HSRC[0], iy=_HSRC[1], width=width)],
        shape=(_HNY, _HNX), probes=[_HPROBE],
        boundaries={"left": "absorbing", "right": "absorbing",
                    "top": "rigid", "bottom": "rigid"},
        absorbing_layer_cells=_HSPONGE,
        obstacle_mask=_helmholtz_mask(cavity_width),
    )
    p = res.pressures[0]
    n = p.size * _HPAD
    return np.fft.rfftfreq(n, dt), np.fft.rfft(p, n)


@pytest.fixture(scope="module")
def helmholtz_spectra() -> dict[int, tuple[NDArray[np.float64],
                                           NDArray[np.float64]]]:
    """One bare-duct run plus one run per cavity: transmitted pressure ratio.

    Three FDTD runs in total (~1.7 s), shared by every test below.
    """
    freq, bare = _h_transmission(None)
    out = {}
    for width in (_HCAV_NARROW, _HCAV_WIDE):
        _, loaded = _h_transmission(width)
        nominal = _h_nominal_resonance(width)
        band = (freq > 0.45 * nominal) & (freq < 1.45 * nominal)
        out[width] = (freq[band],
                      np.abs(loaded[band]) / np.abs(bare[band]))
    return out


def _h_resonance(freq: NDArray[np.float64],
                 ratio: NDArray[np.float64]) -> float:
    """Resonance from the simple zero of the transmitted amplitude.

    A lossless side branch shorts the duct at ``f_0``, so ``|p/p_bare|`` has a
    simple zero there and its square is locally parabolic. Fitting the parabola
    over the seven bins around the minimum locates the zero far below the bin
    width and, unlike reading off the transmission-loss maximum, does not
    depend on how close a bin happens to fall to a near-singularity.
    """
    i = min(max(int(np.argmin(ratio)), 3), ratio.size - 4)
    window = slice(i - 3, i + 4)
    a, b, _ = np.polyfit(freq[window], ratio[window] ** 2, 2)
    return float(-b / (2.0 * a))


@pytest.mark.xdist_group("fdtd-helmholtz")
@pytest.mark.parametrize("width", [_HCAV_NARROW, _HCAV_WIDE])
def test_fdtd_helmholtz_resonance_within_end_correction_bracket(
    helmholtz_spectra: dict[int, tuple[NDArray[np.float64],
                                       NDArray[np.float64]]],
    width: int,
) -> None:
    # Every effect the lumped model leaves out pushes the resonance *down*:
    # the evanescent field at both ends of the neck adds inertance, and a
    # cavity of finite depth is slightly more compliant than its volume alone
    # (tan(kD)/kD > 1). So `helmholtz_resonator` fed the bare geometric neck
    # length is a rigorous upper bound on what the simulation can show. For the
    # lower bound, charge each open end of the slit a full neck width of end
    # correction, which is generous: a 2D slit of width w opening into a
    # channel of width W gains an inertial length of order (w/pi) ln(W/w),
    # about 0.3 w on the duct side (W/w = 2.7) and 0.7 w on the cavity side
    # (W/w = 4.7 and 9.3) here.
    # Measured: 1.38 w and 1.36 w of total end correction, so the resonance
    # lands at 0.812 of the geometric-length bound for both cavities.
    freq, ratio = helmholtz_spectra[width]
    measured = _h_resonance(freq, ratio)
    geometric = _h_lumped_resonance(_HNECK_L * _HDX, width)
    corrected = _h_lumped_resonance(
        (_HNECK_L + 2 * _HNECK_W) * _HDX, width
    )
    assert corrected < measured < geometric


@pytest.mark.xdist_group("fdtd-helmholtz")
def test_fdtd_helmholtz_follows_inverse_root_volume(
    helmholtz_spectra: dict[int, tuple[NDArray[np.float64],
                                       NDArray[np.float64]]],
) -> None:
    # The content of the Helmholtz formula that survives the end-correction
    # ambiguity: with the neck untouched, f_0 goes as 1 / sqrt(V). The wide
    # cavity is the narrow one with its width doubled and its depth kept, so
    # the volume doubles while the neck, and therefore l_e, does not.
    #
    # Error budget for the 1 % bound. Grid dispersion is negligible here: at
    # 177 cells per wavelength the leapfrog scheme under-reads the frequency by
    # (k dx)^2 / 24 = 5.3e-5, that is 0.005 %. The numerical spread of the
    # measurement is of the same order: sweeping the grid length over
    # 380-500 cells, the record over 5000-8000 steps and the zero-padding over
    # 4-16 moves the ratio by less than 0.05 %. What is left is physical, the
    # slight change of the cavity-side end correction when the cavity widens,
    # and it measures 0.19 % (1.4115 against sqrt(2)). One percent leaves a
    # factor of five of headroom and still rejects any other exponent by a wide
    # margin: V^(-1/3) would give 1.26 and V^(-1) would give 2.
    narrow = _h_resonance(*helmholtz_spectra[_HCAV_NARROW])
    wide = _h_resonance(*helmholtz_spectra[_HCAV_WIDE])
    # The library's own prediction of the same ratio. Any effective neck length
    # serves: it is common to both cavities and cancels, which is exactly why
    # the ratio is the part of the formula the simulation can pin down.
    length = _HNECK_L * _HDX
    predicted = (_h_lumped_resonance(length, _HCAV_NARROW)
                 / _h_lumped_resonance(length, _HCAV_WIDE))
    assert _h_volume(_HCAV_WIDE) == pytest.approx(2.0 * _h_volume(_HCAV_NARROW))
    assert predicted == pytest.approx(math.sqrt(2.0), rel=1e-12)
    assert narrow / wide == pytest.approx(predicted, rel=0.01)


@pytest.mark.xdist_group("fdtd-helmholtz")
@pytest.mark.parametrize("width", [_HCAV_NARROW, _HCAV_WIDE])
def test_fdtd_helmholtz_transmission_loss_curve(
    helmholtz_spectra: dict[int, tuple[NDArray[np.float64],
                                       NDArray[np.float64]]],
    width: int,
) -> None:
    # With the single unknown pinned -- the effective neck length read back
    # from the measured resonance, l_e = S_neck c^2 / (4 pi^2 f_0^2 V) -- the
    # rest of `helmholtz_resonator` is still a prediction: the width of the
    # notch is set by c S_neck / (S_duct l_e) and its level by the shunt and
    # transmission-loss formulae. Both are compared from 0.6 to 1.4 times the
    # resonance, stepping around the resonance itself, where a lossless branch
    # shorts the duct and the modelled transmission loss is singular.
    #
    # The 1.5 dB bound is a modelling difference, not a numerical one. The 1D
    # model treats the cavity as a pure compliance, but at resonance the cavity
    # spans kD = 0.50 rad, so its reactance departs from the lumped value by
    # about (kD)^2 / 3 = 8 %, and 8.7 ln(1.08) = 0.7 dB of transmission loss on
    # the flanks of the notch; the same argument on the neck, kl_e = 0.43 rad,
    # adds a comparable term. Grid dispersion contributes 0.005 % of frequency
    # and nothing measurable in dB. Measured: at most 1.00 dB for the narrow
    # cavity and 0.33 dB for the wide one.
    freq, ratio = helmholtz_spectra[width]
    resonance = _h_resonance(freq, ratio)
    volume = _h_volume(width)
    neck_length = (
        _HS_NECK * _C**2 / (4.0 * math.pi**2 * resonance**2 * volume)
    )

    probe = resonance * np.array(
        [0.6, 0.7, 0.8, 0.9, 0.95, 1.05, 1.1, 1.2, 1.3, 1.4]
    )
    analytic = sl.helmholtz_resonator(
        probe, _HS_DUCT, _HS_NECK, neck_length, volume, speed_of_sound=_C,
    ).transmission_loss
    measured = np.array([
        -20.0 * math.log10(ratio[int(np.argmin(np.abs(freq - f)))])
        for f in probe
    ])
    assert measured == pytest.approx(analytic, abs=1.5)

    # And the branch really does short the duct: a deep notch, not a dent.
    assert measured.max() > 10.0
    assert -20.0 * math.log10(ratio.min()) > 25.0
