#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the image-source room impulse response (Kuttruff 4.1 / Vorlander 11.4).

Oracles, from strongest to softest:

* **Geometry (exact).** The direct-sound and early-reflection arrival times
  ``r / c`` and amplitudes ``prod R^n / (4 pi r)`` are exact geometric
  quantities, checked to machine precision.
* **Audible image count (exact).** A shoebox has ``(2/3)(2 i0^3 + 3 i0^2 +
  4 i0)`` audible images up to order ``i0`` (Kuttruff Equation (9.23)); the
  model sums exactly that many reflections plus the direct source.
* **Reflection density (asymptotic).** The arrival histogram follows
  ``dN/dt = 4 pi c^3 t^2 / V`` (Kuttruff Equation (4.6)).
* **Eyring relation (statistical, documented tolerance).** The initial decay
  rate of the reverberant energy density of a *near-cubic* uniform room
  reproduces the Eyring reverberation time (Kuttruff Equation (5.23)) to within
  about 10 %; an elongated room decays more slowly because the specular field
  is anisotropic (the regime the Fitzroy / Arau-Puchades models correct), so
  the pure specular decay matches Eyring only in the near-cubic / initial-slope
  limit.
* **FDTD cross-checks.** An independent 2D wave solver
  (:func:`phonometry.simulation.fdtd_simulation`) reproduces the rigid-wall
  echo delay predicted by the first image, and its uniform-bulk-damping decay
  (dimension-independent, ``T60 = 3 ln 10 / damping``) recovers the same
  ``T60`` through the shared :func:`phonometry.room.decay_curve` machinery.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry import (
    ImageSourceResult,
    audible_image_count,
    decay_curve,
    eyring_reverberation_time,
    image_source_rir,
    reflection_density,
)
from phonometry.room.image_source import WALL_ORDER
from phonometry.simulation import GaussianPulse, fdtd_simulation

C = 343.0


# ---------------------------------------------------------------------------
# Exact geometry.
# ---------------------------------------------------------------------------


def test_direct_sound_time_and_amplitude() -> None:
    dims = (8.0, 5.0, 3.0)
    src = (2.0, 2.5, 1.5)
    rcv = (6.0, 2.5, 1.5)
    res = image_source_rir(dims, src, rcv, 0.2, fs=48000, max_order=2)
    d = math.dist(src, rcv)  # 4.0 m
    # Sorted by arrival time -> the first entry is the direct sound (order 0).
    assert res.orders[0] == 0
    assert res.distances[0] == pytest.approx(d)
    assert res.times[0] == pytest.approx(d / C)
    assert float(np.atleast_1d(res.amplitudes)[0]) == pytest.approx(
        1.0 / (4.0 * math.pi * d)
    )


def test_first_reflection_amplitude_carries_one_reflection_factor() -> None:
    # Source and receiver on the room mid-plane; the floor/ceiling images are
    # the first reflections. A single reflection scales the amplitude by
    # R = sqrt(1 - alpha) on top of 1 / (4 pi r).
    alpha = 0.3
    dims = (10.0, 8.0, 4.0)
    src = np.array([3.0, 4.0, 2.0])
    rcv = np.array([6.0, 4.0, 2.0])
    res = image_source_rir(dims, src, rcv, alpha, fs=96000, max_order=1)
    # The two first-order z-reflections (floor at z=0, ceiling at z=4) mirror
    # the source to z = -2 and z = +6, both at distance sqrt(3^2 + 4^2) = 5 m.
    r_reflect = math.hypot(3.0, 4.0)
    expected = math.sqrt(1.0 - alpha) / (4.0 * math.pi * r_reflect)
    amp = np.atleast_1d(res.amplitudes)
    order1 = amp[res.orders == 1]
    # Among the first-order images, the floor/ceiling pair sits at r = 5 m.
    assert np.any(np.isclose(order1, expected, rtol=1e-9))


def test_fully_absorbing_wall_annihilates_its_reflections() -> None:
    # A wall with alpha = 1 has R = 0, so every image using it vanishes; only
    # images that never touch that wall survive.
    dims = (6.0, 5.0, 4.0)
    alpha = np.zeros(6)
    alpha[WALL_ORDER.index("z0")] = 1.0  # floor perfectly absorbing
    res = image_source_rir(dims, (2.0, 2.0, 2.0), (4.0, 3.0, 2.0), alpha,
                           fs=48000, max_order=6)
    amp = np.atleast_1d(res.amplitudes)
    # No non-finite or negative amplitudes; the floor-touching images are 0.
    assert np.all(np.isfinite(amp))
    assert np.all(amp >= 0.0)
    assert np.any(amp == 0.0)


# ---------------------------------------------------------------------------
# Image count and reflection density.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("i0", [0, 1, 2, 3, 5, 10])
def test_audible_image_count_matches_kuttruff_9_23(i0: int) -> None:
    dims = (8.0, 5.0, 3.0)
    res = image_source_rir(dims, (2.0, 2.5, 1.6), (5.5, 3.2, 1.4), 0.1,
                           fs=8000, max_order=i0)
    # The reflection table holds the audible images plus the direct source.
    assert res.times.size == audible_image_count(i0) + 1


def test_audible_image_count_formula() -> None:
    # (2/3)(2 i0^3 + 3 i0^2 + 4 i0); i0 = 10 -> 1560 (Kuttruff Eq. 9.23).
    assert audible_image_count(10) == 1560
    assert audible_image_count(0) == 0
    assert audible_image_count(1) == 6


def test_reflection_density_closed_form() -> None:
    V = 120.0
    assert reflection_density(0.1, V) == pytest.approx(
        4.0 * math.pi * C**3 * 0.1**2 / V
    )
    assert reflection_density(0.0, V) == pytest.approx(0.0)


def test_reflection_density_matches_arrival_histogram() -> None:
    # The cumulative image count up to time t approaches the integral of
    # dN/dt = 4 pi c^3 t^2 / V, i.e. (4 pi c^3 / 3V) t^3, in the mid range
    # (before the finite max_order truncates the far/late images).
    dims = (5.0, 4.0, 3.0)
    V = float(np.prod(dims))
    res = image_source_rir(dims, (2.0, 2.0, 1.5), (3.0, 2.5, 1.5), 0.1,
                           fs=48000, max_order=50)
    t_probe = 0.05  # s, well inside the complete region (50 * 3 / c = 0.44 s)
    n_before = int(np.count_nonzero(res.times <= t_probe))
    predicted = 4.0 * math.pi * C**3 / (3.0 * V) * t_probe**3
    assert n_before == pytest.approx(predicted, rel=0.15)


# ---------------------------------------------------------------------------
# Eyring relation (near-cubic uniform room, documented tolerance).
# ---------------------------------------------------------------------------


def _energy_density_t60(res: ImageSourceResult, win: float = 0.008) -> float:
    """Recover T60 from the initial slope of the reverberant energy density."""
    times = res.times
    amp = np.atleast_1d(res.amplitudes)
    if amp.ndim == 2:
        amp = amp[0]
    edges = np.arange(0.0, float(times.max()), win)
    energy, _ = np.histogram(times, bins=edges, weights=amp**2)
    centres = 0.5 * (edges[:-1] + edges[1:])
    good = energy > 0.0
    ref = energy[good][0]
    level = 10.0 * np.log10(np.where(good, energy, 1.0) / ref)
    mask = good & (level <= -1.0) & (level >= -10.0)
    slope = np.polyfit(centres[mask], level[mask], 1)[0]
    return float(-60.0 / slope)


@pytest.mark.parametrize("length,alpha", [(5.0, 0.12), (4.0, 0.15), (6.0, 0.1)])
def test_cubic_room_recovers_eyring(length: float, alpha: float) -> None:
    dims = (length, length, length)
    volume = length**3
    surface = 6.0 * length**2
    eyring = float(eyring_reverberation_time(volume, [(surface / 6.0, alpha)] * 6))
    res = image_source_rir(dims, (1.3, 2.1, 1.7),
                           (length - 1.4, length - 1.9, length - 1.1),
                           alpha, fs=48000, max_order=70)
    recovered = _energy_density_t60(res)
    # The specular decay of a cube reproduces the Eyring rate to ~10 %.
    assert recovered == pytest.approx(eyring, rel=0.12)


def test_elongated_room_decays_slower_than_eyring() -> None:
    # An anisotropic (elongated) room sustains energy along its long axis, so
    # the specular decay runs longer than the diffuse-field Eyring estimate.
    dims = (12.0, 5.0, 3.0)
    volume = float(np.prod(dims))
    surface = 2.0 * (12 * 5 + 12 * 3 + 5 * 3)
    eyring = float(eyring_reverberation_time(volume, [(surface / 6.0, 0.12)] * 6))
    res = image_source_rir(dims, (2.0, 2.0, 1.5), (9.0, 3.0, 1.6), 0.12,
                           fs=48000, max_order=60)
    assert _energy_density_t60(res) > eyring


# ---------------------------------------------------------------------------
# Per-band assembly.
# ---------------------------------------------------------------------------


def test_per_band_result_shapes() -> None:
    freqs = [250.0, 500.0, 1000.0]
    alpha = np.array([[0.1] * 6, [0.2] * 6, [0.4] * 6]).T  # (6, 3)
    res = image_source_rir((5.0, 4.0, 3.0), (1.2, 1.1, 1.3), (3.5, 2.6, 1.7),
                           alpha, fs=16000, max_order=6, frequencies=freqs)
    assert res.ir.ndim == 2 and res.ir.shape[0] == 3
    assert res.amplitudes.shape[0] == 3
    assert res.frequencies is not None and res.frequencies.size == 3
    # A more absorbing band decays to a lower total energy, monotonically.
    energies = np.sum(res.ir**2, axis=1)
    assert energies[0] > energies[1] > energies[2]


def test_per_band_uniform_vector() -> None:
    res = image_source_rir((5.0, 4.0, 3.0), (1.2, 1.1, 1.3), (3.5, 2.6, 1.7),
                           np.array([0.1, 0.3]), fs=16000, max_order=5)
    assert res.ir.shape[0] == 2


def test_length_six_is_per_wall_without_frequencies() -> None:
    # A bare length-6 vector is the six per-wall coefficients (broadband).
    res = image_source_rir((5.0, 4.0, 3.0), (1.2, 1.1, 1.3), (3.5, 2.6, 1.7),
                           np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                           fs=16000, max_order=5)
    assert res.ir.ndim == 1 and res.frequencies is None


def test_length_six_is_per_band_with_six_frequencies() -> None:
    # With six declared bands, a length-6 vector is a per-band curve.
    freqs = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
    res = image_source_rir((5.0, 4.0, 3.0), (1.2, 1.1, 1.3), (3.5, 2.6, 1.7),
                           np.array([0.1, 0.15, 0.2, 0.3, 0.45, 0.6]),
                           fs=16000, max_order=5, frequencies=freqs)
    assert res.ir.shape[0] == 6
    # More absorption at each higher band -> a strictly lower total energy.
    energies = np.sum(res.ir**2, axis=1)
    assert np.all(np.diff(energies) < 0.0)


def test_air_attenuation_reduces_late_energy() -> None:
    dims = (8.0, 6.0, 4.0)
    common = {"fs": 48000, "max_order": 30}
    dry = image_source_rir(dims, (2.0, 2.0, 2.0), (6.0, 4.0, 2.0), 0.1,
                           air_attenuation=0.0, **common)
    humid = image_source_rir(dims, (2.0, 2.0, 2.0), (6.0, 4.0, 2.0), 0.1,
                             air_attenuation=0.01, **common)
    # Air absorption scales exp(-m r / 2); distant (late) images lose more.
    late = dry.times > 0.05
    assert np.all(
        np.atleast_1d(humid.amplitudes)[late]
        <= np.atleast_1d(dry.amplitudes)[late] + 1e-12
    )
    assert np.sum(np.atleast_1d(humid.amplitudes) ** 2) < np.sum(
        np.atleast_1d(dry.amplitudes) ** 2
    )


# ---------------------------------------------------------------------------
# FDTD cross-checks (independent wave solver).
# ---------------------------------------------------------------------------


def test_fdtd_rigid_wall_echo_matches_first_image() -> None:
    # A single rigid wall makes one echo; its delay after the direct sound is
    # the image-source prediction (2 * source-to-wall distance) / c. The 2D
    # FDTD (dimension-independent for a plane geometry) must reproduce it.
    dx = 0.02
    nx, ny = round(3.0 / dx), round(2.0 / dx)
    src_x, probe_x = 0.5, 1.0
    width = 0.4e-3
    src = GaussianPulse(ix=int(src_x / dx), iy=ny // 2, width=width)
    res = fdtd_simulation(
        C, dx, duration=0.012, sources=[src], shape=(ny, nx),
        probes=[(int(probe_x / dx), ny // 2)],
        boundaries={"left": "rigid", "right": "absorbing",
                    "top": "absorbing", "bottom": "absorbing"},
        absorbing_layer_cells=25,
    )
    p = res.pressures[0]
    t = res.times
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(p, height=0.05 * p.max())
    peak_t = t[peaks]
    # The image-source echo lag: source mirrored across x = 0 sits behind the
    # wall, so the echo path is probe_x + src_x against a direct path of
    # probe_x - src_x, a lag of 2 * src_x / c.
    lag = 2.0 * src_x / C
    observed_lag = peak_t[1] - peak_t[0]
    assert observed_lag == pytest.approx(lag, abs=3.0 * dx / C)


def test_fdtd_damping_decay_recovers_t60() -> None:
    # A rigid 2D room with uniform bulk damping decays as exp(-2 damping t)
    # in energy, giving a dimension-independent T60 = 3 ln 10 / damping. Feed
    # the FDTD probe pressure through the same Schroeder machinery the
    # image-source RIR uses (room.decay_curve) and recover that T60.
    dx = 0.05
    nx, ny = round(4.0 / dx), round(3.0 / dx)
    t60_target = 0.6
    damping = 3.0 * math.log(10.0) / t60_target
    src = GaussianPulse(ix=nx // 3, iy=ny // 3, width=1.2e-3)
    res = fdtd_simulation(
        C, dx, duration=1.2, sources=[src], shape=(ny, nx),
        probes=[(2 * nx // 3, 2 * ny // 3)],
        boundaries="rigid", damping=damping,
    )
    p = res.pressures[0]
    fs = round(1.0 / res.dt)
    time, level = decay_curve(p, fs)
    mask = (level <= -5.0) & (level >= -35.0)
    slope = np.polyfit(time[mask], level[mask], 1)[0]
    t30 = -60.0 / slope
    assert t30 == pytest.approx(t60_target, rel=0.05)


# ---------------------------------------------------------------------------
# Validation and plotting.
# ---------------------------------------------------------------------------


def test_source_outside_room_rejected() -> None:
    with pytest.raises(ValueError, match="outside the room"):
        image_source_rir((5.0, 4.0, 3.0), (6.0, 2.0, 1.5), (3.0, 2.0, 1.5),
                         0.2, fs=16000, max_order=2)


def test_absorption_above_one_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        image_source_rir((5.0, 4.0, 3.0), (1.0, 2.0, 1.5), (3.0, 2.0, 1.5),
                         1.2, fs=16000, max_order=2)


def test_absorption_band_count_mismatch_rejected() -> None:
    alpha = np.full((6, 4), 0.2)  # 4 bands vs 3 frequencies
    with pytest.raises(ValueError, match="bands but the result has"):
        image_source_rir((5.0, 4.0, 3.0), (1.0, 1.0, 1.0), (3.0, 2.0, 1.5),
                         alpha, fs=16000, max_order=3,
                         frequencies=[250.0, 500.0, 1000.0])


def test_coincident_source_receiver_rejected() -> None:
    with pytest.raises(ValueError, match="coincide"):
        image_source_rir((5.0, 4.0, 3.0), (2.0, 2.0, 1.5), (2.0, 2.0, 1.5),
                         0.2, fs=16000, max_order=2)


def test_bad_dimension_and_fs() -> None:
    with pytest.raises(ValueError):
        image_source_rir((0.0, 4.0, 3.0), (1.0, 2.0, 1.5), (3.0, 2.0, 1.5),
                         0.2, fs=16000)
    with pytest.raises(ValueError):
        image_source_rir((5.0, 4.0, 3.0), (1.0, 2.0, 1.5), (3.0, 2.0, 1.5),
                         0.2, fs=0)


def test_reflectogram_plot_smoke() -> None:
    import matplotlib

    matplotlib.use("Agg")
    res = image_source_rir((5.0, 4.0, 3.0), (1.2, 1.1, 1.3), (3.5, 2.6, 1.7),
                           0.15, fs=16000, max_order=8)
    ax = res.plot()
    assert ax.get_xlabel() == "Arrival time [ms]"


def test_reflectogram_plot_direct_only() -> None:
    # max_order=0 has no reflections; the reflectogram must not crash on the
    # empty scatter / colorbar.
    import matplotlib

    matplotlib.use("Agg")
    res = image_source_rir((5.0, 4.0, 3.0), (1.2, 1.1, 1.3), (3.5, 2.6, 1.7),
                           0.15, fs=16000, max_order=0)
    ax = res.plot()
    assert ax.get_xlabel() == "Arrival time [ms]"
