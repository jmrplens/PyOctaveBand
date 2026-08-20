#  Copyright (c) 2026. Jose Manuel Requena Plens

"""Real result objects, built through the public API, for the plot tests.

Every ``.plot()`` in the library draws a result object, and a result object is
only honest if a real computation produced it: a hand-filled dataclass would let
a renderer agree with a field the standard never puts there. So each factory
below runs the public entry point on a small, deterministic input and hands back
the result it returns, which is what the plot contract in
``tests/test_result_plots.py`` and the per-domain content assertions both draw.

Two of them build the dataclass directly, and say why in their docstrings: the
wide-band ``IntensityResult`` needs centres two decades apart to exercise the
log-axis bar widths, and the invalid-band ``RoomAcousticsResult`` needs exactly
one band flagged on every decay-time series. Everything else goes through the
same call a user would write.

The tests tree is on ``sys.path`` (``pythonpath = ["tests"]`` in pyproject), so
this module is imported as ``result_factories`` next to ``reference_data``,
``golden_data`` and ``oracle_data``.
"""

from __future__ import annotations

import numpy as np
import pytest

import phonometry as ph

FS = 48000
RNG = np.random.default_rng(20260707)


def _exp_ir(seconds: float = 0.8, t60: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * FS)) / FS
    decay = np.exp(-3.0 * np.log(10.0) / t60 * t)
    return decay * RNG.standard_normal(t.size)


def _zwicker_stationary() -> ph.psychoacoustics.ZwickerLoudness:
    return ph.psychoacoustics.loudness_zwicker_from_spectrum(np.full(28, 40.0))


def _sti() -> ph.speech.STIResult:
    sig = ph.speech.stipa_signal(fs=FS, seconds=16.0, seed=3)
    return ph.speech.stipa(sig, FS)


def _airborne_rating() -> ph.building.WeightedRatingResult:
    measured = np.array(
        [30, 34, 38, 41, 45, 49, 50, 53, 54, 55, 56, 57, 58, 58, 58, 58],
        dtype=float,
    )
    return ph.building.weighted_rating(measured)


def _impact_rating() -> ph.building.ImpactRatingResult:
    measured = np.array(
        [60, 61, 62, 63, 64, 65, 63, 61, 60, 58, 56, 53, 50, 47, 44, 41],
        dtype=float,
    )
    return ph.building.weighted_impact_rating(measured)


def _room(limits: list[float] | None) -> ph.room.RoomAcousticsResult:
    return ph.room.room_parameters(_exp_ir(), FS, limits=limits)


def _sound_power() -> ph.emission.SoundPowerResult:
    freqs = np.array([250.0, 500.0, 1000.0, 2000.0])
    r = 2.0
    s = 2.0 * np.pi * r**2
    lw_bands = np.array([90.0, 92.0, 95.0, 93.0])
    levels = np.tile(lw_bands - 10.0 * np.log10(s), (10, 1))
    return ph.emission.sound_power_pressure(
        levels, "hemisphere", radius=r, frequencies=freqs
    )


def _reverb_power() -> ph.emission.ReverberationSoundPowerResult:
    freqs = np.array([250.0, 500.0, 1000.0, 2000.0])
    t60 = np.array([1.6, 1.5, 1.4, 1.1])
    lp = np.tile(np.array([80.0, 82.0, 84.0, 81.0]), (6, 1))
    return ph.emission.sound_power_reverberation(lp, t60, 200.0, 220.0, freqs)


def _intensity_power_negative() -> ph.emission.SoundPowerIntensityResult:
    areas = np.array([0.5, 0.5, 0.5, 0.5])
    # band 0 positive net, band 1 all-negative net (external source).
    intensity = np.column_stack([np.full(4, 5.0e-4), np.full(4, -5.0e-5)])
    freqs = np.array([500.0, 1000.0])
    with pytest.warns(ph.emission.SoundPowerWarning):
        return ph.emission.sound_power_intensity(intensity, areas, frequencies=freqs)


def _intensity() -> ph.emission.IntensityResult:
    p1 = RNG.standard_normal(FS)
    p2 = np.roll(p1, 1)
    return ph.emission.sound_intensity(
        p1, p2, FS, spacing=0.012, fraction=3, limits=[125, 4000]
    )


def _open_plan() -> ph.room.OpenPlanResult:
    positions = np.array([2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0])
    spl = 57.0 - 7.0 * np.log2(positions / 4.0)
    sti = np.clip(0.9 - 0.055 * positions, 0.0, 1.0)
    return ph.room.open_plan_metrics(positions, spl, sti)


def _outdoor() -> ph.environment.OutdoorAttenuation:
    bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    barrier = ph.environment.Barrier(source_to_edge=101.0, edge_to_receiver=101.0)
    return ph.environment.outdoor_propagation_attenuation(
        200.0,
        1.5,
        1.5,
        bands,
        ground_source=1.0,
        ground_middle=1.0,
        ground_receiver=1.0,
        barrier=barrier,
        temperature=15.0,
        relative_humidity=70.0,
    )


def _cnossos_road() -> ph.environment.RoadEmissionResult:
    return ph.environment.road_source_power(
        [
            ph.environment.RoadTraffic(
                ph.environment.RoadVehicleCategory.LIGHT, 1200.0, 50.0
            ),
            ph.environment.RoadTraffic(
                ph.environment.RoadVehicleCategory.HEAVY, 45.0, 50.0
            ),
        ],
        surface=ph.environment.RoadSurface.THIN_LAYER_A,
        temperature=12.0,
        gradient=3.0,
    )


def _porous_medium() -> ph.materials.PorousMediumResult:
    f = np.linspace(400.0, 4000.0, 40)
    return ph.materials.miki(f, 20000.0)


def _layered_absorber() -> ph.materials.LayeredAbsorberResult:
    f = np.linspace(400.0, 4000.0, 40)
    med = ph.materials.miki(f, 20000.0)
    return ph.materials.layered_absorber(f, [ph.materials.PorousLayer(0.05, med)])


def _diffuse_absorption() -> ph.materials.DiffuseFieldAbsorptionResult:
    f = np.linspace(400.0, 4000.0, 8)
    med = ph.materials.miki(f, 20000.0)
    return ph.materials.diffuse_field_absorption(
        f, [ph.materials.PorousLayer(0.05, med)], quadrature_points=16
    )


def _impedance_tube() -> ph.materials.ImpedanceTubeResult:
    f = np.linspace(200.0, 1600.0, 60)
    r_true = 0.6 * np.exp(-f / 1200.0) * np.exp(0.8j)
    k = 2.0 * np.pi * f / 343.2
    s, x1 = 0.05, 0.12
    phase = np.exp(2j * k * x1)
    h12 = (np.exp(-1j * k * s) + r_true * phase * np.exp(1j * k * s)) / (
        1.0 + r_true * phase
    )
    return ph.materials.two_microphone_impedance(
        h12,
        frequency=f,
        spacing=s,
        x1=x1,
        speed_of_sound=343.2,
        characteristic_impedance=407.0,
    )


_MC_QUANTITIES = (
    ph.metrology.Quantity(74.0, 0.0, name="Reading"),
    ph.metrology.rectangular(0.0, 0.20, name="Calibration"),
    ph.metrology.Quantity(0.0, 0.35, dof=9, name="Position"),
)


def _monte_carlo() -> ph.metrology.MonteCarloResult:
    return ph.metrology.monte_carlo(
        lambda a, b, c: a + b + c,
        _MC_QUANTITIES,
        trials=2000,
        seed=7,
        keep_samples=True,
    )


def _exposure() -> ph.hearing.ExposureResult:
    tasks = [
        ph.hearing.Task((86.4, 86.7, 87.0), 2.0, label="grinding"),
        ph.hearing.Task((80.1, 80.9, 80.5), 3.0, label="welding"),
        ph.hearing.Task((75.0, 74.6, 74.9), 3.0, label="assembly"),
    ]
    return ph.hearing.task_based_exposure(tasks)


def _static_airflow() -> ph.materials.StaticAirflowResult:
    u = np.array([0.2e-3, 0.4e-3, 0.6e-3, 0.8e-3, 1.0e-3])
    dp = 30000.0 * u + 4.0e6 * u**2
    return ph.materials.static_airflow_resistance(u, dp, area=0.01, thickness=0.05)


def _airborne_prediction() -> ph.building.AirbornePredictionResult:
    paths = []
    for name, rw, k_ff, k_side, lf in (
        ("floor", 49.0, 12.4, 8.9, 4.5),
        ("facade", 42.0, 12.6, 6.7, 2.55),
    ):
        ff, df, fd = ph.building.flanking_element(
            label=name,
            r_flanking=rw,
            r_separating=57.0,
            k_ff=k_ff,
            k_fd=k_side,
            k_df=k_side,
            separating_area=11.5,
            coupling_length=lf,
        )
        paths.extend((ff, df, fd))
    return ph.building.predicted_airborne_insulation(
        r_direct=57.0, flanking_paths=paths
    )


def _impact_prediction() -> ph.building.ImpactPredictionResult:
    return ph.building.predicted_impact_insulation(
        ln_w_eq=78.0, delta_l_w=17.0, k_correction=2.0
    )


def _airborne_insulation() -> ph.building.AirborneInsulationResult:
    return ph.building.airborne_insulation(
        [70.0, 72.0, 74.0],
        [40.0, 41.0, 42.0],
        [0.5, 0.5, 0.5],
        area=10.0,
        volume=50.0,
    )


def _impact_insulation() -> ph.building.ImpactInsulationResult:
    return ph.building.impact_insulation(
        [60.0, 61.0, 62.0], [0.5, 0.5, 0.5], volume=50.0
    )


def _band_uncertainty() -> ph.building.BandUncertainty:
    return ph.building.band_uncertainty("airborne", "B")


def _intensity_wide() -> ph.emission.IntensityResult:
    """IntensityResult with band centres spanning two decades (100 Hz-10 kHz)
    so the log-axis bar-width scaling can be checked at the extremes.
    """
    freqs = np.array([100.0, 1000.0, 10000.0])
    n = freqs.size
    return ph.emission.IntensityResult(
        frequency=freqs,
        intensity=np.zeros(n),
        intensity_level=np.full(n, 60.0),
        pressure_level=np.full(n, 62.0),
        pressure_intensity_index=np.full(n, 2.0),
        direction=np.ones(n),
        bias_correction=np.ones(n),
        total_intensity=0.0,
        total_intensity_level=60.0,
        total_pressure_level=62.0,
        total_pressure_intensity_index=2.0,
        total_direction=1,
        max_valid_frequency=5000.0,
    )


_ANNEX_C2_R = [
    20.4,
    16.3,
    17.7,
    22.6,
    22.4,
    22.7,
    24.8,
    26.6,
    28.0,
    30.5,
    31.8,
    32.5,
    33.4,
    33.0,
    31.0,
    25.5,
]
_ANNEX_C2_FREQS = [
    50,
    63,
    80,
    100,
    125,
    160,
    200,
    250,
    315,
    400,
    500,
    630,
    800,
    1000,
    1250,
    1600,
    2000,
    2500,
    3150,
    4000,
    5000,
]


def _extended_rating() -> ph.building.ExtendedWeightedRatingResult:
    return ph.building.weighted_rating_extended(
        [18.7, 19.2, 20.0, *_ANNEX_C2_R, 26.8, 29.2], _ANNEX_C2_FREQS
    )


def _extended_impact_rating() -> ph.building.ExtendedImpactRatingResult:
    li = [
        55.0,
        57.0,
        59.0,
        62.1,
        63.2,
        63.5,
        66.2,
        68.5,
        70.0,
        71.7,
        73.1,
        73.8,
        73.5,
        73.8,
        73.3,
        73.1,
        73.0,
        72.4,
        71.2,
    ]
    freqs = [
        50,
        63,
        80,
        100,
        125,
        160,
        200,
        250,
        315,
        400,
        500,
        630,
        800,
        1000,
        1250,
        1600,
        2000,
        2500,
        3150,
    ]
    return ph.building.weighted_impact_rating_extended(li, freqs)


_PANEL_BANDS = np.array(
    [
        100,
        125,
        160,
        200,
        250,
        315,
        400,
        500,
        630,
        800,
        1000,
        1250,
        1600,
        2000,
        2500,
        3150,
    ],
    dtype=float,
)


def _radiation_efficiency() -> ph.vibration.RadiationEfficiencyResult:
    bp = ph.vibration.plate_bending_stiffness(6.2e10, 0.006, 0.24)
    fc = ph.vibration.coincidence_frequency(2500.0 * 0.006, bp)
    return ph.vibration.radiation_efficiency(_PANEL_BANDS, 1.5, 1.25, fc)


def _single_panel() -> ph.building.SoundReductionResult:
    bp = ph.vibration.plate_bending_stiffness(6.2e10, 0.006, 0.24)
    fc = ph.vibration.coincidence_frequency(15.0, bp)
    return ph.building.single_panel_transmission_loss(
        _PANEL_BANDS, 15.0, critical_frequency=fc, loss_factor=0.024
    )


def _double_wall() -> ph.building.SoundReductionResult:
    return ph.building.double_wall_transmission_loss(_PANEL_BANDS, 12.0, 12.0, 0.1)


def _slit_aperture() -> ph.building.ApertureTransmissionResult:
    return ph.building.slit_transmission_coefficient(_PANEL_BANDS, 0.002, 0.1)


def _modulation() -> ph.electroacoustics.ModulationDistortionResult:
    t = np.arange(FS) / FS
    x = (
        np.sin(2 * np.pi * 250.0 * t)
        + 0.25 * np.sin(2 * np.pi * 8000.0 * t)
        + 0.02 * np.sin(2 * np.pi * 8250.0 * t)
        + 0.02 * np.sin(2 * np.pi * 7750.0 * t)
    )
    return ph.electroacoustics.modulation_distortion(x, FS, f_low=250.0, f_high=8000.0)


def _field_indicators() -> ph.emission.FieldIndicators:
    rng = np.random.default_rng(9614)
    lp = 74.0 + rng.normal(0.0, 0.8, (8, 4))
    i_n = 1.0e-5 * (1.0 + rng.normal(0.0, 0.2, (8, 4)))
    return ph.emission.field_indicators(lp, i_n, [125.0, 250.0, 500.0, 1000.0])


def _room_with_one_invalid_band() -> ph.room.RoomAcousticsResult:
    """A RoomAcousticsResult whose middle band is flagged invalid on every
    decay-time series (built directly so the test is deterministic).
    """
    freq = np.array([250.0, 500.0, 1000.0])
    ones = np.ones(3)
    valid = np.array([True, False, True])  # 500 Hz band invalid
    return ph.room.RoomAcousticsResult(
        frequency=freq,
        edt=ones.copy(),
        t20=ones.copy(),
        t30=ones.copy(),
        c50=np.zeros(3),
        c80=np.zeros(3),
        d50=np.zeros(3),
        ts=np.zeros(3),
        dynamic_range=np.array([60.0, 5.0, 60.0]),
        edt_valid=valid.copy(),
        t20_valid=valid.copy(),
        t30_valid=valid.copy(),
        curvature=np.zeros(3),
    )


def _transfer_matrix() -> tuple[ph.materials.TransferMatrix, np.ndarray, float]:
    f = np.linspace(200.0, 1600.0, 40)
    rho_c = 407.0
    k = 2.0 * np.pi * f / 343.2
    tm = ph.materials.air_layer_transfer_matrix(k, 0.05, rho_c)
    return tm, f, rho_c
