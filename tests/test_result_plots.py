#  Copyright (c) 2026. Jose M. Requena-Plens

"""Tests for the one-line ``.plot()`` methods on result objects.

The library exposes matplotlib as a *soft* dependency: importing
phonometry and computing must work without it, and only ``.plot()``
requires it. These tests run on the non-interactive Agg backend and check
both the smoke path (a figure is produced and the axes returned) and the
content (line/bar data echo the result fields, shading falls only on
unfavourable deviations, invalid/negative bands are marked) plus the
missing-matplotlib ImportError path.
"""

from __future__ import annotations

import ast
import builtins
import inspect

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import phonometry as ph
from phonometry._plot import common as _plotting

FS = 48000
RNG = np.random.default_rng(20260707)


# --------------------------------------------------------------------------
# Fixtures: minimal real result objects via the public API
# --------------------------------------------------------------------------
def _exp_ir(seconds: float = 0.8, t60: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * FS)) / FS
    decay = np.exp(-3.0 * np.log(10.0) / t60 * t)
    return decay * RNG.standard_normal(t.size)


def _zwicker_stationary() -> ph.ZwickerLoudness:
    return ph.loudness_zwicker_from_spectrum(np.full(28, 40.0))


def _sti() -> ph.STIResult:
    sig = ph.stipa_signal(fs=FS, seconds=16.0, seed=3)
    return ph.stipa(sig, FS)


def _airborne_rating() -> ph.WeightedRatingResult:
    measured = np.array(
        [30, 34, 38, 41, 45, 49, 50, 53, 54, 55, 56, 57, 58, 58, 58, 58],
        dtype=float,
    )
    return ph.weighted_rating(measured)


def _impact_rating() -> ph.ImpactRatingResult:
    measured = np.array(
        [60, 61, 62, 63, 64, 65, 63, 61, 60, 58, 56, 53, 50, 47, 44, 41],
        dtype=float,
    )
    return ph.weighted_impact_rating(measured)


def _room(limits: list[float] | None) -> ph.RoomAcousticsResult:
    return ph.room_parameters(_exp_ir(), FS, limits=limits)


def _sound_power() -> ph.SoundPowerResult:
    freqs = np.array([250.0, 500.0, 1000.0, 2000.0])
    r = 2.0
    s = 2.0 * np.pi * r**2
    lw_bands = np.array([90.0, 92.0, 95.0, 93.0])
    levels = np.tile(lw_bands - 10.0 * np.log10(s), (10, 1))
    return ph.sound_power_pressure(levels, "hemisphere", radius=r, frequencies=freqs)


def _reverb_power() -> ph.ReverberationSoundPowerResult:
    freqs = np.array([250.0, 500.0, 1000.0, 2000.0])
    t60 = np.array([1.6, 1.5, 1.4, 1.1])
    lp = np.tile(np.array([80.0, 82.0, 84.0, 81.0]), (6, 1))
    return ph.sound_power_reverberation(lp, t60, 200.0, 220.0, freqs)


def _intensity_power_negative() -> ph.SoundPowerIntensityResult:
    areas = np.array([0.5, 0.5, 0.5, 0.5])
    # band 0 positive net, band 1 all-negative net (external source).
    intensity = np.column_stack([np.full(4, 5.0e-4), np.full(4, -5.0e-5)])
    freqs = np.array([500.0, 1000.0])
    with pytest.warns(ph.SoundPowerWarning):
        return ph.sound_power_intensity(intensity, areas, frequencies=freqs)


def _intensity() -> ph.IntensityResult:
    p1 = RNG.standard_normal(FS)
    p2 = np.roll(p1, 1)
    return ph.sound_intensity(p1, p2, FS, spacing=0.012, fraction=3, limits=[125, 4000])


def _open_plan() -> ph.OpenPlanResult:
    positions = np.array([2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0])
    spl = 57.0 - 7.0 * np.log2(positions / 4.0)
    sti = np.clip(0.9 - 0.055 * positions, 0.0, 1.0)
    return ph.open_plan_metrics(positions, spl, sti)


def _outdoor() -> ph.OutdoorAttenuation:
    bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    barrier = ph.Barrier(source_to_edge=101.0, edge_to_receiver=101.0)
    return ph.outdoor_propagation_attenuation(
        200.0, 1.5, 1.5, bands, ground_source=1.0, ground_middle=1.0,
        ground_receiver=1.0, barrier=barrier, temperature=15.0,
        relative_humidity=70.0,
    )


def _porous_medium() -> ph.PorousMediumResult:
    f = np.linspace(400.0, 4000.0, 40)
    return ph.miki(f, 20000.0)


def _layered_absorber() -> ph.LayeredAbsorberResult:
    f = np.linspace(400.0, 4000.0, 40)
    med = ph.miki(f, 20000.0)
    return ph.layered_absorber(f, [ph.PorousLayer(0.05, med)])


def _diffuse_absorption() -> ph.DiffuseFieldAbsorptionResult:
    f = np.linspace(400.0, 4000.0, 8)
    med = ph.miki(f, 20000.0)
    return ph.diffuse_field_absorption(
        f, [ph.PorousLayer(0.05, med)], quadrature_points=16
    )


def _impedance_tube() -> ph.ImpedanceTubeResult:
    f = np.linspace(200.0, 1600.0, 60)
    r_true = 0.6 * np.exp(-f / 1200.0) * np.exp(0.8j)
    k = 2.0 * np.pi * f / 343.2
    s, x1 = 0.05, 0.12
    phase = np.exp(2j * k * x1)
    h12 = (np.exp(-1j * k * s) + r_true * phase * np.exp(1j * k * s)) / (
        1.0 + r_true * phase
    )
    return ph.two_microphone_impedance(
        h12, frequency=f, spacing=s, x1=x1, speed_of_sound=343.2,
        characteristic_impedance=407.0,
    )


_MC_QUANTITIES = (
    ph.Quantity(74.0, 0.0, name="Reading"),
    ph.rectangular(0.0, 0.20, name="Calibration"),
    ph.Quantity(0.0, 0.35, dof=9, name="Position"),
)


def _monte_carlo() -> ph.MonteCarloResult:
    return ph.monte_carlo(
        lambda a, b, c: a + b + c, _MC_QUANTITIES, trials=2000, seed=7,
        keep_samples=True,
    )


def _exposure() -> ph.ExposureResult:
    tasks = [
        ph.Task((86.4, 86.7, 87.0), 2.0, label="grinding"),
        ph.Task((80.1, 80.9, 80.5), 3.0, label="welding"),
        ph.Task((75.0, 74.6, 74.9), 3.0, label="assembly"),
    ]
    return ph.task_based_exposure(tasks)


def _static_airflow() -> ph.StaticAirflowResult:
    u = np.array([0.2e-3, 0.4e-3, 0.6e-3, 0.8e-3, 1.0e-3])
    dp = 30000.0 * u + 4.0e6 * u**2
    return ph.static_airflow_resistance(u, dp, area=0.01, thickness=0.05)


def _airborne_prediction() -> ph.AirbornePredictionResult:
    paths = []
    for name, rw, k_ff, k_side, lf in (
        ("floor", 49.0, 12.4, 8.9, 4.5),
        ("facade", 42.0, 12.6, 6.7, 2.55),
    ):
        ff, df, fd = ph.flanking_element(
            label=name, r_flanking=rw, r_separating=57.0, k_ff=k_ff,
            k_fd=k_side, k_df=k_side, separating_area=11.5, coupling_length=lf,
        )
        paths.extend((ff, df, fd))
    return ph.predicted_airborne_insulation(r_direct=57.0, flanking_paths=paths)


def _impact_prediction() -> ph.ImpactPredictionResult:
    return ph.predicted_impact_insulation(
        ln_w_eq=78.0, delta_l_w=17.0, k_correction=2.0
    )


def _airborne_insulation() -> ph.AirborneInsulationResult:
    return ph.airborne_insulation(
        [70.0, 72.0, 74.0], [40.0, 41.0, 42.0], [0.5, 0.5, 0.5],
        area=10.0, volume=50.0,
    )


def _impact_insulation() -> ph.ImpactInsulationResult:
    return ph.impact_insulation([60.0, 61.0, 62.0], [0.5, 0.5, 0.5], volume=50.0)


def _band_uncertainty() -> ph.BandUncertainty:
    return ph.band_uncertainty("airborne", "B")


def _intensity_wide() -> ph.IntensityResult:
    """IntensityResult with band centres spanning two decades (100 Hz-10 kHz)
    so the log-axis bar-width scaling can be checked at the extremes."""
    freqs = np.array([100.0, 1000.0, 10000.0])
    n = freqs.size
    return ph.IntensityResult(
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


# --------------------------------------------------------------------------
# Soft-dependency contract: lazy import + ImportError guidance
# --------------------------------------------------------------------------
def test_plotting_module_has_no_toplevel_matplotlib_import() -> None:
    """matplotlib must be imported inside functions, never at module scope."""
    tree = ast.parse(inspect.getsource(_plotting))
    # Imports inside a function body are lazy; imports under an
    # ``if TYPE_CHECKING:`` guard never run at import time. Both are allowed;
    # a plain module-scope runtime import of matplotlib is not.
    allowed: set[ast.AST] = set()
    for node in ast.walk(tree):
        is_func = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        is_typecheck = (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        )
        if is_func or is_typecheck:
            for sub in ast.walk(node):
                allowed.add(sub)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", None) or ""
            names = [module, *[a.name for a in node.names]]
            if any("matplotlib" in n for n in names):
                assert node in allowed, "matplotlib imported at runtime module scope"


def test_plot_raises_helpful_error_without_matplotlib(monkeypatch) -> None:
    """Without matplotlib, .plot() fails with an actionable message."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    res = _zwicker_stationary()
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"pip install phonometry\[plot\]"):
        res.plot()


# --------------------------------------------------------------------------
# Zwicker loudness
# --------------------------------------------------------------------------
def test_zwicker_stationary_returns_single_axes_with_specific_curve() -> None:
    res = _zwicker_stationary()
    ax = res.plot()
    assert not isinstance(ax, np.ndarray)
    ydata = ax.lines[0].get_ydata()
    np.testing.assert_allclose(ydata, res.specific)
    xdata = ax.lines[0].get_xdata()
    assert xdata[0] == pytest.approx(0.1) and xdata[-1] == pytest.approx(24.0)
    assert "Bark" in ax.get_xlabel()
    assert "sone/Bark" in ax.get_ylabel()
    plt.close("all")


def test_zwicker_time_varying_returns_two_panels() -> None:
    sig = RNG.standard_normal(FS) * 0.02  # 1 s of noise
    res = ph.loudness_zwicker(sig, FS, stationary=False)
    assert res.loudness_vs_time is not None
    axes = res.plot()
    assert isinstance(axes, np.ndarray) and axes.size == 2
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), res.specific)
    np.testing.assert_allclose(axes[1].lines[0].get_ydata(), res.loudness_vs_time)
    plt.close("all")


# --------------------------------------------------------------------------
# Moore-Glasberg loudness (ISO 532-2)
# --------------------------------------------------------------------------
def test_moore_glasberg_returns_single_axes_with_specific_curve() -> None:
    res = ph.loudness_moore_glasberg_from_spectrum([(1000.0, 60.0)])
    ax = res.plot()
    assert not isinstance(ax, np.ndarray)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), res.specific)
    xdata = ax.lines[0].get_xdata()
    assert xdata[0] == pytest.approx(1.8) and xdata[-1] == pytest.approx(38.9)
    assert "Cam" in ax.get_xlabel()
    assert "sone/Cam" in ax.get_ylabel()
    plt.close("all")


# --------------------------------------------------------------------------
# STI
# --------------------------------------------------------------------------
def test_sti_bars_match_mti_and_annotate_rating() -> None:
    res = _sti()
    ax = res.plot()
    heights = [patch.get_height() for patch in ax.patches]
    np.testing.assert_allclose(heights, res.mti)
    assert res.rating in ax.get_title()
    assert ax.get_ylim() == (0.0, 1.0)
    plt.close("all")


# --------------------------------------------------------------------------
# Weighted ratings (airborne / impact)
# --------------------------------------------------------------------------
def test_airborne_rating_carries_curve_fields() -> None:
    res = _airborne_rating()
    assert res.band_centers is not None and res.band_centers.size == 16
    assert res.measured is not None and res.shifted_reference is not None
    # shifted reference read at 500 Hz (index 7) equals the rating.
    assert round(res.shifted_reference[7]) == res.rating


def test_airborne_rating_shades_only_unfavourable_bands() -> None:
    res = _airborne_rating()
    ax = res.plot()
    # measured curve and shifted reference are drawn as lines.
    np.testing.assert_allclose(ax.lines[0].get_ydata(), res.measured)
    np.testing.assert_allclose(ax.lines[1].get_ydata(), res.shifted_reference)
    # a shaded region (fill_between -> collection) is present.
    assert len(ax.collections) >= 1
    mask = _plotting._unfavourable_mask(
        res.measured, res.shifted_reference, impact=False
    )
    # airborne: unfavourable where measured < reference.
    np.testing.assert_array_equal(mask, res.measured < res.shifted_reference)
    assert mask.any()
    plt.close("all")


def test_impact_rating_uses_opposite_sign_mask() -> None:
    res = _impact_rating()
    ax = res.plot()
    mask = _plotting._unfavourable_mask(
        res.measured, res.shifted_reference, impact=True
    )
    np.testing.assert_array_equal(mask, res.measured > res.shifted_reference)
    assert str(res.rating) in ax.get_title()
    plt.close("all")


def test_rating_without_curve_data_raises() -> None:
    bare = ph.WeightedRatingResult(rating=52, c=-1, ctr=-3, unfavourable_sum=10.0)
    with pytest.raises(ValueError, match="no band curve"):
        bare.plot()


def test_rating_plot_forwards_kwargs_without_typeerror() -> None:
    # Regression: _plot_rating used to lack **kwargs, so any styling kwarg
    # forwarded by plot_weighted_rating/plot_impact_rating raised TypeError.
    res = _airborne_rating()
    ax = res.plot(linewidth=2)
    assert ax.lines[0].get_linewidth() == 2.0
    ax2 = _impact_rating().plot(linewidth=2)
    assert ax2.lines[0].get_linewidth() == 2.0
    plt.close("all")


# --------------------------------------------------------------------------
# ISO 717 enlarged-range ratings (Annex B / A.2.1)
# --------------------------------------------------------------------------
_ANNEX_C2_R = [20.4, 16.3, 17.7, 22.6, 22.4, 22.7, 24.8, 26.6,
               28.0, 30.5, 31.8, 32.5, 33.4, 33.0, 31.0, 25.5]
_ANNEX_C2_FREQS = [50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
                   630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000]


def _extended_rating() -> ph.ExtendedWeightedRatingResult:
    return ph.weighted_rating_extended(
        [18.7, 19.2, 20.0, *_ANNEX_C2_R, 26.8, 29.2], _ANNEX_C2_FREQS
    )


def _extended_impact_rating() -> ph.ExtendedImpactRatingResult:
    li = [55.0, 57.0, 59.0, 62.1, 63.2, 63.5, 66.2, 68.5, 70.0, 71.7, 73.1,
          73.8, 73.5, 73.8, 73.3, 73.1, 73.0, 72.4, 71.2]
    freqs = [50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
             630, 800, 1000, 1250, 1600, 2000, 2500, 3150]
    return ph.weighted_impact_rating_extended(li, freqs)


def test_extended_rating_plot_full_range_and_terms_in_title() -> None:
    res = _extended_rating()
    ax = res.plot()
    # The measured curve spans the full enlarged range (21 bands), the
    # shifted reference only the 16 core bands.
    assert ax.lines[0].get_xdata().size == 21
    np.testing.assert_allclose(ax.lines[0].get_ydata(), res.measured)
    assert ax.lines[1].get_xdata().size == 16
    np.testing.assert_allclose(
        ax.lines[1].get_ydata(), res.core.shifted_reference
    )
    # Title carries the core rating and the covered Annex B terms.
    title = ax.get_title()
    assert str(res.rating) in title
    assert "C50-5000" in title and "Ctr,50-5000" in title
    plt.close("all")


def test_extended_impact_rating_plot_title_carries_ci_50_2500() -> None:
    res = _extended_impact_rating()
    assert res.ci_50_2500 is not None
    ax = res.plot()
    title = ax.get_title()
    assert str(res.rating) in title and "CI,50-2500" in title
    # Impact shading uses the opposite sign: measurement above reference.
    assert len(ax.collections) >= 1
    plt.close("all")


def test_extended_rating_plot_forwards_kwargs() -> None:
    ax = _extended_rating().plot(linewidth=2)
    assert ax.lines[0].get_linewidth() == 2.0
    ax2 = _extended_impact_rating().plot(linewidth=2)
    assert ax2.lines[0].get_linewidth() == 2.0
    plt.close("all")


def test_extended_rating_spanish_labels() -> None:
    ax = _extended_rating().plot(language="es")
    labels = [str(ln.get_label()) for ln in ax.lines]
    assert any("Medido" in lbl for lbl in labels)
    assert "reducción acústica" in ax.get_ylabel()
    plt.close("all")


# --------------------------------------------------------------------------
# ISO 717-2 octave-band -5 dB rule: the curve is honest, the rating annotated
# --------------------------------------------------------------------------
_ANNEX_C3_LN_OCTAVE = np.array([65.3, 64.5, 58.0, 55.8, 43.0])


def test_octave_impact_plot_keeps_curve_honest_and_annotates_offset() -> None:
    # ISO 717-2 Annex C, Table C.3 (octave): Ln,w = 54, applying the -5 dB
    # rule of Clause 4.3.2. The drawn shifted-reference curve is genuinely
    # ref - shift (so it reads 59 at 500 Hz); the rating (54) is annotated
    # with the -5 dB note rather than the curve being distorted down.
    res = ph.weighted_impact_rating(_ANNEX_C3_LN_OCTAVE)
    assert res.rating == 54
    idx500 = int(np.argmin(np.abs(res.band_centers - 500.0)))
    read_value = float(res.shifted_reference[idx500])
    assert read_value == pytest.approx(res.rating + 5)  # honest curve
    ax = res.plot()
    # reference line (index 1) is undistorted: reads read_value at 500 Hz.
    assert ax.lines[1].get_ydata()[idx500] == pytest.approx(read_value)
    # a marker records the 500 Hz read value.
    marked = [
        ln for ln in ax.lines
        if ln.get_ydata().size == 1
        and ln.get_ydata()[0] == pytest.approx(read_value)
    ]
    assert marked, "expected a 500 Hz read-value marker"
    # annotation carries both the rating and the -5 dB octave note.
    texts = " ".join(t.get_text() for t in ax.texts).replace("−", "-")
    assert str(res.rating) in texts and "-5" in texts.replace(" ", "")
    plt.close("all")


def test_third_octave_impact_plot_reads_rating_at_500() -> None:
    # For one-third-octave impact there is no -5 dB offset: the curve read
    # value at 500 Hz equals the rating, so no offset note is drawn.
    res = _impact_rating()
    idx500 = int(np.argmin(np.abs(res.band_centers - 500.0)))
    assert round(float(res.shifted_reference[idx500])) == res.rating
    ax = res.plot()
    texts = " ".join(t.get_text() for t in ax.texts).replace("−", "-")
    assert "-5" not in texts.replace(" ", "")
    plt.close("all")


# --------------------------------------------------------------------------
# Every public .plot() must accept and forward a benign styling kwarg
# --------------------------------------------------------------------------
_PANEL_BANDS = np.array(
    [100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
     1000, 1250, 1600, 2000, 2500, 3150], dtype=float
)


def _radiation_efficiency() -> ph.RadiationEfficiencyResult:
    bp = ph.plate_bending_stiffness(6.2e10, 0.006, 0.24)
    fc = ph.coincidence_frequency(2500.0 * 0.006, bp)
    return ph.radiation_efficiency(_PANEL_BANDS, 1.5, 1.25, fc)


def _single_panel() -> ph.SoundReductionResult:
    bp = ph.plate_bending_stiffness(6.2e10, 0.006, 0.24)
    fc = ph.coincidence_frequency(15.0, bp)
    return ph.single_panel_transmission_loss(
        _PANEL_BANDS, 15.0, critical_frequency=fc, loss_factor=0.024
    )


def _double_wall() -> ph.SoundReductionResult:
    return ph.double_wall_transmission_loss(_PANEL_BANDS, 12.0, 12.0, 0.1)


def _slit_aperture() -> ph.ApertureTransmissionResult:
    return ph.slit_transmission_coefficient(_PANEL_BANDS, 0.002, 0.1)


def _modulation() -> ph.ModulationDistortionResult:
    t = np.arange(FS) / FS
    x = (np.sin(2 * np.pi * 250.0 * t) + 0.25 * np.sin(2 * np.pi * 8000.0 * t)
         + 0.02 * np.sin(2 * np.pi * 8250.0 * t)
         + 0.02 * np.sin(2 * np.pi * 7750.0 * t))
    return ph.modulation_distortion(x, FS, 250.0, 8000.0)


def _field_indicators() -> ph.FieldIndicators:
    rng = np.random.default_rng(9614)
    lp = 74.0 + rng.normal(0.0, 0.8, (8, 4))
    i_n = 1.0e-5 * (1.0 + rng.normal(0.0, 0.2, (8, 4)))
    return ph.field_indicators(lp, i_n, [125.0, 250.0, 500.0, 1000.0])


_KWARG_PLOT_CASES = [
    ("zwicker", _zwicker_stationary, "line"),
    ("sti", _sti, "bar"),
    ("airborne", _airborne_rating, "line"),
    ("impact", _impact_rating, "line"),
    ("room", lambda: _room([250, 2000]), "bar"),
    ("sound_power", _sound_power, "bar"),
    ("reverb_power", _reverb_power, "bar"),
    ("intensity_power", _intensity_power_negative, "bar"),
    ("intensity", _intensity, "line"),
    ("decay_curve", lambda: ph.decay_curve(_exp_ir(seconds=1.0, t60=0.6), FS), "line"),
    ("facade", lambda: ph.facade_insulation(
        [70.0, 72.0, 74.0], [40.0, 41.0, 42.0], [0.5, 0.5, 0.5]), "line"),
    ("open_plan", _open_plan, "line"),
    ("outdoor", _outdoor, "line"),
    ("impedance_tube", _impedance_tube, "line"),
    ("porous_medium", _porous_medium, "line"),
    ("layered_absorber", _layered_absorber, "line"),
    ("diffuse_absorption", _diffuse_absorption, "line"),
    ("monte_carlo", _monte_carlo, "bar"),
    ("exposure", _exposure, "bar"),
    ("static_airflow", _static_airflow, "line"),
    ("airborne_prediction", _airborne_prediction, "bar"),
    ("impact_prediction", _impact_prediction, "bar"),
    ("airborne_insulation", _airborne_insulation, "line"),
    ("impact_insulation", _impact_insulation, "line"),
    ("band_uncertainty", _band_uncertainty, "line"),
    ("radiation_efficiency", _radiation_efficiency, "line"),
    ("single_panel", _single_panel, "line"),
    ("double_wall", _double_wall, "line"),
    ("slit_aperture", _slit_aperture, "line"),
    ("modulation_distortion", _modulation, "line"),
    ("field_indicators", _field_indicators, "line"),
]


@pytest.mark.parametrize(
    ("name", "factory", "kind"),
    _KWARG_PLOT_CASES,
    ids=[c[0] for c in _KWARG_PLOT_CASES],
)
def test_every_plot_forwards_kwargs_to_primary_artist(name, factory, kind) -> None:
    res = factory()
    out = res.plot(linewidth=2)
    ax = out[0] if isinstance(out, np.ndarray) else out
    artists = ax.lines if kind == "line" else ax.patches
    assert artists, f"{name}: no primary {kind} artist drawn"
    assert any(a.get_linewidth() == 2.0 for a in artists), (
        f"{name}: linewidth kwarg not forwarded to the primary artist"
    )
    plt.close("all")

    # A user-supplied color must win over the renderer's fixed default rather
    # than raising ``TypeError: got multiple values for keyword 'color'``.
    out = res.plot(color="red")
    ax = out[0] if isinstance(out, np.ndarray) else out
    artists = ax.lines if kind == "line" else ax.patches
    red = plt.matplotlib.colors.to_rgba("red")

    def _is_red(artist) -> bool:
        if kind == "line":
            return plt.matplotlib.colors.to_rgba(artist.get_color()) == red
        return tuple(artist.get_facecolor()) == red

    assert any(_is_red(a) for a in artists), (
        f"{name}: color kwarg did not override the fixed default artist color"
    )
    plt.close("all")


def test_facade_plot_accepts_label_kwarg() -> None:
    # Regression: kwargs used to be forwarded to all four curves, so a user
    # label= collided with the per-curve labels and raised TypeError.
    res = ph.facade_insulation(
        [70.0, 72.0, 74.0], [40.0, 41.0, 42.0], [0.5, 0.5, 0.5]
    )
    ax = res.plot(label="my measurement")
    labels = [ln.get_label() for ln in ax.lines]
    assert "my measurement" in labels
    # The companion curves keep their own labels.
    assert any(label.startswith("$D_{2m}$") for label in labels)
    plt.close("all")


# --------------------------------------------------------------------------
# Room acoustics
# --------------------------------------------------------------------------
def test_room_acoustics_two_panels_and_bar_heights() -> None:
    res = _room(limits=[250, 2000])
    axes = res.plot()
    assert isinstance(axes, np.ndarray) and axes.size == 2
    ax_times = axes[0]
    n = res.t30.size
    # 3 grouped bars (EDT/T20/T30) per band.
    assert len(ax_times.patches) == 3 * n
    assert "time" in ax_times.get_ylabel().lower()
    plt.close("all")


def _room_with_one_invalid_band() -> ph.RoomAcousticsResult:
    """A RoomAcousticsResult whose middle band is flagged invalid on every
    decay-time series (built directly so the test is deterministic)."""
    freq = np.array([250.0, 500.0, 1000.0])
    ones = np.ones(3)
    valid = np.array([True, False, True])  # 500 Hz band invalid
    return ph.RoomAcousticsResult(
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


def test_room_acoustics_invalid_bands_are_hatched() -> None:
    # Deterministic: only the 500 Hz band is flagged invalid on all three
    # decay-time series, so exactly those three bars must be hatched/greyed.
    res = _room_with_one_invalid_band()
    invalid_idx = 1
    n = res.t30.size
    axes = res.plot()
    patches = axes[0].patches
    assert len(patches) == 3 * n  # EDT/T20/T30 grouped bars
    hatched = [p for p in patches if p.get_hatch()]
    greyed = [
        p for p in patches
        if p.get_facecolor()[:3] == plt.matplotlib.colors.to_rgb(_plotting._C_MUTED)
    ]
    # Exactly one bar per series (EDT/T20/T30) is invalid -> 3 hatched/greyed.
    assert len(hatched) == 3
    assert len(greyed) == 3
    # And they are the bars sitting over the invalid (500 Hz) band position.
    for p in hatched:
        assert round(p.get_x() + p.get_width() / 2.0) == invalid_idx
    plt.close("all")


def test_room_acoustics_single_axes_composition() -> None:
    res = _room(limits=[250, 2000])
    _fig, ax = plt.subplots()
    out = res.plot(ax=ax)
    assert out is ax
    plt.close("all")


# --------------------------------------------------------------------------
# Sound power (pressure / reverberation / intensity)
# --------------------------------------------------------------------------
def test_sound_power_bars_match_lw_and_annotate_lwa() -> None:
    res = _sound_power()
    ax = res.plot()
    heights = [p.get_height() for p in ax.patches]
    np.testing.assert_allclose(heights, res.sound_power_level)
    assert f"{res.sound_power_level_a:.1f}" in ax.get_title()
    plt.close("all")


def test_reverberation_power_plot_smoke() -> None:
    res = _reverb_power()
    ax = res.plot()
    heights = [p.get_height() for p in ax.patches]
    np.testing.assert_allclose(heights, np.nan_to_num(res.sound_power_level))
    plt.close("all")


def test_intensity_power_marks_negative_band() -> None:
    res = _intensity_power_negative()
    assert bool(res.negative_band[1])
    ax = res.plot()
    hatched = [p for p in ax.patches if p.get_hatch()]
    assert len(hatched) == int(np.count_nonzero(res.negative_band))
    plt.close("all")


# --------------------------------------------------------------------------
# Sound intensity (Lp vs LI)
# --------------------------------------------------------------------------
def test_intensity_plots_lp_and_li_with_index_twin() -> None:
    res = _intensity()
    ax = res.plot()
    np.testing.assert_allclose(ax.lines[0].get_ydata(), res.pressure_level)
    np.testing.assert_allclose(ax.lines[1].get_ydata(), res.intensity_level)
    # twin axis carries the pressure-intensity index bars.
    twins = [a for a in ax.figure.axes if a is not ax]
    assert twins, "expected a twin axis for the pressure-intensity index"
    plt.close("all")


def test_intensity_bar_width_scales_with_frequency_on_log_axis() -> None:
    # On a log frequency axis a constant linear bar width vanishes at high
    # frequency; the index bars must instead scale their width with each
    # centre frequency so the drawn width/f ratio is one constant.
    res = _intensity_wide()
    ax = res.plot()
    twin = next(a for a in ax.figure.axes if a is not ax)
    assert len(twin.patches) == 3
    centers = [p.get_x() + p.get_width() / 2.0 for p in twin.patches]
    ratios = [p.get_width() / c for p, c in zip(twin.patches, centers, strict=True)]
    # width/f is the same constant at 100 Hz and at 10 kHz (and in between).
    assert min(centers) == pytest.approx(100.0)
    assert max(centers) == pytest.approx(10000.0)
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-9)
    plt.close("all")


def test_intensity_without_band_data_raises() -> None:
    p1 = RNG.standard_normal(FS)
    res = ph.sound_intensity(p1, np.roll(p1, 1), FS, spacing=0.012)
    assert res.frequency is None
    with pytest.raises(ValueError, match="per-band"):
        res.plot()


# --------------------------------------------------------------------------
# Schroeder decay curve (backward compat + plot)
# --------------------------------------------------------------------------
def test_decay_curve_is_dataclass_and_unpacks_like_tuple() -> None:
    ir = _exp_ir()
    dc = ph.decay_curve(ir, FS)
    assert isinstance(dc, ph.DecayCurve)
    time, level = ph.decay_curve(ir, FS)  # backward-compatible unpacking
    np.testing.assert_array_equal(time, dc.time)
    np.testing.assert_array_equal(level, dc.level)
    assert dc.band is None


def test_decay_curve_records_band() -> None:
    dc = ph.decay_curve(_exp_ir(), FS, band=500.0)
    assert dc.band == 500.0


def test_decay_curve_plot_has_curve_and_fit_overlays() -> None:
    dc = ph.decay_curve(_exp_ir(seconds=1.0, t60=0.6), FS)
    ax = dc.plot()
    np.testing.assert_allclose(ax.lines[0].get_ydata(), dc.level)
    labels = [ln.get_label() for ln in ax.lines]
    assert any("fit" in str(lbl) for lbl in labels)
    assert "s]" in ax.get_xlabel()
    plt.close("all")


def test_decay_curve_plot_without_fits() -> None:
    dc = ph.decay_curve(_exp_ir(), FS)
    ax = dc.plot(fits=False)
    labels = [str(ln.get_label()) for ln in ax.lines]
    assert not any("fit" in lbl for lbl in labels)
    plt.close("all")


# --------------------------------------------------------------------------
# Open-plan spatial decay (ISO 3382-3)
# --------------------------------------------------------------------------
def test_open_plan_plot_line_and_markers() -> None:
    res = _open_plan()
    ax = res.plot()
    # the regression line passes through Lp,A,S,4m at 4 m.
    line = ax.lines[0]
    x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
    at4 = float(np.interp(4.0, x, y))
    assert at4 == pytest.approx(res.lp_as_4m, abs=0.05)
    # slope over one doubling equals -D2,S.
    at8 = float(np.interp(8.0, x, y))
    assert at4 - at8 == pytest.approx(res.d2s, abs=0.05)
    # rD / rP are marked as vertical lines at their distances.
    vlines = [
        np.asarray(ln.get_xdata())[0] for ln in ax.lines
        if np.asarray(ln.get_xdata()).size == 2
        and np.asarray(ln.get_xdata())[0] == np.asarray(ln.get_xdata())[1]
    ]
    assert any(v == pytest.approx(res.rd) for v in vlines)
    assert any(v == pytest.approx(res.rp) for v in vlines)
    plt.close("all")


def test_open_plan_plot_without_regression_raises() -> None:
    bare = ph.OpenPlanResult(
        d2s=float("nan"), lp_as_4m=float("nan"), rd=float("nan"), rp=float("nan")
    )
    with pytest.raises(ValueError, match="regression"):
        bare.plot()


# --------------------------------------------------------------------------
# Outdoor attenuation breakdown (ISO 9613-2)
# --------------------------------------------------------------------------
def test_outdoor_plot_stacks_terms_to_total() -> None:
    res = _outdoor()
    ax = res.plot()
    n = res.frequencies.size
    # four stacked terms -> 4 bars per band; signed heights sum to a_total.
    assert len(ax.patches) == 4 * n
    heights = np.array([p.get_height() for p in ax.patches]).reshape(4, n)
    np.testing.assert_allclose(heights.sum(axis=0), res.a_total, atol=1e-9)
    # the ground term is a net gain (negative) at 63 Hz in this scenario.
    assert res.a_gr[0] < 0.0
    # the total line echoes a_total.
    np.testing.assert_allclose(ax.lines[0].get_ydata(), res.a_total)
    plt.close("all")


# --------------------------------------------------------------------------
# Impedance tube (ISO 10534-2)
# --------------------------------------------------------------------------
def test_impedance_tube_plot_alpha_and_reflection() -> None:
    res = _impedance_tube()
    ax = res.plot()
    np.testing.assert_allclose(ax.lines[0].get_ydata(), res.absorption)
    np.testing.assert_allclose(ax.lines[1].get_ydata(), np.abs(res.reflection))
    assert ax.get_ylim() == (0.0, 1.05)
    plt.close("all")


def _transfer_matrix() -> tuple[ph.TransferMatrix, np.ndarray, float]:
    f = np.linspace(200.0, 1600.0, 40)
    rho_c = 407.0
    k = 2.0 * np.pi * f / 343.2
    tm = ph.air_layer_transfer_matrix(k, 0.05, rho_c)
    return tm, f, rho_c


def test_transfer_matrix_plot_tl_and_absorption() -> None:
    tm, f, rho_c = _transfer_matrix()
    ax = tm.plot(f, rho_c)
    # Primary axis: the Eq. (26) transmission loss (0 dB for a pure air layer).
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(), tm.transmission_loss(rho_c), atol=1e-12
    )
    # Twin axis: the Eq. (28) hard-backed absorption companion on a 0..1 scale.
    twins = [
        other for other in ax.figure.axes
        if other is not ax and other.bbox.bounds == ax.bbox.bounds
    ]
    assert len(twins) == 1
    np.testing.assert_allclose(
        twins[0].lines[0].get_ydata(), tm.absorption_hard_backed(rho_c)
    )
    assert twins[0].get_ylim() == (0.0, 1.05)
    plt.close("all")


def test_transfer_matrix_plot_forwards_kwargs_and_composes() -> None:
    tm, f, rho_c = _transfer_matrix()
    ax = tm.plot(f, rho_c, linewidth=2, color="red")
    line = ax.lines[0]
    assert line.get_linewidth() == 2.0
    assert plt.matplotlib.colors.to_rgba(line.get_color()) == (
        plt.matplotlib.colors.to_rgba("red")
    )
    plt.close("all")
    fig, external = plt.subplots()
    out = tm.plot(f, rho_c, ax=external)
    assert out is external
    plt.close(fig)


def test_transfer_matrix_plot_spanish_and_bad_language() -> None:
    tm, f, rho_c = _transfer_matrix()
    ax = tm.plot(f, rho_c, language="es")
    assert "ASTM E2611" in ax.get_title()
    assert "matriz de transferencia" in ax.get_title()
    plt.close("all")
    with pytest.raises(ValueError, match="language"):
        tm.plot(f, rho_c, language="fr")


def test_layered_absorber_plot_alpha_and_reflection() -> None:
    res = _layered_absorber()
    ax = res.plot()
    np.testing.assert_allclose(ax.lines[0].get_ydata(), res.absorption)
    np.testing.assert_allclose(ax.lines[1].get_ydata(), np.abs(res.reflection))
    assert ax.get_ylim() == (0.0, 1.05)
    plt.close("all")


def test_porous_medium_plot_normalized_components() -> None:
    res = _porous_medium()
    ax = res.plot()
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(), res.normalized_impedance.real
    )
    np.testing.assert_allclose(
        ax.lines[1].get_ydata(), -res.normalized_impedance.imag
    )
    assert len(ax.lines) == 4
    plt.close("all")


# --------------------------------------------------------------------------
# Monte Carlo output distribution (GUM Supplement 1)
# --------------------------------------------------------------------------
def test_monte_carlo_plot_histogram_and_interval() -> None:
    res = _monte_carlo()
    assert res.samples is not None and res.samples.size == res.trials
    ax = res.plot()
    bars = [
        p for p in ax.patches
        if "coverage interval" not in str(p.get_label())
    ]
    assert bars, "expected histogram bars"
    # the coverage-interval axvspan matches the result's interval.
    spans = [
        p for p in ax.patches
        if "coverage interval" in str(p.get_label())
    ]
    assert spans, "expected the coverage-interval axvspan"
    low, high = res.interval
    assert spans[0].get_x() == pytest.approx(low)
    assert spans[0].get_x() + spans[0].get_width() == pytest.approx(high)
    plt.close("all")


def test_monte_carlo_plot_without_samples_raises() -> None:
    res = ph.monte_carlo(
        lambda a, b, c: a + b + c, _MC_QUANTITIES, trials=200, seed=7
    )
    assert res.samples is None
    with pytest.raises(ValueError, match="keep_samples"):
        res.plot()


# --------------------------------------------------------------------------
# Occupational exposure (ISO 9612)
# --------------------------------------------------------------------------
def test_exposure_plot_task_bars_and_lex_line() -> None:
    res = _exposure()
    ax = res.plot()
    heights = [p.get_height() for p in ax.patches]
    np.testing.assert_allclose(
        heights, [t.lex_8h_contribution for t in res.tasks]
    )
    hlines = [
        np.asarray(ln.get_ydata())[0] for ln in ax.lines
        if np.asarray(ln.get_ydata()).size == 2
        and np.asarray(ln.get_ydata())[0] == np.asarray(ln.get_ydata())[1]
    ]
    assert any(v == pytest.approx(res.lex_8h) for v in hlines)
    assert any(v == pytest.approx(res.upper_limit) for v in hlines)
    plt.close("all")


def test_exposure_plot_without_tasks_raises() -> None:
    levels = np.full(5, 80.0)
    res = ph.job_based_exposure(levels, 6.0)
    assert not res.tasks
    with pytest.raises(ValueError, match="per-task"):
        res.plot()


# --------------------------------------------------------------------------
# Static airflow resistance (ISO 9053-1)
# --------------------------------------------------------------------------
def test_static_airflow_plot_curve_through_evaluation_point() -> None:
    res = _static_airflow()
    ax = res.plot()
    x, y = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
    # x is in mm/s; the fitted curve passes through the evaluation point.
    at_eval = float(np.interp(res.evaluation_velocity * 1e3, x, y))
    assert at_eval == pytest.approx(res.pressure_drop, rel=1e-3)
    plt.close("all")


# --------------------------------------------------------------------------
# EN 12354 predictions
# --------------------------------------------------------------------------
def test_airborne_prediction_plot_sorted_shares() -> None:
    res = _airborne_prediction()
    ax = res.plot()
    heights = [p.get_height() for p in ax.patches]
    assert heights == sorted(heights, reverse=True)
    assert sum(heights) == pytest.approx(100.0)
    assert f"{res.r_prime_w:.1f}" in ax.get_title()
    plt.close("all")


def test_impact_prediction_plot_terms() -> None:
    res = _impact_prediction()
    ax = res.plot()
    heights = [p.get_height() for p in ax.patches]
    np.testing.assert_allclose(
        heights,
        [res.ln_w_eq, -res.delta_l_w, res.k_correction, res.l_prime_n_w],
    )
    plt.close("all")


# --------------------------------------------------------------------------
# ISO 16283 field insulation spectra
# --------------------------------------------------------------------------
def test_airborne_insulation_plot_curves() -> None:
    res = _airborne_insulation()
    ax = res.plot()
    np.testing.assert_allclose(ax.lines[0].get_ydata(), res.dnt)
    np.testing.assert_allclose(ax.lines[1].get_ydata(), res.d)
    assert res.r_prime is not None
    np.testing.assert_allclose(ax.lines[2].get_ydata(), res.r_prime)
    plt.close("all")


def test_impact_insulation_plot_curves_and_label_kwarg() -> None:
    res = _impact_insulation()
    ax = res.plot(label="my measurement")
    np.testing.assert_allclose(ax.lines[0].get_ydata(), res.l_n_t)
    labels = [str(ln.get_label()) for ln in ax.lines]
    # user label styles only the primary curve; companions keep theirs.
    assert "my measurement" in labels
    assert any("L'_n" in lbl for lbl in labels)
    plt.close("all")


# --------------------------------------------------------------------------
# ISO 12999-1 band uncertainty
# --------------------------------------------------------------------------
def test_band_uncertainty_plot_spectrum() -> None:
    res = _band_uncertainty()
    ax = res.plot()
    freqs, u = res.to_arrays()
    np.testing.assert_allclose(ax.lines[0].get_xdata(), freqs)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), u)
    assert "12999" in ax.get_title()
    plt.close("all")


# --------------------------------------------------------------------------
# Common contract: ax=None creates a figure; passing ax composes
# --------------------------------------------------------------------------
def test_single_axes_plots_accept_external_ax() -> None:
    for res in (
        _zwicker_stationary(),
        _sti(),
        _airborne_rating(),
        _extended_rating(),
        _extended_impact_rating(),
        _sound_power(),
        _open_plan(),
        _outdoor(),
        _impedance_tube(),
        _porous_medium(),
        _layered_absorber(),
        _diffuse_absorption(),
        _monte_carlo(),
        _exposure(),
        _static_airflow(),
        _airborne_prediction(),
        _impact_prediction(),
        _airborne_insulation(),
        _impact_insulation(),
        _band_uncertainty(),
    ):
        fig, ax = plt.subplots()
        out = res.plot(ax=ax)
        assert out is ax
        plt.close(fig)
