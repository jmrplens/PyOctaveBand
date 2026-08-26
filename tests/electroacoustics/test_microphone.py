#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the IEC 60268-4 microphone rated characteristics and their
``.report()`` fiche (characteristics model + PDF rendering).

The four computed characteristics are checked against IEC 60268-4's own
definitions as clean-room oracles:

* **Sensitivity level** (11.1): ``L_M = 20 lg(M / 1 V/Pa)``, so 12,5 mV/Pa
  returns ``20 lg 0,0125 = -38,06`` dB re 1 V/Pa (hand-computed) and
  1 000 mV/Pa returns 0 dB exactly.
* **Effective frequency range** (12.2): a response crossing the
  ``+/- tolerance`` limits at chosen frequencies returns exactly those
  frequencies, on either the lower or the upper limit.
* **Directivity index** (13.2.2 via the 11.2.2 a) integral): the ideal
  cardioid ``(1 + cos theta) / 2`` returns ``10 lg 3 = 4,77`` dB.
* **Equivalent noise level** (17.2 d/e): ``20 lg((U_N / M) / 20 uPa)``, so
  2,5 uV over 12,5 mV/Pa is 200 uPa, i.e. 20,0 dB exactly, and the overload
  sound pressure level (15.2.2) is read from a distortion curve that reaches
  the stated limit at a chosen level.

The rendering itself is a feature, so those tests assert only structural facts:
a valid one-page PDF, the rated table content, translated Spanish output and
rejected engines/languages.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
from report_assertions import assert_one_page

from phonometry import ReportMetadata, electroacoustics

if TYPE_CHECKING:
    from pathlib import Path

_M_MV = 12.5
_TOL = 3.0


def _flat_response() -> tuple[np.ndarray, np.ndarray]:
    """A response flat at 0 dB with ramps crossing -3 dB at 40 and 18 000 Hz."""
    f = np.geomspace(20.0, 20000.0, 400)
    rel = np.zeros_like(f)
    f_lo, f_hi = 40.0, 18000.0
    f_a, f_b = 63.0, 15000.0
    below = f < f_a
    rel[below] = -_TOL * (np.log2(f_a / f[below]) / np.log2(f_a / f_lo))
    above = f > f_b
    rel[above] = -_TOL * (np.log2(f[above] / f_b) / np.log2(f_hi / f_b))
    return f, rel


def _extract_text(path: str) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages)


# --- IEC 60268-4 11.1/11.3 sensitivity level ---------------------------------


def test_sensitivity_level_is_20lg_m_over_1v_pa() -> None:
    """12,5 mV/Pa gives 20 lg 0,0125 = -38,06 dB re 1 V/Pa (11.1)."""
    f, rel = _flat_response()
    result = electroacoustics.microphone_characteristics(
        f, rel, _M_MV, tolerance_db=_TOL
    )
    assert result.sensitivity_level_db == pytest.approx(
        20.0 * math.log10(0.0125), abs=1e-12
    )
    assert result.sensitivity_level_db == pytest.approx(-38.0618, abs=5e-5)
    assert result.sensitivity_v_per_pa == pytest.approx(0.0125, rel=1e-12)


def test_reference_sensitivity_gives_zero_level() -> None:
    """M = 1 V/Pa (1 000 mV/Pa) is the reference: L_M = 0 dB exactly (11.1)."""
    f, rel = _flat_response()
    result = electroacoustics.microphone_characteristics(
        f, rel, 1000.0, tolerance_db=_TOL
    )
    assert result.sensitivity_level_db == pytest.approx(0.0, abs=1e-12)


# --- IEC 60268-4 12.2 effective frequency range -------------------------------


def test_effective_range_crosses_lower_limit_at_known_points() -> None:
    """The band edges are the frequencies where the response crosses -tol (12.2)."""
    f, rel = _flat_response()
    result = electroacoustics.microphone_characteristics(
        f, rel, _M_MV, tolerance_db=_TOL
    )
    lo, hi = result.effective_range
    assert lo == pytest.approx(40.0, rel=1e-6)
    assert hi == pytest.approx(18000.0, rel=1e-6)


def test_effective_range_crosses_upper_limit() -> None:
    """A rising response bounds the range where it crosses +tol (12.2)."""
    f, rel = _flat_response()
    rel = rel.copy()
    # A linear-in-log rise above 8 kHz crossing +3 dB at exactly 12 kHz.
    above = f > 8000.0
    rel[above] += _TOL * (np.log2(f[above] / 8000.0) / np.log2(12000.0 / 8000.0))
    result = electroacoustics.microphone_characteristics(
        f, rel, _M_MV, tolerance_db=_TOL
    )
    lo, hi = result.effective_range
    assert lo == pytest.approx(40.0, rel=1e-6)
    assert hi == pytest.approx(12000.0, rel=1e-4)


def test_response_is_normalized_at_reference_frequency() -> None:
    """A constant offset is removed: the response is 0 dB at 1 kHz (12.1.1)."""
    f, rel = _flat_response()
    result = electroacoustics.microphone_characteristics(
        f, rel + 7.0, _M_MV, tolerance_db=_TOL
    )
    idx = int(np.argmin(np.abs(f - 1000.0)))
    assert result.response_db[idx] == pytest.approx(0.0, abs=1e-9)
    assert result.effective_range[0] == pytest.approx(40.0, rel=1e-6)


# --- IEC 60268-4 13.2.2 directivity index -------------------------------------


def test_cardioid_directivity_index_is_10lg3() -> None:
    """The ideal cardioid returns D = 10 lg 3 = 4,77 dB (13.2.2 / 11.2.2 a)."""
    f, rel = _flat_response()
    angles = np.linspace(0.0, 179.9, 1800)
    pattern = 20.0 * np.log10((1.0 + np.cos(np.radians(angles))) / 2.0)
    result = electroacoustics.microphone_characteristics(
        f,
        rel,
        _M_MV,
        tolerance_db=_TOL,
        directivity=electroacoustics.MicrophoneDirectivity(
            polar=(angles, pattern), frequency=1000.0
        ),
    )
    assert result.directivity_index_db == pytest.approx(
        10.0 * math.log10(3.0), abs=5e-3
    )
    # 11.2.2.1: diffuse-field level = free-field level - directivity index.
    assert result.diffuse_field_sensitivity_level_db == pytest.approx(
        result.sensitivity_level_db - result.directivity_index_db, abs=1e-12
    )


def test_omnidirectional_directivity_index_is_zero() -> None:
    """A uniform pattern returns D = 0 dB (13.2.2)."""
    f, rel = _flat_response()
    angles = np.linspace(0.0, 180.0, 721)
    result = electroacoustics.microphone_characteristics(
        f,
        rel,
        _M_MV,
        tolerance_db=_TOL,
        directivity=electroacoustics.MicrophoneDirectivity(
            polar=(angles, np.zeros_like(angles))
        ),
    )
    # Trapezoidal quadrature of the 11.2.2 a) integral over 0,25 degree steps.
    assert result.directivity_index_db == pytest.approx(0.0, abs=1e-4)


def test_stated_directivity_index_is_kept() -> None:
    """A stated directivity index overrides the computed one (13.2.1)."""
    f, rel = _flat_response()
    angles = np.linspace(0.0, 180.0, 721)
    result = electroacoustics.microphone_characteristics(
        f,
        rel,
        _M_MV,
        tolerance_db=_TOL,
        directivity=electroacoustics.MicrophoneDirectivity(
            polar=(angles, np.zeros_like(angles)), index_db=4.5
        ),
    )
    assert result.directivity_index_db == pytest.approx(4.5)


def test_front_only_pattern_gives_no_directivity_index() -> None:
    """A pattern that stops at 90 degrees cannot feed the 11.2.2 a) integral."""
    f, rel = _flat_response()
    angles = np.linspace(0.0, 90.0, 91)
    result = electroacoustics.microphone_characteristics(
        f,
        rel,
        _M_MV,
        tolerance_db=_TOL,
        directivity=electroacoustics.MicrophoneDirectivity(
            polar=(angles, np.zeros_like(angles))
        ),
    )
    assert result.directivity_index_db is None


def test_full_circle_cardioid_gives_same_directivity_index() -> None:
    """A 0..360 pattern folds onto 0..180: the cardioid still gives 10 lg 3."""
    f, rel = _flat_response()
    angles = np.arange(0.0, 360.0, 0.25)
    angles = angles[angles != 180.0]  # the exact null is -inf dB
    pattern = 20.0 * np.log10((1.0 + np.cos(np.radians(angles))) / 2.0)
    result = electroacoustics.microphone_characteristics(
        f,
        rel,
        _M_MV,
        tolerance_db=_TOL,
        directivity=electroacoustics.MicrophoneDirectivity(polar=(angles, pattern)),
    )
    assert result.directivity_index_db == pytest.approx(
        10.0 * math.log10(3.0), abs=5e-3
    )


def test_front_quarter_beyond_270_gives_no_directivity_index() -> None:
    """Angles 270..360 fold onto 0..90, too short for the 11.2.2 a) integral."""
    f, rel = _flat_response()
    angles = np.linspace(270.0, 360.0, 91)
    result = electroacoustics.microphone_characteristics(
        f,
        rel,
        _M_MV,
        tolerance_db=_TOL,
        directivity=electroacoustics.MicrophoneDirectivity(
            polar=(angles, np.zeros_like(angles))
        ),
    )
    assert result.directivity_index_db is None


# --- IEC 60268-4 17.2 equivalent noise level and 15.2 overload SPL ------------


def test_equivalent_noise_level_from_noise_voltage() -> None:
    """2,5 uV over 12,5 mV/Pa is 200 uPa = 20,0 dB SPL exactly (17.2 d/e)."""
    f, rel = _flat_response()
    result = electroacoustics.microphone_characteristics(
        f,
        rel,
        _M_MV,
        tolerance_db=_TOL,
        noise=electroacoustics.MicrophoneNoise(voltage=2.5e-6),
    )
    assert result.equivalent_noise_level_db == pytest.approx(20.0, abs=1e-12)
    # SNR re 1 Pa: 20 lg(1 Pa / 20 uPa) - L_N = 93,98 - 20,0.
    assert result.signal_to_noise_ratio_db == pytest.approx(
        20.0 * math.log10(1.0 / 20e-6) - 20.0, abs=1e-12
    )


def test_overload_spl_read_from_distortion_curve() -> None:
    """The overload SPL is where the THD reaches the stated limit (15.2.2)."""
    f, rel = _flat_response()
    spl = np.linspace(100.0, 140.0, 81)  # includes 130,0 dB exactly
    thd = 0.5 * 10.0 ** ((spl - 130.0) * 0.08)
    result = electroacoustics.microphone_characteristics(
        f,
        rel,
        _M_MV,
        tolerance_db=_TOL,
        overload=electroacoustics.MicrophoneOverload(
            distortion=(spl, thd), thd_percent=0.5
        ),
    )
    assert result.max_spl_db == pytest.approx(130.0, abs=1e-9)


def test_stated_max_spl_is_kept() -> None:
    """A stated overload SPL overrides the distortion-curve reading (15.2.1)."""
    f, rel = _flat_response()
    spl = np.linspace(100.0, 140.0, 81)
    thd = 0.5 * 10.0 ** ((spl - 130.0) * 0.08)
    result = electroacoustics.microphone_characteristics(
        f,
        rel,
        _M_MV,
        tolerance_db=_TOL,
        overload=electroacoustics.MicrophoneOverload(
            distortion=(spl, thd), thd_percent=0.5, spl_db=132.0
        ),
    )
    assert result.max_spl_db == pytest.approx(132.0)


def test_distortion_below_limit_gives_no_max_spl() -> None:
    """A distortion curve that never reaches the limit yields no overload SPL."""
    f, rel = _flat_response()
    spl = np.linspace(100.0, 120.0, 41)
    thd = np.full_like(spl, 0.05)
    result = electroacoustics.microphone_characteristics(
        f,
        rel,
        _M_MV,
        tolerance_db=_TOL,
        overload=electroacoustics.MicrophoneOverload(distortion=(spl, thd)),
    )
    assert result.max_spl_db is None


# --- model validation ----------------------------------------------------------


def test_sensitivity_must_be_positive() -> None:
    f, rel = _flat_response()
    with pytest.raises(ValueError, match="sensitivity_mv_per_pa"):
        electroacoustics.microphone_characteristics(f, rel, 0.0)


def test_mismatched_response_lengths_rejected() -> None:
    with pytest.raises(ValueError, match="equal length"):
        electroacoustics.microphone_characteristics(
            [100.0, 200.0, 400.0], [0.0, 0.0], _M_MV
        )


def test_reference_frequency_outside_band_rejected() -> None:
    f, rel = _flat_response()
    with pytest.raises(ValueError, match="reference_frequency"):
        electroacoustics.microphone_characteristics(
            f, rel, _M_MV, reference_frequency=30000.0
        )


def test_noise_voltage_and_stated_level_conflict() -> None:
    f, rel = _flat_response()
    conflicting_noise = electroacoustics.MicrophoneNoise(
        voltage=1e-6, equivalent_level_db=14.0
    )
    with pytest.raises(ValueError, match="not both"):
        electroacoustics.microphone_characteristics(
            f, rel, _M_MV, noise=conflicting_noise
        )


def test_nonpositive_frequencies_rejected() -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        electroacoustics.microphone_characteristics(
            [0.0, 100.0, 1000.0], [0.0, 0.0, 0.0], _M_MV
        )


def test_empty_distortion_rejected() -> None:
    f, rel = _flat_response()
    empty_distortion = electroacoustics.MicrophoneOverload(distortion=([], []))
    with pytest.raises(ValueError, match="at least two"):
        electroacoustics.microphone_characteristics(
            f, rel, _M_MV, overload=empty_distortion
        )


def test_empty_noise_spectrum_rejected() -> None:
    f, rel = _flat_response()
    empty_spectrum = electroacoustics.MicrophoneNoise(spectrum=([], []))
    with pytest.raises(ValueError, match="at least two"):
        electroacoustics.microphone_characteristics(f, rel, _M_MV, noise=empty_spectrum)


def test_empty_polar_rejected() -> None:
    f, rel = _flat_response()
    empty_polar = electroacoustics.MicrophoneDirectivity(polar=([], []))
    with pytest.raises(ValueError, match="at least two angle points"):
        electroacoustics.microphone_characteristics(
            f, rel, _M_MV, directivity=empty_polar
        )


def test_mismatched_polar_pair_rejected() -> None:
    """A pattern with more angles than levels names both shapes."""
    f, rel = _flat_response()
    ragged_polar = electroacoustics.MicrophoneDirectivity(
        polar=([0.0, 90.0, 180.0], [0.0, -3.0])
    )
    with pytest.raises(ValueError, match="'polar angles'.*same shape"):
        electroacoustics.microphone_characteristics(
            f, rel, _M_MV, directivity=ragged_polar
        )


def test_two_dimensional_polar_rejected() -> None:
    """Angles and levels agreeing on a two-dimensional shape are still refused."""
    f, rel = _flat_response()
    grid_polar = electroacoustics.MicrophoneDirectivity(
        polar=([[0.0, 90.0], [180.0, 270.0]], [[0.0, -3.0], [-6.0, -3.0]])
    )
    with pytest.raises(ValueError, match="'polar' angles and levels must be 1-D"):
        electroacoustics.microphone_characteristics(
            f, rel, _M_MV, directivity=grid_polar
        )


def test_nonfinite_stated_directivity_index_rejected_with_polar() -> None:
    """A non-finite stated DI is rejected whether or not a pattern is given."""
    f, rel = _flat_response()
    angles = np.linspace(0.0, 180.0, 181)
    omnidirectional = np.zeros_like(angles)
    infinite_index_with_polar = electroacoustics.MicrophoneDirectivity(
        polar=(angles, omnidirectional), index_db=float("inf")
    )
    with pytest.raises(ValueError, match=r"directivity\.index_db"):
        electroacoustics.microphone_characteristics(
            f, rel, _M_MV, directivity=infinite_index_with_polar
        )
    nan_index = electroacoustics.MicrophoneDirectivity(index_db=float("nan"))
    with pytest.raises(ValueError, match=r"directivity\.index_db"):
        electroacoustics.microphone_characteristics(
            f, rel, _M_MV, directivity=nan_index
        )


def test_no_noise_input_gives_no_noise_rows() -> None:
    f, rel = _flat_response()
    result = electroacoustics.microphone_characteristics(
        f, rel, _M_MV, tolerance_db=_TOL
    )
    assert result.equivalent_noise_level_db is None
    assert result.signal_to_noise_ratio_db is None


# --- rendering -------------------------------------------------------------------


def _example_result() -> electroacoustics.MicrophoneCharacteristics:
    f, rel = _flat_response()
    angles = np.linspace(0.0, 179.0, 359)
    pattern = 20.0 * np.log10((1.0 + np.cos(np.radians(angles))) / 2.0)
    spl = np.linspace(100.0, 140.0, 81)
    thd = 0.5 * 10.0 ** ((spl - 130.0) * 0.08)
    nf = np.geomspace(20.0, 20000.0, 31)
    nl = 18.0 - 5.4 * np.log2(nf / 20.0)
    return electroacoustics.microphone_characteristics(
        f,
        rel,
        _M_MV,
        tolerance_db=_TOL,
        directivity=electroacoustics.MicrophoneDirectivity(
            polar=(angles, pattern), frequency=1000.0
        ),
        noise=electroacoustics.MicrophoneNoise(voltage=1.25e-6, spectrum=(nf, nl)),
        overload=electroacoustics.MicrophoneOverload(
            distortion=(spl, thd), thd_percent=0.5
        ),
        electrical=electroacoustics.MicrophoneElectrical(
            rated_impedance=150.0,
            minimum_load_impedance=1000.0,
            powering="Phantom P48 (IEC 61938)",
            supply_current_ma=3.1,
        ),
    )


def test_report_renders_one_page_with_rated_table(tmp_path: Path) -> None:
    """The fiche renders a valid one-page PDF listing the rated characteristics."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    result = _example_result()
    md = ReportMetadata(
        manufacturer="Example audio",
        specimen="Cardioid condenser microphone",
        measurement_standard="IEC 60268-4",
        report_id="PHN-60268-4",
        requirement=16.0,
    )
    out = tmp_path / "microphone.pdf"
    returned = result.report(str(out), metadata=md)
    assert returned == str(out)
    assert_one_page(str(out))
    # The table cell labels can wrap across lines in the PDF text layer, so the
    # assertions use single-line fragments.
    text = _extract_text(str(out))
    assert "Microphone characteristics" in text
    assert "Free-field sensitivity" in text
    assert "Rated impedance" in text
    assert "Signal-to-noise ratio" in text
    assert "PASS" in text


def test_report_without_optional_panels(tmp_path: Path) -> None:
    """A response-only result (no polar/noise/distortion) still renders."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    f, rel = _flat_response()
    result = electroacoustics.microphone_characteristics(
        f, rel, _M_MV, tolerance_db=_TOL
    )
    out = tmp_path / "microphone_min.pdf"
    result.report(str(out))
    assert_one_page(str(out))


def test_spanish_report_renders_translated_fiche(tmp_path: Path) -> None:
    """language="es" renders a one-page Spanish fiche."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    result = _example_result()
    out = tmp_path / "microphone_es.pdf"
    result.report(str(out), language="es")
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Características del micrófono" in text
    assert "Sensibilidad en campo libre" in text


def test_plot_each_quantity_returns_single_axes() -> None:
    """Every quantity plots one concept on one axes; directivity is polar."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = _example_result()
    assert result.plot().get_title() == "Free-field response"
    plt.close("all")
    for quantity in ("response", "directivity", "noise", "distortion"):
        ax = result.plot(quantity=quantity)
        assert not isinstance(ax, np.ndarray)
        expected = "polar" if quantity == "directivity" else "rectilinear"
        assert ax.name == expected
        plt.close("all")


def test_plot_on_external_axes_returns_it() -> None:
    """Passing an axes draws on it and returns that same axes."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _fig, ax = plt.subplots()
    out = _example_result().plot(quantity="noise", ax=ax)
    assert out is ax
    assert ax.get_title() == "Inherent noise spectrum"
    plt.close("all")


def test_plot_rejects_unknown_quantity_and_missing_data() -> None:
    """An unknown quantity, and a quantity with no data, raise ValueError."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    result = _example_result()
    with pytest.raises(ValueError, match="unknown quantity"):
        result.plot(quantity="bogus")
    f, rel = _flat_response()
    bare = electroacoustics.microphone_characteristics(f, rel, _M_MV, tolerance_db=_TOL)
    with pytest.raises(ValueError, match="no directional pattern"):
        bare.plot(quantity="directivity")


def test_unknown_engine_rejected(tmp_path: Path) -> None:
    """An unknown rendering engine raises ValueError."""
    result = _example_result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        result.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path: Path) -> None:
    """An unknown fiche language raises ValueError."""
    result = _example_result()
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match="language"):
        result.report(out, language="xx")


# --------------------------------------------------------------------------
# A curve the data sheet could not have measured
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("frequencies", float("nan")),
        ("frequencies", float("inf")),
        ("response_db", float("nan")),
        ("polar_db", float("nan")),
        ("noise_band_levels_db", float("nan")),
    ],
)
def test_a_non_finite_curve_is_refused_where_the_result_is_built(
    field: str, bad: float
) -> None:
    """No producer emits one, and the fiche cannot say what it would mean.

    Every curve reaches the result through the shared curve validator, which
    already refuses a non-finite frequency or value, so a NaN here is never a
    measurement outcome. Written in afterwards it used to travel to the sheet
    unmeasured: a NaN frequency makes the span of the response unmeasurable
    and the panel is scaled by the decades that span covers, so the render
    stopped inside matplotlib with ``Axis limits cannot be NaN or Inf``,
    naming neither the field nor this result; a NaN level was quieter, drawn
    as a gap indistinguishable from a frequency never measured.
    """
    result = _example_result()
    spoilt = np.array(getattr(result, field), dtype=float)
    spoilt[0] = bad
    with pytest.raises(
        ValueError,
        match=rf"MicrophoneCharacteristics: '{field}' must contain only finite",
    ):
        dataclasses.replace(result, **{field: spoilt})


def test_a_non_finite_response_axis_no_longer_reaches_the_fiche(
    tmp_path: Path,
) -> None:
    """The render that used to fail anonymously cannot now be asked for.

    This is the route the sheet took: construction accepted the axis, the
    panel scaling turned it into a ``nan`` box aspect, and matplotlib stopped
    the fiche naming nothing. The refusal now lands before the PDF is opened.
    """
    result = _example_result()
    axis = np.array(result.frequencies, dtype=float)
    axis[10] = float("nan")
    out = tmp_path / "non_finite_axis.pdf"
    with pytest.raises(ValueError, match=r"'frequencies' must contain only finite"):
        dataclasses.replace(result, frequencies=axis).report(str(out))
    assert not out.exists()


# --------------------------------------------------------------------------
# A rated number the sheet could not have measured
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "field",
    [
        "sensitivity_mv_per_pa",
        "sensitivity_level_db",
        "reference_frequency",
        "polar_frequency",
        "equivalent_noise_level_db",
        "tolerance_db",
        "directivity_index_db",
    ],
)
def test_a_non_finite_rated_number_is_refused_where_the_result_is_built(
    field: str,
) -> None:
    """The single numbers are pinned by name, as the curves already were.

    Every one of them is computed or validated by
    :func:`microphone_characteristics`, so no producer emits a NaN here. The
    curve guard never saw them: only the arrays were checked, and a scalar
    written in afterwards travelled to the sheet untouched.
    """
    result = _example_result()
    with pytest.raises(
        ValueError,
        match=rf"MicrophoneCharacteristics: '{field}' must be finite",
    ):
        dataclasses.replace(result, **{field: float("nan")})


def test_a_non_finite_sensitivity_no_longer_reaches_the_fiche(
    tmp_path: Path,
) -> None:
    """The rated table and the boxed result used to disagree with themselves.

    A NaN sensitivity passed the shape-only guard and printed ``nan mV/Pa`` in
    the rated-characteristics table beside a boxed headline that still gave
    the level re 1 V/Pa to a tenth of a decibel: an accredited sheet stating
    a sensitivity it did not have. The refusal now lands before the PDF is
    opened.
    """
    result = _example_result()
    out = tmp_path / "non_finite_sensitivity.pdf"
    with pytest.raises(ValueError, match=r"'sensitivity_mv_per_pa' must be finite"):
        dataclasses.replace(result, sensitivity_mv_per_pa=float("nan")).report(str(out))
    assert not out.exists()


def test_a_non_finite_polar_frequency_no_longer_reaches_the_fiche(
    tmp_path: Path,
) -> None:
    """The other end of the same gap: a NaN hertz killed the render outright.

    The directivity-index label rounds its frequency to whole hertz, so a NaN
    stopped the fiche with ``cannot convert float NaN to integer`` -- an error
    naming neither the field nor this result.
    """
    result = _example_result()
    out = tmp_path / "non_finite_polar_frequency.pdf"
    with pytest.raises(ValueError, match=r"'polar_frequency' must be finite"):
        dataclasses.replace(result, polar_frequency=float("nan")).report(str(out))
    assert not out.exists()


def test_an_effective_range_that_is_not_a_pair_is_refused() -> None:
    """The fiche unpacks the range positionally into its range text.

    ``_range_text(*result.effective_range, language=...)`` turns a triple into
    a ``TypeError`` about the helper's own ``language`` argument, naming
    neither the field nor the result; the guard names the field instead.
    """
    result = _example_result()
    with pytest.raises(
        ValueError,
        match=r"MicrophoneCharacteristics: 'effective_range' must be a \(lo, hi\) pair",
    ):
        dataclasses.replace(result, effective_range=(45.0, 18000.0, 3.0))


# --------------------------------------------------------------------------
# A weighting the measurement standard does not define
# --------------------------------------------------------------------------
@pytest.mark.parametrize("weighting", ["A<b", "Z", "a"])
def test_a_weighting_outside_iec_60268_1_is_refused_at_construction(
    weighting: str,
) -> None:
    """The tag reaches the fiche inside ``dB(...)`` markup, so it is pinned.

    IEC 60268-1 defines the A-weighted r.m.s. and the CCIR quasi-peak
    wide-band inherent-noise measurements, and the fiche interpolates whatever
    it is given between the parentheses of its noise rows and its verdict. An
    arbitrary tag was at best a wrong accredited label; a tag-like one reached
    reportlab's paragraph parser and ended the render with ``parse ended with
    1 unclosed tags``, naming neither the field nor the result.
    """
    f, rel = _flat_response()
    with pytest.raises(ValueError, match=r"'noise_weighting' must be one of"):
        electroacoustics.microphone_characteristics(
            f,
            rel,
            _M_MV,
            tolerance_db=_TOL,
            noise=electroacoustics.MicrophoneNoise(
                equivalent_level_db=14.0, weighting=weighting
            ),
        )


def test_the_ccir_quasi_peak_weighting_is_accepted(tmp_path: Path) -> None:
    """The other measurement of IEC 60268-1 6.2 still renders its own label.

    The guard pins the tag to the two wide-band measurements the standard
    defines, so the psophometric one must reach the sheet intact.
    """
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    f, rel = _flat_response()
    result = electroacoustics.microphone_characteristics(
        f,
        rel,
        _M_MV,
        tolerance_db=_TOL,
        noise=electroacoustics.MicrophoneNoise(
            equivalent_level_db=14.0, weighting="CCIR"
        ),
    )
    out = tmp_path / "ccir.pdf"
    result.report(str(out))
    assert "dB(CCIR)" in _extract_text(str(out))
