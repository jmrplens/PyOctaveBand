#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for the IEC 60268-5 loudspeaker rated characteristics and their
``.report()`` fiche (characteristics model + PDF rendering).

The two computed characteristics are checked against IEC 60268-5's own
definitions as clean-room oracles:

* **Characteristic sensitivity level** (20.3/20.4): a flat ``L0`` response
  driven at ``sqrt(R)`` volts and measured at 1 m returns ``L0`` exactly (1 W
  into ``R``); doubling the drive voltage subtracts 6,02 dB, and doubling the
  distance while doubling the voltage cancels back to ``L0``.
* **Effective frequency range** (21.2): a response crossing the ``reference −
  10 dB`` threshold at chosen frequencies returns exactly those frequencies,
  and a trough narrower than 1/9 octave at that level is neglected.

The rendering itself is a feature, so those tests assert only structural facts:
a valid one-page PDF, the rated table content, translated Spanish output and
rejected engines/languages.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from phonometry import (
    ReportMetadata,
    loudspeaker_characteristics,
    radiating_piston,
)

_PDF_MAGIC = b"%PDF"
_R = 8.0
_L0 = 90.0


def _flat_response() -> tuple[np.ndarray, np.ndarray]:
    """A response flat at ``_L0`` with ramps crossing ``_L0 - 10`` at 50/18000 Hz."""
    f = np.geomspace(20.0, 20000.0, 400)
    spl = np.full_like(f, _L0)
    f_lo, f_hi = 50.0, 18000.0
    f_a, f_b = 80.0, 15000.0
    below = f < f_a
    spl[below] = _L0 - 10.0 * (np.log2(f_a / f[below]) / np.log2(f_a / f_lo))
    above = f > f_b
    spl[above] = _L0 - 10.0 * (np.log2(f[above] / f_b) / np.log2(f_hi / f_b))
    return f, spl


def _assert_one_page(path: str) -> None:
    from pypdf import PdfReader

    with open(path, "rb") as handle:
        assert handle.read(4) == _PDF_MAGIC
    assert os.path.getsize(path) > 0
    assert len(PdfReader(path).pages) == 1


def _extract_text(path: str) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages)


# --- IEC 60268-5 20.3/20.4 characteristic sensitivity ------------------------


def test_characteristic_sensitivity_is_band_mean_at_1w_1m() -> None:
    """A flat L0 driven at sqrt(R) volts at 1 m gives sensitivity level L0 (20.3)."""
    f, spl = _flat_response()
    result = loudspeaker_characteristics(f, spl, _R, sensitivity_band=(200.0, 4000.0))
    assert result.sensitivity_level_db == pytest.approx(_L0, abs=1e-9)
    # Default drive is sqrt(R): 1 W into R (the 2,83 V @ 8 ohm convention).
    assert result.input_voltage == pytest.approx(math.sqrt(_R))


def test_sensitivity_drive_voltage_correction() -> None:
    """Doubling the drive voltage lowers the 1 W sensitivity by 20 lg 2 dB."""
    f, spl = _flat_response()
    result = loudspeaker_characteristics(
        f, spl, _R, input_voltage=2.0 * math.sqrt(_R), sensitivity_band=(200.0, 4000.0)
    )
    assert result.sensitivity_level_db == pytest.approx(_L0 - 20.0 * math.log10(2.0), abs=1e-9)


def test_sensitivity_distance_correction_cancels() -> None:
    """A 2 m distance with a doubled voltage cancels back to L0 (20.3.2)."""
    f, spl = _flat_response()
    result = loudspeaker_characteristics(
        f, spl, _R, input_voltage=2.0 * math.sqrt(_R), distance=2.0,
        sensitivity_band=(200.0, 4000.0),
    )
    assert result.sensitivity_level_db == pytest.approx(_L0, abs=1e-9)


def test_characteristic_sensitivity_pressure() -> None:
    """The characteristic sensitivity in Pa is 20 uPa * 10 ** (L/20)."""
    f, spl = _flat_response()
    result = loudspeaker_characteristics(f, spl, _R, sensitivity_band=(200.0, 4000.0))
    assert result.characteristic_sensitivity_pa == pytest.approx(
        20e-6 * 10.0 ** (_L0 / 20.0), rel=1e-9
    )


# --- IEC 60268-5 21.2 effective frequency range ------------------------------


def test_effective_range_crosses_threshold_at_known_points() -> None:
    """The band edges are the frequencies where the response crosses ref - 10 dB."""
    f, spl = _flat_response()
    result = loudspeaker_characteristics(f, spl, _R, sensitivity_band=(200.0, 4000.0))
    assert result.reference_level_db == pytest.approx(_L0, abs=1e-9)
    lo, hi = result.effective_range
    assert lo == pytest.approx(50.0, rel=1e-6)
    assert hi == pytest.approx(18000.0, rel=1e-6)


def test_narrow_trough_is_neglected() -> None:
    """A single-sample trough (< 1/9 octave) below the threshold is ignored (21.2)."""
    f, spl = _flat_response()
    spl = spl.copy()
    spl[int(np.argmin(np.abs(f - 1000.0)))] = _L0 - 15.0
    result = loudspeaker_characteristics(f, spl, _R, sensitivity_band=(200.0, 4000.0))
    lo, hi = result.effective_range
    assert lo == pytest.approx(50.0, rel=1e-6)
    assert hi == pytest.approx(18000.0, rel=1e-6)


def test_minimum_impedance_over_effective_range() -> None:
    """Without a rated range the minimum impedance falls back to the effective range."""
    f, spl = _flat_response()
    fz = np.geomspace(20.0, 20000.0, 200)
    z = 7.0 + 20.0 * np.exp(-((np.log2(fz / 40.0)) ** 2) / 0.1)  # peak below the range
    result = loudspeaker_characteristics(
        f, spl, _R, sensitivity_band=(200.0, 4000.0), impedance=(fz, z)
    )
    # Within [50, 18000] Hz the modulus floor is ~7 ohm (>= 80 % of 8 = 6.4).
    assert result.minimum_impedance == pytest.approx(7.0, abs=0.2)
    assert result.minimum_impedance >= 0.8 * _R


def _dip_below_effective_range() -> tuple[np.ndarray, np.ndarray]:
    """An impedance curve whose modulus dips to 4 ohm at 40 Hz, 7 ohm elsewhere.

    40 Hz is below the ~50 Hz lower edge of the effective range computed from
    ``_flat_response`` but inside a woofer-style rated range starting at 30 Hz.
    """
    fz = np.unique(np.append(np.geomspace(20.0, 20000.0, 200), 40.0))
    z = 7.0 - 3.0 * np.exp(-((np.log2(fz / 40.0)) ** 2) / 0.02)
    return fz, z


def test_minimum_impedance_uses_rated_range_when_supplied() -> None:
    """16.1 scans the rated frequency range: a dip outside the effective range counts."""
    f, spl = _flat_response()
    fz, z = _dip_below_effective_range()
    result = loudspeaker_characteristics(
        f, spl, _R, sensitivity_band=(200.0, 4000.0),
        rated_frequency_range=(30.0, 20000.0), impedance=(fz, z),
    )
    lo_eff, _ = result.effective_range
    assert lo_eff > 40.0  # the dip sits outside the computed effective range
    assert result.minimum_impedance == pytest.approx(4.0, abs=1e-9)
    assert result.minimum_impedance < 0.8 * _R  # the 16.1 check must fail here


def test_minimum_impedance_without_rated_range_misses_out_of_band_dip() -> None:
    """The effective-range fallback ignores a dip below its lower edge (19.1 NOTE 2)."""
    f, spl = _flat_response()
    fz, z = _dip_below_effective_range()
    result = loudspeaker_characteristics(
        f, spl, _R, sensitivity_band=(200.0, 4000.0), impedance=(fz, z)
    )
    assert result.minimum_impedance == pytest.approx(7.0, abs=0.2)
    assert result.minimum_impedance >= 0.8 * _R


# --- model validation --------------------------------------------------------


def test_rated_impedance_must_be_positive() -> None:
    f, spl = _flat_response()
    with pytest.raises(ValueError, match="rated_impedance"):
        loudspeaker_characteristics(f, spl, 0.0)


def test_mismatched_response_lengths_rejected() -> None:
    with pytest.raises(ValueError, match="equal length"):
        loudspeaker_characteristics([100.0, 200.0, 400.0], [90.0, 90.0], _R)


def test_sensitivity_band_out_of_range_rejected() -> None:
    f, spl = _flat_response()
    with pytest.raises(ValueError, match="no on-axis response samples"):
        loudspeaker_characteristics(f, spl, _R, sensitivity_band=(30000.0, 40000.0))


def test_distortion_from_swept_sine_result() -> None:
    """A SweptSineDistortionResult feeds the THD panel (thd ratio -> %)."""
    import phonometry as ph

    fs = 48000
    sweep = ph.synchronized_sweep_signal(fs, 100.0, 5000.0, 1.0)
    a2, a3 = 0.05, 0.02
    y = sweep + a2 * sweep**2 + a3 * sweep**3
    swept = ph.swept_sine_distortion(y, fs, 100.0, 5000.0, 1.0)
    f, spl = _flat_response()
    result = loudspeaker_characteristics(
        f, spl, _R, sensitivity_band=(200.0, 4000.0), distortion=swept
    )
    assert result.thd_percent is not None
    assert np.all(result.thd_percent >= 0.0)


# --- rendering ---------------------------------------------------------------


def _example_result():
    f, spl = _flat_response()
    angles = np.radians(np.linspace(0.0, 90.0, 40))
    pist = radiating_piston(0.075, np.array([1000.0, 2000.0, 4000.0]), angles=angles)
    fz = np.geomspace(20.0, 20000.0, 200)
    z = 6.6 + 20.0 * np.exp(-((np.log2(fz / 55.0)) ** 2) / 0.12)
    ft = np.geomspace(50.0, 5000.0, 100)
    thd = 0.4 + 2.0 * np.exp(-((np.log2(ft / 70.0)) ** 2) / 0.4)
    return loudspeaker_characteristics(
        f, spl, _R, sensitivity_band=(200.0, 4000.0),
        rated_frequency_range=(45.0, 20000.0), rated_noise_power=80.0,
        resonance_frequency=55.0, impedance=(fz, z), distortion=(ft, thd),
        directivity=pist, polar_frequency=2000.0,
    )


def test_report_renders_one_page_with_rated_table(tmp_path) -> None:
    """The fiche renders a valid one-page PDF listing the rated characteristics."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    result = _example_result()
    md = ReportMetadata(
        manufacturer="Example audio", specimen="Two-way loudspeaker",
        measurement_standard="IEC 60268-5", report_id="PHN-60268-5", requirement=84.0,
    )
    out = tmp_path / "loudspeaker.pdf"
    returned = result.report(str(out), metadata=md)
    assert returned == str(out)
    _assert_one_page(str(out))
    # The table cell labels can wrap across lines in the PDF text layer, so the
    # assertions use single-line fragments.
    text = _extract_text(str(out))
    assert "Loudspeaker characteristics" in text
    assert "Rated impedance" in text
    assert "Effective frequency" in text
    assert "Characteristic sensitivity" in text
    assert "PASS" in text


def test_report_without_optional_panels(tmp_path) -> None:
    """A response-only result (no impedance/THD/polar) still renders."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    f, spl = _flat_response()
    result = loudspeaker_characteristics(f, spl, _R, sensitivity_band=(200.0, 4000.0))
    out = tmp_path / "loudspeaker_min.pdf"
    result.report(str(out))
    _assert_one_page(str(out))


def test_spanish_report_renders_translated_fiche(tmp_path) -> None:
    """language="es" renders a one-page Spanish fiche."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    result = _example_result()
    out = tmp_path / "loudspeaker_es.pdf"
    result.report(str(out), language="es")
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Características del altavoz" in text
    assert "Impedancia nominal" in text


def test_plot_each_quantity_returns_single_axes() -> None:
    """Every quantity plots one concept on one axes; directivity is polar."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = _example_result()
    # The default is the on-axis response.
    assert result.plot().get_title() == "On-axis response"
    plt.close("all")
    for quantity in ("response", "impedance", "thd", "directivity"):
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
    out = _example_result().plot(quantity="impedance", ax=ax)
    assert out is ax
    assert ax.get_title() == "Impedance"
    plt.close("all")


def test_plot_rejects_unknown_quantity_and_missing_data() -> None:
    """An unknown quantity, and a quantity with no data, raise ValueError."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    result = _example_result()
    with pytest.raises(ValueError, match="unknown quantity"):
        result.plot(quantity="bogus")
    f, spl = _flat_response()
    bare = loudspeaker_characteristics(f, spl, _R, sensitivity_band=(200.0, 4000.0))
    with pytest.raises(ValueError, match="no impedance"):
        bare.plot(quantity="impedance")


def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ValueError."""
    result = _example_result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        result.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path) -> None:
    """An unknown fiche language raises ValueError."""
    result = _example_result()
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match="language"):
        result.report(out, language="xx")
