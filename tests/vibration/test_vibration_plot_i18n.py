#  Copyright (c) 2026. Jose M. Requena-Plens
"""EN/ES language option of the vibration ``.plot()`` renderers."""

from __future__ import annotations

import numpy as np
import pytest

from phonometry import rigid_mass_calibration_check, sdof_mobility_result


def _result():
    return sdof_mobility_result(np.linspace(1.0, 50.0, 200), 2.0, 8000.0, 5.0)


def test_spanish_labels() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    res = _result()

    ax_en = res.plot(language="en")
    assert ax_en.get_xlabel() == "Frequency [Hz]"
    assert ax_en.get_title() == "ISO 7626-1 mechanical mobility"

    ax_es = res.plot(language="es")
    assert ax_es.get_xlabel() == "Frecuencia [Hz]"
    assert ax_es.get_title() == "ISO 7626-1 movilidad mecánica"


def test_rigid_mass_spanish_labels() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    f = np.array([20.0, 100.0, 500.0])
    res = rigid_mass_calibration_check([0.100, 0.102, 0.097], f, mass=10.0)

    axes_en = res.plot(language="en")
    assert axes_en[0].get_title() == "ISO 7626-2 rigid-mass calibration check (PASS)"
    assert axes_en[1].get_ylabel() == "Deviation [%]"

    axes_es = res.plot(language="es")
    assert axes_es[0].get_title() == (
        "ISO 7626-2 verificación de calibración con masa rígida (CORRECTO)"
    )
    assert axes_es[1].get_ylabel() == "Desviación [%]"
    assert axes_es[1].get_xlabel() == "Frecuencia [Hz]"


def test_multiple_shock_spanish_title_translates_sex() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from phonometry.vibration.multiple_shock_vibration import (
        MZ_MALE,
        RISK_THRESHOLDS_MALE,
        MultipleShockResult,
        compression_dose,
        dose_from_peaks,
        injury_probability,
        injury_risk,
    )

    peaks = np.full(5, 40.0)
    dz = dose_from_peaks(peaks)
    sd = compression_dose(dz, mz=MZ_MALE)
    r = injury_risk(sd, start_age=20.0, years=20, days_per_year=120.0, sex="male")
    res = MultipleShockResult(
        sex="male",
        acceleration_dose=dz,
        daily_dose=dz,
        compression_dose=sd,
        risk=r,
        probability=float(injury_probability(r, sex="male")),
        start_age=20.0,
        years=20,
        days_per_year=120.0,
        peaks=peaks,
        risk_thresholds=RISK_THRESHOLDS_MALE,
    )
    ax_en = res.plot(language="en")
    assert ax_en.get_title() == "ISO 2631-5 injury probability — male"
    ax_es = res.plot(language="es")
    assert ax_es.get_title() == "ISO 2631-5 probabilidad de lesión — hombre"


def test_unknown_language_raises() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    result = _result()
    with pytest.raises(ValueError, match="Unknown language"):
        result.plot(language="xx")


def test_fault_frequency_overlay_labels() -> None:
    """The fault-line overlay localises its axes, title and family legend."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from phonometry import bearing_fault_frequencies

    res = bearing_fault_frequencies(2000.0, 15, 6.0, 34.0,
                                    contact_angle_deg=12.96)

    ax_en = res.plot()
    assert ax_en.get_xlabel() == "Frequency [Hz]"
    assert "rolling-contact bearing" in ax_en.get_title()
    assert ax_en.get_ylabel() == "Predicted fault line"
    labels_en = [t.get_text() for t in ax_en.get_legend().get_texts()]
    assert {"shaft", "bearing"} <= set(labels_en)

    ax_es = res.plot(language="es")
    assert ax_es.get_xlabel() == "Frecuencia [Hz]"
    assert "rodamiento de contacto rodante" in ax_es.get_title()
    labels_es = [t.get_text() for t in ax_es.get_legend().get_texts()]
    assert {"eje", "rodamiento"} <= set(labels_es)

    with pytest.raises(ValueError, match="Unknown language"):
        res.plot(language="xx")


def test_fault_frequency_overlay_on_a_measured_spectrum() -> None:
    """With a spectrum the curve is drawn underneath and named."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from phonometry import bearing_fault_frequencies, envelope_spectrum

    res = bearing_fault_frequencies(2000.0, 15, 6.0, 34.0,
                                    contact_angle_deg=12.96)
    fs = 8192.0
    t = np.arange(int(fs)) / fs
    signal = (1.0 + 0.5 * np.cos(2.0 * np.pi * res["BPFO"] * t)) * np.cos(
        2.0 * np.pi * 2000.0 * t)
    spectrum = envelope_spectrum(signal, fs)

    ax = res.plot(spectrum=spectrum, max_frequency=600.0)
    assert ax.get_ylabel() == "Envelope amplitude"
    assert ax.get_xlim() == (0.0, 600.0)
    labels = [t_.get_text() for t_ in ax.get_legend().get_texts()]
    assert "envelope spectrum" in labels

    empty = res.within(1.0e6, 2.0e6)
    with pytest.raises(ValueError, match="no fault lines"):
        empty.plot()


def test_crowded_fault_labels_do_not_overlap() -> None:
    """Names of nearby lines are pushed apart instead of stacking up.

    On a wide axis the low-frequency bearing lines (FTF 13,8 Hz, FTF_rel
    19,5 Hz and the 33,3 Hz shaft) fall within a few points of each other and
    their rotated labels used to be drawn on top of one another.
    """
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from phonometry import bearing_fault_frequencies
    from phonometry._plot.vibration import _LABEL_WIDTH_PT, _label_offsets

    res = bearing_fault_frequencies(2000.0, 15, 6.0, 34.0,
                                    contact_angle_deg=12.96)
    ax = res.plot(max_frequency=1200.0)
    assert len(ax.texts) == len(res.lines)

    # Every label must clear the previous one across the axis.
    width_pt, f_max = 450.0, 1200.0
    freqs = [line.frequency for line in res.lines]
    offsets = _label_offsets(freqs, f_max, width_pt)
    placed = sorted(
        f * width_pt / f_max + offsets[i] for i, f in enumerate(freqs)
    )
    assert min(np.diff(placed)) >= _LABEL_WIDTH_PT - 1e-9
    # An isolated line keeps its label where it was: BPFI has no near neighbour.
    assert offsets[freqs.index(res["BPFI"])] == pytest.approx(2.0)


def test_power_injection_labels() -> None:
    """The SEA loss-factor budget localises its axes and title."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from phonometry import power_injection_clf

    f = np.array([250.0, 500.0, 1000.0])
    res = power_injection_clf(f, 0.087, 0.013, 4.4e-3, 2.4e-3, 0.557, 0.606)

    ax_en = res.plot()
    assert ax_en.get_xlabel() == "Frequency [Hz]"
    assert ax_en.get_ylabel() == "Loss factor"
    assert "single-drive" in ax_en.get_title()

    ax_es = res.plot(language="es")
    assert ax_es.get_xlabel() == "Frecuencia [Hz]"
    assert ax_es.get_ylabel() == "Factor de pérdidas"
    assert "excitación única" in ax_es.get_title()

    with pytest.raises(ValueError, match="Unknown language"):
        res.plot(language="xx")
