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
