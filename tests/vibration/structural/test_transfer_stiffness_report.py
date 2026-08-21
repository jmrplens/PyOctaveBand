#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ISO 10846 dynamic-transfer-stiffness ``.report()`` fiche.

The rendered values are checked against the module's closed-form oracle: a
viscously damped resilient element (Kelvin-Voigt, k = 1 MN/m, c = 80 N.s/m) has
a transfer stiffness k2,1(f) = k + j*omega*c. The direct method (ISO 10846-2)
measures k2,1 = F2,b/u1; synthesising F2,b = k2,1 * u1 and feeding it back
through ``transfer_stiffness_direct`` recovers the closed form, so the printed
low-frequency plateau (|k2,1| = 1.00 MN/m, L_k = 20 lg(|k2,1|/k0) = 120.0 dB re
1 N/m, loss factor eta = Im/Re = 0.010) is a library-independent oracle. The
indirect (blocking-mass) method (ISO 10846-3) is exercised for its method label
and blocking mass. Values are read back from the PDF via pypdf text extraction;
structural facts (one page, rejected engines/languages) complete the rendering
contract.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from report_assertions import assert_one_page

from phonometry import PhonometryWarning, ReportMetadata, vibration

_FREQS = np.array(
    [
        20,
        25,
        31.5,
        40,
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
    ],
    dtype=float,
)
_K, _C = 1.0e6, 80.0


def _extract_text(path: str) -> str:
    """Whitespace-normalized page text (PDF line wraps fold to single spaces)."""
    from pypdf import PdfReader

    raw = "\n".join(page.extract_text() for page in PdfReader(path).pages)
    return " ".join(raw.split())


def _direct_result() -> vibration.TransferStiffnessResult:
    """A Kelvin-Voigt element characterised by the direct method (ISO 10846-2)."""
    omega = 2.0 * np.pi * _FREQS
    k21 = _K + 1j * omega * _C
    u1 = 1.0e-6 + 0.0j
    measured = vibration.transfer_stiffness_direct(k21 * u1, u1)
    return vibration.TransferStiffnessResult(
        frequencies=_FREQS, transfer_stiffness=measured, blocking_mass=None
    )


def test_low_frequency_plateau_oracle() -> None:
    """At the 20 Hz plateau |k2,1| = k and L_k = 120 dB (module oracle)."""
    res = _direct_result()
    assert float(res.magnitude[0]) == pytest.approx(_K, rel=1e-4)
    assert float(res.level[0]) == pytest.approx(
        float(vibration.transfer_stiffness_level(_K)), abs=1e-3
    )
    assert float(vibration.transfer_stiffness_level(_K)) == pytest.approx(
        120.0, abs=1e-3
    )


def test_report_renders_plateau_and_basis(tmp_path) -> None:
    """The fiche prints the low-frequency plateau, the level and ISO 10846."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _direct_result()
    out = tmp_path / "transfer.pdf"
    returned = res.report(str(out))
    assert returned == str(out)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Dynamic transfer stiffness of a resilient element" in text
    assert "ISO 10846-1:2008" in text
    assert "direct method (ISO 10846-2:2008)" in text
    # Low-frequency plateau: |k2,1| = 1.00 MN/m, L_k = 120.0 dB re 1 N/m.
    assert "1.00" in text
    assert "120.0" in text
    # Loss factor eta = omega*c/k at 20 Hz = 0.010.
    assert "0.010" in text


def test_indirect_method_labels_blocking_mass(tmp_path) -> None:
    """The indirect method names ISO 10846-3 and prints the blocking mass."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    freqs = np.array(
        [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000], dtype=float
    )
    # A blocking mass well above the mass/spring resonance keeps |T| <= 0.1, so
    # the ISO 10846-3 validity advisory does not fire.
    transmissibility = vibration.base_transmissibility(freqs, 50.0, _K, 40.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PhonometryWarning)
        res = vibration.indirect_transfer_stiffness_result(
            freqs, transmissibility, blocking_mass=50.0
        )
    out = tmp_path / "indirect.pdf"
    res.report(str(out))
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "indirect blocking-mass method (ISO 10846-3:2002)" in text
    assert "Blocking mass" in text
    assert "50.0" in text


def test_metadata_header_renders(tmp_path) -> None:
    """Supplied metadata renders the client, the specimen and the instrumentation."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _direct_result()
    metadata = ReportMetadata(
        specimen="Rubber vibration isolator",
        client="Example elastomers client",
        test_room="Transfer-stiffness rig",
        instrumentation="Force transducer + accelerometers",
        laboratory="Vibration laboratory",
        report_id="TS-10846",
    )
    out = tmp_path / "meta.pdf"
    res.report(str(out), metadata=metadata)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Example elastomers client" in text
    assert "Rubber vibration isolator" in text
    assert "Force transducer + accelerometers" in text
    assert "TS-10846" in text


def test_spanish_report_renders_translated_fiche(tmp_path) -> None:
    """language="es" renders the transfer-stiffness vocabulary and comma decimals."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _direct_result()
    out = tmp_path / "es.pdf"
    res.report(str(out), language="es")
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Rigidez dinámica de transferencia de un elemento resiliente" in text
    assert "método directo" in text
    assert "baja frecuencia" in text


def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ValueError."""
    res = _direct_result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        res.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path) -> None:
    """An unknown fiche language raises ValueError."""
    res = _direct_result()
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match="language"):
        res.report(out, language="xx")
