#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ISO 10534-2 impedance-tube test report (``.report()`` -> PDF).

The report is a rendering feature, so these tests assert only structural
facts: a valid single-page PDF is written for a two-microphone measurement,
the displayed absorption and normalised-impedance values match the closed-form
ISO 10534-2 oracle, the verbose reflection-magnitude column stays on one page,
unknown engines/languages are rejected, and XML specials in metadata do not
break reportlab. Pixel or layout content is never inspected.

The clean-room oracle is a locally-reacting resistive screen (normalised flow
resistance theta = 1) backed by a rigidly-terminated air cavity of depth
L = c0 / (4 * 1000 Hz): its normalised surface impedance is
z(f) = theta - j*cot(k0*L), so r = (z - 1)/(z + 1) (Eq. (19) inverted) and
alpha = 1 - |r|^2 (Eq. (18)). The transfer function H12 is synthesised from r
via the Annex D field model (Eq. (D.7)) and reduced back through
``two_microphone_impedance`` (Eq. (17)). At the 1000 Hz quarter-wave resonance
z = 1, r = 0 and alpha = 1.00; at 500 Hz cot(pi/4) = 1, z = 1 - j and
alpha = 1 - 1/5 = 0.80 with |r| = 1/sqrt(5) = 0.45.
"""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("reportlab")

import numpy as np
import result_factories
from report_assertions import assert_one_page

from phonometry import ReportMetadata
from phonometry.materials import (
    air_density_iso,
    characteristic_impedance,
    speed_of_sound_iso,
    tube_wavenumber,
    two_microphone_impedance,
)

if TYPE_CHECKING:
    from pathlib import Path

    from phonometry.materials import ImpedanceTubeResult

_TEMPERATURE_K = 293.15  # 20 degC
_DIAMETER, _SPACING, _X1 = 0.100, 0.050, 0.100
_FREQS = np.array([400, 500, 630, 800, 1000, 1250, 1600], dtype=float)


def _result() -> ImpedanceTubeResult:
    c0 = float(speed_of_sound_iso(_TEMPERATURE_K))
    rho = float(air_density_iso(_TEMPERATURE_K, 101.0))
    rc = characteristic_impedance(rho, c0)
    cavity = c0 / (4.0 * 1000.0)
    k0 = 2.0 * np.pi * _FREQS / c0
    z = 1.0 - 1j / np.tan(k0 * cavity)
    r = (z - 1.0) / (z + 1.0)
    kk = np.asarray(tube_wavenumber(_FREQS, c0))
    x2 = _X1 - _SPACING
    h12 = (np.exp(1j * kk * x2) + r * np.exp(-1j * kk * x2)) / (
        np.exp(1j * kk * _X1) + r * np.exp(-1j * kk * _X1)
    )
    return two_microphone_impedance(
        h12,
        frequency=_FREQS,
        spacing=_SPACING,
        x1=_X1,
        speed_of_sound=c0,
        characteristic_impedance=rc,
        diameter=_DIAMETER,
        shape="circular",
    )


def _metadata(**overrides: object) -> ReportMetadata:
    base = {
        "specimen": "Resistive facing over an 86 mm rigidly-backed air cavity",
        "client": "Acoustic Test Client Ltd.",
        "manufacturer": "Acoustics Works Inc.",
        "tube_diameter": _DIAMETER,
        "mic_spacing": _SPACING,
        "mounting": "Deliberate 86 mm backing air cavity",
        "test_room": "Impedance tube R1",
        "measurement_standard": "ISO 10534-2",
        "temperature": 20.0,
        "pressure": 101.0,
        "test_date": "2026-07-21",
        "laboratory": "Phonometry Reference Laboratory",
        "operator": "Jose Manuel Requena Plens",
        "report_id": "PHN-2026-10534",
    }
    base.update(overrides)
    return ReportMetadata(**base)


def _text(path: str) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages).replace(
        "\n", " "
    )


def test_report_writes_one_page_pdf(tmp_path: Path) -> None:
    out = tmp_path / "iso10534.pdf"
    returned = _result().report(str(out))
    assert returned == str(out)
    assert_one_page(str(out))


def test_report_with_metadata_one_page(tmp_path: Path) -> None:
    out = tmp_path / "iso10534_meta.pdf"
    _result().report(str(out), metadata=_metadata())
    assert_one_page(str(out))


def test_unknown_engine_rejected(tmp_path: Path) -> None:
    result = _result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        result.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path: Path) -> None:
    result = _result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="Unknown language"):
        result.report(out, language="xx")


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize(
    "field", ["frequency", "absorption", "reflection", "normalized_impedance"]
)
def test_short_per_frequency_array_rejected(
    field: str, verbose: bool, tmp_path: Path
) -> None:
    """One short per-frequency array raises before the fiche is written.

    ``ImpedanceTubeResult`` is a frozen dataclass with no validation, so a
    caller can hand the renderer a result whose arrays disagree in length; the
    value table would then be silently cut short, in the plain four-column
    branch and in the verbose ``|r|`` branch alike. The renderer names every
    per-frequency field beside the shape it was given, and leaves no PDF
    behind.
    """
    result = _result()
    ragged = dataclasses.replace(result, **{field: getattr(result, field)[:-1]})
    out = tmp_path / "iso10534_ragged.pdf"
    metadata = _metadata()
    with pytest.raises(ValueError, match=rf"'{field}'.*same shape"):
        ragged.report(str(out), metadata=metadata, verbose=verbose)
    assert not out.exists()


def test_displayed_absorption_matches_oracle(tmp_path: Path) -> None:
    """The fiche prints the closed-form alpha, band labels and geometry."""
    out = tmp_path / "iso10534.pdf"
    _result().report(str(out), metadata=_metadata())
    text = _text(str(out))
    assert "0.80" in text  # alpha(500 Hz) = 1 - 1/5
    assert "1.00" in text  # alpha(1000 Hz) = 1 (quarter-wave resonance)
    assert "500" in text  # band labels
    assert "1000" in text
    assert "400 to 1600" in text  # measured frequency range
    # Anchor the geometry to its label so a bare "100"/"50" cannot match a band
    # centre (1000) or another frequency (500): the value follows its label.
    assert re.search(r"Tube diameter d \[mm\]:\s*100\b", text)  # d = 100 mm
    assert re.search(r"Microphone spacing s\s*\[mm\]:\s*50\b", text)  # s = 50 mm


def test_verbose_shows_reflection_column_one_page(tmp_path: Path) -> None:
    """verbose=True adds the |r| column and stays one page."""
    out = tmp_path / "iso10534_verbose.pdf"
    _result().report(str(out), metadata=_metadata(), verbose=True)
    assert_one_page(str(out))
    text = _text(str(out))
    assert "0.45" in text  # |r|(500 Hz) = 1/sqrt(5)


def test_metadata_xml_specials_do_not_break(tmp_path: Path) -> None:
    out = tmp_path / "iso10534_xml.pdf"
    _result().report(str(out), metadata=_metadata(specimen='Panel <A> & <B> "edge"'))
    assert_one_page(str(out))


def test_no_metadata_still_renders(tmp_path: Path) -> None:
    """Without metadata the header still shows the measured frequency range."""
    out = tmp_path / "iso10534_bare.pdf"
    _result().report(str(out))
    assert_one_page(str(out))
    assert "400 to 1600" in _text(str(out))


def test_spanish_fiche_uses_comma_decimal(tmp_path: Path) -> None:
    out = tmp_path / "iso10534_es.pdf"
    _result().report(str(out), metadata=_metadata(), language="es")
    assert_one_page(str(out))
    assert "0,80" in _text(str(out))  # Spanish decimal comma


@pytest.mark.parametrize("field", ["tube_diameter", "mic_spacing"])
@pytest.mark.parametrize(
    "value", [-0.05, 0.0, float("nan"), float("inf"), -float("inf")]
)
def test_metadata_rejects_invalid_geometry(field: str, value: float) -> None:
    """The impedance-tube geometry fields must be finite and strictly positive.

    A negative, zero or non-finite (NaN / +-inf) ``tube_diameter`` or
    ``mic_spacing`` violates the finite-positive contract and raises.
    """
    with pytest.raises(ValueError, match=field):
        ReportMetadata(**{field: value})


# --------------------------------------------------------------------------
# A table longer than the single page a fiche is
# --------------------------------------------------------------------------
@pytest.mark.parametrize("points", [38, 60], ids=["runs-on", "layout-error"])
def test_a_table_longer_than_the_page_is_refused(tmp_path: Path, points: int) -> None:
    """A fiche is one page, and nothing used to check that it came to one.

    Up to about forty rows the sheet quietly ran on, leaving page 1 with the
    title over a blank half-page and the table, the result and the laboratory
    identity on pages nobody stamped. Past that reportlab gave up with a
    LayoutError naming a flowable and a frame in points.
    """
    base = result_factories._impedance_tube()
    size = np.size(base.frequency)
    trimmed = dataclasses.replace(
        base,
        **{
            field.name: np.asarray(getattr(base, field.name))[:points]
            for field in dataclasses.fields(base)
            if isinstance(getattr(base, field.name), np.ndarray)
            and np.ndim(getattr(base, field.name)) == 1
            and np.size(getattr(base, field.name)) == size
        },
    )
    out = tmp_path / "too_long.pdf"
    with pytest.raises(ValueError, match="does not fit the single page"):
        trimmed.report(str(out))
    assert not out.exists()  # a fiche that does not fit is not left behind
