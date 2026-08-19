#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for the ISO 3382-1/-2 room acoustic parameters report (``.report()``).

The report is a rendering feature, so these tests assert only structural
facts: a valid single-page PDF is written for a room-acoustics result, the
per-band table and the mid-frequency descriptor carry the closed-form values
of a documented synthetic decay, unknown engines/languages are rejected, XML
specials in metadata do not break reportlab, the optional target-RT verdict
renders both ways, and the Spanish fiche is translated with comma decimals.
Pixel or layout content is never inspected.

Oracle. The impulse response is a deterministic single-slope decay: one sine
carrier per octave band, each modulated by its own exponential energy envelope
exp(-A60*t/T) with A60 = 6*ln(10). For a pure exponential energy decay the
Schroeder curve is an exact straight line L(t) = -60*t/T dB (ISO 3382-1:2009,
5.3.3), so the closed-form reverberation time is T20 = T30 = EDT = T per band.
The chosen per-band T therefore appear verbatim in the reverberation-time
columns and fix the mid-frequency descriptor T_mid = (T30@500 + T30@1000)/2.
"""

from __future__ import annotations

import pytest

pytest.importorskip("reportlab")
pytest.importorskip("svglib")
pytest.importorskip("pypdf")

import numpy as np

from phonometry import ReportMetadata, room

_PDF_MAGIC = b"%PDF"
_FS = 48000
#: 6*ln(10): energy-decay-rate constant so exp(-A60*t/T) falls 60 dB in T s.
_A60 = 6.0 * np.log(10.0)

#: Octave bands and their per-band reverberation times (a small-hall profile
#: falling with frequency); T_mid = (1.20 + 1.10)/2 = 1.15 s.
_BANDS = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)
_T60 = (1.40, 1.30, 1.20, 1.10, 1.00, 0.85)
_T_MID = 0.5 * (_T60[2] + _T60[3])


def _synthetic_ir(seconds: float = 5.0) -> np.ndarray:
    """One sine carrier per octave band, each with its own exponential decay."""
    time = np.arange(round(seconds * _FS)) / _FS
    ir = np.zeros_like(time)
    for freq, decay in zip(_BANDS, _T60):
        ir += np.sin(2.0 * np.pi * freq * time) * np.exp(-0.5 * _A60 * time / decay)
    return ir


def _result() -> room.RoomAcousticsResult:
    return room.room_parameters(_synthetic_ir(), _FS)


def _assert_one_page(path: str) -> None:
    """The fiche is a single-page PDF beginning with ``%PDF``."""
    import os

    with open(path, "rb") as handle:
        assert handle.read(4) == _PDF_MAGIC
    assert os.path.getsize(path) > 0
    from pypdf import PdfReader

    assert len(PdfReader(path).pages) == 1


def _extract_text(path: str) -> str:
    """The concatenated, single-line text of every page."""
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages).replace(
        "\n", " "
    )


def _full_metadata(**overrides) -> ReportMetadata:
    base = {
        "specimen": "Small auditorium, unoccupied, fully furnished",
        "client": "Acoustic Test Client Ltd.",
        "test_room": "Auditorium A",
        "room_volume": 2830.0,
        "area": 340.0,
        "source_positions": 2,
        "receiver_positions": 8,
        "instrumentation": "Omnidirectional source + 1/2 in. microphone",
        "measurement_standard": "ISO 3382-1",
        "temperature": 21.0,
        "relative_humidity": 45.0,
        "pressure": 101.1,
        "test_date": "2026-07-20",
        "laboratory": "Phonometry Reference Laboratory",
        "operator": "Jose Manuel Requena Plens",
        "report_id": "PHN-2026-3382",
    }
    base.update(overrides)
    return ReportMetadata(**base)


# --------------------------------------------------------------------------
# Oracle: the synthetic decay reproduces its closed-form reverberation times.
# --------------------------------------------------------------------------
def test_synthetic_decay_matches_closed_form() -> None:
    """T20 = T30 = EDT = T per band (within +/-1 %, far below the 5 % JND)."""
    res = _result()
    assert res.frequency is not None
    assert len(res.frequency) == len(_BANDS)
    np.testing.assert_allclose(res.t20, _T60, rtol=0.01)
    np.testing.assert_allclose(res.t30, _T60, rtol=0.01)
    np.testing.assert_allclose(res.edt, _T60, rtol=0.01)
    assert bool(np.all(res.t30_valid))


# --------------------------------------------------------------------------
# Structural rendering
# --------------------------------------------------------------------------
def test_report_writes_pdf(tmp_path) -> None:
    """A room-acoustics result renders a PDF fiche and returns its path."""
    out = tmp_path / "room.pdf"
    returned = _result().report(str(out))
    assert returned == str(out)
    _assert_one_page(str(out))


def test_report_with_full_metadata_one_page(tmp_path) -> None:
    """A full ReportMetadata renders a one-page accredited room fiche."""
    out = tmp_path / "room_meta.pdf"
    _result().report(str(out), metadata=_full_metadata())
    _assert_one_page(str(out))


def test_report_bare_without_metadata(tmp_path) -> None:
    """metadata=None renders a bare characterisation fiche (no header)."""
    out = tmp_path / "bare.pdf"
    _result().report(str(out), metadata=None)
    _assert_one_page(str(out))


def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ``ValueError``."""
    res = _result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        res.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path) -> None:
    """An unknown fiche language raises ``ValueError``."""
    res = _result()
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match="language"):
        res.report(out, language="xx")


# --------------------------------------------------------------------------
# Displayed values (the oracle appears in the rendered text)
# --------------------------------------------------------------------------
def test_band_labels_and_reverberation_times_render(tmp_path) -> None:
    """Nominal band labels and the closed-form T30/EDT appear in the PDF."""
    out = tmp_path / "values.pdf"
    _result().report(str(out), metadata=_full_metadata())
    text = _extract_text(str(out))
    for label in ("125", "250", "500", "1000", "2000", "4000"):
        assert label in text
    # Reverberation-time columns carry the closed-form values (2 decimals).
    assert "1.40" in text  # 125 Hz T20/T30/EDT
    assert "1.20" in text  # 500 Hz
    assert "0.85" in text  # 4000 Hz
    # Column headers for the energy parameters are present.
    assert "EDT" in text
    assert "C80" in text


def test_mid_frequency_descriptor_and_verdict(tmp_path) -> None:
    """The boxed T_mid = 1.15 s and a PASS/FAIL verdict both render."""
    passing = tmp_path / "pass.pdf"
    failing = tmp_path / "fail.pdf"
    _result().report(str(passing), metadata=_full_metadata(requirement=1.30))
    _result().report(str(failing), metadata=_full_metadata(requirement=1.00))
    _assert_one_page(str(passing))
    _assert_one_page(str(failing))
    assert abs(_T_MID - 1.15) < 1e-9
    pass_text = _extract_text(str(passing))
    assert "1.15" in pass_text
    assert "PASS" in pass_text
    assert "FAIL" in _extract_text(str(failing))


def test_report_without_requirement_has_no_verdict(tmp_path) -> None:
    """With no target RT the characterisation fiche omits the verdict row."""
    out = tmp_path / "noverdict.pdf"
    _result().report(str(out), metadata=_full_metadata())
    text = _extract_text(str(out))
    assert "PASS" not in text
    assert "FAIL" not in text


def _synthetic_result(
    frequency: np.ndarray | None,
    t30: np.ndarray,
    edt: np.ndarray,
) -> room.RoomAcousticsResult:
    """A hand-built result: T30/EDT as given, other parameters mirrored."""
    t30 = np.asarray(t30, dtype=np.float64)
    edt = np.asarray(edt, dtype=np.float64)
    n = t30.size
    finite_t30 = np.isfinite(t30)
    finite_edt = np.isfinite(edt)
    return room.RoomAcousticsResult(
        frequency=frequency,
        edt=edt,
        t20=t30.copy(),
        t30=t30,
        c50=np.zeros(n),
        c80=np.zeros(n),
        d50=np.full(n, 0.5),
        ts=np.full(n, 0.1),
        dynamic_range=np.full(n, 60.0),
        edt_valid=finite_edt,
        t20_valid=finite_t30,
        t30_valid=finite_t30,
        curvature=np.zeros(n),
    )


# --------------------------------------------------------------------------
# Band-set variants
# --------------------------------------------------------------------------
def test_third_octave_report_renders(tmp_path) -> None:
    """A one-third-octave analysis renders and labels T_mid honestly.

    With one-third-octave data only the 500 Hz and 1 kHz one-third-octave
    bands are averaged (not the two full octaves), so the boxed descriptor
    must say so instead of the octave "500-1000 Hz" label.
    """
    res = room.room_parameters(
        _synthetic_ir(), _FS, limits=(100.0, 5000.0), fraction=3
    )
    out = tmp_path / "thirds.pdf"
    res.report(str(out), metadata=_full_metadata())
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "500 Hz and 1 kHz one-third-octave bands" in text
    assert "500-1000 Hz" not in text


def test_band_range_without_mid_octaves_names_the_band(tmp_path) -> None:
    """A range missing the mid bands boxes the first finite T30 band by name."""
    res = room.room_parameters(_synthetic_ir(), _FS, limits=(2000.0, 4000.0))
    assert res.frequency is not None
    assert len(res.frequency) == 2
    out = tmp_path / "high_bands.pdf"
    res.report(str(out), metadata=_full_metadata())
    text = _extract_text(str(out))
    assert "(2000 Hz)" in text  # the boxed T30 names its band explicitly
    assert "Tmid" not in text
    assert "T_mid" not in text


def test_edt_label_follows_edts_own_band_coverage(tmp_path) -> None:
    """EDT_mid vs EDT is chosen from EDT's own bands, not T30's.

    Here T30 is finite in both mid octaves (T_mid = 1.15 s) but the 500 Hz
    EDT is not evaluable, so the extended term must be the plain EDT of its
    1000 Hz band, never a false "EDT_mid".
    """
    freq = np.array([500.0, 1000.0])
    res = _synthetic_result(
        freq, t30=np.array([1.2, 1.1]), edt=np.array([np.nan, 1.0])
    )
    out = tmp_path / "edt_divergent.pdf"
    res.report(str(out), metadata=_full_metadata())
    text = _extract_text(str(out))
    assert "1.15" in text  # T_mid from the two finite T30 bands
    assert "EDTmid" not in text
    assert "EDT_mid" not in text
    assert "EDT (1000 Hz) = 1.00 s" in text


def test_verdict_uses_display_rounded_values(tmp_path) -> None:
    """T_mid = 1.1502 s prints as 1.15 s and must PASS a 1.15 s target."""
    freq = np.array([500.0, 1000.0])
    res = _synthetic_result(
        freq, t30=np.array([1.2004, 1.1]), edt=np.array([1.2, 1.1])
    )
    out = tmp_path / "boundary.pdf"
    res.report(str(out), metadata=_full_metadata(requirement=1.15))
    text = _extract_text(str(out))
    assert "1.15" in text
    assert "PASS" in text
    assert "FAIL" not in text


def test_single_band_is_not_labeled_broadband(tmp_path) -> None:
    """A narrow single-band selection is a single band, not broadband.

    Selecting a single octave band leaves one frequency entry (not ``None``),
    so the caption must read "Single-band parameters", never "Broadband".
    """
    res = room.room_parameters(
        _synthetic_ir(), _FS, limits=(490.0, 510.0), fraction=1
    )
    assert res.frequency is not None
    assert len(res.frequency) == 1
    out = tmp_path / "single_band.pdf"
    res.report(str(out))
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Single-band parameters" in text
    assert "Broadband" not in text


def test_octave_report_many_bands_renders(tmp_path) -> None:
    """A wide octave analysis (>6 bands) renders one page as an octave set.

    The one-third-octave triplet grouping is gated on the band structure, so a
    wide octave set (11 bands) is still labelled an octave-band set and grouped
    by octave, not by spurious one-third-octave triplets.
    """
    res = room.room_parameters(
        _synthetic_ir(), _FS, limits=(16.0, 16000.0), fraction=1
    )
    assert res.frequency is not None
    assert len(res.frequency) > 6
    out = tmp_path / "octave_wide.pdf"
    res.report(str(out), metadata=_full_metadata())
    _assert_one_page(str(out))
    assert "Octave-band parameters" in _extract_text(str(out))


def test_broadband_report_renders(tmp_path) -> None:
    """A broadband (single-band) analysis renders a one-page fiche."""
    res = room.room_parameters(_synthetic_ir(), _FS, limits=None)
    assert res.frequency is None
    out = tmp_path / "broadband.pdf"
    res.report(str(out))
    _assert_one_page(str(out))
    assert "Broadband" in _extract_text(str(out))


def test_broadband_makes_no_mid_frequency_claim(tmp_path) -> None:
    """A broadband result must NOT box a false "500-1000 Hz" T_mid.

    With no frequency bands there is no 500/1000 Hz averaging, so the boxed
    descriptor is the plain broadband T30 and the verdict is broadband-aware;
    neither the result box nor the verdict may claim "500-1000 Hz".
    """
    res = room.room_parameters(_synthetic_ir(), _FS, limits=None)
    assert res.frequency is None
    out = tmp_path / "broadband_claim.pdf"
    res.report(str(out), metadata=_full_metadata(requirement=1.30))
    text = _extract_text(str(out))
    assert "500-1000" not in text
    assert "T_mid" not in text
    assert "Tmid" not in text
    # A verdict is still emitted against the broadband reverberation time.
    assert ("PASS" in text) or ("FAIL" in text)


# --------------------------------------------------------------------------
# Robustness and localisation
# --------------------------------------------------------------------------
def test_report_escapes_xml_specials_in_metadata(tmp_path) -> None:
    """Metadata with XML specials (& < >) renders without crashing reportlab."""
    md = ReportMetadata(
        client="Ac & Co <Ltd>",
        specimen="hall <A> & foyer",
        test_room="Room & Stage",
        laboratory="Lab & Sons",
        operator="A <B>",
        report_id="R&D-3382",
        measurement_standard="ISO 3382-1 & Annex",
    )
    out = tmp_path / "xml.pdf"
    _result().report(str(out), metadata=md)
    _assert_one_page(str(out))


def test_spanish_report_renders_translated_fiche(tmp_path) -> None:
    """language="es" renders a one-page Spanish fiche with comma decimals."""
    import re

    out = tmp_path / "room_es.pdf"
    _result().report(str(out), metadata=_full_metadata(requirement=1.30), language="es")
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Parámetros acústicos de salas" in text
    assert "CUMPLE" in text
    assert re.search(r"\d,\d", text) is not None  # comma decimal separator


# --------------------------------------------------------------------------
# ReportMetadata room-specific field validation
# --------------------------------------------------------------------------
def test_room_volume_must_be_positive() -> None:
    """A non-positive room volume raises ``ValueError``."""
    with pytest.raises(ValueError, match="room_volume"):
        ReportMetadata(room_volume=0.0)


@pytest.mark.parametrize("field", ["source_positions", "receiver_positions"])
@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_position_counts_must_be_positive_integers(field, bad) -> None:
    """A position count must be a positive integer (bools and floats rejected)."""
    with pytest.raises(ValueError, match=field):
        ReportMetadata(**{field: bad})


def test_position_counts_accept_positive_integers() -> None:
    """Positive integer position counts are accepted."""
    md = ReportMetadata(source_positions=2, receiver_positions=8, room_volume=2830.0)
    assert md.source_positions == 2
    assert md.receiver_positions == 8
    assert md.room_volume == 2830.0
