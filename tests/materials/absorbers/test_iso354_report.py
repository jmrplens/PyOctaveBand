#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ISO 354 sound-absorption test report (``.report()`` -> PDF).

The report is a rendering feature, so these tests assert only structural
facts: a valid single-page PDF is written for a reverberation-room measurement,
the displayed one-third-octave values match the closed-form ISO 354 oracle,
the verbose detail table stays on one page, a hand-built result whose per-band
columns disagree in length is rejected before anything is written, a band whose
reverberation time could not be evaluated prints the em dash rather than a
literal ``nan``, unknown engines are rejected, and XML specials in metadata do
not break reportlab. Pixel or layout content is never inspected.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("reportlab")

import numpy as np
from report_assertions import assert_one_page

from phonometry import ReportMetadata
from phonometry.materials import measure_sound_absorption

if TYPE_CHECKING:
    from pathlib import Path

    from phonometry.materials import SoundAbsorptionMeasurement

# The committed clean-room example (V = 200 m3, S = 10.8 m2, 20 degC -> c = 343,
# m = 0); alpha_s(500 Hz) = 0.33 and alpha_s(1000 Hz) = 0.61 by Eq. (8)/(9).
_FREQS = np.array(
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
        4000,
        5000,
    ],
    dtype=float,
)
_T1 = np.array(
    [
        9.0,
        9.0,
        8.8,
        8.6,
        8.4,
        8.2,
        8.0,
        7.8,
        7.5,
        7.2,
        6.9,
        6.6,
        6.2,
        5.8,
        5.4,
        5.0,
        4.6,
        4.2,
    ]
)
_T2 = np.array(
    [
        8.4,
        8.2,
        7.7,
        7.2,
        6.5,
        5.7,
        4.9,
        4.2,
        3.6,
        3.15,
        2.85,
        2.65,
        2.55,
        2.5,
        2.55,
        2.6,
        2.7,
        2.85,
    ]
)


def _result() -> SoundAbsorptionMeasurement:
    return measure_sound_absorption(
        _FREQS,
        _T1,
        _T2,
        volume=200.0,
        area=10.8,
        temperature=20.0,
        humidity=54.0,
    )


def _metadata(**overrides: object) -> ReportMetadata:
    base = {
        "specimen": "50 mm porous absorber over a 100 mm air gap",
        "client": "Acoustic Test Client Ltd.",
        "manufacturer": "Acoustics Works Inc.",
        "mounting": "Type A (against a rigid wall)",
        "test_room": "Reverberation room R1",
        "measurement_standard": "ISO 354",
        "pressure": 101.0,
        "test_date": "2026-07-21",
        "laboratory": "Phonometry Reference Laboratory",
        "operator": "Jose Manuel Requena Plens",
        "report_id": "PHN-2026-354",
    }
    base.update(overrides)
    return ReportMetadata(**base)


def _text(path: str) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages).replace(
        "\n", " "
    )


def test_report_writes_one_page_pdf(tmp_path: Path) -> None:
    out = tmp_path / "iso354.pdf"
    returned = _result().report(str(out))
    assert returned == str(out)
    assert_one_page(str(out))


def test_report_with_metadata_one_page(tmp_path: Path) -> None:
    out = tmp_path / "iso354_meta.pdf"
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


def test_displayed_alpha_s_matches_oracle(tmp_path: Path) -> None:
    """The fiche prints the closed-form alpha_s and the band labels."""
    out = tmp_path / "iso354.pdf"
    _result().report(str(out), metadata=_metadata())
    text = _text(str(out))
    assert "0.33" in text  # alpha_s(500 Hz)
    assert "0.61" in text  # alpha_s(1000 Hz)
    assert "500" in text  # band labels
    assert "1000" in text
    assert "343" in text  # speed of sound c (Eq. (6))


def test_verbose_shows_areas_and_times_one_page(tmp_path: Path) -> None:
    """verbose=True adds the T1/T2/A1/A2 columns and stays one page."""
    out = tmp_path / "iso354_verbose.pdf"
    _result().report(str(out), metadata=_metadata(), verbose=True)
    assert_one_page(str(out))
    text = _text(str(out))
    assert "7.80" in text  # T1(500 Hz)
    assert "7.7" in text  # A2(500 Hz) = 7.677 m2 rounded to 0.1


@pytest.mark.parametrize(
    "field",
    [
        "t_empty",
        "t_specimen",
        "absorption_area_empty",
        "absorption_area_with_specimen",
    ],
)
def test_a_short_band_column_is_refused(field: str) -> None:
    """Every per-band column must be as long as ``frequencies``.

    A column one band short would leave the detail table silently truncated.
    The refusal used to be the renderer's, and only on the verbose sheet; the
    measurement refuses to exist that way now, which covers the plain sheet
    and the plot with it.
    """
    result = _result()
    one_short = getattr(result, field)[:-1]
    with pytest.raises(ValueError, match=f"'{field}'"):
        replace(result, **{field: one_short})


def test_metadata_xml_specials_do_not_break(tmp_path: Path) -> None:
    out = tmp_path / "iso354_xml.pdf"
    _result().report(str(out), metadata=_metadata(specimen='Panel <A> & <B> "edge"'))
    assert_one_page(str(out))


def test_no_metadata_still_renders(tmp_path: Path) -> None:
    """Without metadata the body still shows the result's physical conditions."""
    out = tmp_path / "iso354_bare.pdf"
    _result().report(str(out))
    assert_one_page(str(out))
    assert "343" in _text(str(out))  # speed of sound from the result


def test_spanish_fiche_uses_comma_decimal(tmp_path: Path) -> None:
    out = tmp_path / "iso354_es.pdf"
    _result().report(str(out), metadata=_metadata(), language="es")
    assert_one_page(str(out))
    assert "0,33" in _text(str(out))  # Spanish decimal comma


def test_non_evaluable_band_prints_the_em_dash(tmp_path: Path) -> None:
    """A band whose decay could not be evaluated is em-dashed, never "nan".

    ``measure_sound_absorption`` accepts a non-finite reverberation time by
    design: that is how a band flagged non-evaluable by an ISO 3382-2 decay
    analysis arrives, and refusing it here would refuse the library's own
    output. It propagates through the Sabine inversion to ``A2`` and
    ``alpha_s``, and the accredited table prints the house em dash for it, as
    the sibling fiches do. The verbose sheet em-dashes all three cells of that
    band's row (T2, A2 and alpha_s).
    """
    t2 = _T2.copy()
    t2[5] = np.nan  # 315 Hz band
    result = measure_sound_absorption(
        _FREQS, _T1, t2, volume=200.0, area=10.8, temperature=20.0
    )
    assert not np.isfinite(result.alpha_s[5])
    plain = tmp_path / "iso354_nan.pdf"
    verbose = tmp_path / "iso354_nan_verbose.pdf"
    result.report(str(plain))
    result.report(str(verbose), verbose=True)
    plain_text = _text(str(plain))
    verbose_text = _text(str(verbose))
    assert "nan" not in plain_text
    assert "nan" not in verbose_text
    assert plain_text.count("—") == 1  # the alpha_s cell of the 315 Hz band
    assert verbose_text.count("—") == 3  # T2, A2 and alpha_s of that band
