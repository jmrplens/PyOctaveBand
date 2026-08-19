#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for the ISO 9613-2 outdoor-propagation prediction reports (``.report()``).

Two prediction fiches share the ISO 9613-2 family renderer: the octave-band
attenuation breakdown (:class:`OutdoorAttenuation`), which boxes the A-weighted
downwind receiver level when a :class:`SourceEmission` is supplied at report
time, and the barrier insertion loss (:class:`BarrierInsertionLoss`). Their
per-band values come from the tested clause-7 / wave-acoustics functions (see
``test_outdoor_propagation.py`` and ``test_ground_barriers.py``), so these tests
stay structural (a one-page ``%PDF``) plus pypdf text-extraction checks of the
boxed single number, a couple of band terms, the mandatory "prediction / not a
measurement" wording and the verdict direction, like the sibling report tests.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("reportlab")

import phonometry as ph
from phonometry import ReportMetadata, environment

_PDF_MAGIC = b"%PDF"
_BANDS = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])


def _attenuation() -> ph.OutdoorAttenuation:
    """A porous-ground attenuation over 200 m (a tested clause-7 geometry)."""
    return ph.outdoor_propagation_attenuation(
        200.0, 2.0, 2.0, _BANDS, 1.0, 1.0, 1.0
    )


def _emission() -> environment.SourceEmission:
    """A flat 100 dB octave-band source power (matches a tested composition)."""
    return environment.SourceEmission(sound_power_level=np.full(8, 100.0))


def _barrier(method: str = "exact") -> ph.BarrierInsertionLoss:
    """A 4 m thin screen, source 1 m at 50 m, receiver 1.5 m at 100 m."""
    return ph.barrier_insertion_loss(
        _BANDS, 1.0, 50.0, 4.0, 100.0, 1.5, method=method
    )


def _assert_one_page(path: str) -> None:
    """A written report is a non-empty single-page PDF."""
    from pypdf import PdfReader

    with open(path, "rb") as handle:
        assert handle.read(4) == _PDF_MAGIC
    assert os.path.getsize(path) > 0
    assert len(PdfReader(path).pages) == 1


def _extract_text(path: str) -> str:
    """The concatenated, whitespace-normalised text of every page."""
    from pypdf import PdfReader

    return " ".join(
        "\n".join(page.extract_text() for page in PdfReader(path).pages).split()
    )


# --------------------------------------------------------------------------- #
# Attenuation prediction fiche
# --------------------------------------------------------------------------- #
def test_attenuation_with_emission_boxes_receiver_level(tmp_path) -> None:
    """With a source emission the fiche boxes the A-weighted downwind level."""
    out = tmp_path / "atten.pdf"
    assert _attenuation().report(str(out), source_emission=_emission()) == str(out)
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "LAT(DW) = 46.5 dB" in text  # boxed A-weighted downwind level
    assert "ISO 9613-2:1996" in text
    assert "prediction" in text
    assert "not a measurement" in text
    # A couple of band terms from the table.
    assert "57.0" in text  # Adiv = 20 lg(200) + 11
    assert "47.2" in text  # LfT(DW) at 63 Hz = 100 - a_total


def test_attenuation_without_emission_boxes_total_range(tmp_path) -> None:
    """Without a source emission the fiche boxes the total-attenuation range."""
    out = tmp_path / "bare.pdf"
    _attenuation().report(str(out))
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Total attenuation A" in text
    assert "52.8" in text  # min total attenuation
    assert "72.3" in text  # max total attenuation
    assert "LAT(DW)" not in text  # no receiver level without a source emission


def test_attenuation_emission_length_mismatch_raises(tmp_path) -> None:
    """A source power that does not match the band count is rejected."""
    result = _attenuation()
    bad = environment.SourceEmission(sound_power_level=np.full(4, 100.0))
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="one value per frequency"):
        result.report(out, source_emission=bad)


def test_attenuation_verbose_adds_a_weighted_band(tmp_path) -> None:
    """``verbose=True`` adds the A-weighted band-level column."""
    result = _attenuation()
    emission = _emission()
    plain = tmp_path / "plain.pdf"
    result.report(str(plain), source_emission=emission)
    assert "dB(A)" not in _extract_text(str(plain))

    verbose = tmp_path / "verbose.pdf"
    result.report(str(verbose), source_emission=emission, verbose=True)
    assert "dB(A)" in _extract_text(str(verbose))  # the L_A column header


def test_attenuation_requirement_lower_is_better(tmp_path) -> None:
    """The downwind level passes at or below the declared limit level."""
    result = _attenuation()  # LAT(DW) = 46.5 dB with the flat 100 dB source
    emission = _emission()
    passing = tmp_path / "pass.pdf"
    result.report(
        str(passing), metadata=ReportMetadata(requirement=50.0),
        source_emission=emission,
    )
    assert "PASS" in _extract_text(str(passing))

    failing = tmp_path / "fail.pdf"
    result.report(
        str(failing), metadata=ReportMetadata(requirement=40.0),
        source_emission=emission,
    )
    assert "FAIL" in _extract_text(str(failing))


def test_attenuation_metadata_header_and_distance(tmp_path) -> None:
    """A populated metadata header and the recovered distance print."""
    metadata = ReportMetadata(
        specimen="Industrial fan plant",
        client="Acoustic Consultants Ltd.",
        test_room="Nearest dwelling",
        temperature=10.0,
        laboratory="Phonometry Reference Laboratory",
    )
    out = tmp_path / "meta.pdf"
    _attenuation().report(str(out), metadata=metadata, source_emission=_emission())
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Industrial fan plant" in text
    assert "Phonometry Reference Laboratory" in text
    assert "200" in text  # distance d recovered from Adiv


def test_attenuation_spanish_fiche(tmp_path) -> None:
    """``language="es"`` renders the Spanish attenuation fiche."""
    import re

    out = tmp_path / "es.pdf"
    _attenuation().report(
        str(out),
        metadata=ReportMetadata(requirement=50.0, laboratory="Ejemplo"),
        source_emission=_emission(),
        language="es",
    )
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "CUMPLE" in text  # 46.5 dB <= 50 dB
    assert "prevista" in text  # "predicted"
    assert "no una medici" in text  # "not a measurement"
    assert re.search(r"\d+,\d", text)  # comma decimal separator


# --------------------------------------------------------------------------- #
# Barrier insertion-loss prediction fiche
# --------------------------------------------------------------------------- #
def test_barrier_report_structure_and_numbers(tmp_path) -> None:
    """The barrier fiche boxes the mean insertion loss and cites the model."""
    out = tmp_path / "bar.pdf"
    assert _barrier().report(str(out)) == str(out)
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "IL = 12.9 dB" in text  # boxed mean insertion loss
    assert "prediction" in text
    assert "not a measurement" in text
    # The actual diffraction model is cited, not ISO 9613-2's Dz formula.
    assert "wave-theoretic" in text
    assert "MacDonald" in text
    assert "ISO 9613-2:1996" in text  # named as the complemented term
    assert "21.0" in text  # IL at 8 kHz from the table


def test_barrier_verbose_adds_fresnel_number(tmp_path) -> None:
    """``verbose=True`` adds the Fresnel-number column."""
    plain = tmp_path / "plain.pdf"
    _barrier().report(str(plain))
    assert "7.05" not in _extract_text(str(plain))

    verbose = tmp_path / "verbose.pdf"
    _barrier().report(str(verbose), verbose=True)
    assert "7.05" in _extract_text(str(verbose))  # N at 8 kHz


def test_barrier_requirement_higher_is_better(tmp_path) -> None:
    """The insertion loss passes at or above the required minimum."""
    result = _barrier()  # mean IL = 12.9 dB
    passing = tmp_path / "pass.pdf"
    result.report(str(passing), metadata=ReportMetadata(requirement=8.0))
    assert "PASS" in _extract_text(str(passing))

    failing = tmp_path / "fail.pdf"
    result.report(str(failing), metadata=ReportMetadata(requirement=20.0))
    assert "FAIL" in _extract_text(str(failing))


def test_barrier_kurze_anderson_model_cited(tmp_path) -> None:
    """The Kurze-Anderson method is named in the basis line."""
    out = tmp_path / "ka.pdf"
    _barrier(method="kurze_anderson").report(str(out))
    _assert_one_page(str(out))
    assert "Kurze-Anderson" in _extract_text(str(out))


def test_barrier_lightweight_without_metadata(tmp_path) -> None:
    """A barrier fiche with no metadata is still a valid one-page prediction."""
    out = tmp_path / "bare.pdf"
    _barrier().report(str(out))
    _assert_one_page(str(out))
    assert "prediction" in _extract_text(str(out))


def test_barrier_spanish_fiche(tmp_path) -> None:
    """``language="es"`` renders the Spanish barrier fiche."""
    out = tmp_path / "es.pdf"
    _barrier().report(str(out), language="es")
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "prevista" in text  # "predicted"
    assert "no una medici" in text  # "not a measurement"


# --------------------------------------------------------------------------- #
# Shared validation
# --------------------------------------------------------------------------- #
def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ``ValueError`` for both fiches."""
    attenuation = _attenuation()
    out_a = str(tmp_path / "a.pdf")
    with pytest.raises(ValueError, match="engine"):
        attenuation.report(out_a, engine="weasyprint")
    barrier = _barrier()
    out_b = str(tmp_path / "b.pdf")
    with pytest.raises(ValueError, match="engine"):
        barrier.report(out_b, engine="weasyprint")


def test_unknown_language_rejected(tmp_path) -> None:
    """An unsupported language raises ``ValueError`` for both fiches."""
    attenuation = _attenuation()
    out_a = str(tmp_path / "a.pdf")
    with pytest.raises(ValueError, match="language"):
        attenuation.report(out_a, language="fr")
    barrier = _barrier()
    out_b = str(tmp_path / "b.pdf")
    with pytest.raises(ValueError, match="language"):
        barrier.report(out_b, language="fr")
