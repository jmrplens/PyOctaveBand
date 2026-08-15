#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for the ISO 9614-3 precision sound-power-by-intensity ``.report()`` fiche.

The rendered values are checked against a clean-room oracle derived from the
standard, independent of the library's own combination path. A machine is
enclosed by a surface of ``N`` equal partial surfaces of area ``Si``; the normal
intensity is uniform over the surface, so every partial surface reports the same
signed intensity ``In`` per band. The partial powers ``Pi = In*Si`` sum to the
band power ``P = In*S`` (ISO 9614-3:2002 Eq. 5/8), so the band sound-power level
is ``LW = 10*lg(In*S/P0)``, ``P0 = 1 pW`` (Eq. 9), the normalized level is
``LW0 = LW - 15*lg((B/101325)*(296,15/(273,15 + theta)))`` (Eq. 10), and the
A-weighted total is ``LWA = 10*lg(sum 10^((LW + Ck)/10))`` with the standard
one-third-octave A-weighting corrections Ck, summed over the qualified bands
only (clause 10 f) 2)). The per-band expanded uncertainty is twice the Table 1
standard deviation of reproducibility (clause 4.3).

The tests recompute those numbers from the closed forms and assert they appear
in the PDF, together with the mandatory statement naming the bands the
A-weighted determination leaves out and why. Values are read back via pypdf
text extraction; structural facts (one page, the paired wide table, rejected
engines, languages and band counts) complete the rendering contract.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from phonometry import ReportMetadata
from phonometry.emission import (
    SoundPowerWarning,
    precision_field_indicators,
    precision_qualification,
    sound_power_intensity_precision,
)

_PDF_MAGIC = b"%PDF"

# Standard one-third-octave A-weighting corrections Ck (dB), IEC 61672 /
# ISO 3744 Annex E Table E.1, at the example band centres.
_CK_THIRD = {
    100: -19.1, 125: -16.1, 160: -13.4, 200: -10.9, 250: -8.6, 315: -6.6,
    400: -4.8, 500: -3.2, 630: -1.9, 800: -0.8, 1000: 0.0, 1250: 0.6,
    1600: 1.0, 2000: 1.2, 2500: 1.3, 3150: 1.2,
}
# ISO 9614-3:2002 Table 1 standard deviations of reproducibility (dB).
_SIGMA_R0 = {
    100: 2.0, 125: 2.0, 160: 2.0, 200: 1.5, 250: 1.5, 315: 1.5, 400: 1.0,
    500: 1.0, 630: 1.0, 800: 1.0, 1000: 1.0, 1250: 1.0, 1600: 1.0, 2000: 1.0,
    2500: 1.0, 3150: 1.0,
}

_FREQS = np.array([200, 250, 315, 400, 500], dtype=float)
# Uniform per-band signed normal intensity (W/m^2), shared by every segment.
_INTENSITY = np.array([1.0e-5, 2.0e-5, 3.0e-5, 2.5e-5, 1.5e-5])
_N_SEG = 4
_SEG_AREA = 0.5
_SURFACE = _N_SEG * _SEG_AREA  # 2.0 m^2
# Test atmosphere: warm and at altitude, so the Eq. 10 normalization is
# visible at the one decimal the fiche prints.
_TEMPERATURE = 30.0
_PRESSURE_PA = 95_000.0
# Instrument pressure-residual intensity index, so Ld = 15 - 10 = 5 dB.
_RESIDUAL_INDEX = 15.0


def _assert_one_page(path: str) -> None:
    from pypdf import PdfReader

    with open(path, "rb") as handle:
        assert handle.read(4) == _PDF_MAGIC
    assert os.path.getsize(path) > 0
    assert len(PdfReader(path).pages) == 1


def _extract_text(path: str) -> str:
    """Whitespace-normalized page text (PDF line wraps fold to single spaces)."""
    from pypdf import PdfReader

    raw = "\n".join(page.extract_text() for page in PdfReader(path).pages)
    return " ".join(raw.split())


def _determination(intensity: np.ndarray = _INTENSITY, freqs: np.ndarray = _FREQS):
    """A uniform-field precision determination, so LW and LW0 are derivable."""
    scan = np.tile(intensity, (_N_SEG, 1))
    return sound_power_intensity_precision(
        scan,
        np.full(_N_SEG, _SEG_AREA),
        frequencies=freqs,
        temperature=_TEMPERATURE,
        barometric_pressure=_PRESSURE_PA,
    )


def _qualification(margin: np.ndarray, freqs: np.ndarray = _FREQS):
    """Annex B indicators and Annex C criteria for the uniform-field scan.

    ``margin`` is the per-band amount by which the segment pressure levels
    exceed the segment intensity levels, so it is the pressure-intensity
    indicator of the band; a band whose margin exceeds ``Ld`` fails criterion 2.
    The two scans coincide to 0,1 dB, inside the s/2 of Table 1 everywhere.
    """
    scan = np.tile(_INTENSITY[: freqs.size], (_N_SEG, 1))
    pressure_levels = 10.0 * np.log10(np.abs(scan) / 1.0e-12) + margin[None, :]
    indicators = precision_field_indicators(scan, pressure_levels)
    level = 10.0 * np.log10(np.abs(np.mean(scan, axis=0)) / 1.0e-12)
    criteria = precision_qualification(
        indicators,
        scan_intensity_level_1=level,
        scan_intensity_level_2=level + 0.1,
        pressure_residual_index=_RESIDUAL_INDEX,
        frequencies=freqs,
    )
    return indicators, criteria


def _oracle_lw(intensity: np.ndarray = _INTENSITY) -> np.ndarray:
    """Closed-form band LW = 10*lg(In*S/P0), P0 = 1 pW (Eq. 5/8/9)."""
    return 10.0 * np.log10(intensity * _SURFACE / 1.0e-12)


def _oracle_normalization() -> float:
    """Closed-form Eq. 10 normalization term, in dB (LW0 = LW - this)."""
    return 15.0 * np.log10(
        (_PRESSURE_PA / 101325.0) * (296.15 / (273.15 + _TEMPERATURE))
    )


def _oracle_lwa(
    keep: np.ndarray | None = None, freqs: np.ndarray = _FREQS
) -> float:
    """Closed-form LWA over the kept bands, via the Table E.1 corrections."""
    lw = _oracle_lw(_INTENSITY[: freqs.size])
    ck = np.array([_CK_THIRD[int(f)] for f in freqs])
    contributions = 10.0 ** ((lw + ck) / 10.0)
    if keep is not None:
        contributions = contributions[keep]
    return float(10.0 * np.log10(np.sum(contributions)))


# --- clean-room oracle --------------------------------------------------------


def test_hand_oracle_matches_library() -> None:
    """The library LW / LW0 / LWA equal the closed-form Eq. 5/8/9/10 values."""
    res = _determination()
    np.testing.assert_allclose(res.sound_power_level, _oracle_lw(), atol=1e-9)
    np.testing.assert_allclose(
        res.sound_power_level_normalized,
        _oracle_lw() - _oracle_normalization(),
        atol=1e-9,
    )
    assert res.sound_power_level_a == pytest.approx(_oracle_lwa(), abs=1e-9)


def test_report_renders_oracle_values(tmp_path) -> None:
    """The fiche prints the hand-derived LWA, band LW, LW0 and Table 1 U."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _determination()
    out = tmp_path / "iso9614_3.pdf"
    returned = res.report(str(out))
    assert returned == str(out)
    _assert_one_page(str(out))
    text = _extract_text(str(out))

    # A-weighted total to one decimal, boxed with its reference power.
    assert f"{_oracle_lwa():.1f}" in text
    assert "re 1 pW" in text
    # Two band levels and their normalized counterparts (315 Hz and 400 Hz).
    lw = _oracle_lw()
    lw0 = lw - _oracle_normalization()
    for index in (2, 3):
        assert f"{lw[index]:.1f}" in text
        assert f"{lw0[index]:.1f}" in text
    # The Eq. 10 normalization is stated in the basis strip, to two decimals.
    assert f"{_oracle_normalization():.2f}".replace("-", "−") in text
    # Nominal band labels, the band-set caption and the range it covers.
    assert "One-third-octave-band sound power levels, 200 Hz to 500 Hz" in text
    # Method, grade and the measurement surface S = 2.00 m2.
    assert "ISO 9614-3:2002" in text
    assert "precision grade (accuracy grade 1)" in text
    assert "2.00" in text


def test_uncertainty_column_follows_table_1(tmp_path) -> None:
    """Every band prints U = 2*sigma_R0 from its ISO 9614-3 Table 1 row."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    # Six consecutive one-third-octave bands crossing all three Table 1 tiers:
    # 160 Hz -> 2*2,0 = 4,0 dB; 200 to 315 Hz -> 2*1,5 = 3,0 dB; 400 and
    # 500 Hz -> 2*1,0 = 2,0 dB.
    freqs = np.array([160, 200, 250, 315, 400, 500], dtype=float)
    res = _determination(np.full(freqs.size, 2.0e-5), freqs)
    out = tmp_path / "uncertainty.pdf"
    res.report(str(out))
    text = _extract_text(str(out))
    for freq in freqs:
        assert f"{2.0 * _SIGMA_R0[int(freq)]:.1f}" in text
    assert "Table 1 standard deviation of reproducibility" in text


def test_octave_bands_have_no_tabulated_uncertainty(tmp_path) -> None:
    """Table 1 tabulates one-third-octave bands; an octave set prints an em dash."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    freqs = np.array([250, 500, 1000, 2000], dtype=float)
    res = _determination(_INTENSITY[:4], freqs)
    out = tmp_path / "octave.pdf"
    res.report(str(out))
    text = _extract_text(str(out))
    assert "Octave-band sound power levels, 250 Hz to 2000 Hz" in text
    assert "—" in text


# --- Annex B tabulation (verbose) ---------------------------------------------


def test_verbose_tabulates_annex_b_indicators(tmp_path) -> None:
    """verbose=True adds the four Annex B indicators and the grade cell."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _determination()
    indicators, criteria = _qualification(np.full(_FREQS.size, 2.0))
    out = tmp_path / "verbose.pdf"
    res.report(str(out), verbose=True, indicators=indicators, criteria=criteria)
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    flat = "".join(text.split())
    for header in ("FT", "Fp|In|", "FpIn", "FS"):
        assert header in flat
    assert "Grade" in text
    # Every band clears Ld = 5 dB with a 2 dB indicator, so each grade cell
    # reports the single accuracy grade of the method.
    assert criteria.qualified is not None
    assert criteria.qualified.tolist() == [True] * _FREQS.size


# --- clause 10 f) 2) omission -------------------------------------------------


def test_failing_band_is_omitted_and_named(tmp_path) -> None:
    """A band failing criterion 2 leaves LWA and is named with its criterion."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _determination()
    # A 200 Hz pressure-intensity indicator of 8 dB exceeds the probe's
    # Ld = 5 dB; the other bands sit at 2 dB and qualify.
    margin = np.array([8.0, 2.0, 2.0, 2.0, 2.0])
    indicators, criteria = _qualification(margin)
    assert criteria.qualified is not None
    assert criteria.qualified.tolist() == [False, True, True, True, True]

    out = tmp_path / "omitted.pdf"
    res.report(
        str(out),
        indicators=indicators,
        criteria=criteria,
        residual_index=_RESIDUAL_INDEX,
    )
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    # The boxed LWA is the total over the qualified bands, not the result's own.
    keep = np.array([False, True, True, True, True])
    assert f"{_oracle_lwa(keep):.1f}" in text
    assert "Bands omitted from LWA: 200 Hz (criterion 2)" in text
    # The strip states the dynamic capability the criterion was judged against.
    assert "5.0" in text


def test_not_applicable_band_named_with_clause_9_2(tmp_path) -> None:
    """A net-negative band prints an em dash and is named under clause 9.2."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    scan = np.tile(_INTENSITY, (_N_SEG, 1)).copy()
    # Drive the 250 Hz band net-negative (more energy flowing in than out).
    scan[:, 1] = -_INTENSITY[1]
    with pytest.warns(SoundPowerWarning, match="non-positive in one or more bands"):
        res = sound_power_intensity_precision(
            scan,
            np.full(_N_SEG, _SEG_AREA),
            frequencies=_FREQS,
            temperature=_TEMPERATURE,
            barometric_pressure=_PRESSURE_PA,
        )
    indicators, criteria = _qualification(np.full(_FREQS.size, 2.0))
    out = tmp_path / "not_applicable.pdf"
    res.report(str(out), indicators=indicators, criteria=criteria)
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "250 Hz (clause 9.2)" in text
    # The determinable 315 Hz band still prints its level.
    assert f"{_oracle_lw()[2]:.1f}" in text


def test_without_criteria_the_fiche_says_so(tmp_path) -> None:
    """No Annex C qualification: the result's own LWA, and the strip says why."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _determination()
    out = tmp_path / "unqualified.pdf"
    res.report(str(out))
    text = _extract_text(str(out))
    assert "The Annex C qualification was not supplied" in text
    assert f"{_oracle_lwa():.1f}" in text


def test_clause_9_2_band_is_named_without_any_qualification(tmp_path) -> None:
    """A not-applicable band is named even when no Annex C data was supplied.

    The omission statement is the standard's, not the qualification's: a band
    with non-positive net power leaves the A-weighted determination whether or
    not the criteria were ever evaluated, so it is named either way.
    """
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    scan = np.tile(_INTENSITY, (_N_SEG, 1)).copy()
    scan[:, 1] = -_INTENSITY[1]
    with pytest.warns(SoundPowerWarning, match="non-positive in one or more bands"):
        res = sound_power_intensity_precision(
            scan,
            np.full(_N_SEG, _SEG_AREA),
            frequencies=_FREQS,
            temperature=_TEMPERATURE,
            barometric_pressure=_PRESSURE_PA,
        )
    out = tmp_path / "unqualified_negative.pdf"
    res.report(str(out))
    text = _extract_text(str(out))
    assert "The Annex C qualification was not supplied" in text
    assert "Bands omitted from LWA: 250 Hz (clause 9.2)" in text


def test_qualified_set_states_that_nothing_is_omitted(tmp_path) -> None:
    """Every band qualifying is stated too, not left to silence."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _determination()
    indicators, criteria = _qualification(np.full(_FREQS.size, 2.0))
    out = tmp_path / "all_qualified.pdf"
    res.report(str(out), indicators=indicators, criteria=criteria)
    text = _extract_text(str(out))
    assert "Every band satisfies the Annex C criteria" in text


# --- the wide (one-third-octave) band set -------------------------------------


def test_wide_band_set_is_paired_onto_one_page(tmp_path) -> None:
    """Sixteen one-third-octave bands print in two column groups, on one page."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    freqs = np.array(list(_CK_THIRD), dtype=float)
    intensity = np.full(freqs.size, 2.0e-5)
    res = _determination(intensity, freqs)
    out = tmp_path / "wide.pdf"
    res.report(str(out))
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    for label in ("100", "3150"):
        assert label in text
    assert "One-third-octave-band sound power levels, 100 Hz to 3150 Hz" in text
    # The paired layout repeats the column headings for the second group.
    assert text.count("U [dB]") == 2


# --- verdict against a declared limit -----------------------------------------


@pytest.mark.parametrize(("limit", "verdict"), [(120.0, "PASS"), (60.0, "FAIL")])
def test_verdict_against_declared_limit(tmp_path, limit: float, verdict: str) -> None:
    """A declared limit yields a PASS/FAIL verdict (lower is better)."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _determination()
    out = tmp_path / f"verdict_{limit}.pdf"
    res.report(str(out), metadata=ReportMetadata(requirement=limit))
    text = _extract_text(str(out))
    assert verdict in text
    assert "declared limit" in text


def test_verdict_follows_the_boxed_level(tmp_path) -> None:
    """The verdict compares the qualified-band LWA the fiche boxes, not the raw one."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _determination()
    indicators, criteria = _qualification(np.array([8.0, 2.0, 2.0, 2.0, 2.0]))
    keep = np.array([False, True, True, True, True])
    qualified = _oracle_lwa(keep)
    # A limit between the two totals: the determination fails on every band and
    # passes on the qualified ones, so the verdict follows the printed number.
    assert qualified < _oracle_lwa()
    limit = 0.5 * (qualified + _oracle_lwa())
    out = tmp_path / "verdict_qualified.pdf"
    res.report(
        str(out),
        metadata=ReportMetadata(requirement=limit),
        indicators=indicators,
        criteria=criteria,
    )
    text = _extract_text(str(out))
    assert f"{qualified:.1f}" in text
    assert "PASS" in text


# --- metadata header ----------------------------------------------------------


def test_metadata_header_renders(tmp_path) -> None:
    """Supplied metadata renders the source, environment and identity fields."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _determination()
    metadata = ReportMetadata(
        client="Example works",
        specimen="Air compressor",
        test_room="Machine hall",
        instrumentation="Class 1 p-p probe, s/n 7",
        temperature=_TEMPERATURE,
        pressure=95.0,
        laboratory="Acoustics lab",
        report_id="SP-9614-3",
    )
    out = tmp_path / "meta.pdf"
    res.report(str(out), metadata=metadata)
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    for value in (
        "Example works",
        "Air compressor",
        "Machine hall",
        "Class 1 p-p probe, s/n 7",
        "Acoustics lab",
        "95",
    ):
        assert value in text


# --- Spanish fiche ------------------------------------------------------------


def test_spanish_fiche_translates_every_fixed_string(tmp_path) -> None:
    """No English template survives into the Spanish precision fiche.

    ``_i18n.t`` falls back to the English text when a key is missing, so a typo
    in the translation table shows up as a stray English sentence rather than
    as an error. The Spanish wording of each block is asserted, and the English
    original of each is asserted absent.
    """
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _determination()
    indicators, criteria = _qualification(np.array([8.0, 2.0, 2.0, 2.0, 2.0]))
    out = tmp_path / "es.pdf"
    res.report(
        str(out),
        language="es",
        indicators=indicators,
        criteria=criteria,
        residual_index=_RESIDUAL_INDEX,
    )
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    for spanish in (
        "Determinación de la potencia acústica",
        "cualificada mediante el procedimiento del Anexo C",
        "grado de precisión (grado de exactitud 1)",
        "Niveles de potencia acústica en bandas de tercio de octava, de 200 Hz a 500 Hz",
        "normalizado",
        "Superficie de medición",
        "Incertidumbre expandida",
        "El Anexo C cualifica una banda",
        "Bandas omitidas de LWA: 200 Hz (criterio 2)",
    ):
        assert spanish in text, spanish
    for english in (
        "qualified by the Annex C procedure",
        "precision grade (accuracy grade 1)",
        "One-third-octave-band sound power levels",
        "Expanded uncertainty",
        "Annex C qualifies a band",
        "Bands omitted from",
        "criterion 2",
    ):
        assert english not in text, english
    # Comma decimal separator on the A-weighted total of the qualified bands.
    keep = np.array([False, True, True, True, True])
    assert f"{_oracle_lwa(keep):.1f}".replace(".", ",") in text


# --- rendering contract -------------------------------------------------------


def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ValueError."""
    res = _determination()
    with pytest.raises(ValueError, match="engine"):
        res.report(str(tmp_path / "x.pdf"), engine="weasyprint")


def test_unknown_language_rejected(tmp_path) -> None:
    """An unknown fiche language raises ValueError."""
    res = _determination()
    with pytest.raises(ValueError, match="language"):
        res.report(str(tmp_path / "bad.pdf"), language="xx")


def test_indicators_over_other_bands_rejected(tmp_path) -> None:
    """Indicators measured over a different band set are refused, not printed."""
    res = _determination()
    other, _criteria = _qualification(np.full(3, 2.0), _FREQS[:3])
    with pytest.raises(ValueError, match="indicators"):
        res.report(str(tmp_path / "mismatch.pdf"), indicators=other)


def test_criteria_over_other_bands_rejected(tmp_path) -> None:
    """Criteria evaluated over a different band set are refused too."""
    res = _determination()
    _indicators, other = _qualification(np.full(3, 2.0), _FREQS[:3])
    with pytest.raises(ValueError, match="criteria"):
        res.report(str(tmp_path / "mismatch.pdf"), criteria=other)


def test_residual_index_over_other_bands_rejected(tmp_path) -> None:
    """A per-band residual index must span the determination's bands."""
    res = _determination()
    with pytest.raises(ValueError, match="residual_index"):
        res.report(str(tmp_path / "mismatch.pdf"), residual_index=[15.0, 15.0])
