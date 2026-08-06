#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for :mod:`phonometry.speech.sii` (Speech Intelligibility Index, ANSI S3.5-1997).

All four band procedures are validated against the reference implementation
of ASA Working Group S3-79, the committee that maintains ANSI S3.5
(``SII.C`` and its eight official test-input files with published results,
from the WG support site sii.to): ``CB.TST``/``CB_1.TST`` for the
critical-band procedure, ``ECB.TST``/``ECB_1.TST`` for the
equally-contributing critical-band procedure, ``TO.TST``/``TO_1.TST`` for
the one-third-octave procedure and ``OCTAVE.TST``/``OCTAVE_1.TST`` for the
octave-band procedure, the ``_1`` file of each pair exercising an
alternative band-importance function. Two of the eight, ``CB_1.TST`` and
``ECB_1.TST``, are the same confirmation twice (see
``test_sii_wg_s3_79_official_test_cases``), so the eight cases carry seven
independent confirmations. They are also validated against the
standard's own tabulated constants and its Annex C.2 worked example (with
the official errata applied), and against the independent Hornsby worksheet
and R CRAN "SII" implementations where they overlap. See
``tests/reference_data/`` for the provenance of every constant.
"""

from __future__ import annotations

import numpy as np
import pytest
from reference_data import (
    ANSIS3_5_ANNEX_C1,
    ANSIS3_5_ANNEX_C1_LEVEL_DISTORTION_I5,
    ANSIS3_5_ANNEX_C1_NOISE,
    ANSIS3_5_ANNEX_C1_SPEECH,
    ANSIS3_5_ANNEX_C2,
    ANSIS3_5_ANNEX_C2_MASKING,
    ANSIS3_5_BAND_IMPORTANCE_SUM,
    ANSIS3_5_CRITICAL_IMPORTANCE_SUM,
    ANSIS3_5_CRITICAL_TABLE1,
    ANSIS3_5_DISTURBANCE_5000HZ,
    ANSIS3_5_EQUAL_IMPORTANCE_SUM,
    ANSIS3_5_LOUD_1KHZ,
    ANSIS3_5_NOISE_PLUS_LOSS,
    ANSIS3_5_OCTAVE_IMPORTANCE_SUM,
    ANSIS3_5_OCTAVE_TABLE4,
    ANSIS3_5_OCTAVE_TABLE4_SHARED,
    ANSIS3_5_STANDARD_QUIET,
    ANSIS3_5_THIRD_OCTAVE_TABLE3,
    ANSIS3_5_WG_CB1_IMPORTANCE,
    ANSIS3_5_WG_CB1_SII,
    ANSIS3_5_WG_CB1_SII_EXACT,
    ANSIS3_5_WG_CB_NOISE,
    ANSIS3_5_WG_CB_SII,
    ANSIS3_5_WG_CB_SII_EXACT,
    ANSIS3_5_WG_CB_SPEECH,
    ANSIS3_5_WG_CB_THRESHOLD,
    ANSIS3_5_WG_ECB1_IMPORTANCE,
    ANSIS3_5_WG_ECB1_SII,
    ANSIS3_5_WG_ECB1_SII_EXACT,
    ANSIS3_5_WG_ECB_NOISE,
    ANSIS3_5_WG_ECB_SII,
    ANSIS3_5_WG_ECB_SII_EXACT,
    ANSIS3_5_WG_ECB_SPEECH,
    ANSIS3_5_WG_ECB_THRESHOLD,
    ANSIS3_5_WG_FLAT_CASES,
    ANSIS3_5_WG_OCTAVE1_IMPORTANCE,
    ANSIS3_5_WG_OCTAVE1_SII,
    ANSIS3_5_WG_OCTAVE1_SII_EXACT,
    ANSIS3_5_WG_OCTAVE_NOISE,
    ANSIS3_5_WG_OCTAVE_SII,
    ANSIS3_5_WG_OCTAVE_SII_EXACT,
    ANSIS3_5_WG_OCTAVE_SPEECH,
    ANSIS3_5_WG_OCTAVE_THRESHOLD,
    ANSIS3_5_WG_TO1_IMPORTANCE,
    ANSIS3_5_WG_TO1_SII,
    ANSIS3_5_WG_TO1_SII_EXACT,
    ANSIS3_5_WG_TO_NOISE,
    ANSIS3_5_WG_TO_SII,
    ANSIS3_5_WG_TO_SII_EXACT,
    ANSIS3_5_WG_TO_SPEECH,
    ANSIS3_5_WG_TO_THRESHOLD,
)

from phonometry.speech import sii


def test_band_importance_sums_to_one() -> None:
    # ANSI S3.5-1997 Table 3: the band-importance function is normalised.
    assert sii.BAND_IMPORTANCE.sum() == pytest.approx(
        ANSIS3_5_BAND_IMPORTANCE_SUM, abs=1e-12
    )
    assert sii.BAND_IMPORTANCE.size == 18
    np.testing.assert_allclose(
        sii.BAND_CENTERS,
        [160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0,
         1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0,
         8000.0],
    )


def test_sii_standard_speech_in_quiet() -> None:
    # Standard normal-effort spectrum, quiet field, normal hearing, at the
    # full precision of the official worksheet value.
    result = sii.speech_intelligibility_index("normal")
    assert result.sii == pytest.approx(ANSIS3_5_STANDARD_QUIET, abs=1e-6)
    assert 0.0 <= result.sii <= 1.0
    # Clause 5.6: in quiet the disturbance is the reference internal noise
    # itself, Di = max(Zi, Xi') = Xi' = -23.6 dB at 5000 Hz. An energy-sum
    # Di would read above this wherever Zi is comparable with Xi'.
    assert result.disturbance[15] == pytest.approx(
        ANSIS3_5_DISTURBANCE_5000HZ, abs=1e-2
    )


def test_sii_noise_plus_hearing_loss() -> None:
    # Discriminating oracle for the clause 5.6 maximum (ANSI S3.5-1997):
    # normal speech, flat 30 dB noise, flat 40 dB hearing loss. The WG S3-79
    # reference implementation (SII.C) gives 0.2184539329 (Hornsby worksheet:
    # 0.2185); an energy-sum disturbance reads 0.1841 -- an error large
    # enough to flip an intelligibility grade.
    result = sii.speech_intelligibility_index(
        "normal",
        noise_spectrum=np.full(18, 30.0),
        threshold=np.full(18, 40.0),
    )
    assert result.sii == pytest.approx(ANSIS3_5_NOISE_PLUS_LOSS, abs=1e-6)


def test_sii_annex_c2_worked_example() -> None:
    # ANSI S3.5-1997 Annex C.2 worked example (one-third-octave method):
    # speech 54 dB in every band, noise 40, 30 and 20 dB in the first three
    # bands, normal hearing. The WG S3-79 reference implementation (SII.C)
    # gives 0.8513748619; the R CRAN package "SII" prints 0.8513749.
    result = sii.speech_intelligibility_index(
        np.full(18, 54.0),
        np.array([40.0, 30.0, 20.0] + [0.0] * 15),
        threshold=np.zeros(18),
    )
    assert result.sii == pytest.approx(ANSIS3_5_ANNEX_C2, abs=1e-6)
    # Table C.2's printed Zi column, first three rows (errata-consistent):
    # the officially corrected first-row slope Ci = -46.59 (not the printed
    # -45.59, WG S3-79 errata) is required to reproduce Z2 = 34.66 dB. The
    # 250 Hz cell is printed 25.04 while the exact chain gives 25.0468 (the
    # print truncates rather than rounds), hence the half-unit tolerance of
    # the two-decimal print.
    np.testing.assert_allclose(
        result.masking[:3], ANSIS3_5_ANNEX_C2_MASKING, atol=1e-2
    )


def test_sii_wg_s3_79_official_test_case_to() -> None:
    # ASA WG S3-79 official test input TO.TST for the one-third-octave
    # procedure (DevelopmentKit, sii.to). Published result: SII = 0.445; the
    # committee's SII.C, compiled unmodified, prints 0.4453910059.
    result = sii.speech_intelligibility_index(
        np.array(ANSIS3_5_WG_TO_SPEECH),
        np.array(ANSIS3_5_WG_TO_NOISE),
        threshold=np.array(ANSIS3_5_WG_TO_THRESHOLD),
    )
    assert result.sii == pytest.approx(ANSIS3_5_WG_TO_SII, abs=5e-4)
    assert result.sii == pytest.approx(ANSIS3_5_WG_TO_SII_EXACT, abs=1e-9)


def test_sii_wg_s3_79_official_test_case_to1() -> None:
    # ASA WG S3-79 official test input TO_1.TST: the same procedure with an
    # alternative band-importance function. Published result: SII = 0.438;
    # SII.C prints 0.4382176540. The alternative-importance index is the dot
    # product of the alternative Ii with the per-band audibility Ai (which
    # already carries the level-distortion factor), exactly as SII.C forms it.
    result = sii.speech_intelligibility_index(
        np.array(ANSIS3_5_WG_TO_SPEECH),
        np.array(ANSIS3_5_WG_TO_NOISE),
        threshold=np.array(ANSIS3_5_WG_TO_THRESHOLD),
    )
    sii_alt = float(
        np.sum(np.array(ANSIS3_5_WG_TO1_IMPORTANCE) * result.band_audibility)
    )
    assert sii_alt == pytest.approx(ANSIS3_5_WG_TO1_SII, abs=5e-4)
    assert sii_alt == pytest.approx(ANSIS3_5_WG_TO1_SII_EXACT, abs=1e-9)


def test_masking_spectrum_matches_reference() -> None:
    # Equivalent masking spectrum level Zi for the standard spectrum in quiet
    # (first four one-third-octave bands), as printed by the WG S3-79
    # reference implementation SII.C for this input: z[0..3] = 8.410000,
    # -1.664717, 0.705214, 0.381701.
    result = sii.speech_intelligibility_index("normal")
    reference_zi = np.array([8.41, -1.6647, 0.7052, 0.3817])
    np.testing.assert_allclose(result.masking[:4], reference_zi, atol=1e-4)


def test_masking_slope_keeps_the_printed_summation_order() -> None:
    """The clause 5.4 slope is summed as printed: ``(Bi + 10 lg fi) - 6.353``.

    Floating-point addition is not associative, so folding the printed 6.353
    into the bandwidth term first, as ``Bi + (10 lg fi - 6.353)``, shifts the
    equivalent masking spectrum level in its last bits and with it the shipped
    index and the report fiche. This asserts bit equality against the slope
    written out in the standard's own order, independently of how the module
    happens to factor it, so the association cannot drift again unnoticed.
    """
    speech = np.linspace(20.0, 70.0, 18)
    noise = np.linspace(45.0, 5.0, 18)
    result = sii.speech_intelligibility_index(speech, noise)

    f = sii.BAND_CENTERS
    b = np.maximum(noise, speech - 24.0)
    c = -80.0 + 0.6 * (b + 10.0 * np.log10(f) - 6.353)
    expected = np.empty(18)
    expected[0] = b[0]
    for i in range(1, 18):
        contrib = 10.0 ** (
            0.1 * (b[:i] + 3.32 * c[:i] * np.log10(0.89 * f[i] / f[:i]))
        )
        expected[i] = 10.0 * np.log10(10.0 ** (0.1 * noise[i]) + np.sum(contrib))

    # Bit equality, not a tolerance: a reassociation shows up here as a few
    # units in the last place and would pass any reasonable atol.
    assert result.masking.tolist() == expected.tolist()


def test_noise_reduces_index_monotonically() -> None:
    # More masking noise can only lower the index.
    quiet = sii.speech_intelligibility_index("normal").sii
    mild = sii.speech_intelligibility_index("normal", np.full(18, 20.0)).sii
    loud = sii.speech_intelligibility_index("normal", np.full(18, 40.0)).sii
    assert quiet > mild > loud >= 0.0


def test_hearing_loss_reduces_index() -> None:
    # A raised hearing threshold lifts the internal noise and lowers the index.
    normal = sii.speech_intelligibility_index("normal").sii
    impaired = sii.speech_intelligibility_index(
        "normal", threshold=np.full(18, 40.0)
    ).sii
    assert impaired < normal


def test_extreme_speech_level_stays_bounded() -> None:
    # The level-distortion factor is clipped to [0, 1], so even an absurdly loud
    # speech level cannot drive the audibility (or the index) negative.
    result = sii.speech_intelligibility_index(np.full(18, 200.0))
    assert np.all(result.band_audibility >= 0.0)
    assert 0.0 <= result.sii <= 1.0


def test_standard_speech_spectrum_values() -> None:
    spectrum = sii.standard_speech_spectrum("normal")
    assert spectrum[0] == pytest.approx(32.41)
    assert spectrum[8] == pytest.approx(25.01)
    assert spectrum[17] == pytest.approx(1.13)
    # A returned copy must not alias the module constant.
    spectrum[0] = 0.0
    assert sii.standard_speech_spectrum("normal")[0] == pytest.approx(32.41)


def test_vocal_effort_spectra_spot_values() -> None:
    # ANSI S3.5-1997 Table 3, cross-verified against reference implementations
    # (Google speech_intelligibility_index, R CRAN SII) at 1 kHz (band 8).
    assert sii.standard_speech_spectrum("raised")[8] == pytest.approx(33.86)
    assert sii.standard_speech_spectrum("loud")[8] == pytest.approx(ANSIS3_5_LOUD_1KHZ)
    assert sii.standard_speech_spectrum("shout")[8] == pytest.approx(51.31)
    assert sii.VOCAL_EFFORTS == ("normal", "raised", "loud", "shout")


def test_vocal_effort_overall_level_increases() -> None:
    # The overall speech level grows with vocal effort (Table 3): reconstruct
    # each spectrum's overall free-field SPL and check it is monotone.
    f = sii.BAND_CENTERS
    bw = (2.0 ** (1 / 6) - 2.0 ** (-1 / 6)) * f
    overall = [
        10.0 * np.log10(
            np.sum(10.0 ** ((sii.standard_speech_spectrum(e) + 10 * np.log10(bw)) / 10))
        )
        for e in sii.VOCAL_EFFORTS
    ]
    assert overall == sorted(overall)  # normal < raised < loud < shout
    # Matches the known ANSI vocal-effort overall levels (dB SPL).
    assert overall[0] == pytest.approx(62.35, abs=0.1)
    assert overall[1] == pytest.approx(68.3, abs=0.1)
    assert overall[2] == pytest.approx(74.86, abs=0.1)
    assert overall[3] == pytest.approx(82.36, abs=0.1)


def test_higher_effort_raises_index_in_noise() -> None:
    # In a fixed noise, speaking louder improves intelligibility.
    noise = np.full(18, 40.0)
    indices = [
        sii.speech_intelligibility_index(e, noise).sii for e in sii.VOCAL_EFFORTS
    ]
    assert indices == sorted(indices)
    assert indices[3] > indices[0]


def test_custom_speech_spectrum_accepted() -> None:
    result = sii.speech_intelligibility_index(np.full(18, 40.0))
    assert 0.0 <= result.sii <= 1.0
    assert result.speech_spectrum.shape == (18,)


def test_invalid_inputs_raise() -> None:
    short_threshold = np.zeros(5)
    with pytest.raises(ValueError, match="18"):
        sii.speech_intelligibility_index([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="18"):
        sii.speech_intelligibility_index("normal", noise_spectrum=[1.0, 2.0])
    with pytest.raises(ValueError, match="18"):
        sii.speech_intelligibility_index("normal", threshold=short_threshold)
    with pytest.raises(ValueError, match="vocal_effort"):
        sii.standard_speech_spectrum("whisper")


def test_result_fields_present() -> None:
    result = sii.speech_intelligibility_index("normal")
    assert result.band_audibility.shape == (18,)
    assert result.band_importance.shape == (18,)
    assert result.disturbance.shape == (18,)
    assert result.masking.shape == (18,)
    np.testing.assert_allclose(result.frequencies, sii.BAND_CENTERS)


def test_plot_returns_axes() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax = sii.speech_intelligibility_index("normal").plot()
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_standard_speech_spectra_family_matches_table3() -> None:
    """The result wrapper only stacks the bare Table 3 arrays."""
    res = sii.standard_speech_spectra()
    assert isinstance(res, sii.StandardSpeechSpectrum)
    assert res.vocal_efforts == sii.VOCAL_EFFORTS
    assert res.levels.shape == (4, 18)
    np.testing.assert_allclose(res.frequencies, sii.BAND_CENTERS)
    for i, effort in enumerate(res.vocal_efforts):
        np.testing.assert_allclose(
            res.levels[i], sii.standard_speech_spectrum(effort)
        )
    # ANSI S3.5-1997 Table 3 anchor values, in dB SPL.
    i1k = int(np.flatnonzero(np.isclose(res.frequencies, 1000.0))[0])
    i8k = int(np.flatnonzero(np.isclose(res.frequencies, 8000.0))[0])
    assert res.levels[0, i1k] == pytest.approx(25.01)          # normal, 1 kHz
    assert res.levels[2, i1k] == pytest.approx(42.16)          # loud, 1 kHz
    assert res.levels[3, i8k] == pytest.approx(20.72)          # shout, 8 kHz


def test_standard_speech_spectra_single_effort() -> None:
    res = sii.standard_speech_spectra("raised")
    assert res.vocal_efforts == ("raised",)
    assert res.levels.shape == (1, 18)
    assert res.levels[0, 0] == pytest.approx(33.81)            # raised, 160 Hz


def test_standard_speech_spectra_rejects_unknown_and_empty() -> None:
    with pytest.raises(ValueError, match="vocal_effort"):
        sii.standard_speech_spectra("whisper")
    with pytest.raises(ValueError, match="empty"):
        sii.standard_speech_spectra([])


def test_standard_speech_spectra_plot_returns_axes() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    res = sii.standard_speech_spectra()
    ax_en = res.plot()
    assert isinstance(ax_en, Axes)
    assert ax_en.get_xlabel() == "One-third-octave band [Hz]"
    assert ax_en.get_ylabel() == "Speech spectrum level [dB SPL]"
    assert "ANSI S3.5-1997" in ax_en.get_title()
    # One labelled line per vocal effort; nominal band labels on the x axis.
    assert len(ax_en.lines) == len(res.vocal_efforts)
    labels = [t.get_text() for t in ax_en.get_xticklabels()]
    assert labels[0] == "160"
    assert labels[-1] == "8k"
    plt.close("all")

    ax_es = res.plot(language="es")
    assert ax_es.get_xlabel() == "Banda de tercio de octava [Hz]"
    assert ax_es.get_ylabel() == "Nivel del espectro de voz [dB SPL]"
    plt.close("all")


def test_standard_speech_spectra_plot_forwards_kwargs_and_rejects_language() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = sii.standard_speech_spectra("normal")
    ax = res.plot(linewidth=3)
    assert any(line.get_linewidth() == 3.0 for line in ax.lines)
    plt.close("all")
    with pytest.raises(ValueError, match="Unknown language"):
        res.plot(language="xx")
    plt.close("all")


# ---------------------------------------------------------------------------
# The other three band procedures of ANSI S3.5-1997 (Tables 1, 2 and 4).
# ---------------------------------------------------------------------------


def test_sii_methods_are_the_four_procedures_of_the_standard() -> None:
    # ANSI S3.5-1997 defines four band procedures; the library exposes them
    # in the order of the standard's Tables 1 to 4.
    assert sii.SII_METHODS == (
        "critical-band", "equally-contributing", "one-third-octave", "octave",
    )
    sizes = {
        method: sii.sii_procedure(method).band_importance.size
        for method in sii.SII_METHODS
    }
    assert sizes == {
        "critical-band": 21, "equally-contributing": 17,
        "one-third-octave": 18, "octave": 6,
    }


def test_band_importance_sums_of_every_procedure() -> None:
    # Tables 1 and 4 are normalised like Table 3; Table 2 prints 0.0588 in
    # each of its 17 bands, which sums to 0.9996 rather than to one.
    assert sii.sii_procedure("critical-band").band_importance.sum() == pytest.approx(
        ANSIS3_5_CRITICAL_IMPORTANCE_SUM, abs=1e-12
    )
    assert sii.sii_procedure("octave").band_importance.sum() == pytest.approx(
        ANSIS3_5_OCTAVE_IMPORTANCE_SUM, abs=1e-12
    )
    assert sii.sii_procedure(
        "equally-contributing"
    ).band_importance.sum() == pytest.approx(ANSIS3_5_EQUAL_IMPORTANCE_SUM, abs=1e-12)


@pytest.mark.parametrize(
    ("method", "table"),
    [("critical-band", ANSIS3_5_CRITICAL_TABLE1),
     ("octave", ANSIS3_5_OCTAVE_TABLE4)],
    ids=["Table 1", "Table 4"],
)
def test_band_table_transcription(
    method: str, table: tuple[tuple[float, ...], ...]
) -> None:
    """Every cell of the shipped Tables 1 and 4, asserted directly.

    This is a transcription pin, not an independent oracle: most of these
    digits come from the same WG S3-79 reference implementation the procedures
    are anchored on (rows 1 to 6 of Table 1 are additionally corroborated by an
    independent transcription; see ``tests/reference_data/`` for the
    per-column provenance). It exists because the eight official ``.TST`` cases
    leave most of these cells inert: their strong maskers and their -10 dB
    bands mean that corrupting, for instance, the reference internal noise of
    critical band 11 by a whole decibel, or an octave band limit by 100 Hz,
    changes no published result. Without this test those cells would be
    unpinned, and a typo in them would ship.
    """
    proc = sii.sii_procedure(method)
    assert proc.frequencies.size == len(table)
    assert proc.band_edges.size == len(table) + 1
    for i, (fc, lo, hi, imp, speech, noise) in enumerate(table):
        assert proc.frequencies[i] == pytest.approx(fc), f"centre, row {i + 1}"
        assert proc.band_edges[i] == pytest.approx(lo), f"lower edge, row {i + 1}"
        assert proc.band_edges[i + 1] == pytest.approx(hi), f"upper edge, row {i + 1}"
        assert proc.band_importance[i] == pytest.approx(imp), f"Ii, row {i + 1}"
        assert proc.speech_spectrum[i] == pytest.approx(speech), f"Ui, row {i + 1}"
        assert proc.internal_noise[i] == pytest.approx(noise), f"Xi, row {i + 1}"


def test_third_octave_table_transcription() -> None:
    """Every cell of Table 3, the procedure that was already shipping.

    Table 3 has an independent digit source (the Hornsby worksheet), so unlike
    the Tables 1 and 4 pin this is a check against a third-party transcription
    as well as a guard. It is asserted for the same practical reason: a
    mutation campaign found the reference internal noise at 2500 Hz and
    3150 Hz surviving a whole-decibel corruption, because the band audibility
    there is clipped at 1 in every case the rest of the suite runs.
    """
    proc = sii.sii_procedure("one-third-octave")
    assert proc.frequencies.size == len(ANSIS3_5_THIRD_OCTAVE_TABLE3)
    for i, (fc, imp, speech, noise) in enumerate(ANSIS3_5_THIRD_OCTAVE_TABLE3):
        assert proc.frequencies[i] == pytest.approx(fc), f"centre, row {i + 1}"
        assert proc.band_importance[i] == pytest.approx(imp), f"Ii, row {i + 1}"
        assert proc.speech_spectrum[i] == pytest.approx(speech), f"Ui, row {i + 1}"
        assert proc.internal_noise[i] == pytest.approx(noise), f"Xi, row {i + 1}"


def test_octave_table_is_the_same_spectrum_level_as_table_3() -> None:
    # ANSI S3.5-1997 Table 4 tabulates Ui and Xi at all six of its centres as
    # the figures Table 3 gives at those same centres: both are spectrum
    # (per-hertz) levels, so they do not depend on the analysis bandwidth.
    # This is one of the three corroborations the limitations section leans on,
    # so it is enforced at every shared centre rather than only at 1 kHz.
    octave = sii.sii_procedure("octave")
    third = sii.sii_procedure("one-third-octave")
    for fc, speech, noise in ANSIS3_5_OCTAVE_TABLE4_SHARED:
        k = int(np.flatnonzero(np.isclose(octave.frequencies, fc))[0])
        j = int(np.flatnonzero(np.isclose(third.frequencies, fc))[0])
        assert octave.speech_spectrum[k] == pytest.approx(speech), f"Ui at {fc:g} Hz"
        assert octave.internal_noise[k] == pytest.approx(noise), f"Xi at {fc:g} Hz"
        # ... and equal to Table 3's own entry at that centre.
        assert octave.speech_spectrum[k] == pytest.approx(third.speech_spectrum[j])
        assert octave.internal_noise[k] == pytest.approx(third.internal_noise[j])


@pytest.mark.parametrize(
    ("method", "regime", "speech", "noise", "committee"),
    ANSIS3_5_WG_FLAT_CASES,
    ids=[f"{m}-{r}" for m, r, *_ in ANSIS3_5_WG_FLAT_CASES],
)
def test_sii_flat_spectrum_cases_against_committee_code(
    method: str, regime: str, speech: float, noise: float, committee: float
) -> None:
    """Flat-input cases from SII.C that bring every band's Ui and Xi into play.

    The eight official ``.TST`` cases exercise the chain but not the whole of
    each table. These do: in the quiet regime the disturbance is the reference
    internal noise in every band, and in the loud regime the clause 5.7
    level-distortion factor is below unity in every band, so each table's Xi
    and Ui column respectively moves the answer cell by cell. Expected values
    are printed by the committee's SII.C, compiled unmodified, on the stated
    flat input.
    """
    proc = sii.sii_procedure(method)
    n = proc.frequencies.size
    result = sii.speech_intelligibility_index(
        np.full(n, speech), np.full(n, noise), method=method
    )
    assert result.sii == pytest.approx(committee, abs=1e-9), regime


def test_equally_contributing_is_the_300_to_6400_hz_span_of_table_1() -> None:
    # ANSI S3.5-1997 Table 2 is critical bands 3 to 19 of Table 1 with an
    # equal importance in every band.
    critical = sii.sii_procedure("critical-band")
    equal = sii.sii_procedure("equally-contributing")
    np.testing.assert_allclose(equal.frequencies, critical.frequencies[2:19])
    np.testing.assert_allclose(equal.band_edges, critical.band_edges[2:20])
    np.testing.assert_allclose(equal.speech_spectrum, critical.speech_spectrum[2:19])
    np.testing.assert_allclose(equal.internal_noise, critical.internal_noise[2:19])
    np.testing.assert_allclose(equal.band_importance, 0.0588)
    assert equal.band_edges[0] == 300.0
    assert equal.band_edges[-1] == 6400.0


#: The six official ASA WG S3-79 test cases of the three procedures added
#: here (DevelopmentKit SOURCES/*.TST, from the WG support site sii.to). Per
#: case: the ``.TST`` file name, the ``method=`` it exercises, the equivalent
#: speech spectrum level, the equivalent noise spectrum level, the equivalent
#: hearing threshold level, an alternative band-importance function or
#: ``None``, the SII published in the kit's readme (three decimals) and the
#: value the committee's ``SII.C`` prints when compiled unmodified and run on
#: the file. The one-third-octave pair ``TO.TST``/``TO_1.TST`` keeps its own
#: two tests above.
_WG_OFFICIAL_CASES = (
    ("CB.TST", "critical-band", ANSIS3_5_WG_CB_SPEECH, ANSIS3_5_WG_CB_NOISE,
     ANSIS3_5_WG_CB_THRESHOLD, None,
     ANSIS3_5_WG_CB_SII, ANSIS3_5_WG_CB_SII_EXACT),
    ("CB_1.TST", "critical-band", ANSIS3_5_WG_CB_SPEECH, ANSIS3_5_WG_CB_NOISE,
     ANSIS3_5_WG_CB_THRESHOLD, ANSIS3_5_WG_CB1_IMPORTANCE,
     ANSIS3_5_WG_CB1_SII, ANSIS3_5_WG_CB1_SII_EXACT),
    ("ECB.TST", "equally-contributing", ANSIS3_5_WG_ECB_SPEECH,
     ANSIS3_5_WG_ECB_NOISE, ANSIS3_5_WG_ECB_THRESHOLD, None,
     ANSIS3_5_WG_ECB_SII, ANSIS3_5_WG_ECB_SII_EXACT),
    ("ECB_1.TST", "equally-contributing", ANSIS3_5_WG_ECB_SPEECH,
     ANSIS3_5_WG_ECB_NOISE, ANSIS3_5_WG_ECB_THRESHOLD,
     ANSIS3_5_WG_ECB1_IMPORTANCE,
     ANSIS3_5_WG_ECB1_SII, ANSIS3_5_WG_ECB1_SII_EXACT),
    ("OCTAVE.TST", "octave", ANSIS3_5_WG_OCTAVE_SPEECH,
     ANSIS3_5_WG_OCTAVE_NOISE, ANSIS3_5_WG_OCTAVE_THRESHOLD, None,
     ANSIS3_5_WG_OCTAVE_SII, ANSIS3_5_WG_OCTAVE_SII_EXACT),
    ("OCTAVE_1.TST", "octave", ANSIS3_5_WG_OCTAVE_SPEECH,
     ANSIS3_5_WG_OCTAVE_NOISE, ANSIS3_5_WG_OCTAVE_THRESHOLD,
     ANSIS3_5_WG_OCTAVE1_IMPORTANCE,
     ANSIS3_5_WG_OCTAVE1_SII, ANSIS3_5_WG_OCTAVE1_SII_EXACT),
)


@pytest.mark.parametrize(
    ("case", "method", "speech", "noise", "threshold", "importance",
     "published", "committee"),
    _WG_OFFICIAL_CASES,
    ids=[case[0] for case in _WG_OFFICIAL_CASES],
)
def test_sii_wg_s3_79_official_test_cases(
    case: str,
    method: str,
    speech: tuple[float, ...],
    noise: tuple[float, ...],
    threshold: tuple[float, ...],
    importance: tuple[float, ...] | None,
    published: float,
    committee: float,
) -> None:
    """Each official ``.TST`` case against its published SII and SII.C value.

    The published results are printed to three decimals in the ASA WG S3-79
    DevelopmentKit readme, hence the 5e-4 tolerance on the first assertion;
    the second pins the full precision of the committee's own C program, which
    the library reproduces to within one unit in the last place of a double.

    These six cases are not six independent confirmations. ``CB_1.TST`` and
    ``ECB_1.TST`` are the same one twice: the equally-contributing bands are
    critical bands 3 to 19, and the two alternative importance functions weight
    the same physical bands. The two extra critical bands below 300 Hz that the
    critical-band procedure adds change nothing in those weighted bands, not
    because they are quiet (their masker is the input's 10 dB noise line) but
    because their upward spread has decayed to about 1e-19 of the local masking
    energy by the time it arrives, which is far below double precision. Both
    procedures therefore return the identical 0.4104741231, published as 0.410
    for each. Counting honestly, the eight official cases give seven
    independent confirmations. The redundant one still earns its place: it
    would break the moment either procedure's band mapping went wrong.
    """
    result = sii.speech_intelligibility_index(
        np.array(speech),
        np.array(noise),
        threshold=np.array(threshold),
        method=method,
        band_importance=None if importance is None else np.array(importance),
    )
    assert result.method == method
    assert result.sii == pytest.approx(published, abs=5e-4), case
    assert result.sii == pytest.approx(committee, abs=1e-9), case
    if importance is not None:
        np.testing.assert_allclose(result.band_importance, importance)


def test_octave_procedure_has_no_spread_of_masking() -> None:
    # ANSI S3.5-1997's octave-band procedure omits the upward spread of
    # masking, so the equivalent masking spectrum level Zi is the equivalent
    # noise spectrum level Ni' itself. Checked on the official OCTAVE.TST
    # input, whose noise spans 85 dB across the six bands.
    result = sii.speech_intelligibility_index(
        np.array(ANSIS3_5_WG_OCTAVE_SPEECH),
        np.array(ANSIS3_5_WG_OCTAVE_NOISE),
        threshold=np.array(ANSIS3_5_WG_OCTAVE_THRESHOLD),
        method="octave",
    )
    np.testing.assert_allclose(result.masking, ANSIS3_5_WG_OCTAVE_NOISE)



def test_sii_annex_c1_worked_example() -> None:
    # ANSI S3.5-1997 Annex C.1 worked example (octave-band procedure), whose
    # input the WG DevelopmentKit readme prints in full. SII.C gives
    # 0.5039555062. Table C.1's own Li column, row i = 5, reads 1.00 with the
    # official WG S3-79 erratum applied (printed 0.10); the level-distortion
    # factor of clause 5.7 for that row is 0.99581, which prints as 1.00.
    result = sii.speech_intelligibility_index(
        np.array(ANSIS3_5_ANNEX_C1_SPEECH),
        np.array(ANSIS3_5_ANNEX_C1_NOISE),
        method="octave",
    )
    assert result.sii == pytest.approx(ANSIS3_5_ANNEX_C1, abs=1e-9)
    assert result.level_distortion[4] == pytest.approx(
        ANSIS3_5_ANNEX_C1_LEVEL_DISTORTION_I5, abs=5e-3
    )
    # Where the speech stays at or below the standard normal-effort spectrum
    # plus 10 dB the factor is exactly unity: band 2 has Ei' = 40 dB against
    # Ui = 34.27 dB, and band 6 has Ei' = 0 dB against Ui = 1.13 dB.
    assert result.level_distortion[1] == 1.0
    assert result.level_distortion[5] == 1.0
    assert np.all(result.level_distortion > 0.0)
    assert np.all(result.level_distortion <= 1.0)


def test_alternative_importance_reweights_the_same_audibility() -> None:
    # The alternative-importance index is the dot product of the alternative
    # Ii with the band audibility Ai of the same run, so band_importance=
    # changes only the weighting, never the audibility chain.
    common = {
        "noise_spectrum": np.array(ANSIS3_5_WG_TO_NOISE),
        "threshold": np.array(ANSIS3_5_WG_TO_THRESHOLD),
    }
    base = sii.speech_intelligibility_index(np.array(ANSIS3_5_WG_TO_SPEECH), **common)
    alt = sii.speech_intelligibility_index(
        np.array(ANSIS3_5_WG_TO_SPEECH),
        band_importance=np.array(ANSIS3_5_WG_TO1_IMPORTANCE),
        **common,
    )
    np.testing.assert_allclose(alt.band_audibility, base.band_audibility)
    assert alt.sii == pytest.approx(
        float(np.sum(np.array(ANSIS3_5_WG_TO1_IMPORTANCE) * base.band_audibility))
    )
    assert alt.sii == pytest.approx(ANSIS3_5_WG_TO1_SII_EXACT, abs=1e-9)


def test_default_method_is_unchanged_one_third_octave() -> None:
    # The default call signature keeps computing the one-third-octave
    # procedure, band for band.
    default = sii.speech_intelligibility_index("normal")
    explicit = sii.speech_intelligibility_index("normal", method="one-third-octave")
    assert default.method == "one-third-octave"
    assert default.sii == explicit.sii
    np.testing.assert_allclose(default.frequencies, sii.BAND_CENTERS)


def test_speech_intelligibility_index_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown SII method"):
        sii.speech_intelligibility_index("normal", method="bark")
    with pytest.raises(ValueError, match="Unknown SII method"):
        sii.sii_procedure("bark")


def test_speech_intelligibility_index_checks_band_count_per_method() -> None:
    # An 18-band vector is the wrong length for the 6-band octave procedure,
    # and the message names the procedure and its frequency span.
    eighteen_bands = np.full(18, 50.0)
    twentyone_bands = np.full(21, 50.0)
    eighteen_band_noise = np.full(18, 30.0)
    six_bands = np.full(6, 50.0)
    short_importance = np.full(5, 0.2)
    with pytest.raises(ValueError, match="6 octave band values"):
        sii.speech_intelligibility_index(eighteen_bands, method="octave")
    with pytest.raises(ValueError, match="21 critical-band band values"):
        sii.speech_intelligibility_index(
            twentyone_bands, eighteen_band_noise, method="critical-band"
        )
    with pytest.raises(ValueError, match="band_importance"):
        sii.speech_intelligibility_index(
            six_bands, method="octave", band_importance=short_importance
        )


def test_vocal_effort_names_outside_the_one_third_octave_procedure() -> None:
    # Tables 1, 2 and 4 are carried for normal vocal effort only.
    for method in ("critical-band", "equally-contributing", "octave"):
        proc = sii.sii_procedure(method)
        quiet = sii.speech_intelligibility_index("normal", method=method)
        np.testing.assert_allclose(quiet.speech_spectrum, proc.speech_spectrum)
        with pytest.raises(ValueError, match="normal vocal effort only"):
            sii.speech_intelligibility_index("shout", method=method)
        with pytest.raises(ValueError, match="Unknown vocal_effort"):
            sii.speech_intelligibility_index("whisper", method=method)


def test_sii_procedure_returns_copies() -> None:
    proc = sii.sii_procedure("critical-band")
    proc.band_importance[0] = 99.0
    assert sii.sii_procedure("critical-band").band_importance[0] == pytest.approx(
        0.0103
    )


def test_sii_procedure_plot_returns_axes() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    ax = sii.sii_procedure("octave").plot()
    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "Frequency [Hz]"
    assert ax.get_ylabel() == "Band importance $I_i$"
    assert "ANSI S3.5-1997" in ax.get_title()
    # The four procedures overlay on one axes, one labelled step each.
    for method in sii.SII_METHODS:
        sii.sii_procedure(method).plot(ax=ax)
    assert len(ax.lines) == 1 + len(sii.SII_METHODS)
    labels = [line.get_label() for line in ax.lines]
    assert "Critical band (21)" in labels
    plt.close("all")

    ax_es = sii.sii_procedure("critical-band").plot(language="es")
    assert ax_es.get_ylabel() == "Importancia de banda $I_i$"
    assert [line.get_label() for line in ax_es.lines] == ["Banda crítica (21)"]
    plt.close("all")
    octave = sii.sii_procedure("octave")
    with pytest.raises(ValueError, match="Unknown language"):
        octave.plot(language="xx")
    plt.close("all")


def test_sii_result_plot_for_every_procedure() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for method in sii.SII_METHODS:
        proc = sii.sii_procedure(method)
        noise = np.full(proc.frequencies.size, 20.0)
        result = sii.speech_intelligibility_index("normal", noise, method=method)
        ax = result.plot()
        assert len(ax.patches) == 2 * proc.frequencies.size
        plt.close("all")
