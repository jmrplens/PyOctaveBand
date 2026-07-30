#  Copyright (c) 2026. Jose M. Requena-Plens
"""Tests for :mod:`phonometry.hearing.sii` (Speech Intelligibility Index, ANSI S3.5-1997).

The one-third-octave-band procedure is validated against the reference
implementation of ASA Working Group S3-79, the committee that maintains
ANSI S3.5 (``SII.C`` and its official test-input files ``TO.TST`` and
``TO_1.TST`` with published results, from the WG support site sii.to),
against the standard's own tabulated constants and its Annex C.2 worked
example (with the official errata applied), and against the independent
Hornsby worksheet and R CRAN "SII" implementations where they overlap.
See ``tests/reference_data.py`` for the provenance of every constant.
"""

from __future__ import annotations

import numpy as np
import pytest
from reference_data import (
    ANSIS3_5_ANNEX_C2,
    ANSIS3_5_ANNEX_C2_MASKING,
    ANSIS3_5_BAND_IMPORTANCE_SUM,
    ANSIS3_5_DISTURBANCE_5000HZ,
    ANSIS3_5_LOUD_1KHZ,
    ANSIS3_5_NOISE_PLUS_LOSS,
    ANSIS3_5_STANDARD_QUIET,
    ANSIS3_5_WG_TO1_IMPORTANCE,
    ANSIS3_5_WG_TO1_SII,
    ANSIS3_5_WG_TO1_SII_EXACT,
    ANSIS3_5_WG_TO_NOISE,
    ANSIS3_5_WG_TO_SII,
    ANSIS3_5_WG_TO_SII_EXACT,
    ANSIS3_5_WG_TO_SPEECH,
    ANSIS3_5_WG_TO_THRESHOLD,
)

from phonometry.hearing import sii


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
    with pytest.raises(ValueError, match="18"):
        sii.speech_intelligibility_index([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="18"):
        sii.speech_intelligibility_index("normal", noise_spectrum=[1.0, 2.0])
    with pytest.raises(ValueError, match="18"):
        sii.speech_intelligibility_index("normal", threshold=np.zeros(5))
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
    assert labels[0] == "160" and labels[-1] == "8k"
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
