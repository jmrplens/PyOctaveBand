#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
End-to-end STIPA verification against the IEC 60268-16:2020 (rev 5)
certified test bench signals from stipa.info (Embedded Acoustics BV).

The bench is 49 mono 48 kHz WAV files, 133 MB of PCM that does not compress
as audio. ``tests/data/stipa/`` carries a **27-signal extract** of it, stored
losslessly (see that folder's ``README.md`` for the selection, the encoding
and the licence), so all five verification suites run everywhere including CI;
none of their assertions skips there. Only the inventory guard on the full
download does, and it says so.
A full local copy wins when there is one, resolved by ``tests/oracle_data.py``
(``$STIPA_VERIFICATION_DATA`` first, then ``tests/data-local/``); every
parametrized case then runs, otherwise the parameter lists shrink to the
committed signals. The run header names the copy actually used.

Expected values come from the accompanying signal description (Jan Verhave,
Embedded Acoustics, v1.0 June 2020) and from the filenames; only the WAVs,
filenames and that description were used (the bundled reference .m sources
were consulted solely to resolve the envelope convention of C.3.2: the
signals encode a STIPA channel of MTF = m, i.e. envelope index 0,55 m, so an
analyzer normalizing by 0,55 must read back m itself).

Suites:

- Annex C.3.2 - direct-method modulation depth: sine-carrier STIPA
  signals at m = 0,0 .. 1,0; the analyzer must recover m per band and the
  published m <-> STI staircase.
- Annex C.3.3 - indirect-method modulation depth: exponentially decayed
  sine carriers (RT60 = 0,125 .. 8 s) against the closed-form Schroeder
  MTF m(F) = 1/sqrt(1 + (2 pi F T/13,8)^2).
- Annex C.4.2 - filter-bank slope: modulated carrier plus an unmodulated
  adjacent-octave tone 41 dB louder; normative criterion m >= 0,5 in the
  observed band (needs > 41 dB effective slope, steeper than class 1).
- Annex A.2.2 - weighting/redundancy factors: modulated octave-band
  pairs; STI = alpha_k + alpha_{k+1} - beta_k.
- Annex A.3.1.2 - filter-bank phase distortion: two sine carriers per
  band at the half-octave edges fc*2^(+/-1/4); normative criterion
  |STI bias| < 0,01 over TI = 0,1 .. 0,9.

Measured worst-case deviations with this implementation (zero-phase
analysis bank), on which the tolerances are based:
C.3.2 |dSTI| 0,0031 / per-m 0,004; C.3.3 |dSTI| 0,0002 / per-m 0,018;
C.4.2 min m 0,937; A.2.2 |dSTI| 0,0002 vs the exact alpha/beta identity;
A.3.1.2 worst bias -0,0029.

With the committed extract alone, C.3.3 and C.4.2 are covered in full and
the other three suites keep their worst-case point (C.3.2 m = 0,9 and the
m = 0 / m = 1 ends, A.3.1.2 TI = 0,9, A.2.2 the outer band pairs). The
intermediate points of C.3.2, A.2.2 and A.3.1.2 - i.e. the shape of the
staircase between its ends, and the phase bias below TI = 0,9 - can only be
asserted where the full bench is present. No tolerance is relaxed either way.
"""

import hashlib
import json
import pathlib
import warnings
import zipfile
from collections.abc import Mapping
from typing import Any

import numpy as np
import oracle_data
import pytest
from scipy.io import wavfile

from phonometry import speech
from phonometry.speech.sti import _MOD_FREQS, _NUM_BANDS, _sti_from_mtf

FS = 48000
_BANDS = (125, 250, 500, 1000, 2000, 4000, 8000)

# Committed oracle: a lossless extract of the certified bench, always present.
# It is the fallback CI takes (see tests/data/stipa/README.md).
_EXTRACT = oracle_data.DATA / "stipa" / "stipa_certified_extract.zip"
with zipfile.ZipFile(_EXTRACT) as _archive:
    _MANIFEST = {
        entry["path"]: entry
        for entry in json.loads(_archive.read("manifest.json"))["signals"]
    }

# Full bench (49 signals) when a local copy is available: $STIPA_VERIFICATION_DATA
# first, then tests/data-local/stipa-verification/ (see tests/oracle_data.py).
_BENCH = oracle_data.resolve(oracle_data.STIPA_BENCH)
FULL_BENCH = _BENCH.path
_FULL_BENCH_PRESENT = _BENCH.is_full_set


def _decode(relative: str) -> np.ndarray:
    """Reconstruct an extracted signal: ``order`` cumulative sums of the
    stored difference, checked against the SHA-256 of the original samples."""
    entry = _MANIFEST[relative]
    with zipfile.ZipFile(_EXTRACT) as archive:
        blob = archive.read(entry["member"])
    d = np.frombuffer(blob, dtype="<i4").astype(np.int64)
    for _ in range(entry["order"]):
        d = np.cumsum(d)
    pcm = d.astype(np.int16)
    digest = hashlib.sha256(pcm.astype("<i2").tobytes()).hexdigest()
    assert digest == entry["sha256"], f"{relative}: extract is corrupt"
    assert entry["fs"] == FS, f"{relative}: expected 48 kHz, got {entry['fs']}"
    return pcm.astype(np.float64) / 32768.0


def _read_wav(path: pathlib.Path) -> np.ndarray:
    """Read a bench WAV as float64 in [-1, 1) at 48 kHz mono."""
    fs, x = wavfile.read(path)
    assert fs == FS, f"{path.name}: expected 48 kHz, got {fs}"
    assert x.ndim == 1, f"{path.name}: expected mono"
    y = x.astype(np.float64) / 32768.0 if x.dtype == np.int16 else np.asarray(x, float)
    # Every bench signal carries at least an unmodulated carrier; a silent
    # file (corrupt download) must fail loudly, not read as m = 0.
    assert float(np.sqrt(np.mean(y**2))) > 1e-3, f"{path.name}: silent file"
    return y


def _load(relative: str) -> np.ndarray:
    """The bench signal ``relative``, from the full copy when there is one."""
    if FULL_BENCH is None:
        return _decode(relative)
    path = FULL_BENCH / relative
    if not path.is_file():
        # All five annex directories were there, so the copy was taken for
        # the full bench, but a signal inside one of them is missing. Say so
        # rather than dying on a bare FileNotFoundError - and do not quietly
        # substitute the committed extract, which would hide the gap.
        raise AssertionError(
            f"{relative} is missing from the local bench at {FULL_BENCH}. "
            "The download is incomplete: remove or complete it, or point "
            "STIPA_VERIFICATION_DATA elsewhere, to fall back to the "
            "committed extract."
        )
    return _read_wav(path)


def _available(cases: Mapping[Any, str]) -> list[Any]:
    """Sorted keys of ``cases`` whose signal can be read here."""
    if _FULL_BENCH_PRESENT:
        return sorted(cases)
    return sorted(key for key, path in cases.items() if path in _MANIFEST)


def _stipa_quiet(x: np.ndarray) -> speech.STIResult:
    """stipa() with the expected verification-bench warnings silenced
    (dead bands and junk m > 1,3 in the two-band C.4.2 signals)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return speech.stipa(x, FS)


# ---------------------------------------------------------------------------
# Annex C.3.2 - direct-method modulation depth (sine carriers, m = 0 .. 1)
# ---------------------------------------------------------------------------

# m <-> STI staircase from the signal description (Table with related m,
# SNR and STI values); m and TI = (10 lg(m/(1-m)) + 15)/30 in 0,1 steps.
_C32_EXPECTED = {
    0.0: 0.00,
    0.1: 0.18,
    0.2: 0.30,
    0.3: 0.38,
    0.4: 0.44,
    0.5: 0.50,
    0.6: 0.56,
    0.7: 0.62,
    0.8: 0.70,
    0.9: 0.82,
    1.0: 1.00,
}
_C32_FILES = {m: f"Annex C.3.2/STIPA-sinecarrier-M={m:g}.wav" for m in _C32_EXPECTED}


@pytest.mark.parametrize("m", _available(_C32_FILES))
def test_c32_direct_method_modulation_depth(m: float) -> None:
    res = speech.stipa(_load(_C32_FILES[m]), FS)
    # Published staircase (worst measured |dSTI| = 0,0031 -> tol 0,01).
    assert res.sti == pytest.approx(_C32_EXPECTED[m], abs=0.01)
    # MTF extraction: every band/modulation-frequency cell must read back
    # the encoded m (worst measured deviation 0,004 -> tol 0,01). At
    # m = 1,0 the cells are clipped to at most 1.
    assert res.mtf.shape == (7, 2)
    np.testing.assert_allclose(res.mtf, m, atol=0.01)


# ---------------------------------------------------------------------------
# Annex C.3.3 - indirect-method modulation depth (exponential decays)
# ---------------------------------------------------------------------------

_C33_FILES = {
    rt60: f"Annex C.3.3/STIPA-expdecay-RT60={rt60:g}.wav"
    for rt60 in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
}


def _schroeder_m(rt60: float) -> np.ndarray:
    """Closed-form MTF of an exponential intensity decay of 60 dB in RT60:
    I(t) ~ e^(-a t), a = 6 ln(10)/RT60, m(F) = 1/sqrt(1 + (2 pi F/a)^2)."""
    a = 6.0 * np.log(10.0) / rt60
    return np.asarray(1.0 / np.sqrt(1.0 + (2.0 * np.pi * _MOD_FREQS / a) ** 2))


@pytest.mark.parametrize("rt60", _available(_C33_FILES))
def test_c33_indirect_method_exponential_decay(rt60: float) -> None:
    res = speech.sti_from_impulse_response(_load(_C33_FILES[rt60]), FS)
    m_expected = _schroeder_m(rt60)
    # Per-band, per-modulation-frequency MTF against the closed form
    # (worst measured deviation 0,018 at RT60 = 0,125 s -> tol 0,03).
    np.testing.assert_allclose(res.mtf, np.tile(m_expected, (_NUM_BANDS, 1)), atol=0.03)
    # STI derived from the closed-form MTF through the standard TI chain
    # (worst measured |dSTI| = 0,0002 -> tol 0,005).
    sti_expected = _sti_from_mtf(np.tile(m_expected, (_NUM_BANDS, 1))).sti
    assert res.sti == pytest.approx(sti_expected, abs=0.005)


# ---------------------------------------------------------------------------
# Annex C.4.2 - filter-bank slope (m >= 0,5 with a +41 dB adjacent tone)
# ---------------------------------------------------------------------------

_C42_FILES = {
    (slope, band): f"Annex C.4.2/Filtertest_{slope} {band}.wav"
    for slope in ("lowslope", "highslope")
    for band in _BANDS
}


@pytest.mark.parametrize("case", _available(_C42_FILES))
def test_c42_filter_slope(case: tuple[str, int]) -> None:
    slope, band = case
    res = _stipa_quiet(_load(_C42_FILES[case]))
    m_observed = res.mtf[_BANDS.index(band)]
    # Normative pass criterion of the bench: m >= 0,5 in the observed
    # band (an unmodulated tone one octave away, 41 dB louder, must not
    # leak enough to halve the modulation depth).
    assert np.all(m_observed >= 0.5), (
        f"m = {m_observed} in the {band} Hz band ({slope})"
    )
    # Regression lock well above the criterion: the zero-phase bank
    # achieves m >= 0,937 on all 14 signals.
    assert np.all(m_observed >= 0.85)


# ---------------------------------------------------------------------------
# Annex A.2.2 - weighting and redundancy factors (octave-band pairs)
# ---------------------------------------------------------------------------

# Exact Ed.5 Table A.1 identity alpha_k + alpha_{k+1} - beta_k; the
# filename STI values are these rounded to two decimals.
_A22_EXPECTED = {
    (125, 250): 0.127,
    (250, 500): 0.279,
    (500, 1000): 0.398,
    (1000, 2000): 0.531,
    (2000, 4000): 0.486,
    (4000, 8000): 0.302,
}
_A22_FILES = {
    pair: (
        "Annex A.2.2 - weight factor test/"
        f"STIPA-sine-pair[{pair[0]}+{pair[1]}]STI={round(sti, 2):g}.wav"
    )
    for pair, sti in _A22_EXPECTED.items()
}


@pytest.mark.parametrize("pair", _available(_A22_FILES))
def test_a22_weighting_factor_pairs(pair: tuple[int, int]) -> None:
    res = speech.stipa(_load(_A22_FILES[pair]), FS)
    # Worst measured deviation vs the exact identity: 0,0002 (the visible
    # 0,004 vs the filename is its 2-decimal rounding) -> tol 0,005.
    assert res.sti == pytest.approx(_A22_EXPECTED[pair], abs=0.005)


# ---------------------------------------------------------------------------
# Annex A.3.1.2 - filter-bank phase distortion (half-octave edge carriers)
# ---------------------------------------------------------------------------

# TI -> encoded m from the filenames (m = 1/(1 + 10^(-SNR/10)),
# SNR = 30 TI - 15; endpoints clipped to 0 and 1 by the bench).
_A312_NAMES = {
    0.0: "STIPA-sine-edge-carriers-TI=0[m=0].wav",
    0.1: "STIPA-sine-edge-carriers-TI=0.1[m=0.059351].wav",
    0.2: "STIPA-sine-edge-carriers-TI=0.2[m=0.11182].wav",
    0.3: "STIPA-sine-edge-carriers-TI=0.3[m=0.20076].wav",
    0.4: "STIPA-sine-edge-carriers-TI=0.4[m=0.33386].wav",
    0.5: "STIPA-sine-edge-carriers-TI=0.5[m=0.5].wav",
    0.6: "STIPA-sine-edge-carriers-TI=0.6[m=0.66614].wav",
    0.7: "STIPA-sine-edge-carriers-TI=0.7[m=0.79924].wav",
    0.8: "STIPA-sine-edge-carriers-TI=0.8[m=0.88818].wav",
    0.9: "STIPA-sine-edge-carriers-TI=0.9[m=0.94065].wav",
    1.0: "STIPA-sine-edge-carriers-TI=1[m=1].wav",
}
_A312_FILES = {
    ti: f"Annex A.3.1.2 - filter bank phase test/{name}"
    for ti, name in _A312_NAMES.items()
}


@pytest.mark.parametrize("ti", _available(_A312_FILES))
def test_a312_filter_bank_phase(ti: float) -> None:
    res = speech.stipa(_load(_A312_FILES[ti]), FS)
    # Normative criterion: |STI bias| < 0,01 over TI = 0,1 .. 0,9; the
    # endpoints (clipped m) hold trivially and are asserted at the same
    # tolerance. Worst measured bias with the zero-phase bank: -0,0029.
    assert res.sti == pytest.approx(ti, abs=0.01)


# ---------------------------------------------------------------------------
# Inventory guards
# ---------------------------------------------------------------------------

#: The exact 27 signals the extract must hold. Pinned name by name, not by
#: count: swapping a worst-case point for an easier one of the same suite
#: would keep the counts right and quietly weaken the committed coverage.
_COMMITTED = (
    frozenset(_C32_FILES[m] for m in (0.0, 0.9, 1.0))
    | frozenset(_C33_FILES.values())
    | frozenset(_C42_FILES.values())
    | frozenset(_A22_FILES[pair] for pair in ((125, 250), (1000, 2000)))
    | frozenset({_A312_FILES[0.9]})
)


def test_committed_extract_inventory() -> None:
    """The committed extract must hold exactly the 27 declared signals and
    every one of them must decode to the digest of the original samples."""
    assert set(_MANIFEST) == set(_COMMITTED), (
        "the extract no longer matches the documented selection "
        "(see tests/data/stipa/README.md)"
    )
    assert len(_COMMITTED) == 27, f"expected 27 signals, found {len(_COMMITTED)}"
    for relative in _MANIFEST:
        # _decode() verifies the SHA-256 and the sample rate itself.
        assert _decode(relative).size > 0


@pytest.mark.skipif(not _FULL_BENCH_PRESENT, reason="full stipa.info bench absent")
def test_certified_bench_inventory() -> None:
    """Guard against a silent partial download: 11 + 7 + 14 + 6 + 11."""
    assert FULL_BENCH is not None
    counts = {
        "Annex C.3.2": 11,
        "Annex C.3.3": 7,
        "Annex C.4.2": 14,
        "Annex A.2.2 - weight factor test": 6,
        "Annex A.3.1.2 - filter bank phase test": 11,
    }
    for sub, n in counts.items():
        found = len(list((FULL_BENCH / sub).glob("*.wav")))
        assert found == n, f"{sub}: expected {n} WAVs, found {found}"
