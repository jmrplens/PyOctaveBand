#  Copyright (c) 2026. Jose M. Requena-Plens
"""The EBU loudness test set's authentic-programme cases (Tech 3341 7-8,
Tech 3342 5-6), which use programme material that cannot be synthesized.

The audio itself can never be committed. The EBU terms of use for its test
sequences state that you "may not copy, modify, merge, publish, distribute,
sublicense, and/or sell copies", and cases 7/8 and 5/6 are licensed feature
film excerpts on top of that (see ``tests/data/broadcast/README.md``). What is
committed instead is a **derived measurement series**: the per-block loudness
envelopes measured from those two files, which are our own measurements and
cannot reconstruct the audio.

So this module runs on two levels:

* **always, including CI** - the committed series in
  ``tests/data/broadcast/ebu_programme_block_loudness.npz`` drive the gating
  and loudness-range stages and must reproduce the EBU-published targets
  (integrated -23,0 LUFS; LRA 5 LU for NLR and 15 LU for WLR). These four
  assertions never skip;
* **where a full local copy of the set exists** - resolved by
  ``tests/oracle_data.py`` (``$EBU_LOUDNESS_TEST_SET`` first, then
  ``tests/data-local/ebu-loudness-test-set/``) - the same targets are asserted
  end to end from the WAV files, at the same tolerances. These six cases read
  audio that cannot be committed, so they skip everywhere else, CI included,
  and the skip reason says why.

What the committed series therefore cannot assert: the stages *upstream* of
the block envelope, i.e. 24-bit WAV decoding, the BS.1770 K-weighting front
end, the 400 ms / 100 ms block segmentation and the channel weighting, on
authentic programme material. Those stages are covered in CI by the
synthesizable EBU cases in ``test_program_loudness.py``, which build their
signals from the specifications; only their behaviour on real programme
material is local-only. No tolerance differs between the two levels.
"""

import pathlib

import numpy as np
import oracle_data
import pytest
from scipy.io import wavfile

from phonometry import broadcast
from phonometry.broadcast import loudness_range
from phonometry.broadcast.program_loudness import _integrated_from_blocks

_NLR = "seq-3341-7_seq-3342-5-24bit.wav"
_WLR = "seq-3341-2011-8_seq-3342-6-24bit-v02.wav"

#: The 100 ms gate hop is every tenth reading of the 10 ms momentary series.
_GATE_STRIDE = 10

# Committed derived oracle (always present).
_NPZ = oracle_data.DATA / "broadcast" / "ebu_programme_block_loudness.npz"
with np.load(_NPZ) as _f:
    _SERIES = {key: _f[key] for key in _f.files}

# Full local copy of the test set, when there is one.
_SET = oracle_data.resolve(oracle_data.EBU_LOUDNESS_SET)
_AUDIO_ROOT = _SET.path
_requires_audio = pytest.mark.skipif(
    _AUDIO_ROOT is None,
    reason="EBU loudness test set absent (its licence forbids redistribution; "
    "drop a copy in tests/data-local/ebu-loudness-test-set/ or point "
    "EBU_LOUDNESS_TEST_SET at it). The committed block-loudness series cover "
    "the gating and loudness-range stages here.",
)


def _load(name: str) -> tuple[np.ndarray, float]:
    """Read a test-set WAV as float64 ``[channels, samples]`` in [-1, 1)."""
    assert _AUDIO_ROOT is not None
    fs, x = wavfile.read(pathlib.Path(_AUDIO_ROOT) / name)
    assert x.ndim == 2, f"{name}: expected a stereo programme file"
    if x.dtype == np.int32:  # 24-bit PCM arrives left-justified in int32
        y = x.astype(np.float64) / 2147483648.0
    elif x.dtype == np.int16:
        y = x.astype(np.float64) / 32768.0
    else:
        y = np.asarray(x, dtype=np.float64)
    # A silent file means a corrupt download; fail loudly instead of gating
    # everything away and "passing" on noise.
    assert float(np.sqrt(np.mean(y**2))) > 1e-4, f"{name}: silent file"
    return y.T, float(fs)


# ---------------------------------------------------------------------------
# Committed series: gating and loudness range (runs everywhere)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "case"),
    [("nlr", "Tech 3341 case 7 (NLR)"), ("wlr", "Tech 3341 case 8 (WLR)")],
)
def test_gated_integrated_loudness_from_series(key: str, case: str) -> None:
    """The BS.1770-5 two-stage gate over the committed 400 ms momentary
    series must land on the EBU-published -23,0 LUFS."""
    integrated, _ = _integrated_from_blocks(_SERIES[f"{key}_momentary"])
    assert integrated == pytest.approx(-23.0, abs=0.1), case


@pytest.mark.parametrize(
    ("key", "expected", "case"),
    [
        ("nlr", 5.0, "Tech 3342 case 5 (NLR)"),
        ("wlr", 15.0, "Tech 3342 case 6 (WLR)"),
    ],
)
def test_loudness_range_from_series(key: str, expected: float, case: str) -> None:
    """EBU Tech 3342 loudness range over the committed 3 s short-term series."""
    assert loudness_range(_SERIES[f"{key}_short_term"]) == pytest.approx(
        expected, abs=1.0
    ), case


# ---------------------------------------------------------------------------
# Full chain from the audio (runs where a local copy of the set exists)
# ---------------------------------------------------------------------------


@_requires_audio
@pytest.mark.parametrize(
    ("name", "case"),
    [(_NLR, "Tech 3341 case 7 (NLR)"), (_WLR, "Tech 3341 case 8 (WLR)")],
)
def test_authentic_programme_integrated_loudness(name: str, case: str) -> None:
    x, fs = _load(name)
    res = broadcast.program_loudness(x, fs)
    assert res.integrated == pytest.approx(-23.0, abs=0.1), case


@_requires_audio
@pytest.mark.parametrize(
    ("name", "expected", "case"),
    [
        (_NLR, 5.0, "Tech 3342 case 5 (NLR)"),
        (_WLR, 15.0, "Tech 3342 case 6 (WLR)"),
    ],
)
def test_authentic_programme_loudness_range(
    name: str, expected: float, case: str
) -> None:
    x, fs = _load(name)
    res = broadcast.program_loudness(x, fs)
    assert res.loudness_range == pytest.approx(expected, abs=1.0), case


@_requires_audio
@pytest.mark.parametrize("key", ["nlr", "wlr"])
def test_committed_series_match_the_audio(key: str) -> None:
    """The committed series must still be what the full chain measures.

    This is the join between the two levels: where the audio is available,
    re-measuring it has to reproduce the committed envelopes block for block,
    so the derived oracle cannot silently drift away from the material it
    claims to represent.
    """
    x, fs = _load(_NLR if key == "nlr" else _WLR)
    res = broadcast.program_loudness(x, fs)
    # The committed momentary envelope is the BS.1770 gate input, i.e. the
    # 400 ms blocks at the 100 ms hop, which is every tenth reading of the
    # finer 10 ms display series; the short-term envelope is already at the
    # 100 ms hop. Both are stored rounded to 0,0001 LU.
    np.testing.assert_allclose(
        res.momentary[::_GATE_STRIDE], _SERIES[f"{key}_momentary"], atol=5e-4
    )
    np.testing.assert_allclose(
        res.short_term, _SERIES[f"{key}_short_term"], atol=5e-4
    )
