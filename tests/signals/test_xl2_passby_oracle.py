#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Level and loudness oracles on real NTi XL2 pass-by recordings.

Source: "Psychoacoustic indicators of pass-by road traffic noise",
A. Grangeiro de Barros and C. Vuye (University of Antwerp), Zenodo,
DOI 10.5281/zenodo.7904680, CC BY 4.0 (see ``tests/data/audio/README.md``).
The campaign is a statistical pass-by measurement per ISO 11819-1 (method
in A. Barros and C. Vuye, Proc. Mtgs. Acoust. 51, 040001 (2023),
DOI 10.1121/2.0001775): an XL2 sound level meter recorded continuously at
7.5 m from the lane centre, and the authors published, per vehicle
pass-by, indicators computed with PsySound3 over a 4 s window centred on
the LA,max peak.

What runs here, all through :func:`phonometry.io.read`:

* the meter's own calibration take, whose ``bext`` chunk declares the
  digital full scale ("0dBFS = 129.3 dBSPL"): applying that declaration
  to the file's RMS must land on the 113.7 dB the deposit publishes for
  the calibrator tone;
* five 4 s pass-by excerpts (all four vehicle categories, two sites),
  against the published LA,eq, LA,max, L10 and N50 of their rows in the
  deposit's master workbook ("Psychoacoustic indicators of pass-by road
  traffic noise.xlsx", row indices in the file names' provenance table
  below). The excerpts carry no ``bext`` of their own (they are cuts, and
  the instrument did not write them); each session's full scale is
  transcribed here from the ``bext`` of the hour-long recording the cut
  came from.

Tolerances. The published levels are PsySound3's computation, with no
stated uncertainty; recomputing them through this library agrees to
within 0.25 dB with a small systematic offset, so LA,eq, LA,max and L10
are asserted to **0.3 dB**. N50 is the median of the ISO 532-1 method B
time-varying loudness (the thesis behind the dataset states method B,
free field, frontal incidence), asserted to **5 %** -- ISO 532-1 itself
tolerates 5 % against its own reference on arbitrary signals. Two
published columns are deliberately **not** asserted: L90 and Delta-L both
depend on where the Fast envelope starts relative to the 4 s window
(differences of several dB for an integrator seeded outside the excerpt),
so they would pin a windowing convention, not a measurement.

The published workbook values are transcribed at three decimals; the
rounding is three orders of magnitude below the tolerances.
"""

from __future__ import annotations

import re

import numpy as np
import oracle_data
import pytest

from phonometry import io, signals
from phonometry.filters import time_weighting, weighting_filter
from phonometry.psychoacoustics import loudness_zwicker

_DATA = oracle_data.DATA / "audio" / "xl2"

#: Reference sound pressure, Pa.
_P0 = 2e-5

#: Published level of the calibration recording (deposit file name and
#: description: "pure-tone 113.7 dB calibration file recorded from the
#: sound level meter and microphone used in the measurements").
_PUBLISHED_CALIBRATION_SPL = 113.7

#: Pass-by excerpts: file name, digital full scale of the source session
#: (0 dBFS in dB SPL, from the bext Description of the hour-long WAV each
#: excerpt was cut from), and the published row of the master workbook:
#: LA,eq, LA,max, L10 (dB) and N50 (sone).
_PASS_BYS = [
    ("passby_0566_van_60kmh.wav", 129.4, 73.905, 78.196, 77.358, 27.445),
    ("passby_0571_van_90kmh.wav", 129.4, 76.928, 82.154, 81.522, 25.821),
    ("passby_0668_passenger_car_74kmh.wav", 129.4, 75.546, 80.353, 79.525, 25.999),
    ("passby_0960_heavy_dual_axle_66kmh.wav", 129.4, 80.600, 85.993, 85.114, 38.271),
    ("passby_1558_heavy_multi_axle_62kmh.wav", 129.5, 77.537, 81.616, 81.090, 33.002),
]

_LEVEL_TOL_DB = 0.3
_N50_TOL = 0.05


def _calibration_factor(full_scale_spl: float) -> float:
    """Digital-to-pascal multiplier from a '0 dBFS = x dB SPL' declaration."""
    return _P0 * 10 ** (full_scale_spl / 20)


def test_calibration_take_carries_the_instrument_bext() -> None:
    """The meter wrote the file: its bext names the XL2 and the full scale."""
    meta = io.info(_DATA / "calibration_113_7dB.wav")
    assert meta.container == "WAV"
    assert meta.format_name == "PCM"
    assert meta.bit_depth == 24
    assert meta.fs == 48000
    assert meta.channels == 1
    bext = meta.bext
    assert bext is not None
    assert bext.originator == "NTi Audio XL2 A2A-17367-E0"
    assert bext.description.startswith("0dBFS = 129.3 dBSPL")
    assert bext.origination_date == "2022-09-06"
    assert bext.coding_history == "PCM: mono, 24 bits, 48 kHz"


def test_calibration_chain_reproduces_the_published_113_7_db() -> None:
    """bext full scale + file RMS = the published calibrator level.

    The full scale is parsed out of the file's own bext Description, so
    this closes the whole chain: instrument WAV -> reader -> declared
    calibration -> published value. 113.7 dB is published to one decimal;
    the assertion allows half that last digit.
    """
    sig = io.read(_DATA / "calibration_113_7dB.wav")
    assert sig.provenance is not None
    match = re.match(r"0dBFS = (\d+\.\d) dBSPL", sig.provenance.description)
    assert match is not None
    full_scale_spl = float(match.group(1))
    assert full_scale_spl == 129.3

    spl = float(
        signals.leq(sig, calibration_factor=_calibration_factor(full_scale_spl))
    )
    assert spl == pytest.approx(_PUBLISHED_CALIBRATION_SPL, abs=0.05)
    # Same number straight from dB full scale, by construction.
    dbfs = float(signals.leq(sig, dbfs=True))
    assert dbfs + full_scale_spl == pytest.approx(spl, abs=1e-9)


@pytest.mark.parametrize(
    ("name", "full_scale_spl", "laeq_pub", "lamax_pub", "l10_pub", "n50_pub"),
    _PASS_BYS,
    ids=[row[0].removeprefix("passby_").removesuffix(".wav") for row in _PASS_BYS],
)
def test_passby_levels_and_loudness_match_the_published_row(
    name: str,
    full_scale_spl: float,
    laeq_pub: float,
    lamax_pub: float,
    l10_pub: float,
    n50_pub: float,
) -> None:
    sig = io.read(_DATA / name)
    assert sig.fs == 48000
    assert sig.n_channels == 1
    assert sig.duration == 4.0
    factor = _calibration_factor(full_scale_spl)
    x = np.asarray(sig)

    laeq = float(signals.laeq(sig, calibration_factor=factor))
    assert laeq == pytest.approx(laeq_pub, abs=_LEVEL_TOL_DB)

    # LA,max with Fast time weighting. The workbook does not name the
    # weighting; Fast reproduces every published LA,max within 0.25 dB
    # while Slow misses by several dB, which settles it.
    envelope = time_weighting(weighting_filter(x, sig.fs, "A"), sig.fs, mode="fast")
    lafmax = 10 * np.log10(float(np.max(envelope)) * factor**2 / _P0**2)
    assert lafmax == pytest.approx(lamax_pub, abs=_LEVEL_TOL_DB)

    l10 = signals.ln_levels(
        sig, n=(10,), mode="fast", weighting="A", calibration_factor=factor
    )[10]
    assert float(l10) == pytest.approx(l10_pub, abs=_LEVEL_TOL_DB)

    # N50: the value exceeded 50 % of the time, i.e. the median of the
    # time-varying total loudness (ISO 532-1:2017 method B).
    result = loudness_zwicker(x, sig.fs, calibration_factor=factor)
    assert result.loudness_vs_time is not None
    n50 = float(np.percentile(result.loudness_vs_time, 50))
    assert n50 == pytest.approx(n50_pub, rel=_N50_TOL)
