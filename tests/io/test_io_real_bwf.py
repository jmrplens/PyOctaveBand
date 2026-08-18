#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The bext transcription oracle: a real Sound Devices 702T BWF.

Every other bext test in this suite reads chunks this repository forged, so
a bug in the writer and the same bug in the reader would cancel out.
``A101_3.WAV`` breaks that circle: a field recorder wrote its ``bext``
version 1 and its ``iXML`` in 2018, long before this library existed, and
the expected values below are transcribed from what that machine wrote --
not from anything this code base produced.

Source and licence: the wavinfo project (github.com/iluvcapra/wavinfo),
MIT License, (c) Jamie Hardt; see ``tests/data/audio/README.md``. The
wavinfo documentation ("Broadcast WAV Metadata") prints this recorder's
bext output for the sibling take ``A101_1.WAV`` of the same session, with
the same Originator and CodingHistory shape; the values for this take were
cross-read with an independent reader (ffprobe) at import time, field by
field. Every comparison is exact equality: a transcription oracle that
needs a tolerance is not transcribing.
"""

from __future__ import annotations

import numpy as np
import oracle_data

from phonometry import io

_PATH = oracle_data.DATA / "audio" / "sounddevices" / "A101_3.WAV"

#: The Description field the 702T wrote: Sound Devices' sTAG=value lines,
#: CRLF-separated, one per metadata slot of the recorder's slate.
_DESCRIPTION = (
    "sSPEED=023.976-ND\r\n"
    "sTAKE=3\r\n"
    "sUBITS=$12311803\r\n"
    "sSWVER=2.67\r\n"
    "sPROJECT=BMH\r\n"
    "sSCENE=A101\r\n"
    "sFILENAME=A101_3.WAV\r\n"
    "sTAPE=18Y12M31\r\n"
    "sTRK1=MKH516 A\r\n"
    "sTRK2=Boom\r\n"
    "sNOTE=\r\n"
)

_ORIGINATOR = "Sound Dev: 702T S#GR1112089007"
#: EBU R99-style unique identifier: country, manufacturer, serial, then the
#: origination time and a random tail, assembled by the recorder.
_ORIGINATOR_REFERENCE = "USSDVGR1112089007124014008228301"
_CODING_HISTORY = "A=PCM,F=48000,W=24,M=stereo,R=48000,T=2 Ch"
#: Samples since midnight at 48 kHz. The recorder stamps its timecode
#: counter here; it need not equal the wall-clock OriginationTime.
_TIME_REFERENCE = 2191661476


def test_info_transcribes_the_702t_bext_field_by_field() -> None:
    meta = io.info(_PATH)
    assert meta.container == "WAV"
    assert meta.format_name == "PCM"
    assert meta.bit_depth == 24
    assert meta.fs == 48000
    assert meta.channels == 2
    assert meta.frames == 48044
    assert not meta.lossy

    bext = meta.bext
    assert bext is not None
    assert bext.description == _DESCRIPTION
    assert bext.originator == _ORIGINATOR
    assert bext.originator_reference == _ORIGINATOR_REFERENCE
    assert bext.origination_date == "2018-12-31"
    assert bext.origination_time == "12:40:06"
    assert bext.time_reference == _TIME_REFERENCE
    assert bext.version == 1
    assert bext.coding_history == _CODING_HISTORY
    # Version 1 defines the UMID (this machine filled it with zeros) but
    # not the v2 loudness fields, which must come back as absent rather
    # than as a decoded reserved area.
    assert bext.umid == bytes(64)
    assert bext.loudness_value is None
    assert bext.loudness_range is None
    assert bext.max_true_peak_level is None
    assert bext.max_momentary_loudness is None
    assert bext.max_short_term_loudness is None


def test_info_reports_the_ixml_chunk() -> None:
    assert io.info(_PATH).has_ixml


def test_read_returns_the_recording_with_its_provenance() -> None:
    sig = io.read(_PATH)
    x = np.asarray(sig)
    assert x.shape == (2, 48044)
    assert sig.fs == 48000
    # A silent file would mean a corrupt import; this is a real recording.
    assert float(np.sqrt(np.mean(x**2))) > 1e-3
    assert float(np.max(np.abs(x))) < 1.0
    # The provenance riding on the Signal is the same transcription.
    assert sig.provenance is not None
    assert sig.provenance.originator == _ORIGINATOR
    assert sig.provenance.time_reference == _TIME_REFERENCE
