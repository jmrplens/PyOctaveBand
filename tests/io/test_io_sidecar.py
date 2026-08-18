#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the calibration sidecar: schema v1, auto-application, refusals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phonometry.io import (
    CalibrationSidecar,
    Signal,
    read,
    read_sidecar,
    sidecar_path,
    write,
    write_sidecar,
)
from phonometry.io._sidecar import SIDECAR_SCHEMA, SIDECAR_VERSION

FS = 48000


def test_sidecar_name_appends_to_the_full_audio_filename() -> None:
    # Appending keeps take1.wav and take1.flac from sharing a sidecar.
    assert sidecar_path("m/take1.wav").name == "take1.wav.phonometry.json"
    assert sidecar_path("m/take1.flac").name == "take1.flac.phonometry.json"


def test_every_field_round_trips_through_the_json(tmp_path: Path) -> None:
    audio = tmp_path / "meas.wav"
    written_to = write_sidecar(
        audio,
        0.0123,
        reference_spl=94.0,
        calibrator_frequency=1000.0,
        calibrator_model="B&K 4231",
        channel_labels=("mic A", "mic B"),
    )
    assert written_to == sidecar_path(audio)
    got = read_sidecar(audio)
    assert got is not None
    assert got.calibration_factor == 0.0123
    assert got.reference_spl == 94.0
    assert got.calibrator_frequency == 1000.0
    assert got.calibrator_model == "B&K 4231"
    assert got.channel_labels == ("mic A", "mic B")
    assert got.phonometry_version is not None
    # The on-disk shape is the documented schema: fixed keys, always there.
    payload = json.loads(written_to.read_text())
    assert payload["schema"] == SIDECAR_SCHEMA
    assert payload["schema_version"] == SIDECAR_VERSION
    assert set(payload) == {
        "schema", "schema_version", "phonometry_version",
        "calibration_factor", "reference_spl", "calibrator",
        "channel_labels",
    }


def test_missing_sidecar_is_none_not_an_error(tmp_path: Path) -> None:
    assert read_sidecar(tmp_path / "lonely.wav") is None


def test_read_applies_the_sidecar_calibration(tmp_path: Path) -> None:
    audio = tmp_path / "meas.wav"
    write(audio, np.full(16, 0.25), FS, subtype="DOUBLE")
    write_sidecar(audio, 2.5, channel_labels=("outdoor mic",))
    sig = read(audio)
    assert sig.calibration_factor == 2.5
    assert sig.channel_labels == ("outdoor mic",)


def test_explicit_calibration_argument_beats_the_sidecar(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "meas.wav"
    write(audio, np.full(16, 0.25), FS)
    write_sidecar(audio, 2.5)
    assert read(audio, calibration_factor=7.0).calibration_factor == 7.0


def test_mismatched_label_count_fails_loudly(tmp_path: Path) -> None:
    audio = tmp_path / "stereo.wav"
    write(audio, np.zeros((2, 8)), FS)
    write_sidecar(audio, 1.0, channel_labels=("only one",))
    with pytest.raises(ValueError, match="channel labels"):
        read(audio)


def test_foreign_json_at_the_reserved_name_is_refused(tmp_path: Path) -> None:
    audio = tmp_path / "meas.wav"
    write(audio, np.zeros(8), FS)
    sidecar_path(audio).write_text('{"unrelated": true}')
    with pytest.raises(ValueError, match="phonometry-calibration"):
        read(audio)


def test_newer_schema_versions_are_refused_by_name(tmp_path: Path) -> None:
    audio = tmp_path / "meas.wav"
    write_sidecar(audio, 1.0)
    target = sidecar_path(audio)
    payload = json.loads(target.read_text())
    payload["schema_version"] = SIDECAR_VERSION + 1
    target.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="upgrade phonometry"):
        read_sidecar(audio)


def test_malformed_fields_are_refused(tmp_path: Path) -> None:
    audio = tmp_path / "meas.wav"
    target = sidecar_path(audio)
    base = {
        "schema": SIDECAR_SCHEMA, "schema_version": 1,
        "phonometry_version": None, "calibration_factor": 1.0,
        "reference_spl": None, "calibrator": None, "channel_labels": None,
    }
    for corruption, message in (
        ({"calibration_factor": "loud"}, "must be a number"),
        ({"calibration_factor": -1.0}, "finite and positive"),
        ({"channel_labels": [1, 2]}, "array of strings"),
        ({"reference_spl": "94"}, "must be a number"),
    ):
        target.write_text(json.dumps(base | corruption))
        with pytest.raises(ValueError, match=message):
            read_sidecar(audio)
    target.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_sidecar(audio)


def test_the_dataclass_itself_rejects_a_nonpositive_factor() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        CalibrationSidecar(calibration_factor=0.0)


def test_write_sidecar_true_requires_a_calibrated_signal(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="calibration_factor"):
        write(tmp_path / "x.wav", np.zeros(8), FS, sidecar=True)
    with pytest.raises(ValueError, match="calibration_factor"):
        write(tmp_path / "x.wav", Signal(data=np.zeros(8), fs=FS),
              sidecar=True)


def test_write_sidecar_true_writes_the_signals_calibration(
    tmp_path: Path,
) -> None:
    sig = Signal(
        data=np.full(16, 0.125), fs=FS, calibration_factor=3.5,
        channel_labels=("courtyard",),
    )
    audio = tmp_path / "cal.wav"
    write(audio, sig, subtype="DOUBLE", sidecar=True)
    assert sidecar_path(audio).exists()
    reread = read(audio)
    assert reread.calibration_factor == 3.5
    assert reread.channel_labels == ("courtyard",)
