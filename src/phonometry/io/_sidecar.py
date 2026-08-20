#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""The calibration sidecar: a versioned JSON file beside the audio.

No audio container carries a microphone calibration. The equipment survey
behind this module found the same pattern everywhere: sound level meters
export linear WAV and keep the sensitivity in a proprietary companion
(NTi's report text, Svantek's .svl/.svt), and ``bext`` has no calibration
field -- its loudness values are EBU R 128 *programme* loudness, and
pressing them into service as a sensitivity would corrupt both meanings.
Seismology met the identical problem decades ago and settled it with a
standardised sidecar (the StationXML inventory beside the miniSEED
waveform, in obspy's model); this module is that answer at phonometry's
scale: a small versioned JSON written beside the audio file, so the one
number that turns digital full scale into pascals travels with the
recording through filesystems, archives and colleagues, in a format any
tool can read.

**Naming.** The sidecar of ``measurement.wav`` is
``measurement.wav.phonometry.json``: the full audio filename plus a
suffix that names the producing library. Appending (rather than swapping
the extension) keeps files that differ only by extension from sharing a
sidecar, and the ``.phonometry.json`` tail makes the file self-describing
in a directory listing and collision-proof against generic ``.json``
companions other tools drop beside recordings.

**Schema, version 1.** A single JSON object; every key is always present
(``null`` when unknown), so consumers parse one fixed shape:

======================  ======================================================
Key                     Meaning
======================  ======================================================
``schema``              The constant ``"phonometry-calibration"``; a reader
                        must refuse a file claiming anything else.
``schema_version``      Integer, this layout is ``1``. Readers accept equal
                        or older versions and refuse newer ones loudly (a
                        newer writer may have changed a key's meaning).
``phonometry_version``  The library version that wrote the file, for
                        forensics; never used to gate reading.
``calibration_factor``  The digital-to-pascal multiplier (0 dBFS = RMS
                        1.0 convention of ``signals.levels``), as derived
                        by :func:`phonometry.metrology.sensitivity` from a
                        calibrator recording read through the same reader.
                        Required, finite and positive.
``reference_spl``       The calibrator's known SPL the factor was derived
                        against (dB SPL, typically 94.0 per IEC 60942), or
                        ``null``.
``calibrator``          Object ``{"frequency": Hz | null, "model": str |
                        null}`` describing the calibrator tone (nominally
                        1000 Hz, where all IEC 61672 weightings are 0 dB).
``channel_labels``      Array of one label per channel, or ``null``. When
                        present it overrides labels derived from the
                        file's channel mask: the sidecar is curated by
                        whoever made the measurement, the mask by whatever
                        firmware wrote the file.
======================  ======================================================

:func:`phonometry.io.read` looks for the sidecar automatically and applies
its calibration when the caller did not pass one explicitly -- the
explicit argument always wins, because the person at the keyboard knows
more than a file on disk -- so the pair "WAV + sidecar" behaves as a
calibrated measurement with no ceremony at the call site.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

#: The schema identifier and the layout version this module writes.
SIDECAR_SCHEMA = "phonometry-calibration"
SIDECAR_VERSION = 1

#: The tail appended to the full audio filename to name its sidecar.
_SIDECAR_TAIL = ".phonometry.json"


@dataclass(frozen=True)
class CalibrationSidecar:
    """The calibration record of one audio file (schema v1, module docstring).

    ``calibration_factor`` is the digital-to-pascal multiplier and the
    only mandatory field; the rest document how it was obtained
    (``reference_spl``, ``calibrator_frequency``, ``calibrator_model``)
    and what the channels are (``channel_labels``).
    ``phonometry_version`` records the writing library version.
    """

    calibration_factor: float
    reference_spl: float | None = None
    calibrator_frequency: float | None = None
    calibrator_model: str | None = None
    channel_labels: tuple[str, ...] | None = None
    phonometry_version: str | None = None

    def __post_init__(self) -> None:
        factor = float(self.calibration_factor)
        if not (math.isfinite(factor) and factor > 0):
            msg = (
                f"calibration_factor must be finite and positive; got "
                f"{self.calibration_factor!r}"
            )
            raise ValueError(msg)


def sidecar_path(audio_path: str | Path) -> Path:
    """The sidecar filename of an audio file (see the module docstring)."""
    audio = Path(audio_path)
    return audio.with_name(audio.name + _SIDECAR_TAIL)


def write_sidecar(
    audio_path: str | Path,
    calibration_factor: float,
    *,
    reference_spl: float | None = None,
    calibrator_frequency: float | None = None,
    calibrator_model: str | None = None,
    channel_labels: tuple[str, ...] | None = None,
) -> Path:
    """Write the calibration sidecar beside an audio file.

    Serialises schema v1 with every key present (the module docstring's
    table); an existing sidecar is replaced, which is the update semantics
    a recalibration wants. The audio file itself is never touched.

    :param audio_path: The audio file the sidecar belongs to (it need not
        exist yet; writing the sidecar first is fine).
    :param calibration_factor: Digital-to-pascal multiplier (required,
        finite, positive).
    :param reference_spl: The calibrator's known SPL, dB (e.g. 94.0).
    :param calibrator_frequency: The calibrator tone's nominal frequency,
        Hz (e.g. 1000.0).
    :param calibrator_model: Free-text calibrator identification.
    :param channel_labels: One label per channel of the audio file.
    :return: The path the sidecar was written to.
    """
    from .._version import __version__

    record = CalibrationSidecar(
        calibration_factor=float(calibration_factor),
        reference_spl=reference_spl,
        calibrator_frequency=calibrator_frequency,
        calibrator_model=calibrator_model,
        channel_labels=channel_labels,
        phonometry_version=__version__,
    )
    target = sidecar_path(audio_path)
    payload = {
        "schema": SIDECAR_SCHEMA,
        "schema_version": SIDECAR_VERSION,
        "phonometry_version": record.phonometry_version,
        "calibration_factor": record.calibration_factor,
        "reference_spl": record.reference_spl,
        "calibrator": {
            "frequency": record.calibrator_frequency,
            "model": record.calibrator_model,
        },
        "channel_labels": (
            None if record.channel_labels is None else list(record.channel_labels)
        ),
    }
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def _optional_number(payload: dict[str, object], key: str, path: Path) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{path}: {key} must be a number or null; got {value!r}"
        raise ValueError(  # noqa: TRY004 - ValueError keeps the module validation errors uniform
            msg
        )
    return float(value)


def _load_sidecar_payload(source: Path) -> dict[str, object]:
    """Parse the sidecar's JSON and check its schema declaration."""
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{source}: sidecar is not valid JSON"
        raise ValueError(msg) from exc
    if not isinstance(payload, dict) or payload.get("schema") != SIDECAR_SCHEMA:
        msg = (
            f"{source}: not a {SIDECAR_SCHEMA!r} sidecar; refusing to "
            "guess at its meaning"
        )
        raise ValueError(msg)
    version = payload.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        msg = f"{source}: schema_version must be an integer"
        raise ValueError(  # noqa: TRY004 - ValueError keeps the module validation errors uniform
            msg
        )
    if version > SIDECAR_VERSION:
        msg = (
            f"{source}: sidecar schema version {version} is newer than the "
            f"version {SIDECAR_VERSION} this phonometry understands; "
            "upgrade phonometry to read it"
        )
        raise ValueError(msg)
    return payload


def _required_factor(payload: dict[str, object], source: Path) -> float:
    """The mandatory calibration factor, validated as a number."""
    factor = payload.get("calibration_factor")
    if isinstance(factor, bool) or not isinstance(factor, int | float):
        msg = f"{source}: calibration_factor must be a number; got {factor!r}"
        raise ValueError(  # noqa: TRY004 - ValueError keeps the module validation errors uniform
            msg
        )
    return float(factor)


def _calibrator_fields(
    payload: dict[str, object], source: Path
) -> tuple[dict[str, object], str | None]:
    """The calibrator object (``{}`` for null) and its validated model."""
    calibrator = payload.get("calibrator")
    if calibrator is None:
        calibrator = {}
    if not isinstance(calibrator, dict):
        msg = f"{source}: calibrator must be an object or null"
        raise ValueError(  # noqa: TRY004 - ValueError keeps the module validation errors uniform
            msg
        )
    model = calibrator.get("model")
    if model is not None and not isinstance(model, str):
        msg = f"{source}: calibrator model must be a string or null"
        raise ValueError(msg)
    return calibrator, model


def _channel_labels(payload: dict[str, object], source: Path) -> tuple[str, ...] | None:
    """The channel labels as a tuple, or ``None`` when absent or null."""
    labels = payload.get("channel_labels")
    if labels is None:
        return None
    if not isinstance(labels, list) or not all(
        isinstance(label, str) for label in labels
    ):
        msg = f"{source}: channel_labels must be an array of strings or null"
        raise ValueError(msg)
    return tuple(labels)


def read_sidecar(audio_path: str | Path) -> CalibrationSidecar | None:
    """Read an audio file's calibration sidecar, if one exists.

    Returns ``None`` when there is no sidecar -- the common case, never an
    error. A file *at the sidecar's reserved name* that is not a valid
    phonometry calibration record raises instead of being ignored: a
    corrupted or foreign file squatting on ``*.phonometry.json`` beside a
    measurement is a problem to surface, not to read past (silently
    dropping it would silently drop the calibration).

    :param audio_path: The audio file whose sidecar to look for.
    :return: The parsed record, or ``None`` when no sidecar exists.
    :raises ValueError: If the sidecar exists but is not valid JSON, does
        not declare this schema, was written by a newer schema version, or
        carries malformed fields.
    """
    source = sidecar_path(audio_path)
    if not source.exists():
        return None
    payload = _load_sidecar_payload(source)
    factor = _required_factor(payload, source)
    calibrator, model = _calibrator_fields(payload, source)
    labels = _channel_labels(payload, source)
    try:
        return CalibrationSidecar(
            calibration_factor=factor,
            reference_spl=_optional_number(payload, "reference_spl", source),
            calibrator_frequency=_optional_number(calibrator, "frequency", source),
            calibrator_model=model,
            channel_labels=labels,
            phonometry_version=(
                str(payload["phonometry_version"])
                if payload.get("phonometry_version") is not None
                else None
            ),
        )
    except ValueError as exc:
        msg = f"{source}: {exc}"
        raise ValueError(msg) from exc
