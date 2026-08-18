#  Copyright (c) 2026. Jose Manuel Requena Plens
"""io domain of phonometry (see module docstrings)."""

from __future__ import annotations

from ._backends import LossyCompressionWarning, info, read
from ._chunks import BroadcastMetadata, CuePoint
from ._sidecar import (
    CalibrationSidecar,
    read_sidecar,
    sidecar_path,
    write_sidecar,
)
from ._signal import Signal, SignalSource
from ._wav import AudioFileInfo
from ._write import ClippingWarning, write

__all__ = [
    "AudioFileInfo",
    "BroadcastMetadata",
    "CalibrationSidecar",
    "ClippingWarning",
    "CuePoint",
    "LossyCompressionWarning",
    "Signal",
    "SignalSource",
    "info",
    "read",
    "read_sidecar",
    "sidecar_path",
    "write",
    "write_sidecar",
]
