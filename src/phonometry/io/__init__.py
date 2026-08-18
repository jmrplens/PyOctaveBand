#  Copyright (c) 2026. Jose Manuel Requena Plens
"""io domain of phonometry (see module docstrings)."""

from __future__ import annotations

from ._backends import LossyCompressionWarning, info, read
from ._chunks import BroadcastMetadata, CuePoint
from ._signal import Signal, SignalSource
from ._wav import AudioFileInfo
from ._write import ClippingWarning, write

__all__ = [
    "AudioFileInfo",
    "BroadcastMetadata",
    "ClippingWarning",
    "CuePoint",
    "LossyCompressionWarning",
    "Signal",
    "SignalSource",
    "info",
    "read",
    "write",
]
