#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
The :class:`Signal` result object: samples plus the metadata they need.

A measurement WAV is only interpretable together with its sample rate, its
calibration and its provenance, yet the ecosystem's readers return a bare
``(x, fs)`` tuple and leave the rest to a notebook variable that does not
survive the session. :class:`Signal` keeps them in one immutable object,
following the library's result-object pattern (a frozen dataclass with
``__array__``, e.g. :class:`~phonometry.room.impulse_response.ImpulseResponseResult`)
rather than subclassing :class:`numpy.ndarray`: NumPy's own subclassing guide
warns that functions may return the base class and *forget* the attached
attributes, and for metrological metadata a silently dropped ``fs`` or
calibration is a wrong result that looks right -- ``x[::2]`` keeping ``fs``
intact is an automatic time-calibration bug. With composition, a NumPy
operation returns a plain array and the loss of metadata is visible in the
type, while every existing ``(x, fs)`` function accepts the object today via
:func:`numpy.asarray`.

The channel convention is the library's: ``data`` is ``(channels, samples)``
float64, reductions run over ``axis=-1``. ``__array__`` returns the mono
channel 1-D and a multichannel block 2-D, matching what every function in
the library expects from a bare array.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from ._chunks import BroadcastMetadata


@dataclass(frozen=True)
class SignalOrigin:
    """Where a :class:`Signal` came from, as read from the file itself.

    The name is deliberately not ``SignalSource``: phonometry already exports
    a :class:`~phonometry.simulation.fdtd.SignalSource` (an FDTD excitation
    driven by an arbitrary sample sequence), and two public classes sharing a
    name across subpackages would collide the moment both are imported flat
    from ``phonometry``. This one is a passive record -- an origin, not a
    source of sound.

    ``bit_depth`` is the *valid* bits per sample (an EXTENSIBLE container
    holding 20 valid bits in 24 reports 20), or ``None`` where the notion
    does not apply (lossy codecs decode to float, not to a bit depth).
    ``lossy`` records that a lossy decoder produced the samples: levels
    computed from such a signal are not metrologically defensible, and the
    flag keeps that fact attached to the data after the read-time warning
    has scrolled away.
    """

    path: str
    container: str
    format_name: str
    bit_depth: int | None
    lossy: bool


@dataclass(frozen=True)
class Signal:
    """A sampled acoustic signal with its rate, calibration and provenance.

    Returned by :func:`phonometry.io.read`; can also be constructed directly
    around an array. The object is a drop-in replacement for the bare array:
    it implements :meth:`__array__`, so ``np.asarray(signal)`` yields the
    samples -- 1-D for one channel, ``(channels, samples)`` for several --
    and the object can be passed straight to the ``(x, fs, ...)`` functions
    of the library. Indexing, ``len()`` and the ``size``/``ndim``/``shape``/
    ``dtype`` attributes forward to that same view, so the object and the
    array it stands for never disagree about geometry.

    ``data`` is always stored ``(channels, samples)`` float64 (a 1-D input
    is stored as one channel); ``calibration_factor`` is the multiplier
    converting digital full-scale units to pascals, the same convention as
    ``signals.levels`` (0 dBFS = RMS 1.0), and stays ``None`` until a
    calibration is actually known -- the object never invents one.
    ``channel_labels`` names each channel (e.g. loudspeaker positions from
    an EXTENSIBLE channel mask); ``provenance`` carries the ``bext``
    broadcast metadata when the file had it; ``source`` records the file,
    container, codec, bit depth and lossy flag of the origin.
    """

    data: NDArray[np.float64]
    fs: int
    calibration_factor: float | None = None
    channel_labels: tuple[str, ...] | None = None
    provenance: BroadcastMetadata | None = None
    source: SignalOrigin | None = None

    def __post_init__(self) -> None:
        data = np.atleast_2d(np.asarray(self.data, dtype=np.float64))
        if data.ndim != 2:
            raise ValueError(
                f"data must be 1-D or (channels, samples); got {data.ndim}-D"
            )
        object.__setattr__(self, "data", np.ascontiguousarray(data))
        if self.channel_labels is not None and len(self.channel_labels) != data.shape[0]:
            raise ValueError(
                f"{len(self.channel_labels)} channel labels for "
                f"{data.shape[0]} channels"
            )
        # Finiteness is checked alongside the sign: NaN and infinity pass
        # every comparison against zero, and an fs or calibration of NaN
        # would flow into durations and levels as a wrong number that
        # looks computed.
        if not np.isfinite(self.fs) or self.fs <= 0:
            raise ValueError(f"fs must be a positive finite number; got {self.fs}")
        if self.calibration_factor is not None and (
            not np.isfinite(self.calibration_factor) or self.calibration_factor <= 0
        ):
            raise ValueError(
                "calibration_factor must be a positive finite number; "
                f"got {self.calibration_factor}"
            )

    @property
    def _view(self) -> NDArray[np.float64]:
        """The array the object stands for: 1-D mono, 2-D multichannel."""
        return self.data[0] if self.data.shape[0] == 1 else self.data

    def __array__(self, dtype: Any = None) -> np.ndarray:
        """Return the samples as an array (optionally recast)."""
        return np.asarray(self._view, dtype=dtype)

    def __len__(self) -> int:
        return int(self._view.shape[0])

    def __getitem__(self, key: Any) -> Any:
        return self._view[key]

    @property
    def size(self) -> int:
        """Total number of samples across all channels."""
        return int(self.data.size)

    @property
    def ndim(self) -> int:
        return int(self._view.ndim)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._view.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data.dtype

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return int(self.data.shape[0])

    @property
    def n_samples(self) -> int:
        """Samples per channel (frames)."""
        return int(self.data.shape[-1])

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return self.n_samples / float(self.fs)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the waveform, calibrated to pascals when a calibration is set.

        Draws each channel's time-domain waveform; with a
        ``calibration_factor`` the amplitude axis is in pascals, otherwise
        in digital full-scale units. Requires matplotlib
        (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.io import plot_signal

        check_language(language)
        return plot_signal(self, ax=ax, language=language, **kwargs)
