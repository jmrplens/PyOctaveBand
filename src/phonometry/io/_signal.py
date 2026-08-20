#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""The :class:`Signal` result object: samples plus the metadata they need.

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

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from ._chunks import BroadcastMetadata


@dataclass(frozen=True)
class SignalOrigin:
    """Where a :class:`Signal` came from, as read from the file itself.

    The name is deliberately not ``SignalSource``, which
    :class:`~phonometry.simulation.SignalSource` already is (an FDTD
    excitation driven by an arbitrary sample sequence). The two could share a
    spelling now that each is reached through its own package, and the reason
    for the distinction is the better one anyway: this is a passive record --
    an origin, not a source of sound.

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

    Written ``io.Signal`` after ``from phonometry import io``, or
    ``phonometry.Signal``: the top level publishes this one class because
    seven packages accept one, and those are the two spellings. The module
    holding this definition is private, and a name read out of it is the same
    object with nothing promising it will stay reachable that way.

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
        if (
            self.channel_labels is not None
            and len(self.channel_labels) != data.shape[0]
        ):
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

    def pick(self, channels: int | Sequence[int]) -> Signal:
        """The chosen channels, as a new Signal.

        Indexing a Signal yields the samples, which is what makes it a
        drop-in for the array it stands for; the cost is that ``sig[0]``
        drops the rate, the calibration and the labels on the floor. This
        keeps them, so a multichannel take can be narrowed to the channel
        under test and stay a measurement.

        :param channels: One channel index, or a sequence of them, in the
            order they should appear in the result.
        :return: A :class:`Signal` with those channels, in that order.
        :raises IndexError: If a channel is out of range.
        """
        wanted = (
            [channels] if isinstance(channels, (int, np.integer)) else list(channels)
        )
        picked = [int(c) for c in wanted]
        for c in picked:
            if not -self.data.shape[0] <= c < self.data.shape[0]:
                raise IndexError(
                    f"channel {c} out of range for a {self.data.shape[0]}-channel signal"
                )
        labels = (
            tuple(self.channel_labels[c] for c in picked)
            if self.channel_labels is not None
            else None
        )
        return Signal(
            data=self.data[picked],
            fs=self.fs,
            calibration_factor=self.calibration_factor,
            channel_labels=labels,
            provenance=self.provenance,
            source=self.source,
        )

    def crop(self, tmin: float | None = None, tmax: float | None = None) -> Signal:
        """The samples between *tmin* and *tmax* seconds, as a new Signal.

        The edges are seconds from the start of the record, and follow the
        half-open convention of a Python slice: the sample at *tmax* is not
        included, so cropping ``[0, t)`` and ``[t, end)`` partitions the
        record with nothing counted twice. ``None`` means the record's own
        edge.

        :param tmin: Start time, in seconds (default: the beginning).
        :param tmax: End time, in seconds, exclusive (default: the end).
        :return: A :class:`Signal` over that span, at the same rate.
        :raises ValueError: If an edge is negative, not finite, or if
            *tmax* is not after *tmin*.
        """
        start = 0.0 if tmin is None else float(tmin)
        stop = self.duration if tmax is None else float(tmax)
        if not np.isfinite(start) or not np.isfinite(stop):
            raise ValueError("'tmin' and 'tmax' must be finite times in seconds")
        if start < 0.0:
            raise ValueError(f"'tmin' must not be negative; got {start}")
        if stop <= start:
            raise ValueError(f"'tmax' ({stop}) must be greater than 'tmin' ({start})")
        # Ceiling, not rounding: the span is half-open, so a sample belongs
        # to it when tmin <= i/fs < tmax, which is i >= tmin*fs. Rounding
        # would pull in the sample just before tmin whenever the edge falls
        # in the first half of a sample period. The rounding to nanoseconds
        # is for the products that land on 2.9999999999999996 rather than 3.
        first = math.ceil(round(start * self.fs, 9))
        last = min(math.ceil(round(stop * self.fs, 9)), self.data.shape[1])
        if first >= last:
            raise ValueError(
                f"the span [{start}, {stop}) s holds no samples at {self.fs} Hz"
            )
        return Signal(
            data=self.data[:, first:last],
            fs=self.fs,
            calibration_factor=self.calibration_factor,
            channel_labels=self.channel_labels,
            provenance=self.provenance,
            source=self.source,
        )

    def plot(
        self,
        ax: Axes | None = None,
        *,
        language: str = "en",
        scale: str = "linear",
        **kwargs: Any,
    ) -> Axes:
        """Plot the waveform, calibrated to pascals when a calibration is set.

        Draws each channel's time-domain waveform; with a
        ``calibration_factor`` the amplitude axis is in pascals, otherwise
        in digital full-scale units. ``scale="db"`` draws the magnitude of
        each sample as ``20 lg(|p| / 20 uPa)`` instead, which needs a
        calibrated record to mean anything. That is a waveform in decibels
        and not a sound pressure level: an ``L_p`` is defined on a mean
        square over a stated time weighting, and
        :func:`~phonometry.filters.time_weighting` is what produces one.
        Requires matplotlib
        (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.io import plot_signal

        check_language(language)
        return plot_signal(self, ax=ax, language=language, scale=scale, **kwargs)
