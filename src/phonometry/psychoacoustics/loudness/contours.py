#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Normal equal-loudness-level contours per ISO 226:2023.

Implements Formula (1) (clause 4.1: SPL of a pure tone from its loudness
level) and Formula (2) (clause 4.2: the inverse) with the Table 1 (p. 4)
parameters at the 29 preferred third-octave frequencies of ISO 266.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._internal.validation import require_ranks, require_same_length

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

# ISO 226:2023 Table 1 (p. 4): frequency Hz -> (alpha_f, L_U dB, T_f dB).
# alpha_f: exponent for loudness perception; L_U: magnitude of the linear
# transfer function normalized at 1 kHz; T_f: threshold of hearing.
_TABLE1: dict[float, tuple[float, float, float]] = {
    20.0: (0.635, -31.5, 78.1),
    25.0: (0.602, -27.2, 68.7),
    31.5: (0.569, -23.1, 59.5),
    40.0: (0.537, -19.3, 51.1),
    50.0: (0.509, -16.1, 44.0),
    63.0: (0.482, -13.1, 37.5),
    80.0: (0.456, -10.4, 31.5),
    100.0: (0.433, -8.2, 26.5),
    125.0: (0.412, -6.3, 22.1),
    160.0: (0.391, -4.6, 17.9),
    200.0: (0.373, -3.2, 14.4),
    250.0: (0.357, -2.1, 11.4),
    315.0: (0.343, -1.2, 8.6),
    400.0: (0.330, -0.5, 6.2),
    500.0: (0.320, 0.0, 4.4),
    630.0: (0.311, 0.4, 3.0),
    800.0: (0.303, 0.5, 2.2),
    1000.0: (0.300, 0.0, 2.4),
    1250.0: (0.295, -2.7, 3.5),
    1600.0: (0.292, -4.2, 1.7),
    2000.0: (0.290, -1.2, -1.3),
    2500.0: (0.290, 1.4, -4.2),
    3150.0: (0.289, 2.3, -6.0),
    4000.0: (0.289, 1.0, -5.4),
    5000.0: (0.289, -2.3, -1.5),
    6300.0: (0.293, -7.2, 6.0),
    8000.0: (0.303, -11.2, 12.6),
    10000.0: (0.323, -10.9, 13.9),
    12500.0: (0.354, -3.5, 12.3),
}

_FREQUENCIES = np.array(sorted(_TABLE1))

# Validity range of ISO 226:2023 Formula (1), clause 4.1: contours are
# specified from 20 phon to 90 phon between 20 Hz and 4 kHz, and only up to
# 80 phon between 5 kHz and 12.5 kHz - above that level a returned contour
# is truncated to the grid rows at or below 4 kHz.
_PHON_MIN = 20.0
_PHON_MAX = 90.0
_PHON_MAX_ABOVE_4KHZ = 80.0
_F_MAX_ABOVE_80_PHON = 4000.0  # Hz


def _params(frequency: float) -> tuple[float, float, float]:
    """Table 1 parameters for a preferred third-octave frequency.

    The 1e-6 relative tolerance only absorbs floating-point representation
    error; adjacent table rows are >20 % apart, so a near-miss can never
    resolve to the wrong row - anything else raises.
    """
    for f, params in _TABLE1.items():
        if abs(frequency - f) <= 1e-6 * f:
            return params
    msg = (
        f"ISO 226:2023 Table 1 defines parameters only at the 29 preferred "
        f"third-octave frequencies (20 Hz to 12.5 kHz); got frequency "
        f"{frequency!r} Hz. The standard specifies no interpolation."
    )
    raise ValueError(msg)


def _spl_from_phon(frequency: float, phon: float) -> float:
    """ISO 226:2023 Formula (1), clause 4.1 (p. 2)."""
    alpha_f, l_u, t_f = _params(frequency)
    term = (4.0e-10) ** (0.3 - alpha_f) * (10 ** (0.03 * phon) - 10**0.072) + 10 ** (
        alpha_f * (t_f + l_u) / 10
    )
    return float(10 / alpha_f * np.log10(term) - l_u)


def equal_loudness_contour(phon: float) -> tuple[np.ndarray, np.ndarray]:
    """Normal equal-loudness-level contour (ISO 226:2023 Formula 1).

    Returns the sound pressure levels of pure tones judged equally loud as
    a 1 kHz tone at ``phon`` dB SPL, at the 29 preferred third-octave
    frequencies of Table 1.

    Validity (clause 4.1): 20 phon to 90 phon between 20 Hz and 4 kHz, and
    up to 80 phon between 5 kHz and 12.5 kHz; above 80 phon the returned
    contour therefore stops at 4 kHz.

    :param phon: Loudness level in phons (20 to 90).
    :return: Tuple ``(frequencies, spl)`` in Hz and dB re 20 uPa.
    """
    if not _PHON_MIN <= phon <= _PHON_MAX:
        msg = (
            "ISO 226:2023 Formula (1) is specified for 20 phon to 90 phon "
            f"(80 phon above 4 kHz); got {phon!r}."
        )
        raise ValueError(msg)
    freqs = (
        _FREQUENCIES
        if phon <= _PHON_MAX_ABOVE_4KHZ
        else _FREQUENCIES[_FREQUENCIES <= _F_MAX_ABOVE_80_PHON]
    )
    spl = np.array([_spl_from_phon(f, phon) for f in freqs])
    return freqs.copy(), spl


def loudness_level(spl: float, frequency: float) -> float:
    """Loudness level of a pure tone (ISO 226:2023 Formula 2).

    :param spl: Sound pressure level of the tone in dB re 20 uPa.
    :param frequency: Tone frequency in Hz; must be one of the 29 preferred
        third-octave frequencies of Table 1 (the standard specifies no
        interpolation between them).
    :return: Loudness level in phons. Values outside 20-90 phon (80 above
        4 kHz) are extrapolations the standard labels as informative only.
    """
    alpha_f, l_u, t_f = _params(frequency)
    b = (10 ** (alpha_f * (spl + l_u) / 10) - 10 ** (alpha_f * (t_f + l_u) / 10)) / (
        4.0e-10
    ) ** (0.3 - alpha_f) + 10**0.072
    return float(100.0 / 3.0 * np.log10(b))


def hearing_threshold() -> tuple[np.ndarray, np.ndarray]:
    """Threshold of hearing T_f (ISO 226:2023 Table 1).

    :return: Tuple ``(frequencies, threshold)`` in Hz and dB re 20 uPa.
    """
    tf = np.array([_TABLE1[f][2] for f in _FREQUENCIES])
    return _FREQUENCIES.copy(), tf


#: Default loudness levels of the classic ISO 226:2023 contour family: the
#: 20 phon to 90 phon range of Formula (1) in 10 phon steps.
_DEFAULT_PHONS: tuple[float, ...] = (
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    70.0,
    80.0,
    90.0,
)


@dataclass(frozen=True)
class EqualLoudnessContours:
    """The ISO 226:2023 normal equal-loudness-level contour family.

    Bundles a set of equal-loudness contours (ISO 226:2023 Formula 1) with
    the threshold of hearing (Table 1) over a shared frequency grid, so the
    iconic chart can be drawn with :meth:`plot`.

    ``frequencies`` is the frequency grid in Hz (by default the 29 preferred
    third-octave frequencies of Table 1). ``phons`` lists the loudness levels
    of the contours, in phon. ``contours`` holds the sound pressure levels in
    dB re 20 uPa as a ``(len(phons), len(frequencies))`` array, row ``i`` being
    the contour for ``phons[i]``; entries the standard does not define are
    ``nan`` (Formula 1 is valid only up to 4 kHz above 80 phon, so those rows
    stop at 4 kHz). ``threshold`` is the threshold of hearing T_f in dB re
    20 uPa on the same grid.
    """

    frequencies: np.ndarray
    phons: tuple[float, ...]
    contours: np.ndarray
    threshold: np.ndarray

    def __post_init__(self) -> None:
        """Reject a family whose contours do not span the grid they are drawn on.

        Two independent axes meet in ``contours``, and both are read by
        position: every row is drawn against the whole of ``frequencies``,
        and row ``i`` belongs to ``phons[i]``. A row of the wrong width, or a
        threshold of the wrong length, cannot be drawn against that grid at
        all, and says so from inside matplotlib, about a first dimension,
        naming neither the field nor this result; a row count that disagrees
        with ``phons`` pairs each contour with a level that is not its own,
        which the chart's strict pairing catches only once someone draws it
        and a reader indexing the two together never catches.

        The two lengths are checked apart because nothing relates them: a
        grid of eight frequencies would otherwise be taken as agreement with
        the eight levels of the default family.

        :raises ValueError: if a contour row, the threshold or the level list
            disagrees with the axis it is drawn against.
        """
        require_ranks(self, frequencies=1, phons=1, contours=2, threshold=1)
        require_same_length(
            self, "frequencies", ("contours", 1), "threshold", axis="frequency"
        )
        require_same_length(self, "phons", ("contours", 0), axis="loudness level")

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the equal-loudness contour family with the hearing threshold.

        Draws sound pressure level in dB against a logarithmic frequency axis:
        one line per loudness level in :attr:`phons`, plus the threshold of
        hearing. Requires matplotlib (``pip install phonometry[plot]``);
        returns the :class:`~matplotlib.axes.Axes`.

        :param ax: Existing axes to draw on, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the contour ``plot`` calls.
        :return: The axes.
        """
        from ..._i18n import check_language
        from ..._plot.psychoacoustics import plot_equal_loudness_contours

        return plot_equal_loudness_contours(
            self, ax=ax, language=check_language(language), **kwargs
        )


def equal_loudness_contours(
    phons: Sequence[float] = _DEFAULT_PHONS,
    frequencies: Sequence[float] | np.ndarray | None = None,
) -> EqualLoudnessContours:
    """Build the ISO 226:2023 equal-loudness-level contour family.

    Evaluates :func:`equal_loudness_contour` at each level in ``phons`` and
    :func:`hearing_threshold`, and bundles them into an
    :class:`EqualLoudnessContours` result that exposes ``.plot()``. The maths
    is unchanged; this is a thin, plottable wrapper around the existing
    functions.

    :param phons: Loudness levels of the contours, in phon; each must be in
        the 20 phon to 90 phon validity range of Formula (1). Defaults to the
        classic family from 20 phon to 90 phon in 10 phon steps.
    :param frequencies: Frequency grid in Hz; ``None`` (default) uses the 29
        preferred third-octave frequencies of Table 1. Any value given must be
        one of those preferred frequencies (the standard specifies no
        interpolation between them).
    :return: An :class:`EqualLoudnessContours`.
    """
    if frequencies is None:
        grid = _FREQUENCIES.copy()
    else:
        grid = np.asarray(frequencies, dtype=float)
        for f in grid:
            _params(float(f))  # reject any non-preferred frequency

    phon_tuple = tuple(float(p) for p in phons)
    contours = np.full((len(phon_tuple), grid.size), np.nan)
    for i, phon in enumerate(phon_tuple):
        freqs_p, spl_p = equal_loudness_contour(phon)
        for f, s in zip(freqs_p, spl_p, strict=True):
            j = np.flatnonzero(np.isclose(grid, f, rtol=1e-6))
            if j.size:
                contours[i, j[0]] = s

    tf_freqs, tf_vals = hearing_threshold()
    threshold = np.array(
        [
            tf_vals[int(np.flatnonzero(np.isclose(tf_freqs, f, rtol=1e-6))[0])]
            for f in grid
        ]
    )
    return EqualLoudnessContours(
        frequencies=grid, phons=phon_tuple, contours=contours, threshold=threshold
    )
