#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Sound absorption in enclosed spaces (EN 12354-6:2003).

Estimates the total equivalent sound absorption area of a room and the
resulting reverberation time from the absorption of its surfaces and objects
(the normative Clause 4 model).

The total equivalent absorption area sums the surface contributions, the
equivalent absorption areas of objects and object arrays and the air
absorption (Formula 1)::

    A = sum_i alpha_s,i * S_i + sum_j Aobj,j + sum_k alpha_s,k * S_k + Aair

with the air term ``Aair = 4*m*V*(1 - psi)`` (Formula 2), the object fraction
``psi = sum Vobj / V`` (Formula 3) and, for hard irregular objects, the
empirical equivalent area ``Aobj = Vobj**(2/3)`` (Formula 4). The reverberation
time follows from ``T = 55.3/c0 * V*(1 - psi) / A`` (Formula 5); with the
standard's ``c0 = 345.6 m/s`` the factor ``55.3/c0`` is the familiar ``0.16``.

The informative Annex D method for irregular spaces / uneven absorption
distribution is out of scope.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

from numpy.typing import ArrayLike

from .._internal.types import as_float_or_array
from .._internal.validation import require_fraction, require_positive

# ---------------------------------------------------------------------------
# Normative constants.
# ---------------------------------------------------------------------------

#: Octave-band centre frequencies of Table 1, in hertz.
OCTAVE_BANDS: np.ndarray = np.array(
    [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0], dtype=np.float64
)

#: Speed of sound giving ``55.3/c0 = 0.16`` (clause 4.4 NOTE), in m/s.
SPEED_OF_SOUND = 345.6

#: Reverberation-time constant of Formula 5 (``55.3``).
_RT_CONSTANT = 55.3

#: The empirical exponent of Formula 4 for a hard object's equivalent area.
_OBJECT_EXPONENT = 2.0 / 3.0

#: Power attenuation coefficient ``m`` in air, in Neper per metre, by
#: temperature/humidity condition over :data:`OCTAVE_BANDS` (Table 1). The
#: table lists ``10**-3 Neper/m``; the values here are already in Neper/m.
AIR_ATTENUATION: dict[str, np.ndarray] = {
    "10C_30-50": np.array([0.1, 0.2, 0.5, 1.1, 2.7, 9.4, 29.0]) * 1e-3,
    "10C_50-70": np.array([0.1, 0.2, 0.5, 0.8, 1.8, 5.9, 21.1]) * 1e-3,
    "10C_70-90": np.array([0.1, 0.2, 0.5, 0.7, 1.4, 4.4, 15.8]) * 1e-3,
    "20C_30-50": np.array([0.1, 0.3, 0.6, 1.0, 1.9, 5.8, 20.3]) * 1e-3,
    "20C_50-70": np.array([0.1, 0.3, 0.6, 1.0, 1.7, 4.1, 13.5]) * 1e-3,
    "20C_70-90": np.array([0.1, 0.3, 0.6, 1.1, 1.7, 3.5, 10.6]) * 1e-3,
}

#: The recommended default air condition (clause 4.3): 20 C, 50 %-70 % humidity.
DEFAULT_AIR_CONDITION = "20C_50-70"


# ---------------------------------------------------------------------------
# Clause 4.3 - total equivalent absorption area.
# ---------------------------------------------------------------------------


def object_fraction(object_volumes: ArrayLike, volume: float) -> float:
    """Object fraction ``psi`` of an enclosed space (Formula 3).

    :param object_volumes: Volumes of the objects and object arrays, m3.
    :param volume: Volume of the empty enclosed space ``V``, m3.
    :return: The object fraction ``psi = sum(Vobj) / V``.
    """
    volume = require_positive(volume, "volume")
    vols = np.asarray(object_volumes, dtype=np.float64)
    if np.any(vols < 0.0):
        raise ValueError("object volumes must be non-negative.")
    psi = float(np.sum(vols)) / volume
    if psi >= 1.0:
        raise ValueError("the object volumes cannot exceed the room volume.")
    return psi


def hard_object_absorption(object_volume: ArrayLike) -> np.ndarray:
    """Equivalent absorption area of a hard object (Formula 4).

    An empirical estimate for hard, irregularly shaped objects (machinery,
    furniture) whose equivalent area is not otherwise available.

    :param object_volume: Volume ``Vobj`` of the hard object(s), m3.
    :return: The equivalent absorption area ``Aobj = Vobj**(2/3)``, m2.
    """
    vol = np.asarray(object_volume, dtype=np.float64)
    if np.any(vol < 0.0):
        raise ValueError("object volumes must be non-negative.")
    return vol**_OBJECT_EXPONENT


def air_absorption_area(
    m: ArrayLike, volume: float, object_fraction: float = 0.0
) -> np.ndarray | float:
    """Equivalent absorption area of the air (Formula 2).

    :param m: Power attenuation coefficient of air, in Neper per metre (see
        :data:`AIR_ATTENUATION`).
    :param volume: Volume of the empty enclosed space ``V``, m3.
    :param object_fraction: Object fraction ``psi`` (0-1).
    :return: The air absorption area ``Aair = 4*m*V*(1 - psi)``, m2.
    """
    volume = require_positive(volume, "volume")
    object_fraction = require_fraction(object_fraction, "object_fraction")
    m_arr = np.asarray(m, dtype=np.float64)
    if np.any(m_arr < 0.0):
        raise ValueError("the attenuation coefficient m must be non-negative.")
    area = 4.0 * m_arr * volume * (1.0 - object_fraction)
    return as_float_or_array(area)


def equivalent_absorption_area(
    surfaces: Sequence[tuple[float, ArrayLike]],
    *,
    objects: ArrayLike = (),
    air_area: ArrayLike = 0.0,
) -> np.ndarray | float:
    """Total equivalent sound absorption area (Formula 1).

    :param surfaces: Sequence of ``(area, absorption_coefficient)`` pairs, one
        per surface (or object array treated as a surface). The absorption
        coefficient may be a scalar or a per-band array.
    :param objects: Equivalent absorption areas of the objects ``Aobj``, m2
        (see :func:`hard_object_absorption`): a single value, a 1-D sequence of
        one value per object, or a 2-D array ``(n_objects, n_bands)`` for
        per-band values; summed over the objects.
    :param air_area: Air absorption area ``Aair``, m2 (see
        :func:`air_absorption_area`); scalar or per-band.
    :return: The total equivalent absorption area ``A``, m2; a float for
        all-scalar inputs, otherwise a per-band array.
    """
    total = np.asarray(air_area, dtype=np.float64)
    if np.any(total < 0.0):
        raise ValueError("air_area must be non-negative.")
    for area, alpha in surfaces:
        if area < 0.0:
            raise ValueError("surface areas must be non-negative.")
        alpha_arr = np.asarray(alpha, dtype=np.float64)
        if np.any(alpha_arr < 0.0):
            raise ValueError("absorption coefficients must be non-negative.")
        total = total + area * alpha_arr
    obj = np.atleast_1d(np.asarray(objects, dtype=np.float64))
    if obj.size:
        if np.any(obj < 0.0):
            raise ValueError("object absorption areas must be non-negative.")
        total = total + obj.sum(axis=0)
    return as_float_or_array(total)


# ---------------------------------------------------------------------------
# Clause 4.4 - reverberation time.
# ---------------------------------------------------------------------------


def reverberation_time(
    absorption_area: ArrayLike,
    volume: float,
    *,
    object_fraction: float = 0.0,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> np.ndarray | float:
    """Reverberation time from the equivalent absorption area (Formula 5).

    :param absorption_area: Total equivalent absorption area ``A``, m2.
    :param volume: Volume of the empty enclosed space ``V``, m3.
    :param object_fraction: Object fraction ``psi`` (0-1).
    :param speed_of_sound: Speed of sound ``c0``, m/s (default
        :data:`SPEED_OF_SOUND`, giving the factor ``0.16``).
    :return: The reverberation time ``T = 55.3/c0 * V*(1 - psi) / A``, s.
    """
    volume = require_positive(volume, "volume")
    speed_of_sound = require_positive(speed_of_sound, "speed_of_sound")
    object_fraction = require_fraction(object_fraction, "object_fraction")
    area = np.asarray(absorption_area, dtype=np.float64)
    if np.any(area <= 0.0):
        raise ValueError("absorption_area must be positive.")
    t = _RT_CONSTANT / speed_of_sound * volume * (1.0 - object_fraction) / area
    return as_float_or_array(t)


@dataclass(frozen=True)
class ReverberationResult:
    """Absorption area and reverberation time of an enclosed space (Clause 4).

    :ivar frequencies: Octave-band centre frequencies, in hertz.
    :ivar absorption_area: Total equivalent absorption area ``A`` per band, m2.
    :ivar reverberation_time: Reverberation time ``T`` per band, s.
    :ivar volume: Volume of the empty enclosed space, m3.
    :ivar object_fraction: Object fraction ``psi`` (0-1).
    """

    frequencies: np.ndarray
    absorption_area: np.ndarray
    reverberation_time: np.ndarray
    volume: float
    object_fraction: float

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the reverberation time over the octave bands.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.room import plot_enclosed_space_absorption

        check_language(language)
        return plot_enclosed_space_absorption(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an enclosed-space absorption/reverberation fiche to a PDF.

        Writes a one-page report characterising an enclosed space by the
        EN 12354-6:2003 Clause 4 model: a standard-basis line, an optional
        metadata header block (client, room, description, room volume, object
        fraction, climate ...), a per-band table of the equivalent sound
        absorption area ``A`` and the reverberation time ``T`` beside the
        result's own reverberation-time plot (:meth:`plot`), the boxed
        mid-frequency reverberation time with the mid-frequency absorption area
        alongside, and a footer with the fixed disclaimer. EN 12354-6 gives a
        diffuse-field estimate rather than a measurement; a supplied
        ``metadata.requirement`` is printed as a target reverberation-time
        reference line without a PASS/FAIL verdict, since a room reverberation
        time is a target range rather than a strictly higher/lower-is-better
        quantity.

        :param path: Destination path of the PDF file.
        :param metadata: Optional
            :class:`~phonometry.ReportMetadata`; ``None`` produces a bare
            characterisation fiche (body, result and disclaimer only).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for parity with the other fiches; the band
            table already shows both A and T, so it has no effect.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"``.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from .._report.reverberation import render_enclosed_space_report

        return render_enclosed_space_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def enclosed_space_reverberation(
    surfaces: Sequence[tuple[float, ArrayLike]],
    volume: float,
    *,
    objects: ArrayLike = (),
    object_fraction: float = 0.0,
    air_condition: str | None = None,
    frequencies: ArrayLike = OCTAVE_BANDS,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> ReverberationResult:
    """Predict the absorption area and reverberation time per octave band.

    Chains the total equivalent absorption area (Formula 1, with the air term
    of Formula 2 from :data:`AIR_ATTENUATION` when ``air_condition`` is given)
    and the reverberation time (Formula 5).

    :param surfaces: Sequence of ``(area, absorption_coefficient)`` pairs; each
        coefficient a per-band array aligned with ``frequencies``.
    :param volume: Volume of the empty enclosed space ``V``, m3.
    :param objects: Equivalent absorption areas of the objects ``Aobj``, m2.
    :param object_fraction: Object fraction ``psi`` (0-1).
    :param air_condition: A key of :data:`AIR_ATTENUATION` (e.g.
        ``"20C_50-70"``) to include air absorption, or ``None`` to neglect it.
    :param frequencies: Octave-band centre frequencies, in hertz. The built-in
        ``air_condition`` profiles require the standard :data:`OCTAVE_BANDS`.
    :param speed_of_sound: Speed of sound ``c0``, m/s.
    :return: The :class:`ReverberationResult`.
    """
    freq = np.asarray(frequencies, dtype=np.float64)
    if air_condition is not None and not np.array_equal(freq, OCTAVE_BANDS):
        raise ValueError(
            "the built-in air_condition profiles cover the standard "
            "OCTAVE_BANDS only; for custom frequencies compute the air term "
            "with air_absorption_area and pass it to equivalent_absorption_area."
        )
    if air_condition is None:
        air_area: np.ndarray | float = 0.0
    elif air_condition in AIR_ATTENUATION:
        air_area = air_absorption_area(
            AIR_ATTENUATION[air_condition], volume, object_fraction
        )
    else:
        raise ValueError(
            f"air_condition must be one of {tuple(AIR_ATTENUATION)} or None; "
            f"got {air_condition!r}."
        )
    area = np.broadcast_to(
        equivalent_absorption_area(surfaces, objects=objects, air_area=air_area),
        freq.shape,
    ).astype(np.float64)
    t = reverberation_time(
        area, volume, object_fraction=object_fraction, speed_of_sound=speed_of_sound
    )
    return ReverberationResult(
        frequencies=freq,
        absorption_area=area,
        reverberation_time=np.asarray(t, dtype=np.float64),
        volume=volume,
        object_fraction=object_fraction,
    )
