#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Reverberation-time prediction from room geometry and absorption.

Predicts the reverberation time ``T`` of an enclosed space from its volume,
boundary areas and the sound-absorption coefficients of its surfaces, through
the classical statistical-acoustics formulae. Given a diffuse field and an
exponential energy decay, ``T`` is the time for the level to fall by 60 dB.

Five models are provided, in order of increasing account of a non-uniform
absorption distribution:

* **Sabine** -- the original diffuse-field estimate,
  :math:`T = k V / (A + 4 m V)`
  with the total equivalent absorption area
  :math:`A = \sum_i S_i \alpha_i` and the
  air term :math:`4 m V`. Exact only for low, uniform absorption.
* **Eyring** (Norris-Eyring) -- replaces ``A`` by
  :math:`-S \ln(1 - \bar{\alpha})` with
  the mean absorption :math:`\bar{\alpha} = A / S` over the total surface
  ``S``; correct in the strong-absorption limit where Sabine overestimates
  ``T``.
* **Millington-Sette** -- :math:`-\sum_i S_i \ln(1 - \alpha_i)` sums the
  Eyring term
  per surface, so a single perfectly absorbing surface drives ``T`` to zero.
* **Fitzroy** -- an *area-weighted arithmetic* mean of three axial Eyring
  reverberation times, one per pair of opposing walls; captures rooms with the
  absorption concentrated on one axis (e.g. a carpeted, otherwise hard room).
* **Arau-Puchades** -- an *area-weighted geometric* mean of the same three
  axial Eyring times (Arau-Puchades, *Acustica* 65 (1988) 163):
  :math:`T = \prod_i T_i^{S_i / S}`. Recommended by its author over Fitzroy
  for anisotropic rooms.

The Sabine constant is :math:`k = 24 \ln 10 / c_0`
(:math:`= 55.26 / c_0`); with the
default :math:`c_0 = 343` m/s it takes the familiar textbook value
``0.161``. (The
:mod:`~phonometry.room.enclosed_space_absorption` EN 12354-6 model instead rounds
``k`` to ``55.3`` and uses :math:`c_0 = 345.6` to pin the factor at exactly
``0.16``.)

Air absorption enters every model through the ``air_attenuation`` power
coefficient ``m`` (in neper per metre) as the additive term :math:`4 m V`;
obtain a physical ``m`` from temperature and humidity with
:func:`~phonometry.environment.propagation.air_absorption.air_attenuation_m`.

Each model enforces its own mathematical domain on the absorption
coefficients. Sabine's linear sum is finite for any non-negative coefficient,
so it accepts measured ISO 354 values at or above 1 (up to a unit-error guard
at 2). The logarithmic models are stricter exactly where the maths requires
it: Millington-Sette needs *every* coefficient below 1, while Eyring, Fitzroy
and Arau-Puchades need each *mean* entering :math:`\ln(1 - \alpha)` below 1.

The Fitzroy and Arau-Puchades models require a rectangular (shoebox) room and
take the room ``dimensions`` together with the mean absorption of each of the
three wall pairs. All five reduce to Eyring for a uniform absorption
distribution, and Eyring reduces to Sabine as the absorption tends to zero --
the identities the conformance suite anchors on, absent a machine-readable
worked example in the source texts.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

from numpy.typing import ArrayLike, NDArray

from .._internal.types import as_float_or_array
from .._internal.validation import require_non_negative, require_positive

# ---------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------

#: Sabine numerator ``24 ln 10 = 55.26...`` (the ``0.161`` constant with
#: ``c0 = 343 m/s``). ``T = 24 ln 10 / c0 * V / A``.
_SABINE_NUMERATOR = 24.0 * log(10.0)

#: Default speed of sound ``c0`` (20 degC dry air), in m/s, giving ``k = 0.161``.
DEFAULT_SPEED_OF_SOUND = 343.0

#: Absorption-coefficient ceiling for the Eyring-family logarithm ``ln(1 -
#: Sanity ceiling for any individual absorption coefficient. Measured ISO 354
#: reverberation-room coefficients legitimately exceed 1 (edge diffraction
#: makes 1.05 to 1.20 routine for thick porous absorbers), so Sabine accepts
#: them as supplied; a value above this bound is almost certainly a unit error
#: (a percentage passed where a fraction is expected) and is rejected by the
#: shared surface validator. The axial models (Fitzroy, Arau-Puchades) never
#: reach it: their stricter below-1 wall-pair-mean check fires first.
_MAX_ABSORPTION = 2.0

#: Number of orthogonal axes -- equivalently wall pairs -- of the rectangular
#: room the axial models describe: the three room lengths ``(Lx, Ly, Lz)``.
_N_ROOM_AXES = 3


Surface = tuple[float, ArrayLike]


# ---------------------------------------------------------------------------
# Surface bookkeeping.
# ---------------------------------------------------------------------------


def _broadcast_or_raise(
    shapes: Sequence[tuple[int, ...]], name: str
) -> tuple[int, ...]:
    """Common broadcast shape of *shapes*, or a clear error naming the mismatch.

    Turns NumPy's terse "operands could not be broadcast" into the library's
    usual descriptive :class:`ValueError` for a per-band count mismatch.
    """
    try:
        return np.broadcast_shapes(*shapes)
    except ValueError:
        msg = (
            f"the per-band {name} have incompatible shapes "
            f"{[tuple(s) for s in shapes]}; each must be a scalar or share a "
            "common band count."
        )
        raise ValueError(msg) from None


def _accumulate_surfaces(
    surfaces: Sequence[Surface],
) -> tuple[float, NDArray[np.float64], list[tuple[float, NDArray[np.float64]]]]:
    r"""Total area ``S``, total absorption area ``A`` and the validated surfaces.

    Returns ``(S, A, pairs)`` where :math:`A = \sum_i S_i \alpha_i` is a
    per-band
    array (0-d for scalar coefficients) and ``pairs`` is the list of validated
    ``(area, alpha)`` items for model-specific processing.

    Validation here is the domain shared by *all* models: coefficients must be
    finite, non-negative and at most :data:`_MAX_ABSORPTION` (the unit-error
    guard). The stricter logarithmic domains are enforced where the maths
    requires them: per surface in :func:`_millington_absorption`, on the mean
    in :func:`_eyring_absorption`.

    :raises ValueError: for an empty surface list, a negative area, a
        non-finite or negative coefficient, a coefficient above
        :data:`_MAX_ABSORPTION`, or per-band coefficients whose band counts do
        not broadcast together.
    """
    if not surfaces:
        msg = "at least one surface is required."
        raise ValueError(msg)
    areas: list[float] = []
    alphas: list[NDArray[np.float64]] = []
    for area, alpha in surfaces:
        areas.append(require_non_negative(area, "surface area"))
        alpha_arr = np.asarray(alpha, dtype=np.float64)
        if not np.all(np.isfinite(alpha_arr)):
            msg = "absorption coefficients must be finite."
            raise ValueError(msg)
        if np.any(alpha_arr < 0.0):
            msg = "absorption coefficients must be non-negative."
            raise ValueError(msg)
        if np.any(alpha_arr > _MAX_ABSORPTION):
            msg = (
                f"absorption coefficients above {_MAX_ABSORPTION} look like a "
                "unit error (a percentage passed instead of a fraction); "
                "measured ISO 354 coefficients do not exceed about 1.2."
            )
            raise ValueError(msg)
        alphas.append(alpha_arr)
    shape = _broadcast_or_raise(
        [a.shape for a in alphas], "surface absorption coefficients"
    )
    total_area = float(sum(areas))
    if total_area <= 0.0:
        msg = "the total surface area must be positive."
        raise ValueError(msg)
    absorption_area = np.zeros(shape, dtype=np.float64)
    for area, alpha_arr in zip(areas, alphas):
        absorption_area = absorption_area + area * alpha_arr
    return total_area, absorption_area, list(zip(areas, alphas))


def _air_term(air_attenuation: ArrayLike, volume: float) -> NDArray[np.float64]:
    """Air absorption area ``4 m V`` (m2), validated finite and non-negative."""
    m = np.asarray(air_attenuation, dtype=np.float64)
    if not np.all(np.isfinite(m)) or np.any(m < 0.0):
        msg = "'air_attenuation' must be finite and non-negative."
        raise ValueError(msg)
    return np.asarray(4.0 * m * volume, dtype=np.float64)


def _add_air(
    absorption_term: NDArray[np.float64], air_attenuation: ArrayLike, volume: float
) -> NDArray[np.float64]:
    """Add the air term ``4 m V`` to an absorption term, band counts checked."""
    air = _air_term(air_attenuation, volume)
    _broadcast_or_raise(
        [absorption_term.shape, air.shape], "absorption and air attenuation"
    )
    return np.asarray(absorption_term + air, dtype=np.float64)


def _millington_absorption(
    pairs: Sequence[tuple[float, NDArray[np.float64]]],
) -> NDArray[np.float64]:
    r"""Millington equivalent absorption (per band).

    :math:`-\sum_i S_i \ln(1 - \alpha_i)`.

    :raises ValueError: for any coefficient at or above 1; the per-surface
        :math:`\ln(1 - \alpha_i)` diverges there, so Millington-Sette (alone
        among the five models) requires every individual coefficient below 1.
    """
    total = np.asarray(0.0, dtype=np.float64)
    for area, alpha_arr in pairs:
        if np.any(alpha_arr >= 1.0):
            msg = (
                "Millington-Sette requires every absorption coefficient below "
                "1: its per-surface ln(1 - alpha) diverges at alpha = 1. For "
                "measured ISO 354 coefficients at or above 1 use Sabine, or "
                "Eyring if the mean absorption stays below 1."
            )
            raise ValueError(msg)
        total = total - area * np.log1p(-alpha_arr)
    return np.asarray(total, dtype=np.float64)


def _eyring_absorption(
    total_area: float, mean_absorption: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Eyring equivalent absorption :math:`-S \ln(1 - \bar{\alpha})` (per band)."""
    # A mean of exactly 1 (fully absorbing on average) has no finite Eyring
    # time: ln(1 - mean) diverges, so fail with a clear message instead.
    if np.any(mean_absorption >= 1.0):
        msg = (
            "the mean absorption coefficient must be below 1 for the "
            "Eyring-family models: ln(1 - mean) diverges at a mean of 1. "
            "Individual coefficients at or above 1 are accepted (up to the "
            "shared unit-error ceiling of 2) as long as the mean stays "
            "below 1; Sabine does not constrain the mean."
        )
        raise ValueError(msg)
    return np.asarray(-total_area * np.log1p(-mean_absorption), dtype=np.float64)


def _reverberation_time(
    volume: float, absorption: NDArray[np.float64], speed_of_sound: float
) -> np.ndarray | float:
    r""":math:`T = 24 \ln 10 / c_0 \cdot V / \text{absorption}`, with a
    positive-absorption guard.
    """
    if np.any(absorption <= 0.0):
        msg = (
            "the total absorption is non-positive; a perfectly reflecting room "
            "has no finite reverberation time."
        )
        raise ValueError(msg)
    t = _SABINE_NUMERATOR / speed_of_sound * volume / absorption
    return as_float_or_array(t)


def mean_absorption(surfaces: Sequence[Surface]) -> np.ndarray | float:
    r"""Area-weighted mean absorption coefficient
    :math:`\bar{\alpha} = A / S`.

    :param surfaces: Sequence of ``(area, absorption_coefficient)`` pairs; each
        coefficient a scalar or a per-band array.
    :return: The mean absorption
        :math:`\sum_i S_i \alpha_i / \sum_i S_i`; a float for
        scalar coefficients, otherwise a per-band array.
    """
    total_area, absorption_area, _ = _accumulate_surfaces(surfaces)
    return as_float_or_array(absorption_area / total_area)


# ---------------------------------------------------------------------------
# Statistical models over an arbitrary surface list.
# ---------------------------------------------------------------------------


def sabine_reverberation_time(
    volume: float,
    surfaces: Sequence[Surface],
    *,
    air_attenuation: ArrayLike = 0.0,
    speed_of_sound: float = DEFAULT_SPEED_OF_SOUND,
) -> np.ndarray | float:
    r"""Sabine reverberation time :math:`T = k V / (A + 4 m V)`.

    :math:`A = \sum_i S_i \alpha_i` is finite for any non-negative
    coefficient, so
    unlike the logarithmic models Sabine accepts coefficients at or above 1:
    measured ISO 354 reverberation-room values of 1.05 to 1.20 (the edge
    effect) and the exact 1.0 that the ISO 11654 practical rating caps at are
    legitimate inputs. Coefficients above 2 are rejected as a probable unit
    error (a percentage passed instead of a fraction).

    :param volume: Room volume ``V``, m3.
    :param surfaces: Sequence of ``(area, absorption_coefficient)`` pairs; each
        coefficient a scalar or a per-band array in ``[0, 2]``.
    :param air_attenuation: Air power-attenuation coefficient ``m``, in neper
        per metre (scalar or per-band); see
        :func:`~phonometry.environment.propagation.air_absorption.air_attenuation_m`. Default ``0``
        (air absorption neglected).
    :param speed_of_sound: Speed of sound ``c0``, m/s (default
        :data:`DEFAULT_SPEED_OF_SOUND`, giving the factor ``0.161``).
    :return: The reverberation time ``T``, s; a float for scalar inputs,
        otherwise a per-band array.
    """
    volume = require_positive(volume, "volume")
    speed_of_sound = require_positive(speed_of_sound, "speed_of_sound")
    _, absorption_area, _ = _accumulate_surfaces(surfaces)
    absorption = _add_air(absorption_area, air_attenuation, volume)
    return _reverberation_time(volume, absorption, speed_of_sound)


def eyring_reverberation_time(
    volume: float,
    surfaces: Sequence[Surface],
    *,
    air_attenuation: ArrayLike = 0.0,
    speed_of_sound: float = DEFAULT_SPEED_OF_SOUND,
) -> np.ndarray | float:
    r"""Eyring (Norris-Eyring) reverberation time.

    .. math::

       T = \frac{k V}{-S \ln(1 - \bar{\alpha}) + 4 m V}

    with the total surface ``S``
    and its area-weighted mean absorption ``alpha_bar``.

    The formula constrains only the *mean*: :math:`\ln(1 - \bar{\alpha})`
    requires :math:`\bar{\alpha} < 1`, while individual coefficients at or
    above 1 (a measured
    ISO 354 outcome) are accepted as long as the mean stays below 1 and each
    coefficient stays within the shared unit-error ceiling of 2.

    :param volume: Room volume ``V``, m3.
    :param surfaces: Sequence of ``(area, absorption_coefficient)`` pairs.
    :param air_attenuation: Air power-attenuation coefficient ``m`` (1/m).
    :param speed_of_sound: Speed of sound ``c0``, m/s.
    :return: The reverberation time ``T``, s.
    """
    volume = require_positive(volume, "volume")
    speed_of_sound = require_positive(speed_of_sound, "speed_of_sound")
    total_area, absorption_area, _ = _accumulate_surfaces(surfaces)
    mean = absorption_area / total_area
    absorption = _add_air(_eyring_absorption(total_area, mean), air_attenuation, volume)
    return _reverberation_time(volume, absorption, speed_of_sound)


def millington_sette_reverberation_time(
    volume: float,
    surfaces: Sequence[Surface],
    *,
    air_attenuation: ArrayLike = 0.0,
    speed_of_sound: float = DEFAULT_SPEED_OF_SOUND,
) -> np.ndarray | float:
    r"""Millington-Sette reverberation time.

    .. math::

       T = \frac{k V}{-\sum_i S_i \ln(1 - \alpha_i) + 4 m V}

    The Eyring absorption
    term is summed surface by surface rather than through a single mean.
    A surface
    approaching total absorption (:math:`\alpha_i \to 1`) drives ``T`` to
    zero.
    Because the logarithm applies per surface, *every* coefficient must be
    strictly below 1; measured ISO 354 coefficients at or above 1 are outside
    this model's domain (use Sabine, or Eyring while the mean stays below 1).

    :param volume: Room volume ``V``, m3.
    :param surfaces: Sequence of ``(area, absorption_coefficient)`` pairs.
    :param air_attenuation: Air power-attenuation coefficient ``m`` (1/m).
    :param speed_of_sound: Speed of sound ``c0``, m/s.
    :return: The reverberation time ``T``, s.
    """
    volume = require_positive(volume, "volume")
    speed_of_sound = require_positive(speed_of_sound, "speed_of_sound")
    _, _, pairs = _accumulate_surfaces(surfaces)
    absorption = _add_air(_millington_absorption(pairs), air_attenuation, volume)
    return _reverberation_time(volume, absorption, speed_of_sound)


# ---------------------------------------------------------------------------
# Axial models for a rectangular room (Fitzroy, Arau-Puchades).
# ---------------------------------------------------------------------------


def _axial_geometry(
    dimensions: Sequence[float],
) -> tuple[float, float, NDArray[np.float64]]:
    r"""Volume ``V``, total surface ``S`` and the three wall-pair areas ``S_i``.

    ``S_i`` is the combined area of the two walls perpendicular to axis ``i``:
    :math:`S_x = 2 L_y L_z`, :math:`S_y = 2 L_x L_z`,
    :math:`S_z = 2 L_x L_y`.
    """
    if len(dimensions) != _N_ROOM_AXES:
        msg = "'dimensions' must be the three room lengths (Lx, Ly, Lz)."
        raise ValueError(msg)
    lx, ly, lz = (require_positive(float(d), "dimension") for d in dimensions)
    volume = lx * ly * lz
    pair_areas = np.array(
        [2.0 * ly * lz, 2.0 * lx * lz, 2.0 * lx * ly], dtype=np.float64
    )
    total_area = float(pair_areas.sum())
    return volume, total_area, pair_areas


def _axial_eyring_times(
    dimensions: Sequence[float],
    absorptions: Sequence[ArrayLike],
    air_attenuation: ArrayLike,
    speed_of_sound: float,
) -> tuple[NDArray[np.float64], list[np.ndarray | float]]:
    """Per-axis Eyring reverberation times and their area weights ``S_i / S``.

    Each axial time uses the *whole* room surface ``S`` and the mean absorption
    ``alpha_bar_i`` of the wall pair perpendicular to that axis (Arau-Puchades'
    construction), plus the shared air term ``4 m V``.
    """
    if len(absorptions) != _N_ROOM_AXES:
        msg = (
            "'absorptions' must give the mean absorption of the three wall pairs "
            "(perpendicular to x, y and z)."
        )
        raise ValueError(msg)
    volume, total_area, pair_areas = _axial_geometry(dimensions)
    speed_of_sound = require_positive(speed_of_sound, "speed_of_sound")
    air = _air_term(air_attenuation, volume)
    means: list[NDArray[np.float64]] = []
    for alpha in absorptions:
        mean = np.asarray(alpha, dtype=np.float64)
        if not np.all(np.isfinite(mean)):
            msg = "absorption coefficients must be finite."
            raise ValueError(msg)
        if np.any(mean < 0.0):
            msg = "absorption coefficients must be non-negative."
            raise ValueError(msg)
        if np.any(mean >= 1.0):
            msg = (
                "each wall-pair mean absorption must be below 1: the axial "
                "Eyring term ln(1 - alpha_i) diverges at alpha_i = 1. The "
                "inputs of the Fitzroy and Arau-Puchades models are "
                "themselves the means entering the logarithm."
            )
            raise ValueError(msg)
        means.append(mean)
    # The three axial times are combined (arithmetic/geometric mean), so their
    # band counts -- and the shared air term -- must broadcast together.
    _broadcast_or_raise(
        [m.shape for m in means] + [air.shape], "axis absorptions and air attenuation"
    )
    times: list[np.ndarray | float] = []
    for mean in means:
        absorption = _eyring_absorption(total_area, mean) + air
        times.append(_reverberation_time(volume, absorption, speed_of_sound))
    weights = pair_areas / total_area
    return weights, times


def fitzroy_reverberation_time(
    dimensions: Sequence[float],
    absorptions: Sequence[ArrayLike],
    *,
    air_attenuation: ArrayLike = 0.0,
    speed_of_sound: float = DEFAULT_SPEED_OF_SOUND,
) -> np.ndarray | float:
    r"""Fitzroy reverberation time -- area-weighted arithmetic mean of axial times.

    :math:`T = \sum_i (S_i / S)\, T_i` with :math:`T_i` the Eyring time of
    the wall pair
    perpendicular to axis ``i`` (Fitzroy, *J. Acoust. Soc. Am.* 31 (1959) 893).
    Equivalent to
    :math:`T = \frac{k V}{S^2} \sum_i \frac{S_i}{-\ln(1 - \alpha_i)}`
    without air. Reduces to Eyring for a uniform absorption distribution.
    Each input is itself a mean entering :math:`\ln(1 - \alpha_i)`, so each
    must be below 1.

    :param dimensions: Room lengths ``(Lx, Ly, Lz)``, m.
    :param absorptions: Mean absorption ``(alpha_x, alpha_y, alpha_z)`` of the
        three wall pairs (perpendicular to x, y, z); each a scalar or per-band
        array.
    :param air_attenuation: Air power-attenuation coefficient ``m`` (1/m).
    :param speed_of_sound: Speed of sound ``c0``, m/s.
    :return: The reverberation time ``T``, s.
    """
    weights, times = _axial_eyring_times(
        dimensions, absorptions, air_attenuation, speed_of_sound
    )
    total = sum(w * np.asarray(t, dtype=np.float64) for w, t in zip(weights, times))
    return as_float_or_array(total)


def arau_puchades_reverberation_time(
    dimensions: Sequence[float],
    absorptions: Sequence[ArrayLike],
    *,
    air_attenuation: ArrayLike = 0.0,
    speed_of_sound: float = DEFAULT_SPEED_OF_SOUND,
) -> np.ndarray | float:
    r"""Arau-Puchades reverberation time -- area-weighted geometric mean of axial times.

    :math:`T = \prod_i T_i^{S_i / S}` with :math:`T_i` the Eyring time of the
    wall pair perpendicular to axis ``i`` (Arau-Puchades, *Acustica* 65
    (1988) 163, Formula 18). Preferred by its author over Fitzroy for rooms
    with an anisotropic absorption distribution. Reduces to Eyring for a
    uniform distribution. Each input is itself a mean entering
    :math:`\ln(1 - \alpha_i)`, so each must be below 1.

    :param dimensions: Room lengths ``(Lx, Ly, Lz)``, m.
    :param absorptions: Mean absorption ``(alpha_x, alpha_y, alpha_z)`` of the
        three wall pairs (perpendicular to x, y, z); each a scalar or per-band
        array.
    :param air_attenuation: Air power-attenuation coefficient ``m`` (1/m).
    :param speed_of_sound: Speed of sound ``c0``, m/s.
    :return: The reverberation time ``T``, s.
    """
    weights, times = _axial_eyring_times(
        dimensions, absorptions, air_attenuation, speed_of_sound
    )
    log_t = sum(
        w * np.log(np.asarray(t, dtype=np.float64)) for w, t in zip(weights, times)
    )
    return as_float_or_array(np.exp(log_t))


# ---------------------------------------------------------------------------
# Bundled model comparison for a rectangular room.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReverberationModelResult:
    """Predicted reverberation time of a rectangular room by five models.

    :ivar frequencies: Band centre frequencies, in hertz.
    :ivar sabine: Sabine reverberation time per band, s.
    :ivar eyring: Eyring (Norris-Eyring) reverberation time per band, s.
    :ivar millington_sette: Millington-Sette reverberation time per band, s.
    :ivar fitzroy: Fitzroy reverberation time per band, s.
    :ivar arau_puchades: Arau-Puchades reverberation time per band, s.
    :ivar volume: Room volume ``V``, m3.
    :ivar surface_area: Total boundary area ``S``, m2.
    """

    frequencies: np.ndarray
    sabine: np.ndarray
    eyring: np.ndarray
    millington_sette: np.ndarray
    fitzroy: np.ndarray
    arau_puchades: np.ndarray
    volume: float
    surface_area: float

    @property
    def models(self) -> dict[str, np.ndarray]:
        """The five reverberation-time curves keyed by model name."""
        return {
            "Sabine": self.sabine,
            "Eyring": self.eyring,
            "Millington-Sette": self.millington_sette,
            "Fitzroy": self.fitzroy,
            "Arau-Puchades": self.arau_puchades,
        }

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the reverberation-time curves of the five models.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.room import plot_reverberation_models

        check_language(language)
        return plot_reverberation_models(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render a reverberation-time prediction fiche to a PDF.

        Writes a one-page report of the reverberation time predicted by the
        five statistical-acoustics models (Sabine, Eyring, Millington-Sette,
        Fitzroy and Arau-Puchades): a standard-basis line marking it a
        design-stage prediction, an optional metadata header block (client,
        room, description, room volume, total surface area, climate ...), a
        per-band table with one reverberation-time column per model beside the
        result's own model-comparison plot (:meth:`plot`), the boxed
        mid-frequency reverberation time from Arau-Puchades (the recommended
        model for a non-uniform absorption distribution) with the per-model
        spread alongside, and a footer with the fixed disclaimer. This is a
        prediction, not a measurement: the five models bracket the
        reverberation time likely to occur. A supplied
        ``metadata.requirement`` is printed as a target reverberation-time
        reference line without a PASS/FAIL verdict, since a room reverberation
        time is a target range rather than a strictly higher/lower-is-better
        quantity.

        :param path: Destination path of the PDF file.
        :param metadata: Optional
            :class:`~phonometry.ReportMetadata`; ``None`` produces a bare
            prediction fiche (body, result and disclaimer only).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for parity with the other fiches; the model
            table already shows every computed value, so it has no effect.
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
            msg = f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            raise ValueError(msg)
        from .._report.reverberation import render_reverberation_models_report

        return render_reverberation_models_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def reverberation_time_models(
    dimensions: Sequence[float],
    absorptions: Sequence[ArrayLike],
    *,
    air_attenuation: ArrayLike = 0.0,
    frequencies: ArrayLike | None = None,
    speed_of_sound: float = DEFAULT_SPEED_OF_SOUND,
) -> ReverberationModelResult:
    """Predict the reverberation time of a rectangular room by all five models.

    A convenience front-end that builds the six boundary surfaces of the room
    from ``dimensions`` and the three wall-pair mean absorptions, then evaluates
    :func:`sabine_reverberation_time`, :func:`eyring_reverberation_time`,
    :func:`millington_sette_reverberation_time`, :func:`fitzroy_reverberation_time`
    and :func:`arau_puchades_reverberation_time` on a common footing. Because
    the bundle evaluates the logarithmic models too, the inputs must satisfy
    the strictest of the five domains: every absorption below 1.

    :param dimensions: Room lengths ``(Lx, Ly, Lz)``, m.
    :param absorptions: Mean absorption ``(alpha_x, alpha_y, alpha_z)`` of the
        three wall pairs (perpendicular to x, y, z); each a scalar or a per-band
        array aligned with ``frequencies``.
    :param air_attenuation: Air power-attenuation coefficient ``m`` (1/m),
        scalar or per-band.
    :param frequencies: Band centre frequencies, in hertz, used only to label
        the result and its plot; defaults to an integer index over the bands.
    :param speed_of_sound: Speed of sound ``c0``, m/s.
    :return: The :class:`ReverberationModelResult`.
    """
    volume, total_area, pair_areas = _axial_geometry(dimensions)
    if len(absorptions) != _N_ROOM_AXES:
        msg = "'absorptions' must give the mean absorption of the three wall pairs."
        raise ValueError(msg)
    # Two equal-area opposing walls per axis share the axis mean absorption.
    surfaces: list[Surface] = []
    for area, alpha in zip(pair_areas, absorptions):
        surfaces.append((float(area) / 2.0, alpha))
        surfaces.append((float(area) / 2.0, alpha))

    sabine = np.atleast_1d(
        sabine_reverberation_time(
            volume,
            surfaces,
            air_attenuation=air_attenuation,
            speed_of_sound=speed_of_sound,
        )
    )
    eyring = np.atleast_1d(
        eyring_reverberation_time(
            volume,
            surfaces,
            air_attenuation=air_attenuation,
            speed_of_sound=speed_of_sound,
        )
    )
    millington = np.atleast_1d(
        millington_sette_reverberation_time(
            volume,
            surfaces,
            air_attenuation=air_attenuation,
            speed_of_sound=speed_of_sound,
        )
    )
    fitzroy = np.atleast_1d(
        fitzroy_reverberation_time(
            dimensions,
            absorptions,
            air_attenuation=air_attenuation,
            speed_of_sound=speed_of_sound,
        )
    )
    arau = np.atleast_1d(
        arau_puchades_reverberation_time(
            dimensions,
            absorptions,
            air_attenuation=air_attenuation,
            speed_of_sound=speed_of_sound,
        )
    )

    curve_bands = max(arr.size for arr in (sabine, eyring, millington, fitzroy, arau))
    if frequencies is None:
        n_bands = curve_bands
        freq = np.arange(1.0, n_bands + 1.0, dtype=np.float64)
    else:
        freq = np.asarray(frequencies, dtype=np.float64)
        if curve_bands not in (1, freq.size):
            msg = (
                f"'frequencies' has {freq.size} bands but the per-band "
                f"absorption has {curve_bands}; they must match."
            )
            raise ValueError(msg)
        n_bands = freq.size

    def _fit(arr: np.ndarray) -> np.ndarray:
        return np.broadcast_to(arr, (n_bands,)).astype(np.float64)

    return ReverberationModelResult(
        frequencies=freq,
        sabine=_fit(sabine),
        eyring=_fit(eyring),
        millington_sette=_fit(millington),
        fitzroy=_fit(fitzroy),
        arau_puchades=_fit(arau),
        volume=volume,
        surface_area=total_area,
    )
