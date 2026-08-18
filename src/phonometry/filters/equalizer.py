#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Parametric equalizer biquads per the RBJ Audio EQ Cookbook.

Second-order (biquad) IIR sections designed from the closed-form recipes of
Robert Bristow-Johnson's *Audio EQ Cookbook*, the de-facto reference for
parametric equalization (published as a W3C Working Group Note:
https://www.w3.org/TR/audio-eq-cookbook/). Every filter type of the cookbook
is available:

* ``peaking`` - the bell of a parametric EQ: gain ``G`` dB exactly at ``f0``,
  0 dB exactly at DC and Nyquist.
* ``lowshelf`` / ``highshelf`` - shelving filters: gain ``G`` dB exactly at
  DC (low shelf) or Nyquist (high shelf), 0 dB at the opposite end, with the
  transition centred on ``f0`` (the midpoint-gain frequency).
* ``lowpass`` / ``highpass`` - second-order Butterworth-style sections with
  a resonance set by ``Q`` (:math:`Q = 1/\sqrt{2}` is the Butterworth
  alignment; the magnitude at ``f0`` is exactly ``Q``).
* ``bandpass`` - constant 0 dB peak gain at ``f0``.
* ``bandpass_skirt`` - the cookbook's constant-skirt-gain variant (peak
  gain ``Q``).
* ``notch`` - a null exactly at ``f0``, 0 dB at DC and Nyquist.
* ``allpass`` - unit magnitude everywhere; only the phase turns (360
  degrees across the band, steepest at ``f0``).

Each section is parameterized exactly as the cookbook defines: sample rate
``fs``, centre/corner frequency ``f0``, gain ``gain_db`` (peaking and
shelves only) and one of

* ``q`` - the quality factor (default :math:`1/\sqrt{2}`),
* ``bw`` - the bandwidth in octaves, mapped through the cookbook's
  digital-domain relation
  :math:`\alpha = \sin(\omega_0) \sinh\!\left( \frac{\ln 2}{2} \, BW \,
  \frac{\omega_0}{\sin \omega_0} \right)`
  (the :math:`\omega_0/\sin \omega_0` factor compensates the bilinear
  frequency warping), or
* ``slope`` - the shelf-slope parameter ``S`` (shelves only;
  :math:`S = 1` is the steepest slope that stays monotonic).

For the peaking filter the bandwidth is measured between the midpoint-gain
(:math:`G/2` dB) frequencies; for band-pass and notch, between the -3 dB
frequencies - both as the cookbook states.

The design is exact, not approximate: the cookbook's formulas are the
bilinear transform of second-order analog prototypes with the frequency
prewarped so the analog ``f0`` lands exactly on the digital ``f0``.
Closed-form consequences pinned by the test suite: the peaking gain at
``f0`` is exactly ``G`` dB, shelves reach exactly ``G`` dB at their shelved
end and exactly 0 dB at the other, the all-pass magnitude is exactly 1 at
every frequency, and the half-gain bandwidth honours the cookbook's Q
definition.

Reference: Bristow-Johnson, R. *Audio EQ Cookbook*. Republished as a W3C
Working Group Note (ed. R. Toy), 8 June 2021.
https://www.w3.org/TR/audio-eq-cookbook/
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from scipy import signal

from .._internal.utils import _sos_initial_state, _sos_state_mismatch
from ..io._resolve import refuse_foreign_rate, resolve_fs, resolve_samples
from ..io._signal import Signal

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

__all__ = [
    "EQFilterType",
    "EQResponseResult",
    "EQSection",
    "ParametricEQ",
    "parametric_eq",
]

#: The biquad types of the RBJ Audio EQ Cookbook.
EQFilterType = Literal[
    "peaking",
    "lowshelf",
    "highshelf",
    "lowpass",
    "highpass",
    "bandpass",
    "bandpass_skirt",
    "notch",
    "allpass",
]

_SHELF_TYPES = ("lowshelf", "highshelf")
_GAIN_TYPES = ("peaking", "lowshelf", "highshelf")
_BANDWIDTH_TYPES = ("peaking", "bandpass", "bandpass_skirt", "notch")

#: Default quality factor when neither ``q``, ``bw`` nor ``slope`` is given:
#: the Butterworth alignment of the second-order prototypes.
_DEFAULT_Q = 1.0 / math.sqrt(2.0)


@dataclass(frozen=True)
class EQSection:
    r"""One biquad of the RBJ Audio EQ Cookbook.

    :ivar filter_type: ``'peaking'``, ``'lowshelf'``, ``'highshelf'``,
        ``'lowpass'``, ``'highpass'``, ``'bandpass'`` (constant 0 dB peak
        gain), ``'bandpass_skirt'`` (constant skirt gain, peak gain ``Q``),
        ``'notch'`` or ``'allpass'``.
    :ivar f0: Centre/corner frequency, in Hz (must sit below Nyquist).
    :ivar gain_db: Gain ``G`` in dB - peaking and shelving types only.
    :ivar q: Quality factor. Exactly one of ``q``, ``bw`` and ``slope`` may
        be given; with none, :math:`q = 1/\sqrt{2}` (Butterworth
        alignment).
    :ivar bw: Bandwidth in octaves (peaking: between the midpoint-gain
        frequencies; band-pass/notch: between the -3 dB frequencies).
    :ivar slope: Shelf-slope parameter ``S`` (shelves only; :math:`S = 1`
        is the steepest monotonic slope, and ``S`` must stay below the
        gain-dependent bound :math:`(A + 1/A)/(A + 1/A - 2)` with
        :math:`A = 10^{\text{gain\_db}/40}`, beyond which the cookbook's
        alpha turns complex; at 0 dB gain :math:`A = 1`, the bound is
        unbounded and every positive slope is admissible).
    """

    filter_type: EQFilterType
    f0: float
    gain_db: float = 0.0
    q: float | None = None
    bw: float | None = None
    slope: float | None = None

    def __post_init__(self) -> None:
        _validate_section(self)


def _validate_section(section: EQSection) -> None:
    """Validate one :class:`EQSection` against the cookbook's parameter rules."""
    if section.filter_type not in _DESIGNERS:
        raise ValueError(
            f"Unknown filter_type {section.filter_type!r}; expected one of "
            f"{sorted(_DESIGNERS)}."
        )
    for name, value in (
        ("f0", section.f0),
        ("gain_db", section.gain_db),
        ("q", section.q),
        ("bw", section.bw),
        ("slope", section.slope),
    ):
        if value is not None and not math.isfinite(value):
            raise ValueError(f"'{name}' must be finite, got {value!r}.")
    if section.f0 <= 0:
        raise ValueError("Centre frequency 'f0' must be positive.")
    _gain_amplitude(section.gain_db)
    _validate_section_parameterization(section)


def _gain_amplitude(gain_db: float) -> float:
    r"""The cookbook's amplitude :math:`A = 10^{\text{gain\_db}/40}`.

    A finite but extreme ``gain_db`` overflows the power (about
    +12320 dB) or underflows it to zero (about -12320 dB), where the
    designers would divide by zero; both are rejected with the documented
    :class:`ValueError` instead of leaking raw arithmetic errors.
    """
    try:
        big_a = float(10.0 ** (gain_db / 40.0))
    except OverflowError:
        big_a = math.inf
    if not (big_a > 0.0 and math.isfinite(big_a)):
        raise ValueError(
            f"'gain_db' = {gain_db:g} dB is outside the representable "
            "range: 10^(gain_db/40) must be a positive finite number."
        )
    return big_a


def _validate_shelf_slope(section: EQSection) -> None:
    r"""Bound the shelf slope so the cookbook's alpha stays real.

    The alpha recipe takes :math:`\sqrt{(A + 1/A)(1/S - 1) + 2}`, which
    is real only for :math:`S < (A + 1/A)/(A + 1/A - 2)` (any ``S`` when
    the gain is 0 dB, i.e. :math:`A = 1`); at the bound itself alpha is 0
    and the poles land on the unit circle.
    """
    if section.slope is None:
        return
    big_a = _gain_amplitude(section.gain_db)
    a_sum = big_a + 1.0 / big_a
    if a_sum <= 2.0:  # A = 1 (gain 0 dB): every positive slope is fine
        return
    bound = a_sum / (a_sum - 2.0)
    if section.slope >= bound:
        raise ValueError(
            f"Shelf slope 'slope' = {section.slope:g} is too steep for "
            f"gain_db = {section.gain_db:g}: the cookbook's alpha is real "
            f"only for slopes below (A + 1/A)/(A + 1/A - 2) = {bound:.6g}."
        )


def _validate_section_parameterization(section: EQSection) -> None:
    """Check the ``gain_db`` / ``q`` / ``bw`` / ``slope`` combination."""
    given = [
        name
        for name, value in (("q", section.q), ("bw", section.bw), ("slope", section.slope))
        if value is not None
    ]
    if len(given) > 1:
        raise ValueError(f"Give only one of 'q', 'bw' and 'slope', not {given}.")
    if section.q is not None and section.q <= 0:
        raise ValueError("Quality factor 'q' must be positive.")
    if section.bw is not None and section.bw <= 0:
        raise ValueError("Bandwidth 'bw' must be positive (octaves).")
    if section.slope is not None and section.slope <= 0:
        raise ValueError("Shelf slope 'slope' must be positive.")
    if section.slope is not None and section.filter_type not in _SHELF_TYPES:
        raise ValueError("'slope' applies to the shelving types only.")
    if section.bw is not None and section.filter_type not in _BANDWIDTH_TYPES:
        raise ValueError(
            "'bw' applies to 'peaking', 'bandpass', 'bandpass_skirt' and 'notch' only."
        )
    if abs(section.gain_db) > 0 and section.filter_type not in _GAIN_TYPES:
        raise ValueError(
            "'gain_db' applies to 'peaking', 'lowshelf' and 'highshelf' only."
        )
    _validate_shelf_slope(section)


# ---------------------------------------------------------------------------
# Cookbook coefficient recipes (one small designer per filter type)
# ---------------------------------------------------------------------------

#: (b0, b1, b2), (a0, a1, a2) - unnormalized, exactly as the cookbook prints.
_Coefficients = tuple[tuple[float, float, float], tuple[float, float, float]]

#: Designer signature: (cos_w0, sin_w0, alpha, big_a) -> coefficients.
_Designer = Callable[[float, float, float, float], _Coefficients]


def _design_lowpass(c: float, s: float, alpha: float, big_a: float) -> _Coefficients:
    return ((1 - c) / 2, 1 - c, (1 - c) / 2), (1 + alpha, -2 * c, 1 - alpha)


def _design_highpass(c: float, s: float, alpha: float, big_a: float) -> _Coefficients:
    return ((1 + c) / 2, -(1 + c), (1 + c) / 2), (1 + alpha, -2 * c, 1 - alpha)


def _design_bandpass(c: float, s: float, alpha: float, big_a: float) -> _Coefficients:
    # Constant 0 dB peak gain.
    return (alpha, 0.0, -alpha), (1 + alpha, -2 * c, 1 - alpha)


def _design_bandpass_skirt(c: float, s: float, alpha: float, big_a: float) -> _Coefficients:
    # Constant skirt gain; the peak gain is Q (b0 = sin(w0)/2 = Q*alpha).
    return (s / 2, 0.0, -s / 2), (1 + alpha, -2 * c, 1 - alpha)


def _design_notch(c: float, s: float, alpha: float, big_a: float) -> _Coefficients:
    return (1.0, -2 * c, 1.0), (1 + alpha, -2 * c, 1 - alpha)


def _design_allpass(c: float, s: float, alpha: float, big_a: float) -> _Coefficients:
    return (1 - alpha, -2 * c, 1 + alpha), (1 + alpha, -2 * c, 1 - alpha)


def _design_peaking(c: float, s: float, alpha: float, big_a: float) -> _Coefficients:
    return (
        (1 + alpha * big_a, -2 * c, 1 - alpha * big_a),
        (1 + alpha / big_a, -2 * c, 1 - alpha / big_a),
    )


def _design_lowshelf(c: float, s: float, alpha: float, big_a: float) -> _Coefficients:
    sq = 2 * math.sqrt(big_a) * alpha
    b = (
        big_a * ((big_a + 1) - (big_a - 1) * c + sq),
        2 * big_a * ((big_a - 1) - (big_a + 1) * c),
        big_a * ((big_a + 1) - (big_a - 1) * c - sq),
    )
    a = (
        (big_a + 1) + (big_a - 1) * c + sq,
        -2 * ((big_a - 1) + (big_a + 1) * c),
        (big_a + 1) + (big_a - 1) * c - sq,
    )
    return b, a


def _design_highshelf(c: float, s: float, alpha: float, big_a: float) -> _Coefficients:
    sq = 2 * math.sqrt(big_a) * alpha
    b = (
        big_a * ((big_a + 1) + (big_a - 1) * c + sq),
        -2 * big_a * ((big_a - 1) + (big_a + 1) * c),
        big_a * ((big_a + 1) + (big_a - 1) * c - sq),
    )
    a = (
        (big_a + 1) - (big_a - 1) * c + sq,
        2 * ((big_a - 1) - (big_a + 1) * c),
        (big_a + 1) - (big_a - 1) * c - sq,
    )
    return b, a


_DESIGNERS: dict[str, _Designer] = {
    "peaking": _design_peaking,
    "lowshelf": _design_lowshelf,
    "highshelf": _design_highshelf,
    "lowpass": _design_lowpass,
    "highpass": _design_highpass,
    "bandpass": _design_bandpass,
    "bandpass_skirt": _design_bandpass_skirt,
    "notch": _design_notch,
    "allpass": _design_allpass,
}


def _section_alpha(section: EQSection, w0: float, big_a: float) -> float:
    """The cookbook's ``alpha`` for the section's Q / BW / S parameterization."""
    sin_w0 = math.sin(w0)
    if section.bw is not None:
        # Digital-domain bandwidth relation: the w0/sin(w0) factor
        # compensates the bilinear-transform frequency warping. Near the
        # Nyquist frequency that factor diverges, so a wide 'bw' can push
        # the sinh argument past the floating-point range.
        try:
            return sin_w0 * math.sinh(
                math.log(2.0) / 2 * section.bw * w0 / sin_w0
            )
        except OverflowError:
            raise ValueError(
                f"Bandwidth 'bw' = {section.bw:g} octaves is too wide for "
                f"f0 = {section.f0:g} Hz this close to the Nyquist "
                "frequency: the cookbook's bandwidth-to-alpha relation "
                "overflows. Reduce 'bw', lower 'f0' or parameterize the "
                "section with 'q' instead."
            ) from None
    if section.slope is not None:
        return (sin_w0 / 2) * math.sqrt(
            (big_a + 1 / big_a) * (1 / section.slope - 1) + 2
        )
    q = section.q if section.q is not None else _DEFAULT_Q
    return sin_w0 / (2 * q)


def _check_section_stability(
    fs: float, section: EQSection, a1: float, a2: float
) -> None:
    r"""Require the realized poles strictly inside the unit circle.

    In exact arithmetic every cookbook section with :math:`\alpha > 0` is
    stable, but at the extremes (a very wide ``bw`` or ``f0`` within a few
    ppm of DC or the Nyquist frequency) the rounded coefficients can land
    exactly on the stability-triangle boundary -- e.g. ``a2 == -1.0``, a
    pole pair on the unit circle -- and the filter would ring forever
    without any arithmetic error to flag it.
    """
    if abs(a2) < 1.0 and abs(a1) < 1.0 + a2:
        return
    raise ValueError(
        f"The designed {section.filter_type!r} section at f0 = "
        f"{section.f0:g} Hz (fs = {fs:g} Hz) is not strictly stable after "
        f"rounding (a1 = {a1:.17g}, a2 = {a2:.17g}): its poles do not lie "
        "strictly inside the unit circle. This happens when 'bw' (or "
        "1/'q') is very large or f0 sits within a few ppm of DC or the "
        "Nyquist frequency; bring the section parameters back from the "
        "extreme."
    )


def _section_sos(fs: float, section: EQSection) -> NDArray[np.float64]:
    """One normalized SOS row ``[b0, b1, b2, 1, a1, a2]`` for the section."""
    if section.f0 >= fs / 2:
        raise ValueError(
            f"Centre frequency f0 = {section.f0} Hz must sit below the "
            f"Nyquist frequency {fs / 2} Hz."
        )
    w0 = 2 * math.pi * section.f0 / fs
    big_a = _gain_amplitude(section.gain_db)
    alpha = _section_alpha(section, w0, big_a)
    b, a = _DESIGNERS[section.filter_type](math.cos(w0), math.sin(w0), alpha, big_a)
    a0 = a[0]
    a1, a2 = a[1] / a0, a[2] / a0
    _check_section_stability(fs, section, a1, a2)
    return np.array(
        [b[0] / a0, b[1] / a0, b[2] / a0, 1.0, a1, a2],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Frequency response result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EQResponseResult:
    """Frequency response of a parametric-EQ cascade (RBJ Audio EQ Cookbook).

    :ivar frequencies: Evaluation frequencies, in Hz (log-spaced).
    :ivar magnitude_db: Cascade magnitude, in dB.
    :ivar phase_rad: Cascade phase, unwrapped, in radians.
    :ivar section_magnitude_db: Per-section magnitude, in dB, shape
        ``(n_sections, n_frequencies)`` - the cascade magnitude is their sum.
    :ivar sos: The designed cascade, shape ``(n_sections, 6)`` (scipy SOS
        layout, normalized so every section's ``a0`` is 1).
    :ivar fs: Sample rate, in Hz.
    :ivar sections: The :class:`EQSection` specifications, in cascade order.
    """

    frequencies: NDArray[np.float64]
    magnitude_db: NDArray[np.float64]
    phase_rad: NDArray[np.float64]
    section_magnitude_db: NDArray[np.float64]
    sos: NDArray[np.float64]
    fs: float
    sections: tuple[EQSection, ...]

    def plot(
        self, ax: Axes | None = None, *, language: str = "en",
        show_sections: bool = True, **kwargs: Any
    ) -> Axes | NDArray[Any]:
        """Plot the cascade magnitude and phase response.

        With ``ax`` given, only the magnitude panel is drawn on it.

        :param ax: Existing axes for the magnitude panel, or ``None`` for a
            fresh two-panel (magnitude + phase) figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param show_sections: Overlay each section's individual response under
            the cascade curve (default ``True``).
        :return: The magnitude axes (``ax`` given) or the array of two axes.
        """
        from .._i18n import check_language
        from .._plot.filters import plot_parametric_eq

        check_language(language)
        return plot_parametric_eq(
            self, ax=ax, language=language, show_sections=show_sections, **kwargs
        )


# ---------------------------------------------------------------------------
# The equalizer
# ---------------------------------------------------------------------------


class ParametricEQ:
    """Cascade of RBJ Audio EQ Cookbook biquads.

    Designs one second-order section per :class:`EQSection` and runs them in
    series as a numerically robust SOS cascade, following the house style of
    :class:`~phonometry.filters.weighting.WeightingFilter`
    (reusable coefficients, optional stateful block processing).
    """

    def __init__(
        self,
        fs: float,
        sections: EQSection | Sequence[EQSection],
        stateful: bool = False,
        steady_ic: bool = False,
    ) -> None:
        """
        :param fs: Sample rate in Hz.
        :param sections: One :class:`EQSection` or a sequence of them; the
            cascade applies them in order.
        :param stateful: If True, :meth:`filter` carries the filter state
            across calls (block processing).
        :param steady_ic: If True, initialize the state at steady state.
        """
        if fs <= 0:
            raise ValueError("Sample rate 'fs' must be positive.")
        if isinstance(sections, EQSection):
            sections = (sections,)
        if len(sections) == 0:
            raise ValueError("Give at least one EQSection.")

        self.fs = float(fs)
        self.sections: tuple[EQSection, ...] = tuple(sections)
        self.stateful = stateful
        self.sos = np.vstack([_section_sos(self.fs, s) for s in self.sections])
        if self.stateful:
            self.zi = np.array([])
            self._steady_ic = steady_ic

    def filter(self, x: Signal | list[float] | np.ndarray) -> np.ndarray:
        """Apply the EQ cascade to a signal.

        :param x: Input signal (1D or 2D ``[channels, samples]``), or a
            :class:`phonometry.io.Signal`. A Signal whose rate disagrees
            with the one this cascade was designed for is refused; a
            calibrated one is equalized in pascals.
        :return: Equalized signal.
        :raises ValueError: If a Signal's rate is not the cascade's.
        """
        refuse_foreign_rate(x, self.fs, "EQ cascade")
        x_proc = resolve_samples(x)
        if not self.stateful:
            return cast(np.ndarray, signal.sosfilt(self.sos, x_proc, axis=-1))

        if _sos_state_mismatch(self.zi, x_proc):
            self.zi = _sos_initial_state(self.sos, x_proc, self._steady_ic)
        y, self.zi = signal.sosfilt(self.sos, x_proc, axis=-1, zi=self.zi)
        return cast(np.ndarray, y)

    def response(
        self,
        n_points: int = 2048,
        *,
        f_min: float = 10.0,
        f_max: float | None = None,
    ) -> EQResponseResult:
        """Evaluate the cascade frequency response on a log-spaced grid.

        :param n_points: Number of evaluation frequencies.
        :param f_min: Lowest frequency, in Hz.
        :param f_max: Highest frequency, in Hz (default: Nyquist).
        :return: An :class:`EQResponseResult` (carries the SOS cascade and
            plots the magnitude/phase response).
        """
        if n_points < 2:
            raise ValueError("'n_points' must be at least 2.")
        nyquist = self.fs / 2
        if f_max is None:
            f_max = nyquist
        if not 0 < f_min < f_max <= nyquist:
            raise ValueError(
                "Need 0 < f_min < f_max <= fs/2; got "
                f"f_min = {f_min}, f_max = {f_max}, fs/2 = {nyquist}."
            )

        freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_points)
        # Per-section responses; the cascade is their product.
        h_sections = np.empty((self.sos.shape[0], n_points), dtype=np.complex128)
        for idx in range(self.sos.shape[0]):
            _, h_sections[idx] = signal.sosfreqz(
                self.sos[idx : idx + 1], worN=freqs, fs=self.fs
            )
        h = np.prod(h_sections, axis=0)

        tiny = np.finfo(np.float64).tiny  # floor keeps notch nulls finite
        return EQResponseResult(
            frequencies=freqs,
            magnitude_db=20 * np.log10(np.abs(h) + tiny),
            phase_rad=np.unwrap(np.angle(h)),
            section_magnitude_db=20 * np.log10(np.abs(h_sections) + tiny),
            sos=self.sos.copy(),
            fs=self.fs,
            sections=self.sections,
        )


def parametric_eq(
    x: Signal | list[float] | np.ndarray,
    fs: float | None = None,
    *,
    sections: EQSection | Sequence[EQSection],
) -> np.ndarray:
    """Apply a parametric-EQ cascade to a signal (RBJ Audio EQ Cookbook).

    Convenience wrapper around :class:`ParametricEQ` for one-shot use.

    :param x: Input signal (1D or 2D ``[channels, samples]``), or a
        :class:`phonometry.io.Signal` read from a measurement file. A
        calibrated Signal is equalized in pascals, so the result is in
        pascals too.
    :param fs: Sample rate in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises.
    :param sections: One :class:`EQSection` or a sequence of them.
        Keyword-only and required: it sits behind an optional ``fs``, and a
        default here would be a signature that lies about what the call
        needs.
    :return: Equalized signal.
    """
    fs = resolve_fs(x, fs)
    return ParametricEQ(fs, sections).filter(x)
