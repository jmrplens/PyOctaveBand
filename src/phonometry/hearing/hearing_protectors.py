#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""What a hearing protector leaves at the ear (ISO 4869-2:2018).

A protector is measured on people, not on a coupler: ISO 4869-1 seats it on at
least 16 subjects and records the threshold shift it produces in each octave
band. What comes out is a **distribution**, one attenuation per subject per
band, and ISO 4869-2 is the standard that turns that distribution into a level
someone can act on.

**The distribution first (Clause 5).** Every method here starts from the
assumed protection value, the mean attenuation reduced by a multiple of its own
spread (Formula (1)):

.. math::

   APV_{fx} = m_f - \alpha\, s_f

The constant :math:`\alpha` is the inverse standard normal cumulative
distribution at the protection performance :math:`x` (Table 1), so
:math:`APV_{f84}` with :math:`\alpha = 1` is the attenuation 84 % of wearers
reach or beat, and :math:`APV_{f98}` with :math:`\alpha = 2` is what all but
one in fifty reach. A protector is never quoted at its mean.

**Then one of three methods**, in decreasing order of what they need to know
about the noise:

- The **octave-band method** (Clause 6) subtracts the assumed protection value
  band by band from the A-weighted noise spectrum, Formula (2). It needs the
  spectrum and is the most faithful.
- The **HML method** (Clause 7) collapses the protector to three numbers, its
  high-, medium- and low-frequency attenuation values, each the predicted noise
  level reduction for a reference noise of a stated :math:`(L_{p,C} - L_{p,A})`.
  It needs only the C- and A-weighted levels of the noise.
- The **SNR method** (Clause 8) collapses it to one number against a pink noise
  and subtracts it from the C-weighted level. It needs only that level.

The three answer the same question and rarely agree exactly: on the worked
example of Annexes B, C and D the same protector in the same noise gives 81 dB,
82 dB and 82 dB. Clause 1's own NOTE puts differences of 3 dB or less between
comparable protectors below the resolution of the exercise.

The octave-band method starts at 63 Hz when both the noise and the protector
have data there and at 125 Hz when either does not (Clause 6). The ``HML``
(Clause 7) and ``SNR`` (Clause 8) computations start at 125 Hz always,
whatever is available at 63 Hz, which is why the reference spectra of Tables 2
and 3 begin there.

Clause, formula and table numbers refer to ISO 4869-2:2018(E).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

__all__ = [
    "AssumedProtectionResult",
    "HMLRatingResult",
    "PROTECTION_PERFORMANCES",
    "ProtectedLevelResult",
    "SNRRatingResult",
    "assumed_protection_value",
    "hml_protected_level",
    "hml_rating",
    "octave_band_protected_level",
    "snr_protected_level",
    "snr_rating",
]

#: Values of the constant ``alpha`` for the protection performances Table 1
#: tabulates, keyed by the performance ``x`` in per cent. ``alpha`` is the
#: inverse of the standard normal cumulative distribution at ``x``, so a
#: larger performance subtracts more of the spread.
PROTECTION_PERFORMANCES: dict[int, float] = {
    50: 0.00,
    75: 0.67,
    80: 0.84,
    84: 1.00,
    90: 1.28,
    95: 1.64,
    98: 2.00,
}

#: The eight octave bands of Formula (2), in hertz. ``f(1) = 63`` Hz through
#: ``f(8) = 8000`` Hz; the reference spectra of Tables 2 and 3 start at
#: ``k = 2``.
PROTECTOR_OCTAVE_BANDS: tuple[float, ...] = (
    63.0,
    125.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
    8000.0,
)

#: Frequency weighting A at the eight octave-band mid-frequencies, in dB,
#: from IEC 61672-1:2013 Table 3. Formula (2) adds these to the octave-band
#: levels of the noise; ISO 4869-2 reprints them in its own Table B.1.
#: ``tests`` pins them against the library's own copy of that table.
PROTECTOR_A_WEIGHTING: tuple[float, ...] = (
    -26.2,
    -16.1,
    -8.6,
    -3.2,
    0.0,
    1.2,
    1.0,
    -1.1,
)

#: Table 2: the A-weighted octave-band sound pressure levels of the eight
#: reference noises the ``HML`` method is fitted on, 125 Hz to 8000 Hz, each
#: normalized to an A-weighted level of 100 dB.
HML_REFERENCE_NOISES: tuple[tuple[float, ...], ...] = (
    (62.6, 70.8, 81.0, 90.4, 96.2, 94.7, 92.3),
    (68.9, 78.3, 84.3, 92.8, 96.3, 94.0, 90.0),
    (71.1, 80.8, 88.0, 95.0, 94.4, 94.1, 89.0),
    (77.2, 84.5, 89.8, 95.5, 94.3, 92.5, 88.8),
    (77.4, 86.5, 92.5, 96.4, 93.0, 90.4, 83.7),
    (82.0, 89.3, 93.3, 95.6, 93.0, 90.1, 83.0),
    (84.2, 90.1, 93.6, 96.2, 91.3, 87.9, 81.9),
    (88.0, 93.4, 93.8, 94.2, 91.4, 87.9, 79.9),
)

#: Table 2, ``(LpC - LpA)`` column: what each reference noise is worth on the
#: axis the ``HML`` method interpolates along, in dB.
HML_REFERENCE_C_MINUS_A: tuple[float, ...] = (
    -1.2,
    -0.5,
    0.1,
    1.6,
    2.3,
    4.3,
    6.1,
    8.4,
)

#: Table 2, ``d`` column: the empirically derived weights of Formulas (12) to
#: (14). Note 1 to the table gives no closed form for them.
HML_REFERENCE_D: tuple[float, ...] = (
    -1.20,
    -0.49,
    0.14,
    1.56,
    -2.98,
    -1.01,
    0.85,
    3.14,
)

#: Table 3: the A-weighted octave-band levels, 125 Hz to 8000 Hz, of the pink
#: noise the ``SNR`` method is defined against, whose C-weighted level is
#: 100 dB.
PINK_NOISE_A_WEIGHTED: tuple[float, ...] = (
    75.9,
    83.4,
    88.8,
    92.0,
    93.2,
    93.0,
    90.9,
)

#: The reference level of Formulas (15) and (22), in dB. Both reference noise
#: families are normalized to 100 dB, which the NOTEs call arbitrary and
#: chosen for computational simplicity.
_REFERENCE_TOTAL = 100.0

#: The ``(LpC - LpA)`` at which Formulas (16) and (17) change branch, in dB.
#: It is also where ``M`` itself is defined (Clause 3.6).
_HML_BREAK = 2.0

#: A distribution needs two dimensions and more than one subject before it
#: has a spread at all; ISO 4869-1 asks for at least sixteen.
_GRID_RANK = 2
_MINIMUM_SUBJECTS = 2

#: Denominators of Formulas (16) and (17): 4 dB from the ``H`` anchor at
#: -2 dB to the break, and 8 dB from the break to the ``L`` anchor at +10 dB.
_HML_HIGH_SPAN = 4.0
_HML_LOW_SPAN = 8.0


def _alpha_for(performance: int) -> float:
    """The Table 1 constant for a protection performance, or a refusal.

    Whole numbers only, and no coercion: ``int(84.5)`` is 84, so truncating
    would answer a question nobody asked, and ``"84"`` would let a string
    select a normative constant.

    :param performance: The protection performance ``x``, in per cent.
    :return: The constant ``alpha``.
    :raises ValueError: for a performance Table 1 does not tabulate.
    """
    listed = ", ".join(str(x) for x in sorted(PROTECTION_PERFORMANCES))
    msg = (
        "'performance' must be one of the protection performances Table 1 "
        f"tabulates ({listed} per cent); got {performance!r}."
    )
    if isinstance(performance, (bool, np.bool_)) or not isinstance(
        performance, (int, float, np.integer, np.floating)
    ):
        raise ValueError(msg)
    whole = int(performance)
    if whole != performance or whole not in PROTECTION_PERFORMANCES:
        raise ValueError(msg)
    return PROTECTION_PERFORMANCES[whole]


def _subject_attenuations(
    attenuation: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """The ``(subjects, bands)`` attenuation grid ISO 4869-1 produces.

    :param attenuation: One row per test subject, one column per octave band.
    :return: The grid as a float array.
    :raises ValueError: if it is not two-dimensional, if it holds fewer than
        two subjects, or if any value is not finite.
    """
    grid = np.asarray(attenuation, dtype=np.float64)
    if grid.ndim != _GRID_RANK:
        msg = (
            "'attenuation' must be a (subjects, bands) grid: ISO 4869-1 "
            "measures one attenuation per subject per band, and the spread "
            "across subjects is what Formula (1) subtracts."
        )
        raise ValueError(msg)
    if grid.shape[0] < _MINIMUM_SUBJECTS:
        msg = (
            "'attenuation' needs at least two subjects for a standard "
            "deviation; ISO 4869-1 asks for at least 16."
        )
        raise ValueError(msg)
    if grid.size == 0 or not np.all(np.isfinite(grid)):
        msg = "'attenuation' must contain only finite values."
        raise ValueError(msg)
    return grid


def _round_half_up(value: float) -> int:
    """Round to the nearest integer, halves away from zero.

    Clauses 6, 7.3 and 8.3 all end "shall be rounded to the nearest integer",
    which is not what :func:`round` does: it sends halves to the even
    neighbour, so 82,5 dB would come back 82 dB.

    :param value: The level to round.
    :return: The nearest integer, with .5 going away from zero.
    """
    return int(np.floor(value + 0.5)) if value >= 0 else -int(np.floor(-value + 0.5))


@dataclass(frozen=True)
class AssumedProtectionResult:
    r"""Assumed protection values of a hearing protector (Clause 5).

    :ivar apv: :math:`APV_{fx} = m_f - \alpha s_f` per octave band, in dB.
    :ivar mean_attenuation: The mean attenuation :math:`m_f` per band, in dB.
    :ivar standard_deviation: The standard deviation :math:`s_f` per band, in
        dB, over the test subjects.
    :ivar performance: The protection performance ``x``, in per cent.
    :ivar alpha: The Table 1 constant that ``performance`` selected.
    :ivar frequencies: Octave-band mid-frequencies, in hertz.
    :ivar subjects: Number of test subjects the distribution came from.
    """

    apv: np.ndarray
    mean_attenuation: np.ndarray
    standard_deviation: np.ndarray
    performance: int
    alpha: float
    frequencies: np.ndarray
    subjects: int

    def plot(self, ax: Axes | None = None, language: str = "en", **kwargs: Any) -> Axes:
        """Draw the mean attenuation, its spread and the assumed protection.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the ``APV`` curve.
        :return: The axes.
        """
        from .._plot.hearing import plot_assumed_protection

        return plot_assumed_protection(self, ax=ax, language=language, **kwargs)


def assumed_protection_value(
    attenuation: Sequence[Sequence[float]] | np.ndarray,
    *,
    performance: int = 84,
    frequencies: Sequence[float] | np.ndarray | None = None,
) -> AssumedProtectionResult:
    r"""Assumed protection values of a hearing protector (Formula (1)).

    The attenuation of a protector is a distribution over people, and Clause 5
    reduces it to the level a stated share of wearers reaches or beats:

    .. math::

       APV_{fx} = m_f - \alpha\, s_f \tag{1}

    with :math:`m_f` and :math:`s_f` the mean and standard deviation of the
    per-subject attenuations of ISO 4869-1 and :math:`\alpha` the inverse
    standard normal cumulative distribution at the protection performance
    (Table 1). The standard deviation is the sample one, over ``N - 1``.

    :param attenuation: A ``(subjects, bands)`` grid of sound attenuation
        values, in dB, one row per subject, measured to ISO 4869-1.
    :param performance: The protection performance ``x``, in per cent, from
        Table 1: 50, 75, 80, 84 (the default), 90, 95 or 98.
    :param frequencies: Octave-band mid-frequencies, in hertz, or ``None`` for
        the eight bands of Formula (2) when the grid has eight columns.
    :return: :class:`AssumedProtectionResult`.
    :raises ValueError: if the grid is not two-dimensional or holds fewer than
        two subjects, if any value is not finite, if ``performance`` is not one
        Table 1 tabulates, or if ``frequencies`` does not match the grid.

    .. note::

       Annex A prints its ``APV`` row as the difference of the *rounded*
       ``m_f`` and ``s_f`` it displays above it, which differs from Formula (1)
       applied to the underlying data by 0,1 dB in three of its eight bands.
       This returns Formula (1) applied to the data; round afterwards if you
       need the annex's table back.
    """
    grid = _subject_attenuations(attenuation)
    alpha = _alpha_for(performance)
    freqs = _octave_axis(frequencies, grid.shape[1], "assumed_protection_value")
    mean = np.asarray(grid.mean(axis=0), dtype=np.float64)
    spread = np.asarray(grid.std(axis=0, ddof=1), dtype=np.float64)
    return AssumedProtectionResult(
        apv=np.asarray(mean - alpha * spread, dtype=np.float64),
        mean_attenuation=mean,
        standard_deviation=spread,
        performance=int(performance),
        alpha=alpha,
        frequencies=freqs,
        subjects=int(grid.shape[0]),
    )


def _octave_axis(
    frequencies: Sequence[float] | np.ndarray | None, count: int, owner: str
) -> np.ndarray:
    """The band axis of a result, defaulted to Formula (2)'s eight bands.

    :param frequencies: The caller's frequencies, or ``None``.
    :param count: How many bands the data carries.
    :param owner: Name used in the error message.
    :return: The mid-frequencies as a float array.
    :raises ValueError: if the count does not match, or if no default fits.
    """
    if frequencies is None:
        if count == len(PROTECTOR_OCTAVE_BANDS):
            return np.asarray(PROTECTOR_OCTAVE_BANDS, dtype=np.float64)
        if count == len(PROTECTOR_OCTAVE_BANDS) - 1:
            return np.asarray(PROTECTOR_OCTAVE_BANDS[1:], dtype=np.float64)
        msg = (
            f"{owner}: pass 'frequencies' explicitly for {count} bands; only "
            "the eight octaves of Formula (2), or those seven without 63 Hz, "
            "are assumed."
        )
        raise ValueError(msg)
    freqs = np.asarray(frequencies, dtype=np.float64)
    if freqs.ndim != 1 or freqs.size != count:
        msg = (
            f"{owner}: 'frequencies' must be one mid-band frequency per band; "
            f"got {freqs.size} for {count} bands."
        )
        raise ValueError(msg)
    return freqs


@dataclass(frozen=True)
class ProtectedLevelResult:
    r"""The A-weighted level left at the ear behind a protector.

    :ivar effective_level: :math:`L'_{p,Ax}`, in dB, unrounded. Clauses 6, 7.3
        and 8.3 all report it to the nearest integer, which
        :attr:`reported_level` does.
    :ivar noise_reduction: :math:`PNR_x = L_{p,A} - L'_{p,Ax}`, in dB, or
        ``None`` where it cannot be formed. The ``SNR`` method given only a
        C-weighted level never learns :math:`L_{p,A}`, and the difference
        between the C-weighted level and the answer is the rating itself
        rather than a noise reduction.
    :ivar performance: The protection performance ``x``, in per cent, or
        ``None`` when the rating that produced it did not carry one.
    :ivar method: ``"octave-band"``, ``"HML"`` or ``"SNR"``.
    :ivar band_levels: The A-weighted band levels behind the protector, in dB,
        for the octave-band method, and ``None`` for the other two, which
        never see a spectrum.
    :ivar frequencies: Octave-band mid-frequencies, in hertz, or ``None``.
    """

    effective_level: float
    noise_reduction: float | None
    performance: int | None
    method: str
    band_levels: np.ndarray | None = None
    frequencies: np.ndarray | None = None

    @property
    def reported_level(self) -> int:
        """:attr:`effective_level` rounded the way the standard reports it.

        :return: The nearest integer, halves away from zero.
        """
        return _round_half_up(self.effective_level)

    def plot(self, ax: Axes | None = None, language: str = "en", **kwargs: Any) -> Axes:
        """Draw the band levels the protector leaves, where there are any.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the protected-level bars.
        :return: The axes.
        :raises ValueError: for an ``HML`` or ``SNR`` result, which carries no
            spectrum to draw.
        """
        from .._plot.hearing import plot_protected_level

        return plot_protected_level(self, ax=ax, language=language, **kwargs)


def octave_band_protected_level(
    noise_levels: Sequence[float] | np.ndarray,
    apv: Sequence[float] | np.ndarray | AssumedProtectionResult,
    *,
    frequencies: Sequence[float] | np.ndarray | None = None,
    a_weighting: Sequence[float] | np.ndarray | None = None,
) -> ProtectedLevelResult:
    r"""Effective A-weighted level by the octave-band method (Formula (2)).

    The most faithful of the three methods, and the only one that sees the
    shape of the noise:

    .. math::

       L'_{p,Ax} = 10 \lg \sum_{k=1}^{8}
       10^{0,1\left(L_{p,f(k)} + A_{f(k)} - APV_{f(k)x}\right)} \mathrm{dB} \tag{2}

    The summation runs over the eight octaves from 63 Hz, or over seven from
    125 Hz when 63 Hz data is missing for either the noise or the protector
    (Clause 6). Pass seven values to both arguments for that case.

    :param noise_levels: Octave-band sound pressure levels of the noise,
        :math:`L_{p,f(k)}`, in dB. Unweighted: the A weighting is added here.
    :param apv: Assumed protection values, in dB, or the
        :class:`AssumedProtectionResult` that carries them.
    :param frequencies: Octave-band mid-frequencies, in hertz, or ``None`` for
        the eight bands of Formula (2), or those seven without 63 Hz.
    :param a_weighting: Frequency weighting A at those bands, in dB, or
        ``None`` for :data:`PROTECTOR_A_WEIGHTING`, which is IEC 61672-1:2013 Table 3.
    :return: :class:`ProtectedLevelResult`.
    :raises ValueError: if the band counts disagree, if any value is not
        finite, or if the bands are neither the eight of Formula (2) nor those
        seven without 63 Hz and ``frequencies`` was not given.
    """
    levels = np.asarray(noise_levels, dtype=np.float64)
    performance: int | None = None
    if isinstance(apv, AssumedProtectionResult):
        performance = apv.performance
        protection = np.asarray(apv.apv, dtype=np.float64)
    else:
        protection = np.asarray(apv, dtype=np.float64)
    for name, values in (("noise_levels", levels), ("apv", protection)):
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            msg = (
                f"'{name}' must be a non-empty one-dimensional array of finite levels."
            )
            raise ValueError(msg)
    if levels.size != protection.size:
        msg = (
            "'noise_levels' and 'apv' must cover the same octave bands; got "
            f"{levels.size} and {protection.size}."
        )
        raise ValueError(msg)
    freqs = _octave_axis(frequencies, levels.size, "octave_band_protected_level")
    if isinstance(apv, AssumedProtectionResult):
        carried = np.asarray(apv.frequencies, dtype=np.float64)
        if carried.size != freqs.size or not np.allclose(carried, freqs):
            msg = (
                "'apv' was computed over a different band axis than this call: "
                f"{carried.tolist()} against {freqs.tolist()}. Formula (2) "
                "subtracts band by band, so the two have to be the same bands "
                "in the same order."
            )
            raise ValueError(msg)
    weighting = _a_weighting_axis(a_weighting, freqs)
    band = levels + weighting - protection
    effective = float(10.0 * np.log10(np.sum(10.0 ** (0.1 * band))))
    l_p_a = float(10.0 * np.log10(np.sum(10.0 ** (0.1 * (levels + weighting)))))
    return ProtectedLevelResult(
        effective_level=effective,
        noise_reduction=l_p_a - effective,
        performance=performance,
        method="octave-band",
        band_levels=np.asarray(band, dtype=np.float64),
        frequencies=freqs,
    )


def _a_weighting_axis(
    a_weighting: Sequence[float] | np.ndarray | None, freqs: np.ndarray
) -> np.ndarray:
    """Frequency weighting A over the bands in play, defaulted from the table.

    The default is keyed to the *frequencies* and not to how many there are: a
    band set that is not the standard octaves has no tabulated weighting, and
    picking one by count alone would silently apply the 63 Hz to 8 kHz values
    to a different axis.

    :param a_weighting: The caller's weighting, or ``None``.
    :param freqs: The band axis the levels sit on, in hertz.
    :return: The weighting as a float array.
    :raises ValueError: if the count does not match, if any value is not
        finite, or if no tabulated default covers the given bands.
    """
    standard = np.asarray(PROTECTOR_OCTAVE_BANDS, dtype=np.float64)
    if a_weighting is None:
        for offset in (0, 1):
            bands = standard[offset:]
            if freqs.size == bands.size and np.allclose(freqs, bands):
                return np.asarray(PROTECTOR_A_WEIGHTING[offset:], dtype=np.float64)
        msg = (
            "pass 'a_weighting' explicitly for these bands: only the eight "
            "octaves of Formula (2), or those seven without 63 Hz, have a "
            f"tabulated weighting; got {freqs.tolist()}."
        )
        raise ValueError(msg)
    weighting = np.asarray(a_weighting, dtype=np.float64)
    if weighting.ndim != 1 or weighting.size != freqs.size:
        msg = (
            "'a_weighting' must be one value per band; got "
            f"{weighting.size} for {freqs.size} bands."
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(weighting)):
        msg = "'a_weighting' must contain only finite values, in dB."
        raise ValueError(msg)
    return weighting


@dataclass(frozen=True)
class HMLRatingResult:
    r"""The three ``HML`` attenuation values of a protector (Clause 7.2).

    :ivar high: :math:`H_x`, the high-frequency value, in dB, unrounded.
    :ivar medium: :math:`M_x`, the medium-frequency value, in dB, unrounded.
    :ivar low: :math:`L_x`, the low-frequency value, in dB, unrounded.
    :ivar subject_h: :math:`H_j` per test subject, in dB (Formula (12)).
    :ivar subject_m: :math:`M_j` per test subject, in dB (Formula (13)).
    :ivar subject_l: :math:`L_j` per test subject, in dB (Formula (14)).
    :ivar predicted_reduction: :math:`PNR_{ji}` per subject and reference
        noise, in dB, a ``(subjects, 8)`` grid (Formula (15)).
    :ivar performance: The protection performance ``x``, in per cent.
    :ivar alpha: The Table 1 constant that ``performance`` selected.
    """

    high: float
    medium: float
    low: float
    subject_h: np.ndarray
    subject_m: np.ndarray
    subject_l: np.ndarray
    predicted_reduction: np.ndarray
    performance: int
    alpha: float

    @property
    def reported(self) -> tuple[int, int, int]:
        """``(H, M, L)`` rounded the way Clause 7.2 reports them.

        :return: The three values as integers, halves away from zero.
        """
        return (
            _round_half_up(self.high),
            _round_half_up(self.medium),
            _round_half_up(self.low),
        )

    def plot(self, ax: Axes | None = None, language: str = "en", **kwargs: Any) -> Axes:
        """Draw the predicted noise level reduction against ``LpC - LpA``.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the ``PNR`` curve.
        :return: The axes.
        """
        from .._plot.hearing import plot_hml_rating

        return plot_hml_rating(self, ax=ax, language=language, **kwargs)


def hml_rating(
    attenuation: Sequence[Sequence[float]] | np.ndarray, *, performance: int = 84
) -> HMLRatingResult:
    r"""The ``H``, ``M`` and ``L`` values of a protector (Formulas (3) to (15)).

    Clause 7.2 fits the protector against the eight reference noises of
    Table 2. For each subject and each noise it forms the predicted noise level
    reduction

    .. math::

       PNR_{ji} = 100\ \mathrm{dB} - 10 \lg \sum_{k=2}^{8}
       10^{0,1\left(L_{p,\mathrm{A}f(k)i} - a_{jf(k)}\right)} \mathrm{dB} \tag{15}

    and collapses the eight into three, weighted by the empirical constants
    :math:`d_i` of Table 2:

    .. math::

       H_j = 0{,}25 \sum_{i=1}^{4} PNR_{ji} - 0{,}48 \sum_{i=1}^{4} d_i PNR_{ji} \tag{12}

       M_j = 0{,}25 \sum_{i=5}^{8} PNR_{ji} - 0{,}16 \sum_{i=5}^{8} d_i PNR_{ji} \tag{13}

       L_j = 0{,}25 \sum_{i=5}^{8} PNR_{ji} + 0{,}23 \sum_{i=5}^{8} d_i PNR_{ji} \tag{14}

    The four quiet-spectrum noises carry ``H`` and the four loud-spectrum ones
    carry ``M`` and ``L``, which is the split the formulas print. Each index is
    then reduced by its own spread across subjects exactly as Formula (1)
    reduces the attenuation (Formulas (3) to (11)).

    :param attenuation: A ``(subjects, bands)`` grid of sound attenuation
        values, in dB. Formula (15) reads 125 Hz to 8000 Hz, so a grid of eight
        bands is taken to start at 63 Hz and its first column is dropped; a
        grid of seven is taken to start at 125 Hz.
    :param performance: The protection performance ``x``, in per cent, from
        Table 1.
    :return: :class:`HMLRatingResult`.
    :raises ValueError: if the grid is not two-dimensional, holds fewer than
        two subjects, carries neither seven nor eight bands, or if
        ``performance`` is not one Table 1 tabulates.
    """
    grid = _subject_attenuations(attenuation)
    alpha = _alpha_for(performance)
    bands = _from_125_hz(grid, "hml_rating")
    noises = np.asarray(HML_REFERENCE_NOISES, dtype=np.float64)
    weights = np.asarray(HML_REFERENCE_D, dtype=np.float64)
    # (subjects, noises): Formula (15) with the reference level of the table.
    summed = np.sum(10.0 ** (0.1 * (noises[None, :, :] - bands[:, None, :])), axis=2)
    pnr = _REFERENCE_TOTAL - 10.0 * np.log10(summed)
    quiet, loud = pnr[:, 0:4], pnr[:, 4:8]
    d_quiet, d_loud = weights[0:4], weights[4:8]
    subject_h = 0.25 * quiet.sum(axis=1) - 0.48 * (d_quiet * quiet).sum(axis=1)
    subject_m = 0.25 * loud.sum(axis=1) - 0.16 * (d_loud * loud).sum(axis=1)
    subject_l = 0.25 * loud.sum(axis=1) + 0.23 * (d_loud * loud).sum(axis=1)
    return HMLRatingResult(
        high=float(subject_h.mean() - alpha * subject_h.std(ddof=1)),
        medium=float(subject_m.mean() - alpha * subject_m.std(ddof=1)),
        low=float(subject_l.mean() - alpha * subject_l.std(ddof=1)),
        subject_h=subject_h,
        subject_m=subject_m,
        subject_l=subject_l,
        predicted_reduction=pnr,
        performance=int(performance),
        alpha=alpha,
    )


def _from_125_hz(grid: np.ndarray, owner: str) -> np.ndarray:
    """The seven bands from 125 Hz that Formulas (15) and (22) read.

    :param grid: A ``(subjects, bands)`` attenuation grid.
    :param owner: Name used in the error message.
    :return: The seven columns from 125 Hz to 8000 Hz.
    :raises ValueError: for any band count but seven or eight.
    """
    if grid.shape[1] == len(PROTECTOR_OCTAVE_BANDS):
        return grid[:, 1:]
    if grid.shape[1] == len(PROTECTOR_OCTAVE_BANDS) - 1:
        return grid
    msg = (
        f"{owner}: 'attenuation' must carry the eight octave bands from 63 Hz "
        "or the seven from 125 Hz, because the reference spectra it is fitted "
        f"against are tabulated over 125 Hz to 8000 Hz; got {grid.shape[1]}."
    )
    raise ValueError(msg)


def hml_protected_level(
    l_p_a: float, l_p_c: float, rating: HMLRatingResult
) -> ProtectedLevelResult:
    r"""Effective A-weighted level by the ``HML`` method (Formulas (16) to (18)).

    Two straight segments through the three anchors, in
    :math:`(L_{p,C} - L_{p,A})`:

    .. math::

       PNR_x = M_x - \frac{H_x - M_x}{4}(L_{p,C} - L_{p,A} - 2\ \mathrm{dB})
       \quad\text{for } (L_{p,C} - L_{p,A}) \leq 2\ \mathrm{dB} \tag{16}

       PNR_x = M_x - \frac{M_x - L_x}{8}(L_{p,C} - L_{p,A} - 2\ \mathrm{dB})
       \quad\text{for } (L_{p,C} - L_{p,A}) > 2\ \mathrm{dB} \tag{17}

       L'_{p,Ax} = L_{p,A} - PNR_x \tag{18}

    Both branches pass through :math:`M_x` at :math:`+2` dB, which is where the
    medium-frequency value is defined (Clause 3.6). The three values that enter
    them are the **rounded** ones: Clause 7.2 rounds :math:`H_x`, :math:`M_x`
    and :math:`L_x` to the nearest integer, so that is what a protector is
    published with and what this consumes, whatever the unrounded fit behind
    them was. Clause 7.3 allows the unweighted level in place of the
    C-weighted one, which for very low-frequency noise returns a higher, safer
    :math:`L'_{p,Ax}`.

    :param l_p_a: A-weighted sound pressure level of the noise, in dB.
    :param l_p_c: C-weighted sound pressure level of the noise, in dB.
    :param rating: The protector's :class:`HMLRatingResult`.
    :return: :class:`ProtectedLevelResult`.
    :raises ValueError: if either level is not finite.
    """
    a_level, c_level = _finite_levels(l_p_a=l_p_a, l_p_c=l_p_c)
    difference = c_level - a_level
    # Clause 7.2 rounds H, M and L to the nearest integer before they are
    # used, so the three that go into Formulas (16) and (17) are the reported
    # ones and not the unrounded fit behind them.
    high, medium, low = rating.reported
    if difference <= _HML_BREAK:
        slope = (high - medium) / _HML_HIGH_SPAN
    else:
        slope = (medium - low) / _HML_LOW_SPAN
    reduction = medium - slope * (difference - _HML_BREAK)
    return ProtectedLevelResult(
        effective_level=a_level - reduction,
        noise_reduction=reduction,
        performance=rating.performance,
        method="HML",
    )


def _finite_levels(**levels: float) -> tuple[float, ...]:
    """Every named level as a finite float, or a refusal naming the first bad one.

    :param levels: The levels to check, by argument name.
    :return: The levels as floats, in the order given.
    :raises ValueError: for a value that is not finite.
    """
    out = []
    for name, value in levels.items():
        number = float(value)
        if not np.isfinite(number):
            msg = f"'{name}' must be a finite sound pressure level, in dB."
            raise ValueError(msg)
        out.append(number)
    return tuple(out)


@dataclass(frozen=True)
class SNRRatingResult:
    r"""The single number rating of a protector (Clause 8.2).

    :ivar snr: :math:`SNR_x`, in dB, unrounded.
    :ivar subject_snr: :math:`SNR_j` per test subject, in dB (Formula (22)).
    :ivar mean: :math:`SNR_m`, in dB (Formula (20)).
    :ivar standard_deviation: :math:`SNR_s`, in dB (Formula (21)).
    :ivar performance: The protection performance ``x``, in per cent.
    :ivar alpha: The Table 1 constant that ``performance`` selected.
    """

    snr: float
    subject_snr: np.ndarray
    mean: float
    standard_deviation: float
    performance: int
    alpha: float

    @property
    def reported(self) -> int:
        """:attr:`snr` rounded the way Clause 8.2 reports it.

        :return: The nearest integer, halves away from zero.
        """
        return _round_half_up(self.snr)

    def plot(self, ax: Axes | None = None, language: str = "en", **kwargs: Any) -> Axes:
        """Draw the per-subject ratings the single number was reduced from.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the per-subject bars.
        :return: The axes.
        """
        from .._plot.hearing import plot_snr_rating

        return plot_snr_rating(self, ax=ax, language=language, **kwargs)


def snr_rating(
    attenuation: Sequence[Sequence[float]] | np.ndarray, *, performance: int = 84
) -> SNRRatingResult:
    r"""The single number rating of a protector (Formulas (19) to (22)).

    One reference noise instead of eight, and one number instead of three:

    .. math::

       SNR_j = 100\ \mathrm{dB} - 10 \lg \sum_{k=2}^{8}
       10^{0,1\left(L_{p,\mathrm{A}f(k)} - a_{jf(k)}\right)} \tag{22}

       SNR_x = SNR_m - \alpha\, SNR_s \tag{19}

    where :math:`L_{p,\mathrm{A}f(k)}` is the pink noise of Table 3, whose
    C-weighted level is 100 dB. Because the reference noise is fixed, the
    rating says nothing about the shape of the noise it will meet, which is
    what the ``HML`` method's three values recover.

    :param attenuation: A ``(subjects, bands)`` grid of sound attenuation
        values, in dB, over the eight octaves from 63 Hz or the seven from
        125 Hz.
    :param performance: The protection performance ``x``, in per cent, from
        Table 1.
    :return: :class:`SNRRatingResult`.
    :raises ValueError: if the grid is not two-dimensional, holds fewer than
        two subjects, carries neither seven nor eight bands, or if
        ``performance`` is not one Table 1 tabulates.
    """
    grid = _subject_attenuations(attenuation)
    alpha = _alpha_for(performance)
    bands = _from_125_hz(grid, "snr_rating")
    pink = np.asarray(PINK_NOISE_A_WEIGHTED, dtype=np.float64)
    per_subject = _REFERENCE_TOTAL - 10.0 * np.log10(
        np.sum(10.0 ** (0.1 * (pink[None, :] - bands)), axis=1)
    )
    mean = float(per_subject.mean())
    spread = float(per_subject.std(ddof=1))
    return SNRRatingResult(
        snr=mean - alpha * spread,
        subject_snr=per_subject,
        mean=mean,
        standard_deviation=spread,
        performance=int(performance),
        alpha=alpha,
    )


def snr_protected_level(
    rating: SNRRatingResult,
    *,
    l_p_c: float | None = None,
    l_p_a: float | None = None,
    c_minus_a: float | None = None,
) -> ProtectedLevelResult:
    r"""Effective A-weighted level by the ``SNR`` method (Formulas (23) and (24)).

    .. math::

       L'_{p,Ax} = L_{p,C} - SNR_x \tag{23}

       L'_{p,Ax} = L_{p,A} + (L_{p,C} - L_{p,A}) - SNR_x \tag{24}

    Formula (24) is Formula (23) with the C-weighted level reassembled from an
    A-weighted measurement and an estimate of the difference, for the common
    case where only the A-weighted level was recorded. Pass ``l_p_c``, or pass
    ``l_p_a`` together with ``c_minus_a``. Clause 8.3 allows the unweighted
    level in place of the C-weighted one, which for very low-frequency noise
    returns a higher, safer :math:`L'_{p,Ax}`.

    The rating is used as Clause 8.2 reports it, rounded to the nearest
    integer.

    :param rating: The protector's :class:`SNRRatingResult`.
    :param l_p_c: C-weighted sound pressure level of the noise, in dB, for
        Formula (23).
    :param l_p_a: A-weighted sound pressure level of the noise, in dB, for
        Formula (24).
    :param c_minus_a: The difference :math:`(L_{p,C} - L_{p,A})`, in dB, for
        Formula (24).
    :return: :class:`ProtectedLevelResult`.
    :raises ValueError: if neither pairing is complete, if both are given, or
        if any level is not finite.
    """
    a_level: float | None = None
    if l_p_c is not None and l_p_a is None and c_minus_a is None:
        (c_level,) = _finite_levels(l_p_c=l_p_c)
    elif l_p_c is None and l_p_a is not None and c_minus_a is not None:
        a_level, difference = _finite_levels(l_p_a=l_p_a, c_minus_a=c_minus_a)
        c_level = a_level + difference
    else:
        msg = (
            "give either 'l_p_c' (Formula (23)) or both 'l_p_a' and "
            "'c_minus_a' (Formula (24)), and not both pairings: the second is "
            "the first with the C-weighted level reassembled."
        )
        raise ValueError(msg)
    reported = rating.reported
    effective = c_level - reported
    # Formula (23) is handed the C-weighted level alone, so LpA is unknown and
    # there is no predicted noise level reduction to report: c_level - effective
    # is the rating back again, which is not what this field means.
    reduction = None if a_level is None else a_level - effective
    return ProtectedLevelResult(
        effective_level=effective,
        noise_reduction=reduction,
        performance=rating.performance,
        method="SNR",
    )
