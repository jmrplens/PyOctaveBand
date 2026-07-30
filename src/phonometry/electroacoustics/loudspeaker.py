#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Rated loudspeaker characteristics (IEC 60268-5).

A loudspeaker measurement/rating report gathers the *rated characteristics*
IEC 60268-5:2003+A1:2007 defines around a measured on-axis response: the
**characteristic sensitivity** referred to 1 W input at 1 m (clauses 20.3/20.4),
the **effective frequency range** read from the on-axis response against a
-10 dB band (clause 21.2), the **rated impedance** and its modulus curve
(clause 16), the **total harmonic distortion** against frequency (clause 24.1)
and the **directional / polar response** (clause 23). This module bundles those
into a single :class:`LoudspeakerCharacteristics` result whose ``report`` method
renders the IEC 60268-5 rated-characteristics fiche, with the characteristic
graphs laid out to the IEC 60263:1982 scale conventions.

Two of the rated characteristics are *computed* from the on-axis response so
the report never merely repeats a manufacturer number:

* **Characteristic sensitivity level** (20.3/20.4). The on-axis response is
  measured at a constant voltage ``U`` and distance ``d``; the sensitivity
  level referred to 1 W into the rated impedance ``R`` at 1 m is

      L_M = L_band + 20 lg(d / d0) + 20 lg(U_p / U),   d0 = 1 m,

  where ``L_band`` is the energetic mean of the on-axis level over a stated
  band (20.1.2.4: the r.m.s. of the band pressures) and ``U_p = sqrt(R * P0)``
  with ``P0 = 1 W`` is the voltage that drives 1 W into ``R`` (20.3.2). With the
  default drive ``U = sqrt(R)`` at ``d = 1 m`` the two corrections vanish and
  the sensitivity level equals the band mean, which for ``R = 8`` ohm is the
  familiar "dB / 2.83 V @ 1 m" figure. This is the clean-room oracle: a flat
  ``L0`` response driven at ``sqrt(R)`` volts and 1 m returns ``L0`` exactly,
  and a doubled voltage returns ``L0 - 6,02`` dB.

* **Effective frequency range** (21.2). The range of frequencies for which the
  on-axis response is not more than 10 dB below the level averaged over a
  one-octave band in the region of maximum sensitivity. Troughs narrower than
  1/9 octave at the -10 dB level are neglected. The band edges are the
  interpolated frequencies where the response last crosses the -10 dB
  threshold on either side of the peak, which is the second clean-room oracle:
  a response crossing the threshold at chosen frequencies returns exactly
  those frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.validation import require_positive

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray

    from .._report.metadata import ReportMetadata
    from .piston import RadiatingPistonResult
    from .swept_sine import SweptSineDistortionResult

#: Standard reference sound pressure, 20 uPa (IEC 60268-5 20.2).
_P_REF = 20e-6
#: One-octave half-width factor sqrt(2): a one-octave band centred at ``f`` is
#: ``[f / sqrt(2), f * sqrt(2)]`` (IEC 60268-5 21.2 reference band).
_OCTAVE_HALF = float(np.sqrt(2.0))
#: Width below which a trough in the response is neglected when the effective
#: frequency range is determined: one-third of a 1/3 octave (IEC 60268-5 21.2).
_MIN_TROUGH_OCTAVES = 1.0 / 9.0
#: Level drop, in decibels, defining the effective frequency range (21.2).
_EFFECTIVE_DROP_DB = 10.0


def _as_curve(
    frequencies: ArrayLike, values: ArrayLike, name: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate a paired (frequencies, values) curve and sort it by frequency."""
    f = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    v = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if f.ndim != 1 or v.ndim != 1 or f.shape != v.shape:
        raise ValueError(f"'{name}' frequencies and values must be 1-D and equal length.")
    if f.size < 2:
        raise ValueError(f"'{name}' needs at least two frequency points.")
    if np.any(f <= 0.0) or not np.all(np.isfinite(f)):
        raise ValueError(f"'{name}' frequencies must be positive and finite.")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"'{name}' values must be finite.")
    order = np.argsort(f)
    return f[order], v[order]


def _energetic_mean_db(levels: NDArray[np.float64]) -> float:
    """Energetic (r.m.s.-pressure) mean of a set of levels, in dB (20.1.2.4).

    The mean is unweighted over the samples handed in, so the response grid
    itself is the frequency weighting: a logarithmically spaced grid weights
    every octave equally, while a linearly spaced grid concentrates its
    samples in, and therefore over-weights, the top octaves of a band.
    """
    return float(10.0 * np.log10(np.mean(10.0 ** (levels / 10.0))))


def _reference_level(
    frequencies: NDArray[np.float64], spl_db: NDArray[np.float64]
) -> float:
    """Level averaged over a one-octave band in the region of maximum sensitivity.

    The reference is the largest energetic (r.m.s.-pressure) mean over any
    one-octave band ``[f / sqrt(2), f * sqrt(2)]`` centred on a response sample
    (IEC 60268-5 21.2: "averaged over a bandwidth of one octave in the region
    of maximum sensitivity"). The clause does not pin down how the "region of
    maximum sensitivity" is located; this implementation reads it as the
    one-octave band that maximises the average. That is a conservative
    interpretation: it can only raise the reference, hence the -10 dB
    threshold, hence never widens the effective range beyond what any other
    placement of the octave would give. It also makes the reference
    independent of where a flat plateau's argmax happens to fall.
    """
    best = -np.inf
    for f_c in frequencies:
        in_band = (frequencies >= f_c / _OCTAVE_HALF) & (frequencies <= f_c * _OCTAVE_HALF)
        if np.any(in_band):
            best = max(best, _energetic_mean_db(spl_db[in_band]))
    return float(best)


def _log2(x: float) -> float:
    return float(np.log2(x))


def _threshold_crossing(
    f_in: float, l_in: float, f_out: float, l_out: float, threshold: float
) -> float:
    """Frequency where the level equals ``threshold``, linear in log-freq vs dB.

    ``(f_in, l_in)`` is the in-band point (``l_in >= threshold``) and
    ``(f_out, l_out)`` the neighbouring out-of-band point (``l_out < threshold``).
    """
    if l_in == l_out:  # pragma: no cover - defensive (guarded by the >= / < split)
        return f_in
    frac = (l_in - threshold) / (l_in - l_out)
    return float(2.0 ** (_log2(f_in) + frac * (_log2(f_out) - _log2(f_in))))


def _fill_narrow_troughs(
    above: NDArray[np.bool_],
    frequencies: NDArray[np.float64],
    spl_db: NDArray[np.float64],
    threshold: float,
) -> None:
    """Mark below-threshold runs narrower than 1/9 octave as in-band (21.2).

    Mutates ``above`` in place. A run is measured by its two threshold crossings
    (crossing to crossing); interior runs only, since an edge run has no outer
    crossing to bracket it and never counts as a trough.
    """
    n = above.size
    i = 0
    while i < n:
        if above[i]:
            i += 1
            continue
        j = i
        while j < n and not above[j]:
            j += 1
        if 0 < i and j < n:
            lo_f = _threshold_crossing(
                frequencies[i - 1], spl_db[i - 1], frequencies[i], spl_db[i], threshold
            )
            hi_f = _threshold_crossing(
                frequencies[j - 1], spl_db[j - 1], frequencies[j], spl_db[j], threshold
            )
            if _log2(hi_f / lo_f) < _MIN_TROUGH_OCTAVES:
                above[i:j] = True
        i = j


def _band_edge(
    frequencies: NDArray[np.float64],
    spl_db: NDArray[np.float64],
    threshold: float,
    edge: int,
    neighbour: int,
) -> float:
    """Interpolated threshold crossing between the in-band ``edge`` and ``neighbour``.

    When the in-band region already reaches the measured band end (no outer
    neighbour), the measured edge frequency is returned.
    """
    if not 0 <= neighbour < frequencies.size:
        return float(frequencies[edge])
    return _threshold_crossing(
        frequencies[edge], spl_db[edge], frequencies[neighbour], spl_db[neighbour],
        threshold,
    )


def _effective_range(
    frequencies: NDArray[np.float64], spl_db: NDArray[np.float64], reference: float
) -> tuple[float, float]:
    """Effective frequency range against the ``reference - 10 dB`` band (21.2).

    Below-threshold excursions narrower than 1/9 octave at the -10 dB level are
    neglected; the edges are the interpolated threshold crossings on either
    side of the peak, or the measured band edge where the response never drops.
    """
    threshold = reference - _EFFECTIVE_DROP_DB
    above = np.asarray(spl_db >= threshold)
    _fill_narrow_troughs(above, frequencies, spl_db, threshold)

    n = above.size
    peak = int(np.argmax(spl_db))
    if not above[peak]:  # pragma: no cover - the peak is always >= threshold
        return float(frequencies[peak]), float(frequencies[peak])
    lo_idx = peak
    while lo_idx > 0 and above[lo_idx - 1]:
        lo_idx -= 1
    hi_idx = peak
    while hi_idx < n - 1 and above[hi_idx + 1]:
        hi_idx += 1
    lower = _band_edge(frequencies, spl_db, threshold, lo_idx, lo_idx - 1)
    upper = _band_edge(frequencies, spl_db, threshold, hi_idx, hi_idx + 1)
    return lower, upper


@dataclass(frozen=True)
class LoudspeakerCharacteristics:
    """Rated loudspeaker characteristics for an IEC 60268-5 report.

    The on-axis response and the rated impedance are the required inputs; the
    impedance modulus curve, the total-harmonic-distortion curve and the
    directional/polar response are optional panels rendered when supplied. The
    characteristic sensitivity level and the effective frequency range are
    computed from the on-axis response (see the module docstring).

    :ivar frequencies: On-axis response frequency axis, in Hz.
    :ivar spl_db: On-axis sound pressure level (re 20 uPa), in dB, measured at
        :attr:`input_voltage` and :attr:`distance`.
    :ivar rated_impedance: Rated impedance ``R``, in ohm (16.1).
    :ivar input_voltage: Constant drive voltage of the response, in V.
    :ivar distance: Measuring distance of the response, in m.
    :ivar sensitivity_band: Stated band ``(lo, hi)`` for the characteristic
        sensitivity, in Hz.
    :ivar tolerance_db: Half-width of the response tolerance band, in dB.
    :ivar reference_level_db: Level averaged over a one-octave band in the
        region of maximum sensitivity (the effective-range reference), in dB.
    :ivar sensitivity_level_db: Characteristic sensitivity level referred to
        1 W into ``R`` at 1 m (20.3/20.4), in dB.
    :ivar effective_range: Computed effective frequency range ``(lo, hi)``, Hz.
    :ivar rated_frequency_range: Manufacturer-stated rated frequency range
        ``(lo, hi)`` in Hz (19.1), or ``None``.
    :ivar rated_noise_power: Rated noise power, in W (18.1), or ``None``.
    :ivar rated_sinusoidal_power: Rated sinusoidal power, in W (18.4), or
        ``None``.
    :ivar resonance_frequency: Resonance frequency, in Hz (19.2), or ``None``.
    :ivar impedance_frequencies: Impedance-curve frequency axis, Hz, or
        ``None``.
    :ivar impedance_modulus: Impedance modulus ``|Z|``, ohm, or ``None``.
    :ivar thd_frequencies: THD-curve frequency axis, Hz, or ``None``.
    :ivar thd_percent: Total harmonic distortion, in %, or ``None``.
    :ivar polar_angles_deg: Polar-response angles, in degrees, or ``None``.
    :ivar polar_db: Polar response relative to the on-axis level, in dB, or
        ``None``.
    :ivar polar_frequency: Frequency of the polar response, in Hz, or ``None``.
    :ivar directivity_index_db: Directivity index at :attr:`polar_frequency`,
        in dB (23.3), or ``None``.
    """

    frequencies: NDArray[np.float64]
    spl_db: NDArray[np.float64]
    rated_impedance: float
    input_voltage: float
    distance: float
    sensitivity_band: tuple[float, float]
    tolerance_db: float
    reference_level_db: float
    sensitivity_level_db: float
    effective_range: tuple[float, float]
    rated_frequency_range: tuple[float, float] | None
    rated_noise_power: float | None
    rated_sinusoidal_power: float | None
    resonance_frequency: float | None
    impedance_frequencies: NDArray[np.float64] | None
    impedance_modulus: NDArray[np.float64] | None
    thd_frequencies: NDArray[np.float64] | None
    thd_percent: NDArray[np.float64] | None
    polar_angles_deg: NDArray[np.float64] | None
    polar_db: NDArray[np.float64] | None
    polar_frequency: float | None
    directivity_index_db: float | None

    @property
    def characteristic_sensitivity_pa(self) -> float:
        """Characteristic sensitivity as a pressure, in Pa (20.3).

        The sound pressure at 1 m for 1 W into the rated impedance:
        ``p_M = p_ref * 10 ** (L_M / 20)``.
        """
        return float(_P_REF * 10.0 ** (self.sensitivity_level_db / 20.0))

    @property
    def minimum_impedance(self) -> float | None:
        """Lowest impedance modulus over the rated range, ohm (16.1), or ``None``.

        IEC 60268-5 16.1 requires the lowest value of the impedance modulus
        *in the rated frequency range* to be not less than 80 % of the rated
        impedance, so the scan uses :attr:`rated_frequency_range` when it is
        supplied. When no rated range is stated, the computed
        :attr:`effective_range` stands in for it; note that the two ranges may
        differ (19.1 NOTE 2), particularly for tweeters or woofers, so an
        impedance dip outside the effective range is only caught when the
        rated range is given.
        """
        if self.impedance_frequencies is None or self.impedance_modulus is None:
            return None
        lo, hi = (
            self.rated_frequency_range
            if self.rated_frequency_range is not None
            else self.effective_range
        )
        in_range = (self.impedance_frequencies >= lo) & (self.impedance_frequencies <= hi)
        band = self.impedance_modulus[in_range]
        if band.size == 0:
            band = self.impedance_modulus
        return float(np.min(band))

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render the IEC 60268-5 loudspeaker-characteristics fiche to a PDF.

        Writes a one-page rated-characteristics data sheet: the standard-basis
        line (IEC 60268-5:2003+A1:2007, graphs to IEC 60263:1982), an optional
        metadata header, the rated-characteristics table beside the on-axis
        response with its tolerance band and effective-range markers, the
        impedance, total-harmonic-distortion and polar-directivity panels for
        the data supplied, a boxed sensitivity/range result, an optional
        sensitivity verdict when a requirement is given, and the footer.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the header identity and, through ``requirement``, a characteristic
            sensitivity level (dB) the verdict row compares against.
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for a uniform ``.report()`` signature; the
            fiche has one layout, so it has no effect.
        :param language: Fiche language: ``"en"`` (default) or ``"es"``.
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"`` or ``language``
            is unknown.
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
        del verbose  # uniform signature; the fiche has a single layout
        from .._report.iec60268_5 import render_iec60268_5_report

        return render_iec60268_5_report(
            self, path, metadata=metadata, language=language
        )

    def plot(
        self, quantity: str = "response", ax: Axes | None = None, *,
        language: str = "en", **kwargs: Any,
    ) -> Axes:
        """Plot one IEC 60268-5 loudspeaker rated characteristic on one axes.

        One concept per figure, drawn by the same shared renderer the
        ``.report()`` fiche composes: ``"response"`` (the on-axis SPL response
        with its tolerance band and effective-range markers, the default),
        ``"impedance"`` (the modulus ``|Z|`` with the rated and 80 %-of-rated
        lines), ``"thd"`` (total harmonic distortion against frequency) and
        ``"directivity"`` (the polar response on the 25 dB reference circle).

        :param quantity: Which characteristic to plot (see above).
        :param ax: Existing axes to draw on, or ``None`` for a fresh figure (a
            polar axes is created for ``"directivity"``).
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :return: The axes the characteristic was drawn on.
        :raises ValueError: If ``quantity`` or ``language`` is unknown, or the
            result carries no data for ``quantity``.
        :raises ImportError: If matplotlib is not installed
            (``pip install phonometry[plot]``).
        """
        from .._i18n import check_language
        from .._plot.electroacoustics import plot_loudspeaker_characteristics

        check_language(language)
        return plot_loudspeaker_characteristics(
            self, quantity, ax=ax, language=language, **kwargs
        )


def _polar_from_directivity(
    directivity: RadiatingPistonResult, polar_frequency: float | None
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float | None]:
    """Derive (angles_deg, relative_db, frequency, DI) from a piston result."""
    if directivity.angles is None or directivity.directivity is None:
        raise ValueError(
            "'directivity' must be a radiating-piston result computed with "
            "'angles' so it carries a directivity pattern."
        )
    freqs = np.asarray(directivity.frequencies, dtype=np.float64)
    target = float(polar_frequency) if polar_frequency is not None else float(freqs[freqs.size // 2])
    idx = int(np.argmin(np.abs(freqs - target)))
    pattern = np.asarray(directivity.directivity[idx], dtype=np.float64)
    with np.errstate(divide="ignore"):
        rel_db = 20.0 * np.log10(np.abs(pattern))
    rel_db[~np.isfinite(rel_db)] = -120.0
    angles_deg = np.degrees(np.asarray(directivity.angles, dtype=np.float64))
    di = float(np.asarray(directivity.directivity_index, dtype=np.float64)[idx])
    return angles_deg, rel_db, float(freqs[idx]), di


def _optional_positive(value: float | None, name: str) -> float | None:
    """Validate a supplied rated value as finite and positive, else keep ``None``."""
    return require_positive(value, name) if value is not None else None


def _resolve_sensitivity_band(
    f: NDArray[np.float64],
    spl: NDArray[np.float64],
    sensitivity_band: tuple[float, float] | None,
) -> tuple[float, float]:
    """Resolve the characteristic-sensitivity band, defaulting to the peak octave."""
    if sensitivity_band is None:
        f_peak = float(f[int(np.argmax(spl))])
        return (f_peak / _OCTAVE_HALF, f_peak * _OCTAVE_HALF)
    lo, hi = float(sensitivity_band[0]), float(sensitivity_band[1])
    if not 0.0 < lo < hi:
        raise ValueError("'sensitivity_band' must be a positive (lo, hi) with lo < hi.")
    return (lo, hi)


def _characteristic_sensitivity_level(
    f: NDArray[np.float64],
    spl: NDArray[np.float64],
    band: tuple[float, float],
    r: float,
    u: float,
    d: float,
) -> float:
    """Band-mean on-axis level referred to 1 W into ``R`` at 1 m (20.3/20.4)."""
    in_band = (f >= band[0]) & (f <= band[1])
    if not np.any(in_band):
        raise ValueError("'sensitivity_band' selects no on-axis response samples.")
    band_level = _energetic_mean_db(spl[in_band])
    # The drive-voltage and distance corrections; U_p = sqrt(R) drives 1 W into R.
    return float(
        band_level + 20.0 * np.log10(d) + 20.0 * np.log10(float(np.sqrt(r)) / u)
    )


def _resolve_impedance(
    impedance: tuple[ArrayLike, ArrayLike] | None,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Validate the optional impedance-modulus curve (16.2)."""
    if impedance is None:
        return None, None
    imp_f, imp_z = _as_curve(impedance[0], impedance[1], "impedance")
    if np.any(imp_z <= 0.0):
        raise ValueError("'impedance' modulus must be positive.")
    return imp_f, imp_z


def _resolve_distortion(
    distortion: SweptSineDistortionResult | tuple[ArrayLike, ArrayLike] | None,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Resolve the optional THD-against-frequency curve, in % (24.1)."""
    if distortion is None:
        return None, None
    if isinstance(distortion, tuple):
        thd_f, thd_p = _as_curve(distortion[0], distortion[1], "distortion")
    else:
        thd_f = np.asarray(distortion.thd_frequencies, dtype=np.float64)
        thd_p = np.asarray(distortion.thd, dtype=np.float64) * 100.0
    if np.any(thd_p < 0.0):
        raise ValueError("'distortion' THD values must be non-negative.")
    return thd_f, thd_p


def _resolve_polar(
    directivity: RadiatingPistonResult | None,
    polar: tuple[ArrayLike, ArrayLike] | None,
    polar_frequency: float | None,
    directivity_index_db: float | None,
) -> tuple[
    NDArray[np.float64] | None, NDArray[np.float64] | None, float | None, float | None
]:
    """Resolve the optional polar response and directivity index (23.1/23.3)."""
    if directivity is not None:
        return _polar_from_directivity(directivity, polar_frequency)
    if polar is None:
        return None, None, polar_frequency, directivity_index_db
    p_ang = np.atleast_1d(np.asarray(polar[0], dtype=np.float64))
    p_db = np.atleast_1d(np.asarray(polar[1], dtype=np.float64))
    if p_ang.ndim != 1 or p_ang.shape != p_db.shape:
        raise ValueError("'polar' angles and levels must be 1-D and equal length.")
    if not (np.all(np.isfinite(p_ang)) and np.all(np.isfinite(p_db))):
        raise ValueError("'polar' angles and levels must be finite.")
    return p_ang, p_db, polar_frequency, directivity_index_db


def loudspeaker_characteristics(
    frequencies: ArrayLike,
    spl_db: ArrayLike,
    rated_impedance: float,
    *,
    input_voltage: float | None = None,
    distance: float = 1.0,
    sensitivity_band: tuple[float, float] | None = None,
    tolerance_db: float = 3.0,
    rated_frequency_range: tuple[float, float] | None = None,
    rated_noise_power: float | None = None,
    rated_sinusoidal_power: float | None = None,
    resonance_frequency: float | None = None,
    impedance: tuple[ArrayLike, ArrayLike] | None = None,
    distortion: SweptSineDistortionResult | tuple[ArrayLike, ArrayLike] | None = None,
    directivity: RadiatingPistonResult | None = None,
    polar: tuple[ArrayLike, ArrayLike] | None = None,
    polar_frequency: float | None = None,
    directivity_index_db: float | None = None,
) -> LoudspeakerCharacteristics:
    """Assemble the rated loudspeaker characteristics for an IEC 60268-5 report.

    The characteristic sensitivity level (20.3/20.4) and the effective frequency
    range (21.2) are computed from the on-axis response; the optional impedance,
    distortion and directivity data feed the corresponding report panels.

    :param frequencies: On-axis response frequency axis, in Hz (1-D, > 0).
        Logarithmically spaced samples are strongly recommended: the band
        averages behind the sensitivity level and the effective-range
        reference weight each sample equally, so a linearly spaced grid
        over-weights the high-frequency end of every band.
    :param spl_db: On-axis sound pressure level, in dB re 20 uPa.
    :param rated_impedance: Rated impedance ``R``, in ohm (16.1).
    :param input_voltage: Constant drive voltage of the response, in V; defaults
        to ``sqrt(R)`` (1 W into ``R``, the 2,83 V @ 8 ohm convention).
    :param distance: Measuring distance of the response, in m (default 1).
    :param sensitivity_band: Stated band ``(lo, hi)`` for the characteristic
        sensitivity, in Hz; defaults to the one-octave band in the region of
        maximum sensitivity.
    :param tolerance_db: Half-width of the plotted response tolerance band, in
        dB (default 3).
    :param rated_frequency_range: Manufacturer-stated rated frequency range
        ``(lo, hi)`` in Hz (19.1). When supplied it is also the range over
        which the 16.1 minimum impedance modulus is evaluated.
    :param rated_noise_power: Rated noise power, in W (18.1).
    :param rated_sinusoidal_power: Rated sinusoidal power, in W (18.4).
    :param resonance_frequency: Resonance frequency, in Hz (19.2).
    :param impedance: Impedance curve as ``(frequencies, modulus)`` in
        ``(Hz, ohm)`` (16.2).
    :param distortion: THD against frequency, either as a
        :class:`~phonometry.electroacoustics.SweptSineDistortionResult` (its
        ``thd`` ratio is converted to %) or a ``(frequencies, thd_percent)``
        pair (24.1).
    :param directivity: A
        :class:`~phonometry.electroacoustics.RadiatingPistonResult` computed
        with ``angles`` to supply the polar response and directivity index
        (23.1/23.3).
    :param polar: Polar response as ``(angles_deg, relative_db)`` when it is not
        taken from ``directivity``.
    :param polar_frequency: Frequency of the polar response, in Hz.
    :param directivity_index_db: Directivity index at ``polar_frequency``, in dB
        (23.3), when not taken from ``directivity``.
    :return: A :class:`LoudspeakerCharacteristics`.
    :raises ValueError: If the inputs are invalid.
    """
    f, spl = _as_curve(frequencies, spl_db, "on-axis response")
    r = require_positive(rated_impedance, "rated_impedance")
    u = (
        require_positive(input_voltage, "input_voltage")
        if input_voltage is not None
        else float(np.sqrt(r))
    )
    d = require_positive(distance, "distance")
    tol = require_positive(tolerance_db, "tolerance_db")

    reference = _reference_level(f, spl)
    effective = _effective_range(f, spl, reference)
    band = _resolve_sensitivity_band(f, spl, sensitivity_band)
    sensitivity_level = _characteristic_sensitivity_level(f, spl, band, r, u, d)
    imp_f, imp_z = _resolve_impedance(impedance)
    thd_f, thd_p = _resolve_distortion(distortion)
    p_ang, p_db, p_freq, di = _resolve_polar(
        directivity, polar, polar_frequency, directivity_index_db
    )
    rated_range = (
        (float(rated_frequency_range[0]), float(rated_frequency_range[1]))
        if rated_frequency_range is not None
        else None
    )

    return LoudspeakerCharacteristics(
        frequencies=f,
        spl_db=spl,
        rated_impedance=r,
        input_voltage=u,
        distance=d,
        sensitivity_band=band,
        tolerance_db=tol,
        reference_level_db=reference,
        sensitivity_level_db=sensitivity_level,
        effective_range=effective,
        rated_frequency_range=rated_range,
        rated_noise_power=_optional_positive(rated_noise_power, "rated_noise_power"),
        rated_sinusoidal_power=_optional_positive(
            rated_sinusoidal_power, "rated_sinusoidal_power"
        ),
        resonance_frequency=_optional_positive(
            resonance_frequency, "resonance_frequency"
        ),
        impedance_frequencies=imp_f,
        impedance_modulus=imp_z,
        thd_frequencies=thd_f,
        thd_percent=thd_p,
        polar_angles_deg=p_ang,
        polar_db=p_db,
        polar_frequency=p_freq,
        directivity_index_db=di,
    )
