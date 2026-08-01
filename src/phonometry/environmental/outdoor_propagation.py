#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Outdoor sound propagation: ISO 9613-2:1996 general method of calculation.

This part of ISO 9613 predicts octave-band attenuation of sound propagating
outdoors from a point source to a receiver under conditions favourable to
propagation (moderate downwind, or the equivalent moderate temperature
inversion; ISO 9613-2:1996, clause 5). The equivalent-continuous downwind
octave-band sound pressure level is (ISO 9613-2:1996):

.. math::

   L_{fT}(DW) = L_W + D_c - A \tag{Eq. 3}

with :math:`L_W` the octave-band sound power level, :math:`D_c` the
directivity correction
(directivity index plus a solid-angle index ``DOmega``) and ``A`` the
octave-band attenuation, itself a sum of physical mechanisms:

.. math::

   A = A_{div} + A_{atm} + A_{gr} + A_{bar} + A_{misc} \tag{Eq. 4}

Implemented here are the four general terms of clause 7:

* ``Adiv`` geometrical divergence, :math:`20 \log_{10}(d/d_0) + 11` (Eq. (7));
* ``Aatm`` atmospheric absorption, :math:`\alpha d` (Eq. (8)) with ``alpha``
  the ISO 9613-1 coefficient supplied by :mod:`phonometry.environmental.air_absorption`;
* ``Agr`` ground effect, both the general per-region method of 7.3.1 with the
  Table 3 functions ``a'/b'/c'/d'`` (Eq. (9)) and the alternative simplified
  method of 7.3.2 (Eq. (10));
* ``Abar`` screening by a barrier, :math:`D_z - A_{gr}` with the ``Dz``
  diffraction
  formula of Eq. (14) including the ``C2``/``C3`` factors, the pathlength
  difference ``z`` (Eq. (16)/(17)), the meteorological factor ``Kmet``
  (Eq. (18)) and the 20 dB (single) / 25 dB (double) limits.

The long-term average level follows from the meteorological correction ``Cmet``
(Eq. (6), (21), (22), clause 8). ``Amisc`` (foliage, industrial sites, housing;
annex A) and reflections from vertical obstacles (clause 7.5) are informative and
left to the caller. Accuracy of the method is stated in Table 5 (clause 9): within
+/-1 dB to +/-3 dB for broadband noise up to 1000 m.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._internal.warnings import _warn_renamed
from .air_absorption import air_attenuation

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

#: Reference distance ``d0`` in the divergence term (ISO 9613-2:1996, Eq. (7)), m.
_D0 = 1.0
#: Speed of sound used for the barrier wavelength ``lambda = c/f``
#: (ISO 9613-2:1996, clause 7.5 writes ``lambda = 340/f``), in m/s.
_C_SOUND = 340.0
#: Nominal octave-band midband frequencies of the method (clause 1), in hertz.
DEFAULT_FREQUENCIES: tuple[float, ...] = (63.0, 125.0, 250.0, 500.0, 1000.0,
                                          2000.0, 4000.0, 8000.0)
#: Barrier attenuation limit for single diffraction (clause 7.4), in dB.
_DZ_LIMIT_SINGLE = 20.0
#: Barrier attenuation limit for double diffraction (clause 7.4), in dB.
_DZ_LIMIT_DOUBLE = 25.0

_NOMINAL_BANDS = np.array(DEFAULT_FREQUENCIES, dtype=np.float64)


@dataclass(frozen=True)
class Barrier:
    r"""Screening obstacle for the ISO 9613-2 barrier term (clause 7.4).

    The barrier is described by the diffraction geometry that feeds the
    pathlength-difference equations (16)/(17) directly, which is the cleanest
    match to the ``Dz`` formula of Eq. (14).

    :param source_to_edge: Distance ``dss`` from the source to the (first)
        diffraction edge, in metres (ISO 9613-2:1996, Eq. (16)).
    :param edge_to_receiver: Distance ``dsr`` from the (second) diffraction edge
        to the receiver, in metres.
    :param parallel_distance: Component ``a`` of the source-receiver separation
        parallel to the barrier edge, in metres (0 for a purely 2-D section).
    :param edge_separation: Spacing ``e`` between the two diffraction edges for
        double (thick-barrier) diffraction, in metres; ``None`` selects single
        diffraction (Eq. (16), :math:`C_3 = 1`). When given, Eq. (17) and the
        ``C3`` factor of Eq. (15) are used with the 25 dB limit.
    :param ground_reflections_by_image: When ``True`` the ground reflections are
        assumed to be handled separately by image sources, so
        :math:`C_2 = 40`; otherwise :math:`C_2 = 20` (Eq. (14)).
    :param lateral: When ``True`` the diffraction is around a vertical edge
        (Eq. (13)): :math:`A_{bar} = D_z` (the ground term is not cancelled)
        and :math:`K_{met} = 1`. Default ``False`` selects top-edge
        diffraction (Eq. (12)).
    :param line_of_sight_clear: When ``True`` the line of sight between source
        and receiver passes *above* the top edge: ISO 9613-2:1996 (text after
        Eq. (16)) then gives the path difference ``z`` a negative sign, and
        Eq. (14) is still evaluated (with :math:`K_{met} = 1`, Eq. (18)), so
        ``Dz`` falls continuously from :math:`10 \log_{10} 3 = 4.8` dB at grazing
        to 0 for deeper geometries. The edge distances stay the unsigned
        geometric lengths; only the sign convention of ``z`` changes.
    """

    source_to_edge: float
    edge_to_receiver: float
    parallel_distance: float = 0.0
    edge_separation: float | None = None
    ground_reflections_by_image: bool = False
    lateral: bool = False
    line_of_sight_clear: bool = False

    def __post_init__(self) -> None:
        if self.source_to_edge < 0.0 or self.edge_to_receiver < 0.0:
            raise ValueError("Barrier edge distances must be non-negative.")
        if self.parallel_distance < 0.0:
            raise ValueError("'parallel_distance' must be non-negative.")
        if self.edge_separation is not None and self.edge_separation <= 0.0:
            raise ValueError(
                "'edge_separation' must be positive; use None for single diffraction."
            )

    @property
    def is_double(self) -> bool:
        """Whether double diffraction (Eq. (17)/(15)) applies (``e`` given)."""
        return self.edge_separation is not None


@dataclass(frozen=True)
class SourceEmission:
    r"""Source emission terms for the ISO 9613-2 downwind receiver level (Eq. (3)).

    Passed to :meth:`OutdoorAttenuation.report` so the prediction fiche can box
    the A-weighted downwind level at the receiver from an octave-band
    attenuation breakdown. The level is composed as
    :math:`L_{fT}(DW) = L_W + D_c - A`
    with the directivity correction :math:`D_c = D_i + D_\Omega`
    (ISO 9613-2:1996, Eq. (3)); an optional meteorological correction ``cmet``
    is subtracted for the long-term average level (Eq. (6)).

    This report-time object keeps the emission out of
    :func:`outdoor_propagation_attenuation` (which stays purely an attenuation
    calculation), so the receiver level is a presentation concern of the fiche.

    :param sound_power_level: Octave-band source sound power level ``Lw`` (dB re
        1 pW), one value per band of the attenuation result.
    :param directivity_index: Source directivity index ``Di``, in decibels.
    :param d_omega: Solid-angle index ``DOmega``, in decibels (see
        :func:`directivity_omega` for the alternative ground method).
    :param cmet: Optional meteorological correction ``Cmet`` (dB), obtained from
        :func:`meteorological_correction`; ``None`` reports the downwind level
        ``LfT(DW)`` directly (:math:`C_{met} = 0`).
    """

    sound_power_level: ArrayLike
    directivity_index: float = 0.0
    d_omega: float = 0.0
    cmet: float | None = None


@dataclass(frozen=True)
class OutdoorAttenuation:
    """Per-octave-band ISO 9613-2 attenuation breakdown (clause 7).

    Every array is aligned with :attr:`frequencies`. The terms sum, band by band,
    to :attr:`a_total` (ISO 9613-2:1996, Eq. (4) without the informative
    ``Amisc``), so users can see the divergence, atmospheric, ground and barrier
    contributions separately.

    :ivar frequencies: Nominal octave-band midband frequencies, in hertz.
    :ivar a_div: Geometrical divergence ``Adiv`` (Eq. (7)), in dB, per band
        (identical across bands).
    :ivar a_atm: Atmospheric absorption ``Aatm`` (Eq. (8)), in dB, per band.
    :ivar a_gr: Ground effect ``Agr`` (Eq. (9) or (10)), in dB, per band. A
        negative value denotes a net gain from ground reflection.
    :ivar a_bar: Screening ``Abar`` (Eq. (12)/(13)), in dB, per band (>= 0).
    :ivar a_total: Total attenuation ``A`` (Eq. (4)), in dB, per band.
    :ivar d_omega: Solid-angle directivity index ``DOmega`` (Eq. (11)), in dB;
        non-zero only for the alternative ground method of 7.3.2.
    """

    frequencies: NDArray[np.float64]
    a_div: NDArray[np.float64]
    a_atm: NDArray[np.float64]
    a_gr: NDArray[np.float64]
    a_bar: NDArray[np.float64]
    a_total: NDArray[np.float64]
    d_omega: NDArray[np.float64]

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the stacked per-band attenuation terms with the total.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.environmental import plot_outdoor_attenuation

        return plot_outdoor_attenuation(self, ax=ax, language=check_language(language), **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
        source_emission: SourceEmission | None = None,
    ) -> str:
        """Render a one-page ISO 9613-2 outdoor-propagation prediction fiche.

        Writes a prediction sheet (clearly labelled a prediction, not a
        measurement) laid out like an environmental-noise propagation
        calculation: the standard-basis line naming ISO 9613-2:1996 (general
        method, conditions favourable to propagation), an optional metadata
        header (source/situation, client, receiver position, meteorological
        conditions, date), a per-band table of the attenuation terms
        (``Adiv``, ``Aatm``, ``Agr``, ``Abar`` and the total ``A``) and the
        attenuation-breakdown plot, closed by a boxed single result and a footer
        identity/disclaimer block.

        When a ``source_emission`` is supplied, the fiche also lists the source
        power level ``Lw`` and the composed downwind level ``LfT(DW)`` per band
        and boxes the A-weighted downwind level ``LAT(DW)`` at the receiver, with
        an optional PASS/FAIL verdict against a declared limit level (a lower
        level is better). Without it the fiche boxes the octave-band range of
        the total attenuation ``A``.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the header identity (``specimen`` the source/situation, ``client``,
            ``test_room`` the receiver position), the ``temperature`` /
            ``relative_humidity`` / ``pressure`` conditions and the footer
            identity. A supplied ``requirement`` is read as the maximum
            acceptable A-weighted downwind level in dB (used only when a
            ``source_emission`` is given).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When True and a ``source_emission`` is supplied, the
            per-band table adds the A-weighted band level (``LfT(DW)`` plus the
            band A-weighting), whose energy sum is the boxed ``LAT(DW)``.
        :param language: Fiche language: ``"en"`` (default) or ``"es"``.
        :param source_emission: Optional :class:`SourceEmission` (the source
            sound power ``Lw`` and directivity, plus an optional meteorological
            correction) that turns the attenuation breakdown into the boxed
            A-weighted downwind level at the receiver.
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"``, ``language``
            is unknown, or a supplied ``source_emission`` sound power does not
            match the number of frequency bands.
        :raises ImportError: If reportlab or matplotlib is not installed
            (``pip install "phonometry[report,plot]"``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from .._report.iso9613 import render_outdoor_attenuation_report

        return render_outdoor_attenuation_report(
            self, path, metadata=metadata, verbose=verbose, language=language,
            source_emission=source_emission,
        )


# --------------------------------------------------------------------------- #
# Geometrical divergence (7.1) and atmospheric absorption (7.2)
# --------------------------------------------------------------------------- #
def geometric_divergence(distance: float) -> float:
    r"""Attenuation due to geometrical divergence (ISO 9613-2:1996, Eq. (7)).

    Spherical spreading in the free field from a point source:

    .. math::

       A_{div} = 20 \log_{10}(d/d_0) + 11~\text{dB}, \qquad d_0 = 1~\text{m}

    The ``+11`` (:math:`= 10 \log_{10} 4\pi`) sets the sound pressure level at the
    reference distance :math:`d_0 = 1` m from an omnidirectional point source
    (Note 7).

    :param distance: Straight-line source-to-receiver distance ``d``, in metres.
    :return: ``Adiv``, in decibels (51 dB at 100 m, 11 dB at 1 m).
    :raises ValueError: If ``distance`` is not positive.
    """
    if distance <= 0.0:
        raise ValueError("'distance' must be positive.")
    return float(20.0 * np.log10(distance / _D0) + 11.0)


def _resolve_humidity(
    func: str, relative_humidity: float | None, humidity: float | str
) -> float:
    """Resolve the deprecated ``humidity`` alias onto ``relative_humidity``.

    ``stacklevel=4`` skips this helper *and* the public function so the
    :class:`DeprecationWarning` points at the caller's line.
    """
    if not isinstance(humidity, str):
        _warn_renamed(
            f"the 'humidity' keyword of {func}()",
            "'relative_humidity'",
            stacklevel=4,
        )
        if relative_humidity is not None:
            raise ValueError(
                f"{func}() got both 'relative_humidity' and its deprecated "
                "alias 'humidity'; pass only 'relative_humidity'."
            )
        relative_humidity = humidity
    return 70.0 if relative_humidity is None else relative_humidity


def atmospheric_absorption(
    distance: float,
    frequencies: ArrayLike = DEFAULT_FREQUENCIES,
    temperature: float = 20.0,
    relative_humidity: float | None = None,
    pressure: float = 101.325,
    *,
    humidity: float | str = "deprecated",
) -> NDArray[np.float64]:
    r"""Attenuation due to atmospheric absorption (ISO 9613-2:1996, Eq. (8)).

    :math:`A_{atm} = \alpha d` with ``alpha`` the ISO 9613-1 atmospheric
    attenuation
    coefficient (here in dB/m, from :func:`phonometry.air_absorption.air_attenuation`)
    at each octave-band midband frequency. Eq. (8) writes ``alpha`` in dB/km
    with :math:`A_{atm} = \alpha_{\text{dB/km}} \, d / 1000`; the two forms
    are identical.

    ``alpha`` is evaluated at the *exact* base-10 midband frequency behind
    each nominal band label (e.g. 7 943.3 Hz for the "8 kHz" band), the
    convention behind the ISO 9613-2 Table 2 coefficients (they come from
    ISO 9613-1 Table 1 at exact midbands; at 8 kHz the nominal-frequency
    evaluation would run ~1.3 % high). Each supplied frequency is snapped to
    the nearest exact midband.

    :param distance: Source-to-receiver distance ``d``, in metres.
    :param frequencies: Octave-band midband frequencies, in hertz.
    :param temperature: Air temperature, in degrees Celsius.
    :param relative_humidity: Relative humidity, in percent (default 70).
    :param pressure: Atmospheric pressure, in kilopascals.
    :param humidity: Deprecated alias of ``relative_humidity`` (remove in 4.0).
    :return: ``Aatm`` per band, in decibels.
    """
    relative_humidity = _resolve_humidity(
        "atmospheric_absorption", relative_humidity, humidity
    )
    alpha = air_attenuation(
        frequencies, temperature, relative_humidity, pressure, exact_midband=True
    )
    return np.asarray(alpha * distance, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Ground effect (7.3)
# --------------------------------------------------------------------------- #
def _a_prime(h: float, dp: float) -> float:
    """Table 3 function ``a'(h)`` for the 125 Hz band (ISO 9613-2:1996)."""
    return float(
        1.5
        + 3.0 * np.exp(-0.12 * (h - 5.0) ** 2) * (1.0 - np.exp(-dp / 50.0))
        + 5.7 * np.exp(-0.09 * h**2) * (1.0 - np.exp(-2.8e-6 * dp**2))
    )


def _b_prime(h: float, dp: float) -> float:
    """Table 3 function ``b'(h)`` for the 250 Hz band (ISO 9613-2:1996)."""
    return float(1.5 + 8.6 * np.exp(-0.09 * h**2) * (1.0 - np.exp(-dp / 50.0)))


def _c_prime(h: float, dp: float) -> float:
    """Table 3 function ``c'(h)`` for the 500 Hz band (ISO 9613-2:1996)."""
    return float(1.5 + 14.0 * np.exp(-0.46 * h**2) * (1.0 - np.exp(-dp / 50.0)))


def _d_prime(h: float, dp: float) -> float:
    """Table 3 function ``d'(h)`` for the 1000 Hz band (ISO 9613-2:1996)."""
    return float(1.5 + 5.0 * np.exp(-0.9 * h**2) * (1.0 - np.exp(-dp / 50.0)))


# Float keys are safe: lookups always come from _nearest_nominal, which returns
# these exact literals.
_PRIME_BY_BAND = {125.0: _a_prime, 250.0: _b_prime, 500.0: _c_prime,
                  1000.0: _d_prime}


def _nearest_nominal(freq: float) -> float:
    """Nearest nominal octave band to ``freq`` (Table 3 is octave-band only)."""
    idx = int(np.argmin(np.abs(np.log2(_NOMINAL_BANDS / freq))))
    return float(_NOMINAL_BANDS[idx])


def _region_attenuation(nominal: float, g: float, h: float, dp: float) -> float:
    """As or Ar contribution for one band (ISO 9613-2:1996, Table 3, col. 2)."""
    if nominal <= 63.0:
        return -1.5
    if nominal in _PRIME_BY_BAND:
        return -1.5 + g * _PRIME_BY_BAND[nominal](h, dp)
    return -1.5 * (1.0 - g)


def ground_attenuation(
    distance: float,
    source_height: float,
    receiver_height: float,
    frequencies: ArrayLike = DEFAULT_FREQUENCIES,
    ground_source: float = 0.0,
    ground_middle: float = 0.0,
    ground_receiver: float = 0.0,
    projected_distance: float | None = None,
) -> NDArray[np.float64]:
    r"""Ground attenuation by the general per-region method (7.3.1, Eq. (9)).

    :math:`A_{gr} = A_s + A_r + A_m` (source, receiver and middle regions),
    each evaluated
    with the Table 3 expressions and its ground factor ``G`` (0 = hard, 1 =
    porous, in between = porous fraction). For the source region
    :math:`G = G_s` and :math:`h = h_s`; for the receiver region
    :math:`G = G_r` and :math:`h = h_r` (Table 3,
    note 1). The middle-region term uses the overlap factor ``q`` of note 2:

    .. math::

       q = 0 \quad \text{if } d_p \le 30 (h_s + h_r)

       q = 1 - \frac{30 (h_s + h_r)}{d_p} \quad
       \text{if } d_p > 30 (h_s + h_r)

    with :math:`A_m = -3q` at 63 Hz and :math:`A_m = -3q(1 - G_m)` above.

    :param distance: Straight-line source-to-receiver distance ``d``, in metres.
    :param source_height: Source height ``hs`` above ground, in metres.
    :param receiver_height: Receiver height ``hr`` above ground, in metres.
    :param frequencies: Octave-band midband frequencies, in hertz. Table 3 is
        defined for the eight nominal octave bands 63 Hz-8 kHz only; any other
        requested frequency is snapped to the nearest nominal octave band for
        the Table 3 lookup.
    :param ground_source: Ground factor ``Gs`` of the source region ([0, 1]).
    :param ground_middle: Ground factor ``Gm`` of the middle region ([0, 1]).
    :param ground_receiver: Ground factor ``Gr`` of the receiver region ([0, 1]).
    :param projected_distance: Ground-plane projected distance ``dp``, in metres;
        defaults to :math:`\sqrt{d^2 - (h_s - h_r)^2}`.
    :return: ``Agr`` per band, in decibels (negative denotes a net gain).
    :raises ValueError: If a ground factor is outside ``[0, 1]``, ``distance``
        is not positive, a height is negative, or a frequency is not positive.
    """
    for name, g in (("ground_source", ground_source),
                    ("ground_middle", ground_middle),
                    ("ground_receiver", ground_receiver)):
        if not 0.0 <= g <= 1.0:
            raise ValueError(f"'{name}' must be within [0, 1].")
    if distance <= 0.0:
        raise ValueError("'distance' must be positive.")
    if source_height < 0.0 or receiver_height < 0.0:
        raise ValueError("Source and receiver heights must be non-negative.")
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if np.any(freqs <= 0.0):
        raise ValueError("'frequencies' must be positive.")
    dp = _projected_distance(distance, source_height, receiver_height,
                             projected_distance)

    threshold = 30.0 * (source_height + receiver_height)
    q = 0.0 if dp <= threshold else 1.0 - threshold / dp

    out = np.empty_like(freqs)
    for i, f in enumerate(freqs):
        nominal = _nearest_nominal(float(f))
        a_s = _region_attenuation(nominal, ground_source, source_height, dp)
        a_r = _region_attenuation(nominal, ground_receiver, receiver_height, dp)
        a_m = -3.0 * q if nominal <= 63.0 else -3.0 * q * (1.0 - ground_middle)
        out[i] = a_s + a_r + a_m
    return out


def ground_attenuation_alternative(
    distance: float,
    mean_height: float,
) -> float:
    r"""Ground attenuation by the alternative A-weighted method (7.3.2, Eq. (10)).

    Valid only when the A-weighted receiver level alone is of interest, the sound
    propagates over porous or mostly-porous ground and is not a pure tone
    (ISO 9613-2:1996, 7.3.2):

    .. math::

       A_{gr} = 4.8 - \frac{2 h_m}{d} \left[ 17 + \frac{300}{d} \right]
       \ge 0~\text{dB}

    Negative results are replaced by zero. When this method is used, add the
    solid-angle index :func:`directivity_omega` (Eq. (11)) to ``Dc`` in Eq. (3).

    :param distance: Source-to-receiver distance ``d``, in metres.
    :param mean_height: Mean height ``hm`` of the propagation path above the
        ground (:math:`h_m = F/d`, figure 3), in metres.
    :return: ``Agr``, in decibels (>= 0).
    :raises ValueError: If ``distance`` is not positive.
    """
    if distance <= 0.0:
        raise ValueError("'distance' must be positive.")
    agr = 4.8 - (2.0 * mean_height / distance) * (17.0 + 300.0 / distance)
    return float(max(agr, 0.0))


def directivity_omega(
    source_height: float,
    receiver_height: float,
    projected_distance: float,
) -> float:
    r"""Solid-angle directivity index ``DOmega`` (ISO 9613-2:1996, Eq. (11)).

    Accounts for the apparent increase in source power from ground reflection
    near the source when the alternative ground method (Eq. (10)) is used:

    .. math::

       D_\Omega = 10 \log_{10}\!\left\{ 1 +
       \frac{d_p^2 + (h_s - h_r)^2}{d_p^2 + (h_s + h_r)^2} \right\}
       ~\text{dB}

    :param source_height: Source height ``hs``, in metres.
    :param receiver_height: Receiver height ``hr``, in metres.
    :param projected_distance: Ground-plane projected distance ``dp``, in metres.
    :return: ``DOmega``, in decibels (0 to ~3 dB).
    """
    num = projected_distance**2 + (source_height - receiver_height) ** 2
    den = projected_distance**2 + (source_height + receiver_height) ** 2
    return float(10.0 * np.log10(1.0 + num / den))


# --------------------------------------------------------------------------- #
# Screening / barriers (7.4)
# --------------------------------------------------------------------------- #
def _c3_double(lam: float, edge_separation: float) -> float:
    """Double-diffraction factor ``C3`` (ISO 9613-2:1996, Eq. (15))."""
    ratio = (5.0 * lam / edge_separation) ** 2
    return (1.0 + ratio) / ((1.0 / 3.0) + ratio)


def barrier_attenuation(
    barrier: Barrier,
    distance: float,
    frequencies: ArrayLike = DEFAULT_FREQUENCIES,
) -> NDArray[np.float64]:
    r"""Barrier diffraction attenuation ``Dz`` (ISO 9613-2:1996, Eq. (14)).

    .. math::

       D_z = 10 \log_{10}\!\left[ 3 + \frac{C_2}{\lambda} \, C_3 \, z \, K_{met}
       \right] ~\text{dB}

    with :math:`C_2 = 20` (or 40 when ground reflections are handled by image
    sources), :math:`C_3 = 1` for single diffraction or Eq. (15) for double,
    the pathlength difference ``z`` (Eq. (16)/(17)),
    :math:`\lambda = 340/f` and the
    meteorological factor ``Kmet`` (Eq. (18), 1 for :math:`z \le 0`). ``Dz``
    is limited to 20 dB (single) or 25 dB (double). When the line of sight
    passes above the top edge (``Barrier(line_of_sight_clear=True)``) ``z``
    takes a negative sign (ISO 9613-2:1996, text after Eq. (16)) and Eq. (14)
    still applies: ``Dz`` falls continuously from :math:`10 \log_{10} 3 = 4.8` dB
    at grazing (:math:`z = 0`) towards 0 as the clearance deepens, clamped at
    0 (the logarithm's argument is floored at 1 -- a barrier below the sight
    line never amplifies).

    :param barrier: Barrier geometry (:class:`Barrier`).
    :param distance: Straight-line source-to-receiver distance ``d``, in metres.
    :param frequencies: Octave-band midband frequencies, in hertz.
    :return: ``Dz`` per band, in decibels (>= 0).
    :raises ValueError: If ``distance`` or any frequency is not positive.
    """
    if distance <= 0.0:
        raise ValueError("'distance' must be positive.")
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if np.any(freqs <= 0.0):
        raise ValueError("'frequencies' must be positive.")
    dss = barrier.source_to_edge
    dsr = barrier.edge_to_receiver
    a = barrier.parallel_distance
    e = barrier.edge_separation

    if e is not None:
        z = float(np.hypot(dss + dsr + e, a) - distance)
        limit = _DZ_LIMIT_DOUBLE
    else:
        z = float(np.hypot(dss + dsr, a) - distance)
        limit = _DZ_LIMIT_SINGLE
    if barrier.line_of_sight_clear:
        # Eq. (16) sign convention: the diffracted path is still longer than
        # the direct one, but z is given a negative sign when the sight line
        # clears the top edge.
        z = -z

    c2 = 40.0 if barrier.ground_reflections_by_image else 20.0
    if barrier.lateral or z <= 0.0:
        kmet = 1.0  # Eq. (18): Kmet = 1 for z <= 0 and for lateral diffraction
    else:
        kmet = float(np.exp(-(1.0 / 2000.0)
                            * np.sqrt(dss * dsr * distance / (2.0 * z))))

    out = np.empty_like(freqs)
    for i, f in enumerate(freqs):
        lam = _C_SOUND / float(f)
        c3 = _c3_double(lam, e) if e is not None else 1.0
        # For z >= 0 the argument is >= 3; for negative z it decreases below 3
        # and is floored at 1 so Dz >= 0 (Dz -> 0 around z ~ -lambda/10).
        arg = max(3.0 + (c2 / lam) * c3 * z * kmet, 1.0)
        dz = 10.0 * np.log10(arg)
        out[i] = min(dz, limit)
    return out


# --------------------------------------------------------------------------- #
# Meteorological correction (clause 8)
# --------------------------------------------------------------------------- #
def meteorological_correction(
    projected_distance: float,
    source_height: float,
    receiver_height: float,
    c0: float,
) -> float:
    r"""Meteorological correction ``Cmet`` (ISO 9613-2:1996, Eq. (21)/(22)).

    .. math::

       C_{met} = 0 \quad \text{if } d_p \le 10 (h_s + h_r)

       C_{met} = C_0 \left[ 1 - \frac{10 (h_s + h_r)}{d_p} \right] \quad
       \text{if } d_p > 10 (h_s + h_r)

    ``C0`` (dB) reflects local wind and temperature-gradient statistics; practical
    values lie in 0..~5 dB (note 22). Subtract ``Cmet`` from ``LAT(DW)`` for the
    long-term average level (Eq. (6)).

    :param projected_distance: Ground-plane projected distance ``dp``, in metres.
    :param source_height: Source height ``hs``, in metres.
    :param receiver_height: Receiver height ``hr``, in metres.
    :param c0: Meteorological factor ``C0``, in decibels.
    :return: ``Cmet``, in decibels (>= 0 for :math:`C_0 \ge 0`).
    """
    threshold = 10.0 * (source_height + receiver_height)
    if projected_distance <= threshold:
        return 0.0
    return float(c0 * (1.0 - threshold / projected_distance))


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def _projected_distance(
    distance: float,
    source_height: float,
    receiver_height: float,
    projected_distance: float | None,
) -> float:
    """Ground-plane projected distance ``dp`` (default from ``d``, ``hs``, ``hr``)."""
    if projected_distance is not None:
        return projected_distance
    return float(np.sqrt(max(distance**2 - (source_height - receiver_height) ** 2,
                             0.0)))


def _compose_receiver_level(
    sound_power_level: NDArray[np.float64],
    directivity_index: float,
    d_omega: float,
    a_total: NDArray[np.float64],
    cmet: float | None,
) -> NDArray[np.float64]:
    r"""Compose the octave-band receiver level from ``Lw``, ``Dc`` and ``A``.

    :math:`L_{fT} = L_W + D_c - A` (ISO 9613-2:1996, Eq. (3)) with
    :math:`D_c = D_i + D_\Omega`; when a meteorological correction ``cmet`` is
    given it is subtracted band by band to approximate the long-term average
    level (Eq. (6)). Used by :func:`predicted_receiver_level` and by the
    :class:`OutdoorAttenuation` report when a :class:`SourceEmission` is given.
    """
    level = sound_power_level + (directivity_index + d_omega) - a_total
    if cmet is not None:
        level = level - cmet
    return np.asarray(level, dtype=np.float64)


def outdoor_propagation_attenuation(
    distance: float,
    source_height: float,
    receiver_height: float,
    frequencies: ArrayLike = DEFAULT_FREQUENCIES,
    ground_source: float = 0.0,
    ground_middle: float = 0.0,
    ground_receiver: float = 0.0,
    barrier: Barrier | None = None,
    temperature: float = 20.0,
    relative_humidity: float | None = None,
    pressure: float = 101.325,
    projected_distance: float | None = None,
    *,
    humidity: float | str = "deprecated",
) -> OutdoorAttenuation:
    r"""Total octave-band outdoor attenuation (ISO 9613-2:1996, Eq. (4)).

    Assembles the four general terms of clause 7 into
    :math:`A = A_{div} + A_{atm} + A_{gr} + A_{bar}` (the informative
    ``Amisc`` is omitted). The ground effect uses the
    general per-region method (7.3.1). With a barrier, the top-edge insertion
    loss :math:`A_{bar} = D_z - A_{gr}` (Eq. (12)) folds the ground effect of
    the screened path into ``Dz`` (note 13); for a lateral (vertical-edge)
    barrier :math:`A_{bar} = D_z`
    (Eq. (13)) and the ground term is retained.

    :param distance: Straight-line source-to-receiver distance ``d``, in metres.
    :param source_height: Source height ``hs`` above ground, in metres.
    :param receiver_height: Receiver height ``hr`` above ground, in metres.
    :param frequencies: Octave-band midband frequencies, in hertz. The ground
        term snaps each frequency to the nearest nominal octave band (Table 3
        is octave-band only) and the atmospheric term evaluates the exact
        base-10 midband behind it (see :func:`atmospheric_absorption`).
    :param ground_source: Ground factor ``Gs`` of the source region ([0, 1],
        0 = hard, 1 = porous).
    :param ground_middle: Ground factor ``Gm`` of the middle region ([0, 1]).
    :param ground_receiver: Ground factor ``Gr`` of the receiver region ([0, 1]).
    :param barrier: Optional screening obstacle (:class:`Barrier`).
    :param temperature: Air temperature, in degrees Celsius.
    :param relative_humidity: Relative humidity, in percent (default 70).
    :param pressure: Atmospheric pressure, in kilopascals.
    :param projected_distance: Ground-plane projected distance ``dp``, in metres;
        defaults to :math:`\sqrt{d^2 - (h_s - h_r)^2}`.
    :param humidity: Deprecated alias of ``relative_humidity`` (remove in 4.0).
    :return: :class:`OutdoorAttenuation` with the per-band term breakdown.
    :raises ValueError: If ``distance`` is not positive.
    """
    relative_humidity = _resolve_humidity(
        "outdoor_propagation_attenuation", relative_humidity, humidity
    )
    if distance <= 0.0:
        raise ValueError("'distance' must be positive.")
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))

    a_div = np.full_like(freqs, geometric_divergence(distance))
    a_atm = atmospheric_absorption(distance, freqs, temperature,
                                   relative_humidity, pressure)
    a_gr = ground_attenuation(distance, source_height, receiver_height, freqs,
                              ground_source, ground_middle, ground_receiver,
                              projected_distance)

    if barrier is None:
        a_bar = np.zeros_like(freqs)
    else:
        dz = barrier_attenuation(barrier, distance, freqs)
        if barrier.lateral:
            a_bar = np.maximum(dz, 0.0)
        else:
            a_bar = np.maximum(dz - a_gr, 0.0)

    a_total = a_div + a_atm + a_gr + a_bar
    return OutdoorAttenuation(
        frequencies=freqs,
        a_div=a_div,
        a_atm=a_atm,
        a_gr=a_gr,
        a_bar=a_bar,
        a_total=a_total,
        d_omega=np.zeros_like(freqs),
    )


def predicted_receiver_level(
    sound_power_level: ArrayLike,
    distance: float,
    source_height: float,
    receiver_height: float,
    frequencies: ArrayLike = DEFAULT_FREQUENCIES,
    ground_source: float = 0.0,
    ground_middle: float = 0.0,
    ground_receiver: float = 0.0,
    barrier: Barrier | None = None,
    temperature: float = 20.0,
    relative_humidity: float | None = None,
    pressure: float = 101.325,
    directivity_index: float = 0.0,
    d_omega: float = 0.0,
    c0: float | None = None,
    projected_distance: float | None = None,
    *,
    humidity: float | str = "deprecated",
) -> NDArray[np.float64]:
    r"""Predicted octave-band receiver level (ISO 9613-2:1996, Eq. (3)/(6)).

    Composes the downwind octave-band sound pressure level:

    .. math::

       L_{fT}(DW) = L_W + D_c - A, \qquad D_c = D_i + D_\Omega

    from the total attenuation :func:`outdoor_propagation_attenuation`. When ``c0``
    is given, the meteorological correction ``Cmet`` (Eq. (21)/(22)) is subtracted
    band by band to approximate the long-term average level ``LfT(LT)`` (Eq. (6));
    the standard applies ``Cmet`` to the A-weighted level, so this is a per-band
    convenience.

    :param sound_power_level: Octave-band sound power level ``Lw``, in decibels
        (re 1 pW), one value per frequency.
    :param distance: Straight-line source-to-receiver distance ``d``, in metres.
    :param source_height: Source height ``hs``, in metres.
    :param receiver_height: Receiver height ``hr``, in metres.
    :param frequencies: Octave-band midband frequencies, in hertz.
    :param ground_source: Ground factor ``Gs`` ([0, 1]).
    :param ground_middle: Ground factor ``Gm`` ([0, 1]).
    :param ground_receiver: Ground factor ``Gr`` ([0, 1]).
    :param barrier: Optional screening obstacle (:class:`Barrier`).
    :param temperature: Air temperature, in degrees Celsius.
    :param relative_humidity: Relative humidity, in percent (default 70).
    :param pressure: Atmospheric pressure, in kilopascals.
    :param directivity_index: Source directivity index ``Di``, in decibels.
    :param d_omega: Solid-angle index ``DOmega``, in decibels (see
        :func:`directivity_omega` for the alternative ground method).
    :param c0: Meteorological factor ``C0``, in decibels; ``None`` returns the
        downwind level ``LfT(DW)`` (:math:`C_{met} = 0`).
    :param projected_distance: Ground-plane projected distance ``dp``, in metres.
    :param humidity: Deprecated alias of ``relative_humidity`` (remove in 4.0).
    :return: Predicted octave-band level per frequency, in decibels.
    """
    relative_humidity = _resolve_humidity(
        "predicted_receiver_level", relative_humidity, humidity
    )
    lw = np.atleast_1d(np.asarray(sound_power_level, dtype=np.float64))
    attenuation = outdoor_propagation_attenuation(
        distance, source_height, receiver_height, frequencies,
        ground_source, ground_middle, ground_receiver, barrier,
        temperature, relative_humidity, pressure, projected_distance,
    )
    cmet: float | None = None
    if c0 is not None:
        dp = _projected_distance(distance, source_height, receiver_height,
                                 projected_distance)
        cmet = meteorological_correction(dp, source_height, receiver_height, c0)
    return _compose_receiver_level(lw, directivity_index, d_omega,
                                   attenuation.a_total, cmet)
