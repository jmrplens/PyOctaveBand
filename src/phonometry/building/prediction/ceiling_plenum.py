#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Suspended-ceiling plenum flanking path (Vigran 9.2.3 after Mechel 1980;
ISO 140-9 / ISO 10848-2; ASTM E1414 / ASTM E413).

Two offices separated by a partition that stops at the suspended ceiling share
one continuous plenum above it. Sound leaves the source room through the
ceiling tiles, travels sideways over the partition and comes back down through
the tiles of the receiving room. That path is often the weakest link in an
open-plan fit-out, and it is *not* what a partition's ``Rw`` describes.

**The one-dimensional model (Vigran Eqs. (9.14) to (9.20)).** Mechel's
one-dimensional variant treats the plenum as a duct lined on one side. The
ceiling on each side has a transmission factor
:math:`\tau_S = \tau_{S,pl} \, \tau_{S,a}` (plates times the prospective
plenum absorber, Eq. (9.14)); the power injected into the plenum splits, a
fraction :math:`s_S` heading for the partition, and decays as
:math:`\exp(-m x)` with the power attenuation coefficient
:math:`m = 2 \operatorname{Re}\{\Gamma\} = -2 \operatorname{Im}\{k'\}`
(Eqs. (9.15) and (9.16)). Integrating over the ceiling length on both sides
gives

.. math::

   \tau_{cl} = \frac{s_S s_R \tau_S \tau_R L_R}{m_S L_S \, m'_R L_R \, h}
   \left( 1 - \exp(-\epsilon m_S L_S) \right)
   \left( 1 - \exp(-\epsilon m'_R L_R) \right) \tag{Eq. 9.18}

with the receiving-side coefficient increased by the leakage back into the
room, :math:`m'_R = m_R + s_R \tau_R / h` (Eq. (9.17)). Vigran prints the
exponents with a factor 2 for totally reflecting plenum sidewalls and states
that totally absorbing ones give the same expression "without the factor 2",
so the factor is the same :math:`\epsilon` that the compact form carries.

.. warning::

   Vigran prints the **unprimed** :math:`m_R` in that denominator. That is a
   misprint (see ``docs/ERRATA.md``): the receiving-side integral
   :math:`\int_0^{L_R} \exp(-\epsilon m'_R x) \, dx =
   (1 - \exp(-\epsilon m'_R L_R))/(\epsilon m'_R)` carries :math:`m'_R`,
   exactly as the source-side one carries :math:`m_S`, and the printed
   reading leaves :math:`\tau_{cl}` unbounded as :math:`m_R \to 0`, where the
   derived reading stays bounded by the leakage term. This module implements
   the derived :math:`m'_R`.

For a plenum with little attenuation on both sides (:math:`m_S L_S \ll 1`
**and** :math:`m'_R L_R \ll 1`, which needs a fairly insulating ceiling as
well as a weakly damped plenum, since :math:`m'_R` never falls below
:math:`s_R \tau_R / h`) and :math:`s_S = s_R = 0.5` it collapses to the
result that makes the geometry visible:

.. math::

   \tau_{cl} = \frac{\epsilon^2 \tau_S \tau_R L_R}{4 h} \tag{Eq. 9.19}

   R_{cl} = R_S + R_R - 10 \log_{10}\!\left[ \frac{\epsilon^2 L_R}{4 h} \right]
   \tag{Eq. 9.20}

with :math:`\epsilon = 1` for totally absorbing plenum sidewalls and
:math:`\epsilon = 2` for totally reflecting ones. Referred to the partition
area instead of the ceiling, :math:`R_{cl,p} = R_{cl} + 10 \log_{10}(H_S/L_S)`
(Eq. (9.13)). A deep plenum helps (the :math:`-10 \log_{10}` term shrinks), a long
room hurts, and doubling the tile insulation helps twice over because
:math:`R_S` and :math:`R_R` both appear.

**The measured quantity (ISO 140-9:1985 clause 3.3, ISO 10848-2).** A ceiling
is not rated by :math:`R_{cl}` but by the **normalized ceiling attenuation**
:math:`D_{n,c} = D - 10 \log_{10}(A/A_0)`, with ``A`` the receiving-room equivalent
absorption area and the reference :math:`A_0 = 10` m2. The laboratory has two
rooms of at least 50 m3 whose volumes differ by at least 10 %, a dividing
wall tapered to at most 100 mm at the top, and a plenum 650 mm to 760 mm deep
with one sidewall and both end walls lined; the standard prints the required
lining absorption :math:`\alpha_s \ge 0.65` at 125 Hz and :math:`\ge 0.80`
from 250 Hz to 4000 Hz, and requires :math:`\alpha < 0.10` on the other
sidewall and on the plenum ceiling. The North American counterpart,
ASTM E1414, uses :math:`A_0 = 12` m2, so an ASTM value runs about
:math:`10 \log_{10}(12/10) = 0.79` dB higher than the ISO one.

**Single number.** ISO rates :math:`D_{n,c}` with the ISO 717-1 curve
(:func:`phonometry.weighted_rating`, giving ``Dn,c,w``); ASTM E1414 rates it
through ASTM E413 as the **ceiling attenuation class** (CAC). E413 rounds the
data to the nearest integer (clause 5.2), shifts its reference contour upward in
1 dB steps while the sum of the deficiencies stays at or below 32 dB and no
single deficiency exceeds 8 dB (clauses 5.3 and 5.4), and reads the rating off
the shifted contour at 500 Hz (clause 5.5). See
:func:`ceiling_attenuation_class`.

.. note::

   The one-dimensional plenum model has **no published numeric output**: every
   result in Vigran (Figs. 9.11 to 9.13) and in Mechel's *Formulas of Acoustics*
   (Sections I.21 and I.22) is a figure. The functions here are anchored on the
   closed forms, on the derivation of Eq. (9.18) from the two side integrals,
   and on structural properties that a wrong reading breaks: the bound
   :math:`\tau_{cl} \le 1`, a bare plenum no worse than the undamped
   Eq. (9.20) value, and the small-attenuation limit taken where it applies
   rather than only where the Eq. (9.17) leakage term happens to vanish.
   The measurement chain
   (:func:`normalized_ceiling_attenuation`, :func:`ceiling_attenuation_class`)
   *is* anchored on accredited ASTM E1414 laboratory reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from ..._internal.validation import (
    require_choice,
    require_finite_array,
    require_positive,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "CEILING_ATTENUATION_CONTOUR",
    "CeilingAttenuationResult",
    "PlenumFlankingResult",
    "ceiling_attenuation_class",
    "normalized_ceiling_attenuation",
    "partition_referenced_reduction_index",
    "plenum_flanking_reduction_index",
]

#: ASTM E413-22 Table 1, the reference sound insulation contour, in dB, keyed by
#: one-third-octave band centre frequency in Hz. The contour has a rating of
#: zero (Note 2); other contours are derived by adding the same integer to every
#: value. The same array rates STC (E90), NIC/NNIC/ASTC (E336), NIC (E596) and
#: the ceiling attenuation class CAC (E1414).
CEILING_ATTENUATION_CONTOUR: dict[float, float] = {
    125.0: -16.0, 160.0: -13.0, 200.0: -10.0, 250.0: -7.0,
    315.0: -4.0, 400.0: -1.0, 500.0: 0.0, 630.0: 1.0,
    800.0: 2.0, 1000.0: 3.0, 1250.0: 4.0, 1600.0: 4.0,
    2000.0: 4.0, 2500.0: 4.0, 3150.0: 4.0, 4000.0: 4.0,
}

#: Reference equivalent absorption area ``A0`` of ISO 140-9:1985 clause 3.3 and
#: ISO 10848-2, in m2.
_A0_ISO = 10.0

#: Maximum sum of deficiencies allowed by ASTM E413-22 clause 5.4.1, in dB.
_MAX_DEFICIENCY_SUM = 32.0

#: Maximum deficiency at any one frequency, ASTM E413-22 clause 5.4.2, in dB.
_MAX_SINGLE_DEFICIENCY = 8.0

#: Sidewall reflection cases of Vigran Eqs. (9.18) and (9.19): the constant
#: ``eps``, which is both the prefactor of the compact form and the factor in
#: the exponents of the attenuated one.
_SIDEWALLS: dict[str, float] = {"absorbing": 1.0, "reflecting": 2.0}


def _require_transmission_factor(tau: np.ndarray) -> None:
    r"""Reject a ceiling/plenum transmission factor above unity.

    :math:`\tau_{cl}` is a ratio of powers referred to the ceiling area, so a
    value above 1 means the model has been pushed outside the range where its
    diffuse-field power balance means anything, which happens once the ceiling
    itself is nearly transparent (roughly :math:`R_S + R_R` below the
    geometry term). Failing loudly beats returning a negative sound reduction
    index.
    """
    worst = float(np.max(tau))
    if worst > 1.0:
        raise ValueError(
            f"the ceiling/plenum transmission factor reaches {worst:.3g}, above "
            "unity: the one-dimensional model is outside its validity range. "
            "Check that 'reduction_index_source' and 'reduction_index_receiving' "
            "exceed the geometry term 10 lg(eps^2 LR/(4h))."
        )


def _require_split(value: float, name: str) -> float:
    r"""Require a power-split ratio in ``(0, 1]`` (Vigran Section 9.2.3.2).

    :math:`s_S` and :math:`s_R` are the fraction of the power injected into
    the plenum that heads towards the partition, so a value above 1 is not a
    split.
    """
    ratio = require_positive(value, name)
    if ratio > 1.0:
        raise ValueError(f"'{name}' must be a power split in (0, 1].")
    return ratio


def normalized_ceiling_attenuation(
    level_source: ArrayLike,
    level_receiving: ArrayLike,
    absorption_area: ArrayLike,
    *,
    reference_area: float = _A0_ISO,
) -> np.ndarray:
    r"""Normalized ceiling attenuation ``Dn,c`` (ISO 140-9:1985, clause 3.3).

    :math:`D_{n,c} = (L_1 - L_2) - 10 \log_{10}(A/A_0)`, the level difference
    between two rooms sharing a common ceiling plenum, normalized to a
    reference equivalent absorption area. ISO 140-9 and ISO 10848-2 use
    :math:`A_0 = 10` m2; ASTM E1414 uses :math:`A_0 = 12` m2, which makes an
    ASTM value about 0.79 dB higher for the same rooms.

    :param level_source: Source-room sound pressure level ``L1`` per band, in dB.
    :param level_receiving: Receiving-room level ``L2`` per band, in dB.
    :param absorption_area: Receiving-room equivalent absorption area ``A``
        per band, in m2 (> 0); a scalar is broadcast over the bands.
    :param reference_area: Reference area ``A0``, in m2 (Default: 10 m2).
    :return: The normalized ceiling attenuation ``Dn,c`` per band, in dB.
    :raises ValueError: for mismatched shapes or a non-positive area.
    """
    l1 = require_finite_array(level_source, "level_source")
    l2 = require_finite_array(level_receiving, "level_receiving")
    if l1.shape != l2.shape:
        raise ValueError("'level_source' and 'level_receiving' must share their shape.")
    area = require_finite_array(absorption_area, "absorption_area")
    if area.size == 1:
        area = np.full(l1.shape, float(area[0]))
    if area.shape != l1.shape:
        raise ValueError("'absorption_area' must be a scalar or match the levels.")
    if np.any(area <= 0.0):
        raise ValueError("'absorption_area' must be positive.")
    a0 = require_positive(reference_area, "reference_area")
    return np.asarray(l1 - l2 - 10.0 * np.log10(area / a0), dtype=np.float64)


@dataclass(frozen=True)
class CeilingAttenuationResult:
    """Ceiling attenuation class (ASTM E1414 rated through ASTM E413).

    :ivar frequencies: One-third-octave band centre frequencies, in Hz.
    :ivar measured: Normalized ceiling attenuation ``Dn,c`` per band, in dB,
        as supplied.
    :ivar rounded: The same data rounded to the nearest integer (clause 5.2),
        which is what the contour is fitted to.
    :ivar shifted_reference: The fitted reference contour, in dB.
    :ivar deficiencies: Per-band deficiency (shifted contour minus data, floored
        at zero), in dB.
    :ivar deficiency_sum: Sum of the deficiencies, in dB (at most 32).
    :ivar max_deficiency: Largest single deficiency, in dB (at most 8).
    :ivar rating: The ceiling attenuation class CAC, read off the shifted
        contour at 500 Hz (clause 5.5), in dB.
    """

    frequencies: np.ndarray
    measured: np.ndarray
    rounded: np.ndarray
    shifted_reference: np.ndarray
    deficiencies: np.ndarray
    deficiency_sum: float
    max_deficiency: float
    rating: int

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot ``Dn,c`` against the fitted ASTM E413 contour.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_ceiling_attenuation

        check_language(language)
        return plot_ceiling_attenuation(self, ax=ax, language=language, **kwargs)


def ceiling_attenuation_class(
    attenuation: ArrayLike, frequency: ArrayLike | None = None
) -> CeilingAttenuationResult:
    """Ceiling attenuation class CAC (ASTM E413-22 clause 5, via ASTM E1414).

    Rounds the data to the nearest integer (clause 5.2), then raises the
    :data:`CEILING_ATTENUATION_CONTOUR` in 1 dB steps to the highest position
    at which the sum of the deficiencies is at most 32 dB (clause 5.4.1) and no
    single deficiency exceeds 8 dB (clause 5.4.2). The rating is the shifted
    contour read at 500 Hz (clause 5.5).

    :param attenuation: Normalized ceiling attenuation ``Dn,c`` in the 16
        one-third-octave bands 125 Hz to 4000 Hz, in dB.
    :param frequency: Optional band centre frequencies, in Hz; when given they
        must be exactly the 16 contour bands.
    :return: A :class:`CeilingAttenuationResult`.
    :raises ValueError: for a wrong number of bands or mismatched frequencies.
    """
    bands = np.asarray(sorted(CEILING_ATTENUATION_CONTOUR), dtype=np.float64)
    contour = np.asarray(
        [CEILING_ATTENUATION_CONTOUR[float(b)] for b in bands], dtype=np.float64
    )
    values = require_finite_array(attenuation, "attenuation")
    if values.shape != bands.shape:
        raise ValueError(
            f"'attenuation' must hold {bands.size} one-third-octave values "
            f"(125 Hz to 4000 Hz)."
        )
    if frequency is not None:
        given = require_finite_array(frequency, "frequency")
        if given.shape != bands.shape or not np.allclose(given, bands):
            raise ValueError(
                "'frequency' must be the 16 ASTM E413 contour bands "
                "125 Hz to 4000 Hz."
            )
    # Clause 5.2: the contour is fitted to integer-rounded data.
    rounded = np.asarray(np.floor(values + 0.5), dtype=np.float64)

    def _deficiencies(shift: int) -> np.ndarray:
        return np.maximum(contour + shift - rounded, 0.0)

    # Clause 5.3: start below every possible fit and raise until it fails.
    shift = int(np.floor(np.min(rounded - contour)))
    while True:
        trial = _deficiencies(shift + 1)
        if float(np.sum(trial)) > _MAX_DEFICIENCY_SUM or float(
            np.max(trial)
        ) > _MAX_SINGLE_DEFICIENCY:
            break
        shift += 1
    deficiencies = _deficiencies(shift)
    reference = contour + shift
    return CeilingAttenuationResult(
        frequencies=bands,
        measured=values,
        rounded=rounded,
        shifted_reference=reference,
        deficiencies=deficiencies,
        deficiency_sum=float(np.sum(deficiencies)),
        max_deficiency=float(np.max(deficiencies)),
        # Clause 5.5: the contour is zero at 500 Hz, so the rating is the shift.
        rating=int(shift),
    )


@dataclass(frozen=True)
class PlenumFlankingResult:
    r"""Ceiling/plenum flanking path of a suspended ceiling (Vigran 9.2.3).

    :ivar frequencies: Band centre frequencies, in Hz, or ``None``.
    :ivar reduction_index: Sound reduction index ``Rcl`` of the ceiling/plenum
        path per band, in dB.
    :ivar transmission_factor: The transmission factor ``tau_cl`` per band.
    :ivar reduction_index_source: Source-side ceiling ``RS`` per band, in dB.
    :ivar reduction_index_receiving: Receiving-side ceiling ``RR`` per band, in dB.
    :ivar geometry_term: The geometry penalty
        :math:`10 \log_{10}[\epsilon^2 L_R/(4h)]`, in dB, or ``None`` for the
        attenuated model, whose penalty is per band.
    :ivar penalty: The per-band difference :math:`R_S + R_R - R_{cl}`, in dB:
        what the plenum takes off the sum of the two ceilings.
    :ivar model: ``"undamped"`` (Eq. (9.20)) or ``"attenuated"`` (Eq. (9.18)).
    :ivar epsilon: The sidewall constant ``eps`` (1 absorbing, 2 reflecting).
    :ivar plenum_height: Plenum height ``h``, in m.
    :ivar ceiling_length: Receiving-side ceiling length ``LR``, in m.
    """

    frequencies: np.ndarray | None
    reduction_index: np.ndarray
    transmission_factor: np.ndarray
    reduction_index_source: np.ndarray
    reduction_index_receiving: np.ndarray
    geometry_term: float | None
    penalty: np.ndarray
    model: str
    epsilon: float
    plenum_height: float
    ceiling_length: float

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot ``Rcl`` against the two ceiling reduction indices.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_plenum_flanking

        check_language(language)
        return plot_plenum_flanking(self, ax=ax, language=language, **kwargs)


def plenum_flanking_reduction_index(
    reduction_index_source: ArrayLike,
    reduction_index_receiving: ArrayLike,
    *,
    ceiling_length: float,
    plenum_height: float,
    sidewalls: str = "reflecting",
    frequency: ArrayLike | None = None,
    attenuation_source: ArrayLike | None = None,
    attenuation_receiving: ArrayLike | None = None,
    source_length: float | None = None,
    split_source: float = 0.5,
    split_receiving: float = 0.5,
) -> PlenumFlankingResult:
    r"""Ceiling/plenum flanking reduction index ``Rcl`` (Vigran
    Eqs. (9.18)-(9.20)).

    With no attenuation coefficients this is the compact undamped form
    :math:`R_{cl} = R_S + R_R - 10 \log_{10}[\epsilon^2 L_R/(4h)]` (Eq. (9.20)).
    Supplying the plenum power attenuation coefficients :math:`m_S` and
    :math:`m_R` (Eq. (9.16), :math:`m = -2 \operatorname{Im}\{k'\}` of the
    lined duct) switches to the full Eq. (9.18), whose receiving side carries
    the leakage term :math:`m'_R = m_R + s_R \tau_R/h` (Eq. (9.17)) in both
    the exponent and the denominator. Vigran prints the denominator with the
    unprimed :math:`m_R`; that is a misprint and the derived reading is
    implemented here (see ``docs/ERRATA.md``).

    Because :math:`m'_R` never falls below :math:`s_R \tau_R / h`, the
    attenuated form approaches Eq. (9.20) only when the ceiling is insulating
    enough for :math:`m'_R L_R` to be small as well; at :math:`R = 20` dB
    with the geometry of Vigran's own example the leakage term is still worth
    about 0.24 dB.

    :param reduction_index_source: Source-side ceiling ``RS`` per band, in dB
        (the ceiling plates and any plenum absorber together, Eq. (9.14)).
    :param reduction_index_receiving: Receiving-side ceiling ``RR`` per band, in dB.
    :param ceiling_length: Receiving-side ceiling length ``LR``, in m (> 0).
    :param plenum_height: Plenum height ``h``, in m (> 0).
    :param sidewalls: ``"reflecting"`` (:math:`\epsilon = 2`, Default) or
        ``"absorbing"`` (:math:`\epsilon = 1`); :math:`\epsilon` scales both
        the geometry penalty of Eq. (9.20) and the exponents of Eq. (9.18).
    :param frequency: Optional band centre frequencies, in Hz.
    :param attenuation_source: Optional plenum power attenuation coefficient
        ``mS`` per band, in 1/m (> 0); switches to Eq. (9.18).
    :param attenuation_receiving: Optional ``mR`` per band, in 1/m (> 0);
        required together with *attenuation_source*.
    :param source_length: Source-side ceiling length ``LS``, in m
        (Default: equal to *ceiling_length*).
    :param split_source: Power split ``sS`` towards the partition, in
        ``(0, 1]`` (Default: 0,5).
    :param split_receiving: Power split ``sR`` on the receiving side, in
        ``(0, 1]`` (Default: 0,5).
    :return: A :class:`PlenumFlankingResult`.
    :raises ValueError: for mismatched shapes, an unknown sidewall case, or a
        non-positive dimension.
    """
    rs = require_finite_array(reduction_index_source, "reduction_index_source")
    rr = require_finite_array(reduction_index_receiving, "reduction_index_receiving")
    if rs.shape != rr.shape:
        raise ValueError(
            "'reduction_index_source' and 'reduction_index_receiving' must "
            "share their shape."
        )
    lr = require_positive(ceiling_length, "ceiling_length")
    h = require_positive(plenum_height, "plenum_height")
    ls = lr if source_length is None else require_positive(source_length, "source_length")
    case = require_choice(sidewalls, "sidewalls", tuple(_SIDEWALLS))
    eps = _SIDEWALLS[case]
    ss = _require_split(split_source, "split_source")
    sr = _require_split(split_receiving, "split_receiving")
    freqs: np.ndarray | None = None
    if frequency is not None:
        freqs = require_finite_array(frequency, "frequency")
        if freqs.shape != rs.shape:
            raise ValueError("'frequency' must match the reduction indices.")

    tau_s = 10.0 ** (-rs / 10.0)
    tau_r = 10.0 ** (-rr / 10.0)
    if (attenuation_source is None) != (attenuation_receiving is None):
        raise ValueError(
            "'attenuation_source' and 'attenuation_receiving' must be given "
            "together."
        )
    if attenuation_source is None or attenuation_receiving is None:
        geometry = float(10.0 * np.log10(eps**2 * lr / (4.0 * h)))
        tau = eps**2 * tau_s * tau_r * lr / (4.0 * h)
        _require_transmission_factor(tau)
        rcl = rs + rr - geometry
        return PlenumFlankingResult(
            frequencies=freqs,
            reduction_index=np.asarray(rcl, dtype=np.float64),
            transmission_factor=np.asarray(tau, dtype=np.float64),
            reduction_index_source=rs,
            reduction_index_receiving=rr,
            geometry_term=geometry,
            penalty=np.full(rs.shape, geometry),
            model="undamped",
            epsilon=eps,
            plenum_height=h,
            ceiling_length=lr,
        )

    ms = require_finite_array(attenuation_source, "attenuation_source")
    mr = require_finite_array(attenuation_receiving, "attenuation_receiving")
    if ms.shape != rs.shape or mr.shape != rs.shape:
        raise ValueError("the attenuation coefficients must match the reduction indices.")
    if np.any(ms <= 0.0) or np.any(mr <= 0.0):
        raise ValueError("the attenuation coefficients must be positive.")
    # Eq. (9.17): the receiving side also loses power back into the room.
    mr_eff = mr + sr * tau_r / h
    # Vigran's Eq. (9.18) prints the *unprimed* mR in this denominator while
    # carrying m'R in the exponent. The derivation gives m'R in both: the
    # receiving-side integral is int_0^LR exp(-eps m'R x) dx =
    # (1 - exp(-eps m'R LR))/(eps m'R), exactly as the source-side one is in
    # mS. Taken literally the printed form is not a transmission factor at
    # all: it diverges as mR -> 0, where the derived reading stays bounded by
    # the leakage term, and it exceeds unity for ordinary inputs. See
    # docs/ERRATA.md, "Vigran (2008), Eq. (9.18)".
    tau = (
        ss * sr * tau_s * tau_r * lr / (ms * ls * mr_eff * lr * h)
        * (1.0 - np.exp(-eps * ms * ls))
        * (1.0 - np.exp(-eps * mr_eff * lr))
    )
    _require_transmission_factor(tau)
    rcl = -10.0 * np.log10(tau)
    return PlenumFlankingResult(
        frequencies=freqs,
        reduction_index=np.asarray(rcl, dtype=np.float64),
        transmission_factor=np.asarray(tau, dtype=np.float64),
        reduction_index_source=rs,
        reduction_index_receiving=rr,
        geometry_term=None,
        penalty=np.asarray(rs + rr - rcl, dtype=np.float64),
        model="attenuated",
        epsilon=eps,
        plenum_height=h,
        ceiling_length=lr,
    )


def partition_referenced_reduction_index(
    reduction_index: ArrayLike, room_height: float, room_length: float
) -> np.ndarray:
    r"""Refer ``Rcl`` to the partition area instead of the ceiling
    (Eq. (9.13)).

    :math:`R_{cl,p} = R_{cl} + 10 \log_{10}(H_S/L_S)`, with ``HS`` the height and
    ``LS`` the length of the sending room. Referring every path to one common
    area (the partition) is what lets the ceiling path be added to the direct
    path as transmission factors.

    :param reduction_index: Ceiling/plenum path ``Rcl`` per band, in dB.
    :param room_height: Sending-room height ``HS``, in m (> 0).
    :param room_length: Sending-room length ``LS``, in m (> 0).
    :return: The partition-referenced ``Rcl,p`` per band, in dB.
    :raises ValueError: for a non-positive dimension.
    """
    rcl = require_finite_array(reduction_index, "reduction_index")
    hs = require_positive(room_height, "room_height")
    ls = require_positive(room_length, "room_length")
    return np.asarray(rcl + 10.0 * np.log10(hs / ls), dtype=np.float64)
