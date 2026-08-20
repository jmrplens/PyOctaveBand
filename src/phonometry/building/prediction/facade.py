#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Façade sound insulation and outdoor radiation prediction (EN 12354-3/-4:2000).

Two companion prediction models for the building envelope, both built on the
same energy summation of element transmission factors
:math:`\tau = 10^{-R/10}`, area-weighted by :math:`S_i/S` (small elements /
air paths enter through their element-normalized level difference ``Dn,e``
with the reference area :math:`A_0 = 10` m²):

**EN 12354-3, outdoor → indoor (façade sound insulation).** The apparent
sound reduction index of a façade for diffuse incidence (Formula 10):

.. math::

   R' = -10 \log_{10}\!\left( \sum \tau_{\mathrm{e},i} \right)

   \tau_{\mathrm{e},i} = \frac{S_i}{S}\, 10^{-R_i/10} \tag{Formula 15}

   \tau_{\mathrm{e},i} = \frac{A_0}{S}\, 10^{-D_{\mathrm{n,e},i}/10} \tag{Formula 14}

from which the loudspeaker- and traffic-referenced indices
:math:`R_{45} = R' + 1` (Formula 11) and :math:`R_\mathrm{tr,s} = R'` (Formula 12),
and the primary output, the standardized level difference at 2 m
(Formula 13):

.. math::

   D_{2\mathrm{m,nT}} = R' + \Delta L_\mathrm{fs}
   + 10 \log_{10}\!\left( \frac{V}{6 T_0 S} \right)
   \qquad T_0 = 0.5~\text{s}

with the façade-shape term ``ΔLfs`` (Annex C; 0 dB for a flat reflecting
façade).

**EN 12354-4, indoor → outdoor (sound radiated to the outside).** The sound
power level radiated by a segment (Formulas 2-3):

.. math::

   R' = -10 \log_{10}\!\left( \sum \frac{S_i}{S}\, 10^{-R_i/10}
   + \sum \frac{A_0}{S}\, 10^{-D_{\mathrm{n,e},i}/10} \right)

   L_W = L_{p,\mathrm{in}} + C_\mathrm{d} - R'
   + 10 \log_{10}\!\left( \frac{S}{S_0} \right)
   \qquad S_0 = 1~\text{m}^2

with the inside-field diffusivity term ``Cd`` (Annex B; -6 dB ideal diffuse,
-5 dB average industrial). An opening is modelled here as an element whose "R"
is the silencer insertion loss ``D`` (a bare opening is :math:`D = 0`),
combined in the *same* energy sum as the structural elements over the segment
area ``S`` -- a practical extension for a mixed wall-plus-opening segment.
This is NOT the standard's Formula (4), which treats a segment made up *only*
of openings with a different area normalization (``S`` = the opening area) and
sums its ``LW`` with the envelope segments only at the final energetic stage.
The exterior level follows from the simplified Annex E attenuation ``Atot`` of
a finite radiating side and :math:`L_p = L_W - A_\mathrm{tot}`.

Single-number ratings reuse EN ISO 717-1 via
:func:`phonometry.building.weighted_rating` (exact for
:math:`R'_\mathrm{w} + C_\mathrm{tr}`, a good approximation for ``R'w``, Part 3
NOTE 7).

Clause/formula citations refer to EN 12354-3:2000 or EN 12354-4:2000.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, log10, pi
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

    from ..._report.metadata import ReportMetadata

#: Reference absorption area ``A₀`` (Formulas 8, 14), in m².
_A0 = 10.0
#: Reference area ``S₀`` (Formula 2), in m².
_S0 = 1.0
#: Reference reverberation time ``T₀`` for dwellings (Formula 13), in s.
_T0 = 0.5


def _as_array(value: float | Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Return *value* as a 1-D float array, raising on non-finite entries."""
    arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if not np.all(np.isfinite(arr)):
        msg = f"'{name}' must contain only finite numbers."
        raise ValueError(msg)
    return arr


@dataclass(frozen=True)
class FacadeElement:
    """One façade element as a transmission path (EN 12354-3/-4).

    Provide exactly one of ``r`` (an area element, Formula 15 / Part 4 Formula 3),
    ``dn_e`` (a small element or air path, Formula 14) or ``insertion_loss`` (an
    opening in Part 4, modelled as an element whose reduction is the silencer's
    insertion loss ``D`` and combined in the same energy sum -- a practical
    extension, not the standard's separate segment-of-openings Formula 4). Per-band
    values may be scalars or equal-length arrays.

    :ivar name: Label used in results and plots.
    :ivar area: Element area ``Sᵢ`` in m² (required for ``r`` / ``insertion_loss``;
        ignored for ``dn_e`` small elements, which use ``A₀`` instead).
    :ivar r: Sound reduction index ``Rᵢ`` in dB.
    :ivar dn_e: Element-normalized level difference ``Dn,e,i`` in dB.
    :ivar insertion_loss: Opening silencer insertion loss ``Dᵢ`` in dB.
    """

    name: str
    area: float | None = None
    r: float | Sequence[float] | np.ndarray | None = None
    dn_e: float | Sequence[float] | np.ndarray | None = None
    insertion_loss: float | Sequence[float] | np.ndarray | None = None

    def _kind(self) -> str:
        given = [
            k for k in ("r", "dn_e", "insertion_loss") if getattr(self, k) is not None
        ]
        if len(given) != 1:
            msg = (
                f"Element '{self.name}': give exactly one of r / dn_e / insertion_loss."
            )
            raise ValueError(msg)
        return given[0]

    def tau(self, total_area: float, n_bands: int) -> np.ndarray:
        """Transmission factor ``τ`` of this element for the whole façade area."""
        if total_area <= 0:
            msg = "'total_area' (façade area S) must be positive."
            raise ValueError(msg)
        kind = self._kind()
        raw = getattr(self, kind)  # the one non-None quantity, per _kind()
        label = f"{self.name} ({kind})"
        if kind == "dn_e":
            level = _as_array(raw, label)
            weight = _A0 / total_area
        else:
            if self.area is None or self.area <= 0:
                msg = f"Element '{self.name}': 'area' must be positive."
                raise ValueError(msg)
            level = _as_array(raw, label)
            weight = self.area / total_area
        if level.size == 1:
            level = np.full(n_bands, float(level[0]))
        if level.size != n_bands:
            msg = f"Element '{self.name}': expected {n_bands} bands."
            raise ValueError(msg)
        return weight * 10.0 ** (-level / 10.0)


def _band_count(elements: Sequence[FacadeElement]) -> int:
    """Number of frequency bands implied by the elements (1 if all scalar)."""
    sizes = set()
    for el in elements:
        kind = el._kind()  # exactly one of r / dn_e / insertion_loss
        sizes.add(_as_array(getattr(el, kind), f"{el.name} ({kind})").size)
    sizes.discard(1)
    if len(sizes) > 1:
        msg = "Elements have inconsistent band counts."
        raise ValueError(msg)
    return sizes.pop() if sizes else 1


@dataclass(frozen=True)
class FacadePredictionResult:
    r"""Predicted façade airborne insulation (EN 12354-3:2000).

    :ivar r_prime: Apparent sound reduction index ``R'`` per band, in dB
        (Formula 10).
    :ivar r_45: :math:`R_{45} = R' + 1` (loudspeaker method, Formula 11),
        in dB.
    :ivar r_tr_s: :math:`R_\mathrm{tr,s} = R'` (traffic, Formula 12), in dB.
    :ivar d_2m_nt: Standardized level difference ``D2m,nT`` per band, in dB
        (Formula 13).
    :ivar element_r: Per-element partial index :math:`R_\mathrm{p} = -10 \log_{10} \tau` per
        band, in dB.
    :ivar r_tr_s_w: Single-number ``Rtr,s,w`` (ISO 717-1); ``None`` if the bands
        are not the ISO 717-1 octave/third-octave set.
    :ivar d_2m_nt_w: Single-number ``D2m,nT,w`` (ISO 717-1); ``None`` as above.
    :ivar c_tr: Spectrum adaptation term ``Ctr`` of ``R'`` (ISO 717-1).
    :ivar frequencies: Band centre frequencies (Hz) for plotting; ``None`` labels
        the axis by band index.
    :ivar elements: The element sequence the prediction was run with,
        retained so :meth:`plot_geometry` can draw the elevation; appended
        after the original fields and ``None`` for hand-built results.
    """

    r_prime: np.ndarray
    r_45: np.ndarray
    r_tr_s: np.ndarray
    d_2m_nt: np.ndarray
    element_r: dict[str, np.ndarray]
    r_tr_s_w: int | None = None
    d_2m_nt_w: int | None = None
    c_tr: int | None = None
    frequencies: np.ndarray | None = None
    elements: tuple[FacadeElement, ...] | None = None

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the per-element partial indices and the façade ``R'`` / ``D2m,nT``."""
        from ..._i18n import check_language
        from ..._plot.building import plot_facade_prediction

        check_language(language)
        return plot_facade_prediction(self, ax=ax, language=language, **kwargs)

    def plot_geometry(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Draw the composite facade elevation, element areas to scale.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :raises ValueError: If the result does not retain its geometry.
        """
        from ..._i18n import check_language
        from ..._plot.geometry import plot_facade_result_geometry

        check_language(language)
        return plot_facade_result_geometry(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render a predicted façade sound insulation report to a PDF (EN 12354-3).

        Writes a one-page **prediction** report for the predicted standardized
        level difference of a façade ``D2m,nT`` estimated by the EN/ISO
        12354-3:2000 model (Formula 13): a standard-basis line that states the
        sheet is a prediction from element data and not a measurement, an
        optional metadata header block, a two-panel body with the façade-element
        table (each element's weighted partial index ``Rp,w``) beside the
        per-element partial-index and ``R'`` / ``D2m,nT`` plot, the boxed
        predicted rating ``D2m,nT,w``, the prediction statement and, when a
        requirement is supplied, a PASS/FAIL verdict (the level difference
        passes at or above the requirement), followed by a footer.

        The applicable :class:`~phonometry.ReportMetadata` fields describe the
        predicted situation: ``specimen`` (the façade element set), ``area``
        (the exposed façade area ``S``), ``receiving_volume`` (the receiving
        room volume ``V``), ``test_room`` (the traffic / outdoor situation),
        ``client``, ``manufacturer``, ``measurement_standard``, ``laboratory``
        (the calculator / laboratory), ``operator``, ``report_id`` and
        ``test_date``. A summary of the façade shape (``ΔLfs``) and the model
        assumptions is recorded in ``notes`` (free text), and ``requirement``
        supplies the target ``D2m,nT,w``.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`; ``None``
            produces a lightweight fiche (body, rating, statement, disclaimer).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True``, the element table also shows each
            element's share of the transmitted sound energy.
        :param language: Fiche language: ``"en"`` (default) or ``"es"``.
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is unknown, ``language`` is not
            supported, or the result lacks the ISO 717-1 single-number ratings
            (build it on the 5 octave or 16 one-third-octave bands).
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        from .detailed_model import _check_report_request

        _check_report_request(engine, language)
        from ..._report.iso12354 import render_iso12354_facade_report

        return render_iso12354_facade_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


@dataclass(frozen=True)
class RadiatedPowerResult:
    """Predicted sound power radiated to the outside by a segment (EN 12354-4).

    :ivar l_w: Radiated sound power level ``LW`` per band, in dB re 1 pW
        (Formula 2).
    :ivar r_prime: Apparent sound reduction index ``R'`` per band, in dB
        (Formula 3).
    :ivar l_w_dba: A-weighted ``LW`` in dB(A), if the bands are known octave
        bands; else ``None``.
    :ivar frequencies: Band centre frequencies (Hz) for plotting; ``None`` labels
        the axis by band index.
    """

    l_w: np.ndarray
    r_prime: np.ndarray
    l_w_dba: float | None = None
    frequencies: np.ndarray | None = None

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the radiated sound power level ``LW`` per band."""
        from ..._i18n import check_language
        from ..._plot.building import plot_radiated_power

        check_language(language)
        return plot_radiated_power(self, ax=ax, language=language, **kwargs)


# A-weighting at octave-band centres 63 Hz .. 8 kHz (IEC 61672-1), for the
# A-weighted single-number outputs of the worked examples.
_A_WEIGHT_OCTAVE = {
    63: -26.2,
    125: -16.1,
    250: -8.6,
    500: -3.2,
    1000: 0.0,
    2000: 1.2,
    4000: 1.0,
    8000: -1.1,
}


def _apparent_reduction(
    elements: Sequence[FacadeElement], total_area: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    r""":math:`R' = -10 \log_{10}(\sum \tau)` (F. 10 / Part 4 F. 3), plus ``Rp``."""
    if not elements:
        msg = "At least one façade element is required."
        raise ValueError(msg)
    if total_area <= 0:
        msg = "'area' (total façade area S) must be positive."
        raise ValueError(msg)
    names = [el.name for el in elements]
    if len(set(names)) != len(names):
        msg = "Façade element names must be unique (they key the per-element results)."
        raise ValueError(msg)
    n = _band_count(elements)
    # Sum over the element list (not a name-keyed dict), so no element is ever
    # silently dropped, then key the per-element index by the unique names.
    taus = [el.tau(total_area, n) for el in elements]
    tau_total = np.sum(taus, axis=0)
    r_prime = -10.0 * np.log10(tau_total)
    element_r = {name: -10.0 * np.log10(t) for name, t in zip(names, taus, strict=True)}
    return r_prime, element_r


def _single_number(values: np.ndarray, bands: str | None) -> tuple[int, int] | None:
    """(rating, Ctr) via ISO 717-1 for 5 octave or 16 third-octave bands."""
    from ..measurement.insulation import weighted_rating

    if values.size not in (5, 16):
        return None
    result = weighted_rating(values, bands=bands)
    return result.rating, result.ctr


def facade_sound_reduction(
    elements: Sequence[FacadeElement],
    *,
    area: float,
    volume: float,
    delta_l_fs: float = 0.0,
    bands: str | None = None,
    frequencies: Sequence[float] | None = None,
) -> FacadePredictionResult:
    """Predict façade airborne sound insulation ``D2m,nT`` (EN 12354-3:2000).

    Energetically combines the element transmission factors (Formula 10) into the
    apparent sound reduction index ``R'``, then derives the loudspeaker/traffic
    indices (Formulas 11-12) and the standardized level difference (Formula 13).

    :param elements: Façade elements (see :class:`FacadeElement`); per-band
        arrays must share a common length (5 octave or 16 third-octave bands to
        get single-number ratings).
    :param area: Total façade area ``S`` seen from inside, in m².
    :param volume: Receiving-room volume ``V``, in m³ (Formula 13).
    :param delta_l_fs: Façade-shape term ``ΔLfs`` in dB (Annex C; 0 for a flat
        reflecting façade).
    :param bands: ``"octave"``, ``"third-octave"`` or ``None`` (auto) for the
        single number ratings, passed to :func:`weighted_rating`.
    :param frequencies: Optional band centre frequencies (Hz), stored on the
        result for plotting; must match the element band count.
    :return: A :class:`FacadePredictionResult`.
    """
    delta = float(delta_l_fs)
    v = float(volume)
    if v <= 0:
        msg = "'volume' must be positive."
        raise ValueError(msg)
    r_prime, element_r = _apparent_reduction(elements, float(area))
    if frequencies is not None and len(frequencies) != r_prime.size:
        msg = (
            f"'frequencies' has {len(frequencies)} entries but the elements imply "
            f"{r_prime.size} bands."
        )
        raise ValueError(msg)
    r_45 = r_prime + 1.0
    r_tr_s = r_prime.copy()
    d_2m_nt = r_prime + delta + 10.0 * log10(v / (6.0 * _T0 * float(area)))

    r_tr = _single_number(r_tr_s, bands)
    d_num = _single_number(d_2m_nt, bands)
    return FacadePredictionResult(
        r_prime=r_prime,
        r_45=r_45,
        r_tr_s=r_tr_s,
        d_2m_nt=d_2m_nt,
        element_r=element_r,
        r_tr_s_w=None if r_tr is None else r_tr[0],
        d_2m_nt_w=None if d_num is None else d_num[0],
        c_tr=None if r_tr is None else r_tr[1],
        frequencies=None
        if frequencies is None
        else np.asarray(frequencies, dtype=np.float64),
        elements=tuple(elements),
    )


def radiated_sound_power(
    elements: Sequence[FacadeElement],
    *,
    lp_in: float | Sequence[float] | np.ndarray,
    area: float,
    c_d: float = -6.0,
    r_prime_cap: float | None = None,
    octave_bands: Sequence[int] | None = None,
) -> RadiatedPowerResult:
    r"""Predict the sound power radiated outside by a segment (EN 12354-4).

    ``R'`` combines the element transmission factors (Formula 3); the radiated
    power level is :math:`L_W = L_{p,\mathrm{in}} + C_\mathrm{d} - R' + 10 \log_{10}(S/S_0)`
    (Formula 2). Openings
    may be included as :class:`FacadeElement` entries with an ``insertion_loss``
    (0 for a bare opening); see the module docstring for how this differs from
    the standard's separate segment-of-openings Formula (4).

    :param elements: Segment elements (see :class:`FacadeElement`).
    :param lp_in: Inside sound pressure level ``Lp,in`` per band, in dB.
    :param area: Segment area ``S``, in m².
    :param c_d: Inside-field diffusivity term ``Cd`` in dB (Annex B: -6 ideal
        diffuse, -5 average industrial building).
    :param r_prime_cap: Optional practical maximum on ``R'`` per band, in dB.
        This cap is **not** part of Formula (2)/(3): it appears only as a
        footnote of the Annex G worked example ("R' limited to 40 dB" for
        field situations with unavoidable leaks). Pass ``40.0`` to reproduce
        Annex G; the default ``None`` computes the bare formulas.
    :param octave_bands: Optional octave-band centre frequencies (Hz) matching
        the per-band data; enables the A-weighted single number.
    :return: A :class:`RadiatedPowerResult`.
    """
    r_prime, _ = _apparent_reduction(elements, float(area))
    if r_prime_cap is not None:
        r_prime = np.minimum(r_prime, float(r_prime_cap))
    lp = _as_array(lp_in, "lp_in")
    if lp.size == 1:
        lp = np.full(r_prime.size, float(lp[0]))
    if lp.size != r_prime.size:
        msg = "'lp_in' must match the element band count."
        raise ValueError(msg)
    if octave_bands is not None and len(octave_bands) != r_prime.size:
        msg = (
            f"'octave_bands' has {len(octave_bands)} entries but the elements imply "
            f"{r_prime.size} bands."
        )
        raise ValueError(msg)
    l_w = lp + float(c_d) - r_prime + 10.0 * log10(float(area) / _S0)

    l_w_dba: float | None = None
    if octave_bands is not None:
        try:
            a_weights = np.array([_A_WEIGHT_OCTAVE[int(f)] for f in octave_bands])
        except KeyError:
            a_weights = None  # unknown band centre; skip the A-weighted number
        if a_weights is not None:
            l_w_dba = float(10.0 * np.log10(np.sum(10.0 ** ((l_w + a_weights) / 10.0))))
    freqs = None if octave_bands is None else np.asarray(octave_bands, dtype=np.float64)
    return RadiatedPowerResult(
        l_w=l_w, r_prime=r_prime, l_w_dba=l_w_dba, frequencies=freqs
    )


def outdoor_attenuation(width: float, height: float, distance: float) -> float:
    r"""Simplified attenuation ``Atot`` of a finite side (EN 12354-4 Annex E).

    Reception point in front of the centre of a rectangular side
    :math:`S = \mathrm{width} \cdot \mathrm{height}` at perpendicular
    ``distance`` d. Uses the finite-side Formula (E.2a)
    up to the largest side dimension and the point-source Formula (E.2b) beyond
    it, following the Annex E Note 3 switching rule. The two branches do not
    join continuously at the switch distance: for a square 10 m x 10 m side
    the step is about -0,7 dB (about -0,3 dB for a 60 m x 10 m side), an
    artefact of the standard's own simplification. The ``+6 dB`` for radiation
    into the quarter-space over hard ground is built into the formula.

    :param width: Side width ``L``, in m.
    :param height: Side height ``H``, in m.
    :param distance: Perpendicular distance ``d`` to the reception point, in m.
    :return: ``Atot`` in dB (subtract from ``LW`` to get the exterior ``Lp``).
    """
    w, h, d = float(width), float(height), float(distance)
    if min(w, h, d) <= 0:
        msg = "width, height and distance must be positive."
        raise ValueError(msg)
    area = w * h
    if d > max(w, h):
        return float(-10.0 * log10(_S0 / (pi * d * d)))  # (E.2b)
    corridor = (4.0 * _S0 / (pi * area)) * atan(w / (2.0 * d)) * atan(h / (2.0 * d))
    return float(-10.0 * log10(corridor))  # (E.2a)


#: EN 12354-3:2000 Annex C, Figure C.2: façade-shape level difference ΔLfs
#: (dB). Keyed by shape; each value maps the line-of-sight height bin
#: (0: < 1,5 m; 1: 1,5 m to 2,5 m; 2: > 2,5 m) to the (αw ≤ 0,3; 0,6; ≥ 0,9)
#: triple, or None where the figure prints "does not apply". The 2017 edition
#: (DIN EN ISO 12354-3, Tabelle C.1) tabulates identical values.
_DELTA_LFS: dict[str, tuple[tuple[float, float, float] | None, ...]] = {
    "plane_facade": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "gallery_2": ((-1.0, -1.0, 0.0), None, None),
    "gallery_3": ((-1.0, -1.0, 0.0), (-1.0, 0.0, 2.0), (1.0, 1.0, 2.0)),
    "gallery_4": ((0.0, 0.0, 1.0), (0.0, 1.0, 3.0), (2.0, 2.0, 3.0)),
    "gallery_5": (None, None, (3.0, 4.0, 6.0)),
    "balcony_6": ((-1.0, -1.0, 0.0), (-1.0, 1.0, 3.0), (1.0, 2.0, 3.0)),
    "balcony_7": ((0.0, 0.0, 1.0), (0.0, 2.0, 4.0), (2.0, 3.0, 4.0)),
    "balcony_8": ((1.0, 1.0, 2.0), (1.0, 1.0, 2.0), (1.0, 1.0, 2.0)),
    "terrace_open": ((1.0, 1.0, 1.0), (3.0, 4.0, 5.0), (4.0, 4.0, 5.0)),
    "terrace_closed": ((3.0, 3.0, 3.0), (5.0, 6.0, 7.0), (6.0, 6.0, 7.0)),
}

#: αw grid of the Figure C.2 columns (≤ 0,3; 0,6; ≥ 0,9).
_DLFS_ALPHAS = (0.3, 0.6, 0.9)

#: Line-of-sight height bin edges of the Figure C.2 rows, in m
#: (below 1,5 m; 1,5 m to 2,5 m; above 2,5 m).
_DLFS_HEIGHT_EDGES = (1.5, 2.5)


def facade_shape_level_difference(
    shape: str,
    *,
    line_of_sight: float = 0.0,
    absorption: float = 0.3,
) -> float:
    """Façade-shape level difference ``ΔLfs`` (EN 12354-3:2000 Annex C).

    Looks up Figure C.2 for the level difference caused by the exterior shape
    of the façade (gallery, balcony or terrace), as a function of the height
    of the line of sight from the source on the façade plane and the weighted
    sound absorption coefficient ``αw`` (EN ISO 11654) of the underside of the
    balcony/roof above. The value feeds ``delta_l_fs`` of
    :func:`facade_sound_reduction` (Formula 13). Intermediate ``αw`` values
    are interpolated linearly between the tabulated 0,3 / 0,6 / 0,9 columns,
    as the annex allows; outside that range the edge column applies. The
    2017 edition tabulates the same values (Tabelle C.1).

    Shapes follow the Figure C.2 numbering: ``"plane_facade"`` (1, always
    0 dB), ``"gallery_2"`` to ``"gallery_5"`` (2-5), ``"balcony_6"`` to
    ``"balcony_8"`` (6-8) and ``"terrace_open"`` / ``"terrace_closed"`` (9,
    open or closed fence).

    :param shape: Façade shape key (see above).
    :param line_of_sight: Height of the line of sight from the source at the
        façade plane, in m (bins: below 1,5 m; 1,5 m to 2,5 m; above 2,5 m).
    :param absorption: Weighted absorption coefficient ``αw`` of the underside
        above the façade (default 0,3; reflecting).
    :return: ``ΔLfs``, in dB.
    :raises ValueError: For an unknown shape, a negative height/absorption, or
        a shape/height combination the figure marks "does not apply".
    """
    try:
        rows = _DELTA_LFS[shape]
    except KeyError:
        valid = ", ".join(sorted(_DELTA_LFS))
        msg = f"Unknown facade shape {shape!r}. Valid: {valid}."
        raise ValueError(msg) from None
    h = float(line_of_sight)
    aw = float(absorption)
    if not (np.isfinite(h) and h >= 0.0):
        msg = "'line_of_sight' must be a non-negative height in m."
        raise ValueError(msg)
    if not (np.isfinite(aw) and 0.0 <= aw <= 1.0):
        msg = "'absorption' must be an αw between 0 and 1."
        raise ValueError(msg)
    low, high = _DLFS_HEIGHT_EDGES
    if h < low:
        row_index = 0
    elif h <= high:
        row_index = 1
    else:
        row_index = 2
    row = rows[row_index]
    if row is None:
        msg = (
            f"Figure C.2 marks shape {shape!r} as 'does not apply' for a "
            f"line-of-sight height of {h:g} m."
        )
        raise ValueError(msg)
    return float(np.interp(aw, _DLFS_ALPHAS, row))


def outdoor_level(
    l_w: float | Sequence[float],
    attenuation: float | Sequence[float],
) -> float:
    r"""Exterior level from one or more radiating sides (EN 12354-4 Formula E.1).

    :math:`L_p = 10 \log_{10}( \sum 10^{L_{W,k}/10} ) - A_\mathrm{tot}` for sides sharing
    a reception point,
    or the per-side ``LW - Atot`` energetically summed. Pass matching sequences
    of side power levels and their attenuations, or scalars for a single side; a
    scalar broadcasts against an array (e.g. several sides, one common ``Atot``).

    :param l_w: Radiated power level(s) ``LW`` (dB), scalar or per side.
    :param attenuation: Attenuation(s) ``Atot`` (dB), scalar or per side.
    :return: Exterior sound pressure level ``Lp`` in dB.
    """
    lw = np.atleast_1d(np.asarray(l_w, dtype=np.float64))
    at = np.atleast_1d(np.asarray(attenuation, dtype=np.float64))
    try:
        diff = lw - at  # NumPy broadcasting (scalar vs array is allowed)
    except ValueError as exc:
        msg = "'l_w' and 'attenuation' must have broadcast-compatible shapes."
        raise ValueError(msg) from exc
    return float(10.0 * np.log10(np.sum(10.0 ** (diff / 10.0))))
