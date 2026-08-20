#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Directional and random-incidence diffusion coefficients in a free field.

**ISO 17497-2:2012.** From the set of reflected sound-pressure levels ``L_i``
on a semicircle or hemisphere the autocorrelation diffusion coefficient
``d_theta`` is formed for equal-area receivers (Clause 8.1, Formula (5)) or
with per-receiver area weights ``N_i`` (Formula (6)); the area weights follow
from the solid-angle factors of Clause 8.3 (Formula (8)). Finite-panel effects
are removed by normalising to the reference flat surface (Clause 8.2,
Formula (7)), and the random-incidence coefficient is the (weighted) average of
the directional coefficients over the source positions (Clause 8.4).

One subject: the free-field polar response of a surface and the single number
Clause 8 distils from it, band by band. Part 1 of ISO 17497 is a different
measurement, made in a reverberation room, and lives in
:mod:`phonometry.materials.diffusers.reverberation_room_scattering`; the two
parts share no formula, and the helpers are named per part so they are never
mixed. Neither part contains a numeric worked example.

The Part 2 diffusion coefficient is the design target of subwavelength diffuser
panels such as the metadiffusers of Jiménez, Cox, Romero-García & Groby
(2017, *Scientific Reports* 7, 5389, doi:10.1038/s41598-017-05710-5), whose
slow-sound resonant slots reach the diffusion of a Schroeder or
quadratic-residue diffuser in a fraction of the depth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .reverberation_room_scattering import _positive_scalar

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

    from ..._internal.types import Real
    from ..._report.metadata import ReportMetadata


__all__ = [
    "TWO_DIMENSIONAL_SOURCE_WEIGHTS",
    "DiffusionResult",
    "DiffusionSpectrum",
    "area_factors",
    "diffusion_spectrum",
    "directional_diffusion",
    "directional_diffusion_coefficient",
    "normalized_diffusion_coefficient",
    "random_incidence_diffusion",
]


#: Source-position weights for the two-dimensional (single-plane) random
#: incidence average of ISO 17497-2 Clause 8.4: the 0 deg source is weighted 1
#: and each of the four +/-30 deg, +/-60 deg sources is weighted 3.
TWO_DIMENSIONAL_SOURCE_WEIGHTS: tuple[int, ...] = (1, 3, 3, 3, 3)


@dataclass(frozen=True)
class DiffusionResult:
    """A measured polar response and its diffusion coefficient (ISO 17497-2).

    :ivar angles: Receiver angles of the polar response, in degrees.
    :ivar levels: Reflected sound-pressure level at each angle, in decibels.
    :ivar coefficient: Autocorrelation diffusion coefficient ``d`` (Formula (5)).
    """

    angles: Real
    levels: Real
    coefficient: float

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the polar response with the diffusion coefficient annotated.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        polar :class:`~matplotlib.axes.Axes` and never calls ``plt.show``.
        """
        from ..._i18n import check_language
        from ..._plot.materials import plot_diffusion_polar

        check_language(language)
        return plot_diffusion_polar(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an ISO 17497-2 polar-response test-report fiche to a PDF.

        Writes a one-page accredited free-field diffusion report for a single
        source position (ISO 17497-2:2012, Clause 8.5): the standard-basis line,
        an optional metadata header block, a two-panel body with the corrected
        polar-response table (receiver angle and reflected sound-pressure level
        ``L``, rounded to 0,1 dB) beside the semicircular polar plot, a boxed
        directional diffusion coefficient ``d_theta`` (Formula (5)/(6)) and a
        footer with the fixed disclaimer. ISO 17497-2 is a characterisation, so
        there is no pass/fail verdict.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`; ``None``
            produces a body-and-disclaimer fiche. The applicable descriptive
            fields are ``client``, ``manufacturer``, ``specimen``, ``mounting``,
            ``test_room``, ``test_date``, ``temperature``, ``relative_humidity``,
            ``pressure``, ``measurement_standard``, ``laboratory``, ``operator``,
            ``report_id`` and ``notes``. The ``requirement`` field is ignored
            (ISO 17497-2 has no verdict).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for signature parity; the polar-response fiche
            has no extended table, so it renders the same body.
        :param language: Fiche language: ``"en"`` (default, English, decimal
            point) or ``"es"`` (Spanish, decimal comma).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"``.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        from ..._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from ..._report.iso17497 import render_diffusion_polar_report

        return render_diffusion_polar_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


@dataclass(frozen=True)
class DiffusionSpectrum:
    """A diffusion-coefficient spectrum ``d(f)`` (ISO 17497-2, Clause 8.5).

    Where :class:`DiffusionResult` holds the polar response of a single
    one-third-octave band, this holds the diffusion coefficient across the
    measured bands, so it can be tabulated and plotted against frequency as
    Clause 8.5 requires. The per-band coefficient is a *directional* diffusion
    coefficient ``d_theta`` (Formula (5)/(6)) when it comes from one source
    position, or a *random-incidence* diffusion coefficient ``d`` when it is the
    per-band average of the directional coefficients over the source positions
    (Clause 8.4); the standard defines both as frequency-dependent quantities,
    so this carries a spectrum rather than a single number.

    :ivar frequencies: One-third-octave band centre frequencies, in hertz.
    :ivar diffusion: Diffusion coefficient ``d`` per band (directional per
        source, or random-incidence when averaged over source positions).
    :ivar normalized: Optional normalised diffusion coefficient ``d_n`` per band
        (Formula (7)), or ``None`` when the reference flat surface was not
        measured.
    """

    frequencies: Real
    diffusion: Real
    normalized: Real | None = None

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the diffusion coefficient ``d`` versus frequency.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes` and never calls ``plt.show``.
        """
        from ..._i18n import check_language
        from ..._plot.materials import plot_diffusion_report

        check_language(language)
        return plot_diffusion_report(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an ISO 17497-2 diffusion-coefficient test-report fiche to a PDF.

        Writes a one-page accredited free-field diffusion report
        (ISO 17497-2:2012, Clause 8.5): the standard-basis line, an optional
        metadata header block, a two-panel body with the per-band table
        (frequency, the diffusion coefficient ``d`` and, when present, the
        normalised ``d_n``) beside the ``d(f)`` curve on a categorical band
        axis, a boxed characterisation headline over the tested frequency range,
        and a footer with the fixed disclaimer. ISO 17497-2 is a
        characterisation, so there is no pass/fail verdict.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`; ``None``
            produces a body-and-disclaimer fiche whose header shows only the
            measured frequency range. The applicable descriptive fields are
            ``client``, ``manufacturer``, ``specimen``, ``mounting``,
            ``test_room``, ``test_date``, ``temperature``, ``relative_humidity``,
            ``pressure``, ``measurement_standard``, ``laboratory``, ``operator``,
            ``report_id`` and ``notes``. The ``requirement`` field is ignored
            (ISO 17497-2 has no verdict).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True`` and a normalised spectrum is present, the
            value table adds the normalised ``d_n`` column.
        :param language: Fiche language: ``"en"`` (default, English, decimal
            point) or ``"es"`` (Spanish, decimal comma).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"``.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        from ..._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from ..._report.iso17497 import render_diffusion_spectrum_report

        return render_diffusion_spectrum_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def diffusion_spectrum(
    frequencies: ArrayLike,
    diffusion: ArrayLike,
    *,
    normalized: ArrayLike | None = None,
) -> DiffusionSpectrum:
    """Diffusion-coefficient spectrum ``d(f)`` (ISO 17497-2, Clause 8.5).

    Pairs the per-band diffusion coefficients ``d`` with their band centres and
    returns a plottable, reportable :class:`DiffusionSpectrum`. The coefficient
    is the *directional* coefficient ``d_theta`` (Formula (5)/(6)) when it comes
    from a single source position, or the *random-incidence* coefficient ``d``
    when it is the per-band average of the directional coefficients over the
    source positions (Clause 8.4, e.g. via :func:`random_incidence_diffusion`
    band by band). The optional normalised coefficients ``d_n`` (Formula (7))
    are carried through when supplied.

    :param frequencies: One-third-octave band centres, in hertz (1-D).
    :param diffusion: Diffusion coefficient ``d`` per band.
    :param normalized: Optional normalised diffusion coefficient ``d_n`` per
        band; ``None`` when the reference flat surface was not measured.
    :return: A :class:`DiffusionSpectrum` with ``.plot()`` and ``.report()``.
    :raises ValueError: if the inputs differ in length, are empty or not 1-D.
    """
    freq = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    d = np.atleast_1d(np.asarray(diffusion, dtype=np.float64))
    if freq.ndim != 1 or freq.size == 0 or freq.shape != d.shape:
        raise ValueError(
            "'frequencies' and 'diffusion' must be non-empty, 1-D and equal-length."
        )
    d_n: Real | None = None
    if normalized is not None:
        d_n = np.atleast_1d(np.asarray(normalized, dtype=np.float64))
        if d_n.shape != freq.shape:
            raise ValueError("'normalized' must match 'frequencies' in length.")
    return DiffusionSpectrum(
        frequencies=freq,
        diffusion=d,
        normalized=d_n,
    )


def directional_diffusion(
    angles: ArrayLike,
    levels: ArrayLike,
    *,
    weights: ArrayLike | None = None,
) -> DiffusionResult:
    """Diffusion coefficient of a polar response (ISO 17497-2, Formula (5)/(6)).

    Convenience wrapper over :func:`directional_diffusion_coefficient` that
    keeps the receiver angles alongside the levels and returns a plottable
    :class:`DiffusionResult`.

    :param angles: Receiver angles of the polar response, in degrees (1-D).
    :param levels: Reflected sound-pressure level at each angle, in decibels.
    :param weights: Optional area weights ``N_i`` (Formula (8)); ``None`` uses
        the equal-area Formula (5).
    :return: A :class:`DiffusionResult` with ``.plot()``.
    :raises ValueError: if ``angles`` and ``levels`` differ in length or are
        shorter than two receivers.
    """
    ang = np.atleast_1d(np.asarray(angles, dtype=np.float64))
    lev = np.atleast_1d(np.asarray(levels, dtype=np.float64))
    if ang.shape != lev.shape:
        raise ValueError("'angles' and 'levels' must have the same length.")
    d = float(directional_diffusion_coefficient(lev, area_weights=weights))
    return DiffusionResult(angles=ang, levels=lev, coefficient=d)


# ---------------------------------------------------------------------------
# ISO 17497-2: directional and random-incidence diffusion (Clause 8).
# ---------------------------------------------------------------------------
def directional_diffusion_coefficient(
    levels: ArrayLike,
    *,
    area_weights: ArrayLike | None = None,
) -> float:
    r"""Directional diffusion coefficient ``d_theta`` (ISO 17497-2, Formulas (5)/(6)).

    For a fixed source position and one-third-octave band, from the ``n``
    reflected sound-pressure levels ``L_i`` (dB). With equal-area receivers
    (``area_weights is None``, Formula (5)):

    .. math::

       d_\theta = \frac{\left( \sum_i p_i \right)^{2} - \sum_i p_i^{2}}
       {(n - 1) \sum_i p_i^{2}}

    where :math:`p_i = 10^{L_i / 10}`. When each receiver samples a different
    area (Formula (6)) the per-receiver weights ``N_i`` (from
    :func:`area_factors`) enter:

    .. math::

       d_\theta = \frac{\left( \sum_i p_i N_i \right)^{2}
       - \sum_i N_i p_i^{2}}
       {\left( \sum_i N_i - 1 \right) \sum_i N_i p_i^{2}}

    which reduces to Formula (5) for uniform weights. The coefficient is 0 when
    only one receiver has non-zero scattered energy and 1 when all receivers
    are equal.

    :param levels: The :math:`n \ge 2` reflected sound-pressure levels
        ``L_i``, in
        decibels (a level of ``-inf`` denotes a receiver with zero energy).
    :param area_weights: Optional per-receiver area weights ``N_i`` (Formula (8));
        ``None`` selects the equal-area Formula (5).
    :return: Directional diffusion coefficient ``d_theta`` (a scalar).
    :raises ValueError: for fewer than two receivers, a non-1-D input, a length
        mismatch, or non-positive total weight.
    """
    lvl = np.asarray(levels, dtype=np.float64)
    if lvl.ndim != 1:
        raise ValueError("'levels' must be a 1-D sequence of receiver SPLs.")
    n = lvl.size
    if n < 2:
        raise ValueError("'levels' needs at least two receivers (n >= 2).")
    p = np.power(10.0, lvl / 10.0)
    if area_weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.asarray(area_weights, dtype=np.float64)
        if weights.ndim != 1 or weights.size != n:
            raise ValueError(
                "'area_weights' must match the number of receivers "
                f"({n}), got shape {weights.shape}."
            )
        if np.any(weights <= 0.0):
            raise ValueError("'area_weights' values must be positive.")
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1.0:
        raise ValueError("The total area weight must exceed 1.")
    weighted_energy = np.sum(p * weights)
    weighted_energy_sq = np.sum(weights * p**2)
    if weighted_energy_sq <= 0.0:
        raise ValueError(
            "The polar response carries no energy (all levels -inf); the "
            "diffusion coefficient is undefined."
        )
    numerator = weighted_energy**2 - weighted_energy_sq
    denominator = (weight_sum - 1.0) * weighted_energy_sq
    return float(numerator / denominator)


def normalized_diffusion_coefficient(
    d_theta: ArrayLike,
    d_theta_reference: ArrayLike,
) -> Real:
    r"""Normalised directional diffusion coefficient (ISO 17497-2, Formula (7)).

    :math:`d_{\theta,\mathrm{n}} = (d_\theta - d_{\theta,\mathrm{r}}) / (1 - d_{\theta,\mathrm{r}})`,
    removing the
    finite-panel diffusion of the reference flat surface ``d_theta_r`` (same
    projected footprint as the test surface). It maps
    :math:`d_\theta = d_{\theta,\mathrm{r}}`
    to 0 and :math:`d_\theta = 1` to 1.

    :param d_theta: Directional diffusion coefficient of the test surface.
    :param d_theta_reference: Directional diffusion coefficient of the
        reference flat surface ``d_theta_r``.
    :return: Normalised directional diffusion coefficient ``d_theta_n``.
    :raises ValueError: if any reference coefficient equals 1 (undefined ratio).
    """
    d = np.asarray(d_theta, dtype=np.float64)
    d_ref = np.asarray(d_theta_reference, dtype=np.float64)
    denom = 1.0 - d_ref
    if np.any(np.isclose(denom, 0.0)):
        raise ValueError("'d_theta_reference' must not equal 1 (division by zero).")
    return np.asarray((d - d_ref) / denom, dtype=np.float64)


def area_factors(
    elevations: ArrayLike,
    *,
    delta_theta: float,
    delta_phi: float | None = None,
) -> Real:
    r"""Per-receiver area weights ``N_i`` (ISO 17497-2, Clause 8.3, Formula (8)).

    For a hemispherical measurement the solid-angle area sampled by a receiver
    at elevation ``theta`` (with angular spacings ``delta_theta``,
    ``delta_phi``) is:

    .. math::

       \begin{aligned}
       A_i &= (4 \pi / \Delta\phi) \sin^{2}(\Delta\theta / 4)
       && \text{for } \theta = 0^\circ \\
       A_i &= 2 \sin(\theta) \sin(\Delta\theta / 2)
       && \text{for } \theta \ne 0^\circ, 90^\circ \\
       A_i &= \sin(\Delta\theta / 2)
       && \text{for } \lvert \theta \rvert = 90^\circ
       \end{aligned}

    and :math:`N_i = A_i / A_{\min}` (Formula (8)), with ``A_min`` the
    smallest ``A_i``. All angles are handled internally in **radians**; the
    :math:`\theta = 0` form in particular requires ``delta_phi`` in radians to
    be dimensionally consistent with the :math:`4 \pi` factor.

    :param elevations: Receiver elevation angles ``theta`` from the reference
        normal, in **degrees** (1-D), over the measurement domain
        :math:`0 \le \theta \le 90` (Figure 7). Formula (8) assumes a single
        receiver
        at :math:`\theta = 0` (the zenith); duplicate zenith entries would
        each take
        the full zenith area.
    :param delta_theta: Elevation spacing between adjacent receivers, in degrees
        (typically 5).
    :param delta_phi: Azimuth spacing between adjacent receivers, in degrees;
        defaults to ``delta_theta``. Required (implicitly) for the
        :math:`\theta = 0`
        receiver.
    :return: Per-receiver area weights ``N_i`` (dimensionless, min value 1).
    :raises ValueError: for a non-1-D input or non-positive spacings.
    """
    theta_deg = np.asarray(elevations, dtype=np.float64)
    if theta_deg.ndim != 1 or theta_deg.size == 0:
        raise ValueError("'elevations' must be a non-empty 1-D sequence of angles.")
    d_theta = _positive_scalar(delta_theta, "delta_theta")
    d_phi = d_theta if delta_phi is None else _positive_scalar(delta_phi, "delta_phi")
    theta = np.radians(theta_deg)
    d_theta_rad = math.radians(d_theta)
    d_phi_rad = math.radians(d_phi)

    at_zenith = np.isclose(theta_deg, 0.0)
    at_pole = np.isclose(np.abs(theta_deg), 90.0)
    general = ~at_zenith & ~at_pole

    areas = np.empty_like(theta_deg)
    areas[at_zenith] = 4.0 * np.pi / d_phi_rad * np.sin(d_theta_rad / 4.0) ** 2
    areas[at_pole] = np.sin(d_theta_rad / 2.0)
    areas[general] = 2.0 * np.sin(theta[general]) * np.sin(d_theta_rad / 2.0)
    a_min = np.min(areas)
    if a_min <= 0.0:
        raise ValueError("Area factors must be positive; check the angles.")
    return np.asarray(areas / a_min, dtype=np.float64)


def random_incidence_diffusion(
    directional_coefficients: ArrayLike,
    *,
    weights: ArrayLike | None = None,
) -> float:
    """Random-incidence diffusion coefficient ``d`` (ISO 17497-2, Clause 8.4).

    The (normalised or non-normalised) directional coefficients are averaged
    over the source positions. Hemispherical measurements use **equal**
    weightings (``weights is None``); two-dimensional (single-plane)
    measurements use the source weighting of Clause 8.4 - weight 1 for the
    0 deg source and weight 3 for each of the four +/-30 deg, +/-60 deg sources
    (see :data:`TWO_DIMENSIONAL_SOURCE_WEIGHTS`).

    :param directional_coefficients: Directional diffusion coefficients
        ``d_theta`` (or ``d_theta_n``), one per source position (1-D).
    :param weights: Optional source-position weights; ``None`` averages with
        equal weight.
    :return: Random-incidence diffusion coefficient ``d`` (a scalar).
    :raises ValueError: for an empty or non-1-D input, a length mismatch, or
        non-positive total weight.
    """
    d = np.asarray(directional_coefficients, dtype=np.float64)
    if d.ndim != 1:
        raise ValueError("'directional_coefficients' must be a 1-D sequence.")
    if d.size == 0:
        raise ValueError("'directional_coefficients' must not be empty.")
    if weights is None:
        w = np.ones(d.size, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or w.size != d.size:
            raise ValueError(
                "'weights' must match the number of source positions "
                f"({d.size}), got shape {w.shape}."
            )
        if np.any(w < 0.0):
            raise ValueError("'weights' values must be non-negative.")
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("The total source weight must be positive.")
    return float(np.sum(w * d) / total)
