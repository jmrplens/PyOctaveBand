#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Metadiffusers: deep-subwavelength Schroeder-like sound diffusers.

A metadiffuser is a rigidly backed slotted panel whose slits are each loaded
by an array of Helmholtz resonators (Jimenez, Cox, Romero-Garcia and Groby,
*Metadiffusers: Deep-subwavelength sound diffusers*, Sci. Rep. 7, 5389,
2017). The resonators slow the sound inside each slit, so the slit reaches
its quarter-wavelength condition at a fraction of the depth a plain well
would need; by giving every slit a different geometry the panel presents a
spatially dependent complex reflection coefficient ``R_n(f)`` along its
face. Tuning that profile to a Schroeder phase grating reproduces the
scattering of a quadratic-residue or primitive-root diffuser from a panel
1/46 to 1/20 of the design wavelength thick, and driving single slits to
critical coupling adds the perfectly absorbing ``0`` state that ternary
sequences require.

Each slit is modelled with the transfer-matrix chain of
:func:`~phonometry.materials.absorbers.slow_sound.slit_helmholtz_absorber`
(visco-thermal effective parameters, resonator end corrections and slit
radiation correction); the panel is locally reacting, so the wells do not
couple internally and the far field follows from the Fraunhofer integral of
the per-well reflection sequence
(:func:`~phonometry.materials.diffusers.design.predict_diffuser_polar_response`)
reduced to the ISO 17497-2 directional diffusion coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._internal.validation import require_positive, require_positive_array
from ..absorbers.porous import DEFAULT_AIR, AirProperties, Complex
from ..absorbers.slow_sound import HelmholtzResonator, slit_helmholtz_absorber
from .design import (
    DEFAULT_POLAR_ANGLES,
    DiffuserPolarResponse,
    predict_diffuser_polar_response,
)
from .scattering_diffusion import (
    DiffusionSpectrum,
    diffusion_spectrum,
    normalized_diffusion_coefficient,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

    from ..._internal.types import Real

__all__ = [
    "MetadiffuserResult",
    "MetadiffuserWell",
    "metadiffuser_diffusion_spectrum",
    "metadiffuser_polar_response",
    "metadiffuser_reflection",
]


@dataclass(frozen=True)
class MetadiffuserWell:
    r"""One slit of a metadiffuser panel.

    ``slit_height`` is the slit opening ``h_n`` along the panel face and
    ``resonators`` the Helmholtz resonators loading the slit, ordered from
    the panel face towards the rigid backing; the resonator lattice step is
    :math:`a = L / M` for a panel of depth ``L`` and ``M`` resonators. All
    lengths are in metres. A ``None`` entry in a well sequence stands for a
    flat rigid strip of the panel face (:math:`R = 1`), the ``+1`` state of
    ternary-sequence designs.
    """

    slit_height: float
    resonators: tuple[HelmholtzResonator, ...]

    def __post_init__(self) -> None:
        """Reject a non-positive ``slit_height`` and a well with no resonators."""
        require_positive(self.slit_height, "slit_height")
        if not self.resonators:
            msg = "'resonators' must contain at least one resonator."
            raise ValueError(msg)


@dataclass(frozen=True)
class MetadiffuserResult:
    r"""Spectra of a metadiffuser panel, one reflection row per well.

    ``reflection`` has shape ``(N, len(frequency))`` with the complex
    pressure reflection factor of each well (flat strips are exactly ``1``),
    ``absorption`` is the face-averaged energy absorption
    :math:`\alpha(f) = 1 - \operatorname{mean}_n \lvert R_n \rvert^2` and
    ``well_absorption`` the per-well
    :math:`\alpha_n = 1 - \lvert R_n \rvert^2`. The trailing fields retain the geometry the
    prediction was run with (``wells``, ``depth``, ``period``) so
    :meth:`plot_geometry` can draw the panel section; they default to
    ``None`` for hand-built results.
    """

    frequency: Real
    reflection: Complex
    absorption: Real
    well_absorption: Real
    wells: tuple[MetadiffuserWell | None, ...] | None = None
    depth: float | None = None
    period: float | None = None

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the per-well and face-averaged absorption spectra.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.materials import plot_metadiffuser_absorption

        check_language(language)
        return plot_metadiffuser_absorption(self, ax=ax, language=language, **kwargs)

    def plot_geometry(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Draw the panel cross-section to scale (slits and resonators).

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :raises ValueError: If the result does not retain its geometry.
        """
        from ..._i18n import check_language
        from ..._plot.geometry import plot_metadiffuser_geometry

        check_language(language)
        return plot_metadiffuser_geometry(self, ax=ax, language=language, **kwargs)


def _check_panel(
    wells: Any, depth: float, period: float
) -> tuple[MetadiffuserWell | None, ...]:
    """Validate the well sequence against the panel depth and period."""
    require_positive(depth, "depth")
    require_positive(period, "period")
    cells = tuple(wells)
    if len(cells) < 2:
        msg = "'wells' must contain at least two wells."
        raise ValueError(msg)
    for i, well in enumerate(cells):
        if well is None:
            continue
        if not isinstance(well, MetadiffuserWell):
            msg = (
                f"wells[{i}] must be a MetadiffuserWell or None, "
                f"got {type(well).__name__}."
            )
            raise TypeError(msg)
        if well.slit_height >= period:
            msg = f"wells[{i}].slit_height must be smaller than the period."
            raise ValueError(msg)
    return cells


def metadiffuser_reflection(
    frequency: ArrayLike,
    wells: Sequence[MetadiffuserWell | None],
    *,
    depth: float,
    period: float,
    angle: float = 0.0,
    resonator_geometry: str = "slit",
    air: AirProperties = DEFAULT_AIR,
) -> MetadiffuserResult:
    r"""Per-well reflection spectra of a metadiffuser panel (Sci. Rep. Eq. (6)).

    Runs the rigidly backed slit transfer-matrix chain of
    :func:`~phonometry.materials.absorbers.slow_sound.slit_helmholtz_absorber`
    once per well: each :class:`MetadiffuserWell` becomes a slit of height
    ``h_n`` and depth ``L`` loaded by its ``M`` resonators on the lattice
    :math:`a = L / M`, and ``None`` wells are flat rigid strips with
    :math:`R = 1`.
    The panel is locally reacting, so a well's reflection does not depend on
    its neighbours and the incidence ``angle`` enters only through the front
    air impedance.

    :param frequency: Frequency vector ``f``, in hertz.
    :param wells: Sequence of :class:`MetadiffuserWell` (or ``None`` for a
        flat rigid strip) describing one period of the panel face.
    :param depth: Panel depth ``L`` common to all slits, in metres.
    :param period: Well pitch ``d`` along the panel face, in metres.
    :param angle: Polar angle of incidence ``theta``, in radians.
    :param resonator_geometry: ``"slit"`` (default) for the paper's
        two-dimensional resonators, ``"square"`` for square-duct necks
        and cavities.
    :param air: State of the air the panel radiates into and the slits and
        resonators are filled with
        (:class:`~phonometry.materials.absorbers.porous.AirProperties`): its
        speed of sound ``c0``, density ``rho0``, viscosity ``eta``, Prandtl
        number ``Pr``, ratio of specific heats ``gamma`` and static pressure
        ``P0``.
    :return: A :class:`MetadiffuserResult` with one reflection row per well.
    """
    f = require_positive_array(frequency, "frequency")
    cells = _check_panel(wells, depth, period)
    rows = np.empty((len(cells), f.size), dtype=np.complex128)
    for i, well in enumerate(cells):
        if well is None:
            rows[i] = 1.0
            continue
        prediction = slit_helmholtz_absorber(
            f,
            well.resonators,
            slit_height=well.slit_height,
            lattice_step=depth / len(well.resonators),
            period=period,
            angle=angle,
            resonator_geometry=resonator_geometry,
            air=air,
        )
        rows[i] = prediction.reflection
    well_alpha = 1.0 - np.abs(rows) ** 2
    return MetadiffuserResult(
        frequency=f,
        reflection=rows,
        absorption=np.asarray(well_alpha.mean(axis=0), dtype=np.float64),
        well_absorption=np.asarray(well_alpha, dtype=np.float64),
        wells=cells,
        depth=float(depth),
        period=float(period),
    )


def metadiffuser_polar_response(
    frequency: float,
    wells: Sequence[MetadiffuserWell | None],
    *,
    depth: float,
    period: float,
    angles: ArrayLike = DEFAULT_POLAR_ANGLES,
    source_angle: float = 0.0,
    periods: int = 1,
    resonator_geometry: str = "slit",
    air: AirProperties = DEFAULT_AIR,
) -> DiffuserPolarResponse:
    """Far-field polar response of a metadiffuser at one frequency.

    Computes the per-well complex reflection sequence at ``frequency`` with
    :func:`metadiffuser_reflection` (the panel is locally reacting, so the
    slit chains see the incidence angle ``source_angle``) and evaluates the
    Fraunhofer far field and ISO 17497-2 directional diffusion coefficient
    with
    :func:`~phonometry.materials.diffusers.design.predict_diffuser_polar_response`.

    :param frequency: Frequency of the prediction ``f``, in hertz.
    :param wells: Sequence of :class:`MetadiffuserWell` (or ``None`` for a
        flat rigid strip) describing one period of the panel face.
    :param depth: Panel depth ``L`` common to all slits, in metres.
    :param period: Well pitch ``d`` along the panel face, in metres; it is
        the ``well_width`` of the far-field model.
    :param angles: Receiver reflection angles ``theta``, in degrees.
    :param source_angle: Angle of incidence ``psi`` of the source, in
        degrees; also applied to the local slit reflection.
    :param periods: Number of repetitions ``N_p`` of the single period; the
        grating lobes of a Schroeder-like design require ``periods >= 2``.
    :param resonator_geometry: ``"slit"`` (default) for the paper's
        two-dimensional resonators, ``"square"`` for square-duct necks
        and cavities.
    :param air: State of the air the panel radiates into and the slits and
        resonators are filled with
        (:class:`~phonometry.materials.absorbers.porous.AirProperties`); its
        speed of sound ``c0`` also carries the far field.
    :return: A
        :class:`~phonometry.materials.diffusers.design.DiffuserPolarResponse`.
    """
    f = require_positive(frequency, "frequency")
    result = metadiffuser_reflection(
        np.asarray([f]),
        wells,
        depth=depth,
        period=period,
        angle=float(np.radians(source_angle)),
        resonator_geometry=resonator_geometry,
        air=air,
    )
    # Sci. Rep. Eq. (1) is the bare Fraunhofer integral of R(x): the
    # piecewise-constant wells carry the aperture factor, but there is no
    # Kirchhoff obliquity term, so it is disabled here for fidelity.
    return predict_diffuser_polar_response(
        period,
        f,
        reflection=result.reflection[:, 0],
        angles=angles,
        source_angle=source_angle,
        periods=periods,
        speed_of_sound=air.speed_of_sound,
        include_obliquity=False,
    )


def metadiffuser_diffusion_spectrum(
    frequencies: ArrayLike,
    wells: Sequence[MetadiffuserWell | None],
    *,
    depth: float,
    period: float,
    angles: ArrayLike = DEFAULT_POLAR_ANGLES,
    source_angle: float = 0.0,
    periods: int = 1,
    resonator_geometry: str = "slit",
    air: AirProperties = DEFAULT_AIR,
) -> DiffusionSpectrum:
    r"""Normalized diffusion-coefficient spectrum ``d_n(f)`` of a metadiffuser.

    Evaluates the far-field polar response at each frequency with
    :func:`metadiffuser_polar_response`, forms the ISO 17497-2 directional
    diffusion coefficient band by band and normalises it against a flat
    rigid reference of the same footprint (all wells :math:`R = 1`) with
    :func:`~phonometry.materials.diffusers.scattering_diffusion.normalized_diffusion_coefficient`,
    exactly as the paper reports ``delta_n``.

    :param frequencies: Frequencies of the spectrum, in hertz (1-D).
    :param wells: Sequence of :class:`MetadiffuserWell` (or ``None`` for a
        flat rigid strip) describing one period of the panel face.
    :param depth: Panel depth ``L`` common to all slits, in metres.
    :param period: Well pitch ``d`` along the panel face, in metres.
    :param angles: Receiver reflection angles ``theta``, in degrees.
    :param source_angle: Angle of incidence ``psi``, in degrees.
    :param periods: Number of repetitions ``N_p`` of the single period.
    :param resonator_geometry: ``"slit"`` (default) for the paper's
        two-dimensional resonators, ``"square"`` for square-duct necks
        and cavities.
    :param air: State of the air the panel radiates into and the slits and
        resonators are filled with
        (:class:`~phonometry.materials.absorbers.porous.AirProperties`); its
        speed of sound ``c0`` also carries the far field.
    :return: A
        :class:`~phonometry.materials.diffusers.scattering_diffusion.DiffusionSpectrum`
        carrying the raw ``d(f)`` and the normalised ``d_n(f)``.
    """
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if freqs.ndim != 1 or freqs.size == 0:
        msg = "'frequencies' must be a non-empty 1-D sequence."
        raise ValueError(msg)
    cells = _check_panel(wells, depth, period)
    flat = np.ones(len(cells), dtype=np.complex128)
    result = metadiffuser_reflection(
        freqs,
        cells,
        depth=depth,
        period=period,
        angle=float(np.radians(source_angle)),
        resonator_geometry=resonator_geometry,
        air=air,
    )
    raw = np.empty(freqs.size, dtype=np.float64)
    norm = np.empty(freqs.size, dtype=np.float64)
    for i, f in enumerate(freqs):
        surface = predict_diffuser_polar_response(
            period,
            float(f),
            reflection=result.reflection[:, i],
            angles=angles,
            source_angle=source_angle,
            periods=periods,
            speed_of_sound=air.speed_of_sound,
            include_obliquity=False,
        )
        reference = predict_diffuser_polar_response(
            period,
            float(f),
            reflection=flat,
            angles=angles,
            source_angle=source_angle,
            periods=periods,
            speed_of_sound=air.speed_of_sound,
            include_obliquity=False,
        )
        raw[i] = surface.coefficient
        norm[i] = float(
            normalized_diffusion_coefficient(surface.coefficient, reference.coefficient)
        )
    return diffusion_spectrum(freqs, raw, normalized=norm)
