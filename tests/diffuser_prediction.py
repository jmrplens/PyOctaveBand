#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Shared ISO 17497-2 diffuser-prediction helpers (tests + conformance report).

Reproduces, with the library's own Fraunhofer far-field phase-grating model
(:mod:`phonometry.materials.diffusers.design`), the published geometry behind
the ``ISO17497_2_*`` reference levels and the Cox & D'Antonio Appendix B
external anchor of :mod:`reference_data`: an N = 7 quadratic-residue diffuser,
6 periods, 3.6 m total width, 0.2 m maximum well depth (Cox & D'Antonio,
"Acoustic Absorbers and Diffusers", 3rd ed., Appendix B section 7; the
commercial N = 7 QRD of Hargreaves, Cox, Lam & D'Antonio, J. Acoust. Soc. Am.
108(4), 1710-1720 (2000), Table I is the same diffuser family). The flat
reference is the equal-footprint zero-depth panel, exactly as the module's
``predicted_diffusion_spectrum`` normalisation pathway models it.

Both ``tests/materials/diffusers/test_scattering_diffusion.py`` and
``scripts/conformance_report.py`` import these helpers, so the committed
reference levels and the Appendix B anchor can never be recomputed two
different ways.
"""

from __future__ import annotations

import numpy as np
import reference_data as ref

from phonometry.materials.diffusers.design import (
    DiffuserPolarResponse,
    predict_diffuser_polar_response,
    qrd_well_depths,
)
from phonometry.materials.diffusers.scattering_diffusion import (
    directional_diffusion_coefficient,
    normalized_diffusion_coefficient,
)


def published_qrd_depths() -> np.ndarray:
    """Well depths of the published N = 7 QRD (deepest well exactly 0.2 m)."""
    return np.asarray(
        qrd_well_depths(
            ref.ISO17497_2_QRD_N,
            ref.ISO17497_2_QRD_DESIGN_FREQUENCY,
            speed_of_sound=ref.ISO17497_2_SPEED_OF_SOUND,
        ),
        dtype=np.float64,
    )


def predicted_arc(
    frequency: float, *, flat: bool = False, source_angle: float = 0.0
) -> DiffuserPolarResponse:
    """Model polar response of the published geometry (or its flat reference).

    Single-frequency Fraunhofer prediction on the standard 37-point,
    5-degree semicircular arc, normal incidence by default.
    """
    depths = published_qrd_depths()
    if flat:
        depths = np.zeros_like(depths)
    return predict_diffuser_polar_response(
        ref.ISO17497_2_QRD_WELL_WIDTH,
        frequency,
        depths=depths,
        periods=ref.ISO17497_2_QRD_PERIODS,
        source_angle=source_angle,
        speed_of_sound=ref.ISO17497_2_SPEED_OF_SOUND,
    )


def predicted_band_normalized_diffusion(
    band_center: float, *, source_angle: float = 0.0
) -> float:
    """One-third-octave band-averaged normalised diffusion coefficient d_n.

    Follows the construction of the Cox & D'Antonio Appendix B table
    (section 5.2.5): each band polar response is the energy average of seven
    single-frequency responses (here spread geometrically across the
    base-two band, ``fc * 2**(+/-1/6)``), reduced with ISO 17497-2
    Formula (5) and normalised against the equal-footprint flat reference
    with Formula (7).
    """
    freqs = np.geomspace(
        band_center * 2.0 ** (-1.0 / 6.0), band_center * 2.0 ** (1.0 / 6.0), 7
    )
    def band_levels(flat: bool) -> np.ndarray:
        energy = np.zeros(37, dtype=np.float64)
        for f in freqs:
            arc = predicted_arc(float(f), flat=flat, source_angle=source_angle)
            energy += 10.0 ** (np.asarray(arc.levels, dtype=np.float64) / 10.0)
        return np.asarray(10.0 * np.log10(energy / freqs.size))
    d = directional_diffusion_coefficient(band_levels(flat=False))
    d_ref = directional_diffusion_coefficient(band_levels(flat=True))
    return float(normalized_diffusion_coefficient(d, d_ref))
