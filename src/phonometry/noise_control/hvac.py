#  Copyright (c) 2026. Jose M. Requena-Plens
"""
HVAC duct acoustics: end reflection, bends, plenums and flow-generated noise.

A ventilation duct network attenuates fan noise through several mechanisms
that add up along the path, and it *regenerates* noise wherever the airflow is
disturbed. This module gathers the engineering methods of Bies, Hansen &
Howard, *Engineering Noise Control* 5th ed., Chapter 8, for the passive
attenuations -- **duct end reflection** (§8.13, Table 8.14), **bends/elbows**
(§8.11, Table 8.11) and **plenum chambers** (§8.17, Wells' method) -- and for
the **flow-generated (self) noise** of straight ducts and bends (§8.15).

The end-reflection and elbow methods are empirical look-up tables (ASHRAE);
they are interpolated over the duct size and, for the elbows, over the
frequency-to-width ratio ``W / lambda``. The plenum and flow-noise methods are
closed forms evaluated directly.

.. note::
   Bies 5th ed. gives the duct end reflection only as the ASHRAE Table 8.14
   look-up (there is no closed form in this edition); this module reproduces
   that table and interpolates it. Rectangular ducts use the equivalent
   diameter ``D = sqrt(4 S / pi)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._internal.validation import require_positive
from ..room.steady_field import room_constant

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

_C_AIR = 343.0

# ---------------------------------------------------------------------------
# Bies Table 8.14 -- duct end reflection loss (dB), ASHRAE.
# Rows: internal diameter (mm). Columns: octave band centre (Hz).
# Two termination conditions: "flush" (duct flush with a wall/ceiling) and
# "free" (free space / suspended in the room).
# ---------------------------------------------------------------------------
_END_REFLECTION_BANDS: NDArray[np.float64] = np.array(
    [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0]
)
_END_REFLECTION_DIAMETERS_MM: NDArray[np.float64] = np.array(
    [150, 200, 250, 300, 400, 510, 610, 710, 810, 910, 1220, 1830], dtype=float
)
_END_REFLECTION_FLUSH: NDArray[np.float64] = np.array(
    [
        [18, 12, 7, 3, 1, 0],
        [15, 10, 5, 2, 1, 0],
        [14, 8, 4, 1, 0, 0],
        [12, 7, 3, 1, 0, 0],
        [10, 5, 2, 1, 0, 0],
        [8, 4, 1, 0, 0, 0],
        [7, 3, 1, 0, 0, 0],
        [6, 2, 1, 0, 0, 0],
        [5, 2, 1, 0, 0, 0],
        [4, 2, 0, 0, 0, 0],
        [3, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
    ],
    dtype=float,
)
_END_REFLECTION_FREE: NDArray[np.float64] = np.array(
    [
        [20, 14, 9, 5, 2, 1],
        [18, 12, 7, 3, 1, 0],
        [16, 11, 6, 2, 1, 0],
        [14, 9, 5, 2, 1, 0],
        [12, 7, 3, 1, 0, 0],
        [10, 6, 2, 1, 0, 0],
        [9, 5, 2, 1, 0, 0],
        [8, 4, 1, 0, 0, 0],
        [7, 3, 1, 0, 0, 0],
        [6, 3, 1, 0, 0, 0],
        [5, 2, 0, 0, 0, 0],
        [3, 1, 0, 0, 0, 0],
    ],
    dtype=float,
)

# ---------------------------------------------------------------------------
# Bies Table 8.11 -- elbow/bend insertion loss (dB per bend) vs W / lambda.
# The five columns are the supported (bend_type, vanes, lined, round) cases;
# each row is the value for the W/lambda band whose upper edge is the key. The
# printed rows read "W/lambda < 0.14", "0.14 <= W/lambda < 0.28" and so on, so
# each edge opens its row rather than closing the one below (see
# :func:`elbow_insertion_loss`).
# ---------------------------------------------------------------------------
_ELBOW_WL_UPPER: NDArray[np.float64] = np.array(
    [0.14, 0.28, 0.55, 1.11, 2.22, np.inf]
)
_ELBOW_TABLE: dict[str, NDArray[np.float64]] = {
    # square, no vanes, unlined
    "square": np.array([0, 1, 5, 8, 4, 3], dtype=float),
    # square, no vanes, lined
    "square_lined": np.array([0, 1, 6, 11, 10, 10], dtype=float),
    # square, with vanes, unlined
    "square_vanes": np.array([0, 1, 4, 6, 4, 4], dtype=float),
    # square, with vanes, lined
    "square_vanes_lined": np.array([0, 1, 4, 7, 7, 7], dtype=float),
    # round, no vanes, unlined
    "round": np.array([0, 1, 2, 3, 3, 3], dtype=float),
}


def _frequencies(frequencies: ArrayLike) -> NDArray[np.float64]:
    f = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if f.ndim != 1 or f.size == 0:
        raise ValueError("'frequencies' must be a non-empty 1-D array.")
    if np.any(f <= 0.0) or not np.all(np.isfinite(f)):
        raise ValueError("'frequencies' must be positive and finite.")
    return f


@dataclass(frozen=True)
class HvacSpectrumResult:
    """A per-frequency HVAC quantity (attenuation or regenerated power level).

    :ivar frequencies: Frequencies ``f``, Hz.
    :ivar values: The quantity per frequency (dB, or dB re 1e-12 W for a
        sound power level).
    :ivar quantity: What ``values`` holds (``"attenuation"`` or
        ``"sound_power_level"``).
    :ivar label: A short human label of the element.
    """

    frequencies: np.ndarray
    values: np.ndarray
    quantity: str
    label: str

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the quantity against a continuous log-frequency axis.

        Requires matplotlib (``pip install phonometry[plot]``).
        """
        from .._i18n import check_language
        from .._plot.noise_control import plot_hvac_spectrum

        check_language(language)
        return plot_hvac_spectrum(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an HVAC duct-noise-spectrum fiche to ``path``.

        Writes a one-page HVAC-noise sheet: the method-basis line naming the
        reported quantity and the Bies, Hansen & Howard chapter (Engineering
        Noise Control 5th ed., Chapter 8), an optional metadata header (client,
        duct element, test environment, instrumentation, climate, date), a
        per-band table (nominal frequency and the reported quantity) beside the
        spectrum, the boxed single-number result (for a regenerated-noise
        spectrum the A-weighted sound power level ``L_WA`` re 1 pW with the
        overall unweighted total; for an attenuation spectrum the mean
        attenuation with its band range), an optional verdict row against a
        declared limit, and a method-basis strip stating the reported quantity's
        relation.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the header (``client``, ``specimen`` the duct element, ``test_room``
            the test environment, ``instrumentation``, ``temperature``,
            ``relative_humidity``, ``pressure``, ``test_date``), the footer
            identity (``laboratory``, ``operator``, ``report_id``, ``notes``)
            and, via ``requirement``, a declared maximum A-weighted sound power
            level for a regenerated-noise spectrum (lower is better) or a
            declared minimum mean attenuation for an attenuation spectrum (more
            is better).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True`` a regenerated-noise table adds the
            A-weighting correction and the A-weighted band level columns.
        :param language: Fiche language: ``"en"`` (default) or ``"es"``.
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"`` or ``language``
            is unknown.
        :raises ImportError: If reportlab (or, for the figure, matplotlib) is
            not installed (``pip install phonometry[report]``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from .._report.hvac import render_hvac_report

        return render_hvac_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def end_reflection_loss(
    frequencies: ArrayLike,
    diameter: float,
    *,
    termination: str = "flush",
    speed_of_sound: float = _C_AIR,
) -> HvacSpectrumResult:
    """Duct end reflection loss (Bies Table 8.14, ASHRAE).

    The low-frequency reflection of sound back up a duct at its open
    termination into a room. Interpolated over ``log`` diameter and ``log``
    frequency from Table 8.14; it passes exactly through the tabulated
    ``(diameter, octave band)`` nodes.

    :param frequencies: Frequencies ``f``, Hz (1-D array).
    :param diameter: Duct internal diameter ``D``, m (use
        ``D = sqrt(4 S / pi)`` for a rectangular duct of area ``S``).
    :param termination: ``"flush"`` (duct flush with a wall/ceiling) or
        ``"free"`` (free space / suspended in the room).
    :param speed_of_sound: Speed of sound ``c``, m/s (kept for signature
        symmetry; the table is indexed by frequency directly).
    :return: A :class:`HvacSpectrumResult` of the reflection loss, dB.
    """
    f = _frequencies(frequencies)
    d_mm = require_positive(diameter, "diameter") * 1000.0
    if termination == "flush":
        table = _END_REFLECTION_FLUSH
    elif termination == "free":
        table = _END_REFLECTION_FREE
    else:
        raise ValueError("'termination' must be 'flush' or 'free'.")
    require_positive(speed_of_sound, "speed_of_sound")

    log_d = np.log(_END_REFLECTION_DIAMETERS_MM)
    log_f_band = np.log(_END_REFLECTION_BANDS)
    # Interpolate the table in log-diameter for each band, then in log-freq.
    per_band = np.array(
        [np.interp(np.log(d_mm), log_d, table[:, j]) for j in range(table.shape[1])]
    )
    values = np.interp(np.log(f), log_f_band, per_band)
    return HvacSpectrumResult(
        frequencies=f, values=values, quantity="attenuation",
        label=f"End reflection ({termination}, D = {diameter * 1000:.0f} mm)",
    )


def elbow_insertion_loss(
    frequencies: ArrayLike,
    width: float,
    *,
    bend_type: str = "square",
    vanes: bool = False,
    lined: bool = False,
    speed_of_sound: float = _C_AIR,
) -> HvacSpectrumResult:
    """Duct bend/elbow insertion loss per bend (Bies Table 8.11, ASHRAE).

    Indexed by the frequency-to-width ratio ``W / lambda`` (``lambda = c / f``).
    Lined bends assume the lining extends at least three duct diameters up- and
    downstream. Round bends are treated as unlined with no vanes.

    :param frequencies: Frequencies ``f``, Hz (1-D array).
    :param width: Duct width ``W`` in the plane of the bend, m.
    :param bend_type: ``"square"`` or ``"round"``.
    :param vanes: Turning vanes fitted (square bends only).
    :param lined: Acoustically lined bend (square bends only).
    :param speed_of_sound: Speed of sound ``c``, m/s.
    :return: A :class:`HvacSpectrumResult` of the insertion loss, dB per bend.
    """
    f = _frequencies(frequencies)
    w = require_positive(width, "width")
    c = require_positive(speed_of_sound, "speed_of_sound")
    if bend_type == "round":
        if vanes or lined:
            raise ValueError("round bends take neither vanes nor lining.")
        key = "round"
    elif bend_type == "square":
        key = "square" + ("_vanes" if vanes else "") + ("_lined" if lined else "")
    else:
        raise ValueError("'bend_type' must be 'square' or 'round'.")
    col = _ELBOW_TABLE[key]
    wl = w * f / c
    # Table 8.11 bounds each row as "a <= W/lambda < b", so a ratio exactly on
    # a bound belongs to the row it opens: side="right" places W/lambda = 0.14
    # in the 0.14-0.28 row (1 dB), not in the row below it (0 dB).
    idx = np.searchsorted(_ELBOW_WL_UPPER, wl, side="right")
    idx = np.clip(idx, 0, col.size - 1)
    values = col[idx]
    return HvacSpectrumResult(
        frequencies=f, values=values, quantity="attenuation",
        label=f"Elbow ({key.replace('_', ', ')}, W = {w * 1000:.0f} mm)",
    )


def plenum_attenuation(
    exit_area: float,
    line_of_sight: float,
    wall_area: float,
    mean_absorption: ArrayLike,
    *,
    angle: float = 0.0,
) -> np.ndarray | float:
    """Plenum-chamber transmission loss by Wells' method (Bies Eq. (8.275)).

    ``TL = -10 log10[ S_out ( cos(theta) / (pi r^2) + (1 - alpha) / (S_w alpha) ) ]``,
    where the reverberant term uses the plenum room constant
    ``R = S_w alpha / (1 - alpha)`` (:func:`phonometry.room.room_constant`). The
    method holds above the inlet cut-on and when the plenum is large compared
    with the wavelength; it underpredicts the low-frequency loss by 5-10 dB.

    :param exit_area: Outlet-opening area ``S_out``, m2.
    :param line_of_sight: Straight-line inlet-to-outlet distance ``r``, m.
    :param wall_area: Total internal wall area ``S_w``, m2.
    :param mean_absorption: Mean Sabine wall absorption ``alpha`` in ``(0, 1)``
        (scalar or per-band).
    :param angle: Angle ``theta`` between the inlet axis and the line to the
        outlet, rad (default 0).
    :return: The transmission loss, dB (float for scalar absorption, else a
        per-band array).
    """
    s_out = require_positive(exit_area, "exit_area")
    r = require_positive(line_of_sight, "line_of_sight")
    s_w = require_positive(wall_area, "wall_area")
    alpha = np.asarray(mean_absorption, dtype=np.float64)
    if np.any(alpha <= 0.0) or np.any(alpha >= 1.0) or not np.all(np.isfinite(alpha)):
        raise ValueError("'mean_absorption' must lie strictly in (0, 1).")
    r_const = np.asarray(room_constant(s_w, alpha), dtype=np.float64)
    direct = np.cos(angle) / (np.pi * r**2)
    reverberant = 1.0 / r_const
    tl = -10.0 * np.log10(s_out * (direct + reverberant))
    return float(tl) if tl.ndim == 0 else tl


def flow_noise_straight_duct(
    frequencies: ArrayLike,
    flow_velocity: float,
    area: float,
) -> HvacSpectrumResult:
    """Flow-generated octave-band sound power of a straight duct (Bies Eq. (8.251)).

    ``L_WB = 7 + 50 log10(U) + 10 log10(S) - 2 - 26 log10(1.14 + 0.02 f / U)``
    in dB re 1e-12 W (VDI 2081-1), for airflow speed ``U`` in a duct of area
    ``S``.

    :param frequencies: Octave-band centre frequencies ``f``, Hz (1-D array).
    :param flow_velocity: Mean flow speed ``U``, m/s.
    :param area: Duct cross-sectional area ``S``, m2.
    :return: A :class:`HvacSpectrumResult` of the band sound power level,
        dB re 1e-12 W.
    """
    f = _frequencies(frequencies)
    u = require_positive(flow_velocity, "flow_velocity")
    s = require_positive(area, "area")
    lw = (
        7.0
        + 50.0 * np.log10(u)
        + 10.0 * np.log10(s)
        - 2.0
        - 26.0 * np.log10(1.14 + 0.02 * f / u)
    )
    return HvacSpectrumResult(
        frequencies=f, values=lw, quantity="sound_power_level",
        label=f"Straight-duct flow noise (U = {u:.1f} m/s)",
    )


def flow_noise_bend(
    frequencies: ArrayLike,
    flow_velocity: float,
    area: float,
    height: float,
    *,
    density: float = 1.206,
) -> HvacSpectrumResult:
    """Flow-generated octave-band sound power of a mitred bend (Bies Eqs. (8.252), (8.254)).

    ``L_WB = L_Ws - 10 log10(1 + 0.165 N_s^2) + 30 log10(U) - 103`` with the
    stream power level ``L_Ws = 30 log10(U) + 10 log10(S) + 10 log10(rho) + 117``
    (Bies Eq. (8.252)) and the Strouhal number ``N_s = f H / U`` (``H`` the duct
    height in the plane of the bend). The radiated sound power grows as the
    sixth power of the stream speed at low ``N_s`` (the inner-corner drag
    dipole) and the eighth power at high ``N_s`` (the outer-corner shear
    quadrupole); equivalently, the *efficiency* referenced to the stream power
    grows as ``U^3`` and ``U^5`` respectively.

    :param frequencies: Octave-band centre frequencies ``f``, Hz (1-D array).
    :param flow_velocity: Mean flow speed ``U``, m/s.
    :param area: Duct cross-sectional area ``S``, m2.
    :param height: Duct height ``H`` in the plane of the bend, m.
    :param density: Air density ``rho``, kg/m3.
    :return: A :class:`HvacSpectrumResult` of the band sound power level,
        dB re 1e-12 W.
    """
    f = _frequencies(frequencies)
    u = require_positive(flow_velocity, "flow_velocity")
    s = require_positive(area, "area")
    h = require_positive(height, "height")
    rho = require_positive(density, "density")
    lws = 30.0 * np.log10(u) + 10.0 * np.log10(s) + 10.0 * np.log10(rho) + 117.0
    ns = f * h / u
    lw = lws - 10.0 * np.log10(1.0 + 0.165 * ns**2) + 30.0 * np.log10(u) - 103.0
    return HvacSpectrumResult(
        frequencies=f, values=lw, quantity="sound_power_level",
        label=f"Mitred-bend flow noise (U = {u:.1f} m/s)",
    )
