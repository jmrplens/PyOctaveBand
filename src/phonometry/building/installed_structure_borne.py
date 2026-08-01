#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Installed structure-borne sound from service equipment (EN 12354-5:2009).

EN 12354-5 predicts the sound pressure level in a receiving room caused by
building service equipment that injects **structure-borne sound** into the
building. The chain closes the structural-vibroacoustics series:

1. The source strength is its *characteristic structure-borne sound power level*
   ``L_Ws,c``. It is **not** the raw reception-plate power of EN 15657
   Formula (14): that plate-injected level must first be converted to the
   plate-independent ``L_Ws,n`` (EN 15657 Formulae (15)/(17); see
   :mod:`phonometry.building.structure_borne_power`) and then referred to the
   actual receiver with the Annex I mobility correction
   (:func:`installed_power_from_reception_plate`),
   :math:`L_{Ws,inst,i} = L_{Ws,n} + 10 \log_{10}( Y_{\infty,i} / Y_{\infty,rec} )`
   with the reference plate mobility
   :math:`Y_{\infty,rec} = 5 \cdot 10^{-6}` m/(N.s), or equivalently to the
   characteristic level
   :math:`L_{Ws,c} = L_{Ws,n} + 10 \log_{10}( Y_s / Y_{\infty,rec} )` with the
   source mobility (Annex I.3, Table I.8), from which ``D_C`` is subtracted.
2. Only part of that power is actually injected into the supporting element; the
   loss is the **coupling term** ``D_C`` (clause 4.4.3), positive in the usual
   mobility-mismatched cases (see :func:`coupling_term` for the exception),
   set by the source mobility ``Y_s`` and the receiver mobility ``Y_i``
   (Formula 19b):
   :math:`D_{C,i} = 10 \log_{10}\left( |Y_s + Y_i|^2 / (|Y_s|
   \operatorname{Re}\{Y_i\}) \right)`, which reduces to
   :math:`10 \log_{10}( |Y_s| / \operatorname{Re}\{Y_i\} )` for a force source
   (high source mobility,
   Formula 19c) and to :math:`-10 \log_{10}( |Y_s| \operatorname{Re}\{Z_i\} )`
   for a velocity source (low
   source mobility, Formula 19d). An elastic support adds its transfer
   mobility ``Y_k`` inside the modulus (Formula 19e).
3. The **installed** power level is then
   :math:`L_{Ws,inst,i} = L_{Ws,c} - D_{C,i}`
   (Formula 18b).
4. The normalised sound pressure level in the receiving room for one path (i->j)
   follows from the installed power, the structure-to-airborne adjustment term
   ``D_sa`` (clause 4.4.4), the flanking sound reduction index ``R_ij,ref`` and
   the element area (Formula 18a):
   :math:`L_{n,s,ij} = L_{Ws,inst,i} - D_{sa,i} - R_{ij,ref}
   - 10 \log_{10}(S_i/S_0) - 10 \log_{10}(A_0/4)`
   with :math:`S_0 = A_0 = 10` m²; the paths combine energetically
   (Formula 17).

The source and receiver mobilities/impedances are those of
:mod:`phonometry.mechanical_mobility` and :mod:`phonometry.transfer_stiffness`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

from numpy.typing import ArrayLike

from .._internal.validation import require_positive

#: Reference area ``S0 = A0`` of EN 12354-5 (Formula 18a), m^2.
REFERENCE_AREA: float = 10.0

#: Characteristic mobility ``Y_inf,rec`` of the EN 15657 reference reception
#: plate (10 cm concrete; EN 12354-5 Annex I / EN 15657:2018 7.2.4), m/(N.s).
REFERENCE_PLATE_MOBILITY: float = 5.0e-6


def _positive_real_part(values: ArrayLike, name: str) -> np.ndarray:
    """Validate that the real part of a mobility/impedance is positive."""
    arr = np.asarray(values, dtype=np.complex128)
    re = np.real(arr)
    if (not np.all(np.isfinite(arr.real)) or not np.all(np.isfinite(arr.imag))
            or np.any(re <= 0.0)):
        raise ValueError(
            f"'{name}' must be finite with a positive real part (a passive "
            "receiver dissipates power)."
        )
    return arr


def _nonzero_magnitude(values: ArrayLike, name: str) -> np.ndarray:
    """Validate a finite, non-zero (complex) mobility magnitude."""
    arr = np.asarray(values, dtype=np.complex128)
    mag = np.abs(arr)
    if not np.all(np.isfinite(mag)) or not np.all(mag > 0.0):
        raise ValueError(f"'{name}' must be finite and non-zero.")
    return arr


def coupling_term(
    source_mobility: ArrayLike,
    receiver_mobility: ArrayLike,
    *,
    transfer_mobility: ArrayLike = 0.0,
) -> np.ndarray:
    r"""Coupling term ``D_C`` for a point excitation (EN 12354-5, Formula 19b/19e).

    :math:`D_C = 10 \log_{10}\left( |Y_s + Y_i + Y_k|^2 / (|Y_s|
    \operatorname{Re}\{Y_i\}) \right)` -- the loss between
    the characteristic and the injected structure-borne power. ``Y_k`` is the
    transfer mobility of an elastic support (Formula 19e; 0 for a rigid
    connection, Formula 19b).

    :param source_mobility: Source point mobility ``Y_s`` (complex, non-zero),
        in m/(N.s).
    :param receiver_mobility: Receiver point mobility ``Y_i`` (complex,
        positive real part).
    :param transfer_mobility: Elastic-support transfer mobility ``Y_k``
        (Default: 0.0).
    :return: The coupling term ``D_C``, in dB. Positive whenever the source
        and receiver mobilities are well mismatched (the usual installed
        case), but **not** guaranteed non-negative: near a mounting
        resonance where ``Y_s`` and ``Y_i`` are of comparable magnitude and
        opposite phase the numerator
        :math:`\lvert Y_s + Y_i \rvert^2` collapses and ``D_C``
        goes negative (the installed power then exceeds the characteristic
        level; e.g. :math:`Y_s = j \cdot 10^{-4}`,
        :math:`Y_i = 10^{-5} - j \cdot 10^{-4}` m/(N·s) gives
        :math:`D_C \approx -10` dB).
    :raises ValueError: if ``Y_s`` is zero/non-finite or ``Re{Y_i}`` is not
        positive and finite.
    """
    ys = _nonzero_magnitude(source_mobility, "source_mobility")
    yi = _positive_real_part(receiver_mobility, "receiver_mobility")
    yk = np.asarray(transfer_mobility, dtype=np.complex128)
    numerator = np.abs(ys + yi + yk) ** 2
    denominator = np.abs(ys) * np.real(yi)
    return np.asarray(10.0 * np.log10(numerator / denominator), dtype=np.float64)


def coupling_term_force_source(
    source_mobility: ArrayLike, receiver_mobility: ArrayLike
) -> np.ndarray:
    r"""Coupling term for a force source, high source mobility (Formula 19c).

    .. math::

       D_C = 10 \log_{10}\frac{|Y_s|}{\operatorname{Re}\{Y_i\}}

    :param source_mobility: Source point mobility ``Y_s`` (complex, non-zero).
    :param receiver_mobility: Receiver point mobility ``Y_i`` (complex,
        positive real part).
    :return: The coupling term ``D_C``, in dB.
    :raises ValueError: if ``Y_s`` is zero/non-finite or ``Re{Y_i}`` is not
        positive and finite.
    """
    ys = np.abs(_nonzero_magnitude(source_mobility, "source_mobility"))
    yi = np.real(_positive_real_part(receiver_mobility, "receiver_mobility"))
    return np.asarray(10.0 * np.log10(ys / yi), dtype=np.float64)


def coupling_term_velocity_source(
    source_mobility: ArrayLike, receiver_impedance: ArrayLike
) -> np.ndarray:
    r"""Coupling term for a velocity source, low source mobility (Formula 19d).

    .. math::

       D_C = -10 \log_{10}\left( |Y_s| \operatorname{Re}\{Z_i\} \right)

    :param source_mobility: Source point mobility ``Y_s`` (complex, non-zero).
    :param receiver_impedance: Receiver point impedance ``Z_i`` (complex,
        positive real part).
    :return: The coupling term ``D_C``, in dB.
    :raises ValueError: if ``Y_s`` is zero/non-finite or ``Re{Z_i}`` is not
        positive and finite.
    """
    ys = np.abs(_nonzero_magnitude(source_mobility, "source_mobility"))
    zi = np.real(_positive_real_part(receiver_impedance, "receiver_impedance"))
    return np.asarray(-10.0 * np.log10(ys * zi), dtype=np.float64)


def installed_power_from_reception_plate(
    reception_plate_level: ArrayLike,
    receiver_mobility: ArrayLike,
    *,
    plate_mobility: float = REFERENCE_PLATE_MOBILITY,
) -> np.ndarray:
    r"""Mobility correction of the reception-plate power (EN 12354-5, Annex I).

    :math:`L_{Ws,inst,i} = L_{Ws,n,i} + 10 \log_{10}( Y_{\infty,i} /
    Y_{\infty,rec} )`, which refers the
    characteristic reception-plate power level ``L_Ws,n`` (EN 15657
    Formula (17), re the 10 cm concrete plate
    :math:`Y_{\infty,rec} = 5 \cdot 10^{-6}` m/(N.s))
    to the characteristic mobility ``Y_inf,i`` of the actual receiving
    element (floor, wall), yielding the installed power of that element as in
    the Annex I.2 whirlpool example. The same correction with the *source*
    mobility instead of ``Y_inf,i`` yields the characteristic level
    ``L_Ws,c`` (Annex I.3, Table I.8), from which
    :func:`installed_structure_borne_power_level` subtracts ``D_C``.

    :param reception_plate_level: Power level to re-refer (per band), in dB re
        1 pW: either the characteristic level ``L_Ws,n`` (EN 15657 Formula 17,
        referred to the default 5e-6 m/(N.s) plate) or a raw Formula (14)
        plate power together with the mobility of the plate it was measured
        on, passed as ``plate_mobility``.
    :param receiver_mobility: Characteristic mobility ``Y_inf,i`` of the
        receiving element (per band; complex values use their magnitude), in
        m/(N.s).
    :param plate_mobility: Mobility the input level is referred to
        (Default: the EN 15657 reference plate,
        :math:`Y_{\infty,rec} = 5 \cdot 10^{-6}` m/(N.s);
        pass the measured plate mobility when the input is a raw Formula (14)
        level).
    :return: The mobility-corrected power level, in dB re 1 pW.
    :raises ValueError: for a non-positive receiver or plate mobility.
    """
    plate_mobility = require_positive(plate_mobility, "plate_mobility")
    lw = np.asarray(reception_plate_level, dtype=np.float64)
    y_i = np.abs(np.asarray(receiver_mobility, dtype=np.complex128))
    if not np.all(np.isfinite(y_i)) or np.any(y_i <= 0.0):
        raise ValueError("'receiver_mobility' must be finite and non-zero.")
    return np.asarray(
        lw + 10.0 * np.log10(y_i / plate_mobility), dtype=np.float64
    )


def installed_structure_borne_power_level(
    characteristic_power_level: ArrayLike, coupling_term: ArrayLike
) -> np.ndarray:
    r"""Installed structure-borne power level (EN 12354-5, Formula 18b).

    .. math::

       L_{Ws,inst,i} = L_{Ws,c} - D_{C,i}

    :param characteristic_power_level: Characteristic level ``L_Ws,c`` (per
        band), in dB: the EN 15657 reception-plate level converted with
        Formulae (15)/(17) and the source-mobility correction (see the module
        docstring), **not** the raw plate-injected Formula (14) level.
    :param coupling_term: Coupling term ``D_C,i`` (per band), in dB.
    :return: The installed structure-borne power level ``L_Ws,inst``, in dB.
    """
    lw = np.asarray(characteristic_power_level, dtype=np.float64)
    dc = np.asarray(coupling_term, dtype=np.float64)
    return np.asarray(lw - dc, dtype=np.float64)


def structure_borne_pressure_level_path(
    installed_power_level: ArrayLike,
    adjustment_term: ArrayLike,
    flanking_reduction_index: ArrayLike,
    element_area: float,
    *,
    reference_area: float = REFERENCE_AREA,
) -> np.ndarray:
    r"""Normalised structure-borne SPL for one path i->j (Formula 18a).

    .. math::

       L_{n,s,ij} = L_{Ws,inst,i} - D_{sa,i} - R_{ij,ref}
       - 10 \log_{10}\frac{S_i}{S_0} - 10 \log_{10}\frac{A_0}{4}

    :param installed_power_level: Installed power level ``L_Ws,inst,i``, in dB.
    :param adjustment_term: Structure-to-airborne adjustment ``D_sa,i`` (clause
        4.4.4 / Annex F), in dB.
    :param flanking_reduction_index: Flanking sound reduction index
        ``R_ij,ref`` re ``S0`` (EN 12354-1), in dB.
    :param element_area: Supporting-element area ``S_i``, in m^2 (> 0).
    :param reference_area: Reference area :math:`S_0 = A_0` (Default: 10 m^2).
    :return: The normalised path sound pressure level ``L_n,s,ij``, in dB.
    :raises ValueError: for a non-positive area.
    """
    element_area = require_positive(element_area, "element_area")
    reference_area = require_positive(reference_area, "reference_area")
    lw = np.asarray(installed_power_level, dtype=np.float64)
    dsa = np.asarray(adjustment_term, dtype=np.float64)
    rij = np.asarray(flanking_reduction_index, dtype=np.float64)
    lp = (
        lw - dsa - rij
        - 10.0 * np.log10(element_area / reference_area)
        - 10.0 * np.log10(reference_area / 4.0)
    )
    return np.asarray(lp, dtype=np.float64)


def total_structure_borne_pressure_level(path_levels: ArrayLike) -> np.ndarray:
    r"""Combine path sound pressure levels energetically (Formula 17).

    .. math::

       L_{n,s} = 10 \log_{10}\!\left( \sum_j 10^{L_{n,s,ij}/10} \right)

    :param path_levels: Path levels ``L_n,s,ij``; sum is over the first axis
        (paths), broadcasting any trailing band axis.
    :return: The total normalised sound pressure level ``L_n,s``, in dB.
    """
    lp = np.asarray(path_levels, dtype=np.float64)
    return np.asarray(
        10.0 * np.log10(np.sum(10.0 ** (0.1 * lp), axis=0)), dtype=np.float64
    )


@dataclass(frozen=True)
class InstalledSourceResult:
    """Installed structure-borne sound prediction (EN 12354-5).

    :ivar frequencies: Band centre frequencies, in hertz, or ``None``.
    :ivar path_levels: Per-path normalised SPL ``L_n,s,ij`` (paths x bands), dB.
    :ivar total_level: Combined normalised SPL ``L_n,s`` per band, in dB.
    :ivar installed_power_level: Installed power level ``L_Ws,inst`` per band, dB.
    """

    path_levels: np.ndarray
    total_level: np.ndarray
    installed_power_level: np.ndarray
    frequencies: np.ndarray | None = None

    @property
    def overall_level(self) -> float:
        r"""Band-summed total level :math:`10 \log_{10}(\sum 10^{0.1 L_{n,s}})`,
        in dB."""
        lt = np.atleast_1d(np.asarray(self.total_level, dtype=np.float64))
        return float(10.0 * np.log10(np.sum(10.0 ** (0.1 * lt))))

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the per-path and total normalised sound pressure levels.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.building import plot_installed_structure_borne

        check_language(language)
        return plot_installed_structure_borne(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an EN 12354-5 installed structure-borne prediction fiche.

        Writes a one-page **prediction** sheet (an estimate, not a
        measurement): a prediction-basis line naming EN 12354-5:2009, an
        optional metadata header (client, source equipment, receiving room,
        instrumentation, climate, date), a per-band table (nominal
        octave/one-third-octave frequency, the installed structure-borne power
        level ``L_Ws,inst``, each transmission path's normalised SPL
        ``L_n,s,ij`` and the combined total ``L_n,s``), the per-path and total
        ``L_n,s(f)`` spectra, the boxed band-summed total ``L_n,s`` (dB) with
        the installed power total and the path count, an optional verdict row
        against a declared limit, and a basis strip stating Formulae 18a/17 and
        the prediction disclaimer.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the header (``client``, ``specimen`` the source equipment,
            ``test_room`` the receiving room, ``instrumentation``,
            ``temperature``, ``relative_humidity``, ``pressure``,
            ``test_date``), the footer identity (``laboratory``, ``operator``,
            ``report_id``, ``notes``) and, via ``requirement``, a declared
            upper limit on the overall ``L_n,s`` (lower is better).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True`` the per-band table adds one column per
            transmission path (up to five); otherwise only the installed power
            and the combined total are shown.
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
        from .._report.en12354_5 import render_installed_structure_borne_report

        return render_installed_structure_borne_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def installed_source_prediction(
    characteristic_power_level: ArrayLike,
    coupling_term: ArrayLike,
    paths: list[dict[str, Any]],
    *,
    frequencies: ArrayLike | None = None,
) -> InstalledSourceResult:
    """Predict the installed structure-borne SPL over several paths (EN 12354-5).

    The band count is set by the widest per-band input (the characteristic
    power level, the ``coupling_term`` or any path's ``adjustment_term`` /
    ``flanking_reduction_index``); every
    per-band input must carry one value or that count, and single values
    broadcast across the bands (a single-number source level with per-band
    path data is valid, and the result's ``installed_power_level`` is
    broadcast to the band count).

    :param characteristic_power_level: Characteristic level ``L_Ws,c`` (per
        band or a single value), in dB.
    :param coupling_term: Coupling term ``D_C`` (per band or a single
        value), in dB.
    :param paths: One dict per transmission path with keys ``adjustment_term``
        (``D_sa``), ``flanking_reduction_index`` (``R_ij,ref``) and
        ``element_area`` (``S_i``), each per band where applicable.
    :param frequencies: Band centre frequencies, in hertz, or ``None``.
    :return: The :class:`InstalledSourceResult`.
    :raises ValueError: if ``paths`` is empty, a path is missing a required
        key, or a per-band input matches neither one value nor the band
        count.
    """
    if not paths:
        raise ValueError("'paths' must contain at least one transmission path.")
    lw_inst = np.atleast_1d(
        np.asarray(
            installed_structure_borne_power_level(
                characteristic_power_level, coupling_term
            ),
            dtype=np.float64,
        )
    )
    required = ("adjustment_term", "flanking_reduction_index", "element_area")
    # The band count is set by the widest per-band input (the characteristic
    # level, the coupling term or any path's per-band terms); every other
    # per-band input must carry one value or that count, and single values
    # broadcast across the bands.
    per_band_keys = ("adjustment_term", "flanking_reduction_index")
    n_bands = lw_inst.size
    for k, p in enumerate(paths):
        missing = [key for key in required if key not in p]
        if missing:
            raise ValueError(
                f"path {k} is missing required key(s): {', '.join(missing)}."
            )
        for key in per_band_keys:
            n_bands = max(
                n_bands, np.atleast_1d(np.asarray(p[key], dtype=np.float64)).size
            )
    if lw_inst.size not in (1, n_bands):
        raise ValueError(
            f"the source levels carry {lw_inst.size} bands, expected 1 or "
            f"{n_bands} to match the transmission paths."
        )
    rows = []
    for k, p in enumerate(paths):
        for key in per_band_keys:
            values = np.atleast_1d(np.asarray(p[key], dtype=np.float64))
            if values.size not in (1, n_bands):
                raise ValueError(
                    f"path {k}: {key!r} has {values.size} bands, expected 1 "
                    f"or {n_bands} to match the other per-band inputs."
                )
        rows.append(
            np.broadcast_to(
                structure_borne_pressure_level_path(
                    lw_inst, p["adjustment_term"],
                    p["flanking_reduction_index"], p["element_area"],
                ),
                (n_bands,),
            )
        )
    path_levels = np.asarray(rows, dtype=np.float64)
    total = total_structure_borne_pressure_level(path_levels)
    freq = (
        None
        if frequencies is None
        else np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    )
    if freq is not None and freq.size != n_bands:
        raise ValueError(
            f"frequencies carries {freq.size} values, expected {n_bands} to "
            "match the per-band inputs."
        )
    return InstalledSourceResult(
        path_levels=path_levels,
        total_level=np.asarray(total, dtype=np.float64),
        installed_power_level=np.broadcast_to(lw_inst, (n_bands,)).copy(),
        frequencies=freq,
    )
