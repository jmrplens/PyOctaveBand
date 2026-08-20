#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""End-to-end duct-borne noise calculation: fan to room, element by element.

The classic consulting workflow of HVAC acoustics is a bookkeeping exercise.
Start from the sound power the fan puts into the duct, walk down the path, and
at every element subtract what it attenuates and add back what its own airflow
regenerates. What arrives at the terminal device is converted into a sound
pressure level by the room effect, the supply and return paths are combined,
and the result is compared with the room criterion. If it fails, the sheet
itself says which element to change.

This module implements that cascade as a result object. It follows the
worked sheet of Long, *Architectural Acoustics* 2nd ed., Table 14.9 (a
5000 cfm forward-curved fan feeding a room through a supply and a return path,
checked against NC 30) and the same row structure as the industry procedure of
AHRI Standard 885, *Procedure for Estimating Occupied Space Sound Levels in the
Application of Air Terminals and Air Outlets*, Table 8: one row per physical
element, octave bands across the columns, regenerated noise as its own row
combined logarithmically rather than subtracted, and a cumulative received-level
row at the foot.

Three arithmetic conventions matter and are worth stating once:

* **Attenuations are positive.** Every element model in
  :mod:`phonometry.noise_control.hvac` returns a loss as a positive number of
  decibels, and the cascade subtracts it. Published worksheets print the same
  quantity as a negative level change; :meth:`DuctPathResult.table` follows the
  worksheet sign so the printed sheet reads like the reference.
* **Regenerated noise adds on a power basis.** The self-noise of an element is
  a sound power level in its own right and is combined with the level arriving
  at that point as :math:`10 \log_{10}(10^{L/10} + 10^{L_\mathrm{sn}/10})`.
* **There is a self-noise floor.** Long's worked sheet uses a 0 dB self-noise
  sound power level as the default and whenever a calculated level would be
  negative, which is why his received spectrum bottoms out near 0 dB rather
  than running to minus infinity. ``self_noise_floor`` reproduces that and can
  be switched off with ``None``.

The elements themselves come from :mod:`phonometry.noise_control.hvac` (fan
sound power, straight lined and unlined ducts, elbows, flexible duct, branch
splits, end reflection, silencer self-noise, plenums and the room effect) and
from manufacturer data, which is what a real calculation uses for silencers and
air terminal devices. :mod:`phonometry.noise_control.duct_modes` supplies the
cut-on frequency above which the plane-wave elements stop being valid, and the
cascade raises a
:class:`~phonometry.noise_control.duct_modes.PlaneWaveWarning` when a duct
section is declared and the analysis runs past it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ._criterion import (
    as_band_spectrum,
    curve_of,
    exceedance_of,
    meets_target_of,
    rating_of,
    validate_target,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray

    from .._report.metadata import ReportMetadata
    from ..room.noise_criteria import NCResult, RCResult

#: Default self-noise sound power level, dB re 1 pW, used where an element has
#: no regenerated-noise data and as the floor for a negative computed level
#: (Long, Table 14.9 commentary).
DEFAULT_SELF_NOISE_FLOOR = 0.0


def _combine(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Energy sum of two spectra, :math:`10 \log_{10}(10^{a/10} + 10^{b/10})`."""
    return 10.0 * np.log10(10.0 ** (a / 10.0) + 10.0 ** (b / 10.0))


@dataclass(frozen=True)
class DuctElement:
    """One element of a duct path: what it attenuates and what it regenerates.

    :ivar label: Human description of the element, e.g.
        ``"Silencer, 3 ft, standard pressure drop"``.
    :ivar attenuation: Octave-band attenuation as a **positive** loss, dB. A
        scalar is broadcast over the bands; an
        :class:`~phonometry.noise_control.hvac.HvacSpectrumResult` (or anything
        exposing ``values``) is accepted directly. ``None`` means no
        attenuation.
    :ivar self_noise: Octave-band regenerated sound power level of the element,
        dB re 1 pW, or ``None`` when the element regenerates nothing (the
        cascade then applies its self-noise floor).
    :ivar code: A short worksheet code for the printed sheet (AHRI 885 style,
        e.g. ``"D1"``, ``"S"``); defaults to the row number.
    """

    label: str
    attenuation: Any = None
    self_noise: Any = None
    code: str = ""


@dataclass(frozen=True)
class DuctPathStage:
    """The computed rows of one element of a duct path.

    The four spectra are exactly the four printed rows of a duct-borne
    calculation sheet: what the element takes out, what the level becomes,
    what the element puts back and where the path stands afterwards.

    :ivar label: Human description of the element.
    :ivar code: The worksheet code of the row.
    :ivar attenuation: Attenuation of the element as a positive loss, dB.
    :ivar attenuated: Level after subtracting the attenuation (the *Sum* row),
        dB.
    :ivar self_noise: Regenerated sound power level of the element, dB re 1 pW,
        after the self-noise floor has been applied.
    :ivar level: Level leaving the element, the energy sum of ``attenuated``
        and ``self_noise`` (the *Combined* row), dB.
    """

    label: str
    code: str
    attenuation: np.ndarray
    attenuated: np.ndarray
    self_noise: np.ndarray
    level: np.ndarray


@dataclass(frozen=True)
class DuctPathResult:
    """The end-to-end duct-borne noise calculation of one path (or of a sum).

    Built by :func:`duct_path` for a single fan-to-room path, and by
    :func:`combine_duct_paths` for the logarithmic sum of several such paths
    (whose ``stages`` are empty and whose ``contributions`` carry one entry per
    path).

    :ivar frequencies: Octave-band centre frequencies, Hz.
    :ivar source_level: Sound power level entering the path, dB re 1 pW.
    :ivar source_label: Description of the source, e.g. the fan.
    :ivar stages: One :class:`DuctPathStage` per element, in path order.
    :ivar room_effect: The room effect applied after the last element, as a
        positive attenuation in dB, or ``None`` when the path was not taken
        into a room (``received_level`` is then still a sound power level).
    :ivar received_level: The spectrum at the receiver: a sound pressure level
        in dB when ``room_effect`` was applied, otherwise the sound power level
        leaving the last element.
    :ivar criterion: ``"NC"`` or ``"RC"``, the room-criterion family used.
    :ivar target: The design criterion value (e.g. ``30`` for NC 30), or
        ``None`` when no target was declared.
    :ivar contributions: ``(label, received_level)`` per contributing path for
        a combination; empty for a single path.
    :ivar label: A short human label of the path.
    """

    frequencies: np.ndarray
    source_level: np.ndarray
    source_label: str
    stages: tuple[DuctPathStage, ...]
    room_effect: np.ndarray | None
    received_level: np.ndarray
    criterion: str
    target: float | None
    label: str
    contributions: tuple[tuple[str, np.ndarray], ...] = field(default_factory=tuple)

    # -- ratings ----------------------------------------------------------
    @property
    def rating(self) -> NCResult | RCResult:
        """The room-criterion rating of :attr:`received_level`.

        An :class:`~phonometry.room.noise_criteria.NCResult` when
        :attr:`criterion` is ``"NC"``, otherwise an
        :class:`~phonometry.room.noise_criteria.RCResult` (ANSI/ASA S12.2-2019).
        """
        return rating_of(self)

    @property
    def criterion_curve(self) -> np.ndarray | None:
        """The design criterion curve at :attr:`frequencies`, dB, or ``None``.

        The NC or RC curve of :attr:`target`, sampled at the analysis bands, so
        it can be compared band by band with :attr:`received_level`.
        """
        return curve_of(self)

    @property
    def exceedance(self) -> np.ndarray | None:
        """Band-by-band excess of the received level over the criterion, dB.

        Positive where the design criterion is exceeded. ``None`` when no
        target was declared.
        """
        return exceedance_of(self)

    @property
    def meets_target(self) -> bool | None:
        """``True`` when no band exceeds the design criterion curve.

        ``None`` when no target was declared. This is the band-by-band test a
        design sheet applies, not the NC *rating* of :attr:`rating`, which the
        standard derives by its own two-step procedure.
        """
        return meets_target_of(self)

    # -- tabular view -----------------------------------------------------
    def table(self) -> list[dict[str, Any]]:
        """The per-element sheet, one entry per printed row.

        The rows follow the layout of a published duct-borne calculation sheet
        (Long Table 14.9; AHRI 885 Table 8): the source, then for each element
        its attenuation, the *Sum* after subtracting it, its regenerated noise
        and the *Combined* level, then the room effect and the received
        spectrum. Each entry has ``code``, ``label``, ``kind`` (one of
        ``"source"``, ``"attenuation"``, ``"sum"``, ``"self_noise"``,
        ``"level"``, ``"contribution"``, ``"room_effect"``, ``"received"``,
        ``"criterion"``) and ``values`` (a per-band array). An attenuation row
        carries the worksheet sign, i.e. the **negative** of the positive loss
        the models return; a ``"contribution"`` row is the received spectrum of
        one path of a combination.

        :return: The list of row dictionaries, in printing order.
        """
        rows: list[dict[str, Any]] = [
            {
                "code": "S",
                "label": self.source_label,
                "kind": "source",
                "values": self.source_level,
            }
        ]
        for stage in self.stages:
            rows.append(
                {
                    "code": stage.code,
                    "label": stage.label,
                    "kind": "attenuation",
                    "values": -stage.attenuation,
                }
            )
            rows.append(
                {
                    "code": "",
                    "label": "Sum",
                    "kind": "sum",
                    "values": stage.attenuated,
                }
            )
            rows.append(
                {
                    "code": "",
                    "label": "Self-noise",
                    "kind": "self_noise",
                    "values": stage.self_noise,
                }
            )
            rows.append(
                {
                    "code": "",
                    "label": "Combined",
                    "kind": "level",
                    "values": stage.level,
                }
            )
        if self.contributions:
            for name, spectrum in self.contributions:
                rows.append(
                    {
                        "code": "",
                        "label": name,
                        "kind": "contribution",
                        "values": spectrum,
                    }
                )
        if self.room_effect is not None:
            rows.append(
                {
                    "code": "R",
                    "label": "Room effect",
                    "kind": "room_effect",
                    "values": -self.room_effect,
                }
            )
        rows.append(
            {
                "code": "Lp",
                "label": "Received level",
                "kind": "received",
                "values": self.received_level,
            }
        )
        curve = self.criterion_curve
        if curve is not None:
            rows.append(
                {
                    "code": self.criterion,
                    "label": f"{self.criterion} {self.target:g}",
                    "kind": "criterion",
                    "values": curve,
                }
            )
        return rows

    # -- presentation -----------------------------------------------------
    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the spectrum cascading down the path against the criterion curve.

        One line per element shows where the spectrum stands after that
        element, from the source at the top to the received level at the
        bottom, with the design criterion curve overlaid. Requires matplotlib
        (``pip install phonometry[plot]``).

        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the received-level ``Axes.plot``.
        :return: The axes.
        """
        from .._i18n import check_language
        from .._plot.noise_control import plot_duct_path

        check_language(language)
        return plot_duct_path(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render the duct-borne noise calculation sheet to a PDF at ``path``.

        Writes a one-page duct-path sheet in the layout of the published
        procedures (AHRI Standard 885 Table 8; Long Table 14.9): the
        method-basis line, an optional metadata header (client, system, room,
        instrumentation, climate, date), the element table with the octave
        bands across the columns and one row per element (attenuations printed
        with the worksheet's negative sign, regenerated noise shaded as its own
        row, the running level after each element and the room effect and
        received spectrum at the foot), the cascade chart against the criterion
        curve, the boxed room-criterion rating, the verdict against the design
        criterion, and a method-basis strip.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the header (``client``, ``specimen`` the air system, ``test_room``
            the served room, ``instrumentation``, ``temperature``,
            ``relative_humidity``, ``pressure``, ``test_date``) and the footer
            identity. The design criterion comes from the result's own
            ``target``; a ``requirement`` in the metadata overrides it.
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True`` the table adds the running level after
            every element; when ``False`` only the source, the element
            attenuations, the regenerated-noise rows and the received spectrum
            are printed, which keeps a long path on one page.
        :param language: Fiche language: ``"en"`` (default) or ``"es"``.
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"`` or
            ``language`` is unknown.
        :raises ImportError: If reportlab (or, for the figure, matplotlib) is
            not installed (``pip install phonometry[report]``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from .._report.duct_path import render_duct_path_report

        return render_duct_path_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def duct_path(
    frequencies: ArrayLike,
    source_level: ArrayLike,
    elements: list[DuctElement] | tuple[DuctElement, ...],
    *,
    room_effect: ArrayLike | None = None,
    source_label: str = "Fan",
    criterion: str = "NC",
    target: float | None = None,
    self_noise_floor: float | None = DEFAULT_SELF_NOISE_FLOOR,
    section: dict[str, float] | None = None,
    flow_velocity: float = 0.0,
    speed_of_sound: float = 343.0,
    label: str = "Duct path",
) -> DuctPathResult:
    """Cascade a duct path from the source sound power to the received level.

    Walks the elements in order, subtracting each attenuation and combining
    each regenerated sound power level on a power basis, then applies the room
    effect and rates the result against the design criterion. This is the
    calculation of Long Table 14.9 and of AHRI 885 Table 8.

    :param frequencies: Octave-band centre frequencies, Hz (1-D array). The
        published sheets use 63 Hz to 8 kHz.
    :param source_level: Sound power level entering the path, dB re 1 pW; e.g.
        from :func:`phonometry.noise_control.hvac.fan_sound_power`.
    :param elements: The path elements in order, as :class:`DuctElement`
        entries. Their ``attenuation`` and ``self_noise`` accept a scalar, a
        per-band array or an
        :class:`~phonometry.noise_control.hvac.HvacSpectrumResult`.
    :param room_effect: The room effect as a positive attenuation, dB (from
        :func:`phonometry.noise_control.hvac.room_effect`). ``None`` leaves the
        received spectrum as a sound power level at the terminal device.
    :param source_label: Description of the source for the printed sheet.
    :param criterion: Room-criterion family, ``"NC"`` (default) or ``"RC"``.
    :param target: The design criterion value (e.g. ``30``), or ``None``.
    :param self_noise_floor: Default regenerated sound power level, dB re 1 pW,
        applied where an element declares none and as a floor under every
        element's self-noise. Long's worked sheet uses 0 dB; pass ``None`` to
        disable the floor entirely.
    :param section: Optional duct cross section for the plane-wave validity
        check, as ``{"diameter": d}``, ``{"width": a, "height": b}`` or
        ``{"area": s}``, in metres. When given, a
        :class:`~phonometry.noise_control.duct_modes.PlaneWaveWarning` is
        raised for analysis frequencies above the first cut-on.
    :param flow_velocity: Mean axial flow speed in that section, m/s, used for
        the cut-on Mach correction.
    :param speed_of_sound: Speed of sound, m/s, for the same check.
    :param label: A short human label of the path.
    :return: A :class:`DuctPathResult`.
    :raises ValueError: If the spectra do not share one value per band or the
        criterion family is unknown.
    :raises TypeError: If an entry of ``elements`` is not a
        :class:`DuctElement`.
    """
    f = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if f.ndim != 1 or f.size == 0:
        raise ValueError("'frequencies' must be a non-empty 1-D array.")
    n = f.size
    level = as_band_spectrum(source_level, n, "source_level")
    family, goal = validate_target(criterion, target)

    if section is not None:
        from .duct_modes import plane_wave_limit, warn_above_plane_wave_limit

        limit = plane_wave_limit(
            flow_velocity=flow_velocity, speed_of_sound=speed_of_sound, **section
        )
        warn_above_plane_wave_limit(f, limit, label)

    stages: list[DuctPathStage] = []
    for position, element in enumerate(elements, start=1):
        if not isinstance(element, DuctElement):
            raise TypeError(
                "'elements' must hold DuctElement entries; got "
                f"{type(element).__name__} at position {position}."
            )
        attenuation = as_band_spectrum(
            element.attenuation, n, f"elements[{position - 1}].attenuation"
        )
        attenuated = level - attenuation
        self_noise = as_band_spectrum(
            element.self_noise,
            n,
            f"elements[{position - 1}].self_noise",
            default=-np.inf if self_noise_floor is None else self_noise_floor,
        )
        if self_noise_floor is not None:
            self_noise = np.maximum(self_noise, self_noise_floor)
        level = _combine(attenuated, self_noise)
        stages.append(
            DuctPathStage(
                label=element.label,
                code=element.code or str(position),
                attenuation=attenuation,
                attenuated=attenuated,
                self_noise=self_noise,
                level=level,
            )
        )

    room = (
        None if room_effect is None else as_band_spectrum(room_effect, n, "room_effect")
    )
    received = level if room is None else level - room
    return DuctPathResult(
        frequencies=f,
        source_level=as_band_spectrum(source_level, n, "source_level"),
        source_label=source_label,
        stages=tuple(stages),
        room_effect=room,
        received_level=received,
        criterion=family,
        target=goal,
        label=label,
    )


def combine_duct_paths(
    paths: list[DuctPathResult] | tuple[DuctPathResult, ...],
    *,
    criterion: str | None = None,
    target: float | None = None,
    label: str = "Combined paths",
) -> DuctPathResult:
    """Combine the received spectra of several duct paths into one result.

    A room is usually fed by more than one path (Long's worked sheet combines a
    supply and a return path before comparing with NC 30). The received levels
    add on a power basis, and the combination is rated against the same design
    criterion.

    :param paths: Two or more :class:`DuctPathResult` objects sharing the same
        analysis bands.
    :param criterion: Room-criterion family; ``None`` (default) takes it from
        the first path.
    :param target: Design criterion value; ``None`` (default) takes it from the
        first path that declares one.
    :param label: A short human label of the combination.
    :return: A :class:`DuctPathResult` whose ``contributions`` hold the
        per-path received spectra and whose ``stages`` are empty.
    :raises ValueError: If fewer than one path is given or the bands differ.
    """
    items = list(paths)
    if not items:
        raise ValueError("'paths' must hold at least one DuctPathResult.")
    f = np.asarray(items[0].frequencies, dtype=np.float64)
    for other in items[1:]:
        if not np.array_equal(np.asarray(other.frequencies, dtype=np.float64), f):
            raise ValueError("all paths must share the same analysis bands.")
    family, goal = validate_target(
        items[0].criterion if criterion is None else criterion,
        next((p.target for p in items if p.target is not None), None)
        if target is None
        else target,
    )
    total = 10.0 * np.log10(
        np.sum([10.0 ** (np.asarray(p.received_level) / 10.0) for p in items], axis=0)
    )
    return DuctPathResult(
        frequencies=f,
        source_level=total,
        source_label=label,
        stages=(),
        room_effect=None,
        received_level=total,
        criterion=family,
        target=goal,
        label=label,
        contributions=tuple(
            (p.label, np.asarray(p.received_level, dtype=np.float64)) for p in items
        ),
    )
