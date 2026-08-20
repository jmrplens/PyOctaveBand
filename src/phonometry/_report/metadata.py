#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Test-report metadata for the ISO 717 accredited-laboratory fiche.

:class:`ReportMetadata` is the shared, frozen container of the descriptive
fields an accredited sound-insulation report carries around the ISO 717 rating:
the specimen, the client, the room and climatic conditions, the laboratory
identity and, optionally, the requirement the result is checked against. It is
passed to a rating result's ``report(..., metadata=...)`` method; every field is
optional and only the supplied ones are rendered, so the same object drives both
a full accredited fiche and a lightweight prediction fiche.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class ReportMetadata:
    """Descriptive metadata for the accredited ISO 717 report fiche.

    All fields are optional (default ``None``); the report renders only the
    fields that are supplied, so a partially populated instance is valid. The
    numeric fields are validated on construction by physical range: the
    dimension, mass, volume and pressure fields must be finite and strictly
    positive; the temperature and requirement fields need only be finite (0
    degrees Celsius or below is a valid test condition, and a programme-loudness
    target in LUFS is negative); and the relative-humidity fields must lie
    within 0..100 %. A violation raises :class:`ValueError`.

    :ivar specimen: Specimen description printed in the header (the tested
        element, e.g. ``"200 mm concrete wall"``).
    :ivar client: Client the test was carried out for.
    :ivar mounted_by: Who mounted the specimen in the test opening.
    :ivar manufacturer: Manufacturer of the tested element.
    :ivar area: Specimen area ``S``, in m^2 (the free test opening area).
    :ivar mass_per_area: Measured mass per unit area, in kg/m^2.
    :ivar source_volume: Source-room volume, in m^3.
    :ivar receiving_volume: Receiving-room volume, in m^3.
    :ivar room_volume: Volume of the single room under test, in m^3. Room
        acoustics (ISO 3382-1/-2) characterises one enclosure rather than a
        source/receiving pair, and ISO 3382-2:2008 Clause 9 requires the room
        volume to be reported; the room-acoustics fiche prints it in the
        header. Distinct from the ``source_volume``/``receiving_volume`` pair,
        which describe a sound-transmission measurement.
    :ivar source_positions: Number of source (loudspeaker/omnidirectional)
        positions used in the measurement, an integer (ISO 3382-1:2009 Table 1
        and ISO 3382-2:2008 Clause 8 require reporting the number of source
        positions). Printed by the room-acoustics fiche.
    :ivar receiver_positions: Number of microphone (receiver) positions used,
        an integer (ISO 3382-1:2009 Table 1 and ISO 3382-2:2008 Clause 8
        require reporting the number of microphone positions). Printed by the
        room-acoustics fiche.
    :ivar temperature: Air temperature during the test, in degrees Celsius (a
        single representative value; use the per-room fields below when the
        source and receiving rooms are reported separately).
    :ivar relative_humidity: Relative humidity during the test, in %.
    :ivar source_temperature: Source-room air temperature, in degrees Celsius.
    :ivar source_relative_humidity: Source-room relative humidity, in %.
    :ivar receiving_temperature: Receiving-room air temperature, in degrees
        Celsius.
    :ivar receiving_relative_humidity: Receiving-room relative humidity, in %.
    :ivar pressure: Ambient (static) air pressure during the test, in kPa.
    :ivar test_room: Test-room / facility identification.
    :ivar instrumentation: Identification and class of the instrumentation used
        (manufacturer, model, serial number), as free text. The occupational
        noise-exposure fiche prints it for ISO 9612:2009 Clause 15 c; when it
        is not supplied that fiche falls back to the result's own instrument
        class.
    :ivar calibration: Calibration traceability, as free text (calibrator,
        date and result of the most recent verification, the before/after
        field checks). Printed by the occupational noise-exposure fiche
        (ISO 9612:2009 Clause 15 c).
    :ivar tube_diameter: Impedance-tube inner diameter ``d`` (circular tube) or
        maximum lateral dimension (rectangular tube), in metres. Printed by the
        impedance-tube fiche (ISO 10534-2), where it fixes the upper plane-wave
        cut-on frequency.
    :ivar mic_spacing: Microphone spacing ``s`` between the two measurement
        positions of the impedance tube, in metres. Printed by the
        impedance-tube fiche (ISO 10534-2), where it bounds the working
        frequency range.
    :ivar tube_shape: Cross-section of the impedance tube: ``"circular"``,
        ``"rectangular"`` or ``"square"``. Printed by the impedance-tube fiche
        (ISO 10534-2) next to the tube diameter, which it qualifies (inner
        diameter for a circular tube, maximum lateral dimension otherwise).
    :ivar thickness: Specimen thickness under the applied static load, in
        metres. Printed by the dynamic-stiffness fiche (EN 29052-1 /
        ISO 9052-1), where EN 29052-1:1992 Clause 9 b) requires reporting the
        thickness of the resilient layer under load; it is shown in millimetres.
    :ivar mounting: Mounting condition of the specimen (e.g. the ISO 10140-1
        mounting code or a short description).
    :ivar measurement_standard: Measurement standard the spectrum was obtained
        under (e.g. ``"ISO 10140-2"`` or ``"ISO 16283-1"``); it forms the
        report's standard-basis line together with the ISO 717 rating part.
    :ivar test_date: Date of the test, as a free-form string.
    :ivar laboratory: Testing laboratory / institute name (footer).
    :ivar operator: Operator who carried out the test (footer signature line).
    :ivar report_id: Report / test number (footer).
    :ivar requirement: Target single-number value the verdict row compares the
        rating against, expressed in the rating's own unit (e.g. dB, a
        dimensionless absorption coefficient, sone, or a programme-loudness
        level in LUFS). It need only be finite (a loudness target in LUFS is
        negative), so its sign is not constrained. The pass direction is
        defined by each rating's ``report`` method: quantities where more is
        better (airborne insulation, absorption) pass at or above the
        requirement, and quantities where less is better (impact level,
        loudness, aircraft noise) pass at or below it; the programme-loudness
        fiche reads it as the target level and passes within a tolerance.
    :ivar required_class: Target performance-class index for a class-compliance
        verdict (the IEC 61260-1 filter fiche): ``0``, ``1`` or ``2``, where
        class 0 is the strictest. When supplied, the fiche's verdict passes if
        the achieved overall class is at least as strict as this class (a
        smaller or equal class index). ``None`` (the default) prints no verdict
        row.
    :ivar notes: Free-form remarks printed in the footer.
    :raises ValueError: If a supplied dimension/mass/volume/pressure is not
        finite and strictly positive, a temperature or requirement is not
        finite, a relative humidity is outside 0..100 %, a required class is
        not one of 0, 1, 2, a position count is not a finite, positive
        integer, or a tube shape is not one of ``"circular"``,
        ``"rectangular"``, ``"square"``.
    """

    specimen: str | None = None
    client: str | None = None
    mounted_by: str | None = None
    manufacturer: str | None = None
    area: float | None = None
    mass_per_area: float | None = None
    source_volume: float | None = None
    receiving_volume: float | None = None
    room_volume: float | None = None
    source_positions: int | None = None
    receiver_positions: int | None = None
    temperature: float | None = None
    relative_humidity: float | None = None
    source_temperature: float | None = None
    source_relative_humidity: float | None = None
    receiving_temperature: float | None = None
    receiving_relative_humidity: float | None = None
    pressure: float | None = None
    tube_diameter: float | None = None
    mic_spacing: float | None = None
    thickness: float | None = None
    test_room: str | None = None
    instrumentation: str | None = None
    calibration: str | None = None
    mounting: str | None = None
    measurement_standard: str | None = None
    test_date: str | None = None
    laboratory: str | None = None
    operator: str | None = None
    report_id: str | None = None
    requirement: float | None = None
    required_class: int | None = None
    notes: str | None = None
    # Appended after the original fields so the positional constructor order
    # of this public dataclass is preserved.
    tube_shape: str | None = None

    #: Numeric fields that must be finite and strictly positive.
    _POSITIVE_FIELDS = (
        "area",
        "mass_per_area",
        "source_volume",
        "receiving_volume",
        "room_volume",
        "pressure",
        "tube_diameter",
        "mic_spacing",
        "thickness",
    )
    #: Count fields that must be finite, positive integers (numbers of source
    #: and receiver positions in a room-acoustics measurement).
    _POSITIVE_INT_FIELDS = (
        "source_positions",
        "receiver_positions",
    )
    #: Fields that need only be finite, of any sign: the test temperatures (0 C
    #: or below is a valid condition, e.g. an unheated outdoor facade) and the
    #: requirement (a target such as a programme-loudness level in LUFS is
    #: legitimately negative, so only its finiteness is checked here; each
    #: rating's ``report`` gives it a physical meaning and pass direction).
    _FINITE_FIELDS = (
        "temperature",
        "source_temperature",
        "receiving_temperature",
        "requirement",
    )
    #: Relative-humidity fields: finite and within 0..100 %.
    _HUMIDITY_FIELDS = (
        "relative_humidity",
        "source_relative_humidity",
        "receiving_relative_humidity",
    )
    #: Upper bound of the relative-humidity percentage scale (saturation).
    _MAX_RELATIVE_HUMIDITY_PERCENT = 100.0

    def _require(
        self, name: str, ok: Callable[[float], bool], description: str
    ) -> None:
        """Raise ``ValueError`` unless a supplied numeric field satisfies ``ok``."""
        value = getattr(self, name)
        if value is None:
            return
        if not ok(float(value)):
            msg = (
                f"ReportMetadata.{name} must be {description} when given; "
                f"got {value!r}."
            )
            raise ValueError(msg)

    def __post_init__(self) -> None:
        """Validate the supplied numeric fields by physical range."""
        for name in self._POSITIVE_FIELDS:
            self._require(
                name,
                lambda x: math.isfinite(x) and x > 0.0,
                "a finite, positive number",
            )
        for name in self._FINITE_FIELDS:
            self._require(name, math.isfinite, "finite")
        for name in self._HUMIDITY_FIELDS:
            self._require(
                name,
                lambda x: (
                    math.isfinite(x) and 0.0 <= x <= self._MAX_RELATIVE_HUMIDITY_PERCENT
                ),
                "a relative humidity in 0..100 %",
            )
        for name in self._POSITIVE_INT_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            # A count is a whole number of positions: reject non-integers
            # (a bool is an int in Python, so it is excluded explicitly) and
            # anything not strictly positive.
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                msg = (
                    f"ReportMetadata.{name} must be a positive integer when "
                    f"given; got {value!r}."
                )
                raise ValueError(msg)
        if self.required_class is not None and self.required_class not in (0, 1, 2):
            msg = (
                "ReportMetadata.required_class must be 0, 1 or 2 when given; "
                f"got {self.required_class!r}."
            )
            raise ValueError(msg)
        if self.tube_shape is not None and self.tube_shape not in (
            "circular",
            "rectangular",
            "square",
        ):
            msg = (
                "ReportMetadata.tube_shape must be 'circular', 'rectangular' "
                f"or 'square' when given; got {self.tube_shape!r}."
            )
            raise ValueError(msg)

    def is_empty(self) -> bool:
        """Return ``True`` when no field is set (an all-``None`` instance)."""
        return all(getattr(self, f.name) is None for f in fields(self))
