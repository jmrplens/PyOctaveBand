#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The state of a propagating fluid, and what a model does and does not fix.

A :class:`Fluid` is what a model returned for one set of conditions. It carries
the state it was asked for, the quantities the model determined, and the name of
the model, so a result can always say where its numbers came from.

Not every model determines every quantity. Reading one it did not raises
:class:`FluidPropertyUnavailable` naming the model and the quantity, rather than
returning a plausible number nobody printed.
"""

from __future__ import annotations

import dataclasses
import types
from typing import TYPE_CHECKING

from .._internal.validation import require_positive
from .._internal.warnings import PhonometryWarning

if TYPE_CHECKING:
    from collections.abc import Mapping


class FluidWarning(PhonometryWarning):
    """A fluid state outside the domain its model states for itself."""


class FluidAssumptionWarning(PhonometryWarning):
    """A fluid was built from a default the caller did not supply.

    The default is a documented value, not a measurement of the caller's air or
    water, so it is announced. Passing every argument silences it. Python shows
    a warning once per call site by default, and
    ``warnings.filterwarnings("error", category=FluidAssumptionWarning)`` turns
    it into a hard failure for anyone who wants one.
    """


class FluidPropertyUnavailable(AttributeError):
    """A quantity the model that built this fluid does not determine."""


def characteristic_impedance(density: float, speed_of_sound: float) -> float:
    """Characteristic impedance ``rho c``, in pascal seconds per metre.

    The product of a medium's density and its speed of sound. It belongs to the
    medium rather than to any procedure, which is why it lives here and not with
    the impedance tube that used to publish it; ISO 10534-2 Clause 7.2 and
    ASTM E2611-19 Clauses 8.2/8.3 both reach for the same product, and so does
    every reflection coefficient in the library.

    :class:`Fluid` exposes the same quantity as a property, closed from the two
    it was built with. This function is for a caller who has a density and a
    speed of sound and no fluid to go with them.

    The arguments carry no unit in their names, unlike the temperatures and
    pressures elsewhere in the library: those name a unit because two are in
    play and a caller can supply the wrong one. A density in this tree is
    kilograms per cubic metre everywhere, and a speed of sound is metres per
    second everywhere, so there is no second unit to be confused with.

    :param density: Density ``rho``, in kg/m3.
    :param speed_of_sound: Speed of sound ``c``, in m/s.
    :return: Characteristic impedance ``rho c``, in Pa*s/m (rayl).
    :raises ValueError: if either argument is not a positive finite number.
    """
    # A bare `<= 0.0` lets NaN through, since every comparison with NaN is
    # false, and the product then comes back as NaN rather than raising. The
    # shared guard rejects non-finite values as well, and names the argument
    # that was wrong instead of both.
    return float(
        require_positive(density, "density")
        * require_positive(speed_of_sound, "speed_of_sound")
    )


@dataclasses.dataclass(frozen=True)
class Fluid:
    """One fluid at one state, and the properties its model fixed there.

    :param temperature_c: Temperature, in degrees Celsius.
    :param static_pressure_pa: Absolute static pressure, in pascals.
    :param composition: What distinguishes this fluid from another at the same
        temperature and pressure: the relative humidity of air, the salinity and
        depth of sea water. Read it to know what was assumed.
    :param model: The model that produced the properties, named so a result can
        cite it.
    :param validity: The domain the model states for itself, in words, or the
        empty string where it states none. Prose, because sources state these in
        prose and reducing them to a box loses the conditions attached to them.
    :param properties: The quantities the model determined, in SI. Reached
        through the named accessors, which raise
        :class:`FluidPropertyUnavailable` for a quantity that is absent.
    """

    temperature_c: float
    static_pressure_pa: float
    composition: Mapping[str, float]
    model: str
    validity: str
    properties: Mapping[str, float]

    def __post_init__(self) -> None:
        """Freeze the two mappings, which ``frozen=True`` does not reach.

        A frozen dataclass stops the attribute from being rebound and says
        nothing about what it points at, so a plain ``dict`` here stays
        writable. That matters most for the shared states: ``PUBLISHED_AIR`` is
        a module-level constant every visco-thermal model defaults to, and one
        ``air.properties["density"] = 999`` anywhere in a process would have
        moved every one of those defaults, silently and for good.

        Copied before wrapping, so a caller who keeps a reference to the dict
        it passed in cannot reach back through it either.
        """
        object.__setattr__(
            self, "composition", types.MappingProxyType(dict(self.composition))
        )
        object.__setattr__(
            self, "properties", types.MappingProxyType(dict(self.properties))
        )

    def _fixed(self, quantity: str) -> float:
        """Return ``quantity``, or say which model failed to determine it."""
        try:
            return self.properties[quantity]
        except KeyError:
            known = ", ".join(sorted(self.properties)) or "nothing"
            msg = (
                f"{self.model!r} does not determine {quantity!r}; it determines "
                f"{known}. Supply the quantity yourself, or use a model that "
                f"prints it."
            )
            raise FluidPropertyUnavailable(msg) from None

    @property
    def density(self) -> float:
        """Density ``rho``, in kilograms per cubic metre."""
        return self._fixed("density")

    @property
    def speed_of_sound(self) -> float:
        """Speed of sound ``c``, in metres per second."""
        return self._fixed("speed_of_sound")

    @property
    def heat_capacity_ratio(self) -> float:
        """Ratio of specific heats ``kappa`` (``gamma``), dimensionless."""
        return self._fixed("heat_capacity_ratio")

    @property
    def viscosity(self) -> float:
        """Dynamic viscosity ``eta``, in pascal seconds."""
        return self._fixed("viscosity")

    @property
    def thermal_diffusivity(self) -> float:
        """Thermal diffusivity ``alpha_t``, in square metres per second."""
        return self._fixed("thermal_diffusivity")

    @property
    def thermal_conductivity(self) -> float:
        """Thermal conductivity ``k_a``, in watts per metre kelvin."""
        return self._fixed("thermal_conductivity")

    @property
    def specific_heat_capacity(self) -> float:
        """Specific heat capacity at constant pressure ``C_P``, in J/(kg K)."""
        return self._fixed("specific_heat_capacity")

    @property
    def characteristic_impedance(self) -> float:
        """``rho c``, in pascal seconds per metre.

        Closed by identity from two quantities the model fixed, so it is
        available whenever both of those are.
        """
        return self.density * self.speed_of_sound

    @property
    def prandtl_number(self) -> float:
        """``Pr = eta / (rho alpha_t)``, dimensionless.

        A model that prints its own Prandtl number keeps it: a published fit
        carries the value it was fitted with, and closing the identity from a
        better air would silently change the model rather than correct it. A
        model that does not print one has it closed from the three that it did.
        """
        carried = self.properties.get("prandtl_number")
        if carried is not None:
            return float(carried)
        return self.viscosity / (self.density * self.thermal_diffusivity)

    @property
    def kinematic_viscosity(self) -> float:
        """``nu = eta / rho``, in square metres per second."""
        return self.viscosity / self.density
