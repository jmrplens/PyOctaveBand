← [Documentation index](../README.md)

# Humid air

A reciprocity calibration needs the air it is done in to five or six figures,
which is why the standard that defines that calibration also carries the most
carefully stated model of humid air in the acoustic literature. Annex F is
that model, and this page is what the library does with it.

## The shortest useful call

```python
from phonometry import fluids

f = fluids.air(
    temperature_c=23.0,
    static_pressure_pa=101325.0,
    relative_humidity_percent=50.0,
)
print(f.density)                    # 1.1860847889882964 kg/m3
print(f.speed_of_sound)             # 345.86651725321 m/s
print(f.characteristic_impedance)   # 410.2270151343906 Pa.s/m
```

Those are the conditions of the first row of Table F.1, and the numbers are
the ones the annex prints. Every one of the ten figures it tabulates comes
back inside the rounding of its last printed digit.

## The three conditions are not equally important

The library asks for the temperature and assumes the other two. That is not
laziness in one direction and rigour in the other; it follows from how much
each is worth. Sweeping each condition across its plausible range, with the
annex's own equations:

| Condition | Swept over | Moves the density by |
| :--- | :--- | ---: |
| Static pressure | 80 to 105 kPa | 25 % |
| Temperature | 15 to 27 °C | 4,5 % |
| Relative humidity | 0 to 100 % | 1,05 % |

The pressure dominates, and it is also the one people believe they already
know. A test site 1000 metres up sits at roughly 90 kPa. That is 11 % below
the standard atmosphere the library would otherwise assume, and it puts the
density about 13 % high, which is half a decibel on a sound power level with
nothing on the page to say so.

Humidity is the opposite case. It moves the density least, and almost nobody
knows theirs without measuring it.

So both are assumed and both are announced, once, in the same warning:

```python
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    fluids.air(temperature_c=20.0)

print(caught[0].category.__name__)              # FluidAssumptionWarning
print(str(caught[0].message))
# air() assumed 101325 Pa and 50 % relative humidity. Density is the sensitive
# one: over 80 kPa to 105 kPa it moves by a quarter, so a test site 1000 m up
# is about 11 % away from the assumed pressure, while the whole span of
# humidity is worth about 1 %. Pass the conditions to silence this.
```

One warning, not two, because a caller who supplied neither has one thing to
fix rather than two.

Supplying both is silent, which is the property somebody who measured their
air should get. Python shows the warning once per call site, so exploring at a
prompt is not drowned in it, and it derives from `PhonometryWarning`, so one
rule escalates it to an error for anyone who wants their runs to fail rather
than warn:

```python
from phonometry import PhonometryWarning

warnings.filterwarnings("error", category=PhonometryWarning)
```

The temperature has no default at all. It is the one condition a caller has
actually measured, and there is no value that would be defensible to invent
for it.

The carbon dioxide fraction defaults quietly, to 0,000 4. That is not a guess
about the caller's air: Clause F.2 names it as the value to use for laboratory
conditions in the absence of a measurement, so the library is quoting the
annex rather than filling a hole.

## What the model fixes, and what it will not invent

Table F.1 prints five quantities. Clause F.6 gives expressions for two more,
and three follow by identity from the ones above:

```python
f.density                    # printed by Table F.1
f.speed_of_sound             # printed, at zero frequency
f.heat_capacity_ratio        # printed
f.viscosity                  # printed
f.thermal_diffusivity        # printed
f.thermal_conductivity       # Clause F.6 expression
f.specific_heat_capacity     # Clause F.6 expression
f.characteristic_impedance   # rho c
f.prandtl_number             # eta / (rho alpha_t)
f.kinematic_viscosity        # eta / rho
```

Anything else raises rather than returning a number nobody printed:

```python
f.properties          # exactly what this model determined
f.model               # 'IEC 61094-2:2009 Annex F (CIPM-2007)'
f.composition         # the humidity and CO2 it was computed with
```

That looks like pedantry with one fluid in the library, and stops looking like
it with two: sea water has no ratio of specific heats to give, and a
`Fluid` that invented one would be handing a caller a number with the shape of
a measurement and none of the substance.

## The printed domain, and the difference between a fit and a fact

Annex F states, on its own page, where its equations were validated: 15 °C to
27 °C, 60 kPa to 110 kPa, 10 % to 90 % relative humidity. It states no range
for the carbon dioxide fraction.

Outside that box the library warns and still answers:

```python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    hot = fluids.air(temperature_c=60.0, static_pressure_pa=101325.0,
                     relative_humidity_percent=50.0)

print(caught[0].category.__name__)                    # FluidWarning
print("temperature 60 degC (stated 15 to 27)" in str(caught[0].message))  # True
print(round(hot.speed_of_sound, 2))                   # 371.52
```

The distinction matters and it is the rule the whole library follows. Air at
60 °C in a duct exists; a fit past its validated range is still arithmetic, and
refusing it would refuse a real measurement. What is refused is a state that
cannot be: a temperature at or below absolute zero, a pressure that is not
positive, a humidity outside nought to saturation.

## How the numbers are pinned

The oracle is Table F.1: five quantities at two condition sets, ten figures in
all. The tolerance is not chosen. A value printed to seven significant figures
is stated to within half of its last figure, so that is what reproducing it
means, and the test derives that bound from the printed string rather than
carrying a constant somebody picked:

<details>
<summary>The tolerance, derived from the print</summary>

```python
from decimal import Decimal

def half_of_the_last_printed_figure(printed: str) -> float:
    """Half a unit in the last place the annex printed."""
    exponent = int(Decimal(printed).as_tuple().exponent)
    return float(Decimal(5) * Decimal(10) ** (exponent - 1))

half_of_the_last_printed_figure("1.1860848")    # 5e-08
half_of_the_last_printed_figure("2.115317e-5")  # 5e-12
```

Nobody can loosen it by accident, and a future edition quoting more figures
tightens it by itself.

</details>

Three conformance rows record how much of that allowance each set uses: 55 %
for the first, 96 % for the second, and none at all for the identity that ties
the thermal conductivity and the specific heat capacity to the diffusivity
printed between them. The 96 % is the density of the second set, and it is the
annex's own rounding rather than any slack of the library's. Reporting it as a
percentage means that if a change ever pushes it out, it is visible coming.

## Where this air is, and is not, used

Passing a `Fluid` to a measurement function is the subject of the packages
that consume it, and it comes with a rule worth stating here: where a standard
*fixes* the air of its own procedure, using better air is a departure from the
standard, and the library says so rather than letting a measurement quietly
stop reproducing the document it cites.

That is why the simplified air formulas of ISO 10534-2, ASTM E2611 and
ISO 17497-1 stay in their own modules with their own clauses, and why the
Johnson-Champoux-Allard Prandtl number of 0,71 stays frozen at 0,71 even
though the air at that state has 0,728.
