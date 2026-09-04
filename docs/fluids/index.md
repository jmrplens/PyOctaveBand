← [Documentation index](../README.md)

# The medium

Every other area of this library measures something that happened *in* a
medium. This one is the medium. A density and a speed of sound stand behind
every sound power level, every absorption coefficient, every transmission
loss and every propagation calculation in the tree, and for most of the
library's life they arrived the same way: as a number somebody typed once.

That is what this area exists to stop. `phonometry.fluids` computes the state
of the fluid from the conditions that were actually measured, keeps the
conditions beside the result, and says which model produced it.

## Why it is not a domain

Nineteen of the twenty packages are domains of application: you go to
`building` because you are measuring a building, to `underwater` because you
are working in water. You never go to `fluids` because you are measuring a
fluid. You go there because whatever else you are measuring happens in one.

So it sits with `filters`, `signals` and `metrology` in the transverse
toolbox: any package may import it without an architecture edge, because a
medium is not a subject some domains have and others do not. It is the fourth
member of that set, and the first added since the library split the original
toolbox in three.

## Three kinds of number, deliberately kept apart

The reason a library ends up with a dozen different values for the density of
air is that three different things wear the same clothes.

**The physics of the fluid** is what lives here. It answers "what is this air,
at these conditions", and better physics is an improvement: when the model
gets more accurate, every caller who asked for air should get the more
accurate answer.

**A standard's own simplified formula** is not that. When ISO 10534-2 prints
$c_0 = 343{,}2\sqrt{T/293}$, that expression is part of the procedure, and a
measurement that claims to follow the standard has to use it. Those stay in
the module that implements their clause, with the citation beside them, and
they never move here.

**A constant frozen by a conformance row** is a third thing again. The
Johnson-Champoux-Allard model carries a Prandtl number of 0,71 as a published
constant of the model. The air at the reference state has 0,728. Substituting
the physical value into the model would not correct an error, it would change
the model, and it moves the impedance it computes by 1,5 parts in a thousand.
That constant stays frozen where it was published.

Keeping the three apart is what lets better physics reach a caller without a
single measurement silently ceasing to reproduce the standard it cites.

**No solids.** A `Fluid` carries no shear speed, and the elastic materials of
the wave solvers keep their own type with its own precondition. The two are
different quantities that happen to share the word "medium", and a solid's
properties are tabulated where a fluid's are computed.

**No fields, only states.** A `Fluid` is one fluid at one point. The
stratified profiles that ray tracers march through stay in the packages that
own their marchers, in the ocean and in the atmosphere, because a profile is
a description of a place rather than of a substance.

**No frequency dependence.** The speed of sound here is the zero-frequency
one. Molecular relaxation makes sound speed depend on frequency, and the
model that describes it lives with the atmospheric absorption that needs it,
in `phonometry.environment`.

## What is here

- [Humid air](humid-air.md): the CIPM-2007 formulation of
  IEC 61094-2:2009 Annex F, what it fixes and what it does not, how much each
  condition is worth, and why the library asks for the temperature but assumes
  the pressure out loud.
