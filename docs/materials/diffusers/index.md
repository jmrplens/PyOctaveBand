← [Documentation index](../../README.md)

# Diffusers and surfaces

Where the [Absorbers](../absorbers/index.md) subsection asks
how much energy a material removes from the field, this one asks what a
*surface* does with the sound it returns: how much it throws off the specular
direction, how evenly it spreads it, and, out on a pavement, how much it
absorbs where no laboratory can follow. Three guides walk that ground.

[Diffusers and Their Coefficients](diffusers.md) is the
measurement and design core: the random-incidence **scattering coefficient**
$s$ of ISO 17497-1, measured on a reverberation-room turntable, the
**diffusion coefficient** $d$ of ISO 17497-2, measured on a free-field
goniometer, the Schroeder quadratic-residue design rules with the Fraunhofer
far-field prediction that grades a well-depth sequence before it is built,
and the closing argument for why the two coefficients must never be swapped.

[Metadiffusers](metadiffusers.md) shrinks the Schroeder
diffuser by an order of magnitude: slits loaded by Helmholtz resonators slow
the sound until a 2 cm panel reproduces the reflection phases of wells 27 cm
deep, with critical coupling supplying the perfectly absorbing `0` state that
ternary sequences need. The published quadratic-residue design is evaluated
end to end, transfer-matrix chain to FDTD cross-check.

[In-situ Road-Surface Absorption](../surfaces/road-absorption.md) takes
the absorption question outdoors: the ISO 13472-1 subtraction technique
separates the incident and road-reflected components of an impulse with the
Adrienne window, and the ISO 13472-2 spot tube presses a portable
impedance tube onto the pavement for reflective surfaces, with the choice
between the two methods spelled out.

The neighbours are close: the diffuser panels are surface relatives of the
[metamaterial absorbers](../absorbers/metamaterial-absorbers.md) built
from the same slit and resonator cell, the scattering coefficient feeds the
room predictions of
[Room acoustics](../../buildings/rooms/index.md), and the road
methods serve the outdoor-noise interest of
[Environment and transport](../../environment/index.md).

## Pages in this section

- [Diffusers and Their Coefficients](diffusers.md): the
  ISO 17497-1 scattering and ISO 17497-2 diffusion coefficients, Schroeder
  design and the far-field prediction model.
- [Metadiffusers](metadiffusers.md): deep-subwavelength
  Schroeder diffusers from resonator-loaded slits, with slow sound and
  ternary sequences.

## See also

Pages elsewhere on the site that this section leans on:

- [In-situ Road-Surface Absorption](../surfaces/road-absorption.md):
  the ISO 13472-1 subtraction technique and the ISO 13472-2 spot method.
