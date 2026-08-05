← [Documentation index](../../README.md)

# Structure-borne sources

A machine fixed to a building radiates sound twice: directly from its own
vibrating surface, and indirectly by injecting **structure-borne power** into
the structure, which carries it away and re-radiates it in distant rooms. The
six pages of this section cover both paths: one estimates the direct
radiation from the surface vibration itself, and the other five characterise
the second, sneakier structure-borne path end to end, from describing the
vibration and characterising the isolators to quantifying the power and
predicting the level a listener finally hears.

The language comes first.
[Mechanical mobility and the FRF family (ISO 7626-1)](mechanical-mobility.md)
defines the motion-per-force frequency-response functions (receptance,
mobility, accelerance and their reciprocals) that every later standard speaks,
with the closed-form SDOF resonator as the reference and the ISO 7626-2
measurement acceptance criteria. Source and receiver *mobilities* are what
decide how much power actually couples across an interface, which is why this
vocabulary matters.

Three pages then characterise the path elements.
[Bending-wave transmission at plate junctions (Cremer/Craik/Hopkins)](junction-transmission.md)
follows the power across the structure itself, with the wave-approach
transmission coefficients for rigid X, T, L and in-line junctions, their
diffuse-field angular average, and the coupling loss factor and vibration
reduction index Kij they yield.
[Transfer stiffness of resilient elements (ISO 10846)](transfer-stiffness.md)
measures the dynamic transfer stiffness of the isolators, mounts and hoses
inserted precisely to break the transmission path, by the direct and indirect
(transmissibility) methods.
[Sound power from surface vibration (ISO/TS 7849)](../../devices/emission/vibration-sound-power.md)
handles the direct radiation: the airborne power estimated from surface
velocity and a radiation factor, without any acoustic measurement.

The last two pages close the chain on the source and the receiver.
[Structure-borne sound power of equipment (EN 15657)](../../buildings/design/structure-borne-power.md)
measures what a machine injects, via the reception-plate method, and derives
the plate-independent source quantities (blocked force, characteristic power
level, free velocity).
[Installed structure-borne sound (EN 12354-5)](../../buildings/design/installed-structure-borne.md)
consumes exactly those quantities, couples them through source and receiver
mobilities, and predicts the sound pressure level in the receiving room,
which is where this section meets the
[sound insulation](../../buildings/insulation/index.md) models.

## Pages in this section

- [Mechanical mobility and the FRF family (ISO 7626-1)](mechanical-mobility.md):
  the FRF family, conversions and the SDOF reference resonator.
- [Bending-wave transmission at plate junctions (Cremer/Craik/Hopkins)](junction-transmission.md):
  the wave-approach transmission coefficients for rigid X, T, L and in-line
  junctions, their angular average and the derived coupling loss factor and Kij.
- [Transfer stiffness of resilient elements (ISO 10846)](transfer-stiffness.md):
  dynamic transfer stiffness of isolators by the direct and indirect methods.

## See also

Pages elsewhere on the site that this section leans on:

- [Sound power from surface vibration (ISO/TS 7849)](../../devices/emission/vibration-sound-power.md):
  radiated airborne power from surface velocity and a radiation factor.
- [Structure-borne sound power of equipment (EN 15657)](../../buildings/design/structure-borne-power.md):
  the reception-plate method and plate-independent source quantities.
- [Installed structure-borne sound (EN 12354-5)](../../buildings/design/installed-structure-borne.md):
  the predicted receiving-room level from installed equipment.
