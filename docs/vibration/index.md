← [Documentation index](../README.md)

# Vibration and structure-borne sound

Vibration matters to acoustics twice. First as a **source of sound**: a pump
or fan bolted to a building injects structure-borne power that travels through
walls and floors and re-radiates as audible noise rooms away. Second as a
**human exposure** in its own right: vibration transmitted to a standing,
seated or hand-gripping person is measured, weighted and limited much like
noise, with its own metrics and legal action values.

The **structure-borne sources** pages follow the source chain in order. The
frequency-response-function family of ISO 7626 (receptance, mobility,
accelerance) is the vocabulary; the transfer stiffness of ISO 10846
characterises the resilient elements that interrupt the path; ISO/TS 7849
estimates the airborne power a vibrating surface radiates directly; EN 15657
measures the structure-borne power a machine injects into a reception plate;
and EN 12354-5 assembles all of it into the sound pressure level predicted in
a receiving room. That final prediction is also where this section hands over
to the [sound insulation](../buildings/insulation/index.md) models
of the buildings section.

The **human vibration** pages share the measurement philosophy of a sound
level meter, applied to acceleration: frequency weightings that reflect body
response, running and integrated averages, and dose quantities compared
against the action and limit values of Directive 2002/44/EC, plus the
dedicated spinal-response model for vibration containing repeated shocks.

Start with
[Mechanical mobility and the FRF family](structural/mechanical-mobility.md)
if you care about machines and buildings, or with
[Human Vibration](human/human-vibration.md) if you care about
people.

## [Structure-borne sources](structural/index.md)

From FRF vocabulary to the predicted level in a receiving room.

- [Mechanical mobility and the FRF family (ISO 7626-1)](structural/mechanical-mobility.md):
  receptance, mobility and accelerance with their reciprocals, and the SDOF
  reference resonator.
- [Bending-wave transmission at plate junctions (Cremer/Craik/Hopkins)](structural/junction-transmission.md):
  the frequency-independent wave-approach coefficients for rigid X, T, L and
  in-line junctions, their angular average and the derived coupling loss factor
  and Kij.
- [Transfer stiffness of resilient elements (ISO 10846)](structural/transfer-stiffness.md):
  the dynamic transfer stiffness of vibration isolators by the direct and
  indirect methods.
- [Sound power from surface vibration (ISO/TS 7849)](../devices/emission/vibration-sound-power.md):
  radiated airborne power from surface velocity and a radiation factor.
- [Structure-borne sound power of equipment (EN 15657)](../buildings/design/structure-borne-power.md):
  the reception-plate method and the plate-independent source quantities.
- [Installed structure-borne sound (EN 12354-5)](../buildings/design/installed-structure-borne.md):
  the receiving-room sound pressure level predicted from source and receiver
  mobilities.

## [Human vibration](human/index.md)

Vibration transmitted to the human body, from daily exposure to spinal injury
risk.

- [Human Vibration](human/human-vibration.md): whole-body and
  hand-arm weightings (ISO 8041-1), r.m.s. and dose measures (ISO 2631-1),
  daily exposure A(8) (ISO 5349) and the Directive 2002/44/EC values.
- [Multiple-shock whole-body vibration (ISO 2631-5)](human/multiple-shock-vibration.md):
  the spinal-response model and the probability of lumbar injury for vibration
  containing multiple shocks.

## Machinery

Turning a vibration spectrum into a diagnosis of the machine that made it.

- [Machine fault frequencies](machinery/machine-diagnostics.md):
  the characteristic bearing, gear and shaft frequencies, and the envelope
  analysis that finds them under the broadband noise of a running machine.
