← [Documentation index](../../README.md)

# Noise control

A noise-control problem is a **budget**, not a component choice. Between a
machine and the person who hears it there is a path, and each element on that
path removes a known number of decibels per band; the design question is which
combination closes the gap between what the source emits and what the criterion
allows. Machinery noise is attacked at the source, the path and the receiver in
turn, and this section holds the path — both halves of it: the element models,
and the two composed chains that spend their output against a criterion.

Everything here is a **prediction** from declared geometry and declared
material data. That matters when a catalogue is open beside the screen: the
figure a supplier publishes for the same device is a *measured* insertion loss,
obtained under the conditions of a measurement standard — ISO 7235 for a ducted
silencer on a laboratory rig with and without airflow, which also gives the
regenerated flow noise and the pressure loss, ISO 11691 for the survey method
without flow, ISO 11820 for a silencer in situ, and ISO 11546-1 and -2 for an
enclosure in the laboratory and in situ. A computed transmission loss and a
catalogue insertion loss are not the same quantity. Neither is wrong; they
answer different questions, and a design that mixes them without saying so is
not defensible.

[Duct-Borne Noise: Fan to Room](duct-path.md)
follows an airborne path from the fan through the duct run into the room:
attenuation in straight duct, at bends and takeoffs, end reflection at the
terminal, regenerated flow noise added back, the room effect at the receiver,
and the result laid against the room criterion. It also states the limit every
element model in this section shares — the frequency above which higher-order
modes cut on and the plane-wave assumption stops holding.
[Room to Room: Partition, Receiving Room, Criterion](room-to-room.md)
follows the airborne room-to-room path instead: a source-room level built from a
sound power and the room constant, a partition with its transmission loss, a
receiving room with its absorption, the received spectrum and its verdict — and
the inverse problem, the transmission loss a partition or a lined enclosure must
have for the receiving room to meet its criterion, solved backwards.

The two element pages supply what those chains call.
[Silencers](silencers.md) covers the reactive
four-pole elements (expansion chambers, Helmholtz, quarter-wave and
extended-tube resonators) with their transmission and insertion loss, and the
choice between reflection and dissipation, while
[Industrial Noise Control](noise-control.md)
keeps the HVAC duct attenuation and flow noise of an installation and the
insertion loss of a machine enclosure.

If the noise travels in a duct, start at
[Duct-Borne Noise](duct-path.md); if it travels
through a wall, start at
[Room to Room](room-to-room.md); open the
element pages when a chain asks for a number you do not have.

Both ends of the problem are settled outside this section, and a path
calculation with either end missing has no verdict. At the **source** end, what
a quieting measure is judged against is the emission of the machine itself,
determined by the [Sound power and intensity](../emission/index.md)
pages — and reducing it there is almost always cheaper than treating a path. At
the **receiver** end sit the criteria: the NC and RC Mark II families of
[Room noise criteria](../../buildings/rooms/room-noise.md), plus whatever
occupational limit applies, in [Occupational exposure
(ISO 9612)](../../perception/hearing/occupational-exposure.md).

## Pages in this section

- [Silencers](silencers.md): reactive silencers by the
  four-pole method and the reactive-versus-dissipative choice.
- [Duct-Borne Noise: Fan to Room](duct-path.md): the
  end-to-end fan-to-room calculation against a room criterion, and the
  higher-order-mode cut-on that limits every plane-wave method.
- [HVAC Noise the German Way (VDI 2081)](vdi2081-air-systems.md):
  the same chain by the German guideline, against the worked sheet of its own
  Part 2.
- [Control Valve Noise (IEC 60534-8-3)](control-valve-noise.md):
  the five flow regimes of a throttling valve and the pipe wall that radiates
  what they make.
- [Room to Room: Partition, Receiving Room, Criterion](room-to-room.md):
  the composed source-room to receiving-room chain and the transmission loss a
  partition or an enclosure needs to meet a noise criterion.
- [Industrial Noise Control: HVAC and Enclosures](noise-control.md):
  duct attenuation, flow noise and machine-enclosure insertion loss.

## What this section does not cover

Nothing here is a measurement: every number is predicted from geometry and
declared data, and the measurement standards named above are cited as the
source of a supplier's figures, not implemented. Within the predictions, three
limits are structural. **Only reactive silencer elements are computed** —
dissipative duct-lining silencers are discussed for selection but are not
modelled from liner properties anywhere in the library, and on the HVAC page
the lined-elbow figure is a table lookup (Bies Table 8.11) and the plenum
attenuation is Wells' closed form driven by a declared mean absorption — neither is a liner
model. **Mean flow is outside the element matrices**:
convection, temperature gradients and the flow-dependent impedance of
perforates do not appear, so a silencer carrying significant flow is predicted
as though it were not. And `enclosure_insertion_loss` **never predicts the
panel's transmission loss**: you supply R measured or from another model, and
the module combines it with the interior correction — predicting R itself is
[Insulation design](../../buildings/design/index.md). Above the
higher-order-mode cut-on frequency the plane-wave assumption every duct model
rests on stops holding, which the duct-path page states and which no method
here works around.
