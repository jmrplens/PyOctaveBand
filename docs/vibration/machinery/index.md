← [Documentation index](../../README.md)

# Machinery

A rotating machine has a **kinematic signature**. Every periodicity in its
vibration belongs to something that turns, meshes or passes, and the geometry
fixes the frequency before any measurement is made: a bearing with fifteen
elements running at a given shaft speed has an outer-race pass frequency that
can be written down, not searched for. Three families cover most machines. For a
**rolling-contact bearing**, the outer-race and inner-race element-pass
frequencies, the cage rate and the rolling-element spin rate, all scaling with
the shaft speed and set by the element and pitch diameters and the contact
angle; their sum is exactly the element count times the shaft rate, which
catches a mistyped geometry instantly. For a **gear pair**, the mesh frequency
and the sideband families that separate a chipped tooth from an eccentric
wheel. For **motors and bladed rotors**, the supply, slip, pole-pass and
rotor-slot lines of an induction motor and the blade-passing tones of fans,
blowers and pumps, with the lobed interaction patterns of a ducted axial fan.

The standard answer to finding one of those lines under the broadband noise of
a running machine is **envelope analysis**, and it is three steps in three
places. Band-pass the record around the high-frequency housing resonance that
the impacts ring, take the envelope spectrum so the repetition rate of the
impacts becomes a discrete line — that step is [Cepstrum, echoes and the
envelope spectrum](../../signals/spectra/cepstrum-echoes.md), one section
away — and overlay the kinematic families, coloured by origin, so a shaft
harmonic can never be misread as bearing evidence. When two shafts have to be
separated before their sidebands can be read, [Time synchronous
averaging](../../signals/spectra/synchronous-averaging.md) does it first.

[Machine fault frequencies](machine-diagnostics.md)
computes the families and draws them on a measured envelope spectrum.

## Pages in this section

- [Machine fault frequencies](machine-diagnostics.md):
  the rolling-contact bearing frequencies, the gear-mesh frequency and its
  sidebands, the induction-motor supply, slip, pole-pass and rotor-slot
  harmonics, and the blade-passing tones of fans, blowers and pumps, all from
  the geometry and the shaft speed (Norton & Karczub, Section 8.4).

## See also

Pages elsewhere on the site that this section leans on:

- [Cepstrum, echoes and the envelope spectrum](../../signals/spectra/cepstrum-echoes.md):
  the envelope spectrum this section's workflow depends on, and the cepstrum
  that reads periodic ripple in the spectrum itself.
- [Time synchronous averaging](../../signals/spectra/synchronous-averaging.md):
  extracting one shaft's contribution before its sidebands are read.
- [Mechanical mobility and the FRF family (ISO 7626-1)](../structural/mechanical-mobility.md):
  the frequency-response vocabulary behind the housing resonance the envelope
  method rings.
- [Bending-wave transmission at plate junctions](../structural/junction-transmission.md):
  where the machine's vibration goes once it has left the machine.

## What this section does not cover

These are **predictions, not detections**. Nothing here decides whether a line
is present, only where it would be if it were: the overlay is a set of
expectations to read a measured spectrum against, and the reading is yours. A
loaded bearing slips a little, so expect the measured peak within a per cent
or two of the prediction rather than exactly on it.

Nor is there a severity verdict. The amplitude criteria that turn a present
line into an assessment — crest-factor and kurtosis trending, and the velocity
severity bands of ISO 10816 / ISO 20816 — are outside this module, and rotor
balancing (ISO 21940) and order tracking are absent from the library
altogether. One published convention differs between sources and is flagged on
the guide rather than hidden: the pole-pass frequency is standard
condition-monitoring practice rather than Norton's, who gives the slip
frequency itself as the broken-bar sideband spacing.
