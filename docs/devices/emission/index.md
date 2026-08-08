← [Documentation index](../../README.md)

# Sound power and intensity

The central quantity of this section is the **sound power**: the total
acoustic energy per second a source radiates. Expressed in decibels as the
sound power level, it is the figure that goes on a datasheet, feeds a room or
outdoor prediction and is checked against noise-emission limits.

[Sound Power](sound-power.md) chooses between the
standardised routes and closes the job with the ISO 4871 emission
declaration, and each route has its own page:
[Sound Power by Pressure Methods](sound-power-pressure.md)
for the enveloping surface of ISO 3744/3746 and the precision anechoic grade
of ISO 3745,
[Sound Power in the Reverberation Room](sound-power-reverberation.md)
for the direct and comparison methods of ISO 3741, and
[Sound Power by Intensity Scanning](sound-power-intensity.md)
for the on-site scanning of ISO 9614-2 and its ISO 9614-3 precision grade.

A fourth route does not measure sound at all.
[Sound power from surface vibration (ISO/TS 7849)](vibration-sound-power.md)
estimates the radiated power from the surface-averaged velocity level and a
radiation factor, which is what remains when the machine cannot be moved to a
qualified room and its environment is too noisy for an enveloping surface:
Part 1 gives an upper-limit value from the velocity alone, Part 2 an
engineering value once the radiation factor has been estimated properly. It
also answers a slightly different question from the four acoustic routes — it
characterises what the *structure* radiates, and stays blind to sound escaping
through openings, intakes and outlets — and it is the natural bridge to the
structure-borne pages, since the same surface velocity is what
[Vibration and structure-borne sound](../../vibration/index.md) measures.

Behind the intensity-based routes sits **sound intensity** itself: the signed
power flux that can localise sources and separate them from background noise,
measured with a two-microphone probe per IEC 61043 and qualified by the
ISO 9614-1 field indicators, covered in
[Sound Intensity (p-p)](intensity.md).

If you are here to measure a machine, start with
[Sound Power](sound-power.md) and let its decision guidance
pick the route; read [Sound Intensity (p-p)](intensity.md)
when that route involves an intensity probe, and go to
[Sound power from surface vibration](vibration-sound-power.md)
when the machine cannot leave its installation and the background is too high
for any pressure method. The determined power level is what the quieting
measures of the [Noise control](../noise-control/index.md) pages are
judged against.

## Pages in this section

- [Sound Intensity (p-p)](intensity.md): two-microphone
  sound intensity per IEC 61043 with the ISO 9614-1 field indicators.
- [Sound Power](sound-power.md): choosing the determination
  method and declaring the noise emission per ISO 4871.
- [Sound Power by Pressure Methods](sound-power-pressure.md):
  the enveloping surface of ISO 3744/3746 and the precision anechoic grade of
  ISO 3745.
- [Sound Power in the Reverberation Room](sound-power-reverberation.md):
  the direct and comparison methods of ISO 3741.
- [Sound power from surface vibration (ISO/TS 7849)](vibration-sound-power.md):
  the radiated power from the surface-averaged velocity level and the
  radiation factor, with the Part 1 upper limit and the Part 2 engineering
  value.
- [Sound Power by Intensity Scanning](sound-power-intensity.md):
  the on-site scanning of ISO 9614-2 and the ISO 9614-3 precision grade.

## What this section does not cover

The determination methods start after the facility and the probe have been
qualified. ISO 3745's free-field qualification of an anechoic or hemi-anechoic
room, ISO 3741's reverberation-room qualification (eigenfrequency counting or a
reference-source comparison) and the IEC 61043 residual-intensity test of a
probe-and-analyser chain are all **assumed, not performed**: the library warns
on the coarse advisory criteria the standards state explicitly — the Table 1
minimum volume, the position count, an inter-position spread above 1.5 dB, the
ISO 3744 K₂ validity — and grades a residual-intensity index you measured
yourself. The C₃ meteorological correction of ISO 3745 likewise needs an
air-absorption coefficient you supply; it is not computed from ISO 9613-1 here.

One route is absent by design: **ISO 9614-1's discrete fixed-point power
summation is not implemented at all**, and only its Annex A field indicators
are, reused by the two scanning parts. On the vibration route, the measurement
clauses 5 to 7 of both parts of ISO/TS 7849 are laboratory practice rather than
code, and only the single-machine radiation factor of Formula 8 is implemented,
so a batch or family determination needs an already-averaged value. Finally,
nothing here reduces a machine's emission: quieting a source is [Noise
control](../noise-control/index.md), and a declared emission value is
the input to that work, not its result.
