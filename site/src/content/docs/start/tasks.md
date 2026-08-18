---
title: "What do you need to measure?"
description: "The task-shaped door into the guides: a job on the left, the guide that answers it on the right, with the standard it implements."
---

The rest of this site is organised by subject, which is right once you know
which subject owns your problem and unhelpful before that. This page is the
other way round: the job on the left, the guide that answers it on the right,
with the standard the guide implements so you can tell before clicking whether
it is the one your client, your reviewer or your regulator will accept.

If your job is not listed, [All guides](/phonometry/start/guides/) is the full
inventory, and the [glossary](/phonometry/reference/glossary/) goes the other
way again: you have a symbol from a report and you want the guide that computes
it.

## From a recording

| The job | Where it is answered |
|---|---|
| I have a WAV recording and I need A-weighted levels, $L_\mathrm{Aeq}$ and the percentile levels | [Build a sound level meter](/phonometry/signals/sound-level-meter/) — IEC 61672-1 and IEC 61260-1, the whole chain on one runnable page, from the calibrator tone to the class check of every stage |
| My numbers are not in pascals and I do not know what they are | [Calibration and dBFS](/phonometry/signals/metrology/calibration/) — the calibrator tone, the sensitivity in pascals per digital unit, the pre/post drift rule and the digital dBFS alternative |
| My file is the meter's own WAV: 24-bit, multichannel, or an overnight RF64, and I want it read and written without touching the level | [Reading and writing measurement audio](/phonometry/io/audio-files/) — the calibrated `Signal`, the `bext` provenance, streaming, BWF writing and the calibration sidecar |
| I need a spectrum rather than a level | [Filter Banks](/phonometry/signals/filters/filter-banks/) for band levels (IEC 61260-1), [Spectral analysis](/phonometry/signals/spectra/spectral-analysis/) for a density estimate |

## Rooms and buildings

| The job | Where it is answered |
|---|---|
| The reverberation time of a room I can get into | [Measuring the Room Impulse Response](/phonometry/buildings/rooms/room-impulse-response/) for the acquisition (ISO 18233 sweeps and the interrupted-noise alternative), then [Room Acoustics](/phonometry/buildings/rooms/room-acoustics/) for $T_{20}$, $T_{30}$, EDT, $C_{50}$, $C_{80}$ and the rest of ISO 3382-1 |
| The reverberation time of a room that does not exist yet | [Reverberation-time prediction](/phonometry/buildings/rooms/reverberation-prediction/) — the Sabine family, with the section that says which model to use and when every model fails; [Sound absorption in enclosed spaces](/phonometry/buildings/rooms/enclosed-space-absorption/) when a design report has to cite EN 12354-6 |
| Whether a room is quiet enough for what happens in it | [Room-noise criteria](/phonometry/buildings/rooms/room-noise/) — NC, RC Mark II and NR, and the section on choosing between them |
| $R'_w$ or $D_\mathrm{nT,w}$ of a partition I measured on site | [Field Insulation Measurement](/phonometry/buildings/insulation/insulation-field/) (ISO 16283-1/-2) for the measurement, then [Insulation Ratings](/phonometry/buildings/insulation/insulation-ratings/) (ISO 717-1/-2) for the single number and the adaptation terms |
| Proof that a wall meets the Spanish CTE DB-HR | [Spanish Building Code](/phonometry/buildings/insulation/spanish-building-code/) — the required quantities, which one each requirement is written against, and the compliance check |
| The insulation of a building I am still designing | [Predicting Sound Insulation (EN 12354)](/phonometry/buildings/design/insulation-prediction/), and [Detailed prediction](/phonometry/buildings/design/detailed-prediction/) when the flanking paths have to be itemised |

## Machines, products and installations

| The job | Where it is answered |
|---|---|
| The sound power of a machine | [Sound Power](/phonometry/devices/emission/sound-power/) — start here whatever the facility: its "Choosing a method" section routes to the pressure (ISO 3744/3745/3746), reverberation-room (ISO 3741), intensity (ISO 9614) or surface-vibration (ISO/TS 7849) method |
| A declared emission value for a datasheet or a CE file | [Declaring the noise emission](/phonometry/devices/emission/sound-power/#declaring-the-noise-emission-iso-4871) — ISO 4871, the declared A-weighted level $L_{WAd}$ and its uncertainty $K_{WA}$ |
| How much a silencer, an enclosure or a duct run will attenuate | [Silencers](/phonometry/devices/noise-control/silencers/), [Industrial noise control](/phonometry/devices/noise-control/noise-control/) for enclosures and HVAC, [Duct paths](/phonometry/devices/noise-control/duct-path/) |
| A loudspeaker, microphone or amplifier measured to its standard | [Electroacoustic measurements](/phonometry/devices/electroacoustics/electroacoustics/) — IEC 60268-3/-4/-5 |
| The loudness of a programme in LUFS | [Programme loudness and true peak](/phonometry/devices/broadcast/program-loudness/) — ITU-R BS.1770-5 and EBU R 128 |

## Environment and transport

| The job | Where it is answered |
|---|---|
| $L_\mathrm{den}$, $L_\mathrm{night}$ or a rating level from an environmental survey | [Environmental noise levels](/phonometry/environment/assessment/environmental-levels/) — ISO 1996-1/-2, the indicators and the adjustments that go on top of them |
| Source power for a road or railway noise map | [CNOSSOS-EU road traffic source emission](/phonometry/environment/sources/cnossos-road-emission/) and [railway source emission](/phonometry/environment/sources/cnossos-rail-emission/) — Annex II of 2002/49/EC, the source side of it |
| How much a distance, a barrier or the weather takes off | [Outdoor propagation](/phonometry/environment/propagation/outdoor-propagation/) (ISO 9613-2), [Ground effect and barriers](/phonometry/environment/propagation/ground-barriers/), [Atmospheric refraction](/phonometry/environment/propagation/atmospheric-refraction/) |
| Whether a wind farm is compliant | [Wind turbine noise](/phonometry/environment/sources/wind-turbine-noise/) — IEC 61400-11 and the tonal audibility assessment |
| The certification level of an aircraft flyover | [Aircraft noise: EPNL](/phonometry/aircraft/aircraft-noise/) — ICAO Annex 16, and [Airport noise contours](/phonometry/aircraft/airport-noise/) for the map around the airport |

## People

| The job | Where it is answered |
|---|---|
| Whether a worker is over the daily exposure limit | [Occupational Noise Exposure](/phonometry/perception/hearing/occupational-exposure/) — ISO 9612, the $L_\mathrm{EX,8h}$ and its uncertainty |
| Loudness in sones, or sharpness, roughness and fluctuation strength | [Loudness](/phonometry/perception/psychoacoustics/loudness/) (ISO 532-1/-2/-3), then [Sound quality metrics](/phonometry/perception/psychoacoustics/sound-quality/) |
| Whether a tone in a noise is audible, and by how much | [Tone audibility](/phonometry/perception/psychoacoustics/tone-audibility/) (ISO/PAS 20065, DIN 45681) and [Tone prominence](/phonometry/perception/psychoacoustics/tone-prominence/), whose section on which metric to use covers both |
| The STI of a public-address system | [Speech Transmission Index](/phonometry/perception/speech/speech-transmission/) — IEC 60268-16, direct and indirect (STIPA) |
| A seat, floor or hand-arm vibration record against the Directive | [Human Vibration](/phonometry/vibration/human/human-vibration/) — ISO 2631-1 and ISO 5349-1, and [Multiple-shock vibration](/phonometry/vibration/human/multiple-shock-vibration/) for ISO 2631-5 |

## Materials, and under water

| The job | Where it is answered |
|---|---|
| The absorption coefficient of a sample | [Sound Absorption Measurement and Rating](/phonometry/materials/absorbers/absorption-measurement/) for the reverberation room (ISO 354, ISO 11654), [Impedance Tube](/phonometry/materials/absorbers/impedance-tube/) for normal incidence (ISO 10534-2) |
| The airflow resistivity or the dynamic stiffness of a layer | [Airflow resistance](/phonometry/materials/absorbers/airflow-resistance/) (ISO 9053-1/-2), [Dynamic stiffness](/phonometry/materials/resilient/dynamic-stiffness/) (EN 29052-1) |
| How much a diffuser scatters | [Diffusers and scattering](/phonometry/materials/diffusers/diffusers/) — ISO 17497-1/-2, and the difference between the two coefficients |
| A ship's radiated noise, or the exposure of a pile-driving campaign | [Underwater acoustics](/phonometry/underwater/underwater-acoustics/) (ISO 17208, ISO 18406) then [Marine-mammal noise exposure](/phonometry/underwater/marine-mammal-exposure/) |
| How far a sound carries in the sea | [Underwater propagation](/phonometry/underwater/underwater-propagation/), and [Underwater propagation solvers](/phonometry/underwater/underwater-solvers/) when a range-dependent field is needed |

## When more than one method is right

Several guides carry the decision themselves, in a section written for a reader
who has the problem and not yet the method. They are worth opening before the
guide you think you need:

- [Choosing F, S or I](/phonometry/signals/levels/time-weighting/#choosing-f-s-or-i) — which exponential time weighting a measurement calls for.
- [Which filter architecture should I choose?](/phonometry/signals/filters/filter-gallery/#which-filter-architecture-should-i-choose) — Butterworth against the four alternatives, and what each trades away.
- [Choosing a criterion: NC, RC Mark II or NR](/phonometry/buildings/rooms/room-noise/#3-choosing-a-criterion-nc-rc-mark-ii-or-nr).
- [Choosing a model, and when every model fails](/phonometry/buildings/rooms/reverberation-prediction/#4-choosing-a-model-and-when-every-model-fails) — Sabine, Eyring, Millington, Fitzroy, Arau.
- [Choosing a method](/phonometry/devices/emission/sound-power/#choosing-a-method) — the four routes to sound power and what each facility allows.
- [Choosing a loudness model](/phonometry/perception/psychoacoustics/advanced-loudness/#choosing-a-loudness-model) — Zwicker, Moore-Glasberg and Sottek.
- [Which measure, and when](/phonometry/perception/speech/objective-intelligibility/#4-which-measure-and-when) — STI against the intrusive and non-intrusive measures.
- [Which tonality metric, and when](/phonometry/perception/psychoacoustics/tone-prominence/#3-which-tonality-metric-and-when).
- [Direct or indirect](/phonometry/perception/speech/speech-transmission/#direct-or-indirect-choosing-between-them) — full-matrix STI against STIPA.
- [Choosing a model](/phonometry/underwater/underwater-solvers/#6-choosing-a-model) — normal modes, rays, Gaussian beams and the parabolic equation.

## Before you measure anything

Two habits save more measurements than any of the pages above. Record the
calibrator tone through the same chain, before and after, and treat the
difference as your drift bound — [Getting
Started](/phonometry/start/getting-started/) shows why an uncalibrated level is
not approximately right but arbitrary. And read the "What this guide covers"
block at the end of whichever guide you land on: it states which clauses,
annexes and methods are implemented and which are not, which is the question a
reviewer will ask first.
