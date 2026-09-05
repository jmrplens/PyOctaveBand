---
title: "Guides"
description: "The 115 guides of phonometry, grouped into the twelve topics the library covers: what each area is for, the standards implemented in it, and a one-line summary of every guide inside it."
head:
  - tag: script
    attrs:
      type: application/ld+json
    content: |
      {
        "@context": "https://schema.org",
        "@type": "ItemList",
        "@id": "https://jmrplens.github.io/phonometry/start/guides/#areas",
        "name": "phonometry documentation areas",
        "description": "The twelve documented areas of the phonometry guides, each with the standards it implements.",
        "inLanguage": "en",
        "numberOfItems": 12,
        "itemListOrder": "https://schema.org/ItemListUnordered",
        "itemListElement": [
          {
            "@type": "ListItem",
            "position": 1,
            "name": "Signal analysis",
            "description": "Filter banks, weighting, levels, spectra, calibration and uncertainty.",
            "url": "https://jmrplens.github.io/phonometry/signals/"
          },
          {
            "@type": "ListItem",
            "position": 2,
            "name": "The medium",
            "description": "The fluid a sound travels through, computed from the measured conditions: humid air from IEC 61094-2:2009 Annex F.",
            "url": "https://jmrplens.github.io/phonometry/fluids/"
          },
          {
            "@type": "ListItem",
            "position": 3,
            "name": "Audio files",
            "description": "Measurement audio in and out: calibrated reading, provenance, streaming, BWF writing and lossless conversion.",
            "url": "https://jmrplens.github.io/phonometry/io/"
          },
          {
            "@type": "ListItem",
            "position": 4,
            "name": "Hearing and perception",
            "description": "Loudness, sound quality, speech intelligibility, hearing and exposure.",
            "url": "https://jmrplens.github.io/phonometry/perception/"
          },
          {
            "@type": "ListItem",
            "position": 5,
            "name": "Rooms and buildings",
            "description": "Room parameters, background noise, field and laboratory insulation, prediction.",
            "url": "https://jmrplens.github.io/phonometry/buildings/"
          },
          {
            "@type": "ListItem",
            "position": 6,
            "name": "Materials and surfaces",
            "description": "Absorption, airflow resistance, impedance tube, porous and metamaterial models, diffusers, scattering.",
            "url": "https://jmrplens.github.io/phonometry/materials/"
          },
          {
            "@type": "ListItem",
            "position": 7,
            "name": "Vibration and structure-borne sound",
            "description": "Mobility and FRFs, isolators, radiated power, junctions, human vibration.",
            "url": "https://jmrplens.github.io/phonometry/vibration/"
          },
          {
            "@type": "ListItem",
            "position": 8,
            "name": "Environment and transport",
            "description": "Outdoor propagation, barriers, refraction, road, rail and wind-turbine sources, and the assessment built on them.",
            "url": "https://jmrplens.github.io/phonometry/environment/"
          },
          {
            "@type": "ListItem",
            "position": 9,
            "name": "Aircraft noise",
            "description": "Certification levels, airport contours and the rotorcraft hemisphere method.",
            "url": "https://jmrplens.github.io/phonometry/aircraft/"
          },
          {
            "@type": "ListItem",
            "position": 10,
            "name": "Underwater acoustics",
            "description": "Levels re 1 microPa, ship radiated noise, pile driving, ambient noise, propagation loss.",
            "url": "https://jmrplens.github.io/phonometry/underwater/"
          },
          {
            "@type": "ListItem",
            "position": 11,
            "name": "Sources and devices",
            "description": "Sound power, intensity, emission declarations, electroacoustics, programme loudness.",
            "url": "https://jmrplens.github.io/phonometry/devices/"
          },
          {
            "@type": "ListItem",
            "position": 12,
            "name": "Wave simulation",
            "description": "Deterministic 2D FDTD solvers, acoustic and elastic P-SV, validated against analytic oracles rather than a standard.",
            "url": "https://jmrplens.github.io/phonometry/simulation/"
          }
        ]
      }
---

Every guide on this site follows the same shape: the standard it implements,
the quantities that standard defines, the assumptions the implementation
makes, then runnable code and the figure it draws. Nothing here is a survey of
the field; each page is the working documentation of a module, written so that
a result can be defended clause by clause rather than trusted.

This page is the map. A hundred and fifteen guides sit in twelve topics, and each topic has its
own overview page with the longer story of how its pieces fit together. If you
are arriving without a specific question, read
[Getting Started](/phonometry/start/getting-started/) first: it runs one signal
through the whole processing chain and gives the vocabulary the rest of the
guides assume. If you already know the quantity you need, the
[glossary](/phonometry/reference/glossary/) lists every symbol with its unit,
its defining standard and the guide that implements it.

Two pointers before the list. Function signatures live in the
[API reference](/phonometry/reference/api/), which is generated from the
docstrings and is not repeated here. Derivations, design decisions and the
numerical evidence live in [Reference](/phonometry/reference/): the theory
pages explain why a formula is the one it is, and the conformance report shows
the standard's own expected value next to the computed one.

A third pointer, for anyone arriving with a measurement to make rather than a
number to compute. Where a method needs a physical arrangement, its guide
carries an engineering setup diagram with the geometry the standard prescribes:
source and microphone positions, separations, mounting, and the specimen or the
enclosing surface. [Field Insulation Measurement](/phonometry/buildings/insulation/insulation-field/)
draws the ISO 16283-1 room pair that way, and
[Impedance Tube](/phonometry/materials/absorbers/impedance-tube/) draws the
ISO 10534-2 tube with both microphone spacings. The prediction and rating
guides take element or material data as their input instead and need no
facility, which is usually the fastest way to tell the two kinds of page apart.
[What do you need to measure?](/phonometry/start/tasks/) sorts the same guides
by the job rather than by the topic.

## [Signal analysis](/phonometry/signals/)

Filter banks, weighting, levels, spectra, calibration and uncertainty. This is
the chain that turns a digital signal into a standards-compliant number, and
every other area consumes it: a loudness model needs calibrated band levels, a
room parameter needs a filtered impulse response, an environmental rating is an
adjusted $L_\mathrm{eq}$. Implements IEC 61260-1, ANSI S1.11, IEC 61672-1, ISO 7196,
IEC 61252, ISO 1996-1, IEC 60942 and the GUM.

- [Build a sound level meter](/phonometry/signals/sound-level-meter/):
  the whole area assembled end to end on one runnable page, from the
  calibrator tone to the reported levels.

**[Octave filtering](/phonometry/signals/filters/)**

- [Filter Banks](/phonometry/signals/filters/filter-banks/): the fractional-octave band
  mathematics, the bank parameters, the parametric EQ, band decomposition and
  zero-phase offline filtering.
- [Filter Architecture Gallery](/phonometry/signals/filters/filter-gallery/): the five
  filter architectures compared at the band edges, the full response gallery
  and per-architecture usage, with the Linkwitz-Riley crossover.
- [Filter Class Verification (IEC 61260-1)](/phonometry/signals/filters/filter-compliance/):
  the Table 1 acceptance mask band by band, the class 0 of the withdrawn 1995
  edition and the compliance fiche.
- [Block Processing](/phonometry/signals/filters/block-processing/): stateful streaming
  analysis that carries filter state across buffers, for signals that never fit
  in memory.
- [Multichannel and Performance](/phonometry/signals/filters/multichannel/): vectorized
  analysis of many channels at once, with the (channels, samples) convention
  and performance notes.

**[Levels and weighting](/phonometry/signals/levels/)**

- [Frequency Weighting (A, C, Z)](/phonometry/signals/levels/weighting/):
  the IEC 61672-1 ear-response curves, the high-frequency accuracy mode and
  the Table 3 class verification.
- [Special Weightings (G, B, D, AU)](/phonometry/signals/levels/special-weightings/):
  ISO 7196 G-weighting for infrasound, the historical B and D curves and AU
  for audible sound in the presence of ultrasound.
- [Time Weighting](/phonometry/signals/levels/time-weighting/): the Fast, Slow and
  Impulse exponential ballistics of IEC 61672-1.
- [Integrated and Statistical Levels](/phonometry/signals/levels/levels/): $L_\mathrm{eq}$ and
  $L_\mathrm{Aeq}$, the percentile levels $L_{10}$/$L_{50}$/$L_{90}$, $L_\mathrm{Cpeak}$ and
  SEL, and the noise dose.
- [Environmental Levels (ISO 1996-1/-2)](/phonometry/environment/assessment/environmental-levels/):
  $L_\mathrm{den}$, $L_\mathrm{dn}$ and the composite rating levels, the tonal adjustment, the
  residual-noise correction and the uncertainty budget. It lives under
  Environment and transport, and is repeated here because it is the level
  definitions above aggregated over a day.
- [Spanish Noise Regulation (RD 1367/2007)](/phonometry/environment/assessment/spanish-noise-regulation/):
  the corrected level $L_\mathrm{Keq}$ with its $K_\mathrm{t}$, $K_\mathrm{f}$ and $K_\mathrm{i}$ corrections, the
  evaluation periods and noise phases, the limit tables and the Article 25
  compliance check. It lives under Environment and transport too.

**[Signals and spectra](/phonometry/signals/spectra/)**

- [Calibrated spectral analysis](/phonometry/signals/spectra/spectral-analysis/): the
  Welch power and cross-spectral estimators with their random errors,
  chi-square confidence intervals and fractional-octave smoothing.
- [Multiple and partial coherence](/phonometry/signals/spectra/miso-coherence/): the
  ordinary, multiple and partial coherence of several correlated sources
  driving one response, and which source dominates each band.
- [Time-frequency analysis](/phonometry/signals/spectra/time-frequency/): the calibrated
  STFT spectrogram in absolute dB SPL, and the zoom FFT that resolves tones
  closer than a practical FFT bin.
- [Cepstrum, echoes and the envelope spectrum](/phonometry/signals/spectra/cepstrum-echoes/):
  quefrency analysis, echo detection with the reflection coefficient read off
  the cepstral peak, liftering and the envelope spectrum.
- [Time synchronous averaging](/phonometry/signals/spectra/synchronous-averaging/):
  extraction of a periodic waveform of known period, the comb filter that
  describes it and the choice of the number of averages.
- [Machine fault frequencies](/phonometry/vibration/machinery/machine-diagnostics/):
  the kinematic fault-frequency families of rotating machinery (Norton &
  Karczub Section 8.4) drawn on top of a measured envelope spectrum: bearing
  BPFO, BPFI, BSF and cage frequencies, gear-mesh sidebands, induction-motor
  slip, pole-pass and rotor-slot harmonics, and blade-passing tones. It lives
  under Vibration and structure-borne sound, and is repeated here because it is
  the envelope spectrum above put to work.
- [Correlation, time delay and envelope](/phonometry/signals/spectra/correlation-delay/):
  correlation with its random errors, time-delay estimation by direct
  correlation and the GCC weightings, and the Hilbert envelope.
- [Test signals and sample-rate tools](/phonometry/signals/spectra/test-signals/):
  IEC 60268-1 tone bursts with exact gating, colored noise with an exact
  slope, resampling with a stated anti-alias specification and fractional
  delay.
- [System measurement](/phonometry/signals/spectra/system-measurement/): complementary
  Golay pairs, sweeps shaped to an arbitrary target magnitude spectrum, and
  Kirkeby-regularized inversion of a measured response.

**[Calibration and uncertainty](/phonometry/signals/metrology/)**

- [Calibration and dBFS](/phonometry/signals/metrology/calibration/): physical SPL
  calibration from a calibrator tone or a known sensitivity, and the digital
  full-scale mode.
- [Compliance and verification](/phonometry/signals/metrology/compliance-verification/):
  what a performance class asserts, the verifiers per stage, the conformance
  report, and the scope of the pattern-evaluation and periodic-test parts.
- [Measurement uncertainty (GUM and Monte Carlo)](/phonometry/signals/metrology/gum-uncertainty/):
  the law of propagation of uncertainty and the Monte Carlo method, with
  expanded uncertainty and coverage intervals.
- [Data qualification](/phonometry/signals/metrology/data-qualification/): the reverse
  arrangement and runs tests for stationarity, and the Rice level-crossing and
  peak statistics with the irregularity factor.

## [The medium](/phonometry/fluids/)

The fluid the sound travels through, computed from the conditions that were
measured rather than assumed. A density and a speed of sound stand behind
every level in the library, and this is where they come from instead of
being typed once. Sits with the filters, the signal analysis and the
metrology in the transverse toolbox, because a medium is not a subject some
domains have and others do not. Implements IEC 61094-2:2009 Annex F
(CIPM-2007).

- [Humid air](/phonometry/fluids/humid-air/): the density, speed of sound,
  ratio of specific heats, viscosity and thermal diffusivity of air from the
  measured temperature, pressure and humidity, what each condition is worth,
  the domain the annex states for itself, and what the model refuses to
  guess.

## [Audio files](/phonometry/io/)

Measurement audio in and out. The file layer of the signal chain: every
linear WAV a sound level meter or field recorder writes comes back as a
calibrated `Signal` with its `bext` provenance, long recordings stream
through the stateful filters block by block, and what leaves the library is
a BWF with its provenance and a sidecar carrying the calibration.
Implements EBU Tech 3285 and ITU-R BS.2088; FLAC archives follow RFC 9639.

- [Reading and writing measurement audio](/phonometry/io/audio-files/): the
  whole workflow on one runnable page, from the meter's WAV to the
  calibrated level, the lossy warning, streaming, BWF writing, the sidecar
  and lossless conversion.

## [Hearing and perception](/phonometry/perception/)

Loudness, sound quality, speech intelligibility, hearing and exposure. Where
the core area asks how much sound there is, this one asks what a listener makes
of it: how loud it seems, how sharp or rough or annoying, how much of a talker
survives the room, and how much hearing a working life in that noise costs.
Implements ISO 532-1/-2/-3, ECMA-418-1/-2, ISO 226, DIN 45692, IEC 60268-16,
ANSI S3.5, DIN 45681, ISO/PAS 20065, ISO 7029, ISO 389-7, ISO 1999 and
ISO 9612.

**[Psychoacoustics](/phonometry/perception/psychoacoustics/)**

- [Loudness](/phonometry/perception/psychoacoustics/loudness/): loudness in sones by the ISO 532-1
  Zwicker method with its one-page fiche, plus the ISO 226 equal-loudness
  contours.
- [Advanced Loudness (ISO 532-2/-3, ECMA-418-2)](/phonometry/perception/psychoacoustics/advanced-loudness/):
  the Moore-Glasberg stationary and time-varying methods and the Sottek
  Hearing Model loudness, with the model-choice table.
- [Sound Quality Metrics](/phonometry/perception/psychoacoustics/sound-quality/): sharpness in
  acum, and the ECMA-418-2 tonality, roughness and fluctuation strength.
- [Prominent Discrete Tones (ECMA-418-1)](/phonometry/perception/psychoacoustics/tone-prominence/):
  the tone-to-noise ratio and prominence ratio, with their frequency-dependent
  prominence criteria.
- [Objective audibility of tones in noise (ISO/PAS 20065)](/phonometry/perception/psychoacoustics/tone-audibility/):
  the critical-band masking level, the masking index and the audibility of a
  tone above the masking threshold.
- [Psychoacoustic annoyance and fluctuation strength](/phonometry/perception/psychoacoustics/psychoacoustic-annoyance/):
  the Fastl and Zwicker annoyance model built from loudness, sharpness,
  roughness and fluctuation strength.

**[Speech](/phonometry/perception/speech/)**

- [Speech Transmission Index (STI)](/phonometry/perception/speech/speech-transmission/):
  the modulation transfer function, the indirect method from an impulse
  response and the direct STIPA measurement.
- [Speech Intelligibility Index](/phonometry/perception/speech/speech-intelligibility/):
  the SII in all four of the standard's band procedures, with the
  band-importance functions, self-masking and upward spread of masking.
- [Objective Intelligibility (STOI and ESTOI)](/phonometry/perception/speech/objective-intelligibility/):
  the two correlation-based measures for time-frequency weighted noisy speech.

**[Hearing and exposure](/phonometry/perception/hearing/)**

- [Hearing threshold (age and reference zero)](/phonometry/perception/hearing/hearing-threshold/):
  the age-related threshold distribution of ISO 7029 and the ISO 389-7
  reference threshold of hearing.
- [Noise-induced hearing loss (ISO 1999)](/phonometry/perception/hearing/noise-induced-hearing-loss/):
  the permanent threshold shift as a function of level, duration and frequency,
  combined with the age component.
- [Occupational Noise Exposure (ISO 9612)](/phonometry/perception/hearing/occupational-exposure/):
  the task-based, job-based and full-day strategies for $L_\mathrm{EX,8h}$, with the
  uncertainty budget and the upper limit.
- [Hearing Protectors (ISO 4869-2)](/phonometry/perception/hearing/hearing-protectors/):
  the octave-band, HML and SNR methods that say what a protector leaves at the
  ear, and the assumed protection value all three start from.

## [Rooms and buildings](/phonometry/buildings/)

Room parameters, background noise, field and laboratory insulation, and
prediction from element data. Two questions run through the area: how a room
treats the sound made inside it, and how much of the sound made next door gets
through. Implements ISO 3382-1/-2/-3, ISO 16283-1/-2/-3, ISO 10140, ISO 10848,
ISO 15186-1/-2, ISO 16251-1, ISO 717-1/-2, EN 12354-1 to -6, ISO 18233,
ISO 12999-1, ISO 10052, ANSI/ASA S12.2 and ASTM E413/E1414.

**[Room acoustics](/phonometry/buildings/rooms/)**

- [Measuring the Room Impulse Response](/phonometry/buildings/rooms/room-impulse-response/):
  the ISO 18233 deterministic acquisition, exponential sweeps with their
  deconvolution, and MLS.
- [Room Acoustics](/phonometry/buildings/rooms/room-acoustics/): the room parameters
  EDT, $T_{20}$, $T_{30}$, $C_{50}$, $C_{80}$, $D_{50}$ and $T_\mathrm{s}$ derived from
  that impulse response.
- [Open-Plan Office Acoustics (ISO 3382-3)](/phonometry/buildings/rooms/open-plan-acoustics/):
  the spatial decay rate of speech and the distraction and privacy distances
  of an open-plan floor.
- [Image sources and the steady-state room field](/phonometry/buildings/rooms/room-image-sources/):
  the deterministic image-source impulse response of a rectangular room, the
  room constant, critical distance and Schroeder frequency.
- [Room-noise criteria (NC / RC Mark II)](/phonometry/buildings/rooms/room-noise/): the
  ANSI/ASA S12.2 Noise Criteria rating by tangency, and the Room Criteria
  Mark II rating with its rumble, hiss or neutral tag.
- [Reverberation-time prediction (Sabine, Eyring, Arau)](/phonometry/buildings/rooms/reverberation-prediction/):
  five statistical-acoustics models from volume, boundary areas and surface
  absorption, including the air term.
- [Sound absorption in enclosed spaces (EN 12354-6)](/phonometry/buildings/rooms/enclosed-space-absorption/):
  the total equivalent absorption area of a room from its surfaces, objects and
  air, and the reverberation time that follows.

**[Sound insulation](/phonometry/buildings/insulation/)**

- [Field Insulation Measurement (ISO 16283)](/phonometry/buildings/insulation/insulation-field/):
  airborne and impact insulation measured in the building, its test report and
  the ISO 12999-1 uncertainty that qualifies it.
- [Small Rooms: the ISO 16283 Low-Frequency Procedure](/phonometry/buildings/insulation/low-frequency-procedure/):
  the corner measurement ISO 16283 makes mandatory below 25 m³, and the 63 Hz
  octave reverberation time that comes with it.
- [Laboratory Insulation Measurement](/phonometry/buildings/insulation/insulation-lab/):
  the ISO 10140 characterisation of an element with flanking suppressed.
- [Sound Insulation by Intensity (ISO 15186)](/phonometry/buildings/insulation/insulation-intensity/):
  the transmitted power read off the radiating face, for the whole element or
  element by element.
- [Sound Insulation Survey Method (ISO 10052)](/phonometry/buildings/insulation/insulation-survey/):
  the octave-band control method with its reverberation index and its
  airborne, impact, façade and service-equipment quantities.
- [Laboratory Flanking Transmission (ISO 10848)](/phonometry/buildings/insulation/flanking-lab/):
  the junction vibration reduction index and the flanking level differences
  measured on a test facility.
- [Heavy and Soft Impact Sources (ISO 16283-2)](/phonometry/buildings/insulation/heavy-impact-sources/):
  the rubber ball and the bang machine: the impact force exposure level that
  specifies them, the laboratory check they have to pass, and the single number
  of ISO 717-2 Annex D.
- [Insulation Ratings (ISO 717)](/phonometry/buildings/insulation/insulation-ratings/):
  the airborne and impact reference-curve engines with $C$, $C_\mathrm{tr}$ and $C_\mathrm{I}$,
  the enlarged-range terms and the ISO 717 fiche.
- [Façade Sound Insulation](/phonometry/buildings/insulation/facade-insulation/): the
  building envelope measured per ISO 16283-3, predicted per EN 12354-3 and
  radiating outwards per EN 12354-4.
- [Spanish Building Code (CTE DB-HR)](/phonometry/buildings/insulation/spanish-building-code/):
  the DB-HR global indices $R_\mathrm{A}$, $R_\mathrm{A,tr}$, $D_\mathrm{nT,A}$ and $D_{2\mathrm{m,nT,Atr}}$,
  the clause 2 requirement tables and the window-size correction.

**[Insulation design](/phonometry/buildings/design/)**

- [Predicting Sound Insulation (EN 12354)](/phonometry/buildings/design/insulation-prediction/):
  in-situ airborne and impact insulation between rooms from element data, with
  their flanking paths.
- [Detailed Per-Band Prediction (ISO 12354)](/phonometry/buildings/design/detailed-prediction/):
  the same prediction band by band rather than as one number: in-situ element and
  junction data, the flanking index and impact level, and the per-path
  contributions behind R'w and L'n,w.
- [Predicting Panel Sound Insulation](/phonometry/buildings/design/panel-sound-insulation/):
  the mass law and coincidence dip, double walls, slits and apertures, plate
  radiation efficiency and point mobilities.
- [Floor-Covering Impact Improvement (ISO 16251-1)](/phonometry/buildings/design/impact-improvement/):
  the weighted improvement of a soft floor covering measured on a small
  heavyweight mock-up.
- [Predicting Resilient-Layer Performance](/phonometry/buildings/design/resilient-layers/):
  the tapping-machine force model, the cut-off frequency of a soft covering,
  the floating-floor improvement laws and the ISO 12354-1 Annex D rating of a
  wall lining.
- [Dynamic stiffness of resilient materials (EN 29052-1)](/phonometry/materials/resilient/dynamic-stiffness/):
  the stiffness per unit area under a floating floor from the load-plate
  resonance, with the enclosed-gas term. It lives under Materials and surfaces,
  and is repeated here because it is the input the layer models above ask for.
- [Structure-borne sound power of equipment (EN 15657)](/phonometry/buildings/design/structure-borne-power/):
  the reception-plate method and the plate-independent source quantities:
  equivalent blocked force, free velocity and source mobility. Also listed under
  Vibration, beside the mobilities it is built from.
- [Installed structure-borne sound (EN 12354-5)](/phonometry/buildings/design/installed-structure-borne/):
  the coupling term from source and receiver mobilities, the installed power and
  the per-path sound pressure level in the receiving room. Also listed under
  Vibration.

## [Materials and surfaces](/phonometry/materials/)

Absorption, airflow resistance, the impedance tube, porous and metamaterial
models, diffusers and scattering. What a surface does to the sound that
reaches it, measured in a laboratory or predicted from the material
parameters. Implements ISO 354, ISO 11654, ISO 10534-1/-2, ISO 9053-1/-2,
ISO 17497-1/-2, ISO 13472-1/-2, EN 29052-1 and ISO 12999-2.

**[Absorbers](/phonometry/materials/absorbers/)**

- [Sound Absorption Measurement and Rating](/phonometry/materials/absorbers/absorption-measurement/):
  the ISO 354 reverberation-room measurement, the weighted rating and its
  class, and the measurement uncertainty of both.
- [Airflow Resistance](/phonometry/materials/absorbers/airflow-resistance/): the static
  and alternating determination of airflow resistance and resistivity.
- [Impedance Tube](/phonometry/materials/absorbers/impedance-tube/): the normal-incidence
  surface impedance, absorption and transmission loss, plus the virtual
  FDTD tube.
- [Porous and Multilayer Absorbers](/phonometry/materials/absorbers/porous-absorbers/): the
  Delany-Bazley, Miki and Johnson-Champoux-Allard models, the transfer-matrix
  multilayer solver with perforated, microperforated and membrane layers, and
  the random-incidence integral.
- [Metamaterial Absorbers](/phonometry/materials/absorbers/metamaterial-absorbers/): the
  critical-coupling condition for perfect absorption and the slow-sound slit
  panel loaded by Helmholtz resonators, with its design solver.
**[Diffusers and surfaces](/phonometry/materials/diffusers/)**

- [Diffusers and Their Coefficients](/phonometry/materials/diffusers/diffusers/): the
  random-incidence scattering coefficient, the autocorrelation diffusion
  coefficient, and Schroeder design with its far-field prediction.
- [Metadiffusers](/phonometry/materials/diffusers/metadiffusers/): deep-subwavelength
  Schroeder diffusers from resonator-loaded slits, slow sound and ternary
  sequences.
- [In-situ Road-Surface Absorption](/phonometry/materials/surfaces/road-absorption/):
  in-situ road-surface absorption by the subtraction technique and the spot
  method. It is the only guide of the [surfaces measured in
  place](/phonometry/materials/surfaces/) overview, which the sidebar files in
  this same group.

**[Resilient layers](/phonometry/materials/resilient/)**

- [Dynamic stiffness of resilient materials (EN 29052-1)](/phonometry/materials/resilient/dynamic-stiffness/):
  the stiffness per unit area under a floating floor from the load-plate
  resonance, with the enclosed-gas term. Also listed under Rooms and buildings,
  beside the floating-floor prediction that consumes it.

## [Vibration and structure-borne sound](/phonometry/vibration/)

Mobility and frequency-response functions, isolators, radiated power,
junctions and human vibration. The area covers the path a machine takes into a
structure and out again as airborne sound, and the separate question of what
vibration does to the person exposed to it. Implements ISO 7626-1/-2,
ISO 10846-1/-2/-3, ISO 9611, ISO/TS 7849-1/-2, EN 15657, EN 12354-5,
ISO 2631-1/-2/-4/-5, ISO 5349-1/-2 and ISO 8041-1.

**[Structure-borne sources](/phonometry/vibration/structural/)**

- [Mechanical mobility and the FRF family (ISO 7626-1)](/phonometry/vibration/structural/mechanical-mobility/):
  receptance, mobility and accelerance with their reciprocals, conversion
  between them, and the closed-form single-degree-of-freedom resonator.
- [Bending-wave transmission at plate junctions](/phonometry/vibration/structural/junction-transmission/):
  the wave-approach transmission coefficients for rigid X, T, L and in-line
  junctions, their diffuse-field average, and the coupling loss factor and
  vibration reduction index.
- [Transfer stiffness of resilient elements (ISO 10846)](/phonometry/vibration/structural/transfer-stiffness/):
  the dynamic transfer stiffness and loss factor of an isolator by the direct
  and indirect methods.
- [Sound power from surface vibration (ISO/TS 7849)](/phonometry/devices/emission/vibration-sound-power/):
  the radiated power from the surface-averaged velocity level and the radiation
  factor, with the Part 1 upper limit and the Part 2 engineering value. It lives
  under Sources and devices, and is repeated here because its input is a
  measured surface velocity.
- [Structure-borne sound power of equipment (EN 15657)](/phonometry/buildings/design/structure-borne-power/):
  the reception-plate method and the plate-independent source quantities,
  equivalent blocked force, free velocity and source mobility. It lives under
  Rooms and buildings, and is repeated here because the source quantities are
  mobilities.
- [Installed structure-borne sound (EN 12354-5)](/phonometry/buildings/design/installed-structure-borne/):
  the coupling term from source and receiver mobilities, the installed power
  and the per-path sound pressure level in the receiving room. It lives under
  Rooms and buildings, for the same reason.

**[Human vibration](/phonometry/vibration/human/)**

- [Human Vibration](/phonometry/vibration/human/human-vibration/): whole-body and
  hand-arm exposure with the ISO 8041-1 weightings, the weighted r.m.s. and
  dose measures, and the daily exposure $A(8)$.
- [Multiple-shock whole-body vibration (ISO 2631-5)](/phonometry/vibration/human/multiple-shock-vibration/):
  the seat-to-spine transfer function, the acceleration dose, and the
  cumulative stress variable behind the lumbar injury probability.

**[Machinery](/phonometry/vibration/machinery/)**

- [Evaluating machine vibration](/phonometry/vibration/machinery/machine-vibration-evaluation/):
  grading a machine from one broad-band measurement into the four evaluation
  zones of ISO 20816-1, with the frequency-shaped velocity criterion, the
  vector reading of a change that a magnitude comparison misses, and the
  printed boundaries and ALARM and TRIP settings ISO 10816-3 gives for
  industrial machines.
- [Machine fault frequencies](/phonometry/vibration/machinery/machine-diagnostics/):
  the characteristic bearing, gear and shaft frequencies and the envelope
  analysis that finds them under the broadband noise of a running machine.
  Also listed under Signal analysis, beside the spectral estimators it uses.

## [Environment and transport](/phonometry/environment/)

Outdoor propagation, barriers, refraction, road, rail and wind-turbine
sources, and the assessment built on them. Everything here concerns sound that
has to travel a long way before it is assessed, so the atmosphere, the ground
and the source's own motion all enter the answer. Implements ISO 9613-1/-2,
ISO 1996-1/-2, ISO/PAS 1996-3, NT ACOU 112, CNOSSOS-EU (2002/49/EC Annex II)
and IEC 61400-11.

One scope boundary is worth stating here rather than one click away. Of
CNOSSOS-EU, what is implemented is the **source** side of Annex II: the road
emission of section 2.2 with the Appendix F coefficients, and the railway
emission of section 2.3 with Appendix G, which give the directional sound power
per metre of source line. The propagation calculation of section 2.5, with its
own ground, diffraction and favourable-conditions machinery, is **not**
implemented; outdoor attenuation here goes through the ISO 9613-2 chain
instead, which is a different model and not interchangeable with it for
regulatory mapping.

**[Outdoor sound](/phonometry/environment/propagation/)**

- [Outdoor Sound Propagation](/phonometry/environment/propagation/outdoor-propagation/):
  atmospheric absorption and the ISO 9613-2 general method, with a per-term
  octave-band attenuation breakdown.
- [Spherical ground effect and advanced barriers](/phonometry/environment/propagation/ground-barriers/):
  the Weyl-Van der Pol reflection coefficient over finite-impedance ground, and
  wave-theoretic barrier diffraction.
- [Atmospheric refraction: rays and the GFPE](/phonometry/environment/propagation/atmospheric-refraction/):
  effective sound-speed profiles, curved rays with a closed-form shadow-zone
  distance, and the GFPE relative-level field.

**[Sources](/phonometry/environment/sources/)**

- [CNOSSOS-EU road traffic source emission](/phonometry/environment/sources/cnossos-road-emission/):
  the common EU road source of Annex II 2.2: rolling and propulsion sound power
  per vehicle category and the directional power per metre of source line.
- [CNOSSOS-EU railway source emission](/phonometry/environment/sources/cnossos-rail-emission/):
  the rail source of Annex II 2.3: roughness and the contact filter, impact
  noise, curve squeal, traction and aerodynamic noise, and the two equivalent
  source lines.
- [Wind-turbine noise: sound power and tonal audibility](/phonometry/environment/sources/wind-turbine-noise/):
  the apparent sound power level referred to the rotor centre, and the tonal
  audibility chain that decides whether a tone is audible.

**[Assessment and regulation](/phonometry/environment/assessment/)**

- [Impulsive-sound prominence (NT ACOU 112)](/phonometry/environment/assessment/impulsive-sound/):
  the predicted prominence of each impulse from its onset rate and level
  difference, and the adjustment added to $L_\mathrm{Aeq}$.

- [Environmental Levels (ISO 1996-1/-2)](/phonometry/environment/assessment/environmental-levels/):
  $L_\mathrm{den}$, $L_\mathrm{dn}$ and the composite rating levels, the tonal adjustment, the
  residual-noise correction and the uncertainty budget. Also listed under
  Signal analysis, beside the level definitions it builds on.
- [Spanish Noise Regulation (RD 1367/2007)](/phonometry/environment/assessment/spanish-noise-regulation/):
  the corrected level $L_\mathrm{Keq}$ with its $K_\mathrm{t}$, $K_\mathrm{f}$ and $K_\mathrm{i}$ corrections, the
  evaluation periods and noise phases, the limit tables and the Article 25
  compliance check. Also listed under Signal analysis, for the same reason.

## [Aircraft noise](/phonometry/aircraft/)

Certification levels, airport contours and the rotorcraft hemisphere
method: the noise of flight measured the way the certification and
airport-planning documents prescribe. Implements ICAO Annex 16, IEC 61265,
SAE ARP 866B/5534 and ECAC Doc 29/32.

- [Aircraft noise: Effective Perceived Noise Level](/phonometry/aircraft/aircraft-noise/):
  perceived noisiness and PNL, the tone correction, the duration correction and
  the measurement-system verifier.
- [Airport Noise (ECAC Doc 29)](/phonometry/aircraft/airport-noise/): the
  noise-power-distance engine, the per-segment single-event chain and the
  ground-grid SEL contour.
- [Rotorcraft noise: the hemisphere method](/phonometry/aircraft/rotorcraft-noise/):
  the hemisphere source model with its propagation adjustments, flight-condition
  interpolation, and the single-event contours.
- [The ANP fleet database](/phonometry/aircraft/anp-fleet/): the EASA tables of
  noise-power-distance curves and default trajectories, and the Doc 29 chain run
  from an aircraft identifier.

## [Underwater acoustics](/phonometry/underwater/)

Levels referenced to 1 micropascal, ship radiated noise, pile driving, ambient
noise and propagation loss. The reference quantities differ from the airborne
ones, so this is the one area where a level cannot be read across without
conversion. Implements ISO 18405, ISO 17208-1/-2, ISO 18406 and
JOMOPANS-ECHO.

- [Underwater acoustics: radiated noise and pile driving](/phonometry/underwater/underwater-acoustics/):
  the ISO 18405 reference levels, the ship radiated noise level and equivalent
  monopole source level, and single-strike and cumulative pile-driving
  exposure.
- [Underwater sound propagation](/phonometry/underwater/underwater-propagation/):
  spreading plus volume absorption, the speed of sound in sea water, the sonar
  equation, seabed reflection loss and the ambient-noise spectrum.
- [Underwater propagation solvers](/phonometry/underwater/underwater-solvers/):
  the normal-mode, ray-tracing, Gaussian beam and parabolic-equation solvers
  of the stratified waveguide, and how to choose a propagation model.
- [Marine-mammal noise exposure](/phonometry/underwater/marine-mammal-exposure/):
  the hearing side of that noise: the group audiograms, the regulatory weighting
  functions with their guidance version, and the exposure of a pile-driving
  campaign against the injury criteria.

## [Sources and devices](/phonometry/devices/)

Sound power, intensity, emission declarations, electroacoustics and programme
loudness. What a source emits rather than what a receiver gets, plus the
electroacoustic chain that reproduces or measures it. Implements ISO 3741,
ISO 3744/3746, ISO 3745, ISO 9614-1/-2/-3, IEC 61043, ISO 4871,
IEC 60268-3/-4/-5, ITU-R BS.1770-5 and EBU R 128.

**[Sound power and intensity](/phonometry/devices/emission/)**

- [Sound Intensity (p-p)](/phonometry/devices/emission/intensity/): two-microphone
  intensity with the field indicators that qualify the measurement.
- [Sound Power](/phonometry/devices/emission/sound-power/): choosing the determination
  method and declaring the noise emission per ISO 4871.
- [Sound Power by Pressure Methods](/phonometry/devices/emission/sound-power-pressure/):
  the enveloping surface of ISO 3744/3746 and the precision anechoic grade of
  ISO 3745.
- [Sound Power in the Reverberation Room](/phonometry/devices/emission/sound-power-reverberation/):
  the direct and comparison methods of ISO 3741.
- [Sound Power in Situ by Comparison](/phonometry/devices/emission/sound-power-in-situ/):
  the ISO 3747 comparison against a reference sound source where the machine
  works, with the sound energy level of an impulsive source.

- [Sound Power in a Duct](/phonometry/devices/emission/sound-power-in-duct/):
  the ISO 5136 in-duct method for fans, with the sampling-tube flow and modal
  correction of Annex A and the plane-wave relation of clause 8.
- [Sound Power by Intensity Scanning](/phonometry/devices/emission/sound-power-intensity/):
  the on-site scanning of ISO 9614-2 and the ISO 9614-3 precision grade.
- [Sound power from surface vibration (ISO/TS 7849)](/phonometry/devices/emission/vibration-sound-power/):
  the surface-velocity route, for the machine that cannot be moved into a
  qualified room: the radiated power from the surface-averaged velocity level
  and the radiation factor, with the Part 1 upper limit and the Part 2
  engineering value. Also listed under Vibration.

**[Electroacoustics](/phonometry/devices/electroacoustics/)**

- [Electroacoustics: distortion and frequency response](/phonometry/devices/electroacoustics/electroacoustics/):
  harmonic and intermodulation distortion, THD+N and SINAD, dynamic range and
  the $H_1$/$H_2$ frequency-response estimators.
- [Loudspeaker Characterisation (IEC 60268-5)](/phonometry/devices/electroacoustics/loudspeakers/):
  the sensitivity conventions, the radiating piston and the characteristics
  fiche.
- [Microphone Characterisation (IEC 60268-4)](/phonometry/devices/electroacoustics/microphones/):
  the sensitivity references, the directional patterns and the inherent noise.
- [Swept-sine distortion and phase utilities](/phonometry/devices/electroacoustics/swept-sine-distortion/):
  harmonic separation from one exponential sweep, THD against excitation
  frequency, and minimum phase, group delay and excess phase.
**[Broadcast](/phonometry/devices/broadcast/)**

- [Programme loudness and true peak](/phonometry/devices/broadcast/program-loudness/):
  K-weighting and gated integrated loudness in LUFS, the momentary and
  short-term meters, the loudness range and the true-peak level. The sidebar
  files it inside Electroacoustics; it has its own section overview.
- [Quasi-peak programme meter](/phonometry/devices/broadcast/quasi-peak/): the
  ITU-R BS.468-4 psophometric noise meter, whose clause 2 prints no time
  constant at all and specifies the detector through eleven tone-burst
  acceptance windows, the 0.775 V calibration that makes a reading dBqps, and
  three fitted time scales the windows pin only to within a factor of three.

**[Noise control](/phonometry/devices/noise-control/)**

- [Silencers](/phonometry/devices/noise-control/silencers/): reactive silencers by the
  four-pole transmission-matrix method and the reactive-versus-dissipative
  choice.
- [Duct-Borne Noise: Fan to Room](/phonometry/devices/noise-control/duct-path/): the
  end-to-end fan-to-room calculation against a room criterion, and the
  higher-order-mode cut-on that limits every plane-wave method.
- [HVAC Noise the German Way (VDI 2081)](/phonometry/devices/noise-control/vdi2081-air-systems/):
  the same chain by the German guideline, from the assembly-type fan model to
  the room step, against the worked sheet of its own Part 2.
- [Room to Room: Partition, Receiving Room, Criterion](/phonometry/devices/noise-control/room-to-room/):
  the source-room level through a partition into the receiving room, the noise
  criterion verdict and the transmission loss a partition or an enclosure needs.
- [Industrial Noise Control: HVAC and Enclosures](/phonometry/devices/noise-control/noise-control/):
  duct attenuation and flow noise, and enclosure insertion loss.

## [Wave simulation](/phonometry/simulation/)

Deterministic 2D finite-difference time-domain solvers, acoustic and elastic
P-SV, validated against analytic oracles rather than a standard. It is the one
area with no governing document, so its evidence is the closed-form solution it
reproduces.

- [2D FDTD wave simulation](/phonometry/simulation/fdtd-simulation/): a staggered
  pressure-velocity grid with Gaussian, tone and arbitrary-signal sources,
  rasterised obstacles, rigid, impedance and absorbing boundaries, and a frozen
  result carrying probe histories and field snapshots.
- [Elastic waves and fluid-solid coupling](/phonometry/simulation/elastic-waves/):
  the P-SV companion solver on the same grid, with Rayleigh waves on free
  surfaces, mode conversion, Scholte interface waves and immersed-plate
  transmission.
