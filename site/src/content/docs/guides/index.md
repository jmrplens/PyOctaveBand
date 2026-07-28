---
title: "Guides"
description: "The 89 guides of phonometry, grouped into the nine areas the library covers: what each area is for, the standards implemented in it, and a one-line summary of every guide inside it."
head:
  - tag: script
    attrs:
      type: application/ld+json
    content: |
      {
        "@context": "https://schema.org",
        "@type": "ItemList",
        "@id": "https://jmrplens.github.io/phonometry/guides/#areas",
        "name": "phonometry documentation areas",
        "description": "The nine documented areas of the phonometry guides, each with the standards it implements.",
        "inLanguage": "en",
        "numberOfItems": 9,
        "itemListOrder": "https://schema.org/ItemListUnordered",
        "itemListElement": [
          {
            "@type": "ListItem",
            "position": 1,
            "name": "Core signal analysis",
            "description": "Filter banks, weighting, levels, spectra, calibration and uncertainty.",
            "url": "https://jmrplens.github.io/phonometry/guides/sections/core-signal-analysis/"
          },
          {
            "@type": "ListItem",
            "position": 2,
            "name": "Hearing and perception",
            "description": "Loudness, sound quality, speech intelligibility, hearing and exposure.",
            "url": "https://jmrplens.github.io/phonometry/guides/sections/hearing-perception/"
          },
          {
            "@type": "ListItem",
            "position": 3,
            "name": "Rooms and buildings",
            "description": "Room parameters, background noise, field and laboratory insulation, prediction.",
            "url": "https://jmrplens.github.io/phonometry/guides/sections/rooms-buildings/"
          },
          {
            "@type": "ListItem",
            "position": 4,
            "name": "Materials and surfaces",
            "description": "Absorption, airflow resistance, impedance tube, porous and metamaterial models, diffusers, scattering.",
            "url": "https://jmrplens.github.io/phonometry/guides/sections/materials-surfaces/"
          },
          {
            "@type": "ListItem",
            "position": 5,
            "name": "Vibration and structure-borne sound",
            "description": "Mobility and FRFs, isolators, radiated power, junctions, human vibration.",
            "url": "https://jmrplens.github.io/phonometry/guides/sections/vibration/"
          },
          {
            "@type": "ListItem",
            "position": 6,
            "name": "Environment and transport",
            "description": "Outdoor propagation, barriers, refraction, aircraft, rotorcraft and wind turbines.",
            "url": "https://jmrplens.github.io/phonometry/guides/sections/environment-transport/"
          },
          {
            "@type": "ListItem",
            "position": 7,
            "name": "Underwater acoustics",
            "description": "Levels re 1 microPa, ship radiated noise, pile driving, ambient noise, transmission loss.",
            "url": "https://jmrplens.github.io/phonometry/guides/sections/underwater/"
          },
          {
            "@type": "ListItem",
            "position": 8,
            "name": "Sources and devices",
            "description": "Sound power, intensity, emission declarations, electroacoustics, programme loudness.",
            "url": "https://jmrplens.github.io/phonometry/guides/sections/sources-devices/"
          },
          {
            "@type": "ListItem",
            "position": 9,
            "name": "Wave simulation",
            "description": "A deterministic 2D FDTD solver, validated against analytic oracles rather than a standard.",
            "url": "https://jmrplens.github.io/phonometry/guides/sections/simulation/"
          }
        ]
      }
---

Every guide on this site follows the same shape: the standard it implements,
the quantities that standard defines, the assumptions the implementation
makes, then runnable code and the figure it draws. Nothing here is a survey of
the field; each page is the working documentation of a module, written so that
a result can be defended clause by clause rather than trusted.

This page is the map. Eighty-nine guides sit in nine areas, and each area has its
own overview page with the longer story of how its pieces fit together. If you
are arriving without a specific question, read
[Getting Started](/phonometry/getting-started/) first: it runs one signal
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

## [Core signal analysis](/phonometry/guides/sections/core-signal-analysis/)

Filter banks, weighting, levels, spectra, calibration and uncertainty. This is
the chain that turns a digital signal into a standards-compliant number, and
every other area consumes it: a loudness model needs calibrated band levels, a
room parameter needs a filtered impulse response, an environmental rating is an
adjusted Leq. Implements IEC 61260-1, ANSI S1.11, IEC 61672-1, ISO 7196,
IEC 61252, ISO 1996-1, IEC 60942, ISO 18233 and the GUM.

- [Build a sound level meter](/phonometry/guides/sound-level-meter/):
  the whole area assembled end to end on one runnable page, from the
  calibrator tone to the reported levels.

**[Octave filtering](/phonometry/guides/sections/octave-filtering/)**

- [Filter Banks](/phonometry/guides/filter-banks/): the five filter
  architectures, their frequency responses, band decomposition and zero-phase
  offline filtering.
- [Filter Class Verification (IEC 61260-1)](/phonometry/guides/filter-compliance/):
  the Table 1 acceptance mask band by band, the class 0 of the withdrawn 1995
  edition and the compliance fiche.
- [Block Processing](/phonometry/guides/block-processing/): stateful streaming
  analysis that carries filter state across buffers, for signals that never fit
  in memory.
- [Multichannel and Performance](/phonometry/guides/multichannel/): vectorized
  analysis of many channels at once, with the (channels, samples) convention
  and performance notes.

**[Levels and weighting](/phonometry/guides/sections/levels-weighting/)**

- [Frequency Weighting (A, B, C, D, G, AU, Z)](/phonometry/guides/weighting/):
  the IEC 61672-1 ear-response curves, ISO 7196 G-weighting for infrasound and
  the historical B and D curves.
- [Time Weighting](/phonometry/guides/time-weighting/): the Fast, Slow and
  Impulse exponential ballistics of IEC 61672-1.
- [Integrated and Statistical Levels](/phonometry/guides/levels/): Leq and
  LAeq, the percentile levels L10/L50/L90, LCpeak and SEL, noise dose,
  Lden and the ISO 1996-1 rating levels.

**[Signals and spectra](/phonometry/guides/sections/signals-spectra/)**

- [Calibrated spectral analysis](/phonometry/guides/spectral-analysis/): the
  Welch power and cross-spectral estimators with their random errors,
  chi-square confidence intervals and fractional-octave smoothing.
- [Multiple and partial coherence](/phonometry/guides/miso-coherence/): the
  ordinary, multiple and partial coherence of several correlated sources
  driving one response, and which source dominates each band.
- [Time-frequency analysis](/phonometry/guides/time-frequency/): the calibrated
  STFT spectrogram in absolute dB SPL, and the zoom FFT that resolves tones
  closer than a practical FFT bin.
- [Cepstrum, echoes and the envelope spectrum](/phonometry/guides/cepstrum-echoes/):
  quefrency analysis, echo detection with the reflection coefficient read off
  the cepstral peak, liftering and the envelope spectrum.
- [Time synchronous averaging](/phonometry/guides/synchronous-averaging/):
  extraction of a periodic waveform of known period, the comb filter that
  describes it and the choice of the number of averages.
- [Correlation, time delay and envelope](/phonometry/guides/correlation-delay/):
  correlation with its random errors, time-delay estimation by direct
  correlation and the GCC weightings, and the Hilbert envelope.
- [Test signals and sample-rate tools](/phonometry/guides/test-signals/):
  IEC 60268-1 tone bursts with exact gating, colored noise with an exact
  slope, resampling with a stated anti-alias specification and fractional
  delay.
- [System measurement](/phonometry/guides/system-measurement/): complementary
  Golay pairs, sweeps shaped to an arbitrary target magnitude spectrum, and
  Kirkeby-regularized inversion of a measured response.

**[Calibration and uncertainty](/phonometry/guides/sections/calibration-uncertainty/)**

- [Calibration and dBFS](/phonometry/guides/calibration/): physical SPL
  calibration from a calibrator tone or a known sensitivity, and the digital
  full-scale mode.
- [Measurement uncertainty (GUM and Monte Carlo)](/phonometry/guides/gum-uncertainty/):
  the law of propagation of uncertainty and the Monte Carlo method, with
  expanded uncertainty and coverage intervals.
- [Data qualification](/phonometry/guides/data-qualification/): the reverse
  arrangement and runs tests for stationarity, and the Rice level-crossing and
  peak statistics with the irregularity factor.

## [Hearing and perception](/phonometry/guides/sections/hearing-perception/)

Loudness, sound quality, speech intelligibility, hearing and exposure. Where
the core area asks how much sound there is, this one asks what a listener makes
of it: how loud it seems, how sharp or rough or annoying, how much of a talker
survives the room, and how much hearing a working life in that noise costs.
Implements ISO 532-1/-2/-3, ECMA-418-1/-2, ISO 226, DIN 45692, IEC 60268-16,
ANSI S3.5, ISO 7029, ISO 1999 and ISO 9612.

**[Psychoacoustics](/phonometry/guides/sections/psychoacoustics/)**

- [Loudness](/phonometry/guides/loudness/): loudness in sones by Zwicker,
  Moore-Glasberg, the time-varying ISO 532-3 model and the ECMA-418-2 Sottek
  Hearing Model, plus the ISO 226 equal-loudness contours.
- [Sound Quality Metrics](/phonometry/guides/sound-quality/): sharpness in
  acum, and the ECMA-418-2 tonality, roughness and fluctuation strength.
- [Prominent Discrete Tones (ECMA-418-1)](/phonometry/guides/tone-prominence/):
  the tone-to-noise ratio and prominence ratio, with their frequency-dependent
  prominence criteria.
- [Objective audibility of tones in noise (ISO/PAS 20065)](/phonometry/guides/tone-audibility/):
  the critical-band masking level, the masking index and the audibility of a
  tone above the masking threshold.
- [Psychoacoustic annoyance and fluctuation strength](/phonometry/guides/psychoacoustic-annoyance/):
  the Fastl and Zwicker annoyance model built from loudness, sharpness,
  roughness and fluctuation strength.

**[Speech](/phonometry/guides/sections/speech/)**

- [Speech Transmission Index (STI)](/phonometry/guides/speech-transmission/):
  the modulation transfer function, the indirect method from an impulse
  response and the direct STIPA measurement.
- [Speech Intelligibility Index](/phonometry/guides/speech-intelligibility/):
  the one-third-octave SII with its band-importance function, self-masking and
  upward spread of masking.
- [Objective Intelligibility (STOI and ESTOI)](/phonometry/guides/objective-intelligibility/):
  the two correlation-based measures for time-frequency weighted noisy speech.

**[Hearing and exposure](/phonometry/guides/sections/hearing-exposure/)**

- [Hearing threshold (age and reference zero)](/phonometry/guides/hearing-threshold/):
  the age-related threshold distribution of ISO 7029 and the ISO 389-7
  reference threshold of hearing.
- [Noise-induced hearing loss (ISO 1999)](/phonometry/guides/noise-induced-hearing-loss/):
  the permanent threshold shift as a function of level, duration and frequency,
  combined with the age component.
- [Occupational Noise Exposure (ISO 9612)](/phonometry/guides/occupational-exposure/):
  the task-based, job-based and full-day strategies for LEX,8h, with the
  uncertainty budget and the upper limit.

## [Rooms and buildings](/phonometry/guides/sections/rooms-buildings/)

Room parameters, background noise, field and laboratory insulation, and
prediction from element data. Two questions run through the area: how a room
treats the sound made inside it, and how much of the sound made next door gets
through. Implements ISO 3382-1/-2/-3, ISO 16283-1/-2/-3, ISO 10140,
ISO 717-1/-2, EN 12354-1 to -6, ISO 12999-1, ISO 10052 and ANSI/ASA S12.2.

**[Room acoustics](/phonometry/guides/sections/room-acoustics/)**

- [Measuring the Room Impulse Response](/phonometry/guides/room-impulse-response/):
  the ISO 18233 deterministic acquisition, exponential sweeps with their
  deconvolution, and MLS.
- [Room Acoustics](/phonometry/guides/room-acoustics/): the room parameters
  EDT, T20, T30, C50, C80, D50 and Ts derived from that impulse response.
- [Open-Plan Office Acoustics (ISO 3382-3)](/phonometry/guides/open-plan-acoustics/):
  the spatial decay rate of speech and the distraction and privacy distances
  of an open-plan floor.
- [Image sources and the steady-state room field](/phonometry/guides/room-image-sources/):
  the deterministic image-source impulse response of a rectangular room, the
  room constant, critical distance and Schroeder frequency.
- [Room-noise criteria (NC / RC Mark II)](/phonometry/guides/room-noise/): the
  ANSI/ASA S12.2 Noise Criteria rating by tangency, and the Room Criteria
  Mark II rating with its rumble, hiss or neutral tag.
- [Reverberation-time prediction (Sabine, Eyring, Arau)](/phonometry/guides/reverberation-prediction/):
  five statistical-acoustics models from volume, boundary areas and surface
  absorption, including the air term.
- [Sound absorption in enclosed spaces (EN 12354-6)](/phonometry/guides/enclosed-space-absorption/):
  the total equivalent absorption area of a room from its surfaces, objects and
  air, and the reverberation time that follows.

**[Sound insulation](/phonometry/guides/sections/sound-insulation/)**

- [Field Insulation Measurement (ISO 16283)](/phonometry/guides/insulation-field/):
  airborne and impact insulation measured in the building, its test report and
  the ISO 12999-1 uncertainty that qualifies it.
- [Laboratory Insulation Measurement](/phonometry/guides/insulation-lab/):
  the ISO 10140 characterisation of an element with flanking suppressed.
- [Sound Insulation by Intensity (ISO 15186)](/phonometry/guides/insulation-intensity/):
  the transmitted power read off the radiating face, for the whole element or
  element by element.
- [Sound Insulation Survey Method (ISO 10052)](/phonometry/guides/insulation-survey/):
  the octave-band control method with its reverberation index and its
  airborne, impact, façade and service-equipment quantities.
- [Laboratory Flanking Transmission (ISO 10848)](/phonometry/guides/flanking-lab/):
  the junction vibration reduction index and the flanking level differences
  measured on a test facility.
- [Insulation Ratings (ISO 717)](/phonometry/guides/insulation-ratings/):
  the airborne and impact reference-curve engines with C, Ctr and CI, the
  enlarged-range terms and the ISO 717 fiche.
- [Façade Sound Insulation](/phonometry/guides/facade-insulation/): the
  building envelope measured per ISO 16283-3, predicted per EN 12354-3 and
  radiating outwards per EN 12354-4.

**[Insulation design](/phonometry/guides/sections/insulation-design/)**

- [Predicting Sound Insulation (EN 12354)](/phonometry/guides/insulation-prediction/):
  in-situ airborne and impact insulation between rooms from element data, with
  their flanking paths.
- [Predicting Panel Sound Insulation](/phonometry/guides/panel-sound-insulation/):
  the mass law and coincidence dip, double walls, slits and apertures, plate
  radiation efficiency and point mobilities.
- [Floor-Covering Impact Improvement (ISO 16251-1)](/phonometry/guides/impact-improvement/):
  the weighted improvement of a soft floor covering measured on a small
  heavyweight mock-up.
- [Dynamic stiffness of resilient materials (EN 29052-1)](/phonometry/guides/dynamic-stiffness/):
  the stiffness per unit area under a floating floor from the load-plate
  resonance, with the enclosed-gas term.

## [Materials and surfaces](/phonometry/guides/sections/materials-surfaces/)

Absorption, airflow resistance, the impedance tube, porous and metamaterial
models, diffusers and scattering. What a surface does to the sound that
reaches it, measured in a laboratory or predicted from the material
parameters. Implements ISO 354, ISO 11654, ISO 10534-2, ISO 9053,
ISO 17497-1/-2, ISO 13472 and EN 29052.

- [Sound Absorption Measurement and Rating](/phonometry/guides/absorption-measurement/):
  the ISO 354 reverberation-room measurement, the weighted rating and its
  class, and the measurement uncertainty of both.
- [Airflow Resistance](/phonometry/guides/airflow-resistance/): the static
  and alternating determination of airflow resistance and resistivity.
- [Impedance Tube](/phonometry/guides/impedance-tube/): the normal-incidence
  surface impedance, absorption and transmission loss, plus the virtual
  FDTD tube.
- [Porous and Multilayer Absorbers](/phonometry/guides/porous-absorbers/): the
  Delany-Bazley, Miki and Johnson-Champoux-Allard models, the transfer-matrix
  multilayer solver with perforated, microperforated and membrane layers, and
  the random-incidence integral.
- [Metamaterial Absorbers](/phonometry/guides/metamaterial-absorbers/): the
  critical-coupling condition for perfect absorption and the slow-sound slit
  panel loaded by Helmholtz resonators, with its design solver.
- [Diffusers and Their Coefficients](/phonometry/guides/diffusers/): the
  random-incidence scattering coefficient, the autocorrelation diffusion
  coefficient, and Schroeder design with its far-field prediction.
- [Metadiffusers](/phonometry/guides/metadiffusers/): deep-subwavelength
  Schroeder diffusers from resonator-loaded slits, slow sound and ternary
  sequences.
- [In-situ Road-Surface Absorption](/phonometry/guides/road-absorption/):
  in-situ road-surface absorption by the subtraction technique and the spot
  method.

## [Vibration and structure-borne sound](/phonometry/guides/sections/vibration/)

Mobility and frequency-response functions, isolators, radiated power,
junctions and human vibration. The area covers the path a machine takes into a
structure and out again as airborne sound, and the separate question of what
vibration does to the person exposed to it. Implements ISO 7626, ISO 10846,
ISO/TS 7849, EN 15657, EN 12354-5, ISO 2631-1/-5, ISO 5349 and ISO 8041.

**[Structure-borne sources](/phonometry/guides/sections/structure-borne/)**

- [Mechanical mobility and the FRF family (ISO 7626-1)](/phonometry/guides/mechanical-mobility/):
  receptance, mobility and accelerance with their reciprocals, conversion
  between them, and the closed-form single-degree-of-freedom resonator.
- [Bending-wave transmission at plate junctions](/phonometry/guides/junction-transmission/):
  the wave-approach transmission coefficients for rigid X, T, L and in-line
  junctions, their diffuse-field average, and the coupling loss factor and
  vibration reduction index.
- [Transfer stiffness of resilient elements (ISO 10846)](/phonometry/guides/transfer-stiffness/):
  the dynamic transfer stiffness and loss factor of an isolator by the direct
  and indirect methods.
- [Sound power from surface vibration (ISO/TS 7849)](/phonometry/guides/vibration-sound-power/):
  the radiated power from the surface-averaged velocity level and the radiation
  factor, with the Part 1 upper limit and the Part 2 engineering value.
- [Structure-borne sound power of equipment (EN 15657)](/phonometry/guides/structure-borne-power/):
  the reception-plate method and the plate-independent source quantities,
  equivalent blocked force, free velocity and source mobility.
- [Installed structure-borne sound (EN 12354-5)](/phonometry/guides/installed-structure-borne/):
  the coupling term from source and receiver mobilities, the installed power
  and the per-path sound pressure level in the receiving room.

**[Human vibration](/phonometry/guides/sections/human-vibration/)**

- [Human Vibration](/phonometry/guides/human-vibration/): whole-body and
  hand-arm exposure with the ISO 8041-1 weightings, the weighted r.m.s. and
  dose measures, and the daily exposure A(8).
- [Multiple-shock whole-body vibration (ISO 2631-5)](/phonometry/guides/multiple-shock-vibration/):
  the seat-to-spine transfer function, the acceleration dose, and the
  cumulative stress variable behind the lumbar injury probability.

## [Environment and transport](/phonometry/guides/sections/environment-transport/)

Outdoor propagation, barriers, refraction, aircraft, rotorcraft and wind
turbines. Everything here concerns sound that has to travel a long way before
it is assessed, so the atmosphere, the ground and the source's own motion all
enter the answer. Implements ISO 9613-1/-2, ISO 1996-1/-2, ISO/PAS 1996-3,
NT ACOU 112, ICAO Annex 16, IEC 61265, SAE ARP 5534, ECAC Doc 29/32 and
IEC 61400-11.

**[Outdoor sound](/phonometry/guides/sections/outdoor-sound/)**

- [Outdoor Sound Propagation](/phonometry/guides/outdoor-propagation/):
  atmospheric absorption and the ISO 9613-2 general method, with a per-term
  octave-band attenuation breakdown.
- [Spherical ground effect and advanced barriers](/phonometry/guides/ground-barriers/):
  the Weyl-Van der Pol reflection coefficient over finite-impedance ground, and
  wave-theoretic barrier diffraction.
- [Atmospheric refraction: rays and the GFPE](/phonometry/guides/atmospheric-refraction/):
  effective sound-speed profiles, curved rays with a closed-form shadow-zone
  distance, and the GFPE relative-level field.
- [Impulsive-sound prominence (NT ACOU 112)](/phonometry/guides/impulse-prominence/):
  the predicted prominence of each impulse from its onset rate and level
  difference, and the adjustment added to LAeq.

**[Aircraft and wind energy](/phonometry/guides/sections/aircraft-wind/)**

- [Aircraft noise: Effective Perceived Noise Level](/phonometry/guides/aircraft-noise/):
  perceived noisiness and PNL, the tone correction, the duration correction and
  the measurement-system verifier.
- [Airport Noise (ECAC Doc 29)](/phonometry/guides/airport-noise/): the
  noise-power-distance engine, the per-segment single-event chain and the
  ground-grid SEL contour.
- [Rotorcraft noise: the hemisphere method](/phonometry/guides/rotorcraft-noise/):
  the hemisphere source model with its propagation adjustments, flight-condition
  interpolation, and the single-event contours.
- [Wind-turbine noise: sound power and tonal audibility](/phonometry/guides/wind-turbine-noise/):
  the apparent sound power level referred to the rotor centre, and the tonal
  audibility chain that decides whether a tone is audible.

## [Underwater acoustics](/phonometry/guides/sections/underwater/)

Levels referenced to 1 micropascal, ship radiated noise, pile driving, ambient
noise and transmission loss. The reference quantities differ from the airborne
ones, so this is the one area where a level cannot be read across without
conversion. Implements ISO 18405, ISO 17208-1/-2, ISO 18406 and
JOMOPANS-ECHO.

- [Underwater acoustics: radiated noise and pile driving](/phonometry/guides/underwater-acoustics/):
  the ISO 18405 reference levels, the ship radiated noise level and equivalent
  monopole source level, and single-strike and cumulative pile-driving
  exposure.
- [Underwater sound propagation](/phonometry/guides/underwater-propagation/):
  spreading plus volume absorption, the speed of sound in sea water, the sonar
  equation, seabed reflection loss and the ambient-noise spectrum.
- [Underwater propagation solvers](/phonometry/guides/underwater-solvers/):
  the normal-mode, ray-tracing and parabolic-equation solvers of the
  stratified waveguide, and how to choose a propagation model.

## [Sources and devices](/phonometry/guides/sections/sources-devices/)

Sound power, intensity, emission declarations, electroacoustics and programme
loudness. What a source emits rather than what a receiver gets, plus the
electroacoustic chain that reproduces or measures it. Implements ISO 3741,
ISO 3744/3746, ISO 3745, ISO 9614-1/-2/-3, IEC 61043, ISO 4871,
IEC 60268-3/-4/-5, ITU-R BS.1770-5 and EBU R 128.

- [Sound Intensity (p-p)](/phonometry/guides/intensity/): two-microphone
  intensity with the field indicators that qualify the measurement.
- [Sound Power](/phonometry/guides/sound-power/): choosing the determination
  method and declaring the noise emission per ISO 4871.
- [Sound Power by Pressure Methods](/phonometry/guides/sound-power-pressure/):
  the enveloping surface of ISO 3744/3746 and the precision anechoic grade of
  ISO 3745.
- [Sound Power in the Reverberation Room](/phonometry/guides/sound-power-reverberation/):
  the direct and comparison methods of ISO 3741.
- [Sound Power by Intensity Scanning](/phonometry/guides/sound-power-intensity/):
  the on-site scanning of ISO 9614-2 and the ISO 9614-3 precision grade.
- [Electroacoustics: distortion and frequency response](/phonometry/guides/electroacoustics/):
  harmonic and intermodulation distortion, THD+N and SINAD, dynamic range and
  the H1/H2 frequency-response estimators.
- [Loudspeaker Characterisation (IEC 60268-5)](/phonometry/guides/loudspeakers/):
  the sensitivity conventions, the radiating piston and the characteristics
  fiche.
- [Microphone Characterisation (IEC 60268-4)](/phonometry/guides/microphones/):
  the sensitivity references, the directional patterns and the inherent noise.
- [Swept-sine distortion and phase utilities](/phonometry/guides/swept-sine-distortion/):
  harmonic separation from one exponential sweep, THD against excitation
  frequency, and minimum phase, group delay and excess phase.
- [Silencers](/phonometry/guides/silencers/): reactive silencers by the
  four-pole transmission-matrix method and the reactive-versus-dissipative
  choice.
- [Industrial Noise Control: HVAC and Enclosures](/phonometry/guides/noise-control/):
  duct attenuation and flow noise, and enclosure insertion loss.
- [Programme loudness and true peak](/phonometry/guides/program-loudness/):
  K-weighting and gated integrated loudness in LUFS, the momentary and
  short-term meters, the loudness range and the true-peak level.

## [Wave simulation](/phonometry/guides/sections/simulation/)

Deterministic 2D finite-difference time-domain solvers, acoustic and elastic
P-SV, validated against analytic oracles rather than a standard. It is the one
area with no governing document, so its evidence is the closed-form solution it
reproduces.

- [2D FDTD wave simulation](/phonometry/guides/fdtd-simulation/): a staggered
  pressure-velocity grid with Gaussian, tone and arbitrary-signal sources,
  rasterised obstacles, rigid, impedance and absorbing boundaries, and a frozen
  result carrying probe histories and field snapshots.
- [Elastic waves and fluid-solid coupling](/phonometry/guides/elastic-waves/):
  the P-SV companion solver on the same grid, with Rayleigh waves on free
  surfaces, mode conversion, Scholte interface waves and immersed-plate
  transmission.
