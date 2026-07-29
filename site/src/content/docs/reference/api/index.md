---
title: "API Reference"
description: "Every public function, class and constant in phonometry, generated from the source docstrings."
---

The complete public API, one page per module. Import the domain subpackage and call through it:

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

```python
from phonometry import metrology, underwater

spl, freq = metrology.octave_filter(x, fs)
snr = underwater.passive_sonar_equation(185.0, 60.0, 50.0)
```

Every documented name can also be imported directly from the top-level package (`from phonometry import leq`).

:::note
The API reference is generated from the English source docstrings and is published in English only.

La referencia de la API se genera a partir de los docstrings del código (en inglés) y se publica únicamente en inglés; las rutas en español muestran esta versión inglesa como alternativa.
:::

## Filters and frequencies

| Module | Summary |
| :--- | :--- |
| [`phonometry`](/phonometry/reference/api/filters/phonometry/) | Package-level names defined in `phonometry/__init__.py` itself. |
| [`metrology.core`](/phonometry/reference/api/filters/core/) | Core processing logic and FilterBank class for phonometry. |
| [`metrology.parametric_filters`](/phonometry/reference/api/filters/parametric-filters/) | Weighting filters (A, B, C, D, G, AU, Z) and time weighting utilities. |
| [`metrology.equalizer`](/phonometry/reference/api/filters/equalizer/) | Parametric equalizer biquads per the RBJ Audio EQ Cookbook. |
| [`metrology.frequencies`](/phonometry/reference/api/filters/frequencies/) | Frequency calculation logic according to ANSI/IEC standards. |
| [`metrology.compliance`](/phonometry/reference/api/filters/compliance/) | IEC 61260-1:2014 filter and IEC 61672-1:2013 weighting class verification. |

## Levels and calibration

| Module | Summary |
| :--- | :--- |
| [`metrology.levels`](/phonometry/reference/api/levels/levels/) | Integrated and statistical sound levels (Leq, LAeq, LN percentiles). |
| [`metrology.calibration`](/phonometry/reference/api/levels/calibration/) | Calibration utilities for mapping digital signals to physical SPL levels. |

## Psychoacoustics

| Module | Summary |
| :--- | :--- |
| [`psychoacoustics.loudness_zwicker`](/phonometry/reference/api/psychoacoustics/loudness-zwicker/) | Zwicker loudness for stationary and time-varying sounds per ISO 532-1:2017. |
| [`psychoacoustics.loudness_moore_glasberg`](/phonometry/reference/api/psychoacoustics/loudness-moore-glasberg/) | Stationary loudness per ISO 532-2:2017 (Moore-Glasberg method). |
| [`psychoacoustics.loudness_moore_glasberg_time`](/phonometry/reference/api/psychoacoustics/loudness-moore-glasberg-time/) | Time-varying loudness per ISO 532-3:2023 (Moore-Glasberg-Schlittenlacher). |
| [`psychoacoustics.loudness_ecma`](/phonometry/reference/api/psychoacoustics/loudness-ecma/) | Psychoacoustic loudness per ECMA-418-2:2025 (4th ed., Sottek Hearing Model). |
| [`psychoacoustics.loudness_contours`](/phonometry/reference/api/psychoacoustics/loudness-contours/) | Normal equal-loudness-level contours per ISO 226:2023. |
| [`psychoacoustics.sharpness`](/phonometry/reference/api/psychoacoustics/sharpness/) | Sharpness per DIN 45692:2009-08. |
| [`psychoacoustics.roughness_ecma`](/phonometry/reference/api/psychoacoustics/roughness-ecma/) | Psychoacoustic roughness per ECMA-418-2:2025 (4th ed., Sottek Hearing Model). |
| [`psychoacoustics.tonality`](/phonometry/reference/api/psychoacoustics/tonality/) | Prominent discrete tone assessment per ECMA-418-1:2024 (3rd edition). |
| [`psychoacoustics.tonality_ecma`](/phonometry/reference/api/psychoacoustics/tonality-ecma/) | Psychoacoustic tonality per ECMA-418-2:2025 (4th ed., Sottek Hearing Model). |
| [`psychoacoustics.tone_audibility`](/phonometry/reference/api/psychoacoustics/tone-audibility/) | Objective audibility of tones in noise -- engineering method (ISO/PAS 20065:2016). |
| [`psychoacoustics.fluctuation_strength`](/phonometry/reference/api/psychoacoustics/fluctuation-strength/) | Fluctuation strength after Fastl & Zwicker / Osses et al. |
| [`psychoacoustics.fluctuation_strength_ecma`](/phonometry/reference/api/psychoacoustics/fluctuation-strength-ecma/) | Psychoacoustic fluctuation strength per ECMA-418-2:2025 (4th ed., Clause 9). |
| [`psychoacoustics.psychoacoustic_annoyance`](/phonometry/reference/api/psychoacoustics/psychoacoustic-annoyance/) | Psychoacoustic annoyance (PA) after Fastl & Zwicker. |
| [`psychoacoustics.erb_scale`](/phonometry/reference/api/psychoacoustics/erb-scale/) | The ERB_N scale: auditory-filter bandwidth and the Cam frequency scale. |

## Speech

| Module | Summary |
| :--- | :--- |
| [`hearing.sti`](/phonometry/reference/api/speech/sti/) | Speech Transmission Index (STI) per IEC 60268-16:2020 (Edition 5). |
| [`hearing.sii`](/phonometry/reference/api/speech/sii/) | Speech Intelligibility Index (SII) per ANSI S3.5-1997 (R2017). |
| [`hearing.objective_intelligibility`](/phonometry/reference/api/speech/objective-intelligibility/) | Short-Time Objective Intelligibility (STOI and ESTOI). |

## Hearing and exposure

| Module | Summary |
| :--- | :--- |
| [`hearing.threshold`](/phonometry/reference/api/hearing/threshold/) | Age-related hearing threshold (ISO 7029:2017) and audiometric reference zero (ISO 389-7:2005). |
| [`hearing.noise_induced_hearing_loss`](/phonometry/reference/api/hearing/noise-induced-hearing-loss/) | Estimation of noise-induced hearing loss (ISO 1999:2013). |
| [`hearing.occupational_exposure`](/phonometry/reference/api/hearing/occupational-exposure/) | Occupational noise exposure: measurement strategies and uncertainty (ISO 9612:2009). |

## Room acoustics

| Module | Summary |
| :--- | :--- |
| [`room.room_acoustics`](/phonometry/reference/api/rooms/room-acoustics/) | Room acoustic parameters from impulse responses per ISO 3382-1:2009 (performance spaces) and ISO 3382-2:2008 (ordinary rooms). |
| [`room.room_ir`](/phonometry/reference/api/rooms/room-ir/) | Impulse-response acquisition per BS EN ISO 18233:2006. |
| [`room.room_noise`](/phonometry/reference/api/rooms/room-noise/) | Room-noise rating curves per ANSI/ASA S12.2-2019. |
| [`room.open_plan`](/phonometry/reference/api/rooms/open-plan/) | Open-plan-office spatial metrics per ISO 3382-3:2012. |
| [`room.reverberation_prediction`](/phonometry/reference/api/rooms/reverberation-prediction/) | Reverberation-time prediction from room geometry and absorption. |
| [`room.enclosed_space_absorption`](/phonometry/reference/api/rooms/enclosed-space-absorption/) | Sound absorption in enclosed spaces (EN 12354-6:2003). |
| [`room.image_source`](/phonometry/reference/api/rooms/image-source/) | Synthetic room impulse response by the image-source method (rectangular room). |
| [`room.steady_field`](/phonometry/reference/api/rooms/steady-field/) | Steady-state sound field in a room: room constant, critical distance, level. |
| [`room.room_modes`](/phonometry/reference/api/rooms/room-modes/) | Normal modes of a rectangular room: frequencies, kinds, count and density. |
| [`room.crowd_noise`](/phonometry/reference/api/rooms/crowd-noise/) | Crowd self-noise in a restaurant: the cocktail-party equilibrium. |

## Building acoustics

| Module | Summary |
| :--- | :--- |
| [`building.insulation`](/phonometry/reference/api/building/insulation/) | Field airborne sound insulation (ISO 16283-1:2014) and impact sound insulation (ISO 16283-2), with single-number weighted ratings and spectrum adaptation terms (ISO 717-1 airborne, ISO 717-2 impact). |
| [`building.panel_transmission`](/phonometry/reference/api/building/panel-transmission/) | Predicted airborne sound reduction index of panels (Bies, Hansen & Howard 2017, Engineering Noise Control 5e, Section 7.2; Sharp 1973). |
| [`building.aperture_transmission`](/phonometry/reference/api/building/aperture-transmission/) | Sound transmission through slits, holes and apertures (Hopkins 2007, Sound Insulation, Section 4.3.10; Gomperts 1964; Wilson & Soroka 1965). |
| [`building.lab_insulation`](/phonometry/reference/api/building/lab-insulation/) | Laboratory sound insulation of building elements (ISO 10140). |
| [`building.survey_insulation`](/phonometry/reference/api/building/survey-insulation/) | Field survey method for sound insulation and service-equipment noise (ISO 10052:2021). |
| [`building.intensity_insulation`](/phonometry/reference/api/building/intensity-insulation/) | Sound insulation measured with sound intensity (ISO 15186). |
| [`building.flanking_transmission`](/phonometry/reference/api/building/flanking-transmission/) | Laboratory measurement of flanking sound transmission (ISO 10848:2006/2010). |
| [`building.facade_prediction`](/phonometry/reference/api/building/facade-prediction/) | Façade sound insulation and outdoor radiation prediction (EN 12354-3/-4:2000). |
| [`building.building_prediction`](/phonometry/reference/api/building/building-prediction/) | Building acoustic performance prediction (EN 12354-1/-2:2000). |
| [`building.detailed_prediction`](/phonometry/reference/api/building/detailed-prediction/) | Detailed per-band building prediction (EN/ISO 12354-1/-2:2017). |
| [`building.building_uncertainty`](/phonometry/reference/api/building/building-uncertainty/) | Measurement uncertainty in building acoustics (ISO 12999-1:2020). |
| [`building.floor_covering_improvement`](/phonometry/reference/api/building/floor-covering-improvement/) | Impact-sound improvement of floor coverings on a small mock-up (ISO 16251-1:2014). |
| [`building.structure_borne_power`](/phonometry/reference/api/building/structure-borne-power/) | Structure-borne sound power of building equipment (EN 15657:2018; ISO 9611). |
| [`building.installed_structure_borne`](/phonometry/reference/api/building/installed-structure-borne/) | Installed structure-borne sound from service equipment (EN 12354-5:2009). |
| [`building.spanish_building_code`](/phonometry/reference/api/building/spanish-building-code/) | Spanish building code CTE DB-HR: global indices and requirement checks. |

## Materials and surfaces

| Module | Summary |
| :--- | :--- |
| [`materials.sound_absorption`](/phonometry/reference/api/materials/sound-absorption/) | Sound absorption in a reverberation room: BS EN ISO 354:2003. |
| [`materials.absorption_rating`](/phonometry/reference/api/materials/absorption-rating/) | Single-number rating of sound absorption (ISO 11654:1997). |
| [`materials.absorption_uncertainty`](/phonometry/reference/api/materials/absorption-uncertainty/) | Measurement uncertainty for sound absorption (ISO 12999-2:2020). |
| [`materials.airflow_resistance`](/phonometry/reference/api/materials/airflow-resistance/) | Airflow resistance of porous materials: ISO 9053-1 and ISO 9053-2. |
| [`materials.dynamic_stiffness`](/phonometry/reference/api/materials/dynamic-stiffness/) | Dynamic stiffness of resilient materials under floating floors (EN 29052-1:1992). |
| [`materials.impedance_tube`](/phonometry/reference/api/materials/impedance-tube/) | Impedance-tube material characterisation. |
| [`materials.porous_absorber`](/phonometry/reference/api/materials/porous-absorber/) | Porous-material models and multilayer absorber prediction. |
| [`materials.slow_sound_absorber`](/phonometry/reference/api/materials/slow-sound-absorber/) | Slow-sound slit panels loaded with Helmholtz resonators (perfect absorbers). |
| [`materials.scattering_diffusion`](/phonometry/reference/api/materials/scattering-diffusion/) | Random-incidence scattering and directional diffusion coefficients. |
| [`materials.diffuser_design`](/phonometry/reference/api/materials/diffuser-design/) | Far-field polar response and diffusion coefficient predicted from a diffuser design. |
| [`materials.metadiffuser`](/phonometry/reference/api/materials/metadiffuser/) | Metadiffusers: deep-subwavelength Schroeder-like sound diffusers. |
| [`materials.road_absorption`](/phonometry/reference/api/materials/road-absorption/) | In-situ sound absorption of road surfaces (ISO 13472-1 / ISO 13472-2). |

## Vibration and structure-borne

| Module | Summary |
| :--- | :--- |
| [`vibration.mechanical_mobility`](/phonometry/reference/api/vibration/mechanical-mobility/) | Mechanical mobility and the frequency-response-function family (ISO 7626-1:2011). |
| [`vibration.point_mobility`](/phonometry/reference/api/vibration/point-mobility/) | Point mobilities and impedances of infinite structures (Cremer, Heckl & Petersson 2005, Chapter 5, Table 5.1). |
| [`vibration.radiation_efficiency`](/phonometry/reference/api/vibration/radiation-efficiency/) | Radiation efficiency of a plate in bending (Hopkins 2007, Sound Insulation, Section 2.9; Leppington et al. |
| [`vibration.junction_transmission`](/phonometry/reference/api/vibration/junction-transmission/) | Bending-wave transmission coefficients for rigid plate junctions (Hopkins 2007, *Sound Insulation*, Section 5.2.1.3; Cremer et al. |
| [`vibration.experimental_sea`](/phonometry/reference/api/vibration/experimental-sea/) | Experimental statistical energy analysis: coupling loss factors from measured energies (Norton & Karczub Ch. |
| [`vibration.machine_diagnostics`](/phonometry/reference/api/vibration/machine-diagnostics/) | Kinematic fault frequencies of rotating machinery (Norton & Karczub Ch. |
| [`vibration.transfer_stiffness`](/phonometry/reference/api/vibration/transfer-stiffness/) | Dynamic transfer stiffness of resilient elements (ISO 10846-1/-2/-3). |
| [`vibration.human_vibration`](/phonometry/reference/api/vibration/human-vibration/) | Human exposure to whole-body and hand-transmitted vibration. |
| [`vibration.multiple_shock_vibration`](/phonometry/reference/api/vibration/multiple-shock-vibration/) | Whole-body vibration containing multiple shocks (ISO 2631-5:2018). |

## Environmental acoustics

| Module | Summary |
| :--- | :--- |
| [`environmental.outdoor_propagation`](/phonometry/reference/api/environment/outdoor-propagation/) | Outdoor sound propagation: ISO 9613-2:1996 general method of calculation. |
| [`environmental.ground_barriers`](/phonometry/reference/api/environment/ground-barriers/) | Spherical-wave ground effect and advanced barrier diffraction. |
| [`environmental.atmospheric_refraction`](/phonometry/reference/api/environment/atmospheric-refraction/) | Atmospheric refraction: ray tracing and the parabolic equation (PE). |
| [`environmental.air_absorption`](/phonometry/reference/api/environment/air-absorption/) | Atmospheric absorption of sound: ISO 9613-1:1993. |
| [`environmental.impulse_prominence`](/phonometry/reference/api/environment/impulse-prominence/) | Prominence of impulsive sounds and the LAeq adjustment (NT ACOU 112:2002). |
| [`environmental.impulsive_sound`](/phonometry/reference/api/environment/impulsive-sound/) | Objective prominence of impulsive sounds and the `LAeq` adjustment (ISO/PAS 1996-3:2022). |
| [`environmental.rating`](/phonometry/reference/api/environment/rating/) | Environmental noise descriptors per ISO 1996-1:2016. |
| [`environmental.measurement`](/phonometry/reference/api/environment/measurement/) | Determination of environmental-noise sound pressure levels (ISO 1996-2:2017). |
| [`environmental.spanish_regulation`](/phonometry/reference/api/environment/spanish-regulation/) | Spanish noise regulation: the corrected level LKeq (Real Decreto 1367/2007). |

## Aircraft and wind energy

| Module | Summary |
| :--- | :--- |
| [`aircraft.aircraft_noise`](/phonometry/reference/api/aeroacoustics/aircraft-noise/) | Aircraft noise certification: Effective Perceived Noise Level (ICAO Annex 16). |
| [`aircraft.atmospheric_absorption`](/phonometry/reference/api/aeroacoustics/atmospheric-absorption/) | One-third-octave-band atmospheric absorption for aircraft noise (SAE ARP 5534). |
| [`aircraft.airport_noise`](/phonometry/reference/api/aeroacoustics/airport-noise/) | Noise-Power-Distance (NPD) event-level interpolation (ECAC Doc 29). |
| [`aircraft.anp_fleet`](/phonometry/reference/api/aeroacoustics/anp-fleet/) | EASA ANP fleet database bridge for the ECAC Doc 29 airport-noise chain. |
| [`aircraft.rotorcraft_noise`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/) | Rotorcraft noise by the hemisphere method (ECAC Doc 32 / NORAH2). |
| [`environmental.wind_turbine_noise`](/phonometry/reference/api/aeroacoustics/wind-turbine-noise/) | Wind-turbine acoustic noise (IEC 61400-11:2012+A1:2018). |

## Underwater acoustics

| Module | Summary |
| :--- | :--- |
| [`underwater.acoustics`](/phonometry/reference/api/underwater/acoustics/) | Underwater-acoustics reference levels (ISO 18405:2017). |
| [`underwater.propagation`](/phonometry/reference/api/underwater/propagation/) | Underwater sound propagation: transmission loss (closed-form). |
| [`underwater.weston_regimes`](/phonometry/reference/api/underwater/weston-regimes/) | Weston's shallow-water propagation regimes (flux theory). |
| [`underwater.sound_speed`](/phonometry/reference/api/underwater/sound-speed/) | Speed of sound in sea water (empirical equations). |
| [`underwater.sonar_equation`](/phonometry/reference/api/underwater/sonar-equation/) | The sonar equation (passive and active), in decibels. |
| [`underwater.ocean_ambient_noise`](/phonometry/reference/api/underwater/ocean-ambient-noise/) | Ocean ambient-noise spectrum levels (Wenz framework). |
| [`underwater.seabed_reflection`](/phonometry/reference/api/underwater/seabed-reflection/) | Plane-wave reflection at the seabed (fluid-fluid Rayleigh model). |
| [`underwater.ship_radiated_noise`](/phonometry/reference/api/underwater/ship-radiated-noise/) | Ship radiated noise and equivalent monopole source level (ISO 17208-1/-2). |
| [`underwater.ship_traffic_noise`](/phonometry/reference/api/underwater/ship-traffic-noise/) | Predicted source-level spectrum of shipping traffic (semi-empirical models). |
| [`underwater.pile_driving_noise`](/phonometry/reference/api/underwater/pile-driving-noise/) | Radiated underwater sound from percussive pile driving (ISO 18406:2017). |
| [`underwater.marine_mammal_audiograms`](/phonometry/reference/api/underwater/marine-mammal-audiograms/) | Marine-mammal hearing thresholds (group audiograms and the orca audiogram). |
| [`underwater.marine_mammal_weighting`](/phonometry/reference/api/underwater/marine-mammal-weighting/) | Regulatory auditory weighting and exposure criteria for marine mammals. |
| [`underwater.numerical_propagation`](/phonometry/reference/api/underwater/numerical-propagation/) | Numerical models of underwater sound propagation (range-independent ocean). |

## Sound power and intensity

| Module | Summary |
| :--- | :--- |
| [`emission.sound_power`](/phonometry/reference/api/power/sound-power/) | Sound power level of a noise source from sound pressure measurements over an enveloping measurement surface: ISO 3744:2010 (engineering, accuracy grade 2) and ISO 3746:2010 (survey, accuracy grade 3). |
| [`emission.sound_power_intensity`](/phonometry/reference/api/power/sound-power-intensity/) | Sound power level of a noise source by sound-intensity **scanning**: ISO 9614-2:1996 (engineering, grade 2; survey/control, grade 3). |
| [`emission.sound_power_reverberation`](/phonometry/reference/api/power/sound-power-reverberation/) | Sound power level of a noise source measured in a reverberation test room: ISO 3741:2010 (precision method, accuracy grade 1). |
| [`emission.intensity`](/phonometry/reference/api/power/intensity/) | Two-microphone (p-p) sound intensity per IEC 61043:1993 and the ISO 9614-1:1993 field indicators. |
| [`metrology.intensity_compliance`](/phonometry/reference/api/power/intensity-compliance/) | IEC 61043:1993 sound-intensity instrument class verification. |
| [`emission.vibration_sound_power`](/phonometry/reference/api/power/vibration-sound-power/) | Airborne sound power from surface vibration (ISO/TS 7849-1/-2:2009). |
| [`emission.declaration`](/phonometry/reference/api/power/declaration/) | ISO 4871:1996 declaration of noise emission values of machinery and equipment. |

## Electroacoustics

| Module | Summary |
| :--- | :--- |
| [`electroacoustics.distortion`](/phonometry/reference/api/electroacoustics/distortion/) | Distortion metrics for electroacoustic equipment (IEC 60268-3 / AES17). |
| [`electroacoustics.frequency_response`](/phonometry/reference/api/electroacoustics/frequency-response/) | Frequency-response and coherence estimators (Bendat & Piersol). |
| [`electroacoustics.swept_sine`](/phonometry/reference/api/electroacoustics/swept-sine/) | Harmonic-distortion separation with exponential sweeps (Farina / Novak). |
| [`electroacoustics.piston`](/phonometry/reference/api/electroacoustics/piston/) | Radiation of a rigid circular piston set in an infinite baffle. |
| [`electroacoustics.loudspeaker`](/phonometry/reference/api/electroacoustics/loudspeaker/) | Rated loudspeaker characteristics (IEC 60268-5). |
| [`electroacoustics.microphone`](/phonometry/reference/api/electroacoustics/microphone/) | Rated microphone characteristics (IEC 60268-4). |
| [`electroacoustics.sound_reinforcement`](/phonometry/reference/api/electroacoustics/sound-reinforcement/) | Gain before feedback of a sound-reinforcement system. |

## Industrial noise control

| Module | Summary |
| :--- | :--- |
| [`noise_control.silencers`](/phonometry/reference/api/noise_control/silencers/) | Reactive silencers by the four-pole (transmission-matrix) method. |
| [`noise_control.hvac`](/phonometry/reference/api/noise_control/hvac/) | HVAC duct acoustics: fan power, duct losses, plenums and flow-generated noise. |
| [`noise_control.duct_path`](/phonometry/reference/api/noise_control/duct-path/) | End-to-end duct-borne noise calculation: fan to room, element by element. |
| [`noise_control.duct_modes`](/phonometry/reference/api/noise_control/duct-modes/) | Higher-order acoustic modes in ducts, with the mean-flow cut-on shift. |
| [`noise_control.enclosures`](/phonometry/reference/api/noise_control/enclosures/) | Insertion loss of a close or free-standing machine enclosure. |

## Program loudness

| Module | Summary |
| :--- | :--- |
| [`broadcast.program_loudness`](/phonometry/reference/api/broadcast/program-loudness/) | Programme loudness and true-peak level (ITU-R BS.1770-5, EBU R 128). |

## Uncertainty and data quality

| Module | Summary |
| :--- | :--- |
| [`metrology.uncertainty`](/phonometry/reference/api/metrology/uncertainty/) | Measurement uncertainty by the GUM and its Monte Carlo supplement. |
| [`metrology.random_data`](/phonometry/reference/api/metrology/random-data/) | Random-data qualification: stationarity tests and Rice crossing statistics. |

## Spectral analysis

| Module | Summary |
| :--- | :--- |
| [`metrology.spectra`](/phonometry/reference/api/spectra/spectra/) | Calibrated spectral-density estimation with statistical error analysis. |
| [`metrology.miso`](/phonometry/reference/api/spectra/miso/) | Multiple and partial coherence of a multiple-input/single-output system. |
| [`metrology.time_frequency`](/phonometry/reference/api/spectra/time-frequency/) | Calibrated time-frequency analysis: STFT spectrogram and zoom FFT. |
| [`metrology.signals`](/phonometry/reference/api/spectra/signals/) | Test signals and sample-rate utilities. |
| [`metrology.phase`](/phonometry/reference/api/spectra/phase/) | Phase utilities: minimum phase, group delay and excess phase. |
| [`metrology.cepstrum`](/phonometry/reference/api/spectra/cepstrum/) | Cepstral analysis: real/power/complex cepstrum, liftering and echo detection. |
| [`metrology.synchronous_average`](/phonometry/reference/api/spectra/synchronous-average/) | Time synchronous averaging (TSA) of a periodic waveform in noise. |
| [`metrology.inversion`](/phonometry/reference/api/spectra/inversion/) | Regularized spectral inversion with frequency-dependent regularization. |

## Wave simulation

| Module | Summary |
| :--- | :--- |
| [`simulation.fdtd`](/phonometry/reference/api/simulation/fdtd/) | 2D acoustic finite-difference time-domain (FDTD) simulation. |
| [`simulation.ntff`](/phonometry/reference/api/simulation/ntff/) | 2D near-to-far-field (NTFF) transformation over a closed contour. |
| [`simulation.elastic_fdtd`](/phonometry/reference/api/simulation/elastic-fdtd/) | 2D elastic finite-difference time-domain (P-SV) simulation. |

## Correlation & envelope

| Module | Summary |
| :--- | :--- |
| [`metrology.correlation`](/phonometry/reference/api/correlation/correlation/) | Correlation analysis and time-delay estimation. |
| [`metrology.envelope`](/phonometry/reference/api/correlation/envelope/) | Envelope and instantaneous phase via the Hilbert transform. |
