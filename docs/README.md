# phonometry Documentation

Full documentation for phonometry. Also available as a website:
**https://jmrplens.github.io/phonometry/**

## Guides

Every guide in the library, grouped by the folder it lives in. Each group
heading links the overview page that says what the area is for and what it
deliberately leaves out; each bold line links a section overview inside it.

Before any of them, [Start](start/index.md) is the four pages meant to be read
once: [Getting Started](start/getting-started.md) installs the library and runs
a first calibrated analysis, [Why phonometry](start/why-phonometry.md) shows
how a metric is checked against its standard, and [About](start/about.md) says
who maintains it, how to report an error and how to cite it.

### [Signal analysis](signals/index.md)

The chain that turns a digital signal into a standards-compliant number:
bands, weightings, ballistics, levels, spectra, calibration and uncertainty.
Every other section consumes it.

- [Build a sound level meter](signals/sound-level-meter.md): an end-to-end walk-through that composes the core API into a working meter: calibrate against an IEC 60942 tone, apply the IEC 61672-1 frequency and time weightings, integrate into $L_\mathrm{eq}$, SEL and percentile levels, split into IEC 61260-1 octave bands, and verify the class of every stage

**[Octave filtering](signals/filters/index.md)**

- [Filter Banks](signals/filters/filter-banks.md): band mathematics, bank parameters, parametric EQ, band decomposition and zero-phase filtering
- [Filter architecture gallery](signals/filters/filter-gallery.md): the five filter architectures compared at the band edges, the full response gallery for 1/1 and 1/3 octave, per-architecture usage and the Linkwitz-Riley crossover
- [Filter class verification](signals/filters/filter-compliance.md): the IEC 61260-1:2014 Table 1 acceptance mask band by band, the class 0 of the withdrawn IEC 61260:1995 / ANSI S1.11-2004 edition, what a performance class buys in a measurement, and the one-page compliance fiche
- [Block Processing](signals/filters/block-processing.md): stateful real-time workflows
- [Multichannel](signals/filters/multichannel.md): vectorized multichannel analysis

**[Levels and weighting](signals/levels/index.md)**

- [Integrated & Statistical Levels](signals/levels/levels.md): $L_\mathrm{eq}$, $L_\mathrm{Aeq}$, $L_{10}$/$L_{50}$/$L_{90}$, $L_\mathrm{Cpeak}$/SEL, noise dose (IEC 61252), octave spectrogram
- [Frequency Weighting](signals/levels/weighting.md): A, C, Z curves
- [Special weightings](signals/levels/special-weightings.md): the ISO 7196 G-weighting for infrasound, the historical B (ANSI S1.4-1983) and D (IEC 537) curves, and AU (IEC 61012) for audible sound in the presence of ultrasound
- [Time Weighting](signals/levels/time-weighting.md): Fast, Slow, Impulse ballistics

**[Signals and spectra](signals/spectra/index.md)**

- [Calibrated spectral analysis](signals/spectra/spectral-analysis.md): the Bendat & Piersol Welch estimators with their statistical quality: PSD and cross-spectral density with the effective number of averages, normalized random errors and chi-square confidence intervals, the coherent output spectrum with the spectral SNR, constant-power 1/n-octave smoothing, colored-noise generators with an exact power-law slope, and the Harris window figures of merit for choosing the taper
- [Time-frequency analysis](signals/spectra/time-frequency.md): the calibrated STFT spectrogram in absolute dB SPL with the exact Welch-module scaling and the time-versus-frequency resolution trade-off, and the zoom FFT that resolves tones closer than a practical FFT bin
- [Multiple and partial coherence](signals/spectra/miso-coherence.md): the Bendat & Piersol multiple-input/output coherence functions for several correlated sources and one output, with the Gaussian-elimination conditioning that separates a genuine cause from a source that merely correlates with it, and the partial coherent output spectra that say which source dominates each band
- [Correlation, time delay and envelope](signals/spectra/correlation-delay.md): auto- and cross-correlation with the Bendat & Piersol normalizations and random errors, time-delay estimation by the direct correlator, the cross-spectrum phase slope and the Knapp & Carter GCC (Roth, SCOT, PHAT, maximum likelihood) with the Eq. 8.129 peak-location uncertainty, sub-sample impulse-response delay and alignment, and the Hilbert envelope with instantaneous phase and frequency
- [Test signals and sample-rate tools](signals/spectra/test-signals.md): IEC 60268-1 tone bursts with exact gating (zero-crossing start, integral full periods, repetitive trains), polyphase resampling behind an explicit anti-alias specification whose designed Kaiser filter travels with the result, and band-limited fractional delay with a linear or circular boundary
- [Cepstrum, echoes and the envelope spectrum](signals/spectra/cepstrum-echoes.md): the power, real and complex cepstrum with quefrency analysis, echo detection with the reflection coefficient read off the cepstral peak, lowpass/highpass liftering of a log spectrum, the homomorphic round trip of the complex cepstrum, and the envelope spectrum that turns amplitude modulations into discrete lines
- [Time synchronous averaging](signals/spectra/synchronous-averaging.md): extraction of a periodic waveform of known period by time domain averaging (McFadden 1987), the comb filter that describes the operation in the frequency domain with unit teeth at the harmonics and nodes between them, the square-root noise-reduction law, and the choice of the number of averages that places a comb node on an interfering order
- [System measurement: Golay, shaped sweeps, inversion](signals/spectra/system-measurement.md): complementary Golay pairs whose periodic autocorrelations sum to an exact delta and deconvolve a noiseless system to machine precision, sweeps synthesized to follow an arbitrary target magnitude spectrum by group-delay shaping (Mueller & Massarani) with a near-ideal crest factor, and the regularized spectral inversion of a measured response with Kirkeby frequency-dependent regularization, achieved flatness and a capped out-of-band gain

**[Calibration and uncertainty](signals/metrology/index.md)**

- [Calibration and dBFS](signals/metrology/calibration.md): physical SPL and digital analysis
- [Compliance and verification](signals/metrology/compliance-verification.md): what a performance class asserts in IEC 61672-1 and IEC 61260-1, the public verifiers that grade weightings, filter banks and intensity spectra against their tolerance tables, how to read the numerical conformance report, and the verified scope of the pattern-evaluation and periodic-test parts (IEC 61672-2/-3, IEC 61260-2/-3) that only a laboratory can run
- [Measurement uncertainty](signals/metrology/gum-uncertainty.md): the GUM law of propagation of uncertainty and the Monte Carlo method (ISO/IEC Guide 98-3:2008 and Supplement 1): combined and expanded uncertainty, Welch–Satterthwaite effective degrees of freedom, and probabilistically symmetric coverage intervals
- [Data qualification](signals/metrology/data-qualification.md): the Bendat & Piersol stationarity tests (reverse arrangements with the Table A.6 acceptance regions, runs about the median with the exact Wald-Wolfowitz distribution) on segment mean squares, and the Rice statistics of level crossings, apparent frequency, peak rates and the irregularity factor that places the peak-height distribution between Rayleigh and Gaussian

### [Audio files](io/index.md)

Measurement audio in and out: every linear WAV a meter writes read into a
calibrated `Signal` with its `bext` provenance, long recordings streamed
block by block, BWF written with provenance and measured loudness, the
calibration sidecar, and lossless conversion.

- [Reading and writing measurement audio](io/audio-files.md): the whole workflow on one runnable page, from the meter's WAV to the calibrated level, the lossy warning, streaming, BWF writing, the sidecar and lossless conversion

### [Hearing and perception](perception/index.md)

What a listener makes of the sound a level merely counts: loudness, sharpness,
roughness and annoyance, speech intelligibility, and what a working life in
noise costs a hearing threshold.

**[Psychoacoustics](perception/psychoacoustics/index.md)**

- [Loudness](perception/psychoacoustics/loudness.md): the ISO 532-1 Zwicker loudness in sones with its one-page fiche, plus the equal-loudness contours (ISO 226)
- [Advanced loudness](perception/psychoacoustics/advanced-loudness.md): the Moore-Glasberg stationary (ISO 532-2) and time-varying (ISO 532-3) methods and the Sottek Hearing Model loudness (ECMA-418-2), with the model-choice table
- [Sound Quality Metrics](perception/psychoacoustics/sound-quality.md): sharpness (DIN 45692) and the ECMA-418-2 Sottek Hearing Model tonality, roughness and fluctuation strength
- [Prominent discrete tones](perception/psychoacoustics/tone-prominence.md): the ECMA-418-1 tone-to-noise and prominence ratios that decide whether a discrete tone is prominent and justify tonal rating adjustments
- [Tonal audibility of tones in noise](perception/psychoacoustics/tone-audibility.md): the ISO/PAS 20065 engineering method for the audibility $\Delta L$ of a tone above the masking threshold: the critical band about the tone, the critical-band masking level, the masking index, and the decisive and mean audibility
- [Psychoacoustic annoyance & fluctuation strength](perception/psychoacoustics/psychoacoustic-annoyance.md): the Fastl & Zwicker annoyance $PA = N_5\left(1 + \sqrt{w_S^2 + w_{FR}^2}\right)$ from loudness, sharpness, roughness and fluctuation strength (Eqs 16.2–16.4), the closed form for AM broadband noise (Eq. 10.2) and the Osses 2016 fluctuation-strength signal model

**[Speech](perception/speech/index.md)**

- [Speech Transmission Index](perception/speech/speech-transmission.md): how much of the speech envelope a room or sound system preserves: the IEC 60268-16 modulation transfer function, indirect method and direct STIPA measurement
- [Speech Intelligibility Index](perception/speech/speech-intelligibility.md): the ANSI S3.5-1997 SII in all four band procedures (critical band, equally contributing, one-third octave, octave): band-importance weighting, self-speech and upward spread of masking, band audibility, and the index in noise and hearing loss
- [Objective intelligibility (STOI & ESTOI)](perception/speech/objective-intelligibility.md): the correlation-based intelligibility measures for time-frequency weighted noisy speech from a clean/degraded pair: STOI (Taal et al. 2011), the clipped per-band envelope correlation, and ESTOI (Jensen & Taal 2016), the row- and column-normalised spectral correlation that tracks modulated maskers

**[Hearing and exposure](perception/hearing/index.md)**

- [Hearing threshold](perception/hearing/hearing-threshold.md): the age-related hearing threshold distribution (ISO 7029:2017) and the free-field/diffuse-field reference threshold of hearing (ISO 389-7:2005)
- [Noise-induced hearing loss](perception/hearing/noise-induced-hearing-loss.md): the ISO 1999:2013 noise-induced permanent threshold shift (NIPTS) and its population distribution, and the combination with age into the hearing threshold level associated with age and noise (HTLAN)
- [Occupational noise exposure](perception/hearing/occupational-exposure.md): the ISO 9612 task-based, job-based and full-day measurement strategies and the Annex C uncertainty budget behind every $L_\mathrm{EX,8h}$ report

### [Rooms and buildings](buildings/index.md)

Sound in the built environment, split along its natural line: what happens
inside one room, and what passes between rooms — measured, rated and
predicted.

**[Room acoustics](buildings/rooms/index.md)**

- [Room Acoustics](buildings/rooms/room-acoustics.md): impulse-response acquisition (ISO 18233), reverberation and room parameters (ISO 3382-1/2), open-plan speech metrics (ISO 3382-3), reverberation-room sound absorption (ISO 354)
- [Room impulse response acquisition (ISO 18233)](buildings/rooms/room-impulse-response.md): the exponential sine sweep and its deconvolution, the MLS correlation method, and the source and microphone placement rules of a valid measurement
- [Open-plan office acoustics (ISO 3382-3)](buildings/rooms/open-plan-acoustics.md): the spatial decay rate $D_{2,\mathrm{S}}$ of A-weighted speech, the level at 4 m, and the distraction and privacy distances derived from STI
- [Sound absorption in enclosed spaces](buildings/rooms/enclosed-space-absorption.md): the EN 12354-6:2003 prediction of a room's total equivalent absorption area and reverberation time from its surfaces and objects (Clause 4)
- [Reverberation-time prediction](buildings/rooms/reverberation-prediction.md): the reverberation time from a room's volume and surface absorption by five statistical models (Sabine, Eyring, Millington-Sette, Fitzroy, Arau-Puchades), with the air-absorption term
- [Image sources and the steady-state room field](buildings/rooms/room-image-sources.md): the deterministic image-source room impulse response of a rectangular room (Kuttruff/Vorländer) and the statistical steady-state level with the room constant, critical distance and Schroeder frequency (Bies)
- [Room-noise criteria](buildings/rooms/room-noise.md): the ANSI/ASA S12.2-2019 room-noise ratings: the NC rating (NC-(SIL) designation with the tangency method when exceeded, Table 1) and the RC Mark II rating with its rumble/hiss/neutral spectral tag (Annex D)

**[Sound insulation](buildings/insulation/index.md)**

- [Field Insulation Measurement (ISO 16283)](buildings/insulation/insulation-field.md): field airborne and impact insulation (ISO 16283-1/2), the Clause 14 test report and the measurement uncertainty that qualifies it (ISO 12999-1)
- [Small Rooms: the ISO 16283 Low-Frequency Procedure](buildings/insulation/low-frequency-procedure.md): the corner procedure ISO 16283-1/-2/-3 make mandatory below 25 m³: the 25 m³ trigger with its rounding, the corner level of Formula (12), the Formula (13) combination and the 63 Hz octave reverberation time of Clause 10.4
- [Laboratory Insulation Measurement](buildings/insulation/insulation-lab.md): the ISO 10140 laboratory sound reduction index and normalized impact level, measured with flanking suppressed, with the background-noise correction and the accredited test fiches
- [Sound Insulation by Intensity (ISO 15186)](buildings/insulation/insulation-intensity.md): the ISO 15186-1/-2 sound reduction index from the intensity scanned over the radiating face, for the whole element or element by element
- [Sound Insulation Survey Method (ISO 10052)](buildings/insulation/insulation-survey.md): the ISO 10052 octave-band control method: the reverberation index and its room-class estimate, the airborne, impact, façade and service-equipment quantities and their survey reports
- [Laboratory Flanking Transmission (ISO 10848)](buildings/insulation/flanking-lab.md): the ISO 10848 junction vibration reduction index $K_{ij}$ and the flanking descriptors $D_\mathrm{n,f}$ and $L_\mathrm{n,f}$ measured on a test facility
- [Insulation Ratings (ISO 717)](buildings/insulation/insulation-ratings.md): the ISO 717-1 airborne and ISO 717-2 impact reference-curve engines with the spectrum adaptation terms $C$, $C_\mathrm{tr}$ and $C_\mathrm{I}$, the enlarged-range and one-decimal variants, and the ISO 717 fiche
- [Spanish Building Code (CTE DB-HR)](buildings/insulation/spanish-building-code.md): the DB-HR global indices $R_\mathrm{A}$, $R_\mathrm{A,tr}$, $D_\mathrm{nT,A}$ and $D_{2\mathrm{m,nT,Atr}}$ from the direct Annex A formula over eighteen bands, the four normalised spectra, the clause 2 requirement tables and the window-size correction
- [Façade Sound Insulation](buildings/insulation/facade-insulation.md): the building envelope measured (ISO 16283-3), predicted from its element indices (EN 12354-3) and radiating an indoor source outwards (EN 12354-4)
- [Heavy and Soft Impact Sources (ISO 16283-2)](buildings/insulation/heavy-impact-sources.md): the rubber ball and the bang machine, their impact force exposure level and octave-band specification, the Fast-weighted standardization of the maximum impact level and the A-weighted single number of ISO 717-2 Annex D

**[Insulation design](buildings/design/index.md)**

- [Structure-borne sound power of building equipment](buildings/design/structure-borne-power.md): the EN 15657 reception-plate method with the plate-injected power level, the spatial mean plate velocity, the loss factor from the structural reverberation time, and the blocked-force / characteristic-level / free-velocity source quantities from the low- and high-mobility plates
- [Installed structure-borne sound from equipment](buildings/design/installed-structure-borne.md): the EN 12354-5 prediction of the receiving-room sound pressure level from service equipment: the coupling term from source and receiver mobilities, the installed structure-borne power, and the per-path transmission with its energetic total
- [Predicting Sound Insulation (EN 12354)](buildings/design/insulation-prediction.md): airborne and impact flanking-transmission prediction between rooms (EN 12354-1/2)
- [Detailed Per-Band Prediction (ISO 12354)](buildings/design/detailed-prediction.md): the per-band detailed model of ISO 12354-1/-2 with in-situ element conversion and per-path contributions
- [Predicting Panel Sound Insulation](buildings/design/panel-sound-insulation.md): theoretical airborne insulation of single panels (mass law and coincidence, Sharp) and double walls (mass-spring-mass, Bies), transmission through slits and apertures (Gomperts/Wilson-Soroka), plate radiation efficiency (Leppington/Maidanik) and the point mobilities of infinite plates and beams (Cremer)
- [Floor-Covering Impact Improvement (ISO 16251-1)](buildings/design/impact-improvement.md): the ISO 16251-1 weighted improvement of impact sound insulation given by a soft floor covering, measured on a small heavyweight mock-up
- [Predicting Resilient-Layer Performance](buildings/design/resilient-layers.md): the prediction side of resilient layers, from the tapping-machine force model to the improvement of soft coverings, floating floors and wall linings

### [Materials and surfaces](materials/index.md)

Where the coefficients every room and insulation model consumes come from: the
laboratory instruments that measure them, the single-number ratings that
summarise them, and the models that predict them.

**[Absorbers](materials/absorbers/index.md)**

- [Sound Absorption Measurement and Rating](materials/absorbers/absorption-measurement.md): reverberation-room absorption measurement (ISO 354), the weighted rating $\alpha_\mathrm{w}$ with its classes (ISO 11654), and the measurement uncertainty (ISO 12999-2)
- [Airflow Resistance](materials/absorbers/airflow-resistance.md): static and alternating determination of airflow resistance, specific resistance and resistivity (ISO 9053-1/-2), and what the resistivity feeds in the porous models
- [Impedance Tube](materials/absorbers/impedance-tube.md): normal-incidence absorption, surface impedance and transmission loss by the standing-wave-ratio, transfer-function and transfer-matrix methods (ISO 10534-1/-2, ASTM E2611), plus the virtual FDTD tube
- [Porous and Multilayer Absorbers](materials/absorbers/porous-absorbers.md): the Delany-Bazley, Miki and Johnson-Champoux-Allard porous models, the transfer-matrix multilayer solver with perforated, microperforated (Maa) and membrane layers, and the random-incidence Paris integral
- [Metamaterial Absorbers](materials/absorbers/metamaterial-absorbers.md): the critical-coupling condition for perfect sound absorption and the slow-sound slit panel loaded by Helmholtz resonators (Jiménez et al. 2016/2017), with the transfer-matrix model and the design solver

**[Diffusers and surfaces](materials/diffusers/index.md)**

- [Diffusers and Their Coefficients](materials/diffusers/diffusers.md): random-incidence scattering (ISO 17497-1), the free-field diffusion coefficient (ISO 17497-2), Schroeder quadratic-residue design and the Fraunhofer far-field prediction that grades a well-depth sequence before it is built
- [Metadiffusers](materials/diffusers/metadiffusers.md): deep-subwavelength Schroeder diffusers from slits loaded by Helmholtz resonators (Jiménez et al. 2017): slow sound, per-well reflection phases, ternary sequences and the published quadratic-residue design evaluated end to end

**[Resilient layers](materials/resilient/index.md)**

- [Dynamic stiffness of resilient materials](materials/resilient/dynamic-stiffness.md): the EN 29052-1:1992 dynamic stiffness per unit area of floating-floor resilient layers from the load-plate resonance, and the floating-floor natural frequency

**[Surfaces measured in place](materials/surfaces/index.md)**

- [In-situ Road-Surface Absorption](materials/surfaces/road-absorption.md): in-situ road-surface absorption by the extended-surface subtraction technique (ISO 13472-1) and the spot method (ISO 13472-2), with the Adrienne window, the sampled-area radius and the plane-wave limits of the spot tube

### [Vibration and structure-borne sound](vibration/index.md)

Vibration as a source of sound, as a human exposure in its own right, and as
the diagnostic signature of a machine.

**[Structure-borne sources](vibration/structural/index.md)**

- [Mechanical mobility and the FRF family](vibration/structural/mechanical-mobility.md): the ISO 7626-1:2011 family of motion-per-force frequency-response functions (receptance, mobility, accelerance and their reciprocals, Table 1), conversion through the receptance pivot, and the single-degree-of-freedom reference resonator (Annex A)
- [Dynamic transfer stiffness of resilient elements](vibration/structural/transfer-stiffness.md): the ISO 10846 dynamic transfer stiffness $k_{21}$ of vibration isolators: the level $L_k$ re 1 N/m and loss factor, the direct and indirect (transmissibility) determination methods, and the Annex-A relation to mechanical impedance and effective mass
- [Bending-wave transmission at plate junctions](vibration/structural/junction-transmission.md): the wave-approach (Cremer/Craik/Hopkins 5.2.1.3) frequency-independent bending-wave transmission coefficients for rigid X, T, L and in-line plate junctions, their diffuse-field angular average, and the derived coupling loss factor and vibration reduction index $K_{ij}$

**[Human vibration](vibration/human/index.md)**

- [Human Vibration](vibration/human/human-vibration.md): whole-body and hand-arm frequency weightings (ISO 8041-1), weighted r.m.s. acceleration, running r.m.s./MTVV/VDV and crest factor (ISO 2631-1), vibration in buildings (ISO 2631-2), vibration total value and daily exposure $A(8)$ (ISO 5349-1/-2), and the exposure action/limit values of Directive 2002/44/EC
- [Multiple-shock whole-body vibration](vibration/human/multiple-shock-vibration.md): the ISO 2631-5:2018 spinal-response model: the seat-to-spine transfer function, the acceleration and daily dose from the response peaks (Clause 5), and the compressive stress, stress variable $R$ and Weibull probability of lumbar injury (Annex C)

**[Machinery](vibration/machinery/index.md)**

- [Machine fault frequencies](vibration/machinery/machine-diagnostics.md): the kinematic fault-frequency families of rotating machinery (Norton & Karczub Section 8.4) overlaid on a measured envelope spectrum: rolling-contact bearing BPFO, BPFI, BSF and cage frequencies, gear-mesh frequency and sideband families, induction-motor supply, slip, pole-pass and rotor-slot harmonics, and fan, blower and pump blade-passing frequencies with the lobed interaction patterns of a ducted axial fan

### [Environment and transport](environment/index.md)

The source-path-receiver problem stretched over open ground: emission models
for traffic and turbines, outdoor propagation, and the ratings and legal
limits a noise map is drawn against.

**[Outdoor sound](environment/propagation/index.md)**

- [Outdoor Sound Propagation](environment/propagation/outdoor-propagation.md): atmospheric absorption $\alpha(f)$ (ISO 9613-1) and the ISO 9613-2 general method: divergence, atmospheric absorption, ground effect and barrier screening
- [Spherical ground effect and advanced barriers](environment/propagation/ground-barriers.md): the Weyl-Van der Pol spherical-wave reflection coefficient over a finite-impedance ground, and wave-theoretic barriers (Kurze-Anderson Fresnel number, exact rigid half-plane, thick barriers and the coherent four-path barrier on the ground)
- [Atmospheric refraction: rays and the GFPE](environment/propagation/atmospheric-refraction.md): effective sound-speed profiles (linear and logarithmic), ray tracing through a refracting atmosphere (curved paths, turning points, closed-form curvature radius and shadow-zone distance) and the Green's Function parabolic equation (GFPE) for the relative-level field over the range-height plane, anchored to the spherical ground effect in the homogeneous limit

**[Environmental sources](environment/sources/index.md)**

- [Wind-turbine noise: apparent sound power & tonal audibility](environment/sources/wind-turbine-noise.md): the IEC 61400-11 apparent sound power level referred to the rotor centre and the tonal-audibility chain (Zwicker critical band, masking-noise level and audibility criterion)
- [CNOSSOS-EU railway source emission](environment/sources/cnossos-rail-emission.md): the common EU method for railway noise emission (Directive 2002/49/EC Annex II, section 2.3 and Appendix G): rail and wheel roughness with the contact filter and the transfer functions, impact noise, curve squeal, traction, aerodynamic noise, bridges, the source directivity and the two equivalent source lines at 0,5 m and 4,0 m
- [CNOSSOS-EU road traffic source emission](environment/sources/cnossos-road-emission.md): the common EU road source of Annex II to Directive 2002/49/EC (section 2.2 and Appendix F): rolling and propulsion sound power per vehicle category with the corrections for road surface, air temperature, studded tyres, gradient and junctions, and the directional sound power per metre of source line

**[Assessment and regulation](environment/assessment/index.md)**

- [Environmental levels](environment/assessment/environmental-levels.md): $L_\mathrm{den}$, $L_\mathrm{dn}$ and the composite rating levels of ISO 1996-1, with the ISO 1996-2 tonal adjustment, residual-noise correction and measurement uncertainty budget
- [Spanish Noise Regulation (RD 1367/2007)](environment/assessment/spanish-noise-regulation.md): the corrected level $L_\mathrm{Keq}$ with its tonal, low-frequency and impulsive corrections, the evaluation periods split into noise phases, the acoustic quality objective and immission limit tables, and the Article 25 compliance check of an activity
- [Impulsive-sound prominence](environment/assessment/impulsive-sound.md): the NT ACOU 112:2002 predicted prominence of impulsive sounds (onset rate and level difference) and the graduated adjustment $K_\mathrm{I}$ added to $L_\mathrm{Aeq}$

### [Aircraft noise](aircraft/index.md)

The two internationally negotiated families: certification, which fixes one
number per aircraft type at reference points around a runway, and contour
methods, which predict what a whole airport does to the land around it.

- [Aircraft noise: Effective Perceived Noise Level](aircraft/aircraft-noise.md): the ICAO Annex 16 Vol. I Appendix 2 EPNL (perceived noisiness and PNL, the tone correction by the slope method, and the 10 dB-down duration correction), the IEC 61265 measurement-system verifier, the SAE ARP 5534 one-third-octave-band atmospheric absorption (SAE Method), and the ECAC Doc 29 noise-power-distance (NPD) event-level interpolation
- [Airport Noise (ECAC Doc 29)](aircraft/airport-noise.md): the noise-power-distance engine, the per-segment single-event chain of lateral attenuation, engine-installation and duration effects, and the ground-grid SEL contour of a full departure or approach
- [The ANP fleet database](aircraft/anp-fleet.md): the EASA/EUROCONTROL Aircraft Noise and Performance database shipped with the package, its noise-power-distance curves and default fixed-point trajectories per aircraft type, and the wiring that runs the ECAC Doc 29 single-event level and ground-grid contour straight from an aircraft identifier
- [Rotorcraft noise: the hemisphere method](aircraft/rotorcraft-noise.md): the ECAC Doc 32 / NORAH2 helicopter noise hemisphere source model, its propagation adjustments (spherical spreading, ISO 9613-1 / Table 4 atmospheric absorption, Chien-Soroka ground effect over CNOSSOS impedance ground), the flight-condition interpolation and track kinematics, the single-event SEL/LASmax/EPNL with ground-grid contours, and the terrain machinery (mean ground plane, log-mean flow resistivity, rubber-band screening, digital elevation models)

### [Underwater acoustics](underwater/index.md)

The same physics on a different scale and a different reference — levels re 1
µPa, a medium that refracts sound into channels carrying it for kilometres,
and the marine mammals that hear it.

- [Underwater acoustics: radiated noise & pile driving](underwater/underwater-acoustics.md): the ISO 18405 reference levels (SPL, SEL, peak re 1 µPa), the ISO 17208 ship radiated noise level and equivalent monopole source level via the Lloyd's-mirror correction, and the ISO 18406 single-strike, peak and cumulative pile-driving sound exposure
- [Underwater sound propagation](underwater/underwater-propagation.md): closed-form propagation loss (geometrical spreading plus volume absorption by Francois-Garrison, Ainslie-McColm or Thorp), Weston's four shallow-water propagation regimes with their transition ranges, the speed of sound in sea water (UNESCO/Chen-Millero, Del Grosso, Mackenzie, Medwin) with the sound-speed profile, the passive/active sonar equation and the detection-range inversion, seabed reflection loss (Rayleigh) and the ocean ambient-noise spectrum (Wenz wind/thermal plus JOMOPANS-ECHO ship-traffic source levels)
- [Underwater propagation solvers](underwater/underwater-solvers.md): the numerical solvers of the range-independent stratified ocean (Jensen et al.): the finite-difference normal-mode sum, Runge-Kutta ray tracing with Snell's invariant, Gaussian beam tracing on those same rays for a field that stays finite at caustics, and the split-step Fourier parabolic equation, each validated against an exact closed form, plus the guidance for choosing between them and the closed forms
- [Marine-mammal noise exposure](underwater/marine-mammal-exposure.md): the hearing side of underwater noise: the Southall et al. group audiograms and Ainslie's orca audiogram, the regulatory auditory weighting functions with the guidance version selectable (NMFS 2024 v3.0, NMFS 2018 v2.0, Southall et al. 2019), the TTS and injury onset criteria with their dual impulsive metric, and the weighted cumulative exposure of a pile-driving campaign against them

### [Sources and devices](devices/index.md)

Emission: numbers that belong to the device rather than to the room or the
distance it is heard at, from sound power to loudspeaker distortion to duct
silencers.

**[Sound power and intensity](devices/emission/index.md)**

- [Sound Intensity (p-p)](devices/emission/intensity.md): two-microphone intensity and field indicators
- [Sound Power](devices/emission/sound-power.md): choosing the determination method for a source and declaring the noise emission (ISO 4871)
- [Sound power by pressure methods](devices/emission/sound-power-pressure.md): the enveloping surface of ISO 3744/3746 and the precision anechoic grade of ISO 3745
- [Sound power in the reverberation room](devices/emission/sound-power-reverberation.md): the direct and comparison methods of ISO 3741
- [Sound power by intensity scanning](devices/emission/sound-power-intensity.md): the on-site scanning of ISO 9614-2 and the ISO 9614-3 precision grade
- [Sound power from surface vibration](devices/emission/vibration-sound-power.md): the ISO/TS 7849 estimation of a machine's radiated airborne sound power from its surface vibratory velocity and a radiation factor: the velocity level and calibration, the surface mean, and the Part 1 upper limit ($\varepsilon = 1$) versus the Part 2 engineering value

**[Electroacoustics](devices/electroacoustics/index.md)**

- [Electroacoustics: distortion & frequency response](devices/electroacoustics/electroacoustics.md): the IEC 60268-3 distortion set (THD, nth-order harmonic, THD+N and SINAD via AES17, SMPTE and CCIF intermodulation, dynamic intermodulation and weighted THD), the AES17 dynamic range and idle channel noise, and the Bendat & Piersol H1/H2 frequency-response estimators with the ordinary coherence $\gamma^2$
- [Loudspeaker characterisation (IEC 60268-5)](devices/electroacoustics/loudspeakers.md): the sensitivity conventions of a loudspeaker datasheet, the radiating piston (radiation impedance and directivity) behind its polar response, and the IEC 60268-5 characteristics fiche
- [Microphone characterisation (IEC 60268-4)](devices/electroacoustics/microphones.md): the IEC 60268-4 sensitivity references, the directional patterns and directivity index, the inherent noise in dB(A) and dB(468), and the IEC 60268-4 characteristics fiche
- [Swept-sine distortion and phase utilities](devices/electroacoustics/swept-sine-distortion.md): harmonic separation from one exponential sweep (Farina 2000) with the synchronized swept-sine of Novak et al. 2015 for coherent harmonic phases, THD as a function of the excitation frequency, and minimum phase from $|H|$ (real cepstrum), group delay and excess phase

**[Broadcast](devices/broadcast/index.md)**

- [Programme loudness & true peak](devices/broadcast/program-loudness.md): the ITU-R BS.1770-5 programme loudness (K-weighting, gated 400 ms blocks, channel weights including the Annex 3 positions) and the oversampled true-peak level in dBTP, with the EBU R 128 −23 LUFS practice, the Tech 3341 EBU Mode momentary/short-term/integrated meters and the Tech 3342 loudness range, validated against the official EBU test signals
- [Quasi-peak programme meter (ITU-R BS.468-4)](devices/broadcast/quasi-peak.md): the psophometric quasi-peak detector of clause 2 — eleven tone-burst acceptance windows where the Recommendation prints no time constant at all, the clause 2.6 calibration that makes a reading dBqps, and the three fitted time scales those windows identify only to within a factor of 1.61 to 2.09

**[Noise control](devices/noise-control/index.md)**

- [Silencers](devices/noise-control/silencers.md): reactive silencers by the four-pole transmission-matrix method (the closed-form expansion chamber, Helmholtz, quarter-wave and extended-tube resonators) with transmission and insertion loss, the independent FDTD cross-check, and the design trade-off against dissipative linings (Bies, Hansen & Howard; Munjal)
- [Duct-borne noise: fan to room](devices/noise-control/duct-path.md): the end-to-end HVAC duct-noise calculation (fan sound power from the operating point, duct runs, elbows, splits, flexible duct, splitter silencers and plenums, the regenerated noise of silencers and air terminal devices, the room effect and the NC verdict) anchored on Long's worked Table 14.9 sheet, with the higher-order-mode cut-on that limits every plane-wave method (Long; Norton & Karczub; ASHRAE; AHRI 885)
- [Room to room: partition, receiving room, criterion](devices/noise-control/room-to-room.md): the composed room-to-room chain (the reverberant source-room level from a sound power level and the room constant, the partition transmission loss, the equivalent absorption area of the receiving room, the received spectrum and its NC verdict), the transmission loss a partition needs to meet a criterion curve and the same inverse for a lined machine enclosure, anchored on the worked octave-band answers of Norton & Karczub problems 4.16, 4.18 and 4.21
- [Industrial noise control: HVAC & enclosures](devices/noise-control/noise-control.md): the HVAC duct methods (end reflection, elbows, plenums and flow-generated noise) and machine-enclosure insertion loss from a supplied panel $R$ and the interior room constant (Bies, Hansen & Howard)

### [Wave simulation](simulation/index.md)

Where the rest of the library predicts a number, this one computes the wave
field itself on a finite-difference grid.

- [2D FDTD wave simulation](simulation/fdtd-simulation.md): the deterministic staggered-grid pressure-velocity FDTD solver (Attenborough & Van Renterghem 2021, chapter 4) with Gaussian, tone and arbitrary-signal sources, pressure probes, rasterised obstacles, rigid/impedance/absorbing boundaries, the near-to-far-field Kirchhoff-Helmholtz chain, and a result object with probe histories and field snapshots
- [Elastic waves and fluid-solid coupling](simulation/elastic-waves.md): the elastic P-SV companion solver (Virieux 1986) on the same staggered grid, with stress-imaging free surfaces and Rayleigh waves, fluids as the shear-free limit, and fluid-solid coupling validated against Brekhovskikh & Godin: mode conversion, Scholte interface waves and immersed-plate transmission

## Reference

- [API Reference](reference/api/index.md): curated quick table of every public function and class
- [Generated API reference](https://jmrplens.github.io/phonometry/reference/api/): one page per public module, generated from the source docstrings (`make api-docs`). English only; the Spanish site serves it via locale fallback
- [Theory](reference/theory/index.md): standards, math and design decisions, split by domain
  - [Signal analysis](reference/theory/signal-analysis.md): filter banks, weightings, time integration, level and exposure metrics, sound intensity, GUM uncertainty
  - [Perception and hearing](reference/theory/perception.md): equal-loudness contours, loudness models, sound quality, tone prominence, STI/SII, hearing statistics
  - [Rooms and buildings](reference/theory/rooms-buildings.md): room acoustics, noise criteria, insulation and ratings, flanking prediction
  - [Materials and surfaces](reference/theory/materials-surfaces.md): scattering and diffusion, in-situ road-surface absorption, absorption ratings, airflow resistance, impedance tube
  - [Environment and transport](reference/theory/environment-transport.md): environmental descriptors, impulsive adjustment, outdoor propagation, occupational exposure, sound power
  - [Vibration](reference/theory/vibration.md): human vibration weightings and metrics, multiple-shock spinal model
- [Glossary](reference/glossary.md): every quantity the guides compute, grouped by domain, each with its symbol, a one-sentence definition, its unit, the standard and clause that defines it and the guide that implements it, plus the table of symbols that collide across domains
- [Bibliography](reference/bibliography.md): the books and papers behind the guides, grouped by domain, every entry with a verified DOI or official publisher link
- [Conformance report](CONFORMANCE.md): auto-generated numerical validation: every check pins a standard clause's expected value against the library's computed value, regenerated in CI
- [Standards errata](ERRATA.md): defects found in the published standards themselves during implementation: misprints, examples contradicting their own normative text, ambiguous wording, each with evidence and the library's disposition

## Development

Run the test suite with `pytest tests/`, the full quality gate with `make check`
(ruff + mypy + bandit + tests), and regenerate the documentation images with
`make graphs`. See [CONTRIBUTING.md](../CONTRIBUTING.md).
