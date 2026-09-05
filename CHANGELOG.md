# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `noise_control.flow_noise_straight_duct_overall`, the two closed forms VDI
  2081 Section 5.2.1 prints for the flow noise of a straight run: Equation (16)
  for the unweighted overall sound power level and Equation (17) for the
  A-weighted one. The guide already said both were covered; only the first was,
  and only spread over the octaves by the relative spectrum of Figure 16, which
  does not sum back to it.

  The A-weighted form is not the other one weighted. Fifty times the logarithm
  of the duct velocity is a fifth power of it and seventy is a seventh, because
  raising the speed also moves the spectrum up into the part of the curve the
  weighting stops attenuating: halving the velocity is worth 15 dB on one
  number and 21 dB on the other.

  The three conformance rows for Equations (16), (17) and (34) called nothing
  and computed their own expected value, so no defect in the library could have
  failed them. All three go through the published functions now, and the limit
  frequency of Equations (33) and (34) is `plane_wave_limit`, which the library
  already had from the duct-mode theory: the rectangular form is exactly
  `c / (2a)` and the round one is the guideline's rounding of the first
  circular mode, so the law is called rather than written a second time.

- Gear units, by ISO 20816-9:2020, which grades them against a **rating**
  rather than against a machine class. `GEAR_UNIT_ZONES` carries the three
  boundary tables, one per quantity, keyed by the rating number that indexes
  them: shaft relative peak-to-peak displacement in micrometres, housing r.m.s.
  velocity in millimetres per second and housing true peak acceleration in
  metres per second squared, which is the vocabulary Table 1 fixes. The three
  tables are one ladder seen three times, with the rating itself as the B/C
  boundary and its neighbours as the other two, and a rating between two
  printed rows is refused rather than interpolated. `GEAR_UNIT_CLASSES` is the
  classification of Table 5, and it carries `None` for the acceleration of
  every subclass b) row, which is what the table prints there.

  `gear_shaft_displacement_limit` and `gear_housing_velocity_limit` are the
  rating curves of Annex A, so a filtered measurement can be judged line by
  line instead of as one broad-band value. The velocity curve turns out to be
  Formula (C.1) of Part 1 with the corners and the slope Part 9 states, 45 Hz
  and 1590 Hz at 14 dB per decade, so it calls `allowable_velocity` rather than
  drawing the same shape twice.
- `noise_control.section_change_loss`, the element of the VDI 2081 chain that
  was described in the guide and missing from the library. Section 6.3 gives
  the reflection at a sudden step in closed form, `10 lg (r+1)²/(4r)` in the
  area ratio, with two rules about when it applies: a sudden reduction
  reflects in every band, a sudden increase only below the limit frequency of
  the duct the sound arrives through. VDI 3733 recommends taking no more than
  5 dB from it, which is the default `cap`, and it binds from an area ratio of
  about 10,5 upwards. A gradual change through a long tapered adapter reflects
  nothing worth counting, and 6.3 says so.

  There is no `model=` here: Long folds the reflection from a change of total
  section into the junction itself, which is what `split_loss` does for
  `model="ashrae"`, while VDI 2081 treats the two as separate elements.

- The numbers behind the machine-vibration grading, for industrial machines.
  ISO 20816-1 fixes the shape of the judgement and leaves the boundaries to the
  machine-specific parts; `INDUSTRIAL_MACHINE_ZONES` carries the four classes
  of ISO 10816-3 Tables A.1 and A.2, keyed by machine group and support class,
  and `industrial_machine_zone` grades against them. The tables state each
  class twice, once in displacement and once in velocity, and 5.2.3 says the
  more restrictive of the two gradings applies: a group 2 machine on a rigid
  support reading 2,0 mm/s and 50 um is zone B on one and zone C on the other,
  and it is a C.

  `is_significant_change`, `alarm_limit` and `trip_limit` are the operational
  limits of 5.3 and 5.4 with their numbers. An ALARM is set from the
  established baseline for that position, a quarter of the upper limit of zone
  B above it and never more than 1,25 times that limit, so a quiet machine gets
  an alarm well inside zone B and a machine whose baseline has crept up cannot
  drag its alarm along behind it. A TRIP is about the mechanical integrity of
  the machine, so it comes from the zone and not from the baseline.

  The edition is ISO 10816-3:2009. Part 3 is the one part of the series not
  held here, so the boundaries come from the direct predecessor of the current
  ISO 20816-3:2022, and the guide says so.

- `phonometry.vibration` grades a machine from one broad-band measurement, the
  way ISO 20816-1:2016 asks for it. `evaluation_zone` places a magnitude in
  zone A, B, C or D against the three boundaries of whichever part of the
  series applies, carried by `ZoneBoundaries`; a magnitude that lands exactly
  on a boundary belongs to the zone below. `allowable_velocity` is the
  frequency-shaped criterion of Figure 9 written as Formula (C.1), flat between
  the two corners with constant displacement below and constant acceleration
  above, and `ZONE_LIMIT_FACTORS` moves the one curve onto the three
  boundaries. For a machine no part of the series covers, Table C.1 arrives as
  `TYPICAL_BOUNDARY_LADDER_MM_S` and `TYPICAL_ZONE_BOUNDARY_RANGES_MM_S`, which
  are ranges rather than limits and overlap on purpose.

  `vibration_vector_change` is Criterion II read the way Annex D reads it. A
  change in vibration is a vector, and the annex makes the point with a machine
  that fell from 3 mm/s at 40 degrees to 2,5 mm/s at 180: the magnitude
  improved by half a millimetre per second while the vibration itself moved by
  5,17. The result carries both numbers and plots the two states and the chord
  between them.

- The last step of the VDI 2081 chain, the level the air finally makes in the
  room. `noise_control.room_effect` and `room.steady_state_spl` already carried
  the relation the guideline's Equation (36) states, but written in the room
  constant `R`; the guideline writes it in the equivalent absorption area `A`,
  and the two are not the same number, `R = A/(1 - alpha_bar)`. Each now has
  its own argument and exactly one may be given, so neither can stand in for
  the other by accident. Both also take a directivity factor that varies across
  the bands, because a diffuser is more directional the shorter the wavelength
  and the guideline reads `Q` off a chart against frequency.

  `room.critical_distance` follows, and with `A` and the hemispherical `Q = 2`
  it is the `rH = 0,2 sqrt(A)` that Section 6.7.3 prints for where the
  reverberant field of a ventilation opening begins.

  New `room.sabine_absorption_area`, the familiar Sabine relation read from the
  time to the area rather than the other way about, which is the direction a
  room survey delivers. Its constant follows the speed of sound: 0,161 at
  343 m/s, and the 0,163 of Equation (37), which is the same expression at
  339 m/s.

  Element 20 of the guideline's worked example now comes out of the library:
  five conformance rows, its eight room attenuations, its band levels and both
  of its sums. Figure 30, the chart the directivity comes from, is not
  implemented; the copy in hand does not resolve it to better than about a
  tenth of what it carries, so `Q` stays an input, which is what a caller does
  with a diffuser's own data.

  Two errata, in both languages. The symbol list under Equation (36) sends the
  reader to Equation (36) for the absorption area it is an input to; it is
  Equation (37) or (39). And the English column of the reverberation-radius
  sentence calls a *halbkugelförmig* propagation spherical, where the printed
  0,2 is the hemispherical constant and the spherical one would be 0,141.


- `noise_control.fan_sound_power` takes `model="vdi2081"`, the German method of
  VDI 2081 Part 1:2001-07 Section 4.3, beside the ASHRAE scaling law it has
  always had. The two answer the same question and disagree on how: ASHRAE
  scales the fan's **static** pressure by a table of per-type band constants,
  VDI 2081 scales the **total** pressure rise by twenty and shapes the spectrum
  in closed form, `-5 - 5 (lg St + c3)^2`, from the Strouhal number of the
  impeller speed. Each model takes only the arguments its own standard is
  written on, declared through `typing.overload`, so the two pressures cannot
  be swapped: they are different quantities and the mistake would be worth
  twenty times the logarithm of their ratio.

  A fan is described by one of the three assembly types of VDI 3731 Part 2
  rather than by a fan-type row: radial with rearwards curved blades, a
  cylindrical rotor with forwards curved blades, or axial with a downstream
  diffuser. Each brings its own specific sound power level, its own spectral
  parameter and its own blade-frequency allowance, and a fan of one's own can
  replace the representative level, which the guideline says a real fan can
  exceed by up to 7 dB at the optimum duty point.

- The duct elements of the same chain follow: `unlined_rectangular_duct_attenuation`,
  `unlined_circular_duct_attenuation`, `elbow_insertion_loss` and `split_loss`
  all take `model="vdi2081"`.

  Table 5 of Section 6.1 is a step table in dB per metre, keyed on the largest
  clear side of a rectangular duct or the bore of a round one, where Reynolds
  fits a continuous power law of the perimeter-to-area ratio. A round duct's
  loss depends on its diameter there and does not here, which is why the German
  model asks for one.

  Table 7 of Section 6.2 is printed once, for a 1250 mm side whose limit
  frequency falls in the 125 Hz octave, and carried along the frequency axis
  for every other size: the octave holding `c / (2 a)`, or `0,586 c / d` for a
  round duct, takes the place of the table's own 125 Hz column and the whole
  row moves with it. It also tabulates lining before, after and on both sides
  of a corner separately, which the ASHRAE table does not distinguish.

  Equation (35) of Section 6.4 is the area split alone. Long folds a reflection
  from the change of total section into the same function; VDI 2081 treats a
  junction and a change of section as two elements of the chain. Where the
  branches sum to the feeder area the two agree, and where they do not the
  difference is exactly that reflection, half a decibel in the guideline's own
  third junction.

  Seven further conformance rows, all of them Table 1 of the worked example:
  both straight runs, the bend and its 1245 Hz limit frequency, and the three
  junctions.

- `flow_noise_bend` takes `model="vdi2081"`, which is Equation (18) of
  Section 5.2.2 with the normalised level of Figure 17 and the rounding
  correction of Figure 18. One law covers a junction and a bend: a bend is the
  same expression with the two velocities equal.

  The approach velocity is the one in the duct **ahead** of the junction, so it
  comes from what the feeder carries rather than from the branch's own duty.
  In the worked example the two differ by a factor of four and reading the
  wrong one loses five decibels at 63 Hz.

  `rounding_ratio` defaults to leaving Figure 18 out rather than to nought.
  Figure 18 is drawn for the rounding of a junction and all its curves cross
  zero at `r / d_a = 0,15`, so nought asks for a sharp-cornered junction and is
  worth over 6 dB at 63 Hz. That is how the guideline's own example treats its
  bend, and passing 0,15 lands back on the uncorrected law.

  Both figures state that they hold only above a Strouhal number of one, so a
  band below that returns negative infinity, the level of no contribution,
  rather than an extrapolation: the fit turns over there and its fractional
  power of `lg St` is not real.

  `flow_noise_straight_duct` needs no second model. Bies Eq. (8.251) already
  reproduces Section 5.2.1 and says so in its own docstring; what it lacked was
  an oracle, and it has one now. Eleven further conformance rows, all Table 1
  of the worked example, take the domain to twenty-nine.

- `silencer_self_noise` and `end_reflection_loss` complete the chain.
  `model="vdi2081"` on the first is Equations (46), (49), (50) and (51) of
  Section 7.2.4.2: the A-weighted level from the speed in the clear section,
  the pressure drop and the approach area, shaped by a quartic in `lg St`.
  `method="vdi2081"` on the second is Figure 28 of Section 6.6 in the closed
  form printed beneath it, which knows all four solid angles a nozzle can
  radiate into, from the middle of a room to a corner, and carries the flat
  15 dB ceiling the guideline puts on the theoretical value because the duct
  walls radiate what the nozzle reflects.

  Ten further conformance rows: the silencer's A-weighted level, its
  regenerated noise octave by octave, and the nozzle's reflection. The domain
  now stands at thirty-nine.

- `noise_control.octave_band_limits` turns an A-weighted room requirement into
  a per-octave one by Equation (1) of VDI 2081 Part 2, and
  `VDI2081_SPECTRAL_CORRECTION` publishes the curve it uses. The curve is the
  inverse A-weighting less 5 dB: a spectrum flat in A-weighted terms would earn
  9 dB for the sum of eight octaves, and the guideline takes 5 because the
  noise of an air-conditioning system does not follow that curve.

  Two further conformance rows, one of which is the point of the whole
  exercise: the fan, the splitter silencer and the first junction chained in
  sequence, each attenuating and then adding what it makes, against the running
  total the worked example prints. Every piece is pinned against its own row
  elsewhere; this is the row that fails if they do not compose. The domain
  closes at forty-one.

- Two more entries in `docs/ERRATA.md`, both from Table 1 element 2 of
  VDI 2081 Part 2. The first is substantive: the table prints a hydraulic
  diameter of 0,171 m for the gap between splitters and computes its Strouhal
  row from 0,200 m instead. Both are defensible for that gap, `4 A / P` and the
  parallel-plate limit `2 s`, but only the second reproduces the eight printed
  Strouhal numbers, and with it the whole element falls out to the last printed
  decimal. The second is a cross-reference: the coefficients of the pressure
  drop are credited to Section 7.3.2, which is the airborne sound insulation of
  a building component; they are printed in Section 7.2.3.2.

- `docs/ERRATA.md` records that the two columns of VDI 2081 Part 1 Section 6.4
  contradict each other. The German says the junction's level reduction is
  independent of frequency and the English, on the same page, says it depends
  on it. The German is the authoritative version by the cover of every VDI
  guideline, and it is the one the rest of the document agrees with: Figure 27
  has no frequency axis, Equation (35) contains no frequency, and the worked
  example prints a single number for each of its three junctions.

  Eleven conformance rows, and the VDI 2081 domain is the sixty-fourth. They
  are the guideline's own worked example, VDI 2081 Part 2:2005-05 Table 1: the
  supply air fan reproduces octave by octave, and so do its unweighted 94,1 dB
  and its A-weighted 84,5 dB. Both prints are superseded, by Part 1:2022-04
  and Part 2:2022-10, and neither is held; the pair in hand is what makes the
  example usable, since Part 2:2005 was written against Part 1:2001 and every
  cross-reference in its tables resolves there. The two current editions are
  listed on the support page.

- A gate against the defect those three VDI rows were, `make conformance-rows`.
  A conformance row says what a standard prints and what the library computes,
  and passes when the two agree; a row that works out the expected value itself
  from the same constants is comparing a formula with a second copy of it and
  reports Pass whatever the library does. The three that did could not have
  failed on any defect in the function they certified.

  Counting them is the hard half. A row usually delegates: to a helper beside
  it, to one imported from `..shared`, or to the library under a name that is
  not `ph`. Reading only each row's own body reports a hundred and two rows on
  this tree, of which ninety-five are fine, and a list that long buries the
  seven that are not. So the check resolves the delegation, through helpers and
  across modules and under any import alias, and reports seven.

  Six of the seven are not defects, and the gate has a list where each says so
  in prose: the ground projection and slant length of the ISO/TR 17534-3 test
  geometry pin how this project read the printed coordinates, the library does
  not own the distance between two points; the Doc 29 event assembly checks the
  printed total against the printed segments it is the sum of, which is the
  oracle vouched for before it is trusted; and the DPSS eigenvalue is a check
  of SciPy, deliberately. An entry that stops being needed fails the gate just
  as loudly as a row that is missing one.

- Two rows for the branch split of the SAE Method, replacing the one that
  computed both sides itself. SAE ARP 5534 Section 3.2.2 splits at a mid-band
  attenuation of 150 dB, Equation (7) below it and Equation (8) above, and the
  two are printed to meet there within 0,003 dB. `sae_band_attenuation` takes a
  path length rather than an attenuation, so the split is reached by scaling
  the path: one call at unit length gives the coefficient, and 150 divided by
  it is the length at which the mid-band attenuation lands exactly on the
  boundary. A hair either side picks one branch or the other, and each is now
  judged against the other branch's printed formula, which is what continuity
  at the split means and what neither branch can fake alone.

### Fixed

- The conformance report filed two documents under one name whenever the
  document's number is separated from its issuing body by a space. `ECAC Doc
  29` is the airport-noise method and `Doc 32` the rotorcraft one, and both
  were `ECAC Doc`; `EBU Tech 3285`, `3341` and `3342` were `EBU Tech`;
  `ISO/PAS 1996-3` and `ISO/PAS 20065` were `ISO/PAS`; `Directive (EU)
  2015/996` and `2021/1226` were `Directive (EU)`. The reference parser reads a
  designation as the body plus one glued token, which is right for `ISO 9613-2`
  and stops one token early for a document published in a numbered series, so
  the series prefixes are now recognised the way the bodies already were.

  `ISO 10140-5:2010+A1` was worse: the amendment marker matched no year, so the
  whole of `10140-5:2010+A1` fell into the clause and the document became the
  bare body `ISO`. An edition now takes its amendment.

  The count of distinct normative designations the report publishes goes from
  125 to 130 and the further sources from 84 to 83, which is what the same
  corpus reads as once the eighty-two affected citations name their document. A
  test holds the class shut: a standard's number always follows its body, so a
  designation with no digit whose clause opens with one has lost it.

- The evaluation zones were shaded with the series palette, which told the
  reader the opposite of what the scale says. Zone B, "unrestricted
  operation", came out red and zone C, "limited operation", came out green, so
  the better of the two looked like the alarming one, and zone D, damage, was
  a neutral grey behind both. Four figures did it: the Table C.1 zone
  criterion, the ISO 10816-3 boundary bars, the ALARM and TRIP settings and
  the twelve-month trend.

  A categorical palette on an ordered scale is the defect, so the zones get a
  ramp of their own that runs one way, calm to alarming, and the letter drawn
  in each band and the order of the bands repeat the same information for a
  reader who does not separate red from green. Nothing else in the corpus
  moved: the other 2388 figures reproduce byte for byte.

  The flow-noise figure's own note is redrawn to match the prose above it:
  what Figure 16 prints is a level difference, not a share.

- `flow_noise_straight_duct` was credited to Bies, Hansen & Howard 5e
  Eq. (8.251), a second-hand citation of a law this project holds the primary
  source for. The printed page settles it: Bies gives that equation as "(VDI
  2081-1, 2003)", and VDI 2081 Part 1 prints the two halves in two places on
  the same page. Equation (16) is the overall level, and the level difference
  the octave spectrum needs is printed inside Figure 16 itself, on an abscissa
  of `f_m / v`. The docstring now cites the guideline first and the handbook as
  the place the two are written on one line.

  With that in hand the "does not sum back" sentence gets the reason it was
  missing. The difference Figure 16 prints tends to `-2 - 26 lg 1,14`, which is
  `-2,5` dB, as `f_m / v` falls, so the energy sum grows with however many
  octaves are taken and no fixed shortfall can be quoted. Over the range the
  figure is actually drawn for it lands within half a decibel of the overall.

- Three railway coefficient rows cited `Appendix G` with no document, which is
  the CNOSSOS-EU case the override file exists for: the appendix belongs to
  Annex II of Directive 2002/49/EC and the tables inside each row are owned by
  the 2015 text and the 2021 replacement between them, so no single directive
  is the answer and the string alone cannot say. Two more cited the Bies
  handbook by its title where the other twenty-one cite it as `Bies 5e`, which
  filed one book as two.

### Changed

- Ten visco-thermal models take a `phonometry.fluids.Fluid` where they took
  loose air floats. `johnson_champoux_allard` went from six of them to one, and
  `effective_kappa` from five. The rule is that a function needing **two or
  more** properties of the medium takes the fluid, and one needing only a speed
  of sound keeps the scalar, because a wavelength or a delay wants a number
  rather than a state. Twenty single-property entry points are unchanged.

  `ground_effect`, `barrier_insertion_loss` and the atmospheric parabolic
  equation follow, since they feed the porous ground models. `environment` had
  its own copy of the air, `343.0` and `1.205` under a comment reading "matches
  the materials domain"; it uses the published air now, which is the same two
  numbers.

  The parabolic equation had conflated two speeds. It passed `c_ref`, the value
  the sound-speed profile has at the ground, as though it were the air's own.
  That is right for the ground impedance and wrong as a description, so it now
  builds the caller's air at that local speed explicitly: a temperature gradient
  is the whole point of a refracting atmosphere.

  `thermal_boundary_layer_thickness` and `effective_kappa` do not default to the
  published air. Their air is the ISO 9053-2:2020 Annex A.3 reference state at
  23 degC, whose constants are transcribed Table F.1 cells; recomputing them from
  `fluids.air(temperature_c=23.0, ...)` would move them in the eighth figure. It
  is `materials.ANNEX_A_AIR`, built from those cells rather than recomputed.

  Nothing else moved: the conformance evidence reproduces to the digit, 619 of
  619 checks across 62 domains.

- The same rule reaches `building` and `simulation`. Nine `building` entry
  points, `mass_law_transmission_loss`, `orthotropic_transmission_loss`,
  `in_situ_element` and `in_situ_total_loss_factor` among them, and
  `simulation.far_field_from_contour`, take a `Fluid` where they took a speed of
  sound and a density side by side. `helmholtz_resonance_frequency` keeps its
  scalar, since a resonance of a neck and a cavity needs only the speed of
  sound.

  Each domain publishes the air it is written around rather than repeating the
  numbers: `building.EN_12354_AIR` is the 340 m/s of EN/ISO 12354 Annex A with
  the 1,29 kg/m³ of Annex B, and `simulation.SIMULATION_AIR` is the `AIR`
  material the solvers run in, written as a fluid. Both are `Fluid` values, so
  a caller who has measured their air passes it instead.

  A plate immersed in water is what showed the argument was never "the air":
  two elastic-solver tests hand `mass_law_transmission_loss` the water they
  integrate, and now say so.

  Nothing moved: the conformance evidence reproduces to the digit.

- A `Fluid` requires every property it carries to be a positive finite number,
  and every composition fraction to be finite and not negative, since dry air
  has no water vapour. Ten models used to re-check the same three or four floats
  each; the invariant belongs to the type that owns them, and the guards the
  models dropped are now tested where the check actually lives.

- `membrane_resonance_frequency` reads the ratio of specific heats from the
  fluid it is given. Its isothermal branch divided the air-spring stiffness by
  the published 1.4 whatever fluid arrived, so an air the caller had computed
  came back with the published air's resonance under the caller's density and
  speed of sound, which is neither air. Warm humid air carries 1.3806 and the
  frequency moves 0.70 %. The default air is unchanged, so nothing that took
  the default moves.

- The curated reference table stops offering arguments the signatures no longer
  take. Ten rows still described the floats a `Fluid` replaced, `speed_of_sound`
  and `air_density` among them, so a reader following the table got a
  `TypeError`. The gate asked only that every public name have a row, never that
  a row describe the function it names; it now refuses a row offering one of
  those floats to a function that does not take it.

- `corrugated_plate_stiffness` computes its `Bx` by calling
  `plate_bending_stiffness` instead of spelling out `E h^3 / (12 (1 - nu^2))` a
  second time. `Bx` is that stiffness divided by the profile factor, so the two
  were the same derivation written twice; `Bxz` keeps its own expression, since
  its denominator is `(1 + nu)` and it shares only the `E h^3 / 12`. Results move
  by at most one unit in the last place, from the order of the divisions.

- The three nominal solids of the elastic solver say where they stand. They
  carry **bulk** wave speeds, which is what the solver integrates, consistent to
  a per cent or two with the textbook modulus, Poisson's ratio and density of
  each material, and transcribed from no standard.

  The note also records the citation not to make. EN 12354-1 Table B.3 is the
  obvious candidate and the wrong one: it tabulates the quasi-longitudinal phase
  velocity of a *plate*, a different quantity that runs eleven per cent below
  the bulk speed for steel and fifteen for aluminium, and it lists neither steel
  nor aluminium at all. A test holds the three constants to the bulk formulas
  and to being above the plate ones, so re-sourcing them from that table would
  fail rather than quietly change the physics.

### Fixed

- The Spanish edition of a figure follows the page, not the author. `ThemeImage`
  derived the dark variant of whatever URL it was handed and nothing else, so a
  Spanish page had to name its own `_es` file, and nine of them did not: the
  caption was in Spanish and the axis labels underneath it were in English.
  `Video.astro` had always read the page's language and added the suffix
  itself; the figure component does the same now, so a Spanish page and its
  English twin carry the identical `src` and a translator copying the English
  line cannot get it wrong. The 687 references that spelled `_es` no longer do,
  and passing one that still does is refused at build time rather than
  resolving to `_es_es`.

- `vibration.industrial_machine_zone` grades a run of measurements one reading
  at a time. It wrapped `evaluation_zone`, which has always taken an array, and
  then applied the most-restrictive rule of 5.2.3 by comparing the two gradings
  as text: with an array in, that text was the array's own repr, so a month of
  daily readings came back as the thirteen-character string `"['A' 'C' 'D']"`.
  The rule is applied to the zone indices now and the letters looked up once at
  the end, so an array in gives one letter per reading and a scalar still gives
  a letter. A scalar and an array together broadcast, which is the shape of
  judging a run of velocities against one displacement.

- The change-of-section formula is published with its fraction. A form feed had
  reached the docstring, and in a raw string it is exactly the backslash and the
  `f` of the command that follows it, so `\frac` became `rac` and the API
  reference printed `10 log10` with the ratio dropped underneath it as prose.
  A new gate, `scripts/check_control_characters.py`, refuses the whole class:
  no C0 control character other than tab and newline anywhere in the tracked
  text corpus.

- `noise_control.section_change_loss` asks for a duct size only when it uses
  one. Figure 26 prints "no effect" in the frequency column of a sudden
  reduction, so no limit frequency and no duct dimension enter that answer; the
  function demanded `upstream_size` for a rectangular duct whatever the ratio,
  and for a reduction any value at all gave the same spectrum. It is required
  now only for the sudden increase, which is the case that has a limit
  frequency. A round duct's diameter and its area are one fact stated twice,
  and giving both now has to give the same duct: the area set the ratio and the
  size set the limit frequency, so a contradictory pair silently returned a
  different spectrum instead of being refused.

- `vibration.gear_housing_velocity_limit` names its own parameter when it
  refuses a rating. It passed the rating straight to `allowable_velocity`, so
  a bad one came back as `'constant_velocity_mm_s' must be positive`: a keyword
  that is in neither its signature nor its documentation, and that nothing
  written against this API could match. Its sibling already validated locally.

- Nine figures on the Spanish pages of this batch pointed at the English artwork. `ThemeImage`
  derives the dark twin of the URL it is given and nothing else, so a Spanish
  page has to name its own `_es` variant, and these did not: the caption was in
  Spanish and the axis labels underneath it were in English.

- Seven statements in the guides and the module comments said something the
  printed page does not. The VDI 2081 fan caption claimed eight decibels
  between a radial machine and an axial one where its own figure prints 94 and
  106; the errata summary counted four VDI defects where the register holds
  five, and described the Section 6.4 mistranslation as being about a duct's
  attenuation when it is about whether a junction's reduction depends on
  frequency at all; the ISO/TR 17534-3 geometry was given as 195 m where the
  report prints 194,16; the 50 MW cap on ISO 10816-3 group 1 was credited to
  clause 4.2, which states no upper bound, instead of to the title of Table
  A.1; `hvac.py` credited the spectral parameter and the blade-frequency
  allowance to VDI 2081 Section 4.3.3, and both are printed in 4.3.4; and the
  note about the three `L_WSM` values being inside their stated tolerance holds
  for two of the three, assembly T being inside the English tolerance and
  outside the normative German one, which the comment now says.

- The ISO/TR 17534-3 errata entry got its Status line back, in both languages.
  Inserting the two VDI 2081 entries above it took the closing line with them,
  so the entry ran straight into the next heading.

- One term, one word, on the Spanish pages: the tolerance band of ISO/TR
  17534-3 is `margen` throughout rather than `sobre` in one section and
  `margen` in another, and the established baseline of ISO 20816-1 is
  `referencia` throughout rather than `línea base` where the criterion is
  introduced and `referencia` where the alarm is set from it.

- Two citations in the machine-vibration module named clauses that do not
  exist. The measurement bandwidth of ISO 10816-3 is stated in its Annex A,
  which has no subclause A.1, and the ladder ISO 20816-1 Table C.1 is drawn on
  is an R5 series, which is what a step of about 1,6 is; R10 steps by 1,26. The
  VDI 2081 reference note named Tables 5 and 8 for the duct attenuation, and
  Table 8 is the sound reduction index of building components: the duct tables
  are 5 and 7.

- `phonometry.environment` publishes all six digits of the CNOSSOS-EU track
  descriptor. `TrackMeasure` (digit 4, the additional measure: rail damper, low
  barrier, absorber plate, embedded rail) and `RailJoints` (digit 5) were
  defined beside the other four and left out of every `__all__` on the way up,
  so `TrackDescriptor.from_code("BMSDSH")` handed back a `TrackMeasure` that the
  caller who received it had no supported way to import.

  The round-trip test had the same hole, asserting digits 1, 2, 3 and 6 and
  skipping the two it could not name either, which is how the gap survived: the
  reference table documented both enums, and the gate behind it asks only that
  every published name have a row, never that every documented row name
  something a reader can reach.

  A new architecture test closes the class rather than the case. It walks the
  fields of every published dataclass and fails on any field whose type is
  defined in the tree and published by no package, so the next type that can be
  received but not named fails the build instead of the reader.

### Removed

- The top-level `phonometry` package publishes twenty-three names: the
  nineteen domain packages, and `Signal`, `ReportMetadata`, `PhonometryWarning`
  and `__version__`. The thirteen hundred functions, classes and constants it
  used to re-export are reached through the package that owns them, so
  `from phonometry import leq` and `phonometry.leq` raise where
  `from phonometry import signals` and `signals.leq` work. Nothing was removed
  from the library: every name is where it always was, and the shortcut is
  what went.

  The four that stay are the ones that belong to no domain. `Signal` is the
  calibrated recording seven packages accept and `phonometry.io` owns;
  `ReportMetadata` stamps the accredited fiches of every domain;
  `PhonometryWarning` is the base of the twenty-seven warning classes across
  thirteen packages, so one `warnings.filterwarnings` rule still reaches them
  all. Measured rather than assumed: of the three hundred and sixty-seven
  public classes, six are imported across a boundary between two public
  packages, and only `Signal` by more than two.

  What the flat namespace cost is easiest to see in the one name it forced a
  rename on. Three domains define an expanded uncertainty over their own
  coverage factors, and one flat namespace holds one spelling, so the
  environmental one was published as `environmental_expanded_uncertainty`, a
  name that existed for no reason but the collision. It is
  `environment.expanded_uncertainty` now. Three more paid the same toll by
  being left out: `hearing` publishes `AUDIOMETRIC_FREQUENCIES`, `FIELDS` and
  `SEXES`, which the top level declined because `FIELDS` and `SEXES` cannot be
  sustained in one flat namespace. All four are ordinary names again.

  To migrate, read the package off the function's page in the generated
  reference, or off the curated table in `docs/reference/api/index.md`, whose
  usage column now shows the namespace to call through. Every name belongs to
  exactly one package, so there is nothing to choose.

- The compatibility layer that served the module paths of every earlier layout
  is gone, and with it `phonometry.environmental`. The hundred and five paths
  the 4.0 taxonomy moved now refuse in the three ways Python has, which are
  worth naming separately because they are not interchangeable: `import
  phonometry.metrology.levels` raises `ModuleNotFoundError`, `from
  phonometry.hearing import sti` raises `ImportError` because the package
  itself imports and only the name inside it is missing, and
  `phonometry.metrology.levels` read as an attribute raises `AttributeError`.
  The shims announced their own removal for 5.0;
  they go now because 4.0 is the release that breaks, and carrying a table of
  105 renamed paths through a whole major version protects nobody: every
  documented path in 4.0 is the real one.

  `_warn_renamed`, the helper behind the notices, goes with them: it had no
  caller left. `PhonometryWarning` is what remains of that module, which is
  what it was there for.

  One import inside the library was resolving through the shims rather than
  through the real module, and only the type checker knew: the intensity-class
  figure read `residual_index_limits`, `residual_index_from_phase_mismatch`
  and `intensity_class_compliance` off `metrology`, which stopped defining
  them when the IEC 61043 check moved to `emission` in 4.0. It reads them from
  `emission` now. Nothing else in the tree, the tests or the documentation was
  on a moved path.

### Added

- `environment.region_ground_factors` and `environment.mean_path_height`, the
  two reductions ISO 9613-2 leaves to the caller. The general ground method
  wants one factor `G` per region and a real path crosses several kinds of
  ground; the alternative method wants the mean path height `hm`, and a terrain
  model holds heights rather than areas. Neither reduction is written down in
  ISO 9613-2, so two implementations that read it differently disagree on the
  same site, which is exactly what ISO/TR 17534-3:2015 exists to stop.

  `region_ground_factors` takes the ground projection cut into segments and
  averages `G` over each of the three regions weighted by length, as 6.2.5 of
  that Technical Report settles it. `mean_path_height` takes the ground profile
  as a polyline and integrates the area between the straight ray and the ground
  under it, which is what Figure 3 of ISO 9613-2 draws and never writes.

  With the two in place the seven quality-assurance test cases the Technical
  Report marks as answerable by ISO 9613-2 alone (T01 to T07) come out of the
  library end to end, band by band and on both totals, inside the +/-0,05 dB
  envelope it declares. They are a kind of oracle the report did not have: a
  whole chain with every intermediate quantity printed, built to expose the
  disagreements between implementations that a single formula never does.
  Thirty-four rows, its own domain, and the run goes from 664 to 698 checks.

  T08 to T19 are not attempted. They build their ray paths by the additional
  recommendations of Clause 5, over barriers and around buildings, which is
  geometry rather than acoustics and would be public surface with no oracle
  behind it.

- `underwater.array_directivity_index` and `underwater.detection_threshold`,
  the two terms of the sonar equation that had to be supplied from outside.
  `passive_sonar_equation` and `active_sonar_equation` accepted `DI` and `DT`
  as numbers and had no way to produce either, so the equation stopped one step
  short of an answer.

  The directivity index of an unshaded line array comes from Ainslie,
  *Principles of Sonar Performance Modelling* (Springer 2010), Equations (6.49)
  to (6.57) on printed folio 267: the beam's solid-angle footprint in closed
  form, through the sine integral. It is also the **array gain** whenever the
  noise is isotropic and the signal a plane wave, which is the case the sonar
  equation is written for, so the one function answers both terms.

  Checked against the three limits Section 6.1.2.1 states, which are exact
  rather than fitted: 10 log10(2L/lambda) at high frequency away from endfire,
  10 log10(4L/lambda) near endfire where the footprint halves, and 0 dB once
  the array is much shorter than a wavelength. And cross-checked against the
  book's own closed-form approximation, Equation (11.20), which Figure 11.1
  plots against the same integral: the two agree within half a decibel across
  the range that figure covers, which is an independent check rather than a
  restatement, since the approximation is fitted and the implementation
  derived.

  The detection threshold at 50 % detection probability comes from
  Equation (11.22) on printed folio 581. The logarithm inside it is base two,
  not a square: the two are typeset alike, they differ by half a decibel at
  p_fa = 1e-4, and the distinction has a test of its own. A false-alarm
  probability of one half is refused rather than returned, since the threshold
  diverges there: half the empty beams are already being declared detections.

  Four conformance rows, and the sonar domain is the sixty-third.

- `fluids.sea_water()` and `fluids.sea_water_density()` give the library a
  density of sea water, which it had nowhere. The density implements Ainslie,
  *Principles of Sonar Performance Modelling* (Springer 2010), Equation (4.6)
  on printed folio 127, which that book attributes to Pierce (1989, p. 34). Three conformance
  rows pin it: the 1027 kg/m3 the book gives for the standard ocean, the
  atmosphere its Equation (4.11) puts at the surface, and the 0,043 855 kg/m3
  the pressure term is worth.

  The four sound-speed equations move here with it, because a medium's own
  properties belong with the medium: `fluids.sea_water_sound_speed` is the
  UNESCO, Del Grosso, Mackenzie and Medwin fits, unchanged, and it now accepts
  a column as readily as a point. `underwater.sound_speed_profile` and
  `SoundSpeedProfile` stay where they are: a profile describes a place rather
  than a substance, and it belongs with the marcher that walks it. It asks for
  the whole column in one call now instead of restating the model dispatch and
  the pressure conversion a second time.

  The two depth-to-pressure conversions are told apart by name, because they
  are not the same quantity. `depth_to_gauge_pressure_mpa` is Leroy & Parthiot,
  gauge, in megapascals, zero at the surface, and it is what the UNESCO and Del
  Grosso sound speeds want. `depth_to_absolute_pressure_pa` is Ainslie's
  Equation (4.11), absolute, in pascals, one atmosphere at the surface, and it
  is what the density wants. Between them lie a factor of a million and an
  offset of an atmosphere, so each name carries its unit and its datum.

  Sea water has no ratio of specific heats, viscosity or thermal diffusivity
  here, because no source in this library prints one. Reading any of them
  raises `FluidPropertyUnavailable` naming the model, which is the case that
  accessor was built for and the reason air and water can share a type at all.

- `phonometry.fluids`, a twentieth domain package holding the state of the
  medium a sound travels through. `fluids.air()` computes humid air from
  IEC 61094-2:2009 Annex F, the CIPM-2007 formulation, and returns a frozen
  `Fluid` carrying the density, speed of sound, ratio of specific heats,
  viscosity and thermal diffusivity Table F.1 tabulates, the thermal
  conductivity and specific heat capacity Clause F.6 gives expressions for, and
  the characteristic impedance, Prandtl number and kinematic viscosity those
  close by identity. All ten printed figures of Table F.1 reproduce, at both
  printed condition sets, inside the rounding of the last figure the annex
  gives; three conformance rows say by how much.

  Every domain may import it without an architecture edge, as they already
  import `filters`, `signals` and `metrology`. A medium is not a domain of
  application but something every domain needs, and eleven identical edges
  would have recorded nothing. That is the fourth transverse package, and the
  first added since 4.0 split the original one in three.

  Three conditions that each exist can still describe air that does not, so
  `air()` refuses the combination as well as the arguments. At 20 degC and
  1 kPa, 50 % relative humidity asks for a water vapour mole fraction of 1,17,
  which is more water vapour than total pressure; every argument passes its own
  guard, and without one on the fraction the CIPM equations carry on and return
  a density and a speed of sound that look like measurements. Saturated air at
  60 degC reaches 0,198 and passes, so the bound is the only one that can be
  set: a mole fraction cannot reach 1.

  The assumption warning is now raised after the refusals rather than before
  them. A caller who promotes it with `simplefilter("error", ...)` used to
  receive it in place of the `ValueError` an invalid humidity earns, and never
  learn which argument was wrong.

  It is frozen through, not only at the surface. `frozen=True` stops the
  attribute from being rebound and says nothing about what it points at, so the
  two mappings are copied and wrapped on construction. Without that, one
  `air.properties["density"] = 999` anywhere in a process would have moved
  every model defaulting to a shared state, silently and for the rest of the
  run.

  The `Fluid` says what it was computed for and what fixed it: the conditions,
  the composition, the model by name, and the domain that model states for
  itself. Reading a quantity the model does not determine raises
  `FluidPropertyUnavailable` naming the model, rather than returning a
  plausible number nobody printed. This matters as soon as a second fluid
  arrives: sea water has no ratio of specific heats to give.

  `temperature_c` is required, because there is no defensible default for the
  one condition a caller has actually measured. The pressure and the humidity
  default, and say so once in a single `FluidAssumptionWarning` naming both and
  what each is worth: a site 1000 m up sits near 90 kPa, and taking that for
  one standard atmosphere puts the density about 13 % high, while the whole
  span of humidity is worth about 1 % of it. Supplying both is silent, which is the
  property a caller who measured their air should get. The carbon dioxide mole
  fraction defaults without warning, because 0,000 4 is a value Clause F.2
  names for laboratory conditions rather than a guess about the caller's air.

  Nothing is refused but what cannot exist. Annex F states 15 °C to 27 °C,
  60 kPa to 110 kPa and 10 % to 90 % relative humidity, and a state outside
  that warns and still answers: the annex states where its equations were
  validated, not what air can be. Air at 60 °C in a duct is real.


- `hearing.assumed_protection_value()`, `hearing.octave_band_protected_level()`,
  `hearing.hml_rating()`, `hearing.hml_protected_level()`,
  `hearing.snr_rating()` and `hearing.snr_protected_level()` estimate what a
  hearing protector leaves at the ear, by the three methods of ISO 4869-2:2018.

  A protector is measured on people rather than on a coupler, so what
  ISO 4869-1 returns is a distribution: one attenuation per subject per octave
  band, with a spread often a third of the mean. Clause 5 refuses to quote the
  mean and every method here starts from the assumed protection value, the mean
  reduced by a multiple of its own spread, with all seven protection
  performances of Table 1 available. The octave-band method subtracts it band
  by band from the A-weighted spectrum; the HML method collapses the protector
  to the three attenuation values a reference noise of stated C minus A gives,
  and applies them through two straight segments meeting where the medium value
  is defined; the SNR method collapses it to one number against a pink noise.
  The two rating methods consume their ratings as the standard publishes them,
  rounded to the nearest integer, which is what makes the worked example's
  predicted reduction come out at 22,5 dB rather than at the unrounded fit, and
  the reported levels round halves away from zero rather than to even.

  The four informative annexes are one example carried through all three
  methods, and every printed number of it is reproduced: the per-band mean and
  spread of Annex A, the band table and 81 dB of Annex B, all sixteen HML
  triples with their statistics and the 82 dB of Annex C, and all sixteen
  single-number ratings with both applications of Annex D. Annex A's own
  assumed-protection row is the difference of the rounded intermediates it
  displays and sits 0,1 dB from the formula applied to the data in three of its
  eight bands, which the documentation states and the suite pins both ways.

  One defect of the printed standard joins the errata registry: Annex C
  reprints the normative Table 2 as its Table C.1 and the reprint disagrees in
  two cells. Table 2 is the reading that reproduces the annex's own results,
  and the suite computes the same row from the reprint to show it missing
  thirteen of the sixteen printed values.

- `building.low_frequency_intensity_reduction()` measures sound insulation
  with sound intensity down to 50 Hz by the method of ISO 15186-3:2002, where
  the source room is too modal for a space average to describe the field
  driving the specimen. The level that goes into Formula (7) is read on the
  surface of the test specimen itself, from microphones within 50 mm of it, so
  the formula subtracts nine decibels and not the six of part 1: close to a
  rigid boundary a diffuse field carries twice the mean-square pressure it
  carries away from one, and a microphone that close reads the 3 dB of it. The Clause 6.4.2
  qualification comes back per band against whichever of its two limits
  applies, 10 dB for a sound-reflecting specimen and 6 dB for one presenting a
  sound-absorbing surface in the receiving room, and a refused band is flagged
  rather than dropped, because the standard's answer to it is to improve the
  measurement environment. Both levels of that indicator are read on the
  measurement surface in the receiving room, which is not where Formula (7)
  reads its own, so they are a separate argument (`l_p`); Clause 6.4.2 asks for
  them only "if possible", so without them the result reports no indicator and
  no verdict instead of guessing one. Clause 6.6 admits 50 Hz, 63 Hz and 80 Hz
  and optionally 100 Hz, 125 Hz and 160 Hz, and nothing outside that range.

  `building.low_frequency_element_normalized_difference()` is Formula (8), the
  same method for small building elements normalized to the 10 m2 reference
  absorption area. It settles a disagreement inside the series: the unit-count
  term reaches the result added here, which is the sign this library derives
  and the opposite of what ISO 15186-1 prints for the same quantity, so the
  errata entry standing against part 1 now cites part 3's own page.

  `building.limp_panel_reduction_index()` is Annex A, which is normative and
  is how a laboratory qualifies itself: the calculated index of a limp panel,
  mass law reduced by the radiation efficiency of forced transmission, with
  the air's characteristic impedance and speed of sound taken at the climate
  of the test, against which a measurement must agree within 4,0 dB from 50 Hz
  to 160 Hz. Its oracle is the plaster-board column of Table A.1, whose six
  printed values reproduce from the stated inputs to the 0,05 dB of their
  printed resolution. The steel-sandwich column beside them does not: no reading of the
  inputs printed beside it reproduces it, and the one that would takes an area
  and a surface mass moved together, the mass landing heavier than solid steel
  of the stated thickness. It joins the errata registry and is deliberately not
  used as an oracle.

  No single-number rating accompanies either result: six bands cannot feed
  ISO 717-1.

- `emission.sound_power_in_duct()` determines the sound power a fan radiates
  into its duct by the in-duct method of ISO 5136:2003, the route for a fan
  that cannot be run unducted. It takes the one-third-octave level a shielded
  microphone read at three circumferential positions of an anechoically
  terminated test duct (or the multiplexed or traversed level), adds the
  combined correction C = C1 + C2 + C3,4 of clause 8 and turns the result into
  sound power through the plane-wave relation of Eq. 12, with the duct air's
  rho c against the 400 N s/m3 reference. C1 and C2 are the microphone's and
  the shield's own responses, supplied per band; C3,4, the combined mean flow
  velocity and modal correction, is computed: the Annex A polynomial in the
  signed mean flow velocity for the sampling tube, from the Tables A.1 to A.6
  that cover 0,15 m to 2 m ducts, and the convective term of Eq. 8 for the
  nose cone and the foam ball. `flow_modal_correction()` gives that correction
  on its own, the A-weighted total follows Annex C over the 27 bands of
  Table C.1, and the result carries the reproducibility of Table 2 doubled
  into the 95 % statement of clause 9.2 (`in_duct_reproducibility()`), with
  every band the standard gives for information only, above 10 kHz or beyond
  40 m/s, flagged as such. The two informative extensions of the Annex A
  footnote do not compose, and neither do they here: the 12,5 kHz to 20 kHz
  rows are printed under a band header of their own that reads |U| <= 40 m/s,
  so a velocity past 40 m/s is refused as soon as a band above 10 kHz is
  asked for, rather than answered from a polynomial with nothing behind it.

  The oracle is Table D.1: all 162 printed values of C3,4 for d = 0,5 m at
  +/-5, +/-15 and +/-30 m/s reproduce from the Table A.4 coefficients to the
  printed 0,1 dB, in the suite and in the conformance report, beside the
  worked example of Eqs (D.2) and (D.3), Eq. 8 at the nose cone's limit, the
  two terms of Eq. 12 and Tables 2 and C.1. Eight defects of the printed
  standard join the errata registry, among them a coefficient of Table A.5
  printed with its leading digit missing, which the library reads from the
  neighbouring tables and says so, and the $a_9$ column of Table A.2 headed
  with a stray subscript zero. The loose prose of 5.3.4.3, which calls
  negative a correction whose equation is positive on the outlet side, is
  filed among the source properties that are not errata: the closing sentence
  of its own paragraph reconciles the two.

  Two things the standard leaves the reader to work out are said at the point
  of use rather than left in a table. Asking for the 5 000 Hz band of the
  table that serves 0,8 m to 1,25 m warns that its $a_3$ is the one
  coefficient of Annex A that is read rather than transcribed, since it moves
  the correction by 0,64 dB per unit of the missing digit at 40 m/s. And a
  determination through a nose cone or a foam ball warns that the
  reproducibility it carries is the sampling tube's, which clause 4 NOTE 5
  says "can be expected to increase for other shields" without putting a
  number on the increase, so the figure is a lower bound.

- The sound power of a machine where it works, by comparison with a reference
  sound source: `emission.sound_power_in_situ` implements ISO 3747:2010 in
  octave bands, correcting the background at every microphone position with
  Eq. 7 and the three rules of clause 8.1 (nothing above a 15 dB margin,
  Eq. 7 between 6 dB and 15 dB, a 1,3 dB cap below 6 dB that turns the band
  into an upper bound the result flags band by band), energy-averaging the
  corrected levels (Eq. 8 to 10) and taking the difference to the calibrated
  power of the reference source for one location (Eq. 11) or for several,
  each energy-averaged first (Eq. 12). An impulsive source goes the same way
  to a sound energy level through `emission.sound_energy_in_situ`, from
  events measured one at a time (Eq. 13 to 15) or once over N of them
  (Eq. 16, 17), averaged over the positions (Eq. 18) and compared (Eq. 19,
  20); Eq. 14 is applied as the whole ISO 3740 family prints it, with an
  `integration_time` to carry the time-averaged background to the event's
  interval where a margin is wanted like for like. The result carries the
  Table 2 grade with its typical reproducibility (1,5 dB engineering, 4,0 dB
  survey), the clause 9 uncertainty (Eq. 22, 23, pinned to the 9.5 EXAMPLE's
  U = 5 dB), the Annex C radiation-impedance correction shared with ISO 3741
  and the Annex D A-weighted total from the Table D.1 corrections, which are
  the ISO 3744 Annex E octave values digit for digit.
  `emission.excess_sound_pressure_level` evaluates the Annex A indicator that
  decides the grade and `emission.static_pressure_from_altitude` Eq. C.2. Five
  defects of the informative Annex E join the errata registry: the sign of
  the direct-field level in E.4.2.6.2, the 0,6 dB altitude correction of
  E.4.2.5 that Annex C's own equation puts at 0,26 dB, the excess that lost
  its delta in the Table E.2 sensitivity coefficient, the sampling
  coefficient of that same table, which E.4.2.6.3 and the budget of E.4.2.12
  both put at 1 where the table prints 0,5, and the equation E.4.2.3 sends
  the reader to for a derivative that is taken of another.
- The sound energy level $L_J$ of a noise burst or transient emission, the
  quantity ISO 3744 and ISO 3741 name in their titles beside the sound power
  level and determine in the clauses the library had left out. A press stroke
  or a door slam has no steady power to report, so clause 3.23 gives it the
  energy it radiates, $L_J = 10 \lg(J/J_0)$ re 1 pJ, and clause 8.3 of ISO 3744
  (clause 8.4 of ISO 3746 for the survey grade) determines it by the
  enveloping-surface chain with the single event time-integrated level $L_E$
  in place of the time-averaged $L_p$: `sound_energy_pressure` takes the
  events measured one at a time (Eq. 19) or one reading encompassing them
  (Eq. 20), the $K_1$ of Eq. 21, the $K_2$ of Annex A and the surface term of
  Eq. 23, and returns a `SoundEnergyResult` with the Annex E $L_{J\mathrm{A}}$
  and `.plot()`. In the reverberation room, `sound_energy_reverberation`
  (Eq. 30) and `sound_energy_comparison` (Eq. 31) are clause 9.2 of ISO 3741,
  with the per-position $K_{1i}$ of Eq. 25, and `octave_band_levels` is the
  Annex F sum of thirds into octaves (Eq. F.1/F.4). `mean_single_event_level`
  is the Eq. 19/20 reduction on its own, and `reference_atmosphere_correction`
  the Annex G correction of either level to 101,325 kPa and 23,0 °C, from a
  measured static pressure or from the altitude through Eq. G.2.

  Neither standard prints a worked example with $L_J$, so the oracle is the
  definitions of clause 3: for a source steady over $T$ the integral of $p^2$
  is $T p^2$, $L_E = L_p + 10 \lg(T/T_0)$ (clause 3.4 NOTE 1), and every
  equation of the energy chain being its power twin gives
  $L_J = L_W + 10 \lg(T/T_0)$ with the same $K_1$, $K_2$, A-weighting and
  uncertainty. The suite and the conformance report pin that identity on
  both families with background and room in play, Eq. 20's $10 \lg N_\mathrm{e}$,
  Annex F's $10 \lg 3$, and the H.4.2.7 statement that the Annex G
  correction vanishes at 120 m and 23 °C.

  One reading had to be decided. Eq. 21 subtracts a time-averaged background
  level (re $p_0^2$) from a time-integrated event level (re $E_0 = p_0^2
  \cdot 1$ s), which is a ratio of energies only when $T = 1$ s. The library
  compares the background as its exposure over the same window,
  $L_{p(\mathrm{B})} + 10 \lg(T/T_0)$, so that the identity above holds for
  every $T$, and asks for `integration_time` with the background of the source
  under test, the only one that is an event level; the reference source of the
  comparison method is steady, so its own background is corrected by the
  time-averaged rule of 9.1.2 and needs no window;
  the errata registry records the reading, for ISO 3744 Eq. 21 and the same
  construction in ISO 3741 Eq. 25.

  Two figures the informative Annex H of ISO 3744 prints for the size of the
  Annex G correction are wrong, and both are now in the errata registry. The
  clause says the correction reaches 0,6 dB at 500 m altitude, where the
  normative Annex G it points at gives 0,394 dB at 23,0 °C; the zero the same
  sentence prints for 120 m comes out exact, which is what shows the equations
  are being read as intended. The clause then divides that 0,6 dB by the square
  root of 6 and prints 0,3 dB, which is 0,6/2. Neither figure feeds anything
  the library publishes, since the Annex H budget is not modelled.

  Bringing the reverberation-room route to the sound energy level regrouped the
  terms the mean room level is raised by, so that they are summed as Eq. 30
  brackets them rather than left to right. That is a reassociation, and it
  moves the sound power level in the last bit: 34 of 84 bands over four cases
  here, by at most 1,4e-14 dB. One committed conformance residual, the known
  $L_W$ of ISO 3741 Eq. 20, goes from 1,42e-14 to zero for the same reason. No
  figure in the corpus moves.

- The ISO 16283 low-frequency procedure, which the standard makes **mandatory**
  and not optional when the receiving room is under 25 m3, taking the volume
  rounded to the nearest cubic metre with halves going up, so 24.4 m3 triggers
  it while 24.5 m3 and 25 m3 exactly do not. That is most bedrooms and
  bathrooms in real dwellings, and until now the field-measurement entry points
  answered for those rooms without it. The one exception is a road-traffic
  facade: Clause 6 of ISO 16283-3 gives those methods the default procedure and
  nothing else, with no corner measurement at any volume.
  `LowFrequencyProcedure`
  carries the corner measurements and the 63 Hz octave reverberation time,
  `apply_low_frequency_procedure` returns a `LowFrequencyResult`, and
  `impact_insulation` and `facade_insulation` take it through a
  `low_frequency` argument, while `airborne_insulation` takes one per room of
  the pair, `source_low_frequency` and `receiver_low_frequency`.

  ISO 16283-1 Clause 8.5, ISO 16283-2 Clause 8.5 and ISO 16283-3 Clause 7.3.4
  print the same procedure, so it is implemented once, and the four places the
  parts genuinely differ are kept apart rather than flattened: which rooms it
  applies to, which methods may use it (Part 3 Clause 6 confines it to the
  loudspeaker methods), whether the energy mean over source positions is a
  numbered formula, and which quantities the 63 Hz octave time feeds.

  A room that triggers the procedure but is measured without it now warns
  instead of answering silently. It warns rather than refuses because the
  default-procedure spectrum is what the ISO 16283 one is compared against: the
  standard says the measurement shall include the corners, not that a
  calculator must refuse arithmetic. On the guide's 23 m3 bedroom the silence
  was worth 3.9, 4.2 and 3.6 dB and moved C_50-3150 from -1 to -2, so it
  changed the rating.

- `speech.sti_adjusted_for_levels()` and `STIResult.adjusted_for_levels()`
  answer what a hall measured empty would have scored with the audience in it
  and the announcement at its operational level. A room is measured out of
  hours at whatever level the test signal ran, and the rating it has to meet is
  the one for the room in use; this is the four-step procedure of
  IEC 60268-16 Annex M, which moves the measurement to that condition without a
  second visit. The result is an ordinary `STIResult`, so `.plot()`,
  `.report()` and `.rating` all read it.

  The method form takes only the target condition, because the result now
  carries the measurement condition itself: `STIResult` gained
  `ambient_levels` beside the `band_levels` it always echoed, and it refuses a
  result computed without `level=`, which carries at most the flat `snr=`
  noise factor and none of the spectra the level-dependent corrections must
  be re-derived from. The function form takes the matrix and all four spectra,
  for a matrix that came from somewhere else.

  Steps 2 and 3 of the annex are one clause A.5.3 correction, inverted and then
  applied, so the auditory masking of Table A.2 and the reception threshold of
  Table A.3 keep the single implementation the forward chain already used and
  the adjustment cannot drift away from the measurement it adjusts. A round
  trip to the levels already measured returns the matrix.

  Checked against the printed intermediates of the annex's Table M.1 worked
  example rather than only the STI it ends on: both 98-value MTF matrices land
  within one unit of the last printed place, every scalar row on the way to
  them reproduces at the precision it is printed to, the effective SNRs agree
  within 0.08 dB, the band MTI row is exact at the annex's own two decimals and
  the index rounds to the printed 0.76. The only prints not used as oracles are
  the step 4 clamp and transmission-index matrices, which the annex derives
  from its own rounded SNR print. Three defects that reading turned up in
  the printed table are recorded in `docs/ERRATA.md`.

  What is implemented is the Edition 4 (2011) procedure. The Edition 5 foreword
  says greater information is given in Annex M about these adjustments, and its
  table of contents grows the annex from three printed pages to ten; the body
  of the Edition 5 annex could not be obtained, so the guides say exactly that
  instead of implying coverage of an edition nobody read.

- `verify_weighting_class(wf, edition="1979")` grades a frequency weighting
  against IEC 651:1979 Table V, which publishes the laboratory-grade **Type 0**
  mask that IEC 61672-1:2013 has no equivalent for, plus Types 1, 2 and 3. The
  argument is the one `verify_filter_class` already takes to reach the
  IEC 61260:1995 class 0, and `weighting_class_limits` takes it too; class `N`
  is the standard's instrument Type `N`, so a verdict carries a
  `margin_class0_db` through `margin_class3_db` per band and for the
  between-nominals sweep. The default stays `"2013"` and its output is
  unchanged.

  The two masks are not the same mask wearing different names, which is the
  reason for carrying both. Type 0 holds +2/-3 dB at 16 kHz and at 20 kHz,
  where class 1 opens to +2.5/-16 dB and imposes no lower limit at all: the
  plain bilinear A design at 48 kHz sits 15.7 dB low at the 20 kHz row, earns
  class 1 for it, and is refused Type 0. The fitted A and C designs earn
  Type 0 at 32, 44.1, 48, 96 and 192 kHz, binding on the +/-0.7 dB pass-band
  cells with about 0.65 dB to spare. The edition covers A, B and C, the
  weightings IEC 651 defines; its Table V footnote makes one mask govern every
  weighting characteristic, so B is held to those limits rather than borrowing
  the ANSI ones. The IEC 651 and ANSI S1.4-1983 Type 0 columns stay two
  separate transcriptions: they disagree at 10, 12.5 and 16 Hz, where ANSI is
  two-sided and stricter and IEC 651 is upper-only.

- `Signal.pick` and `Signal.crop` narrow a recording to a set of channels or
  to a span of seconds and keep it a recording. Indexing a `Signal` yields its
  samples, which is what makes it a drop-in for the array it replaced, and the
  cost is that `sig[0]` drops the rate, the calibration factor and the channel
  labels. These two keep all three. `crop` is half-open like a Python slice,
  so cutting a record in two at the same instant partitions it with nothing
  counted twice.

- `Signal.plot(scale="db")` draws the magnitude of each sample as
  `20 lg(|p| / 20 uPa)`. That is a waveform in decibels and not a sound
  pressure level: an `L_p` is defined on a mean square over a stated time
  weighting, which is what `time_weighting` produces and what
  `TimeWeightedEnvelope.plot` draws. It needs a calibrated record and refuses
  an uncalibrated one by name rather than counting decibels from nothing.

- `TimeWeightedEnvelope`, returned by `time_weighting` and
  `TimeWeighting.process` for a `Signal` input, carries the running mean
  square with the rate, the weighting mode and whether the samples were in
  pascals. A mean square is in pascals squared, so it could not be a `Signal`
  without labelling a Pa^2 quantity a pressure record. It stands in for the
  array it replaced through the array protocol and the geometry attributes,
  and it plots the level trace.

- `LevelHistory`, returned by `sound_pressure_level_history`, carries the
  times, the levels and the interval it actually walked, and plots itself. It
  is iterable, so the callers that wrote `times, levels = ...` keep working,
  the way `DecayCurve` replaced its own tuple return.

### Fixed

- A temperature at or below absolute zero is refused rather than computed
  with.
  `sea_water_sound_speed(-273.15, 35.0, 0.0)` returned -31 457 m/s, a negative
  speed of sound; `sound_speed_profile` checked only that its temperatures were
  finite, so -300 degC went straight through to [-44 562, -45 333] m/s; and
  `seawater_absorption(1e3, temperature=-300.0)` returned 0,495 dB/km, which is
  the dangerous one, because nothing about that number says where it came from
  next to the 0,0185 dB/km of real cold water.

  `require_above_absolute_zero` in the internal validation now says it once,
  with an array companion for the profiles, and every site that had written it
  by hand uses one of the two: the air absorption of ISO 9613-1, the ASTM C423
  absorption, the intensity sound power, the airport impedance adjustment, the
  ocean ambient noise, the two ocean sound-speed entry points and the five air
  helpers of `materials`. One of those was spelling
  its message with a Unicode minus and a degree sign where the rest of the tree
  uses ASCII.

  Three guards deliberately do **not** migrate, and a test pins them by name.
  They protect the pole of the formula their own clause prints, `sqrt(273 + θ)`
  in ISO 3741 and ISO 3745 and `1245/(273 + t)` in Francois-Garrison, with the
  273 the standard prints rather than the physical 273,15. Their pole sits
  0,15 degC higher, so migrating them would leave a band of a sixth of a degree
  that clears the check and then overflows.

  Francois-Garrison needs both: the physical bound in front, and its own pole
  behind it. The band (-273,15, -273,0] used to raise `OverflowError` or
  `ZeroDivisionError`, neither of which names the argument that caused it.


- The weighting filter design gave different coefficients on different CPUs.
  The fit that removes the bilinear warp reaches its answer through a sequence
  of small dense solves, and the reductions inside them, the normal equations,
  their right-hand side and the cost the step test compares, were library
  calls. A BLAS built for many microarchitectures reassociates a reduction to
  suit the vector registers it finds when it loads, and Levenberg-Marquardt in
  a shallow valley turns that last bit into a visible difference: on the A
  weighting at 48 kHz, two kernel sets agreed on every accept decision for
  thirty-nine consecutive steps while the costs they were comparing drifted
  from 3e-13 to 1e-4 apart, and the filters they ended on differ by 5.1e-5 in
  the leading coefficient. Over the whole corpus, seven curves at thirteen
  rates, not one design was the same on all twenty-five kernel sets that could
  produce one here, and every one of them split the same four ways, twelve
  settings against five against four against four.

  Nothing graded moved. The responses agree to 0.010 dB at worst, A and C earn
  IEC 61672-1 class 1 at all eight rates the library claims either way, and no
  published verdict changes. What moved is that the same call gave different
  numbers on two machines, so a measurement could not be reproduced exactly
  anywhere else, and a plotted curve shifted by 0.0225 pt against a figure
  tolerance of 0.0175. The documentation figure job has also been failing
  intermittently on unchanged drawing code, and the figures it fails on are
  drawn from this design. The coefficient difference behind that is reproduced
  exactly, digest for digest, under emulated AVX512; what remains inference is
  only the last step, because the same difference stays inside the image
  comparison tolerance on a developer machine and crosses it on the runner.

  The test suite went with it. `tests/filters` failed on sixteen of those
  twenty-five settings, at the guard that stops an accuracy bound going slack,
  because the AU weighting came out better there, 0.0041 or 0.0042 dB against
  the 0.0044 the guard asks for, than the bound was sized against. The guard
  is right and the measurement it read was not a function of the source.

  The fit no longer asks a library to add up anything that was found to move.
  The reductions that feed each step fold their operands in half and add the
  halves elementwise, in an order the module fixes, and the dozen-by-dozen
  system at the heart of each step is eliminated in the module rather than by
  LAPACK. The two sums that collect a factor group are still numpy's, because
  they were measured not to differ on any kernel set including AVX512, and
  taking them over moved all ninety-one designs for nothing. Two more places
  had the same
  problem for a different reason: the printed curve the fit aims at, and the
  single read-back that sets the shipped gain, were complex expressions, and a
  complex multiply or magnitude fuses a multiply into an add where the CPU
  offers the instruction. Both are now written out in real arithmetic, where
  every step is one operation on two doubles that IEEE 754 rounds to one
  answer. That last pair matters as much as the first: with the reductions
  already fixed, dropping numpy to its baseline kernels still moved all
  ninety-one designs.

  That was still not the whole of it, and finding the rest needed a processor
  nobody here has. AMD does not ship AVX512 before Zen 4 and Intel dropped it
  from the 12th generation onward, so the case that fails could not be run on
  either machine this was developed on, and every measurement of it had been
  taken on hardware that cannot fail that way. Intel's Software Development
  Emulator runs it in software, and under it this machine returns the
  continuous integration runner's digest byte for byte. Measured there,
  numpy's `log`, `exp`, `tan`, `power`, `arctan`, `sin` and `log1p` each
  answer differently from the same host running natively, while the standard
  library's do not. Every transcendental on the design path is now taken from
  `math` one element at a time.

  The root of it then turned out to sit one line before the fit rather than
  anywhere inside it. `numpy.geomspace` is `10 ** linspace(...)`, so the grid
  every residual is evaluated on came out different and each step inherited
  it. The grid is now built from numpy's own formula, term for term, with the
  power taken elementwise and the endpoints pinned as `geomspace` pins them,
  which returns the identical array where AVX512 is absent and therefore moved
  no coefficient, figure or published value.

  Every design in the corpus is now identical bit for bit across sixty-five
  configurations: all twenty-seven OpenBLAS kernel sets, with and without
  numpy's dispatched kernels, six thread counts and five hash seeds. Two of
  those kernel sets used to fault outright. It is also identical between this
  machine and the same machine under emulated AVX512, which is the comparison
  that had been failing.

  The accuracy is where the design already reached. Five of the seven curves
  move by less than 0.0003 dB, two of them downwards, and AU reads 0.0059 dB
  against 0.0053. The BS.468-4 row reads 0.0624 dB where this machine read
  0.0597,
  which is the number the same code already produced on four of the kernel
  sets, and it is 62 % of the bound that row ships with. A and C earn IEC
  61672-1 class 1 at all eight rates the library claims, with the same
  0.700 dB binding margin, and the low-rate worst case does not move at all.

  Reading the cascade back in real arithmetic also made the 0 dB pin at the
  reference frequency real rather than circular. It used to be normalised and
  checked with the same function, so the check could not see an error in it;
  measured against a wider precision, the pin is now good to 1.6e-14 relative
  and `sosfreqz` is the thing that cannot read it, being out by 1.3e-10 at the
  G weighting's 10 Hz reference frequency. A design costs about 200 ms, paid
  once per curve, rate and mode.

- The ANSI S1.4-1983 Table V Type 2 cell at 20 Hz is read as +/-3 dB. It
  prints a bare "+3" where every one-sided cell of that same column prints
  "+5, -inf", and the transcription had taken it as upper-only with the
  alternative reading noted. IEC 651:1979 Table V, which agrees with the
  column at all thirty-three other rows, prints "+/-3" at exactly that cell,
  so the missing bar under the plus sign is a defect of the ANSI print rather
  than a national deviation. The B weighting sits 0.05 dB below nominal there,
  so no shipped verdict moves; what moves is that a filter with a real fault
  at 20 Hz can no longer pass by falling through a limit that was never
  published.

- The ISO 12354 detailed fiches drew the curve they rate without a legend
  entry. The sheet moves the legend above the axes, which means removing the
  one the plot built and making it again, and it collected the handles from
  the primary axes alone. Both detailed models draw their total on a twin
  axis, so `R'` and `L'n`, the quantities the whole page rates, were the half
  that went missing. The rebuild now gathers the twins' handles too, in the
  order the plot used, so the printed legend says what the plot said.

- Two committed fiches printed past the right margin of the page. The
  embedded plot was scaled by the width its SVG declares, and a legend
  anchored outside the axes reaches past that box, so the overhang went
  through at full size: the duct-path sheet asked for a 96 mm drawing, drew
  106.6, and put its legend 10.7 mm beyond the margin, straight through the
  plot's own border. The IEC 61043 sheet ran 3.2 mm over the same way. The
  drawing is scaled by what it covers now, and both sheets end where every
  other one does.

- A fiche is one page, and nothing checked that it came to one. An
  impedance-tube result of forty frequencies rendered three pages, the first
  a title over a blank half and the rest carrying the table, the result and
  the laboratory identity with nothing to identify them; past about fifty
  rows reportlab gave up with a message naming a flowable and a frame in
  points. Both now say the table is longer than the sheet holds, and the
  half-written file is removed rather than left behind.

- The header grid dropped a value whose label was empty, on the right-hand
  side only: the pad for an odd number of pairs was inferred from the empty
  label, so it swallowed a genuine label-less value too. The pad is explicit
  now and the two columns are treated alike.

- A compliance row with an unrecognised verdict printed the informational
  dash, so a mistyped `"FAIL"` came out looking like a row nobody had judged.
  The three the table knows are named, `"info"` among them, and anything else
  is refused.

- `col_widths` lists that do not match the column count are refused. reportlab
  does not object to them: it pads the list by repeating its last width, or
  cuts it short, and prints a table of a width nobody asked for.

- The IEC 60268 fiches scale their response panel by the decades the curve
  spans, and nothing bounded that divisor. Two frequency points a hair apart
  passed every check on the curve and left an empty panel under a range whose
  ends printed as the same number. A response narrower than one-third of an
  octave is refused, which is asked of the response the panel is drawn over
  and not of every curve, so a distortion sweep across a narrow band is still
  allowed.

- An ISO 4871 declaration of more operating modes than the sheet has width
  for is refused rather than printed with columns narrower than the levels
  they hold.

- The band tables decided whether their rows were one-third-octave triplets
  by counting them, and drew a rule after every third row unless there were
  exactly five. Rows that are not frequency bands got the rules anyway: the
  ISO 17497 polar fiche ruled its nineteen receiver angles into triplets, and
  a two-microphone impedance measurement on a linear frequency grid ruled
  that. The caller now says what its rows tabulate and the grouping is
  measured from the band centres, so an octave table of any length is left
  alone and a linear sweep gets no octave rules.

  A result dataclass is frozen, exported and has its constructor in the
  generated reference, so one could be built, or replaced into, with a
  spectrum that did not run over its own band axis. What that produced was
  not a failure: a fiche prints one row per band under a single header, so
  reportlab pads the short row out and the sheet then states a total summing
  a band that appears nowhere in the column above it. An array one *too long*
  is truncated by the same table, just as quietly, and the strips that report
  a field indicator as a range print a range taken over the wrong set of
  bands, which nothing downstream can object to because a range is well
  formed whatever it was taken over.

  Checking where the values are read would have to be repeated at every
  renderer, every plot and every reader not written yet; checking when the
  result is built covers all of them, in both directions. The types are
  `DuctPathResult` (including its element rows and its contributions),
  `SoundPowerIntensityResult` (on its band axis and its segment axis),
  `SoundPowerResult`, `InstalledSourceResult`, `DetailedAirborneResult` and
  `DetailedImpactResult` (on their band axis and their path axis),
  `NCResult`, `RCResult`, `STIResult`, `FilterComplianceResult`,
  `IntensityInstrumentComplianceResult`, `ImpulseProminenceResult` and
  `TonalCorrectionResult`. The message names the type, the axis and every
  count it was given.

  How many axes there are is pinned as well as how long each one is, because
  counting alone is not enough: an array of shape ``(paths, bands, 2)`` has
  the right number of paths on its first axis and the right number of bands
  on its second, so every count agrees and the extra axis travels on into
  whatever indexes the field next.

- Four guards that answered the wrong question, or could not answer at all.

  `sound_power_reverberation` named `background_levels` in its message but
  checked the shape one line *after* broadcasting it, so a spectrum of the
  wrong length died inside `np.broadcast_to`, whose text reports two shapes
  from inside numpy and names neither the argument nor the function. The check
  runs first now and covers both wrong shapes with the one sentence.

  `slit_helmholtz_absorber` let `resonator_geometry` travel two private frames
  down to `helmholtz_resonator_impedance`, which calls the same value
  `geometry`. A caller who mistyped the keyword was answered with the name of
  a parameter that function does not have.

  `_validate_area_inputs` was shared by four call sites and named none of
  them. Reached through `absorption_coefficient(t1, t2, ..., m1=, m2=)`, a bad
  `m1` reported a message about `'m'` and `'t60'`, neither of which is an
  argument of that function. It takes the caller's names now, so three
  messages changed text.

  And `airborne_insulation` diagnosed the wrong defect. Given a reverberation
  time carrying an extra axis, `(1, n)` against `n` bands, it said the band
  counts differed, which was false: they matched, and the extra axis was the
  problem. Its sibling `lab_airborne_insulation` had said so correctly all
  along, because it compares sizes where this one compared shapes. The three
  field-measurement pre-screens compare sizes now, so both halves of the
  package give the same answer, and the dimensionality check they were masking
  is reachable and tested rather than dead.

  Two guards found in the same pass are left exactly as they were, and it is
  worth saying why: the one in `io/_wav.py` cannot be reached from inside this
  repository, but it is promised by its own docstring and it screens a file
  header, and the one in `roughness_ecma` is one of three identical lines
  across parallel front ends. A guard that cannot fire today is not always
  dead weight. One that genuinely was, an early-out in `ecma` whose condition
  the two branches above had already taken, is gone.

### Changed

- `phonometry.noise_control` publishes `HvacWarning`. It fires whenever a
  relative efficiency lands below the 50 % floor Long Table 13.6 is tabulated
  from, and it was the one warning class in the library a package could raise
  and no package published, while its sibling `PlaneWaveWarning` was exported
  beside it. A caller could see the warning and could not write the
  `filterwarnings` rule for it; only the blanket `PhonometryWarning` rule
  reached it, and that one silences twenty-six others too. An architecture test
  now fails on any warning class a package can raise and does not publish.

- The visco-thermal absorber models take a `phonometry.fluids.Fluid` where they
  used to take an `AirProperties` of their own. `AirProperties` and
  `DEFAULT_AIR` are withdrawn; the air the Johnson-Champoux-Allard model was
  published with is `PUBLISHED_AIR`, and the keyword is `fluid=` rather than
  `air=`. Nothing about the numbers changed: `PUBLISHED_AIR` carries the same
  343,0 m/s, 1,205 kg/m3, 1,84 x 10^-5 Pa s, Pr = 0,71, gamma = 1,4 and
  101 325 Pa those models always defaulted to.

  What changes is what a caller can pass instead. `AirProperties` held six
  numbers and no account of where they came from, so an air measured at 30 degC
  and one recited from a textbook were the same object. A `Fluid` carries the
  model that determined it and the conditions it was computed under, which is
  what makes `fluids.air(temperature_c=30.0, static_pressure_pa=90000.0,
  relative_humidity_percent=70.0)` usable here: the absorber is then predicted
  in the air of the room rather than in the air of the paper.

  `PUBLISHED_AIR` keeps its own Prandtl number rather than closing it from
  eta / (rho alpha_t). A published fit carries the value it was fitted with, and
  deriving it from a better air would silently change the model instead of
  correcting it.

- The fan's pressure rise says what it is: `fan_sound_power`'s second argument
  is `fan_static_pressure_pa`, keyword-only. It is a **gauge** pressure rise in
  **pascals**, and it shared its name with the `static_pressure` the ISO 3740
  family takes in **kilopascals absolute** — a different quantity, a different
  unit and a different datum under one word.

  No guard can separate those two, which is why the fix is the name. A fan
  pressure of 101 325 Pa is a legitimate duty for a panel or propeller fan, so
  no plausibility band exists that would refuse the ambient value without also
  refusing a real operating point.

  The rename went first, before the medium migration reads the tree, so that
  `static_pressure` means one quantity when it does. Its usefulness showed up
  immediately: a blanket rename across the figure scripts caught two calls to
  `sound_power_reverberation` in the same file, which are the ambient kPa. One
  word for two quantities confuses even the person removing the confusion.

- `fan_efficiency_correction` takes `relative_efficiency_percent`, and warns
  below the 50 % floor Long Table 13.6 is tabulated from. Eighty per cent is
  `80`, not `0,8`; a caller who wrote the fraction fell into the table's
  catch-all bottom row and got its worst-case 16 dB correction where 80 % earns
  6, which is ten decibels on the sound power level with nothing on the page to
  say it happened. The value is unchanged, because the
  table is what it is; what changed is that the caller is told.

- Two errata registered against Ainslie (2010). Its folio 177 quotes
  1024,2 kg/m3 for sea water at 23 degC, salinity 35 and atmospheric pressure,
  where its own Equation (4.6) read with its own Equation (4.4) gives
  1024,287 9, which prints as 1024,3; the printed value is what the equation
  gives with the pressure term at zero, against the definition the same chapter
  states. And its Equation (4.13), which rearranges (4.6), prints the pressure
  coefficient as 4,3e-5 where (4.6) has 4,3e-7, two orders of magnitude and
  0,42 % on the density. The library implements (4.6) with the absolute
  pressure its own (4.4) defines, because a printed definition outranks a
  rounded quotation three chapters later; the discrepancy is 4,3 parts in a
  hundred thousand and below every tolerance here, so what mattered was not
  picking a side silently.

- `characteristic_impedance` is published by `phonometry.fluids` rather than by
  `phonometry.materials`. `rho c` is a property of the medium, not of the
  impedance-tube procedure that happened to publish it: `room.steady_field`,
  `emission.sound_power_in_duct`, the layered and porous models and the tube
  itself all reached into `materials` for it, four domains for one product.

  The arguments keep the names `density` and `speed_of_sound`. The unit belongs
  in a parameter name where two units are in play and a caller can supply the
  wrong one, which is why the temperatures say `_c` and the pressures say `_pa`;
  a density in this tree is kilograms per cubic metre everywhere and a speed of
  sound is metres per second everywhere, so a suffix here would only make this
  function differ from the two hundred other parameters that already carry
  those names.

  `Fluid` exposes the same quantity as a property, closed by identity from the
  two it was built with: it is their product exactly, not a fit of it. The
  function is for a caller holding a density and a speed of sound with no
  fluid to go with them.

- ISO 3747's Annex C prints its static pressure in pascals, and the library
  says so. It is alone in the ISO 3740 family in doing so: ISO 3741:2010,
  ISO 3744:2010 and ISO 3745:2012 all print kilopascals, and
  `sound_power_in_situ` follows those three so that one unit serves the whole
  family. Nothing moves: C2 carries a pressure term and a temperature term,
  and the pressure enters only as the ratio ps/ps,0, so converting both
  together leaves the correction alone. What changes is that a reader
  comparing the docstring with the printed page is told which is which
  instead of finding an unexplained unit.

- The air behind ISO 9053-2's thermal correction is the air IEC 61094-2:2009
  actually publishes. `thermal_boundary_layer_thickness` and `effective_kappa`
  defaulted to the five properties ISO 9053-2:2020 Annex A.3 prints, which that
  annex credits to IEC 61094-2:2009. Three of them are its Table F.1 cells
  rounded to four figures. The other two are not in that document at all: the
  table gives the thermal diffusivity, not its two constituents, and the pair
  Annex A.3 prints for them is 1,0800 times smaller than Annex F gives at the
  same state. A caller who asked this library for air received a specific heat
  capacity of 938,7 J/(kg·K), which is 27,19 J/(mol·K), below the rigid-rotor
  diatomic floor of 29,10, so it is not air at any temperature in any unit.

  The defaults are now the Annex F values at 23 °C, 101,325 kPa and 50 %
  relative humidity. The two conformance rows that reproduce the Annex A.3
  worked example pass the five values the annex prints, so they reproduce the
  standard rather than merely agree with it, and the citation is visible at the
  call instead of hidden in a default.

  Nothing moves in the printed example either way, because Formula (A.5) uses
  the two constants only through `k_a/(rho0 c0 C_P)` and the common factor
  cancels there: both air states give the printed b = 1,83 mm and kappa' =
  1,370. What changes is the number a caller gets when they do not supply air of
  their own. Anyone reproducing Annex A.3 exactly should now pass its printed
  values explicitly. The defect is registered against ISO 9053-2:2020 in
  `docs/ERRATA.md`.

  Two things are checked that were not. The wavelength the annex prints,
  172,9 m, guards the transcription of the speed of sound, and it was verified
  nowhere before. And the boundary-layer assertion says what it means: b is
  printed to three significant figures, so reproducing it means rounding to it,
  which is what the test now asserts rather than a tolerance that reads like
  spare room. There is none: the true 1,834 8 mm sits near the upper edge of
  that rounding band, and did so before this change too.

- The five air-property helpers of `materials` take degrees Celsius, spell the
  unit in the argument name and refuse a positional call. Two of them took
  kelvin under the same `temperature` name their three siblings read as degrees
  Celsius, so `speed_of_sound_iso(20.0)` returned 89,7 m/s and
  `air_density_iso(20.0)` returned 17,4 kg/m3, fourteen times the density of
  air, with no exception raised and nothing in the signature to say which unit
  was wanted. The helpers are now
  `speed_of_sound_iso10534(temperature_c=...)`,
  `air_density_iso10534(temperature_c=..., atmospheric_pressure_kpa=...)`,
  `speed_of_sound_astm(temperature_c=...)`,
  `air_density_astm(temperature_c=..., atmospheric_pressure_kpa=...)` and
  `speed_of_sound_iso17497(temperature_c=...)`.

  `speed_of_sound` is gone as a bare name and will not be reassigned. It was
  the ISO 17497-1 Clause 8 formula, and nothing in the name said which of the
  three speeds of sound in the package it was, nor what unit it wanted. The
  ISO 10534-2 pair carries its standard's number for the same reason: `_iso`
  did not distinguish it from the ISO 17497-1 one.

  Eqs. (5) and (7) of ISO 10534-2 are still evaluated in kelvin and the printed
  293 K, 343,2 m/s and 1,186 kg/m3 are still literal in the source; the
  conversion happens inside the helper. Both reference states reproduce to the
  bit, because 19,85 degC is 293 K exactly in binary floating point.

  The only temperature refused is one at or below -273,15 degC. Neither
  ISO 10534-2 nor ISO 17497-1 nor ASTM E2611-19 states a temperature range, so
  none is imposed: a bound drawn at laboratory temperatures would refuse an
  extreme but conforming measurement, and would still not catch a caller who
  means a different unit. What catches that is the unit in the argument name,
  which is why the helpers are keyword-only.

- The weighting fit takes its logarithms from `filters._pinned_log` instead of
  from a `math.log` loop, and a cold design costs about 200 ms where it had
  come to cost about 260. The loop was the price of determinism: numpy's `log`
  is dispatched on what the CPU offers and its AVX512 kernels disagree with
  the rest in the last bit, so every transcendental on the design path was
  taken from `math` one element at a time, and `log` -- the one that runs
  inside the Levenberg-Marquardt iteration, three quarters of a million calls
  per design -- carried almost all of the cost. The new module spells the C
  library's own `log`, the table-driven routine from ARM's optimized-routines
  that glibc ships, in numpy operations IEEE 754 pins exactly: plain
  arithmetic, integer bit work and table takes, with the routine's fused
  multiply-adds emulated through error-free transformations. Verified bit for
  bit against glibc's `log` over the 91 658 333 distinct values the whole
  design corpus evaluates, zero mismatches, and the corpus of designs is
  byte-identical before and after: no shipped coefficient, figure or
  conformance value moved. Beyond the corpus the agreement with glibc is
  statistical rather than structural: an adversarial search found about one
  input in thirty million, in the narrow band just outside the near-one
  window, whose true logarithm falls almost exactly between two doubles and
  where the compiled binary's own compiler-fused pairs round the other way;
  a further hundred million draws found no other kind of divergence, and
  none of it can reach a design, since the pinned routine is the definition
  on the design path. The fit also
  batches each evaluation's logarithms into one call and reuses the normal
  equations' index pattern across steps, which is where the rest of the
  recovery comes from; the transcendentals outside the iteration stay on the
  `math` loop, where a design pays them once.

  On platforms whose C library is not glibc the byte-for-byte clause reads
  the other way round, and it is an improvement: Apple's and musl's `log`
  round a handful of inputs to the other neighbouring double, so the retired
  `math.log` loop silently returned different designs on macOS than on
  Linux. The pinned routine returns its own bits everywhere, so `log` no longer
  separates the platforms. The handful of transcendentals a design still
  takes from the C library once per design do: the three CI platforms design
  three different byte strings for the same filter, measured and pinned per
  platform in the test suite and written up in the determinism guide.
  Spelling those calls the way the logarithm is spelled is the remaining
  step.

- The frequency weightings are designed at the sample rate instead of being
  outrun at eight times it. Every curve in `filters.weighting` starts as poles
  and zeros in the s plane, and the bilinear transform that makes a digital
  filter of them is exact in magnitude and wrong in frequency: it puts the
  prototype's response at `2 fs tan(pi f / fs)` rather than at `2 pi f`. The
  library used to hide that by interpolating, filtering at three to eight
  times the rate and decimating back. `high_accuracy=True` now means something
  else: a design routine fits an analog prototype of the same structure whose
  response *at the warped frequencies* is the printed prototype's response at
  the true ones, and transforms that. What runs is one cascade of second-order
  sections at the input rate, with nothing around it.

  Three limitations went with the resampling. The anti-alias filter of those
  stages had its transition band on the input Nyquist frequency and the signal
  crossed it twice, which put a floor of about 1.7 dB on everything above 0.9
  of Nyquist whatever the design rate: at 32 kHz the A weighting read 16.2 dB
  low at the 15 848.9 Hz row of IEC 61672-1 Table 3, over the -16.0 dB class 1
  limit, and verified only to class 2; the AU weighting read 16.3 dB low at the
  same point against a +/-3.0 dB tolerance and verified to no class at all; and
  the 20 kHz row of ITU-R BS.468-4 Table 1 read 2.1 dB low at 44.1 kHz against
  a +/-2.0 dB tolerance and had to be excused from its own test. Those three
  now read -0.008, -0.012 and -0.019 dB. A and C verify to class 1 at every
  sample rate from 8 kHz up, and at every Table 3 row their deviation stays
  inside the 0.05 dB the table itself is rounded to.

  Block processing was the second: a resampler cannot be driven block by
  block, so `stateful=True` forced `high_accuracy=False` and asking for both
  raised. They are independent now, `stateful` defaults to the accurate design
  like everything else, and stitched blocks reproduce a single call bit for
  bit. `WeightingFilter(fs, "468", stateful=True)` built and refused for that
  reason; it builds. `high_accuracy=False` is still refused for that curve,
  because the plain bilinear design is 23 dB out at 16 kHz and BS.468-4 prints
  one mask with no lower grade.

  The third was speed. Weighting one minute of 44.1 kHz audio costs about
  18 ms where it cost 377 ms for A and 775 ms for 468, and holds 21 MB of
  intermediates instead of 169 MB. The design itself costs about 200 ms and is
  cached per curve, rate and mode, so even a first call is quicker than the
  path it replaces: about 215 ms against 377 for A, 220 against 775 for 468.

  `high_accuracy=False` keeps its old meaning: the plain bilinear transform of
  the printed prototype, the closed form a reader can check against the
  standard term by term. It verifies to class 1 for fs >= 44 100 Hz, degrades
  to class 2 at 32 000 and 22 050 Hz, and meets no class at 16 000 Hz.

  What is shipped is the routine, not a table of coefficients: nothing between
  the standard's constants and the sections a caller runs is stored, and a test
  re-derives every filter in the library from the printed prototype and pins
  the residual against it. Each factor is `s^2 + exp(t1) s + exp(t0)`, whose
  Routh-Hurwitz condition holds for any real `t`, so no step of the search can
  produce an unstable filter; the work is a fixed budget of Levenberg-Marquardt
  steps and Lawson reweighting rounds, with no random state and no dependence
  on wall clock or thread count.

  Sample rates below twice the reference frequency are designed too, and they
  need saying because they are the one place the routine does something other
  than fit. There is no digital frequency to read 0 dB at when the frequency
  the standard normalises at is past the Nyquist frequency, and asking the
  bilinear warp `2 fs tan(pi f / fs)` for it anyway does not fail: it answers
  with the alias, which is dc whenever the rate divides the reference
  frequency, and dc is where every one of these curves has a zero. Both the
  fit and the realised cascade are anchored at the middle of the fit band
  instead, and pinned there to the printed curve's own level rather than to
  0 dB, so the absolute gain is still the standard's: a 200 Hz tone reads the
  same decibels at `fs = 903` as at 48 kHz. Where a first fit comes back more
  than 0.2 dB from the printed curve, which happens well below the audio band
  and nowhere in it, the spare factors of the padded prototype are restarted
  from other corners and the best result is kept. Measured over fifteen
  thousand curve and rate pairs from 2 Hz to 200 kHz, every design is finite,
  and over the ten rates from 100 Hz to 2 kHz the worst deviation from the
  printed curve is 0.31 dB against the 0.45 dB the oversampled path reached
  over the same seventy pairs.

### Changed

- The FDTD GPU parity is proved where the GPU is, not only where the process
  is. Those cases needed CuPy in this interpreter, so on a machine without a
  CUDA card the whole GPU half of the file skipped and the engine went
  unchecked. A card in this process is not the only way to reach one:
  `fdtd_gpu_remote` already ships a packed job to a machine that has one,
  which is the path the animation renders take. The same fifteen scenarios now
  go that way and are compared against the library engine.

  Two details decide whether such a test proves anything. It calls
  `run_remote` rather than `submit`, because submitting falls back to a local
  NumPy run when anything goes wrong, which is right for a render and wrong
  for a test: the comparison would be NumPy against NumPy and would pass
  without touching a GPU. And it asserts the backend that ran was `cupy`, for
  the same reason. With no `PHONO_GPU_HOST` configured the cases skip in under
  a second and say what to set, without opening a connection.

- Every requirement is pinned. `requirements.txt` and `requirements-dev.txt`
  used floors, and the two rendering stacks beside them used exact versions,
  so a fresh `make install` landed on a newer reportlab and svglib than the
  committed fiches were rendered with. Nothing was wrong with either version;
  what was wrong is that `make reports` then showed drift in fiches nobody had
  touched. Dependabot already watches these files weekly, so pinning turns an
  upgrade into a pull request the suite votes on rather than a difference
  between two machines.

  The four packages that were behind (reportlab, svglib, lxml, webencodings,
  all in the fiche chain) are now at the current release, and the 71 committed
  fiches come out byte for byte identical under it. The figure stack was
  already current, so no figure, plate or clip needed regenerating.

  This pins the environment this repository is developed in. What the
  published package asks of its users stays an open range in `pyproject.toml`,
  because a library that pins its dependencies cannot be installed beside
  anything else.

- A path is a `Path`, and a loop that only appends is a comprehension. `PTH`
  and `PERF` are selected, and the 193 sites they found were read one at a
  time, because the two spellings of a path are not the same type: a `Path`
  prints like its string and compares unequal to it, so the sites worth
  reading were the ones that hand their value to a sort, a comparison or a
  dictionary key. One did. The CI comment builder sorted a list of artifact
  paths, and sorting `Path` objects compares them component by component
  rather than as text, which reorders the rows whenever one matrix name is a
  prefix of another, as `3.13` is of `3.13-dev`. It sorts as text now, and
  says so.

  Half of what the rules found was one pair of lines copied fifty-four times.
  Every fiche test carried its own `_assert_one_page` and its own
  `_PDF_MAGIC`, in eight spellings, and forty-six of them also asserted a
  non-zero file size, which could not fail: reading the first four bytes and
  finding `%PDF` already proves the file holds four bytes. Converting each
  copy would have carried the duplication into `pathlib`, so the assertion
  that carries weight now lives once, in `tests/report_assertions.py`, and
  the fifty-four copies are gone.

  Fifteen sites keep the older spelling behind a `noqa` that says why: they
  sit in the code that draws the animations, which is hashed into
  `scripts/animation_fingerprints.txt`, so rewriting a line there asks for a
  re-render of clips that would come back frame for frame identical.

- Every test that expects an error says which error. `PT` is selected, and the
  241 `pytest.raises(ValueError)` calls that named no message now name one.
  This was already the habit rather than a new rule: of the 2075 `pytest.raises`
  in the suite, 1795 passed `match=` before this, so what changed is the
  stragglers. The 39 that still name no message are the ones the rule does not
  ask about, and does not need to: `FrozenInstanceError`, `ModuleNotFoundError`
  and `AttributeError` are raised by one thing each here, so the class is
  already the whole assertion. `ValueError` is the one that is not, which is
  why the rule asks about it. Each of the 241 was resolved by reading the guard
  it lands on rather than the surrounding test, because a test that catches any
  `ValueError` at all passes when the wrong guard fires, and the wrong guard
  firing is the regression it was written to catch.

  The fragment is the part of the message that does not move. That means the
  parameter the guard is about, and not the sentence around it: not a bound,
  which changes when the bound does; not the list of allowed values, which
  grows when a value is added; and not wording that belongs to numpy, whose
  shape-mismatch text is assembled inside a C extension and carries no
  stability promise. Where a message is already just a parameter and a
  predicate, as in `'frequencies' must be positive`, it is quoted whole,
  because there is nothing volatile in it to drop.

  Reading the guards turned up two tests that were passing without checking
  what they claimed. `test_short_frequencies_raises_value_error_not_index_error`
  waited on `frequencies`, a word three separate messages in that module
  contain, and a test named for rejecting a non-positive frequency waited on
  `must be positive`, which every positivity guard in its module shares. Both
  now name their guard. Neither was reported by the linter, because both did
  pass a `match=`; only reading them found it.

  Alongside it, 44 `parametrize` decorators take the shape the other 467
  already use, tuples for the names and a list for the values, and ten
  assertions of the form `assert a is not None and b is not None` became two
  assertions, so a failure says which of the two was missing. `B905` is
  selected, and the 162 calls it found were read one at a time: 158 pair
  arrays that are equal by construction and now say `strict=True`, and four
  consume the shorter input on purpose and say `strict=False` with a line
  saying what ends the loop. The four are worth naming, because they are the
  cases a blanket fix would have got wrong: two walk a palette against fewer
  paths than it holds, one walks the fixed seven-band STI table against
  whatever the result carries, and one walks a list of title sizes against
  the same list offset by one, where the smallest size has no step below it.

  `strict=True` is not a formality: it turns a silent truncation into a
  `ValueError`, so it is only the truth where something upstream guarantees
  the lengths. Checking that guarantee, site by site, is what this change was
  for, and it found seven places where nothing guaranteed it. Six accredited
  report fiches and one plot take their per-band arrays straight off a frozen
  dataclass that has no `__post_init__`, is exported from a public package,
  and has its constructor signature in the generated reference, so a result
  built by hand reaches the table with arrays of any lengths at all. The
  guards that existed covered some of the arrays and not the others: ISO 354
  compared the frequencies against the absorption coefficient and left the
  two reverberation times and both absorption areas unchecked, ISO 11654
  checked three of five, and the scattering fiche two of four. Each now
  raises a named `ValueError` before the table is built, the way the sibling
  fiches already did, and the `strict=` behind it is the backstop it should
  have been. Before this, a short array printed a table that quietly listed
  fewer bands than the daily total above it was summing.

  One call stays as it was, with a `noqa` that says so: it sits in the code
  that draws `anim_fdtd_impedance_tube`, where its three operands are the
  same pair of panels built a few lines above, so the argument could only
  ever hold, and adding it would move the clip's recorded fingerprint and
  ask for a re-render of the very same frames.

- A number in a comparison carries a name. `PLR2004` is selected for `src`,
  and the 487 magic values it found there were read one at a time before it
  went on: 380 became named constants, 29 reach an existing symbol that
  already said it, and 85 lines keep their literal under a `noqa` that says
  why. The names state the concept rather than the number, so `if
  len(header) < 4` is `< _BLOCK_HEADER_BYTES` (a FLAC metadata block header
  is one type byte and three length bytes) and `if nominal >= 1000.0` is
  `>= _HZ_PER_KHZ`. Where a name would say less than the expression already
  does, the literal stays: `order == 2` sits under a docstring that says
  "Product order (2 or 3)", and the transcribed CIEDE2000 formula keeps the
  180 and 360 degrees its published equations are written with.

  Reading every site found things a rule never would. Absolute zero is in
  the tree twice, as the exact -273.15 and as the -273.0 that ISO 3741
  writes into its own `273 + theta` arithmetic; they are now
  `_ABS_ZERO_C` and `_ROUNDED_ABS_ZERO_C`, deliberately unmergeable. Four
  modules already spelled the 1 kHz reference `_REFERENCE_FREQUENCY`, so
  the new ones join them instead of inventing a fifth spelling. The RIFF
  wire magics now live once in `io/_chunks.py` and are read from there,
  which closed a duplicate spelling of the bext chunk identifier and a
  sniffer table that had drifted from the name it was meant to match. A
  fiche's private frequency-label helper turned out to be byte-identical to
  `filters.frequencies._format_nominal_freq` and now calls it.

  The strings are not part of this: `PLR2004` exempts `str` and `bytes` by
  default and stays that way, because a comparison against `"cremer"` or
  `"es"` is the API's own vocabulary and the literal is the name. Only the
  format magics among them were named. `tests` and `scripts` are exempt
  too: the tests compare against printed oracle values, and the figure
  scripts lay out halves and angles, where a name for every number is noise.

- Every exception message is assigned to a variable and raised by name:
  2148 sites, `msg = ...` then `raise SomethingError(msg)`. What it buys is
  the traceback: the line the interpreter prints for the raise is
  `raise ClauseError(msg)`, not the first of several wrapped lines of a long
  clause citation, so the message appears once, in the exception itself, and
  the frame stays one line. The one site left alone is a guard in the code
  that draws `anim_fdtd_side_branch`, where the rewrite would move the clip's
  recorded fingerprint and demand a re-render that draws the same frames; it
  says so next to the `noqa`.

- Docstrings hold one shape now. The summary sits on the first line, the
  closing quotes take their own, and 873 docstrings moved to say so without
  changing by a word. Where a docstring carries backslashes (matplotlib
  mathtext, regular expressions) it is a raw string now, 37 sites, under one
  invariant: the value Python stores in `__doc__` stays byte-identical,
  checked by parsing the tree before and after. The check earned its keep
  once. One test docstring wrote `\tau` twice and `\approx` once inside a
  cooked string, so the stored value held a tab and a bell character where
  the source shows mathematics; the raw prefix is the repair, and it is the
  only docstring whose value moved.

  Twenty-one magic methods explain themselves now: mostly the
  `__post_init__` of a dataclass naming what it rejects, and `__len__` and
  `__getitem__` naming what they count and where they forward.

  The docstring rules that judge prose rather than shape stay out, measured
  one by one. `D401` would put 999 summaries in the imperative mood; `D205`
  would fold the deliberate multi-line opening sentence of 516 docstrings;
  `D403` would capitalise `dBFS`, `dL` and `matplotlib`; and `D400`'s nine
  missing periods are four questions that rightly end in their own mark and
  five sentences that finish on the next line. `D210` stands down in the
  tests alone, where a docstring may open with the clause it checks quoted
  verbatim, and the space it objects to is all that separates the quotation
  mark from the triple quote.

- Imports say what they are for, and names say what they are. An import that
  only a type checker needs sits under `if TYPE_CHECKING` now, 313 of them; a
  well-known package is imported under the alias everyone writes, 190 of them;
  and the 83 places that wrote `from ._layout import fiche_paragraph as
  Paragraph` call it `fiche_paragraph`, because the alias made a helper of this
  package look like the class reportlab publishes under that name. Four
  functions in the conformance domains imported numpy a second time, inside the
  function, where the module already had it.

  `INP` and `T20` are selected but find nothing: both are already clean in
  `src`, and they are here to keep it that way. A print is what the scripts are
  for, so they are exempt, and so are the tests.

  The clip freshness gate needed a repair first. It hashes the AST of the code
  that draws each clip, and its own preamble says an alarm nobody believes
  would be worse than no alarm, because a false positive costs a re-render.
  It hashed the whole of each module it reached, so moving an import under
  `if TYPE_CHECKING` looked like a change to the drawing even though nothing
  in that block exists while a frame is rendered: it marked most of the
  forty-two clips stale. It now skips those blocks, and hashes the remaining
  imports as a sorted set, so sorting them is not a change either. The
  imports are still hashed, because the walk stops at the package boundary
  and for `import numpy as np` the statement is the only record it has.

  `TC006`, which quotes the first argument of `typing.cast`, is not selected.
  That argument is never evaluated for its value, so the rewrite is inert, and
  it moved two clips through `figures.media._save_animation`.

- Ruff is the formatter, and there is a configuration to say so. The tree had
  none: no `[tool.ruff]`, no `ruff.toml`, no pre-commit, no `.editorconfig`, and
  CI ran `ruff check .` on ruff's default rules alone, which are four families
  of outright errors. Nothing checked shape, so 673 of the 812 Python files had
  drifted from any single style. They are formatted now, `ruff format --check`
  runs in CI beside the linter and in `make lint`, and an `.editorconfig`
  carries the same numbers to editors that never read `pyproject.toml`.

  The line length is 88, which is both the tooling default and, measured, the
  one that moves this tree least: 100 and 120 each rewrite more files, because
  a longer limit rejoins lines that were split by hand. Only 0,8 % of the lines
  here were over 88 to begin with.

  Markdown stays outside it. Ruff formats fenced Python inside `.md` and does
  not recognise `.mdx` at all, and every snippet in `docs/` is mirrored by hand
  into the site in English and Spanish, so formatting one copy would drift the
  three apart with no gate to notice. Those snippets also align their trailing
  comments to show the output each line prints, which is a teaching device.

  The linter now selects the families this tree already satisfied or nearly
  did, and the fifty-odd findings they raised are fixed. One rule is selected
  by its family and then turned back off: `SIM300` calls anything with an
  upper-case name a literal, and on that reading it gets three idioms here
  backwards. `spectrum[OCTAVE_BANDS == f]` is a numpy mask, not a condition;
  `res.transmission_loss[BANDS <= f0]` reads "the bands at or below f0" until
  it becomes `f0 >= BANDS`; and `value == pytest.approx(x)` is how pytest is
  written. It is right about two of the nineteen it finds and wrong about the
  rest, which SonarCloud noticed as assertions whose argument order no longer
  agrees within a file. Six thousand more from
  `PL`, `RUF`, `TID`, `N`, `T20` and `COM` are deliberately not selected: they
  would rename the symbols standards spell as `Rw` and `LAeq`, flag the
  transcribed normative constants as magic values, and ban the relative imports
  this package uses by design. `B905` is the one worth adding next, and is left
  for its own review because `strict=True` raises where the code truncates.

- `import phonometry` imports the top level and nothing else. The nineteen
  domain packages and the four names it publishes arrive on first use, through
  a module `__getattr__` (PEP 562), so the statement costs **4.4 ms and 56
  entries in `sys.modules`** where it cost **954 ms and 1182**. Exactly one of
  those fifty-six is the library itself, where 235 of the 1182 were; the rest
  are the interpreter's and the standard library's.

  Nothing changes for a caller. `phonometry.building` and
  `from phonometry import building` work as they did, `dir(phonometry)` still
  lists all twenty-three names before anything is touched, and the resolved
  attribute is cached in the module namespace, so the import happens once.
  The type checker sees the same package it always did: the imports are
  written out under `if TYPE_CHECKING`, which is the half of the module a
  checker reads and the interpreter never runs.

  What this buys is a program that uses one domain not paying for the other
  eighteen. A script that reads a WAV imported scipy, matplotlib and the
  aircraft noise-power-distance tables before its first line ran, because the
  top level had to bind every name it published.

- Every example in the documentation reaches a function through the package
  that owns it: `from phonometry import building` and then
  `building.single_panel_transmission_loss(...)`, in both languages in step.
  Two thirds of the pages already led with that form; the remaining third
  imported the function flat off the top level, and the two forms sat side by
  side on the same page.

  Reading the domain at each call site is the point. A page that computes a
  panel's transmission loss from its bending stiffness now says out loud that
  `plate_bending_stiffness` and `coincidence_frequency` come from `vibration`
  while `single_panel_transmission_loss` comes from `building`, which the flat
  form hid. It is also what the published guidance recommends, and it settles
  a collision the pages were already paying for by hand: `envelope`,
  `spectrogram`, `coherence`, `group_delay`, `minimum_phase` and `zoom_fft`
  all exist in `scipy.signal` under the same spelling, which is why several
  pages import the other side as `from scipy import signal as sp_signal`.

- The transforms whose output is itself a pressure record return a `Signal`
  when they are given one, instead of a bare array. A chain of them used to
  lose the rate and the calibration at the first step and the caller rebuilt
  the object by hand. Four transforms that scale like a pressure record are
  deliberately excluded because they are not one: the K-weighting of the
  loudness family is referenced to digital full scale, and the three
  whole-body vibration weightings are accelerations. The octave bank's band
  waveforms come back as `Signal` objects only when the bank read the object's
  calibration in the first place; a dBFS bank, or one carrying a factor of its
  own, filtered digital full-scale units. A bare array in still gives a bare
  array out, everywhere.

  The returned object carries `calibration_factor=1.0` when the input carried
  one, meaning calibrated with the conversion already done. Carrying the
  input's own factor forward would make every function downstream scale the
  samples a second time: measured before the rule was in place, the
  A-weighted level of a record calibrated at 20.0 came back 26.02 dB low, and
  it survived a trip to disk because `write(sidecar=True)` stamps the factor
  into the sidecar.

- The generated API reference publishes every call form a function has,
  instead of the single permissive signature that accepts all of them.
  `inspect.signature` returns the implementation signature, which for an
  overloaded function is not one of its call forms: it is the union that
  admits every head plus the combinations the heads exist to refuse, so the
  site was documenting a contract the library rejects. `octave_filter` shows
  what that cost: its page carried one signature returning
  `tuple[...] | tuple[...] | tuple[...] | tuple[...]` with nothing to say
  which combination of `sigbands` and `nominal` produced which, and it now
  carries its four forms with the return type each one gives. Nineteen pages
  change, covering the functions in the tree that declare overloads. Functions
  that declare none render exactly as before.

### Fixed

- A caller can name a curve. `label` is an ordinary matplotlib keyword and
  arrives through `**kwargs` like `color` or `linewidth`, but fifty-five
  renderers also passed one by name in the same call, so `result.plot(label=
  "mine")` raised `TypeError: got multiple values for keyword argument
  'label'` instead of drawing a labelled curve. Six of the thirty-two result
  types the plotting contract covers failed outright; the rest were one
  argument away from it.

  They use `kwargs.setdefault("label", ...)` now, which is what the same
  renderers already do for `color` and `marker`: the caller's label wins and
  the generated one still appears when there is none. Two renderers draw one
  bar per item, each with its own name, and take a copy of the keywords per
  bar rather than one default before the loop, which would have put the first
  item's label on all of them.

  The plotting contract test now checks `label` alongside `color`, through the
  legend rather than the artist: `ax.bar` puts the label on the container it
  returns, not on each rectangle, so reading `ax.patches` would have missed it
  on every bar renderer and let this back in.

- The `io` guide says how to write the type it talks about. It names `Signal`
  sixteen times in prose and never once said the spelling, so a reader looking
  for it found the private module the library imports it from and wrote
  `pm.io._signal.Signal`, which is what happened. It is `io.Signal` after
  `from phonometry import io`, or `phonometry.Signal`, and the class docstring
  and both index pages say so too, in English and in Spanish.

  `SignalOrigin` carried a reason that stopped being true: it was named that
  way rather than `SignalSource` partly because the top level could not hold
  two classes of one spelling. It can hold neither now, each being reached
  through its own package, and the reason that survives is the better one, that
  this is a passive record and not a source of sound.

- `plot_excitation` is exported by `phonometry.room`, the package that owns it.
  It was the one of the twenty-four plotting helpers that no domain published:
  it is defined in the private `_plot.room`, the top level re-exported it, and
  `phonometry.room` did not, so it was reachable through the flat shortcut and
  through no module path at all. A reader following the domain, which is how
  the documentation teaches the library, could not find it.

  Two architecture rules now hold the invariant it broke. One fails when the
  top level publishes a name no domain package does, naming the orphan. The
  other fails when two domains publish the same spelling, which is what forces
  a rename at the top level and what would make a call through a domain
  ambiguous about the module it reaches. The second was impossible to satisfy
  until the `environmental` alias went: the same module answered to two names,
  so 170 names had two owners.

- `floating_floor_improvement_spectrum` refuses one half of the
  `mass_per_area` / `dynamic_stiffness` pair instead of returning
  `delta_lw=None` without a word. The pair exists only to compute that
  weighted improvement, so answering `None` to a caller who asked for it was
  the one failure that mattered, and it was the only one of the twenty
  parameter groups below that did not already raise.

- `miso_coherence` refuses a list of input records whose `Signal` elements
  were recorded at different rates, instead of taking the rate of whichever
  one happened to be first. Nothing downstream caught it: two records at
  different rates can hold the same number of samples, so the equal-length
  check passed, and the conditioning then cross-spectra them on a single
  frequency axis that was right for at most one of them. The same call
  returned an axis of 4 kHz or 8 kHz for the same data depending on the order
  of the list. The module's own docstring already promised the agreement this
  now enforces.

### Changed

- Nineteen functions, seventeen of them public and two plotting helpers, state
  in their signature the group of arguments they actually require, through
  `typing.overload`, instead of presenting the members as independently
  optional and enforcing the group from the body. A type checker now rejects
  the invalid call before it runs, and the run-time check stays for callers who
  do not type-check.

  The groups come in three shapes, all of them previously invisible to the
  signature.

  - *Exactly one of the members.* `lining_resonance_frequency` (Formula D.1
    against D.2), `ground_effect` (flow resistivity against impedance),
    `gaussian_residual_level` (L90 against L95),
    `predict_diffuser_polar_response` (depths against reflection) and the
    plotting helper `plot_aperture_geometry` (width against radius).
  - *All of them or none*, or one member requiring another.
    `airborne_insulation`, `survey_airborne_insulation`, `adaptation_term_kc`,
    `plenum_flanking_reduction_index`, `plateau_transmission_loss` (three
    members over two routes), `floating_floor_improvement_spectrum`,
    `sound_power_anechoic`, `impulse_response`, `stipa`,
    `sti_from_impulse_response`, `multiple_shock_assessment`, `io.write` and
    the plotting helper `plot_piston_geometry` (angles with directivity).
  - *Required by the value of another argument*, expressed with `Literal`.
    `sensitivity`, whose narrowband estimation needs the rate, and
    `floating_floor_improvement_spectrum`, whose Cremer hammer model needs its
    limiting frequency.

  `sti_from_impulse_response` turned out to carry a second group besides its
  own pair: `snr` and `ambient` are two ways of saying the same thing and the
  run time already refused them together, which no signature said until now.

  One group could not take an overload at all: `SourceRoom` holds its two
  descriptions of the source room as fields, and no overload of
  `room_to_room_transmission` can constrain the inside of its own argument.
  The bundle now refuses itself in `__post_init__`, as `OperatingModeDeclaration`
  already did, so building a source room with both descriptions, with neither,
  or with a power level and no room constant raises at the line that built it
  rather than at the call that used it.

  Two further groups are stated in prose rather than in the signature, because
  no signature can carry them: `db_hr_global_index` requires `frequencies` for
  any input that is not exactly the eighteen DB-HR bands, a condition on the
  length of an array at run time, and `floating_floor_resonance` requires
  `thickness` and `porosity` for the enclosed-gas term, a condition on the
  value of `airflow_resistivity` rather than on a literal. Both now say so
  where the caller reads them.

- `linkwitz_riley` and `parametric_eq` take their mandatory argument as
  keyword-only: `linkwitz_riley(x, fs=None, *, freq, order=4)` and
  `parametric_eq(x, fs=None, *, sections)`. Making `fs` optional had pushed
  `freq` and `sections` into a `= None` default they never meant, with the
  body raising when the default survived, so the signature said optional and
  the call said otherwise. Python enforces them now, and the reader can see
  it without running anything. `fs` keeps its position, so
  `linkwitz_riley(x, fs, freq=1000)` still reads left to right.

- Six more signatures across the library stop advertising an optional
  argument the body then demands. `adrienne_window(fs, *, ...)` and
  `insitu_absorption_spectrum(incident_ir, reflected_ir, fs, *, ...)` take
  their rate as a plain required argument; `measurement_positions(surface, *,
  radius, ...)`, `precision_positions(surface, *, radius, ...)` and
  `sound_power_anechoic(levels_positions, surface, *, radius, ...)` require
  the radius they measure over; and `predicted_diffusion_spectrum(well_width,
  frequencies, *, depths, ...)` requires the well depths it predicts from.
  Each of them had a `= None` default and a hand-written `ValueError` a few
  lines into the body, so the signature and the docstring disagreed about
  what the call needs. Python raises `TypeError: missing required argument`
  for the omitted case now; the range checks that follow (a positive rate, a
  positive radius) keep raising `ValueError` as before.

- Five estimators do the same with the argument that sits behind their now
  optional `fs`: `zoom_fft(x, fs=None, *, f_min, f_max, ...)`,
  `lifter(x, fs=None, *, cutoff, ...)`,
  `time_synchronous_average(x, fs=None, *, period, ...)`,
  `regularized_inverse_filter(response, fs=None, *, f_range, ...)` and
  `resample_signal(x, fs=None, *, fs_new, ...)`. The band, the quefrency
  cutoff, the period, the inversion band and the target rate are what each
  call is for, so they are required rather than defaulted to `None` and
  checked in the body. `fs` keeps its position in all five.

### Added

- The estimators take the `Signal` too, so a record read from a file goes
  through the whole analysis chain without its rate or its calibration being
  retyped: the spectral estimators (`power_spectral_density`,
  `cross_spectral_density`, `coherent_output_spectrum`, `multitaper_psd`,
  `miso_coherence`), the time-frequency pair (`spectrogram`, `zoom_fft`),
  correlation and delay (`correlation`, `time_delay`,
  `impulse_response_delay`, `align_impulse_responses`), the envelope pair
  (`envelope`, `envelope_spectrum`), the cepstrum family (`cepstrum`,
  `lifter`, `echo_detection`), `time_synchronous_average`,
  `regularized_inverse_filter`, `resample_signal`, `fractional_delay`, and the
  data-qualification tests of `metrology` (`stationarity_test`,
  `level_crossing_rate`, `peak_statistics`).

  "Processed in pascals" is one rule but not one unit here, and each function
  now documents which one it lands in rather than repeating the phrase: a
  density comes out in Pa²/Hz (Pa² under `scaling='spectrum'`), a correlation
  in Pa², an amplitude spectrum or a waveform in Pa, and an inverse filter in
  1/Pa, because it undoes the response it was built from. Ratios and times do
  not move at all, so `time_delay`, `impulse_response_delay` and
  `echo_detection` return for a calibrated `Signal` exactly what they return
  for the raw record; a cepstrum takes the factor entirely on its zeroth
  quefrency; and `level_crossing_rate` is the one that asks something back,
  since its `levels` are compared against the samples and so have to be given
  in pascals.

- The filters take the `Signal` the reader returns, as the level functions
  already did: `octave_filter`, `weighting_filter`, `time_weighting`,
  `linkwitz_riley`, `parametric_eq`, and the block-processing objects
  `OctaveFilterBank` (both `.filter()` and `.spectrogram()`),
  `WeightingFilter`, `TimeWeighting` and `ParametricEQ`. The object supplies
  the rate when `fs` is omitted and an explicit one that disagrees raises
  rather than being arbitrated; a bare array still demands its rate by name; and a calibrated
  `Signal` is processed in pascals, which reads as a waveform in pascals, a
  squared envelope in Pa² or a band level in dB SPL depending on what the call
  returns. Two exemptions are stated where they apply: a dBFS reading is
  referenced to digital full scale and never sees pascals, and a bank carrying
  its own `LevelCalibration` is the explicit knob, so the object's factor is
  not applied on top. The resolution rules move out of `signals/levels.py`
  into `phonometry.io._resolve`, beside the `Signal` they describe, so that
  the surfaces on the contract cannot drift apart.

- Real measurement audio, versioned as functional oracles. Five
  licence-verified third-party sources live under `tests/data/audio/`
  (6.9 MB, provenance, licences and CC BY attributions in its README), and
  every one carries a value its authors published that the suite now
  reproduces through `phonometry.io` and the measurement functions: the
  NTi XL2 calibration take whose bext-declared full scale must land on the
  campaign's published 113.7 dB; five 4-second pass-by excerpts (all four
  vehicle categories of the Antwerp ISO 11819-1 campaign) held to the
  published LA,eq, LA,max and L10 within 0.3 dB and to the median ISO
  532-1 loudness N50 within 5 %; a Sound Devices 702T BWF and a genuine
  EBU RF64 as exact-equality transcription oracles for the chunk parser;
  the omnidirectional channel of a measured chapel impulse response held
  to the deposit's ISO 3382-1 parameter table within the standard's own
  just noticeable differences; and the 2015 five-minute LAeq year of two
  Dublin City Council monitors, which must average to the annual report's
  published night-time values before exercising `lden`, `ldn` and
  `composite_rating_level`. The io guide gains a worked example that reads
  the committed calibration take and closes the calibrator-to-measurement
  chain on a real instrument's file, in English and Spanish.
- The bottom may slope: the first range dependence in the underwater module.
  `ray_trace` and `gaussian_beams` accept a `bathymetry` pair of node
  arrays, `(ranges_m, depths_m)`, describing a piecewise-linear depth profile
  (Jensen's faceted boundary, Fig. 3.20) in place of the level bottom, while
  the sound-speed profile stays range independent, deliberately: the sloping
  boundary has an exact published oracle (the ideal wedge's closed fan of
  images) and a range-dependent `c(r, z)` has none. The marcher bisects each
  crossing against the interpolated polyline, vertices included, and reflects
  the ray about the local facet (Eq. 3.121), a rotation of the slowness pair
  that steepens a ray by twice the slope per upslope bounce; the dynamic pair
  crosses the bounce with the impulse of Eqs. (3.122)-(3.124) on a facet of
  zero curvature. A ray steepened past the vertical cannot be carried by a
  range march and is terminated at that bounce, `NaN` from that sample on,
  and a terminated beam retires from the field with its weight zeroed, the
  one-way trade the parabolic equation also makes. The beam solver folds each
  receiver column's image ladder about its own local facet, the dihedral fan
  that is exact for a single facet, with its analytic tails trusted no
  farther than two marched extents so they cannot price wedge geometry the
  polyline never described. Both results record the polyline and draw it
  under `plot()`. Slope zero reproduces the level solvers bit for bit;
  validated against the exact folded wedge trajectory (1e-5 m over 4 km and
  twenty-plus reflections), the single-facet two-path field (0.05 dB), a
  tilted bottom out of every beam's reach (0.001 dB against the free-water
  sum) and the ideal wedge's closed image fan (1.85 dB worst, 0.67 dB mean,
  across dense multipath cells of a 2.8-degree wedge). The lossy fluid
  seabed and `eigenrays` decline sloping runs, each with its reason stated.
  New subsections in the underwater solvers guide, in English and Spanish.
- The eigenrays and the arrival structure. `eigenrays` takes a `ray_trace`
  fan and one receiver and returns the paths that actually connect them, each
  with its travel time, launch and arrival angles, surface and bottom touch
  counts, KMAH caustic count and complex amplitude: the list the sonar
  equation consumes as multipath, the skeleton of a channel impulse response,
  drawn by `EigenrayResult.plot()` as per-path loss stems against delay. The
  search brackets the launch angles whose depth at the receiver range crosses
  the receiver depth and closes each bracket by bisection on fresh marches
  through the same profile, never by interpolating the fan, so every arrival
  is a real ray; a root that slips within the integrations' disagreement of a
  fan rung is recovered by widening the bracket one rung each way, and
  `max_arrivals` caps the unbounded multipath ladder at the earliest
  arrivals, with a warning when it truncates. Amplitudes are the classical
  ray amplitude of Jensen Eq. (3.65) over the 1 m reference, with `(-1)` per
  surface touch, the bottom's coefficient (perfect or Rayleigh, magnitude and
  phase, unconjugated in the module's `exp(-i omega t)` convention) per
  bottom touch, and the `(-i)^m` of Eq. (3.79) per caustic, so one
  frequency-independent list serves every tone as `sum a_j exp(i omega
  tau_j)`. `RayTraceResult` now records the profile it was traced through.
  Validated against the ideal waveguide's image lattice, where the entire
  arrival structure is closed form, to 1.2e-13 s, 1.4e-11 degrees and 2.4e-13
  relative amplitude with the census matching image for image, and against
  the n^2-linear profile's exact parabolic ray, whose quadrature-integrated
  closed-form travel time the upward-refracted eigenray reproduces to 2e-7 s.
  New subsection and figure in the underwater solvers guide, in English and
  Spanish.
- Gaussian beam tracing joins the underwater solvers. `gaussian_beams` hangs a
  beam on every ray of a launch fan (Jensen Eq. 3.88) and sums them into a
  propagation-loss field on the same grid, the same reference and the same
  plot frame as `parabolic_equation`, so the two subtract. What the beams buy
  over the rays they are built on is a finite answer: the dynamic pair `(q, p)`
  is started from the complex conditions of Eq. (3.91), whose conserved
  Wronskian keeps `q` off zero for good, so there is no caustic singularity to
  patch, no KMAH index to count and no minimum-width floor to impose, and a
  shadow zone is entered gradually instead of off a cliff. The `-pi/2` per
  caustic that ray theory has to apply by hand falls out of the branch of the
  complex square root instead, which is checked against Eq. (3.79) rather than
  assumed. Both boundary conditions come out of a ladder of folded receiver
  images rather than being imposed, the beam half-width and wavefront curvature
  are returned along every ray, and the cost is independent of frequency: a
  5000 m Munk column at 100 Hz over 10 km takes 14 s against 177 s for
  `normal_modes`, and raising the frequency does not move the first number.
  Validated against the free field in magnitude and phase, the two-ray
  Lloyd-mirror field, and the image-source sum of the ideal pressure-release
  waveguide to 0.0004 dB. It has no near field: read nothing inside about three
  initial beam widths of the source, where the far-field weighting of
  Eq. (3.92) has nothing to converge to. New section in the underwater solvers
  guide with its own figure, the caustic and the shadow zone behind it, in
  English and Spanish.
- The beams can carry the water's own absorption. `gaussian_beams` accepts
  `absorption` (the bare model name, `"francois-garrison"`,
  `"ainslie-mccolm"` or `"thorp"`, or a `VolumeAbsorption` naming its own
  temperature, salinity and pH; the same names, arguments and defaults as
  `seawater_absorption`, off by default so every published validation number
  stays reproducible bit for bit)
  and multiplies each beam by `exp(-alpha s)` with `s` the arc length along
  its central ray, which is Jensen Sect. 3.6.2 as printed (Eq. 3.116): the
  loss integral runs along the path flown, not along the range axis, and the
  `alpha r` shortcut the section notes "is used in many ray models"
  under-charges every steep or multiply-reflected path by the obliquity of its
  climb. The ray marcher integrates that arc length as a fourth state of the
  very Runge-Kutta stages that place the ray (`ds/dr = 1/(xi c)`), so it
  cannot drift from the geometry actually summed, and `ray_trace` exposes it
  per ray as `arc_lengths` next to the travel times. The applied coefficient
  is recorded on the result (`absorption_coefficient`, dB/km, evaluated at the
  source frequency and depth), and the oracle tests pin the absorbed free
  field against `20 lg R + alpha R` with the alphas transcribed from the
  published formulas independently of the closed-form module, to 2e-4 dB, with
  a steep-receiver case whose arc length exceeds its range by 1.24 dB of loss
  separating the two measures directly.
- The beams can bounce off a lossy seabed. `gaussian_beams` accepts a
  `FluidSeabed` as its `bottom` (the same fluid description
  `seabed_reflection` takes, with `water_density` for the water above; the
  perfect reflector stays the default, bit for bit) and charges every bottom touch of
  each beam with the Rayleigh coefficient at that beam's grazing angle,
  magnitude and phase together, which is Jensen Sect. 3.6.3 as printed
  (Eqs. 3.125-3.126). The angle is the one Snell's invariant fixes at the
  bottom, `cos(phi) = xi c(D)`, the same at every touch of one beam, so the
  coefficient is evaluated once per beam and raised to the marcher's count of
  bottom touches, in the running product and in the receiver-image ladder
  alike; `ray_trace` exposes those cumulative counts per boundary next to its
  arc lengths. The phase matters and is spent conjugated on the way into the
  beam sum, which is assembled in the conjugate time convention: below the
  critical angle `|R| = 1` and only the phase carries the seabed, and the
  oracle test, the image-source sum with `R(theta)^n` at each image's own
  closed-form angle, measures 15.3 dB against the un-conjugated phase and
  0.02 dB against the conjugate at the range where stripping the phase moves
  the exact sum by 5.7 dB. Agreement is 0.07 dB at worst across ranges whose
  dominant images sit above and below the critical angle, the seabed
  parameters are recorded on the result, and the ideal-waveguide check is
  tightened to 5e-4 dB to hold the default where it always was.
- The beams choose their initial width per launch angle, and the quarter-depth
  cap is retired. The default `W0` used to be clamped to a quarter of the
  water depth, which in shallow water pushed beams under the book's own
  ten-wavelength floor and biased the loss quiet by decibels, silently and
  always in the same direction: against the exact Airy modes of an
  `n^2`-linear 200 m guide at 200 Hz the cap's 50 m width measured +3.08 dB in
  the mean, and +4.12 dB in a second 100 m guide at 250 Hz. What a shallow
  channel actually demands is the opposite bound, a beam wide enough to
  resolve the guide's modes in launch angle: adjacent modes stand
  `lambda/(2D)` apart in the sine of that angle while a beam mixes angles over
  its far-field divergence `lambda/(pi W0)`, so the default now raises the
  free-space optimum `sqrt(lambda r_max/pi)` to `4 D cos(theta0)/pi` wherever
  that is larger and the book's 10-50 wavelength band can hold it: one width
  per launch angle, widest for the flat beams whose modes crowd together in
  angle, every beam's vertical footprint the same `4D/pi`. The same two
  oracles measure +0.19 and +0.39 dB at the new default, with no override and
  no warning; the free-field, Lloyd-mirror, ideal-guide and lossy-seabed
  validation numbers stand exactly where they were; and deep water is
  untouched, the guide term standing down for the free-space optimum when
  `4D/pi` exceeds fifty wavelengths. The result records `initial_beam_widths`
  per beam in place of the old scalar, an explicit `beam_width` still applies
  one width to the whole fan bit for bit, and the quarter-depth warning went
  with the cap it policed; the 20 Hz two-mode guide the old guide prose read
  as ray theory failing now measures within 0.03 dB of the image sum, the
  cap's error all along.
- The ISO 9614-3 precision determination renders its own accredited-style
  fiche. `PrecisionIntensityResult.report()` prints the one-page sound-power
  sheet carrying what part 3 asks a report to state (clause 10), which is not
  what part 2 asks: the normalized level `LW0` of Eq. 10 beside `LW` in the
  per-band table, the expanded uncertainty of clause 4.3 band by band (twice
  the Table 1 standard deviation of reproducibility, an em dash where the table
  has no row for the band), a caption naming the frequency range the
  determination covers, the four Annex B field indicators `FT`, `Fp|In|`,
  `FpIn` and `FS` tabulated with `verbose=True` and summarised in the basis
  strip beside the probe's pressure-residual intensity index and the dynamic
  capability it yields, and the five Annex C criteria with the per-band
  qualification cell they decide. Handed that qualification, the sheet makes
  the statement clause 10 f) 2) requires: the bands whose criteria are not
  satisfied leave the A-weighted determination and are named with the criterion
  that rejected them, beside the bands the method is not applicable to at all
  (clause 9.2). The screening is the fiche's own, because the result object
  computes its A-weighted total before any criterion has been evaluated, so the
  boxed `LWA` is the qualified-band level, the requirement verdict is taken on
  the number the reader sees, and the normalized `LWA0` the standard reports
  stands beside it. A one-third-octave band set prints in two column
  groups side by side, the way an accredited sheet fits that many bands on a
  page: the committed example is a sixteen-band determination from 100 Hz to
  3150 Hz over a five-face box measurement surface, whose 100 Hz band fails the
  dynamic-capability criterion and is named as omitted. With this the guide's
  coverage claim that both scanning results render the accredited-style fiche,
  narrowed to the truth by the coverage audit, holds again.
- Silencer layouts built by hand can now be drawn. A layout none of the four
  named devices covers gets a `.plot_geometry()` alongside its transfer matrix,
  through `SilencerChain`. It cascades the same `duct_matrix`, `shunt_matrix`
  and `cascade` calls the four-pole method has always taken, and keeps what
  each element was handed, so one call feeds both the matrix and the drawing
  and the two cannot drift apart. The drawing shows exactly what
  the elements declare and not a millimetre more: the ducts to scale with
  their lengths, bores, area steps and overall run dimensioned, and each side
  branch marked at the station where it joins, because a shunt is handed an
  impedance and an impedance fixes no length, no area and no volume. A branch
  is lettered with its name and with the frequency at which its impedance is
  least, which is the one thing it does say about itself, and the drawing
  states underneath that it is marked rather than drawn. `.result()` closes
  the loop with the transmission loss, the plane-wave limit of the widest
  section of the chain, and the usual `.plot()` and `.report()`.
- The underwater ray solver returns travel times. `ray_trace` now carries
  `travel_times` on its `RayTraceResult`, one cumulative arrival per ray
  sample, integrated as a third state of the Runge-Kutta step that already
  places the ray (`dt/dr = 1/(xi c^2)`, off the sound speed the trajectory
  derivatives interpolate anyway) rather than quadratured over the finished
  path, which is both a fidelity argument and an accuracy one: the stage sum
  reproduces the published closed form to 1e-14 s where the quadrature misses
  it by 3e-5 s at the default step. A reflection is instantaneous, so the time
  crosses surface and bottom bounces unchanged, and a ray bouncing the length
  of an isovelocity waveguide reports range/(c cos theta0) to nine digits. The
  oracle is Medwin and Clay, *Fundamentals of Acoustical Oceanography*
  (Academic Press, 1998), Eq. (3.3.20) on p. 88, read in the library's
  horizontal-angle convention as t = (1/g) ln[(c2/c1)(1 + sin th1)/(1 + sin
  th2)]; it anchors a new conformance row and the tests, which also pin the
  properties the physics owes, including that a ray turning deeper takes
  longer to reach its turn. The airborne twin `atmospheric_ray_paths`, which
  has returned a travel time since it was written, moves to the same
  four-stage integration; nothing rendered depends on either. The documentation
  that promised paths and travel times, and had been corrected the other way
  when the coverage audit found nothing computing them, is true as first
  written.
- Both ray solvers reflect at a boundary instead of folding it. A crossing used
  to be handled by taking a whole Runge-Kutta step and folding whatever came out
  of the medium back into it, which integrates the step through water or ground
  that is not there and applies a mid-step reflection at the step's end: a
  first-order error at every bounce inside a fourth-order integration, carried
  by the geometry and not only by the timing, and invisible to the tests because
  the only reflection they exercised was in isovelocity water where there is
  nothing to get wrong. Each crossing now ends a sub-step exactly on the
  boundary, located by the step's own cubic Hermite and polished against real
  steps, so a reflected ray keeps the order the rest of the path is integrated
  with. Two rays that must be identical, launched up and down from the same
  point on a boundary, went from 3.6 m apart to 1e-12 m in the ocean and to
  exactly equal in the atmosphere, and a path working both the surface and the
  bottom now reproduces the exact circular arc of a constant gradient to 7e-15 s
  over four bounces. The solver guide was printing 583.7 m for a turning depth
  whose analytic value it printed as 583.3 m and offering the gap as its own
  validation; the gap was this defect, and the two figures that draw rays
  against a boundary move with the fix.
- The informative tables of EN 12354-5:2009 are in the library instead of in
  the reader's notebook. `typical_element_mobility` is Table D.1, the six
  closed forms clause D.1.3 builds a source mobility out of (mass, bar end,
  beam, plate, pipe, mass-spring), with `TABLE_D1_QUANTITIES` carrying the
  table's own "describing quantities" column: a row rejects a quantity that
  does not describe it, and the four rows whose expression contains the
  frequency demand it while the other two refuse it.
  `tapping_machine_force_level` is Table F.1, the octave-band force level of
  the ISO tapping machine used as the substitution source of clause D.1.2.3,
  with `tapping_machine_force_level_estimate` for the closed form printed
  beside it (which the standard itself limits to about 1000 Hz, and which is
  the only route to the one-third-octave values it does not tabulate) and
  `tapping_machine_characteristic_power_level` /
  `tapping_machine_coupling_term` for Formulae (D.9a) and (D.9b). Annex F
  supplies the two terms Formula (18a) takes and they are here too:
  `structure_to_airborne_adjustment` is Formula (F.3) for the adjustment term
  `D_sa`, which the documentation previously spelled out as a raw expression
  for the reader to copy, and `multi_junction_adjustment` is the 4 dB and 6 dB
  of clause F.1 with the −5 dB floor on the resulting `Kij`.
- Two defects registered in the errata registry, both from EN 12354-5:2009.
  Table F.1 and clause F.4.2 print the force level "re 1 pN"; the reference is
  10⁻⁶ N, which is what Formulae (D.5a), (D.9a) and (F.10) balance at, what the
  tapping machine's own hammers produce, and what EN 15657:2018 Formula (15)
  and ISO 1683 use. And the Key of Figure D.3 labels all three curves
  `L_Ws,c,A` where its own caption names one characteristic power and two
  installed ones.
- Every piece of vector artwork draws its own text. The figures' SVGs used to
  ship live text positioned glyph by glyph from DejaVu metrics while asking the
  viewer for a generic sans, so any label carrying mathematics arrived
  letter-spaced on every machine without DejaVu, which is all of them; the
  plates asked for Segoe UI first, a face neither the maintainer's machine nor
  CI has ever had. Both now bake outlines, and both take them from the same
  place: the DejaVu Sans inside the pinned matplotlib wheel, so a plate and a
  figure on one documentation page are set in one typeface rather than two and
  a single pin fixes the letters of the whole drawn corpus. The plates keep a
  deterministic per-glyph fallback for what the monospaced cuts lack and a hard
  error for a character no face covers; all 185 distinct characters the 6,552
  plate labels set, in both languages, are carried by the face their own run
  asks for. Three plates rendered under an emptied fontconfig are
  pixel-identical to the normal render, which is the property the whole change
  exists for.
- The plates' type scale is stated in the px of the face that draws them.
  Every size in a builder is px of the outliner's face, and that face changed
  under them: DejaVu runs about 16 % wider than the Helvetica-metric face these
  compositions were measured in, 1.156 over all 6,552 labels of the corpus in
  both languages. A px of the old face is not a px of this one, so every label
  came out 16 % longer than the box, the column or the gap it was written into,
  and the plates filled with labels running under panels, through walls and
  over each other. The scale is converted once, at 0.86 with a floor of 10 px,
  rather than patched where it showed: it restores the line lengths the
  drawings were composed around, and a hand-tuned plate has no other invariant
  to hold to. The title keeps a ladder of 22, 21 and 20 px and gains a margin,
  because clearing the sheet was never the property wanted: a title whose first
  and last letters sit on the sheet border, on a plate whose artwork keeps
  thirty pixels, reads as having run off the page. Two hundred and forty-five
  titles keep the top size, thirty-nine take one step and twelve take two, and
  none reaches the floor, so the floor stays a backstop rather than a size the
  corpus is set at. The census that measures every label against its 900 px
  sheet reports zero.
- A second census measures the labels against the drawings, which is where the
  wider face actually put them: `outline.py --collisions` reports two labels
  sharing a line whose ink meets, a label crossing the border of a stroked box
  it sits inside or reaching into one it sits outside, a rule or a circle drawn
  through a label, and a filled panel drawn after a label and across it. It
  counted 298 collisions that the typeface change had introduced, and it is how
  they were worked off. Thirteen survived the scale conversion and were given
  room one at a time; ten boxes that had been measured on their English
  contents are now cut to whichever of the two languages is the longer, the
  Spanish of this corpus running about a quarter longer throughout; and the
  two-channel FRF plate's signal generator has a box wide enough for its own
  caption for the first time, having been 190 px around a 195 px Spanish
  caption before any of this. The corpus carries 356 hits where the same census
  counts 471 on the release before it, every text-on-text overprint in it is
  gone, and nothing in the 356 is a collision that release did not already
  have.
- The corpus typography is finished end to end: figure titles at text weight
  (the ISO 80000 justification for mixed weights dissolved on reading the
  standard, which regulates slope and never weight), the prime composed as a
  superscript everywhere mathtext draws (typed U+2032 sits at x-height,
  collides with ascenders and takes the following subscript with it), range
  hyphens restored inside mathematics where an ASCII hyphen sets as a spaced
  binary minus, negative readings signed with U+2212 through the shared
  helpers in figures, plots and fiches alike, and the library renderers'
  symbols moved into mathematics with descriptive subscripts upright, their
  Spanish keys moving in lockstep in both translation tables.
- Fifty-five layout collisions repaired across nineteen figures, each found by
  opening all 427 figures in both languages, confirmed at pixel level, fixed,
  and looked at again; two systemic causes fixed at the root (the frequency
  axis labels every third octave past four decades, the seawater relaxation
  labels hold inside the axis). All forty-two clips re-rendered with the
  wave's typography and the visual audit's parked recompositions, every WebM
  variant probed as AV1.
- Two new gates guard what no existing gate could see: every `$...$` in the
  sources must parse (an unparseable label aborts generation and used to be
  reported as staleness), and no Spanish translation value may keep a decimal
  point that the save-time comma pass cannot reach past a dollar sign. Both
  wired into `make figures` and CI ahead of the regeneration, both proven red
  against the real defect before being trusted.
- The scope sections of every guide are set apart from the prose and made
  auditable. Each "what this covers" and "what this does not cover" section now
  renders as a specification panel: one claim per row, a screen-printed legend
  in a fixed gutter (COVERED / NOT COVERED, and the Spanish pair on the Spanish
  pages), and a rule down the left edge that is solid for what the library
  covers and dashed for what it deliberately does not, the drafting convention
  for a boundary. Colour is a second channel only, and both states clear 3:1
  against the panel ground in both themes. The prose inside each row is the
  prose the sections always had, word for word, links and code included; the
  lead-in labels became the legend and nothing else moved. Behind the design
  sits a contract: every claim row carries `data-coverage`, and
  `scripts/list_coverage_claims.py` (`make claims`) enumerates the whole census
  from the authored MDX and from the built HTML with the same parser, 582
  claims over 280 pages at present, the two runs agreeing exactly, so what the
  documentation promises and declines can be checked against the code claim by
  claim. The overview mirror learned to render the components' children into
  the GitHub `docs/` mirrors and the llms artifacts, which also repaired its
  glob, silently matching nothing since the overviews became `.mdx`.
- Every figure on the documentation site opens at the full viewport
  (`starlight-image-zoom`). The reading column is 720 px at its widest and the
  source drawings are 1200 px and more, so a 33-band third-octave filter bank
  or a block diagram with labelled arrows arrived at the page already too
  small to read: the detail was in the file and the layout was the only thing
  hiding it. The figure itself is the control, with no icon painted over the
  drawing; the pointer opens it by clicking anywhere on it, and the plugin's
  floating button is kept, emptied and stretched over the whole figure so the
  keyboard still has something to land on, name and press; its own focus ring
  is suppressed and drawn on the image instead, since the button spans the
  reading column and the figure inside it is narrower, so a ring on the button
  would box in white space. The zoomed view is
  a native `<dialog>`: Escape closes it, so does a click or tap anywhere, and
  focus returns to the figure it came from. Nothing is fetched from a third
  party and no script evaluates a string, which is what lets the page keep the
  existing `script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'` policy
  untouched. The plugin hard-codes its two accessible names in English and
  offers no translation hook, so the site supplies them:
  `src/components/ZoomI18n.astro` rewrites the names the plugin's own
  components render and asserts the English text it replaces is still there, so
  a plugin upgrade fails the build instead of quietly shipping a Spanish page
  with an English control. The plugin finds images by walking the markdown tree
  and never sees the ones a component emits, so `ThemeImage.astro` wraps its
  own pair; every figure on the site is authored through it, and the EN/ES
  parity check now enforces that, since a figure written as markdown on a
  Spanish page is the one route to a zoom control nothing translates. The check
  looks for what opens a markdown image in any of its four forms, so a figure
  written as a reference (`![alt][label]`) cannot slip past the way one written
  inline would have; a sample of the markup inside a fence or inside backticks
  is still just a sample. Each figure ships a light and a dark
  variant and both get wrapped, so `theme-images.css` now hides the wrapper of
  the off-theme half as well; without that the hidden variant left an empty
  full-width box in the flow whose zoom button was still in the tab order and
  opened on a zero-sized image. Captions are disabled: the alt texts here are
  sentence-long descriptions written for a reader who cannot see the plot, and
  painted over the zoomed figure they would cover the thing the reader zoomed
  in to see.

- Page actions in the right-hand column of every documentation page that has a
  markdown copy: a split button that copies the page as markdown, with a menu
  holding "Copy link to Markdown", "View as Markdown", handoffs to ChatGPT and
  Claude and, under a rule of its own, "Copy citation". Every page written as a
  file has published a clean markdown copy of itself at `<page>/index.md` since
  the page-markdown generator landed, and has advertised it with
  `<link rel="alternate">`, but only something parsing the document head could
  find it; this is the first visible handle on it. Which pages those are is
  decided in `src/routeData.ts` from the content collection rather than from
  the address being viewed, and the whole control is withheld where the answer
  is none, so no entry in it can offer a page that was never written. Written
  as a site component rather than installed:
  the two published Starlight plugins for this both declare Astro 5, one
  translates nothing and requires a second markdown-emission package it cannot
  be told to skip, and the other emits its own copy of every page through
  `tidymd`, unconditionally, which on a site served from a base path doubles
  that base in every internal link and leaves the MDX component tags standing
  in the text. Either would have shipped a second, worse answer to "give me
  this page as markdown" beside the one this site already gets right. The
  control is a menu button in the ARIA sense, with `aria-haspopup`,
  `aria-expanded`, arrow-key and Home/End movement between entries, Escape to
  close and focus returned to the button, and it carries both locales. Opening
  it from the keyboard, with Enter, Space or Down, lands on the first entry, and
  Up lands on the last; the entries are out of the tab order, so a keyboard
  reader left standing on the button would have an open menu and no way into it
  that the page ever mentions. A press of the pointer leaves focus alone.
  The first menu entry copies the address of the markdown copy rather than its
  text: pasted into a model that can fetch it, one URL is sturdier and lighter
  than tens of thousands of characters, and it is the same handle the ChatGPT
  and Claude prompts already carry. The last one copies a BibTeX `@software`
  entry, which is what the acousticians and researchers this library is for
  write their bibliographies in. It is assembled during the build from
  `CITATION.cff`, the root `VERSION` file and the dated changelog heading for
  that version, so no author, DOI or release year is written down a second
  time, and it cites the Zenodo concept DOI that the About page and the site's
  structured data already publish, with the release pinned in a `version`
  field. The About page publishes the same citation three times over in both
  locales, as a bold DOI link, an APA reference and that BibTeX entry, and the
  build now fails if any of the six ever disagrees with what it generates: the
  DOI on both sides of the link, and the author, title, year, version and DOI
  the APA reference repeats. Neither new entry fetches anything, and the
  Content-Security-Policy is untouched. Both halves of the
  split ship disabled and the script enables them once it has wired them up, so
  a press that arrives before it, or without it, is refused rather than
  swallowed; the one entry that is a plain navigation is a plain link in a
  `<noscript>` in that case, rather than being unreachable behind a menu no
  script can open.
  It sits above the table of contents, which is where a reader already looks
  for what is *about* a page rather than what is *in* it, and where it costs
  the prose no width; on a phone it is a row under the table-of-contents bar,
  in the flow, so it scrolls away instead of permanently spending height.

- Heavy and soft impact sources (`building/heavy_impact.py`, ISO 16283-2:2020
  Annex A, ISO 10140-5:2010 Annex F, JIS A 1418-2:2019, ISO 717-2:2020
  Annex D): the rubber ball and the bang machine, the two standard sources for
  the slow, low-frequency impacts the tapping machine says nothing about.
  `impact_force_exposure_level` evaluates the defining Formula (A.1)
  `LFE = 10 lg[(1/Tref) integral F^2/F0^2 dt]` from a sampled force record;
  `heavy_impact_source_specification`, `heavy_impact_source_limits` and
  `check_heavy_impact_source` carry the printed octave-band spectra with their
  tolerances and verify a calibration run against them.
  `standardized_maximum_impact_level` implements Formulae (4), (5) and (6),
  where the receiving room cannot be corrected with the usual `10 lg(T/T0)`
  because the rated quantity is a Fast time-weighted *maximum*: the correction
  saturates through `g(C)/g(C0)` with `C = T/1,7275`, exposed on its own as
  `fast_reverberation_correction`, and vanishes exactly at `T = T0`.
  `a_weighted_maximum_impact_level` is the normative single number of
  ISO 717-2 Annex D, an A-weighted energy sum rather than a shifted reference
  curve, and reproduces the Table D.4 worked example to every printed digit
  including its deliberately unrounded 55,350 66... intermediate. The
  rubber-ball spectrum is pinned against three independent printings
  (ISO 16283-2 Table A.1, ISO 10140-5 Table F.1, JIS A 1418-2 Table A.2) and
  the bang machine against JIS A 1418-2 Table A.1; the standardization is
  anchored on its own `T = T0` identity and on a published 25-band
  reproduction of Formula (4). New guide "Heavy and Soft Impact Sources" with
  the source-spectrum and rating figure, in English and Spanish.

- Suspended-ceiling plenum flanking (`building/ceiling_plenum.py`,
  ISO 140-9:1985, ASTM E1414 through ASTM E413-22, Vigran 2008 Section 9.2.3):
  the path where sound leaves one office through the ceiling tiles, crosses
  the plenum over the partition and comes back down into the next.
  `plenum_flanking_reduction_index` implements Mechel's one-dimensional model,
  both the full attenuated form (Eq. (9.18), with the receiving-side leakage
  term of Eq. (9.17), whose printed denominator is a misprint now recorded in
  `docs/ERRATA.md`) and its small-attenuation limit
  `Rcl = RS + RR - 10 lg[eps^2 LR/(4h)]` (Eq. (9.20)), which is what makes the
  geometry visible: a deep plenum helps, a long room hurts and reflecting
  sidewalls cost 6 dB against absorbing ones.
  `partition_referenced_reduction_index` refers the path to the partition area
  (Eq. (9.13)) so it can be added to the direct path.
  `normalized_ceiling_attenuation` is the measured quantity
  `Dn,c = D - 10 lg(A/A0)` with the ISO reference `A0 = 10 m2` or the ASTM
  `12 m2`, and `ceiling_attenuation_class` the ASTM E413 rating that ASTM E1414
  invokes, with the clause 5.2 integer rounding, the 32 dB deficiency sum and
  the 8 dB single-band limit. The rating engine is anchored on three accredited
  ASTM E1414 laboratory reports, reproducing their printed CAC 34, CAC 39 and
  CAC 25 and, band by band, the deficiency column of two of them; the plenum
  model itself has no published numeric output anywhere, so it is anchored on
  its closed forms and on the structural properties a wrong reading breaks:
  monotonicity in the plenum damping, the bound `tau_cl <= 1`, and the
  convergence of Eq. (9.18) to Eq. (9.20). New
  section in the ISO 10848 laboratory flanking guide with its own figure, in
  English and Spanish.

- The wall-tie bridge of masonry cavity walls (`building/masonry_cavity_wall.py`,
  Hopkins 2007 Sections 3.11.3.2 and 4.3.5.4.1): the double-leaf prediction
  treated the cavity as pure air, but a real cavity wall is stitched together
  by ties every few courses. `WALL_TIE_STIFFNESS` and `wall_tie_stiffness`
  carry Hopkins' Table A4, the dynamic stiffness of four ties measured on
  concrete cubes, and `wall_tie_stiffness_per_area` builds the `N k / S` term
  that acts in parallel with the cavity air spring. `mass_spring_mass_resonance`
  and `double_wall_transmission_loss` gain a `tie_stiffness_per_area` keyword
  that folds it in (Eq. 4.89), reproducing the pair Hopkins prints for one
  fully specified wall: 26 Hz without ties, 50 Hz with 2,5 ties/m2 of 2 MN/m,
  which extends the combined-mass branch straight into the rating range.
  `wall_tie_coupling_loss_factor` gives the structure-borne path, the
  point-connection coupling loss factor of Eq. 4.87 with the linear-spring
  connector mobility of Eq. 4.88 and infinite thin-plate driving-point
  mobilities; the plate area cancels, so only the tie density enters, and a
  soft butterfly tie rolls off as `1/f^3` where a rigid connection only falls
  as `1/f`. New section in the panel prediction guide with its own figure, in
  English and Spanish.
- CNOSSOS-EU railway source emission (`environmental/cnossos_rail.py`,
  Directive 2002/49/EC Annex II section 2.3 and Appendix G, consolidated text):
  `railway_source_power` builds the two equivalent source lines of a track, at
  0,5 m and at 4,0 m, from rail and wheel roughness (Tables G-1a and G-1b), the
  contact filter (G-2) and the track, wheel and superstructure transfer
  functions (G-3), adding impact noise at joints and switches (2.3.11, 2.3.12
  with Table G-4), curve squeal, traction noise (G-5), aerodynamic noise above
  200 km/h (2.3.13, 2.3.14 with G-6), bridge noise (2.3.18 with G-7) and the
  source directivity (2.3.15 to 2.3.17), and returns the directional sound
  power per metre in 1/3-octave bands and energy-summed into octave bands.
  `RailwayEmissionResult` keeps the per-source breakdown and draws both heights
  with `.plot()`. The component functions are public, so the roughness
  conversion (`roughness_to_frequency`), the roughness sum
  (`total_effective_roughness`), each rolling component (`rolling_sound_power`),
  the squeal rule (`curve_squeal_excess`), the two directivities and the whole
  Appendix G database can be used on their own; `VehicleDescriptor` and
  `TrackDescriptor` parse the four- and six-digit codes of Tables [2.3.a] and
  [2.3.b]. Which instrument each table comes from is recorded next to it: the
  corrigendum of OJ L 5, 10.1.2018 replaced the whole of Appendix G and put the
  roughness conversion in metres per second, and (EU) 2021/1226 then replaced
  Tables G-1b, G-2, G-3a, G-4 and G-7, the curve-squeal rule, the bridge model
  and the vertical directivity of source A, which `DirectivityEdition` still
  offers in its 2015 form for comparison with pre-2021 studies. Two readings of
  the wavelength-to-frequency resampling, which Annex II describes in prose
  only, are selectable through `RoughnessInterpolation`. Annex II prints no
  worked example anywhere in 2.3, so the equations are pinned end to end on the
  emission test workbook the European Commission published with its reference
  source module: fed the 2015 coefficient database that workbook was computed
  with, the shipped model reproduces all 34 560 published rows of the twenty
  vehicles whose catalogue entries are well formed, 17 280 cases at both source
  heights, to 0,0055 dB, and 123 of those cases are committed and run in CI.
- Poroelastic (Biot) layers in the multilayer absorber solver
  (`materials/biot.py`, Allard & Atalla 2e chapters 6 and 11). Every porous
  model shipped so far collapses the material into an equivalent fluid, so the
  skeleton either stands still or is dragged along by the pore fluid, and
  neither can resonate. `biot_waves` solves the full Biot theory instead: the
  elastic coefficients `P`, `Q`, `R` of Eqs. (6.26)-(6.29), the modified
  densities of Eq. (6.56), and the three waves an isotropic porous material
  really carries, two compressional (Eqs. (6.67)-(6.69)) and one shear
  (Eq. (6.83)), with the fluid-to-frame velocity ratios that identify the
  airborne and the frame-borne branch. `biot_surface_impedance` evaluates the
  hard-backed closed form Eqs. (6.107)-(6.108) at normal incidence,
  `frame_quarter_wave_resonance` the frame resonance of Eq. (6.110), and
  `frame_bulk_modulus` / `frame_elastic_coefficient` the frame moduli in
  vacuum. `PoroelasticLayer` puts the layer into `layered_absorber`, which then
  switches to the global-matrix assembly of Sect. 11.5 with the coupling
  matrices of Sect. 11.4 and the six-variable layer matrix of Table 11.1,
  exposed on its own as `poroelastic_transfer_matrix`. Fluid layers, sheets,
  air gaps, oblique incidence, bonded poroelastic layers and rigid or radiating
  terminations all keep working, and a stack without a poroelastic layer takes
  the old two-variable path unchanged. `BiotWavesResult.plot()` draws the three
  wavenumbers, and a new figure and guide section in English and Spanish show
  the frame resonance no equivalent fluid can produce. No published table of
  `Zs(f)` exists for a Biot layer, so validation rests on the rigid-frame limit
  converging first order on the digit-anchored JCA equivalent fluid at four
  angles of incidence, on the limp limit converging on `limp_frame`, on the
  chapter 6 closed form and the chapter 11 assembly agreeing to machine
  precision, and on the three computed numbers the book prints in prose (the
  airborne branch change at 495 Hz, `|mu_a| > 40` above 50 Hz and `mu_b` from
  1,0 at 50 Hz to 0,82 at 1500 Hz, this last one read as `Re(mu_b)`, see the
  errata below) plus its published 860 Hz impedance peak. Six new conformance
  rows. Fluid runs and poroelastic layers are split into blocks of bounded
  attenuation before the global matrix is assembled, so a stack that
  attenuates by hundreds of nepers stays finite and passive instead of
  overflowing `float64` or returning a negative absorption coefficient, and a
  layer that would need more than 64 such blocks, which is what a shear
  modulus driven to zero asks for, is refused with a clear message.

- Four defects of Allard & Atalla 2e recorded in `docs/ERRATA.md`: the second
  printed form of the shear velocity ratio, Eq. (6.85), whose denominator
  should be `w^2 rho12` and not `w^2 rho22`, as its own Eq. (6.80) shows and
  Eq. (6.84) requires; Eq. (11.48), which attaches both of its terms to the
  antisymmetric amplitude and carries the shear `k33` inside a sum over the two
  compressional waves, contradicting the Table 11.1 it is tabulated into; and
  the thickness of the second glass-wool sample of Sect. 6.6.3, printed as
  5,4 cm in one sentence and as 5,6 cm two sentences later and in the caption
  of Figure 6.11; and the sentence of Sect. 6.5.4 that quotes "the ratio
  modulus `|mu_b|`" decreasing from 1,0 to 0,82, whose two values are the real
  part of `mu_b` (0,811 against the printed 0,82, while the modulus is 0,939).
  The thickness entry is settled on textual grounds only, two printed
  statements against one: the numbers are ambiguous, because the peak-finding
  rule is not printed and the two natural rules point at different
  thicknesses, which the entry now states.
- CNOSSOS-EU road traffic source emission
  (`environmental/cnossos_road.py`, Directive 2002/49/EC Annex II section 2.2
  and Appendix F). The source model every strategic noise map in the European
  Union has been drawn from since 2021: each vehicle is a point source 0,05 m
  above the pavement whose sound power is the energy sum of a **rolling** term
  growing logarithmically with speed (2.2.4) and a **propulsion** term growing
  linearly with it (2.2.11), and a traffic flow is a source line carrying
  `L'W,eq,line = LW + 10 lg[Q/(1000 v)]` per metre (2.2.1). Implemented in full:
  the studded-tyre correction with its 50 and 90 km/h saturation branches
  (2.2.6 to 2.2.9), the air-temperature correction (2.2.10), the road-gradient
  correction with the published asymmetry between the light and heavy
  categories (2.2.13 to 2.2.16), the junction correction that fades out over
  100 m (2.2.17, 2.2.18), and the road-surface effect, which enters rolling
  noise as `alpha + beta lg(v/vref)` and propulsion noise as `min(alpha, 0)`
  (2.2.19, 2.2.20). `road_source_power` assembles a whole traffic mix into a
  `RoadEmissionResult` that keeps the rolling, propulsion, per-vehicle and
  per-category line spectra, A-weights itself with the octave-band table the
  Directive prints in 2.5.5, and draws the category breakdown with `.plot()`;
  `line_source_segment_power` converts the per-metre power into the point
  source of a segment, which the method itself declares out of its own scope.

  The implemented text is the consolidated one: the equations of (EU) 2015/996,
  the 63 Hz to 8 kHz octave-band range restored by the corrigendum of OJ L 5,
  10.1.2018, and Tables F-1 and F-4 as replaced by (EU) 2021/1226, which is
  worth 2,5 to 3,5 dB(A) against the tables published in 2015. Every shipped
  table names the instrument it comes from, and both coefficient objects
  (`RoadEmissionCoefficients`, `RoadSurfaceCoefficients`) can be replaced, which
  is how Appendix F is meant to be used: as a database a Member State may
  substitute.

  Anchored on the European Commission's own test set: 60 of the 4 875 cases of
  the CIRCABC road emission workbook are committed under `tests/data/cnossos/`
  with the 2015 coefficient tables they were computed with, and the shipped
  equations reproduce every published band level to 0,005 dB, inside the two
  decimals the workbook prints. What the workbook cannot reach (the studded
  speed branches, the gradient breakpoints, the 20 km/h floor) is pinned on
  the exact closed forms, and the current Appendix F tables are pinned against
  a machine transcription of the Official Journal. New guide "CNOSSOS-EU road
  traffic source emission" with the source-line spectrum and the
  rolling/propulsion speed-crossover figures, in English and Spanish.
- Prediction of resilient-layer performance
  (`building/resilient_layers.py`, Hopkins, *Sound Insulation* Sections 3.6.3,
  4.3.8, 4.4.3.1 and 4.4.4; ISO 12354-1:2017 Annex D; ISO 12354-2:2017
  Annex C; Vigran, *Building Acoustics* Section 8.4). The library measured
  what a covering or a resilient layer achieved and stopped at the resonance
  frequency; this is the chain from the material data to the improvement
  spectrum, so a construction can be sized before it exists. It starts at the
  excitation, because the force the ISO tapping machine injects is a property
  of the floor as much as of the machine: `tapping_force_spectrum` solves
  Lindblad's mass-spring-dashpot of the hammer, the contact stiffness and the
  floor's driving-point impedance in closed form, splitting the over-critical
  case (a lightweight walking surface, no rebound, force tending to
  `m v0/Ti`) from the under-critical one (concrete, full rebound, `2 m v0/Ti`),
  the 6 dB that separates a tapping-machine level on timber from one on
  concrete. From it follow `force_pulse`, `tapping_cut_off_frequency`,
  `hammer_limiting_frequency`, `plate_contact_stiffness`,
  `covering_contact_stiffness` and `short_pulse_mean_square_force`. A soft
  covering then alters nothing but that force, so `covering_improvement`
  returns the force ratio `20 lg(|Fn|without/|Fn|with)` of Eq. (4.114) both per
  band (`improvement`, summed in mean square over the Fourier lines the band
  contains, since the machine excites a line spectrum) and per line
  (`line_improvement`, which carries the truncation nulls at odd multiples of
  the cut-off), beside the two-line design estimate, 0 dB below the covering's
  cut-off and 12 dB per octave above. Floating floors get their resonance from
  `160 sqrt(s'/m')`
  (`floating_floor_resonance_frequency`), the three improvement laws above it
  through `floating_floor_improvement_spectrum` (the 30 lg law of
  EN 12354-2 for sand-cement screeds, Cremer's 40 lg law for asphalt and dry
  floors, and the same law with the hammer-impedance term for a lightweight
  walking surface), the weighted single numbers from
  `weighted_floating_floor_improvement`, the series combination of layers from
  `combined_dynamic_stiffness`, the two resonances of a stacked pair from
  `double_floating_floor_resonances`, and Ver's discrete-mount SEA model, which
  rises 30 dB per decade rather than 40, from `resilient_mount_improvement`.
  Wall linings close the set: `lining_resonance_frequency` for a layer bonded
  directly or built on studs over a filled cavity, `weighted_lining_improvement`
  for the Annex D table that reads the rating off that resonance (a lining
  below 200 Hz helps, above it costs up to 10 dB), `lining_improvement` for the
  exterior-system and stud fits with their anchor and glued-area corrections,
  and `lining_improvement_in_situ` for the laboratory-to-field transfer.
  Anchored on the 21 printed per-band values of ISO 12354-2:2017 Table G.4 and
  its `DeltaLw = 32,2 dB`, reproduced to the table's printed precision of
  0,1 dB, on Hopkins's printed cut-off frequencies (7 000, 2 300 and 100 Hz),
  on the over/under-critical classification of his four walking surfaces, on
  the double-floating-floor resonances of his Fig. 4.73 (74 Hz and 195 Hz) and
  the lining resonances of his Fig. 4.48 (104, 123, 247 and 542 Hz), and on
  the identity between the dominant term of Vigran Eq. (8.45) and Hopkins
  Eq. (4.118), which state the same SEA model in different algebra.
  Table D.1 is read at the one-third-octave band that *contains* the
  resonance, by ISO 266 band edges rather than by proximity to a nominal
  label, as Clause D.2.2 requires: the two differ by up to 8,8 dB at the
  160 Hz to 200 Hz boundary. Where a formula has no published worked example
  behind it (the cavity stiffness of Formula D.2, the asphalt fit of
  Formula C.5, and Formulae D.3 to D.8) the guide says so. Four results with
  `.plot()`, two figures and a new guide "Predicting Resilient-Layer
  Performance" in English and Spanish.

- Two entries in `docs/ERRATA.md`, and a standing rule at the top of that file.
  ISO 12354-1:2017 Table D.1 covers 1 600 Hz with two rows carrying different
  values, "630 to 1 600 -> -10" and "1 600 <= f0 <= 5 000 -> -5"; the
  predecessor EN 12354-1:2000 Table D.3 printed the second as "> 1 600"
  strictly, so the 2017 rewrite is what created the overlap, and the library
  keeps the 2000 reading of -10 dB at exactly 1 600 Hz. Vigran, *Building
  Acoustics* (2008), gives the carpet-squares stiffness in the caption of
  Fig. 8.37 as 3.2e6 N/m where the body text on p. 321 says it is the same as
  Fig. 8.36's, printed as 3.2e5 N/m, the only value that reproduces the
  "approximately 130 Hz" of p. 320. The standing rule is that no errata claim
  about a constant, coefficient, exponent or symbol may rest on text extracted
  from a PDF: extraction drops radicals silently, so the page is rendered as
  an image at 600 dpi and read.
- Three checks over `docs/ERRATA.md`, and a contributor tool for the sources
  it cites. `scripts/check_errata_evidence.py` requires every entry to cite the
  page its claim rests on (the edition, the PDF page index and the printed
  folio) or to sit
  on an explicit, shrinking allowlist with a reason; rejects an entry that
  describes how that page was read, which is method rather than evidence; and
  separately flags any
  multiplicative claim whose ratio lands within 0,5 % of `√2`, `√3`, `π`, `2π`,
  `1/√2`, `ln 2` or a small integer, which is the arithmetic signature of a
  glyph lost in extraction rather than an author's error. A flagged entry has
  to cite its page; the allowlist does not excuse it. It runs in CI as the
  `errata-evidence` job. `scripts/glyph_census.py` is the companion tool, not a
  gate: it reports whether a document's text layer emits any `√` or `−` at all,
  and counts the C0 control characters and Latin-1 ligature stand-ins that mark
  a broken encoding, so that a source can be screened before anything is quoted
  from it. `CONTRIBUTING.md` states the rule the two enforce.

- A test that can tell `combined_tone_level` (ISO/PAS 20065 Formula (17) with
  Anmerkung 2) from a plain union sum. The Annex E "2 FG" oracle cannot: its
  three tones occupy disjoint line runs, so counting each line once and summing
  the three tone levels give the same 72,15 dB. The new case pairs tones whose
  runs coincide and tones whose runs nest, in the same committed spectrum, and
  pins the exact gap an implementation without the deduplication would open
  (10 lg 2 and 2,91 dB). Removing the deduplication now fails the suite.

- Room-to-room noise reduction (`noise_control/room_to_room.py`, Norton &
  Karczub 2003 Section 4.9): `room_to_room_transmission` composes the whole
  chain a practitioner runs, from the reverberant level a machine builds up in
  the source room, through the transmission loss of the partition and the
  equivalent absorption area of the receiving room, to the received octave-band
  spectrum and its room-criterion verdict. The step that lived nowhere else is
  the steady-state power balance of Eq. (4.101),
  `NR = TL - 10 lg[S_w/(S_2 alpha_2 + tau S_w)]`, which says that the noise
  reduction is not the transmission loss: a large partition into a hard room
  delivers less, a small one into an absorbing room more. The `tau S_w` term is
  optional (`include_partition_transmission`), and `flanking_penalty` records
  the allowance Norton warns about for flanking paths and air leaks.
  `RoomToRoomResult` carries every intermediate spectrum, rates itself against
  an NC or RC target (`rating`, `criterion_curve`, `exceedance`,
  `meets_target`), inverts the chain into the partition `TL` that would just
  meet the criterion (`required_transmission_loss`), prints the rows of the
  hand calculation with `.table()` and draws the whole path with `.plot()`.
  Anchored end to end on the printed octave-band answers to Norton's problems
  4.18 (72.3/60.4/41.4/41.0/33.8/30.7 dB in the operator room, to 0.1 dB) and
  4.21 (the noise reduction of three partitions into one receiving room, to
  0.05 dB). New guide "Room to room: partition, receiving room, criterion" with
  the plant-room chain figure, in English and Spanish.

- Enclosure transmission loss solved the other way
  (`enclosure_required_transmission_loss`, Norton & Karczub 2003 Eq. (4.115)):
  given the insertion loss an enclosure has to deliver, the panel `R` its walls
  need, which is what an enclosure is actually specified from. It returns the
  same `EnclosureResult` as the forward calculation, so it plots and reports
  identically. `enclosure_insertion_loss` gains `model` (`ENCLOSURE_MODELS`) to
  choose between the Bies, Hansen & Howard Eq. (7.111) correction, which
  carries a `0.3` inside the logarithm and floors the insertion loss of a fully
  lined enclosure at `R + 5.2 dB`, and Norton's Eq. (4.115), which has no such
  floor; the two diverge once the lining dominates, and reproducing a published
  answer needs the model its author used. Anchored on Norton's problem 4.16
  (the transmission loss a lined compressor enclosure needs to reach an NC-45
  curve, eight octave bands to 0.15 dB).

- `steady_state_spl` accepts `distance=None` for the reverberant field alone,
  `Lp = Lw + 10 lg(4/R)`, the `r -> infinity` limit of Bies Eq. (6.43) and the
  level a diffuse-field room-to-room calculation works from; and gains
  `source_model` (`SOURCE_POWER_MODELS`) for the three sound power models of
  Norton & Karczub Table 4.5, where a constant-power source radiates the same
  power wherever it stands, a constant-volume source is loaded by nearby
  reflecting boundaries and radiates `10 lg Q` more (the conservative upper
  bound a design estimate uses), and a constant-pressure source `10 lg Q` less.

- Orthotropic (ribbed and corrugated) panel sound insulation
  (`building/panel_transmission.py`, Vigran, *Building Acoustics* Sections
  3.7.3.3 and 6.5.3; Bies, Hansen & Howard 5e Section 7.2.4.5). Cladding that
  is stiff along the corrugations and limp across them has no single
  coincidence frequency but a whole range:
  `orthotropic_critical_frequencies` returns the pair `(fc1, fc2)` from the two
  principal bending stiffnesses (Vigran Eq. 6.107), and
  `orthotropic_transmission_loss` predicts the flattened `R(f)` over it, either
  by numerically averaging the azimuth-dependent transmission coefficient of
  Heckl (1960) and Hansen (1993) over the diffuse field (Vigran
  Eqs. 6.108 to 6.111 = Bies Eqs. 7.30, 7.31 and 7.38, with the size-dependent
  limiting angle of Bies Eq. 7.36 or a fixed one) or by Heckl's closed-form
  design chart (`method="heckl"`, Bies Fig. 7.9(b) and Eqs. 7.59 and 7.60).
  `SoundReductionResult` gains `critical_frequency_upper`, and `.plot()` shades
  the coincidence range. For the common sinusoidal profile,
  `corrugated_plate_stiffness` gives the Timoshenko and Woinowsky-Krieger
  equivalent stiffnesses (Vigran Eq. 3.115) and `corrugated_plate_mass_factor`
  the developed-length increase in surface density that has to accompany them;
  `orthotropic_plate_resonance` is the matching eigenfrequency of a simply
  supported orthotropic plate (Vigran Eq. 3.113 = Bies Eq. 7.27, after Hearmon
  1959), whose lowest mode bounds the validity of both transmission models.
  Anchored on Vigran's own worked example (a 1 mm steel plate 1 m by 1 m: 4,9
  and 19,7 Hz flat, 25,5 and 102 Hz corrugated, which only come out with the
  developed-length mass), on the printed design-chart constants of Bies
  Fig. 7.9(b) (-54, -13,2, -17 and -23 dB for `rho c` = 414), on the 25 `h fc`
  products of Hopkins Table A2, and on the exact reduction of the diffuse
  integral to its elementary mass-law form and of the orthotropic impedance to
  Cremer's isotropic one. New section and figure in the panel-insulation
  guide, in English and Spanish.

- Limp-frame equivalent fluid (`limp_frame` in
  `materials/porous_absorber.py`, Allard & Atalla, *Propagation of Sound in
  Porous Media* 2e Section 11.3.4, Eqs. 11.53 to 11.55 after Panneton 2007).
  Delany-Bazley, Miki and Johnson-Champoux-Allard all assume a motionless
  frame, which only holds above the Zwikker-Kosten decoupling frequency now
  exposed as `decoupling_frequency`; below it a light frame (felts, screens,
  aeronautic-grade fibreglass) is dragged along by the pore fluid and its
  inertia has to be carried. `limp_frame` corrects the effective density of any
  rigid-frame `PorousMediumResult`, keeping its bulk modulus, and returns
  another `PorousMediumResult` that drops straight into a `PorousLayer` of the
  transfer-matrix stack. `limp_frame_applicable` and `LIMP_FRAME_CRITERIA`
  carry the published rules of thumb for when the frame may be treated as limp
  at all (Beranek's `|Kc/Kf| < 0,05` and the 0,2 to which Doutres et al. 2007
  relaxed it, which for air the book states as a "bulk modulus lower than 20
  kPa"). Closed-form rather than digit-oracled: Allard & Atalla publishes no
  table of computed limp densities, so what anchors the model is the printed
  equation itself, transcribed term by term. The book's own two exact limits
  are verified as well (a heavy frame recovers the rigid-frame result digit for
  digit; the density converges on the apparent total density
  `rho_t = rho1 + phi rho0` as the frequency falls, where the rigid model
  diverges), but they corroborate rather than pin the form: a sign-flipped
  variant of the same equation satisfies both. The decoupling frequency of
  their Table 6.1 glass wool, 43,27 Hz, is pure arithmetic. New section and
  figure in the porous-absorber guide, in English and Spanish.

- Kinematic machine fault frequencies (`vibration/machine_diagnostics.py`,
  Norton & Karczub 2003 Section 8.4): `bearing_fault_frequencies` returns the
  seven rolling-contact bearing lines after Shahan and Kamperman (shaft, cage
  and its relative rate, rolling-element rotation and spin, and the outer- and
  inner-race element pass frequencies BPFO and BPFI, Eqs. 8.4 to 8.14);
  `gear_mesh_frequencies` the mesh frequency, its harmonics and their sideband
  families (Eq. 8.3); `induction_motor_frequencies` the supply, slip,
  pole-pass and rotor-slot harmonic lines with their eccentricity sidebands
  (Eqs. 8.19 and 8.20); and `blade_pass_frequencies` the blade-passing tones
  of a fan, blower or pump together with the lobed interaction patterns of a
  ducted axial fan (Eqs. 8.15 to 8.18), named `lobe n=1 m=2` after the blade
  harmonic and the lobe count, since Eq. (8.17) carries `n` and the same lobe
  count reached from another harmonic turns at another speed. The
  `FaultFrequencyResult` carrier selects lines by name, in shaft orders or by
  frequency span, merges families with `combine_fault_lines`, and its
  `.plot(spectrum=...)` draws the predicted lines over a measured envelope
  spectrum, closing the loop with the existing `envelope_spectrum`, `cepstrum`
  and `time_synchronous_average` chain. Anchored on the printed answers to
  Norton's problems 8.5 (BPFO 207 Hz, BPFI 293 Hz), 8.6 and 8.7, plus the
  exact identities `BPFO + BPFI = Z fs` and `BPFO = Z x FTF`. New guide
  "Machine fault frequencies" with the bearing-fault envelope figure, in
  English and Spanish.

- Experimental statistical energy analysis (`vibration/experimental_sea.py`,
  Norton & Karczub 2003 Sections 6.3.3, 6.3.4 and 6.4.1):
  `power_injection_clf` inverts the steady-state two-subsystem power balance
  from measured band energies, giving both coupling loss factors through the
  consistency relationship and the injected power (Eqs. 6.8, 6.10, 6.11 and
  6.15), and `power_injection_matrix` solves the full two-drive power-injection
  problem for all four loss factors with no reciprocity assumed, leaving it as
  a check on the measurement. The modal densities the inversion needs come with
  it: `bar_modal_density`, `beam_modal_density`, `flat_plate_modal_density`,
  `ring_frequency` and `cylindrical_shell_modal_density` (Eqs. 6.23 to 6.29).
  `PowerInjectionResult` reports the transmitted, dissipated and input powers,
  the coupling strength and the implied modal-density ratio, and plots the
  loss-factor budget. Anchored on the printed answer to Norton's problem 6.10
  (eta12 = 4.26e-4, eta21 = 3.92e-4, 1.31 W).

- Two closed-form junction coupling loss factors in
  `vibration/junction_transmission.py`: `right_angle_transmission_coefficient`
  (Norton Eqs. 6.53 to 6.55, the normal- and random-incidence coefficient of
  two plates meeting at right angles, which reproduces Norton Eq. 6.52 when fed
  to the existing `coupling_loss_factor`) and
  `point_connection_coupling_loss_factor` (Eq. 6.56, plates joined at N bolts,
  rivets or spot welds, falling as 1/f instead of the line junction's
  1/sqrt(f)). Both anchored on the printed eta12 columns of Norton's problem
  6.13. New "Experimental SEA" section and comparison figure in the
  plate-junction guide, in English and Spanish.

- Plateau-method transmission-loss estimate (`plateau_transmission_loss`,
  `PLATEAU_MATERIALS` in `building/panel_transmission.py`, Norton & Karczub
  2003 Section 3.9.1 and Table 3.1): the Watters construction of a
  field-incidence mass law, a horizontal coincidence plateau between points A
  and B set by the material's tabulated height and frequency ratio, and a 10
  dB per octave recovery above B, for the eight tabulated materials or an
  explicit panel. The result carries its plateau height and the two
  construction frequencies, and `.plot()` shades the plateau with a
  background-aware wash. `single_panel_transmission_loss` gains
  `coincidence_model="cremer"` (Norton Eq. 3.110) alongside the existing Sharp
  model, floored at `TL = 0 dB`, which is the value Eq. (3.109) gives at the
  critical frequency itself and the hard bound of a passive panel; and both it
  and `mass_law_transmission_loss` accept an explicit `field_correction`.
  Anchored on the printed answers to Norton's problems 3.11 and 3.14. New
  plateau-versus-physical comparison figure in the panel insulation guide, in
  English and Spanish.

- Four entries in `docs/ERRATA.md` for Norton & Karczub (2003): the missing
  square in the denominator of Eq. (6.56), the plate areas dropped from the
  reciprocity step of the problem 6.13 answer, the platform area of problem
  6.10 that its own answers contradict, and the structural loss factor of
  problem 3.14 that is a decade away from the value its answers require.

- Normal modes of a rectangular room (`room/room_modes.py`): `room_modes`
  enumerates every eigenfrequency of a rigid-walled shoebox up to a limit,
  sorted and classified as axial, tangential or oblique, and the pieces
  `room_mode_frequency`, `room_mode_count` and `room_modal_density` expose the
  closed forms directly (Long, *Architectural Acoustics* 2e, Equations (8.43),
  (8.45) and (8.46), the Morse/Pierce lattice count with its wall-plane and
  axis corrections). The result reuses `room.schroeder_frequency` rather than
  reimplementing it, and its `.plot()` draws the mode ladder coloured by
  family above the modal-density curve with the Schroeder frequency marked on
  both panels. Anchored on Long's Table 8.1 for a 7 x 5 x 3 m room (24.6,
  34.5, 42.4, 49.2, 57.4 and 60.1 Hz) and on the 34 modes per hertz he states
  for the same room at 1 kHz; two new conformance checks, a new figure
  `rectangular_room_modes`, and a section in the image-source guide in English
  and Spanish.

- Crowd self-noise of an occupied room (`room/crowd_noise.py`): `crowd_noise`
  evaluates the reverberant level a room's own occupants generate against
  occupancy for a set of absorption areas, with `speech_direct_level`,
  `crowd_noise_level`, `speech_to_noise_ratio` and `absorption_per_table` as
  the building blocks (Long Chapter 17, Equations (17.50) to (17.54)). The
  speech-to-noise ratio drops both the talker power and the number of talkers,
  so the design variable is the absorption *per table*; `.plot()` draws the
  occupancy sweep against the direct speech level and the -6 dB communication
  limit. Anchored on Long's printed restaurant case (60 dB direct field at
  1.2 m, 63 dB with one talker, 76 dB with twenty, 66 dB after an alpha 0.9
  ceiling adds 170 metric sabins) and on his 3.16 rt² privacy constant; two
  new conformance checks, a new figure `restaurant_crowd_noise`, and a section
  in the open-plan guide in English and Spanish. It is a prediction model and
  stays separate from the ISO 3382-3 measurement of `room/open_plan.py`. The
  6.33 constant Long prints for Equation (17.53) is registered in
  `docs/ERRATA.md`: the same closed form that reproduces his own 3.16 gives
  6.31.

- Gain before feedback of a sound-reinforcement system
  (`electroacoustics/sound_reinforcement.py`): `feedback_stability` returns
  the open-loop gain, the feedback-loop gain, the open-microphone correction
  and the resulting headroom against the oscillation threshold, with
  `feedback_loop_gain` and `open_microphone_correction` callable directly
  (Long Chapter 18, Equations (18.16) to (18.24), with the 10 dB stability
  margin he adopts for an equalised system). `.plot()` draws the gain
  structure against the oscillation and margin lines, and
  `plot_sound_reinforcement_geometry` sketches the four points of the loop
  with each path length annotated. Long publishes no fully numeric worked case
  for this chain, so the tests and the three new conformance checks anchor on
  the closed forms and on the special cases he states in words (the
  omnidirectional `L(H-M) <= L(H-L) - 4` and the cardioid `L(H-L) - 2` at
  Zs = -6 dB, and `10 lg Nm`). Note that Equation (18.24) is printed with the
  sign of the microphone directivity flipped relative to Equations (18.20) to
  (18.22); the implementation follows the latter, which is what reproduces
  Long's own special cases, and the defect is registered in `docs/ERRATA.md`.
  New figure `sound_reinforcement_geometry` and a section in the loudspeaker
  guide in English and Spanish.

- Public ERB_N and Cam utilities (`psychoacoustics/erb_scale.py`):
  `erb_bandwidth`, `cam_from_frequency` and `frequency_from_cam` expose the
  Glasberg and Moore (1990) auditory-filter bandwidth and its Cam frequency
  scale (Moore, *An Introduction to the Psychology of Hearing* 6e, pp. 76-77),
  with the constants `ERB_C1`, `ERB_C2` and `CAM_C`. The ISO 532-2 loudness
  model now imports these functions instead of keeping its own private copies,
  so the two cannot drift apart. Two new conformance checks, a new figure
  `erb_bandwidth`, and a section in the advanced-loudness guide in English and
  Spanish.

- Spanish noise regulation RD 1367/2007 (`environmental/spanish_regulation.py`):
  the corrected level `LKeq,T = LAeq,T + Kt + Kf + Ki` of Annex I A.2 c with
  the three reference procedures of Annex IV A.3.3, each graded 0/3/6 dB and
  their sum capped at 9 dB. `tonal_correction` works on the unweighted
  one-third-octave spectrum, comparing each band against the arithmetic mean of
  its two neighbours with the 8/12, 5/8 and 3/5 dB thresholds of the three band
  ranges; `low_frequency_correction` and `impulsive_correction` grade
  `LCeq - LAeq` and `LAIeq - LAeq`. These are the regulation's own variants,
  not the ISO 1996 procedures the library already carries, and the module
  documents where each differs. `NoisePhase`, `evaluation_period_level` and
  `long_term_corrected_level` integrate phases into the day, evening and night
  indices and into the annual `LK,x`, with the regulation's own rounding
  (`round_reported_level`, add 0,5 dB and take the integer part). The Annex II
  quality objectives (outdoor, indoor and vibration) and the Annex III
  immission limits (new transport infrastructure, its `LAmax` variant,
  activities and ports, and noise transmitted to acoustically adjacent
  premises) are exposed as lookups, and `assess_activity` applies the three
  criteria of Article 25 to produce an `ActivityAssessment` with `.plot()` and
  a one-page `.report()` inspection fiche in Spanish. Each correction is
  validated against the 0/3/6 dB grading of the Annex IV A.3.3 tables, and a
  new activity requires the annual index, since Article 25.1 b i is mandatory
  for it and a verdict that never evaluates a mandatory criterion would
  assert more than the assessment has shown. Validated against
  Ejemplos 3.1 to 3.3 of Aviles Lopez & Perera Martin, *Manual de acustica
  ambiental y arquitectonica* (`LKeq,d` 57 dB, `LKeq,e` 51 dB, `LK,d` 56 dB,
  `LK,e` 50 dB and the verdict that flips between a new activity and one
  already in operation), plus the printed limit tables of the regulation
  itself. New figure `rd1367_activity_assessment`, three conformance checks and
  a Spanish-first guide, in Spanish and English.

- Spanish building code CTE DB-HR (`building/spanish_building_code.py`): the
  A-weighted global indices `RA`, `RA,tr`, `DnT,A`, `D2m,nT,A` and
  `D2m,nT,Atr` computed by the direct Annex A formulae (A.5) to (A.7) over the
  eighteen one-third-octave bands 100 Hz to 5 kHz, two more than the sixteen
  of ISO 717-1, with the four normalised source spectra of Tables A.2 to A.5
  (pink noise, road traffic, railway and aircraft). The facade quantity
  follows clause 3.1.3.4 point 1: a railway-dominant site is assessed as
  `D2m,nT,A` through Formula (A.5) and a road-traffic or aircraft one as
  `D2m,nT,Atr` through (A.6), so `d2m_nt_a` and `d2m_nt_atr` are separate
  entry points that each refuse the other's spectra rather than one function
  that could put the wrong quantity name on a report. Results expose the
  exact value, the one-decimal intermediate and the reported integer that
  DB-HR 3.1.3.1 point 4 prescribes. The requirement side covers the facade
  table 2.1 keyed on the day noise index `Ld` (with the sheltered-facade and
  aircraft-noise rules), the airborne and shared-opening requirements of
  2.1.1, the party-wall alternatives of 2.1.1 c), the impact limits of 2.1.2,
  the reverberation and absorption limits of 2.2, and the window-size
  correction of the Catalogo de Elementos Constructivos, checked through
  `check_db_hr_requirement` / `assess_db_hr` with `.plot()`. Validated against
  Ejemplo 7.2 (`R'A` 51,4 dBA), Ejercicio 7.1 (`D2m,nT,Atr` 32,8 dBA) and
  Ejemplo 7.4 of the same Manual; the direct route is cross-checked against the
  ISO 717-1 route of `weighted_rating_extended`, which reproduces the book's
  `R'w` 52 dB with `C` -1 and `Ctr` -5 and agrees to the integer on both
  published spectra. New figure `dbhr_global_index`, four conformance checks
  and a guide in English and Spanish.
- End-to-end HVAC duct-borne noise calculation. `duct_path` (new
  `noise_control/duct_path.py`) cascades a fan-to-room path element by
  element, subtracting each attenuation, combining each element's regenerated
  sound power on an energy basis over a 0 dB self-noise floor, applying the
  room effect and rating the received spectrum against an NC or RC design
  curve; `combine_duct_paths` adds a supply and a return path. The
  `DuctPathResult` exposes the calculation as a printed sheet (`.table()`,
  one row per element with the *Sum*, *Self-noise* and *Combined* rows of the
  published worksheets), a `.plot()` cascade against the criterion curve and a
  one-page `.report()` sheet in the AHRI Standard 885 Table 8 layout, which
  elides its middle rows rather than spilling onto a second page.
  The element models it chains are new in `noise_control/hvac.py`, all from
  Long, *Architectural Acoustics* 2nd ed., Chapters 13-14: `fan_sound_power`
  (Eq. 13.1 with the ASHRAE Tables 13.5-13.7) with
  `fan_efficiency_correction`, `blade_passing_frequency` and
  `fan_casing_attenuation`; `unlined_rectangular_duct_attenuation` and
  `unlined_circular_duct_attenuation` (Eqs. 14.9-14.11, Table 14.1);
  `lined_rectangular_duct_attenuation` and `lined_circular_duct_attenuation`
  (the Reynolds regressions, Eqs. 14.12-14.13 with Tables 14.2-14.3);
  `diffuser_sound_power`, the regenerated noise of a grille, register or
  diffuser from its face area, approach velocity and pressure drop
  (Eqs. 13.27-13.33); `flexible_duct_insertion_loss` (Table 14.4);
  `split_loss` (Eq. 14.17);
  `end_reflection_loss_closed_form` and `equivalent_diameter`
  (Eqs. 14.14-14.16, also reachable as `end_reflection_loss(method="long")`
  beside the existing Bies table look-up); `silencer_self_noise` (Eq. 14.31
  with Table 14.8); `splitter_silencer_insertion_loss`, the dissipative
  parallel-baffle unit as airways combined by Bies Eq. (8.241); `room_effect`;
  and the two ASHRAE (2019) Applications Handbook Chapter 49 screening tables,
  `air_terminal_velocity_limit` (Table 9) and `air_terminal_damper_correction`
  (Table 10). Validated end to end against Long's worked Table 14.9, whose
  supply and return sheets reproduce row for row within the 1 dB the printed
  sheet carries, and element by element against the tables themselves; six
  new conformance rows. The guide records, band by band, which rows of that
  worked sheet the book's own printed tables do and do not reproduce.

- Higher-order duct modes with mean flow (new `noise_control/duct_modes.py`):
  `circular_duct_cut_on` and `rectangular_duct_cut_on` return the cut-on
  frequency and the axial wavenumber at cut-on of each mode, with the uniform
  mean-flow correction (Norton & Karczub, *Fundamentals of Noise and Vibration
  Analysis for Engineers* 2nd ed., Eqs. 7.6-7.10 and the Table 7.1
  eigenvalues), and `DuctModeResult.plot()` draws the cut-on ladder with and
  without flow. `plane_wave_limit` gives the frequency below which a duct
  carries plane waves only, and the plane-wave methods now say where they stop
  being valid: every `ReactiveSilencerResult` carries its `plane_wave_limit`
  and raises the new `PlaneWaveWarning` when the analysis grid reaches past
  it, as does `duct_path` when a duct section is declared. Validated against
  the printed answers to Norton's problems 7.1 and 7.2. Three new conformance
  rows.

- IEC 61043 instrument-class verification for two-microphone sound-intensity
  chains (new `metrology/intensity_compliance.py`). `residual_index_limits`
  returns the Table 2 minimum pressure-residual intensity index per
  one-third-octave band from 50 Hz to 6.3 kHz for a probe, a processor or a
  complete instrument in class 1 and class 2, rescaled to any microphone
  separation by the Note 1 rule `+10 lg(x/25)`; `verify_intensity_class` and
  `intensity_class_compliance` grade a measured `δpI0` spectrum against both
  masks band by band, returning the class the chain meets, the binding
  margin, the bands that miss it and a `range_limited` flag when the verified
  bands cover neither frequency range of clause 6.1;
  `instrument_class_from_components` applies the clause 8 rule for a probe
  and a processor supplied separately. The result exposes `.plot()` (measured
  index over the two Table 2 masks, with the pass region shaded and the
  blocking bands ringed) and `.report()`, a one-page verification fiche laid
  out like a published accredited calibration certificate. New
  `phase_mismatch_from_residual_index` and
  `residual_index_from_phase_mismatch` convert between the index and the
  equivalent channel phase mismatch, `δpI0 = 10 lg(kd/φs)`. Anchored on
  Table 2 of EN 61043:1994 transcribed digit for digit and cross-checked
  against its reproduction as Table 6.1 in Fahy, *Sound Intensity* 2nd ed.,
  and on Fahy's worked check that 20 dB at 1 kHz over 25 mm is a mismatch of
  about 0.26 degrees. Clause 6.1 follows the EN/IEC text, in which class 2 is
  met over either the one-third-octave range or the octave one; the official
  Spanish version UNE-EN 61043:1999 states only the octave alternative, and
  that omission is recorded in `docs/ERRATA.md`. Four new conformance checks,
  a new figure `intensity_class`, and two sections in the intensity guide in
  English and Spanish.

- ISO 9614-1 field indicator F1, the temporal variability of the sound field
  (`temporal_variability_indicator`, equations (A.1)-(A.2)): the coefficient
  of variation of the M short-time-averaged normal-intensity samples taken at
  one typical position in the initial test (clause 8.2). `field_indicators`
  accepts those samples as `temporal_intensity` and fills `f1` on the result
  alongside F2, F3 and F4, `FieldIndicators.field_is_stationary()` checks it
  against the Table B.3 limit of 0.6 (`TEMPORAL_VARIABILITY_LIMIT`), and the
  indicator plot draws F1 and that limit beside F4. ISO 9614-1 publishes no
  worked example for F1, so it is anchored on the closed form of equation
  (A.1) and on identity checks against F4, which is the same statistic taken
  over the measurement positions instead of the time samples.

- Detailed per-band building prediction (ISO 12354-1/-2:2017 Clause 4.2), the
  per-band counterpart of the simplified single-number model: new
  `building/detailed_prediction.py` with the Annex B element performance
  (`bending_radiation_factor`, `forced_radiation_factor`,
  `calculated_sound_reduction_index` with its high-frequency plateau, and
  `bare_floor_impact_level`), the Annex C damping chain
  (`perimeter_absorption_coefficient`, `in_situ_total_loss_factor`,
  `laboratory_total_loss_factor`, `structural_reverberation_time`), the
  in-situ conversions (`in_situ_reduction_index`, `in_situ_impact_level`,
  `in_situ_equivalent_absorption_length`,
  `in_situ_velocity_level_difference`), the per-band paths for Type A
  (`flanking_reduction_index`, `flanking_impact_level`) and Type B elements
  (`flanking_reduction_index_from_normalized_difference`,
  `flanking_reduction_index_from_flanking_level`,
  `flanking_impact_level_from_normalized_difference`,
  `flanking_impact_level_from_flanking_level`,
  `resonant_sound_reduction_index`), and the assembly into `R'` and `L'n` per
  band with their ISO 717 ratings (`detailed_airborne_prediction`,
  `detailed_impact_prediction`). `HomogeneousElement` and `in_situ_element`
  run the whole chain for one element in one call; `DetailedAirborneResult`
  and `DetailedImpactResult` expose each path's share of the transmitted
  energy per band, `.plot()` them as stacked path-contribution bars with the
  apparent quantity overlaid, and `.report()` a one-page detailed prediction
  fiche. Also `floating_floor_improvement` (Part 2 Annex C) and
  `reciprocity_impact_level` (Part 2 Formulae B.3/B.4). Anchored band by band
  on the worked example that ISO 12354-1 Annex L and ISO 12354-2 Annex G share
  (about twenty printed tables for one building) plus its lightweight
  wood-frame second example and ISO 12354-2 Table B.2; three new conformance
  checks, a new figure `detailed_prediction_paths`, and a "Detailed Per-Band
  Prediction (ISO 12354)" guide in English and Spanish.

- Weston's shallow-water propagation regimes (`underwater/weston_regimes.py`):
  `weston_propagation_loss` assembles the composite propagation loss over the
  four successive regimes (spherical, cylindrical, mode stripping at 15 lg r,
  single mode) and returns each regime's own law alongside it, with
  `weston_regime_boundaries` giving the three transition ranges, the effective
  depth, the waveguide cut-off frequency and the number of cut-on modes.
  `reflection_loss_gradient`, `critical_grazing_angle`, `effective_depth`,
  `waveguide_cutoff_frequency` and `loss_parameter` expose the underlying
  closed forms, and the two characteristic seabeds of Ainslie's Table 9.1 ship
  as `WESTON_SEABEDS`. Because the energy-flux result is incoherent it doubles
  as the analytic reference the numerical solvers lacked: with a totally
  reflecting bottom the cylindrical branch reduces to the exact many-mode limit
  F = pi/(rH), which the depth- and range-averaged `normal_modes` loss
  reproduces to within a decibel and `parabolic_equation` to within its known
  paraxial bias. Three new conformance checks, a `weston_regimes` figure and a
  new section of the underwater propagation guide, in English and Spanish.

- Marine-mammal hearing and regulatory noise-exposure criteria
  (`underwater/marine_mammal_audiograms.py` and
  `underwater/marine_mammal_weighting.py`). `group_audiogram` implements the
  Southall et al. (2019) group audiogram in both its absolute and normalised
  fits, and `orca_audiogram` the three-branch killer-whale audiogram of Ainslie
  Eq. (11.159). `auditory_weighting` implements the band-pass auditory
  weighting and exposure functions with the guidance version explicit and
  carried on the result: NMFS 2024 v3.0 by default, with NMFS 2018 v2.0 and
  Southall et al. 2019 selectable. `exposure_criteria` returns the published
  TTS and injury (PTS / AUD INJ) onset criteria including the unweighted
  peak-SPL half of the impulsive dual metric, and `weighted_exposure` runs the
  assessment end to end: weighting a band spectrum, accumulating it over events
  and reporting the margin against every applicable criterion.
  `strike_sel_spectrum` (in `pile_driving_noise.py`) supplies that band
  spectrum from a recorded pile strike by a Parseval split of the sound
  exposure, so a piling campaign now goes from waveform to verdict. Validated
  against the NMFS 2018 Appendix D worked example (W at 1 kHz for the five
  groups), C recomputed as the peak of W for all three parameter sets, the
  published Tw = K + C and injury = TTS + 20 dB identities, Southall's three
  prose audiogram thresholds and both published orca points. Four conformance
  checks, a `marine_mammal_weighting` figure and a new bilingual guide,
  "Marine-mammal noise exposure".

- `detection_range` and `detection_range_from_curve` in
  `underwater/sonar_equation.py`: the figure of merit is the maximum allowable
  transmission loss, so inverting a loss law at TL = FOM gives the range at
  which the detection probability is 50 %. The first inverts the closed-form
  loss (monotone, single root); the second reads the crossing off any computed
  curve, including the oscillatory loss of a real waveguide, where `crossing`
  selects the first or last one. Ainslie's seven numeric sonar worked examples
  are now wired as an end-to-end test suite over the sonar equation and the
  propagation modules, reproducing every published figure of merit and the
  detection ranges of 1.3 km, 0.9 km and 241 / 149 / 117 m.

- Medwin (1975) branch of `sea_water_sound_speed` (`model="medwin"`), the
  simplest member of the sound-speed family, alongside UNESCO/Chen-Millero,
  Del Grosso and Mackenzie. Pinned by the two partial derivatives Ainslie
  prints with it and cross-checked against the other three over their common
  validity range.

- 2D near-to-far-field (NTFF) transformation for the FDTD solver.
  `FDTD2D.add_contour_probe` captures the steady-state pressure and outward
  normal-velocity phasors on a closed rectangular contour of cell faces with
  on-the-fly DFT accumulators (continuous-wave runs store no time
  histories), and `far_field_from_contour` (new `simulation/ntff.py`, with
  the `ContourPhasors` carrier and its `subtract` method) evaluates the 2D
  Kirchhoff-Helmholtz integral with the outgoing free-space Green function
  `-(j/4) H0(2)(kR)`, either in the far-field limit or exactly at a finite
  radius via `scipy.special.hankel2`. Validated against analytic
  line-source oracles (omnidirectionality and absolute level of a monopole
  against the 2D Green function, the antiphase-pair array factor, the
  extinction of a non-enclosing contour) and end to end at 2 kHz against
  the library's own far-field models: the meshed 27.4 cm quadratic-residue
  diffuser against the Fraunhofer prediction, and the meshed Table-1
  metadiffuser of Jiménez et al. (2017) against both that meshed QRD and
  the TMM + Fraunhofer chain, the full-wave counterpart of the paper's
  TMM-vs-FEM cross-check. Two new conformance checks (NTFF monopole
  directivity and level), a new figure `metadiffuser_ntff_polar`, and a
  "From the near field to the far field" section in the FDTD guide, in
  English and Spanish.

- `theme_fill` and `theme_fill_alpha` in `phonometry._plot.common`: a shaded
  area region derived from the page it is drawn on, instead of a fixed
  opacity. Compositing a hue at a fixed alpha cannot be legible on both
  documentation themes at once (10 % of a mid hue is a pale tint over white
  but lands within a couple of levels of the near-black page), so the page
  colour is mixed towards the hue by the smallest amount that puts the result
  a CIEDE2000 distance of 12 away from it: a pale tint over a light page, a
  raised shade over a dark one, the same idea as the background-aware
  wave-field colormap. `theme_fill` returns the wash opaque, so it also
  survives the fiche PDF pipeline, which drops alpha; `theme_fill_alpha`
  returns the equivalent opacity for the few regions that genuinely have to
  blend (overlapping directivity lobes, nested uncertainty bands). Applied to
  every acceptance corridor, period zone, tolerance band and schematic
  material fill in the library renderers and in the documentation figures.
  The library's CIEDE2000 implementation (CIE 142-2001) is verified against
  the CIE reference data published with Sharma, Wu and Dalal (2005).

- `scripts/check_figure_contrast.py`: a legibility gate over the committed
  figures. It parses the SVGs, reconstructs every filled area region together
  with the background behind it, composites the fill with its own opacity as
  a renderer would, and fails when the perceptual distance between the two is
  below a CIEDE2000 difference of 10. Runs in the `Documentation figures up
  to date` CI job and as the last step of `make graphs`;
  `--report` prints the measured distance and WCAG contrast ratio of every
  region.

- `SECURITY.md`: a security policy written for what this library actually is,
  a computation package with no service and no credentials. Supported versions
  (fixes ship forward on the latest 3.x, no backports; Python 3.13 and 3.14 on
  Ubuntu, macOS and Windows), private reporting through GitHub security
  advisories with an acknowledgement and assessment window I can keep, a
  coordinated-disclosure statement, and a scope section that names the real
  attack surface (untrusted files reaching a reader, deserialisation, the
  release supply chain, repository automation) while stating plainly that a
  value disagreeing with a standard is a conformance defect for the issue form,
  and a defect in the standard itself an entry in `docs/ERRATA.md`. Also
  records the checks already running: CodeQL, bandit, GitGuardian, Dependabot,
  the `snyk` target and the conformance staleness gate.

- `.github/pull_request_template.md`: the repository's CI gates written out as a
  compact checklist, so the recurring failures (a stale `docs/CONFORMANCE.md`,
  an unregenerated API reference, a module missing from
  `scripts/api_taxonomy.py`, a figure not produced by `generate_graphs.py`,
  EN/ES drift) are visible before the build runs. It also asks, for any new
  normative implementation, which numeric oracle validated it and confirmation
  that the oracle is independent of the implementation.

- `make lighthouse` (`site/scripts/lighthouse-audit.mjs`): Lighthouse over a
  fixed sample of built pages against a local preview server, with JSON
  reports under `site/lighthouse-results/` and a console summary listing the
  scores and every remaining opportunity. `BASE_URL` audits the live site
  instead; `-- --desktop` switches the throttling profile.

- The three remaining band procedures of ANSI S3.5-1997 in `hearing/sii.py`.
  The standard's title is plural and so is the standard: alongside the
  one-third-octave procedure the library already had, `method=` now selects
  the critical-band procedure (21 bands, 100 Hz to 9500 Hz, Table 1), the
  equally-contributing critical-band procedure (17 bands, 300 Hz to 6400 Hz,
  Table 2) or the octave-band procedure (6 bands, Table 4). The band table and
  the geometry of the upward spread of masking became parameters of one shared
  chain rather than four copies of it: the critical-band and
  equally-contributing procedures spread the masking between tabulated band
  limits, the one-third-octave procedure between band centre frequencies, and
  the octave-band procedure omits the spread entirely because an octave band is
  already wider than the spread being modelled. `band_importance=` applies an
  alternative band-importance function to any procedure (how the Annex B
  functions for particular speech test materials are used), and
  `sii_procedure(method)` returns an `SIIProcedure` carrying that procedure's
  band centre frequencies, band limits, band-importance function, reference
  internal noise spectrum level and normal-effort standard speech spectrum
  level, with a `.plot()` that steps `Ii` over the band limits. `SIIResult`
  carries the `method` used and the fiche names it. The existing call
  signature is unchanged and still computes the one-third-octave procedure.
  All eight official ASA Working Group S3-79 test cases now run as oracles,
  two per procedure: `CB.TST` (published SII 0.273), `CB_1.TST` (0.410),
  `ECB.TST` (0.278), `ECB_1.TST` (0.410), `OCTAVE.TST` (0.491),
  `OCTAVE_1.TST` (0.323), `TO.TST` (0.445) and `TO_1.TST` (0.438); the library
  reproduces the committee's `SII.C` to within one unit in the last place of a
  double on every one. Note that `CB_1.TST` and `ECB_1.TST` are the same
  confirmation twice, since the equally-contributing bands are critical bands 3
  to 19 and their two alternative importance functions weight the same physical
  bands, the extra low critical bands contributing about 1e-19 of the local
  masking energy there, so the eight cases are seven independent ones. Those
  eight exercise the chain but not the whole of each table, so Tables 1, 3 and
  4 are additionally asserted cell by cell and twelve flat-input cases from
  `SII.C` bring every band's `Ui` and `Xi` into the calculation; a mutation
  campaign over all 209 tabulated entries confirms none survives corruption.
  Fourteen new conformance checks, a new figure `sii_band_procedures`
  overlaying the four band-importance functions, and a new section of the
  speech-intelligibility guide in English and Spanish.
  Tables 1, 2 and 4 print all four vocal-effort columns in the standard, but
  are implemented here with their normal-effort speech spectrum column only,
  the one the level-distortion factor of clause 5.7 uses; the raised, loud and
  shout spectra remain one-third-octave only. The one-third-octave procedure
  itself is unchanged bit for bit: the clause 5.4 masking slope is summed in
  the order each procedure's printed form implies, since folding the printed
  `6.353` into the bandwidth term reassociates the sum and moves the index by
  a few units in the last place.

- Brand lockup (`.github/brand/lockup.svg`, `scripts/generate_brand.py`): the
  mark and the wordmark as one asset, trimmed to its ink, with the type
  converted to outlines from the same display face and the same proportions the
  README banner uses. The documentation site inlines it as the site title, so
  the header shows the mark beside a wordmark set in the brand face while
  loading no web font and reserving no space for one, and the mark alone now
  fills the landing hero's second column, which the layout had always left
  empty. `.github/brand/logo.svg` carries the same `p-ink` and `p-grid` classes
  as the favicon and the lockup, so both marks on a page take their two teals
  from the site's theme tokens and stay in step with each other and with the
  palette under the site's own theme switch.

- Brand wordmark (`.github/brand/wordmark.svg`, `scripts/generate_brand.py`):
  the word alone, as outlines in `currentColor` and trimmed to its ink, for
  the one place the name was still live text. The landing hero's H1 set it in
  whatever sans the reader's system offered while the lockup in the header
  above it carried the brand face, so the same word appeared twice on the
  first screen in two different typefaces. The H1 now holds the outlined
  wordmark with the name behind it as visually hidden text, which keeps the
  heading a heading for a screen reader, an assistive outline and the search
  index while the artwork is hidden from the accessibility tree. Still no web
  font: the type is converted from the committed font file at generation time,
  so nothing is fetched and nothing reflows once the page has painted.

### Fixed

- `verify_weighting_class` grades the filter over the whole path a signal
  travels through `WeightingFilter.filter`, not over the designed second-order
  sections alone. With the default `high_accuracy` those sections are reached
  through an interpolation and a decimation stage, and the anti-alias filter of
  those stages carries its transition band on the *input* Nyquist frequency, so
  it dominates the response above roughly `0.9 * fs / 2`. Whenever the highest
  IEC 61672-1 Table 3 row still below Nyquist lands in that band, a verdict read
  from the sections attested a filter nobody runs: at `fs = 16` kHz the
  7 943.3 Hz row came back 0.077 dB from its design goal and earned class 1,
  while a tone through `filter()` at that frequency comes out 12.0 dB low.
  Verdicts change for exactly the rates where that happens. A and C at 16 kHz
  go from class 1 to no class at all (12.0 and 13.7 dB below their design
  goals, against a −2.5 dB class 1 and a −5.0 dB class 2 lower limit), and A
  at 32 kHz from class 1 to class 2 (16.2 dB below its goal, past the −16.0 dB
  class 1 lower limit); C at 32 kHz lands 15.3 dB below its own goal and keeps
  class 1, so the verdict there is per curve. At 44.1 kHz and above nothing
  moves. The published conformance figures follow the same corrected path: the
  A and C deviations at their 1 kHz binding frequency go from +0.000 dB to
  +0.009 and +0.008 dB at 48 kHz (headroom +0.700 dB to +0.691 and +0.692 dB),
  the A row at 96 kHz to +0.002 dB, and the worst D-weighting point from
  −0.131 dB to −0.127 dB at 8 kHz. Every conformance row still passes.

- `ray_trace` resolves a sound-speed node by the direction the ray is going
  through it, so a ray leaving a gradient discontinuity upward gets the
  gradient of the segment above rather than the one below. The node was looked
  up with `searchsorted(..., "right")` whatever the direction, which is correct
  only for a descending ray; an ascending one was handed the wrong side of the
  kink for its first step. On a profile kinked exactly at the source
  (`z = [0, 500, 1000]`, `c = [1500, 1490, 1520]`, source at 500 m) the
  upgoing ray now agrees with an independent `solve_ivp` integration at
  rtol/atol 1e-12 to 4e-12 m at 4 km, where it was 7e-3 m out. This moves
  published numbers wherever the source or a boundary sits on a profile node:
  on a Munk profile with the source on one, ray depths at 20 km move by up to
  0.43 m and travel times by 2.7e-05 s.

- `extended_tube_chamber` cascades its straight chamber element over the length
  the extensions leave, `L - L_a - L_b`, not the full chamber length. The
  junction where each extended pipe ends is where its three ducts meet, so the
  duct between the two annular side branches is the leftover section; Bies
  Example 8.2 states the same arithmetic, `L = L_a + L_b + L_c`, and the
  cross-section drawing in this package already computed it that way. Against
  Bies Figure 8.20 the previous cascade opened deep spurious dips where the
  published curves are smooth (for an area ratio of 16, 5.2 dB at
  `kL/pi = 0.6` and 5.6 dB at 1.15, where the figure holds a dome near 28 dB);
  the corrected curves reproduce the figure, and an independent 2D FDTD
  simulation of the same chamber now agrees with the model to 0.97 dB where
  the old length missed by up to 11.1 dB. Transmission and insertion loss move
  for every call that passes a non-zero extension. Extensions that exactly meet
  are accepted as the limit they are, the two annular branches shunting one
  plane; extensions that would overlap are still rejected, now with a message
  that says the straight section would come out negative.

- The coverage claims tell the truth again, all 582 of them audited claim by
  claim against the code. The errors ran almost entirely in one direction: the
  documentation undersold the library. Five claims denied functionality that
  ships and is tested (the Directive 2003/10/EC exposure verdict of the ISO
  9612 fiche, the Directive 2002/44/EC assessment with its PASS/FAIL banner,
  the IEC 60268 device parts implemented since electroacoustics landed,
  binaural summation, internal position averaging); a cluster of section
  indexes compressed their children's carefully drawn boundaries into false
  generalizations the child pages state correctly; three pages called Wells'
  closed-form plenum an interpolated table; and two Spanish twins had drifted
  from their English originals. The one flattering error, the ray solver's
  travel times that it does not compute, is corrected the other way. Thirty-one
  corrections over 17 English and 19 Spanish pages; no claim changed status,
  only its text became true.

### Changed

- `gaussian_beams` groups its options by the physics they describe, trimming
  the signature from twenty-two parameters back to thirteen. The launch
  fan's half-angle, density and initial width travel together as one
  `BeamFan` (`fan=BeamFan(max_angle_deg=45.0)` where `max_angle_deg=45.0`
  used to stand alone); the lossy fluid seabed is a `FluidSeabed` passed
  directly as `bottom`, carrying the water density with it, so a
  half-described seabed or a seabed alongside `bottom="rigid"` is unwritable
  rather than checked; the volume absorption is the bare model name or a
  `VolumeAbsorption` carrying its temperature, salinity and pH; and the
  bottom profile is one `bathymetry` pair of node arrays, in `ray_trace`
  too. `eigenrays` takes the same `bottom` union in place of its
  seabed-and-density triple. The three classes are frozen dataclasses,
  exported from `phonometry`, `phonometry.underwater` and
  `phonometry.underwater.propagation` alike, and every result field keeps
  its name, so recorded outputs read exactly as before.
- The underwater domain calls its propagation quantity by the name the standard
  gives it. ISO 18405:2017 defines *propagation loss* in 3.4.1.4 as the
  difference between the source level in a specified direction and the
  mean-square sound pressure level at a specified position, `N_PL(x) = L_S -
  L_p(x)`, which is exactly what this library computes; *transmission loss* is
  a different quantity, the reduction in a specified level between two
  specified points (3.4.1.3), and Note 6 to that entry says using it as a
  synonym of propagation loss "is deprecated". Note 7 under 3.4.1.4 deprecates
  the reverse substitution too. The domain named its function and its results
  after the deprecated synonym while citing ISO 18405 on the same pages as the
  source of its definitions.

  So `underwater.transmission_loss` is now `underwater.propagation_loss`,
  `TransmissionLossResult` is `PropagationLossResult` and its `.tl` field is
  `.pl`, and the `.transmission_loss` attribute of every underwater result
  (sonar equation, detection range, normal modes, parabolic equation) is
  `.propagation_loss`, as are the arguments of `passive_sonar_equation`,
  `active_sonar_equation` and `detection_range_from_curve`. The symbol follows
  in the prose, the figures and the diagrams: `PL`, or `N_PL` where the ISO
  level symbol is meant, never `TL`. `underwater_transmission_loss.svg` is
  `underwater_propagation_loss.svg`. Weston's regimes already returned a
  `propagation_loss` and are untouched.

  There are no aliases and no deprecation warnings. An alias would preserve the
  spelling this change exists to remove, and a reader who kept it would keep
  reading a name the standard deprecates.

  Noise control and building acoustics keep `transmission_loss`: for a
  silencer, a partition or an aperture that is the correct term for the
  reduction between two points, which is 3.4.1.3 rather than 3.4.1.4. The
  collision this resolves was real and silent: `phonometry.transmission_loss`
  used to resolve to the ocean function and shadow
  `phonometry.noise_control.transmission_loss`, so a reader following the
  silencer guide got a four-pole call answered by a spreading law. The
  top-level name is now the noise-control one.
- Fourteen names a domain exported publicly were missing from
  `phonometry.__all__`, and the API reference renders `phonometry.__all__`, so
  none of them had a page. Thirteen are now exported at the top level and
  documented: the six four-pole primitives the silencer guide teaches by name
  (`insertion_loss`, `cascade`, `duct_matrix`, `shunt_matrix`,
  `helmholtz_impedance`, `quarter_wave_impedance`), which join the
  `transmission_loss` the previous entry rebound and was already exported, the
  bending-wave speeds,
  beam impedance and rod mobility of `vibration.structural.point_mobility`
  (`beam_bending_wave_speed`, `plate_bending_wave_speed`,
  `infinite_beam_impedance`, `longitudinal_rod_mobility`),
  `plot_metadiffuser_panel_geometry`, `EnvironmentalMeasurementWarning` and
  `DeclarationForm`. Exporting them rather than teaching the generator to
  render domain-level `__all__` keeps the invariant the reference states on its
  own top-level page, that every public name can be imported straight from
  `phonometry`; the alternative would have documented the invariant as broken
  and left the curated quick table, whose gate reads the same list, blind to
  all thirteen.

  The fourteenth, `environment`'s `expanded_uncertainty`, deliberately stays
  where it is. Three domains define an expanded uncertainty over their own
  coverage factors, the top level already qualifies each of them
  (`environmental_expanded_uncertainty`, `insulation_expanded_uncertainty`),
  and claiming the unqualified name for one of the three would recreate exactly
  the shadowing the previous entry removes. Its docstring now says so, and the
  reference documents the function under the qualified name.

- Every entry of `docs/ERRATA.md` was checked again against the page it
  quotes, and the registry now cites the document rather than a file. A
  citation is the edition, the PDF page index and the printed folio, which any
  copy of that edition resolves; a filesystem path pinned the claim to one
  machine and told a reader nothing. The check that enforces it
  (`scripts/check_errata_evidence.py`) reads the same form, and its allowlist,
  which excuses an entry from citing its page, falls from forty-five entries
  to twenty-five because those pages have now been read.

  The pass corrected three of our own arithmetic slips inside accusations that
  were otherwise right: EN 12354-3 Formula (13) against its own Annex F
  example gives 1,5104 dB for the Sabine form and 1,6877 dB for the printed
  formula; Miki against Attenborough gives 1,19 from the coefficient the
  table prints, where Miki's own 0,109 gives 2,88 and the table's own
  Delany-Bazley row gives 3,10; and reading a speed in km/h into
  `f = v/lambda` places the CNOSSOS roughness spectrum a factor 3,6 too high
  in frequency, not too low. Three more entries claimed more than the page
  supports and now say what it does: the Southall table duplication follows
  from the article's own in-air 15 dB offset rather than from a column slip,
  the printed Vigran plenum model is monotonic in the damping and what it
  breaks is the finite bare-plenum limit, and no Official Journal text names a
  "Table G-1a", which was our shorthand for the first table of G-1.

- The multiple-shock time guards go through `require_positive` like the rest
  of the library. They were written as `not t > 0.0`, which rejects NaN where
  `t <= 0.0` would accept it, and that reason lived in a comment. The shared
  validator says the same thing in the form the whole package uses, rejects
  infinities too, and names the parameter that failed instead of both at once.

- The RD 1367 assessment reads as the article it implements. Its two longest
  functions were doing four jobs each: `assess_activity` now validates the
  annual parameters, resolves the period durations, assesses one period and
  derives the annual index in four named steps, and `tonal_correction`
  separates the spectrum it rejects from the band-by-band procedure of
  Annex IV A.3.3. No number changes: the conformance report, the fiches and
  the tests are identical either side of the refactor.

- `phonometry.building` has three subgroups, by what the acoustician is
  doing. `building.measurement` is the building that exists: the ISO 16283,
  10140, 10052, 15186 and 10848 methods, the heavy impact sources, the
  floor-covering improvement, the reception-plate power, and the ISO 717
  ratings and ISO 12999 uncertainty that go with them.
  `building.prediction` is the building that does not exist yet: EN 12354-1
  to -5 and the panel theory behind them. `building.regulation` is the code
  it has to satisfy, which is the Spanish CTE DB-HR today and the shape any
  other country takes tomorrow.

  Five modules lose a word their subgroup now says: `building_prediction` and
  `detailed_prediction` become `prediction.simplified_model` and
  `prediction.detailed_model`, which is what EN 12354 calls the two routes it
  offers for the same calculation, and what the module's own text has always
  called them;
  `facade_prediction` is `prediction.facade`, `building_uncertainty` is
  `measurement.uncertainty` and `spanish_building_code` is
  `regulation.spain`.

  Nothing moves in the namespace: `from phonometry import building` exports
  the same 251 names, and the nineteen pre-split module paths keep working
  until 5.0.

- Four `room` modules and one `aircraft` module lose the prefix their package
  already says: `room.room_acoustics`, `room.room_ir`, `room.room_noise` and
  `room.room_modes` are `room.acoustics`, `room.impulse_response`,
  `room.noise_criteria` and `room.modes`, and `aircraft.aircraft_noise` is
  `aircraft.certification`. Two of them say more than the prefix did: `ir` was
  opaque for the ISO 18233 acquisition, and `noise` alone did not say that the
  module is the ANSI/ASA S12.2 NC and RC criteria. The certification module is
  what ICAO Annex 16 calls it.

- The two impulsive-sound methods are one module,
  `environment.assessment.impulsive_sound`. NT ACOU 112 and ISO/PAS 1996-3
  share the prominence and adjustment formulae (both take them from Pedersen's
  method) and differ only in what the caller supplies: the Nordtest closed form
  takes the onset rate and level difference, the ISO/PAS chain reads them off a
  calibrated recording. The second module was already importing the first for
  its formulae, so the split was between a standard and its own dependency
  rather than between two subjects. Every name is unchanged.

- The IEC 61265 aircraft measurement-system check is
  `phonometry.aircraft.measurement_system`, next to the certification levels
  whose chain it qualifies, rather than in `phonometry.filters` with the
  instrument-conformance code. `verify_aircraft_noise_system` reads the same
  from the top level; from `phonometry.filters` it still resolves until 5.0,
  with the usual notice.

- The `docs/` mirror is laid out like the site it mirrors:
  `docs/signal/filters/filter-banks.md` for `/signal/filters/filter-banks/`,
  with `docs/README.md` still the index. The pairing that publishes a markdown
  copy of every page, and the one that builds `llms.txt`, are by route now
  rather than by file name, so the table of pages whose mirror name differed
  from their route is gone. What it also held stays: the conformance report and
  the errata registry are written at the repository root and transplanted into
  their site routes, which is a mapping the layout cannot express.

- The sidebar is one tree per topic, through `starlight-sidebar-topics`. A
  reader in the underwater chapter sees its four guides and its own API branch,
  not every page in the library folded three levels deep, and the topic list at
  the top of the sidebar says what the other twelve are. The tree a page ships
  drops from the 286 links every page carried to between 17 and 72, the size of
  the topic the reader is in.

  The API reference of a domain lives inside that domain's topic, which the
  package taxonomy is what makes possible: every section of the generated
  reference is the API of one subpackage, so each section mounts under the
  topic that documents it. The last topic is the global index, for a reader who
  wants the whole table rather than one domain. `generate_api_docs.py` emits one
  group per section instead of a single block, so adding a module still changes
  nothing by hand.

  Pagination stops at the edge of a topic. Starlight paginates over the whole
  tree, and the plugin's own trimming does not survive a configured base path,
  so a card that would have sent the reader into another chapter, swapping the
  sidebar under them, is dropped instead.

  The tree is one level shallower as a result, since a topic is what a
  top-level group used to be, and the styling follows: at the top level of a
  topic a page and a subgroup read the same, as they already did further down.
  The landing page of each language keeps a menu on a phone, and it is the
  topic list, which is the right menu for a page whose job is to send the
  reader into a chapter.

- The documentation site is organised by topic rather than by one `guides/`
  drawer. Every page lives under the domain it belongs to, and the URL says so:
  `signal/filters/filter-banks`, `buildings/insulation/insulation-field`,
  `perception/psychoacoustics/loudness`, `materials/absorbers/porous-absorbers`.
  The area overviews that were `guides/sections/<area>` are the index page of
  their topic or subgroup, so the address a reader shortens by hand is a real
  page. Six pages move to the topic that owns their subject rather than the one
  they were filed under: the environmental levels and the Spanish regulation
  leave the signal chapter, machine diagnostics joins vibration, dynamic
  stiffness joins the resilient materials, and the two structure-borne power
  pages join the insulation design they feed. `guides/impulse-prominence` is
  `environment/assessment/impulsive-sound`, following the module merge.

  The old addresses are gone rather than redirected: the site has been public
  for days, is unannounced, and the link validator that fails the build on a
  dead internal link is a better guarantee than a redirect table nobody reads.
  The `llms.txt` shards follow the same shape, so `llms-signals-filters.txt`
  replaces `llms-octave-filtering.txt` and its siblings.

  The signal chapter itself is `/signals/`, plural, matching the
  `phonometry.signals` package, whose own plural exists to keep it apart from
  `scipy.signal` in the code a reader writes. One name for the folder, the
  package, the topic and the shards, with the same clean break as the rest of
  the move.

- The eighteen domain packages are named explicitly in `phonometry/__init__.py`
  rather than bound as a side effect of importing the flat API. Nothing changes
  at runtime, where `phonometry.building` already resolved; what changes is
  that a type checker can follow it, which the removal of the package-level
  `__getattr__` shim would otherwise have taken away.

- `phonometry.underwater` has three families, along the three questions an
  underwater problem asks. `underwater.sources` is what makes the sound:
  ISO 17208 ship radiated noise, shipping traffic, impact pile driving and the
  wind and thermal noise the sea makes on its own. `underwater.propagation` is
  how it gets there: closed-form transmission loss, Weston's regimes, the
  numerical solvers (normal modes, rays, parabolic equation), the seabed it
  reflects off and the sound speed profile it travels through.
  `underwater.bioacoustics` is who hears it: marine mammal audiograms and the
  regulatory auditory weighting and exposure criteria built on them. The
  ISO 18405 quantities and the sonar equation stay at the root, because all
  three families are written in their terms.

  Four modules lose a word the family now says: `ocean_ambient_noise`,
  `marine_mammal_audiograms`, `marine_mammal_weighting` and
  `numerical_propagation` become `sources.ambient_noise`,
  `bioacoustics.audiograms`, `bioacoustics.weighting` and
  `propagation.numerical`. The module that was `underwater.propagation` is
  `propagation.closed_form`, and the name it vacated is the family package,
  which exports everything the module did: that path keeps working without a
  notice, since a package cannot be shadowed by an alias that warns.

  Nothing moves in the namespace: `from phonometry import underwater` exports
  the same 82 names, and the ten pre-split module paths keep working until 5.0.

- The IEC 61043 intensity-instrument class check is
  `phonometry.emission.intensity_compliance`, next to the intensity chain it
  verifies, rather than in `phonometry.metrology`. It was the one section of
  the API reference that had to reach into a second package to stay coherent,
  and now no section does. Reading the names from `metrology` still works
  until 5.0, with the usual notice.

- `phonometry.psychoacoustics` has two families, along the split every text on
  sound quality makes. `psychoacoustics.loudness` holds how loud a sound is:
  the four models (ISO 532-1 Zwicker, ISO 532-2 Moore-Glasberg, the
  time-varying Moore-Glasberg of ISO 532-3 and ECMA-418-2) and the ISO 226
  equal-loudness contours that put a level in phons.
  `psychoacoustics.quality` holds what the sound is like
  once its loudness is known: sharpness, roughness, fluctuation strength,
  tonality, tone audibility and the annoyance models built on them. The ERB
  scale stays at the root, because both families measure on it.

  Five modules lose the word the family now says: `loudness_zwicker`,
  `loudness_moore_glasberg`, `loudness_moore_glasberg_time`, `loudness_ecma`
  and `loudness_contours` become `loudness.zwicker`, `loudness.moore_glasberg`,
  `loudness.moore_glasberg_time`, `loudness.ecma` and `loudness.contours`;
  `psychoacoustic_annoyance` becomes `quality.annoyance`.

  Nothing moves in the namespace: `from phonometry import psychoacoustics`
  exports the same 60 names, and the thirteen pre-split module paths keep
  working until 5.0.

- `phonometry.materials` has four families, by what the material does rather
  than by how it is measured. `materials.absorbers` holds everything that
  absorbs and the methods that characterise it: ISO 354 in the reverberation
  room, ISO 10534-2 in the tube, ISO 9053 for airflow resistance, the porous
  and multilayer models, Biot, the slow-sound metamaterials, and the ISO 11654
  rating and ISO 12999-2 uncertainty. `materials.diffusers` holds what
  scatters: Schroeder design, metadiffusers, and ISO 17497.
  `materials.surfaces` is what is characterised in place, which is the ISO
  13472 road surface today. `materials.resilient` is what isolates, which is
  the EN 29052-1 dynamic stiffness.

  The test method lives with the family it characterises rather than in a
  family of its own, which is how Cox and D'Antonio organise the same material
  and how a reader looks for it: "I have a diffuser" comes before "I have a
  goniometer". Two families start with one module each, `surfaces` and
  `resilient`, because both have known growth and a subaudience that already
  exists.

  Five modules lose a word the family now says: `absorption_rating` and
  `absorption_uncertainty` become `absorbers.rating` and
  `absorbers.uncertainty`, `porous_absorber` becomes `absorbers.porous`,
  `slow_sound_absorber` becomes `absorbers.slow_sound` and `diffuser_design`
  becomes `diffusers.design`.

  Nothing moves in the namespace: `from phonometry import materials` exports
  the same 186 names, and the thirteen pre-split module paths keep working
  until 5.0. Three test files keep their old basenames: `test_rating.py`,
  `test_design.py` and `test_uncertainty.py` are already taken elsewhere in
  the suite, and pytest imports a test module outside a package by its
  basename.

- `phonometry.environmental` is `phonometry.environment`, with three
  subgroups that follow how an environmental study is written: what the source
  emits (`environment.sources`: CNOSSOS road and rail, wind turbines), what the
  path does to it (`environment.propagation`: ISO 9613-1 and -2, ground effect
  and barriers, refraction) and what the law makes of the result at the
  receiver (`environment.assessment`: ISO 1996-1 and -2, impulsive prominence,
  RD 1367/2007). Two modules lose a word the subgroup now says:
  `atmospheric_refraction` is `propagation.refraction` and
  `spanish_regulation` is `assessment.spain`, which is the shape the next
  national code will take.

  The name was the last of the three that did not match: the code said
  `environmental`, the reference said `environment` and the guides said
  Environment. Now they all say the same thing, and the generated reference
  loses a cross-package exception: the wind turbine used to be documented
  under Aircraft and wind energy because that is where its audience was
  looking, and with the sources subgroup it can be documented where it lives.
  That section is now Aircraft noise and draws from one package, which leaves
  two sections in the whole taxonomy that deliberately span two parents.

  Nothing moves in the namespace: `from phonometry import environment` exports
  the same 165 names, and `environmental` keeps working as a package alias
  until 5.0, along with all twelve of its module paths.

- `phonometry.vibration` has three families, by who reads them.
  `vibration.structural` is the structural acoustics that feeds building
  prediction: mobility (ISO 7626), plate junctions, radiation efficiency,
  experimental SEA and transfer stiffness (ISO 10846). `vibration.human` is
  exposure of people: ISO 2631-1 and ISO 5349 in `human.exposure`, ISO 2631-5
  in `human.multiple_shock`. `vibration.machinery` is condition monitoring, in
  `machinery.diagnostics`. Three modules lose a word the package now says:
  `human_vibration` is `human.exposure`, `multiple_shock_vibration` is
  `human.multiple_shock` and `machine_diagnostics` is `machinery.diagnostics`.

  Nothing moves in the namespace. `from phonometry import vibration` still
  exports all 112 names, because the package re-exports its families
  explicitly, so no notice fires for reading a name that stayed in the domain
  it always belonged to. What changed is the module path, and the nine
  pre-split paths still import and delegate with the usual notice until 5.0,
  including the dotted read that follows the import: registering the alias in
  `sys.modules` is enough for `import phonometry.vibration.human_vibration`
  and not for the `phonometry.vibration.human_vibration.daily_exposure` after
  it, so the package carries the same PEP 562 hook the other splits use, with
  nothing to redirect but its own modules.

  The renderers stay in one `_plot/vibration.py`. They mirror the domain, and
  the domain is still vibration: splitting the translation table three ways
  would have been the riskiest edit in the change for no reader's benefit.

- `phonometry.hearing` is two packages. It held two subjects that share a
  word and nothing else: speech intelligibility, which scores a transmission
  channel and is read by whoever designs a room or a public address system,
  and hearing conservation, which is about the ear and is read by an
  occupational hygienist. The generated reference and the sidebar had already
  separated them; the code had not, so the `speech` section of the reference
  drew from a package called something else. STI and STIPA with its signal
  generator, the Speech Intelligibility Index with its four band-importance
  procedures and the standard speech spectra, and STOI with ESTOI, are now
  `phonometry.speech`; the audiometric thresholds, the ISO 1999 hearing loss
  and the ISO 9612 exposure stay in `phonometry.hearing`. No module is
  renamed and no page moves: the reference had these three under `speech`
  already, and the split is what lets its consistency contract say so. Five
  sections still carry a key that is not the name of a single subpackage:
  `rooms` and `environment` wait for the steps that rename their packages, and
  `filters`, `aeroacoustics` and `power` are the three documented sections that
  deliberately span two parents. The renderers follow, so `_plot/hearing.py`
  becomes two modules with their own string tables, and the Spanish labels of
  the speech ones get the test they never had.

  Compatibility is the same as for the other 4.0 splits: `phonometry.hearing.sti`
  and its two neighbours still import and warn, `hearing.stipa` still resolves
  from the namespace it left, and both go in 5.0.

- `phonometry.metrology` is three packages. It had grown to 21 modules across
  four unrelated subjects, and every layer above it had already worked around
  that: the generated reference spread it over six sections, the sidebar over
  four groups, and the name predicted neither `cepstrum` nor `signals`. The
  filter banks, the frequency and time weightings, the parametric EQ and the
  IEC 61260-1 / IEC 61672-1 class verification are now `phonometry.filters`;
  the general signal analysis (levels, Welch and multitaper spectra, coherence,
  time-frequency, correlation, envelope, cepstrum, phase, synchronous
  averaging, test signals) is `phonometry.signals`; and `phonometry.metrology`
  keeps what gives it its name, the calibration, the GUM uncertainty, the data
  qualification and the IEC 61043 intensity-instrument class check. Four
  modules are renamed with the move, each dropping a prefix its package now
  carries: `filter_design` to `filters.design`,
  `parametric_filters` to `filters.weighting` (it is the A/C/G and time
  weightings, which is what people look for), `signals` to
  `signals.test_signals` and `random_data` to
  `metrology.data_qualification`, the name its documentation page already had.
  Nothing moves in the flat API: `from phonometry import leq, octave_filter`
  is what it always was, and that is how the documentation leads.

- Every 3.x module path still imports, and so does every name read through the
  namespace it left. `import phonometry.metrology.levels` and
  `from phonometry.metrology import spectra` resolve to the relocated module
  and warn on attribute access, the same PEP 562 shim the 3.2 modularization
  used; `metrology.leq(...)` after `from phonometry import metrology` warns and
  delegates too, which the module-path shims alone would not have covered, and
  the namespace form is the one the documentation leads with. The aliases are
  removed in 5.0, a release later than the 3.x ones, so the rename notice now
  names the release that removes it instead of assuming 4.0. Resolution goes
  through the public `__all__` of the packages the names moved to, so a name
  that stops being public stops resolving through the old namespace as well,
  and `dir(metrology)` still lists them, since a PEP 562 hook is invisible to
  it and they would otherwise vanish from tab completion a release before they
  stop working. The one form the hook cannot serve is
  `from phonometry.metrology import *`, which now brings the narrowed API
  rather than everything the package used to re-export; the explicit forms all
  keep working.

- The generated reference is keyed by subpackage. Its sections were a fourth
  naming of the same material (`levels`, `spectra`, `correlation` for what the
  code called `metrology`), so a reader who knew where a function lived could
  not predict where its page lived. They are now `filters`, `signals` and
  `metrology`, one per package, and the pages move with them
  (`reference/api/spectra/cepstrum` becomes `reference/api/signals/cepstrum`).
  The consistency contract in `scripts/api_taxonomy.py` is what enforces it:
  five sections drew from `metrology` under four different names and now three
  draw from three packages under their own. The sections that deliberately
  span two parents are untouched, and one of them is why the rule is not yet
  universal: `metrology.intensity_compliance` is still documented with the
  intensity chain it verifies, in the `power` section.

- CI fails when `.github/reports` no longer matches a fresh `make reports`
  run, which is the reason the fiches were able to drift for weeks in the
  first place: the conformance report, the generated API reference, the
  evidence pages and the documentation figures all had a staleness gate, and
  the fiches had none. The new `Example report fiches up to date` job
  regenerates the whole set and compares it through
  `scripts/check_reports.py`. A byte diff is not an option, for the same
  reason it was abandoned for the figures: every fiche embeds its plot as
  vector geometry, and GitHub's heterogeneous runners compute a few of those
  coordinates a last bit apart, which rewrites the page for no visible reason.
  So each fiche is compared on what it says and on what it looks like: the
  extracted text exactly, page by page, and the committed preview within the
  same two-criteria pixel tolerance the figures use, recalibrated for a
  document page. The thresholds are 64 pixels and an RMS of 0.5, and the
  reasoning behind them is written out in the script rather than summarised
  here, because it is not the clean separation it first looked like: moving
  every plotted coordinate by one unit in the last decimal, which is an upper
  bound rather than a forecast, pushes five of the 67 fiches past those
  thresholds, all of them plots whose bar edges sit on a pixel boundary.
  Moving a single coordinate in the worst of them changes nothing at all, and
  the coordinates in question come from exact layout arithmetic, so the bound
  is not what a real machine does. Raising the RMS far enough to clear it
  would stop catching the restyled fills, which is the failure the check
  exists to prevent. The two stale fiches moved 28710 and 27629 pixels and the
  subtler fills moved the RMS to 0.79 and 1.33, so those are caught. They are
  tighter than the figures' on purpose: a fiche preview is a flat document
  page rasterized by a pinned binary rasterizer, a much quieter thing to
  compare than a plot raster, and the figures' looser bound would have let
  the restyled fills through. The rendering stack the job runs is pinned in
  `requirements-reports.txt` alongside the existing figure pins, `make
  reports` now renders into a scratch directory and swaps it in only once the
  whole set is written, so a fiche that is no longer produced is actually
  removed without a generator that dies halfway leaving the working tree
  stripped of the committed examples, and the generator pins the numerical
  thread pools the way the figure generator does, so a runner with a
  different core count cannot reorder a floating-point sum into the rendered
  page. The staleness helpers the two checks share moved to
  `scripts/generated_assets.py`. A unit test also asserts every registered
  fiche has a committed render, which fails in the same second as the commit
  that forgets one, with no regeneration needed.

- The figures now spell the decimal logarithm the way the pages around them
  do. Every axis label, legend entry, title, annotation box and diagram
  caption the generators write still carried the ISO `lg`, so a guide that
  had just stated $\log_{10}$ in its prose showed `lg` in the picture below
  it, and the same quantity was named two ways on one screen. They now write
  `log10` in plain text and $\log_{10}$ in maths, in English and Spanish
  alike; 44 figures were regenerated for it, the ISO 10846 fiche with them,
  because its embedded plot carries the same legend, and the two sound power
  formulas of the free field versus reverberation room clip, which is
  re-encoded in its four variants. Spelling the
  logarithm out costs three characters, which pushed eight diagram lines out
  of the frame that used to hold them: the ISO 3744 formula card, the
  energetic sum of the installed-paths cascade, the survey relation of the
  vibration sound power sheet, the junction and noise-control captions, the
  reception-plate power balance, the detector chain's decibel stage and the
  impact-setup relations each take a smaller face so the line sits inside its
  box again. The renderer of the human-vibration fiche also documented the
  VDV unit as `m/s^1,75`, a decimal comma where the rest of the project
  writes a point, while what it prints has always followed the reader's
  language.

- The theory pages author their figures through `ThemeImage` like every other
  page. Thirty-four figures across the six theory pages, in both languages,
  were still two hand-written `<img>` tags on one line with the alt text typed
  out twice, the dark URL spelled out by hand and no intrinsic size, which is
  the markup that component was introduced to replace. They are now one tag
  each, so the light and dark URLs cannot drift apart, the alt cannot differ
  between the halves, and the browser reserves the right box before the bytes
  arrive. The pages become `.mdx` to import the component; the only prose
  touched is one `<` that markdown reads as a literal and MDX reads as the
  start of a tag. The markdown mirror under `docs/` is unchanged: GitHub
  renders it with no build step, so its figures stay hand-written pairs.

- The landing page counts its own numbers. The guide, API-page, figure and
  fiche totals and the release version were typed into the page by hand, and
  every one of them had fallen behind: the page still said version 3.2.0 after
  3.3.0 shipped, claimed 32 fiches against the 66 in `.github/reports`, 93
  guides against 104 and 120 API pages against 145, and counted 506 figures
  where the tree holds 344 drawings in their four theme and language variants.
  A new `site/src/data/repo-stats.mjs` counts all five from the tree at build
  time, the way `conformance-stats.mjs` already parsed the conformance
  headline, so the only place any of them can now be wrong is the tree itself.
  The About pages cite version 3.3.0 to match, and the landing links PyPI and
  the Zenodo DOI where it names them.
- The standards listed per area on the landing page, checked designation by
  designation against the modules that implement them. The Spanish column
  headed its count "normas implementadas" where the English said "standards
  referenced", which promoted a referenced standard to an implemented one in
  one language only. Part numbers now state what is actually implemented
  (`ISO 10534-1/-2`, `ISO 9053-1/-2`, `ISO 13472-1/-2`, `ISO 7626-1/-2`,
  `ISO 10846-1/-2/-3`, `ISO 2631-1/-2/-4/-5`, `ISO 8041-1`, `EN 29052-1`),
  ISO 18233 moves to the area whose measurements use it, and ISO 9611,
  ISO 10848, ISO 15186-1/-2, ISO 16251-1, ISO 12999-2, ISO/PAS 20065,
  ISO 389-7, DIN 45681 and ASTM E413/E1414 join the areas they are implemented
  in. The area blurbs on the guides hub carried the same lists, disagreed with
  the landing on CNOSSOS-EU, and are now identical to it area by area; the
  Spanish side spells ECAC the way the rest of the Spanish documentation
  does.
- The landing page's fiche example is now the call that renders the fiche
  printed beside it. It showed a hand-written 16-band spectrum for a 200 mm
  reinforced-concrete wall checked against a 42 dB requirement, three claims
  the artwork next to it contradicts: those bands rate Rw = 30 dB, so the
  snippet would have printed FAIL where the caption promised PASS, the curve
  is a glazing curve and not a concrete one, and the committed fiche is a
  predicted 6 mm float glass pane rated against 30 dB. The snippet is now the
  generator's own call with the metadata block cut to the fields the page
  discusses, and it reproduces the printed Rw (C; Ctr) = 31 (-1; -3) dB and
  its PASS exactly. The surrounding copy drops "the layout an accredited
  report uses" for the layout it is, and the Spanish long description no
  longer tells a screen-reader user the English fiche reads APTO.
- Wording on the landing page, English and Spanish. The requirement is
  "Python 3.13 or newer" as `pyproject.toml` says, not "Python 3.13"; the
  Spanish "dataclass inmutable" is the `frozen` dataclass the English
  describes, which is not the same claim; the Spanish links to Primeros pasos
  by the name that page actually carries; the conformance disclaimer drops a
  gerund that does not modify a noun in Spanish; the Spanish description says
  "biblioteca" like the rest of the page rather than "toolkit", and reaches CI
  prepositionally so `CI` cannot be read as the ISO 717-2 spectrum adaptation
  term of the same name.

- The symbols the documentation names as quantities and the relations it
  states in prose are now typeset as maths rather than left as code spans
  or plain text, across the guides, the reference and the errata registry,
  in English and Spanish alike, and in the GitHub mirror under `docs/`.
  Quotations of a printed source keep their original characters, and names
  that are code identifiers stay code. Every formula now writes the decimal
  logarithm as $\log_{10}$, the spelling that needs no key to read, in place
  of the ISO $\lg$ the pages used to mix with it.

- The equations in the library's own docstrings are written in LaTeX now, and
  the API reference renders them. They had been plain-text snippets, so the
  reference published `LWA = 10*lg( sum_k 10^(0,1*(LWk + Ck)) )` as a code
  block where a reader wanted a typeset formula, and the generator understood
  no maths at all, so the handful of docstrings that already used the `:math:`
  role had that markup published verbatim and nothing else could state a
  formula except as a code block. The 287 display blocks and 3703 inline
  expressions across 134 modules now carry the same operands, subscripts and
  signs as the standard each one cites, with the clause number kept as a
  `\tag` so a reader can still find the source, and decimal points in place of
  the commas, the fractions drawn with dashes and the stray Unicode maths
  glyphs the snippets used to mix. The decimal logarithm is written
  $\log_{10}$ here too, so the docstrings, the guides and the reference all
  spell it the same way. The generator translates the `:math:` role and the
  `.. math::` directive into the `$...$` and `$$...$$` the site's KaTeX pass
  already expected, one display block per blank-line-separated equation as in
  Sphinx, and keeps a directive that sits inside a list at the item's own
  indentation, without which the block closed the list and left the equation
  and the prose after it stranded outside the step that introduced them.

- Every editable page now links to its source in the footer, in both
  languages. The generated API reference opts out through the route
  middleware, since those pages are written from docstrings and a
  regeneration would discard any edit made through the web. The head also
  drops a hand-written favicon link that duplicated the one Starlight emits
  from its own default, and the web app manifest describes the toolkit
  instead of the octave-band filter bank it grew out of.

- The test suite runs in about half the wall time. The report generator
  tests render each example fiche exactly once and assert its PDF and WebP
  preview on the same run instead of regenerating the full set twice, the
  conformance checks compute once per test process through a memoized
  registry, the expensive FDTD, elastic-wave and ECMA reference fixtures are
  computed once on a single worker because their tests now share load groups
  instead of scattering and rebuilding the fixture on each one, and the
  heaviest modules dispatch first. No test was removed, no signal shortened
  and no tolerance or oracle touched; coverage is line-for-line identical.

- The CI job that exercises the numba-jitted kernel fans the suite out with
  `pytest -n auto` like every other test job. It was the one job still running
  serially, so the load groups and the heavy-first dispatch order did nothing
  in it and it kept taking about 20 minutes while the parallel jobs halved.
  Compilation was never the reason to keep it serial: the library jits exactly
  one kernel, and the two signatures it is called with cost about 5 s to
  compile from scratch and about 0.6 s to load from numba's on-disk cache
  against a suite of 7800 tests, and the modules that reach the kernel take
  the same time with the JIT on as with it off. A new
  `scripts/check_jit_kernel.py` step compiles both signatures before the
  workers start, so they read that cache instead of each building its own, and
  fails the job when the kernel produced no machine code at all (numba missing
  from the environment, or `NUMBA_DISABLE_JIT` left at the default the rest of
  the suite runs under), which would otherwise leave this job green while
  exercising the interpreted path it does not exist for. The job declares that
  variable once for all of its steps, so the guard runs under exactly the
  environment it vouches for and cannot stay green on a copy of its own while
  the suite loses it. On a 12-core host the suite goes from 2312 s serial to
  789 s on four workers and 574 s on all twelve, with the same 7800 tests
  passing.

- The eleven documentation clips still encoded as VP9 are now AV1, which is
  what the other fifteen had already moved to, so all 104 WebM files finally
  share one codec. Their 44 files come down from 20.1 MB to 15.5 MB, 23 %
  smaller at the same resolution and frame rate. The GitHub GIFs are
  transcoded from the WebM rather than from the frames, so their palettes are
  rebuilt off the new source and they grow by about 10 %; frame-by-frame they
  are indistinguishable from the previous ones.
- Three ways of shading a region against the page colour had grown up side by
  side in `_plot`. `theme_fill` mixes the page towards the hue until it sits a
  fixed CIEDE2000 distance away, which is the one the rest of the renderers
  use; the IEC 61043 pass region instead branched on a hand-computed luminance
  and picked one of two hard-coded greens, and the IEC 61260-1 pass corridor
  had never been migrated at all and still asked for `alpha=0.18`. Both now
  call `theme_fill`, so a single rule decides every themed fill and the two
  pass regions finally agree on a colour. Only the four `intensity_class`
  figures move; the filter-class corridor appears in no committed figure, and
  the fiche it is embedded in renders the same as before. The corridor being
  opaque also removes the alpha that the fiche pipeline is not guaranteed to
  carry, which is the reason the pass region had been given opaque tones in the
  first place.
- Twenty-nine comments in the Spanish guides wrote a value the code prints with
  a decimal comma, so the page promised `132,4 Hz` where running the snippet
  gives `132.4`. Python has no locale in `print`, and the comment next to a
  `print` call is a claim about its output: the reader who types the example in
  and compares is the person the annotation is for. The rest of the Spanish
  guides had already settled this, using a point in 543 such comments against
  49 stragglers. Prose inside the same comment is untouched and keeps the
  Spanish comma, as do the symbols that only look like decimals, `L1,2m` and
  the `(1,1)` mode index.
- Every formula on the site was rendering under a stylesheet written for a
  different version of KaTeX. Two copies decide how a formula looks: the one
  `rehype-katex` renders with, and the one the site imports
  `katex/dist/katex.min.css` from, and nothing tied the two together. The site
  had reached `katex` 0.18.1 while `rehype-katex` still resolves 0.16.47, and
  0.18 renamed the classes on the base box and the strut. So the markup asked
  for `.katex .base` and `.katex .strut` and the stylesheet defined
  `.katex-base` and `.katex-strut`: the base box lost `white-space: nowrap`,
  `display: inline-block` and `position: relative`, and the struts that set the
  line height stopped laying out at all. Measured on the resilient-layer guide,
  a display block came out 142.9 px tall instead of 130.5, and inline formulas
  broke across lines in the middle of an expression, splitting `0.886` from its
  `m/s`. The dependency now follows the line `rehype-katex` resolves, and
  `check-math-render.mjs` gained a third check that compares the classes the
  built markup emits against the classes the built stylesheet defines, so the
  two cannot drift apart again unnoticed. Neither half was invalid on its own,
  which is why nothing complained for as long as it did.
- Nothing in the pull-request pipeline built the documentation site, so a page
  could break the deploy while its pull request showed twenty-six green checks.
  The site build lived in a workflow triggered only by a push to `main`, which
  meant the Astro build, the HTML validation, the formula render check and the
  accessibility audit all ran for the first time after the merge. A repeated
  word in an import statement kept the site from deploying for fifteen hours
  across five merges, and the three over-long page titles that surfaced once
  the build came back had ridden in the same way. The build job now also runs
  on pull requests against `main`, on the same paths that trigger the deploy;
  publishing and the IndexNow submission stay reserved for `main`, and a
  pull-request run no longer shares the deploy's concurrency queue, so it
  cannot cancel a deployment that is already publishing.
- Three documentation pages carried a `<title>` longer than the 70 characters
  `html-validate` allows once the site name is appended, which is what left the
  deploy red. The resilient-layer guide now takes the shorter name it was
  already linked by everywhere else, and the heavy-impact guide names ISO
  16283-2 alone in its title, with JIS A 1418-2 kept in the description and the
  body. The English heavy-impact title had been sitting at exactly the limit.
- The Spanish building-code guide had lost six of its sections, in both
  languages, since the day a display formula was written with its `$$` on the
  same line as the first term. Nothing complained: KaTeX does not fail a build
  on a parse error, it emits an error span and ships the page, and here the
  parser ran past the end of the formula and swallowed the requirements of
  clause 2, windows and composite facades, the plots, what the guide covers,
  the See also and the quick answers. HTML validation and the accessibility
  audit passed on that page throughout, because both were answering a different
  question. The delimiters now sit on their own lines, as the other 416 display
  blocks in the documentation already did, and `site/scripts/check-math-render.mjs`
  fails the build on any `katex-error` span left in the output or any `$$` that
  shares a line with content.
- The maintainer's name is written one way and cited another. The hyphenated
  `Requena-Plens` is a bibliographic citation form, so it stays in the APA and
  BibTeX blocks, `CITATION.cff` and `.zenodo.json`; everywhere the name is a
  name, it is now the full `José Manuel Requena Plens`, which is also what the
  ORCID record and the canonical Person document hold. That covers the `LICENSE`
  copyright line, the `authors` field of both `pyproject.toml` files and a
  documentation example. The copyright header of every file under `src/`,
  `scripts/` and `tests/` carries the same name without the accent, since a
  source file is the one place where a mangled byte does real damage.

- The conformance counts quoted outside the report are generated, not typed.
  `docs/CONFORMANCE.md` has always been the authority for "N/N conformance
  checks pass across D domains and S standards", and the Astro page bodies
  import those integers from `site/src/data/conformance-stats.mjs`, but six
  further places wrote them out and were only validated: `.zenodo.json`,
  `docs/getting-started.md`, the two landing-page meta descriptions and the
  JSON-LD `description` of both conformance pages. Every change that moved a
  count left all six stale until someone edited them, and one of them merged
  silently wrong, because a file quoting two counts whose sides agree on one and
  differ on the other merges without a conflict. `scripts/check_conformance_claims.py`
  now has a `--write` (`--fix`) mode that substitutes the authoritative value in
  place, and `make conformance` runs it straight after regenerating the report,
  so the numbers come out of the pipeline. Checking and writing share one list
  of patterns and cannot diverge, the substitution uses the span of the captured
  digits rather than a textual replace (so an unrelated figure that happens to
  equal the old count is left alone), and the default read-only behaviour is
  unchanged and remains the CI gate, now with the staleness diff widened from
  the report alone to every file the pipeline can rewrite. The four site-side
  claims live in YAML frontmatter, which is parsed before MDX compiles and
  therefore cannot interpolate an import whatever the file extension, so they
  stay with the writer rather than with `conformance-stats.mjs`.

- Documentation site: previous and next links name the section they lead to.
  Thirty-one sidebar rows are called "Overview", one per group, so 62 of the 564
  previous/next cards in each language said only "Overview" and the page between
  two groups said it twice. Each card now carries the group trail above the
  label, looked up in the sidebar the page already renders.

- Documentation site presentation: the heading scale is fluid and one step
  quieter at the top (an H2 goes from 29-35 px to 23-29 px, so a heading with a
  function name in it fits a phone in two lines instead of three), inline code
  inside a heading drops the chip and sizes to the heading around it, "See also"
  sections read as a panel while staying anchored sections in the table of
  contents, the per-page bibliography is set as a reference list rather than as
  more prose, the copy button on a code block is smaller, and the footer credit
  wraps as two units with the code hosts as marks instead of a run of words
  separated by dots.

- The last three mermaid flowcharts are now generated SVGs, and the mermaid
  runtime is gone from the site. The sound level meter guide, the calibration
  guide and the signal analysis theory page each rendered one diagram with
  mermaid, which meant every reader of the site downloaded and ran the mermaid
  bundle for three figures while every other diagram in the documentation is a
  deterministic SVG from `scripts/generate_diagrams.py`. Three new diagrams take
  their place, each in the usual four variants (light and dark, English and
  Spanish): `diagram_slm_pipeline` draws the guide's own chain, from the two
  recordings through the sensitivity factor to the display, integrated and band
  branches and the class verifiers that close it; `diagram_calibration_dataflow`
  draws one factor reaching every level function, and what happens with no
  calibrator, where the factor stays at 1 and the samples are read as pascals
  unless `dbfs=True` asks for levels referred to digital full scale;
  `diagram_bank_dataflow` draws both numerical-stability strategies of the
  filter bank, the biquad cascade and the decimation decision,
  ending in the band level and the optional band signal. The `astro-mermaid`
  integration and the `mermaid` dependency are removed from the site and its
  lockfile.

- "Why phonometry" argues the same thing with current evidence. Every test path
  in its conformance table pointed at the flat `tests/test_*.py` layout the
  suite no longer has, and the markdown mirror had lost the check, domain and
  standards count paragraph the site pages carry; both are fixed, and that
  paragraph's domain sweep now names the areas added since it was written. The
  MoSQITo comparison no longer claims a split that the sound-quality modules
  ended, a `pyroomacoustics` entry joins the ecosystem list now that room and
  wave simulation exist here, the numerics paragraph mentions the class 0 check
  on the default bank, and a new section covers what happens when the standard
  itself is what is wrong, with the errata registry as its evidence.

  Four claims on that page are corrected while it is being rewritten. The
  IEC 61672-1 Table 4 target for the 1 ms toneburst is −21,0 dB, not −20.9;
  Equation (7) gives 10 lg(1 − e^(−0,001/0,125)) = −20,9865 and the library's own
  transcription in `tests/metrology/test_iec_compliance.py` reads −21.0. Printed
  as −20.9 the row reversed its own comparison, showing phonometry at −0.09 and
  python-acoustics at exactly 0.00, when the errors are +0.01 and +0.10. The
  three ecosystem entries are corrected too: acoustic-toolbox declares
  `pyoctaveband>=1.1.3` rather than phonometry, and still hard-codes its own
  one-third-octave weighting tables; pyroomacoustics does ship measurement
  routines, so the defensible contrast is the ISO 3382 parameter set it lacks;
  and MoSQITo already has ECMA-418-2 loudness and roughness and the SII, so what
  is actually added here is ISO 532-2, ISO 532-3, ECMA-418-2 tonality,
  fluctuation strength and the Fastl & Zwicker annoyance model. The ISO 3382
  chain is stated for the image-source solver, which produces an impulse
  response, and not for the FDTD one, which produces probe pressure histories.
  The errata paragraph promised a regression test for every entry, which 38 of
  the 62 do not have (several record that no change was required), so it now
  says what the entries actually carry, as does the introduction of the errata
  registry itself. The conformance table credited ISO 226:2023 Annex B for the
  hearing threshold, which the test pins against the Table 1 `T_f` parameters.

- Every install page recommends `phonometry[full]`. The plain install stays the
  default, with a note that the extra brings matplotlib, numba, reportlab and
  svglib in one go, that the base install computes every metric on NumPy and
  SciPy alone, and that what it leaves unavailable is the `.plot()` figures, the
  `.report()` PDF fiches and the compiled `impulse` kernel. Every install page,
  in both languages, now also states the one consequence of that
  recommendation: numba declares `numpy<2.5`, so `phonometry[full]` resolves
  NumPy below 2.5 where a plain install gets the newest release, and
  `phonometry[plot,report]` is the route that keeps NumPy current, since numba
  only affects speed. The llms export and the transition-package README said the
  `impulse` time weighting is unavailable without numba; it is not, it falls
  back to a pure-Python kernel that is only slower, which is what `README.md`
  already said and what they say now.

- The `[report]` extra no longer claims to be enough for a fiche that embeds a
  figure. 46 of the 49 fiche renderers call `_layout.render_figure_drawing()`
  and two more (`iec60268_4`, `iec60268_5`) their own equivalent, all of which
  import matplotlib and svglib before drawing anything, so only the ISO 4871
  declaration fiche renders on reportlab alone. 35 `.report()` docstrings still
  documented `:raises ImportError: If reportlab is not installed` on renderers
  that need matplotlib too, and CI rebuilt 24 pages of the API reference from
  them on every run; the guides repeated it in prose on 14 topics, in both
  languages, and the llms exports mirrored that. All of them now say that
  reportlab is needed and that matplotlib is needed for the figure the fiche
  embeds, and point at `pip install "phonometry[report,plot]"`.

- One rule now governs how my name is written across the repository: the
  hyphenated "Requena-Plens" is the bibliographic citation form, and everywhere
  else the name is written out in full as two given names and two surnames,
  "José Manuel Requena Plens". `CITATION.cff` and `.zenodo.json` already read
  "Requena-Plens, José M." and are unchanged. The About page did not follow the
  rule: it wrote the inverted citation form with the surnames unjoined, in both
  the APA line and the BibTeX entry and in both languages, so the site
  disagreed with the metadata it points readers at. Those four citation strings
  now match, with the BibTeX field matching modulo the diacritic that block
  already drops. `paper/paper.md` is in the same class, since its `surname:`
  field is what renders the paper's own citation, so it carries the hyphen too.
  Running text, the `LICENSE` copyright line and the `author` meta tag keep the
  full unhyphenated name, which is also what the ORCID record
  `0000-0003-1250-6212` and the canonical Person document fetched from jmrp.io
  hold, so the citation form and the name record are meant to differ and
  neither needs changing.

- `docs/ERRATA.md` is now anchored on the printed page. Every claim that turns
  on the exact characters of a formula, constant, symbol, inequality or table
  cell has been read from an image of the cited page rather than from its text
  layer, and each Evidence bullet cites that page by edition, PDF page index
  and printed folio. The reason is that PDF text extraction deletes glyphs
  without saying so. Of the twenty-five source documents the glyph census
  covers, twenty-two
  emit no `√` at all over their whole text layer, so `f_T/√2` extracts as
  `f_T/2`; eleven emit no `−` either, and one textbook returns every `+` as
  `þ`. Eight further defects were found in the process and registered:
  EN 12354-3:2000 Formula (5), whose reduced form prints `0,32 V` where the
  standard's own `A0 = 10 m²` and `T0 = 0,5 s` give `0,032 V`, a straight 10 dB
  (the library never used the reduced form, and the identical Formula (3) of
  Part 2 prints it correctly); ECMA-418-1 Formula (21), which repeats the
  constant term `C_L,0` where the linear coefficient `C_L,1` belongs, making
  the fit negative everywhere over its middle range; two unresolved
  word-processor field references typeset in bold in clause 11.3 of the same
  standard; ISO 12999-2 Table 5, headed "one-third octave" over the ISO 11654
  octave series its own Table 2 heads "octave"; DIN 45681 Tabellen I.2 and
  I.10, each carrying one column header with the wrong spectrum index; the
  sound speed printed as `q` in Medwin & Clay Eq. (3.4.30) where the same block
  divides by `c`; the displayed male example of ISO 2631-5 Annex C, which drops
  the `− S_stat` term of Formula (C.3) and as displayed gives R = 1,15 against
  its own printed 1,22; and three further cells of ISO 717-2 Table C.2 that the
  800 Hz misprint propagates into, plus the last digit of the Table C.1 energy
  sum. Thirteen entries whose finding stands but whose stated evidence did not
  have their arguments replaced, among them the ISO/PAS 20065 edge-steepness
  entry, which had the two edges the wrong way round because the DIN radicals
  it was compared against had been eaten by extraction, and the Long Eq. 13.28
  entry, which claimed a 100 dB discriminator for a level that is identically
  independent of the velocity in question. Two ANSI S3.5-1997 entries and one
  EN 12354-3 attribution now say plainly what they rest on.

- The filter-bank reuse test asserts the property it is named after instead of a
  stopwatch. It used to compare `time.time()` deltas and require the class-based
  path to stay within 1.5x of the functional one, which on a shared runner under
  a parallel suite measures the scheduler: it failed on a macOS leg at
  `0.351 < 0.233 * 1.5` while passing everywhere else. The design routine is now
  counted directly, so the assertions read as the design cache is supposed to
  behave: ten functional calls with the cache cleared design ten times, a bank
  plus ten `filter()` calls design exactly once, and five identical
  `octave_filter` calls share one design while a different sample rate makes a
  second. Both paths are also compared band for band, so the reuse cannot be a
  shortcut past the work. Timings are still printed on every run, with a
  ten-fold guard that no scheduling noise can trip.

- `make graphs` regenerates the figures and stops there. The legibility check it
  used to end with is now `make figure-contrast`, and `make figures` runs
  generation, legibility and staleness in that order, one step each, the way CI
  does. Chaining the checker inside generation meant an illegible fill aborted
  the generation step and skipped the staleness compare after it, so a single
  bad fill on `main` reported itself as "the figures could not be regenerated"
  and blocked every open branch at once.

- Oracle test data: the two suites that only ran on a machine holding a private
  download now run everywhere. The certified IEC 60268-16 STIPA verification
  bench from stipa.info is 133 MB of PCM that does not compress as audio, so
  `tests/data/stipa/` carries a lossless 27-of-49-signal extract instead (5.8 MB:
  each signal stored as an LZMA-compressed difference series with the SHA-256 of
  its original samples, checked on every load). It keeps the C.3.3 and C.4.2
  suites complete and the worst-case point of the other three, so
  `tests/hearing/test_stipa_certified.py` verifies the standard's filter-slope,
  phase-distortion, modulation-depth and weighting-factor criteria in CI without
  a single tolerance being relaxed. The EBU loudness test set cannot be
  redistributed at all (its terms of use forbid copying and distribution, and
  the authentic-programme cases are licensed film excerpts), so
  `tests/broadcast/test_ebu_material_oracle.py` now takes the committed
  block-loudness series as its primary path and asserts the published -23.0 LUFS
  and 5/15 LU targets everywhere; the full chain from the audio still runs where
  a local copy exists, and there it also re-measures the committed envelopes so
  the derived oracle cannot drift from the material it stands for.

- Non-redistributable reference material moves out of `plan/` into the
  gitignored `tests/data-local/`, and every suite that uses it resolves the same
  way through the new `tests/oracle_data.py`: environment override, then
  `tests/data-local/`, then the committed data under `tests/data/` — the path CI
  takes, whose assertions never skip. `pytest` now prints which copy each dataset resolved
  to in its run header, so a green run says which oracle produced it. The
  convention, and what each committed oracle can and cannot assert, is written
  down in `tests/data/README.md`.

- Frequency-weighting guides: the single chart that drew A, B, C, D, AU and Z
  together is split so each guide shows only the curves it documents.
  `weighting_responses` now holds the three curves of IEC 61672-1, and with
  B/D/AU gone its floor drops from -50 to -72 dB so the A curve no longer
  runs off the bottom-left corner at 10 Hz. The new
  `special_weighting_responses` figure gives the special-weightings guide the
  chart it was missing: B, D and AU against a wide A reference, measured at
  96 kHz so the axis reaches the 40 kHz where IEC 61012 specifies the U
  low-pass of AU, with the +11.5 dB D hump at 3.15 kHz and the 13 dB AU
  cutoff at 16 kHz annotated. Both guides were revised around the split in
  English and Spanish: the A/C/Z page no longer hedges about curves it does
  not show, and the special-weightings page describes the three it now draws.

- Documentation animations: a field that sits tens of dB under the one
  setting the colour scale is now drawn with an explicit display gain
  instead of collapsing into the page background. The gain is measured off
  the field itself (the largest round factor that still keeps the quiet
  region inside the ramp, and none at all above a fifth of it) and every
  panel that uses one prints the factor with its dB equivalent, so the
  compression is stated wherever it applies and the measured levels stay
  the physical ones. Two clips were regenerated in all four variants:
  `anim_elastic_coincidence`, where the air below the plate transmits about
  47 dB under the incident wave and the fixed x20 gain it used still left
  it black, now drawn x150 (+44 dB), and `anim_fdtd_aperture_slit`, where
  the half space past the 25 mm slit is about 23 dB under the standing wave
  facing the wall, now drawn x10 (+20 dB) in the instantaneous panels only,
  the RMS row keeping its single shared scale. The English and Spanish
  captions say that the scale is compressed.

- Documentation site: a display formula that states several equations on one
  line now reflows on narrow screens instead of scrolling sideways. A build-time
  pass splits such a block at the author's own widest top-level separator
  (`\quad` or `\qquad`), makes each equation a part of a wrapping row and
  re-expresses the separator as the row's column gap, so a wide screen renders
  the identical single centred row while a phone stacks one equation per line,
  each centred. It reaches the 232 blocks on 120 pages (60 English, 60 Spanish)
  that state more than one equation per block, needs no change to the markdown,
  and leaves the TeX, the accessible label and the page anchors untouched.

- The Speech Intelligibility Index (`hearing/sii.py`) is now anchored on the
  reference implementation of ASA Working Group S3-79, the committee that
  maintains ANSI S3.5: the library reproduces `SII.C` to double precision on
  the official one-third-octave test cases `TO.TST` (published SII 0.445,
  exact 0.4453910059) and `TO_1.TST` (0.438 / 0.4382176540, alternative
  band-importance function), both added as tests and conformance checks.
  The previous surrogate anchors (Hornsby worksheet, R CRAN `SII`) all agree
  with the committee code and are kept as secondary cross-checks, with every
  reference constant's provenance restated; the quiet-standard and
  noise-plus-hearing-loss anchors now carry the committee-code digits, and
  the Annex C.2 worked example pins the officially corrected worksheet
  (WG S3-79 errata: Table C.1 row i=5 Li 0.10 -> 1.00, Table C.2 first-row
  slope -45.59 -> -46.59, both recorded in `docs/ERRATA.md`). No computed
  value changed; the library already produced the errata-consistent chain.

- Eight defects of the ISO 12354-1:2017 Annex L and ISO 12354-2:2017 Annex G
  worked examples, found while anchoring the detailed model on them, are
  recorded in `docs/ERRATA.md`: the perimeter sums and the external-wall
  internal loss factor of the Table L.3 / G.3 input block (only two of the
  five sums reproduce their own columns), the Table L.4 block labelled "2d"
  that carries the numbers of path 4d, the non-integer weighted ratings of
  Table L.1 / G.1 (a continuously shifted ISO 717 reference curve, printed
  once truncated and once rounded), the 50 Hz to 80 Hz flanking cells of
  Table G.1 that contradict Table G.4 for the same path, the Kij and m'i
  misprints of Table G.8, the mislabelled Table G.6 row and the element label
  of Table L.10 / G.10.

- The home page meta description now fits the ~160-character window search
  engines display (Bing flagged the old 318-character one): the one-sentence
  definition plus the conformance-check claim, in both languages. The full
  scope enumeration stays in the visible page text and in the
  `SoftwareApplication` description.

- `spectrogram_example` is now computed in 1/12-octave bands stepped at 1/8
  of the Fast window instead of 1/3-octave bands at half a window: four times
  the resolution on each axis, so the sweep reads as a line rather than a
  staircase of cells. The Fast (125 ms) integration is unchanged, so the
  figure still shows what it always showed.

- `block_processing_continuity`: the callout marking the filter restart lands
  on top of the dense trace, so it now carries a solid panel and larger text
  instead of relying on the gaps between the waveforms.

- Three new entries in `docs/ERRATA.md`. Ainslie's Equation (9.57) for the
  mode-stripping to single-mode transition range is larger than its own stated
  derivation rule gives by a factor pi times He/H, and the library implements
  the consistent form k^2*He^2*H/(9*pi*eta). NMFS (2024) Table 5 prints
  C = 1.37 dB for the otariid in-water group where 1.36 is correct (NMFS says
  so itself in the table footnote; recomputing C as the peak of W gives
  1.3643), and the library implements 1.36 while keeping the printed digit
  reachable as `c_db_as_printed`. Southall et al. (2019) Table 7 carries four
  peak-SPL misprints corrected by the journal's own errata, and the corrected
  values are the ones implemented.

- Bibliography data in the documentation frontmatter. `Vorländer, M. (eds.)`
  becomes `(Eds.)` on the four pages that carry the Havelock handbook, the
  volume, issue and pages of McFadden (1987) and Harris (1978) move out of the
  journal name into the structured fields the schema already has so they stop
  being italicised as part of it, Hadden and Pierce (1981) uses the same
  journal name as the other sixty-two entries that cite it, ANSI/ASA S12.2-2019
  carries the year its own designation states, and the Spanish page for
  programme loudness names the AES convention as it is published instead of
  translating it.

### Deprecated

- The pre-4.0 `phonometry.metrology.*` module paths and the names read from
  the `metrology` namespace that now live in `filters` or `signals` (see
  Changed). Both warn on use and are removed in 5.0, one release after the
  3.x aliases.

### Removed

- The 3.1 renames, at the release their notices named. The function aliases
  `octavefilter`, `getansifrequencies`, `normalizedfreq` and
  `calculate_sensitivity` are gone; use `octave_filter`, `nominal_frequencies`,
  `normalized_frequencies` and `sensitivity`. The ISO 12999-1 pair that
  shadowed the GUM functions at the package root, the bare `coverage_factor`
  and `expanded_uncertainty`, are gone; use `insulation_coverage_factor` and
  `insulation_expanded_uncertainty`, and read the GUM pair from
  `phonometry.metrology.uncertainty`. The renamed constants
  `OCTAVE_BANDS_HZ`, `THIRD_OCTAVE_BANDS_HZ`, `BASE_PLATE_BANDS_HZ`,
  `BAND_CENTRES` and `ExposureWarning` are gone; use `OCTAVE_BANDS`,
  `THIRD_OCTAVE_BANDS`, `BASE_PLATE_BANDS`, `BAND_CENTERS` and
  `OccupationalExposureWarning`. The deprecated keywords are gone too:
  `sample_rate` is `fs` in the ISO 13472 functions, `humidity` is
  `relative_humidity` in the ISO 9613 ones, and `room_volume` is `volume` in
  the ISO 3744 pair.

  `phonometry._plotting`, the 3.2 re-export of the renderers that moved to
  `phonometry._plot`, is gone with them.

- The flat pre-3.2 module paths, as 3.2 announced when it deprecated them:
  `phonometry.levels`, `phonometry.insulation`, `phonometry.room_ir` and the
  eighty-odd siblings that the modularization grouped into domain subpackages
  raise `ModuleNotFoundError` now, as does `phonometry.loudness`, deprecated
  since 3.1. Only the module paths went. The names they held are still
  exported flat, so `from phonometry import leq` reads as it always has, and
  so does every other name in `phonometry.__all__`. The 4.0 taxonomy aliases
  are untouched and stay until 5.0.

  A pickle written by a 3.x release carries a flat path in the `__module__` of
  its classes, and those no longer resolve; one written by 3.2 or later
  through a subpackage path still does, until 5.0.

  With one generation left, `_compat.py` keeps one table, and the alias
  machinery no longer carries the per-entry wording overrides that the 3.1
  deprecations needed.

### Fixed

- Twenty-nine documentation figures drew a line the dark theme could not
  show. The contrast gate measures filled regions, and a stroke is not a
  fill, so nothing had ever looked at one: the ISO 3745 microphone array
  drew its hemisphere in a half-opacity mid grey on a 0.4 pt line, which
  composites to 1.3:1 against the dark page. It was in the SVG, it passed
  every gate, and a reader could not see it. `scripts/check_stroke_contrast.py`
  measures strokes now and runs in CI, and the corpus it found divided into
  four causes. Most of it was opacity used as a dimming knob: a companion
  curve at `alpha=0.55` of its own hue is a soft tint over the white page and
  lands within a couple of levels of the near-black one, so the same call
  reads on one theme and disappears on the other. That is what `theme_line`
  in `phonometry._plot.common` replaces it with, the line counterpart of
  `theme_fill`: the emphasis the author asked for, expressed as a colour
  rather than an opacity, and raised on whichever page it would otherwise
  fall below the WCAG 2.2 non-text minimum of 3:1. It is opaque, so it also
  stops the line dulling the gridlines it crosses. The three renderers the
  library exposes are fixed with it, so a caller of `.plot()` gets the
  correction too: the `-3 dB` line of the filter bank response, the `+4 dB`
  shelf reference of the K-weighting response, and the `L_pAF` history of
  the ISO/PAS 1996-3 impulsive-sound plot, which was drawn in a fixed
  `"0.3"` grey that can only be neutral on one of the two pages. The second
  cause was a fixed dark literal chosen against the white page alone: the
  CSS `purple` of the time-weighting ballistics, dark enough to read on
  white and not on black, now the palette's own purple. The third was a
  colormap read from end to end: a family of curves sampled from viridis at
  0.05 starts in a near-black violet, so the first two of six were invisible
  on the dark page, and `series_colors` in the figures' theme now reads the
  stretch of the ramp the current page leaves room for. The fourth was the
  3-D drawing itself, whose panes keep their light default under the dark
  style and composite into a mid-grey slab that the wireframe was then drawn
  on, competing with the slab instead of with the page; the panes are
  cleared, the grid is derived from the page so the scaffolding cannot
  outshine the surface it locates, and each microphone number is a chip in
  its own point's colour, above the markers, because a 3-D artist is ordered
  by depth and a point in front of its own label was clipping the digits.
  Three figures are deliberately left as they were: the ISO 1999 compression
  contours and the two Doc 29 / Doc 32 noise contours draw their level lines
  on an opaque field of their own, YlOrRd in one case and viridis in the
  other two, so the page is not what is behind those lines and the gate
  measures them against the field instead.

- The committed example `.report()` fiches were not checked against the code
  that writes them, and two of them had fallen behind it. `iso1999_nipts` and
  `iso1996_tone_audibility` still carried the shaded regions as they were
  drawn before fills started being derived from the page they sit on: the
  fractile band was the old 50 % composite `#ffddbb` instead of the
  `#ffe7d2` wash, and the decisive tone's critical band was the hard-coded
  `#eaf0f8` instead of `#c8deed`. Nothing a reader would misread, since every
  printed value and every verdict was already right, but the two fiches the
  documentation offers as examples of what the library prints were showing a
  paler band than the library prints. `ebu_r128_loudness` and the two
  `iec61260_filter` fiches carried the same kind of drift a shade smaller.
  All five are regenerated here, together with the `iec61043_intensity`
  fiche, which had been registered in the generator and documented in the
  intensity guide without its PDF and preview ever being committed; the guide
  now embeds it in both languages, like every other worked example.

- The `<link rel="alternate" type="text/markdown">` in every page's head
  pointed at a file that does not exist on 147 of them. It was written for
  `<page>/index.md` on every page but the two splash pages, while the copies
  are generated from the content collection, and 146 pages do not have an entry
  of their own: the API reference is generated in English only, and Starlight
  serves the whole of it under `/es/` as fallback content, so those routes
  exist without a file behind them. The 404 page was the 147th, its entry being
  synthesised by Starlight rather than read from the collection. The address
  now comes from the entry actually being rendered, which on a fallback page is
  the English text the reader has in front of them, and is withheld where the
  generator wrote nothing.

- The build published a markdown copy of the English landing page at
  `/index/index.md`, a route no page occupies and nothing links to. The
  generator and the page agree on which pages get a copy, but each worked out
  the route it was talking about for itself, with the same expression Astro
  uses (`/\/index$/`). That expression cannot match at the root of the content
  collection, where the path is `index.mdx` with no directory in front of it,
  so the generator called the landing page `index` and wrote it out while the
  page itself, which Starlight gives the empty route, correctly offered
  nothing. The rule now lives in one place and both sides import it, and the
  file it used to write is gone; the Spanish landing page, whose id Astro does
  normalise, was unaffected either way.

- Eleven figures on seven Spanish guides were the English drawing. The pages
  passed the English asset to `ThemeImage`, which derives only the dark
  variant, so `bearing_fault_envelope.svg` and ten others rendered with
  English axis labels, legends and annotations to a Spanish reader, in both
  themes. Nothing had to be drawn: `generate_graphs.py` fans every figure out
  over language and theme together, so all four variants of all eleven were
  already sitting in `.github/images/` and the pages simply named the wrong
  one. Affected: machine diagnostics, porous absorbers (two), junction
  transmission, resilient layers (two), panel sound insulation (two),
  detailed prediction and CNOSSOS road emission (two). The Spanish tree now
  references no English asset at all.

- The PyOctaveBand transition stub, on both counts its published page got
  wrong, and the reason nothing had noticed. `stub/README.md` is the rendered
  long description of PyOctaveBand on PyPI, and it said the shim re-exports
  the API with a `DeprecationWarning`. It raises a `FutureWarning`, and
  deliberately so, because that is the class Python shows by default; a reader
  acting on the published sentence and calling
  `warnings.simplefilter("ignore", DeprecationWarning)` silenced nothing. The
  3.0.0 entry above repeated the same wrong class and says `FutureWarning`
  now. Second, the stub asked for `phonometry>=3.0.0` with no ceiling, so the
  day the next major ships, `pip install -U PyOctaveBand` would resolve to a
  phonometry that has retired the `octavefilter`, `getansifrequencies`,
  `normalizedfreq` and `calculate_sensitivity` aliases the 3.x `__all__` still
  carries, and `pyoctaveband.octavefilter(...)` would raise `AttributeError`
  with nothing naming the cause. That falsifies exactly the two promises this
  package is for, that a plain rename of the import is a complete migration
  and that upgrading keeps existing code working. The pin is
  `phonometry>=3.0.0,<4` now, which ties the transition package to the line
  whose API is the API of the last release under the old name, and the README
  says so; the stub is 2.1.1. Underneath both defects was the same gap: the
  stub is built straight from the tree by a manual workflow and nothing else
  in the repository imported, type checked or tested it. It now has a test
  module of its own in the ordinary suite, which executes the shipped shim,
  asserts the class it warns with, asserts that the class the README names is
  that same class, and asserts the pin admits every 3.x and no 4.0. `mypy`
  covers `stub/src`, and the stub is built and `twine check`ed on every pull
  request rather than first at publish time, with the cap re-asserted against
  the `Requires-Dist` of the built wheel and sdist, which is what a resolver
  actually reads.

## [3.3.0] - 2026-07-27

### Added

- Documentation site: a "Quick answers" section (Spanish mirror: "Respuestas
  rápidas") at the end of ten core guides (levels, filter banks, weighting,
  calibration, loudness, room acoustics, sound power, field insulation, speech
  intelligibility, FDTD), three question-formed headings each whose one-
  paragraph answers are self-contained restatements of the guide's own
  load-bearing facts. Question-shaped headings are what conversational
  assistants match against user queries; the site had two across 45 sampled
  pages.
- Documentation site: a visible byline under the H1 of every guide and of
  Getting Started ("By José M. Requena Plens · Published … , updated …",
  localized on the Spanish pages), surfacing as text what each page already
  asserted in its JSON-LD, with the author name linking to the About page.
- Documentation site: the About page states the maintainer's education on the
  record (Telecommunications Engineer in Sound and Image, Universidad de
  Alicante, 2011-2018; MSc in Acoustics Engineering, Universitat Politècnica
  de València, 2018-2019; acoustics researcher at the UPV 2020-2023, the
  degrees on the public ORCID record) and adds a direct contact address for
  readers who cannot use the issue tracker.
- Documentation site: the `#person` entity now carries `alumniOf` and
  `hasCredential`. The claim lives in the canonical identity document on
  jmrp.io (published there for every consumer of the entity, not restated
  per site); this site picks it up through its build-time fetch, and the
  committed offline snapshot was refreshed to match.
- Documentation site: a `Content-Security-Policy` meta tag on every page
  (GitHub Pages cannot send it as a header): everything the site loads is
  first-party, and the policy pins that fact.
- Documentation site structured data: the home pages carry a `WebPage` node
  with a `speakable` pointer (title and tagline), the guides and reference
  hubs are dual-typed `CollectionPage` (the guides hub linking its existing
  areas `ItemList` as `mainEntity`), and every article node states
  `wordCount` and `articleSection`.
- The documentation site now renders the numerical conformance report itself:
  `/reference/conformance/` and its Spanish mirror carry every conformance
  table, collapsible section by collapsible section, instead of describing the
  report and linking out to GitHub.
- New page `/reference/errata/` (and `/es/reference/errata/`) publishing the
  registry of defects found in the published sources the library implements
  from: misprints, worked examples that contradict their own normative
  clauses, broken cross-references, each with its evidence, the reading the
  library implements and the test pinning it. Linked from the conformance page
  and from the reference index.
- `make site-reports` (`scripts/generate_site_reports.py`) transplants the
  bodies of `docs/CONFORMANCE.md` and `docs/ERRATA.md` into those four pages,
  below the hand-written introduction each one keeps, rewriting the
  repository-relative links to their GitHub targets. Nothing is hand-copied,
  and a CI job fails the build if a page drifts from its source document.
- Documentation site: an **About** page (English and Spanish) naming who
  maintains the library, with the ORCID, Google Scholar, ResearchGate, GitHub
  and personal-site identities as visible links, how each metric is built from
  the text of its governing standard and checked in CI, how to report an error
  (including the errata registry for defects found in published standards), a
  "Cite this software" block with the Zenodo DOI plus APA and BibTeX entries
  derived from `CITATION.cff`, and the MIT licence.
- Packaging metadata: `Documentation`, `Source`, `Changelog` and `DOI` entries
  in `[project.urls]`, and a `keywords` list covering the domains the library
  spans and the designations of the standards it implements, so a PyPI search
  for a standard number finds the package.
- A conventional `/sitemap.xml` entry point alongside the generated
  `sitemap-index.xml`, for crawlers that probe the default path.
- A clean markdown copy of every page, published next to its HTML as
  `index.md` and advertised with `<link rel="alternate" type="text/markdown">`.
  A rendered page is roughly 90% chrome (the sidebar alone is about 76 kB on
  every page), and the only clean alternative was a single file far too large
  to fetch; `/guides/levels/index.md` is 31 kB against 388 kB of HTML.
- `scripts/check_conformance_claims.py`, run by `make conformance` and by CI:
  it fails the build if any page states a conformance count that disagrees with
  the generated report. The counts had been typed out in several places and two
  of them were a release behind, so the landing page and the page arguing for
  the library's verifiability gave different numbers.
- `ExtendedWeightedRatingResult.plot()` and `ExtendedImpactRatingResult.plot()`
  (ISO 717-1 Annex B / ISO 717-2 A.2.1): the enlarged-range results now carry
  the full measured band curve (`band_centers`, `measured`) and draw it
  against the shifted reference on the 16 core bands, with the unfavourable
  deviations shaded, the enlarged-range bands marked and every covered
  adaptation term in the title (English and Spanish labels).
- Building and structure-borne documentation figures: twelve new deterministic
  figures across the field/laboratory/prediction insulation guides and the
  structure-borne pages (enlarged-range rating, ISO 16283-1 field chain,
  ISO 16283-3 facade quantities, ISO 10052 survey impact, ISO 10140
  laboratory quantities, ISO 15186-1 small elements, ISO 10848 flanking
  Dn,f, EN 12354-2 impact terms, EN 12354-4 radiated power, the rated Sharp
  panel, the wave-approach junction Kij design chart and the ISO 7626
  driving-point mobility reading), each embedded next to its snippet in the
  English guide, the Spanish mirror and the docs mirror with the dual
  one-line `.plot()` / hand-written matplotlib code blocks, plus one-line
  `res.plot()` mentions throughout the building and vibration guides.
- Junction-transmission guide: a worked example feeding the wave-approach
  `K12` of a 100/200 mm concrete X-junction into the EN 12354-1 flanking
  model, and a See-also section; the EN 12354-5 guide cites Hopkins (2007)
  Section 2.8 for the point-excited-plate power input behind the coupling
  term and links the mobility chain. The docs mirrors regain their missing
  ISO 10052 survey-report, ISO 16283-3 facade-report, ISO 10846 fiche and
  panel concept sections.
- `ModulationDistortionResult.plot()` (IEC 60268-3 14.12.7): the result now
  carries the measured carrier amplitude and the four modulation-sideband
  frequencies/amplitudes, and its new `.plot()` draws the carrier (0 dB
  reference) and the `f2 ± f1` / `f2 ± 2f1` sidebands with the per-order
  `d2`/`d3` and the SMPTE combined RMS annotated, the modulation counterpart
  of `HarmonicDistortionResult.plot()`.
- `field_indicators` (ISO 9614-1:1993 Annex A) now accepts 2D
  `(positions, bands)` arrays plus band centre `frequencies` and returns the
  per-band `F2`/`F3`/`F4` arrays; the `FieldIndicators` result gains a
  `frequency` field and a `.plot(dynamic_capability=...)` that draws `F2` and
  `F3` per band against the criterion-1 dynamic capability `Ld` with the
  non-uniformity `F4` on a twin axis. Scalar 1D behaviour is unchanged.
- Documentation figures for the emission and electroacoustics guides
  (four variants each, EN/ES x light/dark): modulation sidebands, baffled
  piston radiation impedance `R1`/`X1`, ISO 9614-1 field indicators vs `Ld`,
  side-branch silencer transmission loss, ASHRAE duct end reflection loss,
  machine-enclosure insertion loss `R − C`, minimum-phase/all-pass
  decomposition, BS.1770 gating and the EBU Tech 3342 loudness range, each
  embedded with its one-line `.plot()` and by-hand matplotlib snippet in the
  English guide, the Spanish mirror and the GitHub docs page.
- `docs/electroacoustics.md` gains the IEC 60268-5 loudspeaker and
  IEC 60268-4 microphone rated-characteristics report sections, and
  `docs/sound-power.md` the measurement-report (`.report()`) and ISO 4871
  noise-emission-declaration sections, matching the site guides.
- `ResampledSignalResult.plot()`: the anti-alias filter the resampler
  actually delivered, drawn against its design specification (passband and
  stopband edges, design attenuation, the rejected band shaded), with
  English and Spanish labels.
- Core metrology documentation figures: eleven new deterministic figures
  across the signal-analysis guides (the resampling anti-alias filter
  against its spec, real/power/complex cepstrum variants, low-time
  liftering, correlation normalizations, cross-correlation impulse-response
  alignment, the Hilbert envelope with instantaneous frequency, the
  cross-spectral density of a delayed path, the coherent output spectrum
  with its SNR, Golay exact recovery, the sqrt(N) noise law of synchronous
  averaging and the reverse-arrangements runs test), each embedded next to
  its snippet in the English guide, the Spanish mirror and the docs mirror
  with the dual one-line `.plot()` / hand-written matplotlib code blocks,
  plus one-line `res.plot()` mentions throughout the metrology guides.
- Core metrology guide prose: the multichannel guide is rewritten around a
  concrete array-measurement story with per-channel guarantees and Bendat &
  Piersol citations; the levels, time-weighting and calibration guides gain
  real introductions (descriptor families and when to use each, detector
  history behind Fast/Slow/Impulse, the field versus laboratory calibration
  workflow); the time-frequency, synchronous-averaging, MISO-coherence,
  GUM-uncertainty and test-signals guides gain See-also sections, practical
  guidance (conditioning order, diagnostics tool choice, window choice) and
  the references behind them.
- Per-guide scope disclosure: 58 English guides and their Spanish mirrors gain
  a "Covered / Not covered" section naming the clauses, formulae, tables and
  annexes of the governing standard that are implemented, and the ones that
  are not, pointing at the page where a missing piece lives when the library
  provides it elsewhere. The eight guides that already stated their negative
  scope in prose keep it as written, so the whole guide set now backs the
  landing-page claim that each guide says what it does not do.
- Query-shaped openers on six guides (field insulation ratings, loudness,
  frequency weighting, occupational exposure, the speech transmission index
  and EN 12354 prediction): a question heading followed by a direct answer
  with the function call and the worked value from the example below it, in
  English and Spanish, with the existing narrative left untouched.
- Getting Started opens with a lead paragraph saying what phonometry is and
  what the first minute of use produces, in English and Spanish, so the page
  no longer starts on an installation tab label.
- New page `/guides/` (and `/es/guides/`), the landing page for the largest
  cluster on the site. It groups the 66 guides into the nine documented areas,
  each with what the area is for, the standards implemented in it and a
  one-line summary of every guide, linking both to the area overview and to
  the guides themselves. The URL was cited as a breadcrumb ancestor and
  referenced from the navigation while returning 404 and staying out of the
  sitemap; it is now a real page, carrying `ItemList` structured data for the
  nine areas.
- New page `/reference/glossary/` (and `/es/reference/glossary/`): a reference
  glossary of every acoustic quantity the guides compute, one row per symbol
  with a one-sentence definition, the unit, the standard and clause that
  defines it and the guide that implements it, across levels, environmental
  and occupational descriptors, weighting, room acoustics, speech, sound
  insulation, materials, vibration, psychoacoustics, electroacoustics,
  aircraft, underwater acoustics and measurement uncertainty. Marked up as a
  `DefinedTermSet` with one `DefinedTerm` per quantity, each carrying its
  symbol as `termCode` and pointing at the implementing guide, so the
  vocabulary is dereferenceable rather than prose. Definitions and clause
  citations are taken from the standards the guides implement; where the
  defining clause is not established, the entry names the standard only.

### Changed

- Documentation site: each KaTeX formula now serialises exactly once for text
  extractors. The MathML block (whose text a DOM walker emitted alongside the
  visual layer, so quoted formulas came out doubled and garbled) is removed
  and the original TeX travels as an `aria-label` on the formula root with
  `role="math"`, the same pattern GitHub uses for rendered math. Screen
  readers announce the TeX source; the visual rendering is unchanged.
- Documentation site: WebP figures now carry intrinsic `width`/`height`
  attributes like the SVG ones already did (the dimensions are read from the
  RIFF header at build time), so the browser reserves the right box before
  the bytes arrive and the page no longer shifts as they load.
- Documentation site: display formulas and collapsible code sections use CSS
  `content-visibility: auto`, so the heaviest guides (the two insulation ones
  run past 10k words) skip layout of what is far below the fold.
- Documentation site: three guide titles that exceeded 65 characters with the
  site suffix (atmospheric refraction, data qualification, dynamic stiffness)
  were shortened so they no longer truncate in search and AI answer snippets.
- The Zenodo deposition metadata (`.zenodo.json`) describes the library as it
  is. Its title, description and keywords were the octave-band ones: a 323
  character summary and six keywords, none of them naming building acoustics,
  psychoacoustics, vibration, environmental, aircraft or underwater noise. The
  record is a top-tier source for scholarly grounding and it was advertising a
  fraction of the package. `scripts/check_conformance_claims.py` now covers
  this file too, since its description states the conformance counts and, like
  the landing page, has no build step to interpolate them through.

- The documentation site serves its own figures, animations and example
  fiches instead of hotlinking `raw.githubusercontent.com`. That host is not a
  CDN, caches for half as long as the Pages origin, is rate limited, is one
  GitHub explicitly asks people not to use as an asset host, and the URLs were
  pinned to `main`, so renaming a figure silently broke every published page
  showing it. A prebuild step stages the media from `.github/images` and
  `.github/reports`: SVG is copied verbatim, the rasters are recompressed with
  the existing image step (36 MB down to 21.7 MB), the WebM animations are
  copied as they are, and the GIFs are skipped since they exist only for the
  markdown mirror, which cannot play WebM. The mirror under `docs/` and the
  README keep their absolute URLs, since GitHub renders those with no build
  step.
- Every page now carries one JSON-LD `@graph` instead of a script per node.
  Google merged the separate blocks before evaluating them, but a plain
  JSON-LD processor expands each `<script>` as its own document, and in that
  reading the `@id` references to the site-wide Person, WebSite and
  SoftwareApplication nodes pointed at nodes with no properties.
- Breadcrumbs are derived from the navigation tree rather than from URL path
  segments. The segment-based trails named grouping prefixes that are not
  routes (`/guides/`, `/guides/sections/`, `/reference/api/<domain>/`), so 414
  of 442 pages advertised a hierarchy that did not resolve, and the labels were
  machine-titlecased in both locales ("Api", "Building"). Labels now come from
  the sidebar, translations included, and a crumb is emitted only where a page
  exists.
- `datePublished` in the per-page structured data is the commit that added the
  page, not the commit that last touched it. The two were the same value, so
  every page claimed it had been published on the day it was last edited, and a
  typo fix moved a guide's publication date forward.
- The `SoftwareApplication` and `WebSite` nodes describe what the library
  covers now. Their description, `featureList` and `keywords` still described
  the fractional octave filter bank the project began as, contradicting the
  visible meta description on the same page; all three are derived from the
  nine documented areas in `src/data/home.ts`, so they follow the landing page.
  The node also declares `alternateName: PyOctaveBand`.
- The `Person` node matches its canonical definition at jmrp.io: the ORCID is
  present as an `identifier`, `knowsAbout` entries are Wikidata-grounded rather
  than free text, and the Bluesky profile is included.
- Guides are typed `TechArticle` + `LearningResource`, and the generated API
  pages use `APIReference`, which is the exact schema.org type for what they
  document. Each page also links its counterpart in the other language through
  `workTranslation` / `translationOfWork`.
- Each page's typed bibliography reaches the structured data: `citation` for
  every entry, `isBasedOn` and `about` for the standards the page implements,
  with one stable `@id` per work minted on the bibliography page so a standard
  is a single entity cited by many pages rather than a repeated string.
- The generated API reference pages lead with what the module does. All 120
  opened with the "auto-generated, do not edit by hand" provenance note, so a
  maintenance instruction was the search snippet and the `speakable` target on
  every one of them; the note now follows the module summary.
- The conformance report page carries `Dataset` structured data, with its
  column schema, its download URL and the DOI. It is the most citable object
  the project has and it was typed as an article about a report rather than as
  the report.
- The generated API reference pages describe themselves with the module's own
  docstring summary. They previously all read "Public API of
  phonometry.<module> (auto-generated)", which restates the URL, while the page
  body already opened with a precise summary naming the standards implemented.
- `llms.txt` is generated from the documentation tree instead of a hand-kept
  list. The list had been written for the original filter-bank library and
  never extended, so it named 42 of 221 English pages and left whole domains
  (aircraft noise, underwater propagation, FDTD simulation, the entire API
  reference) invisible, under a summary that still described a filter bank.
  The generator now fails if a page cannot be placed. The full text is also
  published as one file per area, each small enough to fetch whole, since
  `llms-full.txt` is far past the size most clients will read.
- The documentation `robots.txt` served under the site subpath now declares the
  same four `Content-Signal` keys as the authoritative file at the domain root
  (`search`, `ai-train`, `ai-input`, `ai-retrieval`), and says in a comment
  which of the two is authoritative.
- The docs deploy workflow also triggers on `docs/` and on the llms.txt
  generator, so a change to the markdown mirror republishes the site instead of
  leaving the published `llms.txt` behind the docs.

### Fixed

- Formulas no longer serialize twice into the page text. KaTeX emits the
  original TeX in an `<annotation>` inside the MathML, which is not
  `aria-hidden`, so a text extractor read `L_eq` as `LeqL_{eq}Leq`. The
  annotation is dropped and the MathML kept, on the standards formulas that
  carry most of the documentation's value.
- Figures declare their intrinsic size, read from the SVG at build time, so the
  browser reserves the right box instead of reflowing the page as each lazy
  image arrives. Almost none carried dimensions before (0 of 12 on one sampled
  guide, 2 of 50 on another).
- A figure pair no longer downloads the palette the reader will not see.
  `display: none` suppresses a lazy fetch but not an eager one, so an
  above-the-fold figure fetched both variants and discarded one, about 37 kB on
  the landing page and in either theme. Both variants are lazy now, with
  `fetchpriority="high"` preserving the head start.
- The landing page has a title of its own. Both locales rendered
  "phonometry | phonometry", spending the highest-weighted element on the entry
  page on nothing, and the two splash pages are typed `og:type: website` rather
  than `article`.
- The 404 page no longer emits article structured data headlined "404".
- `.report()` fiches: consecutive lines of wrapped prose no longer collide
  vertically when the text carries `<sub>`/`<super>` markup (e.g. the
  EN 12354-5 formula strip, where the subscripts of `L_n,s,ij` overprinted
  the next line). reportlab shifts a script by half the font size but spaces
  lines by the style's fixed leading alone (its `autoLeading` modes ignore
  the script rise), so every fiche paragraph is now built through a shared
  factory that moderates the shift to the typographic quarter-em, which fits
  inside the fixed leading of every fiche style; the committed example fiches
  are regenerated accordingly.
- Restore the `phonometry._plotting` compatibility re-exports that were lost
  during the lint modernization: the module is again a silent re-export of
  every plot renderer moved to `phonometry._plot` (kept for one deprecation
  cycle, removed in 4.0), now with an explicit `__all__` so the lint fixers
  keep it, and the deprecation test suite asserts the full re-export surface.
- `sound_power_reverberation` / `sound_power_comparison` (ISO 3741:2010): with
  per-position levels the background correction is now applied at each
  microphone position, as clauses 9.1.2/9.1.3 prescribe: `K1i` per position
  (Eq. 14), position levels corrected first (Eq. 15), then energy-averaged
  (Eq. 16). The previous single `K1` from the position-averaged spectra could
  sit up to ~0.2 dB off when positions had spread margins around the 6/10 dB
  criteria, and missed per-position clamps. Pre-averaged 1D spectra keep the
  single-`K1` path, now documented as an approximation.
- `sound_power_intensity` (ISO 9614-2:1996): the A-weighted total now omits
  the bands in which criteria 1 and/or 2 are not satisfied, as clause 10.6 b
  requires, whenever the criteria are evaluable (`pressure_levels` and
  `pressure_residual_index` supplied); the omitted bands are flagged in the
  new `a_weighting_omitted_bands` result field and named in the `.report()`
  basis strip (English and Spanish). Without the criteria inputs the previous
  behavior is kept and a `SoundPowerWarning` notes the missing screening.
- `precision_qualification` (ISO 9614-3:2002): a band whose doubled-density
  scan satisfies criterion 5 now qualifies as a final result even where
  `FS(2) >= 2` fails criterion 4 (C.1.6.2).
- `sound_power_anechoic` (ISO 3745:2012): one-third-octave bands above 10 kHz
  (which the sigma_R0 tables cover up to 20 kHz) no longer abort the whole
  determination because the ISO 3744 Annex E A-weighting table stops at
  10 kHz; the per-band levels and uncertainties stay defined and only the
  A-weighted total is `NaN`.
- `background_noise_correction` (ISO 3744:2010 / ISO 3746:2010): `K1 = 0` only
  strictly above the upper criterion; at exactly `dLp = 15 dB` (engineering,
  8.2.3) or `10 dB` (survey, 8.3.3) Equation (16) still applies, restoring the
  0,14 dB / 0,46 dB correction at the boundary.
- `OperatingModeDeclaration` (ISO 4871:1996): dual-number verification per
  clause 6.2 now compares `L_1` against the sum of the separately rounded
  declared values (`round(L_WA) + round(K_WA)`, clause 3.16) through the new
  `verified_dual` / `dual_number_verification_limit` members, and the
  dual-number `.report()` fiche follows the Annex B.2 layout exactly (the
  derived `L_WAd` row, which mixed the two rounding rules, is dropped and the
  verification row states the dual-number limit).

### Fixed

- `room.noise_criterion` no longer fabricates out-of-family NC designations:
  a spectrum above the NC-70 curve (or entirely below NC-15) used to clamp
  the interpolation to a nonexistent "NC-71"/"NC-14" with an arbitrary
  governing band; it now returns `rating = nan` with an `out_of_range` flag
  (`"above"`/`"below"`), a `label` reading `">NC-70 (band)"`/`"<NC-15"`, and
  the governing band chosen by the maximum exceedance over the NC-70 curve.
  The fiche and the plot render the flagged designation (English and
  Spanish) and the verdict fails above the family and passes below it.
- The room-acoustics fiche (ISO 3382-1/-2) labels the mid-frequency
  descriptor honestly for one-third-octave data: only the 500 Hz and 1 kHz
  one-third-octave bands are averaged, so the box no longer claims the
  octave "500-1000 Hz" span; a band range without both mid bands names the
  band the boxed T30/EDT came from, and the "EDT_mid" vs "EDT" label now
  follows EDT's own band coverage instead of T30's.
- Room fiche verdicts (ISO 3382-1/-2 and ANSI/ASA S12.2) compare the values
  rounded exactly as displayed, so a printed `T_mid = 1.15 s` can no longer
  fail a `1.15 s` requirement on an invisible third decimal.
- Docstring citations corrected: the 5 % reverberation-time JND is
  ISO 3382-1:2009 Table A.1 (ISO 3382-2:2008 Table A.1 tabulates the
  uncertainty constants G and H), and the -20 dB onset trigger of the
  impulse-response onset detector is ISO 3382-1:2009 A.3.4, not A.2.1.
- English fiche strings and docstrings for ISO 3382-3 use point decimals
  ("STI = 0.50"/"0.20"); the Spanish strings keep the comma.
- `metrology.time_synchronous_average`: the result's `times` axis now
  reports the true `1/fs` sampling grid the averaged samples sit on,
  instead of a `T/M` spacing that drifted across the period when
  `fs*period` is not an integer (noiseless recovery read against the old
  axis was wrong by up to ~9e-2 for a 100.37-sample period; on the true
  grid the error is the fractional-delay interpolation level, ~3e-6).
- `metrology.echo_detection`: the peak is now picked on `|cepstrum|`, so
  an inverting reflection (negative reflection coefficient), whose first
  rahmonic is negative, is detected at its true delay instead of being
  silently missed; the signed cepstrum value at the peak is returned as
  the reflection coefficient, and the reported delay is refined by
  three-point parabolic interpolation so off-sample delays are estimated
  to a fraction of a sample (with the bin-splitting caveat on the
  coefficient documented). Note that `delay_samples` stays the whole
  peak-sample index, so `delay == delay_samples/fs` no longer holds
  exactly when the interpolated offset is nonzero.
- `metrology.EQSection`/`ParametricEQ`: informative `ValueError`s replace
  a raw `OverflowError` (bandwidth-derived alpha overflowing near
  Nyquist), a bare `math domain error` (shelf slope beyond the
  gain-dependent bound `(A + 1/A)/(A + 1/A - 2)`) and two silent
  failures (sections whose rounded poles land exactly on the unit
  circle, and non-finite `f0`/`gain_db`/`q`/`bw`/`slope` values passing
  validation).
- `metrology.tone_burst`: warns when `frequency` is incommensurate with
  `fs`, in which case the gate cannot close exactly at the tone's final
  zero crossing (the "integral number of full periods" of IEC 60268-1
  Clause A2.1 is then sample-exact only to the nearest sample) and the
  waveform carries a small residual step at the gate edge.

### Fixed

- `intensity_element_normalized_difference()` (ISO 15186-1): return the
  derivable per-unit value with `+10 lg N` for `N` element units measured
  together. The printed Formula (8) subtracts its `10 lg(N)` term, which
  contradicts ISO 10140-2:2010 Formula (6) and ISO 15186-2:2010 Formula (12)
  (registered in `docs/ERRATA.md`); `n > 1` now warns about the deviation
  from the print.
- ISO 16283-3 façade fiche: a road-traffic measurement is now labelled with
  its own quantity `R'tr,s` (Clause 3.13) instead of the loudspeaker `R'45`
  basis line; `FacadeInsulationResult` carries the measurement `method`.
- ISO 10848 `Dn,f` / `Ln,f` fiches: the basis line names the ISO 10848 part
  governing the tested construction via the new `part` argument (2, 3 or 4)
  instead of hardcoding ISO 10848-2:2006.
- `junction_vibration_reduction()` (EN 12354-1 Annex E): the E.7 double-leaf
  branch (K24) now uses the same per-path `mass_ratio = m'⊥,i/m'i` as every
  other branch (validity `m2/m1 > 3` reads `mass_ratio < 1/3`), evaluating
  `3,0 + 14,1 M + 5,7 M²` as ISO 12354-1:2017 E.3.5 prints it; the 2000
  edition's `−14,1 M` line is the same relation recast in its figure's
  x-axis variable (the K24 errata entry is rewritten accordingly — the
  earlier reading of the 2017 print as a sign misprint was wrong).
- `installed_source_prediction()` (EN 12354-5): a single-number
  characteristic power level with per-band path data now broadcasts across
  the path bands instead of raising or producing a one-element
  `installed_power_level` that crashed the fiche table.
- EN 12354-1/-2/-3/-5 prediction fiches: the footer disclaimer now reads
  "Predicted result: the values relate only to the modelled configuration,
  not to a tested specimen." instead of the measurement wording, and the
  2000 editions are cited as `EN 12354-x:2000` (not `EN/ISO`).
- ISO 10052 survey fiches: the basis line declares the band set the
  reported curve actually carries (octave or one-third-octave) instead of
  hardcoding "octave bands".
- ISO 16251-1 fiche: a result with bands at the 1,3 dB limit of measurement
  is qualified as a lower bound (`≥` on the boxed `ΔLw` plus a note), and a
  hand-built result without `CI,Δ` no longer prints a fabricated `(+0)`.
- ISO 15186-1 intensity fiches: the method statement states the test-object
  and measurement-surface areas `S` / `Sm` (Clause 8 g) and `report()`
  accepts optional per-band `FpI` / `δpI0` spectra (Clause 8 i); the
  Table B.1 `Kc` oracle now pins all 21 printed rows (3150/4000/5000 Hz
  added), and the 6.4.2 qualification wording is "must not exceed 10 dB".
- Spanish impact terminology aligned with the UNE editions: the
  ISO 16283-2 / ISO 10052 basis lines use "nivel de presión acústica
  estandarizado/normalizado de ruido de impactos" (UNE-EN ISO 16283-2:2016,
  3.13/3.14).
- Citation and docstring corrections: `Kij,min` is Formula (29)
  (Clause 4.4.2), the ISO 10140 test report is Clause 9, the ISO 717-1:2020
  band-set statement is Clause 5.3, the ISO 12354-1:2017 both-negative
  lining rule is documented next to `combine_linings()`, and the EN 12354-5
  coupling term `D_C` is documented as not guaranteed non-negative (mounting
  resonance counterexample). New errata entries: ISO 15186-1:2000
  Formula (8), ISO 10848-1:2006 Formula (20) (spurious π), and the extended
  ISO 717-2:2020 Annex C.2 chain (the printed `Ln,sum = 75,2527` is the
  energy sum of the wrong column over the wrong range).

### Changed

- `room.noise_criterion` now follows the full two-step rating procedure of
  ANSI/ASA S12.2-2019, 5.2.2: the speech interference level (clause 3.2,
  average of the 500/1000/2000/4000 Hz bands) selects the NC-(SIL) curve
  and, when no octave band exceeds it, the spectrum is designated NC-(SIL);
  the tangency method (5.2.3) applies otherwise. `NCResult` gains `sil`,
  `tangency_rating`, `method`, `out_of_range` and a `label` property; the
  tangency rating remains available on every in-family result.
- `room.room_criterion` warns when the clause D.4 minimum band set
  (31.5 Hz through 4000 Hz) is incomplete, flags ratings outside the
  tabulated RC-25 to RC-50 family via `RCResult.out_of_family` (the fiche
  adds an extrapolation note), and documents the combined "RH" tag as a
  diagnostic extension beyond the clause D.3.5 letters (N, R, H, RV; the
  RV vibration/rattle tag needs the Table 6 test and is not implemented).
- ISO 1999 `nipts()`/`htlan()` now warn (`NoiseInducedHearingLossWarning`) when
  the exposure conditions leave the domain the standard validates: exposure
  durations outside 1-40 years (clause 6.3.1), fractiles in the distribution
  tails the standard says "should not be estimated" (Q below 5 % or above
  95 %, clause 6.3.2), and exposure levels above the 100 dB its data cover.
  Previously a 130 dB / 60-year / Q = 1 % request returned a 357,9 dB
  threshold shift with no indication that it was an extrapolation; the
  prediction still runs, and both fiches now print the caveat.
- The HVAC elbow insertion loss picked the wrong row of Bies Table 8.11 for a
  `W/lambda` ratio landing exactly on a bin edge (0,14 / 0,28 / 0,55 / 1,11 /
  2,22). The table's rows read "a <= W/lambda < b", so an edge opens its row:
  at W/lambda = 0,14 a square unlined bend is now 1 dB, not 0 dB.
- The STI and SII fiches printed the requirement with one decimal while
  showing the measured value at full precision, so a verdict could contradict
  itself ("STI = 0.50, required >= 0.5 -> FAIL" for a 0,52 requirement) and an
  SII minimum of 0,75 printed as "0.8". Both now print the requirement at the
  quantity's own display precision.
- The ISO 1999 fiches printed the library's fractile as ISO 1999's percentage
  `Q`, inverting its meaning: the standard's `Q` (6.3.2, Formulae (4)/(5)) is
  the fraction of the population with *worse* hearing, so the
  most-susceptible tenth is Q = 10 %, not 0,9. The fiches now print ISO's `Q`
  with its own definition and cite the clause correctly.
- The ISO/PAS 1996-3 fiche justified a zero adjustment with "governing
  P <= 5" even when the adjustment was withheld by the onset-rate gate
  instead, contradicting its own boxed P. The note now states whichever gate
  governs, and the assessment period reflects the interval analysed rather
  than a hardcoded 30 min (the standard's Clause 5 default, now the default of
  the new `assessment_period_min` argument).
- The wind-turbine tonality fiche cited "subclauses 9.5.2-9.5.5" of
  IEC 61400-11 for a chain that runs through 9.5.6 to 9.5.8 (Formulae 31-34);
  the citation now spans 9.5.2 to 9.5.8.
- The noise-control fiches decided verdicts on the half-away-from-zero display
  rounding but printed the value with Python's round-half-to-even formatting,
  which could disagree at a boundary (0,25 printed as 0.2 against a verdict
  taken on 0,3). Both now use the display rounding.
- The shared 10-90 % fractile band of the hearing plots carried a hardcoded
  English legend entry in Spanish figures.

### Added

- `TransferMatrix.plot(frequency, characteristic_impedance)` (ASTM E2611-19):
  the four-pole entries read out as the two spectra a transmission-tube
  laboratory quotes, the normal-incidence transmission loss `TLn(f)`
  (Eq. (26)) with the hard-backed absorption coefficient (Eq. (28)) on a
  companion axis, in English and Spanish like every other result `.plot()`.
- Nine result figures across the rooms & materials documentation (EN and ES
  guides plus the `docs/` mirrors), each generated from a realistic example
  through the public API and its `.plot()`: the ISO 354 reverberation-room
  `alpha_s` spectrum, the ISO 10534-2 two-microphone tube result, the
  ASTM E2611 transfer-matrix transmission loss, the Miki porous-medium
  characteristic values, the Maa microperforated-panel absorption peak, the
  Paris-integral diffuse-field versus normal-incidence absorption, the Bies
  steady-state room field with its critical distance, the ISO 3382 per-band
  decay-time/clarity summary, and the rigid-box FDTD probe spectrum against
  the analytic room modes. Every snippet whose result exposes a `.plot()` now
  mentions the one-liner next to its figure.
- The rooms & materials `docs/` mirrors regain section parity with the site
  guides: the EN 12354-6 enclosed-space report fiche, the reverberation-time
  prediction report fiche and the Schroeder-diffuser prediction section
  (with its figure) are now documented in `docs/` too, and the image-source
  guide gains the Validation summary in both site languages.
- `ToneAssessment.plot()` (ECMA-418-1 tone-to-noise and prominence ratios):
  the assessed tone drawn against the frequency-dependent prominence criterion
  of its own method over the 89.1 Hz to 11.2 kHz range of interest, with the
  margin that decides the verdict marked. The criterion family is recovered
  from `criterion_db`, so the same result object plots correctly whether it
  came from `tone_to_noise_ratio()` or `prominence_ratio()`.
- `ToneAudibilityResult.plot(view="levels")` exposes the level-versus-frequency
  view of an ISO/PAS 20065 assessment (tone levels above their critical-band
  masking noise) that until now only the `.report()` fiche could draw; the
  default `view="audibility"` keeps the per-tone bars.
- Eleven documentation figures for the perception, hearing and speech guides,
  each a single-concept result drawn by its own `.plot()`: the ECMA-418-1 tone
  prominence verdict, the ISO/PAS 20065 tone levels, the ISO/PAS 1996-3 onset
  detection on a level history, the ISO 532-2 specific loudness, the
  ECMA-418-2 specific tonality, the specific fluctuation strength, the SII
  band audibility with a hearing loss, the ISO 7029 fractile band, the
  ISO 1999 NIPTS audiogram, the STOI per-band correlation and the IEC 60268-16
  per-band MTI.
- Environment and underwater documentation: four new figures with their
  reproducible code blocks (the log-linear effective sound-speed profiles,
  the downward-refraction ray fan and the GFPE level-versus-range cut of the
  atmospheric-refraction guide, and the Kurze-Anderson / exact half-plane /
  coherent-ground barrier insertion-loss comparison of the ground-and-barriers
  guide), in English, Spanish, light and dark variants. New prose: a
  "when does refraction matter" introduction (Attenborough & Van Renterghem
  2021, Ch. 11), a "Choosing a model" section in the underwater-propagation
  guide (sound-speed and absorption validity ranges, spreading-law guidance,
  the ray/mode/PE selection table after Jensen et al. 2011 Ch. 1, and a worked
  sonar budget), and the ISO 17208-1 measurement-discipline and ISO 18406
  regulatory-context paragraphs of the underwater-acoustics guide. The
  outdoor-propagation docs mirror gains the missing "Prediction reports"
  section and all six pages are re-synced across the site guides, the Spanish
  mirrors and the `docs/*.md` tree, with `.plot()` one-liners added next to
  every plottable snippet.
- ISO 1999 `combine_age_and_noise()` applies Formula (1)
  (`H' = H + N - H*N/120`) to an age component the caller supplies, which
  clause 6.2 explicitly allows (a database B of the country under
  consideration). It is the composition `htlan()` uses internally, and it
  carries the new ISO 1999 Annex C conformance oracle: the annex's worked
  example (90 dB, 30 years, 1/2/4 kHz at Q = 10 %) reproduces its printed
  13,3 dB compressed shift at 4 kHz and its 31,1 dB combined threshold.
- An independent IEC 60268-16 Annex M conformance oracle: the annex's printed
  MTF matrix, speech spectrum and ambient noise reproduce its published MTI
  row and STI = 0,76 end to end, exercising the whole A.5.3 to A.5.6 chain.
- `docs/ERRATA.md` records two further defects: ISO/PAS 1996-3:2022 Clause 5
  swaps the 3.4/3.5 cross-references of the onset rate and level difference,
  and ISO 9613-2:1996 Table 2 prints 4,1 dB/km at 15 °C / 80 % / 1 kHz where
  ISO 9613-1 gives 4,15 (the library computes the coefficient directly, so it
  is unaffected).

### Changed

- Perception, hearing and speech guides (EN, ES and the `docs/` mirrors): the
  new result figures are embedded next to the snippets that produce them with
  the usual dual-snippet block, every plottable snippet on those pages names
  its `.plot()`, the tone-prominence guide gains a "which tonality metric, and
  when" comparison of the four tonality assessments the library implements
  (ECMA-418-1, ISO/PAS 20065 via ISO 1996-2 Annex J, ECMA-418-2 and
  IEC 61400-11), the hearing-threshold guide explains how the ISO 7029
  fractiles feed the ISO 1999 age component and gains a See-also section, and
  the objective-intelligibility guide gains the "which measure, and when"
  section that until now existed only in the `docs/` mirror, extended with the
  STOI/ESTOI against STI against SII choice.
- The STOI band-score plot labels the one-third-octave band centres at
  whole-hertz resolution (150, 189, 238 Hz) instead of printing the generating
  floats, and the ISO/PAS 20065 levels plot shades the decisive critical band
  in a lightness that follows the figure background, so it reads on both the
  white report page and a dark-theme figure.
- The machine-enclosure, reactive-silencer and HVAC duct-noise fiches now read
  as the model predictions they are, following the EN/ISO 12354 prediction
  exemplar: a prediction-basis line, a prediction statement naming what the
  model does not represent, and a prediction-scoped footer in place of "the
  results relate only to the tested specimen". The committed examples drop
  their measurement framing accordingly.
- The ISO 1999 fiches state the ISO 7029:2017 source of their age component
  `H` (which diverges from the illustrative Table A.3 selection), carry the
  Scope NOTE 1 caveat that the 2/3/4 kHz combination is the user's own choice,
  and close with a population-appropriate scope statement.
- The ISO 1996-2 tonal fiche notes that Table J.1 is defined on the mean
  audibility of the J assessed spectra (ISO/PAS 20065 Clause 5.3.9), which a
  single-spectrum assessment stands in for.
- The Spanish silencer and HVAC fiches translate the device kind and the duct
  element label ("Dispositivo: cámara de expansión", "Codo (rectangular, con
  absorbente, W = 300 mm)"), keeping symbols, numbers and units as written.

- Adopt ruff 0.16 and its broadened default rule set: modernize type
- ISO 17497-2 diffusion oracle: replace the in-house COMSOL simulation arc
  with a fully published-provenance scheme. The committed 37-receiver polar
  levels in `tests/reference_data.py` are now generated by the library's own
  Fraunhofer far-field model (`materials.diffuser_design`) for a published
  geometry, the N = 7 QRD, 6 periods, 3.6 m wide, 0.2 m deep array of
  Cox & D'Antonio, *Acoustic Absorbers and Diffusers*, 3rd ed., Appendix B
  (the commercial N = 7 QRD of Hargreaves, Cox, Lam & D'Antonio, JASA 108(4),
  1710-1720, 2000, Table I), and serve as an arithmetic oracle for
  Formulas (5)/(7) (`d = 0.1099`, flat reference `0.0049`, `d_n = 0.1055`).
  A new external third-party anchor pins the model's one-third-octave
  band-averaged normalised diffusion to the published Appendix B 2D BEM
  values for that row at normal incidence in the 200-400 Hz bands (agreement
  within 0.01, asserted at +/-0.015, in the tests and four new conformance
  rows); across the full published 100-5000 Hz range the model-vs-BEM mean
  absolute deviation is about 0.09, documented as a low-band anchor. The
  surface-scattering guide (EN/ES), the theory pages and the
  `diffusion_polar` figure now use the same published-geometry prediction. modernize type
  annotations (PEP 585 built-in generics, unquoted annotations), sort imports
  and apply the new lint fixes across the tree. No runtime behavior changes.

### Added
- `vibration.wbv_exposure_basis()`: the whole-body daily-exposure basis of
  Directive 2002/44/EC (Annex, Part B, point 1), the highest of the
  frequency-weighted axis values `1,4*a_wx`, `1,4*a_wy`, `a_wz` for a seated
  or standing worker. `daily_vibration_exposure` now documents which magnitude
  each kind must be fed with for a Directive-conforming assessment: the Part A
  vector total `a_hv` for hand-arm, the Part B dominant-axis value for
  whole-body (not the ISO 2631-1 Eq. (10) vector total `a_v`).
- `metrology.envelope_spectrum`: optional `band=(low, high)` zero-phase
  band-pass pre-filter ahead of the envelope detector -- the band-pass
  front end of Bendat & Piersol Figure 13.11 and the classical
  bearing-envelope chain; the band travels with the result. The
  amplitude-calibration docs now state the on-bin condition and the
  taper scalloping loss for off-bin modulation lines.
- `materials.diffuser_design`: far-field polar-response and diffusion-coefficient
  prediction of a Schroeder phase-grating diffuser from its surface design,
  complementing the measurement-only `materials.scattering_diffusion`.
  `predict_diffuser_polar_response()` evaluates the single-plane Fraunhofer
  (Fourier) far-field model of Cox and D'Antonio (Eqs. (5.8)/(9.32)): each
  rigid-bottom well of depth `d_n` contributes a pressure reflection
  coefficient `R_n = exp(-2 j k d_n)` and the scattered pressure at each
  reflection angle is the sum over the wells of the periodic surface, reduced
  to the ISO 17497-2 directional diffusion coefficient through the existing
  `directional_diffusion_coefficient`. `predicted_diffusion_spectrum()` returns
  a `DiffusionSpectrum` of the predicted diffusion `d(f)`, normalised band by
  band against the same-footprint flat reference (Formula (7)). Quadratic
  residue diffuser geometry is provided by `quadratic_residue_sequence()`
  (Eq. (10.2), `s_n = n^2 mod N`) and `qrd_well_depths()` (Eq. (10.3),
  `d_n = s_n * lambda_0 / (2 N)`); an arbitrary user depth sequence or an
  explicit complex per-well `reflection` sequence (for admittance or
  resonator-loaded surfaces) is accepted directly. The `DiffuserPolarResponse`
  result exposes the per-angle predicted levels, the directional diffusion
  coefficient and `.plot()` (a semicircular polar diagram, English and Spanish).
  This is a far-field design estimate, not a substitute for an ISO 17497-2
  measurement.
- Slow-sound slit + Helmholtz-resonator perfect absorbers
  (`materials.slow_sound_absorber`), a clean-room implementation of the
  transfer-matrix model of Jimenez, Groby, Pagneux and Romero-Garcia
  (Appl. Sci. 2017, 7, 618) with the resonator impedance and radiation end
  corrections of Jimenez et al. (Appl. Phys. Lett. 2016, 109, 121902).
  `slit_helmholtz_absorber` predicts the oblique-incidence absorption of a
  periodic slit panel loaded by an array of Helmholtz resonators, including
  the visco-thermal losses of the slit and of the square resonator necks and
  cavities (Stinson 1991), returning a `SlitResonatorAbsorberResult` with the
  surface impedance, reflection factor, absorption, retrieved effective
  parameters and the full chain matrix, plus a `.plot()` of `alpha(f)` with
  `|R|` overlaid. `critical_coupling_design` solves the inverse problem: it
  tunes the cavity length and slit height so the reflection zero sits on the
  real-frequency axis, giving perfect absorption (`alpha = 1`) at a chosen
  frequency and angle. The building blocks `slit_effective_properties`,
  `rectangular_duct_properties` and `helmholtz_resonator_impedance` and the
  `HelmholtzResonator` geometry are exported too. Pinned to the exact analytic
  anchors: perfect absorption at the design frequency, the Poiseuille
  resistivity limits `12 eta / h^2` (slit) and `28.454 eta / w^2` (square
  duct), and the loss-free effective-parameter limits.
- `StaticAirflowResult.report()`: a one-page PDF material airflow-resistance
  test report (ISO 9053-1:2018, static/direct airflow method). The fiche
  carries a standard-basis line, an optional metadata header (client,
  manufacturer, specimen, specimen thickness, test facility, date, climate), a
  two-panel body with a metrics table (the evaluation velocity, the fitted
  pressure difference, the airflow resistance `R`, the specific airflow
  resistance `R_s`, the airflow resistivity `sigma` when a thickness is
  available, and the through-origin fit coefficients `a` and `b`) beside the
  fitted `dp(u)` curve, and a boxed specific airflow resistance `R_s` with the
  airflow resistance `R` and the resistivity `sigma` alongside, all read at the
  clause 7.5 reference velocity of 0.5 mm/s. ISO 9053-1 is a material
  characterisation, so the fiche carries no pass/fail verdict. `language="es"`
  renders the Spanish fiche.
- `WindTurbineTonalityResult.report()`: a one-page PDF wind-turbine
  tonal audibility assessment fiche (IEC 61400-11:2012+A1:2018, subclauses
  9.5.2-9.5.5). The sheet carries a standard-basis line, an optional metadata
  header (source/situation, client, measurement position, instrumentation,
  date), a two-panel body with the critical-band and masking analysis in a
  metrics table (tone frequency, critical bandwidth, tone level `L_pt`,
  masking-noise level `L_pn`, tonality `dL_tn`, audibility criterion `L_a` and
  tonal audibility `dL_a`) beside the narrowband-spectrum plot with the
  critical band, masking level and tone marked, and a boxed decisive tonal
  audibility `dL_a` with the tone frequency and the audibility decision (a
  tone is audible when `dL_a` exceeds 0 dB). A maximum acceptable tonal
  audibility supplied via the metadata `requirement` adds a PASS/FAIL verdict
  (a lower audibility is better). `language="es"` renders the Spanish fiche.
- `IntensityElementNormalizedResult.report()`: a one-page PDF
  element-normalized sound-intensity insulation fiche (ISO 15186-1:2000). It
  reports the element-normalized level difference `DI,n,e` (Clause 3.9,
  Formula (8)) of a small building element measured with sound intensity: a
  standard-basis line, an optional metadata header, the per-band table (16
  one-third-octave or 5 octave bands) beside the measured-versus-shifted-
  reference curve, and the boxed rating `DI,n,e,w (C; Ctr)` (rated by the
  ISO 717-1 airborne machinery). `verbose=True` shows the ISO 717 evaluation
  per band (the reported value, the shifted reference and the unfavourable
  deviation). A target supplied via the metadata `requirement` adds a
  PASS/FAIL verdict (the element insulation passes at or above the target).
  `language="es"` renders the Spanish fiche. The renderer is shared with the
  sibling intensity sound reduction index `RI` fiche.
- `NiptsResult.report()` and `HtlanResult.report()`: one-page PDF
  noise-induced hearing-loss **prediction** fiches (ISO 1999:2013). Both sheets
  are clearly labelled statistical predictions for a noise-exposed population,
  not clinical diagnoses. The NIPTS fiche (clause 6.3) carries a
  prediction-basis line, an optional metadata header (company, worker(s)/group,
  workplace, date), a per-audiometric-frequency table of the median `N50` and
  the NIPTS at the chosen population fractile beside the shift spectrum, and a
  boxed representative shift averaged over the 2/3/4 kHz hearing-handicap set
  with the exposure conditions (`LEX,8h`, exposure years, fractile);
  `verbose=True` adds the upper/lower spread columns. The HTLAN fiche
  (clause 6.1) tables the age component `H`, the noise component `N` and the
  combined threshold `H' = H + N - H*N/120` beside the plot, boxing the
  representative combined threshold; `verbose=True` adds the `H*N/120`
  compression term. A maximum acceptable representative value supplied via the
  metadata `requirement` adds a PASS/FAIL verdict (a lower value is better);
  without it neither fiche prints a verdict. `language="es"` renders the
  Spanish fiches.
- `ReverberationModelResult.report()`: a one-page PDF reverberation-time
  **prediction** fiche. The sheet is clearly labelled a design-stage
  prediction, not a measurement: its basis line names the five classical
  statistical-acoustics models (Sabine, Eyring, Millington-Sette, Fitzroy and
  Arau-Puchades), it carries an optional metadata header (client, room,
  description, room volume, total surface area, climate, date), a per-band
  table with one reverberation-time column per model beside the model
  comparison plot, and a boxed mid-frequency reverberation time from
  Arau-Puchades (the recommended model for a non-uniform absorption
  distribution) with the per-model spread alongside. The five models bracket
  the reverberation time likely to occur. A target reverberation time supplied
  via the metadata `requirement` is printed as a reference line without a
  PASS/FAIL verdict, since a room reverberation time is a target range rather
  than a strictly higher/lower-is-better quantity. `language="es"` renders the
  Spanish fiche.
- `ReverberationResult.report()`: a one-page PDF enclosed-space characterisation
  fiche (EN 12354-6:2003). It carries a basis line naming EN 12354-6:2003, an
  optional metadata header (client, room, description, room volume, object
  fraction, climate, date), a per-band table of the equivalent sound absorption
  area `A` and the reverberation time `T` beside the reverberation-time plot,
  and a boxed mid-frequency reverberation time with the mid-frequency absorption
  area alongside. EN 12354-6 gives a diffuse-field estimate; a target
  reverberation time supplied via the metadata `requirement` is printed as a
  reference line without a PASS/FAIL verdict. `language="es"` renders the
  Spanish fiche.
- `MobilityResult.report()`: a one-page PDF mechanical-mobility measurement
  fiche (ISO 7626-1:2011 frequency-response-function definitions; measurement
  per ISO 7626-2:2015). Mechanical mobility is a continuous frequency-response
  function, not an octave-band quantity, so the sheet presents it honestly as
  the mobility magnitude spectrum `|Y(f)|` plus a compact table of the FRF's
  characteristic points (the FRF type, driving-point or transfer, the frequency
  range, the peak frequency, the peak mobility magnitude and the phase there),
  with a boxed peak mobility `|Y|` at the frequency it occurs at (for a
  driving-point FRF a resonance, where `|Y| = 1/c` measures the damping). It is
  a characterisation, so there is no pass/fail verdict. `language="es"` renders
  the Spanish fiche.
- `TransferStiffnessResult.report()`: a one-page PDF dynamic-transfer-stiffness
  characterisation fiche for a resilient element (ISO 10846-1:2008 definition;
  determined by the direct method, ISO 10846-2:2008, or the indirect
  blocking-mass method, ISO 10846-3:2002). The dynamic transfer stiffness is a
  continuous frequency-response function, so the sheet presents it as the
  transfer-stiffness level spectrum `Lk(f)` plus a compact table of the FRF's
  characteristic points (the determination method, the blocking mass for the
  indirect method, the frequency range, and the low-frequency stiffness plateau
  `|k2,1|`, its level `Lk` and the loss factor there), with a boxed
  low-frequency `Lk` (the plateau that characterises the element below its
  internal resonances). It is a characterisation, so there is no pass/fail
  verdict. `language="es"` renders the Spanish fiche.
- `.report()` for the field and survey building-insulation results: a one-page
  PDF field test report for each. `SurveyAirborneResult`, `SurveyImpactResult`
  and `SurveyFacadeResult` (ISO 10052:2021 survey/control method, octave or
  one-third-octave bands) render the standardized level difference `DnT` (or the
  apparent sound reduction index `R'`), the standardized impact level `L'nT` and
  the standardized facade level difference `D2m,nT`, each with its ISO 717
  weighted single number and the survey-method statement; `FacadeInsulationResult`
  (ISO 16283-3:2016 field facade, one-third-octave bands) renders `D2m,nT`
  (default), `D2m,n` or `R'45` with the ISO 717 weighted rating of the reported
  quantity (`D2m,nT,w`, `D2m,n,w` or `R'45,w`, each with `C; Ctr`) and the
  engineering-method statement. Each sheet carries the standard-basis line, an
  optional metadata header, the per-band table beside the measured-versus-
  shifted-ISO 717-reference curve, the boxed single number, an optional
  requirement verdict (level differences and reduction indices pass at or above
  the target, the impact level at or below it) and a footer; `verbose=True`
  annexes the ISO 717 evaluation per band, and both English and Spanish render.
- `OutdoorAttenuation.report()`: a one-page PDF outdoor sound propagation
  **prediction** fiche (ISO 9613-2:1996, general method of calculation). The
  sheet is clearly labelled a prediction, not a measurement. It carries a
  prediction-basis line naming ISO 9613-2:1996 and the conditions favourable to
  propagation, an optional metadata header (source/situation, client, receiver
  position, meteorological conditions, date, with the propagation distance
  recovered from the divergence term), a per-band table of the divergence,
  atmospheric, ground and barrier attenuation terms with the total attenuation
  `A`, the attenuation-breakdown plot, and a boxed octave-band range of the
  total attenuation. Passing a new `SourceEmission` (the source sound power
  level `Lw` and directivity, plus an optional meteorological correction) to
  `report(..., source_emission=...)` adds the source power and the downwind
  level `LfT(DW)` to the table and boxes the A-weighted downwind level
  `LAT(DW)` at the receiver instead; a declared limit level supplied via the
  metadata `requirement` then adds a PASS/FAIL verdict (a lower level is
  better), and `verbose=True` adds the A-weighted band-level column.
  `language="es"` renders the Spanish fiche.
- `BarrierInsertionLoss.report()`: a one-page PDF barrier insertion-loss
  **prediction** fiche. The sheet is clearly labelled a prediction, not a
  measurement, and its basis line names the actual diffraction model used (the
  wave-theoretic rigid-screen solution or the Kurze-Anderson closed form), a
  wave-acoustics complement to the tabulated ISO 9613-2 screening term. It
  carries an optional metadata header (source/situation, client, receiver
  position, ground model, date), a per-band table of the insertion loss `IL`,
  the insertion-loss spectrum plot and a boxed mean insertion loss over the
  octave bands. A declared minimum required insertion loss supplied via the
  metadata `requirement` adds a PASS/FAIL verdict (a higher insertion loss is
  better). `verbose=True` adds the Fresnel-number column. `language="es"`
  renders the Spanish fiche.
- `NCResult.report()` and `RCResult.report()`: one-page PDF room-noise
  assessment fiches for the two ANSI/ASA S12.2-2019 ratings. Each sheet carries
  the standard-basis line, an optional metadata header (client, room,
  description, room volume, floor area, instrumentation, climate, date), the
  measured octave-band levels beside the measured spectrum plotted against the
  NC/RC curve family (the result's own `plot`), and a boxed rating: `NC-nn`
  with its governing octave band (tangency method, Table 1) or `RC-nn(tag)`
  with the mid-frequency average `LMF` and the neutral/rumble/hiss spectral
  quality (Annex D). A target rating on the metadata `requirement` adds a
  PASS/FAIL verdict, where a lower rating passes. With `verbose=True` the table
  gains the per-band NC contour value read by the tangency method, or the
  reference RC Mark II curve and the measured deviation from it. The fiches
  render in English and Spanish (`language="es"`).
- `STIResult.report()`: a one-page PDF speech-transmission-index fiche
  (IEC 60268-16:2020) for verifying voice-alarm and public-address
  intelligibility. The sheet carries a standard-basis line stating the
  measurement method (the full STI indirect method from an impulse response, or
  the direct STIPA method on a recorded signal), an optional metadata header
  (client, system, manufacturer, test room, date), a per-octave-band table of
  the modulation transfer index `MTI` beside the per-band MTI bars (the
  result's own plot), and a boxed `STI = X` single number with the Annex F
  qualification band. A minimum STI supplied via the metadata `requirement`
  adds a PASS/FAIL verdict (a higher STI passes). `language="es"` renders the
  Spanish fiche.
- `SIIResult.report()`: a one-page PDF speech-intelligibility-index fiche
  (ANSI S3.5-1997, one-third-octave-band method) for a speech-audibility
  assessment. The sheet carries a standard-basis line, an optional metadata
  header (client, listening condition, test room, date), a
  per-one-third-octave-band table of the equivalent speech spectrum `Ei'`, the
  Table 3 band-importance function `Ii` and the band-audibility function `Ai`
  beside the audibility and importance-weighted contribution bars (the result's
  own plot), and a boxed `SII = X` single number. A minimum SII supplied via
  the metadata `requirement` adds a PASS/FAIL verdict (a higher SII passes).
  `verbose=True` adds the equivalent disturbance spectrum level `Di` column;
  `language="es"` renders the Spanish fiche.
- Noise-control performance reports via `.report()` on the three
  `noise_control` result types, each a one-page PDF laid out with a per-band
  table beside the result's own plot, a boxed single-number performance figure
  and an optional PASS/FAIL verdict:
  - `EnclosureResult.report()`: a machine-enclosure insertion-loss fiche (Bies,
    Hansen & Howard, Engineering Noise Control 5th ed., section 7.4.2). The
    per-band table lists the supplied panel transmission loss `R`, the interior
    build-up correction `C` and the net insertion loss `IL = R - C`; the boxed
    figure is the mean insertion loss over the analysis bands with the external
    and internal surface areas. A declared minimum via the metadata
    `requirement` passes when the mean insertion loss meets it (more is
    better); `verbose=True` adds the interior room constant `R_i` column.
  - `ReactiveSilencerResult.report()`: a reactive-silencer transmission-loss
    fiche (Munjal, Acoustics of Ducts and Mufflers 2nd ed., Eq. (3.27); Bies,
    Hansen & Howard, sections 8.8-8.9). The per-band table lists the
    transmission loss `TL` and, when end impedances were supplied, the
    insertion loss `IL`; the boxed figure is the mean transmission loss with
    the peak transmission loss and the device kind. A declared minimum passes
    when the mean transmission loss meets it (more is better).
  - `HvacSpectrumResult.report()`: an HVAC duct-noise-spectrum fiche (Bies,
    Hansen & Howard, Chapter 8; VDI 2081-1). A regenerated-noise spectrum boxes
    the A-weighted sound power level `L_WA` (dB(A) re 1 pW) with the overall
    unweighted total, and a declared limit passes at or below it (lower is
    better); an attenuation spectrum boxes the mean attenuation, and a declared
    minimum passes at or above it. `verbose=True` adds the A-weighting
    correction and A-weighted band-level columns for a regenerated-noise
    spectrum.

  All three accept an optional metadata header (client, equipment, test
  environment, instrumentation, climate, date), state their method basis, and
  render in English or Spanish (`language="es"`).
- `StructureBornePowerResult.report()`: a one-page PDF structure-borne sound
  power characterization fiche for the power a piece of building service
  equipment injects into a reception plate (EN 15657:2018 reception-plate
  method, Formula 14), rendered through the shared sound-power report engine.
  The sheet carries the standard-basis line naming the reception-plate method,
  an optional metadata header (client, source equipment, test environment,
  instrumentation, climate, date), a per-band table of the spatial mean plate
  velocity level `Lv` and the injected structure-borne sound power level
  `L_Ws`, the `L_Ws(f)` spectrum with a nominal band axis, and a boxed
  band-summed total `L_Ws` (dB re 1 pW) with the plate mass per area `m` and
  area `S`. A declared upper limit supplied via the metadata `requirement`
  adds a PASS/FAIL verdict (lower is better). `verbose=True` adds the plate
  loss factor `eta` column; the basis strip states Formula 14 and the
  conversion to the plate-independent source quantities (Formulae 15/17)
  required before EN 12354-5. `language="es"` renders the Spanish fiche.
- `InstalledSourceResult.report()`: a one-page PDF installed structure-borne
  sound **prediction** fiche (EN 12354-5:2009): the normalised structure-borne
  sound pressure level `L_n,s` estimated in a receiving room from building
  service equipment. The sheet is clearly labelled a prediction from the
  characteristic source power and the transmission paths, not a measurement.
  It carries a prediction-basis line naming EN 12354-5:2009, an optional
  metadata header (client, source equipment, receiving room, instrumentation,
  climate, date), a per-band table of the installed structure-borne power
  level `L_Ws,inst`, each transmission path's normalised SPL `L_n,s,ij` and the
  combined total `L_n,s`, the per-path and total `L_n,s(f)` spectra, and a
  boxed band-summed total `L_n,s` (dB) with the installed power total and the
  path count. A declared upper limit supplied via the metadata `requirement`
  adds a PASS/FAIL verdict (lower is better). `verbose=True` adds one column
  per transmission path (up to five); the basis strip states Formulae 18a/17
  and the prediction disclaimer. `language="es"` renders the Spanish fiche.
  Both fiches reuse the shared sound-power report body (per-band table,
  spectrum, boxed result and flow assembly) already used by the
  ISO 3744/3745/3741/9614/7849 fiches.
- `.report()` on the three ISO 10848 laboratory flanking-transmission results,
  each writing a one-page PDF fiche:
  - `VibrationReductionResult.report()`: a junction-characterization report of
    the vibration reduction index `Kij` (ISO 10848-1:2006). The sheet carries
    the standard-basis line, an optional metadata header, the per-band `Kij`
    table beside the `Kij(f)` curve, and a boxed single-number mean `Kij` over
    the Annex A band range (200 Hz to 1250 Hz for one-third-octave bands,
    125 Hz to 1000 Hz for octave bands) with the count of averaged and
    bracketed bands. Bands bracketed for poor modal overlap (`M < 0,25`,
    ISO 10848-4:2010 Clause 9) are printed in brackets and excluded from the
    mean. `verbose=True` adds a column stating whether each band enters the
    mean.
  - `FlankingLevelDifferenceResult.report()`: an airborne measurement report of
    the normalized flanking level difference `Dn,f` (ISO 10848-2:2006), with
    the per-band `Dn,f` table beside the measured-versus-shifted-reference curve
    and the boxed single number `Dn,f,w (C; Ctr)` rated per ISO 717-1.
  - `FlankingImpactLevelResult.report()`: an impact measurement report of the
    normalized flanking impact level `Ln,f` (ISO 10848-2:2006, tapping
    machine), with the per-band `Ln,f` table beside the reference curve and the
    boxed single number `Ln,f,w (CI)` rated per ISO 717-2.

  The two overall descriptors reuse the shared two-panel insulation report
  skeleton; `verbose=True` shows the ISO 717 evaluation per band (the value,
  the shifted reference and the unfavourable deviation). A `requirement`
  supplied via the metadata adds a PASS/FAIL verdict (higher is better for
  `Dn,f,w`, lower for `Ln,f,w`). `language="es"` renders every fiche in Spanish.
- `VibrationSoundPowerResult.report()`: a one-page PDF
  sound-power-from-vibration determination fiche for the airborne sound power a
  machine radiates through its surface vibration (ISO/TS 7849-1/-2:2009),
  rendered through the shared sound-power report engine. The sheet carries the
  standard-basis line naming the applied method (the ISO/TS 7849-1 survey
  method with a fixed radiation factor `ε = 1`, or the ISO/TS 7849-2
  engineering method with a determined radiation factor), an optional metadata
  header (client, machine/source, test environment, instrumentation, climate,
  date), a per-band table of the surface vibratory velocity level `Lv` and the
  radiated band sound-power level `LW`, the sound-power spectrum `LW(f)` with a
  nominal band axis, and a boxed A-weighted sound power level `LWA` (dB re
  1 pW) with the total `LW`, the radiating area `S` and the applied method. A
  declared A-weighted sound-power limit supplied via the metadata `requirement`
  adds a PASS/FAIL verdict (lower is better). `verbose=True` adds the radiation
  factor `ε` column; the basis strip states the sound-power relation
  `LW = Lv + 10 lg(S/S0) + 10 lg(ε) + 10 lg(411/400)` with its fixed impedance
  term and the radiation-factor model. `language="es"` renders the Spanish
  fiche. `VibrationSoundPowerResult` gains a `sound_power_level_a` property (the
  A-weighted total). The fiche reuses the shared sound-power body (per-band
  table, `LW(f)` spectrum, boxed `LWA` and flow assembly) already used by the
  ISO 3744/3745/3741/9614 fiches.
- `FacadePredictionResult.report()`: a one-page PDF **prediction** report for
  the predicted façade sound insulation (EN/ISO 12354-3 standardized level
  difference of a façade `D2m,nT`), computed by energetically combining the
  envelope elements' transmission factors and the room geometry (Formula 13).
  The sheet is clearly labelled a prediction, never a measurement: a
  standard-basis line naming EN/ISO 12354-3 and the ISO 717-1 rating, an
  optional metadata header (façade element set, exposed area, receiving-room
  volume, traffic/outdoor situation, calculator/laboratory, date), a two-panel
  body with the façade-element table on the left (each element's weighted
  partial index `Rp,w`) beside the result's own per-element / `R'` / `D2m,nT`
  plot, the boxed predicted single number `D2m,nT,w` (with `R'tr,s,w` and
  `Ctr`) and a statement reporting the model's ~2 dB standard deviation. A
  supplied requirement adds a PASS/FAIL verdict (the level difference passes at
  or above it); `verbose=True` annexes each element's share of the transmitted
  sound energy, which singles out the limiting element; `language="es"` renders
  the Spanish fiche. The fiche reuses the shared EN/ISO 12354 prediction report
  layout (`phonometry[report]` extra).
- `ReverberationSoundPowerResult.report()`: a one-page PDF reverberation-room
  sound-power determination fiche for the precision method in a qualified
  hard-walled reverberation test room (ISO 3741:2010, accuracy grade 1),
  rendered through the shared sound-power report engine. The sheet carries the
  standard-basis line naming the reverberation-room method (the direct method
  using the room equivalent absorption area, Eq. 20, or the comparison method
  using a reference sound source, Eq. 21) and the precision accuracy grade, an
  optional metadata header (client, noise source, reverberation test room,
  instrumentation, climate, date), a per-band table of the mean room
  sound-pressure level `Lp` and the band sound-power level `LW`, the
  sound-power spectrum `LW(f)` with a nominal band axis, and a boxed A-weighted
  sound power level `LWA` (dB re 1 pW) with the total `LW` and the
  determination method. A declared A-weighted sound-power limit supplied via
  the metadata `requirement` adds a PASS/FAIL verdict (lower is better).
  `verbose=True` adds the background correction `K1` and, for the direct
  method, the equivalent absorption area `A` and the Waterhouse boundary
  correction `Cw`; the basis strip states the correction model (Eq. 20 or
  Eq. 21), the applied meteorological corrections `C1`/`C2` and the speed of
  sound, and cites the Annex F A-weighting. `language="es"` renders the Spanish
  fiche. The fiche reuses the shared sound-power body (per-band table, `LW(f)`
  spectrum, boxed `LWA` and flow assembly) already used by the ISO 3744/3745
  pressure fiche and the ISO 9614-2 intensity fiche.
- `AirbornePredictionResult.report()` and `ImpactPredictionResult.report()`: a
  one-page PDF **prediction** report for the predicted building sound insulation
  between rooms (EN/ISO 12354-1 apparent sound reduction index `R'`) and of
  floors (EN/ISO 12354-2 apparent normalized impact sound pressure level
  `L'n`), computed by the simplified single-number model (direct plus flanking
  transmission). Each sheet is clearly labelled a prediction, never a
  measurement: a standard-basis line naming EN/ISO 12354-1/-2 and the ISO 717
  rating part, an optional metadata header (separating element, area, room
  geometry, bare-floor mass, calculator/laboratory, date), a two-panel body with
  the model-term table on the left (the direct path and each flanking path's
  weighted index `Rij,w` for airborne, or the Formula (21) terms `Ln,w,eq`,
  `ΔLw` and `K` for impact) beside the result's own plot, the boxed predicted
  single number (`R'w` / `L'n,w`) and a statement reporting the model's ~2 dB
  standard deviation. A supplied requirement adds a PASS/FAIL verdict (airborne
  passes at or above it, impact at or below it); `verbose=True` annexes each
  airborne path's share of the transmitted energy; `language="es"` renders the
  Spanish fiche. The fiche reuses the shared report layout toolkit
  (`phonometry[report]` extra).
- `SoundPowerIntensityResult.report()`: a one-page PDF sound-power-by-intensity
  determination fiche for the sound-intensity scanning method
  (ISO 9614-2:1996, engineering grade 2 or survey grade 3), rendered through
  the shared sound-power report engine. The sheet carries the standard-basis
  line naming the scanning method and its measurement grade, an optional
  metadata header (client, noise source, test environment, instrumentation,
  climate, date), a per-band table of the intensity-derived band sound-power
  level `LW` (net-negative, non-determinable bands shown as an em dash), the
  sound-power spectrum `LW(f)` with a nominal band axis, and a boxed A-weighted
  sound power level `LWA` (dB re 1 pW) with the total `LW`, the measurement
  surface area `S` and the determination grade. A declared A-weighted
  sound-power limit supplied via the metadata `requirement` adds a PASS/FAIL
  verdict (lower is better). `verbose=True` adds the field indicators `FpI`
  (surface pressure-intensity) and `F+/-` (negative partial power) and the
  per-band achieved grade; the basis strip states the partial-power model and
  the Annex B qualification criteria. `language="es"` renders the Spanish
  fiche. The shared sound-power body (per-band table, `LW(f)` spectrum, boxed
  `LWA` and flow assembly) is now factored into a common renderer used by both
  the ISO 3744/3745 pressure fiche and this ISO 9614-2 intensity fiche.
- `DynamicStiffnessResult.report()`: a one-page PDF dynamic-stiffness
  test-report fiche rendered through the shared accredited-report engine
  (EN 29052-1:1992, identical to ISO 9052-1:1989, the load-plate resonance
  method for the dynamic stiffness of resilient materials under floating
  floors). The sheet carries the standard-basis line, an optional metadata
  header (client, specimen, the total mass per unit area `m't` used during the
  test, the loaded specimen thickness `d`, test facility, climate), a metrics
  table (the resonant frequency `fr`, the apparent dynamic stiffness `s't` of
  Formula 4, the enclosed-gas term `s'a` of Formula 7 when it applies, the
  installed dynamic stiffness `s'` of Clause 8.2 and the supported-floor natural
  frequency `f0` of Formula 2) beside the `f0(s')` design curve, and a boxed
  apparent dynamic stiffness `s't` with the installed `s'` and the resonance
  `fr` alongside. Clause 9 rounds every dynamic stiffness to the nearest MN/m³;
  it is a characterisation, so there is no pass/fail verdict. `language="es"`
  renders the Spanish fiche. `ReportMetadata` gains a `thickness` field (the
  loaded specimen thickness, EN 29052-1:1992 Clause 9 b).
- `ImpulseProminenceResult.report()`: a one-page PDF impulsive-sound prominence
  assessment fiche rendered through the shared accredited-report engine
  (NT ACOU 112:2002, carried into ISO/PAS 1996-3:2022). The sheet carries the
  standard-basis line, an optional metadata header (source/situation, client,
  measurement position, instrumentation, date, with the 30-minute assessment
  period always shown), a per-impulse table of the onset rate, level difference,
  predicted prominence `P` and the onset-rate qualification beside the `KI(P)`
  adjustment-curve plot with the candidate impulses marked, and the boxed
  governing prominence `P` together with the derived `LAeq` adjustment `KI`
  (Formula 2), with a prominence-category note (prominent / not). A maximum
  acceptable governing prominence supplied via the metadata `requirement` adds a
  PASS/FAIL verdict (a less prominent impulse passes); `language="es"` renders
  the Spanish fiche.
- `IntensityReductionResult.report()`: a one-page PDF intensity
  sound-insulation test-report fiche rendered through the shared
  accredited-report engine (ISO 15186-1:2000, laboratory measurement of sound
  insulation using sound intensity). The sheet carries the standard-basis line,
  an optional metadata header (client, specimen, sample area `S`, room volumes,
  test room, climate), the one-third-octave table of the intensity sound
  reduction index `RI` beside the measured-versus-shifted-reference curve, and
  the boxed single-number rating `RI,w (C; Ctr)` (evaluated per ISO 717-1:2020),
  followed by the statement that the transmitted sound power was measured
  directly over the measurement surface. A minimum rating supplied via the
  metadata `requirement` adds a PASS/FAIL verdict (a higher rating passes).
  `verbose=True` annexes the `Kc`-modified index `RI,M` beside `RI` when an
  Annex B adaptation term is present; `language="es"` renders the Spanish fiche.
- `FloorCoveringImprovementResult.report()`: a one-page PDF floor-covering
  impact-improvement test-report fiche rendered through the shared
  accredited-report engine (BS EN ISO 16251-1:2014, the small-mock-up
  laboratory method for the reduction of transmitted impact sound by soft floor
  coverings). The sheet carries the standard-basis line, an optional metadata
  header (client, floor-covering description, mounting, mass per unit area, the
  measured frequency range, climate), the one-third-octave table of the
  reduction of impact sound pressure level `ΔL` (bands at the 1,3 dB limit of
  measurement prefixed `>`) beside the `ΔL(f)` improvement curve, and the boxed
  single-number weighted improvement `ΔLw (CI,Δ)` (the ISO 16251-1 Clause 8 e)
  statement of results, rated per ISO 717-2:2020; a characterisation headline
  replaces it when the spectrum lacks the 16 rating bands 100 Hz to 3150 Hz). A
  minimum weighted improvement supplied via the metadata `requirement` adds a
  PASS/FAIL verdict (a higher weighted improvement passes). `verbose=True` adds
  the reference-floor-with-covering column `Ln,r = Ln,r,0 - ΔL` (the ISO 717-2
  derivation basis of `ΔLw`); `language="es"` renders the Spanish fiche.
- `ScatteringResult.report()`, `DiffusionSpectrum.report()` and
  `DiffusionResult.report()`: one-page PDF surface-scattering and diffusion
  test-report fiches rendered through the shared accredited-report engine
  (ISO 17497-1:2004+A1:2014 and ISO 17497-2:2012). The scattering fiche carries
  the standard-basis line, an optional metadata header (client, specimen,
  sample area `S`, room volume `V`, test room, climate), a per-one-third-octave
  table of the random-incidence absorption `alpha_s` and the scattering
  coefficient `s` beside the `s(f)` curve on a categorical band axis, and a
  boxed characterisation headline; `verbose=True` adds the specular absorption
  `alpha_spec` column. The diffusion-spectrum fiche tabulates the diffusion
  coefficient `d` per band beside the `d(f)` band-axis curve (the per-band
  random-incidence coefficient being the average of the directional coefficients
  over the source positions, Clause 8.4); `verbose=True` adds the normalised
  `d_n` column. The polar-response fiche tabulates the corrected reflected level
  `L` per receiver angle beside the semicircular polar plot, boxing the
  directional diffusion coefficient. ISO 17497 is a characterisation, so there
  is no pass/fail verdict. `language="es"` renders the Spanish fiche. A new
  `DiffusionSpectrum` result (and its `diffusion_spectrum` constructor) carries
  the diffusion coefficient across the measured bands, with `.plot()` and
  `.report()`.
- `ToneAudibilityResult.report()`: a one-page PDF tonal audibility assessment
  fiche rendered through the shared accredited-report engine, following the
  ISO 1996-2:2017 Annex J engineering method (ISO/PAS 20065:2016). The sheet
  carries the standard-basis line, an optional metadata header (source/situation,
  client, measurement position, instrumentation and date, with the analysis line
  spacing `Δf` read from the result), a full-width table of the key quantities for
  every detected tone (tone frequency `fT`, entry type, tone level `Lpt`,
  critical-band masking-noise level `Lpn`, critical bandwidth `Δfc` and the
  audibility `ΔLta`) above a level-versus-frequency analysis plot with the tones
  and their critical-band masking noise marked, and a boxed decisive audibility
  `ΔLta` with the derived tonal adjustment `K` (ISO 1996-2:2017 Table J.1). A
  maximum-audibility limit supplied via the metadata `requirement` adds a
  PASS/FAIL verdict (a quieter tone passes), and a prominence note states whether
  a prominent tone is present. `verbose=True` adds the per-tone extended
  uncertainty column; `language="es"` renders the Spanish fiche.
- `ImpedanceTubeResult.report()`: a one-page PDF impedance-tube test-report
  fiche rendered through the shared accredited-report engine (BS EN ISO
  10534-2:2001, two-microphone transfer-function method). The sheet carries the
  standard-basis line, an optional metadata header (client, specimen, tube
  diameter `d`, microphone spacing `s`, the measured frequency range, mounting,
  climate), a per-frequency table of the normal-incidence absorption
  coefficient `alpha` with the real and imaginary parts of the normalised
  surface impedance `z = Z/(rho c0)` beside the `alpha(f)` curve on a continuous
  logarithmic frequency axis, and a boxed characterisation headline (ISO 10534-2
  is a characterisation, so there is no pass/fail verdict and no single-number
  rating). `verbose=True` adds the reflection-factor magnitude `|r|` column, and
  `language="es"` renders the Spanish fiche. `ReportMetadata` gains the
  `tube_diameter` and `mic_spacing` geometry fields (in metres, printed in
  millimetres). The `ImpedanceTubeResult.plot()` absorption spectrum now uses a
  continuous logarithmic frequency axis with band-centre labels.
- `SoundPowerResult.report()` and `PrecisionSoundPowerResult.report()`: a
  one-page PDF sound-power determination fiche rendered through the shared
  accredited-report engine. The sheet carries the standard-basis line naming
  the applied method and accuracy grade (ISO 3744:2010 engineering grade 2 /
  ISO 3746:2010 survey grade 3 / ISO 3745:2012 precision grade 1), an optional
  metadata header, a per-band table (nominal octave/one-third-octave frequency,
  the surface sound-pressure level `Lp` and the band sound-power level `LW`,
  grouped by octave for a one-third-octave set), the sound-power spectrum
  `LW(f)` with a nominal band axis, and a boxed A-weighted sound power level
  `LWA` (dB re 1 pW) with the total `LW`, the expanded uncertainty `U` and the
  measurement surface area `S`. A declared limit supplied via the metadata
  `requirement` adds a PASS/FAIL verdict (lower is better). `verbose=True` adds
  the energy-averaged level `Lp'`, and for the ISO 3744/3746 surface result it
  also adds the background (`K1`) and environmental (`K2`) correction columns;
  the ISO 3745 precision result carries no `K1`/`K2`, so its basis strip states
  the meteorological corrections (`C1`/`C2`/`C3`) instead. `language="es"`
  renders the Spanish fiche.
- `LabAirborneInsulationResult.report()` and `LabImpactInsulationResult.report()`:
  the one-page laboratory sound-insulation test report of ISO 10140-2:2010
  (airborne sound reduction index `R`) and ISO 10140-3:2010 (normalized impact
  sound pressure level `Ln`), rendered to PDF and laid out like the accredited
  laboratory reports rated per ISO 717. The sheet carries the standard-basis
  line, an optional metadata header (client, specimen, mounting, sample area,
  room volumes, climatic conditions), the one-third-octave (or octave) table
  beside the measured-versus-shifted-reference curve, the boxed laboratory
  rating `Rw (C; Ctr)` (ISO 717-1) or `Ln,w (CI)` (ISO 717-2) evaluated by the
  existing weighted-rating engine, the statement that the evaluation rests on a
  laboratory precision method, and a footer with the fixed disclaimer.
  `verbose=True` annexes the per-band equivalent absorption area `A = 0,16 V/T`
  (ISO 10140-4:2010) beside the reported quantity. A verdict row appears only
  when a target is supplied through `metadata.requirement` (airborne passes at
  or above it, impact at or below it). `language="es"` renders the Spanish
  fiche with a comma decimal separator.
- `standard_speech_spectra()` and the `StandardSpeechSpectrum` result: a thin,
  plottable wrapper around `standard_speech_spectrum()` (ANSI S3.5-1997 Table 3)
  that carries the 18 one-third-octave band centre frequencies and the standard
  speech spectrum level of one or more vocal efforts (`normal`, `raised`,
  `loud`, `shout`). Its `.plot()` draws the standard speech spectrum level (dB
  SPL) over the categorical one-third-octave band axis, one labelled line per
  vocal effort, so the whole spectrum lifting with vocal effort reads at a
  glance. The band levels are unchanged; `standard_speech_spectrum()` still
  returns the bare per-band array. `language="es"` renders the Spanish labels.
- `seabed_reflection()` and the `SeabedReflection` result: a thin, plottable
  wrapper around `reflection_coefficient()` (fluid-fluid Rayleigh seabed) that
  carries the grazing-angle grid, the complex reflection coefficient, its
  magnitude `|R|`, the bottom loss in dB, the critical angle and the interface
  parameters (`rho1`, `c1`, `rho2`, `c2`). Its `.plot()` draws the
  reflection-coefficient magnitude `|R|` on a linear grazing-angle axis, marking
  the critical angle; `language="es"` renders the Spanish labels. The underlying
  `reflection_coefficient()` maths is unchanged.
- `k_weighting_response()` and `KWeightingResponse`: the K-weighting pre-filter
  of ITU-R BS.1770-5 (Annex 1) evaluated as a magnitude frequency response. The
  constructor evaluates the same Table 1-2 biquads that `k_weighting` applies
  (from `k_weighting_coefficients`, so nothing is re-derived) and returns a
  frozen result carrying the combined magnitude in dB, each of the two stages
  (the +4 dB spherical-head shelf and the RLB high-pass) and the frequency
  grid. Its `.plot()` draws the response on a logarithmic frequency axis, the
  combined curve with the two stages as light companions; `language="es"`
  renders the Spanish labels.
- `piston_directivity_pattern()` and the `PistonDirectivity` result: the
  far-field directivity of a rigid baffled circular piston
  (`D(theta) = 2 J1(ka sin theta)/(ka sin theta)`, Beranek & Mellow Eq. (4.42))
  is now a plottable result carrying the polar-angle grid, the linear
  directivity and its dB form and the `ka` value(s). Its `.plot()` draws the
  classic polar beam pattern in dB over the front hemisphere; passing several
  `ka` shows them as one family, so the main lobe narrowing and the emergence
  of the side lobes read at a glance. `language="es"` renders the Spanish
  labels. The underlying `piston_directivity()` maths is unchanged.
- `atmospheric_attenuation()` and the `AtmosphericAttenuation` result: a thin,
  plottable wrapper around `air_attenuation()` (ISO 9613-1:1993) that carries the
  frequency grid, the pure-tone attenuation coefficient `alpha` and the
  atmospheric conditions (temperature, relative humidity, pressure). Its
  `.plot()` draws the classic `alpha`-versus-frequency curve in dB/km on a
  logarithmic frequency axis, and an optional `distance` exposes the total
  attenuation `A = alpha * d` over the path (`total_attenuation`, the ISO 9613-2
  `Aatm`). The maths is unchanged; `air_attenuation()` still returns the bare
  coefficient array. `language="es"` renders the Spanish labels.
- `TrendTestResult.plot()`: a one-line canonical figure for the nonparametric
  trend test (Bendat & Piersol, *Random Data* Sec. 4.5.2), mirroring the
  companion `StationarityTestResult.plot()`. It draws the tested sequence
  against its sample index and states the outcome in the legend: the
  reverse-arrangement count `A` (or the run count `r`), the Table A.6
  acceptance region and whether the no-trend hypothesis is accepted; for
  `method="runs"` it also marks the sequence median that classifies each
  value. `language="es"` renders the Spanish labels.
- `RigidMassCalibrationResult.plot()`: the ISO 7626-2 (7.5.2) operational
  rigid-mass calibration check as a single-concept, two-panel figure. The upper
  panel draws the measured driving-point FRF magnitude against the known
  rigid-mass line (`|A| = 1/m` for accelerance, `|Y| = 1/(2πf·m)` for mobility)
  with its ±5 % tolerance band; the lower panel draws the relative deviation
  against the same band, where a few-percent tolerance is actually readable.
  Frequencies outside the band are drawn in the out-of-tolerance colour and the
  title carries the overall verdict. Passing an existing axes draws only the
  deviation diagnostic on it; `language="es"` renders Spanish labels.
- `MultipleShockResult.report()`: a one-page PDF whole-body multiple-shock
  health-risk assessment fiche (ISO 2631-5:2018), laid out like a health-risk
  assessment sheet. The sheet carries the standard-basis line (Clause 5 spinal
  response and the Annex C risk model), an optional metadata header (client,
  subject, workplace/vehicle, instrumentation, calibration), the
  exposure-scenario grid (subject sex, the age `b` at which the exposure
  started, the number of exposure years `n`, the number of exposure days per
  year `N` and the number of counted response shocks), the dose-and-stress
  analysis table (the acceleration dose `Dz` of Formula 3, the daily dose `Dzd`
  of Formula 4, the daily compressive stress `Sd` of Formula C.1, the
  cumulative stress variable `R` of Formula C.3 and the probability of lumbar
  injury `P` of Formula C.5), the result's own injury-probability chart, the
  boxed `R` and `P` with the Annex C risk classification, and a classification
  table against the Table C.2 stress variables for 10 / 50 / 90 % risk of
  injury (low / moderate / high / very high probability of an adverse health
  effect) with a risk-band zone row. Because ISO 2631-5:2018 defines no
  exposure limit, the fiche carries the informative risk-band classification
  rather than a PASS/FAIL verdict. `language="es"` renders the Spanish fiche
  with a comma decimal separator.
- `equal_loudness_contours()` and the `EqualLoudnessContours` result type for
  the ISO 226:2023 normal equal-loudness-level contours. The function bundles a
  family of contours (20 phon to 90 phon in 10 phon steps by default) with the
  threshold of hearing over a shared frequency grid, reusing the existing
  `equal_loudness_contour()` and `hearing_threshold()` for all maths. The
  result exposes `.plot()`, which draws the iconic ISO 226 chart (sound
  pressure level versus a logarithmic frequency axis, one line per loudness
  level plus the threshold) in English or Spanish. The bare-array functions are
  unchanged.
- `LoudspeakerCharacteristics.plot(quantity=...)`: single-concept IEC 60268-5
  rated-characteristic figures selected by `quantity`: `"response"` (the on-axis
  SPL response with its tolerance band and effective-range markers, the
  default), `"impedance"` (the modulus `|Z|` with the rated and 80 %-of-rated
  lines), `"thd"` (total harmonic distortion against frequency) and
  `"directivity"` (the polar response on the 25 dB reference circle). Each call
  returns a single Axes and shares its panel drawing with the `.report()` fiche,
  so the plot and the report never diverge. An unknown `quantity` raises
  `ValueError`; `language="es"` labels the figure in Spanish.
- `MicrophoneCharacteristics.plot(quantity=...)`: single-concept IEC 60268-4
  rated-characteristic figures selected by `quantity`: `"response"` (the
  free-field response with its tolerance band, reference-frequency and
  effective-range markers, the default), `"directivity"` (the polar directional
  pattern on the 25 dB reference circle), `"noise"` (the inherent-noise band
  spectrum) and `"distortion"` (total harmonic distortion against sound pressure
  level). Each call returns a single Axes and shares its panel drawing with the
  `.report()` fiche. An unknown `quantity` raises `ValueError`; `language="es"`
  labels the figure in Spanish.
- `OpenPlanResult.report()`: a one-page PDF open-plan office acoustics fiche
  (ISO 3382-3:2012), laid out like a speech-privacy measurement report. The
  sheet carries the standard-basis line, an optional metadata header (client,
  office/zone, description, floor area, source and measurement positions,
  climate), a metrics table of the four single-number quantities of Clause 4
  (the spatial decay rate of A-weighted speech `D2,S`, the nominal 4 m speech
  level `Lp,A,S,4m`, the distraction distance `rD` and the privacy distance
  `rP`), the result's own spatial-decay plot (the Clause 6.2 regression on the
  logarithmic distance axis with the 4 m read-off and the `rD` / `rP`
  crossings), the boxed `D2,S` with the remaining quantities alongside, and a
  footer with the fixed disclaimer. A verdict row appears only when a target
  spatial decay rate is supplied through `metadata.requirement` (read as the
  minimum acceptable `D2,S`, per the informative quality ranges of Annex A; the
  room passes at or above it). `language="es"` renders the Spanish fiche with a
  comma decimal separator.
- `DailyVibrationExposure.report()`: a one-page PDF daily vibration exposure
  assessment fiche for hand-arm (ISO 5349-1/-2:2001) and whole-body
  (ISO 2631-1:1997) exposure, laid out like the hand-arm and whole-body
  exposure calculators of occupational-hygiene practice. The sheet carries the
  standard-basis line naming the applied ISO method and the assessment
  directive, an optional metadata header (company, operator/worker, workplace,
  instrumentation, calibration), the per-operation exposure analysis (each
  operation's vibration total value, its daily exposure time `T_i` and the
  partial exposure `A_i(8)` of ISO 5349-1/-2 Eq. (2), closed by the daily total
  and the combined `A(8)` of Eq. (3)) with the contribution chart, the boxed
  `A(8)` with its exposure zone, and an assessment table against the exposure
  action value (EAV) and exposure limit value (ELV) of Directive 2002/44/EC
  (Article 3): hand-arm 2.5 / 5 m/s², whole-body 0.5 / 1.15 m/s², each marked
  exceeded / not exceeded on the value rounded exactly as displayed, with a
  PASS/FAIL verdict against the limit value and a footer note that the ISO
  standards define no safe exposure limit. `verbose=True` adds each operation's
  share of the daily vibration energy, and `language="es"` renders the Spanish
  fiche (comma decimals). The committed example reproduces the ISO 5349-2:2001
  Annex E.3 forestry worker's day (A(8) = 3.6 m/s², action zone). Rendering
  needs the optional `phonometry[report]` extra (reportlab), plus matplotlib
  for the contribution chart.
- `time_synchronous_average`: extraction of a periodic waveform of known
  period from asynchronous noise by time domain averaging (P. D. McFadden,
  "A revised model for the extraction of periodic waveforms by time domain
  averaging", Mechanical Systems and Signal Processing 1(1), 1987, 83-95).
  Ensemble-averaging N successive periods (Eq. 5) reinforces every component
  synchronous with the period and suppresses the rest, with the residual
  asynchronous-noise standard deviation falling as 1/sqrt(N). The frozen
  `SynchronousAverageResult` carries the averaged one-period waveform, the
  residual and its RMS, the noise reduction in dB (10*log10(N)) with the
  sqrt(N) amplitude gain, the comb-filter response over the first harmonics,
  and a `.plot()` (EN/ES). When the sample rate does not divide the period
  the blocks are aligned to a common integer grid by the band-limited
  fractional delay of `fractional_delay`; an integer number of samples per
  period is recovered to machine precision. `comb_filter_response` evaluates
  the closed-form comb magnitude (Eq. 8/9), a Dirichlet kernel with unit
  teeth at the harmonics k/T and nodes at j/(N*T), so an interfering order is
  best rejected by choosing N to place a node on it (McFadden's 32.05-order
  example: N = 20 beats the power-of-two N = 32).
- `miso_coherence`: multiple and partial coherence of a multiple-input,
  single-output system (Bendat & Piersol, Random Data 4e, Chapter 7). From the
  Welch cross-spectral matrix of several partially correlated inputs and
  one output it reports the ordinary coherence of each input (Eq. 7.109), the
  multiple coherence explained by all inputs jointly (Eq. 7.35), and the
  partial coherences (Eq. 7.87) obtained by the Gaussian-elimination
  conditioning of Section 7.3, so a source that only correlates with the true
  cause is no longer credited for it. The frozen `MISOCoherenceResult` also
  carries the partial coherent output spectra that decompose the output power
  source by source (Eq. 7.86, with `sum(Gvi) + noise = Gyy`), the Section 9.3
  random errors, a `.dominant_input()` helper for the strongest source per
  band, and a `.plot()` (EN/ES). The conditioning is pinned to the exact
  rational values of Problem 7.2 and to the multiple-coherence SNR relation.
- `ExposureResult.report(path)` renders the ISO 9612:2009 Clause 15
  occupational noise-exposure measurement report as a one-page PDF fiche laid
  out like a prevention-service measurement sheet: the standard-basis line
  naming the applied strategy (task-based, job-based or full-day, Clauses
  9-11), a header grid with the new `ReportMetadata.instrumentation` and
  `ReportMetadata.calibration` free-text fields, the work-analysis table (per
  task the mean duration, sample count, task level of Formula 7 and
  contribution of Formula 8, or the Annex C sampling budget for the job-based
  and full-day strategies, with `verbose=True` adding the per-task uncertainty
  columns), the boxed LEX,8h with its expanded uncertainty U at the coverage
  factor k = 1.65, and the assessment against Directive 2003/10/EC (the 80/85
  dB(A) action values and the 87 dB(A) limit value, each marked on the value
  rounded exactly as displayed). English and Spanish. `ReportMetadata` gains
  the optional `instrumentation` and `calibration` fields.
- `metrology.equalizer`: parametric-equalizer biquads implementing the
  closed-form recipes of the RBJ Audio EQ Cookbook (republished as a W3C
  Working Group Note). `EQSection` specifies one biquad with the cookbook's
  exact parameterization (`peaking`, `lowshelf`, `highshelf`, `lowpass`,
  `highpass`, `bandpass`, `bandpass_skirt`, `notch`, `allpass`; `f0`,
  `gain_db`, and exactly one of `q`, `bw` in octaves or shelf `slope`), and
  `ParametricEQ` designs the sections and runs them as a normalized SOS cascade
  in the `WeightingFilter` house style (reusable `.sos`, 1D/2D multichannel
  `filter()`, optional stateful block processing). `ParametricEQ.response()`
  returns a frozen `EQResponseResult` with the per-section and cascade
  magnitude and phase and an i18n `.plot()`, and `parametric_eq(x, fs, sections)`
  is the one-shot helper. Verified against the cookbook's own closed forms (a
  hand-worked coefficient set pinned to 1e-15 and the exact type anchors at
  f0, DC and Nyquist).
- Three system-measurement front ends for the transfer-function toolbox, each
  returning a rich result with a `.plot()` and reusing the existing
  acquisition and deconvolution machinery. `golay_pair(order)` builds a
  complementary binary code pair of length `2**order` whose periodic
  autocorrelations sum to an exact `2L`-delta (Golay 1961; Xiang in Havelock,
  Handbook of Signal Processing in Acoustics, Part I Ch. 6), and
  `golay_impulse_response()` recovers the `ImpulseResponseResult`
  (`method="golay"`) exactly for a noiseless linear time-invariant system.
  `shaped_sweep_signal(...)` synthesizes a swept sine whose group delay grows
  in proportion to a target spectral power (`"pink"`, `"white"` or a
  `(frequencies_hz, magnitude_db)` pair) while keeping a swept sine's crest
  factor (Mueller & Massarani, JAES 49(6), 2001). `regularized_inverse_filter()`
  computes the frequency-dependent Tikhonov inverse of Kirkeby & Nelson
  (JAES 47(7/8), 1999), `H_inv = conj(H) / (|H|^2 + eps(f))`, small in-band and
  large outside, with the out-of-band gain capped at `1 / (2*sqrt(eps))` and a
  modeling delay that makes the mixed-phase inverse causal; `.apply()`
  equalizes a recording with the delay removed.
- `multitaper_psd`: Thomson multitaper spectral density (Thomson 1982;
  Percival & Walden 1993, Chapter 7) as the whole-record alternative to the
  Welch estimator for records too short to segment. K orthogonal Slepian
  (dpss) tapers - computed by `scipy.signal.windows.dpss`, their
  concentrations verified against Percival & Walden's quadruple-precision
  Table 382 - yield K nearly uncorrelated eigenspectra whose adaptively
  weighted average (P&W Eqs. 368a/370a, iterated to convergence) carries
  about 2K chi-square degrees of freedom from a single record. The frozen
  `MultitaperSpectralDensityResult` reports the per-frequency equivalent
  degrees of freedom (Eq. 370b), the chi-square confidence interval, the
  normalized random error, the combination weights and the taper
  concentrations, with `.plot()` (EN/ES) drawing the density and its
  confidence band; calibration (density/spectrum scaling, no detrending)
  matches the Welch estimators exactly.
- Random-data qualification module `phonometry.metrology.random_data`
  (Bendat & Piersol, Random Data 4e, Secs. 4.5.2, 10.3 and 5.5).
  `trend_test()` runs the nonparametric reverse arrangement test with the
  Table A.6 acceptance regions (reproduced exactly at alpha = 0.05 for
  N = 10 to 100, pinned in the conformance suite together with the book's
  Example 4.4) and an exact Mahonian p-value up to N = 100, or the classical
  runs test about the median with the exact Wald-Wolfowitz conditional
  distribution at any N. `stationarity_test()` applies either test to
  per-segment mean squares (or rms/mean/variance) of a single record, the
  Sec. 10.3.1.1 procedure, with a `.plot()` of the segment sequence and the
  verdict. `level_crossing_rate()` counts level crossings (both slopes) and
  compares them with the Rice expectations N0 = 2 sqrt(m2/m0) and
  Na = N0 exp(-a^2/2 sigma^2) from the record's own Welch autospectrum
  moments (Examples 5.12-5.14 verified as conformance rows), and
  `peak_statistics()` measures the rate of local maxima against
  M = sqrt(m4/m2), derives the irregularity factor r = N0/2M and exposes
  the Rice peak-height distribution (`peak_exceedance()`/`peak_density()`)
  that interpolates Rayleigh (r = 1) and Gaussian (r -> 0); both with
  EN/ES `.plot()` renderers. New "Data qualification" guide (EN/ES) under
  Calibration and uncertainty, linked from the GUM and levels guides, with
  three SVG figures; the API section "Uncertainty" is now "Uncertainty and
  data quality".
- B, D and AU frequency weightings in `WeightingFilter` / `weighting_filter`
  (`curve="B"|"D"|"AU"`), joining A/C/G/Z with the same 1 kHz normalization,
  `high_accuracy` oversampling, multichannel and stateful block processing.
  B follows ANSI S1.4-1983 Appendix C (the historical curve dropped from the
  IEC sound-level-meter standards); AU follows IEC 61012:1990 (the A
  weighting cascaded with the Table 2 U low-pass for measuring audible sound
  in the presence of ultrasound, designed toward an internal 288 kHz rate for
  the steep roll-off); D implements the withdrawn IEC 537:1976 aircraft-noise
  weighting from its published rational transfer function, cross-checked
  against SQAT (identical zeros/poles) and librosa (independent closed form,
  agreement within 0.002 dB) and pinned against the IEC 537 table republished
  in NASA CR-3406 (Table SLD-I).
- `verify_weighting_class` now also verifies `B` against ANSI S1.4-1983
  (Table IV design goals, Table V Type 1/2 tolerance masks) and `AU` against
  IEC 61012:1990 Table 1 (nominal A + nominal U with the separate-unit
  tolerances and the subclause 2.2 explicit AU values at 25/31.5/40 kHz),
  including the between-nominals sweep against the analytic design goals.
  The conformance report pins B against the strictest Type 0 mask, AU over
  the full 10 Hz-40 kHz Table 1 range at 96 kHz, and D against the published
  tabulated curve.
- `building.AirborneInsulationResult.report()` and
  `building.ImpactInsulationResult.report()` render the ISO 16283 field
  sound-insulation test report to a one-page PDF: the airborne fiche states
  the standardized level difference DnT (or, with `quantity="r_prime"`, the
  apparent sound reduction index R') per ISO 16283-1:2014 Clause 14 in the
  layout of the recommended Annex B results form, and the impact fiche the
  standardized L'nT (or normalized L'n) per ISO 16283-2:2020 Clause 14 /
  Annex C, each with the quantity tabulated to one decimal place beside the
  curve against the shifted reference (Clause 12), the boxed ISO 717-1 /
  ISO 717-2 field rating evaluated over the 16 core one-third-octave bands,
  the mandatory statement that the evaluation is based on field measurement
  results obtained by an engineering method, an optional requirement verdict
  and the shared metadata header/footer, localised in English and Spanish.
  `verbose=True` swaps the table for the per-band measurement chain (the
  energy-average room levels and the reverberation time beside the reported
  quantity); to support it, `airborne_insulation()` and `impact_insulation()`
  now retain their energy-averaged inputs on the results (`l1`, `l2`/`li`,
  `t2`, `t0`, all defaulting to `None` for backward-compatible construction).
- `metrology.spectrogram` and `metrology.SpectrogramResult`: a calibrated
  STFT power spectrogram (Bendat & Piersol, Random Data 4e, Section
  12.6.4.2) built on the exact segmentation and scaling of
  `power_spectral_density`, so a signal in pascals reads Pa²/Hz or Pa² per
  cell (a tone shows its mean square A²/2, i.e. its dB SPL, in every
  'spectrum' column), the column mean over time reproduces the Welch
  estimate bin by bin, and with a squared-COLA taper (Hann at 75 % overlap)
  the time-integrated 'density' power equals the record energy exactly. The
  result reports the time resolution T_B, the effective noise bandwidth Be,
  the hop, and the per-cell random error of the unaveraged estimate, and
  renders through `.plot()` (single-raster dB image, English and Spanish).
- `metrology.zoom_fft` and `metrology.ZoomFFTResult`: the zoom transform of
  Bendat & Piersol Section 11.5.4 (Eqs. 11.122-11.130) computed as its exact
  single-pass digital equivalent, the chirp-Z evaluation of the tapered
  record's DFT on the zoom grid, verified at machine precision against the
  book's demodulate-decimate-DFT chain. Amplitudes are calibrated per taper
  coherent gain (a sine of peak amplitude A on an analysis frequency reads
  amplitude A and power A²/2 exactly), and the result distinguishes the
  freely chosen `bin_spacing` from the record-and-taper-set
  `resolution_bandwidth`, with a `.plot()` renderer over the zoom band.
  Three conformance rows pin the tone calibration, the Parseval/COLA energy
  identity and the zoom demodulation chain; a new "Time-frequency analysis"
  guide (English and Spanish) documents both estimators.
- `metrology.tone_burst` and `metrology.ToneBurstResult` generate the gated
  sine burst of IEC 60268-1:1985 (Annex A, Clause A2): the tone starts at a
  zero crossing and lasts an integral number of full periods, as a single
  burst or as the repetitive train of Clause A2.2 with a stated repetition
  rate. The result carries the rectangular gating envelope and the exact
  sample bookkeeping (burst samples, onset, repetition period, duty cycle),
  a `.plot()` of the waveform with its envelope (EN/ES), and the closed-form
  gate energy `A^2 N/2` anchors the tests and a conformance row.
- `metrology.resample_signal` and `metrology.ResampledSignalResult` perform
  rational polyphase resampling behind an explicit anti-alias specification:
  the lowpass FIR is designed inside the function by the Kaiser window
  method from the stopband attenuation in dB (default 120) and the
  transition-band fraction of the smaller Nyquist frequency (default 0.05),
  and the designed taps travel with the result so the spec is verifiable
  against the filter itself. The tests measure the returned filter and
  assert the passband deviation and stopband leakage against the design's
  own ripple bound `10^(-A/20)`.
- `metrology.fractional_delay` delays a record by an arbitrary fractional
  number of samples via a frequency-domain phase ramp, with a `linear`
  boundary (zero-padded, bit-identical to the alignment kernel of
  `align_impulse_responses`, now shared) or a `circular` one (exact to
  machine precision on bin-centered tones: each component's phase changes
  by exactly `-2*pi*f*D/fs`).
- `metrology.window_metrics` and `metrology.WindowMetricsResult` compute the
  Harris (1978) figures of merit of any scipy window sampled DFT-even as the
  Welch estimators apply it: equivalent noise bandwidth (exactly 1, 3/2,
  1987/1458 and 1523/882 for rectangular/Hann/Hamming/Blackman), coherent
  gain, scalloping loss, worst-case processing loss, highest sidelobe level
  and the -3 dB main-lobe width, with a `.plot()` of the window and its
  spectrum (EN/ES) and an `enbw_hz()` bridge to the PSD estimator's
  resolution bandwidth.
- New "Test signals and sample-rate tools" guide (EN/ES) under Signals and
  spectra, a "Choosing the window" section in the spectral-analysis guide
  (EN/ES) with the window trade-off figure, and two documentation figures
  (`tone_burst_train`, `window_functions_tradeoff`).
- `metrology.cepstrum` brings cepstral analysis to the signal toolbox
  (Havelock, *Handbook of Signal Processing in Acoustics*, Chs. 27/87/75, and
  Bendat & Piersol Sec. 13.1.4): `cepstrum` computes the power, real or
  complex cepstrum over the quefrency axis (the complex kind unwraps the
  phase, removes and stores its linear component, and `CepstrumResult.invert`
  runs the homomorphic round trip back to the record); `lifter` splits a log
  spectrum into its smooth envelope and its fine structure with exactly
  complementary lowpass/highpass quefrency windows; and `echo_detection`
  reads a reflection's delay and coefficient off the power-cepstrum peak,
  whose height for an in-record echo is analytically the reflection
  coefficient itself (the n = 1 term of the ln(1 + a·e^(-j·theta)) series;
  the rahmonic amplitudes (-1)^(n+1)·a^n/n and the liftered-ripple extrema
  20·lg(1 ± a) dB are pinned as closed-form oracles in the tests and the
  conformance report). All three return frozen results with localised
  `.plot()` renderers. `minimum_phase` now folds its real cepstrum through
  the module's shared core; the refactor is verified bit-exact against the
  previous implementation on fixed responses.
- `metrology.envelope_spectrum` transforms the detected envelope itself,
  following the envelope-detector -> DC-remover chain of Bendat & Piersol
  Sec. 13.3 (Fig. 13.11) with either the Hilbert magnitude envelope or the
  book's square-law detector, scaled (window coherent-gain corrected) so a
  sinusoidal amplitude modulation reads out as a line at its exact
  amplitude: an AM tone (A0, m) puts A0·m at fm for the magnitude detector
  and 2·A0²·m at fm plus A0²·m²/2 at 2·fm for the squared one, the
  closed forms the tests and the conformance report pin. The frozen
  `EnvelopeSpectrumResult` keeps the removed mean level and offers a
  localised `.plot()` with the envelope and its spectrum.
- New "Cepstrum, echoes and the envelope spectrum" guide (English and
  Spanish) in the Signals and spectra section, with the echo-detection and
  envelope-spectrum figures.
- `electroacoustics.loudspeaker_characteristics` and
  `electroacoustics.LoudspeakerCharacteristics` gather the rated loudspeaker
  characteristics of IEC 60268-5:2003+A1:2007 around a measured on-axis
  response: the characteristic sensitivity level referred to 1 W into the rated
  impedance at 1 m (clauses 20.3/20.4) and the effective frequency range against
  the -10 dB band in the region of maximum sensitivity (clause 21.2) are
  computed from the response, alongside the rated impedance, rated frequency
  range, rated powers, resonance frequency, the impedance modulus curve, the
  total-harmonic-distortion curve and the directional/polar response. A
  `.report()` method renders the IEC 60268-5 rated-characteristics fiche: the
  rated-characteristics table beside the on-axis response with its tolerance
  band and effective-range markers, plus the impedance, THD and polar panels,
  the frequency and polar graphs drawn to the IEC 60263:1982 scale conventions
  (one frequency decade equal to 25 dB, and the 25 dB polar reference circle),
  localised in English and Spanish.
- `electroacoustics.microphone_characteristics` and
  `electroacoustics.MicrophoneCharacteristics` gather the rated microphone
  characteristics of IEC 60268-4:2014 around a measured free-field frequency
  response: the sensitivity level `20 lg(M / 1 V/Pa)` of the rated free-field
  sensitivity (clauses 11.1/11.3), the effective frequency range against the
  response tolerance limits (clause 12.2), the directivity index from a
  rotationally symmetric directional pattern through the clause 11.2.2 a)
  diffuse-field integral (clause 13.2.2) and the equivalent sound pressure
  level due to inherent noise from the weighted noise voltage over the rated
  sensitivity (clause 17.2) are computed from the standard's definitions,
  alongside the rated and minimum permitted load impedances (clauses
  10.2/10.3), the overload sound pressure level at a stated THD limit
  (clauses 14.2/15.2), the derived signal-to-noise ratio re 1 Pa and
  diffuse-field sensitivity level, and the rated power supply (clause 9.1). A
  `.report()` method renders the IEC 60268-4 rated-characteristics fiche: the
  rated-characteristics table beside the free-field response with its
  tolerance band, reference-frequency and effective-range markers, plus the
  directional-pattern, inherent-noise-spectrum and distortion-against-level
  panels, the frequency and polar graphs drawn to the IEC 60263:1982 scale
  conventions, localised in English and Spanish.
- `emission.NoiseEmissionDeclaration` and `emission.OperatingModeDeclaration`
  model the ISO 4871:1996 declaration of noise emission values of machinery and
  equipment: the dual-number form (a measured A-weighted sound power level
  `L_WA` and its uncertainty `K_WA`, optionally an emission sound pressure level
  `L_pA` at a work station) and the derived single-number value
  `L_WAd = L_WA + K_WA` (clause 3.15), with the single-machine verification
  verdict of clause 6.2 (`L_1 <= L_WAd`). `SoundPowerResult.declare()` builds a
  declaration straight from a measured sound power. A `.report()` method renders
  the ISO 4871 declaration fiche (the Annex B dual/single-number table across
  the operating-mode columns, the basic-standard and noise-test-code footnote
  and the verification table), localised in English and Spanish.
- `vibration.junction_transmission` implements the wave-approach bending-wave
  transmission coefficients for rigid plate junctions (Cremer et al. 1973;
  Craik 1981/1996; Hopkins 2007, Section 5.2.1.3). `junction_wave_parameters`
  returns the dimensionless wave parameters `chi` and `psi` of a plate pair
  (Eqs 5.10/5.11); `corner_transmission_coefficient` (Eq. 5.12) and
  `straight_transmission_coefficient` (Eq. 5.13) give the frequency-independent
  angle-resolved coefficients `tau12(theta)` (around a corner) and
  `tau13(theta)` (across a straight section) for X, T-junction (1)/(2) and L
  junctions, `inline_transmission_coefficient` gives the normal-incidence
  in-line coefficient (Eq. 5.14), and `angular_average_transmission_coefficient`
  integrates the diffuse-field average (Eq. 5.6). `coupling_loss_factor`
  (Eq. 2.154) and `wave_vibration_reduction_index` (Eq. 5.116) derive the SEA
  coupling loss factor and the wave-approach junction `Kij`. The
  `junction_transmission` builder returns a frozen `JunctionTransmissionResult`
  (corner and straight curves, their angular averages and a `corner_reduction_index`)
  with a `.plot()` of `tau` versus incidence angle.
- `environmental.impulsive_sound` implements the objective impulsive-sound
  prominence method of ISO/PAS 1996-3:2022. `impulsive_sound_adjustment` reads a
  calibrated pressure signal, computes the A-weighted, F time-weighted level
  history `LpAF`, detects onsets (gradient above 10 dB/s, merging events under
  50 ms apart), measures each onset's level difference and least-squares onset
  rate, and returns the governing prominence `P`, the adjustment `KI` and the
  clause 7 source category (not impulsive / regular impulsive / highly
  impulsive) plus the adjusted `LAeq`, in an `ImpulsiveSoundResult` with a
  `.plot()`. `sound_pressure_level_history` and `detect_onsets` expose the
  level-history and onset-analysis stages, the latter accepting an
  `"upper_half"` onset-rate variant for pass-bys.
- EASA ANP fleet database bridge in `phonometry.aircraft.anp_fleet`:
  `load_anp_database` reads the Aircraft Noise and Performance (ANP) CSV tables
  and exposes, for a real aircraft type and operation, its Noise-Power-Distance
  curves (`AnpNpdCurves`, `LAmax`/`SEL` versus distance per power setting) and
  its default fixed-point trajectory (`AnpProfile`, a ready Doc 29 flight path
  with the takeoff/landing ground-roll masks). `AnpDatabase`/`AnpAircraft` feed
  both straight into the existing `event_level`/`noise_contour` chain, so noise
  levels and contours can be computed from actual aircraft instead of only
  synthetic NPD input. Called without a path the loader uses the full EASA ANP
  database (archive version 2.3, 155 aircraft types) bundled with the package;
  pointed at a directory it reads any other ANP CSV export. Result objects
  expose `.plot()`.
- Every `.report()` fiche accepts a `language` argument (`'en'` default, `'es'`):
  Spanish strings and a comma decimal separator.
- The `.plot()` renderers of the building, noise-control and emission result
  objects accept a `language` keyword (`"en"` by default, or `"es"`). Spanish
  renders the axis labels, titles and legends with building-acoustics
  terminology and a comma decimal separator, while English stays exactly as
  before. A shared `phonometry._i18n` helper module backs the locale-aware
  number formatting and tick localisation.
- The `.plot()` methods of the aircraft, underwater and environmental result
  objects accept a `language` keyword (`"en"`, the default, or `"es"`). Spanish
  renders localised axis labels, titles and legends with comma decimal
  separators; English output is unchanged.
- The `.plot()` methods of the psychoacoustics, hearing and vibration results
  accept a `language` keyword (`"en"`, the default, or `"es"`). Spanish renders
  the axis labels, titles and legends in Spanish and switches the decimal
  separator to a comma; English output is unchanged.
- The `.plot()` methods of the metrology, electroacoustics, simulation and
  broadcast results accept a `language` keyword (`"en"`, the default, or
  `"es"`). Spanish renders localised axis labels, titles and legends and uses a
  comma decimal separator; English is unchanged. An unsupported language raises
  a clear `ValueError`.
- The room and materials `.plot()` renderers accept a `language` keyword
  (`"en"` default, `"es"` for Spanish). Passing `language="es"` renders the
  axis labels, titles and legends in Spanish and switches linear-axis tick
  decimals to a comma; the English output is unchanged. An unsupported code
  raises a clear `ValueError`.
- The frequency/band axes, the shifted-reference rating figure, the
  band-level bar chart, the façade x-axis, the ISO 717-2 500 Hz annotation
  and the time axis shared by every domain's `.plot()` renderers now
  localise their own axis labels and legend entries for `language="es"`
  ("Frequency [Hz]" to "Frecuencia [Hz]", "Measured" to "Medido", "Band" to
  "Banda", and so on), instead of leaving them in English whenever a domain
  renderer relied on the shared helper for that label. English output is
  unchanged.
- IEC 61260-1 filter class compliance can now be exported as a one-page PDF
  fiche. A new `filter_class_compliance(bank, *, num_points=..., edition=...)`
  runs the same verification as `verify_filter_class` and returns a frozen
  `FilterComplianceResult` that packages the verdict with the bank's
  second-order sections, mid-band frequencies, per-band decimation factors and
  sampling rate, so it can redraw the measured relative attenuation without
  keeping a reference to the (possibly stateful) bank. The result exposes the
  standard pair: `.plot()` draws the worst-margin band's measured relative
  attenuation over the class acceptance corridor (the pass region shaded green
  and any out-of-tolerance segment marked red), and `.report(path)` renders the
  accredited fiche laid out like an electroacoustic type-test report: a
  standard-basis line naming the IEC 61260 edition, a metadata header block, a
  per-band classification table (mid-band frequency, achieved class and binding
  margin) beside the mask-overlay plot, and the boxed `Class n - COMPLIES`
  result with its binding margin (or a non-compliance statement). `ReportMetadata`
  gains an optional `required_class` field (0, 1 or 2); when supplied, the fiche
  adds a PASS/FAIL verdict row that passes when the achieved overall class is at
  least as strict as the required one. `verify_filter_class` keeps its existing
  dictionary return unchanged. Selecting `edition="1995"` verifies against the
  IEC 61260:1995 / ANSI S1.11-2004 mask, which keeps the stricter class 0 that
  the 2014 edition dropped, so a high-order bank can be certified to class 0.
  Two rendered example fiches (2014-edition class 1 and 1995-edition class 0)
  are kept under `.github/reports/`, regenerated with `make reports`
  (`scripts/generate_reports.py`), and linked from the filter-banks guide.
- ISO 717 sound-insulation ratings can now be exported as a one-page PDF fiche
  through a `report(path)` method, laid out like an accredited-laboratory test
  report: a standard-basis line, a metadata header block, the one-third-octave
  table beside the measured-versus-shifted-reference plot, a boxed single-number
  result and a footer with a fixed disclaimer. `WeightedRatingResult.report()`
  renders the airborne ISO 717-1 fiche (the `Rw (C; Ctr)` result), and
  `ImpactRatingResult.report()` renders the impact ISO 717-2 counterpart with
  the `Ln,w (CI)` result and the opposite deviation sign.
  `SoundReductionResult.report()` is a convenience that rates the predicted
  `R(f)` and writes its fiche in one call. The report metadata is supplied as a
  new `ReportMetadata` frozen dataclass (specimen, client, room and climatic
  conditions, laboratory identity and an optional requirement); every field is
  optional and only the supplied ones are rendered, so the same object drives a
  full accredited fiche or, with `metadata=None`, a lightweight prediction
  fiche. When a requirement is given, the fiche adds a PASS/FAIL verdict row
  (airborne passes at or above the target, impact at or below it). Passing
  `verbose=True` swaps the two-column `f | value` table for the ISO 717 Annex C
  columns (frequency, measured value, shifted reference, unfavourable
  deviation). Rendering uses reportlab, added as the optional
  `phonometry[report]` extra so it stays out of the runtime dependencies; a
  missing reportlab raises a clear `ImportError` with the install command,
  mirroring the matplotlib guard behind `.plot()`. The rating results gain a
  `quantity` field ("airborne" or "impact") that selects the report labels and
  standard reference. Rendered example fiches (airborne and impact) are kept
  under `.github/reports/`, regenerated with `make reports`
  (`scripts/generate_reports.py`), and linked from the documentation.
- ISO 11654 weighted sound absorption ratings can now be exported as a one-page
  PDF fiche through `AbsorptionRatingResult.report(path)`, laid out like an
  accredited ISO 354 absorption certificate: a standard-basis line, a metadata
  header, the octave-band practical-coefficient table beside the
  practical-versus-shifted-reference plot, the boxed `alpha_w` result with its
  shape indicator and absorption class, an optional minimum-`alpha_w` verdict
  and the fixed disclaimer. A new `weighted_absorption_from_third_octave`
  convenience rates the one-third-octave `alpha_s` spectrum and retains it on
  the result, so the fiche can also show the full one-third-octave `alpha_s`
  table (`Frequency | alpha_s | alpha_p`) that accredited certificates carry.
  It reuses the shared `ReportMetadata` and reportlab back end of the ISO 717
  fiche.
- ISO 532-1 Zwicker loudness results can now be exported as a one-page PDF fiche
  through `ZwickerLoudness.report(path)`: a standard-basis line, a metadata
  header, a compact metrics table (total loudness `N`, loudness level `LN`, and
  the `N5`/`N10` percentiles for a time-varying result) beside the
  specific-loudness pattern plot, the boxed `N (LN)` result, an optional
  maximum-loudness verdict and the fixed disclaimer.
- Programme-loudness measurements can now be exported as a one-page EBU R128
  compliance fiche through `ProgramLoudnessResult.report(path)`: a standard-basis
  line (EBU R128 / ITU-R BS.1770-5), a metadata header, a compliance table
  (integrated loudness, loudness range, maximum true peak, and the maximum
  momentary/short-term levels, each with its target/limit and a PASS/FAIL or
  informational result), the loudness-versus-time plot, the boxed integrated
  result, a combined verdict (integrated within the target tolerance and true
  peak at or below -1 dBTP) and a measurement-basis strip. The verdict is driven
  only by the integrated loudness and the true peak; loudness range and the
  momentary/short-term maxima are shown as informational characterising
  measures. `ReportMetadata.requirement` now accepts any finite value, since a
  programme-loudness target in LUFS is negative.
- Aircraft-noise EPNL results can now be exported as a one-page ICAO Annex 16
  certification fiche through `EPNLResult.report(path)`: a standard-basis line
  (ICAO Annex 16 Vol. I Appendix 2), a TCDSN-style metadata header (aircraft,
  manufacturer / type-certificate holder, applicant, measurement point), a
  metrics table of the informational intermediate quantities (the peak `PNLTM`,
  the duration correction `D`, the 10 dB-down record window and, when non-zero,
  the bandsharing adjustment) above the result's own landscape
  `PNLT`-versus-time plot, the boxed `EPNL = X EPNdB` result, a
  Level | Limit | Margin verdict row when a certification limit is supplied
  (the EPNL passes at or below it), a static reference-conditions strip and a
  footer noting the result is computational and not an official State noise
  certificate. It reuses the shared `ReportMetadata` and reportlab back end of
  the other fiches, and does not reproduce any type-certificate data sheet. A
  rendered example fiche is kept under `.github/reports/`, regenerated with
  `make reports`, and linked from the documentation.
- `enclosure_insertion_loss` now accepts a panel prediction result directly for
  its `panel_transmission_loss` argument, in addition to a per-band array or a
  callable. A `SoundReductionResult` or `ApertureTransmissionResult` (from the
  panel and aperture models) is matched structurally, so its per-band `R` and
  band centres feed the enclosure calculation without any dependency of
  `noise_control` on `building`. The predicted panel `R` and a measured `R`
  are therefore interchangeable at the enclosure input.
- Atmospheric refraction: ray tracing and the parabolic equation
  (`phonometry.environmental.atmospheric_refraction`, also re-exported at the
  top level). The module is the refracting-atmosphere counterpart of the ocean
  solvers in `phonometry.underwater.numerical_propagation` and is clean-room
  from Salomons, *Computational Atmospheric Acoustics* (2001) and Attenborough
  & Van Renterghem, *Predicting Outdoor Sound* 2e (2021, Ch. 11). Vertical
  gradients of the effective sound speed `c_eff(z) = c(z) + wind` are built with
  `linear_sound_speed_profile` and `log_linear_sound_speed_profile` (Salomons
  Eq. (4.5)). `atmospheric_ray_paths` integrates Snell's law for sound rays
  (Eq. (4.3)) with a fourth-order Runge-Kutta scheme, reflecting at the ground,
  and returns the curved ray paths, turning points, travel times and ground
  reflections; for a linear gradient every ray is an exact circular arc of
  radius `ray_curvature_radius` and an upward-refracting profile has the
  closed-form `shadow_zone_distance`. `atmospheric_parabolic_equation` solves
  the Green's Function Parabolic Equation (GFPE, Salomons Appendix H) with the
  split-step Fourier algorithm, a Gaussian starter (Eq. (G.64)), a
  finite-impedance ground condition (Eq. (H.28)) and an absorbing top layer, and
  returns the relative sound level (dB re free field) over the range-height
  plane. In a homogeneous atmosphere the GFPE reproduces the exact
  spherical-wave ground effect (`ground_effect`) to about 0.1 dB; it is
  reciprocal and, over an upward-refracting profile, resolves the acoustic
  shadow. Ground impedance is taken directly, from a `PorousMediumResult`, or
  from an effective flow resistivity via the porous models of
  `phonometry.materials`. The results expose `.plot()` (the profile, the ray
  paths and the relative-level field).
- Theoretical panel sound insulation, predicting the airborne sound reduction
  index `R(f)` from physical properties (`phonometry.building.panel_transmission`
  and `phonometry.building.aperture_transmission`, re-exported at the top level).
  `single_panel_transmission_loss` implements the mass law and coincidence dip by
  Sharp's method (Bies, Hansen & Howard 2017, *Engineering Noise Control* 5e,
  Section 7.2.4): the field-incidence mass law `TL = 10 lg(1 + (pi f m''/rho0 c0)^2)
  - 5.5` (rising 6 dB per octave and 6 dB per doubling of mass), the closed-form
  coincidence frequency `fc = (c0^2/2pi) sqrt(m''/B')`, and the loss-factor-
  controlled dip. `double_wall_transmission_loss` adds the mass-spring-mass
  double wall (Section 7.2.6): the resonance `f0 = 60 sqrt((m1+m2)/(m1 m2 d))`,
  the total-mass law below it and the cavity-boosted sum above, with an optional
  porous `cavity_medium` (a `materials.PorousMediumResult`) that lowers `f0`.
  `slit_transmission_coefficient`, `circular_aperture_transmission_coefficient`
  and `composite_transmission_loss` predict transmission through slits (Gomperts)
  and holes (Wilson & Soroka) and combine them with the wall in the area-weighted
  energy sum (Hopkins 2007, *Sound insulation*, Section 4.3.10, Eq. 4.92), so a
  bare opening of relative area `Sa/S` caps the result at `10 lg(S/Sa)`. Every
  prediction returns a `SoundReductionResult` / `ApertureTransmissionResult` with
  `.rating()` (ISO 717-1) and `.plot()`, and is clean-room from the cited books.
- Plate radiation efficiency and theoretical point mobilities
  (`phonometry.vibration.radiation_efficiency` and
  `phonometry.vibration.point_mobility`, also re-exported at the top level).
  `radiation_efficiency` predicts the frequency-averaged radiation efficiency
  `sigma(f)` of a bending plate by Leppington/Maidanik "method no. 1" (Hopkins
  2007, Section 2.9.4, Eqs 2.227-2.230), tending to `(1 - fc/f)^-0.5` above the
  coincidence frequency `coincidence_frequency`, so it supplies the ISO 7849
  radiation factor of `sound_power_from_vibration` without a power measurement.
  `infinite_plate_impedance` / `infinite_beam_mobility` (and the rod, moment and
  bundled-`MobilityResult` variants) give the closed-form point impedances and
  mobilities of infinite structures and the injected power
  `W = 0.5 |F|^2 Re{Y}` (Cremer, Heckl & Petersson 2005, *Structure-Borne Sound*
  3e, Table 5.1), the theoretical companions of the measured ISO 7626 mobilities.
- Industrial noise control: silencers, HVAC and machine enclosures, a new
  `phonometry.noise_control` domain (Bies, Hansen & Howard, *Engineering Noise
  Control* 5th ed.; Munjal, *Acoustics of Ducts and Mufflers*). `expansion_chamber`,
  `helmholtz_resonator`, `quarter_wave_resonator` and `extended_tube_chamber`
  build **reactive silencers** by the one-dimensional four-pole
  (transmission-matrix) method, returning a `ReactiveSilencerResult` with the
  transmission loss, the insertion loss for configurable source/radiation
  impedances, the compound transfer matrix and `.plot()`; the expansion-chamber
  transmission loss matches the closed form `10 lg[1 + (1/4)(m - 1/m)^2 sin^2 kL]`
  (Bies Eq. (8.111)) exactly, and the four-pole machinery is available under
  `phonometry.noise_control` (`duct_matrix`, `shunt_matrix`, `cascade`,
  `transmission_loss`, `insertion_loss`, `helmholtz_impedance`,
  `quarter_wave_impedance`). The
  `noise_control.hvac` module adds the duct end reflection (`end_reflection_loss`,
  ASHRAE Table 8.14), bend insertion loss (`elbow_insertion_loss`, Table 8.11),
  the plenum transmission loss by Wells' method (`plenum_attenuation`) and the
  flow-generated (self) noise of straight ducts and mitred bends
  (`flow_noise_straight_duct`, `flow_noise_bend`, VDI 2081). `enclosure_insertion_loss`
  gives the machine-enclosure insertion loss `IL = R - C` (Bies Eqs. (7.103),
  (7.111)) from a **caller-supplied** panel transmission loss `R` (array or
  callable; never predicted here) and the interior room constant reused from
  `phonometry.room.room_constant`. The four-pole expansion chamber is
  cross-checked against the independent 2D FDTD solver.
- Radiation of a rigid circular piston in an infinite baffle
  (`phonometry.electroacoustics.piston`, re-exported at the top level). `radiating_piston`
  computes the radiation impedance `rho c S (R1 + j X1)` from the piston
  resistance `R1 = 1 - 2 J1(2ka)/2ka` and reactance `X1 = 2 H1(2ka)/2ka`
  (Beranek & Mellow, *Acoustics* 2nd ed., Eqs. (13.117), (13.118)), the
  low-frequency radiation mass `8 rho a^3 / 3` (Eq. (4.151)), the far-field
  directivity `2 J1(ka sin theta)/(ka sin theta)` and the directivity index,
  returning a `RadiatingPistonResult` with `.plot()`; the building blocks
  `piston_resistance`, `piston_reactance` and `piston_directivity` are exposed
  directly.
- Objective speech intelligibility STOI and ESTOI
  (`phonometry.hearing.objective_intelligibility.stoi`, also `phonometry.stoi`).
  `stoi(clean, degraded, fs)` computes the short-time objective intelligibility
  (Taal, Hendriks, Heusdens & Jensen 2011, *An Algorithm for Intelligibility
  Prediction of Time-Frequency Weighted Noisy Speech*, IEEE TASLP 19(7); short
  version Taal et al. 2010, ICASSP): both signals are resampled to 10 kHz, split
  into 256-sample Hann frames at 50 % overlap, cleared of the clean frames more
  than 40 dB below the loudest, grouped into 15 one-third-octave bands from
  150 Hz, and compared over 384 ms segments by a per-band envelope correlation
  after a per-segment normalisation and a `-15` dB signal-to-distortion
  clipping. With `extended=True` it computes ESTOI (Jensen & Taal 2016, IEEE/ACM
  TASLP 24(11)), which mean- and variance-normalises the short-time spectrogram
  over both its rows and columns so the intermediate index is a spectral
  correlation that tracks modulated maskers. The measures return a `STOIResult`
  with the index `value`, the per-segment and (for STOI) per-band intermediate
  scores, and a `.plot()`. The implementation is clean-room from the papers and
  matches `pystoi` (a test-only cross-check, never a runtime dependency) to well
  under `1e-3`.
- AES17-2015 noise measurements for audio equipment
  (`phonometry.electroacoustics.dynamic_range` and `idle_channel_noise`, also
  re-exported at the top level). `dynamic_range` implements clause 6.4.1 (a
  997 Hz, -60 dBFS test tone, the standard notch, and the CCIR-RMS weighting of
  5.2.7, expressed as the ratio of the full-scale sine to the weighted residual)
  and `idle_channel_noise` implements clause 6.4.2 (the CCIR-RMS-weighted output
  with the device idle, in dBFS). Both reuse the standard notch and the ITU-R
  BS.468-4 weighting of the existing THD+N chain; the CCIR-RMS filter is that
  curve with the standard `-5.63` dB offset (unity at 2 kHz).
- Image-source room impulse response and the steady-state room field
  (`phonometry.room.image_source` and `phonometry.room.steady_field`).
  `image_source_rir` synthesises the room impulse response of a rectangular
  room by mirroring the source in its walls (Kuttruff *Room Acoustics* 4.1,
  Equations (4.4)-(4.6); Vorländer *Auralization* 11.4; Allen & Berkley 1979):
  the direct sound plus one delayed, attenuated impulse per image, with the
  `1/(4 pi r)` spreading, the wall pressure reflection factors `R = sqrt(1 -
  alpha)` (Vorländer Equation (11.39)) raised to the per-wall reflection counts,
  and an optional per-band air attenuation `exp(-m r / 2)`. It returns broadband
  or per-octave-band RIRs plus an exact sub-sample reflection table (times,
  distances, orders, amplitudes, image positions) and a `.plot()` reflectogram;
  `audible_image_count` and `reflection_density` give the shoebox image count
  (Equation (9.23)) and the temporal density `4 pi c^3 t^2 / V` (Equation
  (4.6)). `steady_state_field` and its parts `room_constant`,
  `critical_distance`, `schroeder_frequency` and `steady_state_spl` predict the
  steady-state level `Lp = Lw + 10 lg(Q/(4 pi r^2) + 4/R)` with the room
  constant `R = S alpha/(1 - alpha)`, the critical distance and the Schroeder
  frequency `2000 sqrt(T/V)` (Bies *Engineering Noise Control* 6.4, Equations
  (6.43)-(6.44); Kuttruff 5.6 and Equations (3.44), (5.44)). Anchored by the
  exact reflection geometry, the audible-image count, the Eyring reverberation
  time recovered from the synthetic decay (near-cubic limit) and an independent
  2D FDTD cross-check of the rigid-wall echo and the uniform-damping T60, plus
  the closed-form steady-field relations, with a new EN/ES guide and GitHub
  theory page.
- Spherical-wave ground effect and advanced barrier diffraction
  (`phonometry.environmental.ground_barriers`). `ground_effect` predicts the
  excess attenuation over a finite-impedance ground from the Weyl-Van der Pol
  spherical-wave reflection coefficient `Q = Rp + (1 - Rp) F(w)` (Attenborough
  & Van Renterghem 2021 Eq. 2.40; Salomons 2001 Eqs. D.57-D.60), with the
  boundary-loss factor through the scaled complementary error function
  (`scipy.special.wofz`) and the ground impedance taken either directly or from
  the Delany-Bazley / Miki porous models of `phonometry.materials`.
  `barrier_insertion_loss` adds wave-theoretic screening beyond the ISO 9613-2
  `Dz` term: the Kurze-Anderson closed form in the Fresnel number (Bies 2017
  Eq. 5.138), the exact rigid half-plane (the flat-wedge limit of the MacDonald
  / Hadden & Pierce solution, Attenborough Eqs. 9.19-9.20), thick barriers by
  double-edge diffraction (Bies Eq. 5.157) and the coherent four-path barrier
  on the ground combining the source/receiver images with the spherical-wave
  `Q`. Both results expose `.plot()` (excess attenuation and insertion loss vs
  frequency). Anchored by the analytic limits (hard ground `Q -> 1` and the
  `+6 dB` enhancement, grazing `Rp -> -1`, Kurze-Anderson `5 dB` at `N = 0`,
  the exact half-plane `6 dB` at the shadow boundary) as tests and conformance
  checks, with a new EN/ES guide and GitHub theory page.
- Docs site: unified per-page references. Guides can declare a typed
  bibliography in the page frontmatter (standard, book, article, web and
  report entries with DOI and publisher-catalogue links) and the site renders
  one localized, alphabetized APA-7 "References" section at the end of the
  article, wired into the "On this page" panel. After the loudness, aircraft
  noise and psychoacoustic annoyance pilots, every remaining guide and the
  five theory chapters (EN + ES) migrated their manual References and
  Standards sections into the typed bibliography, with a verified
  publisher-catalogue URL or DOI for every entry.
- Docs site: accessible animation embed (`Video.astro`). Wraps the existing
  four-variant WebM convention (EN/ES, light/dark, lazy poster) and adds a
  required accessible name and textual description, a visible localized
  download link, an in-element textual fallback and keyboard-operable
  controls. Every guide animation now embeds through it, each with a real
  textual description of what the clip shows.
- Docs site: theme-aware figure embed (`ThemeImage.astro`). Takes the light
  variant URL and one alt text, derives the `_dark` twin and emits the same
  light-only/dark-only pair the theme switch and the responsive width tiers
  key on, with native lazy-loading (the page hero opts into eager). All
  guide, getting-started and why-phonometry figure pairs (EN + ES) embed
  through it, and the pages became MDX along the way.
- Docs: media placement pass from the documentation audit. Thirty-eight
  figure, animation and code-block moves reunite each medium with the prose,
  formula or snippet that explains it (including the whole inverted
  underwater-propagation layout and a leftover duplicated code block in the
  levels guide), applied in lockstep to the GitHub docs and both site
  languages, with new introductory sentences where a clip had none.
- Correlation, time-delay estimation and Hilbert envelope (Bendat & Piersol,
  Random Data 4e; Knapp & Carter 1976): `correlation` (auto/cross via
  zero-padded FFT with the biased, unbiased and coefficient normalizations,
  the coefficient always carried, and the per-lag random error of
  Eqs. 8.109/8.112 as `.random_error()` / `correlation_random_error`),
  `time_delay` (the direct correlator of Section 5.1.4, the Eq. 5.101b
  cross-spectrum phase slope and the generalized cross-correlation with the
  Knapp & Carter Table I weightings Roth/SCOT/PHAT/ML over the shared Welch
  core, sub-sample peak refinement by parabolic interpolation with optional
  band-limited local upsampling, and the Eq. 8.129 peak-location uncertainty
  with its Eq. 8.130 interval), `impulse_response_delay` and
  `align_impulse_responses` (sub-sample IR arrival/pair delay and exact
  fractional-shift alignment) and `envelope` (analytic-signal envelope,
  unwrapped instantaneous phase and frequency per Chapter 13, with optional
  anti-aliased or ECMA-convention plain decimation), each returning a frozen
  result with `.plot()`. Anchored on exact integer and synthetic fractional
  delays (accuracy pinned from a few hundredths of a sample for the
  parabola alone to 1e-5 at 32x upsampling), the closed-form sine and
  bandwidth-limited-white
  autocorrelations, the Example 8.5 published numbers, the ideal-delta
  GCC-PHAT of a noiseless delay and the exact AM envelope; seven new
  conformance checks.
- Swept-sine harmonic-distortion separation (Farina 2000; Novak, Lotton &
  Simon 2015): `synchronized_sweep_signal` (the synchronized exponential
  sweep of Novak Eqs. 47/49, whose rounded rate makes a `L*ln(n)` time
  shift exactly equivalent to the nth harmonic) and `swept_sine_distortion`
  in `phonometry.electroacoustics`, which deconvolves a recorded sweep
  response (closed-form analytic inverse spectrum for the synchronized
  sweep, extending every harmonic band to `[n*f1, n*f2]`; the classical
  Farina inverse filter for plain ESS recordings) and windows each
  distortion order at its exact `L*ln(n)` advance with sub-sample
  alignment, returning a frozen `SweptSineDistortionResult` with the
  harmonic frequency responses `H1..HN`, their impulse responses, THD as a
  function of the excitation frequency, per-order distortion ratios and
  `.plot()`. With the synchronized sweep the harmonic phases are system
  properties; with `method="farina"` the magnitudes are exact and the
  phases are documented as excitation-dependent. Anchored on the Chebyshev
  identities of a memoryless cubic polynomial (`|H1| = 1+3a3/4`,
  `|H2| = a2/2` at phase -pi/2, `|H3| = a3/4` at phase pi, and the
  closed-form THD), the pure-linear THD floor, a delayed/gained Hammerstein
  composition, and agreement with the tone-by-tone `thd` analyzer on the
  same nonlinearity.
- Phase utilities in `phonometry.metrology` (Bendat & Piersol Sec. 13.1.4):
  `minimum_phase` (the minimum-phase response of a magnitude via the real
  cepstrum, with trigonometric-interpolation oversampling against cepstral
  aliasing and documented sampling/zero-floor precautions), `group_delay`
  (central-difference `-(1/2pi) dphi/df` of an unwrapped one-sided
  response), `excess_phase` (measured minus minimum phase: the all-pass
  part no stable causal equalizer can remove) and `phase_decomposition`
  (frozen result bundling magnitude, the three phases and the total and
  excess group delays, with `.plot()`). Anchored on a strictly
  minimum-phase biquad reconstructed to better than 1e-12 rad, the exact
  first-order-allpass group delay `(1-a^2)/(1+2a*cos(w)+a^2)`, and the
  exact `-2*pi*f*t0` excess phase and constant excess group delay of a
  pure fractional latency. Seven new conformance checks across both
  features (306 total).
- 2D FDTD wave simulation as a public API (new `phonometry.simulation`
  domain, promoting the engine behind the documentation animations):
  `fdtd_simulation` (deterministic staggered-grid pressure-velocity leapfrog
  solver per Attenborough & Van Renterghem 2021 chapter 4, Eqs. 4.11-4.14,
  returning a frozen `FDTDResult` with per-probe pressure histories, optional
  field snapshots and `.plot()` for probe traces and field frames), the
  stepping engine `FDTD2D` (now with rasterised rigid `obstacle_mask`
  geometry and per-side locally reacting real-impedance boundaries,
  Eqs. 4.32-4.35, alongside the existing sponge layers), and the sources
  `GaussianPulse`, `CWSource` and the new arbitrary-waveform `SignalSource`.
  Validated against analytic oracles: rigid-box and 1D-tube
  eigenfrequencies, free-field arrival delay and cylindrical `1/sqrt(r)`
  decay, the rigid-wall image echo, the normal-incidence impedance
  reflection coefficient `(Z - rho c)/(Z + rho c)`, the scheme's discrete
  dispersion relation and second-order convergence under grid refinement;
  two new conformance checks. The figure/animation scripts now import the
  library engine (bit-identical fields, committed clips unchanged).
- Porous-material models and multilayer absorber prediction
  (`phonometry.materials.porous_absorber`): the Delany-Bazley one-parameter
  power law (with the Bies Table D.1 presets for polyester and foams and
  the published 0.01 < X < 1 fit-range warning), Miki's positive-real
  revision and the five-parameter Johnson-Champoux-Allard model, each
  returning a frozen `PorousMediumResult` (characteristic impedance,
  wavenumber, effective density/bulk modulus, `.plot()`); a declarative
  transfer-matrix multilayer solver (`layered_absorber`) over porous, air,
  perforated-plate, microperforated-plate (Maa's exact Bessel impedance
  with the Eq. 5 end corrections) and limp-membrane layers at any incidence
  angle with rigid/free/impedance terminations, evaluated through a
  numerically robust admittance recursion and exposing the reciprocal chain
  matrix; the random-incidence Paris integral
  (`diffuse_field_absorption`) and the locally reacting closed form
  (`statistical_absorption`, maximum 0.951); and the closed-form
  Helmholtz/membrane resonance helpers. Anchored digit-exact on the printed
  coefficient tables (Bies 5e Table D.1 / Mechel 2e G.11; Miki 1990
  Eqs. 30-34), on the Johnson et al. 1987 asymptotic limits, on the
  hard-backed-layer and lossless-cavity closed forms, on Maa 1998 (the
  Eq. 4 approximation, Eq. 9/23 absorption identities, Eq. 11 resonance,
  Table I and the Fig. 5 design example) and cross-checked against the
  ASTM E2611 `TransferMatrix` machinery; ten new conformance checks. Three
  source misprints recorded in the errata registry (Maa 1998 Eq. 5b;
  Attenborough & Van Renterghem 2021 Table 5.1 and Eq. 5.13).
- Programme loudness and true peak (ITU-R BS.1770-5, EBU R 128 with Tech
  3341/3342), a new `phonometry.broadcast` domain: `program_loudness` (the
  full EBU Mode measurement set as a frozen `ProgramLoudnessResult` with
  `.plot()`: integrated loudness with the two-stage -70 LKFS / -10 LU gate,
  ungated momentary and short-term series with their maxima, the Tech 3342
  loudness range with its cascaded -20 LU gate and 10th-95th percentile
  spread, and per-channel true peaks), `integrated_loudness`,
  `loudness_range`, `true_peak_level` (Annex 2 oversampling to 192 kHz,
  sharing the inter-sample-peak machinery of `lc_peak`), `k_weighting` /
  `k_weighting_coefficients` (the Annex 1 biquads, verbatim at 48 kHz and
  redesigned through the analog prototype elsewhere, response within
  0.02 dB at 32 kHz and above) and `channel_weight` (the Annex 3 position-dependent weights,
  Table 4). Validated against every synthesizable EBU Tech 3341 minimum
  requirements signal (cases 1-6 and 9-23) and Tech 3342 LRA signal (cases
  1-4) at their official tolerances, the 997 Hz / -3.01 LKFS anchor and the
  Annex 2 under-read bound; eight new conformance checks and a new
  programme-loudness guide (EN/ES) with a metering figure.
- Calibrated spectral analysis (Bendat & Piersol, Random Data 4e):
  `power_spectral_density` and `cross_spectral_density` (Welch estimators
  that report the effective number of independent averages under overlap,
  the normalized random errors of Eqs. 8.158/9.33/9.52 and the chi-square
  confidence interval of Eq. 8.163), `coherent_output_spectrum` (the
  Gvv = γ²·Gyy / Gnn split with the spectral SNR and the Eq. 9.73 random
  error), `resolution_bias_error` (the Eq. 8.141 first-order peak bias),
  `fractional_octave_smoothing` (constant-power 1/n-octave kernel, exact
  flat-spectrum invariance) and `noise_signal` (seeded white/pink/red/blue/
  violet noise with an exact power-law PSD slope), each estimator returning
  a frozen result with `.plot()`. The random errors and interval coverage
  are pinned by seeded Monte Carlo against the closed forms, and the
  slopes/levels by exact oracles; six new conformance checks.
- The `transfer_function`/`coherence` estimators, the p-p `sound_intensity`
  probe and the ECMA-418-1 tonality spectrum now compute their spectra
  through the shared Welch core in `phonometry.metrology.spectra`
  (bit-identical outputs, verified array-for-array against the previous
  implementations).
- Rotorcraft terrain and screening (NORAH2 guidance §A.4.4-A.4.5, completing
  the ECAC Doc 32 domain): `mean_ground_plane` (the closed-form continuous
  least-squares plane of a terrain section, Eq. 36-40, with orthogonal
  equivalent heights), `mean_flow_resistivity` (the log-mean of Eq. 41),
  `diffraction_attenuation` (Eq. 42-44 with the multiple-edge coefficient and
  the 25 dB bound) and `terrain_screening_adjustment` (the full vertical
  section: mean-plane ground effect on clear paths, rubber-band diffraction
  combined with side ground effects per Eq. 45-47 when terrain blocks the
  line of sight), plus digital-elevation-model support in
  `rotorcraft_event_level`/`rotorcraft_noise_contour` (`terrain`,
  `terrain_resolution`) and per-receiver `flow_resistivity`/`ground_elevation`
  arrays on the contour grid. Anchored on closed-form oracles (flat and
  inclined sections reproduce the flat-ground model exactly, hand-checked
  rubber-band geometry, classical grazing 10·lg 3) and end to end on the
  NORAH2 ARP Case 3 per-receiver elevations (187 microphones, SEL/LASmax to
  0.05 dB) and the mixed-ground Case 2 grid in a single call; four new
  conformance checks. Two further guidance defects recorded in the errata
  registry (the Eq. 46 subscript and the §A.4.5 cross-references).
- Rotorcraft single-event method (ECAC Doc 32 §4.3/§5-6, NORAH2 guidance
  §A.3-A.5) completing the hemisphere chain:
  `flight_condition_weights`/`interpolated_source_level` (the distance-scaled
  Delaunay interpolation across a hemisphere database, Eq. 3-10, with nearest
  neighbour outside the envelope and optional database lookup triangulations),
  `flight_path_kinematics` (ground speed, airspeed, heading, curvature, bank
  and path angle by central differences, Eq. 16-21), `rotorcraft_event_level`
  (the received one-third-octave time history at recorded time and its
  `LASmax`, full-history and 10 dB-down `SEL` and ICAO Annex 16 `EPNL`, with
  per-point overrides for radar-track workflows, bank-tilted emission frames
  and per-point class-substitution level offsets) and `rotorcraft_noise_contour`
  (vectorised single-event `SEL`/`LASmax` ground grids), each result frozen
  with `.plot()`. `RotorcraftHemisphere.mirrored()` handles mirrored-rotor
  class substitutions (Eq. 2). Validated end to end against the NORAH2
  reference implementation's ARP verification cases (emission angles 0.01°,
  hard-ground step levels 0.08 dB(A) out to 18 km, `LASmax` 0.03 dB, `SEL`
  0.05-0.4 dB, `PNLTM` 0.1 dB, 187-microphone contour grid 0.7 dB worst case) and against
  hand-checked interpolation, kinematics and Lorentzian-flyover closed forms;
  four new conformance checks. Two guidance defects recorded in the errata
  registry (Eq. 21 path angle; §A.3.1 triangulation plane).
- Four new physics-based 2D FDTD documentation animations (WebM + poster
  for the site, GIF for the GitHub docs, EN + ES, light + dark), each
  simulated with `scripts/fdtd2d.py` and annotated on canvas:
  `anim_fdtd_barrier` (a thin rigid screen on reflecting ground driven at
  100 Hz vs 500 Hz side by side, with the measured insertion loss at a
  shadow-zone receiver, embedded in the outdoor-propagation guide),
  `anim_fdtd_ground_effect` (a 400 Hz source over a rigid plane forming
  the direct + image-source interference lobes, with the level on an 8 m
  arc converging to the two-path model and its predicted nulls, embedded
  in the outdoor-propagation and aircraft-noise guides),
  `anim_fdtd_ducting` (a low-frequency pulse trapped by a SOFAR-like
  sound-speed channel vs a near-surface source that leaks to depth, with
  the c(z) profile inset, embedded in the underwater-propagation guide)
  and `anim_fdtd_diffusion` (a plane wavefront onto a flat panel vs an
  N = 7 quadratic-residue Schroeder diffuser, with the scattered field
  and the ISO 17497-2 arc diffusion coefficients, embedded in the
  surface-scattering guide).
- Seven new mechanism animations (WebM + poster for the site, GIF for the
  GitHub docs, EN + ES, light + dark): the ISO 10534-2 standing wave in the
  impedance tube (rigid vs porous termination), the EN 12354-1 flanking
  paths Dd/Ff/Fd/Df, the ISO 9614-2 intensity scan accumulating partial
  powers into L_W, the ISO 18233 sweep deconvolution into the impulse
  response, the ISO 532-1 specific-loudness pattern and its integral, the
  same source measured in an anechoic and a reverberation room converging
  to one L_W, and comb filtering from a single floor reflection; embedded
  in the materials, insulation-prediction, sound-power, intensity,
  room-acoustics and loudness guides (EN + ES) with deferred loading, plus
  a `--anim NAME` flag on `scripts/generate_graphs.py` to re-render a
  single clip.
- `scripts/fdtd2d.py`: a small deterministic 2D acoustic FDTD engine
  (staggered-grid pressure-velocity leapfrog, heterogeneous c/rho maps,
  rigid walls, sponge absorbing layers, Gaussian-pulse and CW sources,
  frame subsampling for animation rendering) with physics and determinism
  tests, backing the physics-based documentation animations.
- `anim_fdtd_room_modes`: a 2D FDTD documentation clip driving a rigid
  5 m × 3,5 m room on its (2,1) mode and between modes, showing the
  standing-wave pattern and the RMS mode map growing to dominance on
  resonance while the off-mode response stays weak and disorganised;
  embedded in the room-acoustics and reverberation-prediction guides
  (EN + ES).
- Clause 5.3.8 Step 3 inside `analyze_spectrum` (DIN 45681 / ISO PAS 20065):
  audible tones sharing a critical band are energy-summed (Formula (17),
  shared lines counted once) into a combined FG entry rated at the most
  audible member, with the exactly-two-tones-below-1000-Hz separation
  exception (Formulae (18)/(19)); the new `group_sizes` field on
  `ToneAudibilityResult` tells single tones from FG entries and the decisive
  audibility now follows Step 4 as printed.
- DIN 45681:2005-03 Anhang I oracle: the Tabelle I.9 wind-energy-plant
  spectrum reproduces the decisive Tabelle I.10 row end to end (LS 41,71 /
  LT 68,10 / dL 12,52 / u 3,18 dB), Tabelle I.11 rows j = 45/48 pin the
  Formulae (12)-(14) chain, the printed 5 FG row pins the two-tone
  combination decision, and Tabelle A.1 pins Formula (2) at 24 printed
  bandwidths.
- `verify_weighting_class`: a dense IEC 61672-1 5.5.7 sweep between the
  nominal frequencies (analytic Annex E design goal, larger of the adjacent
  limits) so a resonance or notch between nominals can no longer pass, and a
  `range_limited` flag when Table 3 rows with finite lower limits fall
  beyond Nyquist.
- `Barrier(line_of_sight_clear=True)`: the ISO 9613-2 negative-z sign
  convention when the sight line passes above the top edge; Eq. (14) is
  evaluated with Kmet = 1 and Dz decays continuously from 10 lg 3 at grazing
  to 0, never granting screening credit below the sight line.
- `wind_turbine_tonality`: the IEC 61400-11 9.5.2 possible-tone screening
  and a `has_identified_tone` flag (spectra without an identified tone must
  be excluded from the 9.5.1 bin averaging); `slant_distance` gains the
  vertical-axis Formula 2 (`rotor_axis="vertical"`, R0 = H + D).
- `uncertainty_from_repeated_measurements`: the primary ISO 1996-2
  Formulae (17)+(19) energy-domain route as `standard_uncertainty`, with the
  Note 2 substitute kept as `approximate_uncertainty` and a spread warning.
- `ResidualCorrectionResult.reportable_upper_bound`: the uncorrected
  measured level, the value 10.4 permits reporting when the margin is 3 dB
  or less; `ImpulseProminenceResult.qualifies`: the NT ACOU 112 clause 4.5
  onset-rate qualification mask.
- Published-oracle promotions: GUM Annex H.1 end to end through
  `combine_uncertainty` (uc = 31,71 nm, U99 = 92,1 nm), GUM Annex H.2 as the
  correlated Equation (16) oracle (uc(R/X/Z) = 0,071/0,295/0,236 ohm),
  GUM-S1 Tables 2 and 3 coverage intervals (+/-3,92 Gaussian, +/-3,88
  rectangular with a seeded Monte Carlo conformance entry), the full
  ISO 9613-2 Table 2 grid (6 conditions x 8 octave bands at exact midbands)
  and the IEC 61260-1 Table F.1 breakpoints with both E.3.4 rounding
  examples.
- New warning classes: `WindTurbineNoiseWarning`, `UncertaintyWarning` and
  `ImpulseProminenceWarning`.
- `itu_r_468_weighting`: the ITU-R BS.468-4 Table 1 weighting-network
  response (identical to the IEC 60268-1 Appendix A network that
  IEC 60268-3 14.12.11 requires), interpolated per the recommendation's own
  log-frequency rule; `weighted_thd` now defaults to it (`'468'`), keeps
  `'A'`/`'C'` as labelled options and documents the 31,5 Hz - 400 Hz
  fundamental validity range.
- AES17 measurement-bandwidth chain for `thd_plus_noise`, `sinad` and
  `harmonic_analysis`: both the residual and the total signal are measured
  through a 20 Hz high-pass plus the standard low-pass (clauses 5.2.5 /
  6.3.1) with a configurable `bandwidth` (default 20 kHz;
  `bandwidth=None` keeps the full-Nyquist measurement).
- `ModulationDistortionResult`: `modulation_distortion` now returns the
  IEC 60268-3 14.12.7 per-order values `d2`/`d3` together with the
  SMPTE-analyzer combined RMS as the explicitly labelled `smpte` field.
- Hearing and electroacoustics oracle expansion: the SII quiet condition at
  the official worksheet full precision plus a discriminating
  noise-plus-hearing-loss case and the independent R package Example C.2;
  the IEC 60268-16 Ed.5 C.3.2 STIPA direct-method staircase (m = 0 to 1
  through the full audio path, tolerance 0,05); the clipped-sine THD
  Fourier constants; a full-signal DIM regression (1,536 MHz synthesis of
  the standard test signal through a weak polynomial nonlinearity,
  FIR-decimated to 192 kHz, checked against an exact-bin FFT oracle); the
  H1 input-noise bias SNR/(1+SNR); the printed ITU-R 468 response values
  with the AES17 CCIR-RMS cross-check; and the effective notch-Q
  validation on the applied zero-phase response.
- ISO 532-1 Annex B.4 level-ramp oracles (Test signals 6-9): the tone and
  pink-noise ramps are regenerated from parameters measured on the official
  electronic-attachment WAVs and pinned to the workbook Nmax/N5 with
  tolerances calibrated against those WAVs; the sub-1-sone phon conversion
  and the NX percentile are now confirmed line by line against the
  attachment's Annex A.4 reference program source.
- `docs/ERRATA.md` entry for IEC 60268-3:2013 clause 14.12.9.2 f): the
  printed DIM denominator ("U2") contradicts the 14.12.9.1 definition
  (the output amplitude at f_s), which the library follows.
- ISO/PAS 20065:2016 extended uncertainty of the audibility (Clauses 5.4/6):
  `audibility_uncertainty` propagates the uniform 3 dB narrow-band level
  uncertainty through the audibility chain to the extended uncertainty U
  (90 % bilateral coverage, k = 1,645) and `mean_audibility_uncertainty`
  combines the per-spectrum values through the Formula (20) mean;
  `analyze_spectrum` reports the per-tone U on its result. Verified against
  the printed Table E.2 U column (decisive tone to 0,006 dB), the 2 FG row
  and the Annex E Step 4 mean uncertainty (1,38 dB).
- ISO 532-1 stationary `time_skip` parameter (Annex B.1's TimeSkip: the
  stationary calculation starts from 0,2 s when validating against the
  official recordings, excluding leading silence and the filterbank
  transient from the mean square).
- Psychoacoustics oracle expansion: the shipped ISO 532-1 Annex B.4 Test
  signal 13 and a B.3 pink-noise bounds check; all 21 DIN 45692 Table A.2
  rows and all 20 Table A.3 broadband rows at the normative 5 % / 0,05 acum
  tolerance (plus a non-definitional Table A.2 conformance row at 2,5 kHz);
  the ISO 532-2 Annex B phon columns; the ISO 532-3 Annex C.3 multi-tone
  complexes, the C.1.2/C.1.3 sone columns and the Annex C.1 anchor pinned at
  32/44,1/48 kHz native rates; the ECMA-418-2 decision thresholds as
  constants-tests (audibility 0,01 sone_HMS, prominent tonality 0,4 tu_HMS,
  prominent roughness 0,2 asper); a loudness/tonality N'_tonal
  cross-consistency test on the shared Clause 6.2 intermediate; the
  fluctuation-strength carrier sweep (F&Z Fig. 10.5 trend) and the Osses
  Table 1 AM-broadband-noise row as a trend cross-check; and the full
  ISO/PAS 20065 Tables E.2/E.3/E.4 columns (LG, av, band limits, per-tone
  and mean uncertainties, all-spectra tone records).
- ECMA-418-1 TNR/PR degenerate-input warnings: peaks within one bin of the
  89,1 Hz / 11,2 kHz range edge (bin snapping can flip the prominence
  verdict) and critical bands at numeric-noise power (silence, DC) now warn;
  the `prominent` verdict is documented as the numeric criterion only (the
  clause 11.8/12.8 aural examination and clause 8/9 hearing-threshold screen
  are the caller's responsibility).
- `docs/ERRATA.md` entries for the review findings: the ECMA-418-1 clause
  4.1.2 "11 220 Hz" range misprint, the internally inconsistent
  ECMA-418-2 clause 5.1.5.2 last-block formula, the ISO/PAS 20065 clause
  5.3.4 edge-steepness print that contradicts the executable DIN 45681
  Annex J program, and the Osses 2016 Eq. (3) Bark-constant exponent typo.
- Underwater printed-source oracles promoted to tests and the conformance
  report: the Wong & Zhu (1995) ITS-90 check tables pin the UNESCO and
  Del Grosso sound-speed refits at 17 printed grid points each (agreement
  within the printed 0.001 m/s resolution), the Francois & Garrison (1982,
  Part II) Table IV pins the total seawater absorption at 21 printed points
  spanning 0.4 kHz-1 MHz (within half a unit of the last printed digit), and
  the Wales & Heitmeyer (2002) ensemble merchant-ship mean spectrum is pinned
  against the paper's printed closed form and asymptote statements
  (exponents -3.6/-1.76, 340 Hz breakpoint).
- `docs/ERRATA.md`: a versioned registry of the defects found in published
  standards and guidance documents during implementation (misprints, worked
  examples contradicting their own normative text, ambiguous wording), each
  with the evidence, the library's disposition and a report status.
- ISO 7626-2:2015 measurement-side acceptance criteria:
  `rigid_mass_calibration_check` implements the 7.5.2 operational calibration
  on a rigid block of known mass (|A| = 1/m or |Y| = 1/(2 pi f m) within
  +/-5 %, per-band pass flags in a `RigidMassCalibrationResult`) and
  `random_error_percent` the Annex A normalized random error
  eps = sqrt((1-gamma^2)/(2 n gamma^2)) with the 8.1.3 < 5 % averaging
  criterion; measured-FRF processing per 8.1.3 (H1) and coherence checks are
  documented through the existing `transfer_function`/`coherence` estimators.
- ISO 10846-1:2008 Equation (6): `blocking_force_ratio` quantifies the
  blocking-force approximation F2/F2,b = 1/(1 + k2,2/kt) (within 10 % for
  |k2,2| < 0,1 |kt|, Eq. (7)).
- Vibration oracles promoted to tests and the conformance report: the nine
  printed ISO 8041-1:2017 Annex B design-goal tables (318 bands) with a
  Table 5 tolerance-envelope check and the Wb/Wc/Wd/We/Wf/Wj design-goal
  points, the ISO 2631-5:2018 Annex C NOTE 5 female worked example
  (Sd = 1,40 MPa, R = 0,97) and the Annex D Table D.1 digital-filter
  cross-check of Formula 1, the ISO 7626-2 rigid-mass calibration values and
  Annex A random-error example, the omega = 1000 rad/s rigid-mass decade
  identity, the ISO 10846-3 validity anchors (|T| = 0,1 <->
  DeltaL1,2 = 20 dB, model bias 1,1 within 1 dB / 12 %), the ISO 10846-1
  Eq. (6) force ratio 1/1,1 and the 7.6 linearity criterion (10 dB input
  step within 1,5 dB).
- ISO 717 enlarged frequency ranges: `weighted_rating_extended` computes the
  Annex B spectrum adaptation terms (C50-3150, C50-5000, C100-5000 and the
  Ctr counterparts, Table B.1 spectra) and `weighted_impact_rating_extended`
  the CI,50-2500 term of ISO 717-2:2020 A.2.1; both offer the 0,1 dB
  reference-curve shift and one-decimal reductions for uncertainty statements
  (ISO 12999-1:2020 Annex B), reproducing the printed Rw = 57,4 dB example
  and the A.2.2 constants Ln,r,0,w = 77,6 dB / CI,r,0 = -10,3 dB.
- ISO 16251-1 statement of results: the floor-covering improvement now rates
  any spectrum containing the 16 bands 100-3150 Hz (the clause 6.3 18-band
  range included) and exposes the spectrum adaptation term CI,delta
  (ISO 717-2:2020 Formula (A.4)) as `ci_delta` /
  `impact_improvement_adaptation_term`.
- EN 15657:2018 source-quantity chain: `equivalent_blocked_force_level`
  (Formula 15), `characteristic_reception_plate_power` (Formula 17,
  Y_R,inf,low = 5e-6 m/(N s)), `equivalent_free_velocity_level` (Formula 18)
  and `source_mobility_from_levels` (Formula 19), plus the EN 12354-5 Annex I
  mobility correction `installed_power_from_reception_plate`; the raw
  Formula (14) level is re-documented as the plate-injected power.
- ISO 9611:1996 `mean_free_velocity_level` (equation (9) energy mean over
  contact points, v0 = 5e-8 m/s) with a conformance check.
- EN 12354-1 Annex E junction families E.7 (lightweight double leaf with
  homogeneous elements), E.8 (coupled double-leaf walls) and E.9 (corner and
  thickness change), and the E.5 K24 double-leaf branch (clamp read as
  -4 dB <= K24 <= 0 dB; the 2000 print's "0 <= K24 <= -4 dB" is a misprint).
- EN 12354-1 Formula (5b) `standardized_level_difference` (DnT,w from R'w,
  exact 0,32 V/Ss form) closing both Annex H.3 examples to 54 dB.
- EN 12354-3 Annex C facade-shape term lookup
  `facade_shape_level_difference` (Figure C.2, all nine shapes, alpha_w
  interpolation).
- ISO 10848-4 Clause 9 modal-overlap bracketing: `vibration_reduction_index`
  accepts the per-band modal overlap factor, flags M < 0,25 bands as
  `bracketed` and excludes them from the single-number Kij.
- ISO 15186-1 Clause 6.4.6 negative-direction subareas: `combine_subareas`
  accepts signed subarea areas (minus sign for reverse energy flow) with
  Sm = sum(|Smi|), raising when the signed energy sum is not positive.
- Oracles promoted to tests and the conformance report: ISO 717-1:2020
  Annex C Table C.2 (enlarged 50-5000 Hz range), ISO 717-2 Annex C Table C.1
  covered floor (64, -3, 30,0 dB) and Table C.2 improvement (15 dB,
  CI,delta = -9 dB), ISO 12999-1:2020 Annex B single numbers and
  uncorrelated/correlated uncertainties, all 12 printed EN 12354-1 Annex H.3
  path values with the DnT,w closures, the remaining EN 12354-4 Table G.9
  reception cells, ISO 10140-5 Tables B.1/C.1 reference elements and floors
  end-to-end, the printed ISO 15186-1 Table B.1 Kc rows, and the EN 12354-5
  Annex I whirlpool (Table I.6a) and flushing cistern (Tables I.8/I.9)
  worked examples replacing the former formula-restatement checks.
- Underwater validation: independent image-source absolute TL anchors for the
  normal-mode and parabolic-equation solvers, the published UNESCO EOS-80
  canonical sound-speed check value (Fofonoff & Millard 1983), six additional
  JOMOPANS File S1 oracle rows, the Ainslie-McColm Table I reference oceans
  against Francois-Garrison, Wenz-chart graphical anchors for the wind noise,
  and the RANDI 3.1 report's representative source levels (Table 2) for the
  ship-spectrum model; two new conformance checks (198 total).
- ECAC Doc 32 rotorcraft: `reference_distance` parameter on
  `spherical_spreading_adjustment` and `atmospheric_adjustment` for hemispheres
  recorded at a non-standard polar distance, and an end-to-end propagation-chain
  oracle against the NORAH2 prototype single-event histories (0.1 dB(A) over
  hard ground) plus off-node bilinear spot checks on the reference hemispheres
  of all eleven rotorcraft types.
- ECAC Doc 29 airport noise: landing-rollout modelling (`landing_roll` mask;
  reduced noise fraction Eq. 4-21b, nearest-end geometry, no directivity term)
  and per-segment bank angle (`bank`, §4.5.2 sign convention). ICAO Annex 16
  EPNL: the bandsharing adjustment ΔB (App. 2 §4.4.2/4.4.3), exposed as
  `EPNLResult.bandsharing_adjustment`, and a `start_band` parameter on
  `tone_correction` for the helicopter 50 Hz procedure.
- STIPA certified-bench verification: the IEC 60268-16 rev 5 verification
  test-bench signals from stipa.info (Embedded Acoustics BV) run as a local
  end-to-end oracle over `stipa` / `sti_from_impulse_response` (49 WAVs
  across Annexes C.3.2 modulation-depth staircase, C.3.3 exponential decays,
  C.4.2 filter slope, A.2.2 weighting factors and A.3.1.2 phase distortion;
  the suite skips cleanly when the local data is absent), plus seven synthetic
  conformance checks that re-derive the same constructions in CI (one of them
  replacing the previous C.3.2 m = 0,5 check, so the total grows by six): the
  C.3.2 staircase at m = 0,2/0,5/0,8 (±0,01 STI), the C.3.3 RT60 = 1 s decay
  against the closed-form Schroeder MTF (±0,005), the C.4.2 slope criterion
  (m ≥ 0,5 under a +41 dB unmodulated adjacent-octave tone) and the
  A.2.2/A.3.1.2 audio-path checks (257 checks total).
- `fluctuation_strength_ecma`: the normative psychoacoustic fluctuation
  strength of ECMA-418-2:2025 Clause 9 (Sottek Hearing Model, vacil_HMS),
  completing the standard's metric set next to `loudness_ecma`,
  `tonality_ecma` and `roughness_ecma`. Implements the High-resolution
  Spectral Analysis of the critical-band envelopes (least-squares
  window-kernel line-pair fits, Formulae 121-142), the envelope-dependent
  analysis windows with quieter-period detection (Clause 9.1.3), the
  band-pass modulation-rate weighting, damped-Newton rate fine tuning and
  harmonic-complex analysis (Clauses 9.1.5-9.1.9) and the HSA-based loudness
  scaling (Clause 9.1.10), reusing the shared Clause 5 auditory front-end.
  The frozen `EcmaFluctuationStrength` result exposes `.plot()` (F(l50)
  trace plus specific-fluctuation-strength heatmap). Reproduces the Clause 9
  calibration signal (1 kHz, 100 % AM at 4 Hz, overall 60 dB SPL) to
  0.9958 vacil_HMS converged with the tabulated `c_F` (no reverse fitting), pinned in CI
  together with closed-form HSA recovery anchors and the standard's
  qualitative band-pass behaviours; `docs/ERRATA.md` records four confirmed
  Clause 9 print defects (the Formula (127) kernel phase, the Formula (144)
  bin offset, the unit-less Newton constants of Clause 9.1.7 and a broken
  cross-reference).
- `SoundAbsorptionMeasurement` and `measure_sound_absorption`: a public
  reverberation-room sound absorption measurement result type (BS EN
  ISO 354:2003). It carries the one-third-octave reverberation times of the
  empty room and of the room with the specimen, the room volume, the specimen
  area and the climate, and derives the equivalent sound absorption areas `A1`
  (Eq. (5)) and `A2` (Eq. (7)) and the sound absorption coefficient `alpha_s`
  (Eq. (8)/(9)) by reusing the existing `absorption_area`/`absorption_coefficient`
  functions; `alpha_s` may exceed 1,0 (Clause 3.7 NOTE 2) and is never clamped.
  The result exposes `.plot()` (`alpha_s` versus one-third-octave frequency) and
  `.report()`, which renders a one-page accredited reverberation-room test-report
  PDF (metadata header, the `alpha_s` table beside the curve, `verbose=True`
  adding the `T1`/`T2` and `A1`/`A2` detail columns). ISO 354 is a
  characterisation, so the fiche has no single-number rating and no pass/fail
  verdict; the weighted `alpha_w` remains the separate ISO 11654 rating.
- `RoomAcousticsResult.report()`: a one-page PDF room acoustic parameters
  measurement fiche (ISO 3382-1:2009 performance spaces, ISO 3382-2:2008
  ordinary rooms, both by the integrated impulse-response method). It renders
  the standard-basis line, an optional metadata header, the full-width per-band
  parameter table (T20, T30, EDT, C50, C80, D50 and Ts, one row per octave or
  one-third-octave band) above the result's own per-band decay-time plot, and
  the boxed mid-frequency reverberation time T_mid (the mean of the 500 Hz and
  1000 Hz octave T30). ISO 3382-1/-2 are characterisation standards with no
  intrinsic pass/fail, so the verdict row appears only when a target T_mid is
  supplied through `ReportMetadata.requirement` (read as the maximum acceptable
  T_mid). Uses the same `ReportMetadata` container and reportlab engine as the
  ISO 717 / ISO 11654 fiches, in English or Spanish (`language="es"`). The
  committed example fiche renders an auditorium octave-band measurement whose
  synthetic single-slope decay fixes the reverberation-time columns at their
  closed-form values (T20 = T30 = EDT = T per band, since a pure exponential
  energy decay gives the straight Schroeder line L(t) = -60 t / T of
  ISO 3382-1:2009, 5.3.3).
- `ReportMetadata`: three room-acoustics fields, `room_volume` (the volume of
  the single room under test, in m^3; ISO 3382-2:2008 Clause 9), and the
  `source_positions` / `receiver_positions` counts (ISO 3382-1:2009 Table 1 and
  ISO 3382-2:2008 Clause 8), populating the room fiche header. `room_volume` is
  validated as a finite positive number and the position counts as positive
  integers, matching the existing field-validation style.

### Changed
- Docs materials reference: the Materials and Surfaces theory page now cites
  three open-access works by Jiménez and co-workers alongside Cox & D'Antonio,
  the metadiffuser paper (Jiménez, Cox, Romero-García and Groby, 2017,
  Scientific Reports) on the surface-diffusion section and the two
  critical-coupling perfect-absorption papers (Jiménez, Groby, Pagneux and
  Romero-García, 2017, Applied Sciences; Jiménez, Romero-García and Groby,
  2018, Acta Acustica) on the material-characterisation section, and the
  `scattering_diffusion` and `porous_absorber` module docstrings reference the
  same works. The bibliography page gains the edited volume Acoustic Waves in
  Periodic Structures, Metamaterials, and Porous Media (Jiménez, Umnova and
  Groby, Eds., 2021, Springer) as a metamaterials companion to Cox & D'Antonio.
  Mirrored in the Spanish pages.
- ISO 16251-1 floor-covering impact improvement is now anchored on a real
  measurement. The conformance report and the laboratory-insulation guide use
  the improvement of a textile carpet on the CSTB heavyweight mock-up, taken
  from Foret, Chéné and Guigou-Carter (Forum Acusticum 2011): the per-band ΔL
  was read from the figure's vector chart to +/- 0,5 dB and rates to the
  paper's published ΔLw = 29 dB (the companion ISO 140-8 series reads back to
  its published ΔLw = 30 dB, confirming the calibration). A new conformance
  row and a clean-room test assert the rating, the worked example and its
  figure and example report now show the measured carpet, and the guide cites
  the source (mirrored in Spanish).
- ISO 17497-2 directional diffusion coefficient: the conformance suite and the
  Surface Scattering guide now use a real polar response instead of the former
  hand-made four-level pattern. The oracle is a boundary-element (COMSOL)
  free-field prediction of an N = 7 quadratic-residue diffuser and its flat
  reference panel at 1000 Hz, normal incidence, reduced to the standard
  37-receiver single-plane semicircle (5 deg spacing) in the plane of maximum
  diffusion. It gives the directional diffusion coefficient d_theta = 0.7572
  (QRD) and 0.1391 (flat), and the normalised d_theta,n = 0.7180 (Formula (7)),
  reproduced to 1e-6, with clean-room tests re-deriving Formula (5) from the
  levels independently.
- Docs theory reference: the Vibration page now lists in its `references`
  frontmatter the four structure-borne sources cited on the point-mobility and
  radiation-efficiency section (Cremer, Heckl and Petersson, Structure-borne
  sound, 3rd ed.; Hopkins, Sound insulation; ISO 7626-1:2011; and
  ISO/TS 7849-1:2009 and ISO/TS 7849-2:2009), and the theory index gains the
  two missing table-of-contents anchors for that page (the point-mobility
  section and the multiple-shock subsection). Both changes are mirrored in the
  Spanish pages.
- Raised the `scipy` floor in `pyproject.toml` to `scipy>=1.18.0`, the minimum
  supported scipy version, matching the floor `requirements.txt` already
  carried. The
  `numpy` floor stays at `>=2.4.4` because the optional `perf` extra pulls
  numba, which still requires NumPy 2.4 or older; the pure-Python matrix and the
  figure stack already run on the latest NumPy through their own pins.
- Spanish acoustics terminology in the figure generators now matches the
  official UNE-EN wording used by the `.plot()` string tables: "reducción
  sonora" becomes "reducción acústica" (sound reduction index R, UNE-EN
  ISO 717-1), "reducción vibracional" becomes "reducción de vibraciones"
  (vibration reduction index Kij, UNE-EN ISO 10848), and "potencia sonora"
  becomes "potencia acústica" (sound power level LW, UNE-EN ISO 3740 series).
  The affected Spanish figures and the Spanish guide prose describing them
  were regenerated and corrected to match.
- Plot readability: continuous logarithmic frequency axes now label the octave
  band centres present in the data range (16 Hz, 31.5 Hz, ..., 16 kHz shown as
  `1k`, `2k`, ...) with unlabelled one-third-octave minor ticks, replacing the
  `10^2` / `10^3` power-of-ten ticks that read poorly in acoustics. A shared
  `format_frequency_axis` helper drives the change across every `.plot()`
  renderer and the documentation figures; time, distance, harmonic-order,
  Fresnel-number, flow-resistivity, modulation-frequency and normalized-
  frequency axes are unaffected.
- Docs prose: swept the remaining em-dashes out of the editorial text across
  the GitHub guides, the EN and ES site guides, theory chapters, landings and
  frontmatter descriptions, rewriting each sentence with standard punctuation
  while preserving em-dashes inside official standard titles, citations and
  table placeholders.
- Docs site build and assets: a postbuild step recompresses the raster images
  the site serves (the social card drops from 663 KiB to 337 KiB, sources
  untouched), prerendered pages render four at a time, the pa11y audit covers
  the six migrated guide URLs, and the programme-loudness page titles were
  shortened to satisfy the html-validate long-title gate.
- The multiple-shock whole-body vibration module (ISO 2631-5:2018) now
  documents its axis scope: the spinal-response model is vertical-axis only by
  design (clause 4a neglects the horizontal contributions to spinal
  compression), and horizontal whole-body exposure should be assessed with the
  ISO 2631-1 weighted r.m.s. and VDV metrics already in the vibration domain.
  Docstrings and the guide (EN + ES) gain the clarification; no behaviour
  change.
- Dependency refresh across the toolchains: the development tools move to
  ruff >=0.15.21, mypy >=2.3.0 and types-setuptools >=83.0.0.20260716; the
  docs site moves to Astro 7.0.9 and html-validate 11.5.6; the docs deploy
  workflow moves to actions/setup-node v7. The pinned figure-rendering stack
  (`requirements-figures.txt`) was already at the current releases, so the
  committed figures are unchanged (verified with a full regeneration), and
  the runtime floors (numpy/scipy, optional matplotlib/numba) stay put since
  no newer APIs are required.
- The raster documentation figures (the seven `_PNG_FIGURES`: spectrogram,
  excitation signals, Schroeder decay, calibration stability, impulse
  response, numerical propagation and airport contour, across the EN/ES x
  light/dark variants) now render at 300 DPI instead of 150, doubling their
  pixel dimensions so they stay crisp on high-density displays and when
  zoomed. The docs embeds size them by percentage width, so no layout change
  was needed. Each PNG is also losslessly recompressed at save time with a
  deterministic `oxipng` pass (via the pinned `pyoxipng==9.1.1`), which
  shrinks the files ~25-30% without changing a single pixel; because the
  optimization is a pure function of the input, `make graphs` stays
  byte-reproducible and `scripts/check_figures.py` still matches a fresh
  regeneration.
- The documentation animations now render at 300 DPI (2400x1350, from
  800x450) for crisp playback on high-density displays, keeping the same
  8x4.5 figure so no layout re-tuning is needed. The VP9 WebM encode gains
  screen-content tuning and row multithreading (`-tune-content screen
  -row-mt 1 -deadline good -cpu-used 4`), which roughly halves the encode
  time at the same file size; the GitHub GIF fallback stays at 640 px. All
  16 clips (four language x theme variants each) now render on a spawned
  process pool, one grouped task per clip so each FDTD field is simulated
  once and reused across its variants; `make animations` and the `--jobs`
  flag drive it, while `--anim NAME` keeps the single-clip re-render
  sequential. The site `<video>` embeds and posters follow the new intrinsic
  size. A handful of on-canvas readability fixes ride along: Spanish decimal
  commas in the FDTD time readouts, repositioned time and caption labels so
  they clear tick labels and legends, a legible dark pill behind the ducting
  channel-axis label, a mathtext `L_W` subscript in the two-rooms readout,
  and the onset magnifier kept fully inside its panel.
- The four Tier-1 documentation animations are redesigned as mechanism
  infographics instead of line-plot reveals: `anim_time_weighting` shows the
  tone burst driving the IEC 61672-1 RC detector (square-law rectifier,
  capacitor charging/draining, F/S/I meter needles), `anim_onset_detection`
  scans L_AF with a magnifier and lights the NT ACOU 112 OR/LD/P/KI decision
  chain, `anim_instantaneous_intensity` animates a two-microphone p-p probe
  with p/u phasors and a flipping intensity arrow (active vs reactive), and
  `anim_schroeder` fills the tail energy of the squared impulse response
  while the decay curve and T20/T30 fits emerge on a companion axis. A
  shared scaffold (schematic axes, mic symbols, meter gauges, pipeline
  boxes, updatable arrows) backs all clips; filenames and the WebM/GIF
  variant pipeline are unchanged. Every animation embed now loads deferred
  with its space pre-reserved: `_save_animation` exports a poster still per
  WebM variant (`anim_<name>[_es][_dark]_poster.jpg`, a frame inside the
  closing hold; JPEG so the posters stay outside the SVG/PNG `make graphs` /
  `check_figures.py` pipeline, refreshable via `make posters` /
  `--posters`), the site `<video>` embeds use `preload="none"` plus the
  poster and explicit `width`/`height` (no `autoplay`), and the GitHub-docs
  GIF embeds carry `loading="lazy"` with their intrinsic size, so no clip
  fetches eagerly and none shifts the layout while loading. Future clips
  inherit the convention automatically.
- `scripts/generate_graphs.py` renders the documentation figures on a
  process pool: each (figure, language, theme) combination is an independent
  task (spawned workers, language/theme applied per task), except the five
  psychoacoustic figures whose `lru_cache`-memoised ECMA-418-2 / ISO 532
  analyses render their four variants grouped in one worker so the cache
  reuse of the sequential run is preserved. Output is byte-identical to the
  sequential run (all 624 committed assets hash-equal on the same machine);
  the full four-variant regeneration drops from 196 s to 53 s on a 12-core
  dev box. New CLI flags: `--jobs N` (default: cores minus two, capped at 8;
  `--jobs 1` keeps the old in-process sequential path) and `--figure NAME`
  (repeatable single-figure filter); `--animations` / `--all` are unchanged
  and the clips still render sequentially.
- ECMA-418-2 chain: the per-lag Python loop of the unbiased ACF
  normalization (Formulae 27-29) is now column-vectorised, the Clause 6.2.3
  neighbour-band averaging accumulates without the stacked copy, the ACF
  normalization reuses the block-RMS energy array, and the roughness
  Clause 7.1.7 pchip
  interpolation runs all 53 bands in one call. Outputs are bit-identical
  (guarded by bitwise equivalence tests against retained reference loops and
  the golden baseline); on the 0.5 s bench signal `loudness_ecma` drops from
  2.78 s to 2.00 s, `tonality_ecma` from 2.74 s to 2.01 s and
  `roughness_ecma` from 0.24 s to 0.23 s. The remaining runtime is the
  standard-mandated 16384-point DFTs, which no bit-preserving reformulation
  (rfft, one-sided magnitudes, batched or alternative FFT backends) can
  reduce.
- ECAC Doc 29 `noise_contour` now evaluates the grid with one vectorised pass
  per flight-path segment instead of a per-point Python loop; numerically
  identical (guarded by the golden baseline and a scalar-equivalence test)
  and ~45x faster on small grids, several hundred times on production-size
  grids (30 000 points in ~0.2 s).
- The test suite now runs in parallel: `make test`, `make coverage` and the CI
  test job invoke `pytest -n auto` (pytest-xdist, added to the dev
  requirements) so the 3241 tests fan out across every CPU core. Each worker
  pins its numerical thread pools to a single thread
  (`OMP_NUM_THREADS=1` and the MKL/OpenBLAS/NumExpr/Accelerate equivalents) so
  the per-core workers do not oversubscribe the machine with nested BLAS
  pools. On a 12-core host the full suite drops from 508 s to 138 s wall-clock
  (about 3.7x); pytest-cov combines the per-worker coverage so the Codecov
  upload is unchanged (identical 96% line coverage, same 13914 statements).
  Two supporting test changes: the Annex C.1 Moore-Glasberg time-varying
  loudness checks share one memoised result per distinct tone (the phon and
  sone suites analysed the same signals independently), and the ECMA-418-2
  roughness peak/depth checks reuse the module calibration result instead of
  recomputing it - no oracle, tolerance or covered line changes. The pink-noise
  flatness test no longer writes a throwaway debug PNG to a fixed path (it
  raced under parallel workers and left an untracked file); its numerical
  assertion is untouched.

### Fixed
- The whole-body daily vibration exposure fiche labels its per-operation
  magnitude `a_w,max` per Directive 2002/44/EC Annex Part B point 1 (the
  highest frequency-weighted axis value `max(1,4*a_wx, 1,4*a_wy, a_wz)`)
  instead of presenting it as the ISO 2631-1 Eq. (10) vector total `a_v`, and
  a printed note states that basis (EN/ES). The measurement-chain diagram and
  the human-vibration guide (EN/ES) state the same Part B basis; the hand-arm
  side keeps the Part A vector total `a_hv`.
- The boxed exposure zone of the daily vibration exposure fiche is now derived
  from the same displayed-rounded comparisons as the assessment rows and the
  verdict, so an `A(8)` that prints exactly at a threshold (for example 4,997
  displaying as 5,00) can no longer show a boxed zone one step below its own
  Exceeded row.
- The ISO 10848 `Kij` fiche distinguishes an empty single-number mean caused
  by every in-range band being bracketed for poor modal overlap (M < 0,25)
  from a spectrum with no bands in the Annex A range (EN/ES).
- Spanish ISO 2631-5 outputs: the injury-probability plot title translates the
  subject sex ("hombre"/"mujer") and the fiche's clause references print
  "Fórmula (n)" instead of the untranslated "Formula (n)".
- `shaped_sweep_signal()` no longer returns a NaN or infinite crest factor
  when a very small `seconds` next to a dominant `start_delay` rounds the two
  edges of the constant-envelope window onto the same sample and leaves an
  empty core slice: it now falls back to the whole retained sweep in that
  degenerate case.
- The shaped-sweep spectrum plot no longer crashes when the Welch grid,
  resolved independently of the synthesis grid, leaves no bin inside a narrow
  `(f1, f2)` band on a short signal: the Welch magnitude now falls back to its
  overall positive-frequency maximum as the normalization reference so the plot
  still renders.
- The `regularized_inverse_filter()` result documents `max_gain_db` as the
  achieved peak-normalized out-of-band gain the code actually returns (the
  computation is unchanged and stays conformance-pinned), and its `plot()`
  return type is narrowed to a single `Axes`.
- The documentation figures now render every text run in the viewer's native
  sans-serif font: I normalise all `font-family` declarations in the generated
  SVGs to the generic `sans-serif`, so viewers without DejaVu Sans no longer
  fall back to a serif for the mathtext labels while the surrounding prose
  stays sans. Text remains selectable.
- The EBU R 128 programme-loudness fiche no longer passes the verdict on the
  obsolete blanket ±0.5 LU tolerance (the June 2014 V3 rule, superseded in
  the August 2020 V4 revision and absent from R 128-2023).
  `ProgramLoudnessResult.report()` now selects the tolerance explicitly via a
  `tolerance` keyword: the default `"qc"` applies the ±0.2 LU
  measurement-error allowance of R 128 item i) (loudness workflows such as
  Quality Control) and `"live"` applies the ±1.0 LU tolerance of item h),
  permitted only where the Target Level is not achievable practically; the
  fiche prints the applied rule and its R 128 item. A non-live programme at
  −23.5 LUFS, previously rendered PASS, now correctly fails the default rule.
- Fiche verdicts are now evaluated on the value rounded exactly as displayed,
  so the printed numbers can no longer contradict the pass/fail at a
  tolerance boundary: the programme-loudness delta is compared at the 0.1 LU
  display precision of EBU Tech 3341, the EPNL is determined once to one
  decimal place (as Annex 16 states it) before the level, margin and status
  are derived, and the ISO 532-1 loudness verdict compares the one-decimal
  displayed value. Tiny negative values no longer print as "-0.0".
- `effective_perceived_noise_level()` gains a `procedure` keyword
  (`"aeroplane"` default, `"helicopter"`) that plumbs the tone-correction
  start band of ICAO Annex 16 Vol I App. 2, 4.3.1 Step 1 through the public
  EPNL chain: helicopters and tilt-rotors start the slope analysis at the
  50 Hz band, so rotor tones in the 50-80 Hz bands are no longer silently
  outside the analysis when the spectrum is processed with the default
  aeroplane procedure.
- The ISO 717 fiche now declares the band set actually rated (an octave-band
  rating was captioned "One-third-octave", contradicting the clause 4.4
  statement requirement) and can name the reported single-number quantity via
  a `symbol` keyword on `report()` (e.g. `"DnT,w"`, `"L'nT,w"`), so a field
  measurement is no longer mislabelled with the laboratory `Rw` / `Ln,w`
  descriptor; spectrum-adaptation terms print a sign only when negative,
  matching the standard's own examples.
- The ISO 532-1 fiche now reports the items clause 7 makes mandatory: the
  method used (stationary, clause 5, or time-varying, clause 6), the sound
  field (`ZwickerLoudness` gains a `field` attribute set by the
  constructors), the loudness-versus-time function N(t) for time-varying
  sounds and the "maximum loudness Nmax" labelling of the reported maximum.
- `verify_filter_class()` / `filter_class_compliance()` now report when the
  verification is range-limited: the stop-band mask beyond each band's
  processing Nyquist cannot be exercised at the decimated rate (the
  `range_limited` flag and per-band `checked_to_omega` record it), and the
  IEC 61260-1 fiche qualifies its COMPLIES statement accordingly instead of
  asserting unqualified full-mask conformance. The fiche also labels bands
  with the nominal mid-band frequencies both editions identify filters by
  (125 Hz, not the exact 125.89.. Hz), cites the 1995 edition's own
  relative-attenuation definition instead of a 2014-only formula number, and
  rejects a `required_class` that does not exist in the verified edition
  (class 0 against a 2014 result previously rendered a meaningless FAIL).
- Spanish fiches (`language="es"`) now localise the embedded chart too: the
  fiche language is forwarded to the result's own `plot()`, so axis labels
  and legends render in Spanish instead of English.
- Fiche header grids reprint client-supplied metadata verbatim instead of
  display-rounding it (an area of 1.23 m² was printed as "1.2"), and the
  ISO 11654 fiche writes the shape indicator in the clause 5.3 style
  (`0.60(M)`, no space) with the 5.3 NOTE recommendation printed when an
  indicator applies.
- `LoudspeakerCharacteristics.minimum_impedance` now evaluates the lowest
  impedance modulus over the rated frequency range when
  `rated_frequency_range` is supplied, as IEC 60268-5 clause 16.1 requires
  ("the lowest value of the modulus of the impedance in the rated frequency
  range shall be not less than 80 % of the rated impedance"), falling back to
  the computed effective frequency range only when no rated range is stated.
  The property previously always scanned the effective range, which clause
  19.1 NOTE 2 warns may differ from the rated range (particularly for woofers
  and tweeters), so an impedance dip below 80 % of the rated impedance inside
  the rated range but outside the effective range went undetected.
- `vibration.wave_vibration_reduction_index` now implements Hopkins (2007)
  Eq. 5.116 exactly: `Kij = 10 lg(1/tau_ij) + 5 lg(fc_j/f_ref)` with the
  reference frequency `f_ref = 1000 Hz`, instead of the previous
  `5 lg(fc_j/fc_i)` correction term. The old form broke the required symmetry
  of the junction descriptor (`Kij = Kji`, guaranteed by the Eq. 5.7
  reciprocity of the angular-average transmission coefficients): an L-junction
  of 100 mm and 215 mm concrete plates returned 6.81 dB in one direction and
  8.09 dB in the other where the correct symmetric value is 2.97 dB. The
  function signature changed accordingly (it now takes the receiving plate's
  `critical_frequency_receiver` instead of the optional source/receiver pair),
  `junction_transmission` computes both plates' thin-plate critical frequencies
  (`fc = sqrt(12) c0^2 / (2 pi h cL)`, `c0 = 343 m/s`) and stores them on
  `JunctionTransmissionResult.critical_frequency1`/`critical_frequency2`, and
  `corner_reduction_index` uses the receiving plate's value.
- The ISO 4871 declared single-number values
  (`OperatingModeDeclaration.declared_sound_power_level` and
  `.declared_emission_pressure_level`) now round the sum `L + K` once to the
  nearest decibel, as clause 3.15 defines (`L_d = L + K`, the sum of the
  unrounded quantities rounded to the nearest decibel), instead of adding the
  separately rounded `L` and `K` of the dual-number form (clause 3.16). The
  previous behaviour could understate the declared value by 1 dB (for example
  `L_WA = 91.4`, `K_WA = 2.4` gave 93 dB instead of 94 dB) and thereby flip
  the clause 6.2 verification verdict at the boundary.
- `aircraft.load_anp_database` no longer interleaves distinct fixed-point
  profiles: the parser now keys each trajectory on the full ANP identity
  (aircraft, operation, profile identifier, stage length) instead of dropping
  `Profile_ID`, so aircraft shipping several profiles for the same operation
  and stage length (e.g. `CNA206`/`CNA20T` departures with `DEFAULT` and
  `3000LB` weight variants) load each profile whole instead of a physically
  impossible 18-point merge. `AnpDatabase.profile` and `AnpAircraft.profile`
  gained an optional `profile_id=` keyword; without it the `DEFAULT` profile is
  selected when present, a single available profile is used, and an ambiguous
  request raises listing the identifiers. Duplicate or non-consecutive point
  numbers within a profile now raise instead of silently corrupting the path.
- `aircraft.event_level` and `aircraft.noise_contour` apply the ECAC Doc 29
  Vol 2 §4.5.2 bank-angle sign to the correct observer side: the depression
  angle is now `φ = β + ε` for observers to starboard (right of the flight
  direction) and `φ = β − ε` for observers to port. The sign was inverted, so
  the lateral-directivity (engine-installation) correction of every banked
  segment was mirrored between the inside and outside of a turn.
- `noise_control.silencers.transmission_loss` had the four-pole prefactor
  inverted (`Z1/Zn` instead of `Zn/Z1`), overstating the transmission loss by
  `20 lg(S_out/S_in)` whenever the inlet and outlet port areas differed: a
  zero-length element between 0.01 and 0.02 m2 (a sudden expansion, classic
  `TL = 10 lg[(1+m)^2/(4m)] = 0.512 dB`) came out as 6.532 dB, and the TL of
  an unequal-port chamber was not reciprocal (11.34 vs -0.70 dB, a negative
  loss for a passive element). The formula now follows Munjal, *Acoustics of
  Ducts and Mufflers* 2e, Eq. (3.27), reproduces the sudden-expansion limit
  exactly in both directions and is reciprocal; regression tests pin both.
  The four shipped silencer builders call it with equal port areas, so their
  results are unchanged. Bies 5e Eq. (8.141) as printed also fails the
  sudden-expansion limit (1.938 dB) and is now registered in
  `docs/ERRATA.md`.
- `environmental.ground_effect`, `environmental.spherical_reflection_coefficient`,
  the ground-reflected paths of `environmental.barrier_insertion_loss` and the
  ground condition of `environmental.atmospheric_parabolic_equation` now
  conjugate the impedance obtained from the porous models of
  `phonometry.materials` (the `flow_resistivity=` path and any
  `PorousMediumResult`). The materials domain works in the `e^{+jωt}` time
  convention (`Im(Z) < 0` for a passive medium) while the Salomons formulas
  behind these functions require `e^{-iωt}` (`Im(Z) > 0`); the impedance used
  to cross that boundary unconjugated, so the phase of the spherical-wave
  reflection coefficient `Q` was wrong and the soft-ground interference dip
  almost vanished (Salomons Fig. D.3 geometry, grassland
  `σ = 200 kPa·s/m²`, `hs = hr = 2 m`, `r = 100 m`: the library gave a
  `-0.55 dB` minimum where the correct dip is `-12.7 dB` near `395 Hz`, now
  pinned by test and conformance oracle). User-supplied `impedance=` values are
  now documented as `e^{-iωt}` with `Im(Z) > 0` for a passive ground (the
  docstrings previously asserted the opposite sign), matching the convention of
  `aircraft.rotorcraft_noise.ground_effect_adjustment`.
- `environmental.atmospheric_parabolic_equation` adds the surface-wave term of
  the GFPE range step (the residue of the ground-reflection pole at
  `kz = -k0/Z`, third term of Salomons Eq. (H.49)), which is active for a
  passive finite-impedance ground; without it the homogeneous-atmosphere field
  drifted by up to about 1 dB from the Weyl-Van der Pol oracle once the
  impedance convention was corrected.
- `building.critical_frequency` (flanking, ISO 12354-1:2017 Formula (20)) dropped
  a spurious factor of pi that made the thin-plate critical frequency about 3.14
  times too low (for example roughly 670 Hz instead of about 2100 Hz for 6 mm
  glass). The value now matches the closed-form plate coincidence frequency
  `coincidence_frequency` to within the rounding of the 1.8 constant.
- Reverberation-time prediction no longer rejects absorption coefficients
  at or above 1 for every model with a blanket `[0, 1)` validator; each
  model now enforces its own mathematical domain. Sabine accepts measured
  ISO 354 coefficients at or above 1 (its linear absorption area stays
  finite), including the exact 1.0 of the ISO 11654 practical-coefficient
  cap, up to a unit-error guard at 2 that catches percentages passed
  where fractions are expected. Eyring accepts individual coefficients at
  or above 1 as long as the mean absorption entering `ln(1 - mean)` stays
  below 1, and Millington-Sette keeps the strict per-surface requirement
  (every coefficient below 1) that its per-surface logarithm demands;
  Fitzroy and Arau-Puchades keep requiring each wall-pair mean below 1,
  since their inputs enter the axial logarithm directly. NaN inputs, which
  the old range comparisons silently let through both as coefficients and
  as `air_attenuation`, are now rejected with explicit finiteness checks.
- Documentation site rendering on phones: wide parameter tables (4+
  columns with a long-text notes column) no longer crush every column to a
  few characters per line with enormously tall cells; on narrow screens
  their cells keep a readable minimum width (16rem for the trailing notes
  column) and the table scrolls horizontally inside Starlight's scroll
  container instead (new `site/src/styles/theme-tables.css`, desktop
  rendering unchanged); these wide tables also get `tabindex="0"` and a
  visible focus outline so the scroll container stays keyboard-reachable
  on Safari/WebKit, which unlike Chrome and Firefox does not focus
  scrollable regions by default. Figure widths on phones now cover the whole
  authored tier inventory with a generic selector: every
  `<img style="width:NN%">` and video spans the full reading column on
  screens up to 50rem, where tiers such as 60/70/72/75/78/82/86/90/94/96%
  previously kept wasted side margins and left axis labels illegible.
- G-weighting (ISO 7196) digital design is now sample-rate aware: at the low
  sample rates common for infrasound recordings, 315 Hz approaches Nyquist
  and the un-prewarped bilinear design warped the upper response; the design
  now oversamples toward 48 kHz so the response at 315 Hz stays within a few
  hundredths of a dB of the 48 kHz design at any rate (at 48 kHz and above
  the design is unchanged).
- STI analysis filter bank is now zero-phase (forward-backward filtering):
  the single-pass bank failed two normative verification criteria of
  IEC 60268-16:2020 on the certified test bench - the C.4.2 filter-slope
  test (an unmodulated tone one octave below the observed band, 41 dB
  louder, leaked with only ~31 dB attenuation and dragged the measured m
  down to ~0,1 against the m ≥ 0,5 criterion) and the A.3.1.2
  phase-distortion limit (STI bias reached -0,020 at TI = 0,9 against the
  < 0,01 criterion). Zero-phase filtering removes the phase bias by
  construction and doubles the effective skirt attenuation to ~63 dB one
  octave out; all 49 certified bench signals now pass (worst slope-test
  m = 0,937, worst edge-carrier bias -0,0029). STI values on ordinary
  signals of the recommended >= 15 s duration shift by at most ~0,003
  (larger only below the minimum-duration warning threshold; the
  sti_vs_t60 figures are regenerated accordingly).
- `stipa` no longer raises when an octave band of the recording carries no
  energy: dead bands read m = 0 (TI = 0, the worst-case reading of a real
  meter) under an `STIWarning`, so component-level verification signals
  that deliberately occupy only some bands (IEC 60268-16 C.4.2) remain
  measurable.
- DIN 45681 / ISO PAS 20065 single-line tones: the tone level of a K = 1 run
  is the line level itself (Formula (7)); the Hanning bandwidth correction
  applies only to K > 1 sums (Formula (8)). The unconditional -1,76 dB
  understated every single-line audibility and could flip weak
  near-bin-centred tones to inaudible.
- `verify_weighting_class` and the conformance report's A/C branch evaluate
  the SOS at the exact base-10 frequencies behind the Table 3 nominal labels
  (Table 3 NOTE; IEC 61672-3 13.3), removing a pairing bias of up to
  0,27 dB; `verify_filter_class` evaluates the Table 1 breakpoints exactly
  with `sosfreqz` instead of interpolating off the grid (the 16-point floor
  now reproduces the dense-grid verdict).
- IEC 61400-11 tone classification anchors the 10 dB cut, the reported tone
  frequency and the Formula 34 criterion to the highest classified tone
  line (9.5.3/9.5.4), not the probed candidate; truncated critical bands
  warn and candidates below 20 Hz are rejected.
- ISO 1996-2 residual correction: with a margin of 3 dB or less no
  correction is allowed and the uncorrected level is the reportable upper
  bound (the previous wording inverted the bound);
  `gaussian_residual_level` rejects inverted percentile orderings.
- ISO 9613-2 `atmospheric_absorption` evaluates alpha at the exact base-10
  midbands (the Table 2 convention; ~1,3 % high at 8 kHz before); grazing
  barrier geometry (z = 0) now yields Dz = 10 lg 3 instead of 0.
- GUM `combine_uncertainty`: a correlated budget with finite input degrees
  of freedom carries undefined effective dof (the GUM defines no correlated
  Welch-Satterthwaite form; r = 0,999 with nu = 5 previously produced
  k95 ~ 2e151) and `expanded()` then requires an explicit coverage factor;
  the finite-difference step is the input uncertainty with a 64-ULP floor,
  preserving locality for large-magnitude inputs while a tiny uncertainty
  on a large value no longer underflows to uc = 0; `monte_carlo` rejects
  trials < 2.
- NT ACOU 112: level rises with onset rates at or below 10 dB/s no longer
  produce a KI adjustment (clauses 4.5/8).
- `sensitivity()` rejects non-finite reference samples (a NaN propagated
  silently to a NaN calibration factor); docstring thresholds aligned with
  IEC 60942:2017 Table 2 and the plain-bilinear weighting class notes with
  the measured behaviour (class 1 holds at 44,1/48 kHz, degrades at
  fs <= 32 kHz).
- SII (ANSI S3.5-1997): the equivalent disturbance spectrum level is the
  clause 5.6 maximum of the equivalent masking and internal noise levels,
  not their energy sum; the index for a hearing-impaired listener in
  moderate noise was underestimated by up to 0,042 (0,1841 vs the
  standard's 0,2185 for normal speech in flat 30 dB noise with a flat
  40 dB loss). Confirmed against the official worksheet (=MAX in every
  band) and the R CRAN worked example.
- IEC 60268-3 intermodulation reworked to the clause-exact quantities:
  `modulation_distortion` reports the per-order arithmetic sideband sums
  over the f2 output (14.12.7.2 g-h) instead of one combined RMS;
  `difference_frequency_distortion` references U_2,ref = 2 U_2,f2 and sums
  the third-order products arithmetically (14.12.8.1) instead of halving
  the reference (+6 dB on d2, +3 dB on d3); and
  `total_difference_frequency_distortion` implements 14.12.10.1 directly
  (only the two in-band products over the tone-amplitude sum, standard
  8 kHz / 11,95 kHz tones as defaults) instead of sqrt(d2^2 + d3^2).
- Intermodulation product search windows no longer swallow neighbouring
  components: octave-spaced clean tones read 0 instead of d2 = 1,0, DC
  offsets no longer leak into d3, out-of-band products clamp to
  (0, Nyquist), and the two TDFD products can no longer double-count.
- THD+N/SINAD measured noise over the full Nyquist band: at fs = 192 kHz
  white noise read +7 dB versus the AES17 band-limited value and a DC
  offset was counted as noise; both now pass through the AES17
  measurement-bandwidth chain by default.
- The AES17 standard notch was designed at the nominal Q but applied
  zero-phase (filtfilt squares the response), so the applied quality
  factor was 0,644x nominal (a requested 2,0 acted as 1,29; requests below
  ~1,87 silently fell outside the AES17 [1,2, 3] range). The design Q is
  now compensated by sqrt(1 + sqrt(2)) so the applied response has the
  requested quality factor.
- `thd` returned 0,0 when no harmonic of the fundamental fitted below
  Nyquist (for example a 20 kHz fundamental at fs = 48 kHz); it now raises,
  and captures shorter than 64 samples are rejected.
- THD/SINAD documentation labelled both THD conventions as the
  IEC 60268-3 14.12.2-3 quantity and cited AES17 for SINAD; the R
  convention is now identified as the 14.12.3.2 formula and SINAD as
  derived from the AES17 THD+N (AES17 does not define SINAD).
- Psychoacoustic annoyance combined the weightings as
  N5*sqrt(1 + wS^2 + wFR^2) where Fastl & Zwicker Eq. (16.2) prints
  PA = N5*(1 + sqrt(wS^2 + wFR^2)); every PA with a nonzero
  sharpness/fluctuation/roughness contribution was 13-29 % low, and the
  worked reference value had been hand-computed with the same wrong form.
  The worked tuple now anchors PA = 37,0478.
- ECMA-418-2 roughness omitted the mandatory Clause 5.1.2 5 ms fade-in and
  its calibration signal was synthesized with the carrier (not the overall
  signal) at 60 dB SPL. With both corrected the chain reproduces the
  Clause 7 reference to 0,99990 asper with the tabulated c_R; the former
  "+7,35 % clean-room methodology variance" narrative was wrong and is
  removed everywhere.
- ECMA-418-2 loudness skipped the cross-block-size-group ACF recomputation
  of Clause 6.2.3 (tonal content near the block-size boundaries was
  mis-stated by up to -9,5 %); loudness and tonality now share one full
  Clause 6.2.3 tonal/noise split and report identical N'_tonal. The 1 kHz /
  40 dB calibration computes 0,9845 sone_HMS with the verbatim c_N; the
  residual's origin is documented in the module. The common time grid is
  sized from the 1024-sample stage (no more trailing edge-held samples) and
  the tonality band-range arguments are validated per Formulae 56/57.
- Fluctuation strength transcribed the misprinted Bark constant 0,76e-4 from
  Osses 2016 Eq. (3); the Zwicker-Terhardt formula and the paper's own
  anchors require 0,76e-3. The 47 filter centres now span 50 Hz-13,2 kHz
  (was 491 Hz-20 kHz with 12 duplicate edge bands) and carrier-frequency
  behaviour matches F&Z Fig. 10.5 (C_FS = 0,279, near the paper's 0,2490).
- ISO 532-1 Annex B validation tests had silently skipped since the test
  tree reorganization (broken relative fixtures path behind a skip guard);
  the path is fixed and an in-repo presence assertion prevents a recurrence.
- DIN 45692 Aures/von Bismarck sharpness variants used self-derived
  normalization constants where Formulas (B.1)/(B.2) print the literal
  k = 0,11 (every Aures result was +4,2 % against the published formula);
  the reference sound now lands near, not exactly at, 1 acum
  (0,96/1,02 acum) and is asserted non-circularly.
- ISO 532-1 filterbank tables were mislabelled (Table A.1 is the reference
  section; the per-band deviations and gains are Table A.2), and the
  sone-to-phon mapping documents the reference-program provenance of its
  exact 10/lg 2 factor and 3 phon floor.
- Underwater Francois-Garrison absorption used the Medwin & Clay textbook
  transcription of the boric-acid factor (8.68/c); the original paper prints
  8.86/c and only that value reproduces the paper's own Table IV (the
  transcription biased boric-dominated bands at 0.6-30 kHz by up to 1.7 %).
- ISO 10846-3 indirect method returned out-of-validity results silently:
  `transfer_stiffness_indirect` now computes the per-band transmissibility
  magnitude and emits a `PhonometryWarning` where |T| > 0,1 (Inequality (2):
  DeltaL1,2 < 20 dB, outside the 1 dB / 12 % accuracy of Formula (1)),
  exposes the limit as `TRANSMISSIBILITY_LIMIT`, and documents the
  blocking-mass rigidity criterion 10 lg(m2,eff^2/m2^2) <= 1 dB
  (Inequality (3)) with the Formula (4) effective-mass measurement.
- SDOF clause attributions for ISO 7626-1: the closed-form resonator
  (H = 1/(k - w^2 m + jwc), peak mobility 1/c) was cited as "Annex A", which
  covers matrix impedance/mobility relations; module, tests, docs and
  conformance now cite it as closed form consistent with the Table 1 / 3.1.2
  definitions, and the mobility definition citation is corrected from 3.1 to
  3.1.2.
- Zero-denominator edges in the vibration FRF modules raise an explicit
  ValueError instead of propagating inf/NaN: `transfer_stiffness_direct`
  with a zero input displacement, `transfer_stiffness_level` of a zero
  magnitude, `loss_factor` of a purely imaginary stiffness, and
  `convert_frf` of a zero value through a force-per-motion reciprocal.
- Documentation: `convert_frf`/`MobilityResult.to` reciprocals are the free
  quantities of ISO 7626-1 3.1.4 (blocked matrix quantities do not invert
  element-wise; Table 1 names F/a the effective mass), the ISO 2631-5 input
  record must be conditioned (DC-removed) since the seat-to-spine transfer
  is unity at 0 Hz, and `weighted_acceleration` notes that ISO tables
  tabulate at true one-third-octave centres 10^(n/10), not nominal labels.
- ISO 10848 octave-band single-number Kij averaged the wrong range: the
  one-third-octave 200-1250 Hz mask was reused on octave centres, dropping
  the 125 Hz octave (a +0,5 dB bias per dB/octave of Kij slope); octave
  results now average 125-1000 Hz per Annex A, and `octave_bands()` validates
  that the input groups into whole octave triples.
- EN 12354-2 Formula (3) standardization used the Annex E.3 rounding
  10 lg(V/30); `standardized_impact_level` now applies the exact
  10 lg(0,032 V) form (the E.3 rounding stays documented), removing a
  constant -0,18 dB bias.
- `flanking_element` never applied the mandatory Kij,min floor of EN 12354-1
  Clause 4.4.2 / Formula (29); passing the flanking-element area now clamps
  each path automatically.
- EN 12354-4 `radiated_sound_power` no longer caps R' at 40 dB by default:
  the cap is an Annex G worked-example footnote, not part of Formula (2)/(3)
  (pass `r_prime_cap=40.0` to reproduce Annex G).
- `equivalent_impact_level` warns outside the 100-600 kg/m2 envelope of the
  EN 12354-2 Annex B relation; `structure_borne_power_level` validates the
  loss factor; the EN 12354-5 coupling terms validate source/receiver
  mobilities instead of silently returning NaN/inf;
  `single_number_uncertainty_uncorrelated` validates finiteness of both
  inputs.
- Documentation corrections: the EN 15657 Formula (14) level is the
  reception-plate injected power (not the characteristic source level; the
  EN 12354-5 chain needs the Formula (15)/(17) conversion), the EN 12354-4
  Annex E (E.2a)/(E.2b) branches do not "meet within rounding" (about
  -0,7 dB step for a square side), the `modal_overlap_factor` docstring no
  longer claims an exclusion the code did not apply, and
  `materials/dynamic_stiffness` no longer points to ISO 16251-1 (whose scope
  excludes floating floors).
#### Previously in Unreleased
- Underwater Wenz wind noise was referenced to the historical 20 µPa
  (0.0002 dyn/cm²) pressure while labelled dB re 1 µPa: every wind spectrum
  level (and the ambient composite) was 26.02 dB low and the wind/thermal
  crossover appeared at ~12 kHz instead of ~63 kHz. The rule-of-fives anchor
  is now 25 + 20·lg(20) = 51.02 dB re 1 µPa²/Hz at 1 kHz / 5 kn, matching the
  published Wenz chart; slopes unchanged, figures regenerated.
- Underwater `normal_modes` derives its default depth grid from the physics,
  restricts the eigensolve to the propagating band, discards eigenvalues
  inside the finite-difference error band about cutoff (previously a spurious
  extra mode could appear at under-resolved settings) and warns when a
  retained near-cutoff mode needs a finer grid.
- Underwater `ray_trace` evaluates the sound-speed gradient exactly per
  profile segment instead of smoothing kinks through an interpolated fine
  grid (turning-depth errors of metres on thermocline profiles).
- Underwater documentation accuracy: ISO 17208-2 uncertainty band edges and
  the standard's 16-20 kHz gap, Ainslie-McColm ~10 % agreement caveat at
  domain corners, the inherent Francois-Garrison A3 step at 20 °C, published
  validity domains of the sound-speed equations, wind-speed validity of the
  Wenz law, the paraxial (~±15-20°) validity of the standard PE, and stale
  cross-references; `sound_speed_profile` now evaluates in a single
  vectorised pass.
- ECAC Doc 32 rotorcraft: hemisphere gap-fill now fills the grid first and
  interpolates over all four corners (Eq. 14/15 before Eq. 13), so cells with
  some missing corners keep their measured corners instead of snapping to the
  single nearest bin; nearest-bin ties are compared through dot products
  (well-conditioned near zero angular distance); single-row/column grids are
  handled explicitly; the low-band advisory warning is no longer emitted for
  the standard 10-40 Hz NORAH bands; and the atmospheric docstring states the
  pure-tone choice and its deviation from the guidance's SAE band mapping
  (up to 2.2 dB/km at 8-10 kHz) instead of overclaiming Table 4.
- ECAC Doc 29: behind takeoff ground-roll segments the lateral attenuation and
  installation effect now use the §4.5.5 nearest-end geometry (previously the
  track perpendicular, which zeroed both terms on the extended centreline and
  overpredicted behind-the-runway levels by up to about 10 dB); runway segments
  use the Eq. 4-13b average speed (V = 0 start-of-roll waypoints are now
  accepted); the exposure elevation angle on inclined segments follows the
  equivalent-level-path convention (Fig. 4-6); maximum-level metrics use the
  nearest-end lateral geometry behind/ahead; NPD lookups clamp to the
  recommended 30 m floor (also fixes a crash for receivers exactly under the
  path). Validated against seven branch-covering receptor events of the ECAC
  Doc 29 Vol 3 Part 1 reference workbook.

## [3.2.0] - 2026-07-13

### Added

- Twelve domain namespaces: `phonometry.metrology`, `.psychoacoustics`,
  `.hearing`, `.emission`, `.room`, `.materials`, `.building`, `.vibration`,
  `.environmental`, `.aircraft`, `.underwater` and `.electroacoustics`, each a
  curated subpackage usable as `from phonometry import aircraft as air`. The
  flat top-level API is unchanged and remains the primary surface.

### Changed

- The internal package layout is reorganized into the domain subpackages
  above; every pre-3.2 flat module path (for example `phonometry.insulation`,
  `phonometry.underwater_acoustics`) keeps importing for one deprecation
  cycle and warns on attribute access. They are removed in 4.0.

### Deprecated

- The pre-3.2 flat module paths (see Changed). Pickles created with 3.2
  reference the new module paths and will not unpickle under 3.1.

## [3.1.0] - 2026-07-13

### Added

- `air_attenuation()` and `air_attenuation_m()` (with the
  `AtmosphericAbsorptionWarning`) — pure-tone atmospheric absorption coefficient
  α per ISO 9613-1:1993 (Eq. 3–5) from frequency, temperature, humidity and
  pressure; reproduces Table 1 to under 0,4 % and exposes the ISO 354 power
  attenuation coefficient m = α/(10 lg e). `exact_midband=True` snaps onto the
  Table 1 midbands.
- ISO 9613-2:1996 outdoor sound propagation: `outdoor_propagation_attenuation()`
  and `predicted_receiver_level()` with the `OutdoorAttenuation` per-term
  breakdown, plus the individual terms `geometric_divergence()`,
  `atmospheric_absorption()`, `ground_attenuation()` (general 7.3.1),
  `ground_attenuation_alternative()` / `directivity_omega()` (alternative 7.3.2),
  `barrier_attenuation()` with the `Barrier` geometry dataclass, and
  `meteorological_correction()`. `DEFAULT_FREQUENCIES` gives the nominal octave
  bands.
- `task_based_exposure()`, `job_based_exposure()` and `full_day_exposure()`
  (with `Task`, `ExposureResult` and `OccupationalExposureWarning`) — ISO 9612:2009 daily
  noise exposure LEX,8h by the three measurement strategies and the normative
  Annex C uncertainty budget (k = 1,65 one-sided 95 %, the C.6 I(I−1) vs
  C.12 N−1 sampling asymmetry, Table C.4 and the LEX,8h + U upper limit); the
  Annex D/E/F worked examples reproduce to the standard's printed precision.
- Documentation: new **Outdoor Sound Propagation** guide (ISO 9613-1/2, EN + ES),
  an **ISO 9612 occupational exposure** section in the Levels guide, theory and
  API-reference sections, four new figures each (air absorption α(f), the
  per-term attenuation breakdown and the exposure-uncertainty example) and the
  source–barrier–receiver geometry diagram. Conformance report adds seven
  ISO 9613-2 / ISO 9612 checks (41 total).

- `loudness_moore_glasberg()`, `loudness_moore_glasberg_from_spectrum()` and
  `loudness_moore_glasberg_from_third_octave()` with the `MooreGlasbergLoudness`
  result and its `.plot()` — stationary Moore-Glasberg loudness per
  ISO 532-2:2017 (level-dependent roex excitation pattern on the ERB-number/Cam
  scale, compressive specific loudness and binaural inhibition), anchored so a
  1 kHz / 40 dB SPL tone is 1.000 sone.
- `loudness_moore_glasberg_time()` and `MooreGlasbergTimeVaryingLoudness` (with
  `.plot()`) — time-varying Moore-Glasberg-Schlittenlacher loudness per
  ISO 532-3:2023: the short-term S′(t) and long-term S″(t) loudness traces, the
  peak long-term loudness N_max and percentile long-term loudness.
- `loudness_ecma()` and `EcmaLoudness` (with `.plot()`) — Sottek Hearing Model
  loudness per ECMA-418-2:2025 (sone_HMS) on the 53-band Bark_HMS auditory
  front-end; a 1 kHz / 40 dB SPL tone calibrates to ≈ 1 sone_HMS.
- `tonality_ecma()` and `EcmaTonality` (with `.plot()`) — ECMA-418-2:2025
  tonality (tu_HMS) from the autocorrelation of the band signal, with the
  time-dependent tonality T(l), the average specific tonality T′(z) and the
  per-band tonal frequency; a 1 kHz / 40 dB tone calibrates to ≈ 1 tu_HMS.
- `roughness_ecma()` and `EcmaRoughness` (with `.plot()`) — ECMA-418-2:2025
  roughness (asper), a new capability, from the band-envelope modulation
  analysis; the reference 1 kHz carrier 100 %-AM at 70 Hz, 60 dB SPL calibrates
  to ≈ 1 asper.

- One-line canonical `.plot()` method on every public result object —
  `ZwickerLoudness`, `STIResult`, `RoomAcousticsResult`, `DecayCurve`,
  `WeightedRatingResult`, `ImpactRatingResult`, `SoundPowerResult`,
  `ReverberationSoundPowerResult`, `SoundPowerIntensityResult` and
  `IntensityResult` — reproducing each result's documentation figure in a
  single call (`res.plot()`). matplotlib stays a soft dependency: importing
  and computing work without it, and `.plot()` raises a clear `ImportError`
  with `pip install phonometry[plot]` guidance when it is missing. The
  renderers create their own figure when `ax` is `None`, always return the
  Matplotlib `Axes` (an array for multi-panel figures) and never call
  `plt.show()`, so they compose into larger layouts. To feed those
  figures, `WeightedRatingResult` and `ImpactRatingResult` now also carry
  `band_centers`, `measured` and `shifted_reference`, exposing the ISO 717
  measured curve and shifted reference behind each single-number rating.
- `sweep_signal()`, `inverse_filter()`, `impulse_response()`, `mls_signal()`
  and `mls_impulse_response()` — deterministic-excitation impulse-response
  acquisition per ISO 18233:2006 (exponential sine sweep with spectral and
  Farina deconvolution, maximum-length sequences), verified by recovering
  known impulse responses at the correct sample lags.
- `decay_curve()`, `room_parameters()` and `RoomAcousticsResult` — room
  acoustic parameters per ISO 3382-1:2009 / 3382-2:2008 (Schroeder backward
  integration with noise truncation and tail compensation; EDT/T20/T30,
  C50/C80, D50, Ts with dynamic-range validity flags and curvature),
  verified against the exponential-decay closed forms within their JNDs.
- `open_plan_metrics()` and `OpenPlanResult` — open-plan-office spatial
  speech metrics per ISO 3382-3:2012 (spatial decay rate D2,S and Lp,A,S,4m
  from the log-distance regression, distraction distance rD and privacy
  distance rP from the STI-vs-distance line).
- `airborne_insulation()`, `weighted_rating()`, `energy_average_level()`
  and the `AirborneInsulationResult` / `WeightedRatingResult` dataclasses —
  field airborne sound insulation per ISO 16283-1:2014 (D, DnT, R') and
  single-number weighted ratings with C/Ctr per ISO 717-1 (reference-curve
  method), verified against the ISO 717-1 Annex C worked example.
- `impact_insulation()`, `weighted_impact_rating()` and the
  `ImpactInsulationResult` / `ImpactRatingResult` dataclasses — field impact
  sound insulation per ISO 16283-2 (standardized L'nT and normalized L'n from
  the tapping-machine impact level) and single-number weighted impact ratings
  with the CI adaptation term per ISO 717-2 (reference-curve method, octave
  −5 dB rule), verified against the ISO 717-2 Annex C examples (Ln,w = 79,
  CI = −11; octave 54, CI = 0).
- `facade_insulation()` and `FacadeInsulationResult` (with `.plot()`) — field
  façade sound insulation per ISO 16283-3:2016: the level difference D2m from
  the level 2 m in front of the façade, its standardized D2m,nT and normalized
  D2m,n forms, and the apparent element-method sound reduction index R′45°
  (loudspeaker, −1.5 dB) / R′tr,s (road traffic, −3 dB), rated to a single
  number with the ISO 717-1 airborne engine.
- `lab_airborne_insulation()`, `lab_impact_insulation()`,
  `background_correction()` and the `LabAirborneInsulationResult` /
  `LabImpactInsulationResult` dataclasses (with `.plot()`) plus the
  `LabInsulationWarning` — laboratory sound insulation per ISO 10140: the direct
  sound reduction index R (Part 2) and normalized impact level Ln (Part 3) with
  the Sabine absorption area A = 0,16 V/T (Part 4), background-noise correction
  with the 6/15 dB limit-of-measurement rule, and single-number ratings via the
  reused ISO 717-1/2 engines.
- `predicted_airborne_insulation()`, `predicted_impact_insulation()`,
  `junction_vibration_reduction()`, `junction_min_vibration_reduction()`,
  `flanking_path()`, `flanking_element()`, `combine_linings()`,
  `equivalent_impact_level()`, `impact_flanking_correction()`,
  `standardized_impact_level()` and the `AirbornePredictionResult` /
  `ImpactPredictionResult` / `FlankingPath` / `PathContribution` dataclasses —
  building acoustic performance prediction per EN 12354-1/-2:2000 (simplified
  single-number model): the apparent R′w from the direct path and the twelve
  flanking paths of four elements (Ff/Df/Fd each) with the Annex E junction
  vibration reduction index Kij and the Kij,min
  floor, and the apparent L′n,w from the bare-floor equivalent level, covering
  improvement and Table 1 flanking correction. Verified against the EN 12354-1
  Annex H.3 (R′w = 52 dB) and EN 12354-2 Annex E.3 (L′n,w = 45 dB) worked
  examples.
- `band_uncertainty()`, `single_number_uncertainty()`,
  `single_number_uncertainty_uncorrelated()`,
  `maximum_repeatability_standard_deviation()`, `coverage_factor()`,
  `expanded_uncertainty()`, `uncertain_value()`, `combine_uncertainties()`,
  `prediction_input_uncertainty()`, `reduce_by_independent_measurements()`,
  `satisfies_lower_requirement()`, `satisfies_upper_requirement()`, the
  `BandUncertainty` / `UncertainValue` dataclasses and the `COVERAGE_FACTORS`
  mapping — measurement uncertainty in building acoustics per ISO 12999-1:2020:
  the tabulated standard uncertainties for the three measurement situations
  (A/B/C, Tables 1–7 and Annex D), the expanded uncertainty U = k·u with the
  Table 8 coverage factors, and the combination, reduction and conformity rules
  (Annexes A/B/C, one-sided and two-sided).
- `absorption_area()`, `absorption_coefficient()`, `attenuation_from_alpha()`
  and `AbsorptionWarning` — sound absorption in a reverberation room per
  ISO 354:2003 (equivalent absorption area from the empty and with-specimen
  reverberation times via Sabine's equation with the air-attenuation term, and
  the plane-absorber coefficient αs, left unclamped per Clause 3.7), with
  room-volume and sample-area qualification advisories.
- `sound_power_pressure()`, `measurement_positions()`,
  `background_noise_correction()`, `environmental_correction()` and
  `SoundPowerResult` / `SoundPowerWarning` — sound power level from surface
  sound pressure per ISO 3744:2010 (engineering) and ISO 3746:2010 (survey):
  hemisphere and box measurement surfaces, background (K1) and environmental
  (K2) corrections, A-weighted total, directivity index and expanded
  uncertainty.
- `sound_power_reverberation()`, `sound_power_comparison()` and
  `ReverberationSoundPowerResult` — precision-grade sound power in a
  reverberation room per ISO 3741:2010 (direct method with Sabine absorption
  area, Waterhouse boundary correction and C1/C2 meteorological corrections;
  comparison method against a reference sound source), with room-qualification
  advisories.
- `sound_power_intensity()` and `SoundPowerIntensityResult` — sound power by
  sound-intensity scanning per ISO 9614-2:1996 (partial powers over the
  measurement surface, FpI and F+/− field indicators, per-segment
  repeatability and the per-band achieved engineering/survey grade).
- `loudness_zwicker()` / `loudness_zwicker_from_spectrum()` — Zwicker
  loudness per ISO 532-1:2017 (stationary and time-varying), a clean-room
  port of the normative reference program with the full Annex B validation
  set in CI (25 test cases, per-sample tolerance bands).
- `sharpness_din()` — sharpness in acum per DIN 45692:2009 with the Aures
  and von Bismarck variants, verified against the Table A.2 targets.
- `sti_from_impulse_response()`, `stipa()` and `stipa_signal()` — Speech
  Transmission Index per IEC 60268-16 Ed. 5 (indirect and direct STIPA
  methods, Ed. 5 male spectrum, Annex F ratings), verified against the
  standard's weighting-pair and m-mapping vectors and the Schroeder
  closed form.
- `sound_intensity()`, `field_indicators()`, `dynamic_capability_index()` —
  two-microphone p-p sound intensity per IEC 61043 (cross-spectral
  estimator, finite-difference bias correction validated against Table 3)
  with the ISO 9614-1 Annex A field indicators.
- `weighting_filter(..., curve="G")` — G frequency weighting for infrasound
  (ISO 7196:1995), verified against every Table 2 nominal response value.
- `equal_loudness_contour()`, `loudness_level()`, `hearing_threshold()` —
  ISO 226:2023 normal equal-loudness-level contours (Formulae 1-2, Table 1),
  verified against the Annex B tables.
- `tone_to_noise_ratio()` and `prominence_ratio()` — prominent discrete tone
  assessment per ECMA-418-1:2024 (clauses 10-12), including proximate-tone
  combination and the low-frequency truncated band.
- `lden()`, `ldn()`, `composite_rating_level()` — environmental noise
  descriptors per ISO 1996-1:2016 (3.6.4/3.6.5 and clause 6.5).
- `calculate_sensitivity(..., narrowband=True)` — opt-in coherent
  single-frequency (Goertzel) tone estimator that locks to the calibrator
  tone near `frequency` and rejects broadband hum/noise in the reference
  take, which otherwise inflates the RMS and biases the sensitivity by
  `-10*lg(1 + 1/SNR)` (about -0.42 dB at 10 dB SNR). The default keeps the
  legacy broadband-RMS behaviour.
- `decay_curve(..., zero_phase=...)` and `room_parameters(..., zero_phase=...)`
  — optional forward-backward (time-reversed) octave filtering permitted by
  ISO 3382-2:2008 Clause 7.3 NOTE (relaxing B*T > 16 to B*T > 4). It removes
  the octave filter's group delay before the backward integration, roughly
  halving the 125 Hz short-decay T30 bias. Default off (causal).
- `sound_intensity(..., bias_correct=True)` — optional per-bin
  finite-difference correction `(k*dr)/sin(k*dr)` (IEC 61043:1994, 7.3)
  applied to the band and broadband intensity totals so they no longer
  under-read toward `max_valid_frequency`. Default off (legacy totals).
- **STIPA (IEC 60268-16):** `stipa()` now emits a `UserWarning` for
  recordings shorter than the recommended 15 s, where the recovered
  modulation depths (and STI) are biased low.
- `tone_to_noise_ratio()` / `prominence_ratio()` (ECMA-418-1) now warn when
  the FFT resolution is too coarse to resolve the tone band (fewer than
  ~3 bins across the 15 % tone half-width), which biases TNR/PR at low
  frequencies.
- `mls_impulse_response()` (ISO 18233) now warns when the recovered impulse
  response still carries energy at the end of the MLS period, the symptom of
  an impulse response longer than one period aliasing circularly (choose a
  higher MLS order).
- `practical_absorption_coefficient()`, `weighted_absorption()` and
  `absorption_class()` — practical sound-absorption coefficient and the weighted
  rating αw with shape indicators (L/M/H) and class A–E per ISO 11654:1997.
- Impedance-tube absorption, surface impedance and transmission loss
  (`impedance_tube` module) per ISO 10534-1/-2:1998 (standing-wave and
  transfer-function methods) and the four-microphone transmission-loss method of
  ASTM E2611, with the ISO/ASTM speed-of-sound and air-density models.
- `airflow_resistance()`, `specific_airflow_resistance()`, `airflow_resistivity()`
  and `static_airflow_resistance()` — airflow resistance of porous materials by
  the steady (ISO 9053-1:2018) and alternating (ISO 9053-2:2020) methods.
- `random_incidence_absorption()`, `specular_absorption_coefficient()`,
  `directional_diffusion()` and the ISO 17497-1/-2 scattering/diffusion
  coefficients — scattering and diffusion of surfaces (ISO 17497-1:2004 random-
  incidence scattering, ISO 17497-2:2012 directional diffusion).
- In-situ road-surface absorption (`road_absorption` module) per
  ISO 13472-1/-2 (extended surface and Adrienne temporal-windowing methods) with
  the geometric-spreading and reflection-factor machinery.
- Precision sound power in an anechoic/hemi-anechoic room (ISO 3745:2012) and by
  discrete-point intensity scanning (ISO 9614-3:2002).
- Human vibration exposure (`human_vibration` module) per ISO 8041-1:2017 /
  ISO 2631-1:1997 (whole-body) and ISO 5349-1:2001 (hand-arm): the frequency
  weightings, running RMS/MTVV, vibration dose value (VDV), motion-sickness dose,
  daily exposure A(8), HAV daily exposure and lifetime VWF, and the
  Directive 2002/44/EC action/limit-value assessment.
- `spinal_response()`, `acceleration_dose()`, `daily_dose()` and `daily_dose_multi()`
  (`multiple_shock_vibration` module) — adverse health effect of multiple shocks
  on the lumbar spine per ISO 2631-5:2018 (seat-to-spine models, the Se dose and
  daily equivalent static compressive stress Sed).
- `verify_weighting_class()` — frequency-weighting (A/C/Z) conformance to the
  IEC 61672-1:2013 Table 3 class-1/2 acceptance limits.
- `verify_filter_class()` — one-third-octave and octave band-filter conformance
  to IEC 61260-1:2014 / ANSI S1.11-2004, including class 0.
- `speech_intelligibility_index()` and `standard_speech_spectrum()` — the Speech
  Intelligibility Index per ANSI S3.5-1997 (critical-band, one-third-octave and
  octave procedures) with the standard normal/raised/loud/shout vocal-effort
  speech spectra.
- `noise_criterion()` / `room_criterion()` — NC and RC Mark II room-noise ratings
  per ANSI/ASA S12.2-2019.
- `age_threshold()` and `reference_threshold()` — age-associated hearing threshold
  (median and percentiles) per ISO 7029:2017 with the ISO 389-7:2019 free/diffuse-
  field reference threshold of hearing.
- `nipts()` and `htlan()` — noise-induced permanent threshold shift and the
  hearing threshold level associated with age and noise per ISO 1999:2013.
- Measurement uncertainty (`uncertainty` module): `rectangular()`, `triangular()`,
  `u_shaped()`, `combine_uncertainty()`, `coverage_factor()`,
  `expanded_uncertainty()` and `monte_carlo()` — the GUM framework (ISO/IEC
  Guide 98-3:2008) law-of-propagation-of-uncertainty and the Supplement 1 Monte
  Carlo method.
- `impulse_prominence()`, `impulse_adjustment()` and `rating_level()` — impulsive-
  sound prominence and the LAeq adjustment/rating level per Nordtest NT ACOU 112.
- Environmental-noise determination per ISO 1996-2:2017: the tonal-audibility
  adjustment (engineering method) and the associated measurement uncertainty.
- Reverberation-time prediction from room geometry and surface absorption by the
  Sabine, Eyring, Millington-Sette, Fitzroy and Arau-Puchades models.
- Façade sound insulation and indoor→outdoor radiation prediction per
  EN 12354-3:2000 and EN 12354-4:2000.
- Sound absorption in enclosed spaces (`enclosed_space_absorption` module) —
  `equivalent_absorption_area()` and `enclosed_space_reverberation()` per
  EN 12354-6:2003 (surface and object absorption, air absorption, reverberation
  time).
- Sound insulation by the intensity method per ISO 15186-1:2000
  (intensity-based sound reduction index).
- Field survey method per ISO 10052:2021, including the Table 3 reverberation-
  time estimator.
- Sound-absorption measurement uncertainty (`absorption_uncertainty` module) per
  ISO 12999-2:2020.
- Impact-sound improvement of floor coverings on a lightweight mock-up per
  ISO 16251-1:2014.
- Laboratory flanking transmission per ISO 10848 (the vibration reduction index
  Kij from velocity-level differences).
- `apparent_dynamic_stiffness()`, `installed_dynamic_stiffness()`,
  `natural_frequency()` and `floating_floor_resonance()` — dynamic stiffness of
  resilient materials per EN 29052-1:1992.
- `sdof_mobility()`, `sdof_accelerance()`, `sdof_receptance()`, `convert_frf()`
  and `resonance_frequency()` — mechanical mobility and the frequency-response-
  function family per ISO 7626-1:2011.
- `transfer_stiffness_direct()`, `transfer_stiffness_indirect()`,
  `transfer_stiffness_level()`, `loss_factor()` and `base_transmissibility()` —
  dynamic transfer stiffness of resilient elements per ISO 10846 (direct and
  indirect/transmissibility methods).
- `radiated_sound_power_level()`, `radiation_factor()`, `mean_velocity_level()`
  and `velocity_level_from_acceleration()` — sound power radiated from a vibrating
  surface per ISO/TS 7849-1/-2:2009.
- `structure_borne_power_level()`, `reception_plate_power()` and
  `spatial_mean_velocity_level()` — structure-borne sound power of building
  equipment by the reception-plate method per EN 15657:2017.
- `installed_structure_borne_power_level()`, `installed_source_prediction()` and
  the coupling-term functions — installed structure-borne sound from building
  equipment per EN 12354-5:2009.
- `tonal_audibility()` (`tone_audibility` module) — objective audibility of tones
  in noise by the Aures method per ISO/PAS 20065:2016 (tonal audibility ΔL and
  the tonal adjustment K).
- `psychoacoustic_annoyance()` and `psychoacoustic_annoyance_from_signal()` —
  Fastl & Zwicker psychoacoustic annoyance from loudness, sharpness, roughness
  and fluctuation strength.
- `thd()`, `thd_plus_noise()`, `sinad()`, `weighted_thd()` and
  `modulation_distortion()` — harmonic and intermodulation distortion per
  IEC 60268-3 / AES17, and `transfer_function()` / `coherence()` — H1/H2 frequency-
  response and coherence estimators (Bendat & Piersol).
- Underwater acoustics reference levels and metrics (`underwater_acoustics`):
  `sound_pressure_level()`, `sound_exposure_level()`, `peak_sound_pressure_level()`
  and the in-air↔underwater conversions per ISO 18405:2017; ship radiated-noise
  source levels (`ship_radiated_noise`) per ISO 17208-1/-2 with the ISO 18406
  hydrophone-depth and uncertainty machinery; and pile-driving SEL metrics
  (`pile_driving_noise`), single-strike and cumulative.
- Underwater sound propagation: `transmission_loss()`, `spreading_loss()` and
  `seawater_absorption()` (`underwater_propagation`); `sea_water_sound_speed()`
  and `sound_speed_profile()` (`underwater_sound_speed`); and the passive/active
  `sonar_equation`.
- Underwater seabed reflection (`seabed_reflection`: `critical_angle()`,
  `reflection_coefficient()`, `bottom_reflection_loss()`), ambient noise, and
  shipping-traffic source spectra (`ship_traffic_noise`, JOMOPANS-ECHO).
- Numerical underwater propagation solvers (`numerical_propagation`):
  `normal_modes()`, `ray_trace()` and `parabolic_equation()`.
- Aircraft noise (`aircraft_noise`): `perceived_noisiness()`,
  `perceived_noise_level()`, `tone_correction()`, `epnl_from_pnlt()` and
  `effective_perceived_noise_level()` — the Effective Perceived Noise Level per
  ICAO Annex 16 Vol. I Appendix 2 — plus `verify_aircraft_noise_system()`
  (IEC 61265 measurement-system tolerances).
- Wind-turbine noise (`wind_turbine_noise`): `apparent_sound_power_level()`,
  `slant_distance()` and `wind_turbine_tonality()` per IEC 61400-11.
- `sae_band_attenuation()` — one-third-octave-band atmospheric absorption by the
  SAE Method of SAE ARP 5534:2021 (pure-tone coefficient from ISO 9613-1).
- Airport noise (`airport_noise`, ECAC Doc 29): `npd_level()` / `npd_curve()`
  noise-power-distance interpolation; the single-event `event_level()` and
  ground-grid `noise_contour()` with the segment corrections
  `impedance_adjustment()`, `lateral_attenuation()`,
  `engine_installation_correction()`, `duration_correction()`,
  `noise_fraction()` and `start_of_roll_directivity()`.
- Rotorcraft noise (`rotorcraft_noise`, ECAC Doc 32 / NORAH2): the
  `RotorcraftHemisphere` source with `hemisphere_source_level()`, and the
  propagation adjustments `spherical_spreading_adjustment()`,
  `atmospheric_adjustment()` and `ground_effect_adjustment()` (Chien-Soroka).

### Changed

- Library-wide audit: shared input-validation and energy-summation helpers,
  consistent numeric types and a common warning base; the `.plot()` convention
  completed on every public result object; concept-based renames of the public
  API carried through a deprecation cycle; the three overloaded guides split by
  topic; and full EN/ES bilingual documentation with a complete API reference.
- Documentation figures migrated to deterministic, byte-reproducible SVG (with a
  tolerance-based CI check), new explanatory diagrams and Tier-1 animations added.

- `decay_curve()` now returns a `DecayCurve` dataclass (`time`, `level`,
  `band`) with a `.plot()` method instead of a bare `(time, level)` tuple.
  Backward compatible: `time, level = decay_curve(ir, fs)` still unpacks
  because `DecayCurve` iterates as `(time, level)`.
- `lc_peak()` now polyphase-oversamples the C-weighted signal (new
  `oversample` parameter, default 8) before peak detection, recovering the
  true inter-sample peak. Sustained high-frequency tones previously
  under-read by up to ~1.15 dB at fs = 48 kHz (e.g. an 8 kHz tone, 6.0
  samples/cycle); LCpeak now tracks the analytic peak within about +/-0.5 dB.
  Reported LCpeak values shift upward for HF-rich signals; pass
  `oversample=1` for the legacy on-grid behaviour.
- `OctaveFilterBank` / `octavefilter` default `attenuation` raised from 60 to
  72 dB. scipy's `cheby2` pins the equiripple deep-stopband floor at exactly
  `attenuation`, so the former 60 dB default sat 10 dB inside the IEC
  61260-1:2014 class 1 deep-stopband limit; 72 dB makes the default cheby2
  bank class 1 (same +0.400 dB passband margin as `butter`). Numerical
  outputs change for cheby2 users — and for elliptic (`ellip`) banks, which
  consume `attenuation` as their stopband attenuation `rs` — who relied on
  the previous default.
- Weighting-filter `high_accuracy` internal oversample target raised from
  96 to 144 kHz, so fs = 48 kHz now oversamples x3 (was x2). This halves the
  high-frequency residual vs the analytic A/C curves (48 kHz: -1.11 -> -0.44
  dB @16k, -2.10 -> -0.85 dB @20k) and removes the 48k-worse-than-44.1k
  asymmetry. High-accuracy weighting outputs shift for all rates below
  144 kHz (48/96/128 kHz included) — e.g. 96 kHz by +0.86 dB @16k / +1.63 dB
  @20k — always toward the analytic curve.
- Calibrator stability validation updated from IEC 60942:2003 to
  IEC 60942:2017 (Ed. 4): the deviations |max − mean| and |min − mean| are
  each compared against the limit, using the
  frequency-dependent Table 2 class 1 limits (0.07 dB in 160 Hz - 1.25 kHz);
  `calculate_sensitivity()` gains a `frequency` parameter.
- `ln_levels()` now discards 5*tau (was 2*tau) of the time-weighting
  integrator attack before computing the percentiles. At 2*tau the F
  integrator is only 86 % settled, so the leading ramp dragged the low
  percentiles down (~0.15 dB L10-L90 spread on a steady 2 s tone); 5*tau
  cuts that ~12x and matches the skip already used by the calibration
  stability check. Percentile values on short records shift slightly.

### Fixed

- **Sharpness (DIN 45692):** the informative Aures and von Bismarck variants
  are now anchored to exactly 1.00 acum on the clause 6 reference sound via
  per-variant normalization constants (previously 0.959 / 1.019 from a
  shared hard-coded factor).
- **Loudness (ISO 532-1):** N5/N10 percentiles are now computed on the
  full-rate 2000 Hz weighted-loudness series instead of the 4x-decimated
  500 Hz trace, removing an up-to-3% decimation-phase ambiguity. The
  500 Hz `time`/`loudness_vs_time` output and Nmax are unchanged.
- **Room acoustics (ISO 3382):** T20/T30 reverberation-time validity flags
  now require 46 dB / 54 dB of dynamic range (was 35 dB / 45 dB) to
  absorb the positive bias of the tail compensation and keep a flagged-valid
  decay time within the 5% JND (ISO 3382-2 Table A.1).
- **Impulse-response (ISO 18233):** `impulse_response(method="farina")` now
  raises `ValueError` when handed a zero-padded reference sweep (which
  silently produced a wrong inverse filter) instead of returning garbage;
  pass the unpadded sweep or use `method="spectral"`.

## [3.0.0] - 2026-07-06

### ⚠️ Breaking changes

- **The library is now `phonometry`** (formerly `PyOctaveBand`). Install with
  `pip install phonometry` and `import phonometry`. The API is unchanged — a
  plain rename of the import is a complete migration. The `PyOctaveBand 2.1.0`
  release on PyPI is a transition stub that depends on
  `phonometry` and re-exports it under the old `pyoctaveband` module name with
  a `FutureWarning`, so `pip install -U PyOctaveBand` keeps existing code
  working. The last state under the old name is preserved in the locked
  [`pyoctaveband-v2`](https://github.com/jmrplens/phonometry/tree/pyoctaveband-v2) branch.
- **Python >= 3.13 required** (was >= 3.11). CI now tests 3.13 and 3.14
  (3.15 will be added once it reaches a stable release).

### Added

- `lc_peak()` — C-weighted peak level (IEC 61672-1 §5.13), verified against
  the Table 5 one-cycle/half-cycle tone-burst responses.
- `sel()` — sound exposure level (SEL/LAE), with class 1 conformance tests for
  the Table 4 LAE tone-burst column.
- `sound_exposure()` and `lex_8h()` — occupational noise dose (Pa²·h and
  normalized 8 h exposure level, IEC 61252).
- `calculate_sensitivity(..., validate=True)` — calibration-tone stability
  validation per IEC 60942 (short-term level fluctuation, class 1 limit),
  emitting `CalibrationWarning` on unstable or too-short recordings.

## [2.0.0] - 2026-07-06

### ⚠️ Numerical behavior changes

Results differ from 1.2.x for the following cases — all of them bring the
output in line with the governing standards:

- **Chebyshev II banks** now place their −3 dB points on the ANSI S1.11 band
  edges. Previously the band edges received the full stopband `attenuation`
  (−60 dB by default), underestimating broadband band levels by ~2.8 dB
  vs Butterworth. ([#58](https://github.com/jmrplens/PyOctaveBand/pull/58))
- **Bessel banks** are designed with `norm="mag"`: −3 dB lands on the band
  edges (previously ~−10 dB, a ~2.2 dB broadband error). ([#58](https://github.com/jmrplens/PyOctaveBand/pull/58))
- **A/C frequency weighting** defaults to `high_accuracy=True`: the filter is
  designed and run at an internally oversampled rate (≥ 96 kHz), keeping the
  response within IEC 61672-1 class 1 tolerances up to 16 kHz at common audio
  rates (previously −2.67 dB at 12.5 kHz for fs = 48 kHz, outside class 1).
  Stateful weighting keeps the legacy design; `high_accuracy=True` with
  `stateful=True` raises. ([#58](https://github.com/jmrplens/PyOctaveBand/pull/58))

### Added

- `verify_filter_class()`: verify a bank against the IEC 61260-1:2014
  class 1/2 acceptance limits (Table 1, transcribed from the official text)
  with per-band margins; `class_limits()` exposes the limit curves. ([#64](https://github.com/jmrplens/PyOctaveBand/pull/64))
- Standards-compliance test layer transcribed from the official texts:
  IEC 61672-1:2013 Table 4 tone-burst suite (F/S, 1 s to 1 ms) and Table 3
  full 34-frequency A/C weighting sweep within class 1 limits. ([#64](https://github.com/jmrplens/PyOctaveBand/pull/64))
- `CITATION.cff` and `.zenodo.json` metadata; Codecov coverage upload and
  quality badges. ([#64](https://github.com/jmrplens/PyOctaveBand/pull/64))
- Documentation: four new auto-generated figures (group delay, IEC
  toneburst, block continuity, IEC class mask), Mermaid diagrams, more
  equations, and a full SEO/GEO layer for the docs site (JSON-LD,
  llms.txt, AI-crawler opt-in, WCAG2AA gates). ([#66](https://github.com/jmrplens/PyOctaveBand/pull/66))
- `leq()`, `laeq()` and `ln_levels()` (L10/L50/L90 statistical levels from the
  Fast/Slow/Impulse envelope, with optional A/C weighting). ([#60](https://github.com/jmrplens/PyOctaveBand/pull/60))
- `OctaveFilterBank.spectrogram()`: short-time band levels (bands × frames),
  multichannel-aware, with bounded-memory windowing. ([#60](https://github.com/jmrplens/PyOctaveBand/pull/60))
- `zero_phase=True` option in `OctaveFilterBank.filter()` and `spectrogram()`:
  forward-backward filtering (no group delay) for offline analysis. ([#60](https://github.com/jmrplens/PyOctaveBand/pull/60))
- `TimeWeighting` stateful class for block processing (block outputs match a
  single continuous call exactly). ([#60](https://github.com/jmrplens/PyOctaveBand/pull/60))
- `weighting_filter(..., high_accuracy=...)` and
  `WeightingFilter(..., high_accuracy=...)` parameter. ([#58](https://github.com/jmrplens/PyOctaveBand/pull/58))
- PEP 561 `py.typed` marker: downstream type checkers now receive the
  library's strict annotations.
- Documentation website (English/Español) at
  https://jmrplens.github.io/phonometry/ with theme-aware auto-generated
  figures, plus topic pages under `docs/`. ([#61](https://github.com/jmrplens/PyOctaveBand/pull/61))

### Fixed

- Integer input (e.g. int16 audio from `scipy.io.wavfile.read`) no longer
  overflows silently: `time_weighting` returned negative energies and
  `calculate_sensitivity` produced factors ~315× off. All inputs are converted
  to float64. ([#58](https://github.com/jmrplens/PyOctaveBand/pull/58))
- `cheby2` with `attenuation <= 3.01 dB` now raises instead of silently
  producing NaN coefficients. ([#58](https://github.com/jmrplens/PyOctaveBand/pull/58))
- `zero_phase` no longer crashes on heavily decimated bands shorter than the
  default `sosfiltfilt` padding. ([#60](https://github.com/jmrplens/PyOctaveBand/pull/60))
- The weighting-curves documentation figure measured a truncated impulse
  response through the resampling path (~2.4 dB low); the measurement is now
  guarded by tests. ([#63](https://github.com/jmrplens/PyOctaveBand/pull/63))

### Performance

- `octavefilter()` reuses cached filter-bank designs across calls (~3.3×
  faster in loops). ([#59](https://github.com/jmrplens/PyOctaveBand/pull/59))
- `numba` and `matplotlib` are optional extras (`[perf]`, `[plot]`, `[full]`);
  a pure-Python impulse kernel and lazy plotting imports keep full
  functionality without them. ([#59](https://github.com/jmrplens/PyOctaveBand/pull/59))

## [1.2.3] - 2026-05-30

### Fixed

- Importing the package no longer forces the matplotlib Agg backend. ([#53](https://github.com/jmrplens/PyOctaveBand/pull/53))

## [1.2.2] - 2026-05-09

### Added

- Configurable time weighting initial state (`initial_state`: `'zero'`,
  `'first'`, scalar or per-channel array). ([#49](https://github.com/jmrplens/PyOctaveBand/pull/49))

## [1.2.1] - 2026-03-08

### Added

- IEC 61260-1 nominal frequency labels (`nominal=True`). ([#46](https://github.com/jmrplens/PyOctaveBand/pull/46))

## [1.1.5] - 2026-03-08

### Fixed

- Overloads, `WeightingFilter` multichannel state, version sync and SonarCloud
  CI. ([#45](https://github.com/jmrplens/PyOctaveBand/pull/45))

## [1.1.4] - 2026-03-08

### Added

- Stateful block-wise processing for `OctaveFilterBank` and
  `WeightingFilter`. ([#42](https://github.com/jmrplens/PyOctaveBand/pull/42))
- macOS and Windows in the CI test matrix. ([#40](https://github.com/jmrplens/PyOctaveBand/pull/40))

## [1.1.1] - 2026-01-07

### Changed

- DSP core enhancements: vectorization, IEC standards alignment and
  precision. ([#35](https://github.com/jmrplens/PyOctaveBand/pull/35))

## [1.0.0] - 2026-01-06

Initial PyPI release: fractional octave filter banks (Butterworth,
Chebyshev I/II, Elliptic, Bessel), A/C/Z weighting, Fast/Slow/Impulse time
weighting, Linkwitz-Riley crossover, calibration and dBFS modes,
multichannel support.

[Unreleased]: https://github.com/jmrplens/phonometry/compare/v3.3.0...HEAD
[3.3.0]: https://github.com/jmrplens/phonometry/compare/v3.2.0...v3.3.0
[3.2.0]: https://github.com/jmrplens/phonometry/compare/v3.1.0...v3.2.0
[3.1.0]: https://github.com/jmrplens/phonometry/compare/v3.0.0...v3.1.0
[3.0.0]: https://github.com/jmrplens/phonometry/compare/v2.0.0...v3.0.0
[2.0.0]: https://github.com/jmrplens/phonometry/compare/v1.2.3...v2.0.0
[1.2.3]: https://github.com/jmrplens/PyOctaveBand/compare/v1.2.2...v1.2.3
[1.2.2]: https://github.com/jmrplens/PyOctaveBand/compare/v1.2.1...v1.2.2
[1.2.1]: https://github.com/jmrplens/PyOctaveBand/compare/v1.1.5...v1.2.1
[1.1.5]: https://github.com/jmrplens/PyOctaveBand/compare/v1.1.4...v1.1.5
[1.1.4]: https://github.com/jmrplens/PyOctaveBand/compare/v1.1.1...v1.1.4
[1.1.1]: https://github.com/jmrplens/PyOctaveBand/compare/v1.0.0...v1.1.1
[1.0.0]: https://github.com/jmrplens/PyOctaveBand/releases/tag/v1.0.0
