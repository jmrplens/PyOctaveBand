← [Documentation index](../../README.md)

# Insulation design

The pages of the [Sound insulation](../insulation/index.md)
section answer the question *what does this building achieve?* The pages here
answer the one that comes first: *what will it achieve, and what should I
build?* Both halves speak the same language of $R$, $L_n$ and their weighted
single numbers, but a prediction is assembled from element data instead of
measured in a finished room, so its inputs, its assumptions and its error bars
are its own subject.

[Predicting Sound Insulation (EN 12354)](insulation-prediction.md)
is the normative model: the airborne and impact flanking transmission between
two rooms, path by path, from the direct element and the junction vibration
reduction indices $K_{ij}$. It consumes laboratory element data measured per
ISO 10140 and junction data measured per ISO 10848, both of which live in
[Sound insulation](../insulation/index.md) beside the
field measurement the prediction is checked against, and beside the façade
guide that carries the same family across the building envelope.

[Detailed Per-Band Prediction (ISO 12354)](detailed-prediction.md)
runs the same standard band by band instead of on single numbers: the
laboratory element and junction data are converted to their in-situ values,
every path is formed per band, and the result shows which path dominates each
band rather than only whether the room passes.

**Which of the two do you run?** Run the simplified model when what you have is
catalogue weighted ratings — $R_w$, $\Delta L_w$, a mass per unit area — and the
question is whether the partition meets a limit. Run the detailed one when you
have per-band element and junction spectra, or the material properties the
standard can calculate them from, and the question is *which path to fix in
which band*. The choice is not about accuracy on the rating: on the standard's
own worked building the two agree well inside their stated spread, and the
detailed airborne model carries no bias error and a standard deviation of 1,5 dB
to 2,5 dB (Clause 5) against about 2 dB for the simplified one. What the
detailed model buys is the spectrum behind the single number.

[Predicting Panel Sound Insulation](panel-sound-insulation.md)
goes one level deeper, to where the element $R$ itself comes from: the mass law
and the coincidence dip of a single panel, the plateau shortcut that estimates
the whole curve by hand, the coincidence *range* of a corrugated or ribbed sheet,
the mass-spring-mass behaviour of a double wall and the wall-tie bridge that
limits a masonry cavity one, transmission through slits and apertures, plate
radiation efficiency and point mobilities. It is the physics a catalogue value
expresses in one number.

Two pages here carry the floor half of any design, one measuring and one
predicting.
[Floor-Covering Impact Improvement (ISO 16251-1)](impact-improvement.md)
gives the weighted improvement $\Delta L_w$ of a covering that exists, on a
small heavyweight mock-up, and that is the term EN 12354-2 subtracts from the
bare-floor level.
[Predicting Resilient-Layer Performance](resilient-layers.md)
predicts it for a covering that does not yet exist, from the tapping machine's
own force spectrum, the cut-off frequency of a soft covering, the 30 lg and
40 lg floating-floor laws and the ISO 12354-1 Annex D rating of a wall lining.
Both start from the stiffness per unit area $s'$ of the resilient layer,
measured per EN 29052-1 in
[Dynamic stiffness of resilient materials](../../materials/resilient/dynamic-stiffness.md)
over in the materials section, which sets the resonance the whole improvement
hangs on.

Building service equipment is a chain of its own, and the two pages only read
correctly in order.
[Structure-borne sound power of equipment (EN 15657)](structure-borne-power.md)
characterises a pump, fan or cistern by the power it injects into the
structure, measured on a reception plate of known dissipation and then made
plate-independent.
[Installed structure-borne sound (EN 12354-5)](installed-structure-borne.md)
takes that source description, loses part of it to the coupling term the source
and receiver mobilities set, and carries the rest to a room that may be several
junctions away.

**EN 12354 or ISO 12354?** One bookkeeping note runs through the whole section.
The prediction family was published by CEN as EN 12354-1 to -6 and later
reissued by ISO as a second edition, ISO 12354-1:2017 and ISO 12354-2:2017,
which is not word for word the earlier text: where a formula changed between the
prints, the [errata registry](https://jmrplens.github.io/phonometry/reference/errata/) records both. Parts
3 to 6 keep their EN designation in the editions used here. Every guide names
the edition it was read from — EN 12354-1:2000 and EN 12354-2:2000 for the
simplified models on
[Predicting Sound Insulation](insulation-prediction.md),
including the tabulated flanking correction $K$ that the 2017 impact part
replaced with explicit per-path formulae; ISO 12354-1:2017 and ISO 12354-2:2017
for the per-band models and the Annex L/G worked examples of
[Detailed Per-Band Prediction](detailed-prediction.md);
and EN 12354-3:2000 to EN 12354-6:2003 for façades, service equipment and
enclosed spaces. Check which edition your regulation calls up before quoting a
clause or a correction from either.

And one caveat both parts print, in Clause 5: the models predict the *measured*
performance of buildings **assuming good workmanship and high measurement
accuracy**. A prediction is therefore a statement about a correctly built
construction, not about the one that will be built; the standard's own advice is
to vary the uncertain inputs and read the spread in the answer, which
ISO 12354-1:2017 Annex K systematises into an uncertainty on the result.

Every prediction here starts from measured data that came from somewhere else,
and the design report has to say where. The element $R$ and $L_n$ come from
ISO 10140-2 and -3, together with the laboratory structural reverberation time
printed in the same report, because the in-situ conversion needs it. The
junction indices $K_{ij}$ come from an ISO 10848 measurement or from the
EN 12354-1 Annex E catalogue of junction types. The floor-covering improvement
$\Delta L_w$ comes from ISO 16251-1 or from a full-size ISO 10140-3 test. The
resilient layer's $s'$ comes from EN 29052-1. And for service equipment, the
characteristic structure-borne power comes from the EN 15657 reception plate.
Two of the pages in this section are themselves such measurements, feeding the
others; the built result is finally checked against the ISO 16283 field
measurement in [Sound insulation](../insulation/index.md).

## Pages in this section

- [Predicting Sound Insulation (EN 12354)](insulation-prediction.md):
  the airborne and impact flanking models between rooms (EN 12354-1/2) with
  their junction vibration reduction indices and prediction fiches.
- [Detailed Per-Band Prediction (ISO 12354)](detailed-prediction.md):
  the per-band detailed model of ISO 12354-1/-2 with in-situ element and
  junction conversion, the flanking indices per band and the per-path
  contributions behind the rating.
- [Predicting Panel Sound Insulation](panel-sound-insulation.md):
  the mass law, the plateau shortcut and the coincidence dip of a single panel
  (Sharp, Norton), the coincidence range of a corrugated or ribbed sheet
  (Vigran/Heckl), double walls and the wall-tie bridge of a masonry cavity wall
  (Bies, Hopkins), slits and apertures (Gomperts, Wilson-Soroka), radiation
  efficiency (Leppington/Maidanik) and point mobilities (Cremer).
- [Floor-Covering Impact Improvement (ISO 16251-1)](impact-improvement.md):
  the weighted improvement of a soft floor covering measured on a small
  heavyweight mock-up.
- [Predicting Resilient-Layer Performance](resilient-layers.md):
  the tapping-machine force model, the cut-off frequency of a soft covering,
  the floating-floor improvement laws and the ISO 12354-1 Annex D rating of a
  wall lining.
- [Structure-borne sound power of equipment (EN 15657)](structure-borne-power.md):
  the characteristic power a machine injects into a building element, measured
  on a reception plate.
- [Installed structure-borne sound (EN 12354-5)](installed-structure-borne.md):
  what that power becomes once the machine is mounted on a real element, and
  the level it produces in the receiving room.

## See also

Pages elsewhere on the site that this section leans on:

- [Dynamic stiffness of resilient materials (EN 29052-1)](../../materials/resilient/dynamic-stiffness.md):
  the load-plate resonance measurement, the enclosed-gas term and the
  floating-floor natural frequency.

## What this section does not cover

**A prediction is only as good as the element data you feed it, and the library
takes that data as given.** The element ratings, the junction indices, the
covering improvement and the structure-to-airborne adjustment terms of
EN 12354-5 Annexes D and F are inputs you supply from measurement or from the
standards' own annexes; none of them is derived here. The simplified prediction
page stops at the weighted single numbers by design, and the detailed page is
where the per-band models live.

Every panel model carries a validity range it does not extend past, and the
guides flag each: Sharp's single-panel method is not valid below about 1.5
times the panel's first resonance, Gomperts' slit model holds only while the
slit is narrow against the wavelength, only Leppington's method no. 1 is
implemented for radiation efficiency, and the orthotropic routes are
infinite-panel models that miss the dip real ribbed cladding shows between 2 and
4 kHz. On the resilient-layer side, the tapping-machine force model assumes a
frequency-independent driving-point impedance, so a joisted or battened
lightweight floor is outside it; soft coverings are treated as linear springs;
there is no per-band prediction of a lining's improvement, because Annex D is a
single-number method; and heavy impact sources such as the rubber ball are not
covered by any of these models at all — their rating is
[Heavy and Soft Impact Sources](../insulation/heavy-impact-sources.md).

Two edition boundaries: only the 2009 edition of EN 12354-5 is implemented, not
the 2023 revision, and the simplified and detailed pages follow different
editions of the 12354 family, as the note above says.
