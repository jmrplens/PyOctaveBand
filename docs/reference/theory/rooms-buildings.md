← [Documentation index](../../README.md)

# Theory: Rooms and Buildings

This page collects the theory behind rooms and buildings: impulse-response measurement and the room-acoustic parameters, background-noise criteria, airborne and impact insulation with their single-number ratings and uncertainty, and flanking and absorption prediction. It is part of the [theory reference](index.md); surface scattering and acoustic material characterisation live on [Materials and Surfaces](materials-surfaces.md).

## Room noise criteria (ANSI S12.2)

ANSI/ASA S12.2-2019 rates steady background noise in rooms against families of octave-band curves (16 Hz – 8 kHz). The **NC rating** follows the two-step procedure of clause 5.2.2 on the Table 1 curves (NC-15 to NC-70): the speech interference level $\mathrm{SIL} = \tfrac14(L_{500}+L_{1000}+L_{2000}+L_{4000})$ (clause 3.2) selects the NC-(SIL) curve, and if no band exceeds it the spectrum is designated NC-(SIL); otherwise the tangency method (clause 5.2.3) applies: each measured band is interpolated against the tabulated curve values, the rating is the highest per-band index and the band that sets it is the governing band; the interpolation makes the rating continuous (an NC-42.5 is reported as such, not snapped to a curve). Spectra above NC-70 or below NC-15 fall outside the family and are flagged (>NC-70 with the band of maximum exceedance, or <NC-15) instead of receiving a fabricated number. The **RC Mark II** contour (Annex D) is a pure −5 dB/octave line keyed to its 1000 Hz value with a low-frequency floor of $\max(\mathrm{RC} + 25,\ 55)$ dB at 16/31.5 Hz; the rating is the arithmetic mean of the 500/1000/2000 Hz levels rounded to an integer (clause D.4), and the spectral-quality tag compares the spectrum with the reference contour (clause D.3): rumble "R" when any band at or below 500 Hz exceeds it by more than 5 dB, hiss "H" when any band at or above 1 kHz exceeds it by more than 3 dB (both together "RH"), else neutral "N", reported as e.g. RC-35(N). The generated RC contours reproduce Table D.1 digit for digit, and feeding any Table 1 NC curve back returns its own tangency rating. NCB, RNC (Annex A) and the QAI (clause D.5) are deliberately out of scope.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/room_noise_criteria_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/room_noise_criteria.svg" alt="Two panels for the same ventilation-dominated room spectrum. Left: the measured octave-band levels over the NC curve family, with a red diamond marking the tangent point at 250 Hz that sets the NC-42.5 rating. Right: the same spectrum over the reference RC-35 curve, with the low-frequency bands rising through the shaded rumble tolerance (plus 5 dB below 500 Hz) so the noise is classified RC-35(R), and the hiss tolerance (plus 3 dB at and above 1000 Hz) shaded for comparison" width="96%"></picture>

*The same spectrum rated both ways: NC tangency at the governing band (left) and the RC Mark II reference with the rumble excess (right).*

See the [Room Noise guide](../../buildings/rooms/room-noise.md) for usage.

## Room and building acoustics (ISO 18233, ISO 3382, ISO 16283, ISO 10140, EN 12354, ISO 12999, ISO 717, ISO 354)

### Deterministic-excitation impulse response (ISO 18233)

A room/transmission path is modelled as **linear time-invariant**, so its impulse response $h(t)$ carries everything. ISO 18233 replaces the classical noise-burst decay with a deterministic excitation that is **deconvolved** into $h(t)$, gaining 20–30 dB of effective signal-to-noise ratio. The exponential sine sweep (ESS, Annex B) has instantaneous frequency $f(t) = f_1 (f_2/f_1)^{t/T}$, so its phase is the closed-form integral of $2 \pi f(t)$:

$$
\varphi(t) = \frac{2 \pi f_1 T}{\ln(f_2/f_1)} \left[ \left( \frac{f_2}{f_1} \right)^{t/T} - 1 \right] .
$$

A constant time-per-octave makes the ESS spectrum pink (−3 dB/octave). Deconvolution is done by **linear** (non-circular, zero-padded) spectral division $H = Y\ \overline{X} / (|X|^2 + \varepsilon)$, the Tikhonov term $\varepsilon$ (a fraction of $\max |X|^2$) preventing noise blow-up at the band edges. Since a low-to-high sweep places harmonic-distortion products at negative arrival times, they fall in the wrapped tail and are removed by keeping the causal part (Farina). The MLS method (Annex A) instead exploits that the circular autocorrelation of a maximum-length sequence of length $2^N-1$ is a periodic delta, so $h = \operatorname{xcorr}_{\text{circ}}(\text{recorded}, \text{mls}) / 2^N$; synchronous averaging of $n$ periods adds $10 \log_{10} n$ dB.

### Schroeder backward integration (ISO 3382-1, 5.3.3)

The band decay curve is the **backward-integrated** squared IR (Schroeder):

$$
E(t) = \int_t^{\infty} p^2(\tau)\ d\tau = \int_0^{\infty} p^2\ d\tau - \int_0^t p^2\ d\tau , \qquad L(t) = 10 \log_{10} \frac{E(t)}{E(0)}\ \text{dB},
$$

i.e. a reversed cumulative sum in discrete time. Backward integration cancels the random fluctuation of a single squared IR: for a purely exponential energy decay $p^2(t) = e^{-a t}$ it gives $E(t) = e^{-a t}/a$, an exactly straight line $L(t) = -(10 a / \ln 10)\ t$. Background noise flattens $E(t)$, so integration is truncated at the crossing $t_1$ of the fitted decay line with the noise level and the missing tail is compensated by an exponential with the fitted rate; without that term the finite integral systematically **underestimates** $T$.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/schroeder_decay_dark.webp"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/schroeder_decay.webp" alt="Squared impulse response with its Schroeder backward-integrated decay curve, and the EDT, T20 and T30 regression windows marked" width="80%"></picture>

*A squared impulse response, its Schroeder backward integral and the EDT/T20/T30 regression windows of the next subsection.*

### Regression windows and validity (ISO 3382-2, Clause 6, Annex B/C)

Reverberation time is a least-squares fit $L = a + b t$ over a window, extrapolated to 60 dB via $T = -60/b$ (Annex C): **EDT** on 0 to −10 dB, **T20** on −5 to −25 dB, **T30** on −5 to −35 dB. A single-slope decay gives EDT = T20 = T30; a fast early / slow late double slope gives EDT < T30. Validity uses the dynamic-range rule of 5.3.3: the noise must sit at least 25 dB below the IR peak for EDT (evaluation span + 15 dB), tightened to 46 dB for T20 and 54 dB for T30 so the tail-compensation bias of a flagged-valid value stays within the 5 % JND. The **curvature** $C = 100\ (T_{30}/T_{20} - 1)$ % (Annex B) flags a non-straight decay above 10 %.

### Clarity, definition and centre time (ISO 3382-1, Annex A)

Splitting the energy at an early/late boundary $t_\mathrm{e}$ gives the early-to-late index and the definition ratio:

$$
C_{te} = 10 \log_{10} \frac{\int_0^{t_\mathrm{e}} p^2\ dt}{\int_{t_e}^{\infty} p^2\ dt}\ \text{dB}, \qquad D_{50} = \frac{\int_0^{0.05} p^2\ dt}{\int_0^{\infty} p^2\ dt}, \qquad C_{50} = 10 \log_{10} \frac{D_{50}}{1 - D_{50}},
$$

with $t_\mathrm{e} = 50$ ms (C50, speech) or 80 ms (C80, music), and the **centre time** $T_\mathrm{s} = \int_0^{\infty} t\ p^2\ dt / \int_0^{\infty} p^2\ dt$. For a pure exponential decay these have closed forms $C_{te} = 10 \log_{10}(e^{a t_\mathrm{e}} - 1)$ and $T_\mathrm{s} = 1/a$; at $T = 1$ s ($a = 13.8155$) they evaluate to C80 = 3.05 dB, C50 = −0.02 dB, D50 = 0.499 and Ts = 72.4 ms, the values the implementation reproduces. Table A.1 JNDs (EDT 5 %, C80 1 dB, D50 0.05, Ts 10 ms) bound how finely each is worth reporting.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/room_parameters_bands_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/room_parameters_bands.svg" alt="ISO 3382 per-band parameters of a synthetic room impulse response: grouped EDT, T20 and T30 bars per octave band falling from about 1.4 s at 125 Hz to 0.7 s at 4 kHz, over a second panel where C50 and C80 rise with frequency" width="92%"></picture>

*The closed forms above hold for a single exponential decay; a real room gives
one set of values per band. The upper panel is the decay itself (EDT, T20 and
T30 falling with frequency as air and surfaces absorb more), the lower panel
the early/late split of the same impulse response, and C50 and C80 rise with
frequency for the same reason the decay time falls — the later the energy, the
more of it the room has already removed.*

### Open-plan spatial decay (ISO 3382-3, Clause 6)

The spatial decay rate of A-weighted speech is the ordinary least-squares slope of $L_{p,A,S}$ against $\log_{10}(r/r_0)$ ($r_0 = 1$ m) over the 2–16 m positions, rescaled to a per-doubling figure, and the nominal level is read off the same line at 4 m:

$$
L = a + b\ \log_{10}(r/r_0), \qquad D_{2,\mathrm{S}} = -\log_{10}(2)\ b, \qquad L_{p,A,S,4\text{m}} = a + b\ \log_{10}(4/r_0).
$$

The distraction distance rD and privacy distance rP are the distances where a **linear** (not logarithmic) regression of STI against distance crosses 0.50 and 0.20; a non-negative fitted slope (STI not falling with distance) makes them undefined, realising the standard's "can prove impossible to determine" note.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/open_plan_decay_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/open_plan_decay.svg" alt="Open-plan spatial decay: A-weighted speech level and STI against source distance on a log axis, with the D2,S regression, the Lp,A,S,4m marker at 4 m and the rD and rP distance crossings" width="80%"></picture>

*Two regressions on two different axes, which is what makes this clause hard to
hold in the head. The level line is fitted against $\log_{10}(r/r_0)$ and read
twice — as the slope $D_{2,\mathrm{S}}$ per doubling, and at $r = 4$ m for
$L_{p,A,S,4\text{m}}$. The STI line is fitted against $r$ itself, **linearly**,
and read where it crosses 0.50 and 0.20 for the distraction and privacy
distances. If that second fit comes out flat or rising, the two distances do
not exist rather than being large.*

### Image-source room impulse response (Kuttruff 4.1, Vorländer 11)

A rectangular room reflects a point source in its walls; each reflection equals the free-field sound of a **mirror image** of the source. Mirroring a coordinate in a wall ($S_n = S - 2 d\,\mathbf{n}$, Vorländer Eq. 11.36) turns the source into a regular lattice of images, and the room impulse response is the sum of the direct sound and one delayed, attenuated impulse per image (Kuttruff Eqs. 4.4–4.5),

$$
g(t) = \sum_n A_n\ \delta(t - t_n), \qquad
A_n = \frac{1}{4\pi r_n}\ e^{-m r_n / 2} \prod_{\text{walls}} R_q^{\,k_{q,n}}, \qquad
t_n = \frac{r_n}{c},
$$

with the $1/(4\pi r_n)$ spherical spreading, the product of the wall **pressure reflection factors** $R_q = \sqrt{1 - \alpha_q}$ (Vorländer Eq. 11.39; $|R|^2 = 1-\alpha$ in energy) each raised to the number of reflections $k_{q,n}$ that image made off wall $q$, and the air pressure attenuation $e^{-m r_n/2}$ ($m$ the *intensity* attenuation constant). Along one axis the reflection counts of the image at lattice index $n$ and parity $p$ are $|n-p|$ and $|n|$ off the two walls (Allen & Berkley 1979), so the total order is $\sum_i |2 n_i - p_i|$; a shoebox has $\tfrac{2}{3}(2 i_0^3 + 3 i_0^2 + 4 i_0)$ audible images up to order $i_0$ (Kuttruff Eq. 9.23), and the reflection density grows as $\mathrm{d}N/\mathrm{d}t = 4\pi c^3 t^2 / V$ (Kuttruff Eq. 4.6).

The initial decay rate of the specular reverberant energy recovers the **Eyring** reverberation time $T = -24 V \ln 10 / (c S \ln(1 - \bar\alpha))$ (Kuttruff Eq. 5.23), because the mean reflection rate $cS/4V$ equals $\tfrac{c}{2}(1/L_x + 1/L_y + 1/L_z)$. The match is exact only near cubic geometry; an elongated room sustains energy along its long axis, so the pure specular decay runs slower than Eyring's diffuse-field estimate (the anisotropy the Fitzroy/Arau-Puchades models correct). The model is specular only — no diffraction or diffuse scattering — and exact only for real, angle-independent reflection factors.

### Steady-state room field (Bies 6.4, Kuttruff 5.6)

A source of constant power sets up a steady level made of a direct field falling with distance and a diffuse **reverberant** field that is (approximately) uniform. With the **room constant** $R = S\bar\alpha/(1-\bar\alpha)$ (Bies Eq. 6.44) and the directivity factor $Q$,

$$
L_p = L_W + 10 \log_{10}\!\left( \frac{Q}{4\pi r^2} + \frac{4}{R} \right) \left[ + 10 \log_{10}\frac{\rho c}{400} \right] \quad \text{(Bies Eq. 6.43)},
$$

the optional last term (about $+0.14$ dB at 20 °C) correcting a characteristic impedance away from 400 Pa·s/m. The **critical distance** $r_\mathrm{c} = \sqrt{Q R / 16\pi}$ is where the two fields cross; Kuttruff's reverberation distance (Eq. 5.44, $r_\mathrm{c} = \sqrt{A/16\pi}$ for $Q=1$) uses the Sabine area $A = S\bar\alpha$ instead of $R = A/(1-\bar\alpha)$, the two coinciding for small $\bar\alpha$. The **Schroeder frequency** $f_\mathrm{s} = 2000\sqrt{T/V}$ (Kuttruff Eq. 3.44) roughly marks the modal-to-diffuse transition, a heuristic crossover rather than a sharp cutoff: well below it discrete room modes dominate and these diffuse-field relations grow unreliable, well above it the modes overlap and the relations hold. Borderline rooms still warrant a band-by-band check. See the [Image sources and steady-state field guide](../../buildings/rooms/room-image-sources.md).

### Field insulation and weighted rating (ISO 16283-1, ISO 717-1)

Per one-third-octave band the level difference $D = L_1 - L_2$ (energy-averaged over microphone positions, $L = 10 \log_{10}[(1/n) \sum_i 10^{L_i/10}]$) is normalised two ways: the standardized level difference $D_\mathrm{nT} = D + 10 \log_{10}(T/T_0)$ with $T_0 = 0.5$ s (so $D_\mathrm{nT} = D$ when $T = T_0$), and the apparent sound reduction index $R' = D + 10 \log_{10}(S/A)$ with the Sabine absorption area $A = 0.16\ V / T$, hence $R' = D + 10 \log_{10}[S T / (0.16\ V)]$.

The single-number rating (ISO 717-1, Clause 4.4) shifts the Table 3 **reference curve** in 1 dB steps toward the measured curve until the sum of *unfavourable* deviations $\sum_i \max(0, \text{ref}_i + k - \text{meas}_i)$ is maximal but $\le$ 32.0 dB (16 thirds) or 10.0 dB (5 octaves); the rating $R_\mathrm{w}$ is the shifted reference at 500 Hz. The **spectrum adaptation terms** are $C = X_{\mathrm{A}1} - X_\mathrm{w}$ and $C_\mathrm{tr} = X_{\mathrm{A}2} - X_\mathrm{w}$ with $X_{\mathrm{A}j} = -10 \log_{10} \sum_i 10^{(L_{ij} - X_i)/10}$ (Table 4 spectra No. 1 pink noise, No. 2 urban traffic), each rounded to an integer. The ISO 717-1 Annex C worked example ($R_\mathrm{w} = 30$, $C = -2$, $C_\mathrm{tr} = -3$, unfavourable sum 31.8 dB) is reproduced exactly.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/insulation_rating_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/insulation_rating.svg" alt="Measured one-third-octave sound reduction index with the shifted ISO 717-1 reference curve and the resulting weighted rating at 500 Hz" width="80%"></picture>

*A measured R spectrum against the shifted ISO 717-1 reference: the rating is the shifted reference read at 500 Hz.*

### Impact insulation and absorption (ISO 16283-2, ISO 717-2, ISO 354)

Impact insulation swaps the airborne source for a standardized **tapping
machine** and rates the receiving-room level, so the sign conventions flip. The
standardized and normalized impact levels are $L'_{nT} = L_i - 10 \log_{10}(T/T_0)$
(the reverberation term is *subtracted*, opposite to $D_\mathrm{nT}$) and
$L'_n = L_i + 10 \log_{10}(A/A_0)$ with $A_0 = 10$ m² and $A = 0.16\ V/T$. The
ISO 717-2 rating shifts the Table 3 reference curve until $\sum_i \max(0, \text{meas}_i - (\text{ref}_i + k))$
is maximal but $\le$ 32.0 dB (16 thirds) or 10.0 dB (5 octaves); the
*unfavourable* deviation now counts where the **measurement exceeds** the
reference (impact noise is worse when louder), the mirror image of ISO 717-1.
The rating is the shifted reference at 500 Hz, reduced by a further 5 dB for
octave bands, and the adaptation term is $C_\mathrm{I} = L_\mathrm{n,sum} - 15 - L_\mathrm{n,w}$
with the energetic sum $L_\mathrm{n,sum} = 10 \log_{10} \sum_i 10^{L_i/10}$ over
100–2500 Hz (thirds) or 125–2000 Hz (octaves). The ISO 717-2 Annex C examples
are reproduced exactly (thirds $L_\mathrm{n,w} = 79$, $C_\mathrm{I} = -11$; octaves $54$, $0$),
via the same monotone shift search as ISO 717-1 run on the negated curves.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/impact_rating_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/impact_rating.svg" alt="Measured one-third-octave normalized impact sound pressure level with the shifted ISO 717-2 reference curve and the resulting weighted rating read at 500 Hz" width="80%"></picture>

*The mirror image of the airborne figure above, drawn so the flip is visible
rather than asserted. There the unfavourable deviations were counted where the
measurement fell **below** the reference; here they are counted where it rises
**above** it, because a louder receiving room is a worse floor. Everything else
is the same procedure: the reference curve shifted in 1 dB steps until the
unfavourable sum is as large as it can be without passing 32.0 dB, and the
rating read off the shifted reference at 500 Hz.*

Sound absorption (ISO 354) measures the equivalent absorption area from
Sabine's relation applied to a reverberation room empty and with the specimen:
$A = 55.3\ V/(c\ T) - 4 V m$ (the $4 V m$ term is the air absorption, $m$ the
power attenuation coefficient in 1/m), so the specimen area is
$A_\mathrm{T} = A_2 - A_1$ and its coefficient $\alpha_\mathrm{s} = A_\mathrm{T}/S$. With the speed of
sound from Eq. (6), $c = 331 + 0.6\ t$ (°C), and $m$ converted from an
ISO 9613-1 attenuation coefficient by $m = \alpha / (10 \log_{10} e)$. Because
diffraction and edge scattering intercept more than the flat sample area,
$\alpha_\mathrm{s}$ is left unclamped and may exceed 1.0 (Clause 3.7 NOTE 2).

### Laboratory vs field normalization (ISO 10140, ISO 16283)

The field indices carry a prime because they include flanking transmission
around the partition; the laboratory indices do not, because a qualified
facility suppresses it. The algebra is otherwise identical, differing only in
which quantity is normalised. The airborne pair is the direct laboratory sound
reduction index $R = L_1 - L_2 + 10 \log_{10}(S/A)$ (ISO 10140-2) versus the
apparent field index $R' = L_1 - L_2 + 10 \log_{10}(S/A)$ (ISO 16283-1), the
same closed form evaluated with the facility's known $A$ or the room's measured
$A = 0.16\ V/T$. The impact pair is the normalized laboratory level
$L_\mathrm{n} = L_i + 10 \log_{10}(A/A_0)$ (ISO 10140-3) versus the field $L'_n$
(ISO 16283-2), both referenced to $A_0 = 10$ m². Before either is formed the
receiving-room level is corrected for background noise by the energy
subtraction $L = 10 \log_{10}(10^{L_\mathrm{sb}/10} - 10^{L_\mathrm{b}/10})$ for a 6–15 dB
signal-to-background margin, capped at a fixed $1.3$ dB (the limit of
measurement) at or below 6 dB and omitted at or above 15 dB (ISO 10140-4,
Clause 4.3), the laboratory analogue of the 6/10 dB rule of ISO 16283-1. The
façade extension (ISO 16283-3) replaces the source-room level by the level 2 m
in front of the façade, $D_{2\mathrm{m}} = L_{1,2\mathrm{m}} - L_2$, and adds a fixed
angle-of-incidence correction to the element sound reduction index, $-1.5$ dB
for the 45° loudspeaker method ($R'_{45°}$) and $-3$ dB for the all-angle
road-traffic method ($R'_{tr,s}$); all three carry the ISO 717-1 airborne
single number.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_insulation_lab_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_insulation_lab.svg" alt="Plan view of an ISO 10140 laboratory transmission suite: structurally decoupled source and receiving reverberation rooms of about 59 and 51 cubic metres, the test element mounted in the 10 square metre test opening between them, a corner loudspeaker in the source room and a continuously moving microphone with a sweep radius of at least 1 m in each room" width="92%"></picture>

### Flanking transmission prediction (EN 12354-1/2)

The apparent field index is the energetic sum of the direct path $Dd$ and, for
each flanking element $F=f$ across its junction with the separating element, the
three paths $Ff$, $Df$ and $Fd$ (EN 12354-1, simplified single-number model,
Formula 26):

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_flanking_paths_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_flanking_paths.svg" alt="The direct path Dd through the separating element and the three flanking paths Ff, Df and Fd across each junction between a flanking element and the separating element" width="92%"></picture>

*The four labels are only meaningful on this drawing. Upper case is the source
side and lower case the receiving side, so $Dd$ goes straight through the
separating element, $Ff$ runs along a flanking element on both sides of the
junction, and $Df$ and $Fd$ cross from one to the other. The junction they cross
is where $K_{ij}$ and the coupling length $\ell_\mathrm{f}$ live in the formula below,
and the separating element is where $S_\mathrm{s}$ is measured. Each junction contributes
three flanking terms, so a room with four flanking elements sums thirteen paths.*

$$
R'_w = -10 \log_{10}\Big[ 10^{-R_\mathrm{Dd,w}/10}
       + \sum 10^{-R_\mathrm{Ff,w}/10} + \sum 10^{-R_\mathrm{Df,w}/10}
       + \sum 10^{-R_\mathrm{Fd,w}/10} \Big].
$$

The direct path is $R_\mathrm{Dd,w} = R_\mathrm{s,w} + \Delta R_\mathrm{Dd,w}$ (Formula 27), the
separating-element laboratory index plus any lining improvement. Each flanking
path (Formula 28a) is

$$
R_{ij,\mathrm{w}} = \frac{R_{i,\mathrm{w}} + R_{j,\mathrm{w}}}{2} + \Delta R_{ij,\mathrm{w}} + K_{ij}
         + 10 \log_{10}\frac{S_\mathrm{s}}{l_0\ l_\mathrm{f}},
$$

with $R_{i,\mathrm{w}}$, $R_{j,\mathrm{w}}$ the laboratory indices of the two elements meeting at
the junction ($i$ source side, $j$ receiving side), $\Delta R_{ij,\mathrm{w}}$ the
combined lining improvement, $S_\mathrm{s}$ the separating-element area, $l_\mathrm{f}$ the
junction coupling length and $l_0 = 1$ m the reference coupling length. $K_{ij}$
is the junction **vibration reduction index** (Annex E), an empirical function of
the mass ratio $M = \log_{10}(m'_{\perp,i}/m'_i)$: for a rigid cross-junction
$K_{13} = 8.7 + 17.1 M + 5.7 M^2$ (through) and $K_{12} = 8.7 + 5.7 M^2$
(corner), read at 500 Hz, and floored at $K_{ij,\min} = 10 \log_{10}[l_f\ l_0
(1/S_i + 1/S_j)]$ (Formula 29). Two linings combine as $\max(a,b) + \min(a,b)/2$
(Formulas 30/31). The impact counterpart (EN 12354-2, Formula 21) is the direct
subtraction $L'_{n,w} = L_\mathrm{n,w,eq} - \Delta L_\mathrm{w} + K$, with the bare-floor
equivalent level $L_\mathrm{n,w,eq} = 164 - 35 \log_{10}(m'/m'_0)$ (Annex B), the
covering improvement $\Delta L_\mathrm{w}$ (ISO 717-2) and the flanking correction $K$
from Table 1. The EN 12354-1 Annex H.3 ($R'_w = 52$ dB) and EN 12354-2 Annex E.3
($L'_{n,w} = 45$ dB) worked examples are reproduced exactly; the simplified
model is stated to have about a 2 dB standard deviation (Clause 5).

### Absorption in enclosed spaces (EN 12354-6)

EN 12354-6:2003 predicts the equivalent absorption area of a room from its
parts (the normative Clause 4 model). The total (Formula 1) sums the surfaces,
the objects and the air:

$$
A = \sum_i \alpha_{\mathrm{s},i}\ S_i + \sum_j A_{obj,j} + \sum_k \alpha_{s,k}\ S_k + A_\mathrm{air},
\qquad A_\mathrm{air} = 4\ m\ V\ (1 - \psi),
$$

with $m$ the power attenuation coefficient of air (Formula 2; Table 1
tabulates it for six temperature/humidity climates over the octave bands
125 Hz – 8 kHz), $\psi = \sum V_\mathrm{obj} / V$ the volume fraction occupied by
objects (Formula 3), and a hard irregular object approximated by
$A_\mathrm{obj} = V_\mathrm{obj}^{2/3}$ (Formula 4). The reverberation time follows from
Sabine applied to the free volume (clause 4.4, Formula 5):

$$
T = \frac{55.3}{c_0}\ \frac{V\ (1 - \psi)}{A},
$$

with $c_0 = 345.6$ m/s chosen so that $55.3/c_0$ is the familiar $0.16$
(clause 4.4 NOTE). The three Annex E worked cases are reproduced: the
bare 29.75 m³ room gives $A = 2.26$ m² and $T = 2.1$ s at 1 kHz, and adding
hard objects ($\psi \approx 0.072$) raises $A$ to 5.03 m² and drops $T$ to
0.9 s. The informative Annex D method for irregular spaces and unevenly
distributed absorption is out of scope.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/enclosed_space_absorption_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/enclosed_space_absorption.svg" alt="Two panels for a 60 cubic metre office with a bare versus acoustically-treated ceiling: the equivalent absorption area per octave band, much higher with the acoustic ceiling, and the reverberation time falling from about five seconds at low frequency for the bare room to under one second with the acoustic ceiling" width="96%"></picture>

*What Formula 1 does band by band: the equivalent absorption area on the left
and the reverberation time it implies through Formula 5 on the right, for the
same room bare and treated. The Annex E case quoted above is the same
arithmetic on a smaller room — $A$ from 2.26 to 5.03 m² and $T$ from 2.1 to
0.9 s at 1 kHz — and the figure shows why the two move in opposite directions
and not proportionally.*

See the [Enclosed-Space Absorption guide](../../buildings/rooms/enclosed-space-absorption.md) for usage.

### Measurement uncertainty (ISO 12999-1)

ISO 12999-1 supplies the uncertainty of the quantities above from
inter-laboratory (ISO 5725) reproducibility and repeatability rather than a
GUM functional model. Three **measurement situations** fix the standard
uncertainty $u$: situation **A** (laboratory characterisation) uses the
reproducibility standard deviation $\sigma_\mathrm{R}$; situation **B** (same location,
different teams) the in-situ $\sigma_\mathrm{situ}$; situation **C** (same location,
operator and equipment, repeated) the repeatability $\sigma_\mathrm{r}$. The per-band and
single-number values are tabulated for airborne $R$/$R'$/$D_\mathrm{n}$/$D_\mathrm{nT}$
(Tables 2/3), impact $L_\mathrm{n}$/$L'_n$ (Table 4 bands, situations B/C only; Table 5
ratings adding a situation-A estimate) and the
covering reduction $\Delta L$ (Tables 6/7, situation A only). The expanded
uncertainty is $U = k\ u$ (Formula 2) with the coverage factor $k$ of Table 8
(at 95 %, $k = 1.96$ two-sided, $k = 1.65$ one-sided; a minimum $k = 1$ is
enforced). A two-sided interval $Y = y \pm U$ reports a value (Formula 3); a
one-sided factor declares conformity, $y - U > $ requirement for a lower limit
(Formula 5) or $y + U <$ requirement for an upper limit (Formula 4).
Uncorrelated components combine in quadrature $u_\mathrm{c} = \sqrt{\sum u_i^2}$
(Formula C.2), $m$ independent measurements reduce $u$ to $u/\sqrt{m}$
(Formula A.7), and the uncorrelated single-number uncertainty is the
energy-weighted quadrature sum of the band uncertainties (Formula B.2).

See the [Room Acoustics](../../buildings/rooms/room-acoustics.md) and
[Field Insulation Measurement (ISO 16283)](../../buildings/insulation/insulation-field.md) guides for usage.

### Predicted panel sound insulation (Bies 7.2, Hopkins 2.9/4.3.10, Cremer 5)

Where EN 12354 takes the element $R$ as a measurement, the sound reduction index
of a panel can also be **predicted** from its physical properties. A limp panel
follows the mass law $TL_0 = 10\log_{10}[1 + (\pi f m''/\rho_0 c_0)^2]$ (Bies Eq. 7.40),
which rises 6 dB per octave and 6 dB per doubling of the surface mass $m''$; the
field-incidence value subtracts 5.5 dB (one-third octave). Stiffness adds a
**coincidence dip** at $f_\mathrm{c} = (c_0^2/2\pi)\sqrt{m''/B'}$ (Eq. 7.3), where the free
bending wavelength matches the acoustic trace wavelength. Sharp's method holds
the mass law to $f_\mathrm{c}/2$, drops linearly in $\log f$ to the dip
$TL = 20\log_{10}(f_\mathrm{c} m'') + 10\log_{10}\eta - 44$ and rises again above $f_\mathrm{c}$ with the loss
factor $\eta$ (Eq. 7.44). A **double wall** is a mass-spring-mass system with the
cavity as the spring: below $f_0 = 60\sqrt{(m_1+m_2)/(m_1 m_2 d)}$ (Eq. 7.62) it
follows the mass law of the combined mass, and above it the two leaves' mass laws
add plus the cavity term $20\log_{10}(2kd)$, saturating at +6 dB beyond
$f_\mathrm{l} = c_0/(2\pi d)$ (Eq. 7.64); a porous fill lowers $f_0$. Small air paths cap
any construction: the transmission coefficient of a straight slit (Gomperts,
Hopkins Eq. 4.99, with resonances at $d + 2e = z\lambda/2$) or a circular hole
(Wilson & Soroka, Eq. 4.102) combines with the wall in the area-weighted energy
sum $R = -10\log_{10}[(1/\sum S_n)\sum S_n 10^{-R_n/10}]$ (Eq. 4.92), so a bare opening
of relative area $S_\mathrm{a}/S$ limits the composite to $10\log_{10}(S/S_\mathrm{a})$. The resonant
transmission path and the double-wall radiation draw on the plate radiation
efficiency and point mobilities of the
[vibration theory](vibration.md). The prediction is clean-room from Bies,
Hansen & Howard (2017), Hopkins (2007) and Cremer, Heckl & Petersson (2005).

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/panel_insulation_concept_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/panel_insulation_concept.svg" alt="Four panels: the single-panel mass law with its coincidence dip, the double wall with the mass-spring-mass resonance and cavity gain, the plate radiation efficiency rising to unity above the critical frequency, and a composite wall whose 1 % open slit caps R at the open-area limit" width="92%"></picture>

*The four behaviours of the paragraph above, one per panel. Top left, the mass
law rising 6 dB per octave with Sharp's coincidence dip cut into it at $f_\mathrm{c}$.
Top right, the double wall: no better than the combined mass below $f_0$, then
the cavity term climbing until it saturates. Bottom left, the radiation
efficiency that decides how much of the plate's vibration becomes sound. Bottom
right, the ceiling a leak imposes: a 1 % open area holds the composite at
$10\log_{10}(S/S_\mathrm{a}) = 20$ dB however good the wall is, which is the panel worth
showing a client.*

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/aperture_slit_geometry_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/aperture_slit_geometry.svg" alt="To-scale cross-section of a 2 mm slit through a 100 mm wall: the hatched wall drawn in section with the narrow horizontal air gap at mid-height, an incident-sound arrow pointing at the gap from the left, the 100 mm wall depth and 2 mm slit width dimensioned, and circular transmitted wavefronts sketched spreading from the slit exit on the right" width="80%"></picture>

See the [Predicting Panel Sound Insulation](../../buildings/design/panel-sound-insulation.md) guide for
usage.

## References

- Kuttruff, H. (2016). *Room acoustics* (6th ed.). CRC Press.
  [doi:10.1201/9781315372150](https://doi.org/10.1201/9781315372150).
  The statistical decay theory behind backward integration and the Sabine
  relations used throughout this page, the image-source construction
  (Section 4.1), the Eyring reverberation and reverberation distance
  (Sections 5.5–5.6) and the Schroeder frequency (Section 3.6).
- Vorländer, M. (2020). *Auralization: Fundamentals of acoustics, modelling,
  simulation, algorithms and acoustic virtual reality* (2nd ed.). Springer.
  [doi:10.1007/978-3-030-51202-6](https://doi.org/10.1007/978-3-030-51202-6).
  The image-source / mirror-source model of the impulse-response section
  (Chapter 11).
- Allen, J. B., & Berkley, D. A. (1979). Image method for efficiently
  simulating small-room acoustics. *The Journal of the Acoustical Society of
  America*, 65(4), 943-950.
  [doi:10.1121/1.382599](https://doi.org/10.1121/1.382599).
  The reflection-count decomposition of the rectangular-room image lattice.
- Bies, D. A., Hansen, C. H., & Howard, C. Q. (2017). *Engineering noise
  control* (5th ed.). CRC Press.
  [doi:10.1201/9781351228152](https://doi.org/10.1201/9781351228152).
  The steady-state room field and the room constant (Section 6.4).
- Schroeder, M. R. (1965). New method of measuring reverberation time.
  *The Journal of the Acoustical Society of America*, 37(3), 409-412.
  [doi:10.1121/1.1909343](https://doi.org/10.1121/1.1909343).
  The backward-integration method of the decay-curve section.
- Hak, C. C. J. M., Wenmaekers, R. H. C., & van Luxemburg, L. C. J. (2012).
  Measuring room impulse responses: Impact of the decay range on derived
  room acoustic parameters. *Acta Acustica united with Acustica*, 98(6),
  907-915. [doi:10.3813/aaa.918574](https://doi.org/10.3813/aaa.918574).
  The INR decay-range analysis behind the tightened T20/T30 validity
  thresholds.
- Beranek, L. L. (1957). Revised criteria for noise in buildings. *Noise
  Control*, 3(1), 19-27.
  [doi:10.1121/1.2369239](https://doi.org/10.1121/1.2369239).
  The original NC curves rated by the tangency method of the room-noise
  section.
- Blazier, W. E. (1997). RC Mark II: A refined procedure for rating the
  noise of heating, ventilating, and air-conditioning (HVAC) systems in
  buildings. *Noise Control Engineering Journal*, 45(6), 243-250.
  [doi:10.3397/1.2828446](https://doi.org/10.3397/1.2828446).
  The RC Mark II contour and spectral-quality tag codified by ANSI/ASA
  S12.2 Annex D.
- Hopkins, C. (2007). *Sound insulation*. Butterworth-Heinemann.
  ISBN 978-0-7506-6526-1.
  [doi:10.4324/9780080550473](https://doi.org/10.4324/9780080550473).
  The measurement chains, flanking transmission and EN 12354 prediction
  framework of the insulation sections.
- Vigran, T. E. (2008). *Building acoustics*. CRC Press.
  ISBN 978-0-415-42853-8.
  [doi:10.1201/9781482266016](https://doi.org/10.1201/9781482266016).
  Sound transmission in buildings, from single and double constructions to
  floating floors.
- Acoustical Society of America. (2019). *Criteria for evaluating room
  noise* (ANSI/ASA S12.2-2019).
  [ANSI webstore](https://webstore.ansi.org/standards/asa/ansiasas122019).
  The normative NC tangency method and the Annex D RC Mark II rating with
  its spectral tag.
- International Organization for Standardization. (2006). *Acoustics —
  Application of new measurement methods in building and room acoustics*
  (ISO 18233:2006).
  [iso.org catalogue](https://www.iso.org/standard/40408.html).
  The swept-sine and MLS deconvolution of the deterministic-excitation
  section.
- International Organization for Standardization. (2009). *Acoustics —
  Measurement of room acoustic parameters — Part 1: Performance spaces*
  (ISO 3382-1:2009).
  [iso.org catalogue](https://www.iso.org/standard/40979.html).
  Backward integration, the parameter definitions and the Annex A clarity
  family.
- International Organization for Standardization. (2008). *Acoustics —
  Measurement of room acoustic parameters — Part 2: Reverberation time in
  ordinary rooms* (ISO 3382-2:2008).
  [iso.org catalogue](https://www.iso.org/standard/36201.html).
  The regression windows, dynamic-range rules and curvature check of the
  validity section.
- International Organization for Standardization. (2012). *Acoustics —
  Measurement of room acoustic parameters — Part 3: Open plan offices*
  (ISO 3382-3:2012; since revised as
  [ISO 3382-3:2022](https://www.iso.org/standard/77437.html)).
  [iso.org catalogue](https://www.iso.org/standard/46520.html).
  The open-plan spatial decay and the distraction and privacy distances.
- International Organization for Standardization. (2014). *Acoustics — Field
  measurement of sound insulation in buildings and of building elements —
  Part 1: Airborne sound insulation* (ISO 16283-1:2014).
  [iso.org catalogue](https://www.iso.org/standard/55997.html).
  The field level differences and normalizations of the insulation
  sections.
- International Organization for Standardization. (2020). *Acoustics —
  Rating of sound insulation in buildings and of building elements — Part 1:
  Airborne sound insulation* (ISO 717-1:2020).
  [iso.org catalogue](https://www.iso.org/standard/77435.html).
  The reference-curve shift and the spectrum adaptation terms C and Ctr.
- International Organization for Standardization. (2003). *Acoustics —
  Measurement of sound absorption in a reverberation room* (ISO 354:2003).
  [iso.org catalogue](https://www.iso.org/standard/34545.html).
  The reverberation-room absorption measurement and its air-absorption
  term.
- European Committee for Standardization. (2003). *Building acoustics —
  Estimation of acoustic performance of buildings from the performance of
  elements — Part 6: Sound absorption in enclosed spaces*
  (EN 12354-6:2003).
  [BSI Knowledge record (BS EN 12354-6:2003)](https://knowledge.bsigroup.com/products/building-acoustics-estimation-of-acoustic-performance-of-buildings-from-the-performance-of-elements-sound-absorption-in-enclosed-spaces).
  The Clause 4 absorption model and Annex E worked cases of the
  enclosed-space section.
- International Organization for Standardization. (2020). *Acoustics —
  Determination and application of measurement uncertainties in building
  acoustics — Part 1: Sound insulation* (ISO 12999-1:2020).
  [iso.org catalogue](https://www.iso.org/standard/73930.html).
  The measurement situations, tabulated uncertainties and coverage factors
  of the uncertainty section.
