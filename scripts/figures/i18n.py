#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The Spanish edition of every figure: the translation table and the switch.

Each figure is drawn once, in English, and translated at save time by walking
the finished figure's Text artists (:func:`_translate_figure`), so no generator
carries a second language inside it. That design puts every Spanish string of
the documentation in one place: the exact-match table for whole labels,
titles and captions, and the ordered pattern list for the families of strings
(units, band labels, generated sentences) that no fixed table could enumerate.
The table is the module -- it is data, not code, and it is kept here so a
translation fix never means touching a figure.
"""

from typing import Any

from . import _publish

# ---------------------------------------------------------------------------
# Language support: every figure is also generated in Spanish ("_es" suffix).
# Translation happens at savefig time by walking the figure's Text artists,
# so the generator functions stay single-language (English) internally.
_LANG = "en"
_LANG_SUFFIX = ""

_ES_EXACT = {
    # Materials: diffusers, resilient layers and in-situ surfaces (B11b)
    "s't, frame (Formula 4)": "s't, esqueleto (Fórmula 4)",
    "s'a, enclosed gas (Formula 7, eps = 0.9)":
        "s'a, aire encerrado (Fórmula 7, eps = 0,9)",
    "s' installed = s't + s'a (clause 8.2)":
        "s' instalada = s't + s'a (capítulo 8.2)",
    "clause 8.2:   r >= 100 kPa.s/m2 -> s' = s't\n"
    "              10 <= r < 100     -> s' = s't + s'a\n"
    "              r < 10            -> s' = s't only if s'a is negligible":
        "capítulo 8.2:  r >= 100 kPa.s/m2 -> s' = s't\n"
        "               10 <= r < 100     -> s' = s't + s'a\n"
        "               r < 10            -> s' = s't solo si s'a es "
        "despreciable",
    "the worked determination:\nd = 20 mm, 4.94 + 5.56 = 10.49":
        "la determinación del ejemplo:\nd = 20 mm, 4,94 + 5,56 = 10,49",
    "f_u = 0.58 c0 / d (Clause 5.4)": "f_u = 0,58 c0 / d (capítulo 5.4)",
    "|Hi|, the windowed free-field reference":
        "|Hi|, la referencia de campo libre enventanada",
    "|Hr|, the windowed surface reflection":
        "|Hr|, la reflexión de la superficie enventanada",
    "hi: free field, the rig clear of every surface":
        "hi: campo libre, el equipo lejos de toda superficie",
    "the same window, on the direct sound":
        "la misma ventana, sobre el sonido directo",
    "measured over the road: the two arrivals overlap":
        "medido sobre el pavimento: las dos llegadas se solapan",
    "hr = road - free field, the surface alone":
        "hr = pavimento - campo libre, solo la superficie",
    "ISO 13472-1, subtraction": "ISO 13472-1, sustracción",
    "ISO 13472-2, spot tube": "ISO 13472-2, tubo puntual",
    "Periodic, 6 x N = 7": "Periódico, 6 x N = 7",
    "Modulated, period + inverse": "Modulado, periodo + inverso",
    "small excitation: fr = 25.0 Hz":
        "excitación pequeña: fr = 25,0 Hz",
    "over-driven: peak lower and at 22 Hz":
        "sobreexcitado: pico más bajo y en 22 Hz",
    "alpha_spec - alpha_s  (numerator of Eq. (5))":
        "alpha_spec - alpha_s  (numerador de la Ec. (5))",
    "alpha_spec, rotating turntable (T3, T4)":
        "alpha_spec, plataforma giratoria (T3, T4)",
    "alpha_s, static turntable (T1, T2)":
        "alpha_s, plataforma estática (T1, T2)",
    "Absorption coefficient": "Coeficiente de absorcion",
    "s = (alpha_spec - alpha_s) / (1 - alpha_s)   Eq. (5)":
        "s = (alpha_spec - alpha_s) / (1 - alpha_s)   Ec. (5)",
    "From three impulse responses to one level (ISO 17497-2, Clause 7.4)":
        "De tres respuestas al impulso a un nivel (ISO 17497-2, capítulo 7.4)",
    "(a) h1: sample present": "(a) h1: con la muestra",
    "(b) h2: sample removed": "(b) h2: sin la muestra",
    "(c) h1 - h2: the room is gone": "(c) h1 - h2: la sala desaparece",
    "(d) h4: deconvolved by h3, Formula (1)":
        "(d) h4: deconvolucionada por h3, Fórmula (1)",
    "(e) windowed, Clause 7.4.3": "(e) enventanada, capítulo 7.4.3",
    "room": "sala",
    "The working band of a Schroeder design":
        "La banda útil de un difusor de Schroeder",
    "N = 7 QRD, f0 = 500 Hz, 5 periods":
        "QRD N = 7, f0 = 500 Hz, 5 periodos",
    "Flat panel, same footprint": "Panel plano, misma huella",
    "Reflected polar response at 1 kHz":
        "Respuesta polar reflejada a 1 kHz",
    "Band by band, same 4.2 m panel":
        "Banda a banda, el mismo panel de 4,2 m",
    "Normalised diffusion coefficient": "Coeficiente de difusión normalizado",
    "The absorption a metadiffuser pays for its phases":
        "La absorción que un metadifusor paga por sus fases",
    "face average (what a room model consumes)":
        "media en la cara (lo que consume un modelo de sala)",
    "At the 2 kHz design frequency": "A la frecuencia de diseño de 2 kHz",
    "Across the band: exact only where it was tuned":
        "En toda la banda: exacto solo donde se sintonizó",
    "QRD target, wells to 27.4 cm":
        "Objetivo QRD, pozos de hasta 27,4 cm",
    "Reflection phase [deg]": "Fase de reflexión [grados]",
    "Phase error against the QRD target [deg]":
        "Error de fase frente al objetivo QRD [grados]",
    "Slit index n": "Índice de rendija n",
    "within 10 deg of the target": "dentro de 10 grados del objetivo",
    "The honest bandwidth of the trick":
        "El ancho de banda real del truco",
    "On the rig: reading fr": "En el banco: leer fr",
    "Installed: only well above f0": "Instalado: solo muy por encima de f0",
    "Load-plate response [dB re static]":
        "Respuesta de la placa de carga [dB re estático]",
    "Improvement of impact insulation [dB]":
        "Mejora del aislamiento a impactos [dB]",
    "ideal mass-spring isolation": "aislamiento masa-resorte ideal",
    "The gas spring takes over as the layer gets thinner":
        "El resorte de aire domina según adelgaza la capa",
    "Loaded thickness d [mm]": "Espesor bajo carga d [mm]",
    "Dynamic stiffness per unit area [MN/m³]":
        "Rigidez dinámica por unidad de superficie [MN/m³]",
    "f0 of a 120 kg/m² screed [Hz]": "f0 de un recrecido de 120 kg/m² [Hz]",
    "The one free parameter of ISO 13472-1":
        "El único parámetro libre de la ISO 13472-1",
    "Free field": "Campo libre",
    "Over the road": "Sobre el pavimento",
    "Level [dB re max]": "Nivel [dB re max]",
    "Two in-situ methods, two reported bands":
        "Dos métodos in situ, dos bandas declaradas",
    "The bore that seals is the bore that caps the band":
        "El diámetro que sella es el que limita la banda",
    "Tube diameter d [mm]": "Diámetro del tubo d [mm]",
    "Plane-wave ceiling [Hz]": "Techo de onda plana [Hz]",
    # Resonance-sweep clip (anim_dynamic_stiffness_sweep)
    "Reading fr on the EN 29052-1 rig":
        "Lectura de fr en el banco EN 29052-1",
    "rigid base": "base rígida",
    "load plate, 8 kg": "placa de carga, 8 kg",
    "specimen": "probeta",
    "plate motion": "movimiento de la placa",
    "Response magnitude": "Módulo de la respuesta",
    "Phase [deg]": "Fase [grados]",
    "-90 deg: resonance": "-90 grados: resonancia",
    "below fr: the plate follows the force":
        "por debajo de fr: la placa sigue a la fuerza",
    "at fr: a quarter cycle behind, amplitude peaks":
        "en fr: un cuarto de ciclo por detrás, la amplitud es máxima",
    "above fr: the plate moves against the force":
        "por encima de fr: la placa se mueve contra la fuerza",
    "phase": "fase",
    # Building acoustics: sound insulation (B8 figures)
    "Same weighted rating, different spectrum":
        "El mismo indice ponderado, distinto espectro",
    "Rw alone is not a specification":
        "Rw por si solo no es una especificacion",
    "shifted reference (both, Rw = 49 dB)":
        "curva de referencia desplazada (ambas, Rw = 49 dB)",
    "150 mm dense concrete": "hormigon denso de 150 mm",
    "double leaf, 12 kg/m2 + 90 mm": "doble hoja, 12 kg/m2 + 90 mm",
    "mass-air-mass resonance at 82 Hz, below the\n"
    "rated range: the double leaf enters it still climbing":
        "resonancia masa-aire-masa en 82 Hz, por debajo del\n"
        "rango valorado: la doble hoja entra aun subiendo",
    "concrete": "hormigon",
    "double leaf": "doble hoja",
    "Single number [dB]": "Numero global [dB]",
    "6 dB apart\nagainst traffic": "6 dB de diferencia\nfrente al trafico",
    "Background-noise correction: two standards, two thresholds":
        "Correccion por ruido de fondo: dos normas, dos umbrales",
    "ISO 10140-4 laboratory (6 / 15 dB)":
        "ISO 10140-4 laboratorio (6 / 15 dB)",
    "ISO 16283-1 field (6 / 10 dB)": "ISO 16283-1 in situ (6 / 10 dB)",
    "limit of measurement\n(fixed 1,3 dB, flag the band)":
        "limite de medicion\n(1,3 dB fijos, senalar la banda)",
    "the field rule stops here": "la regla in situ termina aqui",
    "the laboratory rule stops here": "la regla de laboratorio termina aqui",
    "Signal-to-background margin Lsb - Lb [dB]":
        "Margen senal-fondo Lsb - Lb [dB]",
    "Correction applied, Lsb - L [dB]": "Correccion aplicada, Lsb - L [dB]",
    "A Fast detector cannot follow a long decay":
        "Un detector Fast no puede seguir una caida larga",
    "more than 1 dB apart": "mas de 1 dB de diferencia",
    "Fast maximum: 10 lg[g(C)/g(C0)] (ISO 16283-2)":
        "maximo Fast: 10 lg[g(C)/g(C0)] (ISO 16283-2)",
    "energy average: 10 lg(T/T0)": "promedio energetico: 10 lg(T/T0)",
    "T = T0 = 0,5 s\nboth terms vanish":
        "T = T0 = 0,5 s\nambos terminos se anulan",
    "T = 1,7275 s: C = 1, g = 1/e": "T = 1,7275 s: C = 1, g = 1/e",
    "Receiving-room reverberation time T [s]":
        "Tiempo de reverberacion del recinto receptor T [s]",
    "Term subtracted from the measured level [dB]":
        "Termino restado al nivel medido [dB]",
    "The same wall, in the laboratory and in two buildings\n"
    "(EN 12354-1 flanking over twelve paths)":
        "La misma pared, en laboratorio y en dos edificios\n"
        "(transmision por flancos EN 12354-1 sobre doce caminos)",
    "laboratory R (Rw = 49 dB)": "R de laboratorio (Rw = 49 dB)",
    "field R', good junctions (R'w = 47 dB)":
        "R' in situ, uniones buenas (R'w = 47 dB)",
    "field R', flanking dominant (R'w = 42 dB)":
        "R' in situ, flancos dominantes (R'w = 42 dB)",
    "2 dB: normal": "2 dB: normal",
    "7 dB: find the path": "7 dB: busque el camino",
    "The weak element sets the composite\n"
    "(Ejemplo 7.5 geometry: 6 m2 blind part + 2 m2 window)":
        "El elemento debil fija el conjunto\n"
        "(geometria del Ejemplo 7.5: 6 m2 de parte ciega + 2 m2 de ventana)",
    "Window RA [dBA]  (2 m2 of an 8 m2 facade)":
        "RA de la ventana [dBA]  (2 m2 de una fachada de 8 m2)",
    "Overall facade RA [dBA]": "RA global de la fachada [dBA]",
    "+10 dBA on the blind part: +0.4 dBA":
        "+10 dBA en la parte ciega: +0,4 dBA",
    "+5 dBA on the window: +4.1 dBA": "+5 dBA en la ventana: +4,1 dBA",
    "Qualifying the measurement surface (ISO 15186-1, 6.4.2)":
        "Calificacion de la superficie de medicion (ISO 15186-1, 6.4.2)",
    "surface not qualified": "superficie no calificada",
    "FpI = Lp - LIn (Formula (10))": "FpI = Lp - LIn (Formula (10))",
    "10 dB: reflecting specimen (6.4.2)":
        "10 dB: probeta reflectante (6.4.2)",
    "6 dB: absorbing specimen": "6 dB: probeta absorbente",
    "remedy in order: +5 to 10 cm of measurement distance first,\n"
    "then absorption in the receiving room":
        "remedio en orden: primero +5 a 10 cm de distancia de medicion,\n"
        "despues absorcion en el recinto receptor",
    "Surface pressure-intensity indicator FpI [dB]":
        "Indicador superficial presion-intensidad FpI [dB]",
    # ANP fleet database guide
    "ANP NPD Curves - Boeing 747-100 / JT9DBD (SEL, departure)":
        "Curvas NPD ANP - Boeing 747-100 / JT9DBD (SEL, despegue)",
    "ANP Default Departure Profile - Boeing 747-100 / JT9DBD":
        "Perfil de despegue por defecto ANP - Boeing 747-100 / JT9DBD",
    "power parameter: CNT (lb)\nmarkers: tabulated NPD nodes":
        "parámetro de potencia: CNT (lb)\nmarcadores: nodos NPD tabulados",
    "stage length 1, 11 fixed points": "etapa 1, 11 puntos fijos",
    # The axis labels and the legend come from the library renderer, so they
    # are worded exactly as src/phonometry/_plot/aircraft.py translates them.
    "Along-track distance [km]": "Distancia sobre la ruta [km]",
    "Altitude AFE [m]": "Altitud AFE [m]",
    "ground roll": "rodaje en pista",
    "Bearing Fault Lines on a Measured Envelope Spectrum":
        "Líneas de fallo del rodamiento sobre un espectro de envolvente medido",
    "envelope spectrum of the 2-4 kHz band":
        "espectro de envolvente de la banda 2-4 kHz",
    "predicted BPFO and harmonics": "BPFO previsto y sus armónicos",
    "predicted BPFI": "BPFI previsto",
    "predicted BSF": "BSF previsto",
    "shaft rate": "frecuencia de giro del eje",
    "Envelope amplitude": "Amplitud de la envolvente",
    "15 rollers, D = 34 mm, d = 6 mm, φ = 12.96°, 2000 r/min\n"
    "BPFO = 207 Hz, BPFI = 293 Hz\n"
    "the envelope lines fall on BPFO, not on BPFI: outer-race spall":
        "15 rodillos, D = 34 mm, d = 6 mm, φ = 12,96°, 2000 r/min\n"
        "BPFO = 207 Hz, BPFI = 293 Hz\n"
        "las líneas de la envolvente caen en BPFO, no en BPFI: "
        "descascarillado de la pista exterior",
    "Coupling Loss Factors: Prediction and Power Injection":
        "Factores de pérdidas por acoplamiento: predicción e inyección de potencia",
    "Predicted: Weld Against Bolts":
        "Predicción: soldadura frente a tornillos",
    "Measured: Power Injection, 500 Hz Octave":
        "Medida: inyección de potencia, octava de 500 Hz",
    "Coupling loss factor": "Factor de pérdidas por acoplamiento",
    "Loss factor": "Factor de pérdidas",
    r"welded line junction $\eta_{12}$":
        r"unión lineal soldada $\eta_{12}$",
    r"12 bolts, point connections $\eta_{12}$":
        r"12 tornillos, conexiones puntuales $\eta_{12}$",
    r"internal loss factor $\eta_1$": r"factor de pérdidas interno $\eta_1$",
    "Plateau Estimate Against the Physical Panel Model":
        "Estimación por meseta frente al modelo físico del panel",
    "coincidence plateau (A to B)": "meseta de coincidencia (A a B)",
    "physical model (mass law + coincidence + damping)":
        "modelo físico (ley de masas + coincidencia + amortiguamiento)",
    "plateau estimate (Norton Table 3.1)":
        "estimación por meseta (Norton, tabla 3.1)",
    "critical frequency fc": "frecuencia crítica fc",
    "Transmission loss TL [dB]": "Pérdida por transmisión TL [dB]",
    "identical below A; the plateau replaces the whole coincidence region":
        "idénticas por debajo de A; la meseta sustituye toda la región de "
        "coincidencia",
    "Corrugating a Sheet Flattens Its Sound Reduction Index":
        "Grecar una chapa aplana su índice de reducción acústica",
    r"coincidence range $f_{c1}$ to $f_{c2}$":
        r"rango de coincidencia $f_{c1}$ a $f_{c2}$",
    r"flat 1 mm sheet (isotropic, single $f_c$)":
        r"chapa plana de 1 mm (isótropa, un solo $f_c$)",
    "corrugated sheet (orthotropic, diffuse-field integral)":
        "chapa grecada (ortótropa, integral en campo difuso)",
    "Heckl's approximation": "aproximación de Heckl",
    "A Limp Frame Carries Its Own Inertia":
        "Un esqueleto flexible arrastra su propia inercia",
    r"Normalised effective density $\rho_e/\rho_0$":
        r"Densidad efectiva normalizada $\rho_e/\rho_0$",
    "rigid frame, real part": "esqueleto rígido, parte real",
    "rigid frame, imaginary part": "esqueleto rígido, parte imaginaria",
    "limp frame, real part": "esqueleto flexible, parte real",
    "limp frame, imaginary part": "esqueleto flexible, parte imaginaria",
    "Only an Elastic Frame Resonates":
        "Solo un esqueleto elástico resuena",
    r"Normalised surface impedance $Z_s/\rho_0 c_0$":
        r"Impedancia superficial normalizada $Z_s/\rho_0 c_0$",
    "Biot poroelastic, real part": "Biot poroelástico, parte real",
    "Biot poroelastic, imaginary part": "Biot poroelástico, parte imaginaria",
    "Glass wool, 100 mm glued to a rigid wall: porosity 0.94,\n"
    "flow resistivity 40 kPa s/m\u00b2, frame density 130 kg/m\u00b3,\n"
    "shear modulus 2.2 MPa":
        "Lana de vidrio, 100 mm pegada a una pared r\u00edgida: porosidad 0,94,\n"
        "resistividad al flujo 40 kPa s/m\u00b2, densidad del esqueleto 130 kg/m\u00b3,\n"
        "m\u00f3dulo de cizalla 2,2 MPa",
    "2D FDTD wavefront in a hall of columns":
        "Frente de onda FDTD 2D en una sala de columnas",
    "loudspeaker": "altavoz",
    "anechoic termination": "terminación anecoica",
    "rigid plug": "tapón rígido",
    "|p| envelope": "envolvente |p|",
    "The 2 cm metadiffuser scatters like the 27 cm QRD (2 kHz)":
        "El metadifusor de 2 cm dispersa como el QRD de 27 cm (2 kHz)",
    "Metadiffuser, panel 2 cm": "Metadifusor, panel de 2 cm",
    "Far field of the meshed metadiffuser vs the model (2 kHz)":
        "Campo lejano del metadifusor mallado frente al modelo (2 kHz)",
    "TMM + Fraunhofer model": "Modelo TMM + Fraunhofer",
    "FDTD + NTFF, panel meshed at 0.5 mm":
        "FDTD + NTFF, panel mallado a 0,5 mm",
    "Schroeder diffuser vs metadiffuser (2D FDTD)":
        "Difusor de Schroeder frente a metadifusor (FDTD 2D)",
    "QRD, wells down to 27 cm": "QRD, pozos de hasta 27 cm",
    "Metadiffuser, 2 cm panel": "Metadifusor, panel de 2 cm",
    "real slits and resonators meshed at 0.25 mm":
        "rendijas y resonadores mallados a 0,25 mm",
    "a collimated specular beam": "un haz especular colimado",
    "a wide scattered fan": "un abanico dispersado ancho",
    "the same fan, from 2 cm": "el mismo abanico, con 2 cm",
    "QRD, wells up to 27.4 cm": "QRD, pozos de hasta 27,4 cm",
    "The virtual impedance tube: standing waves read the absorption "
    "(2D FDTD)":
        "El tubo de impedancia virtual: las ondas estacionarias leen la "
        "absorción (FDTD 2D)",
    "Rigid end: deep minima, |r| ~ 1":
        "Extremo rígido: mínimos profundos, |r| ~ 1",
    "10 cm lossy sample: shallow minima":
        "Muestra disipativa de 10 cm: mínimos poco profundos",
    "The virtual transmission tube: what gets through (2D FDTD)":
        "El tubo de transmisión virtual: lo que se transmite (FDTD 2D)",
    "Empty tube: the packet crosses unchanged":
        "Tubo vacío: el paquete lo cruza intacto",
    "10 cm lossy layer: reflected + attenuated transmission":
        "Capa disipativa de 10 cm: reflexión + transmisión atenuada",
    "Position along the tube [m]": "Posición a lo largo del tubo [m]",
    "Frequency [Hz]": "Frecuencia [Hz]",
    "Rayleigh wave along a free aluminium surface (elastic FDTD)":
        "Onda de Rayleigh en una superficie libre de aluminio (FDTD elástico)",
    "P wave front": "Frente de onda P",
    "S wave front": "Frente de onda S",
    "Rayleigh wave": "Onda de Rayleigh",
    "Scholte wave along a water-sediment interface (elastic FDTD)":
        "Onda de Scholte en una interfase agua-sedimento (FDTD elástico)",
    "Scholte wave, evanescent on both sides":
        "Onda de Scholte, evanescente a ambos lados",
    "direct water wave": "onda directa en el agua",
    "water 1500 m/s": "agua 1500 m/s",
    "seabed 3500 / 2000 m/s": "sedimento 3500 / 2000 m/s",
    "Particle velocity [m/s]": "Velocidad de partícula [m/s]",
    "Predicted diffusion from design (Cox & D'Antonio Fraunhofer model)":
        "Difusión predicha desde el diseño (modelo de Fraunhofer de Cox y D'Antonio)",
    "Predicted diffusion coefficient d":
        "Coeficiente de difusión predicho d",
    "N = 7 QRD design": "Diseño QRD N = 7",
    "Flat panel": "Panel plano",
    "Atmospheric Refraction: Ray Bending and the Acoustic Shadow":
        "Refracción atmosférica: curvatura de rayos y zona de sombra",
    "Sound rays (upward refraction)": "Rayos sonoros (refracción hacia arriba)",
    "Tone-Burst Test Signal (IEC 60268-1)":
        "Señal de prueba de salvas de tono (IEC 60268-1)",
    "Single 5 ms burst of 5 kHz tone (25 full periods)":
        "Salva única de 5 ms de tono de 5 kHz (25 períodos completos)",
    "Repetitive train: 10 bursts per second (duty cycle 5 %)":
        "Tren repetitivo: 10 salvas por segundo (ciclo de trabajo 5 %)",
    "Gating envelope": "Envolvente de conmutación",
    "Window Functions: The Spectral Trade-off (Harris 1978)":
        "Ventanas de análisis: el compromiso espectral (Harris 1978)",
    "Parametric EQ Biquads (RBJ Audio EQ Cookbook)":
        "Biquads de EQ paramétrico (RBJ Audio EQ Cookbook)",
    "Peaking +6 dB (Q = 1.4)": "Campana +6 dB (Q = 1.4)",
    "Low shelf +6 dB": "Shelving grave +6 dB",
    "High shelf -6 dB": "Shelving agudo -6 dB",
    "Low-pass (Q = 0.707)": "Paso bajo (Q = 0.707)",
    "High-pass (Q = 0.707)": "Paso alto (Q = 0.707)",
    "Band-pass (Q = 2)": "Paso banda (Q = 2)",
    "Notch (Q = 6)": "Rechazo de banda (Q = 6)",
    "Frequency offset [DFT bins]": "Desplazamiento en frecuencia [bins de la DFT]",
    "Level re main lobe [dB]": "Nivel re lóbulo principal [dB]",
    "GFPE relative sound level": "Nivel sonoro relativo GFPE",
    "Shadow-zone boundary": "Límite de la zona de sombra",
    "Effective sound speed [m/s]": "Velocidad efectiva del sonido [m/s]",
    "Theoretical panel sound insulation (Bies / Hopkins / Cremer)":
        "Aislamiento acústico teórico de paneles (Bies / Hopkins / Cremer)",
    "Single panel: mass law and coincidence":
        "Panel simple: ley de masas y coincidencia",
    "Double wall: mass-spring-mass resonance":
        "Pared doble: resonancia masa-muelle-masa",
    "Radiation efficiency of a bending plate":
        "Eficiencia de radiación de una placa en flexión",
    "Composite wall with a small aperture":
        "Pared compuesta con una abertura pequeña",
    "Sound reduction index R [dB]": "Índice de reducción acústica R [dB]",
    r"Radiation efficiency $\sigma$": r"Eficiencia de radiación $\sigma$",
    "field-incidence mass law": "ley de masas de campo",
    "single panel R (Sharp)": "R de panel simple (Sharp)",
    "double wall R": "R de pared doble",
    "single leaf (total mass)": "hoja simple (masa total)",
    "solid wall alone": "pared maciza sola",
    "wall + 1 % open slit": "pared + rendija 1 % abierta",
    "open-area limit": "límite de área abierta",
    "Aircraft Atmospheric Absorption (SAE ARP 5534)":
        "Absorción atmosférica aeronáutica (SAE ARP 5534)",
    "Attenuation [dB]": "Atenuación [dB]",
    "Expansion-chamber transmission loss (Bies Eq. 8.111)":
        "Pérdida de transmisión de cámara de expansión (Bies Ec. 8.111)",
    "Area ratio m = Sexp/Sduct": "Relación de áreas m = Sexp/Sduct",
    "Noise-Power-Distance Curves (ECAC Doc 29)":
        "Curvas nivel-potencia-distancia (ECAC Doc 29)",
    "Aircraft Departure SEL Contour (ECAC Doc 29)":
        "Contorno SEL de despegue (ECAC Doc 29)",
    "Aircraft noise contour (ECAC Doc 29)": "Contorno de ruido de aeronave (ECAC Doc 29)",
    "Start-of-Roll Directivity ΔSOR (ECAC Doc 29 §4.5.7)":
        "Directividad de inicio de rodaje ΔSOR (ECAC Doc 29 §4.5.7)",
    "Turbofan jet (Eq. 4-24a)": "Reactor turbofán (Ec. 4-24a)",
    "Turboprop (Eq. 4-24b)": "Turbohélice (Ec. 4-24b)",
    "90°\nabeam": "90°\ntravés",
    "180° behind": "180° detrás",
    "radial axis: ΔSOR [dB] relative to abeam  ·  dSOR = 300 m":
        "eje radial: ΔSOR [dB] relativo al través  ·  dSOR = 300 m",
    "Rotorcraft Ground Effect (ECAC Doc 32, Chien-Soroka)":
        "Efecto de suelo de rotorcraft (ECAC Doc 32, Chien-Soroka)",
    "Rotorcraft Flyover Time History (ECAC Doc 32)":
        "Historia temporal de sobrevuelo de rotorcraft (ECAC Doc 32)",
    "Recorded time [s]": "Tiempo registrado [s]",
    "Rotorcraft Terrain Screening (ECAC Doc 32 / NORAH2)":
        "Apantallamiento por terreno de rotorcraft (ECAC Doc 32 / NORAH2)",
    "Terrain profile": "Perfil del terreno",
    "Line of sight": "Línea de visión",
    "Diffraction edges": "Aristas de difracción",
    "Terrain screening (ECAC Doc 32 / NORAH2 guidance)":
        "Apantallamiento por terreno (ECAC Doc 32 / guía NORAH2)",
    "Section distance [m]": "Distancia en la sección [m]",
    "Height [m]": "Altura [m]",
    "Flat ground (no hill)": "Suelo plano (sin colina)",
    "Screened by the hill (Eq. 45-47)": "Apantallado por la colina (Ec. 45-47)",
    "Ground and screening adjustment [dB]": "Ajuste de suelo y apantallamiento [dB]",
    "A-weighted sound pressure level [dB(A)]":
        "Nivel de presión acústica ponderado A [dB(A)]",
    "Received level $L_A(t)$": "Nivel recibido $L_A(t)$",
    "One-third-octave-band centre frequency [Hz]":
        "Frecuencia central de banda de 1/3 de octava [Hz]",
    "Ground-effect adjustment ΔLg [dB]": "Ajuste por efecto de suelo ΔLg [dB]",
    "Hard (asphalt/concrete, class G)": "Duro (asfalto/hormigón, clase G)",
    "Soft (grass/pasture, class D)": "Blando (hierba/pasto, clase D)",
    "x [km]": "x [km]",
    "y [km]": "y [km]",
    "Slant distance [m]": "Distancia oblicua [m]",
    "Event level [dB]": "Nivel de evento [dB]",
    "markers: tabulated NPD nodes\nlines: log-linear interpolation":
        "marcadores: nodos NPD tabulados\nlíneas: interpolación log-lineal",
    "25 °C, 70% RH\nsolid: SAE band, dashed: pure-tone mid-band":
        "25 °C, 70% HR\ncontinuo: banda SAE, discontinuo: tono puro medio de banda",
    # Emitted by phonometry.filters.design._showfilter (not by this script);
    # do not remove as "orphans".
    "Filter Bank Frequency Response": "Respuesta en frecuencia del banco de filtros",
    "Amplitude [dB]": "Amplitud [dB]",
    "Level [dB]": "Nivel [dB]",
    "Time [s]": "Tiempo [s]",
    "Amplitude": "Amplitud",
    "Error [dB]": "Error [dB]",
    "Group delay [ms]": "Retardo de grupo [ms]",
    "Level re steady state [dB]": "Nivel re estado estacionario [dB]",
    # ISO 18233 excitation signals + recovered impulse response
    "ISO 18233 excitation signals": "Señales de excitación ISO 18233",
    "Exponential sine sweep — waveform":
        "Barrido sinusoidal exponencial — forma de onda",
    "Sweep spectrogram (exponential rise)":
        "Espectrograma del barrido (ascenso exponencial)",
    "MLS magnitude spectrum (flat)": "Espectro de magnitud de la MLS (plano)",
    "Recovered room impulse response (ISO 18233)":
        "Respuesta al impulso de la sala recuperada (ISO 18233)",
    "Amplitude (norm.)": "Amplitud (norm.)",
    "Level re peak [dB]": "Nivel re pico [dB]",
    "Magnitude [dB]": "Magnitud [dB]",
    "Sample": "Muestra",
    "direct sound": "sonido directo",
    "reflections": "reflexiones",
    "Log-magnitude envelope": "Envolvente log-magnitud",
    "Schroeder decay (EDC)": "Decaimiento de Schroeder (EDC)",
    "Normalized Response": "Respuesta normalizada",
    "Normalized frequency  f / fm": "Frecuencia normalizada  f / fm",
    "Relative attenuation \u0394A [dB]": "Atenuaci\u00f3n relativa \u0394A [dB]",
    "Sound pressure level [dB re 20 \u00b5Pa]": "Nivel de presi\u00f3n ac\u00fastica [dB re 20 \u00b5Pa]",
    "1/3 Octave Band Analysis": "An\u00e1lisis en bandas de octava 1/3",
    "1/12 Octave Spectrogram (Fast windows, 87.5% overlap)":
        "Espectrograma 1/12 de octava (ventanas Fast, 87,5 % de solape)",
    "4 kHz Toneburst Response vs IEC 61672-1 Table 4 (FAST)":
        "Respuesta a r\u00e1fagas de 4 kHz vs Tabla 4 de IEC 61672-1 (FAST)",
    "A-Weighting": "Ponderaci\u00f3n A",
    "A-Weighting (reference)": "Ponderaci\u00f3n A (referencia)",
    "B-Weighting (historical)": "Ponderaci\u00f3n B (hist\u00f3rica)",
    "C-Weighting": "Ponderaci\u00f3n C",
    "D-Weighting (aircraft, withdrawn)":
        "Ponderaci\u00f3n D (aeronaves, retirada)",
    "AU-Weighting (audible + ultrasound)":
        "Ponderaci\u00f3n AU (audible + ultrasonido)",
    "Z-Weighting (Flat)": "Ponderaci\u00f3n Z (plana)",
    "G-weighting (ISO 7196)": "Ponderaci\u00f3n G (ISO 7196)",
    "G Frequency Weighting for Infrasound (ISO 7196:1995)":
        "Ponderaci\u00f3n frecuencial G para infrasonido (ISO 7196:1995)",
    "Bessel": "Bessel",
    "Bilinear error": "Error del dise\u00f1o bilineal",
    "Butterworth": "Butterworth",
    "Butterworth (Flat)": "Butterworth (plano)",
    "Butterworth order 6 (1 kHz octave band)":
        "Butterworth de orden 6 (banda de octava de 1 kHz)",
    "Causal filtering (group delay)": "Filtrado causal (retardo de grupo)",
    "Chebyshev I": "Chebyshev I",
    "Chebyshev II": "Chebyshev II",
    "Class 1 lower limit @ 12.5 kHz": "L\u00edmite inferior de clase 1 @ 12,5 kHz",
    "Class 2 minimum attenuation": "Atenuaci\u00f3n m\u00ednima de clase 2",
    "Continuous (whole signal)": "Continuo (se\u00f1al completa)",
    "Elliptic": "El\u00edptico",
    "FAST envelope": "Envolvente FAST",
    "Fast (125ms)": "Fast (125 ms)",
    "Fast level $L_p(t)$": "Nivel Fast $L_p(t)$",
    "Filter Architecture Comparison (Order 6, 1kHz Band)":
        "Comparativa de arquitecturas de filtro (orden 6, banda de 1 kHz)",
    "Forbidden for class 1 (too little attenuation)":
        "Prohibido para clase 1 (atenuaci\u00f3n insuficiente)",
    "Forbidden for class 1 (too much attenuation)":
        "Prohibido para clase 1 (atenuaci\u00f3n excesiva)",
    # weighting_class_mask figure (IEC 61672-1 Table 3 verifier)
    "Weighting Deviation vs IEC 61672-1:2013 Table 3 Limits":
        "Desviaci\u00f3n de ponderaci\u00f3n vs l\u00edmites de la Tabla 3 de IEC 61672-1:2013",
    "Class 1 acceptance region": "Regi\u00f3n de aceptaci\u00f3n de clase 1",
    "Class 1 upper/lower limit": "L\u00edmite superior/inferior de clase 1",
    "Class 2 upper/lower limit": "L\u00edmite superior/inferior de clase 2",
    "A weighting deviation (48 kHz)": "Desviaci\u00f3n de ponderaci\u00f3n A (48 kHz)",
    "C weighting deviation (48 kHz)": "Desviaci\u00f3n de ponderaci\u00f3n C (48 kHz)",
    "Deviation from design goal [dB]": "Desviaci\u00f3n del objetivo de dise\u00f1o [dB]",
    "Frequency Weighting Curves (IEC 61672-1)":
        "Curvas de ponderaci\u00f3n frecuencial (IEC 61672-1)",
    # special_weighting_responses figure (B, D and AU against the A reference)
    "Special Weighting Curves (B, D, AU)":
        "Curvas de ponderaci\u00f3n especiales (B, D y AU)",
    "AU is 13 dB below A at 16 kHz":
        "AU queda 13 dB por debajo de A en 16 kHz",
    "Group Delay Comparison (1 kHz Octave Band, Order 6)":
        "Comparativa de retardo de grupo (banda de 1 kHz, orden 6)",
    "Hearing threshold $T_f$ (Table 1)": "Umbral de audici\u00f3n $T_f$ (Tabla 1)",
    "Hearing threshold $T_f$": "Umbral de audici\u00f3n $T_f$",
    "High Pass (LR4)": "Paso alto (LR4)",
    "IEC 61672-1 analytic curve": "Curva anal\u00edtica IEC 61672-1",
    "ISO 7196 Table 2 nominals": "Nominales de la Tabla 2 de ISO 7196",
    "Impulse (35ms/1.5s)": "Impulse (35 ms/1,5 s)",
    "Independent blocks (state reset)": "Bloques independientes (estado reiniciado)",
    "Input Burst (Normalized)": "R\u00e1faga de entrada (normalizada)",
    "Input burst (250 Hz)": "R\u00e1faga de entrada (250 Hz)",
    "Left Channel: Pink Noise": "Canal izquierdo: ruido rosa",
    "Linkwitz-Riley Crossover (4th Order @ 1kHz)":
        "Crossover Linkwitz-Riley (4\u00ba orden @ 1 kHz)",
    "Low Pass (LR4)": "Paso bajo (LR4)",
    "Multichannel Analysis (Stereo Input)": "An\u00e1lisis multicanal (entrada est\u00e9reo)",
    "No state: each block restarts the filter transient":
        "Sin estado: cada bloque reinicia el transitorio del filtro",
    "Normal Equal-Loudness-Level Contours (ISO 226:2023)":
        "L\u00edneas isof\u00f3nicas normales (ISO 226:2023)",
    "ISO 226:2023 defines 20 to 90 phon; above 80 phon the contour is defined only up to 4 kHz.":
        "ISO 226:2023 define de 20 a 90 fonios; por encima de 80 fonios la curva solo se define hasta 4 kHz.",
    "Original Signal (250 Hz + 1000 Hz Sum) @ 48 kHz":
        "Se\u00f1al original (suma de 250 Hz + 1000 Hz) @ 48 kHz",
    "Oversampled (high_accuracy=True)": "Sobremuestreado (high_accuracy=True)",
    "Plain bilinear (high_accuracy=False)": "Bilineal simple (high_accuracy=False)",
    "Raw PSD": "PSD sin filtrar",
    "Raw Signal Spectrum (PSD)": "Espectro de la se\u00f1al (PSD)",
    "Relative Attenuation vs IEC 61260-1:2014 Class Limits":
        "Atenuaci\u00f3n relativa vs l\u00edmites de clase de IEC 61260-1:2014",
    "Right Channel: Log Sine Sweep": "Canal derecho: barrido senoidal logar\u00edtmico",
    "Slow (1000ms)": "Slow (1000 ms)",
    "Stateful blocks (state carried)": "Bloques con estado (estado conservado)",
    "Statistical Levels L10 / L50 / L90 (Fast envelope)":
        "Niveles estad\u00edsticos L10 / L50 / L90 (envolvente Fast)",
    "Sum (Flat)": "Suma (plana)",
    "Time Weighting Ballistics (IEC 61672-1)":
        "Ponderaci\u00f3n temporal F/S/I (IEC 61672-1)",
    "Zero-Phase Filtering: Group Delay Elimination (250 Hz Band)":
        "Filtrado de fase cero: eliminaci\u00f3n del retardo de grupo (banda de 250 Hz)",
    "Zoom at -3 dB (Log Scale)": "Zoom en -3 dB (escala log)",
    "Zoom: A-weighting is positive (max +1.27 dB @ 2.5 kHz)":
        "Zoom: la ponderaci\u00f3n A es positiva (m\u00e1x +1,27 dB @ 2,5 kHz)",
    "block boundary:\nfilter transient restarts":
        "frontera de bloque:\nse reinicia el transitorio del filtro",
    "high_accuracy error": "Error con high_accuracy",
    "stateful=True: block outputs equal the continuous result":
        "stateful=True: los bloques igualan el resultado continuo",
    "zero_phase=True (aligned)": "zero_phase=True (alineado)",
    "0 dB @ 10 Hz": "0 dB @ 10 Hz",
    # filter_class0_mask figure (IEC 61260:1995 / ANSI S1.11-2004 class 0)
    "Pass-band Class 0/1/2 Limits (IEC 61260:1995 / ANSI S1.11-2004)":
        "Límites de clase 0/1/2 en banda de paso (IEC 61260:1995 / ANSI S1.11-2004)",
    "Class 0 corridor": "Corredor de clase 0",
    "Class 1 corridor": "Corredor de clase 1",
    "Class 2 corridor": "Corredor de clase 2",
    # intensity_insulation figure (ISO 15186-1)
    "ISO 15186-1 Intensity Sound Reduction Index (RI and RI,M)":
        "Índice de reducción acústica por intensidad ISO 15186-1 (RI y RI,M)",
    "Sound reduction index [dB]": "Índice de reducción acústica [dB]",
    "Kc adaptation": "Adaptación Kc",
    "RI (intensity)": "RI (intensidad)",
    # survey_insulation figure (ISO 10052)
    "ISO 10052 Survey Method: Reverberation-Index Correction":
        "Método de control ISO 10052: corrección por índice de reverberación",
    "Level difference [dB]": "Diferencia de nivel [dB]",
    "D (level difference)": "D (diferencia de nivel)",
    "DnT (standardized)": "DnT (estandarizada)",
    "octave bands, T0 = 0.5 s": "bandas de octava, T0 = 0,5 s",
    # absorption_uncertainty figure (ISO 12999-2)
    "ISO 12999-2 Sound Absorption Coefficient Uncertainty":
        "Incertidumbre del coeficiente de absorción acústica (ISO 12999-2)",
    "+/-U (k = 2), reproducibility": "±U (k = 2), reproducibilidad",
    "alpha_s (ISO 354)": "alpha_s (ISO 354)",
    # floor_covering_improvement figure (ISO 16251-1)
    "ISO 16251-1 Floor-Covering Impact Sound Improvement":
        "Mejora a impacto de revestimientos de suelo (ISO 16251-1)",
    "Improvement of impact sound insulation [dB]":
        "Mejora del aislamiento a impactos [dB]",
    "delta-L (improvement)": "delta-L (mejora)",
    # heavy_impact_sources figure (ISO 16283-2 / JIS A 1418-2 / ISO 717-2)
    "Standard heavy impact sources\n(ISO 16283-2 Table A.1, JIS A 1418-2 Tables A.1/A.2)":
        "Fuentes de impacto pesadas normalizadas\n(ISO 16283-2 Tabla A.1, "
        "JIS A 1418-2 Tablas A.1/A.2)",
    "Impact force exposure level LFE [dB re 1 N]":
        "Nivel de exposición a la fuerza de impacto LFE [dB re 1 N]",
    "rubber ball tolerance": "tolerancia de la pelota de caucho",
    "rubber ball nominal": "pelota de caucho nominal",
    "bang machine tolerance": "tolerancia de la máquina de neumático",
    "bang machine nominal": "máquina de neumático nominal",
    "A-weighted heavy-impact rating\n(ISO 717-2 Annex D, Table D.4 worked example)":
        "Índice de impacto pesado ponderado A\n(ISO 717-2 Anexo D, ejemplo de "
        "la Tabla D.4)",
    "Maximum impact sound pressure level [dB]":
        "Nivel máximo de presión acústica de impactos [dB]",
    "Li,Fmax (measured)": "Li,Fmax (medido)",
    "Li,Fmax + A (Table D.3)": "Li,Fmax + A (Tabla D.3)",
    # ceiling_plenum_flanking figure (Vigran 9.2.3 / ASTM E1414 / ASTM E413)
    "Suspended-ceiling plenum path\n(one-dimensional model, LR = 4.75 m, reflecting sidewalls)":
        "Trayecto por plenum de techo suspendido\n(modelo unidimensional, "
        "LR = 4.75 m, paredes laterales reflectantes)",
    "RS + RR (two ceilings)": "RS + RR (dos techos)",
    "Ceiling attenuation class\n(ASTM E1414/E413, CAC = 34 dB)":
        "Clase de atenuación de techo\n(ASTM E1414/E413, CAC = 34 dB)",
    "Normalized ceiling attenuation Dn,c [dB]":
        "Diferencia de niveles normalizada del techo Dn,c [dB]",
    "Dn,c (measured)": "Dn,c (medido)",
    "ASTM E413 contour, fitted": "curva ASTM E413 ajustada",
    "deficiencies": "deficiencias",
    # masonry_wall_ties figure (Hopkins 3.11.3.2 / 4.3.5.4.1)
    "Wall-tie structure-borne coupling\n(point-connection model, 2.5 ties/m2)":
        "Acoplamiento estructural por llaves de muro\n(modelo de unión puntual, "
        "2.5 llaves/m2)",
    "Coupling loss factor eta_ij": "Factor de pérdidas por acoplamiento eta_ij",
    "rigid connection (Yc = 0)": "unión rígida (Yc = 0)",
    "Ties stiffen the cavity\n(140 kg/m2 leaves, 75 mm cavity, 2.5 ties/m2)":
        "Las llaves rigidizan la cámara\n(hojas de 140 kg/m2, cámara de 75 mm, "
        "2.5 llaves/m2)",
    "cavity wall, no ties": "muro con cámara, sin llaves",
    "2.5 ties/m2, k = 2 MN/m": "2,5 llaves/m2, k = 2 MN/m",
    "combined-mass range added by the ties":
        "rango de masa combinada añadido por las llaves",
    # floating_floor_prediction / soft_covering_prediction figures
    "Floating-Floor Impact Improvement Above the Mass-Spring Resonance":
        "Mejora a impactos del suelo flotante por encima de la resonancia masa-resorte",
    "30 log10(f/f0) (sand-cement screed)": "30 log10(f/f0) (solera de mortero de cemento)",
    "40 log10(f/f0) (asphalt, dry)": "40 log10(f/f0) (asfalto, seco)",
    "40 log10(f/f0) + hammer term (chipboard)":
        "40 log10(f/f0) + término del martillo (tablero de partículas)",
    "ISO 12354-2 Annex G bands": "Bandas del Anexo G de ISO 12354-2",
    "35 mm screed m' = 73.5 kg/m2 on s' = 8 MN/m3\n"
    "delta-Lw = 32.2 dB  (ISO 12354-2 Formula C.4)":
        "solera de 35 mm m' = 73.5 kg/m2 sobre s' = 8 MN/m3\n"
        "delta-Lw = 32.2 dB  (fórmula C.4 de ISO 12354-2)",
    "Soft Floor Covering Improvement From the Hammer Contact Stiffness":
        "Mejora del revestimiento de suelo blando a partir de la rigidez de contacto del martillo",
    "two-line estimate (0 dB, 12 dB/oct)":
        "estimación de dos rectas (0 dB, 12 dB/oct)",
    "140 mm concrete slab, hammer 0.5 kg, r = 15 mm\n"
    "No. 1: fco = 2318 Hz\nNo. 2: fco = 100 Hz":
        "losa de hormigón de 140 mm, martillo 0.5 kg, r = 15 mm\n"
        "N.o 1: fco = 2318 Hz\nN.o 2: fco = 100 Hz",
    # flanking_transmission figure (ISO 10848)
    "ISO 10848 Junction Vibration Reduction Index":
        "Índice de reducción de vibraciones de unión (ISO 10848)",
    "Vibration reduction index Kij [dB]":
        "Índice de reducción de vibraciones Kij [dB]",
    "Kij (ISO 10848)": "Kij (ISO 10848)",
    "mean Kij (200-1250 Hz)": "Kij medio (200-1250 Hz)",
    # tonal_audibility figure (ISO 1996-2)
    "ISO 1996-2 Tonal Adjustment": "Ajuste tonal ISO 1996-2",
    r"Tonal audibility $\Delta L_{ta}$ [dB]":
        r"Audibilidad tonal $\Delta L_{ta}$ [dB]",
    r"Tonal adjustment $K_t$ [dB]": r"Ajuste tonal $K_t$ [dB]",
    r"$K_t(\Delta L_{ta})$ (Formulae C.4-C.6)":
        r"$K_t(\Delta L_{ta})$ (Fórmulas C.4-C.6)",
    "Annex C.5 examples": "ejemplos del Anexo C.5",
    "mid-range tone": "tono de rango medio",
    # reverberation_models figure (Sabine / Eyring / Millington / Fitzroy / Arau)
    "Reverberation-time prediction models":
        "Modelos de predicción del tiempo de reverberación",
    # dynamic_stiffness figure (EN 29052-1)
    "EN 29052-1 Floating-Floor Resonance":
        "Resonancia de suelo flotante EN 29052-1",
    r"Dynamic stiffness per unit area $s'$ [MN/m³]":
        r"Rigidez dinámica por unidad de área $s'$ [MN/m³]",
    r"Natural frequency $f_0$ [Hz]": r"Frecuencia natural $f_0$ [Hz]",
    "design point": "punto de diseño",
    # junction_transmission figure (Hopkins 5.2.1.3, Cremer/Craik)
    "Bending-wave transmission at a rigid X-junction (Hopkins 5.2.1.3)":
        "Transmisión de onda de flexión en una unión X rígida (Hopkins 5.2.1.3)",
    "Incidence angle [degrees]": "Ángulo de incidencia [grados]",
    r"Transmission coefficient $\tau$": r"Coeficiente de transmisión $\tau$",
    r"corner $\tau_{12}(\theta)$": r"esquina $\tau_{12}(\theta)$",
    r"straight $\tau_{13}(\theta)$": r"recta $\tau_{13}(\theta)$",
    "corner average": "media esquina",
    "straight average": "media recta",
    # mechanical_mobility figure (ISO 7626-1)
    "ISO 7626-1 Mechanical Mobility FRFs":
        "FRF de movilidad mecánica ISO 7626-1",
    "Normalized FRF magnitude": "Magnitud FRF normalizada",
    "Receptance $|H|$ (× k)": "Receptancia $|H|$ (× k)",
    r"Mobility $|Y|$ (× k/$\omega_0$)": r"Movilidad $|Y|$ (× k/$\omega_0$)",
    r"Accelerance $|A|$ (× k/$\omega_0^2$)":
        r"Acelerancia $|A|$ (× k/$\omega_0^2$)",
    "resonance $f_0$": "resonancia $f_0$",
    # transfer_stiffness figure (ISO 10846)
    "ISO 10846 Dynamic Transfer Stiffness":
        "Rigidez dinámica de transferencia ISO 10846",
    r"Transfer stiffness level $L_k$ [dB re 1 N/m]":
        r"Nivel de rigidez de transferencia $L_k$ [dB re 1 N/m]",
    r"true $L_k$ of $k_{2,1}=k+j\omega c$":
        r"$L_k$ real de $k_{2,1}=k+j\omega c$",
    r"indirect method $-(2\pi f)^2 m_2 T$":
        r"método indirecto $-(2\pi f)^2 m_2 T$",
    # rigid_mass_calibration figure (ISO 7626-2, 7.5.2)
    "ISO 7626-2 Rigid-Mass Calibration Check":
        "Verificación de calibración con masa rígida ISO 7626-2",
    "Accelerance $|A|$ [1/kg]": "Acelerancia $|A|$ [1/kg]",
    "Deviation [%]": "Desviación [%]",
    r"expected $|A| = 1/m$": r"esperado $|A| = 1/m$",
    r"$\pm$5 % tolerance band": r"banda de tolerancia $\pm$5 %",
    "within tolerance": "dentro de tolerancia",
    "out of tolerance": "fuera de tolerancia",
    # vibration_sound_power figure (ISO/TS 7849)
    "ISO/TS 7849 Sound Power from Surface Vibration":
        "Potencia acústica desde vibración superficial ISO/TS 7849",
    r"Sound power level $L_W$ [dB re 1 pW]":
        r"Nivel de potencia acústica $L_W$ [dB re 1 pW]",
    "Part 1 upper limit ($\\varepsilon$ = 1)":
        "Parte 1 límite superior ($\\varepsilon$ = 1)",
    "Part 2 engineering ($\\varepsilon$ measured)":
        "Parte 2 ingeniería ($\\varepsilon$ medido)",
    # structure_borne_power figure (EN 15657)
    "EN 15657 Reception-Plate Structure-Borne Sound Power":
        "Potencia acústica estructural en placa receptora EN 15657",
    r"Structure-borne power level $L_{Ws}$ [dB re 1 pW]":
        r"Nivel de potencia estructural $L_{Ws}$ [dB re 1 pW]",
    "low-mobility plate": "placa de baja movilidad",
    "high-mobility plate": "placa de alta movilidad",
    # installed_structure_borne figure (EN 12354-5)
    "EN 12354-5 Installed Structure-Borne Sound":
        "Ruido estructural instalado EN 12354-5",
    r"characteristic $L_{Ws,c}$ (EN 15657)":
        r"característica $L_{Ws,c}$ (EN 15657)",
    r"installed $L_{Ws,inst}$ = $L_{Ws,c}-D_C$":
        r"instalada $L_{Ws,inst}$ = $L_{Ws,c}-D_C$",
    "paths $L_{n,s,ij}$": "caminos $L_{n,s,ij}$",
    r"total $L_{n,s}$": r"total $L_{n,s}$",
    # tone_audibility figure (ISO/PAS 20065)
    "ISO/PAS 20065 Tonal Audibility": "Audibilidad tonal ISO/PAS 20065",
    r"Audibility $\Delta L$ [dB]": r"Audibilidad $\Delta L$ [dB]",
    r"threshold $\Delta L = 0$ dB": r"umbral $\Delta L = 0$ dB",
    # facade_prediction figure (EN 12354-3 Annex F)
    "EN 12354-3 Façade Sound Insulation (Annex F example)":
        "Aislamiento acústico de fachada EN 12354-3 (ejemplo del Anexo F)",
    "Reduction index / level difference [dB]":
        "Índice de reducción / diferencia de niveles [dB]",
    "Rp — wall": "Rp — muro",
    "Rp — window": "Rp — ventana",
    "Rp — skylight": "Rp — claraboya",
    "Rp — air inlet": "Rp — entrada de aire",
    "R′ (façade)": "R′ (fachada)",
    "air inlet limits the low bands": "la entrada de aire limita las bandas bajas",
    # Scattering coefficient spectrum (ISO 17497-1)
    "Random-incidence scattering coefficient (ISO 17497-1)":
        "Coeficiente de dispersión de incidencia aleatoria (ISO 17497-1)",
    "Scattering coefficient s": "Coeficiente de dispersión s",
    # In-situ road-surface absorption (ISO 13472-1)
    "In-situ road-surface absorption (ISO 13472-1)":
        "Absorción in situ de pavimentos (ISO 13472-1)",
    "Absorption coefficient alpha": "Coeficiente de absorción alpha",
    "Porous layer 50 mm (sigma = 20 kPa s/m2)":
        "Capa porosa de 50 mm (sigma = 20 kPa s/m2)",
    "Microperforated panel + 48 mm cavity":
        "Panel microperforado + cámara de 48 mm",
    "Perforated panel 6 mm + porous 25 mm + air":
        "Panel perforado 6 mm + poroso 25 mm + aire",
    "Membrane 2 kg/m2 + air + porous 38 mm":
        "Membrana 2 kg/m2 + aire + poroso 38 mm",
    "Helmholtz closed form": "Forma cerrada de Helmholtz",
    "Membrane closed form": "Forma cerrada de membrana",
    "Multilayer Absorber Prediction (Transfer-Matrix Method)":
        "Predicción de absorbentes multicapa (método de matrices de transferencia)",
    "Normal incidence, rigid backing, 50 mm total depth":
        "Incidencia normal, respaldo rígido, 50 mm de profundidad total",
    # Human vibration (ISO 8041-1 / ISO 2631 / ISO 5349 / 2002/44/EC)
    "Whole-body vertical weighting Wk (ISO 8041-1)":
        "Ponderación vertical de cuerpo entero Wk (ISO 8041-1)",
    "Weighting factor [dB]": "Factor de ponderación [dB]",
    "One-third-octave band [Hz]": "Banda de tercio de octava [Hz]",
    "Band audibility": "Audibilidad de banda",
    r"Band audibility $A_i$": r"Audibilidad de banda $A_i$",
    r"Importance-weighted $I_i\,A_i$ (scaled)":
        r"Ponderada por importancia $I_i\,A_i$ (escalada)",
    "r.m.s. acceleration [m/s$^2$]": "Aceleración eficaz [m/s$^2$]",
    "Unweighted $a_i$": "Sin ponderar $a_i$",
    "Weighted $W_i\\,a_i$ (Wk)": "Ponderada $W_i\\,a_i$ (Wk)",
    "Daily exposure A(8) [m/s$^2$]": "Exposición diaria A(8) [m/s$^2$]",
    "brush-saw": "desbrozadora",
    "felling": "tala",
    "stripping": "descortezado",
    # Precision sound power (ISO 3745 / ISO 9614-3)
    "Sound power level LW [dB]": "Nivel de potencia acústica LW [dB]",
    "Non-applicable band": "Banda no aplicable",
    "Stable tone (good coupling)": "Tono estable (buen acoplamiento)",
    "3% AM tone (loose coupling)": "Tono con AM del 3 % (acoplamiento flojo)",
    "IEC 60942:2017 class 1 limit (deviation from mean)":
        "L\u00edmite de clase 1 de IEC 60942:2017 (desviaci\u00f3n de la media)",
    "Calibration Tone Stability Check (IEC 60942:2017, 5.3.3)":
        "Comprobaci\u00f3n de estabilidad del tono de calibraci\u00f3n (IEC 60942:2017, 5.3.3)",
    "F-weighted level re mean [dB]": "Nivel con ponderaci\u00f3n F re media [dB]",
    "Fast level of the event": "Nivel Fast del evento",
    "Leq over the whole event": "Leq de todo el evento",
    "SEL: same energy in 1 s": "SEL: la misma energ\u00eda en 1 s",
    "equal energy": "igual energ\u00eda",
    "Sound Exposure Level: the event normalized to 1 s":
        "Nivel de exposici\u00f3n sonora: el evento normalizado a 1 s",
    "Level [dBFS]": "Nivel [dBFS]",
    "Hourly LAeq": "LAeq horario",
    "Lday (+0 dB)": "Ld\u00eda (+0 dB)",
    "Levening + 5 dB": "Ltarde + 5 dB",
    "Lnight + 10 dB": "Lnoche + 10 dB",
    "Day-Evening-Night Level Lden (ISO 1996-1)":
        "Nivel d\u00eda-tarde-noche Lden (ISO 1996-1)",
    "Hour of day": "Hora del d\u00eda",
    "Averaged FFT spectrum (Hann)": "Espectro FFT promediado (Hann)",
    "Critical band around the tone": "Banda cr\u00edtica en torno al tono",
    "Tone-to-Noise Ratio (ECMA-418-1, clause 11)":
        "Relaci\u00f3n tono-ruido (ECMA-418-1, apartado 11)",
    "Bin power [dB]": "Potencia por bin [dB]",
    "Specific Loudness Pattern (ISO 532-1 Zwicker)":
        "Patr\u00f3n de sonoridad espec\u00edfica (Zwicker, ISO 532-1)",
    "Critical-band rate z [Bark]": "Raz\u00f3n de banda cr\u00edtica z [Bark]",
    "Specific loudness N' [sone/Bark]":
        "Sonoridad espec\u00edfica N' [sonios/Bark]",
    "Shaded area = total loudness N": "\u00c1rea sombreada = sonoridad total N",
    "STI vs Reverberation Time (IEC 60268-16)":
        "STI frente al tiempo de reverberaci\u00f3n (IEC 60268-16)",
    "Reverberation time T60 [s]": "Tiempo de reverberaci\u00f3n T60 [s]",
    "Analytic Schroeder MTF (closed form)":
        "MTF de Schroeder anal\u00edtica (forma cerrada)",
    "Measured (sti_from_impulse_response)":
        "Medido (sti_from_impulse_response)",
    "Annex F rating": "Calificaci\u00f3n del Anexo F",
    "Measured": "Medido",
    "Octave-band center frequency [Hz]":
        "Frecuencia central de banda de octava [Hz]",
    "Octave-band sound pressure level [dB]":
        "Nivel de presi\u00f3n ac\u00fastica por banda de octava [dB]",
    "Rumble tol. (+5 dB)": "Tol. retumbo (+5 dB)",
    "Hiss tol. (+3 dB)": "Tol. siseo (+3 dB)",
    "55 dB floor\n(16 Hz = 31.5 Hz)": "suelo de 55 dB\n(16 Hz = 31,5 Hz)",
    # ISO 18233 acquisition: SNR gain, harmonic separation, d_min bias,
    # modal count per band and time variance
    "Effective signal-to-noise ratio of the recovered impulse response":
        "Relación señal-ruido efectiva de la respuesta al impulso recuperada",
    "noise floor read here": "el ruido de fondo se lee aquí",
    "Harmonic distortion lands before t = 0 (ISO 18233 B.5)":
        "La distorsión armónica cae antes de t = 0 (ISO 18233 B.5)",
    "Arrival time relative to the linear impulse response [s]":
        "Tiempo de llegada respecto a la respuesta al impulso lineal [s]",
    "causal part: what impulse_response() returns":
        "parte causal: lo que devuelve impulse_response()",
    "A microphone inside d_min returns wrong numbers, not noisy ones":
        "Un micrófono dentro de d_min da números equivocados, no ruidosos",
    "Source–receiver distance [m]": "Distancia fuente–receptor [m]",
    "Decay time [s]": "Tiempo de caída [s]",
    "Clarity C80 [dB]": "Claridad C80 [dB]",
    "T30 (500–1000 Hz)": "T30 (500–1000 Hz)",
    "EDT (500–1000 Hz)": "EDT (500–1000 Hz)",
    "C80 (500–1000 Hz)": "C80 (500–1000 Hz)",
    "What the analysis band averages over (7 × 5 × 3 m room, V = 105 m³)":
        "Sobre qué promedia la banda de análisis (sala de 7 × 5 × 3 m, "
        "V = 105 m³)",
    "Modes inside the octave band": "Modos dentro de la banda de octava",
    "Below the Schroeder frequency": "Por debajo de la frecuencia de Schroeder",
    "Above it": "Por encima",
    "Time variance costs the MLS its dynamic range, not the sweep":
        "La variación temporal le cuesta el rango dinámico al MLS, no al barrido",
    "the MLS floor rises to the room's own early decay:\nthe tail is gone":
        "el suelo del MLS sube hasta la primera caída de la sala:\n"
        "la cola ha desaparecido",
    "sweep: the two traces lie on top of each other":
        "barrido: las dos trazas quedan superpuestas",
    # ISO 3382-3 Annex A quality ranges and the absorption-per-table window
    "The same two quantities at the two ends of Annex A":
        "Las mismas dos magnitudes en los dos extremos del Anexo A",
    "Lp,A,S,4m > 50 dB: poor": "Lp,A,S,4m > 50 dB: deficiente",
    "Lp,A,S,4m ≤ 48 dB: good target":
        "Lp,A,S,4m ≤ 48 dB: objetivo bueno",
    "rD ≤ 5 m: good": "rD ≤ 5 m: bueno",
    "rD > 10 m: poor": "rD > 10 m: deficiente",
    "A-weighted speech level [dB]": "Nivel de habla ponderado A [dB]",
    "Speech transmission index": "Índice de transmisión del habla",
    "STI = 0.50": "STI = 0,50",
    "STI = 0.20": "STI = 0,20",
    "The design window, for one layout":
        "La ventana de diseño, para una distribución",
    "Communication: A_tab > 6.31 r_s²  (L_SN > −6 dB)":
        "Comunicación: A_tab > 6,31 r_s²  (L_SN > −6 dB)",
    "Privacy: A_tab < 3.16 r_t²  (L_SN < −9 dB)":
        "Privacidad: A_tab < 3,16 r_t²  (L_SN < −9 dB)",
    "Separation [m]": "Separación [m]",
    "Absorption per occupied table A_tab [m²]":
        "Absorción por mesa ocupada A_tab [m²]",
    "Table spacing over cross-table separation, r_t / r_s":
        "Separación entre mesas frente a la de la mesa, r_t / r_s",
    "Width of the feasible A_tab window [m²]":
        "Anchura de la ventana factible de A_tab [m²]",
    "Two ratings: the RC Mark II tag reads the character":
        "Dos calificaciones: la etiqueta RC Mark II lee el carácter",
    "Tag threshold (D.3): +5 / +3 dB":
        "Umbral de etiqueta (D.3): +5 / +3 dB",
    "Level minus the room's own RC curve [dB]":
        "Nivel menos la curva RC propia de la sala [dB]",
    "ISO 7029 — age-related threshold (male)":
        "ISO 7029 — umbral por edad (hombres)",
    "ISO 389-7 — reference threshold of hearing":
        "ISO 389-7 — umbral de referencia de la audición",
    "Audiometric frequency [Hz]": "Frecuencia audiométrica [Hz]",
    "Median threshold deviation from age 18 [dB]":
        "Desviación mediana del umbral respecto a los 18 años [dB]",
    "Reference threshold [dB]": "Umbral de referencia [dB]",
    "Free-field (frontal)": "Campo libre (frontal)",
    "Diffuse-field": "Campo difuso",
    "ANSI S3.5-1997 — speech spectra by vocal effort":
        "ANSI S3.5-1997 — espectros de voz por esfuerzo vocal",
    "Speech spectrum level [dB SPL]": "Nivel del espectro de voz [dB SPL]",
    "SII vs vocal effort in a fixed noise":
        "SII frente al esfuerzo vocal en un ruido fijo",
    "Onset rate [dB/s]": "Tasa de crecimiento [dB/s]",
    "Predicted prominence $P$": "Prominencia prevista $P$",
    "Adjustment $K_I$ [dB]": "Ajuste $K_I$ [dB]",
    "Adjustment to $L_{Aeq}$": "Ajuste a $L_{Aeq}$",
    "Impulses": "Impulsos",
    "threshold $P = 5$": "umbral $P = 5$",
    "Transmissibility  seat $\\rightarrow$ spine":
        "Transmisibilidad  asiento $\\rightarrow$ columna",
    "Seat-to-spine transfer function":
        "Función de transferencia asiento-columna",
    "Stress variable $R$": "Variable de tensión $R$",
    "Probability of lumbar injury [%]": "Probabilidad de lesión lumbar [%]",
    "Injury probability (Annex C)": "Probabilidad de lesión (Anexo C)",
    "male": "hombre",
    "female": "mujer",
    "Equivalent absorption area $A$ [m$^2$]":
        "Área de absorción equivalente $A$ [m$^2$]",
    "Absorption area (Formula 1)": "Área de absorción (Fórmula 1)",
    "Reverberation time $T$ [s]": "Tiempo de reverberación $T$ [s]",
    "Reverberation time (Formula 5)": "Tiempo de reverberación (Fórmula 5)",
    "bare ceiling": "techo desnudo",
    "acoustic ceiling": "techo acústico",
    "Speech Intelligibility Index": "Índice de inteligibilidad del habla",
    "Normal": "Normal",
    "Raised": "Elevada",
    "Loud": "Fuerte",
    "Shout": "Grito",
    r"ISO 1999 — NIPTS at $L_{EX,8h}$ = 95 dB":
        r"ISO 1999 — NIPTS a $L_{EX,8h}$ = 95 dB",
    "ISO 1999 — HTLAN (male, age 60, 95 dB / 30 yr)":
        "ISO 1999 — HTLAN (hombres, 60 años, 95 dB / 30 años)",
    "Median NIPTS [dB]": "NIPTS mediana [dB]",
    "Hearing threshold level [dB]": "Nivel del umbral de audición [dB]",
    "10-90 % band (40 yr)": "Banda 10-90 % (40 años)",
    "Age (HTLA, ISO 7029)": "Edad (HTLA, ISO 7029)",
    "Noise (NIPTS)": "Ruido (NIPTS)",
    "Age + noise (HTLAN)": "Edad + ruido (HTLAN)",
    # Perception / hearing + speech (B15b)
    "ISO 7029 median at 4 kHz — men against women":
        "Mediana de ISO 7029 a 4 kHz: hombres frente a mujeres",
    "The spread around the median (male)":
        "La dispersión en torno a la mediana (hombres)",
    "Male": "Hombres",
    "Female": "Mujeres",
    "Age [years]": "Edad [años]",
    "Median deviation from age 18 [dB]":
        "Desviación mediana respecto a los 18 años [dB]",
    "Standard deviation at 4 kHz [dB]": "Desviación típica a 4 kHz [dB]",
    "$s_u$ (worse than the median)": "$s_u$ (peor que la mediana)",
    "$s_l$ (better than the median)": "$s_l$ (mejor que la mediana)",
    "$s_u - s_l$: the asymmetry": "$s_u - s_l$: la asimetría",
    "above 70 yr:\ninformative only\n(clause 4.1, f ≥ 3 kHz)":
        "por encima de 70 años:\nsolo informativo\n(capítulo 4.1, f ≥ 3 kHz)",
    "The frames the 40 dB rule drops (shaded), and the segment scores under them":
        "Las tramas que descarta la regla de 40 dB (sombreadas) y las "
        "puntuaciones por segmento debajo",
    "Clean reference": "Referencia limpia",
    "Segment score": "Puntuación por segmento",
    "dropout": "corte",
    "the 30 dB window: Di − 15 to Di + 15":
        "la ventana de 30 dB: de Di − 15 a Di + 15",
    "speech Ei'": "voz Ei'",
    "external noise Ni'": "ruido externo Ni'",
    "equivalent masking Zi": "enmascaramiento equivalente Zi",
    "equivalent disturbance Di": "perturbación equivalente Di",
    "Band audibility Ai": "Audibilidad de banda Ai",
    "Equivalent spectrum level [dB]": "Nivel de espectro equivalente [dB]",
    "The octave procedure carries no spread of masking":
        "El procedimiento por octavas no incluye la extensión del enmascaramiento",
    "Spectrum level of the low-frequency noise below 450 Hz [dB]":
        "Nivel de espectro del ruido de baja frecuencia por debajo de 450 Hz [dB]",
    "critical-band": "por bandas críticas",
    "equally-contributing": "de contribución igual",
    "one-third-octave": "por tercios de octava",
    "octave": "por octavas",
    "Reverberation: a low-pass corner that moves":
        "Reverberación: una frecuencia de corte que se desplaza",
    "Steady noise: the same curve, scaled down":
        "Ruido estacionario: la misma curva, escalada",
    "Modulation frequency F [Hz]": "Frecuencia de modulación F [Hz]",
    "Modulation transfer m (1 kHz band)":
        "Transferencia de modulación m (banda de 1 kHz)",
    "STIPA probes only these two F in this band":
        "STIPA solo sondea estas dos F en esta banda",
    "noise-free (T60 = 0.9 s)": "sin ruido (T60 = 0,9 s)",
    r"$\times\,1/(1 + 10^{-\mathrm{SNR}/10})$: flat in $F$":
        r"$\times\,1/(1 + 10^{-\mathrm{SNR}/10})$: constante en $F$",
    "One impulse response against the level it is played at":
        "Una misma respuesta al impulso frente al nivel de reproducción",
    "Overall speech level at the listener [dB SPL]":
        "Nivel global de voz en el oyente [dB SPL]",
    "with level= and ambient= (Tables A.2/A.3)":
        "con level= y ambient= (tablas A.2/A.3)",
    "reception threshold:\nthe speech is barely above\nthe room's own noise":
        "umbral de recepción:\nla voz apenas supera\nel ruido propio de la sala",
    "auditory masking:\nloud low bands mask\nthe high ones":
        "enmascaramiento auditivo:\nlas bandas graves intensas\nenmascaran las agudas",
    "the standard's fallback level:\n60 dB(A) at 1 m from the source":
        "el nivel por defecto de la norma:\n60 dB(A) a 1 m de la fuente",
    "ISO 9612 Annex D: the Annex C budget, term by term":
        "Anexo D de ISO 9612: el presupuesto del anexo C, término a término",
    "Contribution to $u^2$ [dB²]": "Contribución a $u^2$ [dB²]",
    "sampling  $(c_{1a}u_{1a})^2$": "muestreo  $(c_{1a}u_{1a})^2$",
    "duration  $(c_{1b}u_{1b})^2$": "duración  $(c_{1b}u_{1b})^2$",
    "instrument  $(c_{1a}u_2)^2$": "instrumento  $(c_{1a}u_2)^2$",
    "position  $(c_{1a}u_3)^2$": "posición  $(c_{1a}u_3)^2$",
    "planning/\nbreaks": "planificación/\ndescansos",
    "cutting/\ngrinding": "corte/\namolado",
    "whole day": "jornada completa",
    "1.5 h of the day,\n91 % of the variance":
        "1,5 h de la jornada,\nel 91 % de la varianza",
    "A 3 dB sample scatter: what more samples buy":
        "Con 3 dB de dispersión: lo que aportan más muestras",
    "Samples per task ($I$) or per group ($N$)":
        "Muestras por tarea ($I$) o por grupo ($N$)",
    "Expanded uncertainty $U$ [dB]": "Incertidumbre expandida $U$ [dB]",
    "task-based, $u_{1a}$ (Eq. C.6)": "por tareas, $u_{1a}$ (ec. C.6)",
    "job-based, $c_1u_1$ (Table C.4)": "por puestos, $c_1u_1$ (tabla C.4)",
    "floor with a personal exposimeter or class 2 meter":
        "suelo con exposímetro personal o sonómetro de clase 2",
    "floor with a class 1 meter": "suelo con sonómetro de clase 1",
    "ISO 1999 median NIPTS against level (40 years)":
        "NIPTS mediana de ISO 1999 frente al nivel (40 años)",
    "Median NIPTS $N_{50}$ [dB]": "NIPTS mediana $N_{50}$ [dB]",
    r"$L_{EX,8h}$ [dB(A)]": r"$L_{EX,8h}$ [dB(A)]",
    "each dot is that band's cut-off $L_0$\n(93, 89, 80, 77, 75, 77 dB)":
        "cada punto es el nivel de corte $L_0$ de esa banda\n"
        "(93, 89, 80, 77, 75, 77 dB)",
    "The same 3 dB is worth more the louder the job":
        "Los mismos 3 dB valen más cuanto más ruidoso es el puesto",
    "Median NIPTS at 4 kHz [dB]": "NIPTS mediana a 4 kHz [dB]",
    "ISO 1999 Formula (1): what the compression term removes":
        "Fórmula (1) de ISO 1999: lo que resta el término de compresión",
    "Decibels removed by $HN/120$": "Decibelios que resta $HN/120$",
    "$H + N$ = 40 dB": "$H + N$ = 40 dB",
    "Age component $H$ (HTLA) [dB]": "Componente de edad $H$ (HTLA) [dB]",
    "Noise component $N$ (NIPTS) [dB]":
        "Componente de ruido $N$ (NIPTS) [dB]",
    "the worked case: H = 20.2, N = 24.8\n45.0 dB sum → 40.8 dB HTLAN, "
    "4.2 dB removed":
        "el caso del ejemplo: H = 20,2, N = 24,8\n"
        "suma de 45,0 dB → 40,8 dB de HTLAN, 4,2 dB restados",
    "GUM uncertainty budget": "Presupuesto de incertidumbre (GUM)",
    "Contribution to combined uncertainty [dB]":
        "Contribución a la incertidumbre combinada [dB]",
    "Monte Carlo (Suppl 1)": "Monte Carlo (Supl. 1)",
    "GUM Gaussian": "Gaussiana GUM",
    "95 % coverage interval": "Intervalo de cobertura 95 %",
    "A-weighted level [dB]": "Nivel ponderado A [dB]",
    "Probability density": "Densidad de probabilidad",
    "Reading": "Lectura",
    "Calibration": "Calibración",
    "Instrument": "Instrumento",
    "Position (Type A)": "Posición (Tipo A)",
    "Sound Intensity with a p-p Probe (IEC 61043)":
        "Intensidad acústica con sonda p-p (IEC 61043)",
    "Plane wave: Lp \u2248 LI": "Onda plana: Lp \u2248 LI",
    "Standing wave: reactive field": "Onda estacionaria: campo reactivo",
    "Pressure level Lp": "Nivel de presi\u00f3n Lp",
    "Intensity level LI": "Nivel de intensidad LI",
    "Schroeder Integration and Reverberation Time (ISO 3382)":
        "Integraci\u00f3n de Schroeder y tiempo de reverberaci\u00f3n (ISO 3382)",
    "Raw squared IR level": "Nivel de la RI al cuadrado",
    "Schroeder decay curve": "Curva de ca\u00edda de Schroeder",
    "T20 fit (\u22125 to \u221225 dB)": "Ajuste T20 (\u22125 a \u221225 dB)",
    "T30 fit (\u22125 to \u221235 dB)": "Ajuste T30 (\u22125 a \u221235 dB)",
    "EDT fit (0 to \u221210 dB)": "Ajuste EDT (0 a \u221210 dB)",
    "EDT slope": "Pendiente EDT",
    "ISO 717-1 Weighted Sound Reduction Index (Annex C example)":
        "\u00cdndice ponderado de reducci\u00f3n ac\u00fastica (ISO 717-1, ejemplo del Anexo C)",
    "Apparent sound reduction index R' [dB]":
        "\u00cdndice de reducci\u00f3n ac\u00fastica aparente R' [dB]",
    "Measured R' (third octave)": "R' medido (tercios de octava)",
    "Shifted reference curve (ISO 717-1)":
        "Curva de referencia desplazada (ISO 717-1)",
    "Unfavourable deviations": "Desviaciones desfavorables",
    "Sharpness Weighting g(z) (DIN 45692)":
        "Ponderación de agudeza g(z) (DIN 45692)",
    "Weighting g(z)": "Ponderación g(z)",
    "DIN 45692 g(z)": "g(z) DIN 45692",
    "von Bismarck (Annex B)": "von Bismarck (Anexo B)",
    "DIN knee\n15.8 Bark": "Codo DIN\n15,8 Bark",
    "Bismarck knee\n15 Bark": "Codo Bismarck\n15 Bark",
    "ISO 717-2 Weighted Normalized Impact Sound Level (Annex C example)":
        "Nivel de ruido de impactos normalizado y ponderado "
        "(ISO 717-2, ejemplo del Anexo C)",
    "Normalized impact sound pressure level Ln [dB]":
        "Nivel de presión acústica de impactos normalizado Ln [dB]",
    "Measured Ln (third octave)": "Ln medido (tercios de octava)",
    "Shifted reference curve (ISO 717-2)":
        "Curva de referencia desplazada (ISO 717-2)",
    "Unfavourable deviations (measured above reference)":
        "Desviaciones desfavorables (medido por encima de la referencia)",
    "Open-Plan Spatial Decay of Speech (ISO 3382-3)":
        "Decaimiento espacial del habla en oficina abierta (ISO 3382-3)",
    "Distance from the talker r [m]": "Distancia al hablante r [m]",
    "A-weighted SPL [dB]": "SPL ponderado A [dB]",
    "Measured Lp,A,S": "Lp,A,S medido",
    "STI vs distance": "STI vs distancia",
    # --- ERB_N / Cam auditory-filter scale (Glasberg & Moore 1990) ---
    "Auditory-Filter Bandwidth and the Cam Scale (Glasberg & Moore, 1990)":
        "Ancho de banda del filtro auditivo y escala Cam (Glasberg y Moore, 1990)",
    "Centre frequency [Hz]": "Frecuencia central [Hz]",
    "Equivalent rectangular bandwidth ERB$_N$ [Hz]":
        "Ancho de banda rectangular equivalente ERB$_N$ [Hz]",
    "ERB$_N$ (Glasberg & Moore, 1990)": "ERB$_N$ (Glasberg y Moore, 1990)",
    "One-third octave (23 % of f)": "Tercio de octava (23 % de f)",
    "ERB$_N$ number [Cam]": "Número ERB$_N$ [Cam]",
    "1 kHz = 15.59 Cam": "1 kHz = 15,59 Cam",
    # --- Advanced psychoacoustics figures (plan-17 block A) ---
    "Loudness Models Compared (1 kHz tone)":
        "Modelos de sonoridad comparados (tono de 1 kHz)",
    "Sound pressure level [dB SPL]": "Nivel de presión acústica [dB SPL]",
    "Total loudness N [sone]": "Sonoridad total N [sonios]",
    "Sottek ECMA-418-2": "Sottek ECMA-418-2",
    "Anchor: 1 kHz / 40 dB = 1 sone":
        "Anclaje: 1 kHz / 40 dB = 1 sonio",
    "Models diverge at high levels":
        "Los modelos divergen a niveles altos",
    "Sottek Specific Loudness (ECMA-418-2)":
        "Sonoridad específica de Sottek (ECMA-418-2)",
    "Specific loudness N' [sone_HMS/Bark]":
        "Sonoridad específica N' [sonios_HMS/Bark]",
    "Peak specific loudness": "Sonoridad específica máxima",
    "ECMA-418-2 Tonality T(t)": "Tonalidad T(t) (ECMA-418-2)",
    "Tonality T [tu_HMS]": "Tonalidad T [tu_HMS]",
    "ECMA-418-2 Roughness vs Modulation Frequency":
        "Aspereza vs frecuencia de modulación (ECMA-418-2)",
    "Modulation frequency f_mod [Hz]":
        "Frecuencia de modulación f_mod [Hz]",
    "Roughness R [asper]": "Aspereza R [asper]",
    "1 kHz carrier, 100 % AM": "Portadora de 1 kHz, AM del 100 %",
    "Sound Quality Metrics (ECMA-418-2 Sottek Hearing Model)":
        "Métricas de calidad sonora (modelo auditivo de Sottek, ECMA-418-2)",
    "Slow vs Fast Modulation Perception (ECMA-418-2 Sottek Hearing Model)":
        "Percepción de modulación lenta vs rápida (modelo auditivo de "
        "Sottek, ECMA-418-2)",
    "Fluctuation strength F (Clause 9, slow modulation)":
        "Intensidad de fluctuación F (cláusula 9, modulación lenta)",
    "Roughness R (Clause 7, fast modulation)":
        "Aspereza R (cláusula 7, modulación rápida)",
    "1 kHz carrier, 100 % AM, overall 60 dB SPL":
        "Portadora de 1 kHz, AM del 100 %, 60 dB SPL globales",
    "Time-Varying Loudness (ISO 532-3)":
        "Sonoridad variable en el tiempo (ISO 532-3)",
    "Loudness [sone]": "Sonoridad [sonios]",
    "1 kHz burst, 200 ms": "Ráfaga de 1 kHz, 200 ms",
    "Fast attack / release": "Ataque / relajación rápidos",
    "Slow integration": "Integración lenta",
    # Fluctuation strength + psychoacoustic annoyance (Fastl & Zwicker; Osses 2016)
    "Fluctuation Strength — 4 Hz Band-Pass Characteristic":
        "Intensidad de fluctuación — característica de paso de banda a 4 Hz",
    "Fluctuation strength F [vacil]": "Intensidad de fluctuación F [vacil]",
    "AM-tone F, signal model [vacil]":
        "F de tono AM, modelo de señal [vacil]",
    "4 Hz reference": "referencia 4 Hz",
    "Psychoacoustic Annoyance vs Loudness (Fastl & Zwicker)":
        "Molestia psicoacústica vs sonoridad (Fastl y Zwicker)",
    "Percentile loudness N5 [sone]": "Sonoridad percentil N5 [sonios]",
    "Psychoacoustic annoyance PA": "Molestia psicoacústica PA",
    "Baseline: S = 1.75 acum, F = R = 0":
        "Base: S = 1,75 acum, F = R = 0",
    "Sharp: S = 3.5 acum": "Aguda: S = 3,5 acum",
    "Rough + fluctuating: F = 1.2 vacil, R = 0.7 asper":
        "Áspera + fluctuante: F = 1,2 vacil, R = 0,7 asper",
    # Electroacoustics (IEC 60268-3 distortion; Bendat & Piersol response)
    "Harmonic Distortion of a Single-Tone Test (IEC 60268-3)":
        "Distorsión armónica de un ensayo con tono único (IEC 60268-3)",
    "Magnitude spectrum": "Espectro de magnitud",
    "Harmonics n·f₁": "Armónicos n·f₁",
    "Level re fundamental [dB]": "Nivel respecto al fundamental [dB]",
    "Frequency Response and Coherence (Bendat & Piersol)":
        "Respuesta en frecuencia y coherencia (Bendat y Piersol)",
    "True |H|": "|H| verdadero",
    "Estimated |H| (H1)": "|H| estimado (H1)",
    # Swept-sine harmonic separation (Farina 2000 / Novak et al. 2015)
    "Swept-Sine Harmonic Distortion by Order (Farina / Novak)":
        "Distorsión armónica por orden con barrido sinusoidal "
        "(Farina / Novak)",
    "Excitation frequency [Hz]": "Frecuencia de excitación [Hz]",
    "Distortion re fundamental [%]": "Distorsión respecto al fundamental [%]",
    "Total THD(f)": "THD(f) total",
    "2nd harmonic d₂(f)": "2º armónico d₂(f)",
    "3rd harmonic d₃(f)": "3er armónico d₃(f)",
    "Chebyshev asymptote (a₂/2)/H₁": "Asíntota de Chebyshev (a₂/2)/H₁",
    "Chebyshev asymptote (a₃/4)/H₁": "Asíntota de Chebyshev (a₃/4)/H₁",
    "one sweep separates every distortion order;\n"
    "each rolls off where its product n·f crosses the 3 kHz corner":
        "un solo barrido separa cada orden de distorsión;\n"
        "cada uno cae donde su producto n·f cruza el corte de 3 kHz",
    # Calibrated spectral analysis (Bendat & Piersol PSD/CSD core)
    "Calibrated Spectral Density of Pink Noise (Bendat & Piersol)":
        "Densidad espectral calibrada de ruido rosa (Bendat y Piersol)",
    "95 % chi-square confidence interval":
        "Intervalo de confianza chi-cuadrado del 95 %",
    "Welch PSD estimate": "Estimación de la PSD de Welch",
    "1/3-octave smoothed": "Suavizado en 1/3 de octava",
    "Exact -3.01 dB/octave power law":
        "Ley de potencias exacta de -3,01 dB/octava",
    # Thomson multitaper spectral density (Percival & Walden 1993)
    "Thomson Multitaper Density of a Short Record (Percival & Walden)":
        "Densidad multitaper de Thomson de un registro corto (Percival y Walden)",
    "Single Slepian taper ($K$ = 1, $\\nu$ = 2)":
        "Un solo taper de Slepian ($K$ = 1, $\\nu$ = 2)",
    "Multitaper estimate ($K$ = 7, adaptive)":
        "Estimación multitaper ($K$ = 7, adaptativa)",
    # Time-frequency analysis (Bendat & Piersol spectrogram + zoom FFT)
    "Calibrated Spectrogram in dB SPL (Bendat & Piersol)":
        "Espectrograma calibrado en dB SPL (Bendat y Piersol)",
    "a siren, an impact and a pink-noise floor:\n"
    "every cell reads an absolute level":
        "una sirena, un impacto y un fondo de ruido rosa:\n"
        "cada celda lee un nivel absoluto",
    "Zoom FFT Resolves Tones One Coarse Bin Apart (Bendat & Piersol)":
        "La FFT con zoom resuelve tonos a menos de un bin grueso "
        "(Bendat y Piersol)",
    "1024-point FFT (8 Hz bins)": "FFT de 1024 puntos (bins de 8 Hz)",
    "Zoom FFT of the same record": "FFT con zoom del mismo registro",
    "997 and 1000 Hz, 3 Hz apart:\n"
    "one lump on the 8 Hz grid,\n"
    "two exact lines on the zoom grid":
        "997 y 1000 Hz, separados 3 Hz:\n"
        "un solo bulto en la malla de 8 Hz,\n"
        "dos líneas exactas en la malla del zoom",
    # Cepstral analysis (echo detection) and envelope spectrum
    "Echo Detection on the Power Cepstrum (Quefrency Analysis)":
        "Detección de ecos en el cepstro de potencia (análisis de quefrencia)",
    "Power cepstrum": "Cepstro de potencia",
    "Searched band": "Banda de búsqueda",
    "True echo delay (8 ms)": "Retardo verdadero del eco (8 ms)",
    "Detected peak (height = reflection a)":
        "Pico detectado (altura = reflexión a)",
    "Quefrency [ms]": "Quefrencia [ms]",
    "Cepstrum": "Cepstro",
    "spectral ripple of period 1/(8 ms) collapses to one\n"
    "spike at 8 ms whose height reads the reflection":
        "el rizado espectral de período 1/(8 ms) colapsa en un\n"
        "pico a 8 ms cuya altura mide la reflexión",
    "Envelope Spectrum of an AM Tone (Bendat & Piersol 13.3)":
        "Espectro de la envolvente de un tono AM (Bendat y Piersol 13.3)",
    "Envelope spectrum": "Espectro de la envolvente",
    "Modulation frequency (25 Hz)": "Frecuencia de modulación (25 Hz)",
    r"Exact line amplitude $A_0 m$ = 0.4":
        r"Amplitud exacta de la línea $A_0 m$ = 0,4",
    "Modulation amplitude": "Amplitud de modulación",
    "the carrier is at 1 kHz; its amplitude modulation\n"
    "appears as one line at exactly $f_m$":
        "la portadora está en 1 kHz; su modulación de amplitud\n"
        "aparece como una línea exactamente en $f_m$",
    # Time synchronous averaging (McFadden 1987)
    "Periodic Waveform Extracted from Noise":
        "Forma de onda periódica extraída del ruido",
    "One noisy period": "Un período ruidoso",
    "Average of N = 40 periods": "Promedio de N = 40 períodos",
    "True periodic waveform": "Forma de onda periódica verdadera",
    "averaging N periods lowers the asynchronous\n"
    "noise by $\\sqrt{N}$ in amplitude":
        "promediar N períodos reduce el ruido asíncrono\n"
        "en $\\sqrt{N}$ en amplitud",
    "Rejecting a Tone by Choosing N (McFadden 1987)":
        "Rechazo de un tono eligiendo N (McFadden 1987)",
    "N = 32 (power of two)": "N = 32 (potencia de dos)",
    "N = 20 (node on 32.05)": "N = 20 (nodo en 32,05)",
    "Interfering tone (32.05)": "Tono interferente (32,05)",
    "Frequency [orders]": "Frecuencia [órdenes]",
    "Comb filter magnitude": "Magnitud del filtro peine",
    "N = 20 puts a node on 32.05 orders and removes\n"
    "it; the power-of-two N = 32 lets it through":
        "N = 20 sitúa un nodo en 32,05 órdenes y lo\n"
        "elimina; la potencia de dos N = 32 lo deja pasar",
    # Multiple-input coherence (Bendat & Piersol Ch. 7)
    "Multiple-Input Coherence: Which Source Dominates Each Band "
    "(Bendat & Piersol Ch. 7)":
        "Coherencia de múltiples entradas: qué fuente domina cada banda "
        "(Bendat y Piersol cap. 7)",
    "Measured output": "Salida medida",
    "Input 1 contribution": "Contribución de la entrada 1",
    "Input 2 contribution": "Contribución de la entrada 2",
    "Residual noise": "Ruido residual",
    "Coherent output [dB re 1/Hz]": "Salida coherente [dB re 1/Hz]",
    r"Input 2 ordinary $\gamma^2_{2y}$ (inflated by x1)":
        r"Entrada 2 ordinaria $\gamma^2_{2y}$ (inflada por x1)",
    r"Input 2 partial $\gamma^2_{2y\cdot 1}$ (x1 removed)":
        r"Entrada 2 parcial $\gamma^2_{2y\cdot 1}$ (x1 eliminada)",
    r"Multiple $\gamma^2_{y:x}$": r"Múltiple $\gamma^2_{y:x}$",
    "Coherence": "Coherencia",
    "conditioning removes the shared x1 component:\n"
    "the low-band ordinary coherence of x2 collapses":
        "el condicionamiento elimina la componente x1 compartida:\n"
        "la coherencia ordinaria de x2 en la banda baja se desploma",
    # Data qualification: trend and stationarity tests, Rice crossing statistics
    "Nonparametric Trend Test by Reverse Arrangements (B&P 4.5.2)":
        "Test de tendencia no paramétrico por inversiones de orden (B&P 4.5.2)",
    "B&P Example 4.4: A = 86, accepted (no trend)":
        "Ejemplo 4.4 de B&P: A = 86, aceptado (sin tendencia)",
    "Added rising drift: A = 38, rejected (trend)":
        "Deriva ascendente añadida: A = 38, rechazado (tendencia)",
    "Sample index": "Índice de muestra",
    "Sequence value": "Valor de la secuencia",
    "20 observations; the count A of pairs i < j with x[i] > x[j]\n"
    "must fall in (64, 125] at the 5 % level (Table A.6). A rising\n"
    "trend depresses A below the acceptance region":
        "20 observaciones; el conteo A de pares i < j con x[i] > x[j]\n"
        "debe caer en (64, 125] al nivel del 5 % (Tabla A.6). Una deriva\n"
        "ascendente reduce A por debajo de la región de aceptación",
    "Stationarity Test by Reverse Arrangements (B&P 10.3.1.1)":
        "Test de estacionariedad por inversiones de orden (B&P 10.3.1.1)",
    "Steady noise: A = 91, accepted (stationary)":
        "Ruido estable: A = 91, aceptado (estacionario)",
    "+20 % gain ramp: A = 7, rejected (nonstationary)":
        "Rampa de ganancia del +20 %: A = 7, rechazado (no estacionario)",
    "Segment index": "Índice de segmento",
    "Segment mean square": "Media cuadrática por segmento",
    "20 segment mean squares; the count A of pairs i < j with\n"
    "x[i] > x[j] must fall in (64, 125] at the 5 % level (Table A.6)":
        "20 medias cuadráticas por segmento; el conteo A de pares i < j con\n"
        "x[i] > x[j] debe caer en (64, 125] al nivel del 5 % (Tabla A.6)",
    "Level-Crossing Rates of Bandlimited Gaussian Noise (Rice)":
        "Tasas de cruce por nivel de ruido gaussiano de banda limitada "
        "(Rice)",
    r"Rice: $N_0\,\exp(-a^2/2\sigma_x^2)$ (Eq. 5.196)":
        r"Rice: $N_0\,\exp(-a^2/2\sigma_x^2)$ (Ec. 5.196)",
    "Measured crossing rate": "Tasa de cruces medida",
    "Level a [signal units]": "Nivel a [unidades de la señal]",
    "Crossings per second [1/s]": "Cruces por segundo [1/s]",
    "800-1200 Hz Gaussian band: 2014 zero crossings/s, an\n"
    r"apparent frequency $N_0/2 \approx$ 1007 Hz (B&P Example 5.13)":
        "banda gaussiana de 800-1200 Hz: 2014 cruces por cero/s, una\n"
        r"frecuencia aparente $N_0/2 \approx$ 1007 Hz (B&P Ejemplo 5.13)",
    "Peak-Height Distribution and the Irregularity Factor (Rice)":
        "Distribución de alturas de pico y factor de irregularidad (Rice)",
    "Rayleigh limit (r = 1, narrowband)":
        "Límite de Rayleigh (r = 1, banda estrecha)",
    "Gaussian limit (r = 0, wideband)":
        "Límite gaussiano (r = 0, banda ancha)",
    "Rice mixture at r = 0.746 (Eq. 5.223)":
        "Mezcla de Rice con r = 0,746 (Ec. 5.223)",
    "Empirical peak exceedance (0-2 kHz noise)":
        "Excedencia empírica de picos (ruido de 0-2 kHz)",
    r"Standardized peak height $z = a/\sigma_x$":
        r"Altura de pico estandarizada $z = a/\sigma_x$",
    "Prob[peak > z]": "Prob[pico > z]",
    r"low-pass noise: $r = N_0/2M = \sqrt{5}/3$; negative maxima exist,"
    "\nso the peak law sits between Gaussian and Rayleigh (B&P 5.5.4)":
        r"ruido paso bajo: $r = N_0/2M = \sqrt{5}/3$; existen máximos"
        " negativos,\nasí que la ley de picos queda entre la gaussiana y la"
        " de Rayleigh (B&P 5.5.4)",
    # Correlation and time-delay estimation (B&P / Knapp & Carter GCC)
    "Time-Delay Estimation: GCC-PHAT vs Direct Correlation (Knapp & Carter)":
        "Estimación del retardo: GCC-PHAT frente a correlación directa "
        "(Knapp y Carter)",
    "Direct cross-correlation": "Correlación cruzada directa",
    "True delay (20 samples)": "Retardo verdadero (20 muestras)",
    "Lag [ms]": "Retardo [ms]",
    "Normalized correlation": "Correlación normalizada",
    "colored signal: the plain correlator smears the peak,\n"
    "PHAT prewhitens the cross-spectrum and restores it":
        "señal coloreada: el correlador simple ensancha el pico,\n"
        "PHAT preblanquea el espectro cruzado y lo recupera",
    # Objective intelligibility: STOI vs ESTOI (Taal 2011 / Jensen & Taal 2016)
    "Short-Time Objective Intelligibility: STOI vs ESTOI":
        "Inteligibilidad objetiva de corto plazo: STOI frente a ESTOI",
    "STOI (Taal et al. 2011)": "STOI (Taal et al. 2011)",
    "ESTOI (Jensen & Taal 2016)": "ESTOI (Jensen y Taal 2016)",
    "Stationary masker": "Enmascarador estacionario",
    "Modulated (5 Hz gated) masker":
        "Enmascarador modulado (con puerta a 5 Hz)",
    "SNR [dB]": "SNR [dB]",
    "Intelligibility index": "Índice de inteligibilidad",
    "ESTOI rates the modulated masker higher: it credits the\n"
    "speech glimpsed in the quiet gaps. STOI barely separates them.":
        "ESTOI valora más alto el enmascarador modulado: acredita el\n"
        "habla vislumbrada en los silencios. STOI apenas los separa.",
    # Room acoustics: image-source reflectogram (Kuttruff 4.1 / Vorlander 11.4)
    "Image-Source Room Impulse Response: a 7x5x3 m room (order <= 10)":
        "Respuesta al impulso por fuentes imagen: sala de 7x5x3 m "
        "(orden <= 10)",
    "Reflection order": "Orden de reflexión",
    "Arrival time [ms]": "Tiempo de llegada [ms]",
    "Reflection level re direct [dB]": "Nivel de reflexión rel. directo [dB]",
    r"$1/r$ spreading envelope": r"Envolvente de divergencia $1/r$",
    "Reflections (image sources)": "Reflexiones (fuentes imagen)",
    "Direct sound (order 0)": "Sonido directo (orden 0)",
    "each reflection is a mirror image of the source;\n"
    "amplitude = product of wall reflection factors / (4 pi r)":
        "cada reflexión es una imagen especular de la fuente;\n"
        "amplitud = producto de factores de reflexión de pared / (4 pi r)",
    # Underwater acoustics (ISO 17208 ship radiated noise; ISO 18406 pile driving)
    "Ship Equivalent Monopole Source Level (ISO 17208-2)":
        "Nivel de fuente monopolar equivalente de buque (ISO 17208-2)",
    "Source level Ls": "Nivel de fuente Ls",
    "Radiated noise level": "Nivel de ruido radiado",
    "Surface correction ΔL [dB]": "Corrección de superficie ΔL [dB]",
    "Surface correction ΔL": "Corrección de superficie ΔL",
    "Level [dB re 1 µPa·m]": "Nivel [dB re 1 µPa·m]",
    "Percussive Pile-Driving Strike (ISO 18406)":
        "Golpe de hincado de pilotes por percusión (ISO 18406)",
    "Time [ms]": "Tiempo [ms]",
    "Pressure [Pa]": "Presión [Pa]",
    "FDTD probe pressure": "Presión en las sondas FDTD",
    "Number of strikes N": "Número de golpes N",
    "Cumulative SEL [dB re 1 µPa²·s]": "SEL acumulado [dB re 1 µPa²·s]",
    "ICAO Aircraft Flyover — Effective Perceived Noise Level (Annex 16)":
        "Sobrevuelo de aeronave ICAO — Nivel efectivo de ruido percibido (Anexo 16)",
    "Level [PNdB]": "Nivel [PNdB]",
    "10 dB-down window": "Ventana 10 dB por debajo",
    "Wind-Turbine Tonal Audibility (IEC 61400-11)":
        "Audibilidad tonal de aerogenerador (IEC 61400-11)",
    "Narrowband spectrum": "Espectro de banda estrecha",
    "Critical band": "Banda crítica",
    # Underwater propagation (plan-22 P1): transmission loss, sound speed, sonar.
    "Underwater Transmission Loss (Francois–Garrison)":
        "Pérdida por transmisión submarina (Francois–Garrison)",
    "Range [m]": "Distancia [m]",
    "Transmission loss [dB]": "Pérdida por transmisión [dB]",
    "Total transmission loss": "Pérdida por transmisión total",
    "Geometrical spreading": "Ensanchamiento geométrico",
    "Volume absorption": "Absorción de volumen",
    "Sea-Water Sound-Speed Profile (UNESCO)":
        "Perfil de velocidad del sonido en agua de mar (UNESCO)",
    "Sound speed [m/s]": "Velocidad del sonido [m/s]",
    "Depth [m]": "Profundidad [m]",
    "UNESCO sound speed": "Velocidad del sonido UNESCO",
    "Sound-channel axis": "Eje del canal sonoro",
    # Weston shallow-water propagation regimes (Ainslie section 9.1.1.2).
    "Weston Shallow-Water Propagation Regimes (Ainslie §9.1.1.2)":
        "Regímenes de propagación de Weston en aguas someras (Ainslie §9.1.1.2)",
    "Propagation loss [dB re 1 m²]": "Pérdida de propagación [dB re 1 m²]",
    "Spherical, 20 log10 r": "Esférica, 20 log10 r",
    "Cylindrical, 10 log10 r": "Cilíndrica, 10 log10 r",
    "Mode stripping, 15 log10 r": "Descamado de modos, 15 log10 r",
    "Single mode": "Modo único",
    "Composite propagation loss": "Pérdida de propagación compuesta",
    # Marine-mammal auditory weighting (NMFS 2024 v3.0).
    "Marine-Mammal Auditory Weighting (NMFS 2024, v3.0)":
        "Ponderación auditiva de mamíferos marinos (NMFS 2024, v3.0)",
    "Weighting amplitude W(f) [dB]": "Amplitud de ponderación W(f) [dB]",
    "Passive Sonar Equation": "Ecuación del sonar pasivo",
    "Signal excess [dB]": "Exceso de señal [dB]",
    "Signal excess": "Exceso de señal",
    "Detection limit (SE = 0)": "Límite de detección (SE = 0)",
    "Figure of merit": "Figura de mérito",
    # Underwater propagation (plan-22 P1 PR-2): seabed, ambient noise, traffic.
    "Seabed Reflection Loss (Rayleigh)":
        "Pérdida por reflexión en el fondo (Rayleigh)",
    "Grazing angle [°]": "Ángulo rasante [°]",
    "Bottom loss [dB]": "Pérdida por reflexión [dB]",
    "Bottom loss (sand)": "Pérdida por reflexión (arena)",
    "Seabed Reflection Coefficient (Rayleigh)":
        "Coeficiente de reflexión del fondo (Rayleigh)",
    "Reflection coefficient magnitude |R|":
        "Módulo del coeficiente de reflexión |R|",
    "Reflection coefficient magnitude |R| (sand)":
        "Módulo del coeficiente de reflexión |R| (arena)",
    "Water ρ = 1000, c = 1500\nSand ρ = 1900, c = 1650":
        "Agua ρ = 1000, c = 1500\nArena ρ = 1900, c = 1650",
    "Ocean Ambient Noise (Wenz)": "Ruido ambiental oceánico (Wenz)",
    "Spectrum level [dB re 1 µPa²/Hz]": "Nivel espectral [dB re 1 µPa²/Hz]",
    "Ship Traffic Source Level (JOMOPANS-ECHO)":
        "Nivel de fuente del tráfico marítimo (JOMOPANS-ECHO)",
    "Ray trace (Munk profile)": "Trazado de rayos (perfil de Munk)",
    "Source": "Fuente",
    "Range [km]": "Distancia [km]",
    "Parabolic equation (50 Hz)": "Ecuación parabólica (50 Hz)",
    "Modes vs PE (50 Hz, z = 120 m)": "Modos vs PE (50 Hz, z = 120 m)",
    "Normal modes": "Modos normales",
    "Parabolic equation": "Ecuación parabólica",
    "Source spectral density [dB re 1 µPa²/Hz at 1 m]":
        "Densidad espectral de fuente [dB re 1 µPa²/Hz a 1 m]",
    "Wind": "Viento",
    "Thermal": "Térmico",
    # Building acoustics (EN 12354-1 flanking prediction, ISO 12999-1 uncertainty)
    "EN 12354-1 Flanking Transmission (Annex H.3 example)":
        "Transmisión por flancos EN 12354-1 (ejemplo del Anexo H.3)",
    "Share of transmitted energy [%]": "Cuota de energía transmitida [%]",
    "ISO 12354-1 Detailed Model: Dominant Path per Band (Annex L)":
        "Modelo detallado ISO 12354-1: camino dominante por banda (Anexo L)",
    "other paths": "otros caminos",
    "R' (apparent)": "R' (aparente)",
    "Transmission path": "Camino de transmisión",
    "Dd — direct": "Dd — directo",
    "Ff — flanking–flanking": "Ff — flanco–flanco",
    "Fd — flanking–separating": "Fd — flanco–separador",
    "Df — separating–flanking": "Df — separador–flanco",
    "dominant path": "camino dominante",
    "ISO 12999-1 Measurement Uncertainty (situation B, airborne)":
        "Incertidumbre de medición ISO 12999-1 (situación B, aéreo)",
    "Measured R'": "R' medido",
    "Standard uncertainty ±u": "Incertidumbre típica ±u",
    "Expanded uncertainty ±U (95 %)": "Incertidumbre expandida ±U (95 %)",
    "R'w ± U (single number)": "R'w ± U (valor único)",
    # Outdoor propagation & occupational exposure (PR-C).
    "ISO 9613-1 Atmospheric Absorption α(f)":
        "Absorción atmosférica α(f) (ISO 9613-1)",
    "Attenuation coefficient α [dB/km]":
        "Coeficiente de atenuación α [dB/km]",
    "ISO 9613-1 atmospheric attenuation":
        "Atenuación atmosférica ISO 9613-1",
    r"Attenuation coefficient $\alpha$ [dB/km]":
        r"Coeficiente de atenuación $\alpha$ [dB/km]",
    "ISO 9613-2 Attenuation Breakdown (with a 4 m barrier)":
        "Desglose de la atenuación (ISO 9613-2, con barrera de 4 m)",
    "Octave-band centre frequency [Hz]":
        "Frecuencia central de banda de octava [Hz]",
    "1/3-octave band centre frequency [Hz]":
        "Frecuencia central de banda de tercio de octava [Hz]",
    "CNOSSOS-EU Railway Source Line Power (96 coaches/h at 160 km/h)":
        "Potencia de la línea fuente ferroviaria CNOSSOS-EU "
        "(96 coches/h a 160 km/h)",
    "Line power L'W,eq,line [dB re 1 pW/m]":
        "Potencia de la línea L'W,eq,line [dB re 1 pW/m]",
    "Both source heights": "Ambas alturas de fuente",
    "Source A, 0,5 m (rolling, impact, traction)":
        "Fuente A, 0,5 m (rodadura, impacto, tracción)",
    "Source B, 4,0 m (traction, aerodynamic)":
        "Fuente B, 4,0 m (tracción, aerodinámico)",
    "CNOSSOS-EU Total Effective Roughness against Speed "
    "(f = v/λ, λ in the wavelength domain)":
        "Rugosidad efectiva total CNOSSOS-EU frente a la velocidad "
        "(f = v/λ, λ en el dominio de longitud de onda)",
    "Total effective roughness LR,TOT [dB re 1 μm]":
        "Rugosidad efectiva total LR,TOT [dB re 1 μm]",
    "Attenuation A [dB]": "Atenuación A [dB]",
    "CNOSSOS-EU Road Source Line Power (urban arterial, 50 km/h)":
        "Potencia de la línea fuente viaria CNOSSOS-EU (vía urbana, 50 km/h)",
    "Total source line": "Línea fuente total",
    "Light vehicles (1)": "Vehículos ligeros (1)",
    "Medium heavy vehicles (2)": "Vehículos pesados medios (2)",
    "Heavy vehicles (3)": "Vehículos pesados (3)",
    "Motorcycles (4b)": "Motocicletas (4b)",
    "Light vehicles (1) — total": "Vehículos ligeros (1) — total",
    "Light vehicles (1) — rolling": "Vehículos ligeros (1) — rodadura",
    "Light vehicles (1) — propulsion": "Vehículos ligeros (1) — propulsión",
    "Heavy vehicles (3) — total": "Vehículos pesados (3) — total",
    "Heavy vehicles (3) — rolling": "Vehículos pesados (3) — rodadura",
    "Heavy vehicles (3) — propulsion": "Vehículos pesados (3) — propulsión",
    "CNOSSOS-EU Single-Vehicle Sound Power against Speed (reference conditions)":
        "Potencia acústica de un vehículo frente a la velocidad "
        "(CNOSSOS-EU, condiciones de referencia)",
    "Speed v [km/h]": "Velocidad v [km/h]",
    "A-weighted sound power LW,A [dB(A) re 1 pW]":
        "Potencia acústica ponderada A LW,A [dB(A) re 1 pW]",
    "Adiv — divergence": "Adiv — divergencia",
    "Aatm — atmospheric": "Aatm — atmosférica",
    "Agr — ground": "Agr — suelo",
    "Abar — barrier": "Abar — barrera",
    "A — total": "A — total",
    "Spherical-Wave Ground Effect (Weyl-Van der Pol)":
        "Efecto suelo de onda esférica (Weyl-Van der Pol)",
    "Level re free field [dB]": "Nivel respecto al campo libre [dB]",
    "Fresh snow (10 kPa·s·m⁻²)": "Nieve reciente (10 kPa·s·m⁻²)",
    "Forest floor (50 kPa·s·m⁻²)": "Suelo forestal (50 kPa·s·m⁻²)",
    "Grassland (200 kPa·s·m⁻²)": "Pradera (200 kPa·s·m⁻²)",
    "Asphalt (20 000 kPa·s·m⁻²)": "Asfalto (20 000 kPa·s·m⁻²)",
    "Hard-ground limit (+6 dB)": "Límite de suelo rígido (+6 dB)",
    "ISO 9612 Task-Based Exposure (Annex D)":
        "Exposición por tareas (ISO 9612, Anexo D)",
    "LEX,8h contribution [dB]": "Contribución a LEX,8h [dB]",
    "Measurement task": "Tarea de medición",
    "planning/breaks": "planificación/pausas",
    "welding": "soldadura",
    "cutting/grinding": "corte/amolado",
    "Daily LEX,8h": "LEX,8h diario",
    "LEX,8h + U (one-sided 95 %)": "LEX,8h + U (unilateral 95 %)",
    # Materials: absorption rating, airflow resistance, impedance tube
    "Shifted reference curve (ISO 11654)":
        "Curva de referencia desplazada (ISO 11654)",
    "Practical absorption alpha_p": "Absorción práctica alpha_p",
    "ISO 11654 Weighted Sound Absorption Coefficient (Annex A.2 example)":
        "Coeficiente de absorción acústica ponderado ISO 11654 (ejemplo del Anexo A.2)",
    "Sound absorption coefficient": "Coeficiente de absorción acústica",
    "Through-origin quadratic fit  dp = a u + b u^2":
        "Ajuste cuadrático por el origen  dp = a u + b u^2",
    "Measured pressure drop": "Caída de presión medida",
    "evaluation at 0.5 mm/s": "evaluación a 0,5 mm/s",
    "ISO 9053-1 Static-Method Airflow Resistance":
        "Resistencia al flujo de aire por el método estático (ISO 9053-1)",
    "Linear airflow velocity u [mm/s]": "Velocidad lineal del aire u [mm/s]",
    "Pressure drop dp [Pa]": "Caída de presión dp [Pa]",
    "Absorption coefficient alpha = 1 - |r|^2":
        "Coeficiente de absorción alpha = 1 - |r|^2",
    "Standing-wave level difference L_max - L_min [dB]":
        "Diferencia de nivel de onda estacionaria L_max - L_min [dB]",
    "Sound absorption coefficient alpha": "Coeficiente de absorción acústica alpha",
    "Perfect Absorption by Critical Coupling (Slow-Sound Panel)":
        "Absorción perfecta por acoplo crítico (panel de sonido lento)",
    "Critically coupled (perfect)": "Acoplo crítico (perfecto)",
    "Narrow slit (over-damped)": "Ranura estrecha (sobreamortiguada)",
    "Wide slit (under-damped)": "Ranura ancha (subamortiguada)",
    "design 300 Hz": "diseño 300 Hz",
    "Reflection factor magnitude |r|": "Módulo del factor de reflexión |r|",
    "ISO 10534-1 Standing-Wave-Ratio Method":
        "Método de la razón de onda estacionaria (ISO 10534-1)",
    # --- Tier-1 animation labels ---
    "tone burst": "ráfaga de tono",
    "Fast (125 ms)": "Rápida (125 ms)",
    "Slow (1000 ms)": "Lenta (1000 ms)",
    "Impulse (35 ms / 1.5 s)": "Impulso (35 ms / 1,5 s)",
    "Time-weighting ballistics (IEC 61672-1)":
        "Balística de la ponderación temporal (IEC 61672-1)",
    "Mean-square response (normalized)":
        "Respuesta cuadrática media (normalizada)",
    "RC exponential detector": "Detector exponencial RC",
    "input x(t)": "entrada x(t)",
    "square-law rectifier": "rectificador cuadrático",
    "stored charge (Fast shown)": "carga almacenada (se muestra Fast)",
    "charging": "cargando",
    "draining": "descargando",
    "τ = RC sets attack and decay": "τ = RC fija el ataque y la caída",
    "onset (> 10 dB/s)": "inicio (> 10 dB/s)",
    "Impulse onset detection (NT ACOU 112)":
        "Detección del inicio de impulso (NT ACOU 112)",
    "A-weighted level L_AF [dB]": "Nivel ponderado A L_AF [dB]",
    "L_AF (A-weighted, Fast)": "L_AF (ponderado A, Fast)",
    "detector: onset when dL/dt > 10 dB/s":
        "detector: inicio cuando dL/dt > 10 dB/s",
    "onset rate": "tasa de inicio",
    "level difference": "diferencia de nivel",
    "prominence": "prominencia",
    "adjustment": "ajuste",
    "add KI to the rating level": "sumar KI al nivel de evaluación",
    "pressure p": "presión p",
    "velocity u": "velocidad u",
    "intensity p·u": "intensidad p·u",
    "amplitude (normalized)": "amplitud (normalizada)",
    "Two-microphone p-p probe: instantaneous intensity p·u":
        "Sonda p-p de dos micrófonos: intensidad instantánea p·u",
    "Progressive wave — active": "Onda progresiva — activa",
    "Standing wave — reactive": "Onda estacionaria — reactiva",
    "p and u in phase": "p y u en fase",
    "p and u 90° apart": "p y u desfasados 90°",
    "spacer Δr": "separador Δr",
    "T20 fit": "ajuste T20",
    "T30 fit": "ajuste T30",
    "Schroeder backward integration (ISO 3382)":
        "Integración inversa de Schroeder (ISO 3382)",
    "← integrate from the tail": "← integrar desde la cola",
    "squared impulse response p²": "respuesta al impulso al cuadrado p²",
    "tail energy": "energía de la cola",
    "Room modes in a rigid 5 m × 3.5 m room (2D FDTD)":
        "Modos propios en una sala rígida de 5 m × 3,5 m (FDTD 2D)",
    "instantaneous p(x, y)": "p(x, y) instantánea",
    "RMS pressure (mode map)": "presión RMS (mapa modal)",
    "nodal lines (2,1)": "líneas nodales (2,1)",
    "source": "fuente",
    "same color scale": "misma escala de color",
    # --- Tier-1 animation labels (second batch) ---
    "Standing wave in the impedance tube (ISO 10534-2)":
        "Onda estacionaria en el tubo de impedancia (ISO 10534-2)",
    "Rigid termination": "Terminación rígida",
    "Porous sample": "Muestra porosa",
    "sample": "muestra",
    "rigid wall": "pared rígida",
    "incident": "incidente",
    "reflected": "reflejada",
    "sum p(x, t)": "suma p(x, t)",
    "envelope |p(x)|": "envolvente |p(x)|",
    "deep nodes": "nodos profundos",
    "shallow nodes": "nodos poco profundos",
    "Flanking transmission paths (EN 12354-1)":
        "Caminos de transmisión por flancos (EN 12354-1)",
    "source room": "recinto emisor",
    "receiving room": "recinto receptor",
    "direct, wall to wall": "directo, de muro a muro",
    "floor to floor": "de suelo a suelo",
    "floor to wall": "de suelo a muro",
    "wall to floor": "de muro a suelo",
    "junction: Kij attenuates each transfer":
        "unión: Kij atenúa cada transferencia",
    "R'w sums all paths — always below the wall alone":
        "R'w suma todos los caminos — siempre menor que el muro solo",
    "Intensity scanning over a box surface (ISO 9614-2)":
        "Barrido de intensidad sobre una superficie en caja (ISO 9614-2)",
    "p-p probe": "sonda p-p",
    "normal intensity I·n on the surface":
        "intensidad normal I·n en la superficie",
    "partial powers": "potencias parciales",
    "top": "superior",
    "front": "frontal",
    "back": "trasera",
    "left": "izquierda",
    "right": "derecha",
    "any enclosing surface gives the same P":
        "cualquier superficie envolvente da la misma P",
    "Sweep measurement and deconvolution (ISO 18233)":
        "Medición con barrido y deconvolución (ISO 18233)",
    "mic": "micro",
    "direct + reflections": "directo + reflexiones",
    "Frequency [kHz]": "Frecuencia [kHz]",
    "recorded sweep (spectrogram)": "barrido grabado (espectrograma)",
    "delayed copies = reflections": "copias retardadas = reflexiones",
    "impulse response": "respuesta al impulso",
    "direct": "directo",
    "⊛ inverse filter": "⊛ filtro inverso",
    "same information, different domain: sweep ⊛ inverse filter = impulse"
    " response":
        "la misma información en otro dominio: barrido ⊛ filtro inverso ="
        " respuesta al impulso",
    "Specific loudness N'(z) and its integral (ISO 532-1)":
        "Sonoridad específica N'(z) y su integral (ISO 532-1)",
    "1 kHz ≈ 8.5 Bark": "1 kHz ≈ 8,5 Bark",
    "upward spread of masking":
        "extensión del enmascaramiento hacia agudos",
    "1 kHz narrowband": "banda estrecha de 1 kHz",
    "One source, two rooms, one sound power":
        "Una fuente, dos salas, una única potencia acústica",
    "Anechoic room (ISO 3745)": "Cámara anecoica (ISO 3745)",
    "Reverberation room (ISO 3741)": "Cámara reverberante (ISO 3741)",
    "microphone sphere, r": "esfera de micrófonos, r",
    "direct sound only — no reflections":
        "solo sonido directo — sin reflexiones",
    "rotating microphone": "micrófono giratorio",
    "reflections build a diffuse field":
        "las reflexiones crean un campo difuso",
    "the room changes Lp, not the source power":
        "la sala cambia Lp, no la potencia de la fuente",
    "Comb filtering from a single reflection":
        "Filtrado en peine por una sola reflexión",
    "reflecting floor": "suelo reflectante",
    "image source": "fuente imagen",
    "arrival time [ms]": "tiempo de llegada [ms]",
    "amplitude": "amplitud",
    "delayed copy": "copia retardada",
    "response [dB]": "respuesta [dB]",
    "high mic: dense comb": "micro alto: peine denso",
    "lower: notches move up": "más bajo: los nulos suben",
    "on the floor: copies merge — no comb in band":
        "en el suelo: las copias se funden — sin peine en banda",
    "first notch above 8 kHz": "primer nulo por encima de 8 kHz",
    # --- FDTD animation labels (third batch) ---
    "Barrier diffraction into the shadow zone (2D FDTD)":
        "Difracción en una barrera hacia la zona de sombra (FDTD 2D)",
    "barrier": "barrera",
    "shadow zone": "zona de sombra",
    "rigid ground": "suelo rígido",
    "RMS level [dB re panel max]": "nivel RMS [dB re máx del panel]",
    "diffraction fills the shadow": "la difracción rellena la sombra",
    "deep, clean shadow": "sombra profunda y limpia",
    "each panel on its own dB scale": "cada panel en su propia escala dB",
    "Ground effect: direct + reflected interference (2D FDTD)":
        "Efecto de suelo: interferencia directa + reflejada (FDTD 2D)",
    "source (h = 1.5 m)": "fuente (h = 1,5 m)",
    "image source (ghost)": "fuente imagen (fantasma)",
    "receiver in a dip": "receptor en un mínimo",
    "instantaneous pressure": "presión instantánea",
    "RMS level: interference lobes": "nivel RMS: lóbulos de interferencia",
    "Level on the 8 m arc": "Nivel en el arco de 8 m",
    "elevation angle θ [°]": "ángulo de elevación θ [°]",
    "level [dB re max]": "nivel [dB re máx]",
    "image-source model": "modelo de fuente imagen",
    "predicted nulls": "mínimos previstos",
    "dips land exactly on the predicted nulls":
        "los mínimos caen exactamente en los nulos previstos",
    "FDTD": "FDTD",
    "SOFAR channel: sound trapped by the c(z) minimum (2D FDTD)":
        "Canal SOFAR: sonido atrapado por el mínimo de c(z) (FDTD 2D)",
    "Source on the channel axis (depth 400 m)":
        "Fuente en el eje del canal (400 m de profundidad)",
    "Source near the surface (depth 150 m)":
        "Fuente cerca de la superficie (150 m de profundidad)",
    "channel axis (c minimum)": "eje del canal (mínimo de c)",
    "trapped: wavefronts bend back to the axis":
        "atrapado: los frentes de onda se curvan de vuelta al eje",
    "leaks: energy escapes the channel":
        "fuga: la energía escapa del canal",
    "c(z) [m/s]": "c(z) [m/s]",
    "Flat panel vs Schroeder diffuser (2D FDTD)":
        "Panel plano frente a difusor de Schroeder (FDTD 2D)",
    "Flat rigid panel": "Panel rígido plano",
    "Schroeder diffuser (QRD, N = 7)": "Difusor de Schroeder (QRD, N = 7)",
    "incident plane wavefront": "frente de onda plano incidente",
    "sound field p": "campo sonoro p",
    "scattered field (total − incident)":
        "campo dispersado (total − incidente)",
    "specular beam": "haz especular",
    "scattered fan": "abanico dispersado",
    "receiver arc": "arco de receptores",
    "Programme Loudness Metering (EBU R 128)":
        "Medición de sonoridad de programa (EBU R 128)",
    "Loudness [LUFS]": "Sonoridad [LUFS]",
    "Momentary M (400 ms)": "Momentánea M (400 ms)",
    "Short-term S (3 s)": "Corto plazo S (3 s)",
    "ambience": "ambiente",
    "dialogue": "diálogo",
    "music": "música",
    "fade-out": "fundido",
    # Rooms & materials result figures (WP: rooms & materials)
    r"Normal incidence $\alpha(0°)$": r"Incidencia normal $\alpha(0°)$",
    "Probe pressure spectrum": "Espectro de presión en la sonda",
    "Analytic mode frequencies": "Frecuencias modales analíticas",
    "Rigid-box FDTD probe spectrum vs analytic modes":
        "Espectro FDTD en la sonda de una caja rígida frente a los modos analíticos",
    "Probe spectrum [dB re max]": "Espectro en la sonda [dB re máx]",
    # Building & structure-borne result figures (guide figure coverage).
    "ISO 717-1 Enlarged-Range Rating (Annex B)":
        "Índice ponderado con rango ampliado ISO 717-1 (Anexo B)",
    "enlarged range (Annex B)": "rango ampliado (Anexo B)",
    "measured R": "R medido",
    "shifted reference (100-3150 Hz)": "referencia desplazada (100-3150 Hz)",
    "shifted reference": "referencia desplazada",
    "unfavourable deviations": "desviaciones desfavorables",
    "ISO 16283-1 Field Airborne Insulation":
        "Aislamiento a ruido aéreo in situ ISO 16283-1",
    "ISO 16283-3 Field Facade Insulation":
        "Aislamiento de fachada in situ ISO 16283-3",
    "D2m,nT (standardized)": "D2m,nT (estandarizada)",
    "D2m,n (normalized)": "D2m,n (normalizada)",
    "R'45° (element)": "R'45° (elemento)",
    "Level difference / reduction index [dB]":
        "Diferencia de nivel / índice de reducción [dB]",
    "ISO 10052 Survey Method: Impact Sound":
        "Método de control ISO 10052: ruido de impactos",
    "Li (impact level)": "Li (nivel de impactos)",
    "L'nT (standardized)": "L'nT (estandarizado)",
    "Impact sound pressure level [dB]":
        "Nivel de presión acústica de impactos [dB]",
    "ISO 10140 Laboratory Insulation (flanking suppressed)":
        "Aislamiento en laboratorio ISO 10140 (flancos suprimidos)",
    "normalized Ln": "Ln normalizado",
    "Impact sound pressure level Ln [dB]":
        "Nivel de presión acústica de impactos Ln [dB]",
    "ISO 15186-1 Small-Element Insulation by Intensity":
        "Aislamiento de elementos pequeños por intensidad ISO 15186-1",
    "DI,n,e (element)": "DI,n,e (elemento)",
    "Element normalized level difference [dB]":
        "Diferencia de niveles normalizada de elemento [dB]",
    "ISO 10848 Airborne Flanking Transmission":
        "Transmisión aérea por flancos ISO 10848",
    "Dn,f (flanking)": "Dn,f (flancos)",
    "Normalized flanking level difference [dB]":
        "Diferencia de niveles normalizada de flancos [dB]",
    "EN 12354-2 Impact Sound Prediction (Annex E.3)":
        "Predicción de ruido de impactos EN 12354-2 (Anexo E.3)",
    "Level / correction [dB]": "Nivel / corrección [dB]",
    "EN 12354-4 Radiated Sound Power (Annex G)":
        "Potencia acústica radiada EN 12354-4 (Anexo G)",
    "radiated $L_W$ per octave": "$L_W$ radiada por octava",
    "Radiated sound power level [dB re 1 pW]":
        "Nivel de potencia acústica radiada [dB re 1 pW]",
    "Predicted Single-Panel Insulation Rated per ISO 717-1":
        "Aislamiento previsto de panel simple evaluado según ISO 717-1",
    "predicted R (Sharp)": "R previsto (Sharp)",
    "Wave-Approach Junction $K_{ij}$ (Hopkins Eq. 5.116)":
        "$K_{ij}$ de unión por el enfoque ondulatorio (Hopkins Ec. 5.116)",
    "X corner": "X esquina",
    "X straight": "X recta",
    "T-junction (1) corner": "unión en T (1) esquina",
    "L corner": "L esquina",
    "identical plates (τ = 1/12)": "placas idénticas (τ = 1/12)",
    "Thickness ratio h2/h1": "Relación de espesores h2/h1",
    "Vibration reduction index $K_{ij}$ [dB]":
        "Índice de reducción de vibraciones $K_{ij}$ [dB]",
    "Reading a Driving-Point Mobility (ISO 7626-1)":
        "Lectura de una movilidad en el punto de excitación (ISO 7626-1)",
    "driving-point $|Y(f)|$": "$|Y(f)|$ en el punto de excitación",
    r"stiffness line $\omega/k$": r"línea de rigidez $\omega/k$",
    r"mass line $1/(\omega m)$": r"línea de masa $1/(\omega m)$",
    "peak $|Y| = 1/c$ (damping)": "pico $|Y| = 1/c$ (amortiguamiento)",
    "Mobility $|Y|$ [m/(N·s)]": "Movilidad $|Y|$ [m/(N·s)]",
    # Atmospheric refraction (effective profiles, ray fan, GFPE range cut) and
    # wave-theoretic barrier insertion loss (ground-and-barriers guide).
    "Effective Sound-Speed Profiles (Salomons Eq. 4.5)":
        "Perfiles de velocidad efectiva del sonido (Salomons ec. 4.5)",
    "Downward refraction (b = +1 m/s)": "Refracción hacia abajo (b = +1 m/s)",
    "Upward refraction (b = -1 m/s)": "Refracción hacia arriba (b = -1 m/s)",
    "Sound Rays under Downward Refraction (b = +1 m/s)":
        "Rayos sonoros con refracción hacia abajo (b = +1 m/s)",
    "shallow rays are bent back to the ground\nand bounce on down-range":
        "los rayos rasantes se curvan de vuelta al suelo\ny rebotan a lo largo de la distancia",
    "GFPE Relative Level at the Receiver Height (400 Hz, 2 m)":
        "Nivel relativo GFPE a la altura del receptor (400 Hz, 2 m)",
    "Downward (b = +1 m/s)": "Hacia abajo (b = +1 m/s)",
    "Homogeneous (b = 0)": "Homogénea (b = 0)",
    "Upward (b = -1 m/s)": "Hacia arriba (b = -1 m/s)",
    "Wave-Theoretic Barrier Insertion Loss":
        "Pérdida por inserción de barrera (teoría ondulatoria)",
    "Kurze-Anderson (thin screen)": "Kurze-Anderson (pantalla delgada)",
    "Exact rigid half-plane": "Semiplano rígido exacto",
    "Exact + coherent ground (four paths)":
        "Exacto + suelo coherente (cuatro caminos)",
    "Kurze-Anderson grazing limit (5 dB)":
        "Límite rasante de Kurze-Anderson (5 dB)",
    "Insertion loss [dB]": "Pérdida por inserción [dB]",
    # --- WP emission & electroacoustics figures (result .plot() labels) ---
    "Carrier f₂": "Portadora f₂",
    "Sidebands f₂ ± n·f₁": "Bandas laterales f₂ ± n·f₁",
    "Level re carrier [dB]": "Nivel respecto a la portadora [dB]",
    "$R_1$ (resistance)": "$R_1$ (resistencia)",
    "$X_1$ (reactance)": "$X_1$ (reactancia)",
    r"Normalized radiation impedance $Z_r / \rho c S$":
        r"Impedancia de radiación normalizada $Z_r / \rho c S$",
    "Baffled circular piston radiation impedance":
        "Impedancia de radiación de un pistón circular con pantalla",
    "F2 (surface pressure-intensity)": "F2 (presión-intensidad superficial)",
    "F3 (negative partial power)": "F3 (potencia parcial negativa)",
    "Dynamic capability Ld": "Capacidad dinámica Ld",
    "F4 (non-uniformity)": "F4 (no uniformidad)",
    "Indicator [dB]": "Indicador [dB]",
    "Field non-uniformity F4": "No uniformidad del campo F4",
    "ISO 9614-1 field indicators": "Indicadores de campo ISO 9614-1",
    "Helmholtz resonator": "Resonador de Helmholtz",
    "Quarter-wave tube": "Tubo de cuarto de onda",
    "Side-branch resonators: transmission loss (Bies Eqs. 8.44, 8.46)":
        "Resonadores en derivación: pérdida de transmisión (Bies Ecs. 8.44, 8.46)",
    "Duct end reflection loss (ASHRAE Table 8.14)":
        "Pérdida por reflexión del extremo del conducto (ASHRAE Tabla 8.14)",
    "End reflection loss [dB]": "Pérdida por reflexión del extremo [dB]",
    "Duct diameter": "Diámetro del conducto",
    "Duct-borne noise into the room: supply, return and NC 30":
        "Ruido transmitido por conductos a la sala: impulsión, retorno y NC 30",
    "Supply": "Impulsión",
    "Return": "Retorno",
    "Plant room to operator room: what the wall delivers, and NC 45":
        "Sala de máquinas a sala de control: lo que aporta el muro, y NC 45",
    "Duct higher-order-mode cut-on: 254 mm steam line at 200 m/s":
        "Corte de modos superiores del conducto: línea de vapor de 254 mm a 200 m/s",
    "Panel R": "R del panel",
    "Interior correction C": "Corrección interior C",
    "Insertion loss (R - C)": "Pérdida por inserción (R - C)",
    "Machine enclosure insertion loss":
        "Pérdida por inserción de encapsulado de máquina",
    "Measured phase": "Fase medida",
    "Minimum phase (from |H|)": "Fase mínima (de |H|)",
    "Excess phase (all-pass)": "Fase de exceso (pasa-todo)",
    "Phase [rad]": "Fase [rad]",
    "Minimum-phase / all-pass decomposition":
        "Descomposición fase mínima / pasa-todo",
    "Group delay": "Retardo de grupo",
    "Excess group delay": "Retardo de grupo de exceso",
    "Momentary (400 ms)": "Momentánea (400 ms)",
    "Short-term (3 s)": "Corto plazo (3 s)",
    "Programme loudness (EBU R 128)": "Sonoridad de programa (EBU R 128)",
    # Core-metrology figures: resampling, cepstrum, correlation, spectra,
    # system measurement, synchronous averaging, data qualification.
    "Polyphase Resampling 44.1 kHz → 48 kHz: the Delivered Anti-Alias Filter":
        "Remuestreo polifásico 44,1 kHz → 48 kHz: "
        "el filtro antisolapamiento entregado",
    "Anti-alias filter $|H(f)|$": "Filtro antisolapamiento $|H(f)|$",
    "Passband edge": "Borde de la banda de paso",
    "Stopband edge (alias fold)":
        "Borde de la banda atenuada (pliegue de alias)",
    "Design attenuation -120 dB": "Atenuación de diseño -120 dB",
    "Rejected band (would fold back as aliases)":
        "Banda rechazada (se plegaría como alias)",
    "The Three Cepstrum Variants of One Echo-Carrying Record":
        "Las tres variantes del cepstro de un registro con eco",
    "Real cepstrum (exactly half the power)":
        "Cepstro real (exactamente la mitad del de potencia)",
    "Complex cepstrum": "Cepstro complejo",
    "first rahmonic at 8 ms:\nheight ≈ a on the power cepstrum":
        "primer rahmónico en 8 ms:\naltura ≈ a en el cepstro de potencia",
    "second rahmonic: $-a^2/2$": "segundo rahmónico: $-a^2/2$",
    "Liftering at 4 ms: Envelope Versus Echo Ripple":
        "Liftering a 4 ms: envolvente frente a rizado del eco",
    "Log spectrum of the record": "Espectro logarítmico del registro",
    "Lowpass lifter: spectral envelope":
        "Lifter paso bajo: envolvente espectral",
    "Highpass lifter: the echo ripple alone":
        "Lifter paso alto: solo el rizado del eco",
    "closed-form ripple bounds $20\\log_{10}(1\\pm a)$":
        "cotas del rizado en forma cerrada $20\\log_{10}(1\\pm a)$",
    "Correlation Normalizations of a Two-Sensor Delay Model":
        "Normalizaciones de la correlación de un modelo "
        "de dos sensores con retardo",
    "Coefficient $\\rho_{xy}(\\tau)$ (bounded by $\\pm 1$)":
        "Coeficiente $\\rho_{xy}(\\tau)$ (acotado por $\\pm 1$)",
    "true delay +12.5 ms": "retardo verdadero +12,5 ms",
    "Biased $1/N$ (tapers toward the ends)":
        "Sesgada $1/N$ (se atenúa hacia los extremos)",
    "Unbiased $1/(N-|r|)$ (variance grows at the ends)":
        "Insesgada $1/(N-|r|)$ (la varianza crece en los extremos)",
    "Lag [s]": "Retardo [s]",
    "Correlation": "Correlación",
    "Sub-Sample Impulse-Response Alignment":
        "Alineación submuestra de respuestas al impulso",
    "Reference IR": "RI de referencia",
    "Measured IR (delayed 7.37 samples)":
        "RI medida (retardada 7,37 muestras)",
    "Aligned IR (delay removed)": "RI alineada (retardo eliminado)",
    "Hilbert Envelope and Instantaneous Frequency":
        "Envolvente de Hilbert y frecuencia instantánea",
    "Signal": "Señal",
    "Envelope $A(t)$": "Envolvente $A(t)$",
    "Instantaneous frequency $f(t)$": "Frecuencia instantánea $f(t)$",
    "Instantaneous frequency [Hz]": "Frecuencia instantánea [Hz]",
    "carrier 250 Hz": "portadora de 250 Hz",
    "Cross-Spectral Density of a 2 ms Delay Path":
        "Densidad espectral cruzada de un camino con retardo de 2 ms",
    "$|\\hat{G}_{xy}|$ (Welch estimate)":
        "$|\\hat{G}_{xy}|$ (estimación de Welch)",
    "Unwrapped phase": "Fase desenrollada",
    "$\\pm 1$ s.d. band (Eq. 9.52)": "banda $\\pm 1$ d.e. (Ec. 9.52)",
    "slope $-2\\pi f\\tau$ ($\\tau$ = 2 ms)":
        "pendiente $-2\\pi f\\tau$ ($\\tau$ = 2 ms)",
    "Coherent Output Spectrum and Spectral SNR (Bendat & Piersol 9.2.2)":
        "Espectro de salida coherente y SNR espectral (Bendat y Piersol 9.2.2)",
    "$\\hat{G}_{yy}$ (measured output)": "$\\hat{G}_{yy}$ (salida medida)",
    "$\\hat{G}_{vv} = \\gamma^2\\hat{G}_{yy}$ (coherent part)":
        "$\\hat{G}_{vv} = \\gamma^2\\hat{G}_{yy}$ (parte coherente)",
    "$\\hat{G}_{nn}$ (uncorrelated noise)":
        "$\\hat{G}_{nn}$ (ruido no correlado)",
    "Spectral density [dB re 1/Hz]": "Densidad espectral [dB re 1/Hz]",
    "Spectral SNR [dB]": "SNR espectral [dB]",
    "closed form $10\\log_{10}(2.56)$ = 4.1 dB":
        "forma cerrada $10\\log_{10}(2{,}56)$ = 4,1 dB",
    "Golay-Pair Impulse Response: Exact Complementary Recovery":
        "Respuesta al impulso con par de Golay: "
        "recuperación complementaria exacta",
    "Recovered IR (golay_impulse_response)":
        "RI recuperada (golay_impulse_response)",
    "True system response": "Respuesta verdadera del sistema",
    "noise-free closed-form identity": "identidad exacta sin ruido",
    "TSA Noise Reduction: the $\\sqrt{N}$ Law (McFadden 1987)":
        "Reducción de ruido del promediado síncrono: "
        "la ley $\\sqrt{N}$ (McFadden 1987)",
    "Measured RMS error of the average":
        "Error RMS medido del promedio",
    "Ideal $\\sigma/\\sqrt{N}$": "$\\sigma/\\sqrt{N}$ ideal",
    "Number of averages N": "Número de promedios N",
    "RMS error of the averaged waveform":
        "Error RMS de la forma de onda promediada",
    "Runs Test About the Median (Wald & Wolfowitz)":
        "Test de rachas respecto a la mediana (Wald y Wolfowitz)",
    "Above the median": "Por encima de la mediana",
    "Below the median": "Por debajo de la mediana",
    "Sequence median": "Mediana de la secuencia",
    # anim_fdtd_slit_absorber
    "Slow-sound slit absorber at critical coupling (2D FDTD)":
        "Absorbedor de rendija de sonido lento en acoplamiento crítico "
        "(FDTD 2D)",
    "Critically coupled slit: the wave dies inside":
        "Rendija en acoplamiento crítico: la onda muere dentro",
    "Wide slit (detuned): the reflection returns":
        "Rendija ancha (desintonizada): la reflexión vuelve",
    "inside the cell": "dentro de la celda",
    # anim_fdtd_expansion_chamber
    # Kept short: the canvas clips a suptitle much beyond ~80 characters.
    "Expansion-chamber silencer: pass band vs stop band (2D FDTD)":
        "Silenciador de cámara de expansión: paso frente a rechazo "
        "(FDTD 2D)",
    "the chamber is acoustically invisible":
        "la cámara es acústicamente invisible",
    "the mismatch reflects the wave back up the pipe":
        "el desajuste refleja la onda de vuelta por el conducto",
    "Position along the duct [m]": "Posición a lo largo del conducto [m]",
    # anim_fdtd_aperture_slit
    "Sound through a wall aperture (2D FDTD)":
        "Sonido a través de una abertura en un muro (FDTD 2D)",
    "cylindrical re-radiation from the slit":
        "re-radiación cilíndrica desde la rendija",
    "the front passes: sharp-edged shadow":
        "el frente pasa: sombra de bordes nítidos",
    "RMS level [dB]": "Nivel RMS [dB]",
    # anim_fdtd_refraction
    "Atmospheric refraction: downwind duct, upwind shadow (2D FDTD)":
        "Refracción atmosférica: canal a favor del viento, sombra en "
        "contra (FDTD 2D)",
    "Downwind: sound speed grows with height":
        "A favor del viento: la velocidad del sonido crece con la altura",
    "Upwind: sound speed falls with height":
        "En contra del viento: la velocidad del sonido cae con la altura",
    "bent down: a duct hugs the ground, the receiver stays loud":
        "curvada hacia abajo: un canal pegado al suelo, el receptor sigue "
        "oyendo",
    "bent up: a shadow opens, the receiver goes quiet":
        "curvada hacia arriba: se abre una sombra, el receptor se queda en "
        "silencio",
    "source (h = 2 m)": "fuente (h = 2 m)",
    "receiver 350 m": "receptor a 350 m",
    "c_eff(z) [m/s]": "c_ef(z) [m/s]",
    # anim_elastic_plate_junction
    "Bending waves at an L-junction (elastic 2D FDTD)":
        "Ondas de flexión en una unión en L (FDTD elástico 2D)",
    "Straight plate: the packet just runs on":
        "Placa recta: el paquete sigue de largo",
    "L-junction: reflected, transmitted and mode-converted":
        "Unión en L: reflexión, transmisión y conversión de modo",
    "nothing comes back": "no vuelve nada",
    "free end": "extremo libre",
    "4 kHz tone burst at ▼": "salva de tono de 4 kHz en ▼",
    # anim_elastic_coincidence
    "Coincidence: the same steel plate, below and above f_c "
    "(elastic 2D FDTD)":
        "Coincidencia: la misma placa de acero, bajo y sobre f_c "
        "(FDTD elástico 2D)",
    "10 mm steel plate": "placa de acero de 10 mm",
    # coupling_term_regimes (EN 12354-5, buildings/design)
    "Formula 19b (exact, complex mobilities)":
        "Fórmula 19b (exacta, movilidades complejas)",
    "force-source limit (19c)": "límite de fuente de fuerza (19c)",
    "velocity-source limit (19d)": "límite de fuente de velocidad (19d)",
    "Mobility ratio |Ys| / |Yi|": "Cociente de movilidades |Ys| / |Yi|",
    "Coupling term D_C [dB]": "Término de acoplamiento D_C [dB]",
    "EN 12354-5 Coupling Term and Its Two Limits":
        "Término de acoplamiento de la EN 12354-5 y sus dos límites",
    "D_C = 10 log10(|Ys + Yi + Yk|² / (|Ys| Re{Yi}))\n"
    "left: velocity source (stiff receiver takes more)\n"
    "right: force source (stiff receiver takes less)":
        "D_C = 10 log10(|Ys + Yi + Yk|² / (|Ys| Re{Yi}))\n"
        "izquierda: fuente de velocidad (un receptor rígido acepta más)\n"
        "derecha: fuente de fuerza (un receptor rígido acepta menos)",
    # tapping_force_spectrum (buildings/design/resilient-layers)
    "|Fn|upper = 2 m vh / Ti  (rebound)":
        "|Fn|superior = 2 m vh / Ti  (con rebote)",
    "|Fn|lower = m vh / Ti  (no rebound)":
        "|Fn|inferior = m vh / Ti  (sin rebote)",
    "Line force |Fn| [N]": "Fuerza por línea |Fn| [N]",
    "Tapping-Machine Force: the Floor Decides the Excitation":
        "Fuerza de la máquina de impactos: el forjado decide la excitación",
    "0.5 kg hammer dropped 40 mm, 10 impacts per second\n"
    "building acoustics range shaded":
        "martillo de 0,5 kg desde 40 mm, 10 impactos por segundo\n"
        "rango de acústica de la edificación sombreado",
    # detailed_impact_paths (buildings/design/detailed-prediction)
    "Apparent normalized impact level L'n [dB]":
        "Nivel de impactos normalizado aparente L'n [dB]",
    "L'n (apparent)": "L'n (aparente)",
    "ISO 12354-2 Detailed Model: the Direct Path Governs (Annex G)":
        "Modelo detallado de la ISO 12354-2: manda la vía directa (anexo G)",
    # radiation_efficiency_panels (buildings/design/panel-sound-insulation)
    "0.5 m x 0.4 m pane, same glass":
        "vidrio de 0,5 m x 0,4 m, el mismo material",
    "sigma = 1 (as efficient as a piston)":
        "sigma = 1 (tan eficiente como un pistón)",
    "Radiation efficiency sigma": "Eficiencia de radiación sigma",
    "Edge Radiation, Coincidence and the Slow Return to Unity":
        "Radiación de bordes, coincidencia y el lento retorno a la unidad",
    # structure_borne_conversion (buildings/design/structure-borne-power)
    "One Source, Four Levels: the EN 15657 Conversion Chain":
        "Una fuente, cuatro niveles: la cadena de conversión de la EN 15657",
    "Structure-borne power level [dB re 1 pW]":
        "Nivel de potencia estructural [dB re 1 pW]",
    "L_Ws measured on the test plate (Y = 5.34e-6)":
        "L_Ws medido sobre la placa de ensayo (Y = 5,34e-6)",
    "L_Wsn on the standard plate (Y = 5e-6): what is declared":
        "L_Wsn sobre la placa normalizada (Y = 5e-6): lo que se declara",
    "L_Ws,inst on the receiving wall (Y = 24.1e-6)":
        "L_Ws,inst sobre el muro receptor (Y = 24,1e-6)",
    "L_Ws,c with the source mobility (Y = 1e-3): the input to EN 12354-5":
        "L_Ws,c con la movilidad de la fuente (Y = 1e-3): la entrada de la EN 12354-5",
    "EN 12354-5 Annex I.3 flushing cistern, wall contact\n"
    "markers reproduce the printed Table I.8 columns":
        "Cisterna del anexo I.3 de la EN 12354-5, contacto en el muro\n"
        "los marcadores reproducen las columnas impresas de la tabla I.8",
    # Aircraft: the ECAC Doc 29 single event and its per-segment corrections.
    "Doc 29 Segment Contributions at One Receiver":
        "Aportación de cada segmento en un receptor (Doc 29)",
    "(a) Engine installation (Eq. 4-15/4-16)":
        "(a) Instalación de motores (ec. 4-15/4-16)",
    "(b) Lateral attenuation (Eq. 4-18/4-19)":
        "(b) Atenuación lateral (ec. 4-18/4-19)",
    "(d) Duration correction (Eq. 4-14)": "(d) Corrección de duración (ec. 4-14)",
    "Depression angle φ [°]": "Ángulo de depresión φ [°]",
    "Elevation angle β [°]": "Ángulo de elevación β [°]",
    "Λ(β, ℓ) subtracted [dB]": "Λ(β, ℓ) restada [dB]",
    "Segment speed Vseg [m/s]": "Velocidad del segmento Vseg [m/s]",
    "Wing-mounted": "En ala",
    "Fuselage-mounted": "En fuselaje",
    "Propeller": "Hélice",
    "Λ = 0 above 50°": "Λ = 0 por encima de 50°",
    "observer alongside": "observador junto al segmento",
    "Vref = 82.3 m/s (160 kn)": "Vref = 82,3 m/s (160 kn)",
    "take-off ground roll": "carrera de despegue",
    "default ground track": "traza en tierra por defecto",
    # Rotorcraft: the ECAC Doc 32 hemisphere, its interpolation and its terrain.
    "Fore-aft section (φ = 0°)": "Sección proa-popa (φ = 0°)",
    "measured polar band": "banda polar medida",
    "measured coverage": "cobertura medida",
    "Polar angle θ [°]": "Ángulo polar θ [°]",
    "Azimuth φ [°]": "Acimut φ [°]",
    "Uniform pasture (class D)": "Pastizal uniforme (clase D)",
    "A 600 m hard strip across it": "Una franja dura de 600 m que lo cruza",
    "track": "trayectoria",
    "the event receiver": "el receptor del evento",
    "Mean Ground Plane and Equivalent Heights (ECAC Doc 32 / NORAH2)":
        "Plano medio del terreno y alturas equivalentes (ECAC Doc 32 / NORAH2)",
    "receiver": "receptor",
    "Raw (V, γ) plane — pass it as triangles=":
        "Plano (V, γ) sin normalizar: se pasa como triangles=",
    "Normalised plane — the library default":
        "Plano normalizado: el comportamiento por defecto",
    "Airspeed V [m/s]": "Velocidad aerodinámica V [m/s]",
    "Path angle γ [°]": "Ángulo de trayectoria γ [°]",
    "Ffc · γ / Δγ": "Ffc · γ / Δγ",
    "database conditions": "condiciones de la base de datos",
    "query inside the hull": "consulta dentro de la envolvente",
    "query outside: nearest, unblended":
        "consulta fuera: la más próxima, sin mezclar",
    "Raw radar track, 1 s cadence": "Traza radar sin tratar, cadencia de 1 s",
    "Smoothing spline resampled to 0.5 s":
        "Spline de suavizado remuestreado a 0,5 s",
    # Labels the aircraft result objects draw themselves.
    "Segment SEL [dB]": "SEL del segmento [dB]",
    "Polar angle θ [°]  (0° forward → 180° rearward)":
        "Ángulo polar θ [°]  (0° hacia proa → 180° hacia popa)",
    "Source level at 60 m [dB]": "Nivel de fuente a 60 m [dB]",
    "SEL [dB]": "SEL [dB]",
    "SEL [dB(A)]": "SEL [dB(A)]",
    "Speed [m/s]": "Velocidad [m/s]",
    "Angle [°]": "Ángulo [°]",
    "Airspeed $V_A$": "Velocidad aerodinámica $V_A$",
    "Ground speed $V_g$": "Velocidad respecto al suelo $V_g$",
    "Path angle $\\gamma$": "Ángulo de trayectoria $\\gamma$",
    "Bank angle $\\Phi$": "Ángulo de alabeo $\\Phi$",
}

_ES_PATTERNS = [
    # Aircraft: labels and annotations carrying computed values.
    (r"^closest segment: (.+) dB$", r"segmento más próximo: \1 dB"),
    ((r"^receiver 3 000 m along track, 500 m to the side, 1\.2 m up\.\n"
      r"total SEL (.+) dB; the closest segment alone\n"
      r"carries (.+) % of the energy\. Hatched: the take-off ground roll\.$"),
     ("receptor a 3 000 m en la traza, 500 m al lado y 1,2 m de altura.\n"
      r"SEL total \1 dB; el segmento más próximo aporta por sí solo"
      "\n"
      r"el \2 % de la energía. Rayado: la carrera de despegue.")),
    (r"^ℓ = (.+) m$", r"ℓ = \1 m"),
    (r"^λ = (.+) m$", r"λ = \1 m"),
    (r"^\(c\) Noise fraction, dλ = (.+) m \(Eq\. 4-20\)$",
     r"(c) Fracción de ruido, dλ = \1 m (ec. 4-20)"),
    ((r"^impedance adjustment \(Eq\. 4-6/4-7\), for scale:\n"
      r"15 °C, 101\.3 kPa: (.+) dB\n30 °C, 101\.3 kPa: (.+) dB\n"
      r"15 °C, 95\.0 kPa: (.+) dB$"),
     ("ajuste de impedancia (ec. 4-6/4-7), para dar escala:\n"
      r"15 °C, 101,3 kPa: \1 dB" "\n" r"30 °C, 101,3 kPa: \2 dB" "\n"
      r"15 °C, 95,0 kPa: \3 dB")),
    (r"^ANP Departure SEL Contour - (.+)$", r"Curvas SEL de salida ANP - \1"),
    (r"^stage length (.+), (.+) fixed points, (.+) m at the last one$",
     r"etapa \1, \2 puntos fijos, \3 m en el último"),
    (r"^event_level receiver: SEL (.+) dB$",
     r"receptor de event_level: SEL \1 dB"),
    (r"^(.+) Hz band, the most directive one$",
     r"banda de \1 Hz, la más directiva"),
    ((r"^(.+) dB between the loudest and quietest measured cell;\n"
      r"outside the dashed patch the field is gap-filled, not measured$"),
     (r"\1 dB entre la celda medida más y menos ruidosa;" "\n"
      "fuera del recinto discontinuo el campo está rellenado, no medido")),
    (r"^hs = (.+) m$", r"hs = \1 m"),
    (r"^hr = (.+) m$", r"hr = \1 m"),
    (r"^true heights \((.+) m, (.+) m\)$", r"alturas reales (\1 m, \2 m)"),
    (r"^equivalent heights \((.+) m, (.+) m\)$",
     r"alturas equivalentes (\1 m, \2 m)"),
    ((r"^V = (.+) m/s, γ = (.+)°: the two\n"
      r"triangulations blend (.+) and (.+)$"),
     (r"V = \1 m/s, γ = \2°: las dos triangulaciones" "\n"
      r"mezclan \3 y \4")),
    (r"^peak \|Φ\| = (.+)°  ·  the turn asks for (.+)°$",
     r"|Φ| máximo = \1°  ·  el viraje pide \2°"),
    (r"^Total SEL = (.+) dB$", r"SEL total = \1 dB"),
    (r"^Mean ground plane \(a = (.+)\)$", r"Plano medio del terreno (a = \1)"),
    (r"^(.+) Hz \(φ = 0°\)$", r"\1 Hz (φ = 0°)"),
    # coupling_term_regimes annotations (baked-in computed values).
    (r"^elastic support Yk = (.+) m/\(N s\)  \(19e\)$",
     r"apoyo elástico Yk = \1 m/(N s)  (19e)"),
    (r"^matched mobilities: (.+) dB$", r"movilidades igualadas: \1 dB"),
    (r"^pump on a concrete slab: (.+) dB$",
     r"bomba sobre un forjado de hormigón: \1 dB"),
    # tapping_force_spectrum legend and annotations.
    (r"^(\d+) mm concrete slab \(under-critical, fco = (.+) Hz\)$",
     r"forjado de hormigón de \1 mm (subcrítico, fco = \2 Hz)"),
    (r"^(\d+) mm chipboard walking surface \(over-critical, fco = (.+) Hz\)$",
     r"tablero de partículas de \1 mm (supercrítico, fco = \2 Hz)"),
    (r"^f_limit = (.+) Hz$", r"f_límite = \1 Hz"),
    (r"^(.+) dB in mean square\n\((.+) N to (.+) N\)$",
     r"\1 dB en valor cuadrático medio\n(de \2 N a \3 N)"),
    # detailed_impact_paths annotations.
    (r"^floating floor fo = (.+) Hz$", r"suelo flotante fo = \1 Hz"),
    ((r"^five paths, not thirteen: only the floor is excited\n"
      r"L'n,w \(CI\) = (.+) \((.+)\) dB$"),
     ("cinco vías, no trece: solo se excita el forjado\n"
      r"L'n,w (CI) = \1 (\2) dB")),
    # structure_borne_conversion annotation (baked-in computed values).
    ((r"^\+(.+) dB = 10 lg\(24\.1/5\.0\):\n"
      r"a lighter receiver accepts more power$"),
     "+\\1 dB = 10 lg(24,1/5,0):\nun receptor más ligero acepta más potencia"),
    (r"^L_Ws,c - D_C, D_C = (.+) dB: back to L_Ws,inst$",
     r"L_Ws,c - D_C, D_C = \1 dB: de vuelta a L_Ws,inst"),
    # radiation_efficiency_panels legend and annotations.
    (r"^(.+) m x (.+) m pane$", r"vidrio de \1 m x \2 m"),
    (r"^critical frequency fc = (.+) Hz$", r"frecuencia crítica fc = \1 Hz"),
    (r"^coincidence peak: sigma = (.+)$", r"pico de coincidencia: sigma = \1"),
    ((r"^at (.+) Hz the small pane radiates (.+) times better:\n"
      r"the uncancelled edge strip is a larger fraction of it$"),
     (r"a \1 Hz el vidrio pequeño radia \2 veces mejor:\n"
      r"la franja de borde no cancelada es una fracción mayor de él")),
    # masonry_wall_ties legend entries (tie name + baked-in Table A4 stiffness).
    (r"^butterfly \((.+) MN/m\)$", r"mariposa (\1 MN/m)"),
    (r"^double triangle \((.+) MN/m\)$", r"doble triángulo (\1 MN/m)"),
    (r"^vertical twist \((.+) MN/m\)$", r"torsión vertical (\1 MN/m)"),
    # heavy_impact_sources info box (baked-in ISO 717-2 Table D.4 sum).
    (r"^unrounded sum = (\d+)\.(\d+) dB$",
     r"suma sin redondear = \1,\2 dB"),
    # experimental_sea_clf info box (baked-in input power).
    ((r"^platform driven, cylinder driven only through the joints\n"
      r"input power = (.+) W\n"
      r"coupling stays well below the damping: valid SEA$"),
     ("plataforma excitada, cilindro excitado solo a través de las uniones\n"
      "potencia inyectada = \\1 W\n"
      "el acoplamiento queda muy por debajo del amortiguamiento: SEA válido")),
    # plateau_transmission_loss info box (baked-in panel and plateau values).
    ((r"^6 mm float glass, m'' = (.+) kg/m², η = (.+)\n"
      r"plateau height 27 dB, B/A = 10 → A = (.+) Hz, B = (.+) Hz\n"
      r"identical below A; the plateau replaces the whole coincidence region$"),
     ("vidrio float de 6 mm, m'' = \\1 kg/m², η = \\2\n"
      "altura de meseta 27 dB, B/A = 10 → A = \\3 Hz, B = \\4 Hz\n"
      "idénticas por debajo de A; la meseta sustituye toda la región de "
      "coincidencia")),
    # slow_sound_absorber panel-depth annotation (baked-in wavelength ratio).
    (r"^Normal incidence, rigid backing, panel depth L = lambda/(\d+)$",
     r"Incidencia normal, respaldo rígido, profundidad del panel L = lambda/\1"),
    # psd_confidence_smoothing annotation (mathtext + baked-in numbers).
    (r"^\$n_d\$ = (\d+) averages, \$\\varepsilon_r\$ = (\d+)\.(\d+) %$",
     r"$n_d$ = \1 promedios, $\\varepsilon_r$ = \2,\3 %"),
    # multitaper_psd_confidence annotation (baked-in dof value).
    (r"^171 ms record, \$NW\$ = 4, \$\\bar\\nu\$ = (\d+)\.(\d+) equivalent dof$",
     r"registro de 171 ms, $NW$ = 4, $\\bar\\nu$ = \1,\2 g.d.l. equivalentes"),
    # window_functions_tradeoff legend entries (name + baked-in metrics).
    (r"^([a-z]+): ENBW (.+) bins, sidelobe (.+) dB$",
     r"\1: ENBW \2 bins, lóbulo lateral \3 dB"),
    (r"^alpha = (.+) at (.+) Hz$",
     r"alfa = \1 a \2 Hz"),
    (r"^TL = (.+) dB at (.+) Hz$",
     r"TL = \1 dB a \2 Hz"),
    (r"^Integrated I = (.+) LUFS$", "Integrada I = \\1 LUFS"),
    (r"^LRA = (.+) LU \(P10-P95\)$", "LRA = \\1 LU (P10-P95)"),
    (r"^f = 10 kHz, α = (.+) dB/km\npractical spreading \(R₀ = 1000 m\)$",
     "f = 10 kHz, α = \\1 dB/km\\nensanchamiento práctico (R₀ = 1000 m)"),
    (r"^f = 250 Hz, H = 50 m, medium sand\nψc = (.+)°, η = (.+) Np/rad, (.+) modes$",
     "f = 250 Hz, H = 50 m, arena media\\nψc = \\1°, η = \\2 Np/rad, \\3 modos"),
    # stoi_segment_scores legend and annotation (baked-in values).
    (r"^steady noise at 0 dB \(STOI = (\d)\.(\d+)\)$",
     r"ruido estacionario a 0 dB (STOI = \1,\2)"),
    (r"^a (\d)\.(\d+) s dropout \(STOI = (\d)\.(\d+)\)$",
     r"un corte de \1,\2 s (STOI = \3,\4)"),
    (r"^(\d+) of (\d+) frames fall 40 dB below the loudest and are discarded$",
     r"\1 de \2 tramas quedan 40 dB por debajo de la más intensa y se descartan"),
    # sii_masking_chain / sii_octave_masking_blindness (baked-in values).
    ((r"^The clause 5 chain under a low-frequency masker \(SII = (\d)\.(\d+)\)$"),
     r"La cadena del capítulo 5 con un enmascarante grave (SII = \1,\2)"),
    ((r"^at 1 kHz there is no noise in this band,\nyet Zi = (\d+) dB: the "
      r"masking is spread up\nfrom the low bands, not made here$"),
     ("a 1 kHz no hay ruido en esta banda,\ny sin embargo Zi = \\1 dB: el "
      "enmascaramiento\nsube desde las bandas graves, no se genera aquí")),
    (r"^(\d)\.(\d+) index units apart$", r"\1,\2 unidades de índice de diferencia"),
    # sti_level_dependence legend (baked-in uncorrected STI).
    (r"^without them: a flat (\d)\.(\d+)$", r"sin ellas: un \1,\2 constante"),
    (r"^T60 = (.+) s$", r"T60 = \1 s"),
    (r"^SNR = (.+) dB$", r"SNR = \1 dB"),
    # age_threshold_sex_and_spread annotations (baked-in ISO 7029 values).
    (r"^\$s_u\$ peaks at (\d+) yr \((\d+)\.(\d+) dB\)$",
     r"$s_u$ alcanza su máximo a los \1 años (\2,\3 dB)"),
    (r"^they cross at (\d+) yr$", r"se cruzan a los \1 años"),
    (r"^Low-frequency cetaceans \(AUD INJ (.+) dB\)$",
     r"Cetáceos de baja frecuencia (AUD INJ \1 dB)"),
    (r"^High-frequency cetaceans \(AUD INJ (.+) dB\)$",
     r"Cetáceos de alta frecuencia (AUD INJ \1 dB)"),
    (r"^Very high-frequency cetaceans \(AUD INJ (.+) dB\)$",
     r"Cetáceos de muy alta frecuencia (AUD INJ \1 dB)"),
    (r"^Phocid pinnipeds \(water\) \(AUD INJ (.+) dB\)$",
     r"Pinnípedos fócidos (agua) (AUD INJ \1 dB)"),
    (r"^Otariid pinnipeds \(water\) \(AUD INJ (.+) dB\)$",
     r"Pinnípedos otáridos (agua) (AUD INJ \1 dB)"),
    (r"^SL = 140, NL = 60, DI = 15, DT = 8 dB\nfigure of merit = (.+) dB$",
     "SL = 140, NL = 60, DI = 15, DT = 8 dB\\nfigura de mérito = \\1 dB"),
    (r"^SAE band \((\d+) m\)$", r"banda SAE (\1 m)"),
    (r"^source (\d+) m, receiver (.+) m, offset (\d+) m$",
     r"fuente \1 m, receptor \2 m, offset \3 m"),
    ((r"^SEL (\d+)\.(\d+) dB\(A\)  ·  EPNL (\d+)\.(\d+) EPNdB\n"
     r"level flyover, 60 kt, 150 m, 120 m sideline, grass$"),
     ("SEL \\1,\\2 dB(A)  ·  EPNL \\3,\\4 EPNdB\n"
     "sobrevuelo nivelado, 60 kt, 150 m, 120 m lateral, hierba")),
    (r"^\$L_\{ASmax\}\$ = (\d+)\.(\d+) dB\(A\)$",
     r"$L_{ASmax}$ = \1,\2 dB(A)"),
    (r"^Diffracted path \(δ = (\d+)\.(\d+) m\)$",
     r"Camino difractado (δ = \1,\2 m)"),
    (r"^Diffracted path \(δ = (\d+),(\d+) m\)$",
     r"Camino difractado (δ = \1,\2 m)"),
    (r"^(\d+) yr$", r"\1 años"),
    # anim_elastic_coincidence titles and verdicts (library values baked in).
    (r"^f = f_c/2 = (\d+) Hz, 45° incidence$",
     r"f = f_c/2 = \1 Hz, incidencia a 45°"),
    (r"^f = 2 f_c = (\d+) Hz, 45° incidence$",
     r"f = 2 f_c = \1 Hz, incidencia a 45°"),
    (r"^below f_c: the mass law holds: (.+) dB \(it predicts (.+)\)$",
     r"bajo f_c: manda la ley de masas: \1 dB (predice \2)"),
    (r"^above f_c: trace matches λ_B: (.+) dB, the mass law said (.+)$",
     r"sobre f_c: la traza iguala λ_B: \1 dB, la ley de masas decía \2"),
    (r"^coincidence_frequency: f_c = (\d+) Hz \(10 mm steel\)$",
     r"coincidence_frequency: f_c = \1 Hz (acero de 10 mm)"),
    # Weak-field display-gain notes (_gain_note); the factor is measured off
    # the field, so the number and its dB equivalent ride through.
    (r"^air below the plate drawn ×(\d+) \(\+(\d+) dB\)$",
     r"aire bajo la placa dibujado ×\1 (+\2 dB)"),
    (r"^past the wall drawn ×(\d+) \(\+(\d+) dB\)$",
     r"tras el muro dibujado ×\1 (+\2 dB)"),
    # seabed_reflection / seabed_reflection_coefficient critical-angle legend.
    (r"^Critical angle \((\d+)\.(\d+)°\)$", r"Ángulo crítico (\1,\2°)"),
    # equal_loudness_contours per-contour annotations ("20 phon" ... "90 phon").
    (r"^(\d+) phon$", r"\1 fonios"),
    (r"^total \(limit\) (.+) dB$", r"total (límite) \1 dB"),
    (r"^total \(eng\.\) (.+) dB$", r"total (ing.) \1 dB"),
    # tone_audibility decisive legend (mathtext skips the decimal-comma pass).
    (r"^decisive \$\\Delta L\$ = (\d+)\.(\d+) dB @ (\d+)\.(\d+) Hz$",
     r"decisiva $\\Delta L$ = \1,\2 dB @ \3,\4 Hz"),
    (r"^Governing  \$K_I\$ = (\d+)\.(\d+) dB$", r"Determinante  $K_I$ = \1,\2 dB"),
    # The mathtext ($R$) makes the later decimal-comma pass skip this label, so
    # convert the decimal here as part of the translation.
    (r"^Example  \$R\$ = (\d+)\.(\d+)$", r"Ejemplo  $R$ = \1,\2"),
    (r"^Octave Band: (.+) Hz$", r"Banda de octava: \1 Hz"),
    (r"^(\d+) phon$", r"\1 fonios"),
    (r"^TNR = (.+) dB\n\(criterion (.+) dB\)$", "TNR = \\1 dB\\n(criterio \\2 dB)"),
    (r"^MLS — first (\d+) of (\d+) samples$",
     r"MLS — primeras \1 de \2 muestras"),
    (r"^Measured 1/(\d+) Octave Bands$", r"Bandas de 1/\1 de octava medidas"),
    (r"^IEC target (.+) dB$", r"Objetivo IEC \1 dB"),
    (r"^([\d.]+) ms burst$", "R\u00e1faga de \\1 ms"),
    (r"^A-Weighting High-Frequency Accuracy @ fs=(\d+) kHz$",
     "Precisi\u00f3n en alta frecuencia de la ponderaci\u00f3n A @ fs=\\1 kHz"),
    (r"^Impulse Response \((.+) Hz Band\) - Transient/Stability Comparison$",
     "Respuesta al impulso (banda de \\1 Hz) \u2014 transitorio y estabilidad"),
    (r"^1 kHz narrowband - N = (.+) sone$",
     "Banda estrecha de 1 kHz - N = \\1 sonios"),
    (r"^Flat broadband 60 dB - N = (.+) sone$",
     "Banda ancha plana a 60 dB - N = \\1 sonios"),
    (r"^Pressure-intensity index\n\u03b4pI = (.+) dB$",
     "\u00cdndice presi\u00f3n-intensidad\\n\u03b4pI = \\1 dB"),
    (r"^Reference curve shifted by (.+) dB$",
     r"Curva de referencia desplazada \1 dB"),
    (r"^Sum of unfavourable deviations = (.+) dB  \(limit 32\.0 dB\)$",
     "Suma de desviaciones desfavorables = \\1 dB  (l\u00edmite 32,0 dB)"),
    (r"^Aures \(Annex B, N = (.+) sone\)$",
     r"Aures (Anexo B, N = \1 sonios)"),
    (r"^Spatial decay D2,S = (.+) dB$",
     r"Decaimiento espacial D2,S = \1 dB"),
    (r"^Zwicker \(ISO 532-1\), N = (.+) sone$",
     r"Zwicker (ISO 532-1), N = \1 sonios"),
    (r"^Moore-Glasberg \(ISO 532-2\), N = (.+) sone$",
     r"Moore-Glasberg (ISO 532-2), N = \1 sonios"),
    (r"^Sottek \(ECMA-418-2\), N = (.+) sone$",
     r"Sottek (ECMA-418-2), N = \1 sonios"),
    (r"^1 kHz tone, 60 dB \(N = (.+) sone_HMS\)$",
     r"Tono de 1 kHz, 60 dB (N = \1 sonios_HMS)"),
    (r"^Tone in noise \(T = (.+) tu_HMS\)$",
     r"Tono en ruido (T = \1 tu_HMS)"),
    (r"^Pure noise \(T = (.+) tu_HMS\)$",
     r"Ruido puro (T = \1 tu_HMS)"),
    (r"^Peak R = (.+) asper @ (.+) Hz$",
     r"Máximo R = \1 asper @ \2 Hz"),
    (r"^AM broadband noise \(closed form, 60 dB\), peak (.+) vacil$",
     r"Ruido de banda ancha AM (forma cerrada, 60 dB), máximo \1 vacil"),
    (r"^AM tone \(signal model, 70 dB\), peak (.+) vacil$",
     r"Tono AM (modelo de señal, 70 dB), máximo \1 vacil"),
    (r"^Worked example \(PA = (.+)\)$",
     r"Ejemplo resuelto (PA = \1)"),
    (r"^PA = (.+)\nwS = (.+), wFR = (.+)$",
     "PA = \\1\\nwS = \\2, wFR = \\3"),
    (r"^Short-term loudness STL \(STL peak = (.+) sone\)$",
     r"Sonoridad a corto plazo STL (STL máx = \1 sonios)"),
    (r"^Long-term loudness LTL \(LTL peak = (.+) sone\)$",
     r"Sonoridad a largo plazo LTL (LTL máx = \1 sonios)"),
    (r"^floor-(.+)$", r"suelo-\1"),
    (r"^ceiling-(.+)$", r"techo-\1"),
    (r"^facade-(.+)$", r"fachada-\1"),
    (r"^wall-(.+)$", r"tabique-\1"),
    (r"^(.+) °C, (.+) % RH$", r"\1 °C, \2 % HR"),
    # Materials: absorption rating & airflow resistance annotations
    (r"^Reference curve shifted by ([\d.]+)$",
     r"Curva de referencia desplazada \1"),
    (r"^Sum of unfavourable deviations = (.+)  \(limit 0\.10\)$",
     "Suma de desviaciones desfavorables = \\1  (límite 0,10)"),
    (r"^Absorption class (.+)  \(shape indicator: (.+)\)$",
     r"Clase de absorción \1  (indicador de forma: \2)"),
    (r"^Specific airflow resistance R_s = (.+) Pa s/m$",
     r"Resistencia específica al flujo R_s = \1 Pa s/m"),
    (r"^Airflow resistivity sigma = (.+) Pa s/m\^2$",
     r"Resistividad al flujo sigma = \1 Pa s/m^2"),
    (r"^Linear term a = (.+) Pa s/m  \(= R_s at u -> 0\)$",
     r"Término lineal a = \1 Pa s/m  (= R_s en u -> 0)"),
    # Scattering / diffusion / precision power dynamic titles (numeric d / LWA)
    (r"^Directional diffusion  d = (.+)  \(ISO 17497-2\)$",
     r"Difusión direccional  d = \1  (ISO 17497-2)"),
    (r"^Precision sound power \(ISO 3745\)  LWA = (.+) dB\(A\)$",
     r"Potencia acústica de precisión (ISO 3745)  LWA = \1 dB(A)"),
    (r"^Precision intensity scanning \(ISO 9614-3\)  LWA = (.+) dB\(A\)$",
     r"Barrido de intensidad de precisión (ISO 9614-3)  LWA = \1 dB(A)"),
    (r"^Enveloping-surface sound power \(ISO 3744\)  LWA = (.+) dB\(A\)$",
     r"Potencia acústica por superficie envolvente (ISO 3744)  LWA = \1 dB(A)"),
    (r"^Reverberation-room sound power \(ISO 3741\)  LWA = (.+) dB\(A\)$",
     r"Potencia acústica en cámara reverberante (ISO 3741)  LWA = \1 dB(A)"),
    (r"^Intensity-scanning sound power \(ISO 9614-2\)  LWA = (.+) dB\(A\)$",
     r"Potencia acústica por barrido de intensidad (ISO 9614-2)  LWA = \1 dB(A)"),
    # Human-vibration dynamic titles (numeric a_w / A(8))
    (r"^Weighted seat acceleration \(ISO 2631-1\)  (.+)$",
     r"Aceleración ponderada del asiento (ISO 2631-1)  \1"),
    (r"^Hand-arm daily exposure \(ISO 5349 / 2002-44-EC\)  (.+)$",
     r"Exposición diaria mano-brazo (ISO 5349 / 2002-44-EC)  \1"),
    # Speech intelligibility dynamic title (numeric SII)
    (r"^Speech Intelligibility Index \(ANSI S3\.5-1997\)   SII = (.+)$",
     r"Índice de inteligibilidad del habla (ANSI S3.5-1997)   SII = \1"),
    # Room-noise criteria (ANSI S12.2-2019) dynamic titles/legends
    (r"^Noise Criteria — tangency method   NC-(.+)$",
     r"Criterios de ruido — método de tangencia   NC-\1"),
    (r"^Room Criteria Mark II   RC-(.+)$",
     r"Criterios de sala Mark II   RC-\1"),
    (r"^Tangent @ (.+) Hz$", r"Tangente @ \1 Hz"),
    (r"^Reference RC-(.+)$", r"Referencia RC-\1"),
    # The NC blind spot: one rating, two spectral characters
    (r"^One rating: NC-(.+) for both rooms$",
     r"Una calificación: NC-\1 en las dos salas"),
    (r"^Duct rumble — tangent at (.+) Hz$",
     r"Retumbo de conducto — tangente en \1 Hz"),
    (r"^Diffuser hiss — tangent at (.+) Hz$",
     r"Siseo de difusor — tangente en \1 Hz"),
    (r"^Duct rumble — RC-(.+)$", r"Retumbo de conducto — RC-\1"),
    (r"^Diffuser hiss — RC-(.+)$", r"Siseo de difusor — RC-\1"),
    # ISO 18233 acquisition figures
    (r"^Pistol shot \(no deconvolution\) — (.+) dB$",
     r"Disparo de pistola (sin deconvolución) — \1 dB"),
    (r"^(.+) s sweep, deconvolved — (.+) dB$",
     r"barrido de \1 s, deconvolucionado — \2 dB"),
    (r"^sweep over pistol: \+(.+) dB\ntwo doublings of sweep length: \+(.+) dB$",
     r"barrido frente a pistola: +\1 dB\ndos duplicaciones de la duración: +\2 dB"),
    (r"^H(\d)\n−(.+) s$", r"H\1\n−\2 s"),
    (r"^excluded: r < d_min = (.+) m$", r"excluida: r < d_min = \1 m"),
    (r"^critical distance (.+) m$", r"distancia crítica \1 m"),
    (r"^Schroeder frequency (.+) Hz$", r"Frecuencia de Schroeder \1 Hz"),
    (r"^sweep, stationary$", r"barrido, estacionario"),
    (r"^MLS, stationary$", r"MLS, estacionario"),
    (r"^sweep, \+0\.3 K during the take$", r"barrido, +0,3 K durante la toma"),
    (r"^MLS, \+0\.3 K during the take$", r"MLS, +0,3 K durante la toma"),
    # ISO 3382-3 Annex A quality ranges and the absorption-per-table window
    (r"^Treated: D2,S = (.+) dB, Lp,A,S,4m = (.+) dB$",
     r"Tratada: D2,S = \1 dB, Lp,A,S,4m = \2 dB"),
    (r"^Untreated: D2,S = (.+) dB, Lp,A,S,4m = (.+) dB$",
     r"Sin tratar: D2,S = \1 dB, Lp,A,S,4m = \2 dB"),
    (r"^Treated: rD = (.+) m, rP = (.+) m$", r"Tratada: rD = \1 m, rP = \2 m"),
    (r"^Untreated: rD = (.+) m, rP = (.+) m$",
     r"Sin tratar: rD = \1 m, rP = \2 m"),
    (r"^feasible A_tab: (.+) to (.+) m²$",
     r"A_tab factible: de \1 a \2 m²"),
    (r"^r_s = (.+) m → A_tab > (.+) m²$", r"r_s = \1 m → A_tab > \2 m²"),
    (r"^r_t = (.+) m → A_tab < (.+) m²$", r"r_t = \1 m → A_tab < \2 m²"),
    (r"^window closes at r_t / r_s = (.+)$",
     r"la ventana se cierra en r_t / r_s = \1"),
    (r"^this layout: (.+), (.+) m² wide$",
     r"esta distribución: \1, \2 m² de anchura"),
    (r"^Packed tables close it \(r_s = (.+) m\)$",
     r"Con las mesas juntas se cierra (r_s = \1 m)"),
    (r"^(\d+) yr$", r"\1 años"),
    (r"^10-90 % band \((\d+) yr\)$", r"banda 10-90 % (\1 años)"),
    # Tier-1 animation dynamic labels
    (r"^remaining energy: (.+) %$", r"energía restante: \1 %"),
    (r"^On the \(2,1\) mode: (.+) Hz$", r"En el modo (2,1): \1 Hz"),
    (r"^Off mode: (.+) Hz$", r"Fuera de modo: \1 Hz"),
    (r"^(.+) sone$", r"\1 sonios"),
    (r"^mean Lp = (.+) dB$", r"Lp medio = \1 dB"),
    (r"^first notch (.+) Hz$", r"primer nulo \1 Hz"),
    # FDTD animation dynamic labels (third batch)
    (r"^Low frequency: (.+)$", r"Baja frecuencia: \1"),
    (r"^High frequency: (.+)$", r"Alta frecuencia: \1"),
    (r"^insertion loss (.+) dB$", r"pérdida por inserción \1 dB"),
    (r"^diffusion coefficient d = (.+)$", r"coeficiente de difusión d = \1"),
    (r"^design frequency (.+) Hz$", r"frecuencia de diseño \1 Hz"),
    # 2D FDTD wave simulation (public API concept figure)
    (r"^FDTD pressure field at t = (.+) ms$",
     r"Campo de presión FDTD en t = \1 ms"),
    (r"^probe \((.+)\) m$", r"sonda (\1) m"),
    # Building & structure-borne result figures (dynamic values baked in).
    (r"^Airborne: Rw\(C;Ctr\) = (.+) dB$", r"Aéreo: Rw(C;Ctr) = \1 dB"),
    (r"^Impact: Ln,w\(CI\) = (.+) dB$", r"Impacto: Ln,w(CI) = \1 dB"),
    (r"^coincidence fc = (\d+) Hz$", r"coincidencia fc = \1 Hz"),
    (r"^\$L_\{WA\}\$ = (\d+)\.(\d+) dB\(A\)$", r"$L_{WA}$ = \1,\2 dB(A)"),
    # Their multi-line info boxes are single Text artists, so the whole
    # joined string is matched at once (values stay as capture groups).
    ((r"^Rw\(C;Ctr\) = (.+)\nC50-5000 = (.+)\n"
      r"rating on the core bands, terms on the full range$"),
     ("Rw(C;Ctr) = \\1\nC50-5000 = \\2\n"
      "índice en las bandas básicas, términos en el rango completo")),
    ((r"^Dls,2m,nT,w\(C;Ctr\) = (.+) dB\n45° loudspeaker method "
      r"\(-1\.5 dB on R'\)$"),
     "Dls,2m,nT,w(C;Ctr) = \\1 dB\nmétodo del altavoz a 45° (-1,5 dB en R')"),
    ((r"^L'nT,w\(CI\) = (.+) dB\nnote the minus sign: a live room lowers "
      r"L'nT$"),
     "L'nT,w(CI) = \\1 dB\natención al signo menos: un recinto vivo reduce L'nT"),
    ((r"^LW = Lp,in \+ Cd - R' \+ 10 log10\(S/S0\)\nwall 176 m² \+ industrial "
      r"door 24 m², Cd = -5 dB$"),
     ("LW = Lp,in + Cd - R' + 10 log10(S/S0)\n"
      "muro de 176 m² + puerta industrial de 24 m², Cd = -5 dB")),
    ((r"^Rw\(C;Ctr\) = (.+) dB\n6 mm float glass, m'' = 15 kg/m², "
      r"η = 0\.024$"),
     "Rw(C;Ctr) = \\1 dB\nvidrio flotado de 6 mm, m'' = 15 kg/m², η = 0,024"),
    ((r"^Kij = 10 log10\(1/τ̄\) \+ 5 log10\(fc2/1000\)\nconcrete, plate 1 fixed "
      r"at 100 mm$"),
     "Kij = 10 log10(1/τ̄) + 5 log10(fc2/1000)\nhormigón, placa 1 fija en 100 mm"),
    ((r"^below f0: stiffness-controlled, \|Y\| ~ ω/k\n"
      r"above f0: mass-controlled, \|Y\| ~ 1/\(ωm\)\n"
      r"f0 = (.+) Hz,  1/c = (.+) m/\(N·s\)$"),
     ("por debajo de f0: dominio de la rigidez, |Y| ~ ω/k\n"
      "por encima de f0: dominio de la masa, |Y| ~ 1/(ωm)\n"
      "f0 = \\1 Hz,  1/c = \\2 m/(N·s)")),
    # Programme-loudness .plot() legend lines (loudness_gating/loudness_range).
    (r"^Integrated (.+) LUFS$", r"Integrada \1 LUFS"),
    (r"^Ungated mean (.+) LUFS$", r"Media sin puerta \1 LUFS"),
    # Core-metrology figures: dynamic verdict / error strings
    (r"^r = (\d+) runs, accept \((\d+), (\d+)\]: trend-free$",
     r"r = \1 rachas, aceptación (\2, \3]: sin tendencia"),
    (r"^r = (\d+) runs, accept \((\d+), (\d+)\]: rejected$",
     r"r = \1 rachas, aceptación (\2, \3]: rechazada"),
    (r"^max \|recovered - true\| = (.+)$",
     r"máx |recuperada - verdadera| = \1"),
    (r"^estimated delay removed: (.+) samples$",
     r"retardo estimado eliminado: \1 muestras"),
    # FDTD second-batch clips: baked-number pills and titles
    (r"^slit h = (.+) mm$", r"rendija h = \1 mm"),
    (r"^Pass band: (\d+) Hz, kL = π$", r"Banda de paso: \1 Hz, kL = π"),
    (r"^Stop band peak: (\d+) Hz, kL = π/2$",
     r"Pico de rechazo: \1 Hz, kL = π/2"),
    (r"^Slit w = (\d+) mm \(λ/20\)$", r"Rendija w = \1 mm (λ/20)"),
    (r"^Opening w = (.+) m \(= λ\)$", r"Hueco w = \1 m (= λ)"),
    (r"^slit τ = (.+) \(Gomperts\)$", r"rendija τ = \1 (Gomperts)"),
    (r"^f = (\d+) Hz \(λ = (.+) m\)$", r"f = \1 Hz (λ = \2 m)"),
    (r"^shadow beyond ≈ (\d+) m \(ray model\)$",
     r"sombra más allá de ≈ \1 m (modelo de rayos)"),
    # cnossos_road_speed_law rolling/propulsion crossover annotation.
    (r"^crossover (\d+) km/h$", r"cruce \1 km/h"),
    # orthotropic_transmission_loss info box (library values baked in).
    ((r"^1 mm steel sheet, m'' = (.+) kg/m², flat fc = (.+) kHz\n"
      r"corrugated H = (\d+) mm, L = (\d+) mm, m'' = (.+) kg/m², "
      r"fc1 = (.+) Hz, fc2 = (.+) kHz\n"
      r"worst penalty (.+) dB at (.+) Hz, for a stiffer and only 9 % "
      r"heavier panel$"),
     ("chapa de acero de 1 mm, m'' = \\1 kg/m², fc plana = \\2 kHz\n"
      "grecada H = \\3 mm, L = \\4 mm, m'' = \\5 kg/m², "
      "fc1 = \\6 Hz, fc2 = \\7 kHz\n"
      "penalización máxima \\8 dB a \\9 Hz, con un panel más rígido y solo "
      "un 9 % más pesado")),
    # limp_frame_effective_density annotations (library values baked in).
    (r"^apparent total density rho_t/rho0 = (.+)$",
     r"densidad total aparente rho_t/rho0 = \1"),
    (r"^decoupling frequency (.+) Hz$", r"frecuencia de desacoplo \1 Hz"),
    # biot_frame_resonance annotation (library value baked in).
    (r"^frame quarter-wave resonance (.+) Hz$",
     r"resonancia de cuarto de onda del esqueleto \1 Hz"),
    ((r"^Soft fibrous layer: porosity (.+), flow resistivity (.+) kPa s/m², "
      r"frame density (.+) kg/m³$"),
     ("Capa fibrosa blanda: porosidad \\1, resistividad al flujo \\2 kPa s/m², "
      "densidad del esqueleto \\3 kg/m³")),
]


def set_lang(lang: str) -> None:
    """Switch the output language ('en' or 'es')."""
    global _LANG, _LANG_SUFFIX
    _LANG = lang
    _LANG_SUFFIX = "" if lang == "en" else f"_{lang}"
    _publish(_LANG=_LANG, _LANG_SUFFIX=_LANG_SUFFIX)


def _translate_figure(fig: Any) -> None:
    """Rewrite every Text artist of *fig* into the active language."""
    import re as _re

    import matplotlib.text as _mtext

    if _LANG == "en":
        return
    import re as _re2

    from matplotlib.ticker import FixedFormatter as _FxF
    from matplotlib.ticker import FuncFormatter as _FF
    from matplotlib.ticker import ScalarFormatter as _SF

    def _comma(s: str) -> str:
        # A letter immediately before the number marks a standard designation
        # (e.g. "S3.5"), not a decimal - leave those untouched.
        return _re2.sub(r"(?<![\d.A-Za-z])(\d+)\.(\d+)(?![.\d])", r"\1,\2", s)

    def _tr_words(s: str) -> str:
        """Apply the exact / pattern lookups (no decimal comma) to *s*."""
        if s in _ES_EXACT:
            return _ES_EXACT[s]
        for pat, repl in _ES_PATTERNS:
            new, n = _re.subn(pat, repl, s)
            if n:
                return new
        return s

    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            fmt = axis.get_major_formatter()
            if isinstance(fmt, _FxF):
                # Translate categorical tick labels (e.g. path names) too;
                # numeric labels match nothing and only get the decimal comma.
                fmt.seq = [_comma(_tr_words(s)) for s in fmt.seq]
            elif isinstance(fmt, _FF) and not getattr(fmt, "_phonometry_comma", False):
                # Categorical labels (set_xticklabels installs a FuncFormatter)
                # need the word lookups too; numeric labels are untouched.
                wrapped = _FF(
                    lambda v, pos, _f=fmt: _comma(_tr_words(str(_f(v, pos))))
                )
                wrapped._phonometry_comma = True  # type: ignore[attr-defined]
                axis.set_major_formatter(wrapped)
            elif type(fmt) is _SF and axis.get_scale() == "linear":
                wrapped = _FF(lambda v, pos: _comma(f"{v:g}"))
                axis.set_major_formatter(wrapped)
    for artist in fig.findobj(_mtext.Text):
        s = artist.get_text()
        if not s:
            continue
        if s in _ES_EXACT:
            artist.set_text(_ES_EXACT[s])
        else:
            for pat, repl in _ES_PATTERNS:
                new, n = _re.subn(pat, repl, s)
                if n:
                    artist.set_text(new)
                    break
        # Spanish decimal comma, applied uniformly to every text artist
        # (tick labels included) except mathtext. The substitution itself is
        # conservative -- it only rewrites a bare ``digit.digit`` not adjacent
        # to further digits/dots -- so underscore-bearing unit tokens such as
        # ``sone_HMS`` / ``tu_HMS`` keep their identifier intact while genuine
        # decimals in the same label (e.g. ``8.0 sone_HMS``) still get commas.
        s = artist.get_text()
        if s and "$" not in s and _re.search(r"\d\.\d", s):
            # Clause/version numbers like 5.3.3 and standard designations like
            # "S3.5" (a letter immediately before the number) keep their dots.
            artist.set_text(
                _re.sub(r"(?<![\d.A-Za-z])(\d+)\.(\d+)(?![.\d])", r"\1,\2", s)
            )
