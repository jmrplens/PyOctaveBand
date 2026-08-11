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

import os
import re
import sys
from typing import Any

from . import _publish

# The miss recorder sits at the top of ``scripts/``, next to the checker that
# reads what it writes; the figures package is a subdirectory of the same
# place. Guard the path the way the other cross-imports in ``scripts/`` do
# (see check_figures.py), so an editor importing a single figure module also
# resolves it.
_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import figure_language_audit as _audit

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
        "s' instalada = s't + s'a (apartado 8.2)",
    "clause 8.2:   r >= 100 kPa.s/m2 -> s' = s't\n"
    "              10 <= r < 100     -> s' = s't + s'a\n"
    "              r < 10            -> s' = s't only if s'a is negligible":
        "apartado 8.2:  r >= 100 kPa.s/m2 -> s' = s't\n"
        "               10 <= r < 100     -> s' = s't + s'a\n"
        "               r < 10            -> s' = s't solo si s'a es "
        "despreciable",
    "the worked determination:\nd = 20 mm, 4.94 + 5.56 = 10.49":
        "la determinación del ejemplo:\nd = 20 mm, 4,94 + 5,56 = 10,49",
    "f_u = 0.58 c0 / d (Clause 5.4)": "f_u = 0,58 c0 / d (apartado 5.4)",
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
    "Adrienne window: 0.5 + 5 + 5 ms = 10.5 ms":
        "Ventana de Adrienne: 0,5 + 5 + 5 ms = 10,5 ms",
    "1/T = 95 Hz:\nthe window's own\nlow-frequency limit":
        "1/T = 95 Hz:\nel límite en baja frecuencia\nde la propia ventana",
    "below the reported 250 Hz": "por debajo de los 250 Hz declarados",
    "parasitic arrival, just outside\nthe gate: lengthen the window\n"
    "and it comes in with the road":
        "llegada parásita justo fuera de\nla puerta: si se alarga la ventana,\n"
        "entra con el pavimento",
    "315-1600 Hz: the interval in which\nPart 2 expects the two to agree":
        "315-1600 Hz: el intervalo en el que\nla Parte 2 espera que ambos "
        "coincidan",
    "the 100 mm bore of the worked\nexample: 1989 Hz":
        "el diámetro de 100 mm del\nejemplo resuelto: 1989 Hz",
    "1800 Hz, the top edge of the 1600 Hz band:\n"
    "a bore above 111 mm no longer covers it":
        "1800 Hz, el borde superior de la banda de 1600 Hz:\n"
        "un diámetro de más de 111 mm ya no la cubre",
    "Periodic, 6 x N = 7": "Periódico, 6 x N = 7",
    "Modulated, period + inverse": "Modulado, periodo + inverso",
    "Periodic, 6 x N = 7  (d = 0.22)": "Periódico, 6 x N = 7  (d = 0,22)",
    "Modulated, period + inverse  (d = 0.32)":
        "Modulado, periodo + inverso  (d = 0,32)",
    "f0: the one band the\nperiodic array wins":
        "f0: la única banda en la que\ngana la disposición periódica",
    "small excitation: fr = 25.0 Hz":
        "excitación pequeña: fr = 25,0 Hz",
    "over-driven: peak lower and at 22 Hz":
        "sobreexcitado: pico más bajo y en 22 Hz",
    "this peak is the whole measurement\n(Formula 4), extrapolated to F -> 0":
        "este pico es toda la medición\n(Fórmula 4), extrapolado a F -> 0",
    "amplification below\nsqrt(2) f0 = 67 Hz":
        "amplificación por debajo de\nsqrt(2) f0 = 67 Hz",
    "30 lg(f/f0): 30.8 dB\nagainst 35.6 dB at 500 Hz":
        "30 lg(f/f0): 30,8 dB\nfrente a 35,6 dB a 500 Hz",
    "alpha_spec - alpha_s  (numerator of Eq. (5))":
        "alpha_spec - alpha_s  (numerador de la Ec. (5))",
    "alpha_spec, rotating turntable (T3, T4)":
        "alpha_spec, plataforma giratoria (T3, T4)",
    "alpha_s, static turntable (T1, T2)":
        "alpha_s, plataforma estática (T1, T2)",
    "Absorption coefficient": "Coeficiente de absorción",
    "s = (alpha_spec - alpha_s) / (1 - alpha_s)   Eq. (5)":
        "s = (alpha_spec - alpha_s) / (1 - alpha_s)   Ec. (5)",
    "From three impulse responses to one level (ISO 17497-2, Clause 7.4)":
        "De tres respuestas al impulso a un nivel (ISO 17497-2, apartado 7.4)",
    "(a) h1: sample present": "(a) h1: con la muestra",
    "(b) h2: sample removed": "(b) h2: sin la muestra",
    "(c) h1 - h2: the room is gone": "(c) h1 - h2: la sala desaparece",
    # Measured on the drawn figure: the previous wording ("deconvolucionada
    # por h3, Fórmula (1)") ran 38 px past the left canvas edge -- these row
    # labels are laid out right-aligned against the English lengths.
    "(d) h4: deconvolved by h3, Formula (1)":
        "(d) h4: deconvolución por h3 (Fórm. 1)",
    "(e) windowed, Clause 7.4.3": "(e) enventanada, apartado 7.4.3",
    "room": "sala",
    "h4 arrives 2 ms earlier than h1:\ndividing by h3 removes its delay too":
        "h4 llega 2 ms antes que h1:\ndividir por h3 quita también su retardo",
    "shortest path\n40.0 ms": "camino más corto\n40,0 ms",
    "longest path\n48.9 ms": "camino más largo\n48,9 ms",
    "source 10 m, arc 5 m, receiver 60 deg\n"
    "window 9.7 ms  ->  analysis from about 103 Hz\n"
    "S/N >= 40 dB inside it on the flat reference":
        "fuente a 10 m, arco de 5 m, receptor a 60 grados\n"
        "ventana de 9,7 ms  ->  análisis desde unos 103 Hz\n"
        "S/N >= 40 dB dentro de ella sobre la referencia plana",
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
    "Slit index n": "Índice de ranura n",
    "within 10 deg of the target": "dentro de 10 grados del objetivo",
    "slit 1": "ranura 1",
    "slit 2": "ranura 2",
    "slit 3": "ranura 3",
    "slit 4": "ranura 4",
    "slit 5": "ranura 5",
    "design frequency:\nface average 0.032":
        "frecuencia de diseño:\nmedia en la cara 0,032",
    "tuned here:\n0.32 against 0.32": "sintonizado aquí:\n0,32 frente a 0,32",
    "below c/L = 980 Hz no grating\nlobe exists: neither panel can\n"
    "beat the flat reference":
        "por debajo de c/L = 980 Hz no hay\nlóbulo de red: ningún panel puede\n"
        "superar la referencia plana",
    "N f0 = 3500 Hz: every well\nback in phase, flat again":
        "N f0 = 3500 Hz: todos los pozos\nde nuevo en fase, plano otra vez",
    "f_max = c/(2w) = 1715 Hz:\nthe well stops being\na single-mode waveguide":
        "f_max = c/(2w) = 1715 Hz:\nel pozo deja de ser una\n"
        "guía de onda de un solo modo",
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
    # The insitu_absorption info box is symbols only (Kr, alpha, Hr, Hi and
    # a fraction with no decimal separator): it reads the same in Spanish.
    "Kr = 2/3\nalpha = 1 - (1/Kr^2)|Hr/Hi|^2":
        "Kr = 2/3\nalpha = 1 - (1/Kr^2)|Hr/Hi|^2",
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
    # Bar-group tick label, symbols only: reads the same in Spanish.
    "Rw + Ctr": "Rw + Ctr",
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
    "at T = 5 s the energy average has grown to 10.0 dB\n"
    "while the Fast term has saturated at 5.1 dB":
        "a T = 5 s el promedio energético ha crecido hasta 10,0 dB\n"
        "mientras el término Fast se ha saturado en 5,1 dB",
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
    # machine_fault_families: the three families of Norton 8.4 as patterns.
    # The drawn line names (2xGMF, fsh, lobe n=1 m=2...) are the keys of the
    # FaultFrequencyResult and stay in English by design -- see
    # ENGLISH_BY_DESIGN in scripts/check_figure_language.py.
    "Fault Families Are Recognised by Their Pattern":
        "Las familias de fallos se reconocen por su patrón",
    "Localised fault: one chipped tooth":
        "Fallo localizado: un solo diente astillado",
    "Distributed wear: every tooth": "Desgaste distribuido: todos los dientes",
    "Induction motor: 1x, 2x, 2fe and the rotor-slot family":
        "Motor de inducción: 1x, 2x, 2fe y la familia de ranura de rotor",
    # "rotating" dropped for width: both lobe annotations already say
    # "gira a ... Hz", and the full title overflowed the figure by 34 px.
    "Ducted fan: the blade rate and its rotating lobe patterns":
        "Ventilador entubado: el paso de pala y sus patrones lobulados",
    "Spectrum amplitude": "Amplitud del espectro",
    "vibration spectrum": "espectro de vibración",
    # The family legend words, worded exactly as _plot/vibration.py already
    # translates them for language="es" callers.
    "shaft": "eje",
    "gear": "engranaje",
    "motor": "motor",
    "blade": "álabe",
    "sidebands at ± fs = 25 Hz:\nlow and flat":
        "bandas laterales a ± fs = 25 Hz:\nbajas y planas",
    "sidebands at ± fs = 25 Hz:\ntall groups, and the\nhigher harmonics lift":
        "bandas laterales a ± fs = 25 Hz:\ngrupos altos, y los armónicos\n"
        "superiores se levantan",
    "rotor-slot harmonic\nwith ± fs sidebands:\nthe spacing is the diagnosis":
        "armónico de ranura de rotor\ncon bandas laterales a ± fs:\n"
        "el espaciado es el diagnóstico",
    "mL = 10 turns at 35 Hz,\nbelow the shaft: weak":
        "mL = 10 gira a 35 Hz,\npor debajo del eje: débil",
    "mL = 2 turns at 175 Hz,\n3x the shaft speed: strong":
        "mL = 2 gira a 175 Hz,\n3 veces la velocidad del eje: fuerte",
    # envelope_chain_steps: the envelope route made visible, step by step.
    "What Each Step of the Envelope Route Does to the Signal":
        "Qué le hace cada paso de la ruta de la envolvente a la señal",
    "1. As recorded: noise and unbalance, no visible impacts":
        "1. Tal como se registra: ruido y desequilibrio, sin impactos "
        "visibles",
    "2. Band-passed on the 3 kHz housing resonance: the impact train":
        "2. Filtrado en la banda de la resonancia del soporte de 3 kHz: "
        "el tren de impactos",
    "3. Hilbert envelope: one pulse per impact":
        "3. Envolvente de Hilbert: un pulso por impacto",
    "4. Its spectrum: the period has become a line at BPFO":
        "4. Su espectro: el periodo se ha convertido en una línea en la BPFO",
    "raw record": "registro bruto",
    "2-4 kHz band": "banda 2-4 kHz",
    "envelope": "envolvente",
    "envelope spectrum": "espectro de envolvente",
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
    "Rotorcraft Diffraction Attenuation vs Path Difference "
    "(ECAC Doc 32 / NORAH2)":
        "Atenuación por difracción de rotorcraft frente a la diferencia de "
        "camino (ECAC Doc 32 / NORAH2)",
    "Below line of sight": "Por debajo de la línea de visión",
    "25 dB cap (§A.4.5)": "tope de 25 dB (§A.4.5)",
    "Diffraction attenuation ΔLd [dB]": "Atenuación por difracción ΔLd [dB]",
    "Path difference δ [m]": "Diferencia de camino δ [m]",
    "10 Ch lg 3 at grazing incidence (δ = 0):\n"
    "4.8 dB where Ch = 1, 3.0 dB at 63 Hz":
        "10 Ch lg 3 en incidencia rasante (δ = 0):\n"
        "4,8 dB donde Ch = 1, 3,0 dB en 63 Hz",
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
    "+11.5 dB @ 3.15 kHz": "+11,5 dB @ 3,15 kHz",
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
    # intensity_element_insulation info box: rating and formula, symbols only.
    "DI,n,e,w(C;Ctr) = 29(-1;-2) dB\n"
    "DI,n,e = Lp1 - 6 - [LIn + 10 log10(Sm/A0)] + 10 log10 N":
        "DI,n,e,w(C;Ctr) = 29(-1;-2) dB\n"
        "DI,n,e = Lp1 - 6 - [LIn + 10 log10(Sm/A0)] + 10 log10 N",
    # survey_insulation figure (ISO 10052)
    "ISO 10052 Survey Method: Reverberation-Index Correction":
        "Método de control ISO 10052: corrección por índice de reverberación",
    "Level difference [dB]": "Diferencia de nivel [dB]",
    "D (level difference)": "D (diferencia de nivel)",
    "DnT (standardized)": "DnT (estandarizada)",
    "octave bands, T0 = 0.5 s": "bandas de octava, T0 = 0,5 s",
    "DnT,w = 49 dB  (C = -1)\noctave bands, T0 = 0.5 s":
        "DnT,w = 49 dB  (C = -1)\nbandas de octava, T0 = 0,5 s",
    # absorption_uncertainty figure (ISO 12999-2)
    "ISO 12999-2 Sound Absorption Coefficient Uncertainty":
        "Incertidumbre del coeficiente de absorción acústica (ISO 12999-2)",
    "+/-U (k = 2), reproducibility": "±U (k = 2), reproducibilidad",
    "alpha_s (ISO 354)": "alpha_s (ISO 354)",
    "sigma_R = m alpha_s + n  (Table 1)\nU = k u,  k = 2  (95 %)":
        "sigma_R = m alpha_s + n  (Tabla 1)\nU = k u,  k = 2  (95 %)",
    # floor_covering_improvement figure (ISO 16251-1)
    "ISO 16251-1 Floor-Covering Impact Sound Improvement":
        "Mejora a impacto de revestimientos de suelo (ISO 16251-1)",
    "Improvement of impact sound insulation [dB]":
        "Mejora del aislamiento a impactos [dB]",
    "delta-L (improvement)": "delta-L (mejora)",
    "delta-Lw = 29 dB  (ISO 717-2)\none-third octave, mock-up (a0 = 1e-6 m/s^2)":
        "delta-Lw = 29 dB  (ISO 717-2)\n"
        "tercios de octava, maqueta (a0 = 1e-6 m/s^2)",
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
    # Rating line, symbols only: reads the same in Spanish.
    "LiA,Fmax = 55 dB": "LiA,Fmax = 55 dB",
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
    # "plenum" is the Spanish word too (see the block title above).
    "Rcl, plenum h = 0.43 m": "Rcl, plenum h = 0,43 m",
    "Rcl, plenum h = 0.86 m": "Rcl, plenum h = 0,86 m",
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
    # Resonance markers, symbols only: read the same in Spanish.
    "fmsm = 26 Hz": "fmsm = 26 Hz",
    "fmsm = 50 Hz": "fmsm = 50 Hz",
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
    "rigid T-junction, two heavy walls\nlij = 4 m, Si = 12 m^2, Sj = 10 m^2\n"
    "Formula (13), one-third octave\nmean Kij = 9.5 dB":
        "unión en T rígida, dos muros pesados\n"
        "lij = 4 m, Si = 12 m^2, Sj = 10 m^2\n"
        "Fórmula (13), tercios de octava\nKij medio = 9,5 dB",
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
    # The model names are proper names (also drawn by
    # reverberation_model_absorption): looked at, and the same in Spanish.
    "Sabine": "Sabine",
    "Eyring": "Eyring",
    "Millington-Sette": "Millington-Sette",
    "Fitzroy": "Fitzroy",
    "Arau-Puchades": "Arau-Puchades",
    "room 10 x 7 x 3.5 m\nV = 245 m^3, S = 259 m^2\n"
    "anisotropic: absorptive floor/ceiling\nc0 = 343 m/s, air at 20 C / 50 % RH":
        "sala de 10 x 7 x 3,5 m\nV = 245 m^3, S = 259 m^2\n"
        "anisótropa: suelo y techo absorbentes\n"
        "c0 = 343 m/s, aire a 20 °C / 50 % HR",
    # dynamic_stiffness figure (EN 29052-1)
    "EN 29052-1 Floating-Floor Resonance":
        "Resonancia de suelo flotante EN 29052-1",
    r"Dynamic stiffness per unit area $s'$ [MN/m³]":
        r"Rigidez dinámica por unidad de área $s'$ [MN/m³]",
    r"Natural frequency $f_0$ [Hz]": r"Frecuencia natural $f_0$ [Hz]",
    "design point": "punto de diseño",
    "design point (10 MN/m^3, 46 Hz)": "punto de diseño (10 MN/m^3, 46 Hz)",
    "f0 = (1/2pi) sqrt(s'/m')  (Formula 2)\n"
    "s'  = s't + s'a  (clause 8.2)\n"
    "s't = 4 pi^2 m't fr^2  (Formula 4)\n"
    "s'a = p0/(d eps) ~ 111/d MN/m^3  (NOTE)":
        "f0 = (1/2pi) sqrt(s'/m')  (Fórmula 2)\n"
        "s'  = s't + s'a  (apartado 8.2)\n"
        "s't = 4 pi^2 m't fr^2  (Fórmula 4)\n"
        "s'a = p0/(d eps) ~ 111/d MN/m^3  (NOTA)",
    # junction_transmission figure (Hopkins 5.2.1.3, Cremer/Craik)
    "Bending-wave transmission at a rigid X-junction (Hopkins 5.2.1.3)":
        "Transmisión de onda de flexión en una unión X rígida (Hopkins 5.2.1.3)",
    "Incidence angle [degrees]": "Ángulo de incidencia [grados]",
    r"Transmission coefficient $\tau$": r"Coeficiente de transmisión $\tau$",
    r"corner $\tau_{12}(\theta)$": r"esquina $\tau_{12}(\theta)$",
    r"straight $\tau_{13}(\theta)$": r"recta $\tau_{13}(\theta)$",
    "corner average": "media esquina",
    "straight average": "media recta",
    "X-junction: 100 mm / 200 mm concrete\n"
    "chi = 0.707,  psi = 4.000\n"
    "theta_co = arcsin(chi) = 45.0 deg\n"
    "corner avg = 0.0331\n"
    "straight avg = 0.0072":
        "unión en X: hormigón de 100 mm / 200 mm\n"
        "chi = 0,707,  psi = 4,000\n"
        "theta_co = arcsin(chi) = 45,0 grados\n"
        "media esquina = 0,0331\n"
        "media recta = 0,0072",
    r"$\theta_\mathrm{co} = 45°$: $\tau_{12} = 0$ beyond it":
        r"$\theta_\mathrm{co} = 45°$: $\tau_{12} = 0$ más allá",
    r"cut-off $\theta_\mathrm{co} = \arcsin\chi$":
        r"corte $\theta_\mathrm{co} = \arcsin\chi$",
    # mechanical_mobility figure (ISO 7626-1)
    "ISO 7626-1 Mechanical Mobility FRFs":
        "FRF de movilidad mecánica ISO 7626-1",
    "Normalized FRF magnitude": "Magnitud FRF normalizada",
    "Receptance $|H|$ (× k)": "Receptancia $|H|$ (× k)",
    r"Mobility $|Y|$ (× k/$\omega_0$)": r"Movilidad $|Y|$ (× k/$\omega_0$)",
    r"Accelerance $|A|$ (× k/$\omega_0^2$)":
        r"Acelerancia $|A|$ (× k/$\omega_0^2$)",
    "resonance $f_0$": "resonancia $f_0$",
    "SDOF: m = 2 kg, k = 8000 N/m, c = 5 N.s/m\n"
    "H = 1/(k - w^2 m + j w c)\n"
    "Y = j w H,   A = -w^2 H  (Table 1)\n"
    "f0 = 10.1 Hz,  |Y(f0)| = 1/c":
        "SDOF: m = 2 kg, k = 8000 N/m, c = 5 N.s/m\n"
        "H = 1/(k - w^2 m + j w c)\n"
        "Y = j w H,   A = -w^2 H  (Tabla 1)\n"
        "f0 = 10,1 Hz,  |Y(f0)| = 1/c",
    # mobility_result_lines figure (ISO 7626-1 / ISO 7626-2 A.4)
    "0° at the resonance: $Y$ is real, $|Y| = 1/c$":
        "0° en la resonancia: $Y$ es real, $|Y| = 1/c$",
    "phase of $Y(f)$": "fase de $Y(f)$",
    "Phase [degrees]": "Fase [grados]",
    "a driving-point FRF never leaves ±90°\n(ISO 7626-2, A.4)":
        "una FRF en el punto de excitación nunca sale de ±90°\n"
        "(ISO 7626-2, A.4)",
    # infinite_mobilities figure (Cremer Table 5.1)
    "Point Mobilities of Infinite Structures (Cremer Table 5.1)":
        "Movilidades puntuales de estructuras infinitas (Cremer, tabla 5.1)",
    "SDOF resonator (section 2)": "resonador SDOF (sección 2)",
    "infinite plate, 140 mm concrete": "placa infinita, hormigón de 140 mm",
    "infinite beam, 100 × 200 mm steel":
        "viga infinita, acero de 100 × 200 mm",
    "steel strut, longitudinal": "puntal de acero, longitudinal",
    "plate:  $Y = 1/(8\\sqrt{B' m''})$, real and flat\n"
    "beam:   $Y = (1-\\mathrm{j})/(4 m' c_B)$, $\\propto f^{-1/2}$\n"
    "rod:    $Y = 1/(\\rho c_L S)$, real and flat\n"
    "plate |Y| = 2.55e-06 m/(N·s)":
        "placa: $Y = 1/(8\\sqrt{B' m''})$, real y plana\n"
        "viga:  $Y = (1-\\mathrm{j})/(4 m' c_B)$, $\\propto f^{-1/2}$\n"
        "barra: $Y = 1/(\\rho c_L S)$, real y plana\n"
        "placa |Y| = 2,55e-06 m/(N·s)",
    # mobility_random_error figure (ISO 7626-2 Annex A)
    "How Many Averages the 5 % Criterion Costs (ISO 7626-2, Annex A)":
        "Cuántos promedios cuesta el criterio del 5 % (ISO 7626-2, Anexo A)",
    "Number of averaged spectra $n$": "Número de espectros promediados $n$",
    r"Normalized random error $\varepsilon$ [%]":
        r"Error aleatorio normalizado $\varepsilon$ [%]",
    "the Annex A criterion, 5 %": "el criterio del Anexo A, 5 %",
    "the standard's own example: $\\gamma^2 = 0.8$, n = 75 → 4.08 %":
        "el ejemplo de la propia norma: $\\gamma^2 = 0{,}8$, n = 75 → 4,08 %",
    "the marked n is what each coherence needs to reach 5 %: 11 averages "
    "at 0.95, 200 at 0.5.\nFixing the measurement is cheaper than averaging "
    "through it.":
        "la n marcada es la que cada coherencia necesita para llegar al "
        "5 %: 11 promedios a 0,95, 200 a 0,5.\nArreglar la medición sale "
        "más barato que compensarla promediando.",
    # transfer_stiffness figure (ISO 10846)
    "ISO 10846 Dynamic Transfer Stiffness":
        "Rigidez dinámica de transferencia ISO 10846",
    r"Transfer stiffness level $L_k$ [dB re 1 N/m]":
        r"Nivel de rigidez de transferencia $L_k$ [dB re 1 N/m]",
    r"true $L_k$ of $k_{2,1}=k+j\omega c$":
        r"$L_k$ real de $k_{2,1}=k+j\omega c$",
    r"indirect method $-(2\pi f)^2 m_2 T$":
        r"método indirecto $-(2\pi f)^2 m_2 T$",
    r"$L_{k,\mathrm{ind}} - L_{k,\mathrm{true}}$ [dB]":
        r"$L_{k,\mathrm{ind}} - L_{k,\mathrm{real}}$ [dB]",
    r"transmissibility $|T|$": r"transmisibilidad $|T|$",
    r"loss factor $\eta = \mathrm{Im}(k_{2,1})/\mathrm{Re}(k_{2,1})$":
        r"factor de pérdidas $\eta = \mathrm{Im}(k_{2,1})/\mathrm{Re}(k_{2,1})$",
    "the ±1 dB the criterion buys": "el ±1 dB que garantiza el criterio",
    "Kelvin-Voigt makes η rise with frequency; real\n"
    "elastomers are far flatter, so this is a model, not a material":
        "Kelvin-Voigt hace subir η con la frecuencia; los elastómeros\n"
        "reales son mucho más planos: esto es un modelo, no un material",
    "Kelvin-Voigt: k = 1 MN/m, c = 120 N.s/m\n"
    "blocking mass m2 = 8 kg,  f0 = 56.3 Hz\n"
    "|T| falls to 0.1 at 189 Hz\n"
    "shaded: Inequality (2) not met -> no result":
        "Kelvin-Voigt: k = 1 MN/m, c = 120 N.s/m\n"
        "masa de bloqueo m2 = 8 kg,  f0 = 56,3 Hz\n"
        "|T| cae a 0,1 a 189 Hz\n"
        "sombreado: Desigualdad (2) sin cumplir -> sin resultado",
    # rigid_mass_calibration figure (ISO 7626-2, 7.5.2)
    "ISO 7626-2 Rigid-Mass Calibration Check":
        "Verificación de calibración con masa rígida ISO 7626-2",
    "Accelerance $|A|$ [1/kg]": "Acelerancia $|A|$ [1/kg]",
    "Deviation [%]": "Desviación [%]",
    r"expected $|A| = 1/m$": r"esperado $|A| = 1/m$",
    r"$\pm$5 % tolerance band": r"banda de tolerancia $\pm$5 %",
    "within tolerance": "dentro de tolerancia",
    "out of tolerance": "fuera de tolerancia",
    "calibration block m = 10 kg\n"
    "|A| = 1/m = 0.100 1/kg  (7.5.2)\n"
    "criterion: agree within +/- 5 %\n"
    "high-f drift -> attachment error":
        "bloque de calibración m = 10 kg\n"
        "|A| = 1/m = 0,100 1/kg  (7.5.2)\n"
        "criterio: coincidir dentro de +/- 5 %\n"
        "deriva en alta f -> error de fijación",
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
    "LWs = 10 log10(2 pi f eta m S) + Lv - 60 dB\n"
    "eta = 2.2/(f Ts),  v0 = 1 nm/s\nreception-plate method (clause 7)":
        "LWs = 10 log10(2 pi f eta m S) + Lv - 60 dB\n"
        "eta = 2,2/(f Ts),  v0 = 1 nm/s\n"
        "método de la placa receptora (apartado 7)",
    # installed_structure_borne figure (EN 12354-5)
    "EN 12354-5 Installed Structure-Borne Sound":
        "Ruido estructural instalado EN 12354-5",
    r"characteristic $L_{Ws,c}$ (EN 15657)":
        r"característica $L_{Ws,c}$ (EN 15657)",
    r"installed $L_{Ws,inst}$ = $L_{Ws,c}-D_C$":
        r"instalada $L_{Ws,inst}$ = $L_{Ws,c}-D_C$",
    "paths $L_{n,s,ij}$": "caminos $L_{n,s,ij}$",
    r"total $L_{n,s}$": r"total $L_{n,s}$",
    # Monospace formula box, symbols only: reads the same in Spanish.
    "DC = 10 log10(|Ys+Yi|^2 / (|Ys| Re Yi))\n"
    "Ln,s,ij = LWs,inst - Dsa - Rij - 10 log10(Si/S0) - 10 log10(A0/4)\n"
    "Ln,s = 10 log10(sum 10^(Ln,s,ij/10)),  S0 = A0 = 10 m2":
        "DC = 10 log10(|Ys+Yi|^2 / (|Ys| Re Yi))\n"
        "Ln,s,ij = LWs,inst - Dsa - Rij - 10 log10(Si/S0) - 10 log10(A0/4)\n"
        "Ln,s = 10 log10(sum 10^(Ln,s,ij/10)),  S0 = A0 = 10 m2",
    # tone_audibility figure (ISO/PAS 20065)
    "ISO/PAS 20065 Tonal Audibility": "Audibilidad tonal ISO/PAS 20065",
    r"Audibility $\Delta L$ [dB]": r"Audibilidad $\Delta L$ [dB]",
    r"threshold $\Delta L = 0$ dB": r"umbral $\Delta L = 0$ dB",
    # The formula box: only the parenthetical tail carries language, and
    # the saver's decimal-comma pass restyles the digits.
    "dfc = 25 + 75 (1 + 1.4 (fT/1000)^2)^0.69\n"
    "LG = LS + 10 log10(dfc/df),  av = -2 - log10(1 + (f/502)^2.5)\n"
    "dL = LT - LG - av  (combustion engine, Annex E)":
        "dfc = 25 + 75 (1 + 1.4 (fT/1000)^2)^0.69\n"
        "LG = LS + 10 log10(dfc/df),  av = -2 - log10(1 + (f/502)^2.5)\n"
        "dL = LT - LG - av  (motor de combustión, Anexo E)",
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
    "R′tr,s,w = 31 dB   (Ctr = -3)\nD2m,nT,w = 33 dB\n"
    "air inlet limits the low bands":
        "R′tr,s,w = 31 dB   (Ctr = -3)\nD2m,nT,w = 33 dB\n"
        "la entrada de aire limita las bandas bajas",
    # facade_elevation_geometry element names, drawn by plot_facade_elements;
    # worded as the facade-insulation page's own alt text.
    "Masonry wall": "Muro de fábrica",
    "Window": "Ventana",
    "Roller shutter box": "Cajón de persiana",
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
    # vibration_weighting_family: the nine ISO 8041-1 weightings on one axis.
    "The nine human-vibration weightings (ISO 8041-1 Table 3)":
        "Las nueve ponderaciones de vibración humana (ISO 8041-1, Tabla 3)",
    "Wk — seat surface, vertical (ISO 2631-1)":
        "Wk — superficie del asiento, vertical (ISO 2631-1)",
    "Wd — seat surface, horizontal": "Wd — superficie del asiento, horizontal",
    "Wc — backrest, x": "Wc — respaldo, x",
    "We — rotational (per rad)": "We — rotacional (por rad)",
    "Wj — recumbent, under the head": "Wj — tumbado, bajo la cabeza",
    "Wm — building occupants, all axes (ISO 2631-2)":
        "Wm — ocupantes de edificios, todos los ejes (ISO 2631-2)",
    "Wb — rail ride comfort, vertical (ISO 2631-4)":
        "Wb — confort de marcha ferroviaria, vertical (ISO 2631-4)",
    "Wf — motion sickness, vertical": "Wf — mareo, vertical",
    "Wh — hand-transmitted, all three axes (ISO 5349-1)":
        "Wh — transmitida a la mano, los tres ejes (ISO 5349-1)",
    "whole body: 0.5–80 Hz": "cuerpo entero: 0,5–80 Hz",
    "hand-arm Wh: 6.3–1250 Hz": "mano-brazo Wh: 6,3–1250 Hz",
    "Wf peaks at 0.17 Hz:\nmotion sickness is sub-hertz":
        "Wf tiene su máximo en 0,17 Hz:\nel mareo está por debajo del hercio",
    "Wd peaks 2.5 octaves below Wk:\nthe body is more compliant\n"
    "horizontally at low frequency":
        "Wd tiene su máximo 2,5 octavas bajo Wk:\nel cuerpo es más flexible\n"
        "en horizontal a baja frecuencia",
    # shock_dose_measures: r.m.s., MTVV and VDV read off one record.
    "(a)  A seated off-road record: 4.5 Hz ride plus five impacts":
        "(a)  Un registro sentado en todoterreno: marcha de 4,5 Hz "
        "más cinco impactos",
    "(b)  The 1 s running r.m.s., whose maximum is the MTVV":
        "(b)  El valor eficaz móvil de 1 s, cuyo máximo es el MTVV",
    "(c)  The fourth-power accumulation, whose end point is the VDV":
        "(c)  La acumulación de cuarta potencia, cuyo punto final es el VDV",
    "$a_z(t)$, unweighted": "$a_z(t)$, sin ponderar",
    "$a_w(t)$, Wk-weighted": "$a_w(t)$, ponderada con Wk",
    "acceleration [m/s$^2$]": "aceleración [m/s$^2$]",
    "running r.m.s., 1 s (Eq. (3))": "eficaz móvil, 1 s (Ec. (3))",
    r"$a_w\,t^{1/4}$ (the basic method)":
        r"$a_w\,t^{1/4}$ (el método básico)",
    "dose [m/s$^{1.75}$]": "dosis [m/s$^{1{,}75}$]",
    # spinal_response_peaks: the ISO 2631-5 response and its counted peaks.
    "$a_z(t)$, conditioned seat acceleration":
        "$a_z(t)$, aceleración acondicionada del asiento",
    "$A_z(t)$, spinal response (Formula 2)":
        "$A_z(t)$, respuesta espinal (Fórmula 2)",
    "(a)  The seat-to-spine filter turns an impact into a ringing response":
        "(a)  El filtro asiento-columna convierte un impacto en una "
        "respuesta que resuena",
    "(b)  Each peak's share of $\\sum A_{z,i}^6$ — dose $D_z$ = 32.5 m/s$^2$":
        "(b)  La parte de cada pico en $\\sum A_{z,i}^6$ — dosis $D_z$ = "
        "32,5 m/s$^2$",
    "0.4 s of free fall at $-1\\,g$:\n"
    "the 0.01 Hz high pass of 5.1.3 keeps it":
        "0,4 s de caída libre a $-1\\,g$:\n"
        "el paso alto de 0,01 Hz de 5.1.3 la conserva",
    "28 counted positive peaks $A_{z,i}$":
        "28 picos positivos contabilizados $A_{z,i}$",
    "a third of the largest peak: below this, no contribution":
        "un tercio del pico mayor: por debajo, sin contribución",
    # hav_vwf_lifetime: the ISO 5349-1 Annex C years-to-VWF curve.
    "Group-mean years to a 10 % prevalence of vibration white finger "
    "(ISO 5349-1 Annex C)":
        "Años de media de grupo hasta una prevalencia del 10 % de "
        "dedo blanco (ISO 5349-1, Anexo C)",
    "Daily exposure $A(8)$ [m/s$^2$]": "Exposición diaria $A(8)$ [m/s$^2$]",
    "Exposure duration $D_y$ [years]":
        "Duración de la exposición $D_y$ [años]",
    "EAV 2.5 m/s$^2$\n12.0 years": "VAE 2,5 m/s$^2$\n12,0 años",
    "ELV 5.0 m/s$^2$\n5.8 years": "VLE 5,0 m/s$^2$\n5,8 años",
    "Table C.1: 26 / 14 / 7 / 3.7 m/s$^2$":
        "Tabla C.1: 26 / 14 / 7 / 3,7 m/s$^2$",
    "extrapolation beyond Table C.1": "extrapolación fuera de la Tabla C.1",
    # Precision sound power (ISO 3745 / ISO 9614-3)
    "Sound power level LW [dB]": "Nivel de potencia acústica LW [dB]",
    "Non-applicable band": "Banda no aplicable",
    # precision_positions_arrays: the two ISO 3745 arrays
    "ISO 3745 precision microphone arrays":
        "Conjuntos de micrófonos de precisión de ISO 3745",
    "Hemisphere, 40 positions (Table E.1)":
        "Semiesfera, 40 posiciones (Tabla E.1)",
    "Sphere, 20 positions (Table D.1)":
        "Esfera, 20 posiciones (Tabla D.1)",
    "positions 1-20: always": "posiciones 1-20: siempre",
    "positions 21-40: on escalation": "posiciones 21-40: en la ampliación",
    # k1_k2_corrections: the two ISO 3744 corrections and their caps
    "K1 is a cliff, not a slope": "K1 es un acantilado, no una pendiente",
    "K2 saturates, and inherits every error in A":
        "K2 se satura, y hereda cada error de A",
    "Source-to-background margin ΔLp [dB]":
        "Margen entre fuente y fondo ΔLp [dB]",
    "Background correction K1 [dB]":
        "Corrección por ruido de fondo K1 [dB]",
    "Environmental correction K2 [dB]": "Corrección ambiental K2 [dB]",
    "ISO 3744 criterion\n6 dB": "Criterio de ISO 3744\n6 dB",
    "capped:\nupper bound": "acotada:\ncota superior",
    "A known to ±20 %": "A conocida a ±20 %",
    # reverberation_correction_terms: ISO 3741 Eq. 20, term by term
    "The five terms of ISO 3741 Eq. 20":
        "Los cinco términos de la Ec. 20 de ISO 3741",
    "Contribution to LW − L̄p [dB]": "Aportación a LW − L̄p [dB]",
    "Waterhouse": "Waterhouse",
    "the Waterhouse term rules the low bands\n"
    "(1.68 dB at 100 Hz, 0.02 dB at 10 kHz)":
        "el término de Waterhouse manda en las bandas graves\n"
        "(1,68 dB a 100 Hz, 0,02 dB a 10 kHz)",
    "4.34 A/S is 0.32 dB in this hard room, and 0.79 dB with the same room "
    "damped to T₆₀ = 0.8 s":
        "4,34 A/S es 0,32 dB en esta cámara reflectante, y 0,79 dB con la "
        "misma cámara amortiguada hasta T₆₀ = 0,8 s",
    # partial_power_map: the ISO 9614-2 box unfolded
    "Partial power level per segment [dB re 1 pW], the box unfolded":
        "Nivel de potencia parcial por segmento [dB re 1 pW], "
        "la caja desplegada",
    "LWi [dB]": "LWi [dB]",
    "Face total [dB]": "Total por cara [dB]",
    # spacer_bandwidth: what each p-p spacer costs at both ends
    "High end: the finite-difference bias, and f_max = 0.1 c/Δr":
        "Extremo alto: el sesgo de diferencias finitas, y f_max = 0,1 c/Δr",
    "Low end: doubling the spacer is worth 3 dB of margin":
        "Extremo bajo: doblar el espaciador vale 3 dB de margen",
    "Finite-difference bias [dB]": "Sesgo de diferencias finitas [dB]",
    "δpI0 margin,\nre 25 mm [dB]": "Margen de δpI0,\nre 25 mm [dB]",
    "−0.3 dB: the bound f_max is quoted against":
        "−0,3 dB: el sesgo contra el que se cita la cota f_max",
    "25 mm: the separation Table 2 is written for":
        "25 mm: la separación para la que está escrita la Tabla 2",
    # sound_power_grades_declaration: grades and the ISO 4871 declaration
    "One measurement, three grades: 92.4 dB(A) against a 93 dB limit":
        "Una medición, tres grados: 92,4 dB(A) frente a un límite de 93 dB",
    "ISO 4871 Annex B: LWAd = LWA + K_WA, verified when L1 ≤ LWAd":
        "ISO 4871 Anexo B: LWAd = LWA + K_WA, verificado si L1 ≤ LWAd",
    "Grade 1\nISO 3741 / 3745": "Grado 1\nISO 3741 / 3745",
    "Grade 2\nISO 3744 / 9614-2": "Grado 2\nISO 3744 / 9614-2",
    "Grade 3\nISO 3746": "Grado 3\nISO 3746",
    "declared limit 93 dB(A)": "límite declarado de 93 dB(A)",
    "A-weighted sound power level [dB]":
        "Nivel de potencia acústica ponderado A [dB]",
    "Operating mode 1": "Modo de funcionamiento 1",
    "Operating mode 2": "Modo de funcionamiento 2",
    "verification level L1": "nivel de verificación L1",
    # vibration_sound_power info box (ISO/TS 7849); monospace alignment kept
    "LW = Lv + 10 log10(S/S0) + 10 log10(e) + 10 log10(411/400)\n"
    "S = 1.6 m2,  S0 = 1 m2\n"
    "Part 1: e = 1 -> upper limit LW,max":
        "LW = Lv + 10 log10(S/S0) + 10 log10(e) + 10 log10(411/400)\n"
        "S = 1,6 m2,  S0 = 1 m2\n"
        "Parte 1: e = 1 -> límite superior LW,max",
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
    # anim_modulation_transfer: the envelope clip of speech-transmission.
    "The modulation transfer function on the envelope (IEC 60268-16)":
        "La función de transferencia de modulación sobre la envolvente "
        "(IEC 60268-16)",
    "Intensity envelope, received mean = 1":
        "Envolvente de intensidad, media recibida = 1",
    "mean, the same in every frame":
        "la media, la misma en todos los fotogramas",
    "m": "m",
    "the red point is the 4 Hz probe on the left":
        "el punto rojo es la sonda de 4 Hz de la izquierda",
    "Band MTI": "MTI de banda",
    "Octave band [Hz]": "Banda de octava [Hz]",
    "transmitted, m = 1": "transmitida, m = 1",
    "received": "recibida",
    "Reverberation shrinks the envelope about a fixed mean":
        "La reverberación encoge la envolvente en torno a una media fija",
    "  no noise": "  sin ruido",
    "Now hold the room and let noise take part of the level":
        "Ahora se fija la sala y el ruido se lleva parte del nivel",
    "Noise raises a floor under the same mean: m falls again":
        "El ruido levanta un suelo bajo la misma media: m vuelve a caer",
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
    # Rating statement, symbols only: reads the same in Spanish.
    "Rw (C ; Ctr) = 30 (-2 ; -3) dB": "Rw (C ; Ctr) = 30 (-2 ; -3) dB",
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
    # --- anim_iso717_shift: the fit walked step by step ---
    "Fitting the ISO 717 reference curve, one step at a time":
        "El ajuste de la curva de referencia ISO 717, paso a paso",
    "ISO 717-1: the curve steps down toward the measurement;\n"
    "an unfavourable deviation is a band that falls below it":
        "ISO 717-1: la curva baja hacia la medida;\n"
        "una desviación desfavorable es una banda por debajo de ella",
    "ISO 717-2: the same rule with the sign reversed;\n"
    "the curve steps up, and a band above it is unfavourable":
        "ISO 717-2: la misma regla con el signo cambiado;\n"
        "la curva sube y la banda por encima es la desfavorable",
    "Sum of unfavourable\ndeviations [dB]":
        "Suma de desviaciones\ndesfavorables [dB]",
    "Reference curve read at 500 Hz [dB]":
        "Curva de referencia leída a 500 Hz [dB]",
    "cap 32.0 dB = 2.0 dB per band": "límite 32,0 dB = 2,0 dB por banda",
    "measured spectrum": "espectro medido",
    "reference curve, as shifted": "curva de referencia, ya desplazada",
    # ("unfavourable deviations" is already in this table, further down.)
    "Normalized impact level L'nT [dB]":
        "Nivel de impactos normalizado L'nT [dB]",
    "over the cap: shift again": "pasa del límite: desplazar otra vez",
    "largest sum still under the cap": "la mayor suma bajo el límite",
    "legal, but the sum is smaller:": "válido, pero la suma es menor:",
    "this is one step too far": "esto es un paso de más",
    "Rw": "Rw",
    "L'nT,w": "L'nT,w",
    # --- anim_block_vs_exponential: the alignment the block detector needs ---
    "One burst, two detectors, and the block grid underneath":
        "Una ráfaga, dos detectores y la rejilla de bloques debajo",
    "Sound pressure": "Presión acústica",
    "the shaded strips are the 125 ms blocks":
        "las franjas sombreadas son los bloques de 125 ms",
    "Level re steady tone [dB]": "Nivel re tono estacionario [dB]",
    "exponential Fast envelope": "envolvente exponencial Fast",
    "block Leq, 125 ms slices": "Leq por bloques de 125 ms",
    "IEC 61672-1 Table 4 target": "objetivo de la Tabla 4 de IEC 61672-1",
    "Burst start within the block [ms]":
        "Inicio de la ráfaga dentro del bloque [ms]",
    "Reading [dB]": "Lectura [dB]",
    "the block reading leaves the corridor":
        "la lectura por bloques sale del corredor",
    # --- anim_feedback_howl: the loop as a geometric series ---
    "Gain before feedback is a convergence condition":
        "La ganancia antes del acople es una condición de convergencia",
    "talker,\n0.3 m from the microphone":
        "hablante,\na 0,3 m del micrófono",
    "listener,\n12 m away": "oyente,\na 12 m",
    "microphone, mixer, amplifier": "micrófono, mesa, amplificador",
    "the feedback path": "el camino de realimentación",
    "Round trip number": "Número de vuelta",
    "Copy level re direct [dB]": "Nivel de la copia re directo [dB]",
    "Round trips summed": "Vueltas sumadas",
    "Total re direct [dB]": "Total re directo [dB]",
    "10 dB margin": "margen de 10 dB",
    "four microphones": "cuatro micrófonos",
    "loop at unity": "lazo en la unidad",
    "Long's 10 dB margin: each copy is a third of the last":
        "Margen de 10 dB de Long: cada copia es un tercio de la anterior",
    "Four open microphones add 10 lg 4 = 6 dB to the loop":
        "Cuatro micrófonos abiertos suman 10 lg 4 = 6 dB al lazo",
    "Four more decibels of system gain: the loop reaches unity":
        "Cuatro decibelios más de ganancia: el lazo llega a la unidad",
    "the sum does not converge: this is the howl":
        "la suma no converge: esto es el acople",
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
    # zwicker_time_varying: the clause 6 trace and its percentiles
    "Time-Varying Loudness and the Percentiles (ISO 532-1 clause 6)":
        "Sonoridad variable en el tiempo y los percentiles "
        "(ISO 532-1, apartado 6)",
    "1 kHz bursts stepping 45 to 85 dB":
        "Ráfagas de 1 kHz que suben de 45 a 85 dB",
    "Loudness-vs-time N(t), 2 ms steps":
        "Sonoridad frente al tiempo N(t), pasos de 2 ms",
    "Loudness N [sone]": "Sonoridad N [sonios]",
    "Percentage of the analysis time exceeding N [%]":
        "Porcentaje del tiempo de análisis que supera N [%]",
    # sharpness_pair_and_targets (DIN 45692)
    "Sharpness: Where the Loudness Sits, Not How Much There Is":
        "Agudeza: dónde se sitúa la sonoridad, no cuánta hay",
    "Equally loud, seven times as sharp":
        "Igual de sonoras, siete veces más agudas",
    "Sharpness S [acum]": "Agudeza S [acum]",
    "DIN 45692 Table A.2, 250 Hz to 4 kHz":
        "Tabla A.2 de DIN 45692, de 250 Hz a 4 kHz",
    "Table A.2 hearing-test targets":
        "objetivos de los ensayos de audición (Tabla A.2)",
    "permitted deviation: 5 % or 0.05 acum":
        "desviación permitida: 5 % o 0,05 acum",
    "sharpness_din(), each band set to 4 sone":
        "sharpness_din(), cada banda ajustada a 4 sonios",
    # Sottek specific roughness / fluctuation-strength panels (ECMA-418-2)
    "Roughness of the ECMA-418-2 Reference Sound (1 kHz, 100 % AM at "
    "70 Hz, 60 dB)":
        "Aspereza del sonido de referencia de ECMA-418-2 (1 kHz, AM del "
        "100 % a 70 Hz, 60 dB)",
    "Fluctuation strength of the ECMA-418-2 Reference Sound (1 kHz, "
    "100 % AM at 4 Hz, 60 dB)":
        "Intensidad de fluctuación del sonido de referencia de ECMA-418-2 "
        "(1 kHz, AM del 100 % a 4 Hz, 60 dB)",
    "Average specific roughness": "Aspereza específica media",
    "Average specific fluctuation strength":
        "Intensidad de fluctuación específica media",
    "Critical-band rate z [Bark_HMS]": "Razón de banda crítica z [Bark_HMS]",
    "Specific roughness R'(z) [asper/Bark_HMS]":
        "Aspereza específica R'(z) [asper/Bark_HMS]",
    "Specific fluctuation strength F'(z) [vacil_HMS/Bark_HMS]":
        "Intensidad de fluctuación específica F'(z) [vacil_HMS/Bark_HMS]",
    "Fluctuation strength F [vacil_HMS]":
        "Intensidad de fluctuación F [vacil_HMS]",
    "The single value is a percentile of this trace":
        "El valor único es un percentil de esta traza",
    "R(l50), the running value": "R(l50), el valor dependiente del tiempo",
    "F(l50), the running value": "F(l50), el valor dependiente del tiempo",
    # hms_modulation_bandpass ylabel: symbols and units read the same
    "F [vacil_HMS] / R [asper]": "F [vacil_HMS] / R [asper]",
    # fluctuation_strength: the two models side by side
    "Fluctuation Strength — the 4 Hz Band-Pass, and Which Model to Quote":
        "Intensidad de fluctuación: el paso de banda a 4 Hz y qué modelo "
        "citar",
    "Both models on AM broadband noise, 60 dB":
        "Los dos modelos sobre ruido de banda ancha AM, 60 dB",
    "Signal model on the AM tone, 70 dB":
        "El modelo de señal sobre el tono AM, 70 dB",
    # The Eq. 10.2 info box is all symbols; the saver's decimal-comma pass
    # restyles its digits, so it reads the same in Spanish.
    "F = 5.8 (1.25 m - 0.25)(0.05 L - 1)\n"
    "    / [(fmod/5)^2 + 4/fmod + 1.5]  vacil":
        "F = 5.8 (1.25 m - 0.25)(0.05 L - 1)\n"
        "    / [(fmod/5)^2 + 4/fmod + 1.5]  vacil",
    # annoyance_weightings + psychoacoustic_annoyance (Fastl & Zwicker)
    "The Two Weightings of the Psychoacoustic Annoyance Model":
        "Las dos ponderaciones del modelo de molestia psicoacústica",
    "The kink at the reference sharpness":
        "El codo en la agudeza de referencia",
    "Sharpness weighting wS": "Ponderación de agudeza wS",
    "1.75 acum: below it sharpness costs nothing":
        "1,75 acum: por debajo, la agudeza no cuesta nada",
    # "vs" instead of "frente a": measured, the longer form ran 12 px past
    # the right canvas edge (the title is centred on the right-hand panel).
    "Roughness costs more than fluctuation (0.6 against 0.4)":
        "La aspereza cuesta más que la fluctuación (0,6 vs 0,4)",
    "Sensation magnitude v [asper or vacil]":
        "Magnitud de sensación v [asper o vacil]",
    "all roughness: R = v, F = 0": "todo en aspereza: R = v, F = 0",
    "all fluctuation: F = v, R = 0": "todo en fluctuación: F = v, R = 0",
    "N5 = 30 sone, S = 2.0 acum throughout":
        "N5 = 30 sonios, S = 2,0 acum en todo el panel",
    # The PA info box is all symbols, same story as Eq. 10.2 above.
    "PA = N5 (1 + sqrt(wS^2 + wFR^2))\n"
    "wS  = (S - 1.75) 0.25 log10(N5 + 10)\n"
    "wFR = (2.18 / N5^0.4)(0.4 F + 0.6 R)":
        "PA = N5 (1 + sqrt(wS^2 + wFR^2))\n"
        "wS  = (S - 1.75) 0.25 log10(N5 + 10)\n"
        "wFR = (2.18 / N5^0.4)(0.4 F + 0.6 R)",
    # tnr_pr_comparison (ECMA-418-1)
    "Tone-to-Noise Ratio and Prominence Ratio Compared (ECMA-418-1)":
        "Relación tono-ruido y relación de prominencia comparadas "
        "(ECMA-418-1)",
    "One 250 Hz fan tone, two criteria":
        "Un tono de ventilador de 250 Hz, dos criterios",
    "A tone in the next critical band up":
        "Un tono en la banda crítica contigua superior",
    "TNR criterion (clause 11.5)": "criterio TNR (apartado 11.5)",
    "PR criterion (clause 12.5)": "criterio PR (apartado 12.5)",
    "TNR criterion 8 dB": "criterio TNR de 8 dB",
    "PR criterion 9 dB": "criterio PR de 9 dB",
    "TNR of the 1 kHz tone": "TNR del tono de 1 kHz",
    "PR of the 1 kHz tone": "PR del tono de 1 kHz",
    "Ratio [dB]": "Relación [dB]",
    "Level of a second tone at 1160 Hz, relative to the 1 kHz tone [dB]":
        "Nivel de un segundo tono a 1160 Hz respecto al tono de 1 kHz [dB]",
    # tone_audibility_uncertainty + two_tone_separation (ISO/PAS 20065)
    "Decisive Audibility, Its Uncertainty and the Tonal Adjustment":
        "Audibilidad decisiva, su incertidumbre y el ajuste tonal",
    "Audibility ΔL [dB]": "Audibilidad ΔL [dB]",
    "Measured 3 s spectrum (Annex E, run index j)":
        "Espectro de 3 s medido (Anexo E, índice de medida j)",
    "decisive audibility of each spectrum, ± U (clause 6)":
        "audibilidad decisiva de cada espectro, ± U (apartado 6)",
    "Two Tones in One Critical Band: Separate or Combined (ISO/PAS 20065 "
    "Formula 19)":
        "Dos tonos en una banda crítica: separados o combinados "
        "(Fórmula 19 de ISO/PAS 20065)",
    "Threshold fD (Formula 19)": "Umbral fD (Fórmula 19)",
    "Frequency of the more audible tone fT [Hz]":
        "Frecuencia del tono más audible fT [Hz]",
    "Frequency separation |fT1 − fT2| [Hz]":
        "Separación en frecuencia |fT1 − fT2| [Hz]",
    "minimum 21 Hz at 212 Hz": "mínimo de 21 Hz a 212 Hz",
    "rated separately": "evaluados por separado",
    "energy-summed into one FG entry":
        "sumados en energía en una única entrada FG",
    "Annex E pair: 118.4 and 137.3 Hz,\n"
    "18.9 Hz apart — below fD, so combined":
        "Pareja del Anexo E: 118,4 y 137,3 Hz,\n"
        "separados 18,9 Hz, por debajo de fD: se combinan",
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
    # frequency_response: H1 against H2 with the noise moved between channels
    "Choosing Between H1 and H2 (Bendat & Piersol)":
        "Elegir entre H1 y H2 (Bendat y Piersol)",
    "Noise on the output — H1 is unbiased":
        "Ruido en la salida — H1 es el insesgado",
    "Noise on the input — H2 is unbiased":
        "Ruido en la entrada — H2 es el insesgado",
    # The estimator formulas read the same in both languages.
    "H1 = Gxy / Gxx": "H1 = Gxy / Gxx",
    "H2 = Gyy / Gyx": "H2 = Gyy / Gyx",
    # The measured coherence trace (la coherencia, feminine).
    "measured": "medida",
    # itu_r_468_weighting figure (ITU-R BS.468-4 network and CCIR-RMS form)
    "The ITU-R BS.468-4 Network and Its CCIR-RMS Form":
        "La red ITU-R BS.468-4 y su forma CCIR-RMS",
    "Weighting [dB]": "Ponderación [dB]",
    "A-weighting (reference)": "Ponderación A (referencia)",
    "ITU-R BS.468-4 (0 dB at 1 kHz)": "ITU-R BS.468-4 (0 dB a 1 kHz)",
    "AES17 CCIR-RMS (0 dB at 2 kHz)": "CCIR-RMS de AES17 (0 dB a 2 kHz)",
    "+12.2 dB at 6.3 kHz": "+12,2 dB a 6,3 kHz",
    "0 dB at 1 kHz": "0 dB a 1 kHz",
    "-5.63 dB at 1 kHz": "-5,63 dB a 1 kHz",
    "for a 100 Hz fundamental the network cuts d2 by 13.8 dB\n"
    "and d3 by 10.3 dB, and lifts the 10th order and above":
        "para un fundamental de 100 Hz la red recorta d2 en 13,8 dB\n"
        "y d3 en 10,3 dB, y realza del 10º orden en adelante",
    # intermodulation_tests panel notes (the titles carry computed values
    # and are in the pattern list)
    "denominator: a(f1) + a(f2)": "denominador: a(f1) + a(f2)",
    "only the two in-band products count":
        "solo cuentan los dos productos en banda",
    "denominator: the 15 kHz sine alone":
        "denominador: solo la sinusoide de 15 kHz",
    # The TDFD sideband markers name the Greek letter, same in Spanish.
    "f0-delta": "f0-delta",
    "f0+delta": "f0+delta",
    # microphone_patterns polar family (IEC 60268-4 13.2.2)
    "The First-Order Family and Its Directivity Index (IEC 60268-4 13.2.2)":
        "La familia de primer orden y su índice de directividad "
        "(IEC 60268-4 13.2.2)",
    # microphone_noise_weightings (IEC 60268-4 17.2)
    "One Noise Voltage, Two Networks (IEC 60268-4 17.2)":
        "Una tensión de ruido, dos redes (IEC 60268-4 17.2)",
    "Inherent noise, unweighted": "Ruido inherente, sin ponderar",
    "A-weighted": "Con ponderación A",
    "ITU-R BS.468-4 weighted": "Con ponderación ITU-R BS.468-4",
    "Band-summed level [dB]": "Nivel de la suma de bandas [dB]",
    "this capsule\n(1/f, falling)": "esta cápsula\n(1/f, decreciente)",
    "a hissier capsule\n(rising to 20 kHz)":
        "una cápsula más siseante\n(creciente hasta 20 kHz)",
    "the quasi-peak detector of ITU-R BS.468-4 adds the rest of the "
    "customary ~10 dB, and is not implemented here":
        "el detector de cuasipico de ITU-R BS.468-4 añade el resto de los "
        "~10 dB habituales, y no está implementado aquí",
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
    # swept_sine_harmonic_responses: each order as a full response
    "The Separated Harmonic Frequency Responses (Farina / Novak)":
        "Las respuestas en frecuencia separadas por armónico "
        "(Farina / Novak)",
    "Frequency of the harmonic itself [Hz]":
        "Frecuencia del propio armónico [Hz]",
    "|H1(f)| over [1f1, 1f2]": "|H1(f)| en [1f1, 1f2]",
    "|H2(f)| over [2f1, 2f2]": "|H2(f)| en [2f1, 2f2]",
    "|H3(f)| over [3f1, 3f2]": "|H3(f)| en [3f1, 3f2]",
    "dotted: the Chebyshev levels 1 + 3a3/4, a2/2 and a3/4;\n"
    "each order rolls off at the 3 kHz post-filter, not at n f2":
        "punteado: los niveles de Chebyshev 1 + 3a3/4, a2/2 y a3/4;\n"
        "cada orden cae en el posfiltro de 3 kHz, no en n f2",
    # swept_sine_methods: synchronized against Farina on one recording
    "Same Recording, Two Deconvolutions (Novak et al. 2015, Fig. 6)":
        "La misma grabación, dos deconvoluciones (Novak et al. 2015, Fig. 6)",
    "Chebyshev level a2/2": "Nivel de Chebyshev a2/2",
    "f2 = 6 kHz: the Farina band stops here":
        "f2 = 6 kHz: la banda de Farina se detiene aquí",
    # Only "kHz" carries language here; the label reads the same in Spanish.
    "2 f2 = 12 kHz": "2 f2 = 12 kHz",
    # "arg" is the same operator in Spanish notation; method="..." is API.
    'arg H2 — method="farina"': 'arg H2 — method="farina"',
    'arg H2 — method="synchronized"': 'arg H2 — method="synchronized"',
    "Unwrapped phase [rad]": "Fase desenrollada [rad]",
    "true phase of H2: -pi/2 at every frequency":
        "fase verdadera de H2: -pi/2 en todas las frecuencias",
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
        # Measured on the drawn figure: "de un registro corto" ran 4,9 px
        # past the left canvas edge; the shorter apposition fits.
        "Densidad multitaper de Thomson, registro corto (Percival y Walden)",
    "Single Slepian taper ($K$ = 1, $\\nu$ = 2)":
        "Un solo taper de Slepian ($K$ = 1, $\\nu$ = 2)",
    "Multitaper estimate ($K$ = 7, adaptive)":
        "Estimación multitaper ($K$ = 7, adaptativa)",
    "A 60 dB tone over a pink floor, and what it costs":
        "Un tono de 60 dB sobre un suelo rosa, y lo que cuesta",
    "Equivalent degrees of freedom": "Grados de libertad equivalentes",
    # "la estimación multitaper adaptativa" and "ventana de Hann", as the
    # Spanish page names them (spectral-analysis.mdx).
    "Multitaper, adaptive": "Multitaper, adaptativa",
    "Welch, Hann taper": "Welch, ventana de Hann",
    # A proper name and an API keyword: reads the same in Spanish.
    "Welch, nperseg = 2048": "Welch, nperseg = 2048",
    # Abbreviated like the annotation pattern below ("g.d.l. equivalentes").
    "equivalent dof (right axis)": "g.d.l. equivalentes (eje derecho)",
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
    # anim_image_source_buildup: the expanding sphere sweeping the lattice.
    "The reflectogram is a lattice being swept (image-source method)":
        "El reflectograma es una retícula que se barre (método de las "
        "fuentes imagen)",
    "the plan draws only the images at the source's own height;\nthe floor "
    "and ceiling families arrive between them":
        "la planta dibuja solo las imágenes a la altura de la fuente;\nlas "
        "familias de suelo y techo llegan entre ellas",
    "Level re direct [dB]": "Nivel rel. directo [dB]",
    "Arrivals so far": "Llegadas hasta ahora",
    "1/r spreading": "divergencia 1/r",
    "(4 pi / 3)(c t)^3 / V": "(4 pi / 3)(c t)^3 / V",
    "counted": "contadas",
    "each reflection is a mirror image of the source;\n"
    "amplitude = product of wall reflection factors / (4 pi r)":
        "cada reflexión es una imagen especular de la fuente;\n"
        "amplitud = producto de factores de reflexión de pared / (4 pi r)",
    # decay_signatures panel titles (buildings/rooms/room-acoustics).
    "single slope": "pendiente única",
    "coupled volume": "volumen acoplado",
    "strong early energy": "energía temprana fuerte",
    # decay_range_bias: axes, flags and info box; "indicador" and "rango
    # útil de decaimiento" as the room-acoustics page words them.
    "Error of the fitted decay time [%]":
        "Error del tiempo de decaimiento ajustado [%]",
    "Usable decay range  dynamic_range (INR) [dB]":
        "Rango útil de decaimiento  dynamic_range (INR) [dB]",
    "The bias an undersized decay range leaves behind":
        "El sesgo que deja un rango de decaimiento escaso",
    "ISO min. T20": "mín. ISO T20",
    "ISO min. T30": "mín. ISO T30",
    "flag T20": "indicador T20",
    "flag T30": "indicador T30",
    "synthetic single-slope decay, T = 1.0 s\nwhite noise floor swept, "
    "fs = 48 kHz\ngreen band: the 5 % JND\nred band: flagged invalid for "
    "T20\nbelow ~34 dB the fit returns NaN":
        "decaimiento sintético de pendiente única, T = 1,0 s\nsuelo de "
        "ruido blanco barrido, fs = 48 kHz\nbanda verde: la DAP del 5 %\n"
        "banda roja: marcada no válida para T20\npor debajo de ~34 dB el "
        "ajuste devuelve NaN",
    # reverberation_model_absorption (Sabine against the mean absorption).
    "Model behaviour against the mean absorption":
        "El comportamiento de los modelos frente a la absorción media",
    r"Mean absorption coefficient $\bar\alpha$":
        r"Coeficiente de absorción medio $\bar\alpha$",
    "Departure from\nSabine [%]": "Desviación respecto\nde Sabine [%]",
    "Eyring falls to zero": "Eyring cae a cero",
    "Sabine stays finite:\n0.12 s at $\\alpha = 1$":
        "Sabine se mantiene finito:\n0,12 s en $\\alpha = 1$",
    "room 8 x 5 x 3 m\nV = 120 m^3, S = 158 m^2\nuniform absorption, no "
    "air term\nshaded: outside Sabine's domain":
        "sala de 8 x 5 x 3 m\nV = 120 m^3, S = 158 m^2\nabsorción "
        "uniforme, sin término de aire\nsombreado: fuera del dominio de "
        "Sabine",
    # enclosed_space_air_term (EN 12354-6 clause 4.3): "término de aire",
    # "oficina" and "sala" as the enclosed-space-absorption page words them.
    r"Air term $A_{air} = 4mV(1-\psi)$ [m$^2$]":
        r"Término de aire $A_{air} = 4mV(1-\psi)$ [m$^2$]",
    r"Six climate profiles, $V$ = 2000 m$^3$":
        r"Seis perfiles climáticos, $V$ = 2000 m$^3$",
    "The same absorption in two volumes":
        "La misma absorción en dos volúmenes",
    r"60 m$^3$ office (no air)": r"oficina de 60 m$^3$ (sin aire)",
    r"60 m$^3$ office (20 °C, 50-70 %)":
        r"oficina de 60 m$^3$ (20 °C, 50-70 %)",
    r"2000 m$^3$ hall (no air)": r"sala de 2000 m$^3$ (sin aire)",
    r"2000 m$^3$ hall (20 °C, 50-70 %)":
        r"sala de 2000 m$^3$ (20 °C, 50-70 %)",
    "-1.7 % at 1 kHz": "-1,7 % a 1 kHz",
    "-42 % at 8 kHz": "-42 % a 8 kHz",
    # enclosed_space_objects (EN 12354-6 Annex E case 2): "desnuda" and
    # "amueblada" as the page's own snippet labels them.
    "Where the absorption comes from": "De dónde sale la absorción",
    "The volume the objects displace": "El volumen que desplazan los objetos",
    "surfaces": "superficies",
    "air": "aire",
    "bare": "desnuda",
    "furnished": "amueblada",
    "objects (Formula 4)": "objetos (Fórmula 4)",
    r"furnished, $\psi$ = 0 (absorption only)":
        r"amueblada, $\psi$ = 0 (solo absorción)",
    r"furnished, $\psi$ = 0.072": r"amueblada, $\psi$ = 0,072",
    r"the gap is $\psi$ alone: 7.2 %": r"la diferencia es solo $\psi$: 7,2 %",
    # image_source_order_convergence (buildings/rooms/room-image-sources):
    # "retícula", "corte de orden" and "RIR" as the page words them.
    "The image lattice is a time horizon":
        "La retícula de imágenes es un horizonte temporal",
    "Reflection-order cut-off  max_order":
        "Corte de orden de reflexión  max_order",
    r"Fitted $T_{30}$ [s]": r"$T_{30}$ ajustado [s]",
    "T30 from the synthetic RIR": "T30 de la RIR sintética",
    "Eyring, 0.93 s": "Eyring, 0,93 s",
    "Audible images": "Imágenes audibles",
    "room 7 x 5 x 3 m, alpha = 0.12\nV = 105 m^3, S = 142 m^2, fs = 48 "
    "kHz\nshaded: +/- 10 % around Eyring":
        "sala de 7 x 5 x 3 m, alpha = 0,12\nV = 105 m^3, S = 142 m^2, "
        "fs = 48 kHz\nsombreado: +/- 10 % alrededor de Eyring",
    # image_source_anisotropy: "alargamiento" and "especular (fuentes
    # imagen)" as the page's own snippet labels them.
    "Where the specular decay leaves the diffuse-field estimate":
        "Donde el decaimiento especular abandona la estimación de campo "
        "difuso",
    r"Room elongation  $L_x : L_y = L_z$":
        r"Alargamiento de la sala  $L_x : L_y = L_z$",
    "specular (image source)": "especular (fuentes imagen)",
    "Eyring (diffuse field)": "Eyring (campo difuso)",
    r"$\pm$ 10 % around Eyring": r"$\pm$ 10 % alrededor de Eyring",
    "V = 105 m^3 and mean alpha = 0.12 held fixed\ncube (1:1) through a "
    "6:1 corridor\nmean of 4 source-receiver pairs, max_order = 60":
        "V = 105 m^3 y alpha media = 0,12 fijos\ndel cubo (1:1) a un "
        "pasillo 6:1\nmedia de 4 pares fuente-receptor, max_order = 60",
    # image_source_bands: "continua"/"a trazos" as the page words them.
    "Per-band decay: solid without air, dashed with air":
        "Decaimiento por banda: continua sin aire, a trazos con aire",
    "room 7 x 5 x 3 m, max_order = 60\nwall alpha 0.10 -> 0.50 with "
    "frequency\nair at 20 C / 50 % RH: -0.4 % of T30 at 250 Hz,\n"
    "-4.4 % at 4 kHz":
        "sala de 7 x 5 x 3 m, max_order = 60\nalpha de pared 0,10 -> 0,50 "
        "con la frecuencia\naire a 20 °C / 50 % HR: -0,4 % del T30 a "
        "250 Hz,\n-4,4 % a 4 kHz",
    # room_proportion_modes; "axial" reads the same in Spanish and matches
    # the library's own rectangular_room_modes label.
    r"Three rooms of 105 m$^3$, modes up to 200 Hz":
        r"Tres salas de 105 m$^3$, modos hasta 200 Hz",
    "axial": "axial",
    "tangential": "tangencial",
    "oblique": "oblicuo",
    "cube": "cubo",
    "cube\n4.72 x 4.72 x 4.72 m": "cubo\n4,72 x 4,72 x 4,72 m",
    "Bolt 1 : 1.4 : 1.9": "Bolt 1 : 1,4 : 1,9",
    "Bolt 1 : 1.4 : 1.9\n3.40 x 4.77 x 6.47 m":
        "Bolt 1 : 1,4 : 1,9\n3,40 x 4,77 x 6,47 m",
    "Spacing to the next\ndistinct mode [Hz]":
        "Separación al siguiente\nmodo distinto [Hz]",
    # steady_state_directivity / steady_state_field; "Total" matches the
    # library's own Spanish legend label.
    r"$Q$ moves $r_c$, not the plateau": r"$Q$ mueve $r_c$, no la meseta",
    "Absorption moves the plateau, not the direct field":
        "La absorción mueve la meseta, no el campo directo",
    "Distance from source [m]": "Distancia a la fuente [m]",
    "Sound pressure level [dB]": "Nivel de presión acústica [dB]",
    "12 x 8 x 4 m workshop, S = 352 m^2, Lw = 90 dB re 1 pW":
        "taller de 12 x 8 x 4 m, S = 352 m^2, Lw = 90 dB re 1 pW",
    "Total": "Total",
    # Underwater acoustics (ISO 17208 ship radiated noise; ISO 18406 pile driving)
    # piling_campaign_accumulation: the two units at the FL4/FL8 boundary each
    # thought the other owned it, so its lines were the last of the baseline.
    "Accumulation Against the Criteria (dotted TTS, dashed AUD INJ)":
        "Acumulación frente a los criterios (TTS punteado, AUD INJ discontinuo)",
    "Weighted cumulative SEL [dB re 1 uPa^2 s]":
        "SEL acumulado ponderado [dB re 1 uPa^2 s]",
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
    # Simulation: the one-way plane-wave launcher and the meshed panel.
    "One-way plane-wave launch: flat front, absorbed back side":
        "Lanzamiento de onda plana unidireccional: frente plano, dorso absorbido",
    "Settled field, 1 kHz CW": "Campo estacionario, onda continua de 1 kHz",
    "Cut across the front (row 80)": "Corte transversal del frente (fila 80)",
    "What is left behind the line": "Lo que queda detrás de la línea",
    "envelope over one period": "envolvente sobre un periodo",
    "one snapshot": "una instantánea",
    "injection line": "línea de inyección",
    "sponge": "esponja",
    "Row envelope [dB re the forward field]":
        "Envolvente por fila [dB re el campo hacia delante]",
    "largest difference between neighbouring columns:\n"
    "0.0e+00 Pa — the row is bit-identical":
        "mayor diferencia entre columnas contiguas:\n"
        "0,0e+00 Pa — la fila es idéntica bit a bit",
    "-38.4 dB of the field energy sits\n"
    "in the 20 sponge rows behind the line":
        "-38,4 dB de la energía del campo está\n"
        "en las 20 filas de esponja tras la línea",
    "What the transfer matrix homogenises: five slit resonators, each one "
    "reflection coefficient":
        "Lo que homogeneiza la matriz de transferencia: cinco resonadores de "
        "ranura, cada uno un coeficiente de reflexión",
    "What the solver steps on: the same panel as a boolean obstacle mask at "
    "dx = 0.5 mm":
        "Lo que integra el solver: el mismo panel como máscara booleana de "
        "obstáculos con dx = 0,5 mm",
    "The fifth cell magnified: the 3.2 mm neck is six cells wide, and it is "
    "what sets dx":
        "La quinta celda ampliada: el cuello de 3,2 mm ocupa seis celdas, y es "
        "lo que fija dx",
    "3.2 mm neck = 6 cells": "cuello de 3,2 mm = 6 celdas",
    "20.3 mm slit": "ranura de 20,3 mm",
    "3 mm rigid backing": "fondo rígido de 3 mm",
    "Water over steel at normal incidence: the probe history res.plot() draws":
        "Agua sobre acero con incidencia normal: el historial de sonda que "
        "dibuja res.plot()",
    "probe pressure, 7.5 m below the source":
        "presión en la sonda, 7,5 m bajo la fuente",
    "incident\n1.02 ms": "incidente\n1,02 ms",
    "echo off the steel\n3.04 ms": "eco en el acero\n3,04 ms",
    "echo / incident = 0.938\n(Z₂−Z₁)/(Z₂+Z₁) = 0.938":
        "eco / incidente = 0,938\n(Z₂−Z₁)/(Z₂+Z₁) = 0,938",
    "Number of strikes N": "Número de golpes N",
    "Cumulative SEL [dB re 1 µPa²·s]": "SEL acumulado [dB re 1 µPa²·s]",
    "ICAO Aircraft Flyover — Effective Perceived Noise Level (Annex 16)":
        "Sobrevuelo de aeronave ICAO — Nivel efectivo de ruido percibido (Anexo 16)",
    "Level [PNdB]": "Nivel [PNdB]",
    "10 dB-down window": "Ventana 10 dB por debajo",
    # anim_epnl_flyover: EPNL assembled in the order the standard builds it.
    "EPNL, record by record (ICAO Annex 16 Appendix 2)":
        "EPNL, registro a registro (OACI Anexo 16 Apéndice 2)",
    "PNLTM - 10 dB": "PNLTM - 10 dB",
    "microphone": "micrófono",
    "fitted background SPL''": "SPL'' de fondo ajustado",
    "PNL": "PNL",
    "PNLT = PNL + C": "PNLT = PNL + C",
    "Peak PNLTM": "Máximo PNLTM",
    "Duration correction D": "Corrección de duración D",
    "EPNL": "EPNL",
    "Each record: fit the background, measure the tone excess":
        "Cada registro: se ajusta el fondo y se mide el exceso tonal",
    "The pass is over; only now is the peak PNLTM known":
        "La pasada ha terminado; solo ahora se conoce el máximo PNLTM",
    "The 10 dB-down window is located from that peak":
        "La ventana 10 dB por debajo se sitúa a partir de ese máximo",
    "Divide by the fixed 10 s reference: D, then EPNL":
        "Se divide por la referencia fija de 10 s: D y después EPNL",
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
    "Total (5 kn)": "Total (5 kn)",
    "Total (20 kn)": "Total (20 kn)",
    "containership (18 kn, 300 m)": "portacontenedores (18 kn, 300 m)",
    "cruise (17 kn, 250 m)": "crucero (17 kn, 250 m)",
    "tug (4 kn, 30 m)": "remolcador (4 kn, 30 m)",
    # Underwater propagation, continued: detection range, normal modes, the
    # sonar budget, ray turning, the PE paraxial band, volume absorption and
    # the sound-speed equations. Vocabulary follows the Spanish underwater
    # pages: alcance de detección, cruce, guía de ondas, ceros interiores,
    # ángulo rasante, promediado en distancia, descamar.
    "Closed Form: One Crossing (FOM = 82.7 dB, 50 kHz)":
        "Forma cerrada: un cruce (FOM = 82,7 dB, 50 kHz)",
    "Crossings": "Cruces",
    "Normal-mode TL (30 Hz, 100 m waveguide, 3 modes)":
        "TL de modos normales (30 Hz, guía de 100 m, 3 modos)",
    "Total TL": "TL total",
    "Mode function Psi_m(z)": "Función modal Psi_m(z)",
    "Mode m Has m - 1 Interior Nulls":
        "El modo m tiene m - 1 ceros interiores",
    "Number of propagating modes M": "Número de modos propagantes M",
    "One Mode Cuts On at a Time": "Los modos entran de uno en uno",
    "Propagating modes returned": "Modos propagantes devueltos",
    "mode 4 has a null at the source depth,\nso the source does not excite it":
        "el modo 4 tiene un cero a la profundidad\nde la fuente, que por eso "
        "no lo excita",
    "source depth 50 m": "profundidad de la fuente: 50 m",
    "Modal Transmission Loss (z = 100 m)":
        "Pérdida por transmisión modal (z = 100 m)",
    "10 kHz, spherical only": "10 kHz, solo esférica",
    "10 kHz, practical R0 = 1 km": "10 kHz, ley práctica, R0 = 1 km",
    "20 kHz, practical R0 = 1 km": "20 kHz, ley práctica, R0 = 1 km",
    "A Passive Sonar Budget, End to End":
        "Un balance de sonar pasivo, de punta a punta",
    "SL = 140 dB re 1 uPa^2/Hz at 10 kHz\n"
    "NL = 60 dB,  DI = 15 dB,  DT = 8 dB\nFOM = SL - (NL - DI) - DT":
        "SL = 140 dB re 1 uPa^2/Hz a 10 kHz\n"
        "NL = 60 dB,  DI = 15 dB,  DT = 8 dB\nFOM = SL - (NL - DI) - DT",
    # The ship_source_level info box is mathematics that reads the same in
    # both languages; the save-time pass sets the decimal commas.
    "Ls = LRN + ΔL\nΔL = -10 log10[(2u^4+14u^2)/(14+2u^2+u^4)]\n"
    "u = k d_s,  d_s = 0.7 D = 4.2 m":
        "Ls = LRN + ΔL\nΔL = -10 log10[(2u^4+14u^2)/(14+2u^2+u^4)]\n"
        "u = k d_s,  d_s = 0.7 D = 4.2 m",
    "Linear Gradient": "Gradiente lineal",
    "Every Ray Turns Where c(z_t) = c(z_s)/cos(theta_0)":
        "Cada rayo gira donde c(z_t) = c(z_s)/cos(theta_0)",
    "exact circular arc, R = c0/(g cos th0)":
        "arco circular exacto, R = c0/(g cos th0)",
    "source, 100 m": "fuente, 100 m",
    "2 deg": "2°",
    "4 deg": "4°",
    "6 deg": "6°",
    "8 deg": "8°",
    "10 deg": "10°",
    "50 Hz in 100 m of Water, Receiver at 60 m":
        "50 Hz en 100 m de agua, receptor a 60 m",
    "Normal modes, range-averaged (reference)":
        "Modos normales, promediados en distancia (referencia)",
    "Parabolic equation, range-averaged":
        "Ecuación parabólica, promediada en distancia",
    "Modal grazing angle arccos(k_rm/k) [deg]":
        "Ángulo rasante modal arccos(k_rm/k) [°]",
    "Mode index m": "Índice de modo m",
    "within the paraxial band": "dentro de la banda paraxial",
    "steeper than 20 deg": "más inclinados que 20°",
    "Volume Absorption (10 C, 35 ppt, 100 m)":
        "Absorción de volumen (10 °C, 35 ppt, 100 m)",
    "Absorption coefficient alpha [dB/km]":
        "Coeficiente de absorción α [dB/km]",
    "Departure from Francois-Garrison [%]":
        "Desviación respecto a Francois-Garrison [%]",
    "Where Each Simplification Is Honest":
        "Dónde es honesta cada simplificación",
    "+/-10 % of Francois-Garrison": "+/-10 % de Francois-Garrison",
    "boric acid": "ácido bórico",
    "pure water": "agua pura",
    "MgSO4": "MgSO4",
    "Four Equations, One Profile": "Cuatro ecuaciones, un perfil",
    "Sound speed c [m/s]": "Velocidad del sonido c [m/s]",
    "Difference from UNESCO / Chen-Millero [m/s]":
        "Diferencia respecto a UNESCO / Chen-Millero [m/s]",
    "Medwin: beyond ~1000 m": "Medwin: más allá de ~1000 m",
    # Marine-mammal exposure: assessment, audiograms and the exposure
    # functions. Ponderada/Sin ponderar and "SEL por banda" mirror the
    # library renderer's own Spanish table (_plot/underwater.py).
    "Step 1: Single-Strike SEL by Band":
        "Paso 1: SEL de golpe único por bandas",
    "Band SEL [dB re 1 µPa²·s]": "SEL por banda [dB re 1 µPa²·s]",
    "Unweighted": "Sin ponderar",
    "Weighted (LF, nmfs-2024)": "Ponderada (LF, nmfs-2024)",
    "Weighted (VHF, nmfs-2024)": "Ponderada (VHF, nmfs-2024)",
    "Southall et al. (2019) Group Audiograms":
        "Audiogramas de grupo de Southall et al. (2019)",
    "Threshold [dB re 1 uPa; in-air groups re 20 uPa]":
        "Umbral [dB re 1 uPa; grupos en aire re 20 uPa]",
    "Threshold [dB re 1 uPa]": "Umbral [dB re 1 uPa]",
    "OCA (in air)": "OCA (en aire)",
    "PCA (in air)": "PCA (en aire)",
    "no published fit for LF cetaceans":
        "sin ajuste publicado para los cetáceos LF",
    "orca_audiogram (three branches)": "orca_audiogram (tres tramos)",
    "Killer Whale (Ainslie 2010, Eq. 11.159)":
        "Orca (Ainslie 2010, ec. 11.159)",
    "What a Band Level Is Compared Against":
        "Contra qué se compara un nivel de banda",
    "Exposure function E(f) = K + C - W(f) [dB re 1 uPa^2 s]":
        "Función de exposición E(f) = K + C - W(f) [dB re 1 uPa^2 s]",
    "What b = 5 Changed for LF Cetaceans":
        "Lo que b = 5 cambió para los cetáceos LF",
    "Weighting W(f) [dB]": "Ponderación W(f) [dB]",
    "nmfs-2018  (b = 2)": "nmfs-2018  (b = 2)",
    "nmfs-2024  (b = 5)": "nmfs-2024  (b = 5)",
    "Hearing group": "Grupo auditivo",
    "Onset criterion [dB re 1 uPa^2 s / dB re 1 uPa]":
        "Criterio de inicio [dB re 1 uPa^2 s / dB re 1 uPa]",
    "Impulsive Onset Criteria (NMFS 2024)":
        "Criterios de inicio impulsivos (NMFS 2024)",
    "TTS SEL (weighted)": "SEL de TTS (ponderado)",
    "AUD INJ SEL (weighted)": "SEL de AUD INJ (ponderado)",
    "TTS peak (flat)": "pico de TTS (plano)",
    "AUD INJ peak (flat)": "pico de AUD INJ (plano)",
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
    # prediction_flanking_demo / detailed_prediction_paths info boxes:
    # ratings and formulae, symbols only (decimal commas aside).
    "Rw (Dd) = 57.0 dB\nR'w = 52.2 dB\nR'w − Rw = -4.8 dB\n"
    "Dd 32.9 %   ΣFf,Fd,Df 67.1 %":
        "Rw (Dd) = 57,0 dB\nR'w = 52,2 dB\nR'w − Rw = -4,8 dB\n"
        "Dd 32,9 %   ΣFf,Fd,Df 67,1 %",
    "R' = -10 log10(Σ 10^(-Rij/10))\nR'w (C; Ctr) = 57 (-1; -7) dB":
        "R' = -10 log10(Σ 10^(-Rij/10))\nR'w (C; Ctr) = 57 (-1; -7) dB",
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
    # Outdoor propagation, barriers and refraction figures (FL3).
    "Where 95 dB of Source Power Goes (ISO 9613-2, 200 m)":
        "En qué se gastan los 95 dB de la fuente (ISO 9613-2, 200 m)",
    "LW = 95 dB (source power)": "LW = 95 dB (potencia de la fuente)",
    "LfT(DW) at the receiver": "LfT(DW) en el receptor",
    "− Adiv (divergence)": "− Adiv (divergencia)",
    "− Aatm (air)": "− Aatm (aire)",
    "− Agr (ground)": "− Agr (suelo)",
    "− Abar (barrier)": "− Abar (barrera)",
    "Dz at 500 Hz against barrier height (d = 200 m)":
        "Dz a 500 Hz frente a la altura de barrera (d = 200 m)",
    "Single edge": "Borde simple",
    "Double edge, e = 2 m": "Doble borde, e = 2 m",
    "20 dB cap (single)": "tope de 20 dB (simple)",
    "25 dB cap (double)": "tope de 25 dB (doble)",
    "Barrier height [m]": "Altura de la barrera [m]",
    "Diffraction insertion loss Dz [dB]":
        "Pérdida por inserción por difracción Dz [dB]",
    "Abar = max(Dz − Agr, 0) (Eq. (12))":
        "Abar = max(Dz − Agr, 0) (ec. (12))",
    "Agr, spent on the screened path":
        "Agr, gastada en el camino apantallado",
    "The ground effect is spent, not kept":
        "El efecto del suelo se gasta, no se conserva",
    "The coefficients part as the path grazes":
        "Los coeficientes se separan cuando el camino se hace rasante",
    "What the ground wave keeps alive (500 Hz, 50 m)":
        "Lo que la onda de suelo mantiene vivo (500 Hz, 50 m)",
    "Source = receiver height [m]  (grazing to the right)":
        "Altura de fuente = receptor [m]  (rasante hacia la derecha)",
    "Magnitude": "Módulo",
    "|Rp| — plane wave": "|Rp| — onda plana",
    "|Q| — spherical wave": "|Q| — onda esférica",
    "with Q (spherical wave)": "con Q (onda esférica)",
    "with Rp alone (plane wave)": "solo con Rp (onda plana)",
    "Wave-theoretic model: the path length alone":
        "Modelo ondulatorio: solo la longitud del camino",
    "ISO 9613-2: the path length plus the C3 factor":
        "ISO 9613-2: la longitud del camino más el factor C3",
    "Top width e [m]": "Ancho superior e [m]",
    "Edge separation e [m]": "Separación de bordes e [m]",
    "Gain over the thin screen [dB]": "Ganancia sobre la pantalla delgada [dB]",
    "Gain over the single edge [dB]": "Ganancia sobre el borde simple [dB]",
    "10 lg 3 = 4.77 dB (the C3 ceiling)":
        "10 lg 3 = 4,77 dB (el techo de C3)",
    "Where the Acoustic Shadow Starts (c₀ = 340 m/s)":
        "Dónde empieza la sombra acústica (c₀ = 340 m/s)",
    "Upward sound-speed gradient |g| [1/s]":
        "Gradiente de velocidad del sonido hacia arriba |g| [1/s]",
    "Shadow-zone distance x_shadow [m]":
        "Distancia de zona de sombra x_shadow [m]",
    "Radius of curvature Rc = c₀/|g| [m]":
        "Radio de curvatura Rc = c₀/|g| [m]",
    "representative −0.1 s⁻¹: 233 m": "−0,1 s⁻¹ representativo: 233 m",
    "the page's b = −1 m/s case: 109 m":
        "el caso b = −1 m/s de la página: 109 m",
    "The Homogeneous Limit: 500 Hz, Rigid Ground, hs = hr = 2 m":
        "El límite homogéneo: 500 Hz, suelo rígido, hs = hr = 2 m",
    "GFPE, zero gradient": "GFPE, gradiente nulo",
    "Coherent two-ray closed form": "Forma cerrada coherente de dos rayos",
    "+6 dB (coherent sum)": "+6 dB (suma coherente)",
    "Residual [dB]": "Residuo [dB]",
    "dotted lines: ±0.6 dB": "líneas de puntos: ±0,6 dB",
    # CNOSSOS-EU road corrections, gradient, surfaces and rail figures (FL3).
    "Air temperature (2.2.10)": "Temperatura del aire (2.2.10)",
    "Air temperature τ [°C]": "Temperatura del aire τ [°C]",
    "Change in line power [dB(A)]": "Cambio de la potencia de línea [dB(A)]",
    "Light (1)": "Ligeros (1)",
    "Heavy (3)": "Pesados (3)",
    "Studded tyres (2.2.6-2.2.9)": "Neumáticos con clavos (2.2.6-2.2.9)",
    "Qstud = 0.2, Ts = 4 months": "Qstud = 0,2, Ts = 4 meses",
    "Qstud = 0.5, Ts = 4 months": "Qstud = 0,5, Ts = 4 meses",
    "Junctions (2.2.17, 2.2.18)": "Cruces (2.2.17, 2.2.18)",
    "Crossing with lights": "Cruce con semáforo",
    "Roundabout": "Glorieta",
    "Distance to the junction |x| [m]": "Distancia al cruce |x| [m]",
    "CNOSSOS-EU Road-Gradient Correction (2.2.13-2.2.16)":
        "Corrección por pendiente de la vía CNOSSOS-EU (2.2.13-2.2.16)",
    "Road gradient s [%]  (negative = downhill)":
        "Pendiente de la vía s [%]  (negativa = bajada)",
    "Propulsion-noise correction [dB(A)]":
        "Corrección del ruido de propulsión [dB(A)]",
    "CNOSSOS-EU Road Surfaces (Table F-4, light vehicles)":
        "Pavimentos CNOSSOS-EU (Tabla F-4, vehículos ligeros)",
    "Surface coefficient α [dB]": "Coeficiente de pavimento α [dB]",
    "reference road surface (all speeds)":
        "pavimento de referencia (todas las velocidades)",
    "2-layer ZOAB (fine) (80-130 km/h)": "ZOAB de 2 capas (fino) (80-130 km/h)",
    "1-layer ZOAB (50-130 km/h)": "ZOAB de 1 capa (50-130 km/h)",
    "thin layer A (40-130 km/h)": "capa fina A (40-130 km/h)",
    "hard elements not in herringbone (30-60 km/h)":
        "elementos rígidos no en espiga (30-60 km/h)",
    "Components at 160 km/h, by source height":
        "Componentes a 160 km/h, por altura de fuente",
    "rolling @ 0,5 m": "rodadura @ 0,5 m",
    "traction @ 0,5 m": "tracción @ 0,5 m",
    "traction @ 4,0 m": "tracción @ 4,0 m",
    "Sound power [dB re 1 pW]": "Potencia acústica [dB re 1 pW]",
    "Each source line against speed": "Cada línea fuente frente a la velocidad",
    "Source A (0,5 m)": "Fuente A (0,5 m)",
    "Source B (4,0 m)": "Fuente B (4,0 m)",
    "Aerodynamic threshold (2.3.13)": "Umbral aerodinámico (2.3.13)",
    "Total line power [dB re 1 pW per metre]":
        "Potencia de línea total [dB re 1 pW por metro]",
    "CNOSSOS-EU Railway Source Directivity":
        "Directividad de la fuente ferroviaria CNOSSOS-EU",
    "Vertical correction of source A (2.3.16)":
        "Corrección vertical de la fuente A (2.3.16)",
    "Horizontal dipole (2.3.15)": "Dipolo horizontal (2.3.15)",
    "−20 dB along the track": "−20 dB en el eje de la vía",
    # The two directivity editions: frequency + edition number, the same in
    # Spanish (the 250 Hz pair carries no word and is never reported).
    "4 kHz, 2015/996": "4 kHz, 2015/996",
    "4 kHz, 2021/1226": "4 kHz, 2021/1226",
    # Wind-turbine apparent sound power and audibility (IEC 61400-11) (FL3).
    "Apparent Sound Power against Wind Speed (IEC 61400-11)":
        "Potencia acústica aparente frente al viento (IEC 61400-11)",
    "Hub-height wind speed [m/s]  (0,5 m/s bins)":
        "Velocidad del viento a altura de buje [m/s]  (intervalos de 0,5 m/s)",
    "Formula (29) wind speed at 10 m, z0ref = 0,05 m [m/s]":
        "Velocidad del viento a 10 m de la Fórmula (29), z0ref = 0,05 m [m/s]",
    "Valid bin (margin > 6 dB)": "Intervalo válido (margen > 6 dB)",
    "3-6 dB margin: reported with an asterisk":
        "Margen de 3-6 dB: se declara con asterisco",
    "Margin ≤ 3 dB: bin voided": "Margen ≤ 3 dB: intervalo anulado",
    "Two critical bandwidths on one page":
        "Dos anchos de banda críticos en una página",
    "IEC 61400-11 critical band (Zwicker)":
        "Banda crítica IEC 61400-11 (Zwicker)",
    "ISO 1996-2 Table C.1": "Tabla C.1 de ISO 1996-2",
    "9.5.3: fixed 20-120 Hz band": "9.5.3: banda fija de 20-120 Hz",
    "Critical bandwidth [Hz]": "Ancho de banda crítico [Hz]",
    "Tone frequency fc [Hz]": "Frecuencia del tono fc [Hz]",
    "Tone frequency [Hz]": "Frecuencia del tono [Hz]",
    "The audibility criterion, read as required tonality":
        "El criterio de audibilidad, leído como tonalidad exigida",
    "Tonality needed to be audible (ΔLa > 0)":
        "Tonalidad necesaria para ser audible (ΔLa > 0)",
    "Tonality needed to be reportable (ΔLa ≥ −3 dB)":
        "Tonalidad necesaria para ser declarable (ΔLa ≥ −3 dB)",
    "Required tonality ΔLtn [dB]": "Tonalidad exigida ΔLtn [dB]",
    # RD 1367/2007 assessment figures (FL3); the .plot() labels mirror the
    # library's own translations in src/phonometry/_plot/environment.py.
    "Band level": "Nivel de banda",
    "$L_t$ vs neighbour mean": "$L_t$ frente a la media de contiguas",
    "RD 1367/2007 tonal correction $K_t$ = 6 dB":
        "Corrección tonal $K_t$ = 6 dB (RD 1367/2007)",
    "RD 1367/2007: the Low-Frequency and Impulsive Corrections":
        "RD 1367/2007: las correcciones de baja frecuencia e impulsiva",
    "Kf, from LCeq,Ti − LAeq,Ti": "Kf, a partir de LCeq,Ti − LAeq,Ti",
    "Ki, from LAIeq,Ti − LAeq,Ti": "Ki, a partir de LAIeq,Ti − LAeq,Ti",
    "worked example: Lf = 13 dB → Kf = 3 dB":
        "ejemplo resuelto: Lf = 13 dB → Kf = 3 dB",
    "worked example: Li = 5 dB → Ki = 0 dB":
        "ejemplo resuelto: Li = 5 dB → Ki = 0 dB",
    "Lt = 10.5 dB: both methods agree":
        "Lt = 10,5 dB: los dos métodos coinciden",
    "Lt = 7.5 dB: the verdicts split":
        "Lt = 7,5 dB: los veredictos se separan",
    "Arithmetic mean of the two neighbours":
        "Media aritmética de las dos vecinas",
    "Band sound pressure level [dB]":
        "Nivel de presión acústica por banda [dB]",
    # ISO 1996-2 tonal adjustment: the Kt formula box is symbols only, and
    # reads the same in the Spanish edition.
    "Kt = 0            (dLta < 4)\nKt = dLta - 4  (4 <= dLta <= 10)\n"
    "Kt = 6            (dLta > 10)":
        "Kt = 0            (dLta < 4)\nKt = dLta - 4  (4 <= dLta <= 10)\n"
        "Kt = 6            (dLta > 10)",
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
        "Absorción perfecta por acoplamiento crítico (panel de sonido lento)",
    "Critically coupled (perfect)": "Acoplamiento crítico (perfecto)",
    "Narrow slit (over-damped)": "Ranura estrecha (sobreamortiguada)",
    "Wide slit (under-damped)": "Ranura ancha (subamortiguada)",
    "design 300 Hz": "diseño 300 Hz",
    "Reflection factor magnitude |r|": "Módulo del factor de reflexión |r|",
    "ISO 10534-1 Standing-Wave-Ratio Method":
        "Método de la razón de onda estacionaria (ISO 10534-1)",
    "s = 3 -> |r| = 0.5 -> alpha = 0.75": "s = 3 -> |r| = 0,5 -> alpha = 0,75",
    # tube_working_ranges figure (ISO 10534-2 / ASTM E2611 bores)
    "Plane-Wave Working Range of an Impedance Tube":
        "Rango de trabajo de onda plana de un tubo de impedancia",
    "100 mm bore, s = 100 mm   ·   top end: spacing 0.45 c/s":
        "tubo de 100 mm, s = 100 mm   ·   techo: separación 0,45 c/s",
    "100 mm bore, s = 100 mm (ASTM E2611)   ·   top end: spacing 0.40 c/s":
        "tubo de 100 mm, s = 100 mm (ASTM E2611)   ·   techo: "
        "separación 0,40 c/s",
    "100 mm bore, s = 50 mm   ·   top end: cut-on 0.58 c/d":
        "tubo de 100 mm, s = 50 mm   ·   techo: corte 0,58 c/d",
    "29 mm bore, s = 20 mm   ·   top end: cut-on 0.58 c/d":
        "tubo de 29 mm, s = 20 mm   ·   techo: corte 0,58 c/d",
    "splice band\n(the two tubes must agree)":
        "banda de solape\n(los dos tubos deben coincidir)",
    # standing_wave_envelope figure (ISO 10534-1 probe traverse)
    "What the Probe Carriage Traverses (500 Hz, 100 mm tube)":
        "Lo que recorre el carro de la sonda (500 Hz, tubo de 100 mm)",
    "Distance from the specimen face x [m]  (towards the source)":
        "Distancia a la cara de la probeta x [m]  (hacia la fuente)",
    r"Envelope level $20\log_{10}(|p(x)|/A)$ [dB]":
        r"Nivel de la envolvente $20\log_{10}(|p(x)|/A)$ [dB]",
    r"rigid wall  |r| = 1  ($\Delta L \to \infty$)":
        r"pared rígida  |r| = 1  ($\Delta L \to \infty$)",
    r"the worked sample  |r| = 0.5  ($\Delta L$ = 9.54 dB)":
        r"la muestra resuelta  |r| = 0,5  ($\Delta L$ = 9,54 dB)",
    r"near-anechoic  |r| = 0.1  ($\Delta L$ = 1.74 dB)":
        r"casi anecoica  |r| = 0,1  ($\Delta L$ = 1,74 dB)",
    "the same, with tube loss (k₀'' = 0.013 Np/m)":
        "la misma, con la atenuación del tubo (k₀'' = 0,013 Np/m)",
    "minima at 12, 46 and 81 cm: -5.99, -5.92, -5.85 dB\n"
    "(read the nearest one, and extrapolate to x = 0)":
        "mínimos en 12, 46 y 81 cm: -5,99, -5,92, -5,85 dB\n"
        "(leer el más cercano y extrapolar a x = 0)",
    "specimen face  x = 0": "cara de la probeta  x = 0",
    # sound_absorption_inversion figure (ISO 354 worked example)
    "ISO 354: What the Sabine Inversion Consumes":
        "ISO 354: lo que consume la inversión de Sabine",
    "Reverberation time [s]": "Tiempo de reverberación [s]",
    r"$T_1$  empty room": r"$T_1$  sala vacía",
    r"$T_2$  specimen installed": r"$T_2$  probeta instalada",
    r"$A_1$  empty room": r"$A_1$  sala vacía",
    r"$A_2$  with specimen": r"$A_2$  con la probeta",
    r"$A_2 - A_1$  the specimen": r"$A_2 - A_1$  la probeta",
    r"Equivalent absorption area [m$^2$]":
        r"Área de absorción equivalente [m$^2$]",
    r"Band height as $\alpha_s = (A_2 - A_1)/S$":
        r"Altura de la franja como $\alpha_s = (A_2 - A_1)/S$",
    "the two decays nearly coincide:\n0.6 s out of 9 s":
        "las dos caídas casi coinciden:\n0,6 s de 9 s",
    # effective_kappa figure (ISO 9053-2 Annex A)
    "ISO 9053-2 Annex A Heat-Conduction Correction":
        "Corrección por conducción térmica del Anexo A de ISO 9053-2",
    "Piston frequency f [Hz]  (ISO 9053-2 Clause 6.2: 1 Hz to 4 Hz)":
        "Frecuencia del pistón f [Hz]  (ISO 9053-2, apartado 6.2: de 1 a 4 Hz)",
    r"Effective ratio of specific heats $\kappa'$":
        r"Relación efectiva de calores específicos $\kappa'$",
    "Bias in R if the adiabatic value is used [%]":
        "Error en R si se usa el valor adiabático [%]",
    r"adiabatic $\kappa$ = 1.4008": r"$\kappa$ adiabática = 1,4008",
    "Annex A.3 example (2 Hz, 1.370)": "ejemplo del Anexo A.3 (2 Hz, 1,370)",
    # flow_resistivity_window figure (the sigma-d design window)
    "50 mm hard-backed layer": "Capa de 50 mm con respaldo rígido",
    r"Normal-incidence absorption $\alpha$":
        r"Absorción a incidencia normal $\alpha$",
    "The design window": "La ventana de diseño",
    r"2 kPa s/m$^2$  (too transparent)":
        r"2 kPa s/m$^2$  (demasiado transparente)",
    r"8 kPa s/m$^2$  (window)": r"8 kPa s/m$^2$  (ventana)",
    r"20 kPa s/m$^2$  (window)": r"20 kPa s/m$^2$  (ventana)",
    r"33 kPa s/m$^2$  (window)": r"33 kPa s/m$^2$  (ventana)",
    r"100 kPa s/m$^2$  (too reflecting)":
        r"100 kPa s/m$^2$  (demasiado reflectante)",
    r"$\alpha_{dif}$ at 1 kHz": r"$\alpha_{dif}$ a 1 kHz",
    # porous_model_comparison figure (Delany-Bazley / Miki / JCA).
    # The model names are their authors' names and Re/-Im is notation, so
    # those legend labels read the same in Spanish: identity entries.
    "Delany-Bazley": "Delany-Bazley",
    "Miki": "Miki",
    "Delany-Bazley, Re": "Delany-Bazley, Re",
    "Delany-Bazley, -Im": "Delany-Bazley, -Im",
    "Miki, Re": "Miki, Re",
    "Miki, -Im": "Miki, -Im",
    r"Characteristic impedance, $\sigma$ = 20 kPa s/m$^2$":
        r"Impedancia característica, $\sigma$ = 20 kPa s/m$^2$",
    "Where the extrapolation fails": "Donde falla la extrapolación",
    "Delany-Bazley returns a NEGATIVE resistance\n"
    "below 74.6 Hz: a passive layer generating energy":
        "Delany-Bazley devuelve una resistencia NEGATIVA\n"
        "por debajo de 74,6 Hz: una capa pasiva que genera energía",
    r"Re$(Z_s)/(\rho_0 c_0)$,  50 mm hard-backed layer":
        r"Re$(Z_s)/(\rho_0 c_0)$,  capa de 50 mm con respaldo rígido",
    # oblique_absorption figure (Paris integral against local reaction)
    # Measured on the drawn figure: repeating "reacción" ran 9 px past the
    # left canvas edge; once is enough for the pair.
    "The integrand: solid bulk-reacting, dashed locally reacting":
        "El integrando: continua reacción volumétrica, discontinua local",
    "The two averages, and the tube": "Los dos promedios, y el tubo",
    "Incidence angle θ [°]": "Ángulo de incidencia θ [°]",
    "250 Hz, bulk": "250 Hz, volumétrica",
    "500 Hz, bulk": "500 Hz, volumétrica",
    "1000 Hz, bulk": "1000 Hz, volumétrica",
    "2000 Hz, bulk": "2000 Hz, volumétrica",
    "78° truncation": "truncamiento de 78°",
    r"$\alpha_{dif}$  bulk (Paris integral)":
        r"$\alpha_{dif}$  volumétrica (integral de Paris)",
    r"$\alpha_{st}$  locally reacting (closed form)":
        r"$\alpha_{st}$  reacción local (forma cerrada)",
    r"$\alpha(0°)$  what the tube reads": r"$\alpha(0°)$  lo que lee el tubo",
    "0.951 ceiling of the closed form": "techo de 0,951 de la forma cerrada",
    # biot_waves figure (Allard & Atalla Table 6.1 glass wool)
    r"airborne  $|\mu_a|$": r"del aire  $|\mu_a|$",
    r"frame-borne  $|\mu_b|$": r"del esqueleto  $|\mu_b|$",
    "fluid and frame move together":
        "el fluido y el esqueleto se mueven juntos",
    r"$|\mu|$ = fluid / frame displacement":
        r"$|\mu|$ = desplazamiento fluido / esqueleto",
    "root swap, 495 Hz": "intercambio de raíces, 495 Hz",
    r"$|\mu_a| \geq$ 42 everywhere: the fluid moves and the frame barely does":
        r"$|\mu_a| \geq$ 42 en toda la banda: se mueve el fluido y el "
        "esqueleto apenas",
    # slow_sound_dispersion figure (loaded slit phase speed)
    "Slow Sound: the Phase Speed Inside a Loaded Slit":
        "Sonido lento: la velocidad de fase dentro de una ranura cargada",
    r"Phase speed in the slit  $c_\mathrm{eff}/c_0$":
        r"Velocidad de fase en la ranura  $c_\mathrm{eff}/c_0$",
    "loaded slit": "ranura cargada",
    "empty slit": "ranura vacía",
    "an empty 30 mm slit is a quarter wave at 2858 Hz":
        "una ranura vacía de 30 mm es un cuarto de onda a 2858 Hz",
    "branch closed above the\nresonator resonance (485 Hz)":
        "rama cerrada por encima de la\nresonancia del resonador (485 Hz)",
    "design point: 0.11 $c_0$ = 37 m/s,\n"
    "so the 30 mm depth is a quarter wave at 308 Hz":
        "punto de diseño: 0,11 $c_0$ = 37 m/s,\n"
        "así que los 30 mm de profundidad son un cuarto de onda a 308 Hz",
    # critical_coupling_impedance figure (the sweep and the locus)
    "What the design solver solves": "Lo que resuelve el diseño",
    "The locus over 150-500 Hz": "El lugar geométrico de 150 a 500 Hz",
    "Slit height h [mm]": "Altura de ranura h [mm]",
    r"Normalised surface impedance $z$ at 300 Hz":
        r"Impedancia superficial normalizada $z$ a 300 Hz",
    "solved h = 0.978 mm:\nRe(z) = 1 and Im(z) = 0 together":
        "h resuelta = 0,978 mm:\nRe(z) = 1 e Im(z) = 0 a la vez",
    # Below the gate's radar ("at" is under three letters), but visible on
    # the drawn twin axis; and the mathtext skips the decimal-comma pass, so
    # the lambda/4 ruler needs its comma from the table.
    r"$\alpha$ at 300 Hz": r"$\alpha$ a 300 Hz",
    r"$\lambda/4$ = 17.2 cm": r"$\lambda/4$ = 17,2 cm",
    r"$\alpha(0°)$ at 500 Hz": r"$\alpha(0°)$ a 500 Hz",
    r"$\alpha(0°)$ at 1000 Hz": r"$\alpha(0°)$ a 1000 Hz",
    "0.08 $c_0$ at 379 Hz": "0,08 $c_0$ a 379 Hz",
    r"7.4 m$^2$ / 10.8 m$^2$ = 0.69": r"7,4 m$^2$ / 10,8 m$^2$ = 0,69",
    r"$\alpha$ = 0.8": r"$\alpha$ = 0,8",
    "0.6 h  over-damped": "0,6 h  sobreamortiguada",
    "h  critically coupled": "h  con acoplamiento crítico",
    "1.7 h  under-damped": "1,7 h  subamortiguada",
    "matched  1 + 0j": "adaptado  1 + 0j",
    # graded_slit_absorber figure (chain of resonators)
    "What a Chain of Resonators Buys, and What It Costs":
        "Lo que compra una cadena de resonadores, y lo que cuesta",
    "one cell, L = 30 mm": "una celda, L = 30 mm",
    "four graded, L = 120 mm": "cuatro graduados, L = 120 mm",
    "four identical, L = 120 mm": "cuatro idénticos, L = 120 mm",
    "22 Hz above 0.8": "22 Hz por encima de 0,8",
    "38 Hz above 0.8": "38 Hz por encima de 0,8",
    "51 Hz above 0.8": "51 Hz por encima de 0,8",
    # sheet_transfer_impedance figure (Maa MPP against its cavity)
    "Where a Resonant Sheet Resonates, and How Well It Absorbs":
        "Dónde resuena una lámina resonante, y cuánto absorbe",
    r"panel resistance $r$": r"resistencia del panel $r$",
    r"panel reactance $x$": r"reactancia del panel $x$",
    r"cavity $\cot(\omega D/c_0)$,  D = 60 mm":
        r"cámara $\cot(\omega D/c_0)$,  D = 60 mm",
    r"Normalised transfer impedance $z/\rho_0 c_0$":
        r"Impedancia de transferencia normalizada $z/\rho_0 c_0$",
    r"Absorption $\alpha$ of the stack":
        r"Absorción $\alpha$ de la pila de capas",
    "the reactances meet at 677 Hz\nthere r = 1.53, so 4r/(1+r)² = 0.96":
        "las reactancias se cruzan en 677 Hz\nallí r = 1,53, así que "
        "4r/(1+r)² = 0,96",
    # --- Tier-1 animation labels ---
    "tone burst": "ráfaga de tono",
    # The detector names stay English, as everywhere else in these tables
    # ("Nivel Fast", "envolvente exponencial Fast", "Impulse (35 ms/1,5 s)"):
    # the same frame captions the capacitor with "se muestra Fast" and labels
    # the dials F / S / I, so translating them here gave one weighting two
    # names a few pixels apart.
    "Fast (125 ms)": "Fast (125 ms)",
    "Slow (1000 ms)": "Slow (1000 ms)",
    "Impulse (35 ms / 1.5 s)": "Impulse (35 ms / 1,5 s)",
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
    # anim_loudness_gating: the double gate deciding block by block.
    "The two passes of the EBU R 128 gate (BS.1770-5)":
        "Las dos pasadas de la puerta de EBU R 128 (BS.1770-5)",
    "absolute gate, -70 LUFS": "puerta absoluta, -70 LUFS",
    "Blocks per LU": "Bloques por LU",
    "Loudness range (Tech 3342)": "Rango de sonoridad (3342)",
    "block, counted": "bloque, contado",
    "block, gated out": "bloque, excluido por la puerta",
    "short-term (3 s)": "corto plazo (3 s)",
    "Integrated I (gated)": "Integrada I (con puerta)",
    "Ungated energy mean": "Media energética",
    "What the gate is worth": "Lo que aporta la puerta",
    "Blocks gated out": "Bloques excluidos",
    "The loudness range gates 10 LU deeper and reads the 10th to 95th "
    "percentile spread":
        "El rango de sonoridad usa una puerta 10 LU más profunda: "
        "percentiles 10 a 95",
    "154 of the 597 blocks never counted: the quiet opening and the fade-out":
        "154 de los 597 bloques no cuentan nunca: la entrada silenciosa y "
        "el desvanecimiento",
    "Nothing loud has played yet, so the relative gate sits low and every "
    "block counts":
        "Aún no ha sonado nada fuerte: la puerta relativa está baja y todo "
        "cuenta",
    "Louder material raises the relative gate, and blocks that were counted "
    "stop counting":
        "El material más fuerte sube la puerta relativa, y bloques que "
        "contaban dejan de contar",
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
    # impact_prediction_terms info box: formulae, symbols only.
    "L'n,w = Ln,w,eq - ΔLw + K\n"
    "Ln,w,eq = 164 - 35 log10(m'/m'0) = 76.2 dB\nL'n,w = 45.2 dB → 45 dB":
        "L'n,w = Ln,w,eq - ΔLw + K\n"
        "Ln,w,eq = 164 - 35 log10(m'/m'0) = 76,2 dB\nL'n,w = 45,2 dB → 45 dB",
    "EN 12354-4 Radiated Sound Power (Annex G)":
        "Potencia acústica radiada EN 12354-4 (Anexo G)",
    "radiated $L_W$ per octave": "$L_W$ radiada por octava",
    "Radiated sound power level [dB re 1 pW]":
        "Nivel de potencia acústica radiada [dB re 1 pW]",
    "Predicted Single-Panel Insulation Rated per ISO 717-1":
        "Aislamiento previsto de panel simple evaluado según ISO 717-1",
    "predicted R (Sharp)": "R previsto (Sharp)",
    # Shared ylabel of single_panel_rating, plateau_transmission_loss,
    # orthotropic_transmission_loss and panel_insulation_concept; "pérdida
    # por transmisión" as the panel-sound-insulation page words it.
    r"Sound reduction index $R$ (transmission loss $TL$) [dB]":
        r"Índice de reducción acústica $R$ (pérdida por transmisión $TL$) [dB]",
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
    # silencer_insertion_loss: one TL, as many ILs as installations
    "One chamber, one TL, and as many insertion losses as installations":
        "Una cámara, una TL, y tantas pérdidas por inserción "
        "como instalaciones",
    "Transmission loss (anechoic ports)":
        "Pérdida de transmisión (puertos anecoicos)",
    "Insertion loss, stiff source ($Z_s = 20\\,\\rho c/S$)":
        "Pérdida por inserción, fuente rígida ($Z_s = 20\\,\\rho c/S$)",
    "Insertion loss, matched source ($Z_s = \\rho c/S$)":
        "Pérdida por inserción, fuente adaptada ($Z_s = \\rho c/S$)",
    "near 180 Hz the chamber and the open end it\ndischarges through "
    "resonate together: the\ninstallation is LOUDER with the silencer in it":
        "cerca de 180 Hz la cámara y el extremo abierto por el\nque descarga "
        "resuenan juntos: la instalación es MÁS\nRUIDOSA con el silenciador "
        "puesto",
    "near 645 Hz, just past the half-wave trough\nof the TL, both "
    "insertion losses dip again":
        "cerca de 645 Hz, justo tras el cero de media onda\nde la TL, las "
        "dos pérdidas por inserción vuelven a caer",
    # silencer_selection: reactive against dissipative on one axis
    "Choosing the family: where each one is worth having":
        "Elegir la familia: dónde compensa cada una",
    "Reactive: 0.3 m expansion chamber, m = 4":
        "Reactivo: cámara de expansión de 0,3 m, m = 4",
    "Reactive: Helmholtz branch tuned to 100 Hz":
        "Reactivo: rama de Helmholtz sintonizada a 100 Hz",
    "Dissipative: five-airway splitter unit, 5 ft":
        "Disipativo: unidad de bafles de cinco pasos, 5 ft",
    # silencer_extended_tube: the buried quarter-wave branches
    "Extended-tube chamber: quarter-wave branches buried inside":
        "Cámara de tubos extendidos: ramas de cuarto de onda "
        "enterradas dentro",
    "Plain chamber, 0.4 m": "Cámara simple, 0,4 m",
    "Inlet extension L/4 = 0.10 m": "Extensión de entrada L/4 = 0,10 m",
    "Inlet L/4 and outlet L/2 = 0.20 m":
        "Entrada L/4 y salida L/2 = 0,20 m",
    "c/2L = 429 Hz: the plain chamber is transparent (0.0 dB),\n"
    "the L/4 inlet gives 5.1 dB and the L/4 + L/2 pair 74 dB":
        "c/2L = 429 Hz: la cámara simple es transparente (0,0 dB),\n"
        "la entrada L/4 da 5,1 dB y el par L/4 + L/2, 74 dB",
    # duct_attenuation_elements: the four element panels
    "(a) 36 × 24 in run, 5 ft": "(a) Tramo de 36 × 24 in, 5 ft",
    "(b) One bend, W = 24 in": "(b) Un codo, W = 24 in",
    "(c) Open end, 300 mm flush": "(c) Extremo abierto, 300 mm enrasado",
    "(d) Splitter silencer": "(d) Silenciador de bafles",
    "Bare": "Desnudo",
    "Externally wrapped": "Con manta exterior",
    "25 mm lining, insertion loss":
        "Revestimiento de 25 mm, pérdida por inserción",
    "25 mm lining + side walls":
        "Revestimiento de 25 mm + paredes laterales",
    "Square, lined": "Cuadrado, revestido",
    "Square, bare": "Cuadrado, desnudo",
    "Square, vanes": "Cuadrado, con deflectores",
    "Round, bare": "Circular, desnudo",
    "Geometry-only model, 5 ft, 5 airways":
        "Modelo solo geométrico, 5 ft, 5 pasos",
    "The unit Long's sheet specifies":
        "La unidad que especifica la hoja de Long",
    "10.2 dB and 12.4 dB of low-frequency\nperformance the model "
    "cannot see":
        "10,2 dB y 12,4 dB de rendimiento en baja\nfrecuencia que el modelo "
        "no puede ver",
    # duct_sheet_verification legends (titles carry the computed worst Δ)
    "Long's printed row": "La fila impresa de Long",
    "the library": "la biblioteca",
    # duct_regenerated_noise: the two velocity laws
    "Silencer self-noise (Long Eq. 14.31)":
        "Ruido propio del silenciador (Long Ec. 14.31)",
    "Diffuser sound power (Long Eqs. 13.27-13.33)":
        "Potencia acústica del difusor (Long Ecs. 13.27-13.33)",
    "Airway velocity": "Velocidad en el paso de aire",
    "24 × 24 in device": "Dispositivo de 24 × 24 in",
    "Regenerated $L_W$ [dB re 1 pW]": "$L_W$ regenerado [dB re 1 pW]",
    "55 lg(V/V₀): 16.6 dB per\ndoubling, every band":
        "55 lg(V/V₀): 16,6 dB por\nduplicación, en cada banda",
    "peak band f_P = 48.8 U_G moves up one octave per\ndoubling of the "
    "face velocity; ASHRAE Table 9 caps\nthe neck of an RC 30 supply "
    "outlet at 2.2 m/s":
        "la banda de pico f_P = 48,8 U_G sube una octava por\nduplicación "
        "de la velocidad en la cara; la Tabla 9 de\nASHRAE limita el cuello "
        "de una salida de impulsión RC 30 a 2,2 m/s",
    # fan_sound_power: the efficiency staircase and the blade tone
    "The fan row: 5000 cfm at 2 in w.g., forward curved (Long Eq. 13.1)":
        "La fila del ventilador: 5000 cfm a 2 in c.a., álabes curvados "
        "hacia delante (Long Ec. 13.1)",
    "Casing attenuation (Table 13.8)":
        "Atenuación de la carcasa (Tabla 13.8)",
    "the same fan with its blade tone at 2 kHz":
        "el mismo ventilador con su tono de álabes en 2 kHz",
    "static efficiency [% of peak]": "rendimiento estático [% del máximo]",
    "the C_BFI increment is 2 dB and lands whole in the\noctave "
    "containing the blade tone: 500 Hz at 1200 rev/min\n× 24 blades, "
    "2 kHz if the fan is selected faster":
        "el incremento C_BFI es de 2 dB y cae entero en la\noctava del tono "
        "de álabes: 500 Hz a 1200 rev/min\n× 24 álabes, 2 kHz si el "
        "ventilador se elige más rápido",
    # hvac_elbow_flow_noise: the bend chosen and the noise it regenerates
    "Elbow insertion loss (ASHRAE Table 8.11)":
        "Pérdida por inserción del codo (ASHRAE Tabla 8.11)",
    "Flow-generated sound power (VDI 2081, Bies Eq. 8.252)":
        "Potencia acústica del flujo (VDI 2081, Bies Ec. 8.252)",
    "Insertion loss [dB per bend]": "Pérdida por inserción [dB por codo]",
    "Mitred bend, U = 10 m/s": "Codo a inglete, U = 10 m/s",
    "W / lambda = 1\n(W = 0.30 m, 1143 Hz)":
        "W / lambda = 1\n(W = 0,30 m, 1143 Hz)",
    "one bend at 10 m/s beats\na straight run at 20 m/s\nbelow 500 Hz":
        "un codo a 10 m/s supera a\nun tramo recto a 20 m/s\n"
        "por debajo de 500 Hz",
    # enclosure_required_tl: Norton problem 4.16
    "What the enclosure panels have to be: Norton problem 4.16":
        "Lo que tienen que ser los paneles del encapsulado: "
        "problema 4.16 de Norton",
    "Target IL = $L_{p1}$ − NC 45": "IL objetivo = $L_{p1}$ − NC 45",
    "Interior correction $10\\,\\lg(S_E/R_i)$":
        "Corrección interior $10\\,\\lg(S_E/R_i)$",
    "Required panel R (Norton)": "R necesaria del panel (Norton)",
    "Required panel R (Bies default)":
        "R necesaria del panel (Bies, por omisión)",
    "Norton's printed answer": "La respuesta impresa de Norton",
    "the peak: the lining works and\nthe compressor is still loud":
        "el máximo: el revestimiento funciona y\nel compresor sigue "
        "siendo ruidoso",
    # room_to_room_partitions: Norton problem 4.21
    "What the receiving room adds to (or takes from) the wall":
        "Lo que el recinto receptor añade a la pared (o le quita)",
    "Why: the same room, band by band":
        "El porqué: el mismo recinto, banda a banda",
    "Two 13 mm wallboards, 64 mm gap":
        "Dos placas de yeso de 13 mm, cámara de 64 mm",
    "125 mm plastered brick": "Ladrillo enfoscado de 125 mm",
    "Double brick, 50 mm cavity": "Doble hoja de ladrillo, cámara de 50 mm",
    "the three curves are one curve:\nthe gap belongs to the room, "
    "not to the wall":
        "las tres curvas son una sola:\nla diferencia es del recinto, "
        "no de la pared",
    "crossing: below it the wall delivers less than its TL,\n"
    "above it, more":
        "cruce: por debajo la pared rinde menos que su TL,\n"
        "por encima, más",
    "Receiving-room absorption $S_2\\alpha_2$":
        "Absorción del recinto receptor $S_2\\alpha_2$",
    "Area [m²]": "Área [m²]",
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
    # channel_weight_map: BS.1770 Annex 3 as a map of the layout
    "BS.1770 Annex 3: the weight is a property of the loudspeaker position":
        "Anexo 3 de BS.1770: la ponderación es una propiedad de la "
        "posición del altavoz",
    "1.0 everywhere else": "1,0 en todo lo demás",
    "60° ≤ |azimuth| ≤ 120°,  |elevation| < 30°":
        "60° ≤ |acimut| ≤ 120°,  |elevación| < 30°",
    "Azimuth [°]": "Acimut [°]",
    "Elevation [°]": "Elevación [°]",
    # true_peak_intersample: the intersample peak and the closed-form bound
    "A 12 kHz tone at π/4: every sample misses the peak":
        "Un tono de 12 kHz en π/4: ninguna muestra cae en el pico",
    "band-limited reconstruction": "reconstrucción de banda limitada",
    "samples at 48 kHz": "muestras a 48 kHz",
    "4× oversampled grid": "rejilla sobremuestreada 4×",
    "Amplitude [FS]": "Amplitud [FS]",
    "Worst-case under-read, 20 lg cos(π f_norm / n)":
        "Infravaloración de peor caso, 20 lg cos(π f_norm / n)",
    "Under-read [dB]": "Infravaloración [dB]",
    "Tone frequency / sampling rate":
        "Frecuencia del tono / frecuencia de muestreo",
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
    # regularized_inversion: vocabulary from the Spanish page (respuesta
    # medida, filtro inverso, producto ecualizado, banda ecualizada; "pone
    # tope" for caps, "reforzar" for boosting).
    "Regularized Spectral Inversion (Kirkeby Frequency-Dependent "
    "Regularization)":
        "Inversión espectral regularizada (regularización de Kirkeby "
        "dependiente de la frecuencia)",
    "Measured response $|H|$": "Respuesta medida $|H|$",
    "Inverse filter $|H_{\\mathrm{inv}}|$":
        "Filtro inverso $|H_{\\mathrm{inv}}|$",
    "Equalized $|H \\cdot H_{\\mathrm{inv}}|$":
        "$|H \\cdot H_{\\mathrm{inv}}|$ ecualizado",
    "Equalized band (200 Hz - 4 kHz)": "Banda ecualizada (200 Hz - 4 kHz)",
    "unity in-band; outside, the frequency-dependent\n"
    "regularization caps the gain instead of boosting noise":
        "unidad en banda; fuera, la regularización dependiente de la\n"
        "frecuencia pone tope a la ganancia en vez de reforzar el ruido",
    # shaped_sweep: "barrido conformado" and "tiempo de permanencia", as the
    # Spanish page has them.
    "Shaped Sweep with an Arbitrary Target Spectrum (Group-Delay Synthesis)":
        "Barrido conformado con un espectro objetivo arbitrario "
        "(síntesis por retardo de grupo)",
    "Welch spectrum of the sweep": "Espectro de Welch del barrido",
    "Pink target (-3 dB per octave)": "Objetivo rosa (-3 dB por octava)",
    "Sweep band (50 Hz - 5 kHz)": "Banda del barrido (50 Hz - 5 kHz)",
    "Level re in-band max [dB]": "Nivel respecto al máximo en banda [dB]",
    "nearly constant envelope: the energy shaping lives\n"
    "in the dwell time, not in the amplitude":
        "envolvente casi constante: la conformación de energía\n"
        "vive en el tiempo de permanencia, no en la amplitud",
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
    # anim_fdtd_dispersion
    "Numerical dispersion: one pulse, three grids (2D FDTD)":
        "Dispersión numérica: un pulso, tres mallas (FDTD 2D)",
    "The closed form": "La forma cerrada",
    "Cells per wavelength": "Celdas por longitud de onda",
    "Speed error [%]": "Error de velocidad [%]",
    "group (a pulse)": "grupo (un pulso)",
    "phase (the rule)": "fase (la regla)",
    # anim_fdtd_critical_angle
    "The critical grazing angle: a fast seabed and a slow one (2D FDTD)":
        "El ángulo rasante crítico: un fondo rápido y uno lento (FDTD 2D)",
    "Into the bed": "Hacia el fondo",
    "sand": "arena",
    "mud": "fango",
    "an evanescent skin: bright, and it carries nothing away":
        "una piel evanescente: brilla y no se lleva nada",
    "no critical angle: every angle leaks":
        "sin ángulo crítico: se fuga en todos los ángulos",
    ("Solid: the flux measured into the bed; dashed: "
     "(1 \u2212 |R|\u00b2) sin\u00b2\u03c8. Each curve on its own maximum; "
     "the field compensated for spreading."):
        ("Continuo: el flujo medido hacia el fondo; discontinuo: "
         "(1 \u2212 |R|\u00b2) sin\u00b2\u03c8. Cada curva con su máximo; "
         "campo compensado del ensanchamiento."),
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
    # anim_fdtd_side_branch
    "The quarter-wave side branch, on and off tune (2D FDTD)":
        "La rama lateral de cuarto de onda, en sintonía y fuera (FDTD 2D)",
    "closed stub, built ℓ = 300 mm":
        "tubo cerrado, construido con ℓ = 300 mm",
    "on tune": "en sintonía",
    "off tune": "fuera de sintonía",
    "closed-end\npressure [Pa]": "presión en el\nextremo cerrado [Pa]",
    "Rigid walls, anechoic ends, incident amplitude 1; the charge "
    "bandwidth f/Q is percent-wide, the lossless notch hertz-wide.":
        "Paredes rígidas, extremos anecoicos, amplitud incidente 1; el "
        "ancho de carga f/Q es porcentual; la muesca sin pérdidas, de "
        "hercios.",
    # anim_fdtd_absorption_placement
    "Where the absorption sits: same total, two decays (2D FDTD)":
        "Dónde se coloca la absorción: mismo total, dos decaimientos "
        "(FDTD 2D)",
    "Absorption spread over all four edges":
        "Absorción repartida por los cuatro bordes",
    "The same total, floor and ceiling only":
        "El mismo total, solo en suelo y techo",
    "what survives runs parallel to the absorber":
        "lo que sobrevive corre paralelo al absorbente",
    "each frame on its own scale, 25 dB":
        "cada fotograma con su propia escala, 25 dB",
    "all four edges": "los cuatro bordes",
    "floor + ceiling": "suelo + techo",
    "Total energy [dB]": "Energía total [dB]",
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
    # simply_supported is the boundary= literal, kept as typed.
    "Baffled plate (simply_supported)": "Placa con bafle (simply_supported)",
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
    # Levels: the sound-level-meter walkthrough, energy averaging, percentiles,
    # peak detection, exposure and the C - A spectrum (signals/levels).
    "Level [dB re 20 uPa]": "Nivel [dB re 20 uPa]",
    "Band level [dB re 20 uPa]": "Nivel de banda [dB re 20 uPa]",
    "Band level [dB]": "Nivel de banda [dB]",
    "What step 4 reports, drawn on the recording it read":
        "Lo que informa el paso 4, dibujado sobre la grabación que leyó",
    "LAF(t), Fast A-weighted level": "LAF(t), nivel ponderado A con Fast",
    "the 1 s event": "el evento de 1 s",
    "One-third-octave spectrum of the same ten seconds":
        "Espectro en tercios de octava de los mismos diez segundos",
    "the 1 kHz band holds the event": "la banda de 1 kHz contiene el evento",
    "pink background: equal energy per band":
        "fondo rosa: la misma energía en cada banda",
    "first half": "primera mitad",
    "second half": "segunda mitad",
    "Two equal periods, 60 dB and 80 dB": "Dos periodos iguales, 60 dB y 80 dB",
    "The error of averaging decibels, and it never changes sign":
        "El error de promediar decibelios, y nunca cambia de signo",
    "Standard deviation of the levels [dB]":
        "Desviación típica de los niveles [dB]",
    "Leq minus the arithmetic dB mean [dB]":
        "Leq menos la media aritmética en dB [dB]",
    "Gaussian spread: 0.115 sigma^2": "dispersión gaussiana: 0,115 sigma^2",
    "two levels, one sigma either side": "dos niveles, a una sigma a cada lado",
    "levels spread over 10 dB": "niveles con una dispersión de 10 dB",
    "Two noises with the same LAeq": "Dos ruidos con el mismo LAeq",
    "steady noise": "ruido estacionario",
    "quiet background, three events": "fondo silencioso, tres eventos",
    "Their exceedance curves": "Sus curvas de excedencia",
    "Percentage of the time exceeded [%]":
        "Porcentaje del tiempo superado [%]",
    "The crest falls between two samples": "La cresta cae entre dos muestras",
    "the continuous 8 kHz tone": "el tono continuo de 8 kHz",
    "its samples at 48 kHz (6 per cycle)": "sus muestras a 48 kHz (6 por ciclo)",
    "true peak": "pico verdadero",
    "Worst case over the phase of the tone": "Peor caso sobre la fase del tono",
    "Samples per cycle": "Muestras por ciclo",
    "Under-read of the peak [dB]": "Subestimación del pico [dB]",
    "Equal-energy exchange: every point on a line is the same daily exposure":
        "Canje de igual energía: todos los puntos de una línea son la misma "
        "exposición diaria",
    "Exposure duration [h]": "Duración de la exposición [h]",
    # The dose_exchange tick labels: "min" is the SI symbol in Spanish too.
    "1 min": "1 min",
    "5 min": "5 min",
    "15 min": "15 min",
    "30 min": "30 min",
    "+3 dB for every halving of the duration":
        "+3 dB por cada reducción a la mitad de la duración",
    "90 dB(A) for 8 h = 3.20 Pa2h": "90 dB(A) durante 8 h = 3,20 Pa2h",
    "Every detector under-reads every short event":
        "Todo detector subestima todo evento corto",
    "Toneburst duration [ms]": "Duración de la ráfaga tonal [ms]",
    "Peak level re the steady reading [dB]":
        "Nivel de pico respecto a la lectura estacionaria [dB]",
    "IEC 61672-1 Table 4, Equation (7)": "IEC 61672-1 Tabla 4, Ecuación (7)",
    # The detector names are the `mode=` literals of time_weighting, so only
    # the parenthesis is translated.
    "fast (measured)": "fast (medida)",
    "slow (measured)": "slow (medida)",
    "impulse (measured)": "impulse (medida)",
    "A 50 Hz rumble under a light hiss":
        "Un retumbo de 50 Hz bajo un siseo ligero",
    "Broadband pink noise": "Ruido rosa de banda ancha",
    "Z-weighted bands": "Bandas ponderadas Z",
    "C-weighted bands": "Bandas ponderadas C",
    "A-weighted bands": "Bandas ponderadas A",
    # Filters: class masks, stop-band leakage, pole placement, architectures
    # and the parametric EQ (signals/filters).
    "The same 1 kHz octave band, order 6, on the IEC 61260-1 acceptance mask":
        "La misma banda de octava de 1 kHz, orden 6, sobre la máscara de "
        "aceptación de IEC 61260-1",
    "Butterworth: passes   (overall_class = 1)":
        "Butterworth: cumple   (overall_class = 1)",
    # The four verdicts read as verb phrases in Spanish: it keeps them
    # parallel, and the noun forms overrun the panel the layout sized in
    # English (the figure is laid out before this table is applied).
    "Chebyshev I: passband ripple   (overall_class = None)":
        "Chebyshev I: riza la banda de paso   (overall_class = None)",
    "Elliptic: ripple in both bands   (overall_class = None)":
        "Elíptico: riza las dos bandas   (overall_class = None)",
    "Bessel: roll-off too slow   (overall_class = None)":
        "Bessel: cae demasiado despacio   (overall_class = None)",
    # The corridor panels and the EQ cascade are drawn by the library
    # renderer, so they are worded exactly as src/phonometry/_plot/filters.py
    # translates them.
    "Class 1 pass corridor": "Corredor de aceptación clase 1",
    "Class 2 pass corridor": "Corredor de aceptación clase 2",
    "Out of tolerance": "Fuera de tolerancia",
    "Relative attenuation [dB]": "Atenuación relativa [dB]",
    "Normalised frequency $f\\,/\\,f_m$": "Frecuencia normalizada $f\\,/\\,f_m$",
    "Measured $\\Delta A$": "$\\Delta A$ medida",
    "Parametric EQ response (Audio EQ Cookbook)":
        "Respuesta del EQ paramétrico (Audio EQ Cookbook)",
    "Cascade": "Cascada",
    "lowshelf 100 Hz": "shelf de graves 100 Hz",
    "peaking 1000 Hz": "campana 1000 Hz",
    "highshelf 8000 Hz": "shelf de agudos 8000 Hz",
    "A 1 kHz tone at 100 dB over a pink-noise floor":
        "Un tono de 1 kHz a 100 dB sobre un fondo de ruido rosa",
    "the noise actually present": "el ruido realmente presente",
    "bands on the skirt are measuring\nthe filter, not the sound":
        "las bandas de la falda miden\nel filtro, no el sonido",
    "here the skirt has fallen below the noise:\n"
    "these bands are measuring the sound":
        "aquí la falda ya ha caído por debajo del ruido:\n"
        "estas bandas sí miden el sonido",
    "poles": "polos",
    "zeros (at z = 0)": "ceros (en z = 0)",
    "zoom at z = 1": "zoom en z = 1",
    "Every band of the 1/3-octave bank": "Todo el banco de 1/3 de octava",
    "1 - (largest pole radius)": "1 - (mayor radio de polo)",
    "resample=False (one rate)": "resample=False (una sola frecuencia)",
    "the default multirate bank": "el banco multitasa por defecto",
    "Relative attenuation out of band": "Atenuación relativa fuera de banda",
    "Group delay at the band mid frequency":
        "Retardo de grupo en la frecuencia central de banda",
    "The 1 kHz octave band at 48 kHz, order 6":
        "La banda de octava de 1 kHz a 48 kHz, orden 6",
    "at 2 f_m": "en 2 f_m",
    "at 4 f_m": "en 4 f_m",
    # Block processing and multichannel surveys (signals/filters).
    "Eight 100 ms blocks of a level-stepping signal":
        "Ocho bloques de 100 ms de una señal con escalones de nivel",
    # Symbol and reference notation, shared (cf. "Nivel [dB re 20 uPa]").
    "LAF [dB re 20 uPa]": "LAF [dB re 20 uPa]",
    "one continuous pass": "una sola pasada continua",
    "The settling ramp, magnified": "La rampa de asentamiento, ampliada",
    "5 tau on Fast: 0.63 s": "5 tau en Fast: 0,63 s",
    "A five-position room survey in one octave_filter call":
        "Cinco posiciones de una sala en una sola llamada a octave_filter",
    "the five positions": "las cinco posiciones",
    "energy average (correct)": "promedio energético (correcto)",
    "arithmetic mean of the dB values": "media aritmética de los valores en dB",
    "Energy minus\ndB average [dB]":
        "Promedio energético menos\npromedio en dB [dB]",
    # Metrology: the calibrator take, the two level frames, data qualification
    # and the two uncertainty methods (signals/metrology).
    "Coupler signal-to-noise ratio [dB]":
        "Relación señal-ruido en el acoplador [dB]",
    "Sensitivity error $20\\log_{10}(S/S_\\mathrm{true})$ [dB]":
        "Error de sensibilidad $20\\log_{10}(S/S_\\mathrm{true})$ [dB]",
    "Broadband noise in the calibrator take biases every later level":
        "El ruido de banda ancha en la toma del calibrador sesga todos los "
        "niveles posteriores",
    "IEC 60942 Table 2 class 1 acceptance limit (±0.25 dB)":
        "Límite de aceptación de clase 1 de IEC 60942 Tabla 2 (±0,25 dB)",
    "narrowband=False (full-band RMS)":
        "narrowband=False (RMS de banda completa)",
    "closed form $-10\\log_{10}(1 + 1/\\mathrm{SNR})$":
        "forma cerrada $-10\\log_{10}(1 + 1/\\mathrm{SNR})$",
    "Third-octave band level [dB]": "Nivel de banda de tercio de octava [dB]",
    "dB SPL (factor from the calibrator)": "dB SPL (factor del calibrador)",
    "Time [ms] (onset of a 40-cycle burst)":
        "Tiempo [ms] (inicio de una ráfaga de 40 ciclos)",
    "Band signal [digital units]": "Señal de banda [unidades digitales]",
    "mode='peak' reads the filter's onset transient":
        "mode='peak' lee el transitorio de ataque del filtro",
    "1 kHz band signal": "señal de la banda de 1 kHz",
    "instantaneous frequency": "frecuencia instantánea",
    "Measured rates against each record's own Rice curve":
        "Tasas medidas frente a la curva de Rice de cada registro",
    "Gaussian reference": "Referencia gaussiana",
    "hard-clipped at 2.5 σ": "recortado duro a 2,5 σ",
    "Gaussian + sparse 6 σ spikes": "gaussiano + picos dispersos de 6 σ",
    "spikes lift both tails above the curve":
        "los picos elevan las dos colas por encima de la curva",
    "clipping: no crossings past 2.5 σ":
        "recorte: sin cruces más allá de 2,5 σ",
    "Crossing level a / σ": "Nivel de cruce a / σ",
    "Analysis upper cut-off [kHz]":
        "Frecuencia de corte superior de análisis [kHz]",
    "Irregularity factor $r$": "Factor de irregularidad $r$",
    "A floor 50 dB down, and $m_4$ weighting by $f^4$":
        "Un suelo 50 dB por debajo, y $m_4$ ponderando por $f^4$",
    "ideal low-pass, $\\sqrt{5}/3$ = 0.745":
        "paso bajo ideal, $\\sqrt{5}/3$ = 0,745",
    "the physical band ends at 2 kHz": "la banda física acaba en 2 kHz",
    "Correlation coefficient ρ between the two terms":
        "Coeficiente de correlación ρ entre los dos términos",
    "Combined standard uncertainty [dB]":
        "Incertidumbre típica combinada [dB]",
    "Two terms of 0.3 dB each": "Dos términos de 0,3 dB cada uno",
    "sensitivities of the same sign": "sensibilidades del mismo signo",
    "sensitivities of opposite sign": "sensibilidades de signo opuesto",
    "The same budget, read two ways": "El mismo balance, leído de dos maneras",
    "assumed\nindependent": "supuestos\nindependientes",
    "traceable to the\nsame calibrator": "trazables al\nmismo calibrador",
    "A dominant rectangular input": "Una entrada rectangular dominante",
    "A non-linear model (energy sum)": "Un modelo no lineal (suma energética)",
    "An output against a physical bound": "Una salida contra un límite físico",
    "Correction [dB]": "Corrección [dB]",
    "Combined level [dB]": "Nivel combinado [dB]",
    "Monte Carlo (Supplement 1)": "Monte Carlo (Suplemento 1)",
    "all the mass beyond the bound\npiles up on it":
        "toda la masa más allá del límite\nse acumula en él",
    # Spectral estimation: the Welch segment trade-off and the noise colours
    # (signals/spectra/spectral-analysis).
    "Segment length [samples]": "Longitud de segmento [muestras]",
    "Bias falls, variance rises": "El sesgo baja, la varianza sube",
    "resolution bias, $-10\\lg(1+\\varepsilon_b)$ [dB]":
        "sesgo de resolución, $-10\\lg(1+\\varepsilon_b)$ [dB]",
    "random error, $10\\lg(1+1/\\sqrt{n_d})$ [dB]":
        "error aleatorio, $10\\lg(1+1/\\sqrt{n_d})$ [dB]",
    "measured peak deficit [dB]": "déficit medido del pico [dB]",
    "PSD re its own 1 kHz level [dB]":
        "PSD respecto a su propio nivel en 1 kHz [dB]",
    "The five colours of noise_signal, over three decades":
        "Los cinco colores de noise_signal, en tres décadas",
    # Duct cut-on clip (devices/noise-control/duct-path).
    "Duct cut-on: is one pressure enough? (2D FDTD)":
        "Corte del conducto: ¿basta una sola presión? (FDTD 2D)",
    "Distance along the duct [m]": "Distancia a lo largo del conducto [m]",
    "p across the section": "p en la sección",
    "p / plane mode": "p / modo plano",
    # Lamb's problem clip (simulation/elastic-waves).
    "Lamb's problem: P, S and the surface wave (elastic 2D FDTD)":
        "Problema de Lamb: P, S y la onda de superficie (FDTD elástico 2D)",
    "Free top surface: a Rayleigh train rides it":
        "Superficie superior libre: la recorre un tren de Rayleigh",
    "Rigid top surface: same block, no surface wave":
        "Superficie superior rígida: el mismo bloque, sin onda de superficie",
    "only P and S: a clamped wall carries no surface wave":
        "solo P y S: una pared empotrada no sostiene onda de superficie",
    "each field panel on its own colour scale; the probe traces share one":
        "cada panel de campo con su propia escala de color; las trazas de "
        "sonda comparten una",
    "depth [m]": "profundidad [m]",
    "distance from the impact [m]": "distancia al impacto [m]",
    "surface v_y at 0.15 and 0.30 m":
        "v_y en la superficie a 0,15 y 0,30 m",
    # Mode-conversion clip (simulation/elastic-waves).
    "Mode conversion: water on steel, three incidences (elastic 2D FDTD)":
        "Conversión de modo: agua sobre acero, tres incidencias "
        "(FDTD elástico 2D)",
    "depth from the contact [mm]": "profundidad desde el contacto [mm]",
    "water": "agua",
    "steel": "acero",
    "critical angles: 14.5° (P), 27.5° (SV). Dashed: the Snell direction of "
    "each transmitted wave":
        "ángulos críticos: 14,5° (P), 27,5° (SV). Discontinua: la dirección "
        "de Snell de cada onda transmitida",
    # Radiation-efficiency clip (devices/emission/vibration-sound-power).
    # radiation_efficiency_sigma sat on the FL1/FL8 boundary; see above.
    "ISO/TS 7849-1 assumes ε = 1 here": "ISO/TS 7849-1 supone ε = 1 aquí",
    "the gap Part 1 pays for: 25 dB at 63 Hz,\n"
    "14 dB at 2 kHz, gone only above coincidence":
        "lo que la Parte 1 paga: 25 dB en 63 Hz,\n"
        "14 dB en 2 kHz; solo desaparece sobre la coincidencia",
    "Radiation efficiency: a driven plate below and above f_c":
        "Eficiencia de radiación: una placa excitada bajo y sobre f_c",
    "colour: air pressure / the pressure a piston of the same surface "
    "velocity would make":
        "color: presión del aire / la que haría un pistón con la misma "
        "velocidad superficial",
    "height above the plate [m]": "altura sobre la placa [m]",
    "distance along the plate [m]": "distancia a lo largo de la placa [m]",
    "the whole plate is driven": "se excita la placa entera",
}

_ES_PATTERNS = [
    # piling_campaign_accumulation: the info box carries the computed headroom.
    ((r"^the same campaign, judged five ways: a 200 Hz hammer reaches only\n"
      r"the LF onset, while the porpoise group stays (\d+) dB below its own\n"
      r"TTS criterion even after 10 000 strikes$"),
     ("la misma campaña, juzgada de cinco maneras: un martillo de 200 Hz\n"
      "solo alcanza el inicio LF, y el grupo de las marsopas se queda \\1 dB\n"
      "por debajo de su propio criterio de TTS incluso tras 10 000 golpes")),
    # anim_fdtd_critical_angle: labels and readouts with computed values.
    (r"^critical ray, ψ = (.+)°$", r"rayo crítico, ψ = \1°"),
    (r"^sand: c₂ = (.+) m/s, ρ₂ = (.+) kg/m³$",
     r"arena: c₂ = \1 m/s, ρ₂ = \2 kg/m³"),
    (r"^mud: c₂ = (.+) m/s, ρ₂ = (.+) kg/m³$",
     r"fango: c₂ = \1 m/s, ρ₂ = \2 kg/m³"),
    (r"^ψ = (\d+\.\d+)°$", r"ψ = \1°"),
    (r"^\|R\| = (.+): nothing enters$", r"|R| = \1: no entra nada"),
    (r"^\|R\| = (\d+\.\d+)$", r"|R| = \1"),
    ((r"^beyond (.+) m the net is (.+) % of what entered inside "
      r"\(theory (.+) %\)$"),
     ("más allá de \\1 m el neto es el \\2 % de lo que entró dentro "
      "(teoría \\3 %)")),
    (r"^ · the front meets the bed at ψ = (.+)°$",
     r" · el frente llega al fondo con ψ = \1°"),
    # anim_fdtd_dispersion: labels and readouts carrying computed values.
    (r"^finish line, (.+) m$", r"línea de meta, \1 m"),
    (r"^(\d+) cells per wavelength · Δx = (.+) mm$",
     r"\1 celdas por longitud de onda · Δx = \2 mm"),
    (r"^(.+) m behind \(theory (.+) m\)$",
     r"\1 m por detrás (teoría \2 m)"),
    (r"^crossed at (.+) ms, \+(.+) ms$",
     r"la cruza a \1 ms, +\2 ms"),
    (r"^(.+) % slow \(theory (.+) %\)$",
     r"\1 % más lenta (teoría \2 %)"),
    ((r"^Air at (.+) m/s, a (.+) Hz burst, per-axis Courant number "
      r"S = (.+); grey is the exact continuous wave, dots are the "
      r"grid cells$"),
     ("Aire a \\1 m/s, ráfaga de \\2 Hz, número de Courant por eje "
      "S = \\3; en gris la onda continua exacta, los puntos son las "
      "celdas de la malla")),
    # anim_fdtd_side_branch: titles and readouts carrying computed values.
    (r"^On the tuning frequency: (\d+\.\d) Hz$",
     r"En la frecuencia de sintonía: \1 Hz"),
    (r"^Off tune: (\d+) Hz$", r"Fuera de sintonía: \1 Hz"),
    (r"^×(\d+\.\d) the incident wave, in ≈(\d+) periods$",
     r"×\1 la onda incidente, en ≈\2 periodos"),
    (r"^×(\d+\.\d) at once: it never charges$",
     r"×\1 al instante: nunca se carga"),
    ((r"^the stub rings at (\d+\.\d) Hz, not (\d+\.\d):\n"
      r"ℓ_eff = c/4f = (\d+) mm, so trim it to tune$"),
     ("el tubo resuena a \\1 Hz, no a \\2:\nℓ_ef = c/4f = \\3 mm: se "
      "recorta para afinarlo")),
    # anim_fdtd_absorption_placement: readouts carrying measured values.
    (r"^measured T = (\d+) ms: inside the band$",
     r"T medida = \1 ms: dentro de la banda"),
    (r"^early (\d+) ms, tail (\d+) ms: no single T$",
     r"inicial \1 ms, cola \2 ms: sin una única T"),
    (r"^Sabine, T = (\d+) ms$", r"Sabine, T = \1 ms"),
    (r"^Eyring, T = (\d+) ms$", r"Eyring, T = \1 ms"),
    ((r"^Locally reacting resistive edges; both rooms hold (\d+\.\d) m of "
      r"statistical absorption \(α_st = (\d+\.\d\d) on 21 m vs (\d+\.\d\d) "
      r"on 16 m\); 250 Hz burst\.$"),
     (r"Bordes resistivos de reacción local; ambas salas tienen \1 m de "
      r"absorción estadística (α_st = \2 en 21 m frente a \3 en 16 m); "
      r"ráfaga de 250 Hz.")),
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
    (r"^edge height h0 = (.+) m, single edge$",
     r"altura de arista h0 = \1 m, arista única"),
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
    # composite_facade_weak_element legend (blind-part value baked in);
    # "parte ciega" as the spanish-building-code page words it.
    (r"^blind part RA = (\d+) dBA$", r"parte ciega RA = \1 dBA"),
    # decay_signatures summary boxes (EDT/T20/T30/curvature baked in).
    ((r"^EDT (.+) s\nT20 (.+) s\nT30 (.+) s\nC = (.+) %\n"
      r"T20 = T30, curvature ~ 0$"),
     r"EDT \1 s\nT20 \2 s\nT30 \3 s\nC = \4 %\nT20 = T30, curvatura ~ 0"),
    ((r"^EDT (.+) s\nT20 (.+) s\nT30 (.+) s\nC = (.+) %\n"
      r"T30 > T20: report both$"),
     r"EDT \1 s\nT20 \2 s\nT30 \3 s\nC = \4 %\nT30 > T20: informa de ambos"),
    ((r"^EDT (.+) s\nT20 (.+) s\nT30 (.+) s\nC = (.+) %\n"
      r"EDT << T30: a dry seat$"),
     r"EDT \1 s\nT20 \2 s\nT30 \3 s\nC = \4 %\nEDT << T30: una butaca seca"),
    # image_source_order_convergence crossing annotation (order baked in).
    (r"^crosses Eyring at order (\d+)\nand keeps rising$",
     r"cruza Eyring en el orden \1\ny sigue subiendo"),
    # room_proportion_modes per-room mode counts (all three baked in);
    # "frecuencias distintas" and "hueco" as the page words them.
    (r"^(\d+) modes, (\d+) distinct frequencies; largest gap (.+) Hz$",
     r"\1 modos, \2 frecuencias distintas; mayor hueco \3 Hz"),
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
    # multitaper_psd_confidence legend (mathtext skips the decimal-comma
    # pass, so the pattern writes the comma).
    (r"^Welch 95 % interval \(\$n_d\$ = (\d+)\.(\d+)\)$",
     r"Intervalo del 95 % de Welch ($n_d$ = \1,\2)"),
    # golay_ir readout (baked-in recovery error); wording from the Spanish
    # page and the exact entry above ("identidad exacta sin ruido").
    (r"^max \|recovered - true\| = (.+)\nnoise-free closed-form identity$",
     "máx |recuperada - verdadera| = \\1\nidentidad exacta sin ruido"),
    # dbfs_versus_spl legend (baked-in calibrator offset): symbols shared,
    # comma from the save-time pass.
    (r"^dBFS \+ (.+) dB$", r"dBFS + \1 dB"),
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
    # --- anim_iso717_shift: readouts rewritten on every frame ---
    (r"^(\d+) dB at 500 Hz$", r"\1 dB a 500 Hz"),
    (r"^step (\d+) of (\d+)$", r"paso \1 de \2"),
    (r"^sum = (.+) dB$", r"suma = \1 dB"),
    (r"^C = ([-+]\d+) dB, Ctr = ([-+]\d+) dB$",
     r"C = \1 dB, Ctr = \2 dB"),
    (r"^CI = ([-+]\d+) dB$", r"CI = \1 dB"),
    # --- anim_dynamic_stiffness_sweep: the drive readout, every frame. The
    # padding inside the captures is what keeps the monospace columns still
    # while the numbers change width.
    (r"^f = ( *[\d.]+) Hz    phase = ( *-?[\d.]+)°$",
     r"f = \1 Hz    fase = \2°"),
    # --- anim_block_vs_exponential: readouts rewritten on every frame ---
    (r"^4 kHz burst, (\d+) ms$", r"ráfaga de 4 kHz, \1 ms"),
    (r"^class 1 is (.+) dB about (.+) dB$",
     r"la clase 1 es \1 dB en torno a \2 dB"),
    (r"^burst (\d+) ms, IEC target (.+) dB$",
     r"ráfaga de \1 ms, objetivo IEC \2 dB"),
    (r"^exponential  (.+) dB \((.+)\)$", r"exponencial  \1 dB (\2)"),
    (r"^block Leq    (.+) dB \((.+)\)$", r"Leq bloques  \1 dB (\2)"),
    (r"^spread so far, exponential: (.+) dB$",
     r"dispersión, exponencial: \1 dB"),
    (r"^spread so far, block Leq:   (.+) dB$",
     r"dispersión, Leq bloques: \1 dB"),
    # --- anim_feedback_howl: readouts rewritten on every frame ---
    (r"^Zs = (.+) dB, (\d+) open microphone\(s\)$",
     r"Zs = \1 dB, \2 micrófono(s) abierto(s)"),
    (r"^loop gain Zs \+ Gs = (.+) dB$",
     r"ganancia de lazo Zs + Gs = \1 dB"),
    (r"^each round trip is x (.+)$", r"cada vuelta es x \1"),
    (r"^sum converges to (.+) dB$", r"la suma converge a \1 dB"),
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
    # zwicker_time_varying readouts (values recomputed per run). The two
    # "... sone" lines must precede the generic "(.+) sone" catch-all,
    # which used to swallow them with their English prose intact.
    (r"^Exceedance over the (.+) s record$",
     r"Excedencia sobre el registro de \1 s"),
    (r"^Nmax = (.+) sone \(res\.loudness\)$",
     r"Nmax = \1 sonios (res.loudness)"),
    (r"^arithmetic mean = (.+) sone$",
     r"media aritmética = \1 sonios"),
    (r"^stationary=True on the same record: N = (.+) sone$",
     r"stationary=True en el mismo registro: N = \1 sonios"),
    # sharpness_pair_and_targets band legends and centroids
    (r"^(\d+) Hz critical band — N = (.+) sone, S = (.+) acum$",
     r"Banda crítica de \1 Hz: N = \2 sonios, S = \3 acum"),
    (r"^⟨z⟩ = (.+) Bark$", r"⟨z⟩ = \1 Bark"),
    # Sottek specific panels: percentile line and carrier annotation
    (r"^90th percentile = (.+)$", r"percentil 90 = \1"),
    (r"^the carrier's band, (.+) Bark_HMS$",
     r"la banda de la portadora, \1 Bark_HMS"),
    # hms_modulation_bandpass peak readouts: units read the same
    (r"^R = (.+) asper @ (.+) Hz$", r"R = \1 asper @ \2 Hz"),
    (r"^F = (.+) vacil_HMS @ (.+) Hz$", r"F = \1 vacil_HMS @ \2 Hz"),
    # fluctuation_strength model legends
    (r"^closed form, Eq\. 10\.2 — peak (.+) vacil$",
     r"forma cerrada, Ec. 10.2: máximo \1 vacil"),
    (r"^Osses 2016 signal model — peak (.+) vacil$",
     r"modelo de señal de Osses 2016: máximo \1 vacil"),
    (r"^peak (.+) vacil at (.+) Hz$", r"máximo de \1 vacil a \2 Hz"),
    # annoyance_weightings right-panel gap readout
    (r"^at v = 1\.5, (.+) units apart$",
     r"a v = 1,5, \1 unidades de diferencia"),
    # tone_audibility_uncertainty readouts
    (r"^mean audibility (.+) ± (.+) dB \(Formula 20\)$",
     r"audibilidad media \1 ± \2 dB (Fórmula 20)"),
    (r"^Kt = (\d+) dB, but the interval reaches into Kt = (\d+) dB$",
     r"Kt = \1 dB, pero el intervalo entra en Kt = \2 dB"),
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
    # Electroacoustics figures: titles and readouts carrying computed values
    # (intermodulation_tests, feedback_stability, microphone_patterns,
    # microphone_noise_weightings).
    (r"^\(a\) Difference frequency, 13 / 14 kHz — d\(d,2\) = (.+) %$",
     r"(a) Diferencia de frecuencias, 13 / 14 kHz — d(d,2) = \1 %"),
    ((r"^\(b\) Total difference frequency, 8 / 11\.95 kHz — "
      r"d\(TDFD\) = (.+) %$"),
     ("(b) Diferencia total de frecuencias, 8 / 11,95 kHz — "
      r"d(TDFD) = \1 %")),
    ((r"^\(c\) Dynamic intermodulation, 15 kHz \+ 3\.15 kHz square — "
      r"DIM = (.+) %$"),
     (r"(c) Intermodulación dinámica, 15 kHz + cuadrada de 3,15 kHz — "
      r"DIM = \1 %")),
    (r"^One open microphone — headroom (.+) dB$",
     r"Un micrófono abierto — margen \1 dB"),
    (r"^Four open microphones — headroom (.+) dB$",
     r"Cuatro micrófonos abiertos — margen \1 dB"),
    (r"^Omnidirectional \(b = (.+)\): DI = (.+) dB$",
     r"Omnidireccional (b = \1): DI = \2 dB"),
    (r"^Subcardioid \(b = (.+)\): DI = (.+) dB$",
     r"Subcardioide (b = \1): DI = \2 dB"),
    (r"^Cardioid \(b = (.+)\): DI = (.+) dB$",
     r"Cardioide (b = \1): DI = \2 dB"),
    (r"^Supercardioid \(b = (.+)\): DI = (.+) dB$",
     r"Supercardioide (b = \1): DI = \2 dB"),
    (r"^Hypercardioid \(b = (.+)\): DI = (.+) dB$",
     r"Hipercardioide (b = \1): DI = \2 dB"),
    (r"^Figure-of-eight \(b = (.+)\): DI = (.+) dB$",
     r"Figura en ocho (b = \1): DI = \2 dB"),
    (r"^network alone: (.+) dB$", r"la red sola: \1 dB"),
    # modulation_distortion: the library title reads the same in Spanish;
    # only the decimal comma differs, and the save-time pass applies it.
    ((r"^IEC 60268-3 d₂ = (.+)%, d₃ = (.+)%; SMPTE = (.+)% "
      r"\(f₁ = (.+)Hz, f₂ = (.+)Hz\)$"),
     r"IEC 60268-3 d₂ = \1%, d₃ = \2%; SMPTE = \3% (f₁ = \4Hz, f₂ = \5Hz)"),
    # Broadcast figures: the metered readouts (program_loudness box,
    # true_peak_intersample annotations). Alignment of the monospace box is
    # preserved by the captures.
    ((r"^I     = (.+) LUFS\nLRA   = (.+) LU\nmax M = (.+) LUFS\n"
      r"max S = (.+) LUFS\nTPmax = (.+) dBTP$"),
     ("I     = \\1 LUFS\nLRA   = \\2 LU\nmáx M = \\3 LUFS\n"
      "máx S = \\4 LUFS\nTPmáx = \\5 dBTP")),
    (r"^sample peak (.+) dBFS$", r"pico muestral \1 dBFS"),
    (r"^true peak (.+) dBTP$", r"pico verdadero \1 dBTP"),
    (r"^BS\.1770 asks for n ≥ 4 at 48 kHz:\n(.+) dB left at f_norm = 1/4$",
     "BS.1770 pide n ≥ 4 a 48 kHz:\nquedan \\1 dB en f_norm = 1/4"),
    # Emission figures: computed annotations (k1_k2_corrections,
    # spacer_bandwidth, sound_power_grades_declaration).
    (r"^LWAd = (\d+) dB\nL1 = (\d+) dB\nnot verified$",
     "LWAd = \\1 dB\nL1 = \\2 dB\nno verificado"),
    (r"^LWAd = (\d+) dB\nL1 = (\d+) dB\nverified$",
     "LWAd = \\1 dB\nL1 = \\2 dB\nverificado"),
    ((r"^ISO (3744|3746) limit: K2 = (.+) dB\n"
      r"(\d+) % of the measured energy is room$"),
     ("Límite de ISO \\1: K2 = \\2 dB\n"
      "el \\3 % de la energía medida es de la sala")),
    # A bare "value kHz" marker reads the same in Spanish (the decimal
    # comma is applied by the save-time pass).
    (r"^(\d+(?:\.\d+)?) kHz$", r"\1 kHz"),
    # Noise-control figures: legends and titles carrying computed values
    # (duct_sheet_verification, duct_regenerated_noise, fan_sound_power,
    # hvac_elbow_flow_noise, silencer_selection, silencer_extended_tube,
    # room_to_room_partitions).
    (r"^Fan, Eq\. 13\.1  \(worst Δ (\d+) dB\)$",
     r"Ventilador, Ec. 13.1  (peor Δ \1 dB)"),
    (r"^Flexible duct, 12 in × 6 ft  \(worst Δ (\d+) dB\)$",
     r"Conducto flexible, 12 in × 6 ft  (peor Δ \1 dB)"),
    (r"^Lined duct, (\d+) × (\d+) in, (\d+) ft  \(worst Δ (\d+) dB\)$",
     r"Conducto revestido, \1 × \2 in, \3 ft  (peor Δ \4 dB)"),
    (r"^Supply diffuser, 312 cfm  \(worst Δ (\d+) dB\)$",
     r"Difusor de impulsión, 312 cfm  (peor Δ \1 dB)"),
    (r"^NC 30 curve  \(worst Δ (\d+) dB\)$",
     r"Curva NC 30  (peor Δ \1 dB)"),
    # The diffuser flow legend reads the same in Spanish (comma at save).
    (r"^(\d+) cfm \((.+) m/s\)$", r"\1 cfm (\2 m/s)"),
    (r"^(\d+) % of peak static efficiency \(\+(\d+) dB\)$",
     r"\1 % del rendimiento estático máximo (+\2 dB)"),
    (r"^Straight duct, U = (\d+) m/s$", r"Conducto recto, U = \1 m/s"),
    (r"^Dissipative: 1\.5 m of (\d+) mm lined 36 × 24 in duct$",
     (r"Disipativo: 1,5 m de conducto de 36 × 24 in con \1 mm de "
      r"revestimiento")),
    ((r"^first duct cut-on, (\d+) Hz: the blue four-pole\ncurves stop "
      r"applying beyond it; the dissipative\nregressions carry on$"),
     ("primer corte del conducto, \\1 Hz: las curvas azules\nde cuatro "
      "polos dejan de valer más allá; las\nregresiones disipativas "
      "continúan")),
    (r"^new trough,\n(.+) dB$", "cero nuevo,\n\\1 dB"),
    (r"^Partition area \$S_w\$ = (\d+) m²$",
     r"Área de la partición $S_w$ = \1 m²"),
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
    # anim_loudness_gating: the readouts rewritten every frame.
    (r"^relative gate (.+) LUFS$", r"puerta relativa \1 LUFS"),
    (r"^short-term gate (.+) LUFS$", r"puerta de corto plazo \1 LUFS"),
    (r"^(\d+) of (\d+)$", r"\1 de \2"),
    # anim_epnl_flyover: the per-record tone readout and the window caption.
    (r"^F = (.+) dB at (\d+) Hz$", r"F = \1 dB a \2 Hz"),
    (r"^Sum the energy inside the window, records (\d+) to (\d+)$",
     r"Se suma la energía dentro de la ventana, registros \1 a \2"),
    # anim_image_source_buildup: the running counter and its analytic law.
    (r"^counted (\d+)$", r"contadas \1"),
    (r"^law (\d+)$", r"ley \1"),
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
    # Environment & aircraft figure readouts carrying computed values (FL3).
    (r"^Lden = (\d+)\.(\d+) dB$", r"Lden = \1,\2 dB"),
    (r"^Masking level = (\d+)\.(\d+) dB$",
     r"Nivel de enmascaramiento = \1,\2 dB"),
    (r"^Tone = (\d+)\.(\d+) dB$", r"Tono = \1,\2 dB"),
    (r"^Tonal audibility ΔLₐ = (-?\d+)\.(\d+) dB\naudible$",
     "Audibilidad tonal ΔLₐ = \\1,\\2 dB\naudible"),
    (r"^Tonal audibility ΔLₐ = (-?\d+)\.(\d+) dB\nnot audible$",
     "Audibilidad tonal ΔLₐ = \\1,\\2 dB\nno audible"),
    (r"^PNLTM = (\d+)\.(\d+) PNdB$", r"PNLTM = \1,\2 PNdB"),
    (r"^EPNL = (\d+)\.(\d+) EPNdB\nD = ([+-]?\d+)\.(\d+) dB$",
     "EPNL = \\1,\\2 EPNdB\nD = \\3,\\4 dB"),
    # ground_reflection_coefficient: the grazing-limit annotation.
    (r"^(-?\d+)\.(\d+) dB against (-?\d+)\.(\d+) dB$",
     r"\1,\2 dB frente a \3,\4 dB"),
    # cnossos_road_gradient legend: category and speed per curve.
    (r"^Category ([123]), (\d+) km/h$", r"Categoría \1, \2 km/h"),
    # rd1367_vs_iso_tonal info boxes (True/False are the API's literals).
    ((r"^RD 1367 tonal_correction: Kt = (\d+) dB\n"
      r"ISO 1996-2 survey flag at 250 Hz: (True|False)\n"
      r"Lt at 250 Hz = (\d+)\.(\d+) dB$"),
     ("RD 1367 tonal_correction: Kt = \\1 dB\n"
      "Indicador de cribado ISO 1996-2 en 250 Hz: \\2\n"
      "Lt en 250 Hz = \\3,\\4 dB")),
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
    # Levels: labels and annotations carrying computed values.
    (r"^arithmetic mean of the dB values = (.+) dB$",
     r"media aritmética de los valores en dB = \1 dB"),
    (r"^Leq \(energy mean\) = (.+) dB$", r"Leq (media energética) = \1 dB"),
    (r"^LAeq = (.+) dB \(both\)$", r"LAeq = \1 dB (ambos)"),
    # slm_level_track legend line: the symbol is shared; comma at save time.
    (r"^LAeq = (.+) dB$", r"LAeq = \1 dB"),
    # c_minus_a_spectrum info box: level symbols shared; commas at save time.
    (r"^LAeq (.+) dB\nLCeq (.+) dB\nC - A = (.+) dB$",
     "LAeq \\1 dB\nLCeq \\2 dB\nC - A = \\3 dB"),
    (r"^steady noise:  L10 - L90 = (.+) dB   \|   LAeq - L50 = (.+) dB$",
     r"ruido estacionario:  L10 - L90 = \1 dB   |   LAeq - L50 = \2 dB"),
    ((r"^quiet background, three events:  L10 - L90 = (.+) dB   \|   "
      r"LAeq - L50 = (.+) dB$"),
     ("fondo silencioso, tres eventos:  L10 - L90 = \\1 dB   |   "
      "LAeq - L50 = \\2 dB")),
    (r"^LAE = (.+) dB: the whole event energy in 1 s$",
     r"LAE = \1 dB: toda la energía del evento en 1 s"),
    (r"^Z \(unweighted\): bands sum to (.+) dB$",
     r"Z (sin ponderar): las bandas suman \1 dB"),
    (r"^A-weighted: bands sum to (.+) dB$",
     r"Ponderación A: las bandas suman \1 dB"),
    (r"^largest sample: (.+) dB low$", r"muestra mayor: \1 dB por debajo"),
    (r"^at 100 ms, F reads (.+) dB above S$",
     r"a 100 ms, F lee \1 dB por encima de S"),
    (r"^even Impulse, with its 35 ms attack,\nloses (.+) dB on a 1 ms burst$",
     r"incluso Impulse, con su ataque de 35 ms,\npierde \1 dB en una ráfaga de 1 ms"),
    # Filters: bank order, pole radius and streaming seams (computed values).
    (r"^measured band levels, order (\d+)$",
     r"niveles de banda medidos, orden \1"),
    (r"^Designed at (.+) kHz\n1 - r = (.+)$", r"Diseñada a \1 kHz\n1 - r = \2"),
    (r"^As the bank realizes it, at (.+) Hz\n1 - r = (.+)$",
     r"Tal como la realiza el banco, a \1 Hz\n1 - r = \2"),
    ((r"^The (.+) Hz one-third-octave band, before and after decimation "
      r"by (\d+)$"),
     r"La banda de tercio de octava de \1 Hz, antes y después de diezmar por \2"),
    (r"^stateful=True \(max error (.+) dB\)$",
     r"stateful=True (error máximo \1 dB)"),
    (r"^stateful=False \(up to (.+) dB at a seam\)$",
     r"stateful=False (hasta \1 dB en una unión)"),
    (r"^worst under-read (.+) dB at (.+) Hz$",
     r"subestimación máxima de \1 dB a \2 Hz"),
    # Metrology: calibration, the two level frames, data qualification and the
    # GUM/Monte Carlo comparison (computed values).
    ((r"^below (.+) dB SNR the estimator\nalone spends the whole class 1 "
      r"limit$"),
     ("por debajo de \\1 dB de SNR el estimador\nse gasta él solo todo el "
      "límite de clase 1")),
    (r"^Same shape, different origin: (.+) dB apart$",
     r"La misma forma, distinto origen: \1 dB de diferencia"),
    (r"^drive amplitude (.+)$", r"amplitud de excitación \1"),
    (r"^band peak, \+(.+) dB over the steady tone$",
     r"pico de banda, +\1 dB sobre el tono estacionario"),
    (r"^\(a\) The record: a (.+) Hz to (.+) Hz glide at constant amplitude$",
     r"(a) El registro: un barrido de \1 Hz a \2 Hz con amplitud constante"),
    (r"^the (.+)-(.+) Hz band of panel \(c\)$",
     r"la banda de \1-\2 Hz del panel (c)"),
    (r"^segment values span (.+) to (.+)$",
     r"los valores de segmento van de \1 a \2"),
    ((r"^\(b\) Full band: A = (.+), inside \((.+), (.+)\] — accepted, "
      r"and blind$"),
     r"(b) Banda completa: A = \1, dentro de (\2, \3] — aceptado, y ciego"),
    ((r"^\(c\) Band-limited (.+)-(.+) Hz: A = (.+), outside \((.+), (.+)\] "
      r"— rejected$"),
     ("(c) Limitado a la banda de \\1-\\2 Hz: A = \\3, fuera de (\\4, \\5] "
      "— rechazado")),
    (r"^quadrature value (.+) dB \(assumes ρ = 0\)$",
     r"valor en cuadratura \1 dB (supone ρ = 0)"),
    (r"^\+(.+) % understated$", r"+\1 % subestimado"),
    (r"^MC 95 % interval \[(.+), (.+)\]$", r"Intervalo MC 95 % [\1, \2]"),
    # Spectral estimation: the Welch segment trade-off and the noise-colour
    # legend. The mathtext ($B_e$) makes the later decimal-comma pass skip the
    # segment legend, so its decimal is converted here.
    (r"^A (.+) Hz-wide resonance at (.+) Hz$",
     r"Una resonancia de \1 Hz de ancho a \2 Hz"),
    (r"^nperseg = (\d+), \$B_e\$ = (\d+)\.(\d+) Hz$",
     r"nperseg = \1, $B_e$ = \2,\3 Hz"),
    (r"^the two errors are equal\nnear nperseg = (.+)$",
     r"los dos errores se igualan\ncerca de nperseg = \1"),
    # The colour names are the `color=` literals of noise_signal, so only the
    # measured/exact slope wording is translated.
    (r"^(violet|blue|white|pink|red): measured (.+), exact (.+) dB/octave$",
     r"\1: medida \2, exacta \3 dB/octava"),
    # Duct cut-on clip: panel titles, verdicts and footer (computed values).
    (r"^(.+) Hz: below the (.+) Hz cut-on$",
     r"\1 Hz: por debajo del corte de \2 Hz"),
    (r"^(.+) Hz: above it$", r"\1 Hz: por encima de él"),
    (r"^one pressure, to within (.+) %$",
     r"una sola presión, con un margen del \1 %"),
    (r"^(.+) % of the section's energy is not the plane mode$",
     r"el \1 % de la energía de la sección no está en el modo plano"),
    (r"^section at x = (.+) m$", r"sección en x = \1 m"),
    ((r"^rectangular_duct_cut_on: first cut-on (.+) Hz "
      r"\((.+) × (.+) m duct\)$"),
     r"rectangular_duct_cut_on: primer corte a \1 Hz (conducto de \2 × \3 m)"),
    ((r"^dashed: the section average · vertical scale ×(.+) · each strip on "
      r"its own colour scale$"),
     ("discontinua: la media en la sección · escala vertical ×\\1 · cada "
      "banda con su propia escala")),
    # Lamb's problem clip: the measured Rayleigh speed and the material line.
    (r"^surface train: (.+) m/s measured, (.+) m/s exact$",
     r"tren de superficie: \1 m/s medido, \2 m/s exacto"),
    ((r"^aluminium: \$c_P\$ = (.+), \$c_S\$ = (.+), \$c_R\$ = (.+) m/s "
      r"\(C&H Eq\. 3\.149\)$"),
     r"aluminio: $c_P$ = \1, $c_S$ = \2, $c_R$ = \3 m/s (C&H ec. 3.149)"),
    # Mode-conversion clip: the display gain and the measured reflection.
    (r"^steel drawn ×(\d+) \(\+(\d+) dB\)$",
     r"acero dibujado ×\1 (+\2 dB)"),
    (r"^P and SV both propagate\n\|V\| = (.+)$",
     "P y SV se propagan las dos\n|V| = \\1"),
    (r"^P evanescent, SV alone crosses\n\|V\| = (.+)$",
     "P evanescente, solo SV atraviesa\n|V| = \\1"),
    (r"^both evanescent\n\|V\| = (.+), with a phase$",
     "las dos evanescentes\n|V| = \\1, con una fase"),
    # Radiation-efficiency clip: panel titles, verdicts and footer.
    ((r"^f = f_c/2 = (.+) Hz, below coincidence: the plate wave is slower "
      r"than sound$"),
     ("f = f_c/2 = \\1 Hz, bajo coincidencia: la onda de la placa es más "
      "lenta que el sonido")),
    ((r"^f = 2 f_c = (.+) Hz, above coincidence: the plate wave is faster "
      r"than sound$"),
     ("f = 2 f_c = \\1 Hz, sobre coincidencia: la onda de la placa es más "
      "rápida que el sonido")),
    ((r"^λ_B = (.+) m is shorter than λ = (.+) m in air\n"
      r"no angle solves sin θ = λ/λ_B: the skin dies in (.+) m$"),
     ("λ_B = \\1 m es menor que λ = \\2 m en el aire\n"
      "ningún ángulo cumple sen θ = λ/λ_B: la piel se apaga en \\3 m")),
    ((r"^λ_B = (.+) m is longer than λ = (.+) m in air\n"
      r"the trace match sends a beam out at (.+)°$"),
     ("λ_B = \\1 m es mayor que λ = \\2 m en el aire\n"
      "la coincidencia de traza lanza un haz a \\3°")),
    ((r"^elastic 2D FDTD, 10 mm steel plate, f_c = (.+) Hz · overlaid "
      r"line: its deflection, exaggerated$"),
     ("FDTD elástico 2D, placa de acero de 10 mm, f_c = \\1 Hz · línea "
      "superpuesta: su deformación, exagerada")),
    # Underwater figures: readouts carrying computed values (the detection
    # ranges, the modal wavenumbers, the orca branches, the exposure
    # functions and the sound-speed spread).
    (r"^Detection range ([\d.]+) m$", r"Alcance de detección \1 m"),
    (r"^Figure of merit ([\d.]+) dB$", r"Figura de mérito \1 dB"),
    (r"^Figure of merit = ([\d.]+) dB$", r"Figura de mérito = \1 dB"),
    (r"^Real Waveguide: (\d+) Crossings$", r"Guía de ondas real: \1 cruces"),
    ((r"^first crossing: ([\d.]+) km\n"
      r"\(what detection_range_from_curve returns\)$"),
     "primer cruce: \\1 km\n(lo que devuelve detection_range_from_curve)"),
    (r"^still detectable at ([\d.]+) km$", r"aún detectable a \1 km"),
    (r"^Peak = ([\d.]+) dB re 1 µPa$", r"Pico = \1 dB re 1 µPa"),
    (r"^(\d+) modes \((\d+) Hz\)$", r"\1 modos (\2 Hz)"),
    (r"^m = (\d),  kr = ([\d.]+) \(exact ([\d.]+)\)$",
     r"m = \1,  kr = \2 (exacto \3)"),
    (r"^(\d+) of (\d+) Modes Are Outside It$",
     r"\1 de \2 modos quedan fuera"),
    ((r"^the PE loses ([\d.]+) dB of level at every range,\n"
      r"and the offset does not shrink: an ideal\n"
      r"waveguide strips nothing away$"),
     ("la PE pierde \\1 dB de nivel a cualquier distancia,\n"
      "y el sesgo no se diluye: una guía ideal\n"
      "no descama ningún modo")),
    (r"^analytic z_t = ([\d.]+) m\ntraced      = ([\d.]+) m$",
     "z_t analítica = \\1 m\nz_t trazada   = \\2 m"),
    ((r"^Mackenzie check point: ([\d.]+) m/s\n"
      r"at 25 C, 35 ppt, 1000 m \(not on this profile\)$"),
     ("Punto de comprobación de Mackenzie: \\1 m/s\n"
      "a 25 °C, 35 ppt, 1000 m (fuera de este perfil)")),
    (r"^Spread on This Profile: up to ([\d.]+) m/s$",
     r"Dispersión en este perfil: hasta \1 m/s"),
    (r"^(LF|HF|VHF|PW|OW): cumulative ([\d.]+) dB, margin ([+-][\d.]+) dB$",
     r"\1: acumulado \2 dB, margen \3 dB"),
    (r"^minimum ([\d.]+) dB at ([\d.]+) kHz$", r"mínimo de \1 dB a \2 kHz"),
    (r"^second branch there: ([\d.]+) dB$", r"el segundo tramo allí: \1 dB"),
    (r"^third branch at 50 kHz: ([\d.]+) dB$",
     r"tercer tramo a 50 kHz: \1 dB"),
    ((r"^the third branch starts at 46\.2 kHz;\n"
      r"reading the second one at 50 kHz loses ([\d.]+) dB$"),
     ("el tercer tramo empieza en 46,2 kHz;\n"
      "leer el segundo a 50 kHz pierde \\1 dB")),
    ((r"^each minimum is that group's weighted TTS onset T_w = K \+ C\n"
      r"LF: below f1 the filter falls at 20a = (\d+) dB/decade, "
      r"above f2 at 20b = (\d+) dB/decade$"),
     ("cada mínimo es el inicio de TTS ponderado de su grupo T_w = K + C\n"
      "LF: bajo f1 el filtro cae a 20a = \\1 dB/década, "
      "sobre f2 a 20b = \\2 dB/década")),
]

# The Spanish decimal comma, with the numbers that are NOT decimals carved
# out. A number introduced by a clause/equation/table/annex token ("apartado
# 7.4", "Ec. 8.252", "Tabla 13.8") is a reference into a standard or a book,
# and the Spanish pages keep its dot; a plural token ("Ecs. 8.44, 8.46",
# "apartados 6.2 y 6.3") extends that reading over its whole list, and a
# dash-joined range ("Ecs. 13.27-13.33", "apartado 4.1-4.3") is
# reference-to-reference after either. The carve-out is deliberately no wider:
# after a singular token only the first number binds, so a genuine decimal in
# the same breath ("Ec. 10.2: máximo 3.7 vacil", "Fórmula 7, eps = 0.9",
# "Tabla 2 -0.5 dB") still takes its comma, and the dash of a range must sit
# digit-to-digit so a spaced negative value is never read as a range.
_REF_NUM = r"\d+(?:\.\d+)*(?:[-–−]\d+(?:\.\d+)*)*"
_CLAUSE_REF_RE = re.compile(
    r"\b(?:apartados|cap[ií]tulos|cl[aá]usulas|anexos|tablas|f[oó]rmulas"
    r"|ecs\.)\s*" + _REF_NUM + r"(?:(?:\s*,\s*|\s+y\s+)" + _REF_NUM + r")*"
    r"|(?:\b(?:apartado|cap[ií]tulo|cl[aá]usula|anexo|tabla|f[oó]rmula"
    r"|f[oó]rm\.|ec\.)|§)\s*" + _REF_NUM,
    re.IGNORECASE,
)
# A letter immediately before the number marks a standard designation
# (e.g. "S3.5"), not a decimal; a further digit or dot on either side marks
# a three-part clause/version number ("7.4.3"). Both keep their dots.
_DECIMAL_RE = re.compile(r"(?<![\d.A-Za-z])(\d+)\.(\d+)(?![.\d])")


def _decimal_comma(s: str) -> str:
    """Rewrite the decimal dots of *s* into Spanish commas.

    Reference numbers matched by :data:`_CLAUSE_REF_RE` are passed through
    untouched; everything between them gets :data:`_DECIMAL_RE` applied.
    """
    out: list[str] = []
    last = 0
    for m in _CLAUSE_REF_RE.finditer(s):
        out.append(_DECIMAL_RE.sub(r"\1,\2", s[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(_DECIMAL_RE.sub(r"\1,\2", s[last:]))
    return "".join(out)


def set_lang(lang: str) -> None:
    """Switch the output language ('en' or 'es')."""
    global _LANG, _LANG_SUFFIX
    _LANG = lang
    _LANG_SUFFIX = "" if lang == "en" else f"_{lang}"
    _publish(_LANG=_LANG, _LANG_SUFFIX=_LANG_SUFFIX)


def audit_figure(stem: str) -> None:
    """Name the asset the following lookups belong to, and the pass running.

    Called by the savers just before they translate, so every string
    :func:`lookup` cannot translate is attributed to the file it ships in.
    """
    _audit.visit(stem, _LANG)


def lookup(s: str) -> str:
    """Translate one whole string: the exact table, then the pattern list.

    The single place where "the tables had nothing for this" is known, which
    is why the miss is reported from here rather than from each caller: a
    string that translates to itself (``"2 dB: normal"``, ``"Bessel"``) is a
    deliberate table entry, and comparing input with output cannot tell it
    apart from a string that was never in the table at all.
    """
    import re as _re

    if s in _ES_EXACT:
        return _ES_EXACT[s]
    for pat, repl in _ES_PATTERNS:
        new, n = _re.subn(pat, repl, s)
        if n:
            return new
    _audit.untranslated(s)
    return s


def _audit_english(fig: Any) -> None:
    """Report what the tables cannot translate, on the English pass.

    The English figure is correct by construction, so this walk exists only
    to say which of its strings the tables have no entry for. Intersected
    with the Spanish pass by ``scripts/check_figure_language.py``, that is
    what separates a genuine miss from a label the library renderer already
    translated on its own (see :mod:`figure_language_audit`).

    It reads and never writes: the English SVG is committed and byte-compared
    by ``scripts/check_figures.py``, so this must not be able to perturb it.
    Where the Spanish pass replaces a formatter to reach labels built at draw
    time, this one calls the formatter at the tick locations instead and
    throws the answer away.
    """
    if _audit.audit_dir() is None:
        return

    import matplotlib.text as _mtext
    from matplotlib.category import StrCategoryFormatter as _SCF
    from matplotlib.ticker import FixedFormatter as _FxF
    from matplotlib.ticker import FuncFormatter as _FF

    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            fmt = axis.get_major_formatter()
            if isinstance(fmt, _FxF):
                for label in fmt.seq:
                    lookup(label)
            elif isinstance(fmt, _FF | _SCF):
                for pos, loc in enumerate(axis.get_majorticklocs()):
                    lookup(str(fmt(loc, pos)))
    for artist in fig.findobj(_mtext.Text):
        if artist.get_text():
            lookup(artist.get_text())


def _translate_figure(fig: Any) -> None:
    """Rewrite every Text artist of *fig* into the active language."""
    import re as _re

    import matplotlib.text as _mtext

    if _LANG == "en":
        _audit_english(fig)
        return
    from matplotlib.category import StrCategoryFormatter as _SCF
    from matplotlib.ticker import FixedFormatter as _FxF
    from matplotlib.ticker import FuncFormatter as _FF
    from matplotlib.ticker import ScalarFormatter as _SF

    _comma = _decimal_comma  # the guarded decimal-comma pass, defined above
    _tr_words = lookup  # the exact / pattern lookups, no decimal comma

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
            elif isinstance(fmt, _SCF):
                # A string-category axis (``ax.bar(["first half", ...], ...)``)
                # rebuilds its labels from the unit registry at every draw, so
                # rewriting the tick Text artists below would be undone by the
                # savefig draw; wrap the formatter instead, as above.
                wrapped = _FF(
                    lambda v, pos, _f=fmt: _comma(_tr_words(str(_f(v, pos))))
                )
                wrapped._phonometry_comma = True  # type: ignore[attr-defined]
                axis.set_major_formatter(wrapped)
            elif type(fmt) is _SF and axis.get_scale() == "linear":
                # ``f"{v:g}"`` writes an ASCII hyphen, while the formatter it
                # replaces runs every label through ``fix_minus`` and ships the
                # typographic U+2212 the English figure shows. Keep that pass,
                # or the same axis is drawn with a different minus sign in each
                # language.
                wrapped = _FF(
                    lambda v, pos, _f=fmt: _f.fix_minus(_comma(f"{v:g}"))
                )
                axis.set_major_formatter(wrapped)
    for artist in fig.findobj(_mtext.Text):
        s = artist.get_text()
        if not s:
            continue
        translated = lookup(s)
        if translated != s:
            artist.set_text(translated)
        # Spanish decimal comma, applied uniformly to every text artist
        # (tick labels included) except mathtext. The substitution itself is
        # conservative -- it only rewrites a bare ``digit.digit`` not adjacent
        # to further digits/dots -- so underscore-bearing unit tokens such as
        # ``sone_HMS`` / ``tu_HMS`` keep their identifier intact while genuine
        # decimals in the same label (e.g. ``8.0 sone_HMS``) still get commas.
        s = artist.get_text()
        if s and "$" not in s and _re.search(r"\d\.\d", s):
            # Clause/version numbers like 5.3.3, standard designations like
            # "S3.5" and guarded references ("apartado 7.4") keep their dots.
            artist.set_text(_decimal_comma(s))
