#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Spanish variants of every user-visible string the diagrams draw.

A table rather than code, which is why it is a module of its own: the
builders write their labels in English and :meth:`SVG.tr` looks the
Spanish rendering up here. A string that is absent is deliberate, not
missing; numbers, unit-only labels and code identifiers read the same in
both languages and are shared.
"""

from __future__ import annotations

import os
import sys

# The miss recorder sits at the top of ``scripts/``, next to the checker that
# reads what it writes; guard the path the way the other cross-imports in
# ``scripts/`` do (see check_figures.py).
_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import figure_language_audit as _audit

# Spanish variants of every user-visible string. Strings not in the table
# (numbers, unit-only labels, code identifiers) are shared between languages.
_ES: dict[str, str] = {
    # ISO 9613-2 ground regions (environment/propagation/outdoor-propagation).
    "ISO 9613-2 ground regions and the ground factor G":
        "Regiones de suelo y factor de suelo G (ISO 9613-2)",
    "$G$ is the porous fraction of its region":
        "$G$ es la fracción porosa de su región",
    "grass, $G = 1$": "hierba, $G = 1$",
    "asphalt, $G = 0$": "asfalto, $G = 0$",
    "source region  $30 h_s$ = 45 m": "región de la fuente  $30 h_s$ = 45 m",
    "middle region  110 m": "región intermedia  110 m",
    "receiver region  $30 h_r$ = 45 m":
        "región del receptor  $30 h_r$ = 45 m",
    "$d_p$ = 60 m: the regions overlap, so there is no middle region":
        "$d_p$ = 60 m: las regiones se solapan y no hay región intermedia",
    "below $30(h_s + h_r)$ = 90 m: $q = 0$, $A_m = 0$":
        "por debajo de $30(h_s + h_r)$ = 90 m: $q = 0$, $A_m = 0$",
    "and ground_middle is ignored entirely":
        "y ground_middle se ignora por completo",
    # The four diffracted paths (environment/propagation/ground-barriers).
    "The four diffracted paths of a barrier on finite-impedance ground":
        "Los cuatro caminos difractados sobre suelo de impedancia finita",
    "Four coherent routes over one edge":
        "Cuatro recorridos coherentes sobre un borde",
    "0 bounces": "0 reflexiones",
    "1 bounce, source side": "1 reflexión, lado fuente",
    "1 bounce, receiver side": "1 reflexión, lado receptor",
    "2 bounces": "2 reflexiones",
    "4 m screen": "pantalla de 4 m",
    "image receiver": "receptor imagen",
    "Thick barrier: two edges": "Barrera gruesa: dos bordes",
    # CNOSSOS-EU road source line geometry (environment/sources).
    "CNOSSOS-EU road source line geometry":
        "Geometría de la línea fuente viaria CNOSSOS-EU",
    "Plan — two-lane urban arterial": "Planta — vía urbana de dos carriles",
    "source line,": "línea fuente,",
    "one per lane centre": "una por eje de carril",
    "each segment carries L'W,eq,line,i + 10 lg(dL)":
        "cada segmento lleva L'W,eq,line,i + 10 lg(dL)",
    "signal-controlled junction": "intersección semaforizada",
    "$max(1 − |x|/100, 0)$": "$max(1 − |x|/100, 0)$",
    "dwelling façade": "fachada de vivienda",
    "road surface": "pavimento",
    "equivalent point source": "fuente puntual equivalente",
    "gradient $s$: the flow is split": "pendiente $s$: el flujo se divide",
    "and corrected uphill and downhill": "y se corrige en subida y en bajada",
    "receiver point": "punto receptor",
    # CNOSSOS-EU railway source lines (environment/sources).
    "CNOSSOS-EU railway source lines and directivity angles":
        "Líneas fuente ferroviarias y directividad CNOSSOS-EU",
    "datum: the plane tangent to the two rail heads":
        "referencia: plano tangente a las dos cabezas de carril",
    "A — rolling, impact, squeal,": "A — rodadura, impacto, chirrido,",
    "bridge, low traction": "puente, tracción baja",
    "B — exhausts, roof apparatus,": "B — escapes, equipos de techo,",
    "pantograph recess": "hueco del pantógrafo",
    "receiver, 4 m": "receptor, 4 m",
    "$ψ > 0$": "$ψ > 0$",
    "$ψ ≤ 0$: the vertical correction of A is zero":
        "$ψ ≤ 0$: la corrección vertical de A es nula",
    "track axis": "eje de la vía",
    "0 dB broadside": "0 dB en perpendicular",
    "−20 dB along the track": "−20 dB en el eje de la vía",
    "receiver bearing": "dirección al receptor",
    "Impact noise applies from 50 m": "Ruido de impacto aplicado desde 50 m",
    "before a joint to 50 m after it": "antes de una junta hasta 50 m después",
    "Curve squeal needs ≥ 50 m": "El chirrido en curva exige ≥ 50 m",
    "of continuous curve": "de curva continua",
    # IEC 61400-11 ground board (environment/sources/wind-turbine-noise).
    "IEC 61400-11 ground-board microphone mounting":
        "Micrófono sobre placa en el suelo (IEC 61400-11)",
    "split (if any): off centre, parallel, gap < 1 mm":
        "junta (si la hay): descentrada, paralela, holgura < 1 mm",
    "to the turbine": "hacia el aerogenerador",
    "diameter ≥ 1,0 m": "diámetro ≥ 1,0 m",
    "plywood ≥ 12,0 mm  ·  metal ≥ 2,5 mm":
        "contrachapado ≥ 12,0 mm  ·  metal ≥ 2,5 mm",
    "soil fillet": "recrecido de tierra",
    "capsule diaphragm in the board plane, ≤ 13 mm":
        "diafragma de la cápsula en el plano del tablero, ≤ 13 mm",
    "primary windscreen ≈ 90 mm": "pantalla antiviento primaria ≈ 90 mm",
    "secondary: high wind only —": "secundaria: solo con viento fuerte —",
    "document and correct its insertion loss":
        "documentar y corregir su pérdida por inserción",
    "Position (clause 7.1)": "Posición (capítulo 7.1)",
    "within ±15° of downwind": "dentro de ±15° de la dirección a sotavento",
    "$R_0$ to ±20 %, max ±30 m, measured to ±2 %":
        "$R_0$ con ±20 %, máx. ±30 m, medida con ±2 %",
    "board inclination $φ$ between 25° and 40°":
        "inclinación del tablero $φ$ entre 25° y 40°",
    "reflections from structures < 0,2 dB":
        "reflexiones de estructuras < 0,2 dB",
    # RD 1367/2007 chain (environment/assessment/spanish-noise-regulation).
    "RD 1367/2007: from a noise phase to the three acceptance criteria":
        "RD 1367/2007: de la fase de ruido a los tres criterios de aceptación",
    "LKeq,Ti = LAeq,Ti + Kt + Kf + Ki": "LKeq,Ti = LAeq,Ti + Kt + Kf + Ki",
    "LKeq,x = 10 lg[ (1/T) Σ Ti": "LKeq,x = 10 lg[ (1/T) Σ Ti",
    "                10^(LKeq,Ti/10) ]": "                10^(LKeq,Ti/10) ]",
    "Article 25.2 drops the third criterion for an activity already in "
    "operation":
        "El artículo 25.2 suprime el tercer criterio para una actividad ya "
        "en funcionamiento",
    "1 — the day, split into evaluation periods (Annex I A.1)":
        "1 — el día, dividido en períodos de evaluación (Anexo I A.2)",
    "night 23-07": "noche 23-07",
    "day 07-19": "día 07-19",
    "evening 19-23": "tarde 19-23",
    "2 h shut": "2 h cerrada",
    "6 h, machine": "6 h, máquina",
    "4 h, rest": "4 h, resto",
    "noise phases $T_i$ of uniformly perceived level":
        "fases de ruido $T_i$ de nivel percibido uniforme",
    "2 — each phase, corrected": "2 — cada fase, corregida",
    "$K_t + K_f + K_i ≤ 9$ dB (Annex IV A.3.3)":
        "$K_t + K_f + K_i ≤ 9$ dB (Anexo IV A.3.3)",
    "3 — the period level": "3 — el nivel del período",
    "round_reported_level → 57 dB": "round_reported_level → 57 dB",
    "4 — the annual value": "4 — el valor anual",
    "$L_{K,x}$ over the operating days":
        "$L_{K,x}$ sobre los días de actividad",
    "303 open / 62 closed → 56 dB": "303 abierta / 62 cerrada → 56 dB",
    "worst phase ≤ limit + 5 dB": "peor fase ≤ límite + 5 dB",
    "daily LKeq,x ≤ limit + 3 dB": "LKeq,x diario ≤ límite + 3 dB",
    "annual $L_{K,x}$ ≤ limit": "$L_{K,x}$ anual ≤ límite",
    "Article 25.2 drops the third criterion for an ":
        "El artículo 25.2 suprime el tercer criterio para una ",
    "activity already in operation": "actividad ya en funcionamiento",
    # Facade sound insulation setup (buildings/insulation/facade-insulation).
    "Facade sound insulation setup (ISO 16283-3)":
        "Montaje de aislamiento acústico de fachada (ISO 16283-3)",
    "Receiving room": "Recinto receptor",
    "$S$ = 11.5 m²": "$S$ = 11,5 m²",
    "Loudspeaker": "Altavoz",
    "(on the ground)": "(sobre el suelo)",
    "45° ± 5°": "45° ± 5°",
    "$r ≥ 5$ m element / ≥ 7 m global": "$r ≥ 5$ m elemento / ≥ 7 m global",
    "$D > 3.5$ m (element) / > 5 m (global)":
        "$D > 3,5$ m (elemento) / > 5 m (global)",
    "$L_{1,s}$  element method": "$L_{1,s}$  método de elemento",
    "≤ 10 mm parallel / ≤ 3 mm normal": "≤ 10 mm paralelo / ≤ 3 mm normal",
    "3 to 10 positions, never gridded":
        "de 3 a 10 posiciones, nunca en rejilla",
    "L₁,2m  global method": "L₁,2m  método global",
    "(2.0 ± 0.2) m": "(2,0 ± 0,2) m",
    "1.5 m": "1,5 m",
    "above the": "por encima del",
    "receiving-room floor": "suelo del recinto receptor",
    "Element method → $R′_{45°}$ or $R′_{tr,s}$: one component, comparable "
    "with a laboratory $R$.":
        "Método de elemento → $R′_{45°}$ o $R′_{tr,s}$: un componente, "
        "comparable con una $R$ de laboratorio.",
    "Global method → D2m,nT: the whole facade as built, "
    "not comparable with a laboratory $R$.":
        "Método global → D2m,nT: la fachada tal como está; no comparable con "
        "una $R$ de laboratorio.",
    "Road traffic replaces the loudspeaker at all angles: inside and outside "
    "at once, ≥ 50 pass-bys.":
        "El tráfico rodado incide desde todos los ángulos: dentro y fuera a "
        "la vez, ≥ 50 pasos.",
    "Clauses 9.4, 9.5.1, 9.6.1 and 10.2. None of it is checked "
    "by the functions.":
        "Apartados 9.4, 9.5.1, 9.6.1 y 10.2. Nada de esto lo comprueban "
        "las funciones.",
    # Heavy and soft impact sources (buildings/insulation/heavy-impact-sources).
    "Standard heavy and soft impact sources (ISO 16283-2, JIS A 1418-2)":
        "Fuentes de impacto normalizadas (ISO 16283-2, JIS A 1418-2)",
    "Floor under test (source room)": "Forjado ensayado (recinto emisor)",
    "(a) tapping machine": "(a) máquina de impactos",
    "ISO 10140-5 Annex E": "ISO 10140-5 Anexo E",
    "5 hammers, 500 g each": "5 martillos de 500 g cada uno",
    "(100 ± 20) ms apart": "separados (100 ± 20) ms",
    "40 mm": "40 mm",
    "(b) rubber ball": "(b) pelota de caucho",
    "ISO 16283-2 Annex A / ISO 10140-5 Annex F":
        "ISO 16283-2 Anexo A / ISO 10140-5 Anexo F",
    "180 mm": "180 mm",
    "30 mm wall": "pared de 30 mm",
    "$m_{eff}$ = (2.5 ± 0.1) kg": "$m_{eff}$ = (2,5 ± 0,1) kg",
    "$e$ = 0.8 ± 0.1": "$e$ = 0,8 ± 0,1",
    "(100 ± 1) cm": "(100 ± 1) cm",
    "from the ball's BOTTOM": "desde la BASE de la pelota",
    "(c) bang machine": "(c) máquina de golpes",
    "JIS A 1418-2 only": "solo en JIS A 1418-2",
    "(2.4 ± 0.2)·10⁵ Pa": "(2,4 ± 0,2)·10⁵ Pa",
    "$m_{eff}$ = (7.3 ± 0.2) kg": "$m_{eff}$ = (7,3 ± 0,2) kg",
    "85 cm": "85 cm",
    "source": "fuente",
    "rigid floor +": "suelo rígido +",
    "force plate": "plataforma de fuerza",
    "octave filter": "filtro de octava",
    "analyser → $L_{FE}$": "analizador → $L_{FE}$",
    "JIS A 1418-2 Annex C: the filter goes BEFORE the analyser,":
        "JIS A 1418-2 Anexo C: el filtro va ANTES del analizador,",
    "so $L_{FE}$ is evaluated once per band":
        "de modo que $L_{FE}$ se evalúa una vez por banda",
    "The dimensions above are the standards' informative construction "
    "examples;":
        "Las dimensiones anteriores son ejemplos constructivos informativos "
        "de las normas;",
    "the specification is the force spectrum, not the shape.":
        "la especificación es el espectro de fuerza, no la forma.",
    # ISO 10052 survey sweep (buildings/insulation/insulation-survey).
    "The ISO 10052 survey sweep (Clauses 6.2 and 6.3)":
        "El barrido del método de control ISO 10052 (apartados 6.2 y 6.3)",
    "Plan of the room": "Planta del recinto",
    "separating element": "elemento separador",
    "≥ 0.5 m": "≥ 0,5 m",
    "corner opposite the element,": "esquina opuesta al elemento,",
    "facing into the corner": "orientado hacia la esquina",
    "facing away": "de espaldas",
    "arm's length": "brazo extendido",
    "180° × 4 traverses,": "180° × 4 barridos,",
    "≈ 30 s in total": "≈ 30 s en total",
    "Elevation: the same sweep": "Alzado: el mismo barrido",
    "raise and lower the arm": "subir y bajar el brazo",
    "during each traverse": "durante cada barrido",
    "Alternative (6.3.1): a rotating microphone on a stand, ≥ 10° to the "
    "horizontal, radius ≥ 1 m.":
        "Alternativa (6.3.1): micrófono giratorio en soporte, ≥ 10° sobre la "
        "horizontal, radio ≥ 1 m.",
    "Without a real-time octave analyser, repeat the sweep per band and read "
    "each 30 s $L_{eq}$.":
        "Sin analizador de octavas en tiempo real, repetir el barrido por "
        "banda y leer los $L_{eq}$ de 30 s.",
    "Tapping machine (6.2.3): floor centre, on the diagonal; "
    "three positions at 45° to the ribs.":
        "Máquina de impactos (6.2.3): centro del forjado, en diagonal; "
        "3 a 45° con las viguetas.",
    # Rooms / prediction: the EN 12354-6 take-off plate and the directivity
    # plate (buildings/rooms/enclosed-space-absorption, room-image-sources).
    "Room take-off: one room, three input lists (EN 12354-6)":
        "Levantamiento del recinto: tres listas de entrada (EN 12354-6)",
    "1000 Hz octave band": "banda de octava de 1000 Hz",
    "$V$ = 29.75 m³": "$V$ = 29,75 m³",
    "ceiling  12.39 m²  $α_s$ 0.02": "techo  12,39 m²  $α_s$ 0,02",
    "glass facade  10.90 m²  $α_s$ 0.04":
        "fachada de vidrio  10,90 m²  $α_s$ 0,04",
    "floor  12.39 m²  $α_s$ 0.05": "suelo  12,39 m²  $α_s$ 0,05",
    "short wall  6.55 m²  $α_s$ 0.04  (x2)":
        "pared corta  6,55 m²  $α_s$ 0,04  (x2)",
    # Single separators on this one tag: composed text keeps its
    # whitespace, and with the doubled gaps the longer Spanish wording
    # would push its last digit under the surfaces panel.
    "long wall (brick)  10.90 m²  $α_s$ 0.04":
        "pared larga (ladrillo) 10,90 m² $α_s$ 0,04",
    "objects: 0.15, 0.60, 2 × 0.05, 2 × 0.65 m³":
        "objetos: 0,15, 0,60, 2 × 0,05, 2 × 0,65 m³",
    "$A$ = 2.26 m²   (Formula 1)": "$A$ = 2,26 m²   (fórmula 1)",
    "$A_{obj}$ = 2.77 m²    $ψ$ = 0.072": "$A_{obj}$ = 2,77 m²    $ψ$ = 0,072",
    "(Formula 4, then Formula 3)": "(fórmula 4, después fórmula 3)",
    # The printed surfaces list: code stays code (periods and identifiers as
    # the reader types them, like the page's own snippet); only the trailing
    # comments are prose, worded as the surface tags above.
    "    (12.39, 0.05),   # floor": "    (12.39, 0.05),   # suelo",
    "    (12.39, 0.02),   # ceiling": "    (12.39, 0.02),   # techo",
    "    (10.90, 0.04),   # long wall": "    (10.90, 0.04),   # pared larga",
    "    (10.90, 0.04),   # facade": "    (10.90, 0.04),   # fachada",
    "    (6.55, 0.04),    # short wall": "    (6.55, 0.04),    # pared corta",
    # The one-wall-two-rows inset: the prose around it is translated, the
    # identifiers of the symbols are not (a subscript is part of the name
    # of the quantity, and "wall"/"win" name the same two areas in both).
    "($S_{wall} − S_{win}$, $α_{wall}$)": "($S_{wall} − S_{win}$, $α_{wall}$)",
    "($S_{win}$, $α_{win}$)": "($S_{win}$, $α_{win}$)",
    "One wall, two rows": "Una pared, dos filas",
    "window": "ventana",
    "one wall on the drawing": "una pared en el plano",
    "areas sum to the wall": "las áreas suman la pared",
    "Never average a lining into its wall by hand: the areas are weighted "
    "inside the formula.":
        "Nunca promedie a mano un revestimiento con su pared: la fórmula ya "
        "pondera por área.",
    "Directivity factor Q: four mountings, four critical distances":
        "Factor de directividad Q: cuatro montajes, cuatro distancias críticas",
    "The same compact source, four mountings (workshop with $R$ = 62 m²)":
        "La misma fuente compacta, cuatro montajes (taller con $R$ = 62 m²)",
    "radiates into $4π$ sr": "radia en $4π$ sr",
    "radiates into $2π$ sr": "radia en $2π$ sr",
    "radiates into $π$ sr": "radia en $π$ sr",
    "radiates into $π/2$ sr": "radia en $π/2$ sr",
    "free space": "campo libre",
    "hard floor": "suelo rígido",
    "floor-wall edge": "arista suelo-pared",
    "trihedral corner": "rincón triedro",
    "on a stand": "sobre un trípode",
    "on the slab": "sobre la solera",
    "against a wall on the slab": "contra una pared, sobre la solera",
    "in the corner of the workshop": "en el rincón del taller",
    "$Q$ multiplies the direct term only: the reverberant plateau does not move.":
        "$Q$ solo multiplica el término directo: la meseta reverberante no se "
        "mueve.",
    "$r_c = √(Q·R/16π)$, so two steps of mounting move the crossover by a "
    "factor of 2.":
        "$r_c = √(Q·R/16π)$, así que dos escalones de montaje desplazan el "
        "cruce un factor 2.",
    "Calibration chain — from calibrator to physical units":
        "Cadena de calibración — del calibrador a unidades físicas",
    # Speech Intelligibility Index (ANSI S3.5-1997)
    "Speech Intelligibility Index computation flow (ANSI S3.5-1997)":
        "Flujo de cálculo del índice de inteligibilidad del habla (ANSI S3.5-1997)",
    # Room-noise rating methods (ANSI/ASA S12.2-2019)
    "Room-noise rating methods (ANSI/ASA S12.2-2019): NC and RC Mark II":
        "Calificación del ruido de salas (S12.2-2019): NC y RC Mark II",
    "Octave-band sound pressure levels  $L(f)$":
        "Niveles de presión acústica por banda de octava  $L(f)$",
    "NC — tangency method": "NC — método de tangencia",
    "Table 1 curves": "curvas de la Tabla 1",
    "NC value in each band": "Valor NC en cada banda",
    "curve level = $L(f)$ at that $f$":
        "nivel de la curva = $L(f)$ en esa $f$",
    "NC = highest curve touched": "NC = curva más alta tocada",
    "note the governing band": "anotar la banda determinante",
    "NC-NN (band)": "NC-NN (banda)",
    "RC Mark II  (Annex D)": "RC Mark II  (Anexo D)",
    "−5 dB/octave curves": "curvas de −5 dB/octava",
    "RC = round($L_{MF}$)   (clause D.4)":
        "RC = redondeo($L_{MF}$)   (cláusula D.4)",
    "Spectral tag  (clause D.3)": "Etiqueta espectral  (cláusula D.3)",
    "R  rumble: a band ≤ 500 Hz exceeds RC by > 5 dB":
        "R  retumbo: una banda ≤ 500 Hz supera RC en > 5 dB",
    "H  hiss: a band ≥ 1000 Hz exceeds RC by > 3 dB":
        "H  siseo: una banda ≥ 1000 Hz supera RC en > 3 dB",
    "N  neutral: within both tolerances":
        "N  neutro: dentro de ambas tolerancias",
    # Measuring the rated spectrum (ANSI/ASA S12.2-2019, clause 5.2.5)
    "Measuring the rated spectrum (ANSI/ASA S12.2-2019, clause 5.2.5)":
        "Medir el espectro que se califica (ANSI/ASA S12.2-2019, cláusula 5.2.5)",
    "ceiling plenum": "plénum del falso techo",
    "supply duct": "conducto de impulsión",
    "diffuser": "difusor",
    "air handler": "climatizadora",
    "design condition": "en su régimen de diseño",
    "0.6 m": "0,6 m",
    "1.2 m": "1,2 m",
    "2.4 m": "2,4 m",
    "1.1 m": "1,1 m",
    "0.75 m": "0,75 m",
    "≥ 0.6 m": "≥ 0,6 m",
    "≥ 1.2 m": "≥ 1,2 m",
    "≥ 2.4 m": "≥ 2,4 m",
    "Microphone height (5.2.5)": "Altura del micrófono (5.2.5)",
    "Adult, standing": "Adulto, de pie",
    "Adult, seated": "Adulto, sentado",
    "Child, standing": "Niño, de pie",
    "Child, seated": "Niño, sentado",
    "Standoff (5.2.5)": "Separación (5.2.5)",
    "One reflecting surface": "Una superficie",
    "Two surfaces meeting": "Dos superficies",
    "Three surfaces meeting": "Tres superficies",
    "Instrument and condition": "Instrumento y condición",
    "Integrating-averaging, $L_{EQ}$": "Integrador-promediador, $L_{EQ}$",
    "Class 2 minimum (5.1.1)": "Clase 2 como mínimo (5.1.1)",
    "Octave bands 16 Hz – 8 kHz": "Bandas de octava 16 Hz – 8 kHz",
    "Room unoccupied, plant running": "Sala vacía, instalación en marcha",
    "$L_{EQ}$ at the named position — or scan the whole space at ≤ 0.5 m/s "
    "for ≥ 20 s":
        "$L_{EQ}$ en la posición indicada — o barrer toda la sala a ≤ 0,5 m/s "
        "durante ≥ 20 s",
    "green dashed: microphone exclusion zones (5.2.5)":
        "verde discontinuo: zonas excluidas para el micrófono (5.2.5)",
    "Before rating (5.3.2): is the noise steady?":
        "Antes de calificar (5.3.2): ¿es estacionario el ruido?",
    # Both languages keep the two level differences on this first line, so
    # the pair composes the same mathematics; the Spanish drops the spelt
    # sound level meter to stay inside the panel.
    "screen 16, 31.5 and 63 Hz aurally and on a fast, Z-weighted meter, "
    "then check $L_{MAX} − L_{EQ}$ and $L_{10} − L_{EQ}$":
        "explorar 16, 31,5 y 63 Hz de oído y en rápida con ponderación Z; "
        "comprobar $L_{MAX} − L_{EQ}$ y $L_{10} − L_{EQ}$",
    "against Table 3 — a field that fails belongs to RNC (clause 5.3), "
    "not to NC or RC":
        "frente a la Tabla 3 — si falla, es RNC "
        "(cláusula 5.3), no NC ni RC",
    # Where the ISO 3382-3 measurement line goes (clauses 5.1 and 5.2)
    "Where the ISO 3382-3 measurement line goes (clauses 5.1 and 5.2)":
        "Dónde va la línea de medida de la ISO 3382-3 (cláusulas 5.1 y 5.2)",
    "(a) Plan — 30 × 12 m floor, two ceiling zones":
        "(a) Planta — superficie de 30 × 12 m, dos zonas de techo",
    "absorbent raft ceiling": "techo con islas absorbentes",
    "plain plaster ceiling": "techo de yeso liso",
    "zones measured and reported separately":
        "las zonas se miden y se reportan por separado",
    "≥ 2.0 m from walls and other reflecting surfaces":
        "≥ 2,0 m de paredes y otras superficies reflectantes",
    "1.2 m screens": "mamparas de 1,2 m",
    "P1 at the nearest workstation; the path need not be straight":
        "P1 en el puesto más cercano; la trayectoria no tiene que ser recta",
    "only 2 m to 16 m enter $D_{2,S}$":
        "solo de 2 m a 16 m entran en $D_{2,S}$",
    "≥ 0.5 m from tables": "≥ 0,5 m de las mesas",
    "(b) Section — both heights are 1.2 m (5.2.2)":
        "(b) Sección — ambas alturas son 1,2 m (5.2.2)",
    "omnidirectional, pink noise": "omnidireccional, ruido rosa",
    "seated head position": "posición de la cabeza sentada",
    "Source (5.1.1):": "Fuente (5.1.1):",
    "omnidirectional, pink noise, ISO 3382-1":
        "omnidireccional, ruido rosa, directividad",
    "directivity; a pink-spectrum sweep or":
        "según ISO 3382-1; también vale un barrido",
    "MLS may be used instead": "o un MLS de espectro rosa",
    "Receiver (5.1.2):": "Recepción (5.1.2):",
    "class 1 to IEC 61672-1, IEC 61260 octave":
        "clase 1 según IEC 61672-1, filtros de octava",
    "filters, omnidirectional capsule,":
        "IEC 61260, micrófono omnidireccional,",
    "≥ 10 s integration": "integración ≥ 10 s",
    "Room (5.2.1):": "Sala (5.2.1):",
    "furnished, nobody present but the":
        "amueblada, sin más personas que los",
    "operators, HVAC and any masking system":
        "operadores, climatización y enmascaramiento",
    "at working-day power": "a la potencia de un día normal",
    "Line (5.2.2):": "Línea (5.2.2):",
    "6 to 10 positions preferred, 4 the minimum;":
        "de 6 a 10 posiciones preferible, 4 el mínimo;",
    "≥ 2 source positions, or the line walked":
        "≥ 2 posiciones de fuente, o recorrer la línea",
    "in both directions": "en los dos sentidos",
    # Room-acoustics measurement setup (ISO 3382-1 / ISO 3382-2)
    "Room plan (top view) — 10.0 × 6.0 m, 3.5 m high":
        "Planta de la sala (vista superior) — 10,0 × 6,0 m, 3,5 m de altura",
    "avoid symmetry lines": "evitar las líneas de simetría",
    "2.4 m $> d_{min}$": "2,4 m $> d_{min}$",
    "$d_{min}$": "$d_{min}$",
    "• source height 1.5 m": "• fuente a 1,5 m de altura",
    "• off the symmetry axes": "• fuera de los ejes de simetría",
    "ISO 3382-2 (source clearance):":
        "ISO 3382-2 (separación a la fuente):",
    "$d_{min} = 2√(V/(c·T̂))$ = 2.0 m": "$d_{min} = 2√(V/(c·T̂))$ = 2,0 m",
    "for $V$ = 210 m³, $T̂$ = 0.6 s": "para $V$ = 210 m³, $T̂$ = 0,6 s",
    # The measuring chain in section (ISO 3382-1, 4.2 and 4.3)
    "The measuring chain in section (ISO 3382-1 clauses 4.2 and 4.3)":
        "La cadena de medida en sección (ISO 3382-1, cláusulas 4.2 y 4.3)",
    "Section through the same 10.0 × 6.0 × 3.5 m room":
        "Sección de la misma sala de 10,0 × 6,0 × 3,5 m",
    "dodecahedron": "dodecaedro",
    "acoustic centre": "centro acústico",
    "$d_{min}$ = 2.0 m": "$d_{min}$ = 2,0 m",
    "ISO 3382-1 Table 1 — omnidirectionality over gliding 30° arcs":
        "ISO 3382-1 Tabla 1 — omnidireccionalidad (arcos de 30°)",
    "Hz / dB, measured at ≥ 1.5 m — in practice a dodecahedron, not a monitor":
        "Hz / dB, medido a ≥ 1,5 m — en la práctica un dodecaedro, no un monitor",
    "Level (4.2.1):": "Nivel (4.2.1):",
    "≥ 45 dB over the background": "≥ 45 dB sobre el ruido de fondo",
    "per band for $T_{30}$, ≥ 35 dB for $T_{20}$":
        "por banda para $T_{30}$; 35 dB para $T_{20}$",
    "Receiving chain (4.2.2.2):": "Cadena de recepción (4.2.2.2):",
    "class 1 to IEC 61672-1,": "clase 1 según IEC 61672-1,",
    "IEC 61260 filters, omnidirectional": "filtros IEC 61260, micrófono",
    "capsule, ≤ 13 mm preferred": "omnidireccional, ≤ 13 mm preferible",
    # Dimensioning the ISO 18233 excitation (T = 1.2 s)
    "Dimensioning the excitation for a room with T = 1.2 s (ISO 18233)":
        "Dimensionado de la excitación para una sala con T = 1,2 s (ISO 18233)",
    "1  What you play, and how long you keep recording":
        "1  Qué se reproduce, y cuánto se sigue grabando",
    "sweep, 4.0 s = 3.3 × $T$": "barrido, 4,0 s = 3,3 × $T$",
    "silence $≈ T$": "silencio $≈ T$",
    "record window 5.2 s": "ventana de grabación 5,2 s",
    "B.3.1: sweep 2–4 × $T$, silent gap $≈ T$   |   B.6: +3 dB effective SNR "
    "per doubling":
        "B.3.1: barrido 2–4 × $T$, silencio $≈ T$   |   B.6: +3 dB de S/R "
        "efectiva por duplicación",
    "2  If the excitation repeats, the period must exceed $T$ (6.2.2.2)":
        "2  Si la excitación se repite, el periodo debe superar $T$ (6.2.2.2)",
    "period 1, warm-up: discarded": "periodo 1, arranque: se descarta",
    "period 2, kept": "periodo 2, se conserva",
    "order 17 → 2.73 s $≥ T$": "orden 17 → 2,73 s $≥ T$",
    "0.68 s": "0,68 s",
    "order 15 is shorter than $T$: the tail folds onto the head and $T$ comes "
    "out short":
        "el orden 15 es más corto que $T$: la cola se pliega sobre el inicio y "
        "$T$ sale corto",
    "3  After linear deconvolution (B.5)":
        "3  Tras la deconvolución lineal (B.5)",
    "kept by default: the linear impulse response and its tail":
        "se conserva por defecto: la respuesta al impulso lineal y su cola",
    "discarded, or read as distortion":
        "se descarta, o se lee como distorsión",
    "the linear deconvolution's own decaying noise tail — not the room":
        "la cola de ruido decreciente de la propia deconvolución — no la sala",
    "Arrival time relative to the linear impulse response [s]":
        "Tiempo de llegada respecto a la respuesta al impulso lineal [s]",
    # Hearing threshold (ISO 7029 / ISO 389-7)
    "Hearing-threshold model (ISO 7029 age distribution, ISO 389-7 zero)":
        "Modelo del umbral de audición (ISO 7029 por edad, cero ISO 389-7)",
    "Age $Y$,  sex,  population fractile $Q$":
        "Edad $Y$,  sexo,  fractil poblacional $Q$",
    "audiometric frequencies 125 Hz – 8000 Hz":
        "frecuencias audiométricas 125 Hz – 8000 Hz",
    "Median deviation from age 18   (ISO 7029, 4.2)":
        "Desviación mediana respecto a los 18 años   (ISO 7029, 4.2)",
    "dHmd = a · (Y − 18) ^ b   (Table 1, by sex)":
        "dHmd = a · (Y − 18) ^ b   (Tabla 1, por sexo)",
    "Spread su / sl   (ISO 7029, 4.3)":
        "Dispersión su / sl   (ISO 7029, 4.3)",
    "degree-5 polynomials in $(Y − 18)$   (Tables 2–5)":
        "polinomios de grado 5 en $(Y − 18)$   (Tablas 2–5)",
    "Fractile threshold   (ISO 7029, 4.4)":
        "Umbral del fractil   (ISO 7029, 4.4)",
    "dHQ = dHmd + z(Q) * s   (su if Q >= 0.5, else sl)":
        "dHQ = dHmd + z(Q) * s   (su si Q >= 0.5, si no sl)",
    "Expected hearing threshold level (dB HL)":
        "Nivel del umbral de audición esperado (dB HL)",
    "referenced to the audiometric zero":
        "referido al cero audiométrico",
    "Audiometric zero = ISO 389-7 reference threshold":
        "Cero audiométrico = umbral de referencia ISO 389-7",
    "free-field / diffuse-field (Table 1) — the dB HL / dB SPL zero":
        "campo libre / campo difuso (Tabla 1) — el cero dB HL / dB SPL",
    # Measurement uncertainty (ISO/IEC Guide 98-3 / Supplement 1)
    "Uncertainty: GUM propagation vs Monte Carlo (Guide 98-3)":
        "Incertidumbre: propagación GUM frente a Monte Carlo (Guía 98-3)",
    "Measurement model  $y = f(x_1, …, x_N)$":
        "Modelo de medida  $y = f(x_1, …, x_N)$",
    "input estimates $x_i$ with standard uncertainties $u(x_i)$":
        "estimaciones de entrada $x_i$ con incertidumbres típicas $u(x_i)$",
    "Law of propagation  (GUM 5)": "Ley de propagación  (GUM 5)",
    "sensitivity $c_i = ∂f / ∂x_i$": "sensibilidad $c_i = ∂f / ∂x_i$",
    "Combine in quadrature": "Combinación en cuadratura",
    "$u_c^2 = Σ c_i^2 u^{2}(x_i)$ + correlation":
        "$u_c^2 = Σ c_i^2 u^{2}(x_i)$ + correlación",
    "Effective dof  (Annex G.4)": "Grados de libertad efectivos  (Anexo G.4)",
    "$v_{eff}$ — Welch–Satterthwaite": "$v_{eff}$ — Welch–Satterthwaite",
    "$U = k · u_c$": "$U = k · u_c$",
    "$k = t_{p}(v_{eff})$   (clause 6)":
        "$k = t_{p}(v_{eff})$   (cláusula 6)",
    "Monte Carlo  (Suppl. 1, 7)": "Monte Carlo  (Supl. 1, 7)",
    "draw $x_i$ from its PDF $g(x_i)$":
        "muestrear $x_i$ de su PDF $g(x_i)$",
    "Propagate $M$ trials": "Propagar $M$ ensayos",
    "$y_r = f(x_{1r}, …, x_{Nr})$": "$y_r = f(x_{1r}, …, x_{Nr})$",
    "Sort ${y_r}$, take fractiles": "Ordenar ${y_r}$, tomar fractiles",
    "prob.-symmetric 95 % interval": "intervalo simétrico en prob. al 95 %",
    "coverage interval": "intervalo de cobertura",
    "$[y_{low}, y_{high}]$   (clause 7.7)":
        "$[y_{low}, y_{high}]$   (cláusula 7.7)",
    # Noise-induced hearing loss (ISO 1999)
    "Noise-induced hearing loss (ISO 1999): NIPTS and HTLAN":
        "Pérdida auditiva inducida por ruido (ISO 1999): NIPTS y HTLAN",
    "Age $Y$,  sex,  fractile $Q$": "Edad $Y$,  sexo,  fractil $Q$",
    "database A = ISO 7029": "base de datos A = ISO 7029",
    "Exposure L_EX,8h,  t years": "Exposición L_EX,8h,  t años",
    "normalized to 8 h / 5 days": "normalizada a 8 h / 5 días",
    "Age threshold  $H$  (HTLA)": "Umbral por edad  $H$  (HTLA)",
    "ISO 7029 fractile, dB": "fractil ISO 7029, dB",
    "Median NIPTS  $N_{50}$  (6.3.1)": "NIPTS mediana  $N_{50}$  (6.3.1)",
    "$N_{50} = [u + v·log_{10}(t/t_0)]·(L − L_0)^2$":
        "$N_{50} = [u + v·log_{10}(t/t_0)]·(L − L_0)^2$",
    "Fractile NIPTS  $N$  (6.3.2)": "NIPTS del fractil  $N$  (6.3.2)",
    "N = N50 + z·(du if z ≥ 0 else dl)":
        "N = N50 + z·(du si z ≥ 0, si no dl)",
    "HTLAN   $H′ = H + N − H·N / 120$": "HTLAN   $H′ = H + N − H·N / 120$",
    "threshold from age and noise  (Formula 1, 6.1)":
        "umbral por edad y ruido  (Fórmula 1, 6.1)",
    # Impulsive-sound prominence (NT ACOU 112)
    "Impulsive-sound prominence and LAeq adjustment (NT ACOU 112)":
        "Prominencia de sonidos impulsivos y ajuste de LAeq (NT ACOU 112)",
    "A-weighted level history  $L_{pAF}$  (time weighting F)":
        "Historia del nivel ponderado A  $L_{pAF}$  (ponderación F)",
    "an onset = a stretch where the gradient exceeds 10 dB/s (clauses 4.5-4.7)":
        "un arranque = tramo donde el gradiente supera 10 dB/s (cláusulas 4.5-4.7)",
    "Per impulse: onset rate OR and level difference LD":
        "Por impulso: tasa de crecimiento OR y diferencia de nivel LD",
    "OR = onset slope [dB/s],   $LD = L_e − L_s$ [dB]":
        "OR = pendiente de crecimiento [dB/s],   $LD = L_e − L_s$ [dB]",
    "Predicted prominence  $P$   (clause 7, Formula 1)":
        "Prominencia prevista  $P$   (cláusula 7, Fórmula 1)",
    "$P = 3·log_{10}(OR) + 2·log_{10}(LD)$;   highest $P$ over 30 min governs":
        "$P = 3·log_{10}(OR) + 2·log_{10}(LD)$;   la $P$ más alta en 30 min gobierna",
    "Adjustment  $K_I$   (clause 8, Formula 2)":
        "Ajuste  $K_I$   (cláusula 8, Fórmula 2)",
    "$K_I = 1.8·(P − 5)$ dB for $P > 5$, else 0":
        "$K_I = 1.8·(P − 5)$ dB si $P > 5$, si no 0",
    "Rating level  LAr,T = 10·log10( (1/T) Σ Δt·10^((LAeq+KI)/10) )":
        "Nivel de evaluación  LAr,T = 10·log10( (1/T) Σ Δt·10^((LAeq+KI)/10) )",
    "impulse-adjusted level over the reference time  (Note 1)":
        "nivel ajustado por impulsos sobre el tiempo de referencia  (Nota 1)",
    "Vertical seat acceleration  $a_{z}(t)$":
        "Aceleración vertical del asiento  $a_{z}(t)$",
    "conditioned per 5.1.3:  HP 0.01 Hz (2nd order) / LP 80 Hz (4th order)":
        "acondicionada según 5.1.3:  PA 0,01 Hz (2.º orden) / PB 80 Hz "
        "(4.º orden)",
    "not the ISO 2631-1 0.4 Hz / 100 Hz filters":
        "no los filtros de 0,4 Hz / 100 Hz de ISO 2631-1",
    "Spinal response  $A_{z}(t)$  (clause 5.2, Formula 1/2)":
        "Respuesta de la columna  $A_{z}(t)$  (cláusula 5.2, Fórmula 1/2)",
    "seat-to-spine transfer function $H(f)$: 1 zero, 6 poles":
        "función de transferencia asiento-columna $H(f)$: 1 cero, 6 polos",
    "Acceleration dose  $D_z = 1.07·(Σ A_{z,i}^6)^{1/6}$  (Formula 3)":
        "Dosis de aceleración  $D_z = 1,07·(Σ A_{z,i}^6)^{1/6}$  (Fórmula 3)",
    "$A_{z,i}$ = positive peaks;   daily dose $D_{zd} = D_z·(t_d/t_m)^{1/6}$":
        "$A_{z,i}$ = picos positivos;   dosis diaria "
        "$D_{zd} = D_z·(t_d/t_m)^{1/6}$",
    "Compressive stress  $S_d = m_z·D_{zd}$  (Annex C, Formula C.1)":
        "Tensión compresiva  $S_d = m_z·D_{zd}$  (Anexo C, Fórmula C.1)",
    "$m_z$ = 0.029 (male) / 0.025 (female) MPa per m/s²":
        "$m_z$ = 0,029 (hombre) / 0,025 (mujer) MPa por m/s²",
    "Stress variable  R = [Σ (Sd·N^(1/6) / (Su − Sstat))^6]^(1/6)":
        "Variable de tensión  R = [Σ (Sd·N^(1/6) / (Su − Sstat))^6]^(1/6)",
    "Su = 6.75 − Sage·(b+i) MPa, cumulated over exposure years (C.3/C.4)":
        "Su = 6.75 − Sage·(b+i) MPa, acumulada sobre los años de exposición (C.3/C.4)",
    "Injury probability  $P(R) = 1 − exp(−(R/α)^β)$  (Formula C.5)":
        "Probabilidad de lesión  $P(R) = 1 − exp(−(R/α)^β)$  (Fórmula C.5)",
    "Weibull risk of lumbar injury, by sex (Table C.1/C.2)":
        "riesgo de lesión lumbar de Weibull, por sexo (Tabla C.1/C.2)",
    "Multiple-shock spinal-response dose and injury risk (ISO 2631-5)":
        "Dosis espinal por choques múltiples y riesgo de lesión (ISO 2631-5)",
    "Surfaces  ($S_i$, $α_{s,i}$)": "Superficies  ($S_i$, $α_{s,i}$)",
    "area and absorption per band": "área y absorción por banda",
    "Objects  ($V_{obj}$)": "Objetos  ($V_{obj}$)",
    "$A_{obj} = V_{obj}^{2/3}$  (Formula 4)":
        "$A_{obj} = V_{obj}^{2/3}$  (Fórmula 4)",
    "Equivalent absorption area  $A$  (clause 4.3, Formula 1)":
        "Área de absorción equivalente  $A$  (cláusula 4.3, Fórmula 1)",
    "$A = Σ α_{s,i}·S_i + Σ A_{obj} + A_{air}$;   "
    "$A_{air} = 4·m·V·(1 − ψ)$  (Formula 2)":
        "$A = Σ α_{s,i}·S_i + Σ A_{obj} + A_{air}$;   "
        "$A_{air} = 4·m·V·(1 − ψ)$  (Fórmula 2)",
    "Object fraction  $ψ = Σ V_{obj} / V$   (Formula 3)":
        "Fracción de objetos  $ψ = Σ V_{obj} / V$   (Fórmula 3)",
    "air absorption negligible below 1 kHz for $V < 200$ m³":
        "absorción del aire despreciable bajo 1 kHz si $V < 200$ m³",
    "Reverberation time  $T = 55.3/c_0 · V·(1 − ψ) / A$  (Formula 5)":
        "Tiempo de reverberación  $T = 55,3/c_0 · V·(1 − ψ) / A$  (Fórmula 5)",
    "$c_0$ = 345.6 m/s so $55.3/c_0$ = 0.16  (clause 4.4)":
        "$c_0$ = 345,6 m/s, así $55,3/c_0$ = 0,16  (cláusula 4.4)",
    "Absorption area and reverberation time of a room (EN 12354-6)":
        "Área de absorción y tiempo de reverberación de una sala (EN 12354-6)",
    "Speech  $E′_i$": "Habla  $E′_i$",
    "Noise  $N′_i$": "Ruido  $N′_i$",
    "Threshold  $T′_i$": "Umbral  $T′_i$",
    "spectrum level (dB)": "nivel espectral (dB)",
    "Self-masking + spread of masking": "Automáscara + propagación de la máscara",
    "$Z_i$   (clause 5.4)": "$Z_i$   (cláusula 5.4)",
    "Equivalent disturbance $D_i$": "Perturbación equivalente $D_i$",
    "max(masking, internal noise) (5.6)":
        "máx(máscara, ruido interno) (5.6)",
    "Band audibility $A_i = (E′_i − D_i + 15)/30$":
        "Audibilidad de banda $A_i = (E′_i − D_i + 15)/30$",
    "clipped to [0, 1]   (clause 5.8)": "acotada a [0, 1]   (cláusula 5.8)",
    "band importance $I_i$ (Table 3)  ·  index in [0, 1]  (clause 6)":
        "importancia de banda $I_i$ (Tabla 3)  ·  índice en [0, 1]  (cláusula 6)",
    # Human vibration (ISO 2631-1 / ISO 8041-1 / 2002-44-EC)
    "Whole-body vibration measurement chain (ISO 2631-1 / ISO 8041-1)":
        "Medición de vibración de cuerpo entero (ISO 2631-1 / ISO 8041-1)",
    "vibration input": "entrada de vibración",
    "Seat/body interface": "Interfaz asiento/cuerpo",
    "Triaxial accelerometer": "Acelerómetro triaxial",
    "Band limiting + Wk / Wd": "Limitación de banda + Wk / Wd",
    "weighting (ISO 8041-1)": "ponderación (ISO 8041-1)",
    "Weighted r.m.s. $a_w$  &  VDV": "$a_w$ eficaz ponderada  y  VDV",
    "(ISO 2631-1)": "(ISO 2631-1)",
    "$A(8) = max(1.4·a_{wx} , 1.4·a_{wy} , a_{wz})·√(T/T_0)$":
        "$A(8) = max(1,4·a_{wx} , 1,4·a_{wy} , a_{wz})·√(T/T_0)$",
    "assessed vs EAV / ELV (Directive 2002/44/EC)":
        "evaluada frente a EAV / ELV (Directiva 2002/44/CE)",
    # Hand-transmitted vibration setup (ISO 5349-1/-2, Directive 2002/44/EC)
    "Hand-transmitted vibration: where the accelerometer goes":
        "Vibración mano-brazo: dónde va el acelerómetro",
    "On the tool (ISO 5349-2, 6.1.3)":
        "En la herramienta (ISO 5349-2, 6.1.3)",
    "gripping zone ≈ 100 mm": "zona de agarre ≈ 100 mm",
    "chain-saw front handle,": "empuñadura delantera",
    "Ø 30 mm tube": "de motosierra, tubo Ø 30 mm",
    "cable taped to the handle near the transducer (6.2.3)":
        "cable fijado a la empuñadura junto al transductor (6.2.3)",
    "grip force": "fuerza de agarre",
    "1  middle of the gripping zone, under the hand":
        "1  centro de la zona de agarre, bajo la mano",
    "the most representative location; needs an adaptor":
        "la ubicación más representativa; exige un adaptador",
    "2  either side of the hand — usual practice":
        "2  a ambos lados de la mano — práctica habitual",
    "on a side handle, average the two positions":
        "en empuñadura lateral, se promedian las dos",
    "3  underside of the handle, next to the hand":
        "3  bajo la empuñadura, junto a la mano",
    "grip and push force change the reading: report the":
        "las fuerzas de agarre y empuje alteran la lectura:",
    "posture and the applied forces (7.1, clause 9 g))":
        "documéntense postura y fuerzas (7.1, capítulo 9 g))",
    "basicentric frame (ISO 5349-1 Fig. 1):":
        "sistema basicéntrico (ISO 5349-1 Fig. 1):",
    "rotated so that y_h lies along the":
        "girado para que y_h siga el eje de",
    "handle axis. All three axes are":
        "la empuñadura. Se miden los tres",
    "measured, and every $k = 1$.": "ejes, y todo $k = 1$.",
    "Wh-weighted, one per axis (ISO 5349-1 A.1)":
        "ponderada Wh, una por eje (ISO 5349-1 A.1)",
    "vibration total value (Eq. (1))":
        "valor total de vibración (Ec. (1))",
    "$T_i$ is total contact time per day (5.5)":
        "$T_i$ es el tiempo total de contacto diario (5.5)",
    "one per hand, two significant figures (clause 8)":
        "una por mano, dos cifras significativas (capítulo 8)",
    "EAV 2.5 m/s²   ·   ELV 5 m/s²": "EAV 2,5 m/s²   ·   ELV 5 m/s²",
    "Directive 2002/44/EC, Article 3":
        "Directiva 2002/44/CE, artículo 3",
    "transducer and mount below 5 % of the":
        "transductor y montaje por debajo del",
    "mass they are fixed to (6.1.5)":
        "5 % de la masa sobre la que van (6.1.5)",
    "linear averaging over complete work":
        "promediado lineal sobre ciclos de",
    "cycles (6.1.11)": "trabajo completos (6.1.11)",
    "three samples per operation, a minute":
        "tres muestras por operación, un minuto",
    "of record, none under 8 s (5.4.1)":
        "de registro, ninguna bajo 8 s (5.4.1)",
    "the lowest input range that does not":
        "el menor rango de entrada que no",
    "overload, found by trial (6.1.10)":
        "sature, hallado por tanteo (6.1.10)",
    # Getting the ISO 2631-5 record (clauses 5.1.2 to 5.1.4)
    "Getting the record (ISO 2631-5, clauses 5.1.2 and 5.1.4)":
        "Obtención del registro (ISO 2631-5, capítulos 5.1.2 y 5.1.4)",
    "The seat pan, in section": "El asiento, en sección",
    "suspension travel": "recorrido de la suspensión",
    "semi-rigid mounting disc Ø 250 ± 50 mm, height ≤ 12 mm,":
        "disco de montaje semirrígido Ø 250 ± 50 mm, altura ≤ 12 mm,",
    "80-90 durometer (A), carrying a Ø 75 ± 5 mm × 1.5 mm":
        "80-90 durómetros (A), con un disco metálico de",
    "metal disc for the accelerometers (ISO 10326-1, 5.2.3)":
        "Ø 75 ± 5 mm × 1,5 mm para los acelerómetros (ISO 10326-1)",
    "taped to the cushion so the accelerometers sit midway":
        "pegado al cojín para que los acelerómetros queden",
    "between the ischial tuberosities (5.1.2)":
        "entre las tuberosidades isquiáticas (5.1.2)",
    "$z$ is positive to cranial: the method is about":
        "$z$ es positivo hacia craneal: el método trata de",
    "compressive spinal loading (5.1.3, first step)":
        "la carga de compresión espinal (5.1.3, primer paso)",
    "a contact switch or video detects loss of contact,":
        "un contacto o un vídeo detectan la pérdida de contacto,",
    "which is reported and excluded from the exposure":
        "que se documenta y se excluye de la exposición",
    "The record, split where contact is lost":
        "El registro, partido donde se pierde el contacto",
    "accelerations recorded while contact is lost":
        "las aceleraciones registradas sin contacto no",
    "shall not be counted as exposure, and the":
        "cuentan como exposición, y el impacto al caer",
    "landing impact after a free fall shall be (5.1.2)":
        "tras una caída libre sí cuenta (5.1.2)",
    "no contact": "sin contacto",
    "$a_{z}(t)$, conditioned per 5.1.3":
        "$a_{z}(t)$, acondicionada según 5.1.3",
    "segment 1": "segmento 1",
    "segment 2": "segmento 2",
    "segment 3": "segmento 3",
    "each segment is conditioned separately (5.1.3, second step)":
        "cada segmento se acondiciona por separado (5.1.3, 2.º paso)",
    "flat acceleration response from 0.01 Hz to at least 80 Hz, and "
    "256 samples per second or more (5.1.2)":
        "respuesta plana en aceleración de 0,01 Hz a 80 Hz como mínimo, "
        "y 256 muestras por segundo o más (5.1.2)",
    "equipment adequate for the highest amplitude anticipated; "
    "equipment and calibration method reported (5.1.2)":
        "equipo apto para la mayor amplitud prevista; se documentan el "
        "equipo y su calibración (5.1.2)",
    "long enough to be representative: a complete work cycle for a "
    "repeatable task, longer where the terrain varies (5.1.4)":
        "duración representativa: un ciclo de trabajo completo si la tarea "
        "se repite, más si el terreno varía (5.1.4)",
    "Sound calibrator": "Calibrador acústico",
    "Microphone +": "Micrófono +",
    "preamplifier": "preamplificador",
    "Audio interface": "Interfaz de audio",
    "Pa per": "Pa por",
    "digital unit": "unidad digital",
    "Stability: |max − mean| and |min − mean| ≤ 0.07 dB":
        "Estabilidad: |máx − media| y |mín − media| ≤ 0,07 dB",
    "(IEC 60942:2017 Table 2, class 1) — else CalibrationWarning":
        "(IEC 60942:2017 Tabla 2, clase 1) — si no, CalibrationWarning",
    # Coupling the calibrator on the capsule (signals/metrology/calibration).
    "Coupling the calibrator (IEC 60942:2017)":
        "Acoplamiento del calibrador (IEC 60942:2017)",
    # The calibrator tone: a unit symbol, the same in Spanish. Shared by the
    # calibration-chain, coupling and sound-level-meter-chain plates.
    "1 kHz": "1 kHz",
    "reference plane (3.12)": "plano de referencia (3.12)",
    "effective load": "volumen de carga",
    "volume (3.13)": "efectivo (3.13)",
    "the generated level moves": "el nivel generado cambia",
    "with that volume (6.3 k)": "con ese volumen (6.3 k)",
    "1/2 in capsule": "cápsula de 1/2 in",
    "+ preamplifier": "+ preamplificador",
    "1/4 in capsule": "cápsula de 1/4 in",
    "the adaptor is part of": "el adaptador forma parte",
    "the calibrator (5.1.1)": "del calibrador (5.1.1)",
    "windscreen: off for the": "pantalla antiviento: fuera en",
    "check, back on to measure": "la verificación, puesta al medir",
    "Before switching it on": "Antes de encenderlo",
    "source in use": "fuente en marcha",
    # Instrumenting a synchronous average (signals/spectra/synchronous-averaging).
    "Instrumenting a synchronous average: tacho and accelerometer":
        "Instrumentar un promediado síncrono: tacómetro y acelerómetro",
    "Gearbox — elevation": "Reductora — alzado",
    "pinion, 37 teeth": "piñón, 37 dientes",
    "wheel, 89 teeth": "rueda, 89 dientes",
    "tape": "cinta",
    "tacho": "tacómetro",
    "bearing": "cojinete",
    "accel.": "acel.",
    "(stud)": "(perno)",
    "load direction": "dirección de carga",
    "1800 r/min  →  $T$ = 33.3 ms per revolution":
        "1800 r/min  →  $T$ = 33,3 ms por vuelta",
    "mesh frequency 37 × 30 = 1110 Hz; five mesh harmonics":
        "frecuencia de engrane 37 × 30 = 1110 Hz; cinco armónicos",
    "need $f_s ≥ 2.56 × 5550$ = 14.2 kHz":
        "exigen $f_s ≥ 2,56 × 5550$ = 14,2 kHz",
    "One front end, one clock": "Un solo frontal, un solo reloj",
    "vibration": "vibración",
    "$f_s$ = 25.6 kHz, gains fixed": "$f_s$ = 25,6 kHz, ganancias fijas",
    "one pulse per revolution": "un pulso por vuelta",
    "the pulse is the block boundary": "el pulso marca el bloque",
    "record $N + 1$ revolutions": "registra $N + 1$ vueltas",
    "One tacho pulse per turn, an accelerometer in the load direction: the "
    "period is measured, not assumed":
        "Un pulso de tacómetro por vuelta y un acelerómetro en la carga: el "
        "periodo se mide, no se supone",
    # The geometry behind an echo's quefrency (signals/spectra/cepstrum-echoes).
    "Where the quefrency comes from: the geometry of one reflection":
        "De dónde sale la quefrencia: la geometría de una reflexión",
    # "image source" is already in the table further down; not repeated here.
    "microphone": "micrófono",
    "Floor reflection": "Reflexión en el suelo",
    # The path lengths and the two closed forms carry decimals, so they are
    # listed to get the Spanish comma; the symbols themselves are shared.
    "$r_d$ = 1.00 m": "$r_d$ = 1,00 m",
    "$r_r$ = 2.60 m": "$r_r$ = 2,60 m",
    "$Δd = r_r − r_d$ = 1.60 m": "$Δd = r_r − r_d$ = 1,60 m",
    "$t_0 = Δd / c$ = 4.7 ms": "$t_0 = Δd / c$ = 4,7 ms",
    "$a = R · r_d / r_r = 0.38 R$": "$a = R · r_d / r_r = 0,38 R$",
    "The 8 ms example of this page": "El ejemplo de 8 ms de esta página",
    "$Δd = c · 8 ms$ = 2.74 m": "$Δd = c · 8 ms$ = 2,74 m",
    "$R = a · r_r / r_d = 3.74 a ≤ 1$":
        "$R = a · r_r / r_d = 3,74 a ≤ 1$",
    "a side wall 1.37 m from the direct path":
        "una pared lateral a 1,37 m del camino directo",
    "so $a > 0.27$ is not one specular reflection":
        "así que $a > 0,27$ no es una sola reflexión especular",
    "The reflection has to arrive before the record ends and at least 10 dB "
    "above its noise floor;":
        "La reflexión debe llegar antes de que acabe el registro y al menos "
        "10 dB por encima de su ruido de fondo;",
    "$c$ moves about 0.6 m/s per kelvin, so convert the delay with the "
    "temperature you measured":
        "$c$ cambia unos 0,6 m/s por kelvin: convierte el retardo con la "
        "temperatura medida",
    # Instrumenting a MISO measurement (signals/spectra/miso-coherence).
    "Instrumenting a MISO measurement: one reference per source":
        "Instrumentar una medida MISO: una referencia por fuente",
    "Plant room — plan": "Sala de máquinas — planta",
    "A — fan": "A — ventilador",
    "B — compressor": "B — compresor",
    "receiver": "receptor",
    "1.5 m high": "a 1,5 m de altura",
    "ref 1": "ref 1",
    "ref 2": "ref 2",
    # "ref 1  x1" and "ref 2  x2" are channel identifiers ("ref" abbreviates
    # referencia too), so they read the same in Spanish.
    "ref 1  $x_1$": "ref 1  $x_1$",
    "ref 2  $x_2$": "ref 2  $x_2$",
    "receiver  $y$": "receptor  $y$",
    "leakage": "fuga",
    "One front end": "Un solo frontal",
    "one clock, fixed gains": "un reloj, ganancias fijas",
    "Before reading": "Antes de leer",
    "the split": "el reparto",
    "coherence between": "coherencia entre",
    "$x_1$ and $x_2$ > 0.9": "$x_1$ y $x_2$ > 0,9",
    "⇒ do not attribute": "⇒ no atribuir",
    "ref 1: accelerometer stud-mounted on the fan foot":
        "ref 1: acelerómetro atornillado a la pata del ventilador",
    "ref 2: microphone 0.3 m from the casing":
        "ref 2: micrófono a 0,3 m de la carcasa",
    "the leakage is what correlates the two inputs":
        "la fuga es lo que correlaciona las dos entradas",
    "0.3 m": "0,3 m",
    "Conditioning separates only what the references separate: one sensor per "
    "source, all sampled together":
        "El condicionamiento solo separa lo que separan las referencias: un "
        "sensor por fuente, a la vez",
    # Where an acoustic budget's terms come from (signals/metrology/gum).
    "Where an acoustic budget's terms come from":
        "De dónde salen los términos de un balance acústico",
    "facade": "fachada",
    "calibrator": "calibrador",
    "weather": "meteorología",
    "3 positions, 2 m apart": "3 posiciones, separadas 2 m",
    "The budget it feeds": "El balance que alimenta",
    "Meteorology and ground": "Meteorología y suelo",
    "Type B - from the propagation clause":
        "Tipo B - del capítulo de propagación",
    "Position scatter": "Dispersión entre posiciones",
    "Type A - $s/√n$, $v = n − 1$": "Tipo A - $s/√n$, $v = n − 1$",
    "Instrument class tolerance": "Tolerancia de clase del instrumento",
    "Type B - rectangular, $a$ = 0.3 dB":
        "Tipo B - rectangular, $a$ = 0,3 dB",
    "Calibrator class tolerance": "Tolerancia de clase del calibrador",
    "Type B - rectangular, $a$ = 0.4 dB":
        "Tipo B - rectangular, $a$ = 0,4 dB",
    "one calibrator for two channels makes their":
        "un solo calibrador para dos canales correlaciona",
    "calibration terms correlated, not two rows":
        "sus términos de calibración: no son dos filas",
    "Budgets fail by omission, not by arithmetic: every row here is a piece "
    "of hardware or a decision about geometry":
        "Los balances fallan por omisión, no por aritmética: cada fila es un "
        "equipo o una decisión de geometría",
    "3 m/s  12 °C": "3 m/s  12 °C",
    "68 % RH": "68 % HR",
    "the coupled capsule must": "la cápsula acoplada debe",
    "read ≥ 30 dB below the": "leer ≥ 30 dB por debajo del",
    "calibrator level: under 64 dB": "nivel del calibrador: < 64 dB",
    "for a 94 dB calibrator": "para uno de 94 dB",
    "(B.4.2; 40 dB in A.5.3)": "(B.4.2; 40 dB en A.5.3)",
    "The specified level is the level at the diaphragm of the inserted "
    "microphone (5.3.1.2), and it holds for":
        "El nivel especificado es el nivel en el diafragma del micrófono "
        "insertado (5.3.1.2), y vale para",
    "the microphone models and configurations listed in the manual "
    "(5.3.1.3, 6.3 a) — IEC 60942:2017":
        "los modelos y configuraciones de micrófono que lista el manual "
        "(5.3.1.3, 6.3 a) — IEC 60942:2017",
    "Environmental noise measurement positions (ISO 1996-2)":
        "Posiciones de medida de ruido ambiental (ISO 1996-2)",
    "Building façade": "Fachada del edificio",
    "A — free field": "A — campo libre",
    "B — 2 m from façade": "B — a 2 m de la fachada",
    "C — flush-mounted": "C — enrasado en fachada",
    "4.0 ± 0.2 m": "4,0 ± 0,2 m",
    "Emission measurement positions (ECMA-74)":
        "Posiciones de medida de emisión (ECMA-74)",
    "Operator — seated (P2)": "Operador — sentado (P2)",
    "Bystanders — top view": "Observadores — vista en planta",
    "height 1.50 m": "altura 1,50 m",
    "0.25 m": "0,25 m",
    "1.20 m": "1,20 m",
    "1.00 m": "1,00 m",
    # Loudness capture geometry (perception/psychoacoustics/loudness).
    "Where the microphone goes for a loudness measurement (ISO 532-1)":
        "Dónde va el micrófono en una medida de sonoridad (ISO 532-1)",
    "A — Free field  (NF)": "A — Campo libre  (NF)",
    "hemi-anechoic room, one frontal source":
        "sala semianecoica, una sola fuente frontal",
    "listener absent:": "sin oyente:",
    "the result is diotic": "el resultado es diótico",
    "frontal incidence, 0°": "incidencia frontal, 0°",
    "1.50 m": "1,50 m",
    'field="free"  →  quote N as NF':
        'field="free"  →  cita N como NF',
    "B — Diffuse field  (ND)": "B — Campo difuso  (ND)",
    "reverberant or in-situ room": "sala reverberante o in situ",
    "direct sound plus the reflected field, from every direction":
        "sonido directo más el campo reflejado, desde todas las direcciones",
    'field="diffuse"  →  quote N as ND':
        'field="diffuse"  →  cita N como ND',
    "C — Head-and-torso simulator (Annex D)":
        "C — Simulador de cabeza y torso (Anexo D)",
    "at the listening position": "en la posición de escucha",
    "Equalization matched": "Ecualización acorde",
    "to the room:": "con la sala:",
    "free-field / diffuse-field / ID": "campo libre / campo difuso / ID",
    "left channel": "canal izquierdo",
    "right channel": "canal derecho",
    "both reported": "se informan ambos",
    "free-field equalization only for one frontal source beyond 1.5 m; "
    "diffuse-field in reflective rooms; ID in vehicles":
        "ecualización de campo libre solo con una fuente frontal a más de "
        "1,5 m; de campo difuso en salas reflectantes; ID en vehículos",
    "each channel is analysed separately: report NL and NR, and quote the "
    "maximum or the mean as the single value":
        "cada canal se analiza por separado: se informan NL y NR y se cita "
        "el máximo o la media como valor único",
    # The spectra behind an ISO/PAS 20065 assessment
    # (perception/psychoacoustics/tone-audibility).
    "The spectra an ISO/PAS 20065 assessment is built on":
        "Los espectros de una evaluación ISO/PAS 20065",
    "1 — the source runs through its operating states (clause 5.1: all of "
    "them must be covered)":
        "1 — la fuente recorre sus estados de funcionamiento (capítulo 5.1: "
        "hay que cubrirlos todos)",
    "idle": "ralentí",
    "full load": "plena carga",
    "2 — the analyser's basic spectra (under 1 s each) are merged line by "
    "line into 3 s spectra (clause 4.3)":
        "2 — los espectros básicos (menos de 1 s) se combinan línea a línea "
        "en espectros de 3 s (capítulo 4.3)",
    "3 — each merged spectrum gives one decisive audibility $ΔL_j$ "
    "(clause 5.3.8)":
        "3 — cada espectro combinado da una audibilidad decisiva $ΔL_j$ "
        "(capítulo 5.3.8)",
    "Energy mean of the $J$ decisive audibilities":
        "Media energética de las $J$ audibilidades decisivas",
    "Formula (20); an empty spectrum counts as −10 dB (Formula 21)":
        "Fórmula (20); un espectro sin tono cuenta como −10 dB (Fórmula 21)",
    "mean audibility $ΔL$  →  tonal adjustment $K_t$":
        "audibilidad media $ΔL$  →  ajuste tonal $K_t$",
    "ISO 1996-2:2017 Annex J, Table J.1":
        "ISO 1996-2:2017 Anexo J, Tabla J.1",
    "class 1 chain (IEC 61672-1), lower limit ≤ 20 Hz":
        "cadena de clase 1 (IEC 61672-1), límite inferior ≤ 20 Hz",
    "line spacing $Δf$ between 1.9 Hz and 4.0 Hz":
        "espaciado de líneas $Δf$ entre 1,9 Hz y 4,0 Hz",
    "Hanning window, mandatory": "ventana de Hanning, obligatoria",
    "amplitude resolution ≥ 0.1 dB, anti-aliasing filter":
        "resolución de amplitud ≥ 0,1 dB, filtro antialias",
    "A-weighted spectrum (clause 5.3.2)":
        "espectro ponderado A (capítulo 5.3.2)",
    "$U ≤ 1.5$ dB: below 12 spectra, $U$ must be reported":
        "$U ≤ 1,5$ dB: con menos de 12 espectros, informar $U$",
    # ISO 532-2 capture routes (perception/psychoacoustics/advanced-loudness).
    "Which recording maps to which arguments (ISO 532-2 clause 7.2)":
        "Qué grabación corresponde a qué argumentos (ISO 532-2, cap. 7.2)",
    "1 — single microphone where the head would be, one frontal source":
        "1 — un micrófono donde estaría la cabeza, una fuente frontal",
    "listener absent": "sin oyente",
    "(diotic)": "(diótico)",
    "Table 1 free-field transfer":
        "función de transferencia de campo libre (Tabla 1)",
    "frontal incidence; the default": "incidencia frontal; es el valor por "
        "defecto",
    "2 — the same microphone, reverberant or in-situ field":
        "2 — el mismo micrófono, campo reverberante o in situ",
    "Table 1 diffuse-field transfer":
        "función de transferencia de campo difuso (Tabla 1)",
    "also for diffuse-field earphones":
        "también para auriculares de campo difuso",
    "3 — probe microphone in the ear canal":
        "3 — micrófono de sonda en el canal auditivo",
    "tympanic membrane": "membrana timpánica",
    "probe microphone": "micrófono de sonda",
    "10 mm — 5 mm above 3 kHz": "10 mm — 5 mm por encima de 3 kHz",
    "no transfer function applied":
        "no se aplica función de transferencia",
    "the ear transfer is already in the signal":
        "la transferencia del oído ya está en la señal",
    "4 — head-and-torso simulator": "4 — simulador de cabeza y torso",
    "accurate model of": "¿modelo fiel de un",
    "an average adult?": "adulto medio?",
    "yes: no correction": "sí: sin corrección",
    "no: correction file —": "no: archivo de corrección —",
    "not implemented": "no implementado",
    'field="eardrum"  or  equalize': 'field="eardrum"  o  ecualizar',
    "clause 7.2.5": "capítulo 7.2.5",
    "equalize to the free or diffuse field first":
        "ecualiza antes al campo libre o difuso",
    "presentation: monaural is one ear alone, diotic the same signal at both "
    "ears, binaural two independent ear signals":
        "presentation: monaural es un solo oído, diotic la misma señal en "
        "ambos, binaural dos señales independientes",
    "a diotic sound is about 1.5 times as loud as the same sound at one ear "
    "(clause 8.1)":
        "un sonido diótico suena unas 1,5 veces más fuerte que el mismo "
        "sonido en un solo oído (capítulo 8.1)",
    "phonometry processing chain": "Cadena de procesado de phonometry",
    "Signal": "Señal",
    "Calibrate": "Calibrar",
    "Weighting": "Ponderación",
    "Octave": "Octavas",
    "bands $1/b$": "bandas $1/b$",
    "Ballistics": "Temporal",
    "Metrics": "Métricas",
    # Level symbols, the same in the Spanish edition.
    "$L_{eq}$, $L_N$…": "$L_{eq}$, $L_N$…",
    "Multirate decimation in the octave filter bank":
        "Decimación multitasa en el banco de filtros de octava",
    # Unit symbols, the same in Spanish; "fs = 48 kHz" is shared with the
    # multichannel-capture plate.
    "$f_s$ = 48 kHz": "$f_s$ = 48 kHz",
    "6 kHz": "6 kHz",
    "16 kHz band": "Banda de 16 kHz",
    "1 kHz band": "Banda de 1 kHz",
    "63 Hz band": "Banda de 63 Hz",
    "no decimation": "sin decimación",
    "Anti-alias": "Antialias",
    "Low bands are filtered at a decimated rate: the relative":
        "Las bandas graves se filtran a frecuencia decimada: el ancho",
    "bandwidth stays wide, so the SOS stays numerically healthy.":
        "relativo se mantiene amplio y las SOS siguen bien condicionadas.",
    "Two-microphone (p-p) intensity probe":
        "Sonda de intensidad p-p (dos micrófonos)",
    "measurement axis / intensity direction":
        "eje de medida / dirección de la intensidad",
    "$u$ from the $p_2−p_1$ gradient":
        "$u$ a partir del gradiente $p_2−p_1$",
    "STI measurement chain (IEC 60268-16)":
        "Cadena de medida STI (IEC 60268-16)",
    "Source": "Fuente",
    "STIPA signal": "Señal STIPA",
    "Room": "Sala",
    "reverberation + noise": "reverberación + ruido",
    "Microphone": "Micrófono",
    "Analysis": "Análisis",
    "$m(F)$ drops": "$m(F)$ cae",
    "Airborne sound insulation setup (ISO 16283-1)":
        "Montaje de aislamiento acústico aéreo (ISO 16283-1)",
    "Source room": "Recinto emisor",
    "Test partition": "Partición de ensayo",
    "microphone positions": "posiciones de micrófono",
    "≥ 1.0 m": "≥ 1,0 m",
    "≥ 0.7 m": "≥ 0,7 m",
    "7.6 a) ≥ 0.7 m between microphone positions":
        "7.6 a) ≥ 0,7 m entre posiciones de micrófono",
    "7.6 b) ≥ 0.5 m to room boundaries":
        "7.6 b) ≥ 0,5 m a los límites del recinto",
    "7.6 c) ≥ 1.0 m to the loudspeaker":
        "7.6 c) ≥ 1,0 m al altavoz",
    "7.2.2 ≥ 1.0 m loudspeaker to separating partition":
        "7.2.2 ≥ 1,0 m del altavoz a la partición separadora",
    "Impulse-response measurement chain (ISO 18233)":
        "Cadena de medición de la respuesta al impulso (ISO 18233)",
    "Excitation": "Excitación",
    "ESS sweep / MLS": "Barrido ESS / MLS",
    "Deconvolution": "Deconvolución",
    "correlation /": "correlación /",
    "inverse filter": "filtro inverso",
    "acoustic path": "trayecto acústico",
    "The room response $h(t)$ is recovered by deconvolving the microphone signal.":
        "La respuesta de la sala $h(t)$ se recupera deconvolucionando "
        "la señal del micrófono.",
    # d10 - ISO 3744/3746 sound power measurement surfaces
    "ISO 3744 / 3746 sound power measurement surfaces":
        "Superficies de medición de potencia acústica (ISO 3744 / 3746)",
    "Hemispherical surface": "Superficie hemisférica",
    "Reflecting plane": "Plano reflectante",
    "Measurement surface": "Superficie de medición",
    "Parallelepiped surface": "Superficie de paralelepípedo",
    "radius $r ≥ 2 d_0$": "radio $r ≥ 2 d_0$",
    "measurement distance $d$": "distancia de medición $d$",
    "10 key positions (Table B.1)": "10 posiciones clave (Tabla B.1)",
    "one plane · $S = 2πr^2$": "un plano · $S = 2πr^2$",
    "one plane · $S = 4(a·b+b·c+c·a)$": "un plano · $S = 4(a·b+b·c+c·a)$",
    # d11 - ISO 16283-2 impact sound insulation setup
    "ISO 16283-2 impact sound insulation setup":
        "Montaje de aislamiento de ruido de impactos (ISO 16283-2)",
    "Source room (upper)": "Recinto emisor (superior)",
    "Receiving room (lower)": "Recinto receptor (inferior)",
    "Separating floor": "Forjado separador",
    "Tapping machine": "Máquina de impactos",
    "Microphone positions": "Posiciones de micrófono",
    "structure-borne impact": "impacto estructural",
    "radiated impact sound": "ruido de impactos radiado",
    "Impact sound insulation": "Aislamiento de impactos",
    "$L_i$ = energy-averaged": "$L_i$ = promedio en energía",
    "band level (Formula 10)": "del nivel de banda (Fórmula 10)",
    "$A = 0.16 V/T$  (Sabine)": "$A = 0,16 V/T$  (Sabine)",
    "$T_0$ = 0.5 s , $A_0$ = 10 m²": "$T_0$ = 0,5 s , $A_0$ = 10 m²",
    # d12 - sound power methods comparison
    "Sound power methods compared": "Métodos de potencia acústica comparados",
    "Free field over a reflecting plane":
        "Campo libre sobre plano reflectante",
    "Reverberation test room": "Cámara reverberante de ensayo",
    "In situ — any environment": "In situ — cualquier entorno",
    "Grade 2 / 3 (engineering / survey)":
        "Grado 2 / 3 (ingeniería / control)",
    "Grade 1 (precision)": "Grado 1 (precisión)",
    "Sound pressure · enveloping surface":
        "Presión acústica · superficie envolvente",
    "Sound pressure · diffuse field": "Presión acústica · campo difuso",
    "Sound intensity · scanning": "Intensidad acústica · barrido de intensidad",
    "$K_{2A}$ ≤ 4 dB (3744) / ≤ 7 dB (3746)":
        "$K_{2A}$ ≤ 4 dB (3744) / ≤ 7 dB (3746)",
    "$V$ ≥ 200 m³ , source ≤ 2 % of $V$":
        "$V$ ≥ 200 m³ , fuente ≤ 2 % de $V$",
    "no non-positive bands · $F_{pI} < L_d$":
        "sin bandas no positivas · $F_{pI} < L_d$",
    "Qualified anechoic / hemi-anechoic room":
        "Cámara anecoica o semianecoica cualificada",
    "Sound pressure · fixed 20 / 40 array":
        "Presión acústica · malla fija de 20 / 40",
    "$r ≥ 2 d_0$ , qualified free field":
        "$r ≥ 2 d_0$ , campo libre cualificado",
    "Sound intensity · scanning, tighter":
        "Intensidad acústica · barrido, más exigente",
    "five Annex C criteria per band":
        "cinco criterios del anexo C por banda",
    "Any — no acoustic measurement": "Cualquiera — sin medida acústica",
    "Upper limit ($ε = 1$) / engineering":
        "Límite superior ($ε = 1$) / peritaje",
    "Surface velocity · accelerometers":
        "Velocidad superficial · acelerómetros",
    "$ε$ assumed (-1) or measured (-2)": "$ε$ supuesto (-1) o medido (-2)",
    "Method": "Método",
    "Environment": "Entorno",
    "Accuracy": "Exactitud",
    # The two headline relations read the same in Spanish (log10/lg are
    # shared notation); only "log"/"lg" trip the language check.
    "$L_W = 10 log_{10} |Σ I_i·S_i| / W_0$":
        "$L_W = 10 log_{10} |Σ I_i·S_i| / W_0$",
    "$L_{WA} = L_{vA} + 10 lg(S/S_0) + 10 lg ε$":
        "$L_{WA} = L_{vA} + 10 lg(S/S_0) + 10 lg ε$",
    # d13 - EN 12354 direct and flanking transmission paths
    "Direct and flanking transmission paths (EN 12354)":
        "Caminos de transmisión directa y por flancos (EN 12354)",
    "Separating element (D, d)": "Elemento separador (D, d)",
    "Flanking element (F, f)": "Elemento de flanco (F, f)",
    "junction": "unión",
    "Dd — direct path: separating element both sides":
        "Dd — camino directo: elemento separador en ambos lados",
    "Ff — flanking–flanking: flanking element both sides":
        "Ff — flanco–flanco: elemento de flanco en ambos lados",
    "Fd — flanking (source) → separating (receiving)":
        "Fd — flanco (emisor) → separador (receptor)",
    "Df — separating (source) → flanking (receiving)":
        "Df — separador (emisor) → flanco (receptor)",
    "R'w = −10 log10 Σ 10^(−Rij,w /10) dB   (EN 12354-1, Formula 26)":
        "R'w = −10 log10 Σ 10^(−Rij,w /10) dB   (EN 12354-1, Fórmula 26)",
    # d14 - ISO 9613-2 outdoor propagation geometry
    "ISO 9613-2 source–barrier–receiver geometry":
        "Geometría fuente–barrera–receptor (ISO 9613-2)",
    "Receiver": "Receptor",
    "Barrier": "Barrera",
    "Ground ($G_s$, $G_m$, $G_r$)": "Suelo ($G_s$, $G_m$, $G_r$)",
    "diffracted path": "trayecto difractado",
    "direct path (blocked)": "trayecto directo (bloqueado)",
    "z = dss + dsr − d   (path difference)":
        "z = dss + dsr − d   (diferencia de camino)",
    # ISO 9613-2 Eq. (14) symbols, identical in the Spanish edition.
    "dss": "dss",
    "dsr": "dsr",
    "Dz = 10 log10[ 3 + (C₂/λ) C₃ z Kmet ]   (Eq. 14)":
        "Dz = 10 log10[ 3 + (C₂/λ) C₃ z Kmet ]   (Ec. 14)",
    # Impedance tube (ISO 10534) setup
    "Impedance tube: two-microphone method (ISO 10534-2)":
        "Tubo de impedancia: método de dos micrófonos (ISO 10534-2)",
    "Test specimen": "Probeta de ensayo",
    "Rigid backing": "Terminación rígida",
    # The flush-microphone labels, shared by the two-microphone tube, the
    # ASTM four-microphone tube and the ISO 13472-2 spot tube, abbreviated
    # as "Pos. micróf." already does further down this table.
    "Mic 1": "Micróf. 1",
    "Mic 2": "Micróf. 2",
    "Mic 3": "Micróf. 3",
    "Mic 4": "Micróf. 4",
    "incident": "incidente",
    "reflected": "reflejada",
    "$H_{12}$ → reflection factor $r$ (Eq. 17), "
    "absorption $α = 1 − |r|^2$ (Eq. 18), "
    "$Z/ρc_0 = (1+r)/(1−r)$ (Eq. 19)":
        "$H_{12}$ → factor de reflexión $r$ (Ec. 17), "
        "absorción $α = 1 − |r|^2$ (Ec. 18), "
        "$Z/ρc_0 = (1+r)/(1−r)$ (Ec. 19)",
    "Working range $f_l < f < f_u$ set by the microphone spacing "
    "$s$ and the tube diameter (Clause 6.1)":
        "Rango útil $f_l < f < f_u$ fijado por la separación $s$ de "
        "micrófonos y el diámetro del tubo (apartado 6.1)",
    "ASTM E2611: two further microphones behind the specimen also "
    "give the transmission loss":
        "ASTM E2611: dos micrófonos más tras la probeta dan también "
        "la pérdida por transmisión",
    # Four-microphone tube (ASTM E2611) setup
    "Four-microphone transmission-loss tube (ASTM E2611)":
        "Tubo de pérdida por transmisión de cuatro micrófonos (ASTM E2611)",
    "Termination": "Terminación",
    "(2 loads)": "(2 cargas)",
    "Decompose $A$, $B$ (upstream) and $C$, $D$ (downstream) → "
    "transfer matrix $T$ (Eq. 22)":
        "Descomponer $A$, $B$ (aguas arriba) y $C$, $D$ (aguas abajo) → "
        "matriz de transferencia $T$ (Ec. 22)",
    "$TL = 20 log_{10} |(T_{11} + T_{12}/ρc + ρc·T_{21} + T_{22}) / 2|$"
    "   (Eq. 26)":
        "$TL = 20 log_{10} |(T_{11} + T_{12}/ρc + ρc·T_{21} + T_{22}) / 2|$"
        "   (Ec. 26)",
    "Two-load method: repeat with two terminations; the one-load "
    "variant uses a single anechoic end":
        "Método de dos cargas: repetir con dos terminaciones; la variante "
        "de una carga usa un único extremo anecoico",
    # Airflow resistance (ISO 9053) setup
    "Airflow resistance: static and alternating methods (ISO 9053-1/-2)":
        "Resistencia al flujo: métodos estático y alternante (ISO 9053-1/-2)",
    "Static method (ISO 9053-1)": "Método estático (ISO 9053-1)",
    "specimen (A, d)": "probeta (A, d)",
    "laminar flow  q_v": "flujo laminar  q_v",
    "manom.": "manóm.",
    "$R = Δp / q_v$   (through-origin fit at 0.5 mm/s)":
        "$R = Δp / q_v$   (ajuste por el origen a 0,5 mm/s)",
    "Alternating method (ISO 9053-2)": "Método alternante (ISO 9053-2)",
    "cavity": "cavidad",
    "specimen / airtight": "probeta / cierre estanco",
    "piston  $f$ = 1–4 Hz": "pistón  $f$ = 1–4 Hz",
    "$R$ from $L_{p,s} − L_{p,t}$   ($κ′$ per Annex A)":
        "$R$ por $L_{p,s} − L_{p,t}$   ($κ′$ según Anexo A)",
    "seal": "sellado",
    "grid": "rejilla",
    "specimen  $A$, $d$": "probeta  $A$, $d$",
    "flow source": "fuente de caudal",
    # "≥ 1 diámetro" (78 px) poked 8 px past the plate's left edge at this
    # dimension's 62 px slot; the abbreviation measures 58 px.
    "≥ 1 bore": "≥ 1 diám.",
    "cell ≥ 29 mm bore, ≥ 1 bore of free space above":
        "celda ≥ 29 mm de diámetro, ≥ 1 diámetro libre por encima",
    "$q_v$ and $Δp$ each to ±5 %, $Δp$ readable to 0.1 Pa":
        "$q_v$ y $Δp$ con ±5 % cada uno, $Δp$ legible hasta 0,1 Pa",
    "grid ≥ 50 % open, $R < 1$ %; $d$ measured in position":
        "rejilla ≥ 50 % abierta, $R < 1$ %; $d$ medido en posición",
    "measurement cell → $L_{p,s}$ ($h_s$)":
        "celda de medida → $L_{p,s}$ ($h_s$)",
    "airtight termination → $L_{p,t}$ ($h_t$)":
        "terminación estanca → $L_{p,t}$ ($h_t$)",
    # d24 - ISO 354 reverberation-room sound absorption
    "Reverberation-room sound absorption (ISO 354)":
        "Absorción acústica en cámara reverberante (ISO 354)",
    "Reverberation room · plan": "Cámara reverberante · planta",
    "$V$ = 200 m³ (≥ 150 m³)": "$V$ = 200 m³ (≥ 150 m³)",
    "diffusers  0.8–3 m² each, ≈ 5 kg/m² (Annex A)":
        "difusores  0,8–3 m² cada uno, ≈ 5 kg/m² (Anexo A)",
    "Test specimen  $S$ = 10.8 m²": "Probeta de ensayo  $S$ = 10,8 m²",
    "10–12 m², width/length 0.7–1, edges not parallel to the room":
        "10–12 m², anchura/longitud 0,7–1, bordes no paralelos a la cámara",
    "microphones ≥ 1.5 m apart, ≥ 2 m from a source, ≥ 1 m from any "
    "surface and from the specimen":
        "micrófonos separados ≥ 1,5 m, a ≥ 2 m de una fuente y a ≥ 1 m de "
        "cualquier superficie y de la probeta",
    "≥ 0.75 m": "≥ 0,75 m",
    "≥ 1.5 m": "≥ 1,5 m",
    "The measurement is a difference": "La medida es una diferencia",
    "1 · empty room": "1 · cámara vacía",
    "2 · specimen installed": "2 · probeta instalada",
    "Annex B mounting (part of the result)":
        "Montaje del Anexo B (parte del resultado)",
    "Type A: directly on the rigid floor":
        "Tipo A: directamente sobre el suelo rígido",
    "Type E-400: 400 mm face to floor":
        "Tipo E-400: 400 mm de la cara al suelo",
    "perimeter frame, flush": "marco perimetral, enrasado",
    "$A = 55.3 V/(c T) − 4 V m$   ·   $c = 331 + 0.6 t$  (15–30 °C)":
        "$A = 55,3 V/(c T) − 4 V m$   ·   $c = 331 + 0,6 t$  (15–30 °C)",
    "≥ 12 spatially independent decays = ≥ 3 microphones × ≥ 2 sources "
    "· $T_{20}$ read from −5 dB over 20 dB":
        "≥ 12 curvas de caída espacialmente independientes = ≥ 3 micrófonos "
        "× ≥ 2 fuentes · $T_{20}$ leído desde −5 dB sobre 20 dB",
    "the empty-room $A_1$ must clear the Table 1 ceiling, and $T_1$ is "
    "measured without the specimen frame":
        "$A_1$ de la cámara vacía debe quedar bajo el techo de la Tabla 1, y "
        "$T_1$ se mide sin el marco de la probeta",
    # d25 - ISO 10534-1 standing-wave-ratio apparatus
    # The Spanish is deliberately terser than a literal rendering: the
    # fuller "recorrido de la sonda y los mínimos" measures 87 px wider
    # again. On the DejaVu metrics the plates now set, both titles overrun
    # the 900 px sheet (1054 px here, 1063 px for the English), so the
    # tightness is still doing work and both still want cutting.
    "Standing-wave-ratio tube: probe traverse and the minima (ISO 10534-1)":
        "Tubo de onda estacionaria: recorrido de sonda y mínimos "
        "(ISO 10534-1)",
    "one pure tone at a time": "un tono puro cada vez",
    "Test specimen on the rigid backing":
        "Probeta sobre la terminación rígida",
    "probe microphone on a graduated carriage":
        "micrófono de sonda sobre carro graduado",
    "$|p(x)|$ envelope": "envolvente $|p(x)|$",
    "$Z/ρc_0 = (1 + r)/(1 − r)$ = 1.13 − 1.22j":
        "$Z/ρc_0 = (1 + r)/(1 − r)$ = 1,13 − 1,22j",
    # The rest of the same reduction chain, and the ΔL dimension: decimal
    # commas so the chain does not mix separators line to line.
    "$|r| = (s − 1)/(s + 1)$ = 0.5": "$|r| = (s − 1)/(s + 1)$ = 0,5",
    "$α = 1 − |r|^2$ = 0.75": "$α = 1 − |r|^2$ = 0,75",
    "$Φ = 4π x_{min,1}/λ − π$ = −54.1°":
        "$Φ = 4π x_{min,1}/λ − π$ = −54,1°",
    "$ΔL$ = 9.54 dB": "$ΔL$ = 9,54 dB",
    # 703 px overflowed the note's 680 px slot by 11 px on the left; dropping
    # the article and "en la pared" -> "de pared" brings it to 664 px.
    "minima far from the specimen fill in (wall losses, exaggerated "
    "here): read the nearest one":
        "mínimos lejanos a la probeta se rellenan (pérdidas de pared, "
        "exageradas aquí): leer el más cercano",
    "one channel: the microphone sensitivity cancels and there is no "
    "inter-channel phase mismatch":
        "un solo canal: la sensibilidad del micrófono se cancela y no hay "
        "desajuste de fase entre canales",
    "magnitude from the ratio, phase from the position — which is why "
    "Part 1 is the arbitration method":
        "magnitud por la razón, fase por la posición — por eso la Parte 1 es "
        "el método de arbitraje",
    # d15 - ISO 17497-1 random-incidence scattering (reverberation room)
    "Random-incidence scattering in a reverberation room (ISO 17497-1)":
        "Dispersión a incidencia aleatoria en cámara reverberante (ISO 17497-1)",
    "Reverberation room": "Cámara reverberante",
    "Turntable and base plate": "Plataforma giratoria y placa base",
    "sample on the plate for $T_2$ and $T_4$":
        "probeta sobre la placa en $T_2$ y $T_4$",
    "the only thing that moves": "lo único que se mueve",
    # "≥ 1.0 m" (the turntable wall clearance) is already in the table above.
    "fixed sources (≥ 2)": "fuentes fijas (≥ 2)",
    "fixed microphones (≥ 3)": "micrófonos fijos (≥ 3)",
    "$T_1$ base plate, static  ·  $T_2$ sample, static  →  $α_s$ (Eq. 1)":
        "$T_1$ placa base, estática  ·  $T_2$ probeta, estática  →  "
        "$α_s$ (Ec. 1)",
    "$T_3$ base plate, rotating  ·  $T_4$ sample, rotating  →  "
    "$α_{spec}$ (Eq. 4)":
        "$T_3$ placa base, girando  ·  $T_4$ probeta, girando  →  "
        "$α_{spec}$ (Ec. 4)",
    "$s = (α_{spec} − α_s) / (1 − α_s)$   (Eq. 5)":
        "$s = (α_{spec} − α_s) / (1 − α_s)$   (Ec. 5)",
    "$α$ from $55.3·(V/S)·(1/(c T)) − 4(V/S)m$  ·  the base plate "
    "must pass the Table 1 ceiling":
        "$α$ con $55,3·(V/S)·(1/(c T)) − 4(V/S)m$  ·  la placa base debe "
        "cumplir el límite de la Tabla 1",
    # d16 - ISO 17497-2 free-field diffusion goniometer
    "Free-field diffusion goniometer (ISO 17497-2)":
        "Goniómetro de difusión en campo libre (ISO 17497-2)",
    "Test sample": "Probeta de ensayo",
    "Turntable": "Plataforma giratoria",
    "Fixed source": "Fuente fija",
    "polar response $L_i$": "respuesta polar $L_i$",
    "receiver arc (5° steps)": "arco de receptores (pasos de 5°)",
    "d = [(Σ10^(L_i/10))² − Σ(10^(L_i/10))²] / [(n−1)·Σ(10^(L_i/10))²]   (Formula 5)":
        "d = [(Σ10^(L_i/10))² − Σ(10^(L_i/10))²] / [(n−1)·Σ(10^(L_i/10))²]   (Fórmula 5)",
    "$d_n = (d − d_{ref}) / (1 − d_{ref})$   (Formula 7)":
        "$d_n = (d − d_{ref}) / (1 − d_{ref})$   (Fórmula 7)",
    "5° receiver steps · turntable rotates the sample · source fixed":
        "pasos de 5° entre receptores · la plataforma gira la probeta · fuente fija",
    # d17 - ISO 13472-1 in-situ road absorption, subtraction technique
    "In-situ road absorption — subtraction technique (ISO 13472-1)":
        "Absorción in situ de carreteras — técnica de sustracción (ISO 13472-1)",
    "Road surface": "Superficie de la carretera",
    "direct  $d_s−d_m$": "directo  $d_s−d_m$",
    "reflected  $d_s+d_m$": "reflejado  $d_s+d_m$",
    "to image source ($d_s$ below)":
        "hacia fuente imagen ($d_s$ por debajo)",
    "$d_s$ = 1.25 m": "$d_s$ = 1,25 m",
    "$d_m$ = 0.25 m": "$d_m$ = 0,25 m",
    "Free-field reference": "Referencia en campo libre",
    "$H_i$: no ground reflection in the window":
        "$H_i$: sin reflexión del suelo en la ventana",
    "$K_r = (d_s − d_m)/(d_s + d_m) = 2/3$   (Clause 4.1)":
        "$K_r = (d_s − d_m)/(d_s + d_m) = 2/3$   (apartado 4.1)",
    "$α(f) = 1 − (1/K_r^2)·|H_r/H_i|^2$   ·   $Δτ = 2 d_m / c$":
        "$α(f) = 1 − (1/K_r^2)·|H_r/H_i|^2$   ·   $Δτ = 2 d_m / c$",
    "Adrienne time window isolates the reflected response $H_r$":
        "La ventana temporal Adrienne aísla la respuesta reflejada $H_r$",
    # d18 - ISO 13472-2 in-situ road absorption, spot method
    "In-situ road absorption — spot method (ISO 13472-2)":
        "Absorción in situ de carreteras — método puntual (ISO 13472-2)",
    "Road surface (test sample)": "Superficie de carretera (probeta)",
    "Spot method (ISO 13472-2)": "Método puntual (ISO 13472-2)",
    "$f_u = 0.58 c_0 / d$   (Clause 5.4.1)":
        "$f_u = 0,58 c_0 / d$   (apartado 5.4.1)",
    "$0.05 c_0/f_{min} < s < 0.45 c_0/f_{max}$   (Clause 5.4.2)":
        "$0,05 c_0/f_{min} < s < 0,45 c_0/f_{max}$   (apartado 5.4.2)",
    "Working range: 250–1600 Hz (1/3-octave)":
        "Rango útil: 250–1600 Hz (1/3 de octava)",
    "Two-microphone transfer function $H_{12}$":
        "Función de transferencia de dos micrófonos $H_{12}$",
    "→ ISO 10534-2 decomposition → $α(f)$":
        "→ descomposición ISO 10534-2 → $α(f)$",
    "Tube sealed onto the road; plane waves only below $f_u$":
        "Tubo sellado sobre la carretera; solo ondas planas por debajo "
        "de $f_u$",
    # d19 - ISO 3745 precision sound power (anechoic / hemi-anechoic room)
    "Precision sound power in an anechoic room (ISO 3745)":
        "Potencia acústica de precisión en cámara anecoica (ISO 3745)",
    "Reflecting plane (hemi-anechoic)": "Plano reflectante (semianecoica)",
    "Anechoic wedges": "Cuñas anecoicas",
    "Source (DUT)": "Fuente (DUT)",
    "20 / 40 mic positions": "20 / 40 puntos de micrófono",
    "radius $r$": "radio $r$",
    "$S = 2πr^2$ (hemi-anechoic) · $4πr^2$ (anechoic)":
        "$S = 2πr^2$ (semianecoica) · $4πr^2$ (anecoica)",
    "$K_1$: per-position background correction":
        "$K_1$: corrección de ruido de fondo por posición",
    "$C_1$, $C_2$, $C_3$: meteorological corrections ($p_s$, $θ$, $a(f)$)":
        "$C_1$, $C_2$, $C_3$: correcciones meteorológicas "
        "($p_s$, $θ$, $a(f)$)",
    # d20 - ISO 9614-3 precision sound intensity scanning
    "Precision sound intensity scanning (ISO 9614-3)":
        "Barrido de intensidad acústica de precisión (ISO 9614-3)",
    "Measurement surface (segments $S_i$)":
        "Superficie de medición (segmentos $S_i$)",
    "p-p probe": "sonda p-p",
    "serpentine scan": "barrido en serpentina",
    "$I_n$ (normal intensity)": "$I_n$ (intensidad normal)",
    "$P = Σ I_{n,i} · S_i$   (partial powers per segment)":
        "$P = Σ I_{n,i} · S_i$   (potencias parciales por segmento)",
    "Field indicators: $F_{pIn}$ , $F_T$ , $F_S$":
        "Indicadores de campo: $F_{pIn}$ , $F_T$ , $F_S$",
    "Five acceptance criteria (Annex C); band invalid if $P < 0$":
        "Cinco criterios de aceptación (Anexo C); banda no válida si $P < 0$",
    # d_room - ISO 3382-1/-2 room-acoustics measurement setup
    "Room-acoustics measurement setup (ISO 3382-1 / ISO 3382-2)":
        "Montaje de medición de acústica de salas (ISO 3382-1 / ISO 3382-2)",
    "Room plan (top view)": "Planta de la sala (vista superior)",
    "Microphone position": "Posición de micrófono",
    "Loudspeaker source": "Fuente (altavoz)",
    "ISO 3382-1 (positions):": "ISO 3382-1 (posiciones):",
    "• ≥ 2 source positions": "• ≥ 2 posiciones de fuente",
    "• mics ≥ 2 m apart": "• micrófonos ≥ 2 m entre sí",
    "• ≥ 1 m from surfaces": "• ≥ 1 m de las superficies",
    "• mic height 1.2 m": "• altura del micrófono 1,2 m",
    "ISO 3382-2 — reverberation-time measurement grades":
        "ISO 3382-2 — grados de medición del tiempo de reverberación",
    "Source pos.": "Pos. fuente",
    "Mic pos.": "Pos. micróf.",
    "Source–mic comb.": "Comb. fuente–micróf.",
    "Decays / comb.": "Decaim. / comb.",
    "Survey": "Control",
    "Engineering": "Ingeniería",
    "Precision": "Precisión",
    # --- Tanda 11: new diagrams -------------------------------------------
    "Exponential-detector chain of the time weightings (IEC 61672-1)":
        "El detector exponencial de las ponderaciones temporales (IEC 61672-1)",
    "Block processing: carrying the filter state versus resetting it":
        "Procesado por bloques: conservar el estado del filtro frente a "
        "reiniciarlo",
    "Array-shape flow through a per-channel operation":
        "Flujo de formas de array en una operación por canal",
    "Open-plan office spatial decay of speech (ISO 3382-3)":
        "Caída espacial del habla en oficina diáfana (ISO 3382-3)",
    "what open_plan_metrics returns": "lo que devuelve open_plan_metrics",
    "Clause 4 also requires the average A-weighted background "
    "noise  $L_{p,A,B}$  (Cl. 6.4)":
        "La cláusula 4 exige además el ruido de fondo medio ponderado A  "
        "$L_{p,A,B}$  (cl. 6.4)",
    "Measurement uncertainty from tables to expanded U (ISO 12999-1)":
        "Incertidumbre de medición: de tablas a la U expandida (ISO 12999-1)",
    "Single-number sound-absorption rating (ISO 11654)":
        "Valoración de la absorción acústica en índice único (ISO 11654)",
    "Zwicker loudness model chain (ISO 532-1)":
        "Cadena del modelo de sonoridad de Zwicker (ISO 532-1)",
    # equal-loudness -> A-weighting
    "Why A-weighting: an equal-loudness contour, inverted (ISO 226)":
        "Por qué la ponderación A: una isófona invertida (ISO 226)",
    "Equal-loudness contours (ISO 226)": "Líneas isofónicas (ISO 226)",
    "Frequency [Hz]": "Frecuencia [Hz]",
    "40 phon": "40 fonios",
    "invert": "invertir",
    "0 dB at 1 kHz": "0 dB en 1 kHz",
    "A-weighting (IEC 61672-1)": "Ponderación A (IEC 61672-1)",
    "inverted 40-phon contour": "isófona de 40 fonios invertida",
    "A is the 40-phon contour flipped into a realizable filter: quiet sounds, "
    "where the ear discards bass hardest.":
        "A es la isófona de 40 fonios convertida en un filtro realizable: "
        "niveles bajos, donde el oído descarta más los graves.",
    "The match is deliberately loose (a 1930s convention, not a loudness "
    "model); C mirrors the flatter ~100-phon contour.":
        "Coincidencia laxa a propósito (convención de los años 30, no un "
        "modelo de sonoridad); C sigue la isófona plana de ~100 fonios.",
    # time-weighting
    "a first-order low-pass on the squared signal → the mean-square envelope":
        "un paso bajo de primer orden sobre la señal al cuadrado → la "
        "envolvente cuadrática media",
    "band signal": "señal de banda",
    "square": "cuadrado",
    "one-pole RC": "RC de un polo",
    "time constant $τ$": "constante de tiempo $τ$",
    "to decibels": "a decibelios",
    "time-weighted level": "nivel con ponderación temporal",
    "Fast (F)": "Rápida (F)",
    "Slow (S)": "Lenta (S)",
    "Impulse (I)": "Impulso (I)",
    "35 ms rise · 1500 ms fall": "35 ms subida · 1500 ms bajada",
    # block-processing
    "State carried across blocks — TimeWeighting.process()":
        "Estado conservado entre bloques — TimeWeighting.process()",
    "y[-1] (or the sosfilt zi vector) seeds the next block → identical to "
    "one continuous call":
        "y[-1] (o el vector zi de sosfilt) inicializa el bloque siguiente → "
        "idéntico a una llamada continua",
    "State reset each block — reset() or a fresh call":
        "Estado reiniciado en cada bloque — reset() o una llamada nueva",
    "every block restarts from rest → spurious discontinuities at the seams":
        "cada bloque arranca desde reposo → discontinuidades espurias en las "
        "uniones",
    "block 1": "bloque 1",
    "block 2": "bloque 2",
    "block 3": "bloque 3",
    # multichannel
    "1-D:  (samples,)": "1-D:  (muestras,)",
    "scalar": "escalar",
    "2-D:  (channels, samples)": "2-D:  (canales, muestras)",
    "(channels,)": "(canales,)",
    "reduce along": "reducir sobre",
    "axis = −1  (time)": "eje = −1  (tiempo)",
    "the channel axis 0": "el eje de canal 0",
    "rides through untouched": "pasa intacto",
    "A mono call returns a scalar; a $C$-channel call returns $C$ results.":
        "Una llamada mono devuelve un escalar; una de $C$ canales devuelve "
        "$C$ resultados.",
    "Band metrics widen the reduced axis instead: (…, bands).":
        "Las métricas por banda ensanchan el eje reducido: (…, bandas).",
    # open-plan
    "spatial-decay fit range (2 m to 16 m)":
        "rango de ajuste de caída espacial (2 m a 16 m)",
    "spatial decay rate": "tasa de caída espacial",
    "dB per doubling · Cl. 6.2": "dB por duplicación · Cl. 6.2",
    "speech level at 4 m": "nivel de habla a 4 m",
    "A-weighted · Cl. 3.3": "ponderado A · Cl. 3.3",
    "distraction distance": "distancia de distracción",
    "fitted STI = 0.50 · Cl. 3.6": "STI ajustado = 0,50 · Cl. 3.6",
    "privacy distance": "distancia de privacidad",
    "fitted STI = 0.20 · Cl. 3.7": "STI ajustado = 0,20 · Cl. 3.7",
    # ISO 12999-1
    "Standard uncertainty  $u$  — reproducibility read from the tables":
        "Incertidumbre típica  $u$  — reproducibilidad leída de las tablas",
    "bands: Tables 2/4 · ratings: Tables 3/5 · situation A ($σ_R$) / B ($σ_{situ}$) / "
    "C ($σ_r$)":
        "bandas: Tablas 2/4 · índices: Tablas 3/5 · situación A ($σ_R$) / "
        "B ($σ_{situ}$) / C ($σ_r$)",
    "Reduce by  $m$  independent measurements   $u/√m$   (Formula A.7)":
        "Reducir con  $m$  mediciones independientes   $u/√m$   (Fórmula A.7)",
    "and combine model with reality per Annex A when predicting":
        "y combinar modelo con realidad según el Anexo A al predecir",
    "Combine uncorrelated contributions   $u_c = √(Σ u_i^2)$   (Formula C.2)":
        "Combinar contribuciones no correlacionadas   $u_c = √(Σ u_i^2)$   "
        "(Fórmula C.2)",
    "single-number combination of Annex B uses Formula B.2":
        "la combinación de índice único del Anexo B usa la Fórmula B.2",
    "Expand   $U = k·u$   (Formula 2),   $k$ from Table 8   ($k ≥ 1$)":
        "Expandir   $U = k·u$   (Fórmula 2),   $k$ de la Tabla 8   ($k ≥ 1$)",
    "the coverage factor depends on the reported quantity and situation":
        "el factor de cobertura depende de la magnitud reportada y la situación",
    "Report   $Y = y ± U$   (Formula 3)":
        "Reportar   $Y = y ± U$   (Fórmula 3)",
    "two-sided coverage factor": "factor de cobertura bilateral",
    "Declare conformity   (Formulae 4/5)": "Declarar conformidad   (Fórmulas 4/5)",
    "one-sided coverage factor": "factor de cobertura unilateral",
    # ISO 11654
    "Measured  $α_s$  at one-third octaves, 200 Hz to 5000 Hz":
        "$α_s$ medido en tercios de octava, 200 Hz a 5000 Hz",
    "from a reverberation room (ISO 354)": "en cámara reverberante (ISO 354)",
    "Practical  $α_p$  per octave band, 250 Hz to 4000 Hz  (Clause 4.1)":
        "$α_p$ práctico por banda de octava, 250 Hz a 4000 Hz  "
        "(apartado 4.1)",
    "mean of the three one-third octaves, rounded to 0.05":
        "media de los tres tercios de octava, redondeado a 0,05",
    "read off the shifted curve, always a multiple of 0.05":
        "leído sobre la curva desplazada, siempre múltiplo de 0,05",
    "appended to $α_w$ in parentheses, e.g. 0.60(M)":
        "añadido a $α_w$ entre paréntesis, p. ej. 0,60(M)",
    "Shift the reference curve in 0.05 steps to best fit  (Clause 4.2)":
        "Desplazar la curva de referencia en pasos de 0,05 hasta el mejor "
        "ajuste  (apartado 4.2)",
    "sum of unfavourable deviations kept ≤ 0.10":
        "suma de desviaciones desfavorables ≤ 0,10",
    "Weighted coefficient  $α_w$ = shifted reference at 500 Hz":
        "Coeficiente ponderado  $α_w$ = referencia desplazada a 500 Hz",
    "Shape indicators (L, M, H) where  $α_p$ − reference ≥ 0.25":
        "Indicadores de forma (L, M, H) donde  $α_p$ − referencia ≥ 0,25",
    "Sound absorption class  A to E   (Table B.1, Annex B)":
        "Clase de absorción acústica  A a E   (Tabla B.1, Anexo B)",
    "or “Not classified” when $α_w$ falls below the class-E band":
        "o «No clasificado» cuando $α_w$ cae por debajo de la banda de "
        "clase E",
    # Zwicker
    "28 one-third-octave band levels, 25 Hz to 12.5 kHz":
        "28 niveles de banda de tercio de octava, 25 Hz a 12,5 kHz",
    "from a spectrum, or from a calibrated signal via the Annex A filterbank":
        "de un espectro, o de una señal calibrada mediante el banco de filtros "
        "del Anexo A",
    "Equal-loudness correction and lower critical bands  "
    "(Clause 5.4, Table A.3)":
        "Corrección de igual sonoridad y bandas críticas inferiores  "
        "(Cláusula 5.4, Tabla A.3)",
    "the 11 lowest bands grouped into 3 critical bands, 25-250 Hz":
        "las 11 bandas más bajas agrupadas en 3 bandas críticas, 25-250 Hz",
    "Core loudness of the 20 critical bands  (Tables A.4-A.7)":
        "Sonoridad de núcleo de las 20 bandas críticas  (Tablas A.4-A.7)",
    "a₀ transmission (A.4), diffuse-field DDF (A.5), threshold in quiet "
    "LTQ (A.6)":
        "transmisión a₀ (A.4), DDF de campo difuso (A.5), umbral en silencio "
        "LTQ (A.6)",
    "Specific loudness  $N′(z)$  over 0.1-Bark steps to 24 Bark":
        "Sonoridad específica  $N′(z)$  en pasos de 0,1 Bark hasta 24 Bark",
    "upper masking slopes added band to band (Table A.9)":
        "pendientes de enmascaramiento superior sumadas banda a banda (Tabla A.9)",
    "Total loudness  $N = ∫ N′(z) dz$  [sone]":
        "Sonoridad total  $N = ∫ N′(z) dz$  [sone]",
    "loudness level  $L_N = 40 + 10·log_2 N$  [phon]":
        "nivel de sonoridad  $L_N = 40 + 10·log_2 N$  [phon]",
    # Loudspeaker free-field sensitivity (IEC 60268-5)
    "Loudspeaker free-field sensitivity measurement (IEC 60268-5)":
        "Sensibilidad de altavoz en campo libre (IEC 60268-5)",
    "Reference axis": "Eje de referencia",
    "Measurement microphone": "Micrófono de medición",
    "Amplifier": "Amplificador",
    "2.83 V (8 Ω)": "2,83 V (8 Ω)",
    "Characteristic sensitivity: $L_p$ at 1 m for 1 W into the rated impedance":
        "Sensibilidad característica: $L_p$ a 1 m para 1 W en la impedancia nominal",
    "Up = √(R · 1 W): 2.83 V is 1 W into 8 Ω but 2 W into 4 Ω (+3 dB)":
        "Up = √(R · 1 W): 2,83 V es 1 W en 8 Ω pero 2 W en 4 Ω (+3 dB)",
    "Lp(1 m) = Lp(r) + 20 log10(r / 1 m)   (far field, inverse-distance law)":
        "Lp(1 m) = Lp(r) + 20 log10(r / 1 m)   (campo lejano, ley 1/r)",
    "Microphone (IEC 60268-4): M in mV/Pa, or LM = 20 log10(M / 1 V/Pa) dB":
        "Micrófono (IEC 60268-4): M en mV/Pa, o LM = 20 log10(M / 1 V/Pa) dB",
    # Polar directional response (IEC 60268-5 clause 23)
    "Polar directional response measurement (IEC 60268-5 clause 23)":
        "Medición de la respuesta polar (IEC 60268-5, apartado 23)",
    "Anechoic room, plan view": "Cámara anecoica, vista en planta",
    "$θ$ stepped by 10° or 15°": "$θ$ en pasos de 10° o 15°",
    "reference axis  0°": "eje de referencia  0°",
    "measuring microphone": "micrófono de medición",
    "reference point on the rotation axis:":
        "punto de referencia en el eje de giro:",
    "$r$ never changes as $θ$ is swept": "$r$ no cambia al barrer $θ$",
    "Drive condition (23.1.2.3)": "La excitación (23.1.2.3)",
    # One sentence over three drawn lines; the Spanish redistributes it,
    # keeping the L_p on the same drawn line as the English so the pair
    # composes with the same math structure.
    "input voltage retrimmed at each": "la tensión de entrada se reajusta por",
    "frequency or band so that $L_p$ on": "frecuencia o banda para que $L_p$",
    "the reference axis stays constant":
        "en el eje de referencia no varíe",
    "The cut, on the IEC 60263 circle":
        "El corte, en el círculo de IEC 60263",
    "outer ring = the reference-axis level,":
        "anillo exterior = nivel del eje de referencia,",
    "full radius = 25 dB (clause 3)": "radio completo = 25 dB (apartado 3)",
    # "Índice de directividad" would push the line 50 px past the English
    # extent; D_i in the formula still names the index.
    "Directivity: free field on axis against a reverberation room, "
    "$D_i = L_{ax} − L_p + 10 lg(T/T_0) − 10 lg(V/V_0)$ + 25 dB":
        "Directividad: campo libre en eje contra cámara reverberante, "
        "$D_i = L_{ax} − L_p + 10 lg(T/T_0) − 10 lg(V/V_0)$ + 25 dB",
    "(23.3.2.1), or by integrating these polar curves over the sphere "
    "(23.3.2.2).":
        "(23.3.2.1), o integrando estas curvas polares sobre la esfera "
        "(23.3.2.2).",
    # The three microphone sensitivity fields (IEC 60268-4)
    "The three fields a microphone sensitivity is defined in (IEC 60268-4)":
        "Los tres campos de la sensibilidad de un micrófono (IEC 60268-4)",
    "Free field (11.2.1)": "Campo libre (11.2.1)",
    "Diffuse field (11.2.2)": "Campo difuso (11.2.2)",
    "Pressure (11.2.4)": "Presión (11.2.4)",
    "one source on the reference axis,":
        "una fuente en el eje de referencia,",
    "far enough that $r ≥ d$,": "tan lejos que $r ≥ d$,",
    "$r ≥ d^2/λ$ and $r$ ≥ 3 × the source":
        "$r ≥ d^2/λ$ y $r$ ≥ 3 × la fuente",
    "$M_{ff}$ : the undisturbed": "$M_{ff}$ : la presión de la",
    "pressure of the plane wave": "onda plana sin perturbar",
    "sound from every direction,": "sonido de todas las direcciones,",
    "with equal probability": "con igual probabilidad",
    "(a reverberation room)": "(una cámara reverberante)",
    "$M_{diff}$ : the r.m.s. of $M(θ)$": "$M_{diff}$ : valor eficaz de $M(θ)$",
    "a coupler or a calibrator:": "un acoplador o un calibrador:",
    "the pressure the capsule": "la presión que la propia",
    "itself replaces": "cápsula sustituye",
    "cavity small against $λ$": "cavidad pequeña frente a $λ$",
    "$M_p$ : pressure at": "$M_p$ : la presión en",
    "the acoustic entry": "la entrada acústica",
    "The bench (clauses 5.5.2, 5.6.2, 5.7)":
        "El banco (apartados 5.5.2, 5.6.2, 5.7)",
    "anechoic room; the spherical wave counts as plane at least $λ/2$ "
    "from the centre of curvature at the lowest frequency":
        "cámara anecoica; la onda esférica cuenta como plana a $λ/2$ o más "
        "del centro de curvatura a la frecuencia más baja",
    "substitution: the microphone under test and a calibrated reference "
    "at the same point, one after the other (highest accuracy)":
        "sustitución: el micrófono en ensayo y una referencia calibrada "
        "en el mismo punto, uno tras otro (máxima exactitud)",
    "simultaneous comparison at two nearby points only after showing it "
    "agrees with substitution within ± 1 dB":
        "comparación simultánea en dos puntos próximos solo tras demostrar "
        "que coincide con la sustitución en ± 1 dB",
    "overall accuracy ± 2 dB or better":
        "exactitud global de ± 2 dB o mejor",
    "Polar cuts (13.1.2 a): distance, sound pressure and frequency held "
    "constant while $θ$ is stepped, preferably by 10° or 15°,":
        "Cortes polares (13.1.2 a): distancia, presión acústica y "
        "frecuencia fijas al barrer $θ$, preferentemente a 10° o 15°,",
    "at the octave centres 125 Hz to 16 kHz, with the reference axis as "
    "0° of the polar diagram.":
        "en los centros de octava de 125 Hz a 16 kHz, con el eje de "
        "referencia como 0° del diagrama polar.",
    # The IEC 60268-3 distortion bench
    "The IEC 60268-3 distortion bench and its operating point":
        "El banco de distorsión de IEC 60268-3 y su punto de funcionamiento",
    "1 kHz sine": "sinusoide de 1 kHz",
    # "rated source impedance" over two drawn lines; the Spanish order
    # differs, so the two line fragments swap content.
    "rated source": "impedancia de",
    "impedance": "fuente nominal",
    # "Amplificador en ensayo" breaks the 206 px box that "Amplifier under
    # test" fills exactly (236 px at the drawn size); the chain position and
    # the page name the role, so the head keeps only the device.
    "Amplifier under test": "Amplificador",
    "Class A / B / D": "Clase A / B / D",
    # "Impedancia de carga nominal" runs 29 px past the canvas; the next
    # drawn line ("resistencia no inductiva de 8 Ω,") keeps the quantity.
    "Rated load impedance": "Carga nominal",
    "8 Ω non-inductive resistor,": "resistencia no inductiva de 8 Ω,",
    "never a loudspeaker": "nunca un altavoz",
    "Class D only:": "Solo clase D:",
    "analogue low-pass": "paso bajo analógico",
    "Analyser / ADC": "Analizador / ADC",
    "≥ 10 dB headroom": "≥ 10 dB de margen",
    "Standard measuring conditions (3.1.3)":
        "Condiciones normalizadas (3.1.3)",
    "rated conditions (3.1.2) with the source e.m.f. −10 dB":
        "condiciones nominales (3.1.2), f.e.m. de la fuente −10 dB",
    "Clause 9: hold the amplifier at that operating point for 1 h before "
    "the first reading.":
        "Apartado 9: mantener el amplificador en ese punto de "
        "funcionamiento 1 h antes de la primera lectura.",
    "Three checks, from the method itself":
        "Tres comprobaciones, del propio método",
    "source THD ≥ 10 dB below the lowest distortion to be measured "
    "(14.12.3.2 a)":
        "THD de la fuente ≥ 10 dB por debajo de la distorsión más baja "
        "a medir (14.12.3.2 a)",
    "generator muted: residual < 1/3 of the distortion voltage, or the "
    "result is discarded (14.12.3.2 d)":
        "generador silenciado: residuo < 1/3 de la tensión de distorsión, "
        "o el resultado se descarta (14.12.3.2 d)",
    "highest significant harmonic inside the band: $f_1 ≤ f_{limit} / n$ "
    "(14.12.4.1 — 30 kHz and $n = 5$ give 6 kHz)":
        "armónico significativo más alto dentro de la banda: "
        "$f_1 ≤ f_{limit} / n$ (14.12.4.1 — 30 kHz y $n = 5$ dan 6 kHz)",
    "AES17 band 20 Hz – 20 kHz": "banda AES17 de 20 Hz – 20 kHz",
    # The two sweep benches and the time budget
    "Playing and recording a sweep: the two benches and the time budget":
        "Reproducir y grabar un barrido: dos bancos y el reparto del tiempo",
    "(a) Electrical device under test":
        "(a) Dispositivo eléctrico en ensayo",
    "(b) Loudspeaker in a room": "(b) Altavoz en una sala",
    # "Audio interface" over two drawn lines; the Spanish order differs,
    # so the two line fragments swap content.
    "Audio": "Interfaz",
    "interface": "de audio",
    "Device": "Dispositivo",
    "under test": "en ensayo",
    "out": "salida",
    "in — channel 1": "entrada — canal 1",
    "loopback — channel 2 fixes $t = 0$": "bucle — el canal 2 fija $t = 0$",
    "source and microphone at 1.20 m over a hard floor":
        "fuente y micrófono a 1,20 m sobre suelo duro",
    "reflected path 2.60 m": "camino reflejado de 2,60 m",
    "$d$ = 1.00 m": "$d$ = 1,00 m",
    "Reflection-free time $t_g$ = (2.60 − 1.00) m / (343 m/s) = 4.7 ms: past "
    "that the record is the room, not the box.":
        "Tiempo libre de reflexiones $t_g$ = (2,60 − 1,00) m / (343 m/s) = "
        "4,7 ms: después se graba la sala, no el altavoz.",
    "Recording on the sweep's time axis: $f_1$ = 20 Hz, $f_2$ = 6 kHz, $T$ = "
    "4 s, $L = T / ln(f_2/f_1)$ = 0.70 s":
        "La grabación, en el eje del barrido: $f_1$ = 20 Hz, $f_2$ = 6 kHz, "
        "$T$ = 4 s, $L = T / ln(f_2/f_1)$ = 0,70 s",
    "$h_1$ (linear)": "$h_1$ (lineal)",
    "$h_2$  −0.49 s": "$h_2$  −0,49 s",
    "$h_3$  −0.77 s": "$h_3$  −0,77 s",
    "per-order window: 8192 samples = 0.17 s at 48 kHz":
        "ventana por orden: 8192 muestras = 0,17 s a 48 kHz",
    "$L ln(3/2)$ = 0.28 s: the closest pair of arrivals":
        "$L ln(3/2)$ = 0,28 s: el par de llegadas más próximo",
    "Record until the decay has died, or the pre-arrivals wrap round the "
    "circular deconvolution into $h_1$.":
        "Se graba hasta que muere la caída, o las prellegadas rodean la "
        "deconvolución circular al final de $h_1$.",
    "State the drive amplitude and the fade with the result, and pass "
    "the same fade to the analysis.":
        "La amplitud de excitación y el fundido se declaran con el "
        "resultado, y el mismo fundido se pasa al análisis.",
    # Occupational noise exposure (ISO 9612)
    "Occupational noise exposure measurement (ISO 9612)":
        "Medición de la exposición al ruido en el trabajo (ISO 9612)",
    "Worn instrument (Clause 12.3)": "Instrumento portado (apartado 12.3)",
    "≈ 0.04 m": "≈ 0,04 m",
    "above the shoulder": "sobre el hombro",
    "≥ 0.1 m from the ear canal,": "≥ 0,1 m del canal auditivo,",
    "most-exposed side": "lado del oído más expuesto",
    "Personal sound exposure meter": "Exposímetro sonoro personal",
    "(IEC 61252)": "(IEC 61252)",
    "Measurement strategies (Clauses 9–11)":
        "Estrategias de medición (apartados 9–11)",
    "Working day": "Jornada laboral",
    "Task-based (Clause 9)": "Basada en tareas (apartado 9)",
    "split the day into tasks — ≥ 3 samples (│) per task, plus each duration":
        "dividir la jornada en tareas — ≥ 3 muestras (│) y la duración por tarea",
    "Job-based (Clause 10)": "Basada en la función (apartado 10)",
    "$N ≥ 5$ random samples over the homogeneous exposure group":
        "$N ≥ 5$ muestras aleatorias sobre el grupo de exposición homogéneo",
    "Full-day (Clause 11)": "Jornada completa (apartado 11)",
    "the whole shift, at least 3 times (5 if the days differ by > 3 dB)":
        "toda la jornada, al menos 3 veces (5 si los días difieren en > 3 dB)",
    "Task 1": "Tarea 1",
    "Task 2": "Tarea 2",
    "Task 3": "Tarea 3",
    "day 1": "día 1",
    "choose by work pattern (Table B.1)  →  LEX,8h + Annex C uncertainty":
        "según el patrón de trabajo (Tabla B.1)  →  LEX,8h + U del Anexo C",
    # Dynamic-stiffness resonance rig (EN 29052-1)
    "Dynamic-stiffness resonance rig (EN 29052-1)":
        "Banco de resonancia de rigidez dinámica (EN 29052-1)",
    "The three excitation arrangements (Figures 1 to 3)":
        "Las tres disposiciones de excitación (Figuras 1 a 3)",
    "Rigid base": "Base rígida",
    "load plate measured": "se mide la placa de carga",
    "Isolated baseplate": "Placa base aislada",
    "load plate driven, both measured":
        "se excita la placa de carga; se miden ambas",
    "baseplate driven, both measured":
        "se excita la placa base; se miden ambas",
    "Rigid foundation": "Cimentación rígida",
    "Baseplate ≥ 100 kg": "Placa base ≥ 100 kg",
    "all three are equivalent; sinusoidal excitation is the reference "
    "method in case of dispute (7.1)":
        "las tres son equivalentes; la excitación sinusoidal es el método "
        "de referencia en caso de litigio (7.1)",
    "Specimen and load (Clauses 5 and 6)":
        "Probeta y carga (capítulos 5 y 6)",
    "plaster of Paris ≥ 5 mm on 0.02 mm foil":
        "escayola ≥ 5 mm sobre lámina de 0,02 mm",
    "Load plate, steel": "Placa de carga, acero",
    "(200 ± 3) mm square, flat to 0.5 mm":
        "(200 ± 3) mm de lado, planitud 0,5 mm",
    "8 kg ± 0.5 kg with every device on it":
        "8 kg ± 0,5 kg con todos los equipos encima",
    "Resilient specimen, 200 mm × 200 mm":
        "Probeta resiliente, 200 mm × 200 mm",
    "three of them; irregularities < 3 mm":
        "tres probetas; irregularidades < 3 mm",
    "petroleum-jelly fillet (closed-cell materials)":
        "cordón de vaselina (materiales de celda cerrada)",
    "Mass-spring model": "Modelo masa-resorte",
    "read at the peak, extrapolated to zero force":
        "se lee en el pico, extrapolado a fuerza nula",
    "$s′_t = 4π^2 m′_t f_r^2$   (Formula 4)":
        "$s′_t = 4π^2 m′_t f_r^2$   (Fórmula 4)",
    "$f_0 = (1/2π)·√(s′/m′)$   (Formula 2)":
        "$f_0 = (1/2π)·√(s′/m′)$   (Fórmula 2)",
    # Mechanical-mobility rig (ISO 7626)
    "Mechanical-mobility measurement on a beam (ISO 7626)":
        "Medición de movilidad mecánica sobre una viga (ISO 7626)",
    "soft elastic suspension": "suspensión elástica blanda",
    "Structure under test (free-free beam)":
        "Estructura bajo ensayo (viga libre-libre)",
    "Impedance head": "Cabeza de impedancia",
    "$F$ and $a$ at the drive point": "$F$ y $a$ en el mismo punto",
    "driving point:  $Y_{ii} = v_i / F_i$":
        "punto de excitación:  $Y_{ii} = v_i / F_i$",
    "transfer:  $Y_{ji} = v_j / F_i$": "transferencia:  $Y_{ji} = v_j / F_i$",
    "$Y(f) = v/F$  [m/(N·s)] · attached exciter (Part 2) · impact hammer "
    "(Part 5)":
        "$Y(f) = v/F$  [m/(N·s)] · excitador acoplado (Parte 2) · martillo "
        "de impacto (Parte 5)",
    "same measurement, three FRFs: $x/F$ receptance · $v/F$ mobility · "
    "$a/F$ accelerance":
        "una misma medición, tres FRF: $x/F$ receptancia · $v/F$ movilidad "
        "· $a/F$ acelerancia",
    "Exciter": "Excitador",
    # Dynamic transfer stiffness (ISO 10846)
    "Dynamic transfer stiffness: direct and indirect methods (ISO 10846)":
        "Rigidez de transferencia: métodos directo e indirecto (ISO 10846)",
    "Direct method (Part 2)": "Método directo (Parte 2)",
    "Indirect method (Part 3)": "Método indirecto (Parte 3)",
    "excitation mass": "masa de excitación",
    "isolator under test": "aislador bajo ensayo",
    "force transducer": "transductor de fuerza",
    "blocking mass $m_2$": "masa de bloqueo $m_2$",
    "soft support": "apoyo blando",
    "output blocked:  $u_2 ≈ 0$ → measure $F_{2,b}$":
        "salida bloqueada:  $u_2 ≈ 0$ → se mide $F_{2,b}$",
    "measure $T = u_2 / u_1$  (small)": "se mide $T = u_2 / u_1$  (pequeña)",
    "valid where $ΔL_{1,2} = L_{a1} − L_{a2} ≥ 20$ dB, i.e. $|T| ≤ 0.1$   "
    "(Part 3, Inequality 2)":
        "válido donde $ΔL_{1,2} = L_{a1} − L_{a2} ≥ 20$ dB, es decir "
        "$|T| ≤ 0,1$   (Parte 3, Desigualdad 2)",
    "the blocking force approximates the force delivered to a stiff receiver (Part 1, Eq. 7)":
        "la fuerza de bloqueo aproxima la fuerza entregada a un receptor "
        "rígido (Parte 1, Ec. 7)",
    "$a′_1$: unwanted transverse input,":
        "$a′_1$: entrada transversal no deseada,",
    "≥ 15 dB below $a_1$ (Inequality 3)":
        "≥ 15 dB por debajo de $a_1$ (Desigualdad 3)",
    # The transverse-input symbol: subscripts only, identical in Spanish.
    "$a_{1x}$": "$a_{1x}$",
    "The two ways the static preload is applied (ISO 10846-1, 6.3.3.1)":
        "Las dos formas de aplicar la precarga estática (ISO 10846-1, 6.3.3.1)",
    "a) gravity: the output-side mass is the preload":
        "a) gravedad: la masa de salida es la precarga",
    "load mass": "masa de carga",
    "$W$ = load": "$W$ = carga",
    "simple, but unstable for large isolators at high loads":
        "sencillo, pero inestable con precargas altas",
    "b) frame and actuator, with decoupling springs":
        "b) bastidor y actuador, con muelles de desacoplo",
    "actuator: 100 % of the": "actuador: 100 % de la",
    "permissible static load": "carga estática admisible",
    "auxiliary springs decouple $m_2$ from the frame":
        "los muelles auxiliares desacoplan $m_2$ del bastidor",
    "Transverse translations are standardised too (ISO 10846-2, 5.2)":
        "Las traslaciones transversales también están normalizadas "
        "(ISO 10846-2, 5.2)",
    "force-distribution plate": "placa de reparto de fuerza",
    "test element in shear": "elemento a cortante",
    "roller bearings, or two symmetrical": "los rodamientos, o dos elementos",
    "elements, suppress the unwanted input":
        "simétricos, suprimen la entrada no deseada",
    "output shear force summed from two": "fuerza cortante de salida sumada",
    "transducers,  $F_2 = F_2′ + F_2″$":
        "de dos transductores,  $F_2 = F_2′ + F_2″$",
    "a mount is loaded in shear as well as in compression, and the transverse "
    "stiffness is usually the smaller of the two":
        "un apoyo trabaja a cortante además de a compresión, y la rigidez "
        "transversal suele ser la menor de las dos",
    # Mobility rig: the drive rod and the ISO 7626-2 Figure 4 decision
    "drive rod: stiff axially,": "varilla de arrastre: rígida axialmente,",
    "flexible in every other": "y flexible en todas las demás",
    "direction (6.4.4)": "direcciones (6.4.4)",
    "Where the accelerometer and the force transducer go (clause 6.4.4)":
        "Dónde van el acelerómetro y el transductor de fuerza (capítulo 6.4.4)",
    "accelerometer through the rod": "acelerómetro a través de la varilla",
    "force transducer at the structure": "transductor en la estructura",
    "force transducer at the exciter": "transductor en el excitador",
    "INVALID": "NO VÁLIDO",
    "VALID": "VÁLIDO",
    "WITH CAUTION": "CON PRECAUCIÓN",
    "the exciter attachment must be at least 10× more mobile, laterally and "
    "rotationally, than the structure (6.4.4)":
        "el acoplo del excitador debe ser al menos 10 veces más móvil, "
        "lateral y rotacionalmente, que la estructura (6.4.4)",
    "and the suspension at least 10× more mobile than the structure at each "
    "attachment point (5.3)":
        "y la suspensión al menos 10 veces más móvil que la estructura en "
        "cada punto de sujeción (5.3)",
    # Power injection (Norton & Karczub 6.6.4)
    "Power injection: coupling loss factors from measured energies":
        "Inyección de potencia: de energías medidas a factores de pérdida",
    "Run 1: drive subsystem 1": "Ensayo 1: se excita el subsistema 1",
    "Run 2: drive subsystem 2": "Ensayo 2: se excita el subsistema 2",
    "subsystem 1": "subsistema 1",
    "subsystem 2": "subsistema 2",
    "impedance head": "cabeza de impedancia",
    "shaker": "excitador",
    "$Π_1$ measured,  $Π_2 = 0$": "$Π_1$ medida,  $Π_2 = 0$",
    "$Π_2$ measured,  $Π_1 = 0$": "$Π_2$ medida,  $Π_1 = 0$",
    # Energy bookkeeping of the two runs: symbols only, identical in Spanish.
    "$E_1 = M_1⟨v_1^2⟩,   E_2 = M_2⟨v_2^2⟩$":
        "$E_1 = M_1⟨v_1^2⟩,   E_2 = M_2⟨v_2^2⟩$",
    "$Π_{in} = ½ Re{F v*}$ at the drive point, from an impedance head — "
    "not the amplifier setting":
        "$Π_{in} = ½ Re{F v*}$ en el punto de excitación, con cabeza de "
        "impedancia, no con el ajuste del amplificador",
    "$⟨v^2⟩$ space-averaged over several positions per subsystem, away "
    "from the edges and from the drive point":
        "$⟨v^2⟩$ promediada en varias posiciones por subsistema, lejos de los "
        "bordes y del punto de excitación",
    "one run gives $η_{12}$ only if $η_1$ and $η_2$ come from a decay "
    "measurement; two runs solve all four":
        "un ensayo despeja $η_{12}$ solo si $η_1$ y $η_2$ se miden por "
        "decaimiento; dos resuelven los cuatro",
    "bands wide enough to hold several modes of each subsystem: the modal "
    "densities decide how wide":
        "bandas anchas para contener varios modos de cada subsistema: las "
        "densidades modales deciden cuánto",
    # Machine fault kinematics (Norton & Karczub Section 8.4)
    "Where the fault frequencies come from: bearing, gear pair, ducted fan":
        "Origen de las frecuencias de fallo: rodamiento, engranaje, "
        "ventilador",
    "1. Bearing: end section": "1. Rodamiento: sección frontal",
    "2. Bearing: axial half-section": "2. Rodamiento: media sección axial",
    "3. Gear pair": "3. Par de engranajes",
    "4. Ducted axial fan": "4. Ventilador axial en conducto",
    "spall on the outer race": "descascarillado en la pista exterior",
    "BPFO = 207.0 Hz: one impact per pass":
        "BPFO = 207,0 Hz: un impacto por paso",
    "$D$ = 34 mm (pitch)": "$D$ = 34 mm (primitivo)",
    "$d$ = 6 mm": "$d$ = 6 mm",
    "inner race turns at $f_s$ = 33.33 Hz,":
        "la pista interior gira a $f_s$ = 33,33 Hz,",
    "outer race stationary": "la exterior está fija",
    "cage FTF = 13.8 Hz = 0.41 $f_s$": "jaula FTF = 13,8 Hz = 0,41 $f_s$",
    "bearing axis": "eje del rodamiento",
    "outer ring": "aro exterior",
    "inner ring": "aro interior",
    "radial plane": "plano radial",
    "contact line,  $φ$ = 12.96°": "línea de contacto,  $φ$ = 12,96°",
    "the contact angle exists only in this view: measured":
        "el ángulo de contacto solo existe en esta vista: se mide",
    "from the radial plane: $φ = 0$ for a deep-groove bearing":
        "desde el plano radial: $φ = 0$ en un rodamiento rígido",
    "and $φ > 0$ for angular-contact and tapered-roller types":
        "y $φ > 0$ en los de contacto angular y de rodillos cónicos",
    "chipped tooth": "diente desconchado",
    "28-tooth pinion on a": "piñón de 28 dientes sobre un",
    "1500 r/min shaft:": "eje a 1500 r/min:",
    "$f_s$ = 25 Hz": "$f_s$ = 25 Hz",
    "6 blades (solid), 4 vanes (dashed)":
        "6 álabes (continuos), 4 directrices (a trazos)",
    "$GMF = N f_s = 28 × 25$ = 700 Hz, and a chipped tooth modulates it "
    "once per revolution: sidebands at $± f_s$":
        "$GMF = N f_s = 28 × 25$ = 700 Hz, y un diente desconchado la modula "
        "una vez por vuelta: bandas laterales a $± f_s$",
    "$m_L = n·N ± k·V = 6 ± 4$ → 2 or 10 lobes, turning at $n·N·f_s/m_L$ = "
    "175 or 35 Hz: the faster radiates much more":
        "$m_L = n·N ± k·V = 6 ± 4$ → 2 o 10 lóbulos, que giran a "
        "$n·N·f_s/m_L$ = 175 o 35 Hz: el más rápido radia mucho más",
    # Condition monitoring on a machine train (Norton & Karczub Section 8.4)
    "Condition monitoring on a motor-gearbox train (Norton Section 8.4)":
        "Monitorización de estado en un tren motor-reductora (Norton 8.4)",
    "motor": "motor",
    "gearbox": "reductora",
    "coupling": "acoplamiento",
    "axial": "axial",
    "radial, in the load zone, on the housing itself: no joint between the "
    "bearing and the sensor":
        "radial, en la zona de carga y sobre el propio soporte: sin junta "
        "entre el rodamiento y el sensor",
    "once-per-revolution": "captador de una",
    "pickup on a mark": "marca por vuelta",
    "Analyser": "Analizador",
    "ch 1: acceleration": "canal 1: aceleración",
    "ch 2: tacho pulse": "canal 2: pulso de tacómetro",
    "band-pass → envelope → spectrum": "paso de banda → envolvente → espectro",
    "Mounting sets the usable upper frequency":
        "El montaje fija la frecuencia máxima utilizable",
    "stud into a prepared flat": "espárrago sobre plano",
    "tens of kHz": "decenas de kHz",
    "adhesive / thin cyanoacrylate": "adhesivo / cianoacrilato fino",
    "~10 kHz": "~10 kHz",
    "magnet base": "base magnética",
    "a few kHz": "unos pocos kHz",
    "hand-held probe": "punta manual",
    "~1 kHz": "~1 kHz",
    "this page band-passes 2-4 kHz, so a magnet base is":
        "esta página filtra de 2 a 4 kHz: una base magnética",
    "marginal there and a hand-held probe is useless":
        "queda justa y una punta manual no sirve",
    "Acquisition": "Adquisición",
    "$f_s$ = 20 kHz clears the 3 kHz housing resonance":
        "$f_s$ = 20 kHz supera la resonancia de 3 kHz del soporte",
    "$T$ = 2 s at 2000 r/min = 67 revolutions":
        "$T$ = 2 s a 2000 r/min = 67 vueltas",
    "$Δf = 1/T$ = 0.5 Hz, against $f_s$ = 33.3 Hz":
        "$Δf = 1/T$ = 0,5 Hz, frente a $f_s$ = 33,3 Hz",
    "enough to resolve the $± f_s$ sidebands, which are":
        "suficiente para resolver las bandas laterales a $± f_s$, que son",
    "what separates an inner-race defect from an":
        "lo que distingue un defecto de pista interior de uno",
    "outer-race one": "de pista exterior",
    # Reception plate (EN 15657)
    "Reception-plate measurement of structure-borne power (EN 15657)":
        "Medición en placa receptora de la potencia estructural (EN 15657)",
    "Source under test (pump, fan, boiler …)":
        "Fuente bajo ensayo (bomba, ventilador, caldera …)",
    "Reception plate  ($m$, $S$, $η$)": "Placa receptora  ($m$, $S$, $η$)",
    "velocity positions → $L_v$": "posiciones de velocidad → $L_v$",
    "injected structure-borne power": "potencia estructural inyectada",
    "resilient supports": "apoyos resilientes",
    "Plate power balance": "Balance de potencia de la placa",
    "$η = 2.2 / (f·T_s)$   (Formula 13)":
        "$η = 2,2 / (f·T_s)$   (Fórmula 13)",
    # Formula 14, symbols only: reads the same in Spanish.
    "$L_{Ws} = 10 log_{10}(2πf·η·m·S / f_0 m_0 S_0)$":
        "$L_{Ws} = 10 log_{10}(2πf·η·m·S / f_0 m_0 S_0)$",
    "$+ L_v − 60$   (Formula 14)": "$+ L_v − 60$   (Fórmula 14)",
    "→ source quantities (Formulae 15–19):":
        "→ magnitudes de fuente (Fórmulas 15–19):",
    "equivalent blocked force $L_{Fb,eq}$ ,":
        "fuerza bloqueada equivalente $L_{Fb,eq}$ ,",
    "$L_{Wsn}$ consumed by EN 12354-5": "$L_{Wsn}$ que consume EN 12354-5",
    "spatial average:  Lv = 10 log10[(1/N)·Σ 10^(Lv,i/10)]   (Formula 12)":
        "promedio espacial:  Lv = 10 log10[(1/N)·Σ 10^(Lv,i/10)]   (Fórmula 12)",
    # Installed structure-borne sound (EN 12354-5)
    "Installed structure-borne sound paths (EN 12354-5)":
        "Vías del sonido estructural de equipos instalados (EN 12354-5)",
    "Service equipment (pump)": "Equipo de servicio (bomba)",
    "coupling $D_C$   (Formula 19b)": "acoplamiento $D_C$   (Fórmula 19b)",
    "path along the slab into the wall  ($i$ → $j$)":
        "vía por el forjado hacia la pared  ($i$ → $j$)",
    "excited floor radiates (path $i = j$)":
        "el forjado excitado radia (vía $i = j$)",
    "Prediction cascade": "Cascada de predicción",
    "characteristic power (EN 15657)": "potencia característica (EN 15657)",
    "coupling at the contacts (19b)": "acoplamiento en los contactos (19b)",
    "installed power (18b)": "potencia instalada (18b)",
    "per transmission path (18a)": "por vía de transmisión (18a)",
    "energetic sum $L_{n,s}$ (17)": "suma energética $L_{n,s}$ (17)",
    # Formula 17, symbols only: reads the same in Spanish.
    "10 log10 Σ 10^(L_n,s,ij/10)": "10 log10 Σ 10^(L_n,s,ij/10)",
    "each path $i$ → $j$: excited element $i$, radiating element $j$ "
    "in the receiving room":
        "cada vía $i$ → $j$: elemento excitado $i$, elemento radiante $j$ "
        "en el recinto receptor",
    # Wind-turbine noise measurement geometry (IEC 61400-11)
    "Wind-turbine noise measurement geometry (IEC 61400-11)":
        "Geometría de medida del ruido de aerogenerador (IEC 61400-11)",
    "Wind": "Viento",
    "rotor centre": "centro del rotor",
    "Microphone on a ground board": "Micrófono sobre placa en el suelo",
    "Met mast": "Mástil meteorológico",
    "wind speed + direction": "velocidad y dirección del viento",
    "Plan view (Figure 3)": "Planta (Figura 3)",
    "reference position 1 (downwind)": "referencia 1 (a sotavento)",
    "optional positions 2–4": "posiciones opcionales 2–4",
    "$R_1 = √(H^2 + R_0^2)$   slant distance, rotor centre → microphone":
        "$R_1 = √(H^2 + R_0^2)$   distancia oblicua, centro del rotor → micrófono",
    "$L_{WA,i} = L_{p,i} − 6 + 10 log_{10}(4π R_1^2 / S_0)$   (Formula 26, $S_0$ = 1 m²)":
        "$L_{WA,i} = L_{p,i} − 6 + 10 log_{10}(4π R_1^2 / S_0)$   (Fórmula 26, $S_0$ = 1 m²)",
    "the −6 dB removes the board's pressure doubling; board-to-$R_1$ angle $φ$ = 25°–40°":
        "los −6 dB descuentan la duplicación de presión de la placa; "
        "ángulo placa–$R_1$ $φ$ = 25°–40°",
    # Ground reflection (image source)
    "Ground reflection: direct ray, image source and path difference":
        "Reflexión del suelo: rayo directo, fuente imagen y diferencia de camino",
    "image source": "fuente imagen",
    "direct ray  $r_1$": "rayo directo  $r_1$",
    "reflected ray": "rayo reflejado",
    "equal angles": "ángulos iguales",
    "path difference  $δ = r_2 − r_1$":
        "diferencia de camino  $δ = r_2 − r_1$",
    "phase difference  $Δφ = 2π δ / λ$  (+ $arg Q$)":
        "diferencia de fase  $Δφ = 2π δ / λ$  (+ $arg Q$)",
    "p ∝ e^(jkr1)/r1 + Q · e^(jkr2)/r2   (Q = ground reflection coefficient)":
        "p ∝ e^(jkr1)/r1 + Q · e^(jkr2)/r2   (Q = coeficiente de reflexión del suelo)",
    "in phase ($δ ≈ nλ$): up to +6 dB    ·    out of phase ($δ ≈ λ/2$ on hard ground): a deep dip":
        "en fase ($δ ≈ nλ$): hasta +6 dB    ·    en oposición ($δ ≈ λ/2$ sobre "
        "suelo duro): un mínimo profundo",
    # 2D FDTD wave simulation
    "Domain  $c(x, y)$, $ρ(x, y)$, $dx$":
        "Dominio  $c(x, y)$, $ρ(x, y)$, $dx$",
    "square cells; dt from the Courant number":
        "celdas cuadradas; dt desde el número de Courant",
    "Geometry and boundaries": "Geometría y contornos",
    "rigid, impedance or absorbing edges; obstacles":
        "bordes rígidos, de impedancia o absorbentes; obstáculos",
    "Sources  $s(t)$ injected at cells  (Eq. 4.11-4.12 grid)":
        "Fuentes  $s(t)$ inyectadas en celdas  (malla de Ec. 4.11-4.12)",
    "Gaussian pulse, ramped tone or arbitrary sampled signal":
        "pulso gaussiano, tono con rampa o señal muestreada arbitraria",
    "Staggered-grid leapfrog update  (Eqs. 4.11-4.12)":
        "Actualización leapfrog en malla escalonada  (Ecs. 4.11-4.12)",
    "$v ← v − (dt/ρ·dx)·grad p$,  then  $p ← p − (ρc^2·dt/dx)·div v$":
        "$v ← v − (dt/ρ·dx)·grad p$,  y luego  $p ← p − (ρc^2·dt/dx)·div v$",
    "stable while  $CN = c·dt·√2/dx ≤ 1$  (Eqs. 4.13-4.14)":
        "estable mientras  $CN = c·dt·√2/dx ≤ 1$  (Ecs. 4.13-4.14)",
    "resolve ≥ 10 cells per wavelength to keep dispersion low":
        "resolver ≥ 10 celdas por longitud de onda para baja dispersión",
    "2D acoustic FDTD wave simulation (staggered leapfrog)":
        "Simulación de ondas FDTD acústica 2D (leapfrog escalonado)",
    "FDTDResult:  probe histories $p(t)$, field snapshots, .plot()":
        "FDTDResult:  historias de sonda $p(t)$, instantáneas del campo, "
        ".plot()",
    "deterministic: same inputs, bit-identical outputs":
        "determinista: mismas entradas, salidas idénticas bit a bit",
    # Sound level meter chain (IEC 61672-1)
    "Sound level meter measurement chain (IEC 61672-1)":
        "Cadena de medición del sonómetro (IEC 61672-1)",
    "Sound calibrator (class 1)": "Calibrador acústico (clase 1)",
    "coupled for": "acoplado para",
    "the level check": "verificar el nivel",
    "Windscreen": "Pantalla antiviento",
    "Microphone + preamplifier": "Micrófono + preamplificador",
    "free-field capsule, high-impedance stage":
        "cápsula de campo libre, etapa de alta impedancia",
    "Frequency weighting  A / C / Z": "Ponderación frecuencial  A / C / Z",
    "all three are 0 dB at 1 kHz; class 1: ±0.7 dB":
        "las tres valen 0 dB a 1 kHz; clase 1: ±0,7 dB",
    "Squaring + time weighting  F / S":
        "Cuadrado + ponderación temporal  F / S",
    "exponential detector: $τ_F$ = 125 ms, $τ_S$ = 1 s":
        "detector exponencial: $τ_F$ = 125 ms, $τ_S$ = 1 s",
    "Display": "Pantalla",
    "$L_{AF}(t)$, $L_{AS}(t)$ in dB re 20 µPa":
        "$L_{AF}(t)$, $L_{AS}(t)$ en dB re 20 µPa",
    # Laboratory sound insulation suite (ISO 10140)
    "Laboratory sound insulation suite (ISO 10140)":
        "Cámaras de aislamiento acústico de laboratorio (ISO 10140)",
    "structural break": "junta estructural",
    "Test element in the test opening":
        "Elemento de ensayo en la abertura de ensayo",
    "moving microphone": "micrófono móvil",
    "sweep radius ≥ 1 m": "radio de barrido ≥ 1 m",
    "Test opening ≈ 10 m² (3.75 m × 2.7 m); shorter edge ≥ 2.3 m":
        "Abertura de ensayo ≈ 10 m² (3,75 m × 2,7 m); lado menor ≥ 2,3 m",
    "Room volumes ≥ 50 m³, differing by at least 10 %":
        "Volúmenes de sala ≥ 50 m³, con diferencia de al menos el 10 %",
    "Continuously moving microphone: sweep radius ≥ 1 m, traverse ≥ 15 s":
        "Micrófono en movimiento continuo: radio de barrido ≥ 1 m, "
        "recorrido ≥ 15 s",
    "5.0 m": "5,0 m",
    "4.6 m": "4,6 m",
    "4.1 m": "4,1 m",
    "3.75 m": "3,75 m",
    # Junction vibration measurement (ISO 10848)
    "Junction vibration measurement on L- and T-junctions (ISO 10848)":
        "Medición de vibración en uniones en L y en T (ISO 10848)",
    "L-junction": "Unión en L",
    "T-junction": "Unión en T",
    "Shaker or hammer on element $i$":
        "Excitador o martillo sobre el elemento $i$",
    "accelerometers on i and j": "acelerómetros en i y j",
    "$l_{ij} ≥ 2.3$ m": "$l_{ij} ≥ 2,3$ m",
    "concrete plates 140 mm to 200 mm thick":
        "placas de hormigón de 140 mm a 200 mm de espesor",
    "$l_{ij} ≥ 2.3$ m along the junction; element sizes 3.0 m $≤ l_i <$ "
    "6.0 m":
        "$l_{ij} ≥ 2,3$ m a lo largo de la unión; dimensiones de elemento "
        "3,0 m $≤ l_i <$ 6,0 m",
    "≥ 4 excitation positions on $i$; accelerometers ≥ 0.25 m from edges, "
    "≥ 0.5 m apart":
        "≥ 4 posiciones de excitación en $i$; acelerómetros a ≥ 0,25 m de "
        "los bordes y ≥ 0,5 m entre sí",
    "$K_{ij} = D̄_{v,ij} + 10 log_{10}( l_{ij} / √(a_i·a_j) )$,   $a_i$ = "
    "equivalent absorption length":
        "$K_{ij} = D̄_{v,ij} + 10 log_{10}( l_{ij} / √(a_i·a_j) )$,   "
        "$a_i$ = long. de absorción equiv.",
    # Sound power from surface vibration (ISO/TS 7849)
    "Sound power from surface vibration (ISO/TS 7849)":
        "Potencia acústica a partir de la vibración superficial (ISO/TS 7849)",
    "Vibrating measurement surface $S$": "Superficie de medición vibrante $S$",
    "Machine under test": "Máquina en ensayo",
    "radiated airborne sound": "sonido aéreo radiado",
    "Initial number of positions $N$": "Número inicial de posiciones $N$",
    "one accelerometer per cell of area $S/N$":
        "un acelerómetro por celda de área $S/N$",
    "Survey sound power": "Potencia acústica de control",
    "$ε = 1$ assumed → upper limit $L_{WA,max}$":
        "se asume $ε = 1$ → límite superior $L_{WA,max}$",
    "normal surface velocity, A-weighted r.m.s.":
        "velocidad normal eficaz, ponderada A",
    # The headline relation reads the same in Spanish (shared notation).
    "$L_{WA} = L_{vA} + 10 log_{10}(S/S_0) + 10 log_{10} ε$":
        "$L_{WA} = L_{vA} + 10 log_{10}(S/S_0) + 10 log_{10} ε$",
    "2.5 m": "2,5 m",
    "1.6 m": "1,6 m",
    # Ship radiated-noise measurement geometry (ISO 17208-1)
    "Ship radiated-noise measurement geometry (ISO 17208-1)":
        "Geometría de medición del ruido radiado por buques (ISO 17208-1)",
    "Ship under test": "Buque en ensayo",
    "Surface buoy": "Boya de superficie",
    "vertical array of 3 hydrophones": "array vertical de 3 hidrófonos",
    "ballast": "lastre",
    "sea floor": "fondo marino",
    "Plan view": "Vista en planta",
    "course": "rumbo",
    "data window": "ventana de datos",
    "dCPA ≥ 100 m (or 1·L)": "dCPA ≥ 100 m (o 1·L)",
    # The standard's own symbols, the same in both languages (tan is the
    # Spanish abbreviation of tangente too).
    "dCPA": "dCPA",
    "lDW = 2 dCPA tan 30°": "lDW = 2 dCPA tan 30°",
    "water depth ≥ 150 m (or 1.5·L)": "profundidad ≥ 150 m (o 1,5·L)",
    "Four runs, two per side; levels averaged while the ship crosses the data window":
        "Cuatro pasadas, dos por banda; niveles promediados mientras el buque "
        "cruza la ventana de datos",
    "Hydrophone depths from the 15°, 30° and 45° depression angles at r = dCPA; L = ship length":
        "Profundidades de hidrófono según los ángulos de depresión de 15°, "
        "30° y 45° a r = dCPA; L = eslora",
    "Run schedule (not to scale)": "Secuencia de pasada (sin escala)",
    "reverse course; 4 runs, 2 per side":
        "vuelta en redondo; 4 pasadas, 2 por banda",
    "background: ship stopped, ≥ 2 km, ≥ 30 s,":
        "fondo: buque parado, ≥ 2 km, ≥ 30 s,",
    "at the start and end of each test period":
        "al inicio y al final de cada periodo",
    # SOFAR channel (deep sound channel)
    "The SOFAR channel: a deep-ocean sound waveguide":
        "El canal SOFAR: una guía de ondas del océano profundo",
    "Sound-speed profile $c(z)$": "Perfil de velocidad del sonido $c(z)$",
    "Ray paths near the axis": "Trayectorias de rayos cerca del eje",
    "sea surface": "superficie del mar",
    "source on the channel axis": "fuente en el eje del canal",
    "rays that stay in the channel meet no surface or bottom loss":
        "los rayos que permanecen en el canal no sufren pérdidas en "
        "superficie ni en fondo",
    "$c$ rises toward the surface (temperature) and toward the bottom (pressure); the minimum traps sound":
        "$c$ aumenta hacia la superficie (temperatura) y hacia el fondo "
        "(presión); el mínimo atrapa el sonido",
    "rays launched within about ±12° of the axis stay trapped and can cross entire oceans":
        "los rayos lanzados a menos de unos ±12° del eje quedan atrapados "
        "y pueden cruzar océanos enteros",
    # Percussive pile-driving survey geometry (ISO 18406)
    "Percussive pile-driving survey geometry (ISO 18406)":
        "Geometría de medición del hincado percusivo de pilotes (ISO 18406)",
    "Section: the minimum campaign": "Sección: la campaña mínima",
    "Plan: where else to measure": "Planta: dónde medir además",
    "impact hammer": "martillo de impacto",
    "monopile": "monopilote",
    "penetration": "penetración",
    "bubble curtain, if used": "cortina de burbujas, si se usa",
    "as close as possible to 750 m": "lo más cerca posible de 750 m",
    "lower half of the water column:": "mitad inferior de la columna de agua:",
    "2 m above the bed to ½ depth": "de 2 m sobre el fondo a ½ profundidad",
    "½ depth": "½ prof.",
    "¾ depth": "¾ prof.",
    "bottom-mounted recorder": "registrador fondeado",
    "engines, generator and echo-sounder off;":
        "motores, generador y ecosonda apagados;",
    "flow noise, cable strum and surface heave":
        "ruido de flujo, cable y balanceo",
    "all read as signal": "se leen como señal",
    "seabed": "fondo marino",
    "pile": "pilote",
    "3 × water depth = 90 m:": "3 × profundidad = 90 m:",
    "nothing inside": "sin posiciones",
    "further positions on a transect,": "más posiciones en un transecto,",
    "clear of banks and trenches": "libre de bancos y fosas",
    "(radii not to scale)": "(radios sin escala)",
    "Percussive driving only, in 4 m to 100 m of water: vibro- and sheet-piling are out of scope":
        "Solo hincado percusivo, en 4 m a 100 m de agua: vibrohincado y "
        "tablestacas fuera del alcance",
    "750 m is a comparability convention, not a regulatory limit; the actual range is reported, to 5 %":
        "750 m es un convenio de comparabilidad, no un límite; se informa la "
        "distancia real, con un 5 %",
    "The station records the entire driving sequence, soft start included, at one fixed range":
        "La estación registra toda la secuencia de hincado, arranque suave "
        "incluido, a una distancia fija",
    "Record hydrophone depth, GPS, water depth and tide, seabed class and "
    "energy per blow":
        "Anota profundidad del hidrófono, GPS, profundidad y marea, clase de "
        "fondo y energía por golpe",
    # Sonar equation geometry (underwater/underwater-propagation)
    "Sonar equation geometry: passive and active (ISO 18405)":
        "Geometría de la ecuación del sonar: pasiva y activa (ISO 18405)",
    "Passive: the target radiates, one way":
        "Pasivo: el blanco radia, un solo trayecto",
    "Active, monostatic: out and back": "Activo monoestático: ida y vuelta",
    "ambient noise field NL": "campo de ruido ambiente NL",
    "source level": "nivel de fuente",
    "one-way propagation loss": "pérdida de propagación de ida",
    "array gain": "ganancia de array",
    "detector": "detector",
    "transmit and": "emite y",
    "receive here": "recibe aquí",
    "PL out": "PL ida",
    "PL back": "PL vuelta",
    "target strength": "índice de blanco",
    "RL: surface, volume and bottom scattering":
        "RL: dispersión superficie/volumen/fondo",
    "replaces $NL − DI$ when reverberation-limited":
        "sustituye a $NL − DI$ si domina la reverberación",
    "Field levels are re 1 µPa; a source level carries the squared metre of "
    "its range, re 1 µPa²m²":
        "Niveles de campo re 1 µPa; el nivel de fuente lleva el cuadrado de "
        "su distancia, re 1 µPa²m²",
    "Every term is in the same bandwidth; the figure of merit is the PL that "
    "drives SE to zero, re m²":
        "Todos los términos al mismo ancho de banda; la figura de mérito es "
        "la PL que anula SE, re m²",
    # The waveguide the three numerical solvers share
    "The range-independent waveguide the three solvers share":
        "La guía de ondas que comparten los tres solucionadores",
    "channel axis": "eje del canal",
    "sea surface: pressure release, $p = 0$":
        "superficie: presión nula, $p = 0$",
    "bottom: Ψ(D) = 0 (pressure release) or dΨ/dz = 0 (rigid)":
        "fondo: Ψ(D) = 0 (presión nula) o dΨ/dz = 0 (rígido)",
    "source, $z_s$": "fuente, $z_s$",
    "receiver, $z$": "receptor, $z$",
    "turning depth $z_t$:  $c(z_t) = c(z_s)/cos θ_0$":
        "profundidad de retorno $z_t$:  $c(z_t) = c(z_s)/cos θ_0$",
    "modes:": "modos:",
    "rays:": "rayos:",
    "standing waves in $z$, travelling as $exp(i k_{rm} r)$":
        "ondas estacionarias en $z$, que viajan como $exp(i k_{rm} r)$",
    "trajectories, travel times, convergence zones":
        "trayectorias, tiempos de viaje, zonas de convergencia",
    "the envelope marched in $r$, one step $Δr$ at a time":
        "la envolvente avanzada en $r$, un paso $Δr$ cada vez",
    "All three take the same range-independent $c(z)$: no sediment attenuation, no bathymetry":
        "Los tres toman el mismo $c(z)$ independiente de la distancia: sin "
        "sedimento ni batimetría",
    # Marine-mammal exposure assessment
    "Marine-mammal exposure: measured here, assessed there":
        "Exposición de mamíferos marinos: se mide aquí, se evalúa allí",
    "Section: pile, hydrophone, animal": "Sección: pilote, hidrófono, animal",
    "Plan: the two isopleths": "Planta: las dos isopletas",
    "calibrated hydrophone": "hidrófono calibrado",
    "band SEL measured here": "SEL por bandas medido aquí",
    "range $R$": "distancia $R$",
    "the criterion applies here": "el criterio se aplica aquí",
    "per-band SEL": "SEL por bandas",
    "weighted cumulative SEL  vs  AUD INJ / TTS":
        "SEL acumulado ponderado  frente a  AUD INJ / TTS",
    "unweighted peak SPL  vs  the flat criterion":
        "nivel de pico sin ponderar  frente al criterio plano",
    "peak SPL isopleth": "isopleta del nivel de pico",
    "weighted cumulative SEL isopleth":
        "isopleta del SEL acumulado ponderado",
    "the larger isopleth governs": "manda la isopleta mayor",
    "VHF, impulsive, NMFS 2024:": "VHF, impulsivo, NMFS 2024:",
    "144 / 159 dB re 1 µPa²·s weighted": "144 / 159 dB re 1 µPa²·s ponderado",
    "196 / 202 dB re 1 µPa flat": "196 / 202 dB re 1 µPa plano",
    "An isopleth is the contour on which a criterion is exactly met, so the answer is a radius, not a verdict":
        "Una isopleta es la curva donde el criterio se cumple justo: la "
        "respuesta es un radio, no un veredicto",
    "A cumulative level means nothing without its range and the window the strikes were counted over":
        "Un nivel acumulado no dice nada sin su distancia y la ventana en la "
        "que se contaron los golpes",
    # Atmospheric refraction (Salomons / Attenborough & Van Renterghem)
    "Atmospheric refraction: downwind multipath and the upwind shadow":
        "Refracción atmosférica: multitrayecto y sombra por el viento",
    "wind $u(z)$": "viento $u(z)$",
    "acoustic shadow": "sombra acústica",
    "Upwind: rays bend up; beyond ≈ 220 m a ground shadow opens and the level collapses by over 20 dB":
        "Contra el viento: los rayos suben; desde ≈ 220 m se abre una "
        "sombra y el nivel cae más de 20 dB",
    "Downwind: rays bend down; the receiver hears the direct and the ground-bounced arrival (multipath)":
        "A favor del viento: los rayos bajan y llegan la directa y la "
        "rebotada en el suelo (multitrayecto)",
    "a ±0.1 (m/s)/m gradient curves rays with radius $R_c = c_0/|g|$ ≈ 3.4 km; source $h_s$ = 2 m, receiver $h_r$ = 1.5 m":
        "gradiente de ±0,1 (m/s)/m → radio $R_c = c_0/|g|$ ≈ 3,4 km; fuente "
        "$h_s$ = 2 m, receptor $h_r$ = 1,5 m",
    # Aircraft noise certification points (ICAO Annex 16, Chapter 3)
    "Aircraft noise certification points (ICAO Annex 16, Chapter 3)":
        "Certificación de ruido de aeronaves (Anexo 16 OACI, Capítulo 3)",
    "Side view": "Vista lateral",
    "start of roll": "inicio de rodadura",
    "runway": "pista",
    "take-off": "despegue",
    "approach": "aproximación",
    "Approach reference point": "Punto de referencia de aproximación",
    "Flyover reference point": "Punto de referencia de sobrevuelo",
    "Lateral reference line": "Línea lateral de referencia",
    "where take-off noise is greatest": "donde el ruido de despegue es máximo",
    "symmetric lateral point (measured on both sides)":
        "punto lateral simétrico (se mide a ambos lados)",
    "Microphones 1.2 m above ground; the certification metric at the three points is EPNL, in EPNdB":
        "Micrófonos a 1,2 m del suelo; la métrica de certificación en los "
        "tres puntos es el EPNL (EPNdB)",
    "Lateral: full take-off power · Flyover: 6.5 km from brake release · Approach: 3° ± 0.5° glide path":
        "Lateral: potencia máxima · Sobrevuelo: 6,5 km tras soltar frenos · "
        "Aproximación: senda 3° ± 0,5°",
    "the approach point lies 120 m below the 3° path, which meets the ground 300 m past the threshold":
        "el punto de aproximación queda 120 m bajo la senda de 3°, que toca "
        "suelo 300 m tras el umbral",
    # Helicopter overflight certification (ICAO Annex 16, Chapter 8)
    "Helicopter overflight noise certification (ICAO Annex 16, Chapter 8)":
        "Certificación de ruido de helicópteros (Anexo 16 OACI, Capítulo 8)",
    "level flight at 0.9 $V_H$": "vuelo nivelado a 0,9 $V_H$",
    "centre microphone": "micrófono central",
    "track": "trayectoria",
    "3 microphones on a line perpendicular to the track":
        "3 micrófonos en una línea perpendicular a la trayectoria",
    "Speed: the least of 0.9 $V_H$, 0.9 $V_{NE}$, 0.45 $V_H$ + 120 km/h and 0.45 $V_{NE}$ + 120 km/h":
        "Velocidad: la menor de 0,9 $V_H$, 0,9 $V_{NE}$, 0,45 $V_H$ + "
        "120 km/h y 0,45 $V_{NE}$ + 120 km/h",
    "EPNL in EPNdB at the three points; at least six overflights, headwind and tailwind in equal number":
        "EPNL en EPNdB en los tres puntos; al menos seis pasadas, con viento "
        "en cara y en cola a partes iguales",
    "microphones 1.2 m above ground; the sideline pair sees the overhead helicopter at 45° (slant ≈ 212 m)":
        "micrófonos a 1,2 m del suelo; el par lateral ve el helicóptero en "
        "la vertical con 45° (oblicua ≈ 212 m)",
    # Swept-sine distortion (Farina / Novak)
    "Swept-sine distortion: deconvolution and harmonic pre-arrivals":
        "Distorsión por barrido senoidal: deconvolución y prellegadas "
        "armónicas",
    "Exponential sweep $x(t)$": "Barrido exponencial $x(t)$",
    "20 Hz → 6 kHz in $T$ = 4 s": "20 Hz → 6 kHz en $T$ = 4 s",
    "Device under test": "Dispositivo en ensayo",
    "weakly nonlinear: gain + harmonics":
        "débilmente no lineal: ganancia + armónicos",
    "Recording $y(t)$": "Grabación $y(t)$",
    "sweep + distortion products": "barrido + productos de distorsión",
    "Deconvolve with the inverse filter":
        "Deconvolución con el filtro inverso",
    "time-reversed sweep with a +6 dB/octave tilt":
        "barrido invertido en el tiempo, con +6 dB/octava",
    "$h_1$ (linear), $t = 0$": "$h_1$ (lineal), $t = 0$",
    "harmonic orders arrive early,": "los órdenes armónicos llegan antes,",
    "each in its own window": "cada uno en su propia ventana",
    "$L·ln 2$ = 0.49 s": "$L·ln 2$ = 0,49 s",
    "$L·ln 3$ = 0.77 s": "$L·ln 3$ = 0,77 s",
    "time": "tiempo",
    "$L = T / ln(f_2/f_1)$ = 0.70 s; the order-$n$ products compress $L·ln n$ ahead of the linear response":
        "$L = T / ln(f_2/f_1)$ = 0,70 s; los productos de orden $n$ se "
        "adelantan $L·ln n$ a la respuesta lineal",
    "window each arrival  →  $H_{1}(f)$, $H_{2}(f)$, $H_{3}(f)$, …  →  $THD(f) = √( Σ |H_{n}(n f)|^2 ) / |H_{1}(f)|$":
        "enventanar cada llegada  →  $H_{1}(f)$, $H_{2}(f)$, $H_{3}(f)$, …  →"
        "  $THD(f) = √( Σ |H_{n}(n f)|^2 ) / |H_{1}(f)|$",
    # Two-channel FRF measurement (H1 estimator and coherence)
    "Two-channel FRF measurement: the H1 estimator and coherence":
        "Medición de FRF a dos canales: el estimador H1 y la coherencia",
    "Signal generator": "Generador de señal",
    "broadband noise or a sweep": "ruido de banda ancha o barrido",
    "its gain is in H1": "su ganancia va en H1",
    "Power amplifier": "Amplificador",
    "Loudspeaker under test": "Altavoz en ensayo",
    "measurement microphone": "micrófono de medición",
    "Channel 1: reference $x(t)$": "Canal 1: referencia $x(t)$",
    "the electrical drive signal": "la señal eléctrica de excitación",
    "Channel 2: response $y(t)$": "Canal 2: respuesta $y(t)$",
    "acoustic output at the microphone": "salida acústica en el micrófono",
    "Dual-channel FFT analysis (Welch)":
        "Análisis FFT de dos canales (Welch)",
    # The estimator formulas read the same in both languages.
    "$H_{1}(f) = G_{xy} / G_{xx}$": "$H_{1}(f) = G_{xy} / G_{xx}$",
    "$γ^{2}(f) = |G_{xy}|^2 / (G_{xx}·G_{yy})$":
        "$γ^{2}(f) = |G_{xy}|^2 / (G_{xx}·G_{yy})$",
    "Hann segments, 50 % overlap  →  $G_{xx}(f)$, $G_{yy}(f)$, $G_{xy}(f)$":
        "segmentos Hann, 50 % de solape  →  $G_{xx}(f)$, $G_{yy}(f)$, "
        "$G_{xy}(f)$",
    "unbiased with output noise; $H_2 = G_{yy}/G_{yx}$ for input noise":
        "insesgado ante ruido a la salida ($H_2 = G_{yy}/G_{yx}$ a la "
        "entrada)",
    "1 for a noiseless linear path; less with output noise":
        "1 en un camino lineal sin ruido; menor con ruido a la salida",
    "trust $|H_1|$ only where $γ^2$ stays near 1: coherence dips flag noise, distortion or an unresolved delay":
        "fiarse de $|H_1|$ solo donde $γ^2$ ronda 1: las caídas delatan "
        "ruido, distorsión o retardo sin resolver",
    # Test-signal family panel
    "The test-signal family at a glance":
        "La familia de señales de ensayo de un vistazo",
    "White noise": "Ruido blanco",
    "Pink noise": "Ruido rosa",
    "Sweeps: linear vs exponential": "Barridos: lineal frente a exponencial",
    "Tone burst": "Salva de tono",
    "flat PSD: 0 dB/octave": "DEP plana: 0 dB/octava",
    # log10 is shared notation (as for the reception-plate relations above).
    "$log_{10} f$": "$log_{10} f$",
    "equal power per hertz": "igual potencia por hercio",
    "−3 dB/octave PSD": "DEP de −3 dB/octava",
    "equal power per octave": "igual potencia por octava",
    "flat, line spectrum": "espectro de rayas plano",
    "binary ±1, period $2^m − 1$ samples":
        "binaria ±1, periodo $2^m − 1$ muestras",
    "linear": "lineal",
    "exponential": "exponencial",
    "exponential: equal time (and energy) per octave; linear: equal time per hertz":
        "exponencial: igual tiempo (y energía) por octava; lineal: igual "
        "tiempo por hercio",
    "whole periods, starting at": "periodos completos, empezando en",
    "a zero crossing (IEC 60268-1)": "un paso por cero (IEC 60268-1)",
    "25 periods of 5 kHz = 5 ms": "25 periodos de 5 kHz = 5 ms",
    "every stimulus is deterministic and repeatable; synchronous averaging then lowers uncorrelated noise":
        "todos los estímulos son deterministas y repetibles; promediar "
        "pasadas reduce el ruido no correlacionado",
    "sweeps separate harmonic distortion, MLS smears it across the period, bursts probe dynamics":
        "los barridos separan la distorsión armónica, la MLS la reparte por "
        "el periodo y las salvas sondean la dinámica",
    # Welch PSD pipeline (Bendat & Piersol)
    "The Welch PSD pipeline: segment, taper, average (Bendat & Piersol)":
        "PSD de Welch: segmentar, enventanar, promediar (Bendat & Piersol)",
    "Record $x(t)$ — $f_s$ = 48 kHz, 20 s of pink noise":
        "Registro $x(t)$ — $f_s$ = 48 kHz, 20 s de ruido rosa",
    "960 000 samples, calibrated end to end: pascals in, Pa²/Hz out":
        "960 000 muestras, calibrado de extremo a extremo: pascales dentro, "
        "Pa²/Hz fuera",
    "Split into 50 %-overlapped segments — nperseg = 4096":
        "División en segmentos con 50 % de solape — nperseg = 4096",
    "467 segments of 85.3 ms; bin spacing $Δf = f_s/4096$ = 11.7 Hz":
        "467 segmentos de 85,3 ms; separación de bins $Δf = f_s/4096$ = "
        "11,7 Hz",
    "Hann taper on every segment": "Ventana de Hann en cada segmento",
    "ENBW = 1.5 bins → resolution bandwidth $B_e = 1.5·Δf$ = 17.6 Hz":
        "ENBW = 1,5 bins → ancho de banda de resolución $B_e = 1,5·Δf$ = "
        "17,6 Hz",
    "One-sided $|FFT|^2$ periodogram of each segment, then average":
        "Periodograma unilateral $|FFT|^2$ de cada segmento, y promedio",
    "overlap correlation (Welch 1967): 467 segments → $n_d$ = 442 effective averages":
        "correlación por solape (Welch 1967): 467 segmentos → $n_d$ = 442 "
        "promedios efectivos",
    "$G_{xx}(f)$ with its chi-square confidence interval":
        "$G_{xx}(f)$ con su intervalo de confianza chi-cuadrado",
    "random error $ε_r = 1/√n_d$ = 4.8 %;  $2·n_d ≈ 885$ degrees of freedom":
        "error aleatorio $ε_r = 1/√n_d$ = 4,8 %;  $2·n_d ≈ 885$ grados "
        "de libertad",
    "The trade-off: segment length buys resolution or stability, never both":
        "El compromiso: el segmento compra resolución o estabilidad, "
        "nunca ambas",
    "longer segments → finer $B_e$ but fewer averages (larger $ε_r$); shorter → the reverse":
        "segmentos más largos → $B_e$ más fino pero menos promedios "
        "(mayor $ε_r$); más cortos → lo contrario",
    # MISO coherence (Bendat & Piersol Chapter 7)
    "MISO coherence: from correlated sources to per-source contributions":
        "Coherencia MISO: de fuentes correladas a contribuciones por fuente",
    "Input $x_1$": "Entrada $x_1$",
    "white noise": "ruido blanco",
    "Input $x_2 = 0.7·x_1$ + noise": "Entrada $x_2 = 0,7·x_1$ + ruido",
    "correlated with $x_1$": "correlada con $x_1$",
    "Path $H_{1}(f)$": "Camino $H_{1}(f)$",
    "low-pass, 400 Hz": "paso bajo, 400 Hz",
    "Path $H_{2}(f)$": "Camino $H_{2}(f)$",
    "high-pass, 1.5 kHz": "paso alto, 1,5 kHz",
    "noise $n(t)$": "ruido $n(t)$",
    "Output $y(t)$": "Salida $y(t)$",
    "Welch cross-spectral matrix — $G_{xx}$ (2×2) and $G_{xy}$, nperseg = 2048":
        "Matriz de espectros cruzados de Welch — $G_{xx}$ (2×2) y "
        "$G_{xy}$, nperseg = 2048",
    "conditioning: Schur steps $G_{ij·r!}$ (Eq. 7.94), inputs ordered by descending ordinary coherence":
        "condicionamiento: pasos de Schur $G_{ij·r!}$ (Ec. 7.94), "
        "entradas ordenadas por coherencia ordinaria descendente",
    "Multiple and partial coherence": "Coherencia múltiple y parcial",
    "input 2 in the 100-300 Hz band: ordinary 0.32 → partial 0.00":
        "entrada 2 en la banda de 100-300 Hz: ordinaria 0,32 → parcial "
        "0,00",
    "multiple $γ^2_{y:x} = 1 − G_{nn}/G_{yy}$ ≈ 1.00 (100-300 Hz)":
        "múltiple $γ^2_{y:x} = 1 − G_{nn}/G_{yy}$ ≈ 1,00 (100-300 Hz)",
    "Contribution of each source": "Contribución de cada fuente",
    "$G_{vi} = γ^2_{iy·(i−1)!}·G_{yy}$ per input":
        "$G_{vi} = γ^2_{iy·(i−1)!}·G_{yy}$ por entrada",
    "$ΣG_{vi} + G_{nn} = G_{yy}$, band by band":
        "$ΣG_{vi} + G_{nn} = G_{yy}$, banda a banda",
    "each conditioning step spends one average: the $i$-th ordered input carries $n_d − (i − 1)$; here $n_d$ = 242":
        "cada paso de condicionamiento gasta un promedio: la entrada "
        "$i$-ésima ordenada lleva $n_d − (i − 1)$; aquí $n_d$ = 242",
    "average generously before reading a small partial coherence as zero":
        "promedia con generosidad antes de leer como cero una coherencia "
        "parcial pequeña",
    # Time-frequency tiling trade-off
    "The time-frequency trade-off: two tilings of the same record":
        "El compromiso tiempo-frecuencia: dos teselados del mismo registro",
    "Short window — nperseg = 256": "Ventana corta — nperseg = 256",
    "Long window — nperseg = 1024": "Ventana larga — nperseg = 1024",
    "$T_B$ = 16 ms,  $B_e ≈ 1/T_B$ = 62.5 Hz":
        "$T_B$ = 16 ms,  $B_e ≈ 1/T_B$ = 62,5 Hz",
    "$T_B$ = 64 ms,  $B_e$ ≈ 15.6 Hz": "$T_B$ = 64 ms,  $B_e$ ≈ 15,6 Hz",
    "sharp click, smeared tone": "clic nítido, tono emborronado",
    "sharp tone, smeared click": "tono nítido, clic emborronado",
    "tone": "tono",
    "click": "clic",
    "each cell is one unaveraged estimate: $B_e·T_B ≈ 1$ and $ε_r = 1$ ($n_d = 1$)":
        "cada celda es una estimación sin promediar: $B_e·T_B ≈ 1$ y "
        "$ε_r = 1$ ($n_d = 1$)",
    "the record fixes the product; nperseg only chooses how to spend it ($f_s$ = 16 kHz here)":
        "el registro fija el producto; nperseg solo elige cómo gastarlo "
        "(aquí $f_s$ = 16 kHz)",
    # Cepstrum chain (Havelock Ch. 27)
    "The cepstrum chain: an echo becomes a quefrency spike":
        "La cadena del cepstro: un eco se vuelve un pico en quefrencia",
    "Signal with one echo": "Señal con un eco",
    "Ripply spectrum $|X(f)|$": "Espectro ondulado $|X(f)|$",
    "cosine ripple of period": "ondulación coseno de periodo",
    "Take the log: $ln |X|^2$": "Logaritmo: $ln |X|^2$",
    "the multiplicative echo": "el eco multiplicativo",
    "becomes an additive ripple": "se vuelve ondulación aditiva",
    "Inverse FFT": "FFT inversa",
    "quefrency axis, in seconds": "eje de quefrencia, en segundos",
    "the cepstrum": "el cepstro",
    "quefrency": "quefrencia",
    "source wavelet,": "ondícula de la fuente,",
    "below 2 ms": "bajo 2 ms",
    "$a$ = 0.5,  $t_0$ = 8 ms": "$a$ = 0,5;  $t_0$ = 8 ms",
    "$a$ = 0.5 at $t_0$ = 8 ms": "$a$ = 0,5 en $t_0$ = 8 ms",
    "$−a^2/2$ = −0.125": "$−a^2/2$ = −0,125",
    "lifter cutoff 4 ms": "corte del lifter en 4 ms",
    "lowpass: envelope": "paso bajo: envolvente",
    "highpass: the echo ripple alone":
        "paso alto: solo la ondulación del eco",
    "rahmonics at $n·t_0$ with heights $a$, $−a^2/2$, $a^3/3$, …, whatever the source spectrum does":
        "rahmónicos en $n·t_0$ con alturas $a$, $−a^2/2$, $a^3/3$, …, "
        "haga lo que haga el espectro de la fuente",
    "the highpass ripple swings between $20·log_{10}(1 ± a)$ = +3.5 and −6.0 dB; echo_detection reads $t_0$ and $a$ off the peak":
        "la ondulación paso alto oscila entre $20·log_{10}(1 ± a)$ = "
        "+3,5 y −6,0 dB; echo_detection lee $t_0$ y $a$ del pico",
    # Time synchronous averaging (McFadden 1987)
    "Time synchronous averaging: trigger, slice, average":
        "Promediado síncrono temporal: disparo, troceado, promedio",
    "Tachometer: one trigger pulse per revolution":
        "Tacómetro: un pulso de disparo por revolución",
    "$T$ = 1/32 s = 256 samples": "$T$ = 1/32 s = 256 muestras",
    "Recording $y(t)$ at $f_s$ = 8192 Hz: the synchronous signature buried in noise":
        "Registro $y(t)$ a $f_s$ = 8192 Hz: la firma síncrona sepultada "
        "en ruido",
    "slice at every trigger": "trocear en cada disparo",
    "$N$ aligned blocks": "$N$ bloques alineados",
    "one period $T$ each": "de un periodo $T$ cada uno",
    "Coherent average": "Promedio coherente",
    "$N$ = 40 here": "aquí $N$ = 40",
    "The periodic part survives": "La parte periódica sobrevive",
    "comb teeth of unit gain": "dientes del peine de ganancia uno",
    "at every order $k/T$": "en cada orden $k/T$",
    "Asynchronous noise falls as $1/√N$":
        "El ruido asíncrono cae como $1/√N$",
    "power $−10·log_{10} N$ = −16 dB for $N$ = 40;  amplitude gain $√N$ = 6.3":
        "potencia $−10·log_{10} N$ = −16 dB con $N$ = 40;  ganancia en "
        "amplitud $√N$ = 6,3",
    "Residual": "Residual",
    "record − tiled average:": "registro − promedio repetido:",
    "everything not synchronous": "todo lo no síncrono",
    "a tone on a non-integer order is only attenuated: choose $N$ so a comb node lands on it":
        "un tono en un orden no entero solo se atenúa: elige $N$ para "
        "que un nodo del peine caiga sobre él",
    "McFadden's example: $N$ = 20 nulls the 32.05-order tone (20·32.05 = 641); the habitual $N$ = 32 does not":
        "el ejemplo de McFadden: $N$ = 20 anula el tono de orden 32,05 "
        "(20·32,05 = 641); el habitual $N$ = 32 no",
    # Correlation-based time-delay estimation (Knapp & Carter)
    "Time-delay estimation: two microphones and one correlation peak":
        "Estimación del retardo: dos micrófonos y un pico de correlación",
    "$Δr = c·τ_0$ ≈ 0.84 m  ($c$ = 343 m/s)":
        "$Δr = c·τ_0$ ≈ 0,84 m  ($c$ = 343 m/s)",
    "mic 1 — $x(t)$": "micro 1 — $x(t)$",
    "mic 2 — $y(t)$": "micro 2 — $y(t)$",
    "spacing $d$": "separación $d$",
    # The page's own displayed formula writes \sin (ISO 80000-2 notation),
    # so the plate keeps "sin"; the PHAT weight is shared notation too.
    "$sin θ = c·τ_0 / d$": "$sin θ = c·τ_0 / d$",
    "$ψ(f) = 1/|G_{xy}|$": "$ψ(f) = 1/|G_{xy}|$",
    "cross-correlation against lag — $y(t) = α·x(t − τ_0) + n(t)$":
        "correlación cruzada frente al retardo — $y(t) = α·x(t − τ_0) + "
        "n(t)$",
    "direct correlator: broad peak": "correlador directo: pico ancho",
    "GCC-PHAT: sharp spike": "GCC-PHAT: pico estrecho",
    "$τ_0$ = 20 samples / 8192 Hz = 2.44 ms":
        "$τ_0$ = 20 muestras / 8192 Hz = 2,44 ms",
    "parabolic peak interpolation + ×16 local upsampling → error below 0.002 samples":
        "interpolación parabólica del pico + sobremuestreo local ×16 → "
        "error por debajo de 0,002 muestras",
    "the 'phase' route reads the same $τ_0$ from the slope of the unwrapped cross-spectrum phase":
        "la vía 'phase' lee el mismo $τ_0$ de la pendiente de la fase "
        "desenrollada del espectro cruzado",
    # Data qualification decision flow (Bendat & Piersol 10.3)
    "Data qualification: the stationarity decision (Bendat & Piersol 10.3)":
        "Calificación de datos: decisión de estacionariedad (B&P 10.3)",
    "Time record $x(t)$": "Registro temporal $x(t)$",
    "before trusting any PSD, Leq or GUM average":
        "antes de confiar en cualquier promedio PSD, Leq o GUM",
    "Mean square per interval — $N$ = 20 equal segments":
        "Media cuadrática por intervalo — $N$ = 20 segmentos iguales",
    "each interval long against the record's lowest frequencies; also rms, mean or variance":
        "cada intervalo largo frente a las frecuencias más bajas del "
        "registro; también rms, media o varianza",
    "Reverse arrangement count $A$": "Recuento de inversiones $A$",
    "pairs $i < j$ with $x_i > x_j$; trend-free mean $μ_A = N(N−1)/4$ = 95":
        "pares $i < j$ con $x_i > x_j$; media sin tendencia "
        "$μ_A = N(N−1)/4$ = 95",
    "(Table A.6, $α = 0.05$)": "(Tabla A.6, $α = 0,05$)",
    "yes": "sí",
    "Nonstationary: do not average": "No estacionario: no promediar",
    "+20 % gain ramp: $A = 7$ → rejected":
        "rampa de ganancia del +20 %: $A = 7$ → rechazado",
    "split at the change, or go short-time (spectrogram)":
        "divide en el cambio, o pasa a corto plazo (espectrograma)",
    "Stationary: analyse": "Estacionario: analizar",
    "steady noise: $A = 91$ → accepted":
        "ruido estable: $A = 91$ → aceptado",
    "the chi-square CIs and error formulas hold":
        "los IC chi-cuadrado y las fórmulas de error valen",
    "the runs test (method=\"runs\") is the two-sided companion: too many runs is as suspect as too few":
        "el test de rachas (method=\"runs\") es el compañero bilateral: "
        "demasiadas rachas son tan sospechosas como muy pocas",
    "a frequency glide can hide from the mean square: test statistic=\"mean\" or band-filtered copies too":
        "un deslizamiento en frecuencia puede esconderse de la media "
        "cuadrática: prueba statistic=\"mean\" o copias filtradas por "
        "bandas",
    # Sound-quality metric family (DIN 45692 + ECMA-418-2)
    "Sound quality beyond loudness: four calibrated sensations":
        "Calidad sonora más allá de la sonoridad: cuatro sensaciones",
    "Calibrated signal $x(t)$ in pascals":
        "Señal calibrada $x(t)$ en pascales",
    "any sample rate: each metric resamples to 48 kHz internally":
        "cualquier fs: cada métrica remuestrea internamente a 48 kHz",
    "Specific loudness $N′(z)$": "Sonoridad específica $N′(z)$",
    "Zwicker pattern over 24 Bark": "patrón de Zwicker sobre 24 Bark",
    "Sottek Hearing Model front end (ECMA-418-2)":
        "Etapa de entrada del modelo de Sottek (ECMA-418-2)",
    "outer/middle-ear filter + 53 auditory bands (Bark_HMS)":
        "filtro de oído externo/medio + 53 bandas auditivas (Bark_HMS)",
    "Sharpness $S$": "Agudeza $S$",
    "$g(z)$-weighted first moment": "primer momento ponderado por $g(z)$",
    "of $N′(z)$, with $k$ = 0.108": "de $N′(z)$, con $k$ = 0,108",
    "critical-band-wide noise": "ruido de una banda crítica",
    "at 1 kHz, 60 dB": "a 1 kHz, 60 dB",
    "→ $S$ = 1.00 acum": "→ $S$ = 1,00 acum",
    "Tonality $T$": "Tonalidad $T$",
    "ECMA-418-2 clause 6": "ECMA-418-2, cláusula 6",
    "band autocorrelation finds": "la autocorrelación por banda",
    "periodic components": "detecta componentes periódicas",
    "1 kHz tone at 40 dB": "tono de 1 kHz a 40 dB",
    "→ $T$ = 1.000 tu_HMS (999 Hz)": "→ $T$ = 1,000 tu_HMS (999 Hz)",
    "Roughness $R$": "Aspereza $R$",
    "ECMA-418-2 clause 7": "ECMA-418-2, cláusula 7",
    "fast envelope modulation,": "modulación rápida de la envolvente,",
    "band-pass peaking near 70 Hz": "paso banda con pico hacia 70 Hz",
    "1 kHz, 100 % AM at 70 Hz, 60 dB": "1 kHz, AM 100 % a 70 Hz, 60 dB",
    "→ $R$ = 0.9999 asper": "→ $R$ = 0,9999 asper",
    "Fluctuation strength $F$": "Intensidad de fluctuación $F$",
    "ECMA-418-2 clause 9 (HSA)": "ECMA-418-2, cláusula 9 (HSA)",
    "slow envelope modulation,": "modulación lenta de la envolvente,",
    "band-pass peaking near 4 Hz": "paso banda con pico hacia 4 Hz",
    "1 kHz, 100 % AM at 4 Hz, 60 dB": "1 kHz, AM 100 % a 4 Hz, 60 dB",
    "→ $F$ = 0.9957 vacil_HMS": "→ $F$ = 0,9957 vacil_HMS",
    "Downstream, the sensations combine into annoyance":
        "Aguas abajo, las sensaciones se combinan en molestia",
    "$N_5$, $S$, $R$ and $F$ feed the Fastl and Zwicker psychoacoustic annoyance $PA = N_5·(1 + √(w_S^2 + w_{FR}^2))$":
        "$N_5$, $S$, $R$ y $F$ alimentan la molestia psicoacústica de Fastl y "
        "Zwicker $PA = N_5·(1 + √(w_S^2 + w_{FR}^2))$",
    # Tone audibility (ISO/PAS 20065)
    "Tone audibility: from spectrum to penalty (ISO/PAS 20065)":
        "Audibilidad tonal: del espectro al ajuste (ISO/PAS 20065)",
    "Narrow-band FFT spectrum — line spacing $Δf$ = 2.7 Hz":
        "Espectro FFT de banda estrecha — resolución $Δf$ = 2,7 Hz",
    "Annex E engine spectrum; peak detected at $f_T$ = 137.3 Hz (not on a slope)":
        "espectro del motor del Anexo E; pico detectado en $f_T$ = 137,3 Hz "
        "(no en una ladera)",
    "Critical band about the tone — $Δf_c$ = 101.36 Hz":
        "Banda crítica en torno al tono — $Δf_c$ = 101,36 Hz",
    "geometric placement: corners 95.67 and 197.04 Hz, $√(f_1·f_2) = f_T$":
        "colocación geométrica: esquinas en 95,67 y 197,04 Hz, "
        "$√(f_1·f_2) = f_T$",
    "Levels from the spectrum lines in the band":
        "Niveles desde las líneas del espectro en la banda",
    "masking noise $L_S$ = 49.22 dB (iterative mean); tone $L_T$ = 67.96 dB (energy sum)":
        "ruido enmascarante $L_S$ = 49,22 dB (media iterativa); tono "
        "$L_T$ = 67,96 dB (suma energética)",
    "Masking threshold seen by the ear":
        "Umbral de enmascaramiento visto por el oído",
    "$L_G = L_S + 10·log_{10}(Δf_c/Δf)$ = 64.97 dB;  masking index "
    "$a_v$ = −2.02 dB":
        "$L_G = L_S + 10·log_{10}(Δf_c/Δf)$ = 64,97 dB;  índice de "
        "enmascaramiento $a_v$ = −2,02 dB",
    "Audibility $ΔL = L_T − L_G − a_v$ = 5.01 dB":
        "Audibilidad $ΔL = L_T − L_G − a_v$ = 5,01 dB",
    "the largest $ΔL$ of the nine tones: the decisive audibility of this spectrum":
        "el mayor $ΔL$ de los nueve tonos: la audibilidad decisiva de este "
        "espectro",
    "From audibility to penalty (ISO 1996-2:2017 Annex J)":
        "De la audibilidad al ajuste (ISO 1996-2:2017, Anexo J)",
    "energy mean of the five spectra $ΔL$ = 6.98 dB → tonal adjustment $K_t$ = 4 dB (Table J.1)":
        "media energética de los cinco espectros $ΔL$ = 6,98 dB → ajuste "
        "tonal $K_t$ = 4 dB (Tabla J.1)",
    # Psychoacoustic annoyance (Fastl & Zwicker)
    "Psychoacoustic annoyance: four sensations, one scalar":
        "Molestia psicoacústica: cuatro sensaciones, un escalar",
    "$S$ = 2.0 acum": "$S$ = 2,0 acum",
    "sharpness (DIN 45692)": "agudeza (DIN 45692)",
    "counts only above 1.75 acum": "solo cuenta sobre 1,75 acum",
    "$N_5$ = 30 sone": "$N_5$ = 30 sone",
    "percentile loudness (ISO 532-1)": "sonoridad percentil (ISO 532-1)",
    "exceeded 5 % of the time": "superada el 5 % del tiempo",
    "$F$ = 0.5 vacil": "$F$ = 0,5 vacil",
    "fluctuation strength": "intensidad de fluctuación",
    "slow modulation, ≈ 4 Hz": "modulación lenta, ≈ 4 Hz",
    "$R$ = 0.3 asper": "$R$ = 0,3 asper",
    "roughness": "aspereza",
    "fast modulation, ≈ 70 Hz": "modulación rápida, ≈ 70 Hz",
    "Sharpness weighting $w_S$ = 0.1001":
        "Ponderación de la agudeza $w_S$ = 0,1001",
    "$w_S = (S − 1.75) · 0.25 · log_{10}(N_5 + 10)$":
        "$w_S = (S − 1,75) · 0,25 · log_{10}(N_5 + 10)$",
    "zero for $S ≤ 1.75$ acum": "cero para $S ≤ 1,75$ acum",
    "Roughness and fluctuation $w_{FR}$ = 0.2125":
        "Aspereza y fluctuación $w_{FR}$ = 0,2125",
    "$w_{FR} = 2.18 / N_5^{0.4} · (0.4·F + 0.6·R)$":
        "$w_{FR} = 2,18 / N_5^{0,4} · (0,4·F + 0,6·R)$",
    "roughness weighs more: 0.6 against 0.4":
        "la aspereza pesa más: 0,6 frente a 0,4",
    "$PA = N_5 · (1 + √(w_S^2 + w_{FR}^2))$ = 37.05":
        "$PA = N_5 · (1 + √(w_S^2 + w_{FR}^2))$ = 37,05",
    "Fastl and Zwicker Eq. 16.2 (origin Widmann 1992)":
        "Fastl y Zwicker, Ec. 16.2 (origen Widmann 1992)",
    "a neutral sound ($S ≤ 1.75$ acum, $F = R = 0$) sits on the baseline $PA = N_5$":
        "un sonido neutro ($S ≤ 1,75$ acum, $F = R = 0$) queda en la línea "
        "base $PA = N_5$",
    "sharpness, roughness and fluctuation only ever lift the annoyance above the loudness":
        "la agudeza, la aspereza y la fluctuación solo elevan la "
        "molestia por encima de la sonoridad",
    # Objective intelligibility (STOI / ESTOI)
    "STOI and ESTOI: correlating clean against degraded speech":
        "STOI y ESTOI: correlación entre habla limpia y degradada",
    "Clean reference $x(t)$ and degraded version $y(t)$":
        "Referencia limpia $x(t)$ y versión degradada $y(t)$",
    "the guide's example: speech-like material in a flat masker at 0 dB SNR":
        "el ejemplo de la guía: material tipo habla en un enmascarador "
        "plano a 0 dB de SNR",
    "Resample to 10 kHz and drop the silent frames":
        "Remuestreo a 10 kHz y descarte de tramas silenciosas",
    "frames 40 dB below the loudest clean frame carry no intelligibility":
        "las tramas 40 dB bajo la trama limpia más alta no aportan "
        "inteligibilidad",
    "Short-time DFT: 256-sample Hann frames, 50 % overlap":
        "DFT de corto plazo: tramas Hann de 256 muestras, 50 % de solape",
    "magnitudes grouped into 15 one-third-octave bands from 150 Hz":
        "magnitudes agrupadas en 15 bandas de tercio de octava desde "
        "150 Hz",
    "384 ms segments — 30 frames, the unit of comparison":
        "Segmentos de 384 ms — 30 tramas, la unidad de comparación",
    "long enough to hold the slow modulations that carry speech":
        "lo bastante largos para las modulaciones lentas que llevan el "
        "habla",
    "STOI: envelope correlation": "STOI: correlación de envolventes",
    "per band and segment; normalise,": "por banda y segmento; normaliza,",
    "clip at −15 dB, then average": "recorta en −15 dB y promedia",
    "ESTOI: spectral correlation": "ESTOI: correlación espectral",
    "row- and column-normalised segments;":
        "segmentos normalizados por filas y columnas;",
    "credits glimpses in modulated maskers":
        "acredita los atisbos en enmascaradores modulados",
    "STOI = 0.727 for the example": "STOI = 0,727 para el ejemplo",
    "the lowest band keeps 0.27 of the correlation; above 1.9 kHz it reaches 0.90":
        "la banda más baja conserva 0,27 de la correlación; sobre "
        "1,9 kHz llega a 0,90",
    # Programme loudness (ITU-R BS.1770 / EBU R 128)
    "Programme loudness: the BS.1770 / R 128 metering chain":
        "Sonoridad de programa: la cadena de medición BS.1770 / R 128",
    "Programme $x$ — channel weights $G_i$: 1.0 front, 1.41 surround":
        "Programa $x$ — pesos de canal $G_i$: 1,0 frontales, 1,41 envolventes",
    "anchor: a 0 dB FS 997 Hz sine on one front channel reads −3.01 LKFS":
        "ancla: un seno de 997 Hz a 0 dB FS en un canal frontal marca "
        "−3,01 LKFS",
    "K-weighting: +4 dB spherical-head shelf + RLB high-pass":
        "Ponderación K: estante de +4 dB (cabeza esférica) + paso alto RLB",
    "$L_K = −0.691 + 10·log_{10} Σ G_i·z_i$;  LKFS ≡ LUFS, 1 LU = 1 dB":
        "$L_K = −0,691 + 10·log_{10} Σ G_i·z_i$;  LKFS ≡ LUFS, 1 LU = 1 dB",
    "Mean square in 400 ms blocks, 75 % overlap":
        "Media cuadrática en bloques de 400 ms, 75 % de solape",
    "absolute gate: blocks below −70 LUFS are dropped":
        "puerta absoluta: se descartan los bloques bajo −70 LUFS",
    "Relative gate: −10 LU below the survivors":
        "Puerta relativa: −10 LU bajo los supervivientes",
    "example: 10 s at −23 dBFS + 30 s of quiet → threshold −39.0 LUFS":
        "ejemplo: 10 s a −23 dBFS + 30 s de silencio → umbral −39,0 LUFS",
    "Integrated loudness $I$ = −23.1 LUFS: the tail is gated out":
        "Sonoridad integrada $I$ = −23,1 LUFS: la cola queda fuera",
    "EBU R 128 target −23.0 LUFS; tolerance ±0.2 LU in QC, ±1.0 LU live":
        "objetivo EBU R 128 −23,0 LUFS; tolerancia ±0,2 LU en QC, "
        "±1,0 LU en directo",
    "Loudness range $LRA = P_{95} − P_{10}$":
        "Rango de sonoridad $LRA = P_{95} − P_{10}$",
    "short-term 3 s windows, deeper −20 LU gate":
        "ventanas de corto plazo de 3 s, puerta más honda de −20 LU",
    "10.0 LU on the Tech 3342 two-step case":
        "10,0 LU en el caso de dos escalones de Tech 3342",
    "True peak: 4× oversampling, in dBTP":
        "Pico verdadero: sobremuestreo 4×, en dBTP",
    "the $f_s/4$ tone: sample peak −3.01 dB, true peak +0.12 dBTP":
        "el tono a $f_s/4$: pico muestral −3,01 dB, pico verdadero "
        "+0,12 dBTP",
    "R 128 production ceiling −1 dBTP":
        "techo de producción de R 128: −1 dBTP",
    "K-weighted": "ponderada K",
    "raw signal": "señal sin filtrar",
    "the gates keep quiet passages from dragging the foreground down":
        "las puertas evitan que los pasajes silenciosos arrastren el "
        "primer plano",
    "ungated, the same 40 s example would read near −29 LUFS":
        "sin puertas, el mismo ejemplo de 40 s marcaría cerca de −29 LUFS",
    # Reverberation-time prediction (Sabine / Eyring)
    "Predicting the reverberation time: Sabine against Eyring":
        "Predicción del tiempo de reverberación: Sabine frente a Eyring",
    "Room 10 × 7 × 3.5 m — $V$ = 245 m³, $S$ = 259 m²":
        "Sala de 10 × 7 × 3,5 m — $V$ = 245 m³, $S$ = 259 m²",
    "hard end walls, lightly treated side walls, carpet and acoustic ceiling":
        "testeros duros, laterales con tratamiento ligero, moqueta y "
        "techo acústico",
    "mean absorption $ᾱ$ runs from 0.21 at 125 Hz to 0.51 at 4 kHz":
        "la absorción media $ᾱ$ va de 0,21 a 125 Hz a 0,51 a 4 kHz",
    "$T = 0.161·V / (Σ S_i·α_i + 4·m·V)$":
        "$T = 0,161·V / (Σ S_i·α_i + 4·m·V)$",
    "low, even absorption ($ᾱ$ up to ≈ 0.2);":
        "absorción baja y uniforme ($ᾱ$ hasta ≈ 0,2);",
    "stays finite even at $α$ = 1": "queda finita incluso con $α$ = 1",
    "$T = 0.161·V / (−S·ln(1 − ᾱ) + 4·m·V)$":
        "$T = 0,161·V / (−S·ln(1 − ᾱ) + 4·m·V)$",
    "strong, even absorption;": "absorción fuerte y uniforme;",
    "reaches $T$ = 0 at total absorption":
        "llega a $T$ = 0 con absorción total",
    "Predicted $T_{60}$ per octave band":
        "$T_{60}$ predicho por banda de octava",
    # Model names are proper names; looked at, and the same in Spanish.
    "Sabine": "Sabine",
    "Eyring": "Eyring",
    "Sabine [s]": "Sabine [s]",
    "Eyring [s]": "Eyring [s]",
    "0.74": "0,74", "0.47": "0,47", "0.37": "0,37", "0.31": "0,31",
    "0.30": "0,30", "0.66": "0,66", "0.39": "0,39", "0.29": "0,29",
    "0.23": "0,23", "0.21": "0,21", "0.22": "0,22",
    "Eyring runs 11 to 29 % shorter here: $ᾱ$ is past Sabine's comfort zone":
        "Eyring sale entre un 11 y un 29 % más corto: $ᾱ$ excede la zona "
        "cómoda de Sabine",
    "Domain of validity: a diffuse field that stays diffuse while it decays":
        "Dominio de validez: un campo difuso que sigue difuso mientras "
        "decae",
    "below the Schroeder frequency, in coupled volumes and in corridor-like rooms no single $T_{60}$ exists":
        "bajo la frecuencia de Schroeder, en volúmenes acoplados y en "
        "salas tipo pasillo no existe un $T_{60}$ único",
    # Panel between rooms (mass law and coincidence)
    "Panel between rooms: mass law and the coincidence dip":
        "Panel entre recintos: ley de masas y valle de coincidencia",
    "Panel under test: 12.5 mm plasterboard":
        "Panel en ensayo: yeso laminado de 12,5 mm",
    "diffuse incidence": "incidencia difusa",
    "transmitted": "transmitido",
    "bending wave at $f_c$": "onda de flexión en $f_c$",
    "12.5 mm": "12,5 mm",
    "$m″$ = 8.8 kg/m²": "$m″$ = 8,8 kg/m²",
    "$f_c$ = 2.6 kHz": "$f_c$ = 2,6 kHz",
    "+6 dB/octave": "+6 dB/octava",
    "predicted $R(f)$": "$R(f)$ predicho",
    "Diffuse-field mass law: $R$ rises 6 dB per octave and 6 dB per doubling of $m″$":
        "Ley de masas en campo difuso: $R$ sube 6 dB por octava y 6 dB por "
        "duplicación de $m″$",
    "At $f_c = (c_0^2/2π) √(m″/B′)$ = 2619 Hz the free bending wave matches the trace wavelength":
        "En $f_c = (c_0^2/2π) √(m″/B′)$ = 2619 Hz la onda de flexión iguala "
        "la longitud de onda de traza",
    "Sharp's prediction rates $R_w$ = 27 dB; the dip takes the unfavourable deviations":
        "Sharp predice $R_w$ = 27 dB; el valle concentra las desviaciones "
        "desfavorables",
    # Porous layer on a rigid wall
    "Porous absorber on a rigid wall: microstructure to absorption":
        "Absorbente poroso sobre pared rígida: microestructura y absorción",
    "Porous layer (mineral wool)": "Capa porosa (lana mineral)",
    "plane wave, normal incidence": "onda plana, incidencia normal",
    "reflected: $|R|^2 = 1 − α$ = 0.09":
        "reflejado: $|R|^2 = 1 − α$ = 0,09",
    "microstructure (zoom)": "microestructura (ampliada)",
    "fibre frame": "esqueleto de fibras",
    "air in the pores: $φ$ = 0.98": "aire en los poros: $φ$ = 0,98",
    "$σ$ = 20 kPa·s/m²  (flow resistivity)":
        "$σ$ = 20 kPa·s/m²  (resistividad al flujo)",
    "$φ$ = 0.98  (porosity)": "$φ$ = 0,98  (porosidad)",
    "$α_∞$ = 1.0  (tortuosity)": "$α_∞$ = 1,0  (tortuosidad)",
    "$Λ = Λ′$ = 87 µm  (viscous / thermal lengths)":
        "$Λ = Λ′$ = 87 µm  (longitudes viscosa y térmica)",
    "JCA equivalent fluid: the five parameters give $Z_c$ and $k$; "
    "a hard-backed layer has $Z_s = −j Z_c cot(k·d)$":
        "Fluido equivalente JCA: los cinco parámetros dan $Z_c$ y $k$; con "
        "respaldo rígido $Z_s = −j Z_c cot(k·d)$",
    "$α = 1 − |R|^2$ = 0.91 at 1 kHz for this 50 mm layer":
        "$α = 1 − |R|^2$ = 0,91 a 1 kHz para esta capa de 50 mm",
    "viscous friction in the pores and heat exchange with the frame dissipate the sound energy":
        "la fricción viscosa en los poros y el intercambio de calor con el "
        "esqueleto disipan la energía sonora",
    # Barrier diffraction over ground (Fresnel number)
    "Barrier diffraction over ground: the Fresnel number at work":
        "Difracción en barrera sobre el suelo: el número de Fresnel en acción",
    "$A$ = 50.09 m": "$A$ = 50,09 m",
    "$B$ = 50.06 m": "$B$ = 50,06 m",
    "direct $d$ = 100.00 m (blocked)":
        "directo $d$ = 100,00 m (bloqueado)",
    "1.0 m": "1,0 m",
    "4.0 m": "4,0 m",
    "path difference $δ = A + B − d$ = 0.15 m; Fresnel number $N = 2δ/λ$ = 0.44 at 500 Hz":
        "diferencia de caminos $δ = A + B − d$ = 0,15 m; número de Fresnel "
        "$N = 2δ/λ$ = 0,44 a 500 Hz",
    "Kurze–Anderson: Δbar = 5 + 20 log10( √(2πN) / tanh √(2πN) ) = 10.0 dB at 500 Hz":
        "Kurze–Anderson: Δbar = 5 + 20 log10( √(2πN) / tanh √(2πN) ) = 10,0 dB "
        "a 500 Hz",
    "$N$ grows with frequency: the same screen gives 15.5 dB at 2 kHz (vertical scale exaggerated)":
        "$N$ crece con la frecuencia: la misma pantalla da 15,5 dB a 2 kHz "
        "(escala vertical exagerada)",
    # Image-source lattice in plan
    "Image-source lattice in plan: first reflections of a 7 × 5 m room":
        "Fuentes imagen en planta: primeras reflexiones (sala de 7 × 5 m)",
    "10.7 ms": "10,7 ms",
    "17.3 ms": "17,3 ms",
    "20.5 ms": "20,5 ms",
    "21.6 ms": "21,6 ms",
    "24.6 ms": "24,6 ms",
    "25.6 ms": "25,6 ms",
    "1st order": "1.er orden",
    "2nd order": "2.º orden",
    "the image sees": "la imagen ve",
    "a straight path": "un camino recto",
    "plan at the source plane $z$ = 1.5 m":
        "planta en el plano de la fuente $z$ = 1,5 m",
    "7.0 m": "7,0 m",
    "every reflection is the free-field arrival of a mirror image: "
    "$t = r/c$, $√(1−α)$ per bounce, $1/(4πr)$ spreading":
        "cada reflexión llega como campo libre de su imagen: $t = r/c$, "
        "$√(1−α)$ por rebote, esparcimiento $1/(4πr)$",
    "in-plane images up to order 2 shown; the full lattice adds floor, ceiling and outer mirror rooms":
        "imágenes en planta hasta orden 2; la retícula completa añade suelo, "
        "techo y salas más lejanas",
    # Noise control at source, path and receiver
    "Noise control at the source, along the path and at the receiver":
        "Control de ruido en la fuente, en el camino y en el receptor",
    "1 · At the source": "1 · En la fuente",
    "2 · Along the path": "2 · En el camino",
    "3 · At the receiver": "3 · En el receptor",
    "Enclosure": "Encapsulamiento",
    "Machine": "Máquina",
    "Operator cabin": "Cabina del operario",
    "expansion chamber": "cámara de expansión",
    "lined elbow": "codo revestido",
    "open end": "extremo abierto",
    "enclosure $IL = R − C$": "$IL = R − C$ del encapsulamiento",
    "25 dB at 500 Hz": "25 dB a 500 Hz",
    "silencer TL peak 6.5 dB at 286 Hz ($m = 4$)":
        "pico de TL del silenciador: 6,5 dB a 286 Hz ($m = 4$)",
    "lined elbow 6 dB at 1 kHz; open end 18 dB at 63 Hz":
        "codo revestido: 6 dB a 1 kHz; extremo abierto: 18 dB a 63 Hz",
    "cabin $IL = R − C$": "$IL = R − C$ de la cabina",
    "31 dB at 1 kHz": "31 dB a 1 kHz",
    "0.30 m": "0,30 m",
    "the classic ranking: quiet the source first, treat the path next, shield the receiver last":
        "la jerarquía clásica: primero la fuente, después el camino y por "
        "último el receptor",
    "enclosure and cabin share $IL = R − C$, with $C = 10 log_{10}(0.3 + S_E/R_i)$ = 4.9 dB for a lined interior ($ᾱ = 0.3$)":
        "encapsulado y cabina: $IL = R − C$, con "
        "$C = 10 log_{10}(0,3 + S_E/R_i)$ = 4,9 dB (interior revestido, "
        "$ᾱ = 0,3$)",
    "reactive silencer: $TL = 10 log_{10}[1 + ¼ (m − 1/m)^2 sin^{2}(k·L)]$, "
    "peaking where the 0.3 m chamber is $λ/4$":
        "silenciador reactivo: $TL = 10 log_{10}[1 + ¼ (m − 1/m)^2 "
        "sin^{2}(k·L)]$, máximo donde la cámara de 0,3 m mide $λ/4$",
    # Sound level meter pipeline (IEC 61672-1), one function per stage
    "The sound level meter pipeline: one function per stage":
        "La cadena del sonómetro: una función por etapa",
    "Calibrator tone": "Tono del calibrador",
    "94 dB at 1 kHz  (IEC 60942)": "94 dB a 1 kHz  (IEC 60942)",
    "Measurement recording": "Grabación de medición",
    "same microphone, same gain": "mismo micrófono, misma ganancia",
    "the factor $S$ in pascals per digital unit":
        "el factor $S$ en pascales por unidad digital",
    "Calibrated pressure   $p(t) = S · x(t)$   in pascals":
        "Presión calibrada   $p(t) = S · x(t)$   en pascales",
    "every level function takes $S$ as calibration_factor=":
        "toda función de nivel acepta $S$ como calibration_factor=",
    "Display and statistics": "Pantalla y estadística",
    "exponential detector, $τ_F$ = 125 ms":
        "detector exponencial, $τ_F$ = 125 ms",
    "Integrated levels": "Niveles integrados",
    # IEC 61672 level symbols; the Spanish page prints LCpeak too.
    "$L_{Aeq}$   $L_{AE}$   $L_{Cpeak}$":
        "$L_{Aeq}$   $L_{AE}$   $L_{Cpeak}$",
    "energy average, no ballistics": "promedio energético, sin balística",
    "Band spectrum": "Espectro por bandas",
    "IEC 61260-1 band edges": "bordes de banda IEC 61260-1",
    "dB re 20 µPa": "dB re 20 µPa",
    "one-third-octave band levels": "niveles por tercio de octava",
    "Class verification against the acceptance limits":
        "Verificación de clase frente a los límites de aceptación",
    "verify_weighting_class (IEC 61672-1 Table 3)  ·  verify_filter_class (IEC 61260-1 Table 1)":
        "verify_weighting_class (Tabla 3 de IEC 61672-1)  ·  "
        "verify_filter_class (Tabla 1 de IEC 61260-1)",
    # Calibration data flow (IEC 60942)
    "Calibration data flow: one factor, every level function":
        "Flujo de la calibración: un factor, todas las funciones de nivel",
    "Calibrator recording": "Grabación del calibrador",
    "1 kHz tone through the chain": "tono de 1 kHz por la misma cadena",
    "the same chain, untouched": "la misma cadena, sin tocar",
    "nothing in the chain may change between the two":
        "nada de la cadena puede cambiar entre una y otra",
    "fs enables the IEC 60942 stability check":
        "con fs comprueba la estabilidad (IEC 60942)",
    "pascals per digital unit": "pascales por unidad digital",
    # The reference notation, shared (cf. "dB re 20 µPa" above).
    "re 20 µPa": "re 20 µPa",
    "every level function accepts calibration_factor=":
        "toda función de nivel acepta calibration_factor=",
    "one factor for the whole library": "un solo factor para toda la biblioteca",
    "Levels in dB SPL": "Niveles en dB SPL",
    "No calibrator?": "¿Sin calibrador?",
    "$S = 1$, samples read as Pa": "$S = 1$, muestras como Pa",
    "use dbfs=True for dBFS": "usa dbfs=True para dBFS",
    # Filter bank data flow: decimation decision and band outputs
    "Inside a band: the decimation decision and the biquad cascade":
        "Dentro de una banda: diezmado y cascada de biquads",
    "Input signal  $x(t)$": "Señal de entrada  $x(t)$",
    "sample rate $f_s$": "frecuencia de muestreo $f_s$",
    "Room to decimate?": "¿Cabe diezmar?",
    "$f_s / 2 ≥ 1.25 · f_{upper}$": "$f_s / 2 ≥ 1,25 · f_{sup}$",
    "$M = floor[(f_s / 2) / (1.25 · f_{upper})]$":
        "$M = floor[(f_s / 2) / (1,25 · f_{sup})]$",
    "poles stay clear of $z = 1$": "los polos se alejan de $z = 1$",
    "SOS band filter at $f_s / M$": "Filtro SOS de banda a $f_s / M$",
    "SOS band filter at $f_s$": "Filtro SOS de banda a $f_s$",
    "cascaded biquads": "biquads en cascada",
    "designed on the IEC 61260-1 band edges":
        "diseñado sobre los bordes IEC 61260-1",
    "Every band filter": "Todo filtro de banda",
    "is a biquad cascade": "es una cascada de biquads",
    "not one high-order": "no un par (b, a)",
    "(b, a) pair": "de orden alto",
    "Band level": "Nivel de banda",
    "RMS or peak, in dB re 20 µPa": "RMS o pico, en dB re 20 µPa",
    "sigbands=True also returns the band signal at $f_s$":
        "sigbands=True devuelve además la señal de banda a $f_s$",
    "the decimated branch is interpolated back with resample_poly(M, 1)":
        "la rama diezmada se interpola de vuelta con resample_poly(M, 1)",
    # --- B9: buildings/design plates (EN 15657, ISO 16251-1, ISO 12354) ---
    "EN 15657 low- and high-mobility reception plates":
        "Placas receptoras de baja y alta movilidad de la EN 15657",
    "Low-mobility plate (7.2.2)": "Placa de baja movilidad (7.2.2)",
    "3,15 m x 2,23 m": "3,15 m x 2,23 m",
    "100 mm concrete, $ρ$ = 2 300 ± 200 kg/m³":
        "hormigón de 100 mm, $ρ$ = 2 300 ± 200 kg/m³",
    "$S$ = 7,0 m² (≥ 5 m²), sides ≈ $√2$ : 1":
        "$S$ = 7,0 m² (≥ 5 m²), lados ≈ $√2$ : 1",
    "$η ≥ 0,08$ over 50 Hz to 100 Hz": "$η ≥ 0,08$ de 50 Hz a 100 Hz",
    "≥ 6 velocity positions, ≈ 0,5 m apart":
        "≥ 6 posiciones de velocidad, a ≈ 0,5 m entre sí",
    "and ≥ 0,1 m from any contact point":
        "y a ≥ 0,1 m de cualquier punto de contacto",
    "elastic pads ≤ 100 × 100 mm": "apoyos elásticos ≤ 100 × 100 mm",
    "High-mobility plate (7.3.2)": "Placa de alta movilidad (7.3.2)",
    "source bolted rigidly": "fuente atornillada rígidamente",
    "support frame": "bastidor de sujeción",
    "1 mm steel or 1,5 mm aluminium": "acero de 1 mm o aluminio de 1,5 mm",
    "≈ 50 % perforated, ⌀ ≈ 6 mm holes,":
        "≈ 50 % perforada, agujeros de ⌀ ≈ 6 mm,",
    "so the source's own airborne sound":
        "para que el ruido aéreo de la propia fuente",
    "cannot drive the sheet": "no excite la lámina",
    "$T_s$ and $Y$ measured with the": "$T_s$ e $Y$ medidos con la",
    "source fitted (7.1)": "fuente instalada (7.1)",
    "Three-plate bench (Figure 2)": "Banco de tres placas (figura 2)",
    "whirlpool bath": "bañera de hidromasaje",
    "> 10 dB between plates": "> 10 dB entre placas",
    "up to three isolated plates,": "hasta tres placas aisladas,",
    "for a source that touches": "para una fuente que toca",
    "several building elements": "varios elementos constructivos",
    "the velocity level difference": "la diferencia de nivel de velocidad",
    "is measured per EN ISO 10848-1": "se mide según la EN ISO 10848-1",
    "in every band, with the": "en cada banda, con el",
    "equipment removed": "equipo desmontado",
    "low-mobility plate -> blocked force (15) -> characteristic power $L_{Wsn}$ (17)":
        "placa de baja movilidad -> fuerza bloqueada (15) -> potencia característica $L_{Wsn}$ (17)",
    "high-mobility plate -> free velocity (18) -> source mobility $|Y_{S,eq}|$ (19)":
        "placa de alta movilidad -> velocidad libre (18) -> movilidad de fuente $|Y_{S,eq}|$ (19)",
    "ISO 16251-1 small floor mock-up for floor-covering improvement":
        "Maqueta de suelo de la ISO 16251-1 para la mejora de un revestimiento",
    "Section": "Sección",
    "concrete slab, 200 ± 10 mm": "losa de hormigón, 200 ± 10 mm",
    "covering specimen": "probeta de revestimiento",
    "tapping machine (ISO 10140-5)": "máquina de impactos (ISO 10140-5)",
    "5 hammers, 0,5 kg from 40 mm, 10 s⁻¹":
        "5 martillos, 0,5 kg desde 40 mm, 10 s⁻¹",
    "accelerometer screwed or glued underneath":
        "acelerómetro atornillado o pegado por debajo",
    "elastic pads": "apoyos elásticos",
    "four elastic pads at the corners, each ≤ 100 × 100 mm":
        "cuatro apoyos elásticos en las esquinas, cada uno ≤ 100 × 100 mm",
    "vertical resonance of the slab on its pads < 20 Hz":
        "resonancia vertical de la losa sobre sus apoyos < 20 Hz",
    "top flat to ± 1 mm in a line edge to edge":
        "cara superior plana a ± 1 mm en línea de borde a borde",
    "Plan": "Planta",
    "machine positions above, accelerometers below":
        "posiciones de la máquina arriba, acelerómetros abajo",
    "1 200 × 800 mm (± 50 mm)": "1 200 × 800 mm (± 50 mm)",
    "≥ 2 machine positions, skew to the edges,":
        "≥ 2 posiciones de la máquina, oblicuas a los bordes,",
    "no hammer within 100 mm of an edge, all feet on the specimen":
        "ningún martillo a menos de 100 mm del borde, las patas sobre la probeta",
    "≥ 4 accelerometer positions, uniform but random,":
        "≥ 4 posiciones de acelerómetro, uniformes pero aleatorias,",
    "off the symmetry lines and ≥ 100 mm from every edge":
        "fuera de las líneas de simetría y a ≥ 100 mm de cada borde",
    "three cycles: with specimen  |  without specimen (hammers repeated within ± 20 mm)  |  background":
        "tres ciclos: con probeta  |  sin probeta (martillos repetidos dentro de ± 20 mm)  |  ruido de fondo",
    "≥ 20 s per level; background rule: unchanged ≥ 15 dB, energy subtraction 6-15 dB, −1,3 dB below 6 dB":
        "≥ 20 s por nivel; regla de fondo: sin cambio ≥ 15 dB, resta energética 6-15 dB, −1,3 dB por debajo de 6 dB",
    "$L_a = 10 lg(⟨a^2⟩/a_0^2)$,  $a_0$ = 10⁻⁶ m/s²   (Formula 1)":
        "$L_a = 10 lg(⟨a^2⟩/a_0^2)$,  $a_0$ = 10⁻⁶ m/s²   (fórmula 1)",
    "EN 12354-1 Annex E junction types, path branches and the mass ratio":
        "Uniones del anexo E (EN 12354-1), ramas de vía y cociente de masas",
    "rigid cross": "cruz rígida",
    "rigid T": "T rígida",
    "T with a flexible interlayer": "T con capa elástica intermedia",
    "elastic layer": "capa elástica",
    "corner": "esquina",
    "thickness change": "cambio de espesor",
    "lightweight double leaf": "doble hoja ligera",
    "rigid cross:": "cruz rígida:",
    "rigid T:": "T rígida:",
    "flexible T:": "T flexible:",
    "corner:": "esquina:",
    "thickness change:": "cambio de espesor:",
    "lightweight double leaf:": "doble hoja ligera:",
    "$M = lg(m′_{perp,i} / m′_i)$: $m′_i$ is the element carrying the path, so the ratio is per path, not per junction.":
        "$M = lg(m′_{perp,i} / m′_i)$: $m′_i$ es el elemento que lleva la vía, así que el cociente es por vía, no por unión.",
    "The functions take the RATIO, not $M$. Floor H.3, ratio 1,61: 'through' "
    "-> $K_{13}$ = 12,5 dB, 'corner' -> $K_{12}$ = 8,9 dB":
        "Las funciones toman el COCIENTE, no $M$. Forjado H.3, cociente 1,61: "
        "'through' $K_{13}$ = 12,5 dB; 'corner' 8,9 dB",
    "$ℓ_f$ is the coupling length along the junction line, surface to "
    "surface. Annex E values are read at 500 Hz, ± 3 dB.":
        "$ℓ_f$ es la longitud de acoplamiento en la unión, medida entre superficies. El anexo E se lee a 500 Hz, ± 3 dB.",
    "ISO 12354-1 Annex L worked building: elements, junctions, paths":
        "Edificio resuelto del anexo L (ISO 12354-1): elementos, uniones y vías",
    "Section: two stacked dwellings": "Sección: dos viviendas superpuestas",
    "source dwelling": "vivienda emisora",
    "receiving dwelling": "vivienda receptora",
    "T  rigid T (floor to external wall): $K_{ij}$ = 6,4 / 11,2 dB":
        "T  T rígida (forjado a muro exterior): $K_{ij}$ = 6,4 / 11,2 dB",
    "X  rigid cross (floor to internal wall): $K_{ij}$ = 8,8 / 11,0 dB":
        "X  cruz rígida (forjado a tabique interior): $K_{ij}$ = 8,8 / 11,0 dB",
    "separating floor  220 mm concrete, 484 kg/m², $f_c$ = 76,8 Hz":
        "forjado separador  hormigón de 220 mm, 484 kg/m², $f_c$ = 76,8 Hz",
    "on it  35 mm screed, 73,5 kg/m², on $s′$ = 8 MN/m³":
        "sobre él  solera de 35 mm, 73,5 kg/m², sobre $s′$ = 8 MN/m³",
    "external walls  365 mm AAC, 219 kg/m², $f_c$ = 92,6 Hz":
        "muros exteriores  hormigón celular de 365 mm, 219 kg/m², $f_c$ = 92,6 Hz",
    "internal walls  200 mm calcium silicate, 360 kg/m², $f_c$ = 128,4 Hz":
        "tabiques interiores  silicocalcáreo de 200 mm, 360 kg/m², $f_c$ = 128,4 Hz",
    "Plan: the separating floor": "Planta: el forjado separador",
    "external wall (T)": "muro exterior (T)",
    "internal wall (X)": "tabique interior (X)",
    "two external and two internal walls meet the floor, with":
        "dos muros exteriores y dos interiores llegan al forjado, con",
    "5,00 m of junction along each long edge and 4,00 m along":
        "5,00 m de unión en cada borde largo y 4,00 m en",
    "each short one: perimeter sum 9 m external + 9 m internal":
        "cada borde corto: suma del perímetro 9 m exterior + 9 m interior",
    "13 airborne paths = 1 direct (Dd) + 4 flanking elements × 3 branches (Ff, Df, Fd)":
        "13 vías aéreas = 1 directa (Dd) + 4 elementos de flanco × 3 ramas (Ff, Df, Fd)",
    "5 impact paths = 1 direct + 4 Df: only the floor is excited, so there is no Ff or Fd":
        "5 vías de impacto = 1 directa + 4 Df: solo se excita el forjado, así que no hay Ff ni Fd",
    "Resilient layers in section: floating floor, mounts, wall lining":
        "Capas elásticas en sección: suelo flotante, apoyos y trasdosado",
    "(a) floating floor": "(a) suelo flotante",
    "220 mm structural slab": "forjado estructural de 220 mm",
    "35 mm screed, 73,5 kg/m²": "solera de 35 mm, 73,5 kg/m²",
    "edge strip, both sides": "banda perimetral, en ambos lados",
    "any rigid bridge here": "cualquier puente rígido aquí",
    "short-circuits the spring": "cortocircuita el muelle",
    "$s′$ = 8 MN/m³  →  $f_0$ = 52,8 Hz": "$s′$ = 8 MN/m³  →  $f_0$ = 52,8 Hz",
    "$ΔL = 30 lg(f/f_0)$ or $40 lg(f/f_0)$":
        "$ΔL = 30 lg(f/f_0)$ o $40 lg(f/f_0)$",
    "(ISO 12354-2 C.1 / C.3)": "(ISO 12354-2 C.1 / C.3)",
    "(b) discrete mounts": "(b) apoyos discretos",
    "structural slab": "forjado estructural",
    "50 mm surface, 115 kg/m²": "capa de paso de 50 mm, 115 kg/m²",
    "reverberant bending field": "campo reverberante de flexión",
    "4 mounts per m² of 2 MN/m": "4 apoyos por m² de 2 MN/m",
    "30 dB per decade, not 40": "30 dB por década, no 40",
    "(Vér's two-subsystem SEA model)": "(modelo SEA de dos subsistemas de Vér)",
    "(c) wall lining: two fixings": "(c) trasdosado: dos fijaciones",
    "masonry": "fábrica",
    "adhesive dabs": "pelladas",
    "studs + cavity": "montantes + cámara",
    "(D.1)  $f_0$ = 542 Hz": "(D.1)  $f_0$ = 542 Hz",
    "(D.2)  $f_0$ = 70,8 Hz": "(D.2)  $f_0$ = 70,8 Hz",
    "the same board, two fixings: nearly 23 dB between them, and no formula here can see which one was built":
        "la misma placa, dos fijaciones: casi 23 dB entre ellas, y ninguna fórmula de aquí ve cuál se construyó",
    "$s′$ is the EN 29052-1 value measured WITHOUT pre-load, and the series law (C.6) holds only for an uncut layer":
        "$s′$ es el valor EN 29052-1 medido SIN precarga, y la ley en serie (C.6) solo vale si la capa no está cortada",
    # --- B9 reconstruction of B10b's decay-range plate ---
    "The decay-range budget of one band: INR, truncation and the evaluation windows (ISO 3382)":
        "El presupuesto de rango de caída de una banda: INR, truncamiento y ventanas de evaluación (ISO 3382)",
    # The plate's current registry title; "balance" and "rango de
    # decaimiento" as the embedding page words them.
    "The decay-range budget of one band (ISO 3382)":
        "El balance del rango de decaimiento de una banda (ISO 3382)",
    "Level [dB]": "Nivel [dB]",
    "peak": "pico",
    "background noise": "ruido de fondo",
    "integration truncated here ($t_1$)": "integración truncada aquí ($t_1$)",
    "tail compensated as": "cola compensada como",
    "an exponential decay (C)": "una caída exponencial (C)",
    "INR = 55 dB": "INR = 55 dB",
    "Evaluation windows": "Ventanas de evaluación",
    "hatched: the 15 dB margin ISO 3382-1 asks for beyond each window — EDT needs 25 dB, $T_{20}$ 35 dB, $T_{30}$ 45 dB":
        "rayado: el margen de 15 dB que exige la ISO 3382-1 más allá de cada ventana — EDT necesita 25 dB, $T_{20}$ 35 dB y $T_{30}$ 45 dB",
    "the library flags at 46 dB and 54 dB instead, where the fit's positive bias crosses 5 %":
        "la biblioteca avisa en 46 dB y 54 dB, donde el sesgo positivo del ajuste cruza el 5 %",
    "short of range? $T_{20}$ instead of $T_{30}$ -> a longer sweep or more averages -> EDT; never a fit into the noise":
        "¿falta rango? $T_{20}$ en vez de $T_{30}$ -> un barrido más largo o más promedios -> EDT; nunca un ajuste metido en el ruido",
    # --- ISO 3741 reverberation test room (devices/emission) ---
    "ISO 3741 reverberation test room": "Sala reverberante de ensayo ISO 3741",
    "Direct method (Eq. 20)": "Método directo (Ec. 20)",
    "Comparison method (Eq. 21)": "Método de comparación (Ec. 21)",
    "or one continuous traverse": "o un recorrido continuo",
    "Source under test": "Fuente bajo ensayo",
    "source under test, left in place":
        "la fuente bajo ensayo permanece en la sala",
    "Reference sound": "Fuente sonora de",
    "source (ISO 6926)": "referencia (ISO 6926)",
    "$d_{min} = D_1 √(V / T_{60})$ ,  $D_1$ = 0,08  (0,16 recommended below 5 kHz)":
        "$d_{min} = D_1 √(V / T_{60})$ ,  $D_1$ = 0,08  (0,16 recomendado por debajo de 5 kHz)",
    "$V$ = 200 m³ · $T_{60}$ = 2,0 s  →  $d_{min}$ = 0,8 m,  or 1,6 m at the recommended $D_1$":
        "$V$ = 200 m³ · $T_{60}$ = 2,0 s  →  $d_{min}$ = 0,8 m,  o 1,6 m con el $D_1$ recomendado",
    "six positions: > 1,0 m from every room surface · $> d_{min}$ from the source · spacing $≥ λ/2$ (1,7 m at 100 Hz)":
        "seis posiciones: > 1,0 m de toda superficie de la sala · $> d_{min}$ de la fuente · separación $≥ λ/2$ (1,7 m a 100 Hz)",
    "traverse instead: $≥ d_{min}$ from the source · ≥ 1,0 m from any surface · ≥ 0,5 m from a diffuser":
        "recorrido continuo: $≥ d_{min}$ de la fuente · ≥ 1,0 m de cualquier superficie · ≥ 0,5 m de un difusor",
    "· not within 10° of a room surface · length $≥ 3λ$ or 10,3 m, whichever is smaller":
        "· fuera de todo plano a menos de 10° de una superficie · longitud $≥ 3λ$ o 10,3 m, la menor",
    "comparison method: the same six positions, and Eq. 21 without $A$, $V$, $S$, Waterhouse or $C_1$":
        "método de comparación: las mismas seis posiciones, y la Ec. 21 sin $A$, $V$, $S$, Waterhouse ni $C_1$",
    "averaging ≥ 30 s to 160 Hz, ≥ 10 s from 200 Hz · background at those positions, before or after":
        "promediado ≥ 30 s hasta 160 Hz, ≥ 10 s desde 200 Hz · fondo en esas posiciones, antes o después",
    "hard walls, $α < 0,06$ within one wavelength of the source · $T_{60}$ per ISO 3382-2, first 10 dB or 15 dB only":
        "paredes duras, $α < 0,06$ a una longitud de onda de la fuente · $T_{60}$ (ISO 3382-2) solo de los 10 o 15 dB primeros",
    # --- ISO 3744 parallelepiped measurement surface ---
    "ISO 3744 parallelepiped measurement surface":
        "Superficie de medición paralelepipédica ISO 3744",
    "Top view": "Vista en planta",
    "reference box": "paralelepípedo de referencia",
    "normal to the face": "normal a la cara",
    "at a corner: aimed at $O$": "en una esquina: apuntando a $O$",
    "$S = 4(a·b + b·c + c·a)$ = 36,3 m²   for $l_1 × l_2 × l_3$ = 1,4 × 0,9 × 1,1 m at $d$ = 1 m":
        "$S = 4(a·b + b·c + c·a)$ = 36,3 m²   para $l_1 × l_2 × l_3$ = 1,4 × 0,9 × 1,1 m con $d$ = 1 m",
    "each of the five planes is split on its own into equal partial areas of side $≤ 3d$ (clause C.1)":
        "cada uno de los cinco planos se divide por separado en áreas parciales iguales de lado $≤ 3d$ (apartado C.1)",
    "key positions: the centre of every partial area, plus its corners except those in the reflecting plane":
        "posiciones fundamentales: el centro de cada área parcial y sus esquinas, salvo las del plano reflectante",
    "nine is the minimum, for one partial area per plane; here $2a > 3d$, so the long faces split and the array grows":
        "nueve es el mínimo, con un área parcial por plano; aquí $2a > 3d$, así que las caras largas se dividen y la malla crece",
    "the survey method (ISO 3746) keeps only the partial-area centres":
        "el método de control (ISO 3746) conserva solo los centros de las áreas parciales",
    # --- ISO/TS 7849-2 radiation factor ---
    "Determining the radiation factor (ISO/TS 7849-2)":
        "Determinación del factor de radiación (ISO/TS 7849-2)",
    "$⟨v_j^2⟩$ on the casing": "$⟨v_j^2⟩$ en la carcasa",
    "ISO 9614 measurement surface": "Superficie de medición ISO 9614",
    "Determining $ε_j$ (Part 2)": "Determinación de $ε_j$ (Parte 2)",
    "$P_j$ : ISO 9614 band power": "$P_j$ : potencia de banda ISO 9614",
    "$⟨v_j^2⟩$ : surface-averaged": "$⟨v_j^2⟩$ : velocidad normal promediada",
    "normal velocity, same bands": "en la superficie, mismas bandas",
    "one machine, one run:": "una máquina, un ensayo:",
    "the same operating mode,": "el mismo modo de funcionamiento,",
    "the same mounting, the same bands.": "el mismo montaje, las mismas bandas.",
    "$ε_j$ is a property of the structure":
        "$ε_j$ es una propiedad de la estructura",
    "and its excitation together.": "y de su excitación conjuntamente.",
    "determined once, then the velocity survey alone serves the rest of the family":
        "se determina una vez y luego el muestreo de velocidad basta para el resto de la familia",
    "mean segment-to-source distance ≥ 200 mm (ISO 9614-2, clause 8.2)":
        "distancia media del segmento a la fuente ≥ 200 mm (ISO 9614-2, apartado 8.2)",
    "Part 1 skips this measurement and sets $ε = 1$, which is why it returns an upper limit":
        "La Parte 1 omite esta medición y toma $ε = 1$, por eso devuelve un límite superior",
    # --- Residual-intensity test and the before-use probe check ---
    "Residual-intensity test and the before-use probe check":
        "Ensayo de intensidad residual y comprobación previa de la sonda",
    "1 · Residual-intensity test": "1 · Ensayo de intensidad residual",
    "2 · Pressure check": "2 · Comprobación de presión",
    "3 · Probe reversal, in situ": "3 · Inversión de la sonda, in situ",
    "pink or white noise, 45 Hz to 7,1 kHz":
        "ruido rosa o blanco, de 45 Hz a 7,1 kHz",
    # The indicator definition reads the same in Spanish (shared symbols).
    "$δ_{pI0} = L_p − L_{I0}$": "$δ_{pI0} = L_p − L_{I0}$",
    "both capsules within ± 0,1 dB": "ambas cápsulas dentro de ± 0,1 dB",
    "IEC 60942 calibrator, class 0 or 1": "calibrador IEC 60942, clase 0 o 1",
    "on each microphone in turn": "en cada micrófono por separado",
    "adjust to ± 0,1 dB": "ajustar a ± 0,1 dB",
    "in both channels": "en ambos canales",
    "energy leaving the source": "energía que sale de la fuente",
    "measurement surface": "superficie de medición",
    "acoustic centre held in place": "el centro acústico no se mueve",
    "accepted when the two readings have opposite signs and differ by less than 1,5 dB":
        "se acepta cuando las dos lecturas tienen signos opuestos y difieren en menos de 1,5 dB",
    "in the band of maximum level (ISO 9614-2, clause 6.2.2)":
        "en la banda de nivel máximo (ISO 9614-2, apartado 6.2.2)",
    "same signs → the two channels are swapped, or one is inverted":
        "mismos signos → los dos canales están intercambiados, o uno está invertido",
    "more than 1,5 dB apart → the probe disturbs its own field, or the channels are not matched":
        "más de 1,5 dB de diferencia → la sonda perturba su propio campo, o los canales no están emparejados",
    "$δ_{pI0}$ belongs to the probe, its spacer and the analyser together — not to the microphones":
        "$δ_{pI0}$ pertenece al conjunto sonda, separador y analizador, no a los micrófonos",
    # Long's Table 14.9 installation (devices/noise-control/duct-path).
    "Long's Table 14.9 installation: every row of the sheet as a place":
        "La instalación de la Tabla 14.9 de Long: cada fila de la hoja, en su sitio",
    "Plant room": "Sala de máquinas",
    "Office, 20 × 20 × 8 ft": "Oficina, 20 × 20 × 8 ft",
    "drywall and carpet, NC 30": "cartón-yeso y moqueta, NC 30",
    "1 · Fan, 5000 cfm, 2 in w.g.": "1 · Ventilador, 5000 cfm, 2 in c.a.",
    "same $L_W$ into both runs": "el mismo $L_W$ en ambos ramales",
    "elbow": "codo",
    "silencer": "silenc.",
    "lined duct": "revestido",
    "branch": "deriv.",
    "flex duct": "flexible",
    "elbow, lined": "codo rev.",
    "plenum": "plenum",
    "8 · diffuser 24 × 24 in": "8 · difusor 24 × 24 in",
    "6 · grille": "6 · rejilla",
    "$r$ = 1.83 m": "$r$ = 1,83 m",
    "$Q = 2$, flush in the ceiling": "$Q = 2$, enrasado en el techo",
    "blue: the supply path — red: the return path — each box is one row of the sheet, stamped with its code":
        "azul: ramal de impulsión; rojo: ramal de retorno; cada caja es una fila de la hoja con su código",
    "attenuates only: 4, 5, 6 (supply) and 4, 5 (return) — attenuates and regenerates: 2, 3 — self-noise only: 8 and the return grille":
        "solo atenúan: 4, 5, 6 (impulsión) y 4, 5 (retorno); atenúan y regeneran: 2, 3; solo ruido propio: 8 y la rejilla",
    "the return wins above 1 kHz: its silencer floors the room near 25 dB, and no amount of supply attenuation moves that":
        "por encima de 1 kHz manda el retorno: su silenciador deja un suelo de 25 dB que la impulsión no puede bajar",
    # ISO 7235 silencer measurement (devices/noise-control/silencers).
    "How a silencer is measured: the ISO 7235 substitution method":
        "Cómo se mide un silenciador: el método de sustitución de la ISO 7235",
    "Series I — test object installed": "Serie I: con el objeto de ensayo",
    "Series II — substitution duct": "Serie II: con el conducto de sustitución",
    "test object": "objeto de ensayo",
    "substitution duct": "conducto de sustitución",
    "sealed, lined loudspeaker box": "caja de altavoces estanca",
    "modal filter": "filtro modal",
    "transition": "transición",
    "test duct, anechoic termination": "conducto de ensayo, terminación anecoica",
    "three positions on a line inclined to the axis, at mid-length":
        "tres posiciones en una línea inclinada respecto al eje, a media longitud",
    "$r ≤ 0.3$ planes": "planos $r ≤ 0,3$",
    "$D_i = L_{pI} − L_{pII}$, one third octave at a time":
        "$D_i = L_{pI} − L_{pII}$, tercio de octava a tercio de octava",
    "modal filter: ≥ 3 dB on the fundamental at the low-frequency end,":
        "filtro modal: ≥ 3 dB sobre el modo fundamental en el extremo grave,",
    "≥ 5 dB above the cut-on of higher-order modes (5.2.2.3)":
        "≥ 5 dB por encima del corte de los modos superiores (5.2.2.3)",
    "substitution duct: the empty housing where possible, otherwise":
        "conducto de sustitución: la carcasa vacía si es posible; si no,",
    "matched within 5 % in every linear dimension (5.2.3)":
        "ajustado al 5 % en cada dimensión lineal (5.2.3)",
    "reflection coefficient $r ≤ 0.3$ at the source and receiving":
        "coeficiente de reflexión $r ≤ 0,3$ en los planos de calificación",
    "qualification planes (5.2.2.5, 5.2.4.3)":
        "emisor y receptor (5.2.2.5, 5.2.4.3)",
    "signal ≥ 6 dB and preferably ≥ 10 dB above the background":
        "señal ≥ 6 dB y preferiblemente ≥ 10 dB sobre el ruido de fondo",
    "(5.2.2.2); IEC 61260 third octaves, class 1 chain (5.2.4.6)":
        "(5.2.2.2); tercios de octava IEC 61260, cadena clase 1 (5.2.4.6)",
    "the reported figure is an insertion loss against a substitution duct, not a transmission loss":
        "lo que se declara es una pérdida por inserción frente a un conducto de sustitución, no por transmisión",
    "and the facility's own limiting insertion loss — flanking along the duct walls — caps what it can report at all":
        "y la pérdida por inserción límite de la instalación, por flancos en las paredes del conducto, acota lo que puede declarar",
    # Room-to-room chain in section (devices/noise-control/room-to-room).
    "Plant room to operator room: every symbol of the balance, in section":
        "De la sala de máquinas a la de control: el balance en sección",
    "Plant room 8 × 10 × 3 m": "Sala de máquinas 8 × 10 × 3 m",
    "bare floor, absorbent ceiling": "suelo desnudo, techo absorbente",
    "Operator room 5 × 5 × 3 m": "Sala de control 5 × 5 × 3 m",
    "carpet, same ceiling": "moqueta, mismo techo",
    "flanking over the ceiling void: an allowance, not a model":
        "flancos por el plenum del techo: una reserva, no un modelo",
    "8 m": "8 m",
    "5 m": "5 m",
    "3 m": "3 m",
    "source side — blower on the floor at a wall mid-point: $L_W$ = 105 dB at 125 Hz, $Q = 4$ adds 6.0 dB, $L_{p1}$ = 107.0 dB":
        "lado emisor: soplante en el suelo en el centro de una pared, $L_W$ = 105 dB a 125 Hz, $Q = 4$ suma 6,0 dB, $L_{p1}$ = 107,0 dB",
    "partition — 5 m × 3 m = 15 m², TL = 39 dB at 125 Hz; the $τ S_w$ returned to the source room is off by default":
        "separación: 5 m × 3 m = 15 m², TL = 39 dB a 125 Hz; el $τ S_w$ devuelto al recinto emisor está desactivado por defecto",
    "receiving side — $S_{2}α_{2}$ = 5.5 m² at 125 Hz rising to 39.2 m² at 4 kHz; $L_{p2}$ = 72.4 dB against 60 dB for NC 45":
        "lado receptor: $S_{2}α_{2}$ = 5,5 m² a 125 Hz y 39,2 m² a 4 kHz; $L_{p2}$ = 72,4 dB frente a los 60 dB de NC 45",
    "$NR = TL − 10 log_{10}[S_w / (S_{2}α_{2} + τ S_w)]$":
        "$NR = TL − 10 log_{10}[S_w / (S_{2}α_{2} + τ S_w)]$",
    "$S_{2}α_{2}$ passes the 15 m² of the wall between 250 and 500 Hz: below it the wall delivers less than its TL":
        "$S_{2}α_{2}$ supera los 15 m² del muro entre 250 y 500 Hz: por debajo, el muro entrega menos que su TL",
    "both levels are reverberant-field spatial averages; the balance says "
    "nothing below 163 Hz (Schroeder, 75 m³)":
        "ambos niveles promedian el campo reverberante en el espacio; el "
        "balance calla bajo 163 Hz (Schroeder, 75 m³)",
    # Machine enclosure in section (devices/noise-control/noise-control).
    "Machine enclosure in section: what IL = R − C really depends on":
        "Encapsulamiento de máquina en sección: de qué depende IL = R − C",
    "$S_E$ = 24 m² of exposed shell — a 3.0 × 2.0 × 1.8 m box, five faces":
        "$S_E$ = 24 m² de envolvente expuesta: una caja de 3,0 × 2,0 × 1,8 m, cinco caras",
    "lined cooling outlet": "salida de refrigeración revestida",
    "a short lined duct, never a bare hole":
        "un conducto corto revestido, nunca un agujero desnudo",
    "cable and pipe entry, sealed sleeve":
        "paso de cables y tuberías con manguito estanco",
    "$S_i$ = 30 m², $ᾱ_i = 0.30$  →  $R_i$ = 12.9 m²":
        "$S_i$ = 30 m², $ᾱ_i = 0,30$  →  $R_i$ = 12,9 m²",
    "wall build-up: sheet-steel mass, absorbent lining, perforated facing":
        "cerramiento: chapa como masa, absorbente y chapa perforada",
    "machine on vibration isolators": "máquina sobre aisladores de vibración",
    "no rigid contact with the shell or with its slab":
        "sin contacto rígido con la envolvente ni con su solera",
    "access door": "puerta de acceso",
    "1.28 m², $R$ = 15 dB": "1,28 m², $R$ = 15 dB",
    "compression seal": "junta de compresión",
    "unsealed gap at the foot": "rendija sin sellar al pie",
    "0.24 m² = 1 % of $S_E$": "0,24 m² = 1 % de $S_E$",
    "lined cooling inlet": "entrada revestida",
    "cooling air needs a path": "el aire debe poder circular",
    "$IL = R − C$,   $C = 10 log_{10}(0.3 + S_E/R_i)$ = 3.4 dB":
        "$IL = R − C$,   $C = 10 log_{10}(0,3 + S_E/R_i)$ = 3,4 dB",
    "sealed shell (mean $R$ = 32.3 dB): mean IL = 28.9 dB":
        "envolvente sellada ($R$ medio = 32,3 dB): IL medio = 28,9 dB",
    "with the door: 21.4 dB — with the 1 % gap as well: 15.1 dB, against the $10 log_{10}(S_E/S_a)$ = 20 dB cap":
        "con la puerta: 21,4 dB; con la rendija del 1 % además: 15,1 dB, frente al techo $10 log_{10}(S_E/S_a)$ = 20 dB",
    "3.0 m": "3,0 m",
    "1.8 m": "1,8 m",
    "an enclosure delivers its worst element, not its panels":
        "un encapsulamiento entrega su peor elemento, no sus paneles",
    # ECAC Doc 29 flight-path segment geometry (aircraft/airport-noise).
    "Flight-path segment geometry (ECAC Doc 29, Chapter 4)":
        "Geometría del segmento de trayectoria (ECAC Doc 29, capítulo 4)",
    "(a) Observer alongside the segment": "(a) Observador junto al segmento",
    "(b) Observer behind the segment": "(b) Observador detrás del segmento",
    "(c) In the plane normal to the flight path":
        "(c) En el plano normal a la trayectoria",
    "(d) Behind the take-off roll, in plan":
        "(d) Detrás de la carrera de despegue, en planta",
    "$d_p$ = 526 m": "$d_p$ = 526 m",
    "$d_1$ = 568 m": "$d_1$ = 568 m",
    "$d_2$ = 582 m": "$d_2$ = 582 m",
    "$q$ = 214 m": "$q$ = 214 m",
    "$q$ = −300 m": "$q$ = −300 m",
    "$d_s = d_1$ = 600 m": "$d_s = d_1$ = 600 m",
    "0 ≤ q ≤ λ:  ds = dp, and the NPD lookup for an":
        "0 ≤ q ≤ λ:  ds = dp, y la consulta NPD de un",
    "exposure level uses dp (§4.4.1)": "nivel de exposición usa dp (§4.4.1)",
    "q < 0:  the NPD lookup uses ds behind a take-off":
        "q < 0:  la consulta NPD usa ds detrás de una carrera de",
    "ground roll, and dp everywhere else": "despegue, y dp en todos los demás casos",
    "$0 ≤ q ≤ λ$, so $d_s = d_p$ and an exposure level":
        "$0 ≤ q ≤ λ$: $d_s = d_p$ y un nivel de exposición",
    "reads the NPD table at $d_p$ (§4.4.1)":
        "consulta la tabla NPD en $d_p$ (§4.4.1)",
    "$q < 0$, so the exposure level reads the table":
        "$q < 0$: el nivel de exposición consulta la tabla",
    "at $d_s$ behind a take-off roll and at $d_p$ elsewhere":
        "en $d_s$ tras un recorrido de despegue, si no en $d_p$",
    # Doc 29 symbol (distance from start of roll), identical in Spanish.
    "$d_{SOR}$": "$d_{SOR}$",
    "$β$ elevation of the path over the ground line · $ε$ bank, positive "
    "with the starboard wing up · $φ = β + ε$ to":
        "$β$ elevación del rayo sobre la línea de tierra · $ε$ alabeo, "
        "positivo con el ala de estribor arriba · $φ = β + ε$ a",
    "starboard and $β − ε$ to port · $ψ = arccos(q/d_{SOR})$, 90° abeam to "
    "180° astern, the jet lobe peaking near 120°":
        "estribor y $β − ε$ a babor · $ψ = arccos(q/d_{SOR})$, de 90° al "
        "través a 180° a popa, lóbulo de chorro máximo a 120°",
    "NPD lookup: $d_p$ for exposure levels, $d_s$ for maximum levels, "
    "floored at 30 m; $Δ_{SOR}$ scaled by 762 m/$d_{SOR}$ beyond 762 m":
        "consulta NPD: $d_p$ para exposición, $d_s$ para máximos, cota "
        "inferior 30 m; $Δ_{SOR}$ escalado por 762 m/$d_{SOR}$ desde 762 m",
    "receiver, 1.2 m": "receptor, 1,2 m",
    "wing plane": "plano alar",
    "β elevation of the path over the ground line · ε bank, positive with the starboard wing up":
        "β elevación del rayo sobre la línea de tierra · ε alabeo, positivo con el ala de estribor arriba",
    "φ = β + ε to starboard, β − ε to port; Λ(β, ℓ) uses β and ℓ, ΔI(φ) uses φ":
        "φ = β + ε a estribor y β − ε a babor; Λ(β, ℓ) usa β y ℓ, ΔI(φ) usa φ",
    "0° nose": "0° morro",
    "ψ = arccos(q/dSOR), 90° abeam to 180° astern; the jet lobe peaks near 120°":
        "ψ = arccos(q/dSOR), de 90° por el través a 180° por la popa; el lóbulo del reactor culmina cerca de 120°",
    "ΔSOR is scaled by dSOR,0/dSOR beyond dSOR,0 = 762 m (2 500 ft)":
        "ΔSOR se escala por dSOR,0/dSOR más allá de dSOR,0 = 762 m (2 500 ft)",
    # ICAO Annex 16 certification measurement station (aircraft/aircraft-noise).
    "A noise certification measurement station (ICAO Annex 16, App. 2)":
        "Estación de medición para certificación acústica (OACI Anexo 16, ap. 2)",
    "Site": "Emplazamiento",
    "80° half-angle about the vertical": "semiángulo de 80° en torno a la vertical",
    "inside the cone: site rejected": "dentro del cono: emplazamiento rechazado",
    "tracking, independent of the cockpit":
        "seguimiento, independiente de la cabina",
    "met mast, within 2 000 m": "mástil meteorológico, ≤ 2 000 m",
    "The microphone, at 60 px per metre": "El micrófono, a 60 px por metro",
    "arriving ray": "rayo que llega",
    "windscreen: insertion loss": "pantalla antiviento: pérdida por inserción",
    "within ±1.5 dB, and corrected for": "dentro de ±1,5 dB, y se corrige",
    "sensing element 1.2 m above local ground":
        "elemento sensor a 1,2 m sobre el suelo local",
    "Plan: the capsule axis": "Planta: el eje de la cápsula",
    "the capsule axis is perpendicular to the plane of the":
        "el eje de la cápsula es perpendicular al plano de la",
    "flight path, so every ray arrives at 90°, grazing":
        "trayectoria de vuelo: todo rayo llega a 90°, rasante",
    "24 one-third-octave bands, 50 Hz to 10 kHz":
        "24 bandas de tercio de octava, de 50 Hz a 10 kHz",
    "Plan: capsule orientation": "Planta: orientación de la cápsula",
    "capsule axis ⟂ the flight-path plane:":
        "eje de la cápsula ⟂ al plano de la trayectoria:",
    "every ray arrives at 90°, grazing": "todo rayo llega a 90°, incidencia rasante",
    "reference flight path": "trayectoria de referencia",
    "slant path": "recorrido oblicuo",
    "80° half-angle: no obstruction inside":
        "semiángulo de 80°: sin obstrucciones dentro",
    "inside: site rejected": "dentro: emplazamiento rechazado",
    "windscreen, ±1.5 dB": "pantalla antiviento, ±1,5 dB",
    "tracking": "seguimiento",
    "24 bands, 50 Hz-10 kHz": "24 bandas, 50 Hz-10 kHz",
    "one sample every 500 ms ± 5 ms": "una muestra cada 500 ms ± 5 ms",
    "$T$, RH": "$T$, HR",
    "met mast, within 2 000 m of the station":
        "torre meteorológica, a menos de 2 000 m de la estación",
    "Test window (aeroplanes): no precipitation; −10 to 35 °C and 20 to 95 % RH over the path above 10 m;":
        "Ventana de ensayo (aviones): sin precipitación; de −10 a 35 °C y 20 "
        "a 95 % HR en el recorrido sobre 10 m;",
    "8 kHz attenuation ≤ 12 dB/100 m; wind ≤ 6.2 m/s average and 7.7 m/s peak, crosswind ≤ 3.6 and 5.1 m/s":
        "atenuación a 8 kHz ≤ 12 dB/100 m; viento ≤ 6,2 m/s medio y 7,7 m/s "
        "máximo, cruzado ≤ 3,6 y 5,1 m/s",
    "Helicopters: average wind ≤ 5.1 m/s and crosswind ≤ 2.6 m/s, temperature "
    "and humidity limits at 10 m only":
        "Helicópteros: viento medio ≤ 5,1 m/s y cruzado ≤ 2,6 m/s; "
        "temperatura y humedad solo a 10 m",
    "At least six valid runs per measurement point, with a 90 % confidence limit not exceeding ±1.5 EPNdB":
        "Al menos seis pasadas válidas por punto, con un límite de confianza "
        "del 90 % que no exceda ±1,5 EPNdB",
    # ECAC Doc 32 noise hemisphere (aircraft/rotorcraft-noise).
    "The rotorcraft noise hemisphere and its angles (ECAC Doc 32)":
        "El hemisferio de ruido del giroavión y sus ángulos (ECAC Doc 32)",
    "measured polar band $θ_{t1}$ … $θ_{t2}$,":
        "banda polar medida $θ_{t1}$ … $θ_{t2}$,",
    "the two 10 dB-down instants": "los dos instantes de 10 dB por debajo",
    "measured lateral band $−60° ≤ φ ≤ 60°$,":
        "banda lateral medida $−60° ≤ φ ≤ 60°$,",
    "outside it the bins are gap-filled": "fuera de ella los bins se rellenan",
    "$ΔL_s = −20 lg(r/60)$ · $ΔL_a = −α(f)(r − 60)$ · $ΔL_g$ from the "
    "two-ray model over the ground":
        "$ΔL_s = −20 lg(r/60)$ · $ΔL_a = −α(f)(r − 60)$ · $ΔL_g$ del modelo "
        "de dos rayos sobre el suelo",
    "(a) Polar angle $θ$, centre plane $φ$ = 0":
        "(a) Ángulo polar $θ$, plano central $φ$ = 0",
    "(b) Azimuth $φ$, seen from astern": "(b) Acimut $φ$, visto desde popa",
    "(c) The 60 m sphere on a flight path":
        "(c) La esfera de 60 m sobre una trayectoria",
    "$θ$ = 0 nose": "$θ$ = 0 morro",
    "$θ$ = 90 beneath": "$θ$ = 90 debajo",
    "$θ$ = 180 tail": "$θ$ = 180 cola",
    "measured band": "banda medida",
    "θt1 … θt2, the 10 dB-down instants": "θt1 … θt2, los instantes a 10 dB del máximo",
    "$r_h$ = 60 m": "$r_h$ = 60 m",
    "$φ$ = +90 starboard": "$φ$ = +90 estribor",
    "$φ$ = 0 beneath": "$φ$ = 0 debajo",
    "$φ$ = −90 port": "$φ$ = −90 babor",
    "outside ±60° the bins are gap-filled from the nearest":
        "fuera de ±60° las celdas se rellenan desde la más próxima",
    "slant range $r$": "distancia oblicua $r$",
    "banked by $Φ$ in turns": "alabeado $Φ$ en los virajes",
    "levels[a, p, f] in dB at 60 m under the ICAO reference atmosphere (25 °C, 70 % RH, 101.325 kPa):":
        "levels[a, p, f] en dB a 60 m bajo la atmósfera de referencia OACI (25 °C, 70 % HR, 101,325 kPa):",
    "19 azimuths × 19 polar angles at 10°, 31 one-third-octave bands from 10 Hz to 10 kHz":
        "19 acimutes × 19 ángulos polares cada 10°, 31 bandas de tercio de octava de 10 Hz a 10 kHz",
    "unmeasured bins are NaN, never 0 dB; mirrored-rotor class members read the same data at $−φ$":
        "las celdas no medidas son NaN, nunca 0 dB; los miembros de clase con rotor espejado leen los mismos datos en $−φ$",
    # Sound-field audiometry (ISO 389-7 / ISO 8253-2) — B15b
    "Sound-field audiometry and the ISO 389-7 reference zero":
        "Audiometría en campo sonoro y el cero de referencia de ISO 389-7",
    "A · Free field": "A · Campo libre",
    "pure tone · frontal · binaural": "tono puro · frontal · biaural",
    "B · Diffuse field": "B · Campo difuso",
    "third-octave noise band · binaural":
        "banda de ruido de tercio de octava · biaural",
    "C · Earphone: not this standard": "C · Auricular: no es esta norma",
    "supra-aural or insert · monaural": "supraaural o de inserción · monoaural",
    "level measured here,": "el nivel se mide aquí,",
    "subject and chair absent": "sin el sujeto ni la silla",
    "on the reference axis, 0° azimuth": "sobre el eje de referencia, 0° de acimut",
    "and elevation, ≥ 1 m  (5.2 a)": "y de elevación, ≥ 1 m  (5.2 a)",
    "± 0.15 m: within ± 1 dB to 4 kHz  (5.2 b)":
        "± 0,15 m: dentro de ± 1 dB hasta 4 kHz  (5.2 b)",
    "the same reference point,": "el mismo punto de referencia,",
    "the same absent subject": "el mismo sujeto ausente",
    "several loudspeakers, non-coherent feeds":
        "varios altavoces, con señales no coherentes",
    "≥ 500 Hz: loudest and quietest directions":
        "≥ 500 Hz: las direcciones de nivel máximo y mínimo",
    "within 5 dB  (Table 1)": "dentro de 5 dB  (tabla 1)",
    "the level lives in the coupler,": "el nivel vive en el acoplador,",
    "not at a point in a room": "no en un punto de una sala",
    "coupler /": "acoplador /",
    "ear simulator": "simulador de oído",
    "0 dB HL here is the RETSPL of the earphone":
        "0 dB HL aquí es el RETSPL del auricular",
    "fitted (ISO 389-1 / -2 / -8), referred to a":
        "colocado (ISO 389-1 / -2 / -8), referido a un",
    "coupler — never an ISO 389-7 value":
        "acoplador, nunca un valor de ISO 389-7",
    "Reference point: the midpoint of the line joining the listener's ear-canal openings":
        "Punto de referencia: el punto medio entre las entradas de los conductos auditivos",
    "the listener in the listening position; in A and B the level is measured there with the subject and chair absent":
        "el oyente en la posición de escucha; en A y B el nivel se mide ahí sin el sujeto ni la silla",
    # ISO 9612 Clause 12.4 sound level meter geometry — B15b
    "Sound level meter at a workstation (ISO 9612, Clause 12.4)":
        "Sonómetro en un puesto de trabajo (ISO 9612, capítulo 12.4)",
    "Worker absent": "Sin el trabajador",
    "the preferred Clause 12.4 placement":
        "la colocación preferente del capítulo 12.4",
    "Worker present": "Con el trabajador",
    "hand-held meter, most-exposed ear":
        "sonómetro de mano, oído más expuesto",
    "machine": "máquina",
    "axis ∥ line of sight": "eje ∥ línea de visión",
    "1.55 m": "1,55 m",
    "or sweep in plan:": "o barrido en planta:",
    "at constant speed": "a velocidad constante",
    "capsule at the head position, on the eye line":
        "cápsula en la posición de la cabeza, a la altura de los ojos",
    "standing 1.55 m ± 0.075 m; seated 0.80 m ± 0.05 m":
        "de pie 1,55 m ± 0,075 m; sentado 0,80 m ± 0,05 m",
    "above the middle of the seat plane":
        "sobre el centro del plano del asiento",
    "0.1 m to 0.4 m": "de 0,1 m a 0,4 m",
    "60 mm": "60 mm",
    "windscreen": "pantalla antiviento",
    "held 0.1 m to 0.4 m from the ear-canal entrance":
        "sostenido de 0,1 m a 0,4 m de la entrada del conducto auditivo",
    "on the most exposed side, windscreen ≥ 60 mm (13.3)":
        "en el lado más expuesto, pantalla antiviento ≥ 60 mm (13.3)",
    "beyond 0.4 m, use the worn instrument (12.3)":
        "más allá de 0,4 m, usar el instrumento portado (12.3)",
    "A fixed microphone position under-reads a hand-held tool close to the ear (13.1)":
        "Una posición fija de micrófono subestima una herramienta de mano próxima al oído (13.1)",
    "that is exactly when the worn personal exposure meter of Clause 12.3 is the right instrument":
        "es justo cuando el exposímetro personal portado del capítulo 12.3 es el instrumento adecuado",
    # IEC 60268-16 clause 7 STI setup — B15b
    "Setting up an STI measurement (IEC 60268-16, clause 7)":
        "Montaje de una medida de STI (IEC 60268-16, capítulo 7)",
    "A · Unamplified talker": "A · Hablante sin amplificar",
    "acoustical input, clause 7.2": "entrada acústica, capítulo 7.2",
    "B · Sound system": "B · Sistema de megafonía",
    "electrical input, clause 7.4": "entrada eléctrica, capítulo 7.4",
    "artificial mouth": "boca artificial",
    "(ITU-T P.51 directivity)": "(directividad ITU-T P.51)",
    "1 m": "1 m",
    "2 m": "2 m",
    "60 dB(A) here, or the": "60 dB(A) aquí, o el",
    "Annex J operational level": "nivel de operación del anexo J",
    "source response flat within ± 1 dB in a free field (7.2 b)":
        "respuesta de la fuente plana ± 1 dB en campo libre (7.2 b)",
    "receiver: omnidirectional, diffuse-field, calibrated (7.3)":
        "receptor: omnidireccional, de campo difuso, calibrado (7.3)",
    "ambient noise measured at the same point, source off":
        "ruido de fondo medido en el mismo punto, con la fuente apagada",
    "ceiling loudspeaker line": "línea de altavoces de techo",
    "amp": "amplif.",
    "test signal in,": "entrada de la señal",
    "at the Annex J level": "de ensayo, nivel del anexo J",
    "zone 1": "zona 1",
    "zone 2": "zona 2",
    "system off:": "sistema apagado:",
    "ambient": "ruido de fondo",
    "injected near the normal input, so the whole chain is in (7.4)":
        "inyectada junto a la entrada normal, para incluir toda la cadena (7.4)",
    "one position per coverage zone, at listening height":
        "una posición por zona de cobertura, a la altura de escucha",
    "spread over the served area, worst corners included":
        "repartidas por el área servida, incluidos los peores rincones",
    "The rating of the space is the mean of the positions minus one standard deviation (7.6.4)":
        "La calificación del espacio es la media de las posiciones menos una desviación típica (7.6.4)",
    "a plain mean over the positions overstates coverage; better still, plot the whole distribution":
        "la media simple de las posiciones sobrestima la cobertura; mejor aún, represente toda la distribución",
    # STOI/ESTOI capture bench — B15b
    "Capturing a STOI pair through a real device":
        "Captura de un par para STOI a través de un dispositivo real",
    "clean": "señal",
    "speech file": "de voz limpia",
    "reference path": "camino de referencia",
    "the original file, never a re-recording":
        "el archivo original, nunca una regrabación",
    "playback": "reproducción",
    "amp + loudspeaker": "amplificador + altavoz",
    "test box": "caja de ensayo",
    "device under test": "equipo bajo ensayo",
    "hearing aid on an artificial ear,": "audífono sobre oído artificial,",
    "or a headset on a torso simulator":
        "o auriculares sobre un simulador de torso",
    "capture": "captura",
    "mic + preamp": "micrófono + preamplificador",
    "align and trim": "alinear y recortar",
    "cross-correlate the envelopes": "correlacionar las envolventes",
    "equal length · one clock": "misma longitud · un solo reloj",
    "STOI /": "STOI /",
    "ESTOI": "ESTOI",
    "run it once with the device bypassed":
        "hágalo una vez con el equipo puenteado",
    "the loudspeaker, the box noise and the microphone are scored as degradation too":
        "el altavoz, el ruido de la caja y el micrófono también se puntúan como degradación",
    "Play at the device's operating level, and repeat the capture":
        "Reproduzca al nivel de operación del equipo y repita la captura",
    "the index is invariant to level, the device is not; a single capture carries the acoustic path's run-to-run spread":
        "el índice es invariante al nivel, el equipo no; una sola captura arrastra la dispersión del camino acústico",
    # Near-to-far-field capture geometry (simulation/fdtd-simulation).
    "Near-to-far-field capture: contour, clearances and angle convention":
        "Campo cercano a lejano: contorno, holguras y ángulos",
    "far-field origin: the centre of the panel face":
        "origen de campo lejano: el centro de la cara del panel",
    "The far field is evaluated at $r → ∞$, and lives nowhere on the grid:":
        "El campo lejano se evalúa en $r → ∞$ y no vive en ninguna celda:",
    "far_field_from_contour propagates the captured phasors with the free-space Green function":
        "far_field_from_contour propaga los fasores con la función de Green de campo libre",
    "sponge, 60 cells": "esponja, 60 celdas",
    "capture contour": "contorno de captura",
    "plane-wave injection line": "línea de onda plana",
    "Domain 980 × 346 cells at dx = 0.5 mm (490 × 173 mm); at 2 kHz there are 343 cells per wavelength":
        "Dominio de 980 × 346 celdas, dx = 0,5 mm (490 × 173 mm); a 2 kHz, 343 celdas por longitud de onda",
    "Panel: five 70 mm cells, 20 mm slits on a 3 mm backing; the 3.2 mm neck sets dx, not the wavelength":
        "Panel: cinco celdas de 70 mm, ranuras de 20 mm sobre fondo de 3 mm; dx lo fija el cuello de 3,2 mm",
    "Clearances in cells: 40 ahead of the panel, 20 behind and to the sides, 60 to every sponge":
        "Holguras en celdas: 40 por delante del panel, 20 por detrás y a los lados, 60 hasta cada esponja",
    "Sources outside the contour integrate to zero (extinction): the total-field phasors give the scattered field":
        "Las fuentes fuera del contorno integran a cero (extinción): los fasores dan el campo lejano dispersado",
    "Reference run: the same scene with no panel → ContourPhasors.subtract() removes the grid residual":
        "Referencia: la misma escena sin panel → ContourPhasors.subtract() elimina el residuo numérico",
    # Fluid-solid contact at three incidences (simulation/elastic-waves).
    "A fluid-solid contact at three incidences, and where it sits on the grid":
        "Un contacto fluido-sólido en tres incidencias, y dónde cae en la malla",
    "(a) normal incidence": "(a) incidencia normal",
    "(b) oblique: mode conversion": "(b) oblicua: conversión de modo",
    "(c) beyond both critical angles": "(c) más allá de ambos críticos",
    # "incident" and "reflected" are already in this table, above.
    "P only": "solo P",
    "probe on the normal,": "sonda en la normal,",
    "0.12 m up": "0,12 m por encima",
    "shot, 10 m up": "disparo, 10 m arriba",
    "water 1480": "agua 1480",
    "steel 5900 / 3200": "acero 5900 / 3200",
    "water 1500": "agua 1500",
    "sediment 3500 / 2000": "sedimento 3500 / 2000",
    "evanescent both sides": "evanescente a ambos lados",
    "last fluid row": "última fila de fluido",
    "first solid row": "primera fila de sólido",
    "A region painted from row $i$ down puts the contact on the":
        "Una región pintada desde la fila $i$ hacia abajo sitúa el contacto "
        "en",
    "face plane $y = i·dx$: density averaged arithmetically onto":
        "el plano de caras $y = i·dx$: densidad promediada aritméticamente en",
    "the faces, shear modulus harmonically onto the corners":
        "las caras y módulo de cizalla armónicamente en las esquinas",
    "Sponge bands line the outer edges of every panel; no side is both free and absorbing":
        "Bandas de esponja recubren los bordes exteriores; ningún lado es a la vez libre y absorbente",
    "$V = (Z_2−Z_1)/(Z_2+Z_1)$ = 0.938":
        "$V = (Z_2−Z_1)/(Z_2+Z_1)$ = 0,938",
    "no shear is excited, so the steel":
        "no se excita cizalla, así que el acero",
    "acts as a liquid of its $ρ$ and $c_P$":
        "actúa como un líquido de su $ρ$ y $c_P$",
    "critical angles 14.5° and 27.5°": "ángulos críticos 14,5° y 27,5°",
    "between them P is evanescent and": "entre ellos P es evanescente y",
    "the shear wave carries the power": "la onda de cizalla lleva la potencia",
    "over steel the deficit is 0.03 %": "sobre acero el déficit es del 0,03 %",
    "the tail reaches ~7 $λ$ up, so no":
        "la cola llega a ~7 $λ$, así que ningún",
    "time of flight can separate it": "tiempo de vuelo la separa",
    # Immersed-plate transmission scene (simulation/elastic-waves).
    "Immersed-plate transmission: the strip, the probes and the time gate":
        "Transmisión de placa sumergida: tira, sondas y ventana",
    "Three cells wide:": "Tres celdas de ancho:",
    "a 1D problem in a 2D solver, dx = 0.5 mm":
        "un problema 1D en un solver 2D, dx = 0,5 mm",
    "65 µs gate: the incident pulse alone":
        "ventana de 65 µs: solo el pulso incidente",
    "incident, 47 µs": "incidente, 47 µs",
    "reflection, 88 µs": "reflexión, 88 µs",
    "transmitted ring-down;": "cola transmitida;",
    "no echo in the record": "sin eco en el registro",
    "time [µs]": "tiempo [µs]",
    "$I$ from the gated probe A, $T$ from probe B":
        "$I$ de la sonda A enventanada, $T$ de la sonda B",
    "Inside the plate, at half-wave thickness":
        "Dentro de la placa, a espesor de media onda",
    "$f_n = n c_P / (2h)$ = 295 kHz for 10 mm of steel":
        "$f_n = n c_P / (2h)$ = 295 kHz para 10 mm de acero",
    "the plate goes transparent there, three decades above audio":
        "allí la placa se vuelve transparente, tres décadas sobre el audio",
    "1.5 mm plane pulse at 0.25 m": "pulso plano 1,5 mm en 0,25 m",
    "one-way; usable to 340 kHz": "unidireccional; hasta 340 kHz",
    "probe A, 30 mm above": "sonda A, 30 mm por encima",
    "10 mm STEEL at $y$ = 0.35 m": "STEEL de 10 mm en $y$ = 0,35 m",
    "painted into WATER": "pintado dentro de WATER",
    "probe B, 50 mm below": "sonda B, 50 mm por debajo",
    "probe A": "sonda A",
    "probe B": "sonda B",
    # Mathematics that reads the same in both languages.
    "$TL(f) = 20 log_{10} |I(f) / T(f)|$":
        "$TL(f) = 20 log_{10} |I(f) / T(f)|$",
    # Infrasound measurement chain (signals/levels/special-weightings).
    "Measuring infrasound: the chain that must deliver 0,25 Hz":
        "Medir infrasonido: la cadena que debe llegar a 0,25 Hz",
    "Below 20 Hz the wind is louder": "Por debajo de 20 Hz el viento suena",
    "than the source": "más que la fuente",
    "static-pressure equalisation": "orificio de compensación de",
    "vent: a first-order high-pass,": "presión estática: un paso alto de",
    "and the chain's real corner":
        "primer orden, la esquina real de la cadena",
    "hard board on the ground": "tablero rígido sobre el suelo",
    "capsule flush at its centre": "cápsula enrasada en su centro",
    "primary foam screen (solid)": "pantalla primaria de espuma (continua)",
    "secondary in wind (dashed),": "secundaria con viento (discontinua),",
    "with its loss corrected": "con su pérdida corregida",
    "corner << 0,25 Hz": "esquina << 0,25 Hz",
    "Preamplifier": "Preamplificador",
    "Recorder": "Grabador",
    "low-cut switch OFF": "corte de graves: OFF",
    "G weighting + integrator": "ponderación G + integrador",
    "report $L_{pG}$ with the chain corner,":
        "informa $L_{pG}$ con la esquina de la cadena,",
    "the screens and the averaging time":
        "las pantallas y el tiempo de promediado",
    "What the chain lets through": "Lo que deja pasar la cadena",
    "A.2: 0,25 - 315 Hz": "A.2: 0,25 - 315 Hz",
    "lost": "perdido",
    "green: the G weighting": "verde: la ponderación G",
    "dashed: a chain with a 2 Hz vent corner":
        "discontinua: cadena con esquina de 2 Hz",
    "usable band = the overlap of the two":
        "banda utilizable = el solape de ambas",
    "ISO 7196:1995, Annex A": "ISO 7196:1995, anexo A",
    # Capturing a microphone array (signals/filters/multichannel).
    "Capturing an array: one clock, locked gains, a written row map":
        "Capturar un array: un reloj, ganancias fijas, mapa de filas escrito",
    "Four positions, four sensitivities": "Cada posición, su sensibilidad",
    "mV/Pa, one per capsule": "mV/Pa, una por cápsula",
    "one capsule at a time,": "una cápsula cada vez,",
    "gains locked throughout": "con las ganancias fijas",
    "4-channel preamplifier": "preamplificador de 4 canales",
    "audio interface": "interfaz de audio",
    "single sample clock": "un solo reloj",
    "a second interface": "una segunda interfaz",
    "= two clocks, not one array": "= dos relojes, no un array",
    "$x$, the array you analyse": "$x$, el array que analizas",
    "Write down, with the file:": "Anota, junto al archivo:",
    "one clock  |  locked gains": "un reloj  |  ganancias fijas",
    "the row-to-position map": "el mapa fila-posición",
    "A swapped pair gives perfectly valid levels attributed to the wrong positions,":
        "Intercambiar dos canales da niveles válidos atribuidos a posiciones equivocadas,",
    "and no later check can detect it":
        "y ninguna comprobación posterior lo detecta",
}


def visit(name: str, lang: str) -> None:
    """Name the plate and the pass the following lookups belong to."""
    _audit.visit(name, lang)


def lookup(s: str, *, translate: bool) -> str:
    """The Spanish rendering of *s* when *translate*, otherwise *s* itself.

    Both passes come through here, and the one that does not translate still
    reports: this is the only place that knows the table had no entry for a
    string, and a plate label that never reached :data:`_ES` is drawn in
    English inside ``*_es.svg`` with nothing downstream able to tell that
    from a label that reads the same in both languages. See
    ``scripts/figure_language_audit.py``.
    """
    if s in _ES:
        return _ES[s] if translate else s
    _audit.untranslated(s)
    return s
