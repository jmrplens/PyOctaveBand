/**
 * Every quantity the guides compute, once.
 *
 * The glossary used to exist four times: a five-column table on the English
 * page, the same table translated on the Spanish one, and a hand-written
 * JSON-LD block of 120 `DefinedTerm` nodes in the frontmatter of each. Nothing
 * kept the four in step, and they had drifted. This is the one they are all
 * built from now: `Glossary.astro` renders the page from it and emits the
 * structured data from the same array, in whichever language the page is in.
 *
 * A field that cannot legitimately differ between the two languages is stored
 * once, which is what stops it drifting: the symbol is the notation the
 * standard prints, and the guide is a route. What genuinely translates is
 * stored as `{ en, es }`: the definition always; the unit when the unit is a
 * word ("dimensionless"); the clause, because "clause" is "apartado"; the
 * designation when it carries translated prose ("no governing standard", an
 * author list joined by "and"); and the two entries whose name is a term
 * rather than a symbol.
 *
 * The guide is a locale-free slug and the link's text is the title of that
 * page as the site has it, read at build time. The old tables carried that
 * title by hand, in both languages, and had gone stale.
 *
 * Per term:
 *   id          stable anchor, unique across the glossary
 *   symbol      the notation, inline maths in `$...$`; omitted when the entry
 *               is named rather than symbolised
 *   name        `{ en, es }`, for those entries, instead of `symbol`
 *   qualifier   `{ en, es }`, what tells apart two entries sharing a symbol
 *   unit        a string, or `{ en, es }` when the unit is a translated word
 *   standard    the designation that defines the quantity, or `{ en, es }`
 *   clause      `{ en, es }` or a string: where in that standard
 *   guide       slug of the page that implements it, no locale, no base
 *   definition  `{ en, es }`, one sentence
 */
export const glossary = [
  {
    id: "sound-pressure-power-and-intensity-levels",
    label: {
      en: "Sound pressure, power and intensity levels",
      es: "Niveles de presión, potencia e intensidad",
    },
    terms: [
      {
        id: "l-p",
        symbol: "$L_p$",
        unit: "dB re 20 µPa",
        standard: "IEC 61672-1:2013",
        guide: "signals/metrology/calibration",
        definition: {
          en: "Sound pressure level: twenty times the base-10 logarithm of the r.m.s. sound pressure over the reference pressure.",
          es: "Nivel de presión acústica: veinte veces el logaritmo decimal de la presión acústica eficaz dividida por la presión de referencia.",
        },
      },
      {
        id: "l-eq",
        symbol: "$L_{eq}$",
        unit: "dB re 20 µPa",
        standard: "IEC 61672-1:2013",
        guide: "signals/levels/levels",
        definition: {
          en: "Equivalent continuous sound pressure level: the level of the steady sound carrying the same mean-square pressure over the interval.",
          es: "Nivel de presión acústica continuo equivalente: el nivel del sonido estacionario que transporta la misma presión cuadrática media durante el intervalo.",
        },
      },
      {
        id: "l-aeq",
        symbol: "$L_{Aeq}$",
        unit: "dB re 20 µPa",
        standard: "IEC 61672-1:2013",
        guide: "signals/levels/levels",
        definition: {
          en: "The same integral applied to the A-weighted signal, the default descriptor of environmental and occupational noise.",
          es: "La misma integral aplicada a la señal ponderada A, el descriptor por defecto del ruido ambiental y laboral.",
        },
      },
      {
        id: "l-ae-sel",
        symbol: "$L_{AE}$, SEL",
        unit: "dB re (20 µPa)²·s",
        standard: "IEC 61672-1:2013",
        clause: {
          en: "Equation 8 (Table 4)",
          es: "Ecuación 8 (Tabla 4)",
        },
        guide: "signals/levels/levels",
        definition: {
          en: "Sound exposure level: the whole A-weighted energy of a single event normalised to one second.",
          es: "Nivel de exposición sonora: toda la energía ponderada A de un suceso único normalizada a un segundo.",
        },
      },
      {
        id: "l-cpeak",
        symbol: "$L_{Cpeak}$",
        unit: "dB re 20 µPa",
        standard: "IEC 61672-1:2013",
        clause: {
          en: "subclause 5.13",
          es: "apartado 5.13",
        },
        guide: "signals/levels/levels",
        definition: {
          en: "C-weighted peak sound level: the absolute maximum of the C-weighted pressure, not a time-weighted maximum.",
          es: "Nivel de pico ponderado C: el máximo absoluto de la presión ponderada C, no un máximo con ponderación temporal.",
        },
      },
      {
        id: "l-n-l-10-l-50-l-90",
        symbol: "$L_N$ ($L_{10}$, $L_{50}$, $L_{90}$)",
        unit: "dB re 20 µPa",
        standard: {
          en: "ISO 1996-2:2017 (Annex I uses $L_{90}$ as the residual level)",
          es: "ISO 1996-2:2017 (el Anexo I usa $L_{90}$ como nivel residual)",
        },
        guide: "signals/levels/levels",
        definition: {
          en: "Percentile level: the level exceeded $N$ % of the measurement time, read off the time-weighted level distribution.",
          es: "Nivel percentil: el nivel superado el $N$ % del tiempo de medida, leído en la distribución del nivel con ponderación temporal.",
        },
      },
      {
        id: "l-w-swl",
        symbol: "$L_W$, SWL",
        unit: "dB re 1 pW",
        standard: "ISO 3745:2012",
        clause: {
          en: "Clause 8",
          es: "apartado 8",
        },
        guide: "devices/emission/sound-power",
        definition: {
          en: "Sound power level: the power a source radiates, referred to 1 pW.",
          es: "Nivel de potencia acústica: la potencia que radia una fuente, referida a 1 pW.",
        },
      },
      {
        id: "l-i",
        symbol: "$L_I$",
        unit: "dB re 1 pW/m²",
        standard: "IEC 61043:1993",
        guide: "devices/emission/intensity",
        definition: {
          en: "Sound intensity level: the magnitude of the intensity vector referred to 1 pW/m², with the flow direction reported separately as a sign.",
          es: "Nivel de intensidad acústica: el módulo del vector intensidad referido a 1 pW/m², con el sentido del flujo indicado aparte mediante un signo.",
        },
      },
      {
        id: "l-p---l-i",
        symbol: "$L_p - L_I$",
        unit: "dB",
        standard: "ISO 9614-1:1993",
        clause: {
          en: "Equation (A.3)",
          es: "Ecuación (A.3)",
        },
        guide: "devices/emission/intensity",
        definition: {
          en: "Pressure-intensity index: the difference between the pressure and intensity levels at a position, the field indicator that qualifies an intensity measurement.",
          es: "Índice presión-intensidad: la diferencia entre el nivel de presión y el de intensidad en una posición, el indicador de campo que cualifica una medida de intensidad.",
        },
      },
      {
        id: "k-1",
        symbol: "$K_1$",
        unit: "dB",
        standard: "ISO 3744:2010",
        clause: {
          en: "3.16 and Equation (16)",
          es: "3.16 y ecuación (16)",
        },
        guide: "devices/emission/sound-power-pressure",
        definition: {
          en: "Background noise correction: what is subtracted from the surface level to remove the background's own contribution, $-10\\log_{10}(1 - 10^{-0,1\\Delta L_p})$ from the source-on minus source-off margin. It is a cliff rather than a slope: above a 15 dB margin it is taken as zero, between 6 and 15 dB it is computed, and below 6 dB the standard caps it at 1,3 dB and warns that the result has lost accuracy.",
          es: "Corrección por ruido de fondo: lo que se resta al nivel de la superficie de medición para eliminar la contribución del propio fondo, $-10\\log_{10}(1 - 10^{-0,1\\Delta L_p})$ a partir del margen entre fuente en marcha y fuente parada. Es un escalón y no una pendiente: por encima de 15 dB de margen se toma como cero, entre 6 y 15 dB se calcula, y por debajo de 6 dB la norma la limita a 1,3 dB y advierte de que el resultado ha perdido exactitud.",
        },
      },
      {
        id: "k-2",
        symbol: "$K_2$",
        unit: "dB",
        standard: "ISO 3744:2010",
        clause: {
          en: "3.17 and Formula (A.2)",
          es: "3.17 y fórmula (A.2)",
        },
        guide: "devices/emission/sound-power-pressure",
        definition: {
          en: "Environmental correction: what is subtracted to remove the energy the test room reflects back onto the measurement surface, $10\\log_{10}(1 + 4S/A)$ from the surface area and the room's equivalent absorption area. Its ceiling is the grade of accuracy rather than a preference: an engineering-grade result is only valid where $K_{2A} \\le 4$ dB, the survey method allows 7 dB, and a qualified hemi-anechoic room gives zero.",
          es: "Corrección ambiental: lo que se resta para eliminar la energía que la sala de ensayo refleja de vuelta sobre la superficie de medición, $10\\log_{10}(1 + 4S/A)$ a partir del área de la superficie y del área de absorción equivalente de la sala. Su techo lo fija el grado de exactitud y no una preferencia: un resultado de grado de ingeniería solo es válido si $K_{2A} \\le 4$ dB, el método de control admite 7 dB y una sala semianecoica cualificada da cero.",
        },
      },
      {
        id: "l-wad-k-wa",
        symbol: "$L_{WAd}$, $K_{WA}$",
        unit: "dB re 1 pW",
        standard: "ISO 4871:1996",
        clause: {
          en: "3.15 and 3.16",
          es: "3.15 y 3.16",
        },
        guide: "devices/emission/sound-power",
        definition: {
          en: "The declared noise emission of a machine: either the dual-number form, the measured $L_{WA}$ and its uncertainty $K_{WA}$ stated separately, or the single-number form $L_{WAd} = L_{WA} + K_{WA}$, both rounded to the nearest whole decibel. The declared value is an upper limit a verification measurement is unlikely to exceed, not a best estimate, so it is never the number to feed into a propagation calculation.",
          es: "La emisión sonora declarada de una máquina: o bien la forma de dos números, el $L_{WA}$ medido y su incertidumbre $K_{WA}$ indicados por separado, o bien la forma de un número $L_{WAd} = L_{WA} + K_{WA}$, ambas redondeadas al decibelio entero más próximo. El valor declarado es un límite superior que una medida de verificación difícilmente superará, no una mejor estimación, así que nunca es el número que debe entrar en un cálculo de propagación.",
        },
      },
      {
        id: "l-wa",
        symbol: "$L_{WA}$",
        qualifier: {
          en: "apparent, wind turbine",
          es: "aparente, aerogenerador",
        },
        unit: "dB re 1 pW",
        standard: {
          en: "IEC 61400-11:2012+AMD1:2018",
          es: "IEC 61400-11:2012+AMD1:2018",
        },
        clause: {
          en: "3.1 and Formula (26)",
          es: "3.1 y fórmula (26)",
        },
        guide: "environment/sources/wind-turbine-noise",
        definition: {
          en: "Apparent sound power level of a wind turbine: the A-weighted level of a point source at the rotor centre that would radiate the same downwind emission as the machine measured. It is written like a sound power level but is not one in the usual sense: the ground-board measurement builds a downwind reflection into it, so feeding it to a propagation model that adds a ground effect counts that reflection twice.",
          es: "Nivel de potencia acústica aparente de un aerogenerador: el nivel ponderado A de una fuente puntual situada en el centro del rotor que radiaría a sotavento la misma emisión que la máquina medida. Se escribe como un nivel de potencia acústica pero no lo es en el sentido habitual: la medida sobre placa reflectante incorpora una reflexión a sotavento, de modo que introducirlo en un modelo de propagación que añade efecto del terreno cuenta esa reflexión dos veces.",
        },
      },
    ],
  },
  {
    id: "environmental-and-occupational-descriptors",
    label: {
      en: "Environmental and occupational descriptors",
      es: "Descriptores ambientales y laborales",
    },
    terms: [
      {
        id: "l-den",
        symbol: "$L_{den}$",
        unit: "dB re 20 µPa",
        standard: "ISO 1996-1:2016",
        clause: "3.6.4",
        guide: "environment/assessment/environmental-levels",
        definition: {
          en: "Day-evening-night level: the energy mean of the three periods with 5 dB added to the evening and 10 dB to the night.",
          es: "Nivel día-tarde-noche: la media energética de los tres periodos con 5 dB añadidos a la tarde y 10 dB a la noche.",
        },
      },
      {
        id: "l-dn",
        symbol: "$L_{dn}$",
        unit: "dB re 20 µPa",
        standard: "ISO 1996-1:2016",
        clause: "3.6.5",
        guide: "environment/assessment/environmental-levels",
        definition: {
          en: "Day-night level: the same construction with the 10 dB night penalty only.",
          es: "Nivel día-noche: la misma construcción solo con la penalización de 10 dB nocturna.",
        },
      },
      {
        id: "l-r",
        symbol: "$L_r$",
        unit: "dB re 20 µPa",
        standard: "ISO 1996-1:2016",
        clause: {
          en: "clause 6.5 (Formulae 5 and 6)",
          es: "apartado 6.5 (Fórmulas 5 y 6)",
        },
        guide: "environment/assessment/environmental-levels",
        definition: {
          en: "Rating level: the whole-day composite level after the source-character and time-of-day adjustments.",
          es: "Nivel de evaluación: el nivel compuesto de la jornada completa tras los ajustes por carácter de la fuente y por franja horaria.",
        },
      },
      {
        id: "l-ar-t",
        symbol: "$L_{Ar,T}$",
        unit: "dB re 20 µPa",
        standard: "NT ACOU 112:2002",
        clause: {
          en: "clause 8",
          es: "apartado 8",
        },
        guide: "environment/assessment/impulsive-sound",
        definition: {
          en: "Rating level of an impulsive source over a reference interval, $L_{Aeq}$ plus the graduated impulse adjustment.",
          es: "Nivel de evaluación de una fuente impulsiva en un intervalo de referencia: el $L_{Aeq}$ más el ajuste graduado por impulsos.",
        },
      },
      {
        id: "k-i",
        symbol: "$K_I$",
        unit: "dB",
        standard: "NT ACOU 112:2002",
        clause: {
          en: "clause 8",
          es: "apartado 8",
        },
        guide: "environment/assessment/impulsive-sound",
        definition: {
          en: "Impulse adjustment added to $L_{Aeq}$, graduated by the predicted prominence of the impulses.",
          es: "Ajuste por impulsos que se suma al $L_{Aeq}$, graduado según la prominencia prevista de los impulsos.",
        },
      },
      {
        id: "e",
        symbol: "$E$",
        unit: "Pa²h",
        standard: "IEC 61252:1993",
        clause: "3.1",
        guide: "signals/levels/levels",
        definition: {
          en: "Sound exposure: the time integral of the squared A-weighted sound pressure over the exposure period.",
          es: "Exposición sonora: la integral temporal de la presión acústica ponderada A al cuadrado durante el periodo de exposición.",
        },
      },
      {
        id: "l-ex-8h-l-ep-d",
        symbol: "$L_{EX,8h}$, $L_{EP,d}$",
        unit: "dB re 20 µPa",
        standard: "IEC 61252:1993",
        clause: "3.3",
        guide: "perception/hearing/occupational-exposure",
        definition: {
          en: "Daily noise exposure level: the steady level that, sustained over a nominal 8 h day, carries the same A-weighted sound exposure as the measured one.",
          es: "Nivel de exposición diaria al ruido: el nivel estacionario que, mantenido durante una jornada nominal de 8 h, acumula la misma exposición sonora ponderada A que la medida.",
        },
      },
      {
        id: "l-p-a-eqt",
        symbol: "$L_{p,A,eqT}$",
        unit: "dB re 20 µPa",
        standard: "ISO 9612:2009",
        clause: {
          en: "clauses 9 to 11",
          es: "apartados 9 a 11",
        },
        guide: "perception/hearing/occupational-exposure",
        definition: {
          en: "A-weighted equivalent continuous level of a task, a job sample or a full day, the building block $L_{EX,8h}$ is assembled from.",
          es: "Nivel continuo equivalente ponderado A de una tarea, de una muestra del puesto o de una jornada completa, el ladrillo con el que se construye el $L_{EX,8h}$.",
        },
      },
      {
        id: "nipts",
        symbol: "NIPTS",
        unit: "dB",
        standard: "ISO 1999:2013",
        guide: "perception/hearing/noise-induced-hearing-loss",
        definition: {
          en: "Noise-induced permanent threshold shift: the median hearing loss attributable to a stated exposure level, duration and audiometric frequency.",
          es: "Desplazamiento permanente del umbral inducido por ruido: la pérdida auditiva mediana atribuible a un nivel, una duración y una frecuencia audiométrica dados.",
        },
      },
      {
        id: "htlan",
        symbol: "HTLAN",
        unit: "dB",
        standard: "ISO 1999:2013",
        guide: "perception/hearing/noise-induced-hearing-loss",
        definition: {
          en: "Hearing threshold level associated with age and noise: the NIPTS combined with the age component.",
          es: "Nivel de umbral auditivo asociado a la edad y al ruido: el NIPTS combinado con la componente de la edad.",
        },
      },
      {
        id: "l-keq",
        symbol: "$L_{Keq,T}$",
        unit: "dB re 20 µPa",
        standard: {
          en: "RD 1367/2007 (Spain)",
          es: "RD 1367/2007",
        },
        clause: {
          en: "Annex I A.2 c",
          es: "Anexo I A.2 c",
        },
        guide: "environment/assessment/spanish-noise-regulation",
        definition: {
          en: "Corrected equivalent level: the A-weighted equivalent level of the interval plus the three penalties for tonal, low-frequency and impulsive character, $L_{Aeq,T} + K_t + K_f + K_i$. It is the quantity the Spanish immission limits are written against, so an activity is judged on it and not on the bare $L_{Aeq}$.",
          es: "Nivel corregido equivalente: el nivel equivalente ponderado A del intervalo más las tres penalizaciones por componentes tonales, de baja frecuencia e impulsivas, $L_{Aeq,T} + K_t + K_f + K_i$. Es la magnitud en la que están escritos los límites de inmisión españoles, de modo que una actividad se juzga por él y no por el $L_{Aeq}$ desnudo.",
        },
      },
      {
        id: "k-t-k-f-k-i",
        symbol: "$K_t$, $K_f$, $K_i$",
        unit: "dB",
        standard: {
          en: "RD 1367/2007 (Spain)",
          es: "RD 1367/2007",
        },
        clause: {
          en: "Annex IV A.3.3",
          es: "Anexo IV A.3.3",
        },
        guide: "environment/assessment/spanish-noise-regulation",
        definition: {
          en: "The three character penalties added to $L_{Aeq,T}$: $K_t$ for emergent tonal components, read from an unweighted one-third-octave spectrum against the arithmetic mean of the two adjacent bands; $K_f$ for low-frequency content, from $L_{Ceq} - L_{Aeq}$; and $K_i$ for impulsive content, from $L_{AIeq} - L_{Aeq}$. They are stepped, not continuous, so a spectrum just short of a threshold scores nothing.",
          es: "Las tres penalizaciones por carácter que se suman al $L_{Aeq,T}$: $K_t$ por componentes tonales emergentes, leída en un espectro de tercio de octava sin ponderar frente a la media aritmética de las dos bandas contiguas; $K_f$ por contenido de baja frecuencia, a partir de $L_{Ceq} - L_{Aeq}$; y $K_i$ por contenido impulsivo, a partir de $L_{AIeq} - L_{Aeq}$. Son escalonadas y no continuas, así que un espectro que se queda algo por debajo de un umbral no puntúa nada.",
        },
      },
    ],
  },
  {
    id: "outdoor-propagation",
    label: {
      en: "Outdoor propagation",
      es: "Propagación en exteriores",
    },
    terms: [
      {
        id: "a-3",
        symbol: "$A$",
        qualifier: {
          en: "octave-band attenuation",
          es: "atenuación por banda de octava",
        },
        unit: "dB",
        standard: "ISO 9613-2:1996",
        clause: {
          en: "Equations (3) and (4)",
          es: "Ecuaciones (3) y (4)",
        },
        guide: "environment/propagation/outdoor-propagation",
        definition: {
          en: "Total octave-band attenuation between an outdoor point source and a downwind receiver: the sum $A_{div} + A_{atm} + A_{gr} + A_{bar} + A_{misc}$ subtracted from the sound power level and the directivity correction. Everything the method knows about the path between source and receiver is in this one term.",
          es: "Atenuación total por banda de octava entre una fuente puntual exterior y un receptor a sotavento: la suma $A_{div} + A_{atm} + A_{gr} + A_{bar} + A_{misc}$ que se resta al nivel de potencia acústica y a la corrección por directividad. Todo lo que el método sabe del camino entre fuente y receptor está en este único término.",
        },
      },
      {
        id: "a-div",
        symbol: "$A_{div}$",
        unit: "dB",
        standard: "ISO 9613-2:1996",
        clause: {
          en: "Equation (7)",
          es: "Ecuación (7)",
        },
        guide: "environment/propagation/outdoor-propagation",
        definition: {
          en: "Geometrical divergence: $20\\log_{10}(d/d_0) + 11$ dB, the spreading of a point source radiating into free space with $d_0 = 1$ m. The 11 dB constant is what refers the level to a sound power level rather than to a level measured at one metre.",
          es: "Divergencia geométrica: $20\\log_{10}(d/d_0) + 11$ dB, la propagación de una fuente puntual que radia en campo libre con $d_0 = 1$ m. La constante de 11 dB es lo que refiere el nivel a un nivel de potencia acústica y no a un nivel medido a un metro.",
        },
      },
      {
        id: "a-atm",
        symbol: "$A_{atm}$",
        unit: "dB",
        standard: "ISO 9613-2:1996",
        clause: {
          en: "Equation (8)",
          es: "Ecuación (8)",
        },
        guide: "environment/propagation/outdoor-propagation",
        definition: {
          en: "Atmospheric absorption: the attenuation coefficient of the air times the path length. It is the term that removes the high bands over long distances, and it depends strongly on frequency, temperature and humidity — which is why a long-range prediction has to state the weather it assumed.",
          es: "Absorción atmosférica: el coeficiente de atenuación del aire por la longitud del camino. Es el término que elimina las bandas altas a grandes distancias y depende mucho de la frecuencia, la temperatura y la humedad, y por eso una predicción a larga distancia debe indicar las condiciones meteorológicas supuestas.",
        },
      },
      {
        id: "alpha-atm",
        symbol: "$\\alpha$",
        qualifier: {
          en: "atmospheric",
          es: "atmosférico",
        },
        unit: {
          en: "dB/m (ISO 9613-2 tabulates dB/km)",
          es: "dB/m (la ISO 9613-2 la tabula en dB/km)",
        },
        standard: "ISO 9613-1:1993",
        clause: {
          en: "Equation (5)",
          es: "Ecuación (5)",
        },
        guide: "environment/propagation/outdoor-propagation",
        definition: {
          en: "Atmospheric attenuation coefficient: the excess loss per unit path length from classical absorption and the nitrogen and oxygen relaxation processes, a function of frequency, temperature, humidity and pressure. Watch the length unit: the library returns decibels per metre while ISO 9613-2 Table 2 tabulates decibels per kilometre, a factor of a thousand.",
          es: "Coeficiente de atenuación atmosférica: la pérdida en exceso por unidad de longitud debida a la absorción clásica y a los procesos de relajación del nitrógeno y del oxígeno, función de la frecuencia, la temperatura, la humedad y la presión. Cuidado con la unidad de longitud: la biblioteca devuelve decibelios por metro mientras que la tabla 2 de la ISO 9613-2 los tabula por kilómetro, un factor de mil.",
        },
      },
      {
        id: "a-gr",
        symbol: "$A_{gr}$",
        unit: "dB",
        standard: "ISO 9613-2:1996",
        clause: {
          en: "Equation (9)",
          es: "Ecuación (9)",
        },
        guide: "environment/propagation/outdoor-propagation",
        definition: {
          en: "Ground effect: the interference between the direct path and the path reflected off the ground, split into a source region, a receiver region and the middle between them. Over hard ground it comes out negative — a net gain, not a loss — which is why it cannot be treated as an attenuation that is merely optional.",
          es: "Efecto del terreno: la interferencia entre el camino directo y el reflejado en el suelo, dividida en una región de fuente, una de receptor y la zona intermedia. Sobre terreno duro resulta negativa —una ganancia neta, no una pérdida—, y por eso no puede tratarse como una atenuación simplemente opcional.",
        },
      },
      {
        id: "a-bar",
        symbol: "$A_{bar}$",
        unit: "dB",
        standard: "ISO 9613-2:1996",
        clause: {
          en: "Equation (12)",
          es: "Ecuación (12)",
        },
        guide: "environment/propagation/outdoor-propagation",
        definition: {
          en: "Barrier attenuation: the screening $D_z$ of the diffracting edge minus the ground effect the barrier removes, floored at zero. The subtraction is the point: a barrier over soft ground buys much less than its raw screening, because the ground was already doing part of the work.",
          es: "Atenuación por pantalla: el apantallamiento $D_z$ del borde difractante menos el efecto del terreno que la pantalla suprime, con cero como valor mínimo. La resta es lo esencial: una pantalla sobre terreno blando aporta bastante menos que su apantallamiento bruto, porque el terreno ya hacía parte del trabajo.",
        },
      },
      {
        id: "c-met",
        symbol: "$C_{met}$",
        unit: "dB",
        standard: "ISO 9613-2:1996",
        clause: {
          en: "Equations (21) and (22)",
          es: "Ecuaciones (21) y (22)",
        },
        guide: "environment/propagation/outdoor-propagation",
        definition: {
          en: "Meteorological correction: what is subtracted from the downwind level to obtain a long-term average over many wind directions, driven by the local factor $C_0$ and by the source and receiver heights against the distance. It is zero close to the source and grows only where the path is long compared with the heights.",
          es: "Corrección meteorológica: lo que se resta al nivel a sotavento para obtener un promedio a largo plazo sobre muchas direcciones de viento, gobernada por el factor local $C_0$ y por las alturas de fuente y receptor frente a la distancia. Es nula cerca de la fuente y solo crece cuando el camino es largo comparado con las alturas.",
        },
      },
      {
        id: "d-c",
        symbol: "$D_c$",
        unit: "dB",
        standard: "ISO 9613-2:1996",
        clause: {
          en: "Equation (3)",
          es: "Ecuación (3)",
        },
        guide: "environment/propagation/outdoor-propagation",
        definition: {
          en: "Directivity correction: how far the level from the point source in the chosen direction departs from that of an omnidirectional source of the same sound power. It is the directivity index of the source plus an index for radiation into less than the full sphere, and it is 0 dB for an omnidirectional source in free space.",
          es: "Corrección por directividad: cuánto se aparta el nivel de la fuente puntual en la dirección elegida del de una fuente omnidireccional de la misma potencia acústica. Es el índice de directividad de la fuente más un índice por radiación en menos de la esfera completa, y vale 0 dB para una fuente omnidireccional en campo libre.",
        },
      },
      {
        id: "fresnel-n",
        symbol: "$N$",
        qualifier: {
          en: "Fresnel number",
          es: "número de Fresnel",
        },
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: {
          en: "Bies, Hansen and Howard (2017)",
          es: "Bies, Hansen y Howard (2017)",
        },
        clause: {
          en: "Equation 5.134; no governing standard",
          es: "Ecuación 5.134; sin norma aplicable",
        },
        guide: "environment/propagation/ground-barriers",
        definition: {
          en: "Fresnel number of a screen: twice the extra path length the sound has to travel over the edge, divided by the wavelength. It is the single geometric parameter of the Kurze-Anderson insertion loss, which is why a barrier that works at 1 kHz can be worth almost nothing two octaves lower for the same geometry.",
          es: "Número de Fresnel de una pantalla: el doble de la longitud de camino adicional que el sonido debe recorrer por encima del borde, dividido por la longitud de onda. Es el único parámetro geométrico de la pérdida por inserción de Kurze-Anderson, y por eso una pantalla que funciona a 1 kHz puede no valer casi nada dos octavas por debajo con la misma geometría.",
        },
      },
    ],
  },
  {
    id: "frequency-and-time-weighting",
    label: {
      en: "Frequency and time weighting",
      es: "Ponderación frecuencial y temporal",
    },
    terms: [
      {
        id: "a-c-z",
        symbol: "A, C, Z",
        unit: "dB",
        standard: "IEC 61672-1:2013",
        clause: {
          en: "Annex E (acceptance limits in Table 3)",
          es: "Anexo E (límites de aceptación en la Tabla 3)",
        },
        guide: "signals/levels/weighting",
        definition: {
          en: "The normative frequency weightings: the ear-response curves applied before integration, Z being the flat reference.",
          es: "Las ponderaciones frecuenciales normativas: las curvas de respuesta del oído que se aplican antes de integrar, siendo Z la referencia plana.",
        },
      },
      {
        id: "g",
        symbol: "G",
        unit: "dB",
        standard: "ISO 7196:1995",
        clause: {
          en: "Table 1 (nominal responses in Table 2)",
          es: "Tabla 1 (respuestas nominales en la Tabla 2)",
        },
        guide: "signals/levels/special-weightings",
        definition: {
          en: "Infrasound weighting, defined by its poles and zeros for the 0.25 Hz to 315 Hz range.",
          es: "Ponderación para infrasonido, definida por sus polos y ceros en el intervalo de 0,25 Hz a 315 Hz.",
        },
      },
      {
        id: "b",
        symbol: "B",
        unit: "dB",
        standard: "ANSI S1.4-1983",
        clause: {
          en: "Appendix C (Formula C2)",
          es: "Apéndice C (Fórmula C2)",
        },
        guide: "signals/levels/special-weightings",
        definition: {
          en: "Historical mid-level weighting, withdrawn from the current meter standard.",
          es: "Ponderación histórica para niveles medios, retirada de la norma vigente de sonómetros.",
        },
      },
      {
        id: "d",
        symbol: "D",
        unit: "dB",
        standard: {
          en: "IEC 537:1976 (withdrawn)",
          es: "IEC 537:1976 (retirada)",
        },
        guide: "signals/levels/special-weightings",
        definition: {
          en: "Historical aircraft-noise weighting, derived from the 40-noy perceived-noisiness contour.",
          es: "Ponderación histórica para ruido de aeronaves, derivada de la curva de ruidosidad percibida de 40 noys.",
        },
      },
      {
        id: "au",
        symbol: "AU",
        unit: "dB",
        standard: "IEC 61012:1990",
        clause: {
          en: "subclause 2.2 (Tables 1 and 2)",
          es: "apartado 2.2 (Tablas 1 y 2)",
        },
        guide: "signals/levels/special-weightings",
        definition: {
          en: "Weighting for audible sound measured in the presence of ultrasound.",
          es: "Ponderación para el sonido audible medido en presencia de ultrasonido.",
        },
      },
      {
        id: "f-s-i",
        symbol: "F, S, I",
        unit: {
          en: "s (time constant)",
          es: "s (constante de tiempo)",
        },
        standard: "IEC 61672-1:2013",
        guide: "signals/levels/time-weighting",
        definition: {
          en: "Fast, Slow and Impulse exponential time weightings: the detector ballistics that produce a displayed level.",
          es: "Ponderaciones temporales exponenciales Fast, Slow e Impulse: las balísticas del detector que producen el nivel mostrado.",
        },
      },
      {
        id: "filter-class",
        name: {
          en: "Performance class (0, 1, 2)",
          es: "Clase de prestaciones (0, 1, 2)",
        },
        unit: {
          en: "dB (tolerance)",
          es: "dB (tolerancia)",
        },
        standard: "IEC 61260-1:2014",
        clause: {
          en: "1.2 and Table 1",
          es: "1.2 y tabla 1",
        },
        guide: "signals/filters/filter-compliance",
        definition: {
          en: "Performance class of a filter or an instrument: the width of the tolerance corridor its response has to stay inside, band by band. Class 1 and class 2 share the same design goals and differ in the acceptance limits and in the operating temperature range, class 2 being the looser; class 0, the laboratory reference grade, comes from the withdrawn IEC 61260:1995 and ANSI S1.11-2004 and the current edition no longer defines it. A class is a property of the response, not of the result: a class 2 bank does not make a measurement wrong, it makes its band levels less certain.",
          es: "Clase de prestaciones de un filtro o de un instrumento: la anchura del corredor de tolerancia dentro del cual debe mantenerse su respuesta, banda a banda. Las clases 1 y 2 comparten los mismos objetivos de diseño y se diferencian en los límites de aceptación y en el intervalo de temperatura de funcionamiento, siendo la clase 2 la más holgada; la clase 0, de referencia de laboratorio, procede de las retiradas IEC 61260:1995 y ANSI S1.11-2004 y la edición vigente ya no la define. La clase es una propiedad de la respuesta, no del resultado: un banco de clase 2 no hace que una medida sea errónea, hace que sus niveles de banda sean menos ciertos.",
        },
      },
      {
        id: "k-weighting",
        symbol: "K",
        unit: "dB",
        standard: "ITU-R BS.1770-5",
        clause: {
          en: "Annex 1",
          es: "Anexo 1",
        },
        guide: "devices/broadcast/program-loudness",
        definition: {
          en: "K-weighting: the programme-loudness curve, a two-stage pre-filter that models the high-frequency boost a spherical head gives and then applies the revised low-frequency B-curve high-pass. It is applied per channel before the channel mean squares are summed and gated, and the LKFS designation records it.",
          es: "Ponderación K: la curva de sonoridad de programa, un prefiltro de dos etapas que modela el realce en alta frecuencia que produce una cabeza esférica y aplica después el paso alto de la curva B revisada en baja frecuencia. Se aplica canal a canal antes de sumar y aplicar la puerta a los valores cuadráticos medios, y la designación LKFS lo recoge.",
        },
      },
      {
        id: "itu-r-468",
        name: {
          en: "ITU-R 468 weighting",
          es: "Ponderación ITU-R 468",
        },
        unit: "dB",
        standard: {
          en: "Recommendation ITU-R BS.468-4",
          es: "Recomendación UIT-R BS.468-4",
        },
        clause: {
          en: "Table 1",
          es: "Tabla 1",
        },
        guide: "devices/electroacoustics/electroacoustics",
        definition: {
          en: "The broadcast noise weighting: zero at 1 kHz, peaking at $+12{,}2$ dB at 6,3 kHz and falling to $-29{,}9$ dB at 31,5 Hz, shaped to how audible a noise is rather than how loud a tone is. The Recommendation pairs it with a quasi-peak detector and quotes results as dB(468); AES17 reuses the same curve with an r.m.s. detector, which is a different number from the same filter.",
          es: "La ponderación de ruido para radiodifusión: cero en 1 kHz, con máximo de $+12{,}2$ dB a 6,3 kHz y caída hasta $-29{,}9$ dB a 31,5 Hz, con la forma de lo audible que resulta un ruido y no de lo sonoro que resulta un tono. La Recomendación la combina con un detector de cuasipico y expresa el resultado en dB(468); la AES17 reutiliza la misma curva con un detector eficaz, lo que da otro número con el mismo filtro.",
        },
      },
      {
        id: "w-wholebody",
        symbol: "$W_b$, $W_c$, $W_d$, $W_e$, $W_f$, $W_j$, $W_k$, $W_m$",
        qualifier: {
          en: "whole-body",
          es: "cuerpo entero",
        },
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: "ISO 8041-1:2017",
        clause: {
          en: "5.6.1, Formulae (1) to (5) and Table 3",
          es: "5.6.1, fórmulas (1) a (5) y tabla 3",
        },
        guide: "vibration/human/human-vibration",
        definition: {
          en: "The whole-body frequency weightings, one parameter row each of the same four-stage filter: $W_k$ for the vertical axis and $W_d$ for the two horizontal ones in health and comfort, with $W_b$, $W_c$, $W_e$, $W_f$, $W_j$ and $W_m$ for ride comfort, the seat back, rotational axes, motion sickness and the head. The weighted acceleration $a_w$ is what comes out of them; the standard that names the curve is not the one that says where to apply it.",
          es: "Las ponderaciones en frecuencia para cuerpo entero, cada una una fila de parámetros del mismo filtro de cuatro etapas: $W_k$ para el eje vertical y $W_d$ para los dos horizontales en salud y confort, con $W_b$, $W_c$, $W_e$, $W_f$, $W_j$ y $W_m$ para el confort de marcha, el respaldo, los ejes de rotación, el mareo por movimiento y la cabeza. De ellas sale la aceleración ponderada $a_w$; la norma que define la curva no es la que dice dónde aplicarla.",
        },
      },
      {
        id: "w-h",
        symbol: "$W_h$",
        qualifier: {
          en: "hand-arm",
          es: "mano-brazo",
        },
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: "ISO 8041-1:2017",
        clause: {
          en: "5.6.1, Formulae (1) to (5) and Table 3",
          es: "5.6.1, fórmulas (1) a (5) y tabla 3",
        },
        guide: "vibration/human/human-vibration",
        definition: {
          en: "The hand-arm frequency weighting: one curve, band-limited from 8 Hz to 1 kHz, applied to each of the three axes before they are combined into the vibration total value. Unlike the whole-body case there is no axis multiplier, so the three weighted values enter the total on equal terms.",
          es: "La ponderación en frecuencia para vibración mano-brazo: una sola curva, limitada en banda de 8 Hz a 1 kHz, aplicada a cada uno de los tres ejes antes de combinarlos en el valor total de vibración. A diferencia del caso de cuerpo entero no hay multiplicador por eje, así que los tres valores ponderados entran en el total en igualdad de condiciones.",
        },
      },
    ],
  },
  {
    id: "spectral-and-system-analysis",
    label: {
      en: "Spectral and system analysis",
      es: "Análisis espectral y de sistemas",
    },
    terms: [
      {
        id: "g-xx",
        symbol: "$G_{xx}$, $G_{xy}$",
        unit: {
          en: "(unit of $x$)²/Hz",
          es: "(unidad de $x$)²/Hz",
        },
        standard: {
          en: "Bendat and Piersol (2010)",
          es: "Bendat y Piersol (2010)",
        },
        clause: {
          en: "Sections 5.2 and 9.1; no governing standard",
          es: "Secciones 5.2 y 9.1; sin norma aplicable",
        },
        guide: "signals/spectra/spectral-analysis",
        definition: {
          en: "One-sided auto- and cross-spectral density: mean-square content per hertz, so the power in a band is the integral over it and not the height of a line. Everything else in this group is a ratio of these: the coherences, the two frequency-response estimators and the coherent output spectrum.",
          es: "Densidad espectral unilateral, propia y cruzada: contenido cuadrático medio por hercio, de modo que la potencia de una banda es la integral sobre ella y no la altura de una línea. Todo lo demás de este grupo es un cociente de estas magnitudes: las coherencias, los dos estimadores de respuesta en frecuencia y el espectro de salida coherente.",
        },
      },
      {
        id: "gamma2",
        symbol: "$\\gamma^2_{iy}$",
        qualifier: {
          en: "ordinary coherence",
          es: "coherencia ordinaria",
        },
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: {
          en: "Bendat and Piersol (2010)",
          es: "Bendat y Piersol (2010)",
        },
        clause: {
          en: "Equation (7.109); no governing standard",
          es: "Ecuación (7.109); sin norma aplicable",
        },
        guide: "signals/spectra/miso-coherence",
        definition: {
          en: "Ordinary coherence: the fraction of the output autospectrum, at each frequency, that a linear time-invariant path from one input accounts for, $|G_{xy}|^2/(G_{xx}G_{yy})$. It is one where the pair is noiselessly linearly related, and with additive output noise it settles at $\\mathrm{SNR}/(1+\\mathrm{SNR})$ — so it reads as a quality figure, not as a cause.",
          es: "Coherencia ordinaria: la fracción del autoespectro de salida que, a cada frecuencia, explica un camino lineal e invariante desde una entrada, $|G_{xy}|^2/(G_{xx}G_{yy})$. Vale uno cuando el par está relacionado linealmente y sin ruido, y con ruido aditivo en la salida se estabiliza en $\\mathrm{SNR}/(1+\\mathrm{SNR})$: se lee como una cifra de calidad, no como una causa.",
        },
      },
      {
        id: "gamma2-multiple",
        symbol: "$\\gamma^2_{y:x}$",
        qualifier: {
          en: "multiple coherence",
          es: "coherencia múltiple",
        },
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: {
          en: "Bendat and Piersol (2010)",
          es: "Bendat y Piersol (2010)",
        },
        clause: {
          en: "Equation (7.35); no governing standard",
          es: "Ecuación (7.35); sin norma aplicable",
        },
        guide: "signals/spectra/miso-coherence",
        definition: {
          en: "Multiple coherence: the fraction of the output that all the measured inputs together account for, one minus the residual spectrum over the total. It is the ceiling the partial coherences are apportioned under, and what is left of it is the part of the output no measured input explains.",
          es: "Coherencia múltiple: la fracción de la salida que explican conjuntamente todas las entradas medidas, uno menos el espectro residual dividido por el total. Es el techo bajo el que se reparten las coherencias parciales, y lo que le falta es la parte de la salida que ninguna entrada medida explica.",
        },
      },
      {
        id: "gamma2-partial",
        symbol: "$\\gamma^2_{iy\\cdot(i-1)!}$",
        qualifier: {
          en: "partial coherence",
          es: "coherencia parcial",
        },
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: {
          en: "Bendat and Piersol (2010)",
          es: "Bendat y Piersol (2010)",
        },
        clause: {
          en: "Equation (7.87); no governing standard",
          es: "Ecuación (7.87); sin norma aplicable",
        },
        guide: "signals/spectra/miso-coherence",
        definition: {
          en: "Partial coherence: the coherence of one input with the output after the inputs ranked before it have been conditioned out. It is what separates a genuine source from one that merely correlates with a genuine source, and it depends on the conditioning order, so the order has to be reported with the number.",
          es: "Coherencia parcial: la coherencia de una entrada con la salida después de condicionar y retirar las entradas situadas antes que ella. Es lo que separa una fuente real de otra que solo está correlacionada con una fuente real, y depende del orden de condicionado, así que el orden debe indicarse junto con el número.",
        },
      },
      {
        id: "h-2",
        symbol: "$H_1$, $H_2$",
        qualifier: {
          en: "FRF estimators",
          es: "estimadores de FRF",
        },
        unit: {
          en: "output per input",
          es: "salida por unidad de entrada",
        },
        standard: {
          en: "Bendat and Piersol (2010)",
          es: "Bendat y Piersol (2010)",
        },
        clause: {
          en: "Section 6.1; no governing standard",
          es: "Sección 6.1; sin norma aplicable",
        },
        guide: "devices/electroacoustics/electroacoustics",
        definition: {
          en: "The two averaged estimates of a frequency response function: $H_1 = G_{xy}/G_{xx}$, unbiased when the noise is on the output, and $H_2 = G_{yy}/G_{yx}$, unbiased when it is on the input. Their ratio is exactly the ordinary coherence, so they agree only where the measurement is clean, and the gap between them is a measure of how far it is not.",
          es: "Las dos estimaciones promediadas de una función de respuesta en frecuencia: $H_1 = G_{xy}/G_{xx}$, insesgada cuando el ruido está en la salida, y $H_2 = G_{yy}/G_{yx}$, insesgada cuando está en la entrada. Su cociente es exactamente la coherencia ordinaria, así que coinciden solo donde la medida está limpia, y la distancia entre ambas mide cuánto no lo está.",
        },
      },
      {
        id: "enbw",
        symbol: "ENBW",
        unit: {
          en: "bins (or Hz)",
          es: "líneas (o Hz)",
        },
        standard: {
          en: "Harris (1978)",
          es: "Harris (1978)",
        },
        clause: {
          en: "Table 1; no governing standard",
          es: "Tabla 1; sin norma aplicable",
        },
        guide: "signals/spectra/spectral-analysis",
        definition: {
          en: "Equivalent noise bandwidth of an analysis window: the width of the ideal rectangular filter that would pass the same broadband noise power. It is exactly 1 bin for a rectangular window and 1,5 for a Hann, and it is the factor that turns a windowed line spectrum into a density — a broadband level read off the lines sits $10\\log_{10}(\\mathrm{ENBW})$ dB high without it.",
          es: "Ancho de banda equivalente de ruido de una ventana de análisis: el ancho del filtro rectangular ideal que dejaría pasar la misma potencia de ruido de banda ancha. Vale exactamente 1 línea para la ventana rectangular y 1,5 para la de Hann, y es el factor que convierte un espectro de líneas enventanado en una densidad: sin él, un nivel de banda ancha leído sobre las líneas queda $10\\log_{10}(\\mathrm{ENBW})$ dB alto.",
        },
      },
      {
        id: "cepstrum",
        name: {
          en: "Cepstrum and quefrency",
          es: "Cepstrum y quefrencia",
        },
        unit: {
          en: "quefrency in s",
          es: "quefrencia en s",
        },
        standard: {
          en: "Havelock, Kuwano and Vorländer (2008)",
          es: "Havelock, Kuwano y Vorländer (2008)",
        },
        clause: {
          en: "Chapter 27; no governing standard",
          es: "Capítulo 27; sin norma aplicable",
        },
        guide: "signals/spectra/cepstrum-echoes",
        definition: {
          en: "Cepstrum: the inverse transform of the logarithmic spectrum, in which the periodic ripple an echo or a harmonic family leaves across the spectrum collapses onto a single peak. Quefrency is its independent variable, a time in seconds, at which that peak stands at the echo's own delay — which is why a bearing report quotes a quefrency and not a frequency.",
          es: "Cepstrum: la transformada inversa del espectro logarítmico, en la que el rizado periódico que un eco o una familia de armónicos deja en el espectro se concentra en un único pico. La quefrencia es su variable independiente, un tiempo en segundos, en la que ese pico aparece al retardo propio del eco, y por eso un informe de rodamientos cita una quefrencia y no una frecuencia.",
        },
      },
      {
        id: "crest-factor",
        name: {
          en: "Crest factor",
          es: "Factor de cresta",
        },
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: "ISO 2631-1:1997",
        clause: "6.2.1",
        guide: "vibration/human/human-vibration",
        definition: {
          en: "Crest factor: the modulus of the ratio of the peak to the r.m.s. value over the measurement period. It decides whether an r.m.s. description is honest — above 9 the standard says the basic evaluation method is not sufficient and dose measures have to be reported beside it — and for a test signal it decides whether a device clips before the signal has delivered its energy.",
          es: "Factor de cresta: el módulo del cociente entre el valor de pico y el valor eficaz en el periodo de medida. Decide si una descripción en valor eficaz es honesta —por encima de 9 la norma considera insuficiente el método básico de evaluación y obliga a informar además de medidas de dosis— y, en una señal de ensayo, decide si un equipo recorta antes de que la señal entregue su energía.",
        },
      },
    ],
  },
  {
    id: "room-acoustics",
    label: {
      en: "Room acoustics",
      es: "Acústica de salas",
    },
    terms: [
      {
        id: "t-20",
        symbol: "$T_{20}$",
        unit: "s",
        standard: "ISO 3382-2:2008",
        clause: {
          en: "Clause 6 and Annex C",
          es: "apartado 6 y Anexo C",
        },
        guide: "buildings/rooms/room-acoustics",
        definition: {
          en: "Reverberation time extrapolated to a 60 dB decay from a least-squares fit over −5 dB to −25 dB of the Schroeder curve.",
          es: "Tiempo de reverberación extrapolado a una caída de 60 dB desde un ajuste por mínimos cuadrados entre −5 dB y −25 dB de la curva de Schroeder.",
        },
      },
      {
        id: "t-30",
        symbol: "$T_{30}$",
        unit: "s",
        standard: "ISO 3382-2:2008",
        clause: {
          en: "Clause 6 and Annex C",
          es: "apartado 6 y Anexo C",
        },
        guide: "buildings/rooms/room-acoustics",
        definition: {
          en: "The same extrapolation from a fit over −5 dB to −35 dB, the usual choice when the decay range allows it.",
          es: "La misma extrapolación desde un ajuste entre −5 dB y −35 dB, la opción habitual cuando el margen de caída lo permite.",
        },
      },
      {
        id: "t-60-rt",
        symbol: "$T_{60}$, RT",
        unit: "s",
        standard: "ISO 3382-1:2009",
        guide: "buildings/rooms/room-acoustics",
        definition: {
          en: "Reverberation time as such: the time for the sound energy to fall by 60 dB. Measured in practice as $T_{20}$ or $T_{30}$.",
          es: "El tiempo de reverberación propiamente dicho: el que tarda la energía sonora en caer 60 dB. En la práctica se mide como $T_{20}$ o $T_{30}$.",
        },
      },
      {
        id: "edt",
        symbol: "EDT",
        unit: "s",
        standard: {
          en: "ISO 3382-1:2009 (just-noticeable difference in Table A.1)",
          es: "ISO 3382-1:2009 (diferencia apenas perceptible en la Tabla A.1)",
        },
        guide: "buildings/rooms/room-acoustics",
        definition: {
          en: "Early decay time: the same slope taken over the first 10 dB of decay, which tracks perceived reverberance rather than the tail.",
          es: "Tiempo de caída inicial: la misma pendiente tomada sobre los primeros 10 dB de caída, que sigue la reverberancia percibida y no la cola.",
        },
      },
      {
        id: "c-50",
        symbol: "$C_{50}$",
        unit: "dB",
        standard: "ISO 3382-1:2009",
        guide: "buildings/rooms/room-acoustics",
        definition: {
          en: "Clarity for speech: the energy ratio between the first 50 ms of the impulse response and everything after it.",
          es: "Claridad para la palabra: la relación energética entre los primeros 50 ms de la respuesta al impulso y todo lo que viene después.",
        },
      },
      {
        id: "c-80",
        symbol: "$C_{80}$",
        unit: "dB",
        standard: {
          en: "ISO 3382-1:2009 (just-noticeable difference in Table A.1)",
          es: "ISO 3382-1:2009 (diferencia apenas perceptible en la Tabla A.1)",
        },
        guide: "buildings/rooms/room-acoustics",
        definition: {
          en: "Clarity for music: the same ratio with the boundary at 80 ms.",
          es: "Claridad para la música: la misma relación con la frontera en 80 ms.",
        },
      },
      {
        id: "d-50",
        symbol: "$D_{50}$",
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: {
          en: "ISO 3382-1:2009 (just-noticeable difference in Table A.1)",
          es: "ISO 3382-1:2009 (diferencia apenas perceptible en la Tabla A.1)",
        },
        guide: "buildings/rooms/room-acoustics",
        definition: {
          en: "Definition, or Deutlichkeit: the fraction of the total energy arriving in the first 50 ms.",
          es: "Definición, o Deutlichkeit: la fracción de la energía total que llega en los primeros 50 ms.",
        },
      },
      {
        id: "t-s",
        symbol: "$T_s$",
        qualifier: {
          en: "centre time",
          es: "tiempo central",
        },
        unit: "s",
        standard: "ISO 3382-1:2009",
        clause: {
          en: "Equation (A.13)",
          es: "Ecuación (A.13)",
        },
        guide: "buildings/rooms/room-acoustics",
        definition: {
          en: "Centre time: the centre of gravity of the squared impulse response in time, a boundary-free alternative to the clarity indices. It runs to tens of milliseconds in a room; the building-prediction guides write $T_s$ for something else entirely, the structural reverberation time of a plate, which is seconds.",
          es: "Tiempo central: el centro de gravedad temporal de la respuesta al impulso al cuadrado, una alternativa a los índices de claridad sin frontera arbitraria. En una sala llega a decenas de milisegundos; las guías de predicción en edificación escriben $T_s$ para algo muy distinto, el tiempo de reverberación estructural de una placa, que se mide en segundos.",
        },
      },
      {
        id: "a",
        symbol: "$A$",
        unit: "m²",
        standard: "ISO 354:2003",
        clause: {
          en: "Equations (5) and (7)",
          es: "Ecuaciones (5) y (7)",
        },
        guide: "materials/absorbers/absorption-measurement",
        definition: {
          en: "Equivalent sound absorption area of a room: the area of a perfectly absorbing surface that would give the same reverberation time.",
          es: "Área de absorción acústica equivalente de un recinto: el área de una superficie perfectamente absorbente que daría el mismo tiempo de reverberación.",
        },
      },
      {
        id: "nc",
        symbol: "NC",
        unit: {
          en: "dB (index)",
          es: "dB (índice)",
        },
        standard: "ANSI/ASA S12.2-2019",
        clause: {
          en: "5.2.2 and 5.2.3 (curves in Table 1)",
          es: "5.2.2 y 5.2.3 (curvas en la Tabla 1)",
        },
        guide: "buildings/rooms/room-noise",
        definition: {
          en: "Noise criteria rating of a background spectrum: the speech interference level selects the curve, and the tangency method rates the spectrum when a band exceeds it.",
          es: "Índice de ruido de sala de un espectro de fondo: el nivel de interferencia con la palabra elige la curva, y el método de tangencia valora el espectro cuando alguna banda la supera.",
        },
      },
      {
        id: "sil",
        symbol: "SIL",
        unit: "dB",
        standard: "ANSI/ASA S12.2-2019",
        clause: {
          en: "clause 3.2",
          es: "apartado 3.2",
        },
        guide: "buildings/rooms/room-noise",
        definition: {
          en: "Speech interference level: the average of the 500, 1000, 2000 and 4000 Hz octave-band levels.",
          es: "Nivel de interferencia con la palabra: la media de los niveles en las bandas de octava de 500, 1000, 2000 y 4000 Hz.",
        },
      },
      {
        id: "rc",
        symbol: "RC",
        unit: {
          en: "dB (index)",
          es: "dB (índice)",
        },
        standard: "ANSI/ASA S12.2-2019",
        clause: {
          en: "Annex D (clauses D.3 and D.4)",
          es: "Anexo D (apartados D.3 y D.4)",
        },
        guide: "buildings/rooms/room-noise",
        definition: {
          en: "Room criteria Mark II rating: the average of the 500, 1000 and 2000 Hz levels, with a rumble, hiss or neutral spectral tag.",
          es: "Índice RC Mark II: la media de los niveles de 500, 1000 y 2000 Hz, con una etiqueta espectral de retumbo, siseo o neutro.",
        },
      },
      {
        id: "nr",
        symbol: "NR",
        qualifier: {
          en: "curve family",
          es: "familia de curvas",
        },
        unit: {
          en: "dB (index)",
          es: "dB (índice)",
        },
        standard: {
          en: "Kosten and van Os (1962); no governing standard",
          es: "Kosten y van Os (1962); sin norma aplicable",
        },
        guide: "buildings/rooms/room-noise",
        definition: {
          en: "Noise rating, the European counterpart curve family of NC. Discussed for comparison and deliberately not implemented. Not the NR of the noise-control guides, which is a level drop.",
          es: "Noise Rating, la familia de curvas europea equivalente a NC. Se comenta a efectos de comparación y no se implementa deliberadamente. No es el NR de las guías de control de ruido, que es una caída de nivel.",
        },
      },
      {
        id: "d-2-s",
        symbol: "$D_{2,S}$",
        unit: "dB",
        standard: "ISO 3382-3:2012",
        clause: "3.2",
        guide: "buildings/rooms/open-plan-acoustics",
        definition: {
          en: "Spatial decay rate of speech: the drop in A-weighted speech level per doubling of distance along a line of workstations, taken from the regression of level against the logarithm of distance over positions between 2 m and 16 m. It is a slope only — it says how fast speech dies away, not how loud it starts.",
          es: "Tasa de decaimiento espacial del habla: la caída del nivel de habla ponderado A por cada duplicación de la distancia a lo largo de una fila de puestos de trabajo, obtenida por regresión del nivel frente al logaritmo de la distancia con posiciones entre 2 m y 16 m. Es solo una pendiente: indica con qué rapidez se extingue el habla, no con qué intensidad empieza.",
        },
      },
      {
        id: "l-p-a-s-4m",
        symbol: "$L_{p,A,S,4m}$",
        unit: "dB",
        standard: "ISO 3382-3:2012",
        clause: "3.3",
        guide: "buildings/rooms/open-plan-acoustics",
        definition: {
          en: "A-weighted speech level at 4 m: the nominal level of normal speech four metres from the talker, read off the same regression line rather than measured at that distance. It fixes the absolute height of the decay curve that $D_{2,S}$ only gives the slope of, which is why the two are always reported together.",
          es: "Nivel de habla ponderado A a 4 m: el nivel nominal del habla normal a cuatro metros del hablante, leído sobre la misma recta de regresión y no medido a esa distancia. Fija la altura absoluta de la curva de decaimiento de la que $D_{2,S}$ solo da la pendiente, y por eso ambos se informan siempre juntos.",
        },
      },
      {
        id: "r-d",
        symbol: "$r_D$",
        unit: "m",
        standard: "ISO 3382-3:2012",
        clause: "3.6",
        guide: "buildings/rooms/open-plan-acoustics",
        definition: {
          en: "Distraction distance: the distance from the talker at which the speech transmission index falls below 0,50, beyond which concentration and privacy start to improve rapidly. It is the single number an open-plan office is usually specified on, and the only rating in the corpus that is a distance rather than a level.",
          es: "Distancia de distracción: la distancia al hablante a la que el índice de transmisión del habla cae por debajo de 0,50 y a partir de la cual la concentración y la privacidad mejoran rápidamente. Es el número con el que suele especificarse una oficina diáfana y la única valoración de todo el corpus que es una distancia y no un nivel.",
        },
      },
      {
        id: "r-p",
        symbol: "$r_P$",
        unit: "m",
        standard: "ISO 3382-3:2012",
        clause: "3.7",
        guide: "buildings/rooms/open-plan-acoustics",
        definition: {
          en: "Privacy distance: the distance at which the speech transmission index falls below 0,20, beyond which speech is as private as it would be between separate rooms. In offices with small volume or poor privacy it can be out of reach entirely.",
          es: "Distancia de privacidad: la distancia a la que el índice de transmisión del habla cae por debajo de 0,20 y más allá de la cual el habla es tan privada como lo sería entre recintos separados. En oficinas de volumen reducido o con mala privacidad puede resultar inalcanzable.",
        },
      },
    ],
  },
  {
    id: "speech-and-intelligibility",
    label: {
      en: "Speech and intelligibility",
      es: "Habla e inteligibilidad",
    },
    terms: [
      {
        id: "m-f",
        symbol: "$m(F)$",
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: "IEC 60268-16:2020",
        guide: "perception/speech/speech-transmission",
        definition: {
          en: "Modulation transfer function: the fraction of the speech envelope modulation depth at modulation frequency $F$ that survives the transmission path.",
          es: "Función de transferencia de modulación: la fracción de la profundidad de modulación de la envolvente del habla, a la frecuencia de modulación $F$, que sobrevive al canal de transmisión.",
        },
      },
      {
        id: "sti",
        symbol: "STI",
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: {
          en: "IEC 60268-16:2020, A.5.2 to A.5.6",
          es: "IEC 60268-16:2020, A.5.2 a A.5.6",
        },
        guide: "perception/speech/speech-transmission",
        definition: {
          en: "Speech transmission index: the modulation transfer matrix converted to effective signal-to-noise ratios and weighted into a single value on 0 to 1.",
          es: "Índice de transmisión del habla: la matriz de transferencia de modulación convertida en relaciones señal-ruido efectivas y ponderada en un único valor entre 0 y 1.",
        },
      },
      {
        id: "stipa",
        symbol: "STIPA",
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: "IEC 60268-16:2020",
        clause: {
          en: "clause 6.3 and Table 3 (direct method, Annex B)",
          es: "apartado 6.3 y Tabla 3 (método directo, Anexo B)",
        },
        guide: "perception/speech/speech-transmission",
        definition: {
          en: "The direct STI measurement, made by playing a standardised two-modulation-per-band test signal through the real chain.",
          es: "La medida directa del STI, reproduciendo una señal de prueba normalizada con dos modulaciones por banda a través de la cadena real.",
        },
      },
      {
        id: "sii",
        symbol: "SII",
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: "ANSI S3.5-1997",
        clause: {
          en: "clause 6 (procedure in clause 5, importance function in Table 3)",
          es: "apartado 6 (procedimiento en el apartado 5, función de importancia en la Tabla 3)",
        },
        guide: "perception/speech/speech-intelligibility",
        definition: {
          en: "Speech intelligibility index: the band-importance-weighted audibility of the speech spectrum against noise and the listener's threshold.",
          es: "Índice de inteligibilidad del habla: la audibilidad del espectro de habla frente al ruido y al umbral del oyente, ponderada por la función de importancia de cada banda.",
        },
      },
      {
        id: "stoi",
        symbol: "STOI",
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: "Taal et al. (2011)",
        clause: {
          en: "Equations 5 and 6; no governing standard",
          es: "Ecuaciones 5 y 6; sin norma aplicable",
        },
        guide: "perception/speech/objective-intelligibility",
        definition: {
          en: "Short-time objective intelligibility: the clipped per-band envelope correlation between clean and degraded speech.",
          es: "Inteligibilidad objetiva de corta duración: la correlación recortada de las envolventes por banda entre el habla limpia y la degradada.",
        },
      },
      {
        id: "estoi",
        symbol: "ESTOI",
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: {
          en: "Jensen and Taal (2016)",
          es: "Jensen y Taal (2016)",
        },
        clause: {
          en: "Equation 8; no governing standard",
          es: "Ecuación 8; sin norma aplicable",
        },
        guide: "perception/speech/objective-intelligibility",
        definition: {
          en: "The extended measure, row- and column-normalised so that it tracks modulated maskers.",
          es: "La versión extendida, normalizada por filas y por columnas para que siga a los enmascarantes modulados.",
        },
      },
    ],
  },
  {
    id: "sound-insulation",
    label: {
      en: "Sound insulation",
      es: "Aislamiento acústico",
    },
    terms: [
      {
        id: "nr-2",
        symbol: "NR",
        qualifier: {
          en: "level drop",
          es: "caída de nivel",
        },
        unit: "dB",
        standard: {
          en: "Norton and Karczub (2003), Equation (4.101)",
          es: "Norton y Karczub (2003), ecuación (4.101)",
        },
        guide: "devices/noise-control/room-to-room",
        definition: {
          en: "Noise reduction: the sound pressure level in the source room minus the level in the receiving room, $L_{p1} - L_{p2}$. It is not the transmission loss of the partition: the two differ by a term set by the partition area against the receiving room's absorption, so a small partition into a well-absorbing room delivers more noise reduction than its transmission loss, and a large one into a hard room delivers less.",
          es: "Reducción de ruido: el nivel de presión acústica del recinto emisor menos el del receptor, $L_{p1} - L_{p2}$. No es la pérdida por transmisión del elemento separador: ambas difieren en un término fijado por la superficie del separador frente a la absorción del recinto receptor, de modo que un separador pequeño hacia un recinto muy absorbente ofrece más reducción de ruido que su pérdida por transmisión, y uno grande hacia un recinto reverberante ofrece menos.",
        },
      },
      {
        id: "d-2",
        symbol: "$D$",
        unit: "dB",
        standard: "ISO 16283-1:2014",
        clause: {
          en: "3.12 to 3.15",
          es: "3.12 a 3.15",
        },
        guide: "buildings/insulation/insulation-field",
        definition: {
          en: "Level difference: the energy-averaged source-room level minus the receiving-room level, with no normalisation.",
          es: "Diferencia de niveles: el nivel promediado energéticamente en el recinto emisor menos el del receptor, sin normalizar.",
        },
      },
      {
        id: "d-nt",
        symbol: "$D_{nT}$",
        unit: "dB",
        standard: "ISO 16283-1:2014",
        clause: {
          en: "3.12 to 3.15",
          es: "3.12 a 3.15",
        },
        guide: "buildings/insulation/insulation-field",
        definition: {
          en: "Standardized level difference: the level difference referred to a reference reverberation time, 0.5 s for dwellings.",
          es: "Diferencia de niveles estandarizada: la diferencia de niveles referida a un tiempo de reverberación de referencia, 0,5 s en viviendas.",
        },
      },
      {
        id: "d-n",
        symbol: "$D_n$",
        unit: "dB",
        standard: "ISO 10052:2021",
        guide: "buildings/insulation/insulation-survey",
        definition: {
          en: "Normalized level difference: the level difference referred to a reference absorption area of 10 m².",
          es: "Diferencia de niveles normalizada: la diferencia de niveles referida a un área de absorción de referencia de 10 m².",
        },
      },
      {
        id: "d-n-e",
        symbol: "$D_{n,e}$",
        unit: "dB",
        standard: "EN 12354-3:2000",
        guide: "buildings/design/insulation-prediction",
        definition: {
          en: "Element-normalized level difference of a small element or air path, referred to a reference area of 10 m².",
          es: "Diferencia de niveles normalizada de elemento, para un elemento pequeño o una vía de aire, referida a un área de referencia de 10 m².",
        },
      },
      {
        id: "r",
        symbol: "$R$",
        unit: "dB",
        standard: "ISO 10140-2:2010",
        guide: "buildings/insulation/insulation-lab",
        definition: {
          en: "Sound reduction index: the level difference corrected by the partition area over the receiving-room absorption area, measured in the laboratory with flanking suppressed.",
          es: "Índice de reducción acústica: la diferencia de niveles corregida por el área de la partición partido por el área de absorción del recinto receptor, medida en laboratorio con los flancos suprimidos.",
        },
      },
      {
        id: "r-2",
        symbol: "$R'$",
        unit: "dB",
        standard: "ISO 16283-1:2014",
        clause: {
          en: "3.12 to 3.15",
          es: "3.12 a 3.15",
        },
        guide: "buildings/insulation/insulation-field",
        definition: {
          en: "Apparent sound reduction index: the same construction measured in the building, so it includes every flanking path. The prime is the lab-versus-field marker.",
          es: "Índice de reducción acústica aparente: la misma construcción medida en el edificio, así que incluye todas las trayectorias por flancos. La prima es la marca que distingue el laboratorio del campo.",
        },
      },
      {
        id: "tl",
        symbol: "TL",
        qualifier: {
          en: "panel",
          es: "panel",
        },
        unit: "dB",
        standard: {
          en: "Bies, Hansen and Howard (2017)",
          es: "Bies, Hansen y Howard (2017)",
        },
        clause: {
          en: "Section 7.2; no governing standard",
          es: "Sección 7.2; sin norma aplicable",
        },
        guide: "buildings/design/panel-sound-insulation",
        definition: {
          en: "Transmission loss: the airborne insulation of a partition predicted from its physical properties, ten times the base-10 logarithm of the reciprocal transmission factor, the same quantity as $R$ in a prediction context.",
          es: "Pérdidas por transmisión: el aislamiento a ruido aéreo de un elemento separador predicho a partir de sus propiedades físicas, diez veces el logaritmo decimal del inverso del factor de transmisión, la misma magnitud que $R$ en un contexto de predicción.",
        },
      },
      {
        id: "tl-2",
        symbol: "TL",
        qualifier: {
          en: "duct element",
          es: "elemento de conducto",
        },
        unit: "dB",
        standard: {
          en: "Bies, Hansen and Howard (2017)",
          es: "Bies, Hansen y Howard (2017)",
        },
        clause: {
          en: "Sections 8.8-8.9; no governing standard",
          es: "Secciones 8.8-8.9; sin norma aplicable",
        },
        guide: "devices/noise-control/silencers",
        definition: {
          en: "Transmission loss of a duct element: ten times the base-10 logarithm of the incident plane-wave power over the power transmitted into an anechoic termination, computed from the four-pole transfer matrix and the two port impedances. The anechoic termination is part of the definition, which is why this transmission loss describes the element alone and is not the noise reduction the same silencer delivers once it is installed between a real source and a real outlet.",
          es: "Pérdidas por transmisión de un elemento de conducto: diez veces el logaritmo decimal de la potencia de onda plana incidente dividida por la potencia transmitida a una terminación anecoica, calculada con la matriz de transferencia de cuatro polos y las impedancias de los dos puertos. La terminación anecoica forma parte de la definición, y por eso estas pérdidas por transmisión describen solo al elemento y no son la reducción de ruido que ese mismo silenciador logra una vez instalado entre una fuente y una salida reales.",
        },
      },
      {
        id: "il",
        symbol: "IL",
        unit: "dB",
        standard: {
          en: "Bies, Hansen and Howard (2017)",
          es: "Bies, Hansen y Howard (2017)",
        },
        clause: {
          en: "Section 8.2, Equation (8.1); no governing standard",
          es: "Sección 8.2, ecuación (8.1); sin norma aplicable",
        },
        guide: "devices/noise-control/silencers",
        definition: {
          en: "Insertion loss: the level at a receiver before an element is inserted minus the level after, for a silencer the drop in radiated sound power level when a length of duct is replaced by it. Unlike a transmission loss it depends on the source and the termination as well as on the element, which is what makes it the number a client can hear and the transmission loss the number a catalogue can print.",
          es: "Pérdida por inserción: el nivel en un receptor antes de insertar un elemento menos el nivel después; para un silenciador, la caída del nivel de potencia acústica radiada al sustituir por él un tramo de conducto. A diferencia de las pérdidas por transmisión, depende de la fuente y de la terminación además del elemento, y eso la convierte en el número que un cliente oye y a las pérdidas por transmisión en el número que un catálogo puede imprimir.",
        },
      },
      {
        id: "tau",
        symbol: "$\\tau$",
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: "ISO 12354-1:2017",
        clause: {
          en: "Formula (1)",
          es: "Fórmula (1)",
        },
        guide: "buildings/design/detailed-prediction",
        definition: {
          en: "Transmission factor, or transmission coefficient: the fraction of the incident sound power a partition passes on. It is the quantity the whole group is a logarithm of, since $R = -10\\log_{10}\\tau$ and the transmission loss is the same logarithm; a $\\tau$ of $10^{-5}$ is a 50 dB partition. Transmission factors of parallel paths add, which is why a prediction sums the direct, flanking and indirect airborne factors and only then takes the logarithm.",
          es: "Factor de transmisión, o coeficiente de transmisión: la fracción de la potencia acústica incidente que deja pasar un elemento separador. Es la magnitud de la que es logaritmo todo el grupo, ya que $R = -10\\log_{10}\\tau$ y las pérdidas por transmisión son ese mismo logaritmo; un $\\tau$ de $10^{-5}$ es un separador de 50 dB. Los factores de transmisión de caminos en paralelo se suman, y por eso una predicción suma los factores directo, por flancos e indirecto por vía aérea y solo después toma el logaritmo.",
        },
      },
      {
        id: "r-w-r-w-d-nt-w",
        symbol: "$R_w$, $R'_w$, $D_{nT,w}$",
        unit: "dB",
        standard: "ISO 717-1:2020",
        guide: "buildings/insulation/insulation-ratings",
        definition: {
          en: "The weighted single-number ratings: a fixed reference curve is shifted toward the measured spectrum until the unfavourable deviations reach their allowed sum, and the shifted curve is read at 500 Hz.",
          es: "Los índices globales ponderados: una curva de referencia fija se desplaza hacia el espectro medido hasta que las desviaciones desfavorables alcanzan su suma admisible, y la curva desplazada se lee en 500 Hz.",
        },
      },
      {
        id: "d-n-e-w",
        symbol: "$D_{n,e,w}$",
        unit: "dB",
        standard: "ISO 717-1:2020",
        guide: "buildings/design/insulation-prediction",
        definition: {
          en: "The same reference-curve rating applied to the element-normalized level difference.",
          es: "El mismo índice global de curva de referencia aplicado a la diferencia de niveles normalizada de elemento.",
        },
      },
      {
        id: "c-c-tr",
        symbol: "$C$, $C_{tr}$",
        unit: "dB",
        standard: "ISO 717-1:2020",
        clause: {
          en: "Annex A",
          es: "Anexo A",
        },
        guide: "buildings/insulation/insulation-ratings",
        definition: {
          en: "Spectrum adaptation terms: the corrections that re-rate the measured curve against A-weighted pink noise ($C$) and against A-weighted urban road traffic ($C_{tr}$).",
          es: "Términos de adaptación espectral: las correcciones que vuelven a valorar la curva medida frente a ruido rosa ponderado A ($C$) y frente a tráfico rodado urbano ponderado A ($C_{tr}$).",
        },
      },
      {
        id: "l-n",
        symbol: "$L_n$",
        unit: "dB re 20 µPa",
        standard: "ISO 10140-3:2010",
        guide: "buildings/insulation/insulation-lab",
        definition: {
          en: "Normalized impact sound pressure level: the receiving-room level under the standard tapping machine, referred to a 10 m² absorption area.",
          es: "Nivel de presión acústica de impactos normalizado: el nivel en el recinto receptor bajo la máquina de impactos normalizada, referido a un área de absorción de 10 m².",
        },
      },
      {
        id: "l-nt",
        symbol: "$L'_{nT}$",
        unit: "dB re 20 µPa",
        standard: "ISO 16283-2:2015",
        guide: "buildings/insulation/insulation-field",
        definition: {
          en: "Standardized impact sound pressure level, referred to a reference reverberation time. Note the sign: more reverberation lowers it, the opposite of $D_{nT}$.",
          es: "Nivel de presión acústica de impactos estandarizado, referido a un tiempo de reverberación de referencia. Atención al signo: más reverberación lo baja, al revés que el $D_{nT}$.",
        },
      },
      {
        id: "l-n-w-l-nt-w",
        symbol: "$L_{n,w}$, $L'_{nT,w}$",
        unit: "dB",
        standard: "ISO 717-2:2020",
        guide: "buildings/insulation/insulation-ratings",
        definition: {
          en: "The weighted impact ratings. The reference curve is shifted the same way, but an unfavourable deviation is now one where the measurement exceeds the reference.",
          es: "Los índices globales de impactos. La curva de referencia se desplaza igual, pero ahora una desviación desfavorable es aquella en la que la medida supera a la referencia.",
        },
      },
      {
        id: "c-i",
        symbol: "$C_I$",
        unit: "dB",
        standard: {
          en: "ISO 717-2:2020 (enlarged range in A.2.1 NOTE)",
          es: "ISO 717-2:2020 (rango ampliado en la NOTA de A.2.1)",
        },
        guide: "buildings/insulation/insulation-ratings",
        definition: {
          en: "Impact spectrum adaptation term, from the energetic sum over 100 Hz to 2500 Hz. The enlarged-range $C_{I,50\\text{–}2500}$ extends it down to 50 Hz.",
          es: "Término de adaptación espectral de impactos, a partir de la suma energética entre 100 Hz y 2500 Hz. El término de rango ampliado $C_{I,50\\text{–}2500}$ lo extiende hasta 50 Hz.",
        },
      },
      {
        id: "delta-l-w",
        symbol: "$\\Delta L_w$",
        unit: "dB",
        standard: {
          en: "ISO 717-2:2020 (measurement in ISO 16251-1:2014, Formulae (3) and (4))",
          es: "ISO 717-2:2020 (medición en ISO 16251-1:2014, Fórmulas (3) y (4))",
        },
        guide: "buildings/design/impact-improvement",
        definition: {
          en: "Weighted reduction of impact sound pressure level given by a floor covering, measured as the improvement over the bare reference floor.",
          es: "Reducción ponderada del nivel de presión acústica de impactos que aporta un revestimiento de suelo, medida como la mejora sobre el forjado desnudo de referencia.",
        },
      },
      {
        id: "delta-r-w",
        symbol: "$\\Delta R_w$",
        unit: "dB",
        standard: "EN 12354-1:2000",
        clause: {
          en: "Formulae 27 and 28a",
          es: "Fórmulas 27 y 28a",
        },
        guide: "buildings/design/insulation-prediction",
        definition: {
          en: "Weighted improvement of airborne insulation contributed by a lining or additional layer, added to the element rating in the prediction.",
          es: "Mejora ponderada del aislamiento a ruido aéreo que aporta un trasdosado o una capa adicional, que se suma al índice del elemento en la predicción.",
        },
      },
      {
        id: "k-ij",
        symbol: "$K_{ij}$",
        unit: "dB",
        standard: "ISO 10848-1:2006",
        clause: {
          en: "Formula (13)",
          es: "Fórmula (13)",
        },
        guide: "buildings/insulation/flanking-lab",
        definition: {
          en: "Vibration reduction index of a junction: the direction-averaged velocity level difference corrected by the junction length and the equivalent absorption lengths.",
          es: "Índice de reducción vibratoria de una unión: la diferencia de niveles de velocidad promediada en ambos sentidos, corregida por la longitud de la unión y las longitudes de absorción equivalentes.",
        },
      },
      {
        id: "f-c",
        symbol: "$f_c$",
        unit: "Hz",
        standard: {
          en: "Bies, Hansen and Howard (2017)",
          es: "Bies, Hansen y Howard (2017)",
        },
        clause: {
          en: "Equation 7.3; no governing standard",
          es: "Ecuación 7.3; sin norma aplicable",
        },
        guide: "buildings/design/panel-sound-insulation",
        definition: {
          en: "Critical frequency: the frequency at which the bending wavelength of a panel equals the wavelength in air, where the coincidence dip appears.",
          es: "Frecuencia crítica: aquella en la que la longitud de onda de flexión del panel iguala a la del aire, donde aparece la caída por coincidencia.",
        },
      },
      {
        id: "sigma",
        symbol: "$\\sigma$",
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: "Hopkins (2007)",
        clause: {
          en: "Equations 2.227 to 2.230; no governing standard",
          es: "Ecuaciones 2.227 a 2.230; sin norma aplicable",
        },
        guide: "buildings/design/panel-sound-insulation",
        definition: {
          en: "Radiation efficiency of a plate: the airborne power radiated per unit mean-square surface velocity, normalised by the plane-wave value.",
          es: "Eficiencia de radiación de una placa: la potencia aérea radiada por unidad de velocidad cuadrática media de la superficie, normalizada por el valor de onda plana.",
        },
      },
      {
        id: "d-2m-nt",
        symbol: "$D_{2m,nT}$",
        unit: "dB",
        standard: "ISO 16283-3:2016",
        clause: "3.15",
        guide: "buildings/insulation/facade-insulation",
        definition: {
          en: "Standardized facade level difference: the level 2 m in front of the facade minus the indoor level, standardized to a reference reverberation time of 0,5 s for dwellings. The 2 m position sits in the field the facade itself reflects, so it is not a free-field level, and the notation records the source — $D_{tr,2m,nT}$ for road traffic, $D_{ls,2m,nT}$ for a loudspeaker.",
          es: "Diferencia de niveles estandarizada en fachada: el nivel a 2 m por delante de la fachada menos el nivel interior, estandarizado a un tiempo de reverberación de referencia de 0,5 s en viviendas. La posición a 2 m está en el campo que la propia fachada refleja, así que no es un nivel de campo libre, y la notación indica la fuente: $D_{tr,2m,nT}$ para tráfico rodado y $D_{ls,2m,nT}$ para altavoz.",
        },
      },
      {
        id: "d-ls-2m-nt-w",
        symbol: "$D_{ls,2m,nT,w}$",
        unit: "dB",
        standard: "ISO 717-1:2020",
        guide: "buildings/insulation/facade-insulation",
        definition: {
          en: "The weighted facade rating: $D_{2m,nT}$ measured with a loudspeaker put through the ISO 717-1 reference-curve procedure. The `ls` subscript is not decoration — it records that the source was a loudspeaker at 45 degrees rather than real road traffic, and a facade rated with one source is not interchangeable with the same facade rated with the other.",
          es: "El índice global de fachada: el $D_{2m,nT}$ medido con altavoz pasado por el procedimiento de curva de referencia de la ISO 717-1. El subíndice `ls` no es decorativo: indica que la fuente fue un altavoz a 45 grados y no tráfico rodado real, y una fachada calificada con una fuente no es intercambiable con esa misma fachada calificada con la otra.",
        },
      },
      {
        id: "r-45",
        symbol: "$R'_{45°}$",
        unit: "dB",
        standard: "ISO 16283-3:2016",
        clause: "3.12",
        guide: "buildings/insulation/facade-insulation",
        definition: {
          en: "Apparent sound reduction index of a facade element under loudspeaker sound at 45 degrees: the level difference corrected by the specimen area over the receiving-room absorption area, with a further $-1{,}5$ dB that carries the single-angle geometry. It is apparent, so flanking and any other component of the facade are inside the number.",
          es: "Índice de reducción acústica aparente de un elemento de fachada con altavoz a 45 grados: la diferencia de niveles corregida por el área del elemento frente al área de absorción del recinto receptor, con un término adicional de $-1{,}5$ dB que recoge la geometría de ángulo único. Es aparente, así que las transmisiones por flancos y cualquier otro componente de la fachada quedan dentro del número.",
        },
      },
      {
        id: "r-a",
        symbol: "$R_A$, $R_{A,tr}$",
        unit: "dBA",
        standard: {
          en: "CTE DB-HR",
          es: "CTE DB-HR",
        },
        clause: {
          en: "Annex A, Formulae (A.5) and (A.6)",
          es: "Anejo A, fórmulas (A.5) y (A.6)",
        },
        guide: "buildings/insulation/spanish-building-code",
        definition: {
          en: "Global airborne index: the A-weighted level a partition transmits when it is excited by a normalised spectrum — pink noise for $R_A$, road traffic for $R_{A,tr}$ — summed energetically over eighteen one-third-octave bands from 100 Hz to 5 kHz. It is a close relative of $R_w + C$ and $R_w + C_{tr}$ but is computed directly, not by shifting a reference curve, and it uses two bands the ISO 717-1 range does not.",
          es: "Índice global de reducción acústica: el nivel ponderado A que transmite un elemento cuando se excita con un espectro normalizado —ruido rosa para $R_A$ y tráfico rodado para $R_{A,tr}$— sumado energéticamente en dieciocho bandas de tercio de octava de 100 Hz a 5 kHz. Es pariente cercano de $R_w + C$ y $R_w + C_{tr}$, pero se calcula directamente y no desplazando una curva de referencia, y utiliza dos bandas que el intervalo de la ISO 717-1 no incluye.",
        },
      },
      {
        id: "d-nt-a",
        symbol: "$D_{nT,A}$, $D_{2m,nT,Atr}$",
        unit: "dBA",
        standard: {
          en: "CTE DB-HR",
          es: "CTE DB-HR",
        },
        clause: {
          en: "Annex A, Formulae (A.5) and (A.6); requirements in clause 2",
          es: "Anejo A, fórmulas (A.5) y (A.6); exigencias del capítulo 2",
        },
        guide: "buildings/insulation/spanish-building-code",
        definition: {
          en: "The same global index applied to a standardized level difference: between two rooms ($D_{nT,A}$, pink noise) and between the outside and a protected room ($D_{2m,nT,Atr}$, road traffic or aircraft). These are the quantities the DB-HR requirement tables are written against, so a project is checked in them and not in the ISO 717-1 ratings.",
          es: "El mismo índice global aplicado a una diferencia de niveles estandarizada: entre dos recintos ($D_{nT,A}$, ruido rosa) y entre el exterior y un recinto protegido ($D_{2m,nT,Atr}$, tráfico rodado o aeronaves). Son las magnitudes en las que están escritas las tablas de exigencias del DB-HR, de modo que un proyecto se comprueba en ellas y no en los índices de la ISO 717-1.",
        },
      },
      {
        id: "r-i",
        symbol: "$R_I$",
        unit: "dB",
        standard: "ISO 15186-1:2000",
        clause: {
          en: "3.8, Equation (7)",
          es: "3.8, ecuación (7)",
        },
        guide: "buildings/insulation/insulation-intensity",
        definition: {
          en: "Intensity sound reduction index: the source-room level minus the intensity level scanned over the radiating face, so the transmitted power is measured directly instead of inferred from the receiving room. It is the method of choice where flanking is strong; add the $K_c$ adaptation to get $R_{I,M}$, the value the ISO 10140 pressure method would have produced.",
          es: "Índice de reducción acústica por intensidad: el nivel del recinto emisor menos el nivel de intensidad barrido sobre la cara radiante, de modo que la potencia transmitida se mide directamente en vez de deducirse del recinto receptor. Es el método preferente cuando la transmisión por flancos es fuerte; con la adaptación $K_c$ se obtiene $R_{I,M}$, el valor que habría dado el método de presión de la ISO 10140.",
        },
      },
      {
        id: "d-i-n-e",
        symbol: "$D_{I,n,e}$",
        unit: "dB",
        standard: "ISO 15186-2:2003",
        clause: {
          en: "Formula (12)",
          es: "Fórmula (12)",
        },
        guide: "buildings/insulation/insulation-intensity",
        definition: {
          en: "Intensity element-normalized level difference: the small-element counterpart of $D_{n,e}$, measured by scanning the element and normalised to a reference absorption area of 10 m². It is rated through the same ISO 717-1 procedure, as $D_{I,n,e,w}$, so a ventilator or a transit sealing system can be compared with a wall on one scale.",
          es: "Diferencia de niveles normalizada de elemento por intensidad: la contrapartida para elementos pequeños de $D_{n,e}$, medida barriendo el elemento y normalizada a un área de absorción de referencia de 10 m². Se califica con el mismo procedimiento de la ISO 717-1, como $D_{I,n,e,w}$, de modo que un aireador o un sistema de sellado de pasos puede compararse con un muro en una única escala.",
        },
      },
    ],
  },
  {
    id: "materials-and-surfaces",
    label: {
      en: "Materials and surfaces",
      es: "Materiales y superficies",
    },
    terms: [
      {
        id: "alpha",
        symbol: "$\\alpha$",
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: "ISO 10534-2:1998",
        clause: {
          en: "Equations (17) to (19)",
          es: "Ecuaciones (17) a (19)",
        },
        guide: "materials/absorbers/impedance-tube",
        definition: {
          en: "Sound absorption coefficient at normal incidence: the fraction of incident energy not returned by the surface, obtained in the impedance tube from the reflection factor.",
          es: "Coeficiente de absorción acústica a incidencia normal: la fracción de energía incidente que la superficie no devuelve, obtenida en el tubo de impedancia a partir del factor de reflexión.",
        },
      },
      {
        id: "alpha-s",
        symbol: "$\\alpha_s$",
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: "ISO 354:2003",
        clause: {
          en: "Equations (8) and (9)",
          es: "Ecuaciones (8) y (9)",
        },
        guide: "materials/absorbers/absorption-measurement",
        definition: {
          en: "Random-incidence sound absorption coefficient measured in a reverberation room, from the change in equivalent absorption area with and without the specimen.",
          es: "Coeficiente de absorción acústica a incidencia aleatoria medido en cámara reverberante, a partir del cambio del área de absorción equivalente con y sin la muestra.",
        },
      },
      {
        id: "alpha-p",
        symbol: "$\\alpha_p$",
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: "ISO 11654:1997",
        clause: {
          en: "Clause 4.1",
          es: "apartado 4.1",
        },
        guide: "materials/absorbers/absorption-measurement",
        definition: {
          en: "Practical sound absorption coefficient: the one-third-octave data grouped into octave bands and rounded to steps of 0.05.",
          es: "Coeficiente de absorción acústica práctico: los datos en tercios de octava agrupados en bandas de octava y redondeados a pasos de 0,05.",
        },
      },
      {
        id: "alpha-w",
        symbol: "$\\alpha_w$",
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: "ISO 11654:1997",
        clause: {
          en: "Clause 4.2",
          es: "apartado 4.2",
        },
        guide: "materials/absorbers/absorption-measurement",
        definition: {
          en: "Weighted sound absorption coefficient: the fixed reference curve shifted toward the practical values and read at 500 Hz.",
          es: "Coeficiente de absorción acústica ponderado: la curva de referencia fija desplazada hacia los valores prácticos y leída en 500 Hz.",
        },
      },
      {
        id: "absorption-class",
        name: {
          en: "Absorption class",
          es: "Clase de absorción",
        },
        unit: {
          en: "class letter",
          es: "letra de clase",
        },
        standard: "ISO 11654:1997",
        clause: {
          en: "Table B.1",
          es: "Tabla B.1",
        },
        guide: "materials/absorbers/absorption-measurement",
        definition: {
          en: "The A to E letter class the weighted coefficient maps to, or \"not classified\".",
          es: "La clase, de la A a la E, a la que se asigna el coeficiente ponderado, o «sin clasificar».",
        },
      },
      {
        id: "r-3",
        symbol: "$R$",
        unit: "Pa·s/m³",
        standard: "ISO 9053-1:2018",
        clause: {
          en: "Clause 3",
          es: "apartado 3",
        },
        guide: "materials/absorbers/airflow-resistance",
        definition: {
          en: "Airflow resistance: the pressure difference across a specimen divided by the volumetric airflow rate through it.",
          es: "Resistencia al flujo de aire: la diferencia de presión a través de una probeta dividida por el caudal volumétrico que la atraviesa.",
        },
      },
      {
        id: "r-s",
        symbol: "$R_s$",
        unit: "Pa·s/m",
        standard: "ISO 9053-1:2018",
        clause: {
          en: "Clause 3",
          es: "apartado 3",
        },
        guide: "materials/absorbers/airflow-resistance",
        definition: {
          en: "Specific airflow resistance: the airflow resistance referred to the specimen face area.",
          es: "Resistencia específica al flujo de aire: la resistencia al flujo referida al área de la cara de la probeta.",
        },
      },
      {
        id: "sigma-2",
        symbol: "$\\sigma$",
        unit: "Pa·s/m²",
        standard: "ISO 9053-1:2018",
        clause: {
          en: "Clause 3",
          es: "apartado 3",
        },
        guide: "materials/absorbers/airflow-resistance",
        definition: {
          en: "Airflow resistivity: the specific airflow resistance per unit thickness, the primary input to every empirical porous model.",
          es: "Resistividad al flujo de aire: la resistencia específica por unidad de espesor, la entrada principal de todo modelo poroso empírico.",
        },
      },
      {
        id: "z",
        symbol: "$Z$",
        unit: "Pa·s/m",
        standard: "ISO 10534-2:1998",
        clause: {
          en: "Equations (17) to (19)",
          es: "Ecuaciones (17) a (19)",
        },
        guide: "materials/absorbers/impedance-tube",
        definition: {
          en: "Surface impedance: the complex ratio of sound pressure to particle velocity at the face of the sample, usually reported normalised by the characteristic impedance of air.",
          es: "Impedancia superficial: la relación compleja entre presión acústica y velocidad de partícula en la cara de la muestra, que suele darse normalizada por la impedancia característica del aire.",
        },
      },
      {
        id: "s",
        symbol: "$s$",
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: "ISO 17497-1:2004+A1:2014",
        clause: {
          en: "Formula (5)",
          es: "Fórmula (5)",
        },
        guide: "materials/diffusers/diffusers",
        definition: {
          en: "Scattering coefficient: the fraction of reflected energy that is not returned specularly, measured at random incidence on a turntable in a reverberation room.",
          es: "Coeficiente de dispersión: la fracción de energía reflejada que no vuelve de forma especular, medida a incidencia aleatoria sobre una plataforma giratoria en cámara reverberante.",
        },
      },
      {
        id: "d-3",
        symbol: "$d$",
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: "ISO 17497-2:2012",
        clause: {
          en: "Formula (5) (normalised form in Formula (7))",
          es: "Fórmula (5) (forma normalizada en la Fórmula (7))",
        },
        guide: "materials/diffusers/diffusers",
        definition: {
          en: "Diffusion coefficient: the uniformity of the polar response of a surface, from the autocorrelation of the free-field goniometer measurement.",
          es: "Coeficiente de difusión: la uniformidad de la respuesta polar de una superficie, a partir de la autocorrelación de la medida en goniómetro de campo libre.",
        },
      },
      {
        id: "s-2",
        symbol: "$s'$",
        unit: "MN/m³",
        standard: "EN 29052-1:1992 (ISO 9052-1:1989)",
        clause: {
          en: "Formula 1",
          es: "Fórmula 1",
        },
        guide: "materials/resilient/dynamic-stiffness",
        definition: {
          en: "Dynamic stiffness per unit area of a resilient layer: a dynamic force per unit area divided by the resulting change in thickness.",
          es: "Rigidez dinámica por unidad de superficie de una capa elástica: una fuerza dinámica por unidad de superficie dividida por la variación de espesor que provoca.",
        },
      },
      {
        id: "phi",
        symbol: "$\\phi$",
        qualifier: {
          en: "porosity",
          es: "porosidad",
        },
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: {
          en: "Allard and Atalla (2009)",
          es: "Allard y Atalla (2009)",
        },
        clause: {
          en: "Equation (2.25); no governing standard",
          es: "Ecuación (2.25); sin norma aplicable",
        },
        guide: "materials/absorbers/porous-absorbers",
        definition: {
          en: "Open porosity: the fraction of the material's volume that is air connected to the outside, $\\phi = V_a/V_T$. A closed bubble counts as frame, not as pore, because sound cannot enter it; for most fibrous materials and foams $\\phi$ lies very close to one, which is why a one-parameter model that assumes so can still work.",
          es: "Porosidad abierta: la fracción del volumen del material que es aire conectado con el exterior, $\\phi = V_a/V_T$. Una burbuja cerrada cuenta como esqueleto y no como poro, porque el sonido no puede entrar en ella; en la mayoría de los materiales fibrosos y de las espumas $\\phi$ es muy próxima a uno, y por eso un modelo de un solo parámetro que lo dé por supuesto puede seguir funcionando.",
        },
      },
      {
        id: "alpha-infty",
        symbol: "$\\alpha_\\infty$",
        qualifier: {
          en: "tortuosity",
          es: "tortuosidad",
        },
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: {
          en: "Allard and Atalla (2009)",
          es: "Allard y Atalla (2009)",
        },
        clause: {
          en: "Equation (4.143); no governing standard",
          es: "Ecuación (4.143); sin norma aplicable",
        },
        guide: "materials/absorbers/porous-absorbers",
        definition: {
          en: "Tortuosity: how much longer the winding path through the pores is than the straight line, squared — exactly $1/\\cos^2\\varphi$ for pores inclined at an angle $\\varphi$, and one for straight cylindrical pores. It sets the high-frequency limit of the effective density, and despite the $\\alpha$ it is not an absorption coefficient of any kind.",
          es: "Tortuosidad: cuánto más largo es el camino sinuoso a través de los poros que la línea recta, al cuadrado; exactamente $1/\\cos^2\\varphi$ para poros inclinados un ángulo $\\varphi$, y uno para poros cilíndricos rectos. Fija el límite en alta frecuencia de la densidad efectiva y, pese a la $\\alpha$, no es un coeficiente de absorción de ninguna clase.",
        },
      },
      {
        id: "lambda-visc-therm",
        symbol: "$\\Lambda$, $\\Lambda'$",
        unit: "m",
        standard: {
          en: "Allard and Atalla (2009)",
          es: "Allard y Atalla (2009)",
        },
        clause: {
          en: "Equations (5.24) and (5.27); no governing standard",
          es: "Ecuaciones (5.24) y (5.27); sin norma aplicable",
        },
        guide: "materials/absorbers/porous-absorbers",
        definition: {
          en: "The two pore sizes the Johnson-Champoux-Allard model needs: $\\Lambda$ weights the pore surface by the squared flow velocity, so it is set by the narrow constrictions where the viscous losses happen, and $\\Lambda'$ is the plain surface-to-volume length that governs the thermal exchange with the frame. $\\Lambda'$ is normally the larger, and the two are equal only for identical straight cylindrical pores.",
          es: "Los dos tamaños de poro que necesita el modelo de Johnson-Champoux-Allard: $\\Lambda$ pondera la superficie del poro por el cuadrado de la velocidad del flujo, así que lo fijan los estrechamientos donde se producen las pérdidas viscosas, y $\\Lambda'$ es la longitud superficie-volumen sin ponderar que gobierna el intercambio térmico con el esqueleto. $\\Lambda'$ suele ser la mayor, y ambas coinciden solo con poros cilíndricos rectos idénticos.",
        },
      },
      {
        id: "z-c-k",
        symbol: "$Z_c$, $k$",
        unit: {
          en: "Pa·s/m and 1/m",
          es: "Pa·s/m y 1/m",
        },
        standard: {
          en: "Allard and Atalla (2009)",
          es: "Allard y Atalla (2009)",
        },
        clause: {
          en: "Chapter 5; no governing standard",
          es: "Capítulo 5; sin norma aplicable",
        },
        guide: "materials/absorbers/porous-absorbers",
        definition: {
          en: "Characteristic impedance and complex wavenumber: the pair that describes a porous medium as an equivalent fluid — the ratio of pressure to particle velocity in a travelling wave inside it, and the wavenumber whose imaginary part is the attenuation per metre. Every empirical or semi-phenomenological model produces this pair, and it is what the transfer-matrix method stacks layer by layer.",
          es: "Impedancia característica y número de onda complejo: el par que describe un medio poroso como fluido equivalente, es decir, la relación entre presión y velocidad de partícula en una onda progresiva dentro de él y el número de onda cuya parte imaginaria es la atenuación por metro. Todo modelo empírico o semifenomenológico produce este par, y es lo que el método de matrices de transferencia apila capa a capa.",
        },
      },
    ],
  },
  {
    id: "vibration-and-structure-borne-sound",
    label: {
      en: "Vibration and structure-borne sound",
      es: "Vibración y ruido estructural",
    },
    terms: [
      {
        id: "y",
        symbol: "$Y$",
        unit: "m/(N·s)",
        standard: "ISO 7626-1:2011",
        clause: {
          en: "3.1.2 and Table 1",
          es: "3.1.2 y Tabla 1",
        },
        guide: "vibration/structural/mechanical-mobility",
        definition: {
          en: "Mobility: the complex ratio of a velocity response to the force that produces it.",
          es: "Movilidad: la relación compleja entre una respuesta en velocidad y la fuerza que la produce.",
        },
      },
      {
        id: "z-2",
        symbol: "$Z$",
        unit: "N·s/m",
        standard: "ISO 7626-1:2011",
        clause: {
          en: "Table 1",
          es: "Tabla 1",
        },
        guide: "vibration/structural/mechanical-mobility",
        definition: {
          en: "Mechanical impedance: the reciprocal of mobility, force per unit velocity.",
          es: "Impedancia mecánica: la recíproca de la movilidad, fuerza por unidad de velocidad.",
        },
      },
      {
        id: "h",
        symbol: "$H$",
        qualifier: {
          en: "receptance",
          es: "receptancia",
        },
        unit: "m/N",
        standard: "ISO 7626-1:2011",
        clause: {
          en: "Table 1",
          es: "Tabla 1",
        },
        guide: "vibration/structural/mechanical-mobility",
        definition: {
          en: "Receptance, or dynamic compliance: displacement response per unit force, the pivot the whole family converts through.",
          es: "Receptancia, o flexibilidad dinámica: respuesta en desplazamiento por unidad de fuerza, el pivote por el que convierte toda la familia.",
        },
      },
      {
        id: "a-2",
        symbol: "$A$",
        unit: "1/kg",
        standard: "ISO 7626-1:2011",
        clause: {
          en: "Table 1",
          es: "Tabla 1",
        },
        guide: "vibration/structural/mechanical-mobility",
        definition: {
          en: "Accelerance, or inertance: acceleration response per unit force. Its reciprocal is the apparent mass.",
          es: "Acelerancia, o inertancia: respuesta en aceleración por unidad de fuerza. Su recíproca es la masa aparente.",
        },
      },
      {
        id: "k-21",
        symbol: "$k_{21}$",
        unit: "N/m",
        standard: "ISO 10846-1:2008",
        clause: "3.7",
        guide: "vibration/structural/transfer-stiffness",
        definition: {
          en: "Dynamic transfer stiffness of a resilient element: the blocking force on the output side divided by the displacement on the input side.",
          es: "Rigidez dinámica de transferencia de un elemento resiliente: la fuerza bloqueada del lado de salida dividida por el desplazamiento del lado de entrada.",
        },
      },
      {
        id: "l-k",
        symbol: "$L_k$",
        unit: "dB re 1 N/m",
        standard: {
          en: "ISO 10846-2:2008 and ISO 10846-3:2002",
          es: "ISO 10846-2:2008 e ISO 10846-3:2002",
        },
        clause: "3.17",
        guide: "vibration/structural/transfer-stiffness",
        definition: {
          en: "Level of the dynamic transfer stiffness, referred to 1 N/m.",
          es: "Nivel de la rigidez dinámica de transferencia, referido a 1 N/m.",
        },
      },
      {
        id: "eta",
        symbol: "$\\eta$",
        qualifier: {
          en: "resilient element",
          es: "elemento resiliente",
        },
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: "ISO 10846-1:2008",
        clause: "3.8",
        guide: "vibration/structural/transfer-stiffness",
        definition: {
          en: "Loss factor of a resilient element: the tangent of the phase angle of its dynamic transfer stiffness.",
          es: "Factor de pérdidas de un elemento resiliente: la tangente del ángulo de fase de su rigidez dinámica de transferencia.",
        },
      },
      {
        id: "eta-int",
        symbol: "$\\eta_{int}$",
        qualifier: {
          en: "internal",
          es: "interno",
        },
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: "ISO 12354-1:2017",
        clause: {
          en: "Annex C, Formula (C.1)",
          es: "Anexo C, fórmula (C.1)",
        },
        guide: "buildings/design/detailed-prediction",
        definition: {
          en: "Internal loss factor of a building element: the fraction of its vibrational energy dissipated per radian inside the material itself, before anything is radiated or leaves through the junctions. It is an input to the prediction, not a measured output: Table B.3 of the same standard gives 0,005 for concrete and about 0,01 for most masonry.",
          es: "Factor de pérdidas interno de un elemento constructivo: la fracción de su energía vibratoria que se disipa por radián dentro del propio material, antes de radiarse o de escapar por las uniones. Es un dato de entrada de la predicción, no un resultado medido: la tabla B.3 de la misma norma da 0,005 para el hormigón y en torno a 0,01 para la mayoría de las fábricas.",
        },
      },
      {
        id: "eta-tot",
        symbol: "$\\eta_{tot}$",
        qualifier: {
          en: "total, in situ",
          es: "total, in situ",
        },
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: "ISO 12354-1:2017",
        clause: {
          en: "Annex C, Formula (C.1)",
          es: "Anexo C, fórmula (C.1)",
        },
        guide: "buildings/design/detailed-prediction",
        definition: {
          en: "Total loss factor of an element as built in: the internal loss factor plus the losses radiated into the air and absorbed at the perimeter junctions. It is what damps the resonant transmission above the critical frequency, and it fixes the structural reverberation time through $T_s = 2{,}2/(f\\,\\eta_{tot})$ — so a laboratory value measured in a heavy test frame does not transfer to the building unchanged.",
          es: "Factor de pérdidas total de un elemento ya construido: el factor de pérdidas interno más las pérdidas radiadas al aire y absorbidas en las uniones del perímetro. Es lo que amortigua la transmisión resonante por encima de la frecuencia crítica y fija el tiempo de reverberación estructural mediante $T_s = 2{,}2/(f\\,\\eta_{tot})$, de modo que un valor de laboratorio medido en un marco de ensayo pesado no se traslada al edificio sin más.",
        },
      },
      {
        id: "a-w",
        symbol: "$a_w$",
        unit: "m/s²",
        standard: "ISO 2631-1:1997",
        clause: {
          en: "Equation (9)",
          es: "Ecuación (9)",
        },
        guide: "vibration/human/human-vibration",
        definition: {
          en: "Frequency-weighted acceleration: the root sum of squares of the band accelerations after the human-response weightings.",
          es: "Aceleración ponderada en frecuencia: la raíz de la suma de cuadrados de las aceleraciones por banda tras aplicar las ponderaciones de respuesta humana.",
        },
      },
      {
        id: "a-8",
        symbol: "$A(8)$",
        unit: "m/s²",
        standard: "ISO 5349-1:2001",
        clause: {
          en: "Equations (2) and (3)",
          es: "Ecuaciones (2) y (3)",
        },
        guide: "vibration/human/human-vibration",
        definition: {
          en: "Daily vibration exposure: the exposure magnitude normalised to a reference 8 h day, combined over the operations of the day.",
          es: "Exposición diaria a vibración: la magnitud de exposición normalizada a una jornada de referencia de 8 h, combinada sobre las operaciones del día.",
        },
      },
      {
        id: "vdv",
        symbol: "VDV",
        unit: {
          en: "m/s^1.75",
          es: "m/s^1,75",
        },
        standard: "ISO 2631-1:1997",
        clause: {
          en: "Equation (5)",
          es: "Ecuación (5)",
        },
        guide: "vibration/human/human-vibration",
        definition: {
          en: "Vibration dose value: the fourth-power time integral of the weighted acceleration, which weights shocks far more heavily than an r.m.s. does.",
          es: "Valor de dosis de vibración: la integral temporal de la cuarta potencia de la aceleración ponderada, que pesa los choques mucho más que un valor eficaz.",
        },
      },
      {
        id: "mtvv",
        symbol: "MTVV",
        unit: "m/s²",
        standard: "ISO 2631-1:1997",
        clause: {
          en: "Equation (4)",
          es: "Ecuación (4)",
        },
        guide: "vibration/human/human-vibration",
        definition: {
          en: "Maximum transient vibration value: the largest 1 s running r.m.s. of the weighted acceleration.",
          es: "Valor máximo de vibración transitoria: el mayor valor eficaz corrido de 1 s de la aceleración ponderada.",
        },
      },
      {
        id: "r-4",
        symbol: "$R$",
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: "ISO 2631-5:2018",
        clause: {
          en: "Annex C (Formulae C.1 and C.3 to C.5)",
          es: "Anexo C (Fórmulas C.1 y C.3 a C.5)",
        },
        guide: "vibration/human/multiple-shock-vibration",
        definition: {
          en: "Cumulative stress variable of the multiple-shock model: the daily compressive stresses accumulated over the years of exposure, which the lumbar injury probability is read from.",
          es: "Variable de tensión acumulada del modelo de choques múltiples: las tensiones de compresión diarias acumuladas a lo largo de los años de exposición, de la que se lee la probabilidad de lesión lumbar.",
        },
      },
      {
        id: "l-v",
        symbol: "$L_v$",
        unit: "dB re 50 nm/s",
        standard: "ISO/TS 7849-1:2009",
        clause: {
          en: "Formula 3",
          es: "Fórmula 3",
        },
        guide: "devices/emission/vibration-sound-power",
        definition: {
          en: "Velocity level: twenty times the base-10 logarithm of the surface velocity over the reference velocity.",
          es: "Nivel de velocidad: veinte veces el logaritmo decimal de la velocidad de la superficie dividida por la velocidad de referencia.",
        },
      },
      {
        id: "varepsilon",
        symbol: "$\\varepsilon$",
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: {
          en: "ISO/TS 7849-1:2009 and ISO/TS 7849-2:2009",
          es: "ISO/TS 7849-1:2009 e ISO/TS 7849-2:2009",
        },
        guide: "devices/emission/vibration-sound-power",
        definition: {
          en: "Radiation factor, or radiation efficiency, of a vibrating machine surface: the airborne power radiated per unit mean-square velocity and area.",
          es: "Factor de radiación, o eficiencia de radiación, de la superficie vibrante de una máquina: la potencia aérea radiada por unidad de velocidad cuadrática media y de área.",
        },
      },
      {
        id: "l-ws",
        symbol: "$L_{W\mathrm{s}}$",
        unit: "dB re 1 pW",
        standard: "EN 15657:2018",
        clause: {
          en: "Formula 14",
          es: "Fórmula 14",
        },
        guide: "buildings/design/structure-borne-power",
        definition: {
          en: "Structure-borne sound power level injected by equipment into a reception plate.",
          es: "Nivel de potencia acústica estructural que un equipo inyecta en una placa de recepción.",
        },
      },
      {
        id: "eta-ij",
        symbol: "$\\eta_{ij}$",
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: "Hopkins (2007)",
        clause: {
          en: "Equation 2.154; no governing standard",
          es: "Ecuación 2.154; sin norma aplicable",
        },
        guide: "vibration/structural/junction-transmission",
        definition: {
          en: "Coupling loss factor: the fraction of energy per radian that a statistical energy analysis subsystem loses into a neighbouring one across a junction.",
          es: "Factor de pérdidas por acoplamiento: la fracción de energía por radián que un subsistema de análisis estadístico de energía cede al vecino a través de una unión.",
        },
      },
      {
        id: "tau-ij",
        symbol: "$\\tau_{ij}$",
        qualifier: {
          en: "junction",
          es: "unión",
        },
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: "Hopkins (2007)",
        clause: {
          en: "Equations 5.12 and 5.13; no governing standard",
          es: "Ecuaciones 5.12 y 5.13; sin norma aplicable",
        },
        guide: "vibration/structural/junction-transmission",
        definition: {
          en: "Junction transmission coefficient: the fraction of the bending-wave power arriving at a junction from plate $i$ that continues into plate $j$, angle by angle and then averaged over a diffuse field. Both the coupling loss factor $\\eta_{ij}$ and the vibration reduction index $K_{ij}$ are derived from it, so it is the wave-approach quantity the junction family bottoms out in.",
          es: "Coeficiente de transmisión en la unión: la fracción de la potencia de ondas de flexión que llega a una unión desde la placa $i$ y continúa hacia la placa $j$, ángulo a ángulo y después promediada en campo difuso. De él se derivan tanto el factor de pérdidas por acoplamiento $\\eta_{ij}$ como el índice de reducción vibratoria $K_{ij}$, así que es la magnitud del enfoque ondulatorio en la que se apoya toda la familia de uniones.",
        },
      },
    ],
  },
  {
    id: "psychoacoustics",
    label: {
      en: "Psychoacoustics",
      es: "Psicoacústica",
    },
    terms: [
      {
        id: "n",
        symbol: "$N$",
        unit: {
          en: "sone",
          es: "sonio",
        },
        standard: "ISO 532-1:2017",
        clause: {
          en: "clause 5 (stationary) and clause 6 (time-varying)",
          es: "apartado 5 (estacionaria) y apartado 6 (variable en el tiempo)",
        },
        guide: "perception/psychoacoustics/loudness",
        definition: {
          en: "Loudness: the perceived magnitude of a sound, anchored so that a 1 kHz tone at 40 dB SPL is exactly 1 sone.",
          es: "Sonoridad: la magnitud percibida de un sonido, anclada de modo que un tono de 1 kHz a 40 dB SPL vale exactamente 1 sonio.",
        },
      },
      {
        id: "n-2",
        symbol: "$N'$",
        unit: {
          en: "sone/Bark",
          es: "sonio/Bark",
        },
        standard: {
          en: "ISO 532-1:2017 (sone/Cam form in ISO 532-2:2017, Formula 7)",
          es: "ISO 532-1:2017 (forma en sonio/Cam en ISO 532-2:2017, Fórmula 7)",
        },
        guide: "perception/psychoacoustics/loudness",
        definition: {
          en: "Specific loudness: the loudness density along the critical-band scale, whose integral is $N$.",
          es: "Sonoridad específica: la densidad de sonoridad a lo largo de la escala de bandas críticas, cuya integral es $N$.",
        },
      },
      {
        id: "l-n-2",
        symbol: "$L_N$",
        unit: {
          en: "phon",
          es: "fonio",
        },
        standard: "ISO 226:2023",
        clause: {
          en: "Formula (2) (contours in Formula (1))",
          es: "Fórmula (2) (curvas en la Fórmula (1))",
        },
        guide: "perception/psychoacoustics/loudness",
        definition: {
          en: "Loudness level: the level of the 1 kHz free-field tone judged equally loud as the sound.",
          es: "Nivel de sonoridad: el nivel del tono de 1 kHz en campo libre que se juzga igual de fuerte que el sonido.",
        },
      },
      {
        id: "s-3",
        symbol: "$S$",
        unit: "acum",
        standard: "DIN 45692:2009",
        clause: {
          en: "clause 6",
          es: "apartado 6",
        },
        guide: "perception/psychoacoustics/sound-quality",
        definition: {
          en: "Sharpness: the position of the centre of gravity of the specific loudness on the critical-band scale, normalised so that the reference narrow-band noise is exactly 1 acum.",
          es: "Agudeza (sharpness): la posición del centro de gravedad de la sonoridad específica en la escala de bandas críticas, normalizada para que el ruido de banda estrecha de referencia valga exactamente 1,00 acum.",
        },
      },
      {
        id: "r-5",
        symbol: "$R$",
        unit: "asper",
        standard: "ECMA-418-2:2025",
        clause: {
          en: "clause 7 (Formula 104)",
          es: "apartado 7 (Fórmula 104)",
        },
        guide: "perception/psychoacoustics/sound-quality",
        definition: {
          en: "Roughness: the perceived harshness of fast amplitude modulation, around 70 Hz, normalised so that the reference modulated tone is 1 asper.",
          es: "Aspereza: la sensación áspera de una modulación de amplitud rápida, en torno a 70 Hz, normalizada para que el tono modulado de referencia valga 1 asper.",
        },
      },
      {
        id: "f",
        symbol: "$F$",
        unit: "vacil",
        standard: "ECMA-418-2:2025",
        clause: {
          en: "clause 9 (Formula 163)",
          es: "apartado 9 (Fórmula 163)",
        },
        guide: "perception/psychoacoustics/sound-quality",
        definition: {
          en: "Fluctuation strength: the perceived slow amplitude modulation, around 4 Hz, normalised so that the reference modulated tone is 1 vacil.",
          es: "Intensidad de fluctuación: la modulación de amplitud lenta percibida, en torno a 4 Hz, normalizada para que el tono modulado de referencia valga 1 vacil.",
        },
      },
      {
        id: "t",
        symbol: "$T$",
        unit: "tu",
        standard: "ECMA-418-2:2025",
        clause: {
          en: "clause 6",
          es: "apartado 6",
        },
        guide: "perception/psychoacoustics/sound-quality",
        definition: {
          en: "Tonality: the perceived tonal content of a sound, derived from the autocorrelation of the band envelopes.",
          es: "Tonalidad: el contenido tonal percibido de un sonido, obtenido de la autocorrelación de las envolventes por banda.",
        },
      },
      {
        id: "tnr",
        symbol: "TNR",
        unit: "dB",
        standard: "ECMA-418-1:2024",
        clause: {
          en: "clause 11 (Formulae 9 to 11)",
          es: "apartado 11 (Fórmulas 9 a 11)",
        },
        guide: "perception/psychoacoustics/tone-prominence",
        definition: {
          en: "Tone-to-noise ratio: the level of a discrete tone above the masking noise in the critical band around it.",
          es: "Relación tono-ruido: el nivel de un tono discreto sobre el ruido enmascarante de la banda crítica que lo rodea.",
        },
      },
      {
        id: "pr",
        symbol: "PR",
        unit: "dB",
        standard: "ECMA-418-1:2024",
        clause: {
          en: "clause 12 (Formula 23)",
          es: "apartado 12 (Fórmula 23)",
        },
        guide: "perception/psychoacoustics/tone-prominence",
        definition: {
          en: "Prominence ratio: the level of the critical band containing the tone above the mean of the two adjacent bands.",
          es: "Relación de prominencia: el nivel de la banda crítica que contiene el tono sobre la media de las dos bandas contiguas.",
        },
      },
      {
        id: "delta-l",
        symbol: "$\\Delta L$",
        unit: "dB",
        standard: "ISO/PAS 20065:2016",
        clause: {
          en: "Formula 14",
          es: "Fórmula 14",
        },
        guide: "perception/psychoacoustics/tone-audibility",
        definition: {
          en: "Audibility of a tone in noise: the tone level minus the critical-band masking level minus the masking index.",
          es: "Audibilidad de un tono en ruido: el nivel del tono menos el nivel de enmascaramiento de la banda crítica menos el índice de enmascaramiento.",
        },
      },
      {
        id: "pa",
        symbol: "PA",
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: {
          en: "Fastl and Zwicker (2007)",
          es: "Fastl y Zwicker (2007)",
        },
        clause: {
          en: "Equation 16.2; no governing standard",
          es: "Ecuación 16.2; sin norma aplicable",
        },
        guide: "perception/psychoacoustics/psychoacoustic-annoyance",
        definition: {
          en: "Psychoacoustic annoyance: the percentile loudness scaled by sharpness and by a fluctuation-plus-roughness term.",
          es: "Molestia psicoacústica: la sonoridad percentil escalada por la agudeza y por un término que combina fluctuación y aspereza.",
        },
      },
      {
        id: "critical-band",
        name: {
          en: "Critical band",
          es: "Banda crítica",
        },
        unit: "Hz",
        standard: {
          en: "Fastl and Zwicker (2007)",
          es: "Fastl y Zwicker (2007)",
        },
        clause: {
          en: "Sections 6.1 and 6.2; no governing standard",
          es: "Secciones 6.1 y 6.2; sin norma aplicable",
        },
        guide: "perception/psychoacoustics/loudness",
        definition: {
          en: "Critical band: the frequency span within which the ear sums energy as one event, about 100 Hz wide below 500 Hz and roughly a fifth of the centre frequency above it. Masking, sharpness and the two tone-prominence ratios are all computed band by band on this scale, which is why none of them can be read off a fixed fractional-octave spectrum.",
          es: "Banda crítica: el intervalo de frecuencias dentro del cual el oído suma la energía como un único suceso, de unos 100 Hz de ancho por debajo de 500 Hz y de aproximadamente una quinta parte de la frecuencia central por encima. El enmascaramiento, la agudeza y las dos razones de prominencia tonal se calculan banda a banda en esta escala, y por eso ninguno de ellos puede leerse en un espectro de fracción de octava fija.",
        },
      },
      {
        id: "z-bark",
        symbol: "$z$",
        qualifier: {
          en: "critical-band rate",
          es: "tasa de banda crítica",
        },
        unit: "Bark",
        standard: {
          en: "Fastl and Zwicker (2007)",
          es: "Fastl y Zwicker (2007)",
        },
        clause: {
          en: "Section 6.2, Table 6.1; no governing standard",
          es: "Sección 6.2, tabla 6.1; sin norma aplicable",
        },
        guide: "perception/psychoacoustics/loudness",
        definition: {
          en: "Critical-band rate: the auditory frequency scale on which one unit is one critical band, running 0 to 24 Bark over the audible range. Specific loudness is a density along it, which is why its unit is sone/Bark and why a loudness pattern is plotted against $z$ rather than against frequency.",
          es: "Tasa de banda crítica: la escala auditiva de frecuencia en la que una unidad es una banda crítica, de 0 a 24 Bark en todo el margen audible. La sonoridad específica es una densidad a lo largo de ella, y por eso su unidad es sone/Bark y por eso un patrón de sonoridad se representa frente a $z$ y no frente a la frecuencia.",
        },
      },
      {
        id: "erb-cam",
        symbol: "$ERB_N$",
        qualifier: {
          en: "Cam scale",
          es: "escala Cam",
        },
        unit: {
          en: "Hz (scale in Cam)",
          es: "Hz (escala en Cam)",
        },
        standard: "ISO 532-2:2017",
        clause: {
          en: "3.13 and 3.14",
          es: "3.13 y 3.14",
        },
        guide: "perception/psychoacoustics/advanced-loudness",
        definition: {
          en: "Equivalent rectangular bandwidth of the auditory filter, and the Cam scale built from it by counting one unit per $ERB_N$: about 132 Hz at 1 kHz, so the step from 934 Hz to 1066 Hz is one Cam. It is narrower than the Bark band at low frequencies, and it is the scale ISO 532-2 and ISO 532-3 compute specific loudness on, so a sone/Cam density is not numerically a sone/Bark one.",
          es: "Ancho de banda rectangular equivalente del filtro auditivo, y la escala Cam construida a partir de él contando una unidad por cada $ERB_N$: unos 132 Hz en 1 kHz, de modo que el paso de 934 Hz a 1066 Hz es un Cam. Es más estrecha que la banda de Bark en baja frecuencia y es la escala sobre la que la ISO 532-2 y la ISO 532-3 calculan la sonoridad específica, así que una densidad en sone/Cam no equivale numéricamente a una en sone/Bark.",
        },
      },
      {
        id: "n-5",
        symbol: "$N_5$",
        unit: {
          en: "sone",
          es: "sone",
        },
        standard: {
          en: "Fastl and Zwicker (2007)",
          es: "Fastl y Zwicker (2007)",
        },
        clause: {
          en: "Equation 16.2; no governing standard",
          es: "Ecuación 16.2; sin norma aplicable",
        },
        guide: "perception/psychoacoustics/psychoacoustic-annoyance",
        definition: {
          en: "Percentile loudness: the loudness exceeded 5 % of the time, read off the time-varying loudness of the ISO 532-1 model. It is the stationary stand-in for a fluctuating sound that psychoacoustic annoyance is built on, and it is a loudness in sone, not a loudness level in phon.",
          es: "Sonoridad percentil: la sonoridad superada el 5 % del tiempo, leída sobre la sonoridad variable en el tiempo del modelo de la ISO 532-1. Es el sustituto estacionario de un sonido fluctuante sobre el que se construye la molestia psicoacústica, y es una sonoridad en sone, no un nivel de sonoridad en fon.",
        },
      },
      {
        id: "short-long-term-loudness",
        name: {
          en: "Short-term and long-term loudness",
          es: "Sonoridad a corto y a largo plazo",
        },
        unit: {
          en: "sone",
          es: "sone",
        },
        standard: "ISO 532-3:2023",
        clause: {
          en: "3.10 and 3.11",
          es: "3.10 y 3.11",
        },
        guide: "perception/psychoacoustics/advanced-loudness",
        definition: {
          en: "The two smoothed loudness time series of the Moore-Glasberg-Schlittenlacher model: short-term loudness is the loudness of a brief segment — a syllable, a single note, typically up to 500 ms — and long-term loudness that of a longer one, a whole sentence or musical phrase, typically up to 5 s. The loudness of a sound lasting two or three seconds is predicted by the maximum of the long-term series, not by its average.",
          es: "Las dos series temporales suavizadas de sonoridad del modelo de Moore-Glasberg-Schlittenlacher: la sonoridad a corto plazo es la de un segmento breve —una sílaba, una nota, típicamente hasta 500 ms— y la de largo plazo la de uno más extenso, una frase completa o un fraseo musical, típicamente hasta 5 s. La sonoridad de un sonido de dos o tres segundos se predice con el máximo de la serie a largo plazo, no con su promedio.",
        },
      },
    ],
  },
  {
    id: "electroacoustics-and-programme-loudness",
    label: {
      en: "Electroacoustics and programme loudness",
      es: "Electroacústica y sonoridad de programa",
    },
    terms: [
      {
        id: "thd",
        symbol: "THD",
        unit: {
          en: "% or dB",
          es: "% o dB",
        },
        standard: {
          en: "IEC 60268-3:2013, 14.12.2 to 14.12.11 (the R form in 14.12.3.2)",
          es: "IEC 60268-3:2013, 14.12.2 a 14.12.11 (la forma R en 14.12.3.2)",
        },
        guide: "devices/electroacoustics/electroacoustics",
        definition: {
          en: "Total harmonic distortion: the harmonic content of the output relative to the fundamental ($\\mathrm{THD}_F$) or to the total signal ($\\mathrm{THD}_R$).",
          es: "Distorsión armónica total: el contenido armónico de la salida respecto al fundamental ($\\mathrm{THD}_F$) o respecto a la señal total ($\\mathrm{THD}_R$).",
        },
      },
      {
        id: "thd-n",
        symbol: "THD+N",
        unit: {
          en: "% or dB",
          es: "% o dB",
        },
        standard: "AES17-2015",
        clause: {
          en: "clause 6.3.1 (notch and bandwidth in 5.2.5 and 5.2.8)",
          es: "apartado 6.3.1 (muesca y ancho de banda en 5.2.5 y 5.2.8)",
        },
        guide: "devices/electroacoustics/electroacoustics",
        definition: {
          en: "Total harmonic distortion plus noise: everything left after notching out the fundamental, within the standard measurement bandwidth.",
          es: "Distorsión armónica total más ruido: todo lo que queda tras eliminar el fundamental con el filtro de muesca, dentro del ancho de banda de medida normalizado.",
        },
      },
      {
        id: "sinad",
        symbol: "SINAD",
        unit: "dB",
        standard: "AES17-2015",
        clause: {
          en: "clause 6.3.1",
          es: "apartado 6.3.1",
        },
        guide: "devices/electroacoustics/electroacoustics",
        definition: {
          en: "Signal to noise and distortion ratio, the reciprocal expression of THD+N.",
          es: "Relación entre señal y ruido más distorsión, la expresión recíproca de la THD+N.",
        },
      },
      {
        id: "imd",
        symbol: "IMD, $d_{m,n}$",
        qualifier: {
          en: "modulation",
          es: "por modulación",
        },
        unit: "%",
        standard: "IEC 60268-3:2013",
        clause: "14.12.7",
        guide: "devices/electroacoustics/electroacoustics",
        definition: {
          en: "Modulation intermodulation distortion: the sidebands a strong low-frequency tone produces around a weak high-frequency one. IEC 60268-3 defines several intermodulation families with different test signals, so a bare \"IMD\" has to say which one it is, and the single number an SMPTE-type analyzer prints combines the modulation sidebands in r.m.s. and is none of them.",
          es: "Distorsión de intermodulación por modulación: las bandas laterales que un tono intenso de baja frecuencia genera alrededor de uno débil de alta. La IEC 60268-3 define varias familias de intermodulación con señales de ensayo distintas, así que un «IMD» a secas debe indicar cuál es, y el número único que da un analizador tipo SMPTE combina en valor eficaz las bandas laterales de modulación y no es ninguna de ellas.",
        },
      },
      {
        id: "imd-2",
        symbol: "$d_{d,n}$",
        qualifier: {
          en: "difference frequency",
          es: "por frecuencia diferencia",
        },
        unit: "%",
        standard: "IEC 60268-3:2013",
        clause: "14.12.8",
        guide: "devices/electroacoustics/electroacoustics",
        definition: {
          en: "Difference-frequency intermodulation distortion: the products two closely spaced high-frequency tones of equal amplitude create at their difference and its multiples, referred to the fundamentals. Its test signal and its products are both different from the modulation form, so the two numbers are not comparable.",
          es: "Distorsión de intermodulación por frecuencia diferencia: los productos que dos tonos de alta frecuencia próximos y de igual amplitud generan en su diferencia y en los múltiplos de esta, referidos a los fundamentales. Tanto su señal de ensayo como sus productos difieren de la forma por modulación, así que ambos números no son comparables.",
        },
      },
      {
        id: "tdfd",
        symbol: "TDFD",
        unit: "%",
        standard: "IEC 60268-3:2013",
        clause: "14.12.10",
        guide: "devices/electroacoustics/electroacoustics",
        definition: {
          en: "Total difference-frequency distortion: the second- and third-order difference products of the two-tone test combined into one figure and referred to the sum of the two fundamentals, the single number the difference-frequency family reports.",
          es: "Distorsión total por frecuencia diferencia: los productos diferencia de segundo y tercer orden del ensayo de dos tonos combinados en una sola cifra y referidos a la suma de los dos fundamentales, el número único con el que informa la familia de frecuencia diferencia.",
        },
      },
      {
        id: "h-3",
        symbol: "$H_n$",
        qualifier: {
          en: "harmonic order $n$",
          es: "armónico de orden $n$",
        },
        unit: {
          en: "output per input",
          es: "salida por unidad de entrada",
        },
        standard: {
          en: "Farina (2000)",
          es: "Farina (2000)",
        },
        clause: {
          en: "no governing standard",
          es: "sin norma aplicable",
        },
        guide: "devices/electroacoustics/swept-sine-distortion",
        definition: {
          en: "Harmonic transfer function of order $n$: the impulse response that an exponential-sweep deconvolution places $L\\ln n$ seconds *before* the linear one, so a single sweep separates the linear response and every harmonic order into its own window. The distortion of order $n$ at excitation frequency $f$ is then read as $|H_n(nf)|/|H_1(f)|$.",
          es: "Función de transferencia armónica de orden $n$: la respuesta al impulso que la deconvolución de un barrido exponencial sitúa $L\\ln n$ segundos *antes* de la lineal, de modo que un único barrido separa la respuesta lineal y cada orden armónico en su propia ventana. La distorsión de orden $n$ a la frecuencia de excitación $f$ se lee entonces como $|H_n(nf)|/|H_1(f)|$.",
        },
      },
      {
        id: "dim",
        symbol: "DIM",
        unit: "%",
        standard: "IEC 60268-3:2013",
        clause: "14.12.9",
        guide: "devices/electroacoustics/electroacoustics",
        definition: {
          en: "Dynamic intermodulation distortion, measured with a 15 kHz sine against a filtered 3.15 kHz square wave.",
          es: "Distorsión de intermodulación dinámica, medida con un seno de 15 kHz frente a una onda cuadrada de 3,15 kHz filtrada.",
        },
      },
      {
        id: "l-k-lufs",
        symbol: "$L_K$, LUFS",
        unit: "LUFS",
        standard: "ITU-R BS.1770-5",
        clause: {
          en: "Formula 2 (gating in Formulae 3 to 7)",
          es: "Fórmula 2 (puerta en las Fórmulas 3 a 7)",
        },
        guide: "devices/broadcast/program-loudness",
        definition: {
          en: "Programme loudness: the channel-weighted sum of K-weighted mean-square powers, gated in 400 ms blocks. LUFS and LKFS name the same unit.",
          es: "Sonoridad de programa: la suma ponderada por canales de las potencias cuadráticas medias con ponderación K, con puerta en bloques de 400 ms. LUFS y LKFS nombran la misma unidad.",
        },
      },
      {
        id: "lra",
        symbol: "LRA",
        unit: "LU",
        standard: "EBU Tech 3342",
        guide: "devices/broadcast/program-loudness",
        definition: {
          en: "Loudness range: the spread between the 10th and 95th percentiles of the gated short-term loudness distribution.",
          es: "Rango de sonoridad: la separación entre los percentiles 10 y 95 de la distribución de sonoridad de corto plazo tras la puerta.",
        },
      },
      {
        id: "dbtp",
        symbol: "dBTP",
        unit: "dBTP",
        standard: "ITU-R BS.1770-5",
        clause: {
          en: "Annex 2",
          es: "Anexo 2",
        },
        guide: "devices/broadcast/program-loudness",
        definition: {
          en: "True peak level: the peak of the signal reconstructed by oversampling, which catches the inter-sample peaks a sample-domain maximum misses.",
          es: "Nivel de pico verdadero: el pico de la señal reconstruida por sobremuestreo, que capta los picos entre muestras que un máximo en el dominio de la muestra se pierde.",
        },
      },
    ],
  },
  {
    id: "aircraft-noise",
    label: {
      en: "Aircraft and airport noise",
      es: "Ruido de aeronaves y aeroportuario",
    },
    terms: [
      {
        id: "pnl",
        symbol: "PNL",
        unit: "PNdB",
        standard: {
          en: "ICAO Annex 16, Vol. I",
          es: "Anexo 16 OACI, Vol. I",
        },
        clause: {
          en: "Appendix 2 (noisiness law in Table A2-3)",
          es: "Apéndice 2 (ley de ruidosidad en la Tabla A2-3)",
        },
        guide: "aircraft/aircraft-noise",
        definition: {
          en: "Perceived noise level: the 24 one-third-octave band levels converted to noisiness in noys and recombined.",
          es: "Nivel de ruido percibido: los niveles de las 24 bandas de tercio de octava convertidos a ruidosidad en noys y recombinados.",
        },
      },
      {
        id: "pnlt",
        symbol: "PNLT",
        unit: "PNdB",
        standard: {
          en: "ICAO Annex 16, Vol. I",
          es: "Anexo 16 OACI, Vol. I",
        },
        clause: {
          en: "Appendix 2",
          es: "Apéndice 2",
        },
        guide: "aircraft/aircraft-noise",
        definition: {
          en: "Tone-corrected perceived noise level: PNL plus the penalty for spectral irregularities such as fan and turbine tones.",
          es: "Nivel de ruido percibido corregido por tonos: el PNL más la penalización por irregularidades espectrales como los tonos de ventilador y turbina.",
        },
      },
      {
        id: "epnl",
        symbol: "EPNL",
        unit: "EPNdB",
        standard: {
          en: "ICAO Annex 16, Vol. I",
          es: "Anexo 16 OACI, Vol. I",
        },
        clause: {
          en: "Appendix 2",
          es: "Apéndice 2",
        },
        guide: "aircraft/aircraft-noise",
        definition: {
          en: "Effective perceived noise level: the maximum PNLT plus the duration correction over the 10 dB-down window, the noise-certification metric.",
          es: "Nivel efectivo de ruido percibido: el PNLT máximo más la corrección por duración en la ventana de 10 dB, la métrica de certificación acústica.",
        },
      },
      {
        id: "pnltm",
        symbol: "PNLTM",
        unit: "PNdB",
        standard: {
          en: "ICAO Annex 16, Vol. I",
          es: "Anexo 16 OACI, Vol. I",
        },
        clause: {
          en: "Appendix 2, 4.4",
          es: "Apéndice 2, 4.4",
        },
        guide: "aircraft/aircraft-noise",
        definition: {
          en: "Maximum tone-corrected perceived noise level: the largest PNLT of the half-second time history, after the bandsharing adjustment. It is the peak the certification metric is built on, since $\\mathrm{EPNL} = \\mathrm{PNLTM} + D$ and the 10 dB-down integration window is measured down from it.",
          es: "Nivel máximo de ruido percibido corregido por tonos: el mayor PNLT del historial temporal de medio segundo, tras el ajuste por reparto entre bandas. Es el pico sobre el que se construye la métrica de certificación, ya que $\\mathrm{EPNL} = \\mathrm{PNLTM} + D$ y la ventana de integración de 10 dB se mide hacia abajo desde él.",
        },
      },
      {
        id: "npd",
        symbol: "NPD",
        unit: "dB",
        standard: {
          en: "ECAC Doc 29, 4th ed., Volume 2",
          es: "ECAC Doc 29, 4.ª ed., Volumen 2",
        },
        clause: {
          en: "section 4.2",
          es: "apartado 4.2",
        },
        guide: "aircraft/airport-noise",
        definition: {
          en: "Noise-power-distance table: the event level of one aircraft — $L_{Amax}$ or SEL — tabulated against distance for a handful of engine power settings, measured in steady flight along a notionally infinite straight path at a reference speed. A calculation interpolates it linearly in power and logarithmically in distance, then corrects that baseline segment by segment.",
          es: "Tabla ruido-potencia-distancia: el nivel del suceso de una aeronave — $L_{Amax}$ o SEL — tabulado frente a la distancia para unos pocos regímenes de motor, medido en vuelo estacionario a lo largo de una trayectoria recta nominalmente infinita y a una velocidad de referencia. El cálculo interpola linealmente en potencia y logarítmicamente en distancia, y después corrige esa base segmento a segmento.",
        },
      },
      {
        id: "anp",
        name: {
          en: "ANP database",
          es: "Base de datos ANP",
        },
        standard: {
          en: "ECAC Doc 29, 4th ed., Volume 2",
          es: "ECAC Doc 29, 4.ª ed., Volumen 2",
        },
        clause: {
          en: "Appendix G",
          es: "Apéndice G",
        },
        guide: "aircraft/anp-fleet",
        definition: {
          en: "Aircraft Noise and Performance database: the international collection of NPD tables, aircraft and engine performance coefficients and default departure and approach profiles, supplied mostly by the manufacturers, that an airport-noise calculation is normally run from.",
          es: "Base de datos de ruido y prestaciones de aeronaves (ANP): la colección internacional de tablas NPD, coeficientes de prestaciones de aeronave y motor y perfiles por defecto de despegue y aproximación, aportada en su mayor parte por los fabricantes, desde la que se ejecuta normalmente un cálculo de ruido aeroportuario.",
        },
      },
      {
        id: "slant-distance",
        symbol: "$d_p$",
        unit: "m",
        standard: {
          en: "ECAC Doc 29, 4th ed., Volume 2",
          es: "ECAC Doc 29, 4.ª ed., Volumen 2",
        },
        clause: {
          en: "section 4.5.2",
          es: "apartado 4.5.2",
        },
        guide: "aircraft/airport-noise",
        definition: {
          en: "Slant distance: the perpendicular distance from the receiver to the flight-path segment, which is the abscissa of every NPD table. It is not the distance along the ground and not the aircraft's altitude, and to the side of the track it is the minimum distance to the segment rather than to the whole path.",
          es: "Distancia oblicua: la distancia perpendicular del receptor al segmento de trayectoria, que es la abscisa de toda tabla NPD. No es la distancia sobre el terreno ni la altitud de la aeronave y, a un lado de la traza, es la distancia mínima al segmento y no a toda la trayectoria.",
        },
      },
      {
        id: "l-amax",
        symbol: "$L_{Amax}$",
        unit: "dB re 20 µPa",
        standard: {
          en: "ECAC Doc 29, 4th ed., Volume 2",
          es: "ECAC Doc 29, 4.ª ed., Volumen 2",
        },
        clause: {
          en: "section 4.1",
          es: "apartado 4.1",
        },
        guide: "aircraft/airport-noise",
        definition: {
          en: "Maximum A-weighted level of a single event: the largest instantaneous value $L_A(t)$ reaches while the aircraft passes, the simpler of the two event metrics an NPD table carries. Doc 29 reads $L_A$ on the Slow sound-level-meter scale, so the time weighting is part of the quantity; it is not the peak level, which has no time weighting at all.",
          es: "Nivel máximo ponderado A de un suceso: el mayor valor instantáneo que alcanza $L_A(t)$ durante el paso de la aeronave, la más sencilla de las dos métricas de suceso que recoge una tabla NPD. El Doc 29 lee $L_A$ en la escala Slow del sonómetro, así que la ponderación temporal forma parte de la magnitud; no es el nivel de pico, que no lleva ponderación temporal alguna.",
        },
      },
    ],
  },
  {
    id: "underwater-acoustics",
    label: {
      en: "Underwater acoustics",
      es: "Acústica submarina",
    },
    terms: [
      {
        id: "l-p-2",
        symbol: "$L_p$",
        qualifier: {
          en: "underwater",
          es: "submarino",
        },
        unit: "dB re 1 µPa",
        standard: {
          en: "ISO 18405:2017 (mean-square level in ISO 18406:2017, Formula 7)",
          es: "ISO 18405:2017 (nivel cuadrático medio en ISO 18406:2017, Fórmula 7)",
        },
        guide: "underwater/underwater-acoustics",
        definition: {
          en: "Underwater sound pressure level, referred to 1 µPa rather than 20 µPa. An airborne level never converts to it by subtraction alone.",
          es: "Nivel de presión acústica submarina, referido a 1 µPa y no a 20 µPa. Un nivel aéreo nunca se convierte a él con una simple resta.",
        },
      },
      {
        id: "sel",
        symbol: "SEL",
        qualifier: {
          en: "underwater",
          es: "submarino",
        },
        unit: "dB re 1 µPa²·s",
        standard: "ISO 18405:2017",
        guide: "underwater/underwater-acoustics",
        definition: {
          en: "Underwater sound exposure level, the time integral of squared pressure referred to 1 µPa²·s.",
          es: "Nivel de exposición sonora submarina: la integral temporal de la presión al cuadrado referida a 1 µPa²·s.",
        },
      },
      {
        id: "l-rn",
        symbol: "$L_{RN}$",
        unit: "dB re 1 µPa·m",
        standard: "ISO 17208-1:2016",
        guide: "underwater/underwater-acoustics",
        definition: {
          en: "Radiated noise level of a ship: the level of the product of the far-field r.m.s. pressure and the source distance.",
          es: "Nivel de ruido radiado por un buque: el nivel del producto de la presión eficaz en campo lejano por la distancia a la fuente.",
        },
      },
      {
        id: "l-s",
        symbol: "$L_s$",
        unit: "dB re 1 µPa·m",
        standard: "ISO 17208-2:2019",
        clause: {
          en: "Formula 3",
          es: "Fórmula 3",
        },
        guide: "underwater/underwater-acoustics",
        definition: {
          en: "Equivalent monopole source level: the radiated noise level after the Lloyd's-mirror surface correction, so that one number describes the source itself.",
          es: "Nivel de fuente monopolar equivalente: el nivel de ruido radiado tras la corrección de superficie de Lloyd, de modo que un solo número describa la fuente en sí.",
        },
      },
      {
        id: "tl-3",
        symbol: "PL, $N_{PL}$",
        qualifier: {
          en: "underwater",
          es: "submarino",
        },
        unit: "dB",
        standard: "ISO 18405:2017",
        clause: "3.4.1.4",
        guide: "underwater/underwater-propagation",
        definition: {
          en: "Propagation loss, the quantity the sonar equation uses: the difference between the source level and the mean-square sound pressure level at the receiver, $N_{PL}(x) = L_S - L_p(x)$, in practice a geometrical spreading law plus the volume absorption $\\alpha R$. Clause 3.4.1.3 keeps the name transmission loss for the reduction in level between two stated points, and both entries deprecate using one name as a synonym of the other, which is why so much of the literature calls this one a transmission loss.",
          es: "Pérdida de propagación, la magnitud que usa la ecuación del sonar: la diferencia entre el nivel de fuente y el nivel de presión acústica cuadrático medio en el receptor, $N_{PL}(x) = L_S - L_p(x)$, en la práctica una ley de divergencia geométrica más la absorción de volumen $\\alpha R$. El apartado 3.4.1.3 reserva el nombre de pérdida por transmisión para la reducción de nivel entre dos puntos indicados, y ambas entradas desaconsejan usar un nombre como sinónimo del otro, que es la razón de que buena parte de la literatura llame a esta pérdida por transmisión.",
        },
      },
      {
        id: "sl",
        symbol: "SL, $L_S$",
        unit: "dB re 1 µPa²m²",
        standard: "ISO 18405:2017",
        clause: "3.3.2.1",
        guide: "underwater/underwater-propagation",
        definition: {
          en: "Source level: the level of the source factor, equal to the level 1 m from a hypothetical point source radiating into an infinite lossless medium, so it is back-projected from a measurement made much further away and never measured at 1 m. The reference value carries a squared metre, which is why the widespread \"dB re 1 µPa at 1 m\" is the same number written loosely.",
          es: "Nivel de fuente: el nivel del factor de fuente, igual al nivel a 1 m de una fuente puntual hipotética que radia en un medio infinito sin pérdidas, de modo que se retroproyecta desde una medida hecha mucho más lejos y nunca se mide a 1 m. El valor de referencia lleva un metro cuadrado, y por eso el habitual «dB re 1 µPa a 1 m» es ese mismo número escrito de forma laxa.",
        },
      },
      {
        id: "nl",
        symbol: "NL, $L_N$",
        unit: "dB re 1 µPa",
        standard: "ISO 18405:2017",
        clause: "3.6.2.5",
        guide: "underwater/underwater-propagation",
        definition: {
          en: "Sonar noise level: the background the signal has to be detected against, in the band of interest, from wind, thermal agitation, distant shipping and the platform's own self-noise. It has to be quoted over the same bandwidth as the source level, since an ambient spectrum level and a broadband level differ by $10\\log_{10}B$.",
          es: "Nivel de ruido del sonar: el fondo frente al que hay que detectar la señal en la banda de interés, procedente del viento, la agitación térmica, el tráfico marítimo lejano y el ruido propio de la plataforma. Debe darse en el mismo ancho de banda que el nivel de fuente, ya que un nivel espectral de ruido ambiente y un nivel de banda ancha difieren en $10\\log_{10}B$.",
        },
      },
      {
        id: "di",
        symbol: "DI",
        qualifier: {
          en: "receiving array",
          es: "agrupación receptora",
        },
        unit: "dB",
        standard: "ISO 18405:2017",
        clause: {
          en: "3.6.2.4 (Note 4)",
          es: "3.6.2.4 (nota 4)",
        },
        guide: "underwater/underwater-propagation",
        definition: {
          en: "Directivity index of a receiving array: the array gain in the special case the sonar equation assumes, a plane-wave signal in isotropic background noise. It is the signal-to-noise ratio the beam buys over a single omnidirectional hydrophone, and it enters the equation as a credit against the noise level.",
          es: "Índice de directividad de una agrupación receptora: la ganancia de agrupación en el caso particular que supone la ecuación del sonar, señal de onda plana en ruido de fondo isótropo. Es la relación señal-ruido que aporta el haz frente a un único hidrófono omnidireccional, y entra en la ecuación como un crédito frente al nivel de ruido.",
        },
      },
      {
        id: "dt",
        symbol: "DT",
        unit: "dB",
        standard: "ISO 18405:2017",
        clause: "3.6.2.1",
        guide: "underwater/underwater-propagation",
        definition: {
          en: "Detection threshold: ten times the base-10 logarithm of the signal-to-noise ratio at which a signal counts as just detectable, for a stated probability of detection — often 0,5 — and probability of false alarm. It is where the processing gain and the operator's tolerance for false alarms enter the sonar equation.",
          es: "Umbral de detección: diez veces el logaritmo decimal de la relación señal-ruido a la que una señal se considera apenas detectable, para una probabilidad de detección indicada —a menudo 0,5— y una probabilidad de falsa alarma. Es el punto por el que entran en la ecuación del sonar la ganancia de proceso y la tolerancia del operador a las falsas alarmas.",
        },
      },
      {
        id: "ts",
        symbol: "TS, $N_{TS}$",
        unit: "dB re 1 m²/sr",
        standard: "ISO 18405:2017",
        clause: "3.6.2.8",
        guide: "underwater/underwater-propagation",
        definition: {
          en: "Target strength: the level of the target's differential scattering cross section, the free-field ratio of what it scatters back to what is incident on it. It is the one term that separates the active sonar equation from the passive one, and it depends on both the incidence and the scattering direction, so a backscattering value is the monostatic special case.",
          es: "Índice de blanco: el nivel de la sección eficaz diferencial de dispersión del blanco, la relación en campo libre entre lo que devuelve y lo que incide sobre él. Es el único término que separa la ecuación del sonar activo de la del pasivo y depende tanto de la dirección de incidencia como de la de dispersión, de modo que un valor de retrodispersión es el caso particular monoestático.",
        },
      },
      {
        id: "se",
        symbol: "SE, $\\Delta L_{SE}$",
        unit: "dB",
        standard: "ISO 18405:2017",
        clause: "3.6.2.2",
        guide: "underwater/underwater-propagation",
        definition: {
          en: "Signal excess: the amount by which the signal-to-noise ratio at the processor output exceeds the detection threshold. It is what the sonar equation returns; the target is detectable where it is positive, and the range at which it crosses zero is the detection range.",
          es: "Exceso de señal: la cantidad en que la relación señal-ruido a la salida del procesador supera el umbral de detección. Es lo que devuelve la ecuación del sonar: el blanco es detectable donde es positivo, y el alcance al que cruza el cero es el alcance de detección.",
        },
      },
      {
        id: "fom",
        symbol: "FOM",
        unit: {
          en: "dB (re 1 m² as a propagation factor)",
          es: "dB (re 1 m² como factor de propagación)",
        },
        standard: {
          en: "Ainslie (2010)",
          es: "Ainslie (2010)",
        },
        clause: {
          en: "Equations (3.48) and (3.111); no governing standard",
          es: "Ecuaciones (3.48) y (3.111); sin norma aplicable",
        },
        guide: "underwater/underwater-propagation",
        definition: {
          en: "Figure of merit: the propagation loss a passive system can afford before the signal excess reaches zero, that is, the loss at which SE = 0. It is read as the loss at which the probability of detection falls to 50 % only under the usual convention that the detection threshold is itself referred to that probability; SE = 0 fixes no probability on its own. Inverting any loss law at $PL = \\mathrm{FOM}$ gives the detection range directly, which is why it is quoted instead of the whole curve.",
          es: "Figura de mérito: las pérdidas de propagación que un sistema pasivo puede permitirse antes de que el exceso de señal llegue a cero, es decir, aquellas para las que SE = 0. Solo se leen como las pérdidas a las que la probabilidad de detección baja al 50 % bajo el convenio habitual de referir a esa probabilidad el propio umbral de detección; SE = 0 no fija por sí solo ninguna probabilidad. Invertir cualquier ley de pérdidas en $PL = \\mathrm{FOM}$ da directamente el alcance de detección, y por eso se cita en lugar de toda la curva.",
        },
      },
      {
        id: "w-f",
        symbol: "$W(f)$",
        unit: "dB",
        standard: {
          en: "NMFS (2024) v3.0",
          es: "NMFS (2024) v3.0",
        },
        clause: {
          en: "Equation 1 (parameters in Table 5); no governing standard",
          es: "Ecuación 1 (parámetros en la tabla 5); sin norma aplicable",
        },
        guide: "underwater/marine-mammal-exposure",
        definition: {
          en: "Auditory weighting function: a generic band-pass filter shaped to one hearing group's susceptibility to noise-induced hearing loss, with its gain chosen so the flat central part sits at 0 dB. It is applied to the spectrum before an exposure is summed, so the same physical sound weighs differently for a porpoise and for a baleen whale.",
          es: "Función de ponderación auditiva: un filtro paso banda genérico con la forma de la susceptibilidad de un grupo auditivo a la pérdida de audición inducida por ruido, con la ganancia elegida para que su parte central plana quede en 0 dB. Se aplica al espectro antes de sumar la exposición, de modo que un mismo sonido físico pesa distinto para una marsopa y para una ballena barbada.",
        },
      },
      {
        id: "tts-onset",
        name: {
          en: "TTS onset",
          es: "Umbral de TTS",
        },
        unit: "dB",
        standard: {
          en: "NMFS (2024) v3.0",
          es: "NMFS (2024) v3.0",
        },
        clause: {
          en: "Table 8; no governing standard",
          es: "Tabla 8; sin norma aplicable",
        },
        guide: "underwater/marine-mammal-exposure",
        definition: {
          en: "Onset of temporary threshold shift: the exposure at which a recoverable loss of hearing sensitivity begins, published per hearing group as a weighted sound exposure level and, for impulsive sound, together with an unweighted peak level. Both metrics have to be tested; whichever is reached first decides.",
          es: "Umbral de desplazamiento temporal del umbral auditivo: la exposición a la que empieza una pérdida recuperable de sensibilidad auditiva, publicada por grupo auditivo como nivel de exposición sonora ponderado y, para sonidos impulsivos, junto con un nivel de pico sin ponderar. Hay que comprobar ambas métricas: decide la que se alcanza primero.",
        },
      },
      {
        id: "aud-inj-onset",
        name: {
          en: "AUD INJ onset (PTS onset)",
          es: "Umbral de AUD INJ (umbral de PTS)",
        },
        unit: "dB",
        standard: {
          en: "NMFS (2024) v3.0",
          es: "NMFS (2024) v3.0",
        },
        clause: {
          en: "Table ES3; no governing standard",
          es: "Tabla ES3; sin norma aplicable",
        },
        guide: "underwater/marine-mammal-exposure",
        definition: {
          en: "Onset of auditory injury, the 2024 guidance's name for what earlier versions called permanent threshold shift onset: the exposure above which the shift no longer recovers. For non-impulsive sound it is the TTS onset plus 20 dB for every group; for impulsive sound the guidance sets it 15 dB above in exposure and 6 dB above in peak level.",
          es: "Umbral de lesión auditiva, el nombre que la guía de 2024 da a lo que las versiones anteriores llamaban umbral de desplazamiento permanente: la exposición por encima de la cual el desplazamiento ya no se recupera. Para sonidos no impulsivos son 20 dB por encima del umbral de TTS en todos los grupos; para sonidos impulsivos la guía lo sitúa 15 dB por encima en exposición y 6 dB por encima en nivel de pico.",
        },
      },
      {
        id: "sel-cum",
        symbol: "$SEL_{cum}$",
        unit: "dB re 1 µPa²·s",
        standard: {
          en: "NMFS (2024) v3.0",
          es: "NMFS (2024) v3.0",
        },
        clause: {
          en: "Table ES3; no governing standard",
          es: "Tabla ES3; sin norma aplicable",
        },
        guide: "underwater/marine-mammal-exposure",
        definition: {
          en: "Cumulative sound exposure level: the weighted sound exposure of every event of an activity — every strike of a piling campaign, say — summed over its whole duration. The onset criteria are written against this accumulated quantity, not against a single event, so halving the strike energy and doubling the strike count changes nothing.",
          es: "Nivel de exposición sonora acumulada: la exposición sonora ponderada de todos los sucesos de una actividad —por ejemplo, cada golpe de una campaña de hincado— sumada a lo largo de toda su duración. Los criterios de umbral se escriben frente a esta magnitud acumulada y no frente a un suceso aislado, de modo que reducir a la mitad la energía por golpe y duplicar el número de golpes no cambia nada.",
        },
      },
    ],
  },
  {
    id: "numerical-simulation",
    label: {
      en: "Numerical simulation",
      es: "Simulación numérica",
    },
    terms: [
      {
        id: "courant-number",
        symbol: "$C_N$",
        qualifier: {
          en: "the cfl argument",
          es: "el argumento cfl",
        },
        unit: {
          en: "dimensionless",
          es: "adimensional",
        },
        standard: {
          en: "Attenborough and Van Renterghem (2021)",
          es: "Attenborough y Van Renterghem (2021)",
        },
        clause: {
          en: "Equations (4.13) and (4.14); no governing standard",
          es: "Ecuaciones (4.13) y (4.14); sin norma aplicable",
        },
        guide: "simulation/fdtd-simulation",
        definition: {
          en: "Courant number: how far a wavefront travels in one time step, measured in grid cells, $c\\,\\Delta t\\sqrt{1/\\Delta x^2 + 1/\\Delta y^2}$. An explicit scheme is stable only up to one and meaningless above it, so it is the number the time step is chosen from rather than the other way round; the library defaults to 0,6.",
          es: "Número de Courant: la distancia que recorre un frente de onda en un paso temporal, medida en celdas de la malla, $c\\,\\Delta t\\sqrt{1/\\Delta x^2 + 1/\\Delta y^2}$. Un esquema explícito solo es estable hasta uno y carece de sentido por encima, así que es el número a partir del cual se elige el paso temporal y no al revés; la biblioteca usa 0,6 por defecto.",
        },
      },
      {
        id: "numerical-dispersion",
        name: {
          en: "Numerical dispersion",
          es: "Dispersión numérica",
        },
        unit: {
          en: "% (speed error)",
          es: "% (error de velocidad)",
        },
        standard: {
          en: "Attenborough and Van Renterghem (2021)",
          es: "Attenborough y Van Renterghem (2021)",
        },
        clause: {
          en: "Equation (4.15); no governing standard",
          es: "Ecuación (4.15); sin norma aplicable",
        },
        guide: "simulation/fdtd-simulation",
        definition: {
          en: "Numerical dispersion: the error the discrete scheme makes in the propagation speed, which grows with frequency and depends on the direction of travel — largest along a coordinate axis, and zero along the diagonal of square cells at a Courant number of one. It is why ten cells per shortest wavelength is the working rule, and why an FDTD arrival time drifts if the grid is too coarse.",
          es: "Dispersión numérica: el error que comete el esquema discreto en la velocidad de propagación, que crece con la frecuencia y depende de la dirección de avance: máximo a lo largo de un eje coordenado y nulo en la diagonal de celdas cuadradas con número de Courant igual a uno. Por eso la regla práctica es de diez celdas por longitud de onda más corta, y por eso el tiempo de llegada de una simulación FDTD se desvía si la malla es demasiado gruesa.",
        },
      },
      {
        id: "pml",
        name: {
          en: "PML (perfectly matched layer)",
          es: "PML (capa perfectamente acoplada)",
        },
        standard: {
          en: "Attenborough and Van Renterghem (2021)",
          es: "Attenborough y Van Renterghem (2021)",
        },
        clause: {
          en: "Section 4.2.3; no governing standard",
          es: "Sección 4.2.3; sin norma aplicable",
        },
        guide: "simulation/fdtd-simulation",
        definition: {
          en: "Perfectly matched layer: a boundary region whose absorption is graded so that, in principle, it reflects nothing at any angle or frequency, which is what lets a finite grid stand in for open space. What this library ships is the graded sponge layer, its simple precursor, so a grazing-incidence residue at the edge is expected rather than a bug.",
          es: "Capa perfectamente acoplada: una región de contorno con absorción gradual que, en principio, no refleja nada a ningún ángulo ni frecuencia, y que es lo que permite a una malla finita hacer las veces de espacio abierto. Lo que incluye esta biblioteca es la capa esponja gradual, su precursora sencilla, así que un residuo a incidencia rasante en el borde es esperable y no un fallo.",
        },
      },
    ],
  },
  {
    id: "measurement-uncertainty",
    label: {
      en: "Measurement uncertainty",
      es: "Incertidumbre de medida",
    },
    terms: [
      {
        id: "u-y",
        symbol: "$u(y)$",
        unit: {
          en: "unit of the result",
          es: "unidad del resultado",
        },
        standard: "ISO/IEC Guide 98-3:2008 (JCGM 100:2008)",
        clause: {
          en: "clause 5",
          es: "apartado 5",
        },
        guide: "signals/metrology/gum-uncertainty",
        definition: {
          en: "Combined standard uncertainty of a result, propagated from the standard uncertainties of its inputs by the law of propagation of uncertainty.",
          es: "Incertidumbre típica combinada de un resultado, propagada desde las incertidumbres típicas de sus entradas por la ley de propagación de la incertidumbre.",
        },
      },
      {
        id: "u",
        symbol: "$U$",
        unit: {
          en: "unit of the result",
          es: "unidad del resultado",
        },
        standard: "ISO/IEC Guide 98-3:2008 (JCGM 100:2008)",
        clause: {
          en: "clause 6 and Annex G",
          es: "apartado 6 y Anexo G",
        },
        guide: "signals/metrology/gum-uncertainty",
        definition: {
          en: "Expanded uncertainty: the combined standard uncertainty multiplied by a coverage factor, which defines a coverage interval.",
          es: "Incertidumbre expandida: la incertidumbre típica combinada multiplicada por un factor de cobertura, que define un intervalo de cobertura.",
        },
      },
      {
        id: "sigma-r",
        symbol: "$\\sigma_R$",
        unit: "dB",
        standard: "ISO 12999-1:2020",
        clause: {
          en: "Clause 5.2 (coverage factors in Table 8)",
          es: "apartado 5.2 (factores de cobertura en la Tabla 8)",
        },
        guide: "buildings/insulation/insulation-field",
        definition: {
          en: "Reproducibility standard deviation of a sound-insulation quantity: the standard uncertainty ISO 12999-1 assigns to situation A, the widest of the three. The same clause assigns $\\sigma_{situ}$ to situation B and the repeatability $\\sigma_r$ to situation C — for $R'_w$, 1,2 dB against 0,9 and 0,4 — so the situation has to be stated with the number.",
          es: "Desviación típica de reproducibilidad de una magnitud de aislamiento acústico: la incertidumbre típica que la ISO 12999-1 asigna a la situación A, la más amplia de las tres. El mismo apartado asigna $\\sigma_{situ}$ a la situación B y la repetibilidad $\\sigma_r$ a la situación C —para $R'_w$, 1,2 dB frente a 0,9 y 0,4—, así que hay que indicar la situación junto con el número.",
        },
      },
    ],
  },
];
