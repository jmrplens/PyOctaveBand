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
        unit: "dB",
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
        unit: "dB",
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
        unit: "dB",
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
        unit: "dB",
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
        unit: "dB",
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
        unit: "dB",
        standard: "ISO 1996-1:2016",
        clause: "3.6.4",
        guide: "environment/environmental-levels",
        definition: {
          en: "Day-evening-night level: the energy mean of the three periods with 5 dB added to the evening and 10 dB to the night.",
          es: "Nivel día-tarde-noche: la media energética de los tres periodos con 5 dB añadidos a la tarde y 10 dB a la noche.",
        },
      },
      {
        id: "l-dn",
        symbol: "$L_{dn}$",
        unit: "dB",
        standard: "ISO 1996-1:2016",
        clause: "3.6.5",
        guide: "environment/environmental-levels",
        definition: {
          en: "Day-night level: the same construction with the 10 dB night penalty only.",
          es: "Nivel día-noche: la misma construcción solo con la penalización de 10 dB nocturna.",
        },
      },
      {
        id: "l-r",
        symbol: "$L_r$",
        unit: "dB",
        standard: "ISO 1996-1:2016",
        clause: {
          en: "clause 6.5 (Formulae 5 and 6)",
          es: "apartado 6.5 (Fórmulas 5 y 6)",
        },
        guide: "environment/environmental-levels",
        definition: {
          en: "Rating level: the whole-day composite level after the source-character and time-of-day adjustments.",
          es: "Nivel de valoración: el nivel compuesto de la jornada completa tras los ajustes por carácter de la fuente y por franja horaria.",
        },
      },
      {
        id: "l-ar-t",
        symbol: "$L_{Ar,T}$",
        unit: "dB",
        standard: "NT ACOU 112:2002",
        clause: {
          en: "clause 8",
          es: "apartado 8",
        },
        guide: "environment/assessment/impulsive-sound",
        definition: {
          en: "Rating level of an impulsive source over a reference interval, $L_{Aeq}$ plus the graduated impulse adjustment.",
          es: "Nivel de valoración de una fuente impulsiva en un intervalo de referencia: el $L_{Aeq}$ más el ajuste graduado por impulsos.",
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
        unit: "dB",
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
        unit: "dB",
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
        unit: "s",
        standard: "ISO 3382-1:2009",
        clause: {
          en: "Equation (A.13)",
          es: "Ecuación (A.13)",
        },
        guide: "buildings/rooms/room-acoustics",
        definition: {
          en: "Centre time: the centre of gravity of the squared impulse response in time, a boundary-free alternative to the clarity indices.",
          es: "Tiempo central: el centro de gravedad temporal de la respuesta al impulso al cuadrado, una alternativa a los índices de claridad sin frontera arbitraria.",
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
          en: "Noise rating, the European counterpart curve family of NC. Discussed for comparison and deliberately not implemented.",
          es: "Noise Rating, la familia de curvas europea equivalente a NC. Se comenta a efectos de comparación y no se implementa deliberadamente.",
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
          es: "Función de transferencia de modulación: la fracción de la profundidad de modulación de la envolvente del habla, a la frecuencia de modulación $F$, que sobrevive al camino de transmisión.",
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
          es: "Índice de reducción acústica: la diferencia de niveles corregida por el área del cerramiento partido por el área de absorción del recinto receptor, medida en laboratorio con los flancos suprimidos.",
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
          es: "Índice de reducción acústica aparente: la misma construcción medida en el edificio, así que incluye todas las vías de flanco. La prima es la marca que distingue el laboratorio del campo.",
        },
      },
      {
        id: "tl",
        symbol: "TL",
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
          en: "Transmission loss: the airborne insulation of a panel predicted from its physical properties, the same quantity as $R$ in a prediction context.",
          es: "Pérdidas por transmisión: el aislamiento a ruido aéreo de un panel predicho a partir de sus propiedades físicas, la misma magnitud que $R$ en un contexto de predicción.",
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
        unit: "dB",
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
        unit: "dB",
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
          en: "ISO 717-2:2020 (measurement in ISO 16251-1:2014",
          es: "ISO 717-2:2020 (medición en ISO 16251-1:2014",
        },
        clause: {
          en: "Formulae (3) and (4))",
          es: "Fórmulas (3) y (4))",
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
          es: "Índice de reducción vibracional de una unión: la diferencia de niveles de velocidad promediada en ambos sentidos, corregida por la longitud de la unión y las longitudes de absorción equivalentes.",
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
          es: "La clase, de la A a la E, a la que se asigna el coeficiente ponderado, o \"sin clasificar\".",
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
          es: "Rigidez dinámica por unidad de superficie de una capa resiliente: una fuerza dinámica por unidad de área dividida por la variación de espesor que provoca.",
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
        unit: "dB",
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
        symbol: "$L_{Ws}$",
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
          en: "ISO 532-1:2017 (sone/Cam form in ISO 532-2:2017",
          es: "ISO 532-1:2017 (forma en sonio/Cam en ISO 532-2:2017",
        },
        clause: {
          en: "Formula 7)",
          es: "Fórmula 7)",
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
        symbol: "IMD",
        unit: "%",
        standard: "IEC 60268-3:2013, 14.12.7",
        guide: "devices/electroacoustics/electroacoustics",
        definition: {
          en: "Modulation intermodulation distortion: the sidebands a low-frequency tone produces around a high-frequency one.",
          es: "Distorsión de intermodulación por modulación: las bandas laterales que un tono de baja frecuencia genera alrededor de uno de alta.",
        },
      },
      {
        id: "dim",
        symbol: "DIM",
        unit: "%",
        standard: "IEC 60268-3:2013, 14.12.9",
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
    id: "aircraft-and-underwater",
    label: {
      en: "Aircraft and underwater",
      es: "Aeronaves y acústica submarina",
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
        id: "l-p-2",
        name: {
          en: "$L_p$",
          es: "Lp",
        },
        qualifier: {
          en: "underwater",
          es: "submarino",
        },
        unit: "dB re 1 µPa",
        standard: {
          en: "ISO 18405:2017 (mean-square level in ISO 18406:2017",
          es: "ISO 18405:2017 (nivel cuadrático medio en ISO 18406:2017",
        },
        clause: {
          en: "Formula 7)",
          es: "Fórmula 7)",
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
          en: "Standard uncertainty of a building-acoustics measurement situation, tabulated per quantity and situation.",
          es: "Incertidumbre típica de una situación de medida en acústica de la edificación, tabulada por magnitud y por situación.",
        },
      },
    ],
  },
];
