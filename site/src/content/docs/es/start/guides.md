---
title: "Guías"
description: "Las 105 guías de phonometry, agrupadas en los diez temas que cubre la biblioteca: para qué sirve cada área, las normas que implementa y un resumen de una línea de cada guía que contiene."
head:
  - tag: script
    attrs:
      type: application/ld+json
    content: |
      {
        "@context": "https://schema.org",
        "@type": "ItemList",
        "@id": "https://jmrplens.github.io/phonometry/es/start/guides/#areas",
        "name": "Áreas de la documentación de phonometry",
        "description": "Las diez áreas documentadas de las guías de phonometry, cada una con las normas que implementa.",
        "inLanguage": "es",
        "numberOfItems": 10,
        "itemListOrder": "https://schema.org/ItemListUnordered",
        "itemListElement": [
          {
            "@type": "ListItem",
            "position": 1,
            "name": "Análisis de señal",
            "description": "Bancos de filtros, ponderaciones, niveles, espectros, calibración e incertidumbre.",
            "url": "https://jmrplens.github.io/phonometry/es/signals/"
          },
          {
            "@type": "ListItem",
            "position": 2,
            "name": "Audición y percepción",
            "description": "Sonoridad, calidad sonora, inteligibilidad del habla, audición y exposición.",
            "url": "https://jmrplens.github.io/phonometry/es/perception/"
          },
          {
            "@type": "ListItem",
            "position": 3,
            "name": "Salas y edificación",
            "description": "Parámetros de sala, ruido de fondo, aislamiento en campo y laboratorio, predicción.",
            "url": "https://jmrplens.github.io/phonometry/es/buildings/"
          },
          {
            "@type": "ListItem",
            "position": 4,
            "name": "Materiales y superficies",
            "description": "Absorción, resistencia al flujo de aire, tubo de impedancia, modelos porosos y de metamaterial, difusores, dispersión.",
            "url": "https://jmrplens.github.io/phonometry/es/materials/"
          },
          {
            "@type": "ListItem",
            "position": 5,
            "name": "Vibración y ruido estructural",
            "description": "Movilidad y FRF, aisladores, potencia radiada, uniones, vibración en humanos.",
            "url": "https://jmrplens.github.io/phonometry/es/vibration/"
          },
          {
            "@type": "ListItem",
            "position": 6,
            "name": "Medio ambiente y transporte",
            "description": "Propagación en exteriores, barreras, refracción, aeronaves, helicópteros y aerogeneradores.",
            "url": "https://jmrplens.github.io/phonometry/es/environment/"
          },
          {
            "@type": "ListItem",
            "position": 7,
            "name": "Ruido de aeronaves",
            "description": "Niveles de certificación, contornos de aeropuerto y el método del hemisferio.",
            "url": "https://jmrplens.github.io/phonometry/es/aircraft/"
          },
          {
            "@type": "ListItem",
            "position": 8,
            "name": "Acústica submarina",
            "description": "Niveles re 1 microPa, ruido radiado por buques, hincado de pilotes, ruido ambiente, pérdidas por transmisión.",
            "url": "https://jmrplens.github.io/phonometry/es/underwater/"
          },
          {
            "@type": "ListItem",
            "position": 9,
            "name": "Fuentes y dispositivos",
            "description": "Potencia acústica, intensidad, declaraciones de emisión, electroacústica, sonoridad de programa.",
            "url": "https://jmrplens.github.io/phonometry/es/devices/"
          },
          {
            "@type": "ListItem",
            "position": 10,
            "name": "Simulación de ondas",
            "description": "Un solver FDTD 2D determinista, validado frente a oráculos analíticos y no frente a una norma.",
            "url": "https://jmrplens.github.io/phonometry/es/simulation/"
          }
        ]
      }
---

Todas las guías de este sitio siguen la misma estructura: la norma que
implementan, las magnitudes que esa norma define, las hipótesis que asume la
implementación y, después, el código ejecutable y la figura que dibuja. Aquí no
hay ninguna panorámica del campo: cada página es la documentación de trabajo de
un módulo, escrita para que un resultado se pueda defender apartado por
apartado en lugar de darlo por bueno.

Esta página es el mapa. Ciento cinco guías repartidas en diez temas, y cada
área tiene su propio índice con el relato largo de cómo encajan sus piezas. Si
llegas sin una pregunta concreta, lee primero la
[introducción](/phonometry/es/start/getting-started/): recorre una señal por toda la
cadena de proceso y da el vocabulario que el resto de guías da por supuesto. Si
ya sabes qué magnitud necesitas, el
[glosario](/phonometry/es/reference/glossary/) recoge cada símbolo con su
unidad, la norma que lo define y la guía que lo implementa.

Dos avisos antes de la lista. Las firmas de las funciones están en la
[referencia de la API](/phonometry/es/reference/api/), que se genera a partir de
los docstrings y no se repite aquí. Las deducciones, las decisiones de diseño y
la evidencia numérica están en [Referencia](/phonometry/es/reference/): las
páginas de teoría explican por qué la fórmula es esa, y el informe de
conformidad pone el valor esperado de la propia norma junto al calculado.

## [Análisis de señal](/phonometry/es/signals/)

Bancos de filtros, ponderaciones, niveles, espectros, calibración e
incertidumbre. Esta es la cadena que convierte una señal digital en un número
conforme con las normas, y todas las demás áreas la consumen: un modelo de
sonoridad necesita niveles de banda calibrados, un parámetro de sala necesita
una respuesta al impulso filtrada, una valoración ambiental es un $L_{eq}$
ajustado.
Implementa IEC 61260-1, ANSI S1.11, IEC 61672-1, ISO 7196, IEC 61252,
ISO 1996-1, IEC 60942 y la GUM.

- [Construye un sonómetro](/phonometry/es/signals/sound-level-meter/): el área
  entera montada de principio a fin en una sola página ejecutable, del tono del
  calibrador a los niveles reportados.

**[Filtrado en octavas](/phonometry/es/signals/filters/)**

- [Bancos de filtros](/phonometry/es/signals/filters/filter-banks/): la matemática de
  las bandas de octava fraccional, los parámetros del banco, el ecualizador
  paramétrico, la descomposición en bandas y el filtrado de fase cero fuera de
  línea.
- [Galería de arquitecturas de filtro](/phonometry/es/signals/filters/filter-gallery/):
  las cinco arquitecturas de filtro comparadas en los bordes de banda, la
  galería completa de respuestas y el uso de cada arquitectura, con el
  crossover Linkwitz-Riley.
- [Verificación de clase de filtros (IEC 61260-1)](/phonometry/es/signals/filters/filter-compliance/):
  la máscara de aceptación de la Tabla 1 banda a banda, la clase 0 de la
  edición retirada de 1995 y la ficha de conformidad.
- [Procesado por bloques](/phonometry/es/signals/filters/block-processing/): análisis en
  streaming con estado, que arrastra el estado de los filtros entre búferes,
  para señales que no caben en memoria.
- [Multicanal y rendimiento](/phonometry/es/signals/filters/multichannel/): análisis
  vectorizado de muchos canales a la vez, con el convenio (canales, muestras) y
  notas de rendimiento.

**[Niveles y ponderación](/phonometry/es/signals/levels/)**

- [Ponderación frecuencial (A, C, Z)](/phonometry/es/signals/levels/weighting/):
  las curvas de respuesta del oído de IEC 61672-1, el modo de precisión en
  alta frecuencia y la verificación de clase de la Tabla 3.
- [Ponderaciones especiales (G, B, D, AU)](/phonometry/es/signals/levels/special-weightings/):
  la ponderación G para infrasonido de ISO 7196, las curvas históricas B y D
  y la AU para sonido audible en presencia de ultrasonidos.
- [Ponderación temporal](/phonometry/es/signals/levels/time-weighting/): las balísticas
  exponenciales Fast, Slow e Impulse de IEC 61672-1.
- [Niveles integrados y estadísticos](/phonometry/es/signals/levels/levels/): $L_{eq}$ y
  $L_{Aeq}$, los niveles percentiles $L_{10}$/$L_{50}$/$L_{90}$, $L_{Cpeak}$ y
  SEL, y la dosis de ruido.
- [Niveles ambientales (ISO 1996-1/-2)](/phonometry/es/environment/assessment/environmental-levels/):
  $L_{den}$, $L_{dn}$ y los niveles de valoración compuestos, el ajuste tonal, la
  corrección de ruido residual y el presupuesto de incertidumbre.
- [Normativa española de ruido (RD 1367/2007)](/phonometry/es/environment/assessment/spanish-noise-regulation/):
  el nivel corregido $L_{Keq}$ con sus correcciones $K_t$, $K_f$ y $K_i$, los
  periodos temporales de evaluación y las fases de ruido, las tablas de valores
  límite y la comprobación del artículo 25.

**[Señales y espectros](/phonometry/es/signals/spectra/)**

- [Análisis espectral calibrado](/phonometry/es/signals/spectra/spectral-analysis/): los
  estimadores de Welch de potencia y espectro cruzado con sus errores
  aleatorios, intervalos de confianza chi-cuadrado y suavizado en fracciones de
  octava.
- [Coherencia múltiple y parcial](/phonometry/es/signals/spectra/miso-coherence/): la
  coherencia ordinaria, múltiple y parcial de varias fuentes correladas sobre
  una respuesta, y qué fuente domina cada banda.
- [Análisis tiempo-frecuencia](/phonometry/es/signals/spectra/time-frequency/): el
  espectrograma STFT calibrado en dB SPL absolutos y la FFT con zoom que
  resuelve tonos más próximos que un bin práctico de FFT.
- [Cepstro, ecos y espectro de la envolvente](/phonometry/es/signals/spectra/cepstrum-echoes/):
  el análisis de quefrencia, la detección de ecos con el coeficiente de
  reflexión leído en el pico cepstral, el liftering y el espectro de la
  envolvente.
- [Promediado síncrono en el tiempo](/phonometry/es/signals/spectra/synchronous-averaging/):
  extracción de una forma de onda periódica de período conocido, el filtro peine
  que la describe y la elección del número de promedios.
- [Frecuencias de fallo de máquinas](/phonometry/es/vibration/machinery/machine-diagnostics/):
  las familias cinemáticas de frecuencias de fallo de la maquinaria rotativa
  (Norton y Karczub, sección 8.4) dibujadas sobre un espectro de envolvente
  medido: frecuencias BPFO, BPFI, BSF y de jaula de los rodamientos, bandas
  laterales de engrane, deslizamiento, paso de polos y armónicos de ranura de
  los motores de inducción, y tonos de paso de pala.
- [Correlación, retardo y envolvente](/phonometry/es/signals/spectra/correlation-delay/):
  la correlación con sus errores aleatorios, la estimación de retardo por
  correlación directa y las ponderaciones GCC, y la envolvente de Hilbert.
- [Señales de prueba y herramientas de muestreo](/phonometry/es/signals/spectra/test-signals/):
  salvas de tono IEC 60268-1 con conmutación exacta, ruido de colores con
  pendiente exacta, remuestreo con especificación antialias declarada y retardo
  fraccionario.
- [Medición de sistemas](/phonometry/es/signals/spectra/system-measurement/): pares
  complementarios de Golay, barridos conformados a un espectro de magnitud
  objetivo arbitrario e inversión con regularización de Kirkeby de una respuesta
  medida.

**[Calibración e incertidumbre](/phonometry/es/signals/metrology/)**

- [Calibración y dBFS](/phonometry/es/signals/metrology/calibration/): calibración SPL
  física a partir de un tono de calibrador o de una sensibilidad conocida, y el
  modo digital de escala completa.
- [Incertidumbre de medida (GUM y Monte Carlo)](/phonometry/es/signals/metrology/gum-uncertainty/):
  la ley de propagación de la incertidumbre y el método de Monte Carlo, con
  incertidumbre expandida e intervalos de cobertura.
- [Calificación de datos](/phonometry/es/signals/metrology/data-qualification/): los tests
  de estacionariedad por inversiones de orden y por rachas, y las estadísticas
  de Rice de cruces por nivel y de picos con el factor de irregularidad.

## [Audición y percepción](/phonometry/es/perception/)

Sonoridad, calidad sonora, inteligibilidad del habla, audición y exposición.
Donde el área del núcleo pregunta cuánto sonido hay, esta pregunta qué hace un
oyente con él: cuán fuerte le parece, cuán agudo, rugoso o molesto, cuánto
sobrevive de un locutor tras pasar por la sala, y cuánta audición cuesta una
vida laboral en ese ruido. Implementa ISO 532-1/-2/-3, ECMA-418-1/-2, ISO 226,
DIN 45692, IEC 60268-16, ANSI S3.5, DIN 45681, ISO/PAS 20065, ISO 7029,
ISO 389-7, ISO 1999 e ISO 9612.

**[Psicoacústica](/phonometry/es/perception/psychoacoustics/)**

- [Sonoridad](/phonometry/es/perception/psychoacoustics/loudness/): la sonoridad en sonios según
  el método Zwicker de ISO 532-1 con su ficha de una página, además de las
  curvas isofónicas de ISO 226.
- [Sonoridad avanzada (ISO 532-2/-3, ECMA-418-2)](/phonometry/es/perception/psychoacoustics/advanced-loudness/):
  los métodos Moore-Glasberg estacionario y variable en el tiempo y la
  sonoridad del Sottek Hearing Model, con la tabla de elección de modelo.
- [Métricas de calidad sonora](/phonometry/es/perception/psychoacoustics/sound-quality/): la
  agudeza en acum, y la tonalidad, la aspereza y la intensidad de fluctuación
  de ECMA-418-2.
- [Tonos discretos prominentes (ECMA-418-1)](/phonometry/es/perception/psychoacoustics/tone-prominence/):
  la relación tono-ruido y la relación de prominencia, con sus criterios
  dependientes de la frecuencia.
- [Audibilidad objetiva de tonos en ruido (ISO/PAS 20065)](/phonometry/es/perception/psychoacoustics/tone-audibility/):
  el nivel de enmascaramiento de la banda crítica, el índice de enmascaramiento
  y la audibilidad de un tono sobre el umbral de enmascaramiento.
- [Molestia psicoacústica e intensidad de fluctuación](/phonometry/es/perception/psychoacoustics/psychoacoustic-annoyance/):
  el modelo de molestia de Fastl y Zwicker construido a partir de sonoridad,
  agudeza, aspereza e intensidad de fluctuación.

**[Habla](/phonometry/es/perception/speech/)**

- [Índice de transmisión del habla (STI)](/phonometry/es/perception/speech/speech-transmission/):
  la función de transferencia de modulación, el método indirecto a partir de una
  respuesta al impulso y la medida directa STIPA.
- [Índice de inteligibilidad del habla](/phonometry/es/perception/speech/speech-intelligibility/):
  el SII en los cuatro procedimientos por bandas de la norma, con las
  funciones de importancia por bandas, el autoenmascaramiento y la extensión
  ascendente del enmascaramiento.
- [Inteligibilidad objetiva (STOI y ESTOI)](/phonometry/es/perception/speech/objective-intelligibility/):
  las dos medidas basadas en correlación para habla ruidosa ponderada en
  tiempo-frecuencia.

**[Audición y exposición](/phonometry/es/perception/hearing/)**

- [Umbral de audición (edad y cero de referencia)](/phonometry/es/perception/hearing/hearing-threshold/):
  la distribución del umbral con la edad de ISO 7029 y el umbral de audición de
  referencia de ISO 389-7.
- [Pérdida auditiva inducida por ruido (ISO 1999)](/phonometry/es/perception/hearing/noise-induced-hearing-loss/):
  el desplazamiento permanente del umbral en función del nivel, la duración y la
  frecuencia, combinado con la componente de la edad.
- [Exposición al ruido en el trabajo (ISO 9612)](/phonometry/es/perception/hearing/occupational-exposure/):
  las estrategias por tareas, por puestos y de jornada completa para
  $L_{EX,8h}$, con el presupuesto de incertidumbre y el límite superior.

## [Salas y edificación](/phonometry/es/buildings/)

Parámetros de sala, ruido de fondo, aislamiento en campo y laboratorio, y
predicción a partir de datos de elementos. Dos preguntas recorren el área: qué
hace una sala con el sonido que se produce dentro, y cuánto del sonido que se
produce al lado consigue pasar. Implementa ISO 3382-1/-2/-3,
ISO 16283-1/-2/-3, ISO 10140, ISO 10848, ISO 15186-1/-2, ISO 16251-1,
ISO 717-1/-2, EN 12354-1 a -6, ISO 18233, ISO 12999-1, ISO 10052,
ANSI/ASA S12.2 y ASTM E413/E1414.

**[Acústica de salas](/phonometry/es/buildings/rooms/)**

- [Medición de la respuesta al impulso](/phonometry/es/buildings/rooms/room-impulse-response/):
  la adquisición determinista de ISO 18233, los barridos exponenciales con su
  deconvolución y MLS.
- [Acústica de salas](/phonometry/es/buildings/rooms/room-acoustics/): los parámetros de
  sala EDT, $T_{20}$, $T_{30}$, $C_{50}$, $C_{80}$, $D_{50}$ y $T_s$ derivados
  de esa respuesta al impulso.
- [Acústica de oficinas diáfanas (ISO 3382-3)](/phonometry/es/buildings/rooms/open-plan-acoustics/):
  la tasa de decaimiento espacial del habla y las distancias de distracción y
  de privacidad de una planta diáfana.
- [Fuentes imagen y campo estacionario de sala](/phonometry/es/buildings/rooms/room-image-sources/):
  la respuesta al impulso determinista por fuentes imagen de una sala
  rectangular, la constante de sala, la distancia crítica y la frecuencia de
  Schroeder.
- [Criterios de ruido de salas (NC / RC Mark II)](/phonometry/es/buildings/rooms/room-noise/):
  el índice NC de ANSI/ASA S12.2 por el método de tangencia, y el índice RC
  Mark II con su etiqueta de retumbo, siseo o neutro.
- [Predicción del tiempo de reverberación (Sabine, Arau)](/phonometry/es/buildings/rooms/reverberation-prediction/):
  cinco modelos de acústica estadística a partir del volumen, las superficies y
  su absorción, con el término del aire.
- [Absorción acústica en recintos (EN 12354-6)](/phonometry/es/buildings/rooms/enclosed-space-absorption/):
  el área de absorción acústica equivalente total de un recinto a partir de sus
  superficies, objetos y aire, y el tiempo de reverberación que se deduce.

**[Aislamiento acústico](/phonometry/es/buildings/insulation/)**

- [Medición del aislamiento en campo (ISO 16283)](/phonometry/es/buildings/insulation/insulation-field/):
  el aislamiento a ruido aéreo y de impactos medido en el edificio, su informe de
  ensayo y la incertidumbre de ISO 12999-1 que lo cualifica.
- [Medición del aislamiento en laboratorio](/phonometry/es/buildings/insulation/insulation-lab/):
  la caracterización ISO 10140 de un elemento con los flancos suprimidos.
- [Aislamiento acústico por intensidad (ISO 15186)](/phonometry/es/buildings/insulation/insulation-intensity/):
  la potencia transmitida leída sobre la cara radiante, del elemento completo o
  elemento a elemento.
- [Método de control del aislamiento (ISO 10052)](/phonometry/es/buildings/insulation/insulation-survey/):
  el método de control en bandas de octava con su índice de reverberación y sus
  magnitudes aérea, de impactos, de fachada y de equipos de servicio.
- [Transmisión por flancos en laboratorio (ISO 10848)](/phonometry/es/buildings/insulation/flanking-lab/):
  el índice de reducción de vibraciones de unión y las diferencias de niveles de
  flanco medidas en una instalación de ensayo.
- [Fuentes de impacto pesadas y blandas (ISO 16283-2)](/phonometry/es/buildings/insulation/heavy-impact-sources/):
  la pelota de caucho y la máquina de neumático: el nivel de exposición a la
  fuerza de impacto que las especifica, la comprobación de laboratorio que tienen
  que pasar y el número único del anexo D de ISO 717-2.
- [Índices globales de aislamiento (ISO 717)](/phonometry/es/buildings/insulation/insulation-ratings/):
  los motores de curva de referencia aéreo y de impactos con $C$, $C_{tr}$ y
  $C_I$, los términos de rango ampliado y la ficha de ISO 717.
- [Aislamiento acústico de fachadas](/phonometry/es/buildings/insulation/facade-insulation/):
  el cerramiento del edificio medido según ISO 16283-3, previsto según
  EN 12354-3 y radiando al exterior según EN 12354-4.
- [Código Técnico de la Edificación (CTE DB-HR)](/phonometry/es/buildings/insulation/spanish-building-code/):
  las magnitudes globales $R_A$, $R_{A,tr}$, $D_{nT,A}$ y $D_{2m,nT,Atr}$ del
  DB HR, las tablas de exigencias del apartado 2 y la corrección por tamaño de
  ventana.

**[Diseño del aislamiento](/phonometry/es/buildings/design/)**

- [Predicción del aislamiento acústico (EN 12354)](/phonometry/es/buildings/design/insulation-prediction/):
  el aislamiento in situ a ruido aéreo y de impactos entre recintos a partir de
  datos de elementos, con sus caminos de flancos.
- [Predicción detallada por bandas (ISO 12354)](/phonometry/es/buildings/design/detailed-prediction/):
  la misma predicción banda a banda en vez de como un solo número: datos in situ
  de elemento y de unión, el índice por flancos y el nivel de impactos, y la
  contribución de cada camino a R'w y L'n,w.
- [Predicción del aislamiento de paneles](/phonometry/es/buildings/design/panel-sound-insulation/):
  la ley de masa y la caída de coincidencia, las paredes dobles, rendijas y
  aberturas, la eficiencia de radiación de placas y las movilidades puntuales.
- [Mejora del aislamiento a impacto de suelos (ISO 16251-1)](/phonometry/es/buildings/design/impact-improvement/):
  la mejora ponderada de un revestimiento de suelo blando medida sobre una
  maqueta pesada pequeña.
- [Predicción del comportamiento de capas elásticas](/phonometry/es/buildings/design/resilient-layers/):
  el modelo de fuerza de la máquina de impactos, la frecuencia de corte de un
  revestimiento blando, las leyes de mejora del suelo flotante y la magnitud
  global de un trasdosado según el anexo D de ISO 12354-1.
- [Rigidez dinámica de materiales resilientes (EN 29052-1)](/phonometry/es/materials/resilient/dynamic-stiffness/):
  la rigidez por unidad de superficie bajo un suelo flotante a partir de la
  resonancia con placa de carga, con el término del gas encerrado.

## [Materiales y superficies](/phonometry/es/materials/)

Absorción, resistencia al flujo de aire, tubo de impedancia, modelos porosos
y de metamaterial, difusores y dispersión. Qué le hace una superficie al
sonido que le llega, medido en laboratorio o predicho a partir de los
parámetros del material. Implementa ISO 354, ISO 11654, ISO 10534-1/-2,
ISO 9053-1/-2, ISO 17497-1/-2, ISO 13472-1/-2, EN 29052-1 e ISO 12999-2.

- [Medida y clasificación de la absorción sonora](/phonometry/es/materials/absorbers/absorption-measurement/):
  la medición en cámara reverberante ISO 354, la valoración ponderada con su
  clase y la incertidumbre de medida de ambas.
- [Resistencia al flujo de aire](/phonometry/es/materials/absorbers/airflow-resistance/):
  la determinación estática y alterna de la resistencia y la resistividad al
  flujo de aire.
- [Tubo de impedancia](/phonometry/es/materials/absorbers/impedance-tube/): la impedancia
  superficial, la absorción y la pérdida por transmisión a incidencia normal,
  más el tubo FDTD virtual.
- [Absorbentes porosos y multicapa](/phonometry/es/materials/absorbers/porous-absorbers/): los
  modelos de Delany-Bazley, Miki y Johnson-Champoux-Allard, el solver multicapa
  por matriz de transferencia con capas perforadas, microperforadas y de
  membrana, y la integral de incidencia aleatoria.
- [Metaabsorbentes](/phonometry/es/materials/absorbers/metamaterial-absorbers/): la
  condición de acoplamiento crítico para la absorción perfecta y el panel
  ranurado de sonido lento cargado con resonadores de Helmholtz, con su
  solucionador de diseño.
- [Difusores y sus coeficientes](/phonometry/es/materials/diffusers/diffusers/): el
  coeficiente de dispersión de incidencia aleatoria, el coeficiente de
  difusión por autocorrelación y el diseño de Schroeder con su predicción en
  campo lejano.
- [Metadifusores](/phonometry/es/materials/diffusers/metadiffusers/): difusores de
  Schroeder en sublongitud de onda profunda a partir de rendijas cargadas con
  resonadores, sonido lento y secuencias ternarias.
- [Absorción in situ de firmes de carretera](/phonometry/es/materials/surfaces/road-absorption/):
  la absorción in situ de pavimentos por la técnica de sustracción y el
  método puntual.

## [Vibración y ruido estructural](/phonometry/es/vibration/)

Movilidad y funciones de respuesta en frecuencia, aisladores, potencia radiada,
uniones y vibración en humanos. El área cubre el camino que sigue una máquina
al entrar en una estructura y volver a salir como sonido aéreo, y la cuestión
aparte de qué le hace la vibración a la persona expuesta. Implementa
ISO 7626-1/-2, ISO 10846-1/-2/-3, ISO 9611, ISO/TS 7849-1/-2, EN 15657,
EN 12354-5, ISO 2631-1/-2/-4/-5, ISO 5349-1/-2 e ISO 8041-1.

**[Fuentes de ruido estructural](/phonometry/es/vibration/structural/)**

- [Movilidad mecánica y la familia de FRF (ISO 7626-1)](/phonometry/es/vibration/structural/mechanical-mobility/):
  receptancia, movilidad y acelerancia con sus recíprocas, la conversión entre
  ellas y el resonador de un grado de libertad en forma cerrada.
- [Transmisión de onda de flexión en uniones de placas](/phonometry/es/vibration/structural/junction-transmission/):
  los coeficientes de transmisión por el método de ondas para uniones rígidas en
  X, T, L y en línea, su promedio en campo difuso, y el factor de pérdidas por
  acoplamiento y el índice de reducción vibracional.
- [Rigidez dinámica de transferencia (ISO 10846)](/phonometry/es/vibration/structural/transfer-stiffness/):
  la rigidez dinámica de transferencia y el factor de pérdidas de un aislador por
  los métodos directo e indirecto.
- [Potencia acústica desde vibración (ISO/TS 7849)](/phonometry/es/devices/emission/vibration-sound-power/):
  la potencia radiada a partir del nivel de velocidad promediado en la superficie
  y el factor de radiación, con el límite superior de la Parte 1 y el valor de
  ingeniería de la Parte 2.
- [Potencia acústica estructural de equipos (EN 15657)](/phonometry/es/buildings/design/structure-borne-power/):
  el método de la placa de recepción y las magnitudes de fuente independientes de
  la placa, fuerza bloqueada equivalente, velocidad libre y movilidad de la
  fuente.
- [Ruido estructural instalado (EN 12354-5)](/phonometry/es/buildings/design/installed-structure-borne/):
  el término de acoplamiento a partir de las movilidades de fuente y receptor, la
  potencia instalada y el nivel de presión acústica por vía en el recinto receptor.

**[Vibración en humanos](/phonometry/es/vibration/human/)**

- [Vibración en humanos](/phonometry/es/vibration/human/human-vibration/): exposición de
  cuerpo completo y mano-brazo con las ponderaciones de ISO 8041-1, el valor
  eficaz ponderado y las medidas de dosis, y la exposición diaria $A(8)$.
- [Vibración con choques múltiples (ISO 2631-5)](/phonometry/es/vibration/human/multiple-shock-vibration/):
  la función de transferencia asiento-columna, la dosis de aceleración y la
  variable de tensión acumulada que hay detrás de la probabilidad de lesión
  lumbar.

## [Medio ambiente y transporte](/phonometry/es/environment/)

Propagación en exteriores, barreras, refracción, fuentes viarias,
ferroviarias y de aerogenerador, y la valoración construida sobre ellas. Todo
lo de aquí trata de sonido que tiene que recorrer una distancia larga antes de
valorarse, así que la atmósfera, el suelo y el propio movimiento de la fuente
entran en la respuesta. Implementa ISO 9613-1/-2, ISO 1996-1/-2,
ISO/PAS 1996-3, NT ACOU 112, CNOSSOS-EU (Directiva 2002/49/CE, anexo II) e
IEC 61400-11.

**[Sonido en exteriores](/phonometry/es/environment/propagation/)**

- [Propagación del sonido en exteriores](/phonometry/es/environment/propagation/outdoor-propagation/):
  la absorción atmosférica y el método general de ISO 9613-2, con el desglose de
  la atenuación por bandas de octava y por términos.
- [Emisión de la fuente de tráfico viario CNOSSOS-EU](/phonometry/es/environment/sources/cnossos-road-emission/):
  la fuente viaria común europea del Anexo II 2.2: potencia acústica de rodadura
  y propulsión por categoría de vehículo y potencia direccional por metro de
  línea fuente.
- [Emisión de la fuente ferroviaria CNOSSOS-EU](/phonometry/es/environment/sources/cnossos-rail-emission/):
  la fuente ferroviaria del anexo II 2.3: rugosidad y filtro de contacto, ruido
  de impacto, chirrido en curva, tracción y ruido aerodinámico, y las dos líneas
  de fuente equivalentes.
- [Ruido de aerogeneradores: potencia y audibilidad tonal](/phonometry/es/environment/sources/wind-turbine-noise/):
  el nivel de potencia acústica aparente referido al centro del rotor y la cadena
  de audibilidad tonal que decide si un tono se oye.
- [Efecto suelo esférico y barreras avanzadas](/phonometry/es/environment/propagation/ground-barriers/):
  el coeficiente de reflexión de Weyl-Van der Pol sobre suelo de impedancia
  finita y la difracción en barreras desde la teoría ondulatoria.
- [Refracción atmosférica: rayos y GFPE](/phonometry/es/environment/propagation/atmospheric-refraction/):
  los perfiles de velocidad efectiva del sonido, los rayos curvos con distancia
  de zona de sombra en forma cerrada y el campo de nivel relativo por GFPE.
- [Prominencia de sonidos impulsivos (NT ACOU 112)](/phonometry/es/environment/assessment/impulsive-sound/):
  la prominencia prevista de cada impulso a partir de su tasa de ataque y su
  diferencia de nivel, y el ajuste que se suma al $L_{Aeq}$.

## [Ruido de aeronaves](/phonometry/es/aircraft/)

Niveles de certificación, contornos de aeropuerto y el método del
hemisferio para helicópteros: el ruido del vuelo medido como lo prescriben
los documentos de certificación y de planificación aeroportuaria.
Implementa el Anexo 16 de la OACI, IEC 61265, SAE ARP 866B/5534 y
ECAC Doc 29/32.

- [Ruido de aeronaves: nivel efectivo de ruido percibido](/phonometry/es/aircraft/aircraft-noise/):
  la ruidosidad percibida y el PNL, la corrección por tonos, la corrección por
  duración y el verificador del sistema de medida.
- [Ruido de aeropuertos (ECAC Doc 29)](/phonometry/es/aircraft/airport-noise/):
  el motor nivel-potencia-distancia, la cadena de evento único por segmentos y
  el contorno de SEL en malla de tierra.
- [Ruido de rotorcraft: el método del hemisferio](/phonometry/es/aircraft/rotorcraft-noise/):
  el modelo de fuente en hemisferio con sus ajustes de propagación, la
  interpolación de condiciones de vuelo y los contornos de suceso único.
- [La base de datos ANP de flota](/phonometry/es/aircraft/anp-fleet/): las tablas
  de EASA con curvas nivel-potencia-distancia y trayectorias por defecto, y la
  cadena de Doc 29 ejecutada desde un identificador de aeronave.

## [Acústica submarina](/phonometry/es/underwater/)

Niveles referidos a 1 micropascal, ruido radiado por buques, hincado de
pilotes, ruido ambiente y pérdidas por transmisión. Las magnitudes de
referencia no son las del aire, así que esta es la única área en la que un
nivel no se puede leer de una a otra sin conversión. Implementa ISO 18405,
ISO 17208-1/-2, ISO 18406 y JOMOPANS-ECHO.

- [Acústica submarina: ruido radiado e hincado de pilotes](/phonometry/es/underwater/underwater-acoustics/):
  los niveles de referencia de ISO 18405, el nivel de ruido radiado por buques y
  el nivel de fuente monopolar equivalente, y la exposición por golpe único y
  acumulada del hincado de pilotes.
- [Propagación submarina del sonido](/phonometry/es/underwater/underwater-propagation/):
  la divergencia geométrica más la absorción de volumen, la velocidad del sonido
  en agua de mar, la ecuación del sonar, las pérdidas por reflexión en el fondo y
  el espectro de ruido ambiente.
- [Solvers numéricos de propagación submarina](/phonometry/es/underwater/underwater-solvers/):
  los solvers de modos normales, trazado de rayos y ecuación parabólica de la
  guía de ondas estratificada, y cómo elegir modelo de propagación.

- [Exposición a ruido de mamíferos marinos](/phonometry/es/underwater/marine-mammal-exposure/):
  la cara auditiva de ese ruido: los audiogramas de grupo, las funciones de
  ponderación reglamentarias con su versión de guía, y la exposición de una
  campaña de hincado frente a los criterios de lesión.

## [Fuentes y dispositivos](/phonometry/es/devices/)

Potencia acústica, intensidad, declaraciones de emisión, electroacústica y
sonoridad de programa. Lo que emite una fuente y no lo que recibe un receptor,
más la cadena electroacústica que lo reproduce o lo mide. Implementa ISO 3741,
ISO 3744/3746, ISO 3745, ISO 9614-1/-2/-3, IEC 61043, ISO 4871,
IEC 60268-3/-4/-5, ITU-R BS.1770-5 y EBU R 128.

**[Potencia acústica e intensidad](/phonometry/es/devices/emission/)**

- [Intensidad acústica (p-p)](/phonometry/es/devices/emission/intensity/): la intensidad
  con dos micrófonos y los indicadores de campo que cualifican la medida.
- [Potencia acústica](/phonometry/es/devices/emission/sound-power/): la elección del
  método de determinación y la declaración de emisión de ruido según ISO 4871.
- [Potencia acústica por métodos de presión](/phonometry/es/devices/emission/sound-power-pressure/):
  la superficie envolvente de ISO 3744/3746 y el grado de precisión anecoico de
  ISO 3745.
- [Potencia acústica en cámara reverberante](/phonometry/es/devices/emission/sound-power-reverberation/):
  los métodos directo y de comparación de ISO 3741.
- [Potencia acústica por barrido de intensidad](/phonometry/es/devices/emission/sound-power-intensity/):
  el barrido in situ de ISO 9614-2 y el grado de precisión ISO 9614-3.

**[Electroacústica](/phonometry/es/devices/electroacoustics/)**

- [Electroacústica: distorsión y respuesta en frecuencia](/phonometry/es/devices/electroacoustics/electroacoustics/):
  la distorsión armónica y de intermodulación, THD+N y SINAD, el rango dinámico
  y los estimadores $H_1$/$H_2$ de respuesta en frecuencia.
- [Caracterización de altavoces (IEC 60268-5)](/phonometry/es/devices/electroacoustics/loudspeakers/):
  los convenios de sensibilidad, el pistón radiante y la ficha de
  características.
- [Caracterización de micrófonos (IEC 60268-4)](/phonometry/es/devices/electroacoustics/microphones/):
  las referencias de sensibilidad, los patrones direccionales y el ruido propio.
- [Distorsión con barridos y utilidades de fase](/phonometry/es/devices/electroacoustics/swept-sine-distortion/):
  la separación de armónicos a partir de un solo barrido exponencial, la THD
  frente a la frecuencia de excitación, y la fase mínima, el retardo de grupo y
  la fase en exceso.
- [Sonoridad de programa y pico verdadero](/phonometry/es/devices/broadcast/program-loudness/):
  la ponderación K y la sonoridad integrada con puerta en LUFS, los medidores
  momentáneo y de corto plazo, el rango de sonoridad y el nivel de pico
  verdadero.

**[Control de ruido](/phonometry/es/devices/noise-control/)**

- [Silenciadores](/phonometry/es/devices/noise-control/silencers/): los silenciadores
  reactivos por el método de matrices de cuatro polos y la elección entre
  reactivo y disipativo.
- [Ruido por conductos: del ventilador a la sala](/phonometry/es/devices/noise-control/duct-path/):
  el cálculo completo del ventilador a la sala frente a un criterio de ruido de
  fondo, y el corte de modos superiores que limita todo método de onda plana.
- [Entre recintos: partición, receptor y criterio](/phonometry/es/devices/noise-control/room-to-room/):
  la cadena compuesta del recinto emisor al recinto receptor y la pérdida por
  transmisión que necesita una partición o un encapsulado para cumplir un
  criterio de ruido de fondo.
- [Control de ruido industrial: HVAC y cerramientos](/phonometry/es/devices/noise-control/noise-control/):
  la atenuación y el ruido de flujo en conductos, y la pérdida por inserción de
  cerramientos.

## [Simulación de ondas](/phonometry/es/simulation/)

Solvers deterministas de diferencias finitas en el dominio del tiempo en 2D,
acústico y elástico P-SV, validados frente a oráculos analíticos y no frente a
una norma. Es la única área sin documento aplicable, así que su evidencia es la
solución en forma cerrada que reproduce.

- [Simulación de ondas FDTD 2D](/phonometry/es/simulation/fdtd-simulation/): una
  malla escalonada de presión y velocidad con fuentes gaussianas, tonales y de
  señal arbitraria, obstáculos rasterizados, contornos rígidos, de impedancia y
  absorbentes, y un resultado inmutable con los historiales de sonda y las
  instantáneas del campo.
- [Ondas elásticas y acoplamiento fluido-sólido](/phonometry/es/simulation/elastic-waves/):
  el solucionador compañero P-SV sobre la misma malla, con ondas de Rayleigh
  en superficies libres, conversión de modo, ondas de interfase de Scholte y
  transmisión de placas sumergidas.
