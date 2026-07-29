---
title: "Análisis de señal"
description: "El núcleo de medición de phonometry: bancos de filtros de octava fraccional, ponderación frecuencial y temporal, niveles integrados y estadísticos, análisis espectral y de correlación calibrado, calibración física e incertidumbre de medida, y cómo esas piezas se encadenan en un sonómetro en código."
---

Todo en phonometry empieza aquí. Esta sección cubre la cadena que convierte
una señal digital en bruto en números acústicos conformes con las normas:
dividirla en **bandas de octava fraccional** (ANSI S1.11 / IEC 61260-1),
moldearla con las **ponderaciones frecuenciales** de IEC 61672-1, suavizarla
con las **balísticas temporales Fast/Slow/Impulse** e integrarla en **Leq y
niveles estadísticos**. Es, en la práctica, un sonómetro descompuesto en
funciones componibles, y todas las demás secciones de la documentación se
apoyan en él: un modelo de sonoridad consume niveles de banda calibrados, un
parámetro de sala parte de una respuesta al impulso filtrada, una valoración
ambiental es un Leq ajustado.
[Construye un sonómetro](/phonometry/es/guides/sound-level-meter/) monta esa
cadena de principio a fin en una sola página ejecutable; es el mejor punto de
partida si quieres ver el área entera en funcionamiento antes de abrir las
guías de fondo.

Alrededor de la cadena de niveles están las herramientas generales de
análisis de señal: las **estimaciones espectrales calibradas** (PSD y
densidad espectral cruzada de Welch con intervalos de confianza), la
**correlación y la estimación de retardo** y la **envolvente de Hilbert**,
todas expresadas con el análisis de error de Bendat y Piersol. Y dos
preocupaciones transversales completan el núcleo. La **calibración** decide
qué significan físicamente las muestras digitales: los resultados pueden
referirse a un tono de calibrador medido o a una sensibilidad conocida
(dB SPL), o quedarse en escala digital completa (dBFS). La **incertidumbre
de medida** (la GUM y su suplemento de Monte Carlo) cualifica cualquier
resultado calculado a partir de entradas inciertas, que es lo que hace
defendible un número en un informe.

Si acabas de llegar a la biblioteca, lee primero
[Bancos de filtros](/phonometry/es/guides/filter-banks/): presenta la
descomposición en bandas que el resto de páginas da por supuesta. Después,
[Niveles integrados y estadísticos](/phonometry/es/guides/levels/) muestra las
métricas en las que terminan la mayoría de las mediciones, y
[Calibración y dBFS](/phonometry/es/guides/calibration/) las ancla a unidades
físicas.

## [Filtrado en octavas](/phonometry/es/guides/sections/octave-filtering/)

La descomposición en bandas de octava fraccional y las dos maneras de
escalarla: bloques en streaming y arrays multicanal.

- [Bancos de filtros](/phonometry/es/guides/filter-banks/): la matemática de
  las bandas de octava fraccional, los parámetros del banco, el ecualizador
  paramétrico, la descomposición en bandas y el filtrado de fase cero fuera
  de línea.
- [Galería de arquitecturas de filtro](/phonometry/es/guides/filter-gallery/):
  las cinco arquitecturas de filtro comparadas, la galería completa de
  respuestas y el uso de cada arquitectura, con el crossover Linkwitz-Riley.
- [Verificación de clase de filtros (IEC 61260-1)](/phonometry/es/guides/filter-compliance/):
  la máscara de aceptación de la Tabla 1 banda a banda, la clase 0 de la
  edición retirada de 1995 y la ficha de conformidad.
- [Procesado por bloques](/phonometry/es/guides/block-processing/): análisis
  en streaming con estado, que arrastra el estado de los filtros entre
  búferes, para señales que no caben en memoria.
- [Multicanal y rendimiento](/phonometry/es/guides/multichannel/): análisis
  vectorizado de muchos canales a la vez, con notas de rendimiento.

## [Niveles y ponderación](/phonometry/es/guides/sections/levels-weighting/)

De la señal ponderada al nivel reportado: las ponderaciones frecuenciales,
las balísticas temporales y los niveles integrados, estadísticos y de
valoración.

- [Ponderación frecuencial (A, C, Z)](/phonometry/es/guides/weighting/):
  las curvas de respuesta del oído de IEC 61672-1, el modo de precisión en
  alta frecuencia y la verificación de clase de la Tabla 3.
- [Ponderaciones especiales (G, B, D, AU)](/phonometry/es/guides/special-weightings/):
  la ponderación G para infrasonido de ISO 7196, las curvas históricas B y D
  y la AU según IEC 61012.
- [Ponderación temporal](/phonometry/es/guides/time-weighting/): las
  balísticas exponenciales Fast, Slow e Impulse según IEC 61672-1.
- [Niveles integrados y estadísticos](/phonometry/es/guides/levels/): Leq y
  LAeq, niveles percentiles L10/L50/L90, LCpeak y SEL, dosis de ruido
  (IEC 61252), y espectrogramas de octava.
- [Niveles ambientales (ISO 1996-1/-2)](/phonometry/es/guides/environmental-levels/):
  Lden, Ldn y los niveles de valoración compuestos, el ajuste tonal, la
  corrección de ruido residual y el presupuesto de incertidumbre.
- [Normativa española de ruido (RD 1367/2007)](/phonometry/es/guides/spanish-noise-regulation/):
  el nivel corregido LKeq, las correcciones Kt/Kf/Ki, los periodos temporales
  de evaluación y las fases de ruido, y las tablas de valores límite.

## [Señales y espectros](/phonometry/es/guides/sections/signals-spectra/)

Análisis de grano fino en frecuencia y en tiempo, con cada estimación
calibrada y acompañada de su calidad estadística.

- [Análisis espectral calibrado](/phonometry/es/guides/spectral-analysis/):
  los estimadores de Welch de Bendat y Piersol con su calidad estadística:
  PSD y densidad espectral cruzada con intervalos de confianza chi-cuadrado,
  el espectro de salida coherente con la SNR espectral, suavizado en 1/n de
  octava y generadores de ruido de colores con pendiente exacta.
- [Coherencia múltiple y parcial](/phonometry/es/guides/miso-coherence/): las
  funciones de coherencia de entradas múltiples para varias fuentes
  correladas y una salida, con el condicionamiento que distingue una causa
  real de una fuente que solo correla con ella, y los espectros de salida
  coherente parciales que indican qué fuente domina cada banda.
- [Análisis tiempo-frecuencia](/phonometry/es/guides/time-frequency/): el
  espectrograma STFT calibrado en unidades absolutas (dB SPL para pascales)
  y la FFT con zoom que resuelve tonos más próximos que un bin práctico de
  FFT.
- [Cepstro, ecos y espectro de la envolvente](/phonometry/es/guides/cepstrum-echoes/):
  el cepstro de potencia, real y complejo con análisis de quefrencia,
  detección de ecos con el coeficiente de reflexión leído en el pico
  cepstral, liftering paso bajo/paso alto de un espectro logarítmico, y el
  espectro de la envolvente que convierte las modulaciones de amplitud en
  líneas discretas.
- [Promediado síncrono en el tiempo](/phonometry/es/guides/synchronous-averaging/):
  extracción de una forma de onda periódica de período conocido por promediado
  en el dominio del tiempo, el filtro peine que lo describe en el dominio de la
  frecuencia, la ley de reducción de ruido en raíz cuadrada, y la elección del
  número de promedios que sitúa un nodo del peine sobre un orden interferente
  (McFadden 1987).
- [Correlación, retardo y envolvente](/phonometry/es/guides/correlation-delay/):
  estimaciones de correlación con los errores aleatorios de Bendat y
  Piersol, estimación del retardo por correlación directa, pendiente de fase
  del espectro cruzado y las ponderaciones GCC de Knapp y Carter, retardo
  submuestral y alineación de respuestas al impulso, y la envolvente de
  Hilbert.
- [Señales de prueba y herramientas de muestreo](/phonometry/es/guides/test-signals/):
  salvas de tono IEC 60268-1 con conmutación exacta, remuestreo con
  especificación antialias declarada y retardo fraccionario de banda limitada.
- [Medición de sistemas](/phonometry/es/guides/system-measurement/):
  pares complementarios de Golay, barridos con un espectro de magnitud
  objetivo arbitrario conformando el retardo de grupo, y la inversión con
  regularización de Kirkeby de una respuesta medida.

## [Calibración e incertidumbre](/phonometry/es/guides/sections/calibration-uncertainty/)

Qué significan los números y cuánto fiarse de ellos.

- [Calibración y dBFS](/phonometry/es/guides/calibration/): calibración SPL
  física a partir de un tono de calibrador (IEC 60942) o de una sensibilidad
  conocida, y el modo digital dBFS.
- [Incertidumbre de medida (GUM y Monte Carlo)](/phonometry/es/guides/gum-uncertainty/):
  la ley de propagación de la incertidumbre y el método de Monte Carlo de
  ISO/IEC Guide 98-3, con incertidumbre expandida e intervalos de cobertura.
- [Calificación de datos](/phonometry/es/guides/data-qualification/): los tests
  de estacionariedad por inversiones de orden y por rachas sobre estadísticas
  de segmento, y las estadísticas de Rice de cruces por nivel y de picos con el
  factor de irregularidad.
