---
title: "Análisis de señal"
description: "El núcleo de medición de phonometry: bancos de filtros de fracción de octava, ponderación frecuencial y temporal, niveles integrados y estadísticos, análisis espectral y de correlación calibrado, calibración física e incertidumbre de medida, y cómo esas piezas se encadenan en un sonómetro en código."
---

Todo en phonometry empieza aquí. Esta sección cubre la cadena que convierte
una señal digital en bruto en números acústicos conformes con las normas:
dividirla en **bandas de fracción de octava** (ANSI S1.11 / IEC 61260-1),
moldearla con las **ponderaciones frecuenciales** de IEC 61672-1, suavizarla
con las **balísticas temporales Fast/Slow/Impulse** e integrarla en **Leq y
niveles estadísticos**. Es, en la práctica, un sonómetro descompuesto en
funciones componibles, y todas las demás secciones de la documentación se
apoyan en él: un modelo de sonoridad consume niveles de banda calibrados, un
parámetro de sala parte de una respuesta al impulso filtrada, una valoración
ambiental es un Leq ajustado.

Alrededor de la cadena de niveles están las herramientas generales de
análisis de señal: las **estimaciones espectrales calibradas** (PSD y
densidad espectral cruzada de Welch con intervalos de confianza), la
**correlación y la estimación de retardo** y la **envolvente de Hilbert**,
todas expresadas con el análisis de error de Bendat y Piersol. Y dos
preocupaciones transversales completan el núcleo. La **calibración** decide
qué significan físicamente las muestras digitales: los resultados pueden
referirse a un tono de calibrador medido (dB SPL), o quedarse en escala
digital completa (dBFS). La **incertidumbre de medida** (la GUM y su
suplemento de Monte Carlo) cualifica cualquier resultado calculado a partir de
entradas inciertas, que es lo que hace defendible un número en un informe.

Tres convenios recorren todas las páginas de abajo, y todos los snippets del
sitio los dan por supuestos. Una señal es un array de NumPy de presión
acústica con **el tiempo en el último eje**, así que un canal es `(n,)` y
varios canales en paralelo son `(canales, muestras)`. La frecuencia de
muestreo viaja siempre como argumento `fs` explícito: no se lee nada de la
cabecera de un archivo, porque la biblioteca nunca abre el archivo. Y se
espera que el array contenga **pascales**, que es la razón de que una función
de nivel aplicada a muestras en bruto de tarjeta de sonido devuelva un número
con una referencia arbitraria, y de que toda función de nivel acepte también
un `calibration_factor` en pascales por unidad digital o la salida de
emergencia `dbfs=True`. Las métricas simples vuelven como floats y arrays; las
más ricas vuelven como objetos de resultado inmutables que exponen `.plot()`.
[Calibración y dBFS](/phonometry/es/signals/metrology/calibration/) resuelve el
tercer convenio por completo, y [Multicanal y
rendimiento](/phonometry/es/signals/filters/multichannel/) el primero.

Dos maneras de entrar. Para ver la cadena entera funcionando de una vez,
ejecuta [Construye un sonómetro](/phonometry/es/signals/sound-level-meter/):
calibra contra un tono de calibrador, aplica las ponderaciones frecuencial y
temporal, integra en Leq, SEL y niveles percentiles, divide la señal en bandas
de octava y comprueba la clase de cada etapa, en una sola página ejecutable.
Para aprender las piezas en orden de dependencia, empieza por [Bancos de
filtros](/phonometry/es/signals/filters/filter-banks/), que presenta la
descomposición en bandas que el resto de páginas da por supuesta, después
[Niveles integrados y estadísticos](/phonometry/es/signals/levels/levels/) para
las métricas en las que terminan la mayoría de las mediciones, y [Calibración
y dBFS](/phonometry/es/signals/metrology/calibration/) para anclarlas a
pascales.

## [Construye un sonómetro](/phonometry/es/signals/sound-level-meter/)

- [Construye un sonómetro](/phonometry/es/signals/sound-level-meter/): la cadena
  entera montada en una sola página ejecutable (la calibración, las
  ponderaciones frecuencial y temporal, los niveles integrados y estadísticos,
  la descomposición en bandas y el veredicto de clase de cada etapa) como
  introducción resuelta a las cuatro subsecciones de abajo.

## [Filtrado en octavas](/phonometry/es/signals/filters/)

La descomposición en bandas de fracción de octava y las dos maneras de
escalarla: bloques en streaming y arrays multicanal.

- [Bancos de filtros](/phonometry/es/signals/filters/filter-banks/): la matemática de
  las bandas de fracción de octava, los parámetros del banco, el ecualizador
  paramétrico, la descomposición en bandas y el filtrado de fase cero fuera
  de línea.
- [Galería de arquitecturas de filtro](/phonometry/es/signals/filters/filter-gallery/):
  las cinco arquitecturas de filtro comparadas, la galería completa de
  respuestas y el uso de cada arquitectura, con el crossover Linkwitz-Riley.
- [Verificación de clase de filtros (IEC 61260-1)](/phonometry/es/signals/filters/filter-compliance/):
  la máscara de aceptación de la Tabla 1 banda a banda, la clase 0 de la
  edición retirada de 1995 y la ficha de conformidad.
- [Procesado por bloques](/phonometry/es/signals/filters/block-processing/): análisis
  en streaming con estado, que arrastra el estado de los filtros entre
  búferes, para señales que no caben en memoria.
- [Multicanal y rendimiento](/phonometry/es/signals/filters/multichannel/): análisis
  vectorizado de muchos canales a la vez, con notas de rendimiento.

## [Niveles y ponderación](/phonometry/es/signals/levels/)

De la señal ponderada al nivel declarado: las ponderaciones frecuenciales,
las balísticas temporales y los niveles integrados, estadísticos y de
valoración.

- [Ponderación frecuencial (A, C, Z)](/phonometry/es/signals/levels/weighting/):
  las curvas de respuesta del oído de IEC 61672-1, el modo de precisión en
  alta frecuencia y la verificación de clase de la Tabla 3.
- [Ponderaciones especiales (G, B, D, AU)](/phonometry/es/signals/levels/special-weightings/):
  la ponderación G para infrasonido de ISO 7196, las curvas históricas B y D
  y la AU según IEC 61012.
- [Ponderación temporal](/phonometry/es/signals/levels/time-weighting/): las
  balísticas exponenciales Fast, Slow e Impulse según IEC 61672-1.
- [Niveles integrados y estadísticos](/phonometry/es/signals/levels/levels/): Leq y
  LAeq, niveles percentiles L10/L50/L90, LCpeak y SEL, dosis de ruido
  (IEC 61252), y espectrogramas de octava.
- [Niveles ambientales (ISO 1996-1/-2)](/phonometry/es/environment/assessment/environmental-levels/):
  Lden, Ldn y los niveles de evaluación compuestos, el ajuste tonal, la
  corrección de ruido residual y el balance de incertidumbre.
- [Normativa española de ruido (RD 1367/2007)](/phonometry/es/environment/assessment/spanish-noise-regulation/):
  el nivel corregido LKeq, las correcciones Kt/Kf/Ki, los periodos temporales
  de evaluación y las fases de ruido, y las tablas de valores límite.

## [Señales y espectros](/phonometry/es/signals/spectra/)

Análisis de grano fino en frecuencia y en tiempo, con cada estimación
calibrada y acompañada de su calidad estadística.

- [Análisis espectral calibrado](/phonometry/es/signals/spectra/spectral-analysis/):
  los estimadores de Welch de Bendat y Piersol con su calidad estadística:
  PSD y densidad espectral cruzada con intervalos de confianza chi-cuadrado,
  el espectro de salida coherente con la SNR espectral, suavizado en 1/n de
  octava y generadores de ruido de colores con pendiente exacta.
- [Coherencia múltiple y parcial](/phonometry/es/signals/spectra/miso-coherence/): las
  funciones de coherencia de entradas múltiples para varias fuentes
  correlacionadas y una salida, con el condicionamiento que distingue una causa
  real de una fuente que solo se correlaciona con ella, y los espectros de salida
  coherente parciales que indican qué fuente domina cada banda.
- [Análisis tiempo-frecuencia](/phonometry/es/signals/spectra/time-frequency/): el
  espectrograma STFT calibrado en unidades absolutas (dB SPL para pascales)
  y la FFT con zoom que resuelve tonos más próximos que un bin práctico de
  FFT.
- [Cepstrum, ecos y espectro de la envolvente](/phonometry/es/signals/spectra/cepstrum-echoes/):
  el cepstrum de potencia, real y complejo con análisis de quefrencia,
  detección de ecos con el coeficiente de reflexión leído en el pico
  cepstral, liftering paso bajo/paso alto de un espectro logarítmico, y el
  espectro de la envolvente que convierte las modulaciones de amplitud en
  líneas discretas.
- [Promediado síncrono en el tiempo](/phonometry/es/signals/spectra/synchronous-averaging/):
  extracción de una forma de onda periódica de periodo conocido por promediado
  en el dominio del tiempo, el filtro peine que lo describe en el dominio de la
  frecuencia, la ley de reducción de ruido en raíz cuadrada, y la elección del
  número de promedios que sitúa un nodo del peine sobre un orden interferente
  (McFadden 1987).
- [Frecuencias de fallo de máquinas](/phonometry/es/vibration/machinery/machine-diagnostics/)
  (en la sección de vibraciones): las familias cinemáticas de frecuencias de
  fallo de la maquinaria rotativa
  (Norton y Karczub, sección 8.4) dibujadas sobre un espectro de envolvente
  medido: frecuencias BPFO, BPFI, BSF y de jaula de los rodamientos, bandas
  laterales de engrane, deslizamiento, paso de polos y armónicos de ranura de
  los motores de inducción, y tonos de paso de pala.
- [Correlación, retardo y envolvente](/phonometry/es/signals/spectra/correlation-delay/):
  estimaciones de correlación con los errores aleatorios de Bendat y
  Piersol, estimación del retardo por correlación directa, pendiente de fase
  del espectro cruzado y las ponderaciones GCC de Knapp y Carter, retardo
  submuestral y alineación de respuestas al impulso, y la envolvente de
  Hilbert.
- [Señales de prueba y herramientas de muestreo](/phonometry/es/signals/spectra/test-signals/):
  ráfagas tonales IEC 60268-1 con conmutación exacta, remuestreo con
  especificación antialias declarada y retardo fraccionario de banda limitada.
- [Medición de sistemas](/phonometry/es/signals/spectra/system-measurement/):
  pares complementarios de Golay, barridos con un espectro de magnitud
  objetivo arbitrario conformando el retardo de grupo, y la inversión con
  regularización de Kirkeby de una respuesta medida.

## [Calibración e incertidumbre](/phonometry/es/signals/metrology/)

Qué significan los números y cuánto fiarse de ellos.

- [Calibración y dBFS](/phonometry/es/signals/metrology/calibration/): calibración SPL
  física a partir de un tono de calibrador (IEC 60942), la comprobación de
  estabilidad que aplica a esa grabación, y el modo digital dBFS.
- [Conformidad y verificación](/phonometry/es/signals/metrology/compliance-verification/):
  qué afirma una clase de prestaciones, los verificadores que califican cada
  etapa frente a sus tablas de tolerancias, cómo leer el informe de
  conformidad, y el alcance de IEC 61672-2/-3 e IEC 61260-2/-3.
- [Incertidumbre de medida (GUM y Monte Carlo)](/phonometry/es/signals/metrology/gum-uncertainty/):
  la ley de propagación de la incertidumbre y el método de Monte Carlo de
  ISO/IEC Guide 98-3, con incertidumbre expandida e intervalos de cobertura.
- [Cualificación de datos](/phonometry/es/signals/metrology/data-qualification/): los tests
  de estacionariedad por inversiones de orden y por rachas sobre estadísticas
  de segmento, y las estadísticas de Rice de cruces por nivel y de picos con el
  factor de irregularidad.

## Qué no cubre esta sección

Faltan cuatro cosas que un lector espera razonablemente encontrar aquí, y cada
guía lo dice en su propio bloque «No cubierto». **Aquí no se verifica ningún
instrumento.** `verify_filter_class` y `verify_weighting_class` comprueban una
respuesta digital diseñada frente a las tablas de tolerancias de IEC 61260-1 y
de IEC 61672-1; los ensayos de tipo de IEC 61672-2 que un sonómetro físico
necesita para su aprobación de tipo, los ensayos periódicos de IEC 61672-3
que recibe en servicio, y los ensayos de conformidad del propio calibrador de
IEC 60942, no se ejecutan, así que un veredicto de clase de aquí describe el
algoritmo y no un aparato construido;
[Conformidad y verificación](/phonometry/es/signals/metrology/compliance-verification/)
traza esa frontera parte por parte.
**Aquí no se abre ningún archivo.** Nada en la biblioteca decodifica WAV, FLAC
ni ningún otro contenedor: toda función toma un array que ya has leído, que es
la razón de que `fs` sea siempre un argumento. **Aquí no hay procesado de
arrays.** La correlación y el retardo modelan un único camino común entre
exactamente dos sensores e informan solo del mayor pico; no hay ninguna resolución
de TDOA multisensor, ni beamformer, ni localización de fuentes. **Aquí no hay
rasgos perceptuales.** El cepstrum de aquí es el de frecuencia lineal, sin
escala mel ni variante MFCC, y la sonoridad como sensación pertenece a
[Psicoacústica](/phonometry/es/perception/psychoacoustics/), no a las métricas
energéticas de esta sección.

## Antes y después de estas páginas

Las derivaciones que hay detrás de estas páginas están en la [teoría del
análisis de señal](/phonometry/es/reference/theory/signal-analysis/): la rejilla
de bandas, las curvas de ponderación, la integración temporal, la aproximación
de intensidad y el marco de incertidumbre. Si todavía no has ejecutado nada,
[Primeros pasos](/phonometry/es/start/getting-started/) instala la biblioteca y
calibra un primer análisis.

Si has llegado aquí desde una búsqueda y quieres la forma de la biblioteca
entera, [¿Qué necesitas medir?](/phonometry/es/start/tasks/) la indexa por el
trabajo y [Todas las guías](/phonometry/es/start/guides/) lista todas las
páginas con una línea sobre cada una.
