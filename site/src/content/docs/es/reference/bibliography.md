---
title: "Bibliografía"
description: "Los libros y artículos que sustentan las guías, agrupados por dominio: cada entrada con un DOI verificado o un enlace oficial del editor y una nota sobre qué sustenta."
---

Cada guía de este sitio se cierra con dos secciones de citas: una sección
**Referencias** que lista los libros y artículos que sustentan la física de la
página (estilo APA, un punto por fuente, cada uno con un DOI o un enlace
oficial del editor, y media frase sobre qué sustenta la entrada), seguida de
una sección **Normas** que nombra los documentos normativos que la página
implementa, apartado por apartado. Esta página reúne las entradas de
Referencias de todas las guías en un solo lugar, agrupadas por dominio: una
lista de lectura curada y la fuente única de verdad para la comprobación de
enlaces. Cada entrada lista las guías que la citan; la lista crece a medida
que las guías incorporan sus secciones de Referencias.

## Acústica general

- Kinsler, L. E., Frey, A. R., Coppens, A. B., & Sanders, J. V. (2000).
  *Fundamentals of acoustics* (4.ª ed.). Wiley. ISBN 978-0-471-84789-2.
  [Página del editor](https://www.wiley.com/en-us/Fundamentals+of+Acoustics%2C+4th+Edition-p-9780471847892).
  El primer curso clásico de acústica: ondas planas y esféricas, impedancia
  acústica y las definiciones de nivel sobre las que se apoyan todas las
  guías.
  Citado por [Niveles integrados y estadísticos](/phonometry/es/signal/levels/levels/).
- Rossing, T. D. (Ed.). (2014). *Springer handbook of acoustics* (2.ª ed.).
  Springer. ISBN 978-1-4939-0754-0.
  [doi:10.1007/978-1-4939-0755-7](https://doi.org/10.1007/978-1-4939-0755-7).
  Un panorama en un solo volumen de todos los dominios que toca esta
  biblioteca, desde la acústica de salas hasta la psicoacústica y el sonido
  submarino; la referencia transversal de consulta obligada.
- Beranek, L. L., & Mellow, T. J. (2012). *Acoustics: Sound fields and
  transducers*. Academic Press. ISBN 978-0-12-391421-7.
  [doi:10.1016/C2011-0-05897-0](https://doi.org/10.1016/C2011-0-05897-0).
  Campos sonoros, radiación y transductores electroacústicos; sustenta el
  material de electroacústica y potencia acústica.
  Citado por [Electroacústica](/phonometry/es/devices/electroacoustics/electroacoustics/),
  [Caracterización de altavoces](/phonometry/es/devices/electroacoustics/loudspeakers/) y
  [Potencia acústica](/phonometry/es/devices/emission/sound-power/).

## Procesado de señal

- Oppenheim, A. V., & Schafer, R. W. (2010). *Discrete-time signal processing*
  (3.ª ed.). Pearson. ISBN 978-0-13-198842-2.
  [Ficha en Open Library](https://openlibrary.org/isbn/9780131988422).
  La teoría de filtros digitales que sustenta las cascadas SOS, la transformada
  bilineal y el diezmado multitasa que usan los bancos de filtros.
  Citado por [Bancos de filtros](/phonometry/es/signal/filters/filter-banks/) y
  [Procesado por bloques](/phonometry/es/signal/filters/block-processing/).
- Smith, J. O. *Introduction to digital filters with audio applications*
  (libro en línea). Center for Computer Research in Music and Acoustics
  (CCRMA), Universidad de Stanford.
  [ccrma.stanford.edu/~jos/filters](https://ccrma.stanford.edu/~jos/filters/).
  Tratamiento gratuito y complementario del diseño y análisis de filtros
  digitales, el paso natural tras las guías de bancos de filtros.
  Citado por [Bancos de filtros](/phonometry/es/signal/filters/filter-banks/) y la
  [Galería de arquitecturas de filtro](/phonometry/es/signal/filters/filter-gallery/).
- Bendat, J. S., & Piersol, A. G. (2010). *Random data: Analysis and
  measurement procedures* (4.a ed.). Wiley. ISBN 978-0-470-24877-5.
  [doi:10.1002/9781118032428](https://doi.org/10.1002/9781118032428).
  La referencia de los estimadores espectrales de Welch y su calidad
  estadística, y de las funciones de coherencia de sistemas de entradas
  múltiples del capítulo 7 (coherencia múltiple y parcial, espectros
  condicionados) con las fórmulas de error de la sección 9.3 que implementa
  `miso_coherence`.
  Citado por [Análisis espectral calibrado](/phonometry/es/signal/spectra/spectral-analysis/)
  y [Coherencia múltiple y parcial](/phonometry/es/signal/spectra/miso-coherence/).
- Thomson, D. J. (1982). Spectrum estimation and harmonic analysis.
  *Proceedings of the IEEE*, 70(9), 1055-1096.
  [doi:10.1109/PROC.1982.12433](https://doi.org/10.1109/PROC.1982.12433).
  El método multitaper: ventanas de Slepian, autoespectros propios y los
  pesos adaptativos que implementa `multitaper_psd`.
  Citado por [Análisis espectral calibrado](/phonometry/es/signal/spectra/spectral-analysis/).
- Percival, D. B., & Walden, A. T. (1993). *Spectral analysis for physical
  applications: Multitaper and conventional univariate techniques*.
  Cambridge University Press. ISBN 978-0-521-43541-3.
  [doi:10.1017/CBO9780511622762](https://doi.org/10.1017/CBO9780511622762).
  El desarrollo multitaper (capítulo 7) tras `multitaper_psd` y las tablas
  de autovalores de las secuencias de Slepian que anclan su oráculo de test.
  Citado por [Análisis espectral calibrado](/phonometry/es/signal/spectra/spectral-analysis/).

## Instrumentación de medida

- International Electrotechnical Commission. (2014). *Electroacoustics —
  Octave-band and fractional-octave-band filters — Part 1: Specifications*
  (IEC 61260-1:2014).
  [Catálogo IEC](https://webstore.iec.ch/en/publication/5063).
  Los bordes de banda en base 10 y las máscaras de aceptación de clase de
  los bancos de octava fraccionaria.
  Citado por [Bancos de filtros](/phonometry/es/signal/filters/filter-banks/),
  la [Galería de arquitecturas de filtro](/phonometry/es/signal/filters/filter-gallery/),
  [Verificación de clase de filtros](/phonometry/es/signal/filters/filter-compliance/) y
  [Multicanal y rendimiento](/phonometry/es/signal/filters/multichannel/).
- International Electrotechnical Commission. (2013). *Electroacoustics —
  Sound level meters — Part 1: Specifications* (IEC 61672-1:2013).
  [Catálogo IEC](https://webstore.iec.ch/en/publication/5708).
  Las ponderaciones A/C/Z, las ponderaciones temporales exponenciales y las
  métricas de nivel del sonómetro, con las tablas de tolerancias usadas en
  la verificación.
  Citado por [Niveles integrados y estadísticos](/phonometry/es/signal/levels/levels/),
  [Ponderación frecuencial (A, C, Z)](/phonometry/es/signal/levels/weighting/),
  [Ponderación temporal](/phonometry/es/signal/levels/time-weighting/) y
  [Multicanal y rendimiento](/phonometry/es/signal/filters/multichannel/).
- International Electrotechnical Commission. (2013). *Electroacoustics —
  Sound level meters — Part 3: Periodic tests* (IEC 61672-3:2013).
  [Catálogo IEC](https://webstore.iec.ch/en/publication/5710).
  La verificación periódica de laboratorio de un sonómetro.
  Citado por [Calibración y dBFS](/phonometry/es/signal/metrology/calibration/).
- International Electrotechnical Commission. (2017). *Electroacoustics —
  Sound calibrators* (IEC 60942:2017).
  [Catálogo IEC](https://webstore.iec.ch/en/publication/30045).
  Las clases de calibrador, las tolerancias de nivel y el criterio de
  estabilidad a corto plazo aplicado a las grabaciones de calibración.
  Citado por [Calibración y dBFS](/phonometry/es/signal/metrology/calibration/).
- International Electrotechnical Commission. (2014). *Sound system equipment —
  Part 4: Microphones* (IEC 60268-4:2014).
  [Catálogo IEC](https://webstore.iec.ch/en/publication/32039).
  Las características nominales del micrófono: sensibilidad en campo libre y su
  nivel re 1 V/Pa, la respuesta en frecuencia y el rango de frecuencias
  efectivo frente a los límites de tolerancia, el diagrama direccional y el
  índice de directividad, el nivel de presión acústica de sobrecarga, el nivel de
  presión acústica equivalente debido al ruido inherente, y las impedancias
  nominales y la alimentación.
  Citado por [Caracterización de micrófonos](/phonometry/es/devices/electroacoustics/microphones/).
- International Electrotechnical Commission. (2007). *Sound system equipment —
  Part 5: Loudspeakers* (IEC 60268-5:2003+A1:2007).
  [Catálogo IEC](https://webstore.iec.ch/en/publication/1223).
  Las características nominales del altavoz: impedancia nominal, rango de
  frecuencias nominal, sensibilidad característica referida a 1 W a 1 m, rango
  de frecuencias efectivo frente a la banda de -10 dB, índice de directividad y
  distorsión armónica total frente a la frecuencia.
  Citado por [Caracterización de altavoces](/phonometry/es/devices/electroacoustics/loudspeakers/).
- International Electrotechnical Commission. (1982). *Scales and sizes for
  plotting frequency characteristics and polar diagrams* (IEC 60263:1982).
  [Catálogo IEC](https://webstore.iec.ch/en/publication/1218).
  Las proporciones de escala de las gráficas características: una década de
  frecuencia igual a 25 dB en la ordenada, y el diagrama polar sobre un radio
  de círculo de referencia de 25 dB.
  Citado por [Caracterización de altavoces](/phonometry/es/devices/electroacoustics/loudspeakers/) y
  [Caracterización de micrófonos](/phonometry/es/devices/electroacoustics/microphones/).

## Potencia acústica e intensidad

- Fahy, F. J. (1995). *Sound intensity* (2.ª ed.). E&FN Spon.
  ISBN 978-0-419-19810-9.
  [doi:10.4324/9780203475386](https://doi.org/10.4324/9780203475386).
  La monografía sobre el flujo de energía sonora: intensidad activa y
  reactiva, el estimador p-p y su presupuesto de error por desfase.
  Citado por [Potencia acústica por barrido de intensidad](/phonometry/es/devices/emission/sound-power-intensity/)
  e [Intensidad acústica (p-p)](/phonometry/es/devices/emission/intensity/).
- International Organization for Standardization. (2019). *Acoustics —
  Determination of sound power levels of noise sources — Guidelines for the
  use of basic standards* (ISO 3740:2019).
  [Catálogo iso.org](https://www.iso.org/standard/45107.html).
  La guía de selección de la familia de potencia acústica: grados, entornos y
  criterios de tamaño de fuente y de ruido de fondo.
  Citado por [Potencia acústica](/phonometry/es/devices/emission/sound-power/).
- International Organization for Standardization. (2010). *Acoustics —
  Determination of sound power levels and sound energy levels of noise
  sources using sound pressure — Precision methods for reverberation test
  rooms* (ISO 3741:2010).
  [Catálogo iso.org](https://www.iso.org/standard/52053.html).
  El método de precisión en cámara reverberante.
  Citado por [Potencia acústica en cámara reverberante](/phonometry/es/devices/emission/sound-power-reverberation/).
- International Organization for Standardization. (2010). *Acoustics —
  Determination of sound power levels and sound energy levels of noise
  sources using sound pressure — Engineering methods for an essentially free
  field over a reflecting plane* (ISO 3744:2010).
  [Catálogo iso.org](https://www.iso.org/standard/52055.html).
  El método de ingeniería por superficie envolvente.
  Citado por [Potencia acústica por métodos de presión](/phonometry/es/devices/emission/sound-power-pressure/).
- International Organization for Standardization. (2012). *Acoustics —
  Determination of sound power levels and sound energy levels of noise
  sources using sound pressure — Precision methods for anechoic rooms and
  hemi-anechoic rooms* (ISO 3745:2012).
  [Catálogo iso.org](https://www.iso.org/standard/45362.html).
  El método de precisión en cámara anecoica.
  Citado por [Potencia acústica por métodos de presión](/phonometry/es/devices/emission/sound-power-pressure/).
- International Organization for Standardization. (1996). *Acoustics —
  Declaration and verification of noise emission values of machinery and
  equipment* (ISO 4871:1996).
  [Catálogo iso.org](https://www.iso.org/standard/10868.html).
  La declaración de emisión sonora: las formas de doble/único número,
  $L_{WAd} = L_{WA} + K_{WA}$ y la verificación de la cláusula 6.2.
  Citado por [Potencia acústica](/phonometry/es/devices/emission/sound-power/).
- International Organization for Standardization. (1993). *Acoustics —
  Determination of sound power levels of noise sources using sound
  intensity — Part 1: Measurement at discrete points* (ISO 9614-1:1993).
  [Catálogo iso.org](https://www.iso.org/standard/17427.html).
  Los indicadores de campo y el criterio de capacidad dinámica de la medida
  de intensidad.
  Citado por [Intensidad acústica (p-p)](/phonometry/es/devices/emission/intensity/).
- International Electrotechnical Commission. (1993). *Electroacoustics —
  Instruments for the measurement of sound intensity — Measurements with
  pairs of pressure sensing microphones* (IEC 61043:1993; adoptada en Europa
  como EN 61043:1994).
  [Tienda IEC](https://webstore.iec.ch/en/publication/4353).
  La norma de instrumentación p-p: el estimador por espectro cruzado y el
  índice presión-intensidad residual.
  Citado por [Intensidad acústica (p-p)](/phonometry/es/devices/emission/intensity/).

## Acústica de salas

- Long, M. (2014). *Architectural acoustics* (2.ª ed.). Academic Press.
  [doi:10.1016/C2012-0-03257-5](https://doi.org/10.1016/C2012-0-03257-5).
  El complemento de diseño arquitectónico a las normas de medición: las
  frecuencias propias de la sala rectangular y el recuento de modos de Morse
  y Pierce (capítulo 8), el ruido autogenerado por los ocupantes de una sala
  (capítulo 17) y el criterio de ganancia antes de la realimentación de un
  sistema de refuerzo sonoro (capítulo 18).
  Citado por [Fuentes imagen y campo estacionario en la sala](/phonometry/es/buildings/rooms/room-image-sources/),
  [Acústica de oficinas diáfanas](/phonometry/es/buildings/rooms/open-plan-acoustics/) y
  [Caracterización de altavoces](/phonometry/es/devices/electroacoustics/loudspeakers/).
- Kuttruff, H. (2016). *Room acoustics* (6.ª ed.). CRC Press.
  [doi:10.1201/9781315372150](https://doi.org/10.1201/9781315372150).
  La monografía de referencia sobre el campo sonoro en salas: la teoría
  estadística del decaimiento, la frecuencia de Schroeder, la absorción y
  los parámetros perceptivos de sala.
  Citado por [Acústica de salas](/phonometry/es/buildings/rooms/room-acoustics/),
  [Predicción del tiempo de reverberación](/phonometry/es/buildings/rooms/reverberation-prediction/) y
  [Absorción acústica en recintos](/phonometry/es/buildings/rooms/enclosed-space-absorption/).
- Sabine, W. C. (1922). *Collected papers on acoustics*. Harvard University
  Press.
  [Escaneo libre en Internet Archive](https://archive.org/details/collectedpaperso00sabi).
  Los experimentos fundacionales de la reverberación y la ley de Sabine.
  Citado por [Predicción del tiempo de reverberación](/phonometry/es/buildings/rooms/reverberation-prediction/).
- Eyring, C. F. (1930). Reverberation time in "dead" rooms. *The Journal of
  the Acoustical Society of America*, 1(2A), 217-241.
  [doi:10.1121/1.1915175](https://doi.org/10.1121/1.1915175).
  La fórmula de reverberación por recorrido libre medio para salas muy
  absorbentes.
  Citado por [Predicción del tiempo de reverberación](/phonometry/es/buildings/rooms/reverberation-prediction/).
- Millington, G. (1932). A modified formula for reverberation. *The Journal
  of the Acoustical Society of America*, 4(1), 69-82.
  [doi:10.1121/1.1915588](https://doi.org/10.1121/1.1915588).
  La fórmula de reverberación logarítmica por superficie.
  Citado por [Predicción del tiempo de reverberación](/phonometry/es/buildings/rooms/reverberation-prediction/).
- Fitzroy, D. (1959). Reverberation formula which seems to be more accurate
  with nonuniform distribution of absorption. *The Journal of the
  Acoustical Society of America*, 31(7), 893-897.
  [doi:10.1121/1.1907814](https://doi.org/10.1121/1.1907814).
  La fórmula de reverberación axial para absorción anisótropa.
  Citado por [Predicción del tiempo de reverberación](/phonometry/es/buildings/rooms/reverberation-prediction/).
- Arau-Puchades, H. (1988). An improved reverberation formula. *Acustica*,
  65(4), 163-180.
  [Ficha del editor en Ingenta](https://www.ingentaconnect.com/content/dav/aaua/1988/00000065/00000004/art00003).
  El refinamiento por media geométrica de la fórmula de reverberación
  axial.
  Citado por [Predicción del tiempo de reverberación](/phonometry/es/buildings/rooms/reverberation-prediction/).
- Schroeder, M. R. (1965). New method of measuring reverberation time.
  *The Journal of the Acoustical Society of America*, 37(3), 409-412.
  [doi:10.1121/1.1909343](https://doi.org/10.1121/1.1909343).
  La integración hacia atrás de la respuesta al impulso al cuadrado en una
  curva de decaimiento.
  Citado por [Acústica de salas](/phonometry/es/buildings/rooms/room-acoustics/).
- Hak, C. C. J. M., Wenmaekers, R. H. C., & van Luxemburg, L. C. J. (2012).
  Measuring room impulse responses: Impact of the decay range on derived
  room acoustic parameters. *Acta Acustica united with Acustica*, 98(6),
  907-915. [doi:10.3813/aaa.918574](https://doi.org/10.3813/aaa.918574).
  El análisis de la relación impulso-ruido (INR) sobre los requisitos de
  rango de decaimiento.
  Citado por [Acústica de salas](/phonometry/es/buildings/rooms/room-acoustics/).
- Everest, F. A. (2001). *Master handbook of acoustics* (4.ª ed.).
  McGraw-Hill. ISBN 978-0-07-136097-5.
  [Ficha en Open Library](https://openlibrary.org/isbn/9780071360975).
  Un manual práctico de acústica de salas; su ejemplo resuelto de la
  Fig. 7-22 ancla la batería de conformidad de la predicción de
  reverberación.
  Citado por [Predicción del tiempo de reverberación](/phonometry/es/buildings/rooms/reverberation-prediction/).
- Carrión Isbert, A. (1998). *Diseño acústico de espacios arquitectónicos*.
  Edicions UPC. ISBN 978-84-8301-252-9.
  [Ficha en Open Library](https://openlibrary.org/books/OL23159935M).
  Un manual en español sobre el diseño acústico de salas.
  Citado por [Predicción del tiempo de reverberación](/phonometry/es/buildings/rooms/reverberation-prediction/).
- Beranek, L. L. (1957). Revised criteria for noise in buildings. *Noise
  Control*, 3(1), 19-27.
  [doi:10.1121/1.2369239](https://doi.org/10.1121/1.2369239).
  Las curvas NC originales y su razonamiento de interferencia con el habla.
  Citado por [Criterios de ruido de salas](/phonometry/es/buildings/rooms/room-noise/).
- Kosten, C. W., & van Os, G. J. (1962). Community reaction criteria for
  external noises. In *The Control of Noise* (National Physical Laboratory
  Symposium No. 12, pp. 373-387). Her Majesty's Stationery Office.
  [Ficha en Open Library](https://openlibrary.org/books/OL58781133M).
  La familia de curvas NR que se contrasta con NC.
  Citado por [Criterios de ruido de salas](/phonometry/es/buildings/rooms/room-noise/).
- Blazier, W. E. (1997). RC Mark II: A refined procedure for rating the
  noise of heating, ventilating, and air-conditioning (HVAC) systems in
  buildings. *Noise Control Engineering Journal*, 45(6), 243-250.
  [doi:10.3397/1.2828446](https://doi.org/10.3397/1.2828446).
  El procedimiento RC Mark II codificado después por el Anexo D de
  ANSI/ASA S12.2.
  Citado por [Criterios de ruido de salas](/phonometry/es/buildings/rooms/room-noise/).
- International Organization for Standardization. (2009). *Acoustics —
  Measurement of room acoustic parameters — Part 1: Performance spaces*
  (ISO 3382-1:2009).
  [Catálogo iso.org](https://www.iso.org/standard/40979.html).
  Las definiciones de parámetros de sala, los requisitos de posiciones y
  las diferencias apenas perceptibles.
  Citado por [Acústica de salas](/phonometry/es/buildings/rooms/room-acoustics/).
- International Organization for Standardization. (2008). *Acoustics —
  Measurement of room acoustic parameters — Part 2: Reverberation time in
  ordinary rooms* (ISO 3382-2:2008).
  [Catálogo iso.org](https://www.iso.org/standard/36201.html).
  Los grados de exactitud y los recuentos de posiciones de la medición de
  la reverberación.
  Citado por [Acústica de salas](/phonometry/es/buildings/rooms/room-acoustics/).
- International Organization for Standardization. (2012). *Acoustics —
  Measurement of room acoustic parameters — Part 3: Open plan offices*
  (ISO 3382-3:2012).
  [Catálogo iso.org](https://www.iso.org/standard/46520.html).
  Las magnitudes de privacidad del habla en oficinas diáfanas.
  Citado por [Acústica de oficinas diáfanas](/phonometry/es/buildings/rooms/open-plan-acoustics/).
- International Organization for Standardization. (2006). *Acoustics —
  Application of new measurement methods in building and room acoustics*
  (ISO 18233:2006).
  [Catálogo iso.org](https://www.iso.org/standard/40408.html).
  La adquisición de respuestas al impulso por barrido sinusoidal y MLS.
  Citado por [Medición de la respuesta al impulso](/phonometry/es/buildings/rooms/room-impulse-response/).
- International Organization for Standardization. (2003). *Acoustics —
  Measurement of sound absorption in a reverberation room* (ISO 354:2003).
  [Catálogo iso.org](https://www.iso.org/standard/34545.html).
  La medición de absorción en cámara reverberante que sustenta los datos de
  superficie.
  Citado por
  [Absorción acústica en recintos](/phonometry/es/buildings/rooms/enclosed-space-absorption/) y
  [Medida y clasificación de la absorción sonora](/phonometry/es/materials/absorbers/absorption-measurement/).
- European Committee for Standardization. (2003). *Building acoustics —
  Estimation of acoustic performance of buildings from the performance of
  elements — Part 6: Sound absorption in enclosed spaces*
  (EN 12354-6:2003).
  [Ficha en BSI Knowledge (BS EN 12354-6:2003)](https://knowledge.bsigroup.com/products/building-acoustics-estimation-of-acoustic-performance-of-buildings-from-the-performance-of-elements-sound-absorption-in-enclosed-spaces).
  El miembro de absorción de la familia de predicción EN 12354.
  Citado por [Absorción acústica en recintos](/phonometry/es/buildings/rooms/enclosed-space-absorption/).
- Acoustical Society of America. (2019). *Criteria for evaluating room
  noise* (ANSI/ASA S12.2-2019).
  [Tienda ANSI](https://webstore.ansi.org/standards/asa/ansiasas122019).
  El método de tangencia NC normativo y la calificación RC Mark II de su
  Anexo D informativo, con su etiqueta espectral.
  Citado por [Criterios de ruido de salas](/phonometry/es/buildings/rooms/room-noise/).

## Materiales y superficies

- Allard, J. F., & Atalla, N. (2009). *Propagation of sound in porous media:
  Modelling sound absorbing materials* (2.ª ed.). Wiley.
  ISBN 978-0-470-74661-5.
  [doi:10.1002/9780470747339](https://doi.org/10.1002/9780470747339).
  La teoría del material poroso que enlaza resistividad al flujo, impedancia
  superficial y absorción.
  Citado por
  [Resistencia al flujo de aire](/phonometry/es/materials/absorbers/airflow-resistance/),
  [Tubo de impedancia](/phonometry/es/materials/absorbers/impedance-tube/) y
  [Absorbentes porosos y multicapa](/phonometry/es/materials/absorbers/porous-absorbers/).
- Cox, T. J., & D'Antonio, P. (2017). *Acoustic absorbers and diffusers:
  Theory, design and application* (3.ª ed.). CRC Press.
  ISBN 978-1-4987-4099-9.
  [doi:10.1201/9781315369211](https://doi.org/10.1201/9781315369211).
  La monografía sobre medida y diseño de absorbentes y difusores, de los
  autores del método del coeficiente de difusión de ISO 17497-2.
  Citado por
  [Medida y clasificación de la absorción sonora](/phonometry/es/materials/absorbers/absorption-measurement/),
  [Difusores y sus coeficientes](/phonometry/es/materials/diffusers/diffusers/),
  [Metadifusores](/phonometry/es/materials/diffusers/metadiffusers/) y
  [Metaabsorbentes](/phonometry/es/materials/absorbers/metamaterial-absorbers/).
- Jiménez, N., Umnova, O. y Groby, J.-P. (Eds.). (2021). *Acoustic waves in
  periodic structures, metamaterials, and porous media* (Topics in Applied
  Physics, Vol. 143). Springer.
  [doi:10.1007/978-3-030-84300-7](https://doi.org/10.1007/978-3-030-84300-7).
  Un volumen editado que abarca las estructuras resonantes y periódicas que
  absorben y difunden el sonido, desde la teoría de la matriz de transferencia
  y el acoplamiento crítico de los absorbentes de metamaterial hasta los
  difusores de sublongitud de onda profunda; el complemento moderno sobre
  metamateriales de Cox & D'Antonio.
  Citado por [Metadifusores](/phonometry/es/materials/diffusers/metadiffusers/) y
  [Metaabsorbentes](/phonometry/es/materials/absorbers/metamaterial-absorbers/).
- Hargreaves, T. J., Cox, T. J., Lam, Y. W. y D'Antonio, P. (2000). Surface
  diffusion coefficients for room acoustics: Free-field measures of
  single-plane diffusion. *The Journal of the Acoustical Society of America*,
  108(4), 1710-1720.
  [doi:10.1121/1.1310192](https://doi.org/10.1121/1.1310192).
  El método del coeficiente de difusión en campo libre tras ISO 17497-2 y la
  geometría publicada del QRD N = 7 del ejemplo trabajado.
  Citado por [Difusores y sus coeficientes](/phonometry/es/materials/diffusers/diffusers/).
- Audio Engineering Society. (2001). *AES information document for room
  acoustics and sound reinforcement systems — Characterization and
  measurement of surface scattering uniformity* (AES-4id-2001). *Journal of
  the Audio Engineering Society*, 49(3), 149-165.
  [Normas AES en vigor](https://www.aes.org/publications/standards/list.cfm).
  El procedimiento del coeficiente de difusión de plano único en campo libre
  que ISO 17497-2 normalizó después.
  Citado por [Difusores y sus coeficientes](/phonometry/es/materials/diffusers/diffusers/).
- Jiménez, N., Cox, T. J., Romero-García, V. y Groby, J.-P. (2017).
  Metadiffusers: Deep-subwavelength sound diffusers. *Scientific Reports*,
  7, 5389.
  [doi:10.1038/s41598-017-05710-5](https://doi.org/10.1038/s41598-017-05710-5).
  El modelo de metadifusor: rendijas cargadas con resonadores que reproducen
  perfiles de fase de Schroeder y secuencias ternarias con paneles en
  sublongitud de onda profunda.
  Citado por [Metadifusores](/phonometry/es/materials/diffusers/metadiffusers/).
- Jiménez, N., Cox, T. J., Groby, J.-P. y Romero-García, V. (2019). Beyond
  phase grating diffusers using locally-resonant metamaterials. *Proceedings
  of the 23rd International Congress on Acoustics (ICA 2019)*, Aquisgrán.
  [PDF de las actas](https://pub.dega-akustik.de/ICA2019/data/articles/000706.pdf).
  El acompañante de congreso del artículo de los metadifusores: la cadena de
  matrices de transferencia y la imagen de dispersión de sonido lento.
  Citado por [Metadifusores](/phonometry/es/materials/diffusers/metadiffusers/).
- Jiménez, N., Groby, J.-P., Pagneux, V. y Romero-García, V. (2017).
  Iridescent perfect absorption in critically-coupled acoustic metamaterials
  using the transfer matrix method. *Applied Sciences*, 7(6), 618.
  [doi:10.3390/app7060618](https://doi.org/10.3390/app7060618).
  El modelo por matrices de transferencia de la ranura con resonadores de
  Helmholtz y la condición de acoplamiento crítico.
  Citado por [Metaabsorbentes](/phonometry/es/materials/absorbers/metamaterial-absorbers/)
  y [Metadifusores](/phonometry/es/materials/diffusers/metadiffusers/).
- Jiménez, N., Huang, W., Romero-García, V., Pagneux, V. y Groby, J.-P.
  (2016). Ultra-thin metamaterial for perfect and quasi-omnidirectional
  sound absorption. *Applied Physics Letters*, 109(12), 121902.
  [doi:10.1063/1.4962328](https://doi.org/10.1063/1.4962328).
  La impedancia del resonador y sus correcciones de radiación, y el
  absorbente perfecto de λ/88 publicado.
  Citado por [Metaabsorbentes](/phonometry/es/materials/absorbers/metamaterial-absorbers/)
  y [Metadifusores](/phonometry/es/materials/diffusers/metadiffusers/).
- Stinson, M. R. (1991). The propagation of plane sound waves in narrow and
  wide circular tubes, and generalization to uniform tubes of arbitrary
  cross-sectional shape. *The Journal of the Acoustical Society of America*,
  89(2), 550-558. [doi:10.1121/1.400379](https://doi.org/10.1121/1.400379).
  Los parámetros efectivos viscotérmicos de la ranura y de los cuellos y
  cavidades cuadrados.
  Citado por [Metaabsorbentes](/phonometry/es/materials/absorbers/metamaterial-absorbers/).
- International Organization for Standardization. (1998). *Acoustics —
  Determination of sound absorption coefficient and impedance in impedance
  tubes — Part 2: Transfer-function method* (ISO 10534-2:1998; adoptada en
  Europa como EN ISO 10534-2:2001; revisada después como
  [ISO 10534-2:2023](https://www.iso.org/standard/81294.html)).
  [Catálogo iso.org](https://www.iso.org/standard/22851.html).
  El método de la función de transferencia con dos micrófonos y sus límites
  de onda plana.
  Citado por [Tubo de impedancia](/phonometry/es/materials/absorbers/impedance-tube/).
- ASTM International. (2019). *Standard test method for normal incidence
  determination of porous material acoustical properties based on the
  transfer matrix method* (ASTM E2611-19, la edición implementada aquí;
  revisada después como [ASTM E2611-24](https://store.astm.org/e2611-24.html)).
  [Tienda ASTM](https://store.astm.org/e2611-19.html).
  El método de pérdida por transmisión por matriz de transferencia con
  cuatro micrófonos.
  Citado por [Tubo de impedancia](/phonometry/es/materials/absorbers/impedance-tube/).
- International Organization for Standardization. (2018). *Acoustics —
  Determination of airflow resistance — Part 1: Static airflow method*
  (ISO 9053-1:2018).
  [Catálogo iso.org](https://www.iso.org/standard/69869.html).
  El método estático de resistencia al flujo de aire y su velocidad de
  referencia.
  Citado por
  [Resistencia al flujo de aire](/phonometry/es/materials/absorbers/airflow-resistance/).
- International Organization for Standardization. (2020). *Acoustics —
  Determination of airflow resistance — Part 2: Alternating airflow method*
  (ISO 9053-2:2020).
  [Catálogo iso.org](https://www.iso.org/standard/76744.html).
  El método alterno de resistencia al flujo de aire con la relación efectiva
  de calores específicos del Anexo A.
  Citado por
  [Resistencia al flujo de aire](/phonometry/es/materials/absorbers/airflow-resistance/).
- International Organization for Standardization. (1996). *Acoustics —
  Determination of sound absorption coefficient and impedance in impedance
  tubes — Part 1: Method using standing wave ratio* (ISO 10534-1:1996;
  implementada como su adopción europea BS EN ISO 10534-1:2001).
  [Catálogo iso.org](https://www.iso.org/standard/18603.html).
  El método de la razón de onda estacionaria.
  Citado por [Tubo de impedancia](/phonometry/es/materials/absorbers/impedance-tube/).
- International Organization for Standardization. (1997). *Acoustics — Sound
  absorbers for use in buildings — Rating of sound absorption*
  (ISO 11654:1997).
  [Catálogo iso.org](https://www.iso.org/standard/19583.html).
  La valoración ponderada de absorción acústica, sus indicadores de forma y
  la clase de absorción.
  Citado por
  [Medida y clasificación de la absorción sonora](/phonometry/es/materials/absorbers/absorption-measurement/).
- International Organization for Standardization. (2020). *Acoustics —
  Determination and application of measurement uncertainties in building
  acoustics — Part 2: Sound absorption* (ISO 12999-2:2020).
  [Catálogo iso.org](https://www.iso.org/standard/68749.html).
  Las incertidumbres de reproducibilidad y repetibilidad de las magnitudes
  de cámara reverberante y de sus valoraciones de número único.
  Citado por
  [Medida y clasificación de la absorción sonora](/phonometry/es/materials/absorbers/absorption-measurement/).
- International Organization for Standardization. (2004). *Acoustics —
  Sound-scattering properties of surfaces — Part 1: Measurement of the
  random-incidence scattering coefficient in a reverberation room*
  (ISO 17497-1:2004+A1:2014, la edición implementada aquí).
  [Catálogo iso.org](https://www.iso.org/standard/31397.html).
  El método del coeficiente de dispersión con mesa giratoria.
  Citado por [Difusores y sus coeficientes](/phonometry/es/materials/diffusers/diffusers/).
- International Organization for Standardization. (2012). *Acoustics —
  Sound-scattering properties of surfaces — Part 2: Measurement of the
  directional diffusion coefficient in a free field* (ISO 17497-2:2012).
  [Catálogo iso.org](https://www.iso.org/standard/55293.html).
  El método del coeficiente de difusión con goniómetro.
  Citado por [Difusores y sus coeficientes](/phonometry/es/materials/diffusers/diffusers/).
- International Organization for Standardization. (2002). *Acoustics —
  Measurement of sound absorption properties of road surfaces in situ —
  Part 1: Extended surface method* (ISO 13472-1:2002, la edición
  implementada aquí; revisada después como ISO 13472-1:2022).
  [Catálogo iso.org](https://www.iso.org/standard/35387.html).
  La técnica de sustracción con la ventana de Adrienne y el radio de área
  muestreada.
  Citado por [Absorción in situ de firmes de carretera](/phonometry/es/materials/surfaces/road-absorption/).
- International Organization for Standardization. (2010). *Acoustics —
  Measurement of sound absorption properties of road surfaces in situ —
  Part 2: Spot method for reflective surfaces* (ISO 13472-2:2010, la
  edición implementada aquí; revisada después como ISO 13472-2:2025).
  [Catálogo iso.org](https://www.iso.org/standard/32304.html).
  El método de tubo puntual y sus límites de onda plana y de espaciado.
  Citado por [Absorción in situ de firmes de carretera](/phonometry/es/materials/surfaces/road-absorption/).

## Acústica de edificios

- Hopkins, C. (2007). *Sound insulation*. Butterworth-Heinemann.
  ISBN 978-0-7506-6526-1.
  [doi:10.4324/9780080550473](https://doi.org/10.4324/9780080550473).
  El tratamiento exhaustivo del aislamiento a ruido aéreo y de impactos: las
  cadenas de medición, la transmisión por flancos y el marco de predicción
  EN 12354.
  Citado por [Medición del aislamiento en campo (ISO 16283)](/phonometry/es/buildings/insulation/insulation-field/),
  [Medición del aislamiento en laboratorio](/phonometry/es/buildings/insulation/insulation-lab/) y
  [Predicción del aislamiento acústico (EN 12354)](/phonometry/es/buildings/design/insulation-prediction/).
- Vigran, T. E. (2008). *Building acoustics*. CRC Press.
  ISBN 978-0-415-42853-8.
  [doi:10.1201/9781482266016](https://doi.org/10.1201/9781482266016).
  Un manual compacto sobre la transmisión del sonido en edificios, de las
  construcciones simples y dobles a los suelos flotantes.
  Citado por [Medición del aislamiento en campo (ISO 16283)](/phonometry/es/buildings/insulation/insulation-field/),
  [Medición del aislamiento en laboratorio](/phonometry/es/buildings/insulation/insulation-lab/),
  [Rigidez dinámica de materiales resilientes](/phonometry/es/materials/resilient/dynamic-stiffness/) y
  [Predicción del aislamiento acústico de paneles](/phonometry/es/buildings/design/panel-sound-insulation/).
- International Organization for Standardization. (2020). *Acoustics —
  Rating of sound insulation in buildings and of building elements — Part 1:
  Airborne sound insulation* (ISO 717-1:2020).
  [Catálogo iso.org](https://www.iso.org/standard/77435.html).
  La calificación por curva de referencia y los términos de adaptación
  espectral C y Ctr.
  Citado por [Índices globales de aislamiento (ISO 717)](/phonometry/es/buildings/insulation/insulation-ratings/).
- International Organization for Standardization. (2014). *Acoustics — Field
  measurement of sound insulation in buildings and of building elements —
  Part 1: Airborne sound insulation* (ISO 16283-1:2014).
  [Catálogo iso.org](https://www.iso.org/standard/55997.html).
  El método de medición aérea en campo.
  Citado por [Medición del aislamiento en campo (ISO 16283)](/phonometry/es/buildings/insulation/insulation-field/).
- International Organization for Standardization. (1989). *Acoustics —
  Determination of dynamic stiffness — Part 1: Materials used under floating
  floors in dwellings* (ISO 9052-1:1989).
  [Catálogo iso.org](https://www.iso.org/standard/16620.html).
  El método de resonancia para la rigidez dinámica por unidad de área,
  idéntico a EN 29052-1.
  Citado por [Rigidez dinámica de materiales resilientes](/phonometry/es/materials/resilient/dynamic-stiffness/).

## Sonido estructural

- Cremer, L., Heckl, M., & Petersson, B. A. T. (2005). *Structure-borne
  sound: Structural vibrations and sound radiation at audio frequencies*
  (3.ª ed.). Springer. ISBN 978-3-540-22696-3.
  [doi:10.1007/b137728](https://doi.org/10.1007/b137728).
  La monografía de referencia sobre la vibración estructural y su radiación:
  movilidades, flujo de potencia, aislamiento de vibraciones, eficiencia de
  radiación y transmisión a través de uniones.
  Citado por [Movilidad mecánica y la familia de FRF](/phonometry/es/vibration/structural/mechanical-mobility/),
  [Rigidez dinámica de transferencia](/phonometry/es/vibration/structural/transfer-stiffness/),
  [Potencia acústica desde vibración](/phonometry/es/devices/emission/vibration-sound-power/),
  [Potencia acústica estructural de equipos](/phonometry/es/buildings/design/structure-borne-power/),
  [Ruido estructural instalado](/phonometry/es/buildings/design/installed-structure-borne/)
  y [Ondas elásticas y acoplamiento fluido-sólido](/phonometry/es/simulation/elastic-waves/).
- Cremer, L., Heckl, M., & Ungar, E. E. (1973). *Structure-borne sound:
  Structural vibrations and sound radiation at audio frequencies* (1.ª ed.).
  Springer. ISBN 978-3-540-06002-4.
  [doi:10.1007/978-3-662-10118-6](https://doi.org/10.1007/978-3-662-10118-6).
  La derivación original de los parámetros de onda χ y ψ y de los coeficientes
  de transmisión de onda de flexión para uniones de placas.
  Citado por [Transmisión de onda de flexión en uniones de placas](/phonometry/es/vibration/structural/junction-transmission/).
- Craik, R. J. M. (1996). *Sound transmission through buildings using
  statistical energy analysis*. Gower. ISBN 978-0-566-07572-5.
  El tratamiento SEA de la transmisión aérea y estructural en edificios, con los
  coeficientes de transmisión de onda de flexión tabulados para uniones en X, T,
  L y en línea.
  Citado por [Transmisión de onda de flexión en uniones de placas](/phonometry/es/vibration/structural/junction-transmission/).
- International Organization for Standardization. (2011). *Mechanical
  vibration and shock — Experimental determination of mechanical mobility —
  Part 1: Basic terms and definitions, and transducer specifications*
  (ISO 7626-1:2011).
  [Catálogo iso.org](https://www.iso.org/standard/50426.html).
  La familia de FRF y sus distinciones libre/bloqueada.
  Citado por [Movilidad mecánica y la familia de FRF](/phonometry/es/vibration/structural/mechanical-mobility/).
- International Organization for Standardization. (2015). *Mechanical
  vibration and shock — Experimental determination of mechanical mobility —
  Part 2: Measurements using single-point translation excitation with an
  attached vibration exciter* (ISO 7626-2:2015).
  [Catálogo iso.org](https://www.iso.org/standard/62483.html).
  El método de medición con excitador acoplado y sus criterios de aceptación.
  Citado por [Movilidad mecánica y la familia de FRF](/phonometry/es/vibration/structural/mechanical-mobility/).
- International Organization for Standardization. (2008). *Acoustics and
  vibration — Laboratory measurement of vibro-acoustic transfer properties of
  resilient elements — Part 1: Principles and guidelines* (ISO 10846-1:2008).
  [Catálogo iso.org](https://www.iso.org/standard/38936.html).
  La idealización de fuerza de bloqueo que sustenta la rigidez dinámica de
  transferencia.
  Citado por [Rigidez dinámica de transferencia](/phonometry/es/vibration/structural/transfer-stiffness/).
- International Organization for Standardization. (2009). *Acoustics —
  Determination of airborne sound power levels emitted by machinery using
  vibration measurement — Part 1: Survey method using a fixed radiation
  factor* (ISO/TS 7849-1:2009).
  [Catálogo iso.org](https://www.iso.org/standard/40537.html).
  La potencia acústica de límite superior desde la velocidad superficial con
  ε = 1.
  Citado por [Potencia acústica desde vibración](/phonometry/es/devices/emission/vibration-sound-power/).
- International Organization for Standardization. (2009). *Acoustics —
  Determination of airborne sound power levels emitted by machinery using
  vibration measurement — Part 2: Engineering method including determination
  of the adequate radiation factor* (ISO/TS 7849-2:2009).
  [Catálogo iso.org](https://www.iso.org/standard/40538.html).
  El método de ingeniería con un factor de radiación medido por bandas.
  Citado por [Potencia acústica desde vibración](/phonometry/es/devices/emission/vibration-sound-power/).
- International Organization for Standardization. (1996). *Acoustics —
  Characterization of sources of structure-borne sound with respect to sound
  radiation from connected structures — Measurement of velocity at the
  contact points of machinery when resiliently mounted* (ISO 9611:1996).
  [Catálogo iso.org](https://www.iso.org/standard/17424.html).
  La caracterización por velocidad libre de fuentes montadas
  resilientemente.
  Citado por [Potencia acústica estructural de equipos](/phonometry/es/buildings/design/structure-borne-power/).

## Sonido en exteriores y ruido ambiental

- Salomons, E. M. (2001). *Computational atmospheric acoustics*. Kluwer
  Academic Publishers. ISBN 978-1-4020-0390-5.
  [doi:10.1007/978-94-010-0660-6](https://doi.org/10.1007/978-94-010-0660-6).
  La teoría ondulatoria del sonido en exteriores (ecuación parabólica, fast
  field program, refracción, turbulencia) que sustenta las aproximaciones de
  ingeniería de la ISO 9613-2.
  Citado por [Propagación del sonido en exteriores](/phonometry/es/environment/propagation/outdoor-propagation/).
- Attenborough, K., & Van Renterghem, T. (2021). *Predicting outdoor sound*
  (2.ª ed.). CRC Press.
  [doi:10.1201/9780429470806](https://doi.org/10.1201/9780429470806).
  Los modelos de impedancia del suelo, el coeficiente de reflexión de onda
  esférica que explica el mínimo del suelo y los efectos meteorológicos sobre las
  barreras.
  Citado por [Propagación del sonido en exteriores](/phonometry/es/environment/propagation/outdoor-propagation/).
- Maekawa, Z. (1968). Noise reduction by screens. *Applied Acoustics*, 1(3),
  157-173.
  [doi:10.1016/0003-682X(68)90020-0](https://doi.org/10.1016/0003-682X(68)90020-0).
  El ábaco de atenuación de pantallas frente al número de Fresnel del que
  descienden las fórmulas de ingeniería de barreras.
  Citado por [Propagación del sonido en exteriores](/phonometry/es/environment/propagation/outdoor-propagation/).
- Kephalopoulos, S., Paviotti, M., & Anfosso-Lédée, F. (2012). *Common noise
  assessment methods in Europe (CNOSSOS-EU)* (EUR 25379 EN). Oficina de
  Publicaciones de la Unión Europea.
  [doi:10.2788/31776](https://doi.org/10.2788/31776),
  [repositorio del JRC](https://publications.jrc.ec.europa.eu/repository/handle/JRC72550).
  El marco común de la UE para los mapas de ruido, contrastado con la
  ISO 9613-2; sus clases de suelo por resistividad de flujo las reutiliza el
  efecto de suelo de rotorcraft.
  Citado por [Propagación del sonido en exteriores](/phonometry/es/environment/propagation/outdoor-propagation/)
  y [Ruido de rotorcraft](/phonometry/es/aircraft/rotorcraft-noise/).
- International Organization for Standardization. (1993). *Acoustics —
  Attenuation of sound during propagation outdoors — Part 1: Calculation of
  the absorption of sound by the atmosphere* (ISO 9613-1:1993).
  [Catálogo iso.org](https://www.iso.org/standard/17426.html).
  El coeficiente de atenuación atmosférica de tono puro.
  Citado por [Propagación del sonido en exteriores](/phonometry/es/environment/propagation/outdoor-propagation/).
- International Organization for Standardization. (1996). *Acoustics —
  Attenuation of sound during propagation outdoors — Part 2: General method
  of calculation* (ISO 9613-2:1996; revisada en 2024, el método de 1996 es el
  implementado).
  [Catálogo iso.org](https://www.iso.org/standard/20649.html).
  La cadena de atenuación en exteriores implementada.
  Citado por [Propagación del sonido en exteriores](/phonometry/es/environment/propagation/outdoor-propagation/).
- International Organization for Standardization. (2016). *Acoustics —
  Description, measurement and assessment of environmental noise — Part 1:
  Basic quantities and assessment procedures* (ISO 1996-1:2016).
  [Catálogo iso.org](https://www.iso.org/standard/59765.html).
  El marco de evaluación ambiental y sus ajustes por categoría de la
  Tabla A.1.
  Citado por [Prominencia de sonidos impulsivos](/phonometry/es/environment/assessment/impulsive-sound/).
- International Organization for Standardization. (2017). *Acoustics —
  Description, measurement and assessment of environmental noise — Part 2:
  Determination of sound pressure levels* (ISO 1996-2:2017).
  [Catálogo iso.org](https://www.iso.org/standard/59766.html).
  La norma de medición ambiental: su anexo J adopta el método de ingeniería
  para la audibilidad tonal, y el criterio de audibilidad que reutiliza la
  IEC 61400-11 procede del anexo C de su edición de 2007.
  Citado por [Audibilidad objetiva de tonos](/phonometry/es/perception/psychoacoustics/tone-audibility/) y
  [Ruido de aerogeneradores](/phonometry/es/environment/sources/wind-turbine-noise/).
- Nordtest. (2002). *Acoustics: Prominence of impulsive sounds and for
  adjustment of LAeq* (método Nordtest NT ACOU 112).
  [nordtest.info](https://www.nordtest.info/wp/2002/05/01/acoustics-prominence-of-impulsive-sounds-and-for-adjustment-of-laeq-nt-acou-112/).
  El método de prominencia por tasa de crecimiento, descargable gratis.
  Citado por [Prominencia de sonidos impulsivos](/phonometry/es/environment/assessment/impulsive-sound/).
- International Organization for Standardization. (2022). *Acoustics —
  Description, measurement and assessment of environmental noise — Part 3:
  Objective method for the measurement of prominence of impulsive sounds and
  for adjustment of LAeq* (ISO/PAS 1996-3:2022).
  [Catálogo iso.org](https://www.iso.org/standard/77035.html).
  El sucesor ISO construido sobre la prominencia de la NT ACOU 112.
  Citado por [Prominencia de sonidos impulsivos](/phonometry/es/environment/assessment/impulsive-sound/).
- International Electrotechnical Commission. (2018). *Wind turbines —
  Part 11: Acoustic noise measurement techniques*
  (IEC 61400-11:2012+AMD1:2018 CSV).
  [Tienda IEC](https://webstore.iec.ch/en/publication/63367).
  La geometría de potencia acústica aparente, la clasificación por velocidades
  de viento y la audibilidad tonal de aerogeneradores.
  Citado por [Ruido de aerogeneradores](/phonometry/es/environment/sources/wind-turbine-noise/).
- International Electrotechnical Commission. (2005). *Wind turbines —
  Part 14: Declaration of apparent sound power level and tonality values*
  (IEC TS 61400-14:2005).
  [Tienda IEC](https://webstore.iec.ch/en/publication/5432).
  Los valores declarados y su incertidumbre para un lote de aerogeneradores.
  Citado por [Ruido de aerogeneradores](/phonometry/es/environment/sources/wind-turbine-noise/).

## Ruido de aeronaves

- International Civil Aviation Organization. (2017). *Annex 16 to the
  Convention on International Civil Aviation: Environmental protection —
  Volume I: Aircraft noise* (8.ª ed.).
  [Tienda ICAO](https://store.icao.int/en/annex-16-environmental-protection-volume-i-aircraft-noise).
  La norma de certificación acústica de aeronaves cuyo Apéndice 2 define el
  procedimiento EPNL.
  Citado por [Ruido de aeronaves](/phonometry/es/aircraft/aircraft-noise/).
- International Civil Aviation Organization. (2018). *Environmental technical
  manual — Volume I: Procedures for the noise certification of aircraft*
  (Doc 9501, 3.ª ed.).
  [Tienda ICAO](https://store.icao.int/en/environmental-technical-manual-volume-1-procedures-for-the-noise-certification-of-aircraft-doc-9501-1).
  La guía de certificación cuyos ejemplos trabajados (corrección tonal, EPNL
  por el método integrado) sirven de oráculos numéricos.
  Citado por [Ruido de aeronaves](/phonometry/es/aircraft/aircraft-noise/).
- International Electrotechnical Commission. (1995). *Electroacoustics —
  Instruments for measurement of aircraft noise — Performance requirements for
  systems to measure one-third-octave-band sound pressure levels in noise
  certification of transport-category aeroplanes* (IEC 61265:1995; revisada
  después como [IEC 61265:2018](https://webstore.iec.ch/en/publication/32635),
  la edición de 1995 es la implementada).
  [Catálogo IEC](https://webstore.iec.ch/en/publication/5076).
  Las tolerancias de comportamiento del sistema de medida de ruido de
  aeronaves.
  Citado por [Ruido de aeronaves](/phonometry/es/aircraft/aircraft-noise/).
- SAE International. (2013). *Application of pure-tone atmospheric absorption
  losses to one-third octave-band data* (SAE ARP 5534, reafirmada en 2021).
  [sae.org](https://www.sae.org/standards/content/arp5534/).
  La absorción atmosférica en bandas de tercio de octava por el método SAE
  para espectros de sobrevuelo.
  Citado por [Ruido de aeronaves](/phonometry/es/aircraft/aircraft-noise/).
- SAE International. (2012). *Standard values of atmospheric absorption as a
  function of temperature and humidity* (SAE ARP 866B, estabilizada en 2012).
  [sae.org](https://www.sae.org/standards/content/arp866b/).
  La práctica SAE de absorción atmosférica predecesora, origen del antiguo
  método aproximado limitado a 50 dB.
  Citado por [Ruido de aeronaves](/phonometry/es/aircraft/aircraft-noise/).
- SAE International. (2006). *Method for predicting lateral attenuation of
  airplane noise* (SAE AIR 5662).
  [sae.org](https://www.sae.org/standards/content/air5662/).
  El modelo de atenuación lateral sobre suelo blando que adopta la ECAC
  Doc 29.
  Citado por [Ruido de aeropuertos](/phonometry/es/aircraft/airport-noise/).
- European Civil Aviation Conference. (2016). *Report on standard method of
  computing noise contours around civil airports* (ECAC.CEAC Doc 29, 4.ª ed.),
  Volumen 2: Guía técnica.
  [Página de documentos de ECAC](https://www.ecac-ceac.org/documents/ecac-documents-and-international-agreements),
  [PDF gratuito](https://www.ecac-ceac.org/images/documents/ECAC-Doc_29_4th_edition_Dec_2016_Volume_2.pdf).
  El método europeo de contornos de ruido de aeropuerto: interpolación NPD y
  cálculo de evento único por segmentos.
  Citado por [Ruido de aeropuertos](/phonometry/es/aircraft/airport-noise/).
- European Civil Aviation Conference. (2026). *Report on standard method of
  computing noise contours around civil airports* (ECAC.CEAC Doc 29, 5.ª ed.),
  Volumen 3: Casos de referencia y marco de verificación.
  [Página de documentos de ECAC](https://www.ecac-ceac.org/documents/ecac-documents-and-international-agreements),
  [PDF gratuito](https://www.ecac-ceac.org/images/documents/ECAC-CEAC-DOC_29_5th_Edition-REPORT_ON_STANDARD_METHOD_OF_COMPUTING_NOISE_CONTOURS_AROUND_CIVIL_AIRPORTS-Volume_3-REFERENCE_CASES_AND_VERIFICATION_FRAMEWORK.pdf).
  Los casos de referencia y el workbook con los que se valida la cadena de
  evento único.
  Citado por [Ruido de aeropuertos](/phonometry/es/aircraft/airport-noise/).
- European Civil Aviation Conference. (2026). *Report on standard method of
  computing rotorcraft noise contours* (ECAC.CEAC Doc 32, 1.ª ed.).
  [Página de documentos de ECAC](https://www.ecac-ceac.org/documents/ecac-documents-and-international-agreements),
  [PDF gratuito](https://www.ecac-ceac.org/images/documents/ECAC-CEAC-DOC_32-REPORT_ON_STANDARD_METHOD_OF_COMPUTING_ROTORCRAFT_NOISE_CONTOURS.pdf).
  El método estándar de contornos de rotorcraft construido sobre el
  hemisferio de ruido.
  Citado por [Ruido de rotorcraft](/phonometry/es/aircraft/rotorcraft-noise/).
- Olsen, H., Tuinstra, M., & van Oosten, N. (2024). *Rotorcraft noise
  modelling guidance* (Research Project NOISE SC01, entregable D1.5d,
  contrato EASA.2020.FC.06). European Union Aviation Safety Agency.
  [Página del proyecto en EASA](https://www.easa.europa.eu/en/research-projects/environmental-research-rotorcraft-noise),
  [PDF gratuito](https://www.easa.europa.eu/en/downloads/132005/en).
  La guía de modelado NORAH2 a nivel de ecuación, cuyas tablas y hemisferios
  de referencia sirven de oráculos.
  Citado por [Ruido de rotorcraft](/phonometry/es/aircraft/rotorcraft-noise/).
- Chien, C. F., & Soroka, W. W. (1975). Sound propagation along an impedance
  plane. *Journal of Sound and Vibration*, 43(1), 9-20.
  [doi:10.1016/0022-460X(75)90200-X](https://doi.org/10.1016/0022-460X(75)90200-X).
  La solución de interferencia de dos rayos sobre un plano de impedancia que sustenta
  el efecto de suelo de rotorcraft.
  Citado por [Ruido de rotorcraft](/phonometry/es/aircraft/rotorcraft-noise/).
- Delany, M. E., & Bazley, E. N. (1970). Acoustical properties of fibrous
  absorbent materials. *Applied Acoustics*, 3(2), 105-116.
  [doi:10.1016/0003-682X(70)90031-9](https://doi.org/10.1016/0003-682X(70)90031-9).
  El modelo de impedancia del suelo de un parámetro (resistividad de flujo).
  Citado por [Ruido de rotorcraft](/phonometry/es/aircraft/rotorcraft-noise/).

## Sonido submarino

- Urick, R. J. (1983). *Principles of underwater sound* (3.ª ed.).
  McGraw-Hill; reimpreso en 1996 por Peninsula Publishing.
  ISBN 978-0-932146-62-5.
  [Ficha en Open Library](https://openlibrary.org/books/OL9317725M).
  La monografía clásica del sonido submarino: convenciones de nivel, ruido
  radiado por buques y el marco de la ecuación del sonar.
  Citado por [Acústica submarina](/phonometry/es/underwater/underwater-acoustics/) y
  [Propagación submarina del sonido](/phonometry/es/underwater/underwater-propagation/).
- Ainslie, M. A. (2010). *Principles of sonar performance modelling*.
  Springer.
  [doi:10.1007/978-3-540-87662-5](https://doi.org/10.1007/978-3-540-87662-5).
  El tratamiento sistemático de las magnitudes acústicas submarinas en la
  línea que la ISO 18405 normalizó, los regímenes de propagación de flujo de
  energía de Weston en aguas someras, las ecuaciones del sonar con siete
  ejemplos trabajados totalmente numéricos y el audiograma de orca.
  Citado por [Acústica submarina](/phonometry/es/underwater/underwater-acoustics/),
  [Propagación submarina del sonido](/phonometry/es/underwater/underwater-propagation/)
  y [Exposición a ruido de mamíferos marinos](/phonometry/es/underwater/marine-mammal-exposure/).
- Medwin, H., & Clay, C. S. (1998). *Fundamentals of acoustical oceanography*.
  Academic Press. ISBN 978-0-12-487570-8.
  [Página del editor](https://shop.elsevier.com/books/fundamentals-of-acoustical-oceanography/medwin/978-0-12-487570-8).
  La acústica oceánica desde primeros principios; el coeficiente de reflexión
  de Rayleigh fluido-fluido del modelo de fondo marino.
  Citado por [Propagación submarina del sonido](/phonometry/es/underwater/underwater-propagation/).
- Jensen, F. B., Kuperman, W. A., Porter, M. B., & Schmidt, H. (2011).
  *Computational ocean acoustics* (2.ª ed.). Springer.
  [doi:10.1007/978-1-4419-8678-8](https://doi.org/10.1007/978-1-4419-8678-8).
  La monografía de referencia de la propagación numérica: modos normales,
  trazado de rayos y ecuación parabólica.
  Citado por [Solvers numéricos de propagación submarina](/phonometry/es/underwater/underwater-solvers/).
- Munk, W. H. (1974). Sound channel in an exponentially stratified ocean,
  with application to SOFAR. *The Journal of the Acoustical Society of
  America*, 55(2), 220-226.
  [doi:10.1121/1.1914492](https://doi.org/10.1121/1.1914492).
  El perfil canónico de velocidad del sonido en aguas profundas que usan los
  ejemplos de los solvers.
  Citado por [Solvers numéricos de propagación submarina](/phonometry/es/underwater/underwater-solvers/).
- Francois, R. E., & Garrison, G. R. (1982). Sound absorption based on ocean
  measurements: Part I: Pure water and magnesium sulfate contributions.
  *The Journal of the Acoustical Society of America*, 72(3), 896-907.
  [doi:10.1121/1.388170](https://doi.org/10.1121/1.388170).
  Las componentes de agua pura y sulfato de magnesio del modelo de referencia de
  absorción en agua de mar.
  Citado por [Propagación submarina del sonido](/phonometry/es/underwater/underwater-propagation/).
- Francois, R. E., & Garrison, G. R. (1982). Sound absorption based on ocean
  measurements. Part II: Boric acid contribution and equation for total
  absorption. *The Journal of the Acoustical Society of America*, 72(6),
  1879-1890.
  [doi:10.1121/1.388673](https://doi.org/10.1121/1.388673).
  El término de ácido bórico y la ecuación completa de absorción total.
  Citado por [Propagación submarina del sonido](/phonometry/es/underwater/underwater-propagation/).
- Ainslie, M. A., & McColm, J. G. (1998). A simplified formula for viscous and
  chemical absorption in sea water. *The Journal of the Acoustical Society of
  America*, 103(3), 1671-1672.
  [doi:10.1121/1.421258](https://doi.org/10.1121/1.421258).
  La fórmula simplificada y legible de absorción en agua de mar.
  Citado por [Propagación submarina del sonido](/phonometry/es/underwater/underwater-propagation/).
- Thorp, W. H. (1967). Analytic description of the low-frequency attenuation
  coefficient. *The Journal of the Acoustical Society of America*, 42(1), 270.
  [doi:10.1121/1.1910566](https://doi.org/10.1121/1.1910566).
  La fórmula de absorción de baja frecuencia dependiente solo de la frecuencia.
  Citado por [Propagación submarina del sonido](/phonometry/es/underwater/underwater-propagation/).
- Chen, C.-T., & Millero, F. J. (1977). Speed of sound in seawater at high
  pressures. *The Journal of the Acoustical Society of America*, 62(5),
  1129-1135.
  [doi:10.1121/1.381646](https://doi.org/10.1121/1.381646).
  La ecuación de velocidad del sonido UNESCO, el estándar internacional.
  Citado por [Propagación submarina del sonido](/phonometry/es/underwater/underwater-propagation/).
- Wong, G. S. K., & Zhu, S. (1995). Speed of sound in seawater as a function
  of salinity, temperature, and pressure. *The Journal of the Acoustical
  Society of America*, 97(3), 1732-1736.
  [doi:10.1121/1.413048](https://doi.org/10.1121/1.413048).
  La reformulación ITS-90 de los coeficientes UNESCO, la forma implementada.
  Citado por [Propagación submarina del sonido](/phonometry/es/underwater/underwater-propagation/).
- Del Grosso, V. A. (1974). New equation for the speed of sound in natural
  waters (with comparisons to other equations). *The Journal of the
  Acoustical Society of America*, 56(4), 1084-1091.
  [doi:10.1121/1.1903388](https://doi.org/10.1121/1.1903388).
  La ecuación alternativa de velocidad del sonido basada en presión.
  Citado por [Propagación submarina del sonido](/phonometry/es/underwater/underwater-propagation/).
- Mackenzie, K. V. (1981). Nine-term equation for sound speed in the oceans.
  *The Journal of the Acoustical Society of America*, 70(3), 807-812.
  [doi:10.1121/1.386920](https://doi.org/10.1121/1.386920).
  La ecuación de nueve términos basada en profundidad.
  Citado por [Propagación submarina del sonido](/phonometry/es/underwater/underwater-propagation/).
- Leroy, C. C., & Parthiot, F. (1998). Depth-pressure relationships in the
  oceans and seas. *The Journal of the Acoustical Society of America*, 103(3),
  1346-1352.
  [doi:10.1121/1.421275](https://doi.org/10.1121/1.421275).
  La conversión de profundidad a presión que usan las ecuaciones de
  velocidad del sonido.
  Citado por [Propagación submarina del sonido](/phonometry/es/underwater/underwater-propagation/).
- Wenz, G. M. (1962). Acoustic ambient noise in the ocean: Spectra and
  sources. *The Journal of the Acoustical Society of America*, 34(12),
  1936-1956.
  [doi:10.1121/1.1909155](https://doi.org/10.1121/1.1909155).
  El estudio clásico del ruido ambiental que sustenta las componentes espectrales de
  viento y térmica.
  Citado por [Propagación submarina del sonido](/phonometry/es/underwater/underwater-propagation/).
- Carey, W. M., & Evans, R. B. (2011). *Ocean ambient noise: Measurement and
  theory*. Springer.
  [doi:10.1007/978-1-4419-7832-5](https://doi.org/10.1007/978-1-4419-7832-5).
  El tratamiento moderno del ruido ambiental oceánico: la "regla de los
  cincos" del viento y la derivación del ruido térmico de Mellen.
  Citado por [Propagación submarina del sonido](/phonometry/es/underwater/underwater-propagation/).
- MacGillivray, A., & de Jong, C. (2021). A reference spectrum model for
  estimating source levels of marine shipping based on automated
  identification system data. *Journal of Marine Science and Engineering*,
  9(4), 369.
  [doi:10.3390/jmse9040369](https://doi.org/10.3390/jmse9040369).
  El modelo JOMOPANS-ECHO de nivel de fuente de buques (acceso abierto) y su
  calculadora de referencia.
  Citado por [Propagación submarina del sonido](/phonometry/es/underwater/underwater-propagation/).
- Wales, S. C., & Heitmeyer, R. M. (2002). An ensemble source spectra model
  for merchant ship-radiated noise. *The Journal of the Acoustical Society of
  America*, 111(3), 1211-1231.
  [doi:10.1121/1.1427355](https://doi.org/10.1121/1.1427355).
  El modelo de espectro de conjunto de fuentes de buques mercantes.
  Citado por [Propagación submarina del sonido](/phonometry/es/underwater/underwater-propagation/).
- National Marine Fisheries Service (2018). *2018 Revision to: Technical
  Guidance for Assessing the Effects of Anthropogenic Sound on Marine Mammal
  Hearing (Version 2.0)*. NOAA Technical Memorandum NMFS-OPR-59.
  [NOAA Fisheries](https://www.fisheries.noaa.gov/s3/2023-05/TECHMEMOGuidance508.pdf).
  Los parámetros de ponderación auditiva y los umbrales de inicio de PTS de la
  guía de 2018, con el ejemplo trabajado del Apéndice D.
  Citado por [Exposición a ruido de mamíferos marinos](/phonometry/es/underwater/marine-mammal-exposure/).
- National Marine Fisheries Service (2024). *2024 Update to: Technical
  Guidance for Assessing the Effects of Anthropogenic Sound on Marine Mammal
  Hearing (Version 3.0)*. NOAA Technical Memorandum NMFS-OPR-71.
  [NOAA Fisheries](https://www.fisheries.noaa.gov/s3/2024-11/Tech_Memo-Guidance_-3.0-_OCT-2024-508_OPR1.pdf).
  La guía estadounidense vigente: parámetros de ponderación revisados y los
  criterios de inicio de lesión auditiva que sustituyen a los umbrales de PTS
  de 2018.
  Citado por [Exposición a ruido de mamíferos marinos](/phonometry/es/underwater/marine-mammal-exposure/).
- Southall, B. L., Finneran, J. J., Reichmuth, C., Nachtigall, P. E.,
  Ketten, D. R., Bowles, A. E., Ellison, W. T., Nowacek, D. P., &
  Tyack, P. L. (2019). Marine mammal noise exposure criteria: Updated
  scientific recommendations for residual hearing effects. *Aquatic Mammals*,
  45(2), 125-232.
  [doi:10.1578/AM.45.2.2019.125](https://doi.org/10.1578/AM.45.2.2019.125).
  Los grupos auditivos, los audiogramas de grupo y los criterios de inicio de
  TTS y PTS revisados por pares, con la fe de erratas de 45(5), 569-572.
  Citado por [Exposición a ruido de mamíferos marinos](/phonometry/es/underwater/marine-mammal-exposure/).
- Finneran, J. J. (2016). *Auditory weighting functions and TTS/PTS exposure
  functions for marine mammals exposed to underwater noise*. Technical Report
  3026, SSC Pacific.
  [Página del informe](https://apps.dtic.mil/sti/citations/AD1026445).
  La forma del filtro paso banda de ponderación y la ecuación del audiograma
  que adoptan tanto los criterios NMFS como los de Southall.
  Citado por [Exposición a ruido de mamíferos marinos](/phonometry/es/underwater/marine-mammal-exposure/).

## Simulación de ondas

- Williams, E. G. (1999). *Fourier acoustics: Sound radiation and nearfield
  acoustical holography*. Academic Press.
  [doi:10.1016/B978-0-12-753960-7.X5000-1](https://doi.org/10.1016/B978-0-12-753960-7.X5000-1).
  La ecuación integral de Helmholtz tras la transformación de campo cercano a
  campo lejano, con la función de Green de espacio libre saliente y el límite
  de campo lejano.
  Citado por [Simulación de ondas FDTD 2D](/phonometry/es/simulation/fdtd-simulation/).
- Virieux, J. (1986). P-SV wave propagation in heterogeneous media:
  velocity-stress finite-difference method. *Geophysics*, 51(4), 889-901.
  [doi:10.1190/1.1442147](https://doi.org/10.1190/1.1442147).
  El esquema elástico velocidad-esfuerzo sobre la celda totalmente
  escalonada, su cota de Courant y sus relaciones de dispersión, y el líquido
  como límite sin cizalla.
  Citado por [Ondas elásticas y acoplamiento fluido-sólido](/phonometry/es/simulation/elastic-waves/).
- Moczo, P., Kristek, J., Galis, M., Pazak, P., & Balazovjech, M. (2007).
  The finite-difference and finite-element modeling of seismic wave
  propagation and earthquake motion. *Acta Physica Slovaca*, 57(2), 177-406.
  Los parámetros efectivos de malla en medio heterogéneo (módulo de cizalla
  armónico, densidad aritmética) y la superficie libre por imagen de
  esfuerzos.
  Citado por [Ondas elásticas y acoplamiento fluido-sólido](/phonometry/es/simulation/elastic-waves/).
- Brekhovskikh, L. M., & Godin, O. A. (1990). *Acoustics of layered media I:
  Plane and quasi-plane waves*. Springer.
  [doi:10.1007/978-3-642-52369-4](https://doi.org/10.1007/978-3-642-52369-4).
  Los oráculos fluido-sólido: el coeficiente de reflexión oblicua con
  conversión de modo, la ecuación característica exacta de Scholte y la
  transmisión de la capa de tres medios.
  Citado por [Ondas elásticas y acoplamiento fluido-sólido](/phonometry/es/simulation/elastic-waves/).
- van Vossen, R., Robertsson, J. O. A., & Chapman, C. H. (2002).
  Finite-difference modeling of wave propagation in a fluid-solid
  configuration. *Geophysics*, 67(2), 618-624.
  [doi:10.1190/1.1468623](https://doi.org/10.1190/1.1468623).
  El benchmark fluido-sólido del esquema escalonado: las medias de los
  parámetros efectivos, la configuración de fondo blando de Scholte y la
  regla de puntos por longitud de onda para las ondas de interfase.
  Citado por [Ondas elásticas y acoplamiento fluido-sólido](/phonometry/es/simulation/elastic-waves/).

## Habla

- Houtgast, T., & Steeneken, H. J. M. (1985). A review of the MTF concept in
  room acoustics and its use for estimating speech intelligibility in
  auditoria. *The Journal of the Acoustical Society of America*, 77(3),
  1069-1077. [doi:10.1121/1.392224](https://doi.org/10.1121/1.392224).
  El marco de transferencia de modulación sobre el que se construye el índice
  de transmisión del habla.
  Citado por [Índice de transmisión del habla](/phonometry/es/perception/speech/speech-transmission/).
- French, N. R., & Steinberg, J. C. (1947). Factors governing the
  intelligibility of speech sounds. *The Journal of the Acoustical Society of
  America*, 19(1), 90-119.
  [doi:10.1121/1.1916407](https://doi.org/10.1121/1.1916407).
  Los experimentos por bandas de articulación que sustentan la función de importancia
  de banda del índice de inteligibilidad del habla.
  Citado por [Índice de inteligibilidad del habla](/phonometry/es/perception/speech/speech-intelligibility/).
- Taal, C. H., Hendriks, R. C., Heusdens, R., & Jensen, J. (2011). An
  algorithm for intelligibility prediction of time-frequency weighted noisy
  speech. *IEEE Transactions on Audio, Speech, and Language Processing*,
  19(7), 2125-2136.
  [doi:10.1109/TASL.2011.2114881](https://doi.org/10.1109/TASL.2011.2114881).
  STOI: el frontal común de tercios de octava, la normalización y el recorte
  señal-distorsión, y la correlación de envolventes por banda que promedia el
  índice.
  Citado por [Inteligibilidad objetiva (STOI y ESTOI)](/phonometry/es/perception/speech/objective-intelligibility/).
- Taal, C. H., Hendriks, R. C., Heusdens, R., & Jensen, J. (2010). A
  short-time objective intelligibility measure for time-frequency weighted
  noisy speech. *2010 IEEE International Conference on Acoustics, Speech and
  Signal Processing (ICASSP)*, 4214-4217.
  [doi:10.1109/ICASSP.2010.5495701](https://doi.org/10.1109/ICASSP.2010.5495701).
  La versión corta de congreso del STOI.
  Citado por [Inteligibilidad objetiva (STOI y ESTOI)](/phonometry/es/perception/speech/objective-intelligibility/).
- Jensen, J., & Taal, C. H. (2016). An algorithm for predicting the
  intelligibility of speech masked by modulated noise maskers. *IEEE/ACM
  Transactions on Audio, Speech, and Language Processing*, 24(11), 2009-2022.
  [doi:10.1109/TASLP.2016.2585878](https://doi.org/10.1109/TASLP.2016.2585878).
  ESTOI: el espectrograma de tiempo corto normalizado por filas y columnas y
  su índice intermedio de correlación espectral.
  Citado por [Inteligibilidad objetiva (STOI y ESTOI)](/phonometry/es/perception/speech/objective-intelligibility/).

## Psicoacústica

- Moore, B. C. J. (2013). *An introduction to the psychology of hearing*
  (6.ª ed.). Brill.
  [doi:10.1163/9789004252424](https://doi.org/10.1163/9789004252424).
  El libro de texto de referencia sobre percepción auditiva; sus páginas
  76-77 dan el ancho de banda del filtro auditivo ERB_N de Glasberg y Moore
  (1990) y la escala de frecuencia Cam (número ERB_N) sobre la que se
  escriben los modelos de sonoridad.
  Citado por [Sonoridad avanzada](/phonometry/es/perception/psychoacoustics/advanced-loudness/).
- Fletcher, H., & Munson, W. A. (1933). Loudness, its definition, measurement
  and calculation. *The Journal of the Acoustical Society of America*, 5(2),
  82-108. [doi:10.1121/1.1915637](https://doi.org/10.1121/1.1915637).
  Las mediciones originales de igual sonoridad cuya isófona de 40 fonios se
  convirtió en la curva de ponderación A.
  Citado por [Ponderación frecuencial (A, C, Z)](/phonometry/es/signal/levels/weighting/)
  y [Sonoridad](/phonometry/es/perception/psychoacoustics/loudness/).
- International Organization for Standardization. (2023). *Acoustics —
  Normal equal-loudness-level contours* (ISO 226:2023).
  [Catálogo iso.org](https://www.iso.org/standard/83117.html).
  Las líneas isofónicas modernas, sucesoras de las curvas de Fletcher y
  Munson.
  Citado por [Ponderación frecuencial (A, C, Z)](/phonometry/es/signal/levels/weighting/)
  y [Sonoridad](/phonometry/es/perception/psychoacoustics/loudness/).
- Fastl, H., & Zwicker, E. (2007). *Psychoacoustics: Facts and models*
  (3.ª ed.). Springer.
  [doi:10.1007/978-3-540-68888-4](https://doi.org/10.1007/978-3-540-68888-4).
  El modelo de molestia psicoacústica y la forma cerrada de la intensidad de
  fluctuación para ruido de banda ancha modulado en amplitud.
  Citado por [Molestia psicoacústica](/phonometry/es/perception/psychoacoustics/psychoacoustic-annoyance/),
  [Sonoridad](/phonometry/es/perception/psychoacoustics/loudness/) y
  [Métricas de calidad sonora](/phonometry/es/perception/psychoacoustics/sound-quality/).
- Osses Vecchi, A., García León, R., & Kohlrausch, A. (2016). Modelling the
  sensation of fluctuation strength. *Proceedings of Meetings on Acoustics*,
  28, 050005. [doi:10.1121/2.0000410](https://doi.org/10.1121/2.0000410).
  El modelo de señal de la intensidad de fluctuación y sus valores de la
  Tabla 1 de la literatura.
  Citado por [Molestia psicoacústica](/phonometry/es/perception/psychoacoustics/psychoacoustic-annoyance/).
- Felix Greco, G., Merino-Martínez, R., Osses, A., & Lotinga, M. J. B. (2025).
  *SQAT: a sound quality analysis toolbox for MATLAB* (software de código
  abierto). [github.com/ggrecow/SQAT](https://github.com/ggrecow/SQAT),
  [doi:10.5281/zenodo.7934709](https://doi.org/10.5281/zenodo.7934709).
  La referencia abierta en MATLAB usada como oráculo numérico de las
  comprobaciones de la intensidad de fluctuación.
  Citado por [Molestia psicoacústica](/phonometry/es/perception/psychoacoustics/psychoacoustic-annoyance/).
- Ecma International. (2024). *ECMA-418-1: Psychoacoustic metrics for ITT
  equipment — Part 1: Prominent discrete tones* (3.ª ed.).
  [PDF gratuito](https://ecma-international.org/wp-content/uploads/ECMA-418-1_3rd_edition_december_2024.pdf).
  Los métodos de relación tono-ruido y relación de prominencia, de descarga
  gratuita.
  Citado por [Tonos discretos prominentes](/phonometry/es/perception/psychoacoustics/tone-prominence/).
- Ecma International. (2025). *ECMA-74: Measurement of airborne noise emitted
  by information technology and telecommunications equipment* (22.ª ed.).
  [PDF gratuito](https://ecma-international.org/wp-content/uploads/ECMA-74_22nd_edition_december_2025.pdf).
  La norma de emisión matriz, de descarga gratuita, cuyo anexo D delega la
  evaluación de tonos en ECMA-418-1.
  Citado por [Tonos discretos prominentes](/phonometry/es/perception/psychoacoustics/tone-prominence/).
- International Organization for Standardization. (2016). *Acoustics —
  Objective method for assessing the audibility of tones in noise —
  Engineering method* (ISO/PAS 20065:2016).
  [Catálogo iso.org](https://www.iso.org/standard/66941.html).
  El método de ingeniería para la audibilidad objetiva de tonos.
  Citado por [Audibilidad objetiva de tonos](/phonometry/es/perception/psychoacoustics/tone-audibility/).

## Audición y conservación auditiva

- International Organization for Standardization. (2017). *Acoustics —
  Statistical distribution of hearing thresholds related to age and gender*
  (ISO 7029:2017). [Catálogo iso.org](https://www.iso.org/standard/42916.html).
  El modelo por edad del umbral de audición y su dispersión poblacional.
  Citado por [Umbral de audición](/phonometry/es/perception/hearing/hearing-threshold/).
- International Organization for Standardization. (2005). *Acoustics —
  Reference zero for the calibration of audiometric equipment — Part 7:
  Reference threshold of hearing under free-field and diffuse-field listening
  conditions* (ISO 389-7:2005).
  [Catálogo iso.org](https://www.iso.org/standard/38976.html).
  El cero audiométrico como nivel de presión acústica.
  Citado por [Umbral de audición](/phonometry/es/perception/hearing/hearing-threshold/).
- International Organization for Standardization. (2013). *Acoustics —
  Estimation of noise-induced hearing loss* (ISO 1999:2013).
  [Catálogo iso.org](https://www.iso.org/standard/45103.html).
  El modelo de NIPTS, su distribución y la combinación HTLAN.
  Citado por [Pérdida auditiva inducida por ruido](/phonometry/es/perception/hearing/noise-induced-hearing-loss/).
- Passchier-Vermeer, W. (1974). Hearing loss due to continuous exposure to
  steady-state broad-band noise. *The Journal of the Acoustical Society of
  America*, 56(5), 1585–1593.
  [doi:10.1121/1.1903482](https://doi.org/10.1121/1.1903482).
  Un estudio de campo de las relaciones exposición-respuesta del ruido
  codificadas después en la ISO 1999.
  Citado por [Pérdida auditiva inducida por ruido](/phonometry/es/perception/hearing/noise-induced-hearing-loss/).
- National Institute for Occupational Safety and Health. (1998). *Criteria for
  a recommended standard: Occupational noise exposure — Revised criteria 1998*
  (DHHS/NIOSH Publication No. 98-126).
  [doi:10.26616/NIOSHPUB98126](https://doi.org/10.26616/NIOSHPUB98126),
  [PDF gratuito](https://www.cdc.gov/niosh/docs/98-126/pdfs/98-126.pdf).
  El documento de criterios de libre acceso que sustenta el límite de exposición
  recomendado de 85 dB(A) y la discusión de conservación auditiva y umbral.
  Citado por [Pérdida auditiva inducida por ruido](/phonometry/es/perception/hearing/noise-induced-hearing-loss/) y
  [Exposición al ruido en el trabajo](/phonometry/es/perception/hearing/occupational-exposure/).
- International Organization for Standardization. (2009). *Acoustics —
  Determination of occupational noise exposure — Engineering method*
  (ISO 9612:2009). [Catálogo iso.org](https://www.iso.org/standard/41718.html).
  Las tres estrategias de medición y el presupuesto de incertidumbre del
  anexo C.
  Citado por [Exposición al ruido en el trabajo](/phonometry/es/perception/hearing/occupational-exposure/).
- European Parliament and Council. (2003). *Directive 2003/10/EC on the
  minimum health and safety requirements regarding the exposure of workers to
  the risks arising from physical agents (noise)*. Diario Oficial de la Unión
  Europea. [eur-lex.europa.eu](https://eur-lex.europa.eu/eli/dir/2003/10/oj/eng).
  Los valores de exposición que dan lugar a una acción y el valor límite de la
  UE para el ruido laboral.
  Citado por [Exposición al ruido en el trabajo](/phonometry/es/perception/hearing/occupational-exposure/).

## Vibración en humanos

- Griffin, M. J. (1996). *Handbook of human vibration*. Academic Press.
  ISBN 978-0-12-303041-2.
  [Página del editor](https://shop.elsevier.com/books/handbook-of-human-vibration/griffin/978-0-12-303041-2).
  La monografía de referencia sobre vibración de cuerpo completo y transmitida
  a la mano: la biodinámica, la incomodidad y la evidencia de efectos sobre
  la salud que sustentan las ponderaciones, las medidas de dosis y la
  orientación exposición-respuesta de las guías de vibración.
  Citado por [Vibración en humanos](/phonometry/es/vibration/human/human-vibration/) y
  [Vibración con choques múltiples](/phonometry/es/vibration/human/multiple-shock-vibration/).
- Mansfield, N. J. (2004). *Human response to vibration*. CRC Press.
  ISBN 978-0-415-28239-0.
  [Página del editor](https://www.routledge.com/Human-Response-to-Vibration/Mansfield/p/book/9780415282390).
  Un manual moderno y compacto sobre las cadenas de evaluación de ISO 2631-1 e
  ISO 5349, de la percepción y el confort a los límites de exposición
  laborales.
  Citado por [Vibración en humanos](/phonometry/es/vibration/human/human-vibration/).

## Metrología

- Joint Committee for Guides in Metrology. (2008). *Evaluation of measurement
  data — Guide to the expression of uncertainty in measurement* (JCGM
  100:2008, la GUM). BIPM.
  [doi:10.59161/JCGM100-2008E](https://doi.org/10.59161/JCGM100-2008E),
  [PDF gratuito](https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf).
  La ley de propagación de la incertidumbre implementada por el módulo de
  incertidumbre.
  Citado por [Incertidumbre de medida](/phonometry/es/signal/metrology/gum-uncertainty/).
- Joint Committee for Guides in Metrology. (2008). *Evaluation of measurement
  data — Supplement 1 to the "Guide to the expression of uncertainty in
  measurement" — Propagation of distributions using a Monte Carlo method*
  (JCGM 101:2008). BIPM.
  [doi:10.59161/JCGM101-2008](https://doi.org/10.59161/JCGM101-2008),
  [PDF gratuito](https://www.bipm.org/documents/20126/2071204/JCGM_101_2008_E.pdf).
  La propagación de distribuciones mediante Monte Carlo implementada por el
  motor de incertidumbre de Monte Carlo.
  Citado por [Incertidumbre de medida](/phonometry/es/signal/metrology/gum-uncertainty/).
- International Organization for Standardization. (2020). *Acoustics —
  Determination and application of measurement uncertainties in building
  acoustics — Part 1: Sound insulation* (ISO 12999-1:2020).
  [Catálogo de iso.org](https://www.iso.org/standard/73930.html).
  El balance de reproducibilidad específico de la acústica de la edificación
  para magnitudes de número único, complemento de la maquinaria general de la
  GUM.
  Citado por [Incertidumbre de medida](/phonometry/es/signal/metrology/gum-uncertainty/).
