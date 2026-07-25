---
title: "Fuentes y dispositivos"
description: "Caracterizar lo que emite el sonido: determinación de la potencia acústica por métodos de presión, cámara reverberante e intensidad (series ISO 3740 e ISO 9614), intensidad acústica con dos micrófonos (IEC 61043), las métricas de distorsión y respuesta en frecuencia de equipos de audio de IEC 60268, la separación armónica con barridos y THD(f) (Farina / Novak), y la sonoridad de programa y pico verdadero de UIT-R BS.1770-5 / EBU R 128."
---

Toda predicción del resto de esta documentación parte de un descriptor de
fuente, y esta sección es donde esos descriptores se miden. Su hilo conductor
es la **emisión**: números que pertenecen al dispositivo y no a la sala ni a
la distancia desde la que se escucha.

La magnitud central es la **potencia acústica**: la energía acústica total por
segundo que radia una fuente. Expresada en decibelios como el nivel de
potencia acústica, es la cifra que va en una hoja de datos, alimenta una
predicción de sala o de exteriores y se compara con los límites de emisión
de ruido. [Potencia acústica](/phonometry/es/guides/sound-power/)
cubre las cinco rutas normalizadas para obtenerla, desde la superficie
envolvente de presión de ISO 3744/3746 hasta la cámara reverberante de
ISO 3741, el barrido de intensidad in situ de ISO 9614-2 y los grados de
precisión de ISO 3745 e ISO 9614-3. Detrás de las
rutas por intensidad está la propia **intensidad acústica**: el flujo de
potencia con signo que puede localizar fuentes y separarlas del ruido de
fondo, medido con una sonda de dos micrófonos según IEC 61043 y cualificado
por los indicadores de campo de ISO 9614-1, cubierto en
[Intensidad acústica (p-p)](/phonometry/es/guides/intensity/).

[Electroacústica](/phonometry/es/guides/electroacoustics/) se vuelve hacia
los dispositivos que *deben* producir sonido: amplificadores, altavoces y
micrófonos. Cubre el conjunto de distorsión de IEC 60268-3 (THD, THD+N y
SINAD, intermodulación y DIM), los estimadores de respuesta en frecuencia
H1/H2 con la coherencia, y los convenios de sensibilidad de IEC 60268-4/-5,
donde suelen fallar las comparaciones entre hojas de datos.
[Distorsión con barridos y utilidades de fase](/phonometry/es/guides/swept-sine-distortion/)
amplía el banco de pruebas con la alternativa de un solo barrido: la
separación armónica de Farina / Novak, que convierte un único barrido
exponencial en el conjunto completo de respuestas en frecuencia armónicas y
en una THD medida en función de la frecuencia de excitación, más las
utilidades de fase mínima, retardo de grupo y exceso de fase que diseccionan
de qué está hecha la fase de la respuesta medida.

[Sonoridad de programa](/phonometry/es/guides/program-loudness/) cubre la
señal que transportan esos dispositivos: la sonoridad de un programa de
radiodifusión o streaming según UIT-R BS.1770-5 en LUFS, la normalización a
-23 LUFS de EBU R 128, los medidores momentáneo/corto plazo/integrado del
modo EBU, el rango de sonoridad y el nivel de pico verdadero por
sobremuestreo en dBTP.

Si vienes a medir una máquina, empieza por
[Potencia acústica](/phonometry/es/guides/sound-power/) y deja que su guía de
decisión elija la ruta; lee
[Intensidad acústica (p-p)](/phonometry/es/guides/intensity/) cuando esa ruta
implique una sonda de intensidad. Si vienes a caracterizar equipos de audio,
ve directamente a
[Electroacústica](/phonometry/es/guides/electroacoustics/); si vienes a
nivelar un programa, ve a
[Sonoridad de programa](/phonometry/es/guides/program-loudness/).

## Páginas de esta sección

- [Intensidad acústica (p-p)](/phonometry/es/guides/intensity/): intensidad
  sonora con dos micrófonos según IEC 61043 con los indicadores de campo de
  ISO 9614-1.
- [Potencia acústica](/phonometry/es/guides/sound-power/): el nivel de potencia
  acústica por superficie envolvente, cámara reverberante, barrido de
  intensidad y los métodos de precisión anecoico y de intensidad.
- [Electroacústica: distorsión y respuesta en frecuencia](/phonometry/es/guides/electroacoustics/):
  las métricas de distorsión de IEC 60268-3, la estimación de la respuesta en
  frecuencia con coherencia, y los convenios de sensibilidad de micrófonos y
  altavoces.
- [Distorsión con barridos y utilidades de fase](/phonometry/es/guides/swept-sine-distortion/):
  separación armónica y THD(f) con un solo barrido exponencial (barrido
  sincronizado de Farina / Novak), y fase mínima, retardo de grupo y exceso
  de fase de una respuesta medida.
- [Sonoridad de programa y pico verdadero](/phonometry/es/guides/program-loudness/):
  la sonoridad de programa y el nivel de pico verdadero de UIT-R BS.1770-5
  con la práctica de normalización de EBU R 128, la medición en modo EBU y
  el rango de sonoridad.
