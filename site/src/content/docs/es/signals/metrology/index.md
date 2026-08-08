---
title: "Calibración e incertidumbre"
description: "Las tres disciplinas que convierten un nivel calculado en una medición defendible: calibración SPL física frente a análisis digital en dBFS, la propagación de la incertidumbre de medida por GUM y Monte Carlo, y las estadísticas de estacionariedad y de picos que califican un registro antes de promediar nada de él."
---

Un nivel impreso por un programa no es todavía una medición. Tres cosas
separan lo uno de lo otro: saber que el registro merece **promediarse
siquiera**, saber qué significan **físicamente** las muestras digitales, y
saber cuánto podría razonablemente **equivocarse** el resultado. Esta sección
cubre las tres, y aplican transversalmente a todas las demás páginas de la
documentación.

[Calibración y dBFS](/phonometry/es/signals/metrology/calibration/) se ocupa del
significado físico. phonometry trabaja en dos marcos de referencia: los
**dB SPL** físicos, establecidos a partir de un tono de calibrador grabado (el
ritual de campo de IEC 60942), y los **dBFS** digitales, niveles relativos al
fondo de escala, apropiados cuando no existe referencia física o cuando se
caracteriza la propia cadena digital. La página explica cómo se configura cada
modo y, no menos importante, qué magnitudes tienen sentido en cada marco.

[Incertidumbre de medida (GUM y Monte Carlo)](/phonometry/es/signals/metrology/gum-uncertainty/)
se ocupa de la segunda, implementando la *Guía para la expresión de la
incertidumbre de medida* (**ISO/IEC Guide 98-3:2008**) y su **Suplemento 1**
de Monte Carlo. La ruta GUM propaga las incertidumbres típicas
analíticamente, a través de los coeficientes de sensibilidad, hasta una
incertidumbre combinada y expandida, con los grados de libertad efectivos de
Welch-Satterthwaite; la ruta de Monte Carlo propaga numéricamente las
distribuciones de probabilidad completas y produce intervalos de cobertura
que siguen siendo honestos cuando el modelo es no lineal o las entradas
distan de ser gaussianas. La página muestra ambas sobre los mismos modelos,
incluyendo dónde divergen y por qué.

[Calificación de datos](/phonometry/es/signals/metrology/data-qualification/) guarda la
puerta delante de ambas: todo promedio - un Leq, una PSD de Welch, toda
entrada promediada de un presupuesto de incertidumbre - supone que el
registro es estacionario, y los
tests de inversiones de orden y de rachas de Bendat y Piersol lo deciden
objetivamente a partir de medias cuadráticas por segmento, con las regiones
de aceptación del propio libro. La misma página trae las estadísticas de
Rice de cruces por nivel y de picos - frecuencia aparente, tasas de picos,
el factor de irregularidad - que caracterizan un registro gaussiano
calificado y criban el que no lo es.

La misma disciplina se extiende al dominio de la frecuencia: las páginas de
[Señales y espectros](/phonometry/es/signals/spectra/)
aplican el análisis de error de Bendat y Piersol a las estimaciones
espectrales de Welch, de modo que cada PSD lleva su número efectivo de
promedios, su error aleatorio normalizado y un intervalo de confianza
chi-cuadrado.

Las páginas se encuentran en la práctica: un presupuesto de incertidumbre
de una medición acústica casi siempre contiene un término de calibración, y
varias normas implementadas en otras partes de la biblioteca (ISO 9612,
ISO 12999-1) traen presupuestos de incertidumbre que son especializaciones de
la maquinaria GUM descrita aquí.

## Páginas de esta sección

- [Calibración y dBFS](/phonometry/es/signals/metrology/calibration/): calibración SPL
  física desde un tono de calibrador, la comprobación de estabilidad que
  aplica a esa grabación, y el modo digital de fondo de escala.
- [Incertidumbre de medida (GUM y Monte Carlo)](/phonometry/es/signals/metrology/gum-uncertainty/):
  la ley de propagación de la incertidumbre y el método de Monte Carlo,
  incertidumbre expandida e intervalos de cobertura.
- [Calificación de datos](/phonometry/es/signals/metrology/data-qualification/): los
  tests de estacionariedad por inversiones de orden y por rachas sobre
  estadísticas de segmento, y las estadísticas de Rice de cruces por nivel y
  de picos con el factor de irregularidad.

## Qué no cubre esta sección

Aquí nada comprueba un instrumento ni un calibrador. Los ensayos de
conformidad de IEC 60942 sobre el propio calibrador (nivel generado,
frecuencia, distorsión y las correcciones por presión estática y temperatura)
no están implementados, así que pasa un `target_spl` ya corregido cuando el
manual lo pida, y los ensayos periódicos de IEC 61672-3 se citan como práctica
de laboratorio, no se ejecutan. La parte de dBFS de la página de calibración
queda fuera de cualquier norma y no hace ninguna afirmación física: es un
marco de referencia, no una medición. La calificación de datos implementa solo
el núcleo cuantitativo del apartado 10.3 de Bendat y Piersol: clasificar el
tipo de un registro, validarlo frente a límites físicos y editar transitorios
espurios siguen siendo pasos manuales, tal como los describe el libro. Y un
presupuesto de incertidumbre es un modelo de *tu* medición: la biblioteca
propaga lo que declares y nunca inventa los términos. Donde una norma publica
su propio presupuesto, las guías lo dicen, como hace ISO 9612 para la
[exposición laboral](/phonometry/es/perception/hearing/occupational-exposure/) e
ISO 12999-1 para el [aislamiento acústico](/phonometry/es/buildings/insulation/).
