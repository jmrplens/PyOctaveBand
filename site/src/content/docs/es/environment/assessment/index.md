---
title: "Evaluación y normativa"
description: "Evaluar el ruido ambiental una vez ha llegado: los indicadores y el nivel de evaluación de ISO 1996-1, la cadena de determinación de ISO 1996-2 con su corrección de ruido residual y su balance de incertidumbre, el ajuste por sonidos impulsivos, y una aplicación nacional de toda la cadena."
---

La propagación dice qué llega al receptor. La evaluación dice cómo se
contabiliza, que es otra pregunta con sus propias normas. Es una cadena de tres
eslabones: un **indicador** promedia el sonido sobre un periodo definido; unos
**ajustes** suman decibelios por el carácter que ese promedio no recoge, tono a
tono e impulso a impulso; y un **límite**, siempre nacional, decide. La
ISO 1996 aporta los dos primeros eslabones, y una reglamentación nacional
aporta el tercero junto con su propia versión del segundo.

[Niveles ambientales (ISO
1996-1/-2)](/phonometry/es/environment/assessment/environmental-levels/) es la
capa de indicadores y ajustes, y la página que quiere la mayoría de quienes
llegan a esta subsección. El Lden pondera la tarde con +5 dB y la noche con
+10 dB sobre periodos de 12/4/8 h por defecto, ajustables porque cada Estado
miembro los define distinto; el Ldn es la variante día-noche; y el nivel de
evaluación compuesto del apartado 6.5 generaliza ambos a periodos arbitrarios
con ajustes por fuente y por carácter, desde +5 dB para el sonido impulsivo
regular hasta +12 dB para el altamente impulsivo. La mitad de ISO 1996-2 de la
página *determina* en vez de definir: el ajuste tonal del Anexo C, la
corrección de ruido residual del apartado 10.4 y el balance de incertidumbre
del Anexo F, que dice cuánto vale el nivel de evaluación. Parte de niveles por
periodo que ya tienes; producirlos es
[Niveles integrados y estadísticos](/phonometry/es/signals/levels/levels/).

[Prominencia de sonidos impulsivos (NT ACOU 112)](/phonometry/es/environment/assessment/impulsive-sound/)
es el ajuste para el sonido cuyos impulsos lo hacen más molesto de lo que
sugiere su LAeq, tanto en la forma cerrada del Nordtest como en la cadena de
medición de ISO/PAS 1996-3: de la tasa de crecimiento y la diferencia de nivel
de cada impulso predice una prominencia y la convierte en el ajuste graduado KI
que se suma al LAeq medido. Es la medición que sustituye al juicio de un
técnico evaluador en la casilla del ajuste por carácter de la cadena anterior,
y termina en una ficha de evaluación `.report()`.

[Normativa española de ruido (RD 1367/2007)](/phonometry/es/environment/assessment/spanish-noise-regulation/)
es el aspecto que tiene toda la cadena una vez que un Estado la ha legislado:
el nivel corregido LKeq con sus correcciones tonal, de baja frecuencia e
impulsiva Kt, Kf y Ki, los periodos de evaluación divididos en fases de ruido,
y las tablas de límites con las que se juzga una actividad según el uso del
suelo. Léela como el ejemplo resuelto de una capa nacional aunque no trabajes
en España — enseña qué partes de la ISO 1996 reformula normalmente una
reglamentación, y cuáles sustituye. La Kf, la corrección gobernada por la
diferencia entre el nivel ponderado C y el ponderado A, no tiene ninguna
contraparte en ISO 1996.

Lee primero Niveles ambientales, luego la página de sonidos impulsivos como el
ajuste que la alimenta, y después la normativa española como el montaje
nacional de ambas.

## Páginas de esta sección

- [Niveles ambientales (ISO 1996-1/-2)](/phonometry/es/environment/assessment/environmental-levels/):
  Lden, Ldn y el nivel de evaluación compuesto del apartado 6.5, el ajuste
  tonal del Anexo C, la corrección de ruido residual del apartado 10.4 y el
  balance de incertidumbre del Anexo F.
- [Prominencia de sonidos impulsivos (NT ACOU 112)](/phonometry/es/environment/assessment/impulsive-sound/):
  la prominencia predicha de los sonidos impulsivos, el ajuste graduado KI del
  LAeq, la cadena de medición de ISO/PAS 1996-3 y la ficha de evaluación.
- [Normativa española de ruido (RD 1367/2007)](/phonometry/es/environment/assessment/spanish-noise-regulation/):
  el nivel corregido LKeq, las correcciones Kt/Kf/Ki, los periodos de
  evaluación y las fases de ruido, y las tablas de límites de inmisión.

## Véase también

Páginas de otras áreas del sitio en las que se apoya esta sección:

- [Niveles integrados y estadísticos](/phonometry/es/signals/levels/levels/):
  los niveles LAeq, percentiles y de suceso de cada periodo de referencia, de
  los que parte aquí todo indicador.
- [Audibilidad objetiva de tonos en ruido](/phonometry/es/perception/psychoacoustics/tone-audibility/):
  el método de ingeniería de ISO/PAS 20065 cuya audibilidad media traduce a
  decibelios el ajuste tonal de ISO 1996-2.
- [Propagación del sonido en exteriores](/phonometry/es/environment/propagation/outdoor-propagation/):
  el camino que ha llevado el sonido hasta el receptor que se evalúa.

## Qué no cubre esta sección

La biblioteca empieza donde termina el sonómetro. La ISO 1996-2 fija las
posiciones del receptor y las correcciones de fachada que convierten una medida
en bruto en el nivel que esperan estas funciones, y esos procedimientos de
posición y corrección **no están implementados** — solo la aritmética que sigue
una vez aplicados. Lo mismo vale en el plano nacional: los procedimientos de
medición del anexo IV del RD 1367/2007 (posiciones de micrófono, duración de
las series, número de mediciones) tampoco están implementados, y la
zonificación acústica, los mapas de ruido y los planes de acción de la
Ley 37/2003 son instrumentos de planificación más que cálculos. Faltan a
propósito dos alternativas publicadas: solo está implementado el KI graduado de
la Fórmula 2 de NT ACOU 112, no el valor fijo de 5 dB de reserva de su Nota 4,
y los ajustes por categoría de la Tabla A.1 de ISO 1996-1 aparecen únicamente
como la referencia de juicio del técnico evaluador a la que sustituye la
medición. Por último, no hay ninguna tabla de límites incorporada aparte de la
española: un límite es nacional, y la biblioteca te da el nivel de evaluación
para compararlo con el que corresponda.
