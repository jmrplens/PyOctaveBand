---
title: "Fuentes de ruido estructural"
description: "La cadena desde una máquina vibrante hasta el ruido oído varias salas más allá: la familia de FRF de ISO 7626, la rigidez de transferencia de aisladores de ISO 10846, la potencia radiada desde vibración de ISO/TS 7849, la placa de recepción de EN 15657 y la predicción de instalación de EN 12354-5."
---

Una máquina fijada a un edificio radia sonido dos veces: directamente desde
su propia superficie vibrante, e indirectamente inyectando **potencia
estructural** en la estructura, que la transporta y la vuelve a radiar en
salas lejanas. Seis páginas cubren ambos caminos, tres aquí y tres en otras
partes del sitio: una estima la radiación directa desde la propia vibración
superficial, y las otras cinco caracterizan de extremo a extremo ese segundo
camino
estructural, el más escurridizo, desde describir la vibración y caracterizar
los aisladores hasta cuantificar la potencia y predecir el nivel que
finalmente oye un oyente.

Primero el lenguaje.
[Movilidad mecánica y la familia de FRF (ISO 7626-1)](/phonometry/es/vibration/structural/mechanical-mobility/)
define las funciones de respuesta en frecuencia movimiento-por-fuerza
(receptancia, movilidad, acelerancia y sus recíprocas) que hablan todas las
normas posteriores, con el resonador de un grado de libertad en forma cerrada
como referencia y los criterios de aceptación de medida de ISO 7626-2. Las
*movilidades* de fuente y receptor son las que deciden cuánta potencia se
acopla realmente a través de una interfaz, y por eso importa este
vocabulario.

Tres páginas caracterizan después los elementos del camino.
[Transmisión de onda de flexión en uniones de placas
(Cremer/Craik/Hopkins)](/phonometry/es/vibration/structural/junction-transmission/)
sigue la potencia a través de la propia estructura, con los coeficientes de
transmisión del enfoque ondulatorio para uniones rígidas en X, T, L y en
línea, su media angular de campo difuso, y el factor de pérdidas por
acoplamiento y el índice de reducción vibratoria Kij que se derivan de
ellos.
[Rigidez dinámica de transferencia (ISO 10846)](/phonometry/es/vibration/structural/transfer-stiffness/)
mide la rigidez de transferencia dinámica de los aisladores, soportes y
mangueras que se insertan precisamente para romper la trayectoria de transmisión,
por los métodos directo e indirecto (transmisibilidad).
[Potencia acústica a partir de la vibración (ISO/TS 7849)](/phonometry/es/devices/emission/vibration-sound-power/)
se ocupa de la radiación directa: la potencia aérea estimada desde la
velocidad superficial y un factor de radiación, sin medición acústica
alguna.

Las dos últimas páginas cierran la cadena sobre la fuente y el receptor.
[Potencia acústica estructural de equipos (EN 15657)](/phonometry/es/buildings/design/structure-borne-power/)
mide lo que inyecta una máquina, mediante el método de la placa de recepción,
y deriva las magnitudes de fuente independientes de la placa (fuerza
bloqueada, nivel de potencia característico, velocidad libre).
[Ruido estructural instalado (EN 12354-5)](/phonometry/es/buildings/design/installed-structure-borne/)
consume exactamente esas magnitudes, las acopla a través de las movilidades
de fuente y receptor, y predice el nivel de presión acústica en el recinto
receptor, que es donde esta sección se encuentra con los modelos de
[aislamiento acústico](/phonometry/es/buildings/insulation/).

## Páginas de esta sección

- [Movilidad mecánica y la familia de FRF (ISO 7626-1)](/phonometry/es/vibration/structural/mechanical-mobility/):
  la familia de FRF, las conversiones y el resonador de referencia de un
  grado de libertad.
- [Transmisión de onda de flexión en uniones de placas (Cremer/Craik/Hopkins)](/phonometry/es/vibration/structural/junction-transmission/):
  los coeficientes del enfoque ondulatorio para uniones rígidas en X, T, L y en
  línea, su media angular y el factor de pérdidas por acoplamiento y el Kij.
- [Rigidez dinámica de transferencia (ISO 10846)](/phonometry/es/vibration/structural/transfer-stiffness/):
  rigidez de transferencia dinámica de aisladores por los métodos directo e
  indirecto.

## Véase también

Páginas de otras áreas del sitio en las que se apoya esta sección:

- [Potencia acústica a partir de la vibración (ISO/TS 7849)](/phonometry/es/devices/emission/vibration-sound-power/):
  potencia aérea radiada desde la velocidad superficial y un factor de
  radiación.
- [Potencia acústica estructural de equipos (EN 15657)](/phonometry/es/buildings/design/structure-borne-power/):
  el método de la placa de recepción y las magnitudes de fuente
  independientes de la placa.
- [Ruido estructural instalado (EN 12354-5)](/phonometry/es/buildings/design/installed-structure-borne/):
  el nivel predicho en el recinto receptor desde el equipo instalado.

## Qué no cubre esta sección

Los coeficientes de unión son una **idealización en forma cerrada** para una
unión rígida y simplemente apoyada entre placas homogéneas, no una medición: el
índice de reducción vibratoria empírico obtenido de una diferencia de niveles
de velocidad promediada en dirección es la ISO 10848, en [Transmisión por
flancos en laboratorio](/phonometry/es/buildings/insulation/flanking-lab/). El
coeficiente
de tramo recto no está definido para las geometrías T y L, que no tienen una
tercera placa colineal, así que ahí solo se aplica el camino de esquina.

La página de FRF implementa la ISO 7626-1 y los criterios de aceptación de la
ISO 7626-2 para un excitador acoplado; **la excitación por martillo de impacto
(ISO 7626-5) se nombra solo como contexto**, sin nada que sintetice ni procese
un espectro de impacto, y las conversiones devuelven recíprocas *libres*
elemento a elemento, correctas para el punto de excitación o para una sola vía,
no para una matriz de FRF completa, cuyas magnitudes matriciales bloqueadas no
se construyen.

En la página de aisladores, las Partes 4 y 5 de la ISO 10846 no están
implementadas, y dos de las comprobaciones de validez de la propia norma se
describen en vez de calcularse: la desigualdad de masa de bloqueo rígida y el
criterio de linealidad del apartado 7.6 (dos espectros de entrada separados
10 dB que coincidan dentro de 1,5 dB). Por último, aquí no se diseña ningún
aislador ni ninguna bancada flotante: las páginas caracterizan elementos y
predicen la transmisión, y la decisión de dimensionado se queda contigo.
