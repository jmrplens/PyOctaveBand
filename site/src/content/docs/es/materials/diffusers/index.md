---
title: "Difusores y superficies"
description: "Valorar y diseñar superficies por lo que reflejan: los coeficientes de dispersión ISO 17497-1 y de difusión ISO 17497-2 con el diseño de difusores de Schroeder, los metadifusores en sublongitud de onda profunda, y la absorción in situ de pavimentos de carretera de ISO 13472."
---

Donde la subsección de
[Absorbentes](/phonometry/es/materials/absorbers/) pregunta cuánta
energía retira un material del campo, esta pregunta qué hace una *superficie*
con el sonido que devuelve: cuánto lanza fuera de la dirección especular y con
qué uniformidad lo reparte. Dos guías recorren ese terreno, y una tercera
familia de medidas de superficie — los pavimentos caracterizados allí donde
están — tiene su propia subsección anidada dentro de este grupo.

[Difusores y sus coeficientes](/phonometry/es/materials/diffusers/diffusers/) es el núcleo
de medida y diseño: el **coeficiente de dispersión** de incidencia aleatoria
$s$ de ISO 17497-1, medido sobre la plataforma giratoria de una cámara
reverberante, el **coeficiente de difusión** $d$ de ISO 17497-2, medido en un
goniómetro de campo libre, las reglas de diseño de residuo cuadrático de
Schroeder con la predicción de campo lejano de Fraunhofer que valora una
secuencia de profundidades antes de construirla, y el argumento final de por
qué los dos coeficientes no deben intercambiarse jamás.

[Metadifusores](/phonometry/es/materials/diffusers/metadiffusers/) encoge el difusor de
Schroeder un orden de magnitud: ranuras cargadas con resonadores de
Helmholtz ralentizan el sonido hasta que un panel de 2 cm reproduce las fases
de reflexión de pozos de 27 cm, con el acoplamiento crítico aportando el
estado `0` perfectamente absorbente que necesitan las secuencias ternarias.
El diseño de residuo cuadrático publicado se evalúa de principio a fin, de la
cadena de matrices de transferencia a la comprobación cruzada FDTD.

[Superficies medidas in situ](/phonometry/es/materials/surfaces/) saca la
pregunta de la absorción al exterior, a las superficies que no tienen probeta:
cubre la técnica de sustracción de ISO 13472-1 y el tubo puntual de
ISO 13472-2, y la decisión entre ambos.

Los vecinos están cerca: los paneles difusores son parientes de superficie de
los [metaabsorbentes](/phonometry/es/materials/absorbers/metamaterial-absorbers/)
construidos con la misma celda de ranura y resonador, el coeficiente de
dispersión alimenta las predicciones de sala de
[Acústica de salas](/phonometry/es/buildings/rooms/), y los
métodos de pavimentos sirven al interés por el ruido exterior de
[Medio ambiente y transporte](/phonometry/es/environment/).

## Páginas de esta sección

- [Difusores y sus coeficientes](/phonometry/es/materials/diffusers/diffusers/): los
  coeficientes de dispersión ISO 17497-1 y de difusión ISO 17497-2, el diseño
  de Schroeder y el modelo de predicción en campo lejano.
- [Metadifusores](/phonometry/es/materials/diffusers/metadiffusers/): difusores de
  Schroeder en sublongitud de onda profunda a partir de ranuras cargadas
  con resonadores, con sonido lento y secuencias ternarias.

## Véase también

Páginas de otras áreas del sitio en las que se apoya esta sección:

- [Superficies medidas in situ](/phonometry/es/materials/surfaces/) y su guía
  [Absorción in situ de pavimentos de carretera](/phonometry/es/materials/surfaces/road-absorption/):
  la técnica de sustracción de ISO 13472-1 y el método puntual de
  ISO 13472-2.

## Qué no cubre esta sección

Los dos modelos de predicción de aquí son **estimaciones de diseño, no
mediciones**. El campo lejano de Fraunhofer que comparten las páginas de
difusores y metadifusores pierde exactitud en baja frecuencia, a incidencias
rasantes y sobre superficies muy absorbentes, e ignora la difracción de bordes,
así que valora una secuencia de profundidades antes de construirla pero no
sustituye a una medición ISO 17497-2; el modelo de metadifusor es además de
reacción local, sin acoplamiento entre pozos. Solo la secuencia de
profundidades de residuo cuadrático tiene una función dedicada — las
disposiciones de raíz primitiva y moduladas se comentan como guía de diseño y
entran por los argumentos explícitos de profundidad o de reflexión. El problema
inverso, resolver las geometrías de los resonadores para un perfil de fase
objetivo, no está automatizado: el flujo de trabajo ajusta las fases por
evaluación. Del lado de la medida, la biblioteca reduce los datos pero no
maneja el banco: la plataforma giratoria de ISO 17497-1 y el goniómetro de
ISO 17497-2 aportan los tiempos de reverberación y la respuesta polar, y lo que
está implementado es la aritmética que los convierte en un coeficiente.
