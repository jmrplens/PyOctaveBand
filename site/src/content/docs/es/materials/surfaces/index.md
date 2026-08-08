---
title: "Superficies medidas in situ"
description: "Superficies que no tienen probeta: por qué un firme se caracteriza donde está, la técnica de sustracción de ISO 13472-1 frente al tubo puntual de ISO 13472-2, a qué está acotado cada uno, y las ediciones que implementa la biblioteca."
---

Un coeficiente de cámara reverberante o de tubo de impedancia describe una
*probeta*. Algunas superficies no tienen probeta. Un firme no se puede recortar
y llevar a un laboratorio sin destruir justo aquello que gobierna su absorción
— la estructura de poros conectados de la capa extendida y compactada — y un
testigo extraído de él ya no es la superficie por la que rueda un neumático.
Los métodos in situ responden la pregunta allí donde está la superficie, y lo
pagan con un problema de geometría: el micrófono oye juntos el sonido directo y
la reflexión en la superficie, así que el método se construye alrededor de
separarlos en el **tiempo** y no en el espacio.

[Absorción in situ de firmes de carretera](/phonometry/es/materials/surfaces/road-absorption/)
implementa las dos partes de ISO 13472 y, más útil todavía, dice cuál de ellas
admite un firme dado. La **técnica de sustracción** de la Parte 1 sitúa una
fuente y un micrófono sobre la superficie, resta una medición de referencia en
campo libre y aplica la ventana de Adrienne para quedarse con la reflexión y
descartar todo lo posterior. Cubre el rango completo, de firmes reflectantes a
muy absorbentes, abarca de 250 Hz a 4 kHz y promedia sobre un parche de metros
de lado — una ventana de 5 ms da un radio máximo de área muestreada de unos
1,34 m, del orden de 5,6 m² de pavimento, así que ve textura y juntas en vez de
un solo punto. El **método puntual** de la Parte 2 sella un tubo portátil corto
sobre el pavimento y lo lee con la rutina de función de transferencia de dos
micrófonos. Solo necesita un parche plano y sellable y minutos por punto, así
que puede posarse en una rodada o en una franja estrecha, pero está limitado a
superficies reflectantes, se declara no fiable en cuanto la absorción medida
supera 0,15, y se detiene en 1600 Hz — lo cual importa, porque el ruido de
rodadura al que suele servir la medición tiene su máximo en torno a 1 kHz y
contenido más allá de ese techo.

Son complementarios, no competidores: la propia introducción de la Parte 2
espera que ambos coincidan entre 315 Hz y 1600 Hz, y los dos informan la misma
magnitud, el coeficiente de absorción a incidencia normal en bandas de tercio
de octava. Un carril de baja absorción puede así inspeccionarse con el tubo y
anclarse con una medición por sustracción en unas pocas posiciones. Ese número
es contra el que se escribe una especificación de pavimento de baja sonoridad,
y el que consume el término de suelo de un modelo de propagación en exteriores.

## Páginas de esta sección

- [Absorción in situ de firmes de carretera](/phonometry/es/materials/surfaces/road-absorption/):
  la técnica de sustracción de ISO 13472-1 con la ventana de Adrienne y sus
  funciones de geometría y validez, el tubo puntual de ISO 13472-2 con sus
  límites de aplicabilidad, y la comparación que decide entre ambos.

## Véase también

Páginas de otras áreas del sitio en las que se apoya esta sección:

- [Tubo de impedancia](/phonometry/es/materials/absorbers/impedance-tube/): la
  reducción con dos micrófonos de ISO 10534-2 que el método puntual reutiliza
  sin cambios.
- [Medida y clasificación de la absorción sonora](/phonometry/es/materials/absorbers/absorption-measurement/):
  la vía de laboratorio, para materiales que sí pueden llevarse bajo techo.
- [Medio ambiente y transporte](/phonometry/es/environment/): donde se consume
  la absorción de una carretera, como término de suelo de una predicción en
  exteriores.

## Qué no cubre esta sección

**El estado de las ediciones importa aquí más que en ningún otro sitio de esta
área.** La implementación sigue ISO 13472-1:2002 e ISO 13472-2:2010; ambas se
han revisado desde entonces — en 2022 y en 2025, respectivamente — y esas
revisiones **no** están implementadas, así que un informe que cite la edición
vigente no puede citar estas funciones sin matizarlo. Tampoco se duplica el
tratamiento de señal propio del método puntual: aquí viven solo sus funciones
de geometría, validez y corrección, y la reducción por función de transferencia
de dos micrófonos es la rutina ISO 10534-2 de
[Tubo de impedancia](/phonometry/es/materials/absorbers/impedance-tube/). Nada
de esta subsección mide el ruido que una superficie *genera* — el término de
fuente de rodadura es terreno de CNOSSOS, en [Fuentes
ambientales](/phonometry/es/environment/sources/) — y no se ofrece ningún
método in situ para otra superficie que no sea una carretera: una pared o un
techo medidos donde están quedan fuera de las dos partes de ISO 13472.
