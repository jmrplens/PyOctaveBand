---
title: "Diseño del aislamiento"
description: "El aislamiento acústico antes de construirlo: la predicción de flancos de EN/ISO 12354 entre recintos en sus formas simplificada y por bandas, el aislamiento teórico de un panel a partir de sus propiedades físicas, las capas elásticas de las que depende el diseño de un suelo o de un trasdosado, y el sonido estructural de los equipos de instalaciones desde la placa de recepción de EN 15657 hasta el nivel en el recinto receptor de EN 12354-5."
---

Las páginas de la sección de
[Aislamiento acústico](/phonometry/es/buildings/insulation/)
responden a la pregunta *¿qué consigue este edificio?* Las de aquí responden a
la que viene antes: *¿qué va a conseguir y qué conviene construir?* Ambas
mitades hablan el mismo idioma de $R$, $L_n$ y sus números únicos ponderados,
pero una predicción se ensambla a partir de datos de elemento en lugar de
medirse en una sala terminada, así que sus datos de entrada, sus hipótesis y
sus márgenes son un asunto propio.

[Predicción del aislamiento acústico (EN 12354)](/phonometry/es/buildings/design/insulation-prediction/)
es el modelo normativo: la transmisión por flancos aérea y de impacto entre dos
salas, camino a camino, a partir del elemento directo y de los índices de
reducción de vibraciones de unión $K_{ij}$. Consume datos de elemento de
laboratorio medidos según ISO 10140 y datos de unión medidos según ISO 10848,
que viven ambos en
[Aislamiento acústico](/phonometry/es/buildings/insulation/) junto
a la medición en campo con la que se contrasta la predicción y junto a la guía
de fachadas, que lleva esta misma familia a través del cerramiento del
edificio.

[Predicción detallada por bandas (ISO 12354)](/phonometry/es/buildings/design/detailed-prediction/)
ejecuta la misma norma banda a banda en lugar de sobre números únicos: los
datos de laboratorio de elemento y de unión se convierten a sus valores in
situ, cada camino se forma por banda y el resultado muestra qué camino domina
cada banda, no solo si el recinto cumple.

[Predicción del aislamiento de paneles](/phonometry/es/buildings/design/panel-sound-insulation/)
baja un nivel más, hasta el origen del propio $R$ de elemento: la ley de masa y
la caída de coincidencia de un panel simple, el comportamiento masa-muelle-masa
de una doble hoja, la transmisión por rendijas y aberturas, la eficiencia de
radiación de placas y las movilidades puntuales. Es la física que un valor de
catálogo resume en un número.

Dos páginas de aquí llevan la mitad de suelos de cualquier diseño, una que mide
y otra que predice.
[Mejora del aislamiento a impacto de suelos (ISO 16251-1)](/phonometry/es/buildings/design/impact-improvement/)
da la mejora ponderada $\Delta L_w$ de un revestimiento que ya existe, sobre una
maqueta pesada pequeña, y ese es el término que EN 12354-2 resta al nivel del
forjado desnudo.
[Predicción del comportamiento de capas elásticas](/phonometry/es/buildings/design/resilient-layers/)
la predice para un revestimiento que todavía no existe, a partir del espectro de
fuerza de la propia máquina de impactos, la frecuencia de corte de un
revestimiento blando, las leyes de 30 lg y 40 lg del suelo flotante y la
magnitud global de un trasdosado según el anexo D de ISO 12354-1.
Ambas parten de la rigidez por unidad de superficie $s'$ de la capa elástica,
medida según EN 29052-1 en
[Rigidez dinámica de materiales resilientes](/phonometry/es/materials/resilient/dynamic-stiffness/)
en la sección de materiales, que fija la resonancia de la que depende toda la
mejora.

Los equipos de instalaciones del edificio son una cadena aparte, y sus dos
páginas solo se leen bien en orden.
[Potencia acústica estructural de equipos (EN 15657)](/phonometry/es/buildings/design/structure-borne-power/)
caracteriza una bomba, un ventilador o una cisterna por la potencia que inyecta
en la estructura, medida sobre una placa de recepción de disipación conocida y
convertida después en independiente de la placa.
[Sonido estructural instalado (EN 12354-5)](/phonometry/es/buildings/design/installed-structure-borne/)
toma esa descripción de fuente, pierde parte de ella en el término de
acoplamiento que fijan las movilidades de fuente y de receptor, y lleva el resto
a un recinto que puede estar a varias uniones de distancia.

Una nota de contabilidad recorre toda la sección: la familia existe como
EN 12354:2000 y como ISO 12354:2017, y las dos no son intercambiables en todas
las cláusulas. Los modelos simplificados de
[Predicción del aislamiento acústico](/phonometry/es/buildings/design/insulation-prediction/)
siguen el texto de 2000 — incluida la corrección tabulada por flancos $K$ que la
parte de impactos de 2017 sustituyó por fórmulas explícitas camino a camino —
mientras que
[Predicción detallada por bandas](/phonometry/es/buildings/design/detailed-prediction/)
sigue el texto de 2017. Comprueba qué edición invoca tu reglamento antes de
citar una corrección de cualquiera de las dos.

## Páginas de esta sección

- [Predicción del aislamiento acústico (EN 12354)](/phonometry/es/buildings/design/insulation-prediction/):
  los modelos de flancos aéreo y de impacto entre recintos (EN 12354-1/2) con sus
  índices de reducción de vibraciones de unión y sus fichas de predicción.
- [Predicción detallada por bandas (ISO 12354)](/phonometry/es/buildings/design/detailed-prediction/):
  el modelo detallado por bandas de ISO 12354-1/-2 con la conversión in situ de
  los datos de elemento y de unión, los índices por flancos banda a banda y las
  contribuciones por camino que hay detrás de la valoración.
- [Predicción del aislamiento de paneles](/phonometry/es/buildings/design/panel-sound-insulation/):
  la ley de masa y la coincidencia (Sharp), las dobles hojas (Bies), las
  rendijas y aberturas (Gomperts, Wilson-Soroka), la eficiencia de radiación
  (Leppington/Maidanik) y las movilidades puntuales (Cremer).
- [Mejora del aislamiento a impacto de suelos (ISO 16251-1)](/phonometry/es/buildings/design/impact-improvement/):
  la mejora ponderada de un revestimiento de suelo blando medida sobre una
  maqueta pesada pequeña.
- [Predicción del comportamiento de capas elásticas](/phonometry/es/buildings/design/resilient-layers/):
  el modelo de fuerza de la máquina de impactos, la frecuencia de corte de un
  revestimiento blando, las leyes de mejora del suelo flotante y la magnitud
  global de un trasdosado según el anexo D de ISO 12354-1.
- [Potencia acústica estructural de equipos (EN 15657)](/phonometry/es/buildings/design/structure-borne-power/):
  la potencia característica que una máquina inyecta en un elemento del
  edificio, medida sobre una placa de recepción.
- [Sonido estructural instalado (EN 12354-5)](/phonometry/es/buildings/design/installed-structure-borne/):
  en qué se convierte esa potencia con la máquina montada sobre un elemento
  real, y el nivel que produce en el recinto receptor.

## Véase también

Páginas de otras áreas del sitio en las que se apoya esta sección:

- [Rigidez dinámica de materiales resilientes (EN 29052-1)](/phonometry/es/materials/resilient/dynamic-stiffness/):
  la medición por resonancia de placa de carga, el término del gas encerrado y
  la frecuencia natural del suelo flotante.
