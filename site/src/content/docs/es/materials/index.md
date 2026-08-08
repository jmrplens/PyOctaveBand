---
title: "Materiales y superficies"
description: "Caracterizar materiales y superficies acústicas: absorción en cámara reverberante y su valoración (ISO 354, ISO 11654), resistencia al flujo de aire (ISO 9053), el tubo de impedancia (ISO 10534, ASTM E2611), predicción de absorbentes porosos, multicapa y de metamaterial, difusores con sus coeficientes de dispersión y difusión (ISO 17497), metadifusores, absorción in situ de pavimentos (ISO 13472) y la rigidez dinámica de las capas elásticas (EN 29052-1)."
---

Toda predicción de sala y más de un modelo de aislamiento acaban consumiendo
un coeficiente que describe lo que un material o una superficie hace con el
sonido. Esta sección cubre de dónde salen esos coeficientes: los instrumentos
de laboratorio que los miden, los índices globales que los resumen, los
modelos de predicción que los anticipan y los métodos in situ que los
recuperan fuera del laboratorio.

La subsección de **Absorbentes** cubre cuánta energía retira un material del
campo, un instrumento o una familia de modelos por guía.
[Medida y clasificación de la absorción
sonora](/phonometry/es/materials/absorbers/absorption-measurement/)
es la cámara reverberante: los coeficientes de incidencia aleatoria de
ISO 354 y la valoración ponderada ISO 11654 α_w con su clase por letra, la
cifra que citan las hojas de características de los absorbentes, con la incertidumbre
de medida de ISO 12999-2.
[Resistencia al flujo de
aire](/phonometry/es/materials/absorbers/airflow-resistance/) es
el banco de flujo: resistencia y resistividad según ISO 9053-1/-2, el
parámetro que gobierna el comportamiento en baja frecuencia de un absorbente
poroso y ancla la mayoría de los modelos de material.
[Tubo de impedancia](/phonometry/es/materials/absorbers/impedance-tube/) es el
instrumento
de banco: la impedancia superficial compleja, el factor de reflexión y la
absorción de una muestra pequeña a incidencia normal (ISO 10534-1/-2) y, con
cuatro micrófonos, su pérdida por transmisión (ASTM E2611), más el tubo FDTD
virtual que contrasta la simulación de ondas con las mismas normas.
[Absorbentes porosos y multicapa](/phonometry/es/materials/absorbers/porous-absorbers/)
convierte la resistividad al flujo medida en *predicciones*: los modelos de
fluido equivalente de Delany-Bazley, Miki y Johnson-Champoux-Allard dan la
impedancia característica y el número de onda de un material poroso, y una
pila de capas por matrices de transferencia (porosas, de aire, perforadas,
microperforadas de Maa y de membrana) predice la absorción de una
construcción completa antes de construir nada.
[Metaabsorbentes](/phonometry/es/materials/absorbers/metamaterial-absorbers/) lleva las
mismas matrices de transferencia más allá de las reglas clásicas de espesor:
paneles ranurados de sonido lento cargados con resonadores de Helmholtz
alcanzan la absorción perfecta en el acoplamiento crítico con paneles en
sublongitud de onda profunda.

La subsección de **Difusores y superficies** pasa de las muestras a las
*superficies*, y no pregunta cuánta energía absorbe una superficie sino
adónde envía lo que refleja.
[Difusores y sus coeficientes](/phonometry/es/materials/diffusers/diffusers/) cubre las
dos valoraciones normalizadas, el **coeficiente de dispersión** de
incidencia aleatoria (ISO 17497-1) y el **coeficiente de difusión**
(ISO 17497-2), junto con el diseño de difusores de Schroeder y su predicción
en campo lejano.
[Metadifusores](/phonometry/es/materials/diffusers/metadiffusers/) reconstruye
el difusor
de Schroeder a partir de ranuras cargadas con resonadores, entre uno y dos
órdenes de magnitud más delgado.

Anidada dentro de ese grupo está **Superficies medidas in situ**, para las
superficies que no tienen probeta. Un pavimento no se puede recortar y llevar
bajo techo sin destruir la estructura de poros conectados que gobierna su
absorción, así que la geometría de laboratorio se sustituye por una ventana
temporal sobre una superficie extensa, o por un tubo apoyado sobre la
carretera.
[Absorción in situ de pavimentos de
carretera](/phonometry/es/materials/surfaces/road-absorption/)
la mide allí donde está, por la técnica de sustracción de ISO 13472-1 o con el
tubo puntual de ISO 13472-2, y dice cuál de las dos admite un pavimento dado.

La subsección de **Capas elásticas** cubre la única propiedad de material de
aquí que es mecánica y no acústica: una capa elástica no se caracteriza por
lo que hace con el sonido aéreo, sino por lo blandamente que sostiene una
masa, de modo que su medición es una resonancia y no una absorción. Un suelo
flotante es un sistema masa-resorte, la solera es la masa y la capa es el
resorte, y la rigidez dinámica por unidad de superficie s' de la capa fija la
resonancia por encima de la cual el suelo empieza a funcionar.
[Rigidez dinámica de materiales resilientes (EN
29052-1)](/phonometry/es/materials/resilient/dynamic-stiffness/)
es la medición por resonancia con placa de carga que produce s', con el
término del gas encerrado que hace que una capa permeable al aire sea más
rígida que su esqueleto solo.

Los consumidores de estos números están repartidos por el sitio: los
coeficientes de absorción alimentan las predicciones de reverberación en
[Acústica de salas](/phonometry/es/buildings/rooms/); la rigidez dinámica
medida aquí alimenta el modelo de suelo flotante de
[Aislamiento acústico](/phonometry/es/buildings/insulation/) a través de
[Predicción del comportamiento de capas
elásticas](/phonometry/es/buildings/design/resilient-layers/);
y los métodos de pavimentos conectan con el interés por el ruido exterior de la
sección de
[Medio ambiente y transporte](/phonometry/es/environment/).

## [Absorbentes](/phonometry/es/materials/absorbers/)

Cuánta energía retira un material del campo, un instrumento o una familia de
modelos por guía.

- [Resumen de Absorbentes](/phonometry/es/materials/absorbers/): la
  cadena de medición de la cámara reverberante al banco de flujo y al tubo de
  impedancia, y los modelos de predicción que los atan.
- [Medida y clasificación de la absorción sonora](/phonometry/es/materials/absorbers/absorption-measurement/):
  la medición ISO 354, la valoración ponderada ISO 11654 con su clase y la
  incertidumbre ISO 12999-2.
- [Resistencia al flujo de aire](/phonometry/es/materials/absorbers/airflow-resistance/):
  los métodos estático y alterno de ISO 9053.
- [Tubo de impedancia](/phonometry/es/materials/absorbers/impedance-tube/): la absorción,
  la impedancia y la pérdida por transmisión ASTM E2611 a incidencia normal,
  más el tubo FDTD virtual.
- [Absorbentes porosos y multicapa](/phonometry/es/materials/absorbers/porous-absorbers/):
  los modelos porosos de Delany-Bazley, Miki y JCA, el modelo
  multicapa por matrices de transferencia con capas perforadas,
  microperforadas y de membrana, y la integral de Paris de incidencia
  aleatoria.
- [Metaabsorbentes](/phonometry/es/materials/absorbers/metamaterial-absorbers/): el
  acoplamiento crítico y el panel ranurado de sonido lento con su
  cálculo de diseño.

## [Difusores y superficies](/phonometry/es/materials/diffusers/)

Adónde envía una superficie lo que refleja, y las superficies que solo pueden
medirse allí donde están.

- [Resumen de Difusores y superficies](/phonometry/es/materials/diffusers/):
  qué hace una superficie con el sonido que devuelve, de los coeficientes a
  los paneles de metamaterial.
- [Difusores y sus coeficientes](/phonometry/es/materials/diffusers/diffusers/): los
  coeficientes de dispersión y difusión de ISO 17497-1/2, el diseño de
  Schroeder y la predicción en campo lejano.
- [Metadifusores](/phonometry/es/materials/diffusers/metadiffusers/): difusores de
  Schroeder en sublongitud de onda profunda a partir de ranuras cargadas
  con resonadores.
- [Resumen de Superficies medidas in situ](/phonometry/es/materials/surfaces/):
  superficies que no se pueden llevar al laboratorio, caracterizadas donde
  están.
- [Absorción in situ de pavimentos de carretera](/phonometry/es/materials/surfaces/road-absorption/):
  la absorción in situ de pavimentos de ISO 13472-1/-2.

## [Capas elásticas](/phonometry/es/materials/resilient/)

La propiedad mecánica en torno a la que se diseña un suelo flotante.

- [Resumen de Capas elásticas](/phonometry/es/materials/resilient/):
  lo que hace una capa elástica bajo un suelo flotante, y la rigidez
  dinámica que lo fija.
- [Rigidez dinámica de materiales resilientes (EN 29052-1)](/phonometry/es/materials/resilient/dynamic-stiffness/):
  el método de resonancia que mide lo que hace una capa elástica bajo un
  suelo flotante, y la rigidez aparente que le pide el capítulo de diseño de
  aislamiento.

## Qué no cubre esta sección

Todo lo de aquí caracteriza un **material o una superficie**, nunca una
construcción. La pérdida por transmisión de una pared, la mejora a ruido de
impactos de un suelo y las trayectorias por flancos de una unión son
[Aislamiento acústico](/phonometry/es/buildings/insulation/) y
[Diseño del aislamiento](/phonometry/es/buildings/design/); esta sección
suministra los coeficientes que ellos consumen. Dentro de las propias
mediciones hay dos fronteras que conviene conocer antes de empezar. Los
métodos in situ de carretera implementan ISO 13472-1:2002 e ISO 13472-2:2010;
**sus revisiones de 2022 y 2025 no están implementadas**. Y la medición de
capas elásticas espera una frecuencia de resonancia ya extrapolada a
amplitud de fuerza nula por el apartado 7 de EN 29052-1, un procedimiento que
no está implementado, y una resistividad al flujo de aire suministrada como
entrada en lugar de medida allí mismo. Nada de esta sección predice un
material a partir de su química o de su fabricación: los modelos van hacia
delante desde parámetros macroscópicos medidos — resistividad al flujo,
porosidad, tortuosidad — hasta una impedancia, y no hay ninguna inversión
que recupere esos parámetros de una curva de impedancia medida.

## Antes y después de estas páginas

Todo coeficiente de estas páginas se deriva de niveles de banda o de una
función de transferencia entre micrófonos, así que el filtrado, la ponderación
y la calibración que los producen están en [Análisis de
señal](/phonometry/es/signals/), y [Construye un
sonómetro](/phonometry/es/signals/sound-level-meter/) recorre esa cadena de
principio a fin en una sola página ejecutable. Las derivaciones están en la
[teoría de materiales y
superficies](/phonometry/es/reference/theory/materials-surfaces/):
las magnitudes de caracterización, la sustracción in situ y los coeficientes
de dispersión y de difusión.

Si has llegado aquí desde una búsqueda y quieres la forma de la biblioteca
entera, [¿Qué necesitas medir?](/phonometry/es/start/tasks/) la indexa por el
trabajo y [Todas las guías](/phonometry/es/start/guides/) lista todas las
páginas con una línea sobre cada una.
