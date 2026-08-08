---
title: "Absorbentes"
description: "Caracterizar y predecir absorbentes acústicos: la medición en cámara reverberante y la valoración ISO 11654, la resistencia al flujo de aire ISO 9053, el tubo de impedancia ISO 10534 / ASTM E2611, los modelos de predicción porosos y multicapa, y los metaabsorbentes con acoplamiento crítico."
---

Un absorbente puede caracterizarse a tres escalas, y esta subsección las
recorre desde el producto terminado hacia dentro: el producto montado en una
cámara reverberante, la materia prima en un banco de flujo y la muestra
pequeña en un tubo de impedancia — el banco de flujo antes que el tubo porque
la resistividad que mide es el único parámetro del que parte todo modelo
contra el que se ajusta el tubo. Después vienen los modelos de predicción que
atan las tres escalas, y los diseños de metamaterial que las llevan más allá
de las reglas clásicas de espesor.

[Medida y clasificación de la absorción sonora](/phonometry/es/materials/absorbers/absorption-measurement/)
es la escala de producto: la medición en cámara reverberante ISO 354 del
coeficiente de incidencia aleatoria α_s, la valoración ponderada ISO 11654
α_w con su clase por letra que citan las hojas de datos de los absorbentes, y
la incertidumbre de medida ISO 12999-2 de ambos. También responde la pregunta
recurrente de cuándo un número de cámara reverberante y uno de tubo pueden, y
no pueden, compararse.

[Resistencia al flujo de aire](/phonometry/es/materials/absorbers/airflow-resistance/) es
la escala de material: la determinación estática de ISO 9053-1 y alterna de
ISO 9053-2 de la resistencia al flujo, la resistencia específica y la
resistividad σ, el parámetro que gobierna el comportamiento en baja
frecuencia de un absorbente poroso y ancla todos los modelos porosos aguas
abajo.

[Tubo de impedancia](/phonometry/es/materials/absorbers/impedance-tube/) es la escala de
muestra: el factor de reflexión y la impedancia superficial complejos, y la
absorción, a incidencia normal, por los métodos de la razón de onda
estacionaria de ISO 10534-1 y de la función de transferencia de ISO 10534-2,
más la pérdida por transmisión con cuatro micrófonos de ASTM E2611, y el tubo
FDTD virtual que contrasta el solucionador de ondas con las mismas cadenas de
reducción.

[Absorbentes porosos y multicapa](/phonometry/es/materials/absorbers/porous-absorbers/)
cierra el ciclo con la predicción: los modelos de fluido equivalente de
Delany-Bazley, Miki y Johnson-Champoux-Allard convierten la resistividad
medida en impedancia característica y número de onda, y el solucionador multicapa
por matrices de transferencia predice la absorción de una construcción
completa, a cualquier incidencia y en campo difuso, antes de construir nada.

[Metaabsorbentes](/phonometry/es/materials/absorbers/metamaterial-absorbers/)
es donde los
modelos de predicción dejan atrás las reglas clásicas: paneles ranurados de
sonido lento cargados con resonadores de Helmholtz alcanzan la absorción
perfecta en el acoplamiento crítico con paneles de una cuarentava parte de la
longitud de onda de profundidad, con el modelo de matrices de transferencia,
el solucionador de diseño y la comprobación cruzada FDTD de la celda mallada.

## Páginas de esta sección

- [Medida y clasificación de la absorción sonora](/phonometry/es/materials/absorbers/absorption-measurement/):
  la medición en cámara reverberante ISO 354, la valoración ponderada
  ISO 11654 con su clase y la incertidumbre de medida ISO 12999-2.
- [Resistencia al flujo de aire](/phonometry/es/materials/absorbers/airflow-resistance/):
  los métodos estático y alterno de ISO 9053 para la resistencia y la
  resistividad al flujo de aire.
- [Tubo de impedancia](/phonometry/es/materials/absorbers/impedance-tube/): la absorción,
  la impedancia superficial y la pérdida por transmisión ASTM E2611 a
  incidencia normal, más el tubo FDTD virtual.
- [Absorbentes porosos y multicapa](/phonometry/es/materials/absorbers/porous-absorbers/):
  los modelos de fluido equivalente y el solucionador multicapa por matrices de
  transferencia con capas perforadas, microperforadas y de membrana.
- [Metaabsorbentes](/phonometry/es/materials/absorbers/metamaterial-absorbers/): la
  condición de acoplamiento crítico y el panel ranurado de sonido lento con
  su solucionador de diseño.

## Qué no cubre esta sección

Ninguna de estas páginas cualifica un laboratorio. Los requisitos de sala del
anexo A de ISO 354 — el número de posiciones de altavoz y de micrófono, los
elementos difusores — no se comprueban; las funciones convierten un par de
decaimientos ya medido y solo avisan cuando el volumen de la sala o el área de
la muestra se sale de los límites del apartado 6. Se citan dos ediciones que no
están implementadas: el código sigue el método de la función de transferencia
de ISO 10534-2 de 1998/2001, no la edición de 2023, y ASTM E2611-19, no
E2611-24. Los refinamientos del recorrido de la sonda de ISO 10534-1
(extrapolar los mínimos hasta la cara de la muestra, correcciones por el cuerpo
de la sonda) se describen pero no se automatizan. Los modelos de predicción van
en un solo sentido: convierten una resistividad en una impedancia, y no hay
ningún solucionador inverso que recupere los parámetros del material a partir de una
curva medida. Y ninguna norma de medida gobierna los diseños de metamaterial —
un panel construido se verifica en el tubo de impedancia o en la cámara
reverberante como cualquier otro absorbente, y por eso esa página da una
predicción y no una valoración.
