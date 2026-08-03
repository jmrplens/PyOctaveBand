---
title: "Referencia"
description: "La parte del sitio que no enseña: la teoría que hay detrás de cada módulo, el informe numérico de conformidad que fija cada métrica al apartado de su norma y la bibliografía de todas las fuentes que citan las guías."
---

Las guías muestran cómo medir algo. Esta sección responde a las tres preguntas
que vienen después: **por qué la fórmula es esta**, **si la implementación es
realmente correcta** y **de dónde procede**. Aquí no hay ningún tutorial, así
que nada tiene que leerse en orden: son páginas a las que se llega desde una
guía, desde un informe que se está redactando o desde la pregunta de un
revisor.

Úsala cuando estés defendiendo un resultado y no produciéndolo. Si un colega
pregunta qué apartado de la norma ISO 3382-1 sigue el tiempo de
reverberación, las páginas de teoría lo dicen; si un cliente pide pruebas de
que la librería lo calcula correctamente, el informe de conformidad muestra el
valor esperado de la propia norma junto al calculado; si una revista pide la
fuente de un modelo, la bibliografía tiene el DOI.

Si solo buscas la firma de una función, lo que quieres es la
[referencia de la API](/phonometry/es/reference/api/): se genera a partir de
los docstrings del código y ocupa su propia sección de la barra lateral.

## [Teoría](/phonometry/es/reference/theory/)

Las normas, las matemáticas y las decisiones de diseño detrás de cada módulo,
en seis páginas por dominio: análisis de señal, percepción y audición, salas y
edificación, materiales y superficies, medio ambiente y transporte, y
vibración. Empieza aquí cuando una guía enuncia un resultado y quieres la
deducción, el apartado que implementa o el motivo por el que se eligió una
formulación y no otra. La teoría de los módulos submarinos es la excepción:
está junto a [sus guías](/phonometry/es/underwater/).

## [Informe de conformidad](/phonometry/es/reference/conformance/)

La evidencia numérica. Cada comprobación nombra una norma, un apartado o
tabla, el valor esperado normativo, el valor que calcula la librería, la
desviación y un veredicto de conformidad. Se regenera y se exige en CI en cada
pull request, así que describe el código tal como está ahora y no como se
documentó una vez. Léelo cuando necesites demostrar que un número es
defendible, o para ver exactamente qué partes de una norma están implementadas
y cuáles no.

## [Erratas de las fuentes publicadas](/phonometry/es/reference/errata/)

La otra cara de la misma moneda. Volver a deducir cada fórmula y cada ejemplo
resuelto a partir del documento fuente demuestra de vez en cuando que quien se
equivoca es el *documento*: un ejemplo que contradice su propio articulado,
una constante mal impresa, una referencia cruzada que apunta a la ecuación
equivocada. Cada caso confirmado queda registrado con la edición impresa, la
evidencia, la lectura que implementa la librería y el test que la fija. Léelo
cuando un valor esperado impreso y la librería no coincidan, antes de dar por
hecho que el error es de la librería.

## [Bibliografía](/phonometry/es/reference/bibliography/)

Todos los libros y artículos que citan las guías, en una sola lista agrupada
por dominio, cada entrada con un DOI verificado o un enlace oficial del
editor, media frase sobre qué sustenta y las páginas de guía que la citan.
Sirve a la vez como lista de lectura del campo y como fuente única de verdad
para la comprobación de enlaces.
