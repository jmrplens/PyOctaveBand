---
title: "Inicio"
description: "Por dónde empezar con phonometry: el primer análisis de principio a fin, el índice por tareas, el mapa de todas las guías, por qué existe la biblioteca y quién la mantiene."
---

phonometry calcula magnitudes acústicas a partir del texto de las normas que
las definen — ISO, IEC, ANSI y ASTM, el anexo CNOSSOS-EU de la Directiva
2002/49/CE y los documentos de aeronaves de OACI y ECAC — y cada métrica cita
el apartado que implementa. Qué te aporta eso, y cómo se comprueba, está en
[Por qué phonometry](/phonometry/es/start/why-phonometry/).

Cinco páginas breves, pensadas para leerse una vez antes que nada. Cada una
responde a una pregunta, y están en el orden en que llegan las preguntas.

**¿Puedo instalarla y sacar un número?**
[Primeros pasos](/phonometry/es/start/getting-started/) instala la biblioteca,
ejecuta un primer análisis en tercios de octava sobre una señal sintética,
después ancla ese análisis a un tono de calibrador para que los niveles sean
decibelios re 20 µPa y no decibelios re nada, los reduce a un único nivel
ponderado A y dice qué tiene que cumplir una grabación para que todo eso se
sostenga. Se queda a una etapa de un sonómetro: las balísticas Fast y Slow,
$L_{AE}$, $L_{Cpeak}$ y los niveles percentiles están en [Construye un
sonómetro](/phonometry/es/signals/sound-level-meter/), que recorre la cadena
entera de principio a fin en una sola página, y [Calibración y
dBFS](/phonometry/es/signals/metrology/calibration/) es la guía de fondo del
paso que más importa.

**Tengo un trabajo, no un tema. ¿Cuál es mi página?**
[¿Qué necesitas medir?](/phonometry/es/start/tasks/) indexa la biblioteca por
la tarea en lugar de por el tema: medir un tiempo de reverberación, comprobar
un muro frente a un código de edificación, valorar la potencia acústica de una
máquina, decidir si un trabajador supera el límite de exposición.

**¿Dónde está lo que vine a buscar?**
[Todas las guías](/phonometry/es/start/guides/) es el mapa: todas las guías de
la biblioteca, agrupadas por el tema al que pertenecen, con una línea sobre
cada una.

**¿Me puedo fiar del número?**
[Por qué phonometry](/phonometry/es/start/why-phonometry/) explica para qué
sirve la biblioteca y cómo se valida contra las normas que implementa, con la
comprobación de ráfagas tonales desarrollada frente a los límites de
aceptación.

**¿Quién responde de ella y cómo la cito?**
[Acerca de](/phonometry/es/start/about/) dice quién la mantiene, cómo citarla y
bajo qué licencia.

## Dos cosas que conviene resolver antes de que ninguna guía funcione

**En qué marco de referencia está un nivel.** Un nivel es o bien *físico*, en
dB SPL, anclado por un tono de calibrador grabado o por una sensibilidad de
micrófono conocida, o bien *digital*, en dBFS respecto al fondo de escala. Los
dos no son intercambiables, y casi todas las guías dan por supuesto el primero:
una función de nivel a la que se le pasan muestras en bruto de la tarjeta de
sonido devuelve un número cuya referencia es arbitraria, y eso tiene exactamente
el mismo aspecto que una respuesta válida. [Calibración y
dBFS](/phonometry/es/signals/metrology/calibration/) lo resuelve.

**Que casi todo lo que viene después consume bandas.** Por debajo de la señal
en bruto hay una única descomposición: bandas de fracción de octava cuyos bordes
a −3 dB caen en las frecuencias nominales de ANSI S1.11 / IEC 61260-1. Un
modelo de sonoridad, un parámetro de sala y una valoración ambiental parten
todos de ahí, y por eso la página del filtrado por bandas es el único requisito
previo que aparece en todas partes: [Bancos de
filtros](/phonometry/es/signals/filters/filter-banks/).

## Qué aspecto tiene una guía

Vale la pena saberlo antes de abrir ninguna, porque es lo que permite decidir en
treinta segundos si una página responde a tu pregunta. Toda guía abre con la
norma que implementa, las magnitudes que esa norma define y las hipótesis que
supone la implementación; después viene el código ejecutable y la figura que
dibuja; y cierra con un bloque «Qué cubre esta guía» que dice sin rodeos qué
apartados, anexos y métodos están implementados y cuáles no. Esto último es la
parte por la que pregunta quien revisa, y está escrita a propósito de la forma
más franca.

## Si ya sabes lo que necesitas

- Una primera medición llevada de principio a fin: [Construye un sonómetro](/phonometry/es/signals/sound-level-meter/).
- El inventario completo, por temas: [Todas las guías](/phonometry/es/start/guides/).
- Un símbolo que tienes pero no sabes nombrar: el [glosario](/phonometry/es/reference/glossary/), con su unidad, el apartado que lo define y la guía que lo calcula.
- Pruebas de que un número es defendible: el [informe de conformidad](/phonometry/es/reference/conformance/), que imprime el valor esperado de cada norma junto al calculado.
- Un valor esperado impreso que no concuerda con la biblioteca: el [registro de erratas](/phonometry/es/reference/errata/), que dice cuál de los dos está mal y por qué.

El punto de partida que se supone es Python 3.13 o posterior con NumPy y SciPy
funcionando, y la acústica suficiente para saber qué son una banda de tercio de
octava y un nivel de presión acústica.

## Qué no es Inicio

Esto no es una serie de tutoriales, y no es la API. Las firmas de las funciones
y los tipos de los argumentos están en la referencia de la API generada. Las
deducciones, el informe numérico de conformidad, el registro de erratas de los
defectos encontrados en las propias normas publicadas, el glosario de símbolos
y la bibliografía están todos en [Referencia](/phonometry/es/reference/). Y la
acústica en sí se da por sabida en lugar de enseñarse: las guías explican el
método que prescribe una norma y por qué está escrito así, no qué es un
decibelio.
