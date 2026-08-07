---
title: "Inicio"
description: "Por dónde empezar con phonometry: el primer análisis de principio a fin, el mapa de todas las guías, por qué existe la biblioteca y quién la mantiene."
---

Cuatro páginas breves, pensadas para leerse una vez antes que nada. Cada una
responde a una pregunta, y están en el orden en que llegan las preguntas.

**¿Puedo instalarla y sacar un número?**
[Primeros pasos](/phonometry/es/start/getting-started/) instala la biblioteca y
ejecuta un primer análisis en tercios de octava, sobre una señal sintética y
después sobre un archivo WAV, y dice qué tiene que cumplir una grabación para
que esos números signifiquen algo físico. Se queda a propósito antes de una
medición calibrada:
[Calibración y dBFS](/phonometry/es/signals/metrology/calibration/) es el paso
siguiente, el que convierte los niveles de banda en pascales, y
[Construir un sonómetro](/phonometry/es/signals/sound-level-meter/) recorre la
cadena entera de principio a fin.

**¿Dónde está lo que vine a buscar?**
[Todas las guías](/phonometry/es/start/guides/) es el mapa: todas las guías de
la biblioteca, agrupadas por el tema al que pertenecen, con una línea sobre
cada una.

**¿Me puedo fiar del número?**
[Por qué phonometry](/phonometry/es/start/why-phonometry/) explica para qué
sirve la biblioteca y cómo se valida contra las normas que implementa, con la
comprobación de ráfagas de tono desarrollada frente a los límites de
aceptación.

**¿Quién responde de ella y cómo la cito?**
[Acerca de](/phonometry/es/start/about/) dice quién la mantiene, cómo citarla y
bajo qué licencia.

El punto de partida que se supone es Python 3.13 o posterior con NumPy y SciPy
funcionando, y la acústica suficiente para saber qué son una banda de tercio de
octava y un nivel de presión acústica. Cualquier símbolo que las guías usen sin
presentarlo está en el [glosario](/phonometry/es/reference/glossary/), con su
unidad, la cláusula que lo define y la guía que lo calcula.
