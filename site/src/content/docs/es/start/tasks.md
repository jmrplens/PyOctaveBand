---
title: "¿Qué necesitas medir?"
description: "La puerta de entrada a las guías con forma de tarea: el trabajo a la izquierda, la guía que lo responde a la derecha, con la norma que implementa."
---

El resto de este sitio está organizado por materias, que es lo correcto una vez
sabes qué materia se ocupa de tu problema y no sirve de nada antes. Esta página
va al revés: el trabajo a la izquierda, la guía que lo responde a la derecha, y
la norma que implementa esa guía para que puedas saber antes de hacer clic si
es la que aceptarán tu cliente, quien te revise o el organismo que te regula.

Si tu trabajo no está en la lista, [Todas las
guías](/phonometry/es/start/guides/) es el inventario completo, y el
[glosario](/phonometry/es/reference/glossary/) recorre otra vez el camino
contrario: tienes un símbolo sacado de un informe y quieres la guía que lo
calcula.

## A partir de una grabación

| El trabajo | Dónde se responde |
|---|---|
| Tengo una grabación WAV y necesito niveles ponderados A, $L_\mathrm{Aeq}$ y los niveles percentiles | [Construye un sonómetro](/phonometry/es/signals/sound-level-meter/) — IEC 61672-1 e IEC 61260-1, la cadena entera en una página ejecutable, del tono de calibrador a la comprobación de clase de cada etapa |
| Mis números no están en pascales y no sé qué son | [Calibración y dBFS](/phonometry/es/signals/metrology/calibration/) — el tono de calibrador, la sensibilidad en pascales por unidad digital, la regla de deriva antes/después y la alternativa digital en dBFS |
| Necesito un espectro en lugar de un nivel | [Bancos de filtros](/phonometry/es/signals/filters/filter-banks/) para niveles de banda (IEC 61260-1), [Análisis espectral](/phonometry/es/signals/spectra/spectral-analysis/) para una estimación de densidad |

## Salas y edificación

| El trabajo | Dónde se responde |
|---|---|
| El tiempo de reverberación de una sala en la que puedo entrar | [Medición de la respuesta al impulso](/phonometry/es/buildings/rooms/room-impulse-response/) para la adquisición (barridos de ISO 18233 y la alternativa del ruido interrumpido), y después [Acústica de salas](/phonometry/es/buildings/rooms/room-acoustics/) para $T_{20}$, $T_{30}$, EDT, $C_{50}$, $C_{80}$ y el resto de ISO 3382-1 |
| El tiempo de reverberación de una sala que todavía no existe | [Predicción del tiempo de reverberación](/phonometry/es/buildings/rooms/reverberation-prediction/) — la familia de Sabine, con el apartado que dice qué modelo usar y cuándo fallan todos; [Absorción acústica en recintos](/phonometry/es/buildings/rooms/enclosed-space-absorption/) cuando un informe de diseño tiene que citar EN 12354-6 |
| Si una sala está bastante silenciosa para lo que ocurre dentro | [Criterios de ruido de salas](/phonometry/es/buildings/rooms/room-noise/) — NC, RC Mark II y NR, y el apartado sobre cómo elegir entre ellos |
| El $R'_w$ o el $D_\mathrm{nT,w}$ de un elemento separador que he medido en obra | [Medición del aislamiento en campo](/phonometry/es/buildings/insulation/insulation-field/) (ISO 16283-1/-2) para la medición, y después [Índices globales de aislamiento](/phonometry/es/buildings/insulation/insulation-ratings/) (ISO 717-1/-2) para el número único y los términos de adaptación |
| La prueba de que un muro cumple el CTE DB-HR | [Código Técnico de la Edificación](/phonometry/es/buildings/insulation/spanish-building-code/) — las magnitudes exigidas, contra cuál está escrita cada exigencia y la comprobación de cumplimiento |
| El aislamiento de un edificio que todavía estoy diseñando | [Predicción del aislamiento acústico (EN 12354)](/phonometry/es/buildings/design/insulation-prediction/), y [Predicción detallada por bandas](/phonometry/es/buildings/design/detailed-prediction/) cuando hay que desglosar las trayectorias por flancos |

## Máquinas, productos e instalaciones

| El trabajo | Dónde se responde |
|---|---|
| La potencia acústica de una máquina | [Potencia acústica](/phonometry/es/devices/emission/sound-power/) — empieza aquí sean cuales sean tus instalaciones: su apartado «Elegir un método» encamina hacia el método por presión (ISO 3744/3745/3746), en cámara reverberante (ISO 3741), por intensidad (ISO 9614) o por vibración superficial (ISO/TS 7849) |
| Un valor de emisión declarado para una hoja de características o un expediente CE | [Declarar la emisión sonora](/phonometry/es/devices/emission/sound-power/#declarar-la-emisión-sonora-iso-4871) — ISO 4871, el nivel declarado ponderado A $L_{WAd}$ y su incertidumbre $K_{WA}$ |
| Cuánto van a atenuar un silenciador, un cerramiento o un tramo de conducto | [Silenciadores](/phonometry/es/devices/noise-control/silencers/), [Control de ruido industrial](/phonometry/es/devices/noise-control/noise-control/) para cerramientos y climatización, [Ruido por conductos](/phonometry/es/devices/noise-control/duct-path/) |
| Un altavoz, un micrófono o un amplificador medidos según su norma | [Medidas electroacústicas](/phonometry/es/devices/electroacoustics/electroacoustics/) — IEC 60268-3/-4/-5 |
| La sonoridad de un programa en LUFS | [Sonoridad de programa y pico verdadero](/phonometry/es/devices/broadcast/program-loudness/) — ITU-R BS.1770-5 y EBU R 128 |

## Medio ambiente y transporte

| El trabajo | Dónde se responde |
|---|---|
| El $L_\mathrm{den}$, el $L_\mathrm{night}$ o un nivel de evaluación de un estudio ambiental | [Niveles ambientales](/phonometry/es/environment/assessment/environmental-levels/) — ISO 1996-1/-2, los indicadores y los ajustes que se les suman |
| La potencia de la fuente para un mapa de ruido viario o ferroviario | [Emisión de la fuente de tráfico viario CNOSSOS-EU](/phonometry/es/environment/sources/cnossos-road-emission/) y [emisión de la fuente ferroviaria](/phonometry/es/environment/sources/cnossos-rail-emission/) — el anexo II de la 2002/49/CE, su lado de la fuente |
| Cuánto quitan la distancia, una barrera o la meteorología | [Propagación en exteriores](/phonometry/es/environment/propagation/outdoor-propagation/) (ISO 9613-2), [Efecto suelo y barreras](/phonometry/es/environment/propagation/ground-barriers/), [Refracción atmosférica](/phonometry/es/environment/propagation/atmospheric-refraction/) |
| Si un parque eólico cumple | [Ruido de aerogeneradores](/phonometry/es/environment/sources/wind-turbine-noise/) — IEC 61400-11 y la evaluación de audibilidad tonal |
| El nivel de certificación del sobrevuelo de una aeronave | [Ruido de aeronaves: EPNL](/phonometry/es/aircraft/aircraft-noise/) — Anexo 16 de la OACI, y [Curvas de ruido de aeropuerto](/phonometry/es/aircraft/airport-noise/) para el mapa alrededor del aeropuerto |

## Personas

| El trabajo | Dónde se responde |
|---|---|
| Si un trabajador supera el límite diario de exposición | [Exposición al ruido en el trabajo](/phonometry/es/perception/hearing/occupational-exposure/) — ISO 9612, el $L_\mathrm{EX,8h}$ y su incertidumbre |
| La sonoridad en sonios, o la agudeza, la aspereza y la intensidad de fluctuación | [Sonoridad](/phonometry/es/perception/psychoacoustics/loudness/) (ISO 532-1/-2/-3), y después [Métricas de calidad sonora](/phonometry/es/perception/psychoacoustics/sound-quality/) |
| Si un tono dentro de un ruido es audible, y en qué medida | [Audibilidad de tonos](/phonometry/es/perception/psychoacoustics/tone-audibility/) (ISO/PAS 20065, DIN 45681) y [Tonos prominentes](/phonometry/es/perception/psychoacoustics/tone-prominence/), cuyo apartado sobre qué métrica usar cubre las dos |
| El STI de un sistema de megafonía | [Índice de transmisión del habla](/phonometry/es/perception/speech/speech-transmission/) — IEC 60268-16, directo e indirecto (STIPA) |
| Un registro de vibración de asiento, de suelo o mano-brazo frente a la Directiva | [Vibración en humanos](/phonometry/es/vibration/human/human-vibration/) — ISO 2631-1 e ISO 5349-1, y [Vibración con choques múltiples](/phonometry/es/vibration/human/multiple-shock-vibration/) para ISO 2631-5 |

## Materiales, y bajo el agua

| El trabajo | Dónde se responde |
|---|---|
| El coeficiente de absorción de una muestra | [Medida y clasificación de la absorción sonora](/phonometry/es/materials/absorbers/absorption-measurement/) para la cámara reverberante (ISO 354, ISO 11654), [Tubo de impedancia](/phonometry/es/materials/absorbers/impedance-tube/) para incidencia normal (ISO 10534-2) |
| La resistividad al flujo de aire o la rigidez dinámica de una capa | [Resistencia al flujo de aire](/phonometry/es/materials/absorbers/airflow-resistance/) (ISO 9053-1/-2), [Rigidez dinámica](/phonometry/es/materials/resilient/dynamic-stiffness/) (EN 29052-1) |
| Cuánto dispersa un difusor | [Difusores y dispersión](/phonometry/es/materials/diffusers/diffusers/) — ISO 17497-1/-2, y la diferencia entre los dos coeficientes |
| El ruido radiado por un buque, o la exposición de una campaña de hincado de pilotes | [Acústica submarina](/phonometry/es/underwater/underwater-acoustics/) (ISO 17208, ISO 18406) y después [Exposición a ruido de mamíferos marinos](/phonometry/es/underwater/marine-mammal-exposure/) |
| Hasta dónde llega un sonido en el mar | [Propagación submarina](/phonometry/es/underwater/underwater-propagation/), y [Métodos numéricos de propagación submarina](/phonometry/es/underwater/underwater-solvers/) cuando hace falta un campo dependiente de la distancia |

## Cuando hay más de un método válido

Varias guías llevan dentro la propia decisión, en un apartado escrito para quien
tiene el problema pero todavía no el método. Vale la pena abrirlas antes que la
guía que crees necesitar:

- [Elegir F, S o I](/phonometry/es/signals/levels/time-weighting/#elegir-f-s-o-i) — qué ponderación temporal exponencial pide una medición.
- [¿Qué arquitectura de filtro debería elegir?](/phonometry/es/signals/filters/filter-gallery/#qué-arquitectura-de-filtro-debería-elegir) — Butterworth frente a las cuatro alternativas, y a qué renuncia cada una.
- [Elegir un criterio: NC, RC Mark II o NR](/phonometry/es/buildings/rooms/room-noise/#3-elegir-un-criterio-nc-rc-mark-ii-o-nr).
- [Elegir un modelo, y cuándo fallan todos](/phonometry/es/buildings/rooms/reverberation-prediction/#4-elegir-un-modelo-y-cuándo-fallan-todos) — Sabine, Eyring, Millington, Fitzroy, Arau.
- [Elegir un método](/phonometry/es/devices/emission/sound-power/#elegir-un-método) — las cuatro rutas hacia la potencia acústica y cuál permite cada instalación.
- [Elegir un modelo de sonoridad](/phonometry/es/perception/psychoacoustics/advanced-loudness/#elegir-un-modelo-de-sonoridad) — Zwicker, Moore-Glasberg y Sottek.
- [Qué medida usar y cuándo](/phonometry/es/perception/speech/objective-intelligibility/#4-qué-medida-usar-y-cuándo) — el STI frente a las medidas intrusivas y no intrusivas.
- [Qué métrica de tonalidad usar y cuándo](/phonometry/es/perception/psychoacoustics/tone-prominence/#3-qué-métrica-de-tonalidad-usar-y-cuándo).
- [¿Directo o indirecto?](/phonometry/es/perception/speech/speech-transmission/#directo-o-indirecto-cómo-elegir) — el STI de matriz completa frente a STIPA.
- [Elegir modelo](/phonometry/es/underwater/underwater-solvers/#6-elegir-modelo) — modos normales, rayos, haces gaussianos y ecuación parabólica.

## Antes de medir nada

Dos costumbres salvan más mediciones que ninguna de las páginas de arriba. Graba
el tono de calibrador por la misma cadena, antes y después, y toma la diferencia
como tu cota de deriva: [Primeros
pasos](/phonometry/es/start/getting-started/) enseña por qué un nivel sin
calibrar no es aproximadamente correcto, sino arbitrario. Y lee el bloque «Qué
cubre esta guía» del final de la guía en la que aterrices: dice qué apartados,
anexos y métodos están implementados y cuáles no, que es lo primero por lo que
va a preguntar quien revise.
