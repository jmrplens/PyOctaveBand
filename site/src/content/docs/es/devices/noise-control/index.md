---
title: "Control de ruido"
description: "Control de ruido industrial en el camino: el cálculo completo del ventilador a la sala y la cadena entre recintos frente a un criterio de diseño, más los modelos de elemento a los que llaman: silenciadores reactivos de cuatro polos, atenuación y ruido de flujo en conductos HVAC, y pérdida de inserción de cerramientos de máquina."
---

Un problema de control de ruido es un **presupuesto**, no una elección de
componente. Entre una máquina y quien la oye hay un camino, y cada elemento de
ese camino quita un número conocido de decibelios por banda; la pregunta de
diseño es qué combinación cierra la diferencia entre lo que emite la fuente y lo
que permite el criterio. El ruido de maquinaria se ataca por turnos en la
fuente, el camino y el receptor, y esta sección reúne el camino, y sus dos
mitades: los modelos de elemento, y las dos cadenas compuestas que gastan su
salida frente a un criterio.

Todo lo de aquí es una **predicción** a partir de una geometría declarada y de
unos datos de material declarados. Eso importa cuando hay un catálogo abierto al
lado de la pantalla: la cifra que publica un proveedor para el mismo dispositivo
es una pérdida de inserción *medida*, obtenida en las condiciones de una norma
de medición — la ISO 7235 para un silenciador de conducto sobre un banco de
laboratorio con y sin flujo de aire, que da además el ruido de flujo regenerado
y la pérdida de carga, la ISO 11691 para el método de control sin flujo, la
ISO 11820 para un silenciador in situ, y la ISO 11546-1 y -2 para un
cerramiento en laboratorio y in situ. Una pérdida por transmisión calculada
y una pérdida de inserción de catálogo no son la misma magnitud. Ninguna de las
dos está mal; responden a preguntas distintas, y un diseño que las mezcla sin
decirlo no es defendible.

[Ruido por conductos: del ventilador a la
sala](/phonometry/es/devices/noise-control/duct-path/)
sigue un camino aéreo desde el ventilador, por el tramo de conductos, hasta la
sala: la atenuación en conducto recto, en codos y en derivaciones, la reflexión
del extremo en la unidad terminal, el ruido de flujo regenerado que se vuelve a
sumar, el efecto de sala en el receptor y el resultado contrastado con el
criterio de la sala. Enuncia además el límite que comparte todo modelo de
elemento de esta sección: la frecuencia por encima de la cual entran los modos
de orden superior y deja de valer la hipótesis de onda plana.
[Entre recintos: partición, receptor y
criterio](/phonometry/es/devices/noise-control/room-to-room/)
sigue en cambio el camino aéreo entre recintos: un nivel de recinto emisor
construido a partir de una potencia acústica y de la constante de sala, una
partición con su pérdida por transmisión, un recinto receptor con su absorción,
el espectro recibido y su veredicto — y el problema inverso, la pérdida por
transmisión que necesita una partición o un cerramiento revestido para que
el recinto receptor cumpla su criterio, resuelto hacia atrás.

Las dos páginas de elemento aportan aquello a lo que esas cadenas llaman.
[Silenciadores](/phonometry/es/devices/noise-control/silencers/) cubre los elementos
reactivos de cuatro polos (cámaras de expansión, resonadores de Helmholtz, de
cuarto de onda y de tubo extendido) con su pérdida de transmisión y de
inserción, y la elección entre reflexión y disipación, mientras que
[Control de ruido industrial](/phonometry/es/devices/noise-control/noise-control/) se queda
con la atenuación y el ruido de flujo de los conductos HVAC de una
instalación y con la pérdida de inserción de un cerramiento de máquina.

Si el ruido viaja por un conducto, empieza por
[Ruido por conductos](/phonometry/es/devices/noise-control/duct-path/); si viaja a
través de un muro, empieza por
[Entre recintos](/phonometry/es/devices/noise-control/room-to-room/); abre las
páginas de elemento cuando una cadena pida un número que no tienes.

Los dos extremos del problema se resuelven fuera de esta sección, y un cálculo
de camino al que le falte uno de ellos no tiene veredicto. En el extremo de
**fuente**, la referencia contra la que se juzga una medida correctora es la
emisión de la propia máquina, determinada en las páginas de
[Potencia acústica e intensidad](/phonometry/es/devices/emission/), y reducirla
ahí casi siempre sale más barato que tratar un camino. En el extremo de
**receptor** están los criterios: las familias NC y RC Mark II de
[Criterios de ruido de salas](/phonometry/es/buildings/rooms/room-noise/), más el
límite laboral que corresponda, en [Exposición al ruido en el trabajo
(ISO 9612)](/phonometry/es/perception/hearing/occupational-exposure/).

## Páginas de esta sección

- [Silenciadores](/phonometry/es/devices/noise-control/silencers/): silenciadores reactivos
  por el método de cuatro polos y la elección entre reactivo y disipativo.
- [Ruido por conductos: del ventilador a la sala](/phonometry/es/devices/noise-control/duct-path/):
  el cálculo completo del ventilador a la sala frente a un criterio de ruido de
  fondo, y el corte de modos superiores que limita todo método de onda plana.
- [Entre recintos: partición, receptor y criterio](/phonometry/es/devices/noise-control/room-to-room/):
  la cadena compuesta del recinto emisor al recinto receptor y la pérdida por
  transmisión que necesita una partición o un cerramiento para cumplir un
  criterio de ruido de fondo.
- [Control de ruido industrial: HVAC y cerramientos](/phonometry/es/devices/noise-control/noise-control/):
  atenuación en conductos, ruido de flujo y pérdida de inserción de
  cerramientos de máquina.

## Qué no cubre esta sección

Aquí no hay ninguna medición: todo número se predice a partir de la geometría y
de unos datos declarados, y las normas de medición nombradas arriba se citan
como origen de las cifras de un proveedor, no se implementan. Dentro de las
predicciones, tres límites son estructurales. **Solo se calculan los elementos
de silenciador reactivos**: los silenciadores disipativos de conducto revestido
se comentan para la selección, pero no se modelan a partir de las propiedades
del revestimiento en ningún punto de la biblioteca, y las cifras de codo
revestido y de plenum de la página de HVAC son tablas de instalación
interpoladas y no un modelo de revestimiento. **El flujo medio queda fuera de
las matrices de elemento**: no aparecen la convección, los gradientes de
temperatura ni la impedancia dependiente del flujo de los perforados, así que un
silenciador que transporta un flujo apreciable se predice como si no lo hiciera.
Y `enclosure_insertion_loss` **nunca predice la pérdida por transmisión del
panel**: aportas tú una R medida o venida de otro modelo, y el módulo la combina
con la corrección interior; predecir la propia R es [Diseño del
aislamiento](/phonometry/es/buildings/design/). Por encima de la frecuencia de
corte de los modos de orden superior deja de valer la hipótesis de onda plana en
la que se apoya todo modelo de conducto, cosa que la página de ruido por
conductos enuncia y que ningún método de aquí sortea.
