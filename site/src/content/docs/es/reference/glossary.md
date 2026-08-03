---
title: "Glosario"
description: "Todas las magnitudes acústicas que calculan las guías, con su símbolo, una definición de una frase, su unidad, la norma y el apartado que la define y la guía que la implementa."
head:
  - tag: script
    attrs:
      type: application/ld+json
    content: |
      {
        "@context": "https://schema.org",
        "@type": "DefinedTermSet",
        "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary",
        "name": "Glosario acústico de phonometry",
        "description": "Magnitudes acústicas que calcula la biblioteca phonometry, cada una con su símbolo, unidad, norma y apartado que la define y la guía que la implementa.",
        "url": "https://jmrplens.github.io/phonometry/es/reference/glossary/",
        "inLanguage": "es",
        "hasDefinedTerm": [
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-1",
            "name": "Lp",
            "termCode": "Lp",
            "description": "Nivel de presión acústica: veinte veces el logaritmo decimal de la presión acústica eficaz dividida por la presión de referencia. Unidad: dB re 20 µPa. Definida en: IEC 61672-1:2013.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Calibración",
              "url": "https://jmrplens.github.io/phonometry/es/signal/metrology/calibration/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-2",
            "name": "Leq",
            "termCode": "Leq",
            "description": "Nivel de presión acústica continuo equivalente: el nivel del sonido estacionario que transporta la misma presión cuadrática media durante el intervalo. Unidad: dB. Definida en: IEC 61672-1:2013.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Niveles",
              "url": "https://jmrplens.github.io/phonometry/es/signal/levels/levels/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-3",
            "name": "LAeq",
            "termCode": "LAeq",
            "description": "La misma integral aplicada a la señal ponderada A, el descriptor por defecto del ruido ambiental y laboral. Unidad: dB. Definida en: IEC 61672-1:2013.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Niveles",
              "url": "https://jmrplens.github.io/phonometry/es/signal/levels/levels/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-4",
            "name": "LAE, SEL",
            "termCode": "LAE, SEL",
            "description": "Nivel de exposición sonora: toda la energía ponderada A de un suceso único normalizada a un segundo. Unidad: dB. Definida en: IEC 61672-1:2013, Ecuación 8 (Tabla 4).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Niveles",
              "url": "https://jmrplens.github.io/phonometry/es/signal/levels/levels/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-5",
            "name": "LCpeak",
            "termCode": "LCpeak",
            "description": "Nivel de pico ponderado C: el máximo absoluto de la presión ponderada C, no un máximo con ponderación temporal. Unidad: dB. Definida en: IEC 61672-1:2013, apartado 5.13.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Niveles",
              "url": "https://jmrplens.github.io/phonometry/es/signal/levels/levels/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-6",
            "name": "LN (L10, L50, L90)",
            "termCode": "LN (L10, L50, L90)",
            "description": "Nivel percentil: el nivel superado el N % del tiempo de medida, leído en la distribución del nivel con ponderación temporal. Unidad: dB. Definida en: ISO 1996-2:2017 (el Anexo I usa L90 como nivel residual).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Niveles",
              "url": "https://jmrplens.github.io/phonometry/es/signal/levels/levels/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-7",
            "name": "LW, SWL",
            "termCode": "LW, SWL",
            "description": "Nivel de potencia acústica: la potencia que radia una fuente, referida a 1 pW. Unidad: dB re 1 pW. Definida en: ISO 3745:2012, apartado 8.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Potencia acústica",
              "url": "https://jmrplens.github.io/phonometry/es/devices/emission/sound-power/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-8",
            "name": "LI",
            "termCode": "LI",
            "description": "Nivel de intensidad acústica: el módulo del vector intensidad referido a 1 pW/m², con el sentido del flujo indicado aparte mediante un signo. Unidad: dB re 1 pW/m². Definida en: IEC 61043:1993.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Intensidad",
              "url": "https://jmrplens.github.io/phonometry/es/devices/emission/intensity/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-9",
            "name": "Lp − LI",
            "termCode": "Lp − LI",
            "description": "Índice presión-intensidad: la diferencia entre el nivel de presión y el de intensidad en una posición, el indicador de campo que cualifica una medida de intensidad. Unidad: dB. Definida en: ISO 9614-1:1993, Ecuación (A.3).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Intensidad",
              "url": "https://jmrplens.github.io/phonometry/es/devices/emission/intensity/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-10",
            "name": "Lden",
            "termCode": "Lden",
            "description": "Nivel día-tarde-noche: la media energética de los tres periodos con 5 dB añadidos a la tarde y 10 dB a la noche. Unidad: dB. Definida en: ISO 1996-1:2016, 3.6.4.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Niveles ambientales",
              "url": "https://jmrplens.github.io/phonometry/es/environment/environmental-levels/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-11",
            "name": "Ldn",
            "termCode": "Ldn",
            "description": "Nivel día-noche: la misma construcción solo con la penalización de 10 dB nocturna. Unidad: dB. Definida en: ISO 1996-1:2016, 3.6.5.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Niveles ambientales",
              "url": "https://jmrplens.github.io/phonometry/es/environment/environmental-levels/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-12",
            "name": "Lr",
            "termCode": "Lr",
            "description": "Nivel de valoración: el nivel compuesto de la jornada completa tras los ajustes por carácter de la fuente y por franja horaria. Unidad: dB. Definida en: ISO 1996-1:2016, apartado 6.5 (Fórmulas 5 y 6).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Niveles ambientales",
              "url": "https://jmrplens.github.io/phonometry/es/environment/environmental-levels/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-13",
            "name": "LAr,T",
            "termCode": "LAr,T",
            "description": "Nivel de valoración de una fuente impulsiva en un intervalo de referencia: el LAeq más el ajuste graduado por impulsos. Unidad: dB. Definida en: NT ACOU 112:2002, apartado 8.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Prominencia de impulsos",
              "url": "https://jmrplens.github.io/phonometry/es/environment/assessment/impulsive-sound/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-14",
            "name": "KI",
            "termCode": "KI",
            "description": "Ajuste por impulsos que se suma al LAeq, graduado según la prominencia prevista de los impulsos. Unidad: dB. Definida en: NT ACOU 112:2002, apartado 8.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Prominencia de impulsos",
              "url": "https://jmrplens.github.io/phonometry/es/environment/assessment/impulsive-sound/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-15",
            "name": "E",
            "termCode": "E",
            "description": "Exposición sonora: la integral temporal de la presión acústica ponderada A al cuadrado durante el periodo de exposición. Unidad: Pa²h. Definida en: IEC 61252:1993, 3.1.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Niveles",
              "url": "https://jmrplens.github.io/phonometry/es/signal/levels/levels/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-16",
            "name": "LEX,8h, LEP,d",
            "termCode": "LEX,8h, LEP,d",
            "description": "Nivel de exposición diaria al ruido: el nivel estacionario que, mantenido durante una jornada nominal de 8 h, acumula la misma exposición sonora ponderada A que la medida. Unidad: dB. Definida en: IEC 61252:1993, 3.3.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Exposición laboral",
              "url": "https://jmrplens.github.io/phonometry/es/perception/hearing/occupational-exposure/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-17",
            "name": "Lp,A,eqT",
            "termCode": "Lp,A,eqT",
            "description": "Nivel continuo equivalente ponderado A de una tarea, de una muestra del puesto o de una jornada completa, el ladrillo con el que se construye el LEX,8h. Unidad: dB. Definida en: ISO 9612:2009, apartados 9 a 11.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Exposición laboral",
              "url": "https://jmrplens.github.io/phonometry/es/perception/hearing/occupational-exposure/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-18",
            "name": "NIPTS",
            "termCode": "NIPTS",
            "description": "Desplazamiento permanente del umbral inducido por ruido: la pérdida auditiva mediana atribuible a un nivel, una duración y una frecuencia audiométrica dados. Unidad: dB. Definida en: ISO 1999:2013.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Pérdida auditiva por ruido",
              "url": "https://jmrplens.github.io/phonometry/es/perception/hearing/noise-induced-hearing-loss/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-19",
            "name": "HTLAN",
            "termCode": "HTLAN",
            "description": "Nivel de umbral auditivo asociado a la edad y al ruido: el NIPTS combinado con la componente de la edad. Unidad: dB. Definida en: ISO 1999:2013.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Pérdida auditiva por ruido",
              "url": "https://jmrplens.github.io/phonometry/es/perception/hearing/noise-induced-hearing-loss/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-20",
            "name": "A, C, Z",
            "termCode": "A, C, Z",
            "description": "Las ponderaciones frecuenciales normativas: las curvas de respuesta del oído que se aplican antes de integrar, siendo Z la referencia plana. Unidad: dB. Definida en: IEC 61672-1:2013, Anexo E (límites de aceptación en la Tabla 3).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Ponderación frecuencial",
              "url": "https://jmrplens.github.io/phonometry/es/signal/levels/weighting/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-21",
            "name": "G",
            "termCode": "G",
            "description": "Ponderación para infrasonido, definida por sus polos y ceros en el intervalo de 0,25 Hz a 315 Hz. Unidad: dB. Definida en: ISO 7196:1995, Tabla 1 (respuestas nominales en la Tabla 2).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Ponderaciones especiales",
              "url": "https://jmrplens.github.io/phonometry/es/signal/levels/special-weightings/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-22",
            "name": "B",
            "termCode": "B",
            "description": "Ponderación histórica para niveles medios, retirada de la norma vigente de sonómetros. Unidad: dB. Definida en: ANSI S1.4-1983, Apéndice C (Fórmula C2).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Ponderaciones especiales",
              "url": "https://jmrplens.github.io/phonometry/es/signal/levels/special-weightings/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-23",
            "name": "D",
            "termCode": "D",
            "description": "Ponderación histórica para ruido de aeronaves, derivada de la curva de ruidosidad percibida de 40 noys. Unidad: dB. Definida en: IEC 537:1976 (retirada).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Ponderaciones especiales",
              "url": "https://jmrplens.github.io/phonometry/es/signal/levels/special-weightings/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-24",
            "name": "AU",
            "termCode": "AU",
            "description": "Ponderación para el sonido audible medido en presencia de ultrasonido. Unidad: dB. Definida en: IEC 61012:1990, apartado 2.2 (Tablas 1 y 2).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Ponderaciones especiales",
              "url": "https://jmrplens.github.io/phonometry/es/signal/levels/special-weightings/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-25",
            "name": "F, S, I",
            "termCode": "F, S, I",
            "description": "Ponderaciones temporales exponenciales Fast, Slow e Impulse: las balísticas del detector que producen el nivel mostrado. Unidad: s (constante de tiempo). Definida en: IEC 61672-1:2013.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Ponderación temporal",
              "url": "https://jmrplens.github.io/phonometry/es/signal/levels/time-weighting/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-26",
            "name": "T20",
            "termCode": "T20",
            "description": "Tiempo de reverberación extrapolado a una caída de 60 dB desde un ajuste por mínimos cuadrados entre −5 dB y −25 dB de la curva de Schroeder. Unidad: s. Definida en: ISO 3382-2:2008, apartado 6 y Anexo C.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Acústica de salas",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/rooms/room-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-27",
            "name": "T30",
            "termCode": "T30",
            "description": "La misma extrapolación desde un ajuste entre −5 dB y −35 dB, la opción habitual cuando el margen de caída lo permite. Unidad: s. Definida en: ISO 3382-2:2008, apartado 6 y Anexo C.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Acústica de salas",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/rooms/room-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-28",
            "name": "T60, RT",
            "termCode": "T60, RT",
            "description": "El tiempo de reverberación propiamente dicho: el que tarda la energía sonora en caer 60 dB. En la práctica se mide como T20 o T30. Unidad: s. Definida en: ISO 3382-1:2009.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Acústica de salas",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/rooms/room-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-29",
            "name": "EDT",
            "termCode": "EDT",
            "description": "Tiempo de caída inicial: la misma pendiente tomada sobre los primeros 10 dB de caída, que sigue la reverberancia percibida y no la cola. Unidad: s. Definida en: ISO 3382-1:2009 (diferencia apenas perceptible en la Tabla A.1).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Acústica de salas",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/rooms/room-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-30",
            "name": "C50",
            "termCode": "C50",
            "description": "Claridad para la palabra: la relación energética entre los primeros 50 ms de la respuesta al impulso y todo lo que viene después. Unidad: dB. Definida en: ISO 3382-1:2009.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Acústica de salas",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/rooms/room-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-31",
            "name": "C80",
            "termCode": "C80",
            "description": "Claridad para la música: la misma relación con la frontera en 80 ms. Unidad: dB. Definida en: ISO 3382-1:2009 (diferencia apenas perceptible en la Tabla A.1).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Acústica de salas",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/rooms/room-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-32",
            "name": "D50",
            "termCode": "D50",
            "description": "Definición, o Deutlichkeit: la fracción de la energía total que llega en los primeros 50 ms. Unidad: adimensional. Definida en: ISO 3382-1:2009 (diferencia apenas perceptible en la Tabla A.1).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Acústica de salas",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/rooms/room-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-33",
            "name": "Ts",
            "termCode": "Ts",
            "description": "Tiempo central: el centro de gravedad temporal de la respuesta al impulso al cuadrado, una alternativa a los índices de claridad sin frontera arbitraria. Unidad: s. Definida en: ISO 3382-1:2009, Ecuación (A.13).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Acústica de salas",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/rooms/room-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-34",
            "name": "A",
            "termCode": "A",
            "description": "Área de absorción acústica equivalente de un recinto: el área de una superficie perfectamente absorbente que daría el mismo tiempo de reverberación. Unidad: m². Definida en: ISO 354:2003, Ecuaciones (5) y (7).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Materiales acústicos",
              "url": "https://jmrplens.github.io/phonometry/es/materials/absorbers/absorption-measurement/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-35",
            "name": "NC",
            "termCode": "NC",
            "description": "Índice de ruido de sala de un espectro de fondo: el nivel de interferencia con la palabra elige la curva, y el método de tangencia valora el espectro cuando alguna banda la supera. Unidad: dB (índice). Definida en: ANSI/ASA S12.2-2019, 5.2.2 y 5.2.3 (curvas en la Tabla 1).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Ruido de salas",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/rooms/room-noise/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-36",
            "name": "SIL",
            "termCode": "SIL",
            "description": "Nivel de interferencia con la palabra: la media de los niveles en las bandas de octava de 500, 1000, 2000 y 4000 Hz. Unidad: dB. Definida en: ANSI/ASA S12.2-2019, apartado 3.2.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Ruido de salas",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/rooms/room-noise/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-37",
            "name": "RC",
            "termCode": "RC",
            "description": "Índice RC Mark II: la media de los niveles de 500, 1000 y 2000 Hz, con una etiqueta espectral de retumbo, siseo o neutro. Unidad: dB (índice). Definida en: ANSI/ASA S12.2-2019, Anexo D (apartados D.3 y D.4).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Ruido de salas",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/rooms/room-noise/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-38",
            "name": "NR",
            "termCode": "NR",
            "description": "Noise Rating, la familia de curvas europea equivalente a NC. Se comenta a efectos de comparación y no se implementa deliberadamente. Unidad: dB (índice). Definida en: Kosten y van Os (1962); sin norma aplicable.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Ruido de salas",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/rooms/room-noise/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-39",
            "name": "m(F)",
            "termCode": "m(F)",
            "description": "Función de transferencia de modulación: la fracción de la profundidad de modulación de la envolvente del habla, a la frecuencia de modulación F, que sobrevive al camino de transmisión. Unidad: adimensional. Definida en: IEC 60268-16:2020.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Transmisión del habla",
              "url": "https://jmrplens.github.io/phonometry/es/perception/speech/speech-transmission/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-40",
            "name": "STI",
            "termCode": "STI",
            "description": "Índice de transmisión del habla: la matriz de transferencia de modulación convertida en relaciones señal-ruido efectivas y ponderada en un único valor entre 0 y 1. Unidad: adimensional. Definida en: IEC 60268-16:2020, A.5.2 a A.5.6.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Transmisión del habla",
              "url": "https://jmrplens.github.io/phonometry/es/perception/speech/speech-transmission/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-41",
            "name": "STIPA",
            "termCode": "STIPA",
            "description": "La medida directa del STI, reproduciendo una señal de prueba normalizada con dos modulaciones por banda a través de la cadena real. Unidad: adimensional. Definida en: IEC 60268-16:2020, apartado 6.3 y Tabla 3 (método directo, Anexo B).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Transmisión del habla",
              "url": "https://jmrplens.github.io/phonometry/es/perception/speech/speech-transmission/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-42",
            "name": "SII",
            "termCode": "SII",
            "description": "Índice de inteligibilidad del habla: la audibilidad del espectro de habla frente al ruido y al umbral del oyente, ponderada por la función de importancia de cada banda. Unidad: adimensional. Definida en: ANSI S3.5-1997, apartado 6 (procedimiento en el apartado 5, función de importancia en la Tabla 3).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Inteligibilidad del habla",
              "url": "https://jmrplens.github.io/phonometry/es/perception/speech/speech-intelligibility/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-43",
            "name": "STOI",
            "termCode": "STOI",
            "description": "Inteligibilidad objetiva de corta duración: la correlación recortada de las envolventes por banda entre el habla limpia y la degradada. Unidad: adimensional. Definida en: Taal et al. (2011), Ecuaciones 5 y 6; sin norma aplicable.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Inteligibilidad objetiva",
              "url": "https://jmrplens.github.io/phonometry/es/perception/speech/objective-intelligibility/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-44",
            "name": "ESTOI",
            "termCode": "ESTOI",
            "description": "La versión extendida, normalizada por filas y por columnas para que siga a los enmascarantes modulados. Unidad: adimensional. Definida en: Jensen y Taal (2016), Ecuación 8; sin norma aplicable.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Inteligibilidad objetiva",
              "url": "https://jmrplens.github.io/phonometry/es/perception/speech/objective-intelligibility/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-45",
            "name": "D",
            "termCode": "D",
            "description": "Diferencia de niveles: el nivel promediado energéticamente en el recinto emisor menos el del receptor, sin normalizar. Unidad: dB. Definida en: ISO 16283-1:2014, 3.12 a 3.15.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Aislamiento en campo",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/insulation/insulation-field/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-46",
            "name": "DnT",
            "termCode": "DnT",
            "description": "Diferencia de niveles estandarizada: la diferencia de niveles referida a un tiempo de reverberación de referencia, 0,5 s en viviendas. Unidad: dB. Definida en: ISO 16283-1:2014, 3.12 a 3.15.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Aislamiento en campo",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/insulation/insulation-field/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-47",
            "name": "Dn",
            "termCode": "Dn",
            "description": "Diferencia de niveles normalizada: la diferencia de niveles referida a un área de absorción de referencia de 10 m². Unidad: dB. Definida en: ISO 10052:2021.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Método de control",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/insulation/insulation-survey/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-48",
            "name": "Dn,e",
            "termCode": "Dn,e",
            "description": "Diferencia de niveles normalizada de elemento, para un elemento pequeño o una vía de aire, referida a un área de referencia de 10 m². Unidad: dB. Definida en: EN 12354-3:2000.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Predicción del aislamiento",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/design/insulation-prediction/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-49",
            "name": "R",
            "termCode": "R",
            "description": "Índice de reducción acústica: la diferencia de niveles corregida por el área del cerramiento partido por el área de absorción del recinto receptor, medida en laboratorio con los flancos suprimidos. Unidad: dB. Definida en: ISO 10140-2:2010.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Aislamiento en laboratorio",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/insulation/insulation-lab/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-50",
            "name": "R′",
            "termCode": "R′",
            "description": "Índice de reducción acústica aparente: la misma construcción medida en el edificio, así que incluye todas las vías de flanco. La prima es la marca que distingue el laboratorio del campo. Unidad: dB. Definida en: ISO 16283-1:2014, 3.12 a 3.15.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Aislamiento en campo",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/insulation/insulation-field/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-51",
            "name": "TL",
            "termCode": "TL",
            "description": "Pérdidas por transmisión: el aislamiento a ruido aéreo de un panel predicho a partir de sus propiedades físicas, la misma magnitud que R en un contexto de predicción. Unidad: dB. Definida en: Bies, Hansen y Howard (2017), Sección 7.2; sin norma aplicable.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Aislamiento de paneles",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/design/panel-sound-insulation/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-52",
            "name": "Rw, R′w, DnT,w",
            "termCode": "Rw, R′w, DnT,w",
            "description": "Los índices globales ponderados: una curva de referencia fija se desplaza hacia el espectro medido hasta que las desviaciones desfavorables alcanzan su suma admisible, y la curva desplazada se lee en 500 Hz. Unidad: dB. Definida en: ISO 717-1:2020.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Índices de aislamiento",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/insulation/insulation-ratings/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-53",
            "name": "Dn,e,w",
            "termCode": "Dn,e,w",
            "description": "El mismo índice global de curva de referencia aplicado a la diferencia de niveles normalizada de elemento. Unidad: dB. Definida en: ISO 717-1:2020.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Predicción del aislamiento",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/design/insulation-prediction/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-54",
            "name": "C, Ctr",
            "termCode": "C, Ctr",
            "description": "Términos de adaptación espectral: las correcciones que vuelven a valorar la curva medida frente a ruido rosa ponderado A (C) y frente a tráfico rodado urbano ponderado A (Ctr). Unidad: dB. Definida en: ISO 717-1:2020, Anexo A.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Índices de aislamiento",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/insulation/insulation-ratings/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-55",
            "name": "Ln",
            "termCode": "Ln",
            "description": "Nivel de presión acústica de impactos normalizado: el nivel en el recinto receptor bajo la máquina de impactos normalizada, referido a un área de absorción de 10 m². Unidad: dB. Definida en: ISO 10140-3:2010.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Aislamiento en laboratorio",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/insulation/insulation-lab/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-56",
            "name": "L′nT",
            "termCode": "L′nT",
            "description": "Nivel de presión acústica de impactos estandarizado, referido a un tiempo de reverberación de referencia. Atención al signo: más reverberación lo baja, al revés que el DnT. Unidad: dB. Definida en: ISO 16283-2:2015.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Aislamiento en campo",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/insulation/insulation-field/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-57",
            "name": "Ln,w, L′nT,w",
            "termCode": "Ln,w, L′nT,w",
            "description": "Los índices globales de impactos. La curva de referencia se desplaza igual, pero ahora una desviación desfavorable es aquella en la que la medida supera a la referencia. Unidad: dB. Definida en: ISO 717-2:2020.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Índices de aislamiento",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/insulation/insulation-ratings/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-58",
            "name": "CI",
            "termCode": "CI",
            "description": "Término de adaptación espectral de impactos, a partir de la suma energética entre 100 Hz y 2500 Hz. El término de rango ampliado CI,50-2500 lo extiende hasta 50 Hz. Unidad: dB. Definida en: ISO 717-2:2020 (rango ampliado en la NOTA de A.2.1).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Índices de aislamiento",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/insulation/insulation-ratings/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-59",
            "name": "ΔLw",
            "termCode": "ΔLw",
            "description": "Reducción ponderada del nivel de presión acústica de impactos que aporta un revestimiento de suelo, medida como la mejora sobre el forjado desnudo de referencia. Unidad: dB. Definida en: ISO 717-2:2020 (medición en ISO 16251-1:2014, Fórmulas (3) y (4)).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Mejora de revestimientos",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/design/impact-improvement/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-60",
            "name": "ΔRw",
            "termCode": "ΔRw",
            "description": "Mejora ponderada del aislamiento a ruido aéreo que aporta un trasdosado o una capa adicional, que se suma al índice del elemento en la predicción. Unidad: dB. Definida en: EN 12354-1:2000, Fórmulas 27 y 28a.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Predicción del aislamiento",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/design/insulation-prediction/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-61",
            "name": "Kij",
            "termCode": "Kij",
            "description": "Índice de reducción vibracional de una unión: la diferencia de niveles de velocidad promediada en ambos sentidos, corregida por la longitud de la unión y las longitudes de absorción equivalentes. Unidad: dB. Definida en: ISO 10848-1:2006, Fórmula (13).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Flancos en laboratorio",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/insulation/flanking-lab/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-62",
            "name": "fc",
            "termCode": "fc",
            "description": "Frecuencia crítica: aquella en la que la longitud de onda de flexión del panel iguala a la del aire, donde aparece la caída por coincidencia. Unidad: Hz. Definida en: Bies, Hansen y Howard (2017), Ecuación 7.3; sin norma aplicable.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Aislamiento de paneles",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/design/panel-sound-insulation/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-63",
            "name": "σ",
            "termCode": "σ",
            "description": "Eficiencia de radiación de una placa: la potencia aérea radiada por unidad de velocidad cuadrática media de la superficie, normalizada por el valor de onda plana. Unidad: adimensional. Definida en: Hopkins (2007), Ecuaciones 2.227 a 2.230; sin norma aplicable.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Aislamiento de paneles",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/design/panel-sound-insulation/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-64",
            "name": "α",
            "termCode": "α",
            "description": "Coeficiente de absorción acústica a incidencia normal: la fracción de energía incidente que la superficie no devuelve, obtenida en el tubo de impedancia a partir del factor de reflexión. Unidad: adimensional. Definida en: ISO 10534-2:1998, Ecuaciones (17) a (19).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Materiales acústicos",
              "url": "https://jmrplens.github.io/phonometry/es/materials/absorbers/impedance-tube/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-65",
            "name": "αs",
            "termCode": "αs",
            "description": "Coeficiente de absorción acústica a incidencia aleatoria medido en cámara reverberante, a partir del cambio del área de absorción equivalente con y sin la muestra. Unidad: adimensional. Definida en: ISO 354:2003, Ecuaciones (8) y (9).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Materiales acústicos",
              "url": "https://jmrplens.github.io/phonometry/es/materials/absorbers/absorption-measurement/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-66",
            "name": "αp",
            "termCode": "αp",
            "description": "Coeficiente de absorción acústica práctico: los datos en tercios de octava agrupados en bandas de octava y redondeados a pasos de 0,05. Unidad: adimensional. Definida en: ISO 11654:1997, apartado 4.1.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Materiales acústicos",
              "url": "https://jmrplens.github.io/phonometry/es/materials/absorbers/absorption-measurement/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-67",
            "name": "αw",
            "termCode": "αw",
            "description": "Coeficiente de absorción acústica ponderado: la curva de referencia fija desplazada hacia los valores prácticos y leída en 500 Hz. Unidad: adimensional. Definida en: ISO 11654:1997, apartado 4.2.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Materiales acústicos",
              "url": "https://jmrplens.github.io/phonometry/es/materials/absorbers/absorption-measurement/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-68",
            "name": "Clase de absorción",
            "termCode": "Clase de absorción",
            "description": "La clase, de la A a la E, a la que se asigna el coeficiente ponderado, o \"sin clasificar\". Unidad: letra de clase. Definida en: ISO 11654:1997, Tabla B.1.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Materiales acústicos",
              "url": "https://jmrplens.github.io/phonometry/es/materials/absorbers/absorption-measurement/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-69",
            "name": "R",
            "termCode": "R",
            "description": "Resistencia al flujo de aire: la diferencia de presión a través de una probeta dividida por el caudal volumétrico que la atraviesa. Unidad: Pa·s/m³. Definida en: ISO 9053-1:2018, apartado 3.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Materiales acústicos",
              "url": "https://jmrplens.github.io/phonometry/es/materials/absorbers/airflow-resistance/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-70",
            "name": "Rs",
            "termCode": "Rs",
            "description": "Resistencia específica al flujo de aire: la resistencia al flujo referida al área de la cara de la probeta. Unidad: Pa·s/m. Definida en: ISO 9053-1:2018, apartado 3.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Materiales acústicos",
              "url": "https://jmrplens.github.io/phonometry/es/materials/absorbers/airflow-resistance/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-71",
            "name": "σ",
            "termCode": "σ",
            "description": "Resistividad al flujo de aire: la resistencia específica por unidad de espesor, la entrada principal de todo modelo poroso empírico. Unidad: Pa·s/m². Definida en: ISO 9053-1:2018, apartado 3.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Materiales acústicos",
              "url": "https://jmrplens.github.io/phonometry/es/materials/absorbers/airflow-resistance/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-72",
            "name": "Z",
            "termCode": "Z",
            "description": "Impedancia superficial: la relación compleja entre presión acústica y velocidad de partícula en la cara de la muestra, que suele darse normalizada por la impedancia característica del aire. Unidad: Pa·s/m. Definida en: ISO 10534-2:1998, Ecuaciones (17) a (19).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Materiales acústicos",
              "url": "https://jmrplens.github.io/phonometry/es/materials/absorbers/impedance-tube/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-73",
            "name": "s",
            "termCode": "s",
            "description": "Coeficiente de dispersión: la fracción de energía reflejada que no vuelve de forma especular, medida a incidencia aleatoria sobre una plataforma giratoria en cámara reverberante. Unidad: adimensional. Definida en: ISO 17497-1:2004+A1:2014, Fórmula (5).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Difusores y sus coeficientes",
              "url": "https://jmrplens.github.io/phonometry/es/materials/diffusers/diffusers/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-74",
            "name": "d",
            "termCode": "d",
            "description": "Coeficiente de difusión: la uniformidad de la respuesta polar de una superficie, a partir de la autocorrelación de la medida en goniómetro de campo libre. Unidad: adimensional. Definida en: ISO 17497-2:2012, Fórmula (5) (forma normalizada en la Fórmula (7)).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Difusores y sus coeficientes",
              "url": "https://jmrplens.github.io/phonometry/es/materials/diffusers/diffusers/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-75",
            "name": "s′",
            "termCode": "s′",
            "description": "Rigidez dinámica por unidad de superficie de una capa resiliente: una fuerza dinámica por unidad de área dividida por la variación de espesor que provoca. Unidad: MN/m³. Definida en: EN 29052-1:1992 (ISO 9052-1:1989), Fórmula 1.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Rigidez dinámica",
              "url": "https://jmrplens.github.io/phonometry/es/materials/resilient/dynamic-stiffness/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-76",
            "name": "Y",
            "termCode": "Y",
            "description": "Movilidad: la relación compleja entre una respuesta en velocidad y la fuerza que la produce. Unidad: m/(N·s). Definida en: ISO 7626-1:2011, 3.1.2 y Tabla 1.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Movilidad mecánica",
              "url": "https://jmrplens.github.io/phonometry/es/vibration/structural/mechanical-mobility/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-77",
            "name": "Z",
            "termCode": "Z",
            "description": "Impedancia mecánica: la recíproca de la movilidad, fuerza por unidad de velocidad. Unidad: N·s/m. Definida en: ISO 7626-1:2011, Tabla 1.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Movilidad mecánica",
              "url": "https://jmrplens.github.io/phonometry/es/vibration/structural/mechanical-mobility/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-78",
            "name": "H",
            "termCode": "H",
            "description": "Receptancia, o flexibilidad dinámica: respuesta en desplazamiento por unidad de fuerza, el pivote por el que convierte toda la familia. Unidad: m/N. Definida en: ISO 7626-1:2011, Tabla 1.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Movilidad mecánica",
              "url": "https://jmrplens.github.io/phonometry/es/vibration/structural/mechanical-mobility/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-79",
            "name": "A",
            "termCode": "A",
            "description": "Acelerancia, o inertancia: respuesta en aceleración por unidad de fuerza. Su recíproca es la masa aparente. Unidad: 1/kg. Definida en: ISO 7626-1:2011, Tabla 1.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Movilidad mecánica",
              "url": "https://jmrplens.github.io/phonometry/es/vibration/structural/mechanical-mobility/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-80",
            "name": "k21",
            "termCode": "k21",
            "description": "Rigidez dinámica de transferencia de un elemento resiliente: la fuerza bloqueada del lado de salida dividida por el desplazamiento del lado de entrada. Unidad: N/m. Definida en: ISO 10846-1:2008, 3.7.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Rigidez de transferencia",
              "url": "https://jmrplens.github.io/phonometry/es/vibration/structural/transfer-stiffness/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-81",
            "name": "Lk",
            "termCode": "Lk",
            "description": "Nivel de la rigidez dinámica de transferencia, referido a 1 N/m. Unidad: dB re 1 N/m. Definida en: ISO 10846-2:2008 e ISO 10846-3:2002, 3.17.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Rigidez de transferencia",
              "url": "https://jmrplens.github.io/phonometry/es/vibration/structural/transfer-stiffness/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-82",
            "name": "η",
            "termCode": "η",
            "description": "Factor de pérdidas de un elemento resiliente: la tangente del ángulo de fase de su rigidez dinámica de transferencia. Unidad: adimensional. Definida en: ISO 10846-1:2008, 3.8.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Rigidez de transferencia",
              "url": "https://jmrplens.github.io/phonometry/es/vibration/structural/transfer-stiffness/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-83",
            "name": "aw",
            "termCode": "aw",
            "description": "Aceleración ponderada en frecuencia: la raíz de la suma de cuadrados de las aceleraciones por banda tras aplicar las ponderaciones de respuesta humana. Unidad: m/s². Definida en: ISO 2631-1:1997, Ecuación (9).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Vibración en humanos",
              "url": "https://jmrplens.github.io/phonometry/es/vibration/human/human-vibration/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-84",
            "name": "A(8)",
            "termCode": "A(8)",
            "description": "Exposición diaria a vibración: la magnitud de exposición normalizada a una jornada de referencia de 8 h, combinada sobre las operaciones del día. Unidad: m/s². Definida en: ISO 5349-1:2001, Ecuaciones (2) y (3).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Vibración en humanos",
              "url": "https://jmrplens.github.io/phonometry/es/vibration/human/human-vibration/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-85",
            "name": "VDV",
            "termCode": "VDV",
            "description": "Valor de dosis de vibración: la integral temporal de la cuarta potencia de la aceleración ponderada, que pesa los choques mucho más que un valor eficaz. Unidad: m/s^1,75. Definida en: ISO 2631-1:1997, Ecuación (5).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Vibración en humanos",
              "url": "https://jmrplens.github.io/phonometry/es/vibration/human/human-vibration/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-86",
            "name": "MTVV",
            "termCode": "MTVV",
            "description": "Valor máximo de vibración transitoria: el mayor valor eficaz corrido de 1 s de la aceleración ponderada. Unidad: m/s². Definida en: ISO 2631-1:1997, Ecuación (4).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Vibración en humanos",
              "url": "https://jmrplens.github.io/phonometry/es/vibration/human/human-vibration/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-87",
            "name": "R",
            "termCode": "R",
            "description": "Variable de tensión acumulada del modelo de choques múltiples: las tensiones de compresión diarias acumuladas a lo largo de los años de exposición, de la que se lee la probabilidad de lesión lumbar. Unidad: adimensional. Definida en: ISO 2631-5:2018, Anexo C (Fórmulas C.1 y C.3 a C.5).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Vibración con choques múltiples",
              "url": "https://jmrplens.github.io/phonometry/es/vibration/human/multiple-shock-vibration/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-88",
            "name": "Lv",
            "termCode": "Lv",
            "description": "Nivel de velocidad: veinte veces el logaritmo decimal de la velocidad de la superficie dividida por la velocidad de referencia. Unidad: dB. Definida en: ISO/TS 7849-1:2009, Fórmula 3.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Potencia acústica desde vibración",
              "url": "https://jmrplens.github.io/phonometry/es/devices/emission/vibration-sound-power/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-89",
            "name": "ε",
            "termCode": "ε",
            "description": "Factor de radiación, o eficiencia de radiación, de la superficie vibrante de una máquina: la potencia aérea radiada por unidad de velocidad cuadrática media y de área. Unidad: adimensional. Definida en: ISO/TS 7849-1:2009 e ISO/TS 7849-2:2009.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Potencia acústica desde vibración",
              "url": "https://jmrplens.github.io/phonometry/es/devices/emission/vibration-sound-power/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-90",
            "name": "LWs",
            "termCode": "LWs",
            "description": "Nivel de potencia acústica estructural que un equipo inyecta en una placa de recepción. Unidad: dB re 1 pW. Definida en: EN 15657:2018, Fórmula 14.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Potencia estructural",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/design/structure-borne-power/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-91",
            "name": "ηij",
            "termCode": "ηij",
            "description": "Factor de pérdidas por acoplamiento: la fracción de energía por radián que un subsistema de análisis estadístico de energía cede al vecino a través de una unión. Unidad: adimensional. Definida en: Hopkins (2007), Ecuación 2.154; sin norma aplicable.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Transmisión en uniones",
              "url": "https://jmrplens.github.io/phonometry/es/vibration/structural/junction-transmission/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-92",
            "name": "N",
            "termCode": "N",
            "description": "Sonoridad: la magnitud percibida de un sonido, anclada de modo que un tono de 1 kHz a 40 dB SPL vale exactamente 1 sonio. Unidad: sonio. Definida en: ISO 532-1:2017, apartado 5 (estacionaria) y apartado 6 (variable en el tiempo).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Sonoridad",
              "url": "https://jmrplens.github.io/phonometry/es/perception/psychoacoustics/loudness/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-93",
            "name": "N′",
            "termCode": "N′",
            "description": "Sonoridad específica: la densidad de sonoridad a lo largo de la escala de bandas críticas, cuya integral es N. Unidad: sonio/Bark. Definida en: ISO 532-1:2017 (forma en sonio/Cam en ISO 532-2:2017, Fórmula 7).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Sonoridad",
              "url": "https://jmrplens.github.io/phonometry/es/perception/psychoacoustics/loudness/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-94",
            "name": "LN",
            "termCode": "LN",
            "description": "Nivel de sonoridad: el nivel del tono de 1 kHz en campo libre que se juzga igual de fuerte que el sonido. Unidad: fonio. Definida en: ISO 226:2023, Fórmula (2) (curvas en la Fórmula (1)).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Sonoridad",
              "url": "https://jmrplens.github.io/phonometry/es/perception/psychoacoustics/loudness/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-95",
            "name": "S",
            "termCode": "S",
            "description": "Agudeza (sharpness): la posición del centro de gravedad de la sonoridad específica en la escala de bandas críticas, normalizada para que el ruido de banda estrecha de referencia valga exactamente 1,00 acum. Unidad: acum. Definida en: DIN 45692:2009, apartado 6.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Calidad sonora",
              "url": "https://jmrplens.github.io/phonometry/es/perception/psychoacoustics/sound-quality/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-96",
            "name": "R",
            "termCode": "R",
            "description": "Aspereza: la sensación áspera de una modulación de amplitud rápida, en torno a 70 Hz, normalizada para que el tono modulado de referencia valga 1 asper. Unidad: asper. Definida en: ECMA-418-2:2025, apartado 7 (Fórmula 104).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Calidad sonora",
              "url": "https://jmrplens.github.io/phonometry/es/perception/psychoacoustics/sound-quality/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-97",
            "name": "F",
            "termCode": "F",
            "description": "Intensidad de fluctuación: la modulación de amplitud lenta percibida, en torno a 4 Hz, normalizada para que el tono modulado de referencia valga 1 vacil. Unidad: vacil. Definida en: ECMA-418-2:2025, apartado 9 (Fórmula 163).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Calidad sonora",
              "url": "https://jmrplens.github.io/phonometry/es/perception/psychoacoustics/sound-quality/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-98",
            "name": "T",
            "termCode": "T",
            "description": "Tonalidad: el contenido tonal percibido de un sonido, obtenido de la autocorrelación de las envolventes por banda. Unidad: tu. Definida en: ECMA-418-2:2025, apartado 6.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Calidad sonora",
              "url": "https://jmrplens.github.io/phonometry/es/perception/psychoacoustics/sound-quality/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-99",
            "name": "TNR",
            "termCode": "TNR",
            "description": "Relación tono-ruido: el nivel de un tono discreto sobre el ruido enmascarante de la banda crítica que lo rodea. Unidad: dB. Definida en: ECMA-418-1:2024, apartado 11 (Fórmulas 9 a 11).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Tonos prominentes",
              "url": "https://jmrplens.github.io/phonometry/es/perception/psychoacoustics/tone-prominence/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-100",
            "name": "PR",
            "termCode": "PR",
            "description": "Relación de prominencia: el nivel de la banda crítica que contiene el tono sobre la media de las dos bandas contiguas. Unidad: dB. Definida en: ECMA-418-1:2024, apartado 12 (Fórmula 23).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Tonos prominentes",
              "url": "https://jmrplens.github.io/phonometry/es/perception/psychoacoustics/tone-prominence/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-101",
            "name": "ΔL",
            "termCode": "ΔL",
            "description": "Audibilidad de un tono en ruido: el nivel del tono menos el nivel de enmascaramiento de la banda crítica menos el índice de enmascaramiento. Unidad: dB. Definida en: ISO/PAS 20065:2016, Fórmula 14.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Audibilidad de tonos",
              "url": "https://jmrplens.github.io/phonometry/es/perception/psychoacoustics/tone-audibility/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-102",
            "name": "PA",
            "termCode": "PA",
            "description": "Molestia psicoacústica: la sonoridad percentil escalada por la agudeza y por un término que combina fluctuación y aspereza. Unidad: adimensional. Definida en: Fastl y Zwicker (2007), Ecuación 16.2; sin norma aplicable.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Molestia psicoacústica",
              "url": "https://jmrplens.github.io/phonometry/es/perception/psychoacoustics/psychoacoustic-annoyance/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-103",
            "name": "THD",
            "termCode": "THD",
            "description": "Distorsión armónica total: el contenido armónico de la salida respecto al fundamental (THD_F) o respecto a la señal total (THD_R). Unidad: % o dB. Definida en: IEC 60268-3:2013, 14.12.2 a 14.12.11 (la forma R en 14.12.3.2).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Electroacústica",
              "url": "https://jmrplens.github.io/phonometry/es/devices/electroacoustics/electroacoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-104",
            "name": "THD+N",
            "termCode": "THD+N",
            "description": "Distorsión armónica total más ruido: todo lo que queda tras eliminar el fundamental con el filtro de muesca, dentro del ancho de banda de medida normalizado. Unidad: % o dB. Definida en: AES17-2015, apartado 6.3.1 (muesca y ancho de banda en 5.2.5 y 5.2.8).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Electroacústica",
              "url": "https://jmrplens.github.io/phonometry/es/devices/electroacoustics/electroacoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-105",
            "name": "SINAD",
            "termCode": "SINAD",
            "description": "Relación entre señal y ruido más distorsión, la expresión recíproca de la THD+N. Unidad: dB. Definida en: AES17-2015, apartado 6.3.1.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Electroacústica",
              "url": "https://jmrplens.github.io/phonometry/es/devices/electroacoustics/electroacoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-106",
            "name": "IMD",
            "termCode": "IMD",
            "description": "Distorsión de intermodulación por modulación: las bandas laterales que un tono de baja frecuencia genera alrededor de uno de alta. Unidad: %. Definida en: IEC 60268-3:2013, 14.12.7.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Electroacústica",
              "url": "https://jmrplens.github.io/phonometry/es/devices/electroacoustics/electroacoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-107",
            "name": "DIM",
            "termCode": "DIM",
            "description": "Distorsión de intermodulación dinámica, medida con un seno de 15 kHz frente a una onda cuadrada de 3,15 kHz filtrada. Unidad: %. Definida en: IEC 60268-3:2013, 14.12.9.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Electroacústica",
              "url": "https://jmrplens.github.io/phonometry/es/devices/electroacoustics/electroacoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-108",
            "name": "LK, LUFS",
            "termCode": "LK, LUFS",
            "description": "Sonoridad de programa: la suma ponderada por canales de las potencias cuadráticas medias con ponderación K, con puerta en bloques de 400 ms. LUFS y LKFS nombran la misma unidad. Unidad: LUFS. Definida en: ITU-R BS.1770-5, Fórmula 2 (puerta en las Fórmulas 3 a 7).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Sonoridad de programa",
              "url": "https://jmrplens.github.io/phonometry/es/devices/broadcast/program-loudness/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-109",
            "name": "LRA",
            "termCode": "LRA",
            "description": "Rango de sonoridad: la separación entre los percentiles 10 y 95 de la distribución de sonoridad de corto plazo tras la puerta. Unidad: LU. Definida en: EBU Tech 3342.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Sonoridad de programa",
              "url": "https://jmrplens.github.io/phonometry/es/devices/broadcast/program-loudness/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-110",
            "name": "dBTP",
            "termCode": "dBTP",
            "description": "Nivel de pico verdadero: el pico de la señal reconstruida por sobremuestreo, que capta los picos entre muestras que un máximo en el dominio de la muestra se pierde. Unidad: dBTP. Definida en: ITU-R BS.1770-5, Anexo 2.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Sonoridad de programa",
              "url": "https://jmrplens.github.io/phonometry/es/devices/broadcast/program-loudness/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-111",
            "name": "PNL",
            "termCode": "PNL",
            "description": "Nivel de ruido percibido: los niveles de las 24 bandas de tercio de octava convertidos a ruidosidad en noys y recombinados. Unidad: PNdB. Definida en: Anexo 16 OACI, Vol. I, Apéndice 2 (ley de ruidosidad en la Tabla A2-3).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Ruido de aeronaves",
              "url": "https://jmrplens.github.io/phonometry/es/aircraft/aircraft-noise/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-112",
            "name": "PNLT",
            "termCode": "PNLT",
            "description": "Nivel de ruido percibido corregido por tonos: el PNL más la penalización por irregularidades espectrales como los tonos de ventilador y turbina. Unidad: PNdB. Definida en: Anexo 16 OACI, Vol. I, Apéndice 2.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Ruido de aeronaves",
              "url": "https://jmrplens.github.io/phonometry/es/aircraft/aircraft-noise/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-113",
            "name": "EPNL",
            "termCode": "EPNL",
            "description": "Nivel efectivo de ruido percibido: el PNLT máximo más la corrección por duración en la ventana de 10 dB, la métrica de certificación acústica. Unidad: EPNdB. Definida en: Anexo 16 OACI, Vol. I, Apéndice 2.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Ruido de aeronaves",
              "url": "https://jmrplens.github.io/phonometry/es/aircraft/aircraft-noise/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-114",
            "name": "Lp (submarino)",
            "termCode": "Lp (submarino)",
            "description": "Nivel de presión acústica submarina, referido a 1 µPa y no a 20 µPa. Un nivel aéreo nunca se convierte a él con una simple resta. Unidad: dB re 1 µPa. Definida en: ISO 18405:2017 (nivel cuadrático medio en ISO 18406:2017, Fórmula 7).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Acústica submarina",
              "url": "https://jmrplens.github.io/phonometry/es/underwater/underwater-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-115",
            "name": "SEL (submarino)",
            "termCode": "SEL (submarino)",
            "description": "Nivel de exposición sonora submarina: la integral temporal de la presión al cuadrado referida a 1 µPa²·s. Unidad: dB re 1 µPa²·s. Definida en: ISO 18405:2017.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Acústica submarina",
              "url": "https://jmrplens.github.io/phonometry/es/underwater/underwater-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-116",
            "name": "LRN",
            "termCode": "LRN",
            "description": "Nivel de ruido radiado por un buque: el nivel del producto de la presión eficaz en campo lejano por la distancia a la fuente. Unidad: dB re 1 µPa·m. Definida en: ISO 17208-1:2016.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Acústica submarina",
              "url": "https://jmrplens.github.io/phonometry/es/underwater/underwater-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-117",
            "name": "Ls",
            "termCode": "Ls",
            "description": "Nivel de fuente monopolar equivalente: el nivel de ruido radiado tras la corrección de superficie de Lloyd, de modo que un solo número describa la fuente en sí. Unidad: dB re 1 µPa·m. Definida en: ISO 17208-2:2019, Fórmula 3.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Acústica submarina",
              "url": "https://jmrplens.github.io/phonometry/es/underwater/underwater-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-118",
            "name": "u(y)",
            "termCode": "u(y)",
            "description": "Incertidumbre típica combinada de un resultado, propagada desde las incertidumbres típicas de sus entradas por la ley de propagación de la incertidumbre. Unidad: unidad del resultado. Definida en: ISO/IEC Guide 98-3:2008 (JCGM 100:2008), apartado 5.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Incertidumbre GUM",
              "url": "https://jmrplens.github.io/phonometry/es/signal/metrology/gum-uncertainty/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-119",
            "name": "U",
            "termCode": "U",
            "description": "Incertidumbre expandida: la incertidumbre típica combinada multiplicada por un factor de cobertura, que define un intervalo de cobertura. Unidad: unidad del resultado. Definida en: ISO/IEC Guide 98-3:2008 (JCGM 100:2008), apartado 6 y Anexo G.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Incertidumbre GUM",
              "url": "https://jmrplens.github.io/phonometry/es/signal/metrology/gum-uncertainty/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#term-120",
            "name": "σR",
            "termCode": "σR",
            "description": "Incertidumbre típica de una situación de medida en acústica de la edificación, tabulada por magnitud y por situación. Unidad: dB. Definida en: ISO 12999-1:2020, apartado 5.2 (factores de cobertura en la Tabla 8).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/es/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Aislamiento en campo",
              "url": "https://jmrplens.github.io/phonometry/es/buildings/insulation/insulation-field/"
            }
          }
        ]
      }
---

Cada guía empieza por la norma que implementa, así que una magnitud siempre
queda definida allí donde se usa. Esta página va en sentido contrario: tienes un
símbolo, sacado de un informe, de un pliego o del correo de un colega, y quieres
saber qué es, en qué se mide, qué documento lo define y dónde se calcula en esta
documentación.

Dos convenios rigen la cuarta columna, y pesan más de lo que parece. Cuando
aparece un apartado, una fórmula o una tabla, es el que cita la implementación,
tomado de la norma que implementa la guía. Cuando solo aparece una designación,
la norma está establecida pero el apartado que define la magnitud no se enuncia
en ninguna parte de esta documentación, y inventar uno verosímil sería peor que
omitirlo. Unas pocas magnitudes no tienen norma aplicable: su fuente es el
artículo o el libro del que sale el modelo, citado como tal.

Los símbolos chocan entre dominios, y la tabla no disimula. *R* es el índice de
reducción acústica en edificación, la resistencia al flujo de aire en
materiales, la aspereza en psicoacústica y la variable de tensión acumulada en
ISO 2631-5. Sigma es la resistividad al flujo de aire de un material poroso y la
eficiencia de radiación de una placa. *A* es un área de absorción equivalente y
una acelerancia. *L*<sub>N</sub> es un nivel percentil aquí y un nivel de
sonoridad en fonios allá. El símbolo se lee junto con su dominio, y para eso
sirve la agrupación de abajo.

Si lo que buscas es la fuente de cada definición y no la definición en sí, la
[bibliografía](/phonometry/es/reference/bibliography/) recoge todas las obras
citadas con su DOI o su enlace del editor, y el
[informe de conformidad](/phonometry/es/reference/conformance/) muestra la
comprobación numérica que fija cada magnitud al valor esperado de su propia
norma.

## Niveles de presión, potencia e intensidad

| Símbolo | Magnitud | Unidad | Definida en | Guía |
| --- | --- | --- | --- | --- |
| $L_p$ | Nivel de presión acústica: veinte veces el logaritmo decimal de la presión acústica eficaz dividida por la presión de referencia. | dB re 20 µPa | IEC 61672-1:2013 | [Calibración](/phonometry/es/signal/metrology/calibration/) |
| $L_{eq}$ | Nivel de presión acústica continuo equivalente: el nivel del sonido estacionario que transporta la misma presión cuadrática media durante el intervalo. | dB | IEC 61672-1:2013 | [Niveles](/phonometry/es/signal/levels/levels/) |
| $L_{Aeq}$ | La misma integral aplicada a la señal ponderada A, el descriptor por defecto del ruido ambiental y laboral. | dB | IEC 61672-1:2013 | [Niveles](/phonometry/es/signal/levels/levels/) |
| $L_{AE}$, SEL | Nivel de exposición sonora: toda la energía ponderada A de un suceso único normalizada a un segundo. | dB | IEC 61672-1:2013, Ecuación 8 (Tabla 4) | [Niveles](/phonometry/es/signal/levels/levels/) |
| $L_{Cpeak}$ | Nivel de pico ponderado C: el máximo absoluto de la presión ponderada C, no un máximo con ponderación temporal. | dB | IEC 61672-1:2013, apartado 5.13 | [Niveles](/phonometry/es/signal/levels/levels/) |
| $L_N$ ($L_{10}$, $L_{50}$, $L_{90}$) | Nivel percentil: el nivel superado el $N$ % del tiempo de medida, leído en la distribución del nivel con ponderación temporal. | dB | ISO 1996-2:2017 (el Anexo I usa $L_{90}$ como nivel residual) | [Niveles](/phonometry/es/signal/levels/levels/) |
| $L_W$, SWL | Nivel de potencia acústica: la potencia que radia una fuente, referida a 1 pW. | dB re 1 pW | ISO 3745:2012, apartado 8 | [Potencia acústica](/phonometry/es/devices/emission/sound-power/) |
| $L_I$ | Nivel de intensidad acústica: el módulo del vector intensidad referido a 1 pW/m², con el sentido del flujo indicado aparte mediante un signo. | dB re 1 pW/m² | IEC 61043:1993 | [Intensidad](/phonometry/es/devices/emission/intensity/) |
| $L_p - L_I$ | Índice presión-intensidad: la diferencia entre el nivel de presión y el de intensidad en una posición, el indicador de campo que cualifica una medida de intensidad. | dB | ISO 9614-1:1993, Ecuación (A.3) | [Intensidad](/phonometry/es/devices/emission/intensity/) |

## Descriptores ambientales y laborales

| Símbolo | Magnitud | Unidad | Definida en | Guía |
| --- | --- | --- | --- | --- |
| $L_{den}$ | Nivel día-tarde-noche: la media energética de los tres periodos con 5 dB añadidos a la tarde y 10 dB a la noche. | dB | ISO 1996-1:2016, 3.6.4 | [Niveles ambientales](/phonometry/es/environment/environmental-levels/) |
| $L_{dn}$ | Nivel día-noche: la misma construcción solo con la penalización de 10 dB nocturna. | dB | ISO 1996-1:2016, 3.6.5 | [Niveles ambientales](/phonometry/es/environment/environmental-levels/) |
| $L_r$ | Nivel de valoración: el nivel compuesto de la jornada completa tras los ajustes por carácter de la fuente y por franja horaria. | dB | ISO 1996-1:2016, apartado 6.5 (Fórmulas 5 y 6) | [Niveles ambientales](/phonometry/es/environment/environmental-levels/) |
| $L_{Ar,T}$ | Nivel de valoración de una fuente impulsiva en un intervalo de referencia: el $L_{Aeq}$ más el ajuste graduado por impulsos. | dB | NT ACOU 112:2002, apartado 8 | [Prominencia de impulsos](/phonometry/es/environment/assessment/impulsive-sound/) |
| $K_I$ | Ajuste por impulsos que se suma al $L_{Aeq}$, graduado según la prominencia prevista de los impulsos. | dB | NT ACOU 112:2002, apartado 8 | [Prominencia de impulsos](/phonometry/es/environment/assessment/impulsive-sound/) |
| $E$ | Exposición sonora: la integral temporal de la presión acústica ponderada A al cuadrado durante el periodo de exposición. | Pa²h | IEC 61252:1993, 3.1 | [Niveles](/phonometry/es/signal/levels/levels/) |
| $L_{EX,8h}$, $L_{EP,d}$ | Nivel de exposición diaria al ruido: el nivel estacionario que, mantenido durante una jornada nominal de 8 h, acumula la misma exposición sonora ponderada A que la medida. | dB | IEC 61252:1993, 3.3 | [Exposición laboral](/phonometry/es/perception/hearing/occupational-exposure/) |
| $L_{p,A,eqT}$ | Nivel continuo equivalente ponderado A de una tarea, de una muestra del puesto o de una jornada completa, el ladrillo con el que se construye el $L_{EX,8h}$. | dB | ISO 9612:2009, apartados 9 a 11 | [Exposición laboral](/phonometry/es/perception/hearing/occupational-exposure/) |
| NIPTS | Desplazamiento permanente del umbral inducido por ruido: la pérdida auditiva mediana atribuible a un nivel, una duración y una frecuencia audiométrica dados. | dB | ISO 1999:2013 | [Pérdida auditiva por ruido](/phonometry/es/perception/hearing/noise-induced-hearing-loss/) |
| HTLAN | Nivel de umbral auditivo asociado a la edad y al ruido: el NIPTS combinado con la componente de la edad. | dB | ISO 1999:2013 | [Pérdida auditiva por ruido](/phonometry/es/perception/hearing/noise-induced-hearing-loss/) |

## Ponderación frecuencial y temporal

| Símbolo | Magnitud | Unidad | Definida en | Guía |
| --- | --- | --- | --- | --- |
| A, C, Z | Las ponderaciones frecuenciales normativas: las curvas de respuesta del oído que se aplican antes de integrar, siendo Z la referencia plana. | dB | IEC 61672-1:2013, Anexo E (límites de aceptación en la Tabla 3) | [Ponderación frecuencial](/phonometry/es/signal/levels/weighting/) |
| G | Ponderación para infrasonido, definida por sus polos y ceros en el intervalo de 0,25 Hz a 315 Hz. | dB | ISO 7196:1995, Tabla 1 (respuestas nominales en la Tabla 2) | [Ponderaciones especiales](/phonometry/es/signal/levels/special-weightings/) |
| B | Ponderación histórica para niveles medios, retirada de la norma vigente de sonómetros. | dB | ANSI S1.4-1983, Apéndice C (Fórmula C2) | [Ponderaciones especiales](/phonometry/es/signal/levels/special-weightings/) |
| $D$ | Ponderación histórica para ruido de aeronaves, derivada de la curva de ruidosidad percibida de 40 noys. | dB | IEC 537:1976 (retirada) | [Ponderaciones especiales](/phonometry/es/signal/levels/special-weightings/) |
| AU | Ponderación para el sonido audible medido en presencia de ultrasonido. | dB | IEC 61012:1990, apartado 2.2 (Tablas 1 y 2) | [Ponderaciones especiales](/phonometry/es/signal/levels/special-weightings/) |
| F, S, I | Ponderaciones temporales exponenciales Fast, Slow e Impulse: las balísticas del detector que producen el nivel mostrado. | s (constante de tiempo) | IEC 61672-1:2013 | [Ponderación temporal](/phonometry/es/signal/levels/time-weighting/) |

## Acústica de salas

| Símbolo | Magnitud | Unidad | Definida en | Guía |
| --- | --- | --- | --- | --- |
| $T_{20}$ | Tiempo de reverberación extrapolado a una caída de 60 dB desde un ajuste por mínimos cuadrados entre −5 dB y −25 dB de la curva de Schroeder. | s | ISO 3382-2:2008, apartado 6 y Anexo C | [Acústica de salas](/phonometry/es/buildings/rooms/room-acoustics/) |
| $T_{30}$ | La misma extrapolación desde un ajuste entre −5 dB y −35 dB, la opción habitual cuando el margen de caída lo permite. | s | ISO 3382-2:2008, apartado 6 y Anexo C | [Acústica de salas](/phonometry/es/buildings/rooms/room-acoustics/) |
| $T_{60}$, RT | El tiempo de reverberación propiamente dicho: el que tarda la energía sonora en caer 60 dB. En la práctica se mide como $T_{20}$ o $T_{30}$. | s | ISO 3382-1:2009 | [Acústica de salas](/phonometry/es/buildings/rooms/room-acoustics/) |
| EDT | Tiempo de caída inicial: la misma pendiente tomada sobre los primeros 10 dB de caída, que sigue la reverberancia percibida y no la cola. | s | ISO 3382-1:2009 (diferencia apenas perceptible en la Tabla A.1) | [Acústica de salas](/phonometry/es/buildings/rooms/room-acoustics/) |
| $C_{50}$ | Claridad para la palabra: la relación energética entre los primeros 50 ms de la respuesta al impulso y todo lo que viene después. | dB | ISO 3382-1:2009 | [Acústica de salas](/phonometry/es/buildings/rooms/room-acoustics/) |
| $C_{80}$ | Claridad para la música: la misma relación con la frontera en 80 ms. | dB | ISO 3382-1:2009 (diferencia apenas perceptible en la Tabla A.1) | [Acústica de salas](/phonometry/es/buildings/rooms/room-acoustics/) |
| $D_{50}$ | Definición, o Deutlichkeit: la fracción de la energía total que llega en los primeros 50 ms. | adimensional | ISO 3382-1:2009 (diferencia apenas perceptible en la Tabla A.1) | [Acústica de salas](/phonometry/es/buildings/rooms/room-acoustics/) |
| $T_s$ | Tiempo central: el centro de gravedad temporal de la respuesta al impulso al cuadrado, una alternativa a los índices de claridad sin frontera arbitraria. | s | ISO 3382-1:2009, Ecuación (A.13) | [Acústica de salas](/phonometry/es/buildings/rooms/room-acoustics/) |
| $A$ | Área de absorción acústica equivalente de un recinto: el área de una superficie perfectamente absorbente que daría el mismo tiempo de reverberación. | m² | ISO 354:2003, Ecuaciones (5) y (7) | [Medida de la absorción](/phonometry/es/materials/absorbers/absorption-measurement/) |
| NC | Índice de ruido de sala de un espectro de fondo: el nivel de interferencia con la palabra elige la curva, y el método de tangencia valora el espectro cuando alguna banda la supera. | dB (índice) | ANSI/ASA S12.2-2019, 5.2.2 y 5.2.3 (curvas en la Tabla 1) | [Ruido de salas](/phonometry/es/buildings/rooms/room-noise/) |
| SIL | Nivel de interferencia con la palabra: la media de los niveles en las bandas de octava de 500, 1000, 2000 y 4000 Hz. | dB | ANSI/ASA S12.2-2019, apartado 3.2 | [Ruido de salas](/phonometry/es/buildings/rooms/room-noise/) |
| RC | Índice RC Mark II: la media de los niveles de 500, 1000 y 2000 Hz, con una etiqueta espectral de retumbo, siseo o neutro. | dB (índice) | ANSI/ASA S12.2-2019, Anexo D (apartados D.3 y D.4) | [Ruido de salas](/phonometry/es/buildings/rooms/room-noise/) |
| NR | Noise Rating, la familia de curvas europea equivalente a NC. Se comenta a efectos de comparación y no se implementa deliberadamente. | dB (índice) | Kosten y van Os (1962); sin norma aplicable | [Ruido de salas](/phonometry/es/buildings/rooms/room-noise/) |

## Habla e inteligibilidad

| Símbolo | Magnitud | Unidad | Definida en | Guía |
| --- | --- | --- | --- | --- |
| $m(F)$ | Función de transferencia de modulación: la fracción de la profundidad de modulación de la envolvente del habla, a la frecuencia de modulación $F$, que sobrevive al camino de transmisión. | adimensional | IEC 60268-16:2020 | [Transmisión del habla](/phonometry/es/perception/speech/speech-transmission/) |
| STI | Índice de transmisión del habla: la matriz de transferencia de modulación convertida en relaciones señal-ruido efectivas y ponderada en un único valor entre 0 y 1. | adimensional | IEC 60268-16:2020, A.5.2 a A.5.6 | [Transmisión del habla](/phonometry/es/perception/speech/speech-transmission/) |
| STIPA | La medida directa del STI, reproduciendo una señal de prueba normalizada con dos modulaciones por banda a través de la cadena real. | adimensional | IEC 60268-16:2020, apartado 6.3 y Tabla 3 (método directo, Anexo B) | [Transmisión del habla](/phonometry/es/perception/speech/speech-transmission/) |
| SII | Índice de inteligibilidad del habla: la audibilidad del espectro de habla frente al ruido y al umbral del oyente, ponderada por la función de importancia de cada banda. | adimensional | ANSI S3.5-1997, apartado 6 (procedimiento en el apartado 5, función de importancia en la Tabla 3) | [Inteligibilidad del habla](/phonometry/es/perception/speech/speech-intelligibility/) |
| STOI | Inteligibilidad objetiva de corta duración: la correlación recortada de las envolventes por banda entre el habla limpia y la degradada. | adimensional | Taal et al. (2011), Ecuaciones 5 y 6; sin norma aplicable | [Inteligibilidad objetiva](/phonometry/es/perception/speech/objective-intelligibility/) |
| ESTOI | La versión extendida, normalizada por filas y por columnas para que siga a los enmascarantes modulados. | adimensional | Jensen y Taal (2016), Ecuación 8; sin norma aplicable | [Inteligibilidad objetiva](/phonometry/es/perception/speech/objective-intelligibility/) |

## Aislamiento acústico

| Símbolo | Magnitud | Unidad | Definida en | Guía |
| --- | --- | --- | --- | --- |
| $D$ | Diferencia de niveles: el nivel promediado energéticamente en el recinto emisor menos el del receptor, sin normalizar. | dB | ISO 16283-1:2014, 3.12 a 3.15 | [Aislamiento en campo](/phonometry/es/buildings/insulation/insulation-field/) |
| $D_{nT}$ | Diferencia de niveles estandarizada: la diferencia de niveles referida a un tiempo de reverberación de referencia, 0,5 s en viviendas. | dB | ISO 16283-1:2014, 3.12 a 3.15 | [Aislamiento en campo](/phonometry/es/buildings/insulation/insulation-field/) |
| $D_n$ | Diferencia de niveles normalizada: la diferencia de niveles referida a un área de absorción de referencia de 10 m². | dB | ISO 10052:2021 | [Método de control](/phonometry/es/buildings/insulation/insulation-survey/) |
| $D_{n,e}$ | Diferencia de niveles normalizada de elemento, para un elemento pequeño o una vía de aire, referida a un área de referencia de 10 m². | dB | EN 12354-3:2000 | [Predicción del aislamiento](/phonometry/es/buildings/design/insulation-prediction/) |
| $R$ | Índice de reducción acústica: la diferencia de niveles corregida por el área del cerramiento partido por el área de absorción del recinto receptor, medida en laboratorio con los flancos suprimidos. | dB | ISO 10140-2:2010 | [Aislamiento en laboratorio](/phonometry/es/buildings/insulation/insulation-lab/) |
| $R'$ | Índice de reducción acústica aparente: la misma construcción medida en el edificio, así que incluye todas las vías de flanco. La prima es la marca que distingue el laboratorio del campo. | dB | ISO 16283-1:2014, 3.12 a 3.15 | [Aislamiento en campo](/phonometry/es/buildings/insulation/insulation-field/) |
| TL | Pérdidas por transmisión: el aislamiento a ruido aéreo de un panel predicho a partir de sus propiedades físicas, la misma magnitud que $R$ en un contexto de predicción. | dB | Bies, Hansen y Howard (2017), Sección 7.2; sin norma aplicable | [Aislamiento de paneles](/phonometry/es/buildings/design/panel-sound-insulation/) |
| $R_w$, $R'_w$, $D_{nT,w}$ | Los índices globales ponderados: una curva de referencia fija se desplaza hacia el espectro medido hasta que las desviaciones desfavorables alcanzan su suma admisible, y la curva desplazada se lee en 500 Hz. | dB | ISO 717-1:2020 | [Índices de aislamiento](/phonometry/es/buildings/insulation/insulation-ratings/) |
| $D_{n,e,w}$ | El mismo índice global de curva de referencia aplicado a la diferencia de niveles normalizada de elemento. | dB | ISO 717-1:2020 | [Predicción del aislamiento](/phonometry/es/buildings/design/insulation-prediction/) |
| $C$, $C_{tr}$ | Términos de adaptación espectral: las correcciones que vuelven a valorar la curva medida frente a ruido rosa ponderado A ($C$) y frente a tráfico rodado urbano ponderado A ($C_{tr}$). | dB | ISO 717-1:2020, Anexo A | [Índices de aislamiento](/phonometry/es/buildings/insulation/insulation-ratings/) |
| $L_n$ | Nivel de presión acústica de impactos normalizado: el nivel en el recinto receptor bajo la máquina de impactos normalizada, referido a un área de absorción de 10 m². | dB | ISO 10140-3:2010 | [Aislamiento en laboratorio](/phonometry/es/buildings/insulation/insulation-lab/) |
| $L'_{nT}$ | Nivel de presión acústica de impactos estandarizado, referido a un tiempo de reverberación de referencia. Atención al signo: más reverberación lo baja, al revés que el $D_{nT}$. | dB | ISO 16283-2:2015 | [Aislamiento en campo](/phonometry/es/buildings/insulation/insulation-field/) |
| $L_{n,w}$, $L'_{nT,w}$ | Los índices globales de impactos. La curva de referencia se desplaza igual, pero ahora una desviación desfavorable es aquella en la que la medida supera a la referencia. | dB | ISO 717-2:2020 | [Índices de aislamiento](/phonometry/es/buildings/insulation/insulation-ratings/) |
| $C_I$ | Término de adaptación espectral de impactos, a partir de la suma energética entre 100 Hz y 2500 Hz. El término de rango ampliado $C_{I,50\text{–}2500}$ lo extiende hasta 50 Hz. | dB | ISO 717-2:2020 (rango ampliado en la NOTA de A.2.1) | [Índices de aislamiento](/phonometry/es/buildings/insulation/insulation-ratings/) |
| $\Delta L_w$ | Reducción ponderada del nivel de presión acústica de impactos que aporta un revestimiento de suelo, medida como la mejora sobre el forjado desnudo de referencia. | dB | ISO 717-2:2020 (medición en ISO 16251-1:2014, Fórmulas (3) y (4)) | [Mejora de revestimientos](/phonometry/es/buildings/design/impact-improvement/) |
| $\Delta R_w$ | Mejora ponderada del aislamiento a ruido aéreo que aporta un trasdosado o una capa adicional, que se suma al índice del elemento en la predicción. | dB | EN 12354-1:2000, Fórmulas 27 y 28a | [Predicción del aislamiento](/phonometry/es/buildings/design/insulation-prediction/) |
| $K_{ij}$ | Índice de reducción vibracional de una unión: la diferencia de niveles de velocidad promediada en ambos sentidos, corregida por la longitud de la unión y las longitudes de absorción equivalentes. | dB | ISO 10848-1:2006, Fórmula (13) | [Flancos en laboratorio](/phonometry/es/buildings/insulation/flanking-lab/) |
| $f_c$ | Frecuencia crítica: aquella en la que la longitud de onda de flexión del panel iguala a la del aire, donde aparece la caída por coincidencia. | Hz | Bies, Hansen y Howard (2017), Ecuación 7.3; sin norma aplicable | [Aislamiento de paneles](/phonometry/es/buildings/design/panel-sound-insulation/) |
| $\sigma$ | Eficiencia de radiación de una placa: la potencia aérea radiada por unidad de velocidad cuadrática media de la superficie, normalizada por el valor de onda plana. | adimensional | Hopkins (2007), Ecuaciones 2.227 a 2.230; sin norma aplicable | [Aislamiento de paneles](/phonometry/es/buildings/design/panel-sound-insulation/) |

## Materiales y superficies

| Símbolo | Magnitud | Unidad | Definida en | Guía |
| --- | --- | --- | --- | --- |
| $\alpha$ | Coeficiente de absorción acústica a incidencia normal: la fracción de energía incidente que la superficie no devuelve, obtenida en el tubo de impedancia a partir del factor de reflexión. | adimensional | ISO 10534-2:1998, Ecuaciones (17) a (19) | [Tubo de impedancia](/phonometry/es/materials/absorbers/impedance-tube/) |
| $\alpha_s$ | Coeficiente de absorción acústica a incidencia aleatoria medido en cámara reverberante, a partir del cambio del área de absorción equivalente con y sin la muestra. | adimensional | ISO 354:2003, Ecuaciones (8) y (9) | [Medida de la absorción](/phonometry/es/materials/absorbers/absorption-measurement/) |
| $\alpha_p$ | Coeficiente de absorción acústica práctico: los datos en tercios de octava agrupados en bandas de octava y redondeados a pasos de 0,05. | adimensional | ISO 11654:1997, apartado 4.1 | [Medida de la absorción](/phonometry/es/materials/absorbers/absorption-measurement/) |
| $\alpha_w$ | Coeficiente de absorción acústica ponderado: la curva de referencia fija desplazada hacia los valores prácticos y leída en 500 Hz. | adimensional | ISO 11654:1997, apartado 4.2 | [Medida de la absorción](/phonometry/es/materials/absorbers/absorption-measurement/) |
| Clase de absorción | La clase, de la A a la E, a la que se asigna el coeficiente ponderado, o "sin clasificar". | letra de clase | ISO 11654:1997, Tabla B.1 | [Medida de la absorción](/phonometry/es/materials/absorbers/absorption-measurement/) |
| $R$ | Resistencia al flujo de aire: la diferencia de presión a través de una probeta dividida por el caudal volumétrico que la atraviesa. | Pa·s/m³ | ISO 9053-1:2018, apartado 3 | [Resistencia al flujo](/phonometry/es/materials/absorbers/airflow-resistance/) |
| $R_s$ | Resistencia específica al flujo de aire: la resistencia al flujo referida al área de la cara de la probeta. | Pa·s/m | ISO 9053-1:2018, apartado 3 | [Resistencia al flujo](/phonometry/es/materials/absorbers/airflow-resistance/) |
| $\sigma$ | Resistividad al flujo de aire: la resistencia específica por unidad de espesor, la entrada principal de todo modelo poroso empírico. | Pa·s/m² | ISO 9053-1:2018, apartado 3 | [Resistencia al flujo](/phonometry/es/materials/absorbers/airflow-resistance/) |
| $Z$ | Impedancia superficial: la relación compleja entre presión acústica y velocidad de partícula en la cara de la muestra, que suele darse normalizada por la impedancia característica del aire. | Pa·s/m | ISO 10534-2:1998, Ecuaciones (17) a (19) | [Tubo de impedancia](/phonometry/es/materials/absorbers/impedance-tube/) |
| $s$ | Coeficiente de dispersión: la fracción de energía reflejada que no vuelve de forma especular, medida a incidencia aleatoria sobre una plataforma giratoria en cámara reverberante. | adimensional | ISO 17497-1:2004+A1:2014, Fórmula (5) | [Difusores](/phonometry/es/materials/diffusers/diffusers/) |
| $d$ | Coeficiente de difusión: la uniformidad de la respuesta polar de una superficie, a partir de la autocorrelación de la medida en goniómetro de campo libre. | adimensional | ISO 17497-2:2012, Fórmula (5) (forma normalizada en la Fórmula (7)) | [Difusores](/phonometry/es/materials/diffusers/diffusers/) |
| $s'$ | Rigidez dinámica por unidad de superficie de una capa resiliente: una fuerza dinámica por unidad de área dividida por la variación de espesor que provoca. | MN/m³ | EN 29052-1:1992 (ISO 9052-1:1989), Fórmula 1 | [Rigidez dinámica](/phonometry/es/materials/resilient/dynamic-stiffness/) |

## Vibración y ruido estructural

| Símbolo | Magnitud | Unidad | Definida en | Guía |
| --- | --- | --- | --- | --- |
| $Y$ | Movilidad: la relación compleja entre una respuesta en velocidad y la fuerza que la produce. | m/(N·s) | ISO 7626-1:2011, 3.1.2 y Tabla 1 | [Movilidad mecánica](/phonometry/es/vibration/structural/mechanical-mobility/) |
| $Z$ | Impedancia mecánica: la recíproca de la movilidad, fuerza por unidad de velocidad. | N·s/m | ISO 7626-1:2011, Tabla 1 | [Movilidad mecánica](/phonometry/es/vibration/structural/mechanical-mobility/) |
| $H$ | Receptancia, o flexibilidad dinámica: respuesta en desplazamiento por unidad de fuerza, el pivote por el que convierte toda la familia. | m/N | ISO 7626-1:2011, Tabla 1 | [Movilidad mecánica](/phonometry/es/vibration/structural/mechanical-mobility/) |
| $A$ | Acelerancia, o inertancia: respuesta en aceleración por unidad de fuerza. Su recíproca es la masa aparente. | 1/kg | ISO 7626-1:2011, Tabla 1 | [Movilidad mecánica](/phonometry/es/vibration/structural/mechanical-mobility/) |
| $k_{21}$ | Rigidez dinámica de transferencia de un elemento resiliente: la fuerza bloqueada del lado de salida dividida por el desplazamiento del lado de entrada. | N/m | ISO 10846-1:2008, 3.7 | [Rigidez de transferencia](/phonometry/es/vibration/structural/transfer-stiffness/) |
| $L_k$ | Nivel de la rigidez dinámica de transferencia, referido a 1 N/m. | dB re 1 N/m | ISO 10846-2:2008 e ISO 10846-3:2002, 3.17 | [Rigidez de transferencia](/phonometry/es/vibration/structural/transfer-stiffness/) |
| $\eta$ | Factor de pérdidas de un elemento resiliente: la tangente del ángulo de fase de su rigidez dinámica de transferencia. | adimensional | ISO 10846-1:2008, 3.8 | [Rigidez de transferencia](/phonometry/es/vibration/structural/transfer-stiffness/) |
| $a_w$ | Aceleración ponderada en frecuencia: la raíz de la suma de cuadrados de las aceleraciones por banda tras aplicar las ponderaciones de respuesta humana. | m/s² | ISO 2631-1:1997, Ecuación (9) | [Vibración en humanos](/phonometry/es/vibration/human/human-vibration/) |
| $A(8)$ | Exposición diaria a vibración: la magnitud de exposición normalizada a una jornada de referencia de 8 h, combinada sobre las operaciones del día. | m/s² | ISO 5349-1:2001, Ecuaciones (2) y (3) | [Vibración en humanos](/phonometry/es/vibration/human/human-vibration/) |
| VDV | Valor de dosis de vibración: la integral temporal de la cuarta potencia de la aceleración ponderada, que pesa los choques mucho más que un valor eficaz. | m/s^1,75 | ISO 2631-1:1997, Ecuación (5) | [Vibración en humanos](/phonometry/es/vibration/human/human-vibration/) |
| MTVV | Valor máximo de vibración transitoria: el mayor valor eficaz corrido de 1 s de la aceleración ponderada. | m/s² | ISO 2631-1:1997, Ecuación (4) | [Vibración en humanos](/phonometry/es/vibration/human/human-vibration/) |
| $R$ | Variable de tensión acumulada del modelo de choques múltiples: las tensiones de compresión diarias acumuladas a lo largo de los años de exposición, de la que se lee la probabilidad de lesión lumbar. | adimensional | ISO 2631-5:2018, Anexo C (Fórmulas C.1 y C.3 a C.5) | [Vibración con choques múltiples](/phonometry/es/vibration/human/multiple-shock-vibration/) |
| $L_v$ | Nivel de velocidad: veinte veces el logaritmo decimal de la velocidad de la superficie dividida por la velocidad de referencia. | dB | ISO/TS 7849-1:2009, Fórmula 3 | [Potencia acústica desde vibración](/phonometry/es/devices/emission/vibration-sound-power/) |
| $\varepsilon$ | Factor de radiación, o eficiencia de radiación, de la superficie vibrante de una máquina: la potencia aérea radiada por unidad de velocidad cuadrática media y de área. | adimensional | ISO/TS 7849-1:2009 e ISO/TS 7849-2:2009 | [Potencia acústica desde vibración](/phonometry/es/devices/emission/vibration-sound-power/) |
| $L_{Ws}$ | Nivel de potencia acústica estructural que un equipo inyecta en una placa de recepción. | dB re 1 pW | EN 15657:2018, Fórmula 14 | [Potencia estructural](/phonometry/es/buildings/design/structure-borne-power/) |
| $\eta_{ij}$ | Factor de pérdidas por acoplamiento: la fracción de energía por radián que un subsistema de análisis estadístico de energía cede al vecino a través de una unión. | adimensional | Hopkins (2007), Ecuación 2.154; sin norma aplicable | [Transmisión en uniones](/phonometry/es/vibration/structural/junction-transmission/) |

## Psicoacústica

| Símbolo | Magnitud | Unidad | Definida en | Guía |
| --- | --- | --- | --- | --- |
| $N$ | Sonoridad: la magnitud percibida de un sonido, anclada de modo que un tono de 1 kHz a 40 dB SPL vale exactamente 1 sonio. | sonio | ISO 532-1:2017, apartado 5 (estacionaria) y apartado 6 (variable en el tiempo) | [Sonoridad](/phonometry/es/perception/psychoacoustics/loudness/) |
| $N'$ | Sonoridad específica: la densidad de sonoridad a lo largo de la escala de bandas críticas, cuya integral es $N$. | sonio/Bark | ISO 532-1:2017 (forma en sonio/Cam en ISO 532-2:2017, Fórmula 7) | [Sonoridad](/phonometry/es/perception/psychoacoustics/loudness/) |
| $L_N$ | Nivel de sonoridad: el nivel del tono de 1 kHz en campo libre que se juzga igual de fuerte que el sonido. | fonio | ISO 226:2023, Fórmula (2) (curvas en la Fórmula (1)) | [Sonoridad](/phonometry/es/perception/psychoacoustics/loudness/) |
| $S$ | Agudeza (sharpness): la posición del centro de gravedad de la sonoridad específica en la escala de bandas críticas, normalizada para que el ruido de banda estrecha de referencia valga exactamente 1,00 acum. | acum | DIN 45692:2009, apartado 6 | [Calidad sonora](/phonometry/es/perception/psychoacoustics/sound-quality/) |
| $R$ | Aspereza: la sensación áspera de una modulación de amplitud rápida, en torno a 70 Hz, normalizada para que el tono modulado de referencia valga 1 asper. | asper | ECMA-418-2:2025, apartado 7 (Fórmula 104) | [Calidad sonora](/phonometry/es/perception/psychoacoustics/sound-quality/) |
| $F$ | Intensidad de fluctuación: la modulación de amplitud lenta percibida, en torno a 4 Hz, normalizada para que el tono modulado de referencia valga 1 vacil. | vacil | ECMA-418-2:2025, apartado 9 (Fórmula 163) | [Calidad sonora](/phonometry/es/perception/psychoacoustics/sound-quality/) |
| $T$ | Tonalidad: el contenido tonal percibido de un sonido, obtenido de la autocorrelación de las envolventes por banda. | tu | ECMA-418-2:2025, apartado 6 | [Calidad sonora](/phonometry/es/perception/psychoacoustics/sound-quality/) |
| TNR | Relación tono-ruido: el nivel de un tono discreto sobre el ruido enmascarante de la banda crítica que lo rodea. | dB | ECMA-418-1:2024, apartado 11 (Fórmulas 9 a 11) | [Tonos prominentes](/phonometry/es/perception/psychoacoustics/tone-prominence/) |
| PR | Relación de prominencia: el nivel de la banda crítica que contiene el tono sobre la media de las dos bandas contiguas. | dB | ECMA-418-1:2024, apartado 12 (Fórmula 23) | [Tonos prominentes](/phonometry/es/perception/psychoacoustics/tone-prominence/) |
| $\Delta L$ | Audibilidad de un tono en ruido: el nivel del tono menos el nivel de enmascaramiento de la banda crítica menos el índice de enmascaramiento. | dB | ISO/PAS 20065:2016, Fórmula 14 | [Audibilidad de tonos](/phonometry/es/perception/psychoacoustics/tone-audibility/) |
| PA | Molestia psicoacústica: la sonoridad percentil escalada por la agudeza y por un término que combina fluctuación y aspereza. | adimensional | Fastl y Zwicker (2007), Ecuación 16.2; sin norma aplicable | [Molestia psicoacústica](/phonometry/es/perception/psychoacoustics/psychoacoustic-annoyance/) |

## Electroacústica y sonoridad de programa

| Símbolo | Magnitud | Unidad | Definida en | Guía |
| --- | --- | --- | --- | --- |
| THD | Distorsión armónica total: el contenido armónico de la salida respecto al fundamental ($\mathrm{THD}_F$) o respecto a la señal total ($\mathrm{THD}_R$). | % o dB | IEC 60268-3:2013, 14.12.2 a 14.12.11 (la forma R en 14.12.3.2) | [Electroacústica](/phonometry/es/devices/electroacoustics/electroacoustics/) |
| THD+N | Distorsión armónica total más ruido: todo lo que queda tras eliminar el fundamental con el filtro de muesca, dentro del ancho de banda de medida normalizado. | % o dB | AES17-2015, apartado 6.3.1 (muesca y ancho de banda en 5.2.5 y 5.2.8) | [Electroacústica](/phonometry/es/devices/electroacoustics/electroacoustics/) |
| SINAD | Relación entre señal y ruido más distorsión, la expresión recíproca de la THD+N. | dB | AES17-2015, apartado 6.3.1 | [Electroacústica](/phonometry/es/devices/electroacoustics/electroacoustics/) |
| IMD | Distorsión de intermodulación por modulación: las bandas laterales que un tono de baja frecuencia genera alrededor de uno de alta. | % | IEC 60268-3:2013, 14.12.7 | [Electroacústica](/phonometry/es/devices/electroacoustics/electroacoustics/) |
| DIM | Distorsión de intermodulación dinámica, medida con un seno de 15 kHz frente a una onda cuadrada de 3,15 kHz filtrada. | % | IEC 60268-3:2013, 14.12.9 | [Electroacústica](/phonometry/es/devices/electroacoustics/electroacoustics/) |
| $L_K$, LUFS | Sonoridad de programa: la suma ponderada por canales de las potencias cuadráticas medias con ponderación K, con puerta en bloques de 400 ms. LUFS y LKFS nombran la misma unidad. | LUFS | ITU-R BS.1770-5, Fórmula 2 (puerta en las Fórmulas 3 a 7) | [Sonoridad de programa](/phonometry/es/devices/broadcast/program-loudness/) |
| LRA | Rango de sonoridad: la separación entre los percentiles 10 y 95 de la distribución de sonoridad de corto plazo tras la puerta. | LU | EBU Tech 3342 | [Sonoridad de programa](/phonometry/es/devices/broadcast/program-loudness/) |
| dBTP | Nivel de pico verdadero: el pico de la señal reconstruida por sobremuestreo, que capta los picos entre muestras que un máximo en el dominio de la muestra se pierde. | dBTP | ITU-R BS.1770-5, Anexo 2 | [Sonoridad de programa](/phonometry/es/devices/broadcast/program-loudness/) |

## Aeronaves y acústica submarina

| Símbolo | Magnitud | Unidad | Definida en | Guía |
| --- | --- | --- | --- | --- |
| PNL | Nivel de ruido percibido: los niveles de las 24 bandas de tercio de octava convertidos a ruidosidad en noys y recombinados. | PNdB | Anexo 16 OACI, Vol. I, Apéndice 2 (ley de ruidosidad en la Tabla A2-3) | [Ruido de aeronaves](/phonometry/es/aircraft/aircraft-noise/) |
| PNLT | Nivel de ruido percibido corregido por tonos: el PNL más la penalización por irregularidades espectrales como los tonos de ventilador y turbina. | PNdB | Anexo 16 OACI, Vol. I, Apéndice 2 | [Ruido de aeronaves](/phonometry/es/aircraft/aircraft-noise/) |
| EPNL | Nivel efectivo de ruido percibido: el PNLT máximo más la corrección por duración en la ventana de 10 dB, la métrica de certificación acústica. | EPNdB | Anexo 16 OACI, Vol. I, Apéndice 2 | [Ruido de aeronaves](/phonometry/es/aircraft/aircraft-noise/) |
| Lp (submarino) | Nivel de presión acústica submarina, referido a 1 µPa y no a 20 µPa. Un nivel aéreo nunca se convierte a él con una simple resta. | dB re 1 µPa | ISO 18405:2017 (nivel cuadrático medio en ISO 18406:2017, Fórmula 7) | [Acústica submarina](/phonometry/es/underwater/underwater-acoustics/) |
| SEL (submarino) | Nivel de exposición sonora submarina: la integral temporal de la presión al cuadrado referida a 1 µPa²·s. | dB re 1 µPa²·s | ISO 18405:2017 | [Acústica submarina](/phonometry/es/underwater/underwater-acoustics/) |
| $L_{RN}$ | Nivel de ruido radiado por un buque: el nivel del producto de la presión eficaz en campo lejano por la distancia a la fuente. | dB re 1 µPa·m | ISO 17208-1:2016 | [Acústica submarina](/phonometry/es/underwater/underwater-acoustics/) |
| $L_s$ | Nivel de fuente monopolar equivalente: el nivel de ruido radiado tras la corrección de superficie de Lloyd, de modo que un solo número describa la fuente en sí. | dB re 1 µPa·m | ISO 17208-2:2019, Fórmula 3 | [Acústica submarina](/phonometry/es/underwater/underwater-acoustics/) |

## Incertidumbre de medida

| Símbolo | Magnitud | Unidad | Definida en | Guía |
| --- | --- | --- | --- | --- |
| $u(y)$ | Incertidumbre típica combinada de un resultado, propagada desde las incertidumbres típicas de sus entradas por la ley de propagación de la incertidumbre. | unidad del resultado | ISO/IEC Guide 98-3:2008 (JCGM 100:2008), apartado 5 | [Incertidumbre GUM](/phonometry/es/signal/metrology/gum-uncertainty/) |
| $U$ | Incertidumbre expandida: la incertidumbre típica combinada multiplicada por un factor de cobertura, que define un intervalo de cobertura. | unidad del resultado | ISO/IEC Guide 98-3:2008 (JCGM 100:2008), apartado 6 y Anexo G | [Incertidumbre GUM](/phonometry/es/signal/metrology/gum-uncertainty/) |
| $\sigma_R$ | Incertidumbre típica de una situación de medida en acústica de la edificación, tabulada por magnitud y por situación. | dB | ISO 12999-1:2020, apartado 5.2 (factores de cobertura en la Tabla 8) | [Aislamiento en campo](/phonometry/es/buildings/insulation/insulation-field/) |
