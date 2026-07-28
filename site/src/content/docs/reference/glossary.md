---
title: "Glossary"
description: "Every acoustic quantity the guides compute, with its symbol, a one-sentence definition, its unit, the standard and clause that defines it, and the guide that implements it."
head:
  - tag: script
    attrs:
      type: application/ld+json
    content: |
      {
        "@context": "https://schema.org",
        "@type": "DefinedTermSet",
        "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary",
        "name": "phonometry acoustic glossary",
        "description": "Acoustic quantities computed by the phonometry library, each with its symbol, unit, defining standard and clause, and the guide that implements it.",
        "url": "https://jmrplens.github.io/phonometry/reference/glossary/",
        "inLanguage": "en",
        "hasDefinedTerm": [
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-1",
            "name": "Lp",
            "termCode": "Lp",
            "description": "Sound pressure level: twenty times the base-10 logarithm of the r.m.s. sound pressure over the reference pressure. Unit: dB re 20 µPa. Defined in: IEC 61672-1:2013.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Calibration",
              "url": "https://jmrplens.github.io/phonometry/guides/calibration/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-2",
            "name": "Leq",
            "termCode": "Leq",
            "description": "Equivalent continuous sound pressure level: the level of the steady sound carrying the same mean-square pressure over the interval. Unit: dB. Defined in: IEC 61672-1:2013.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Levels",
              "url": "https://jmrplens.github.io/phonometry/guides/levels/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-3",
            "name": "LAeq",
            "termCode": "LAeq",
            "description": "The same integral applied to the A-weighted signal, the default descriptor of environmental and occupational noise. Unit: dB. Defined in: IEC 61672-1:2013.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Levels",
              "url": "https://jmrplens.github.io/phonometry/guides/levels/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-4",
            "name": "LAE, SEL",
            "termCode": "LAE, SEL",
            "description": "Sound exposure level: the whole A-weighted energy of a single event normalised to one second. Unit: dB. Defined in: IEC 61672-1:2013, Equation 8 (Table 4).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Levels",
              "url": "https://jmrplens.github.io/phonometry/guides/levels/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-5",
            "name": "LCpeak",
            "termCode": "LCpeak",
            "description": "C-weighted peak sound level: the absolute maximum of the C-weighted pressure, not a time-weighted maximum. Unit: dB. Defined in: IEC 61672-1:2013, subclause 5.13.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Levels",
              "url": "https://jmrplens.github.io/phonometry/guides/levels/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-6",
            "name": "LN (L10, L50, L90)",
            "termCode": "LN (L10, L50, L90)",
            "description": "Percentile level: the level exceeded N % of the measurement time, read off the time-weighted level distribution. Unit: dB. Defined in: ISO 1996-2:2017 (Annex I uses L90 as the residual level).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Levels",
              "url": "https://jmrplens.github.io/phonometry/guides/levels/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-7",
            "name": "LW, SWL",
            "termCode": "LW, SWL",
            "description": "Sound power level: the power a source radiates, referred to 1 pW. Unit: dB re 1 pW. Defined in: ISO 3745:2012, Clause 8.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Sound power",
              "url": "https://jmrplens.github.io/phonometry/guides/sound-power/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-8",
            "name": "LI",
            "termCode": "LI",
            "description": "Sound intensity level: the magnitude of the intensity vector referred to 1 pW/m², with the flow direction reported separately as a sign. Unit: dB re 1 pW/m². Defined in: IEC 61043:1993.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Intensity",
              "url": "https://jmrplens.github.io/phonometry/guides/intensity/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-9",
            "name": "Lp − LI",
            "termCode": "Lp − LI",
            "description": "Pressure-intensity index: the difference between the pressure and intensity levels at a position, the field indicator that qualifies an intensity measurement. Unit: dB. Defined in: ISO 9614-1:1993, Equation (A.3).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Intensity",
              "url": "https://jmrplens.github.io/phonometry/guides/intensity/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-10",
            "name": "Lden",
            "termCode": "Lden",
            "description": "Day-evening-night level: the energy mean of the three periods with 5 dB added to the evening and 10 dB to the night. Unit: dB. Defined in: ISO 1996-1:2016, 3.6.4.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Levels",
              "url": "https://jmrplens.github.io/phonometry/guides/levels/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-11",
            "name": "Ldn",
            "termCode": "Ldn",
            "description": "Day-night level: the same construction with the 10 dB night penalty only. Unit: dB. Defined in: ISO 1996-1:2016, 3.6.5.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Levels",
              "url": "https://jmrplens.github.io/phonometry/guides/levels/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-12",
            "name": "Lr",
            "termCode": "Lr",
            "description": "Rating level: the whole-day composite level after the source-character and time-of-day adjustments. Unit: dB. Defined in: ISO 1996-1:2016, clause 6.5 (Formulae 5 and 6).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Levels",
              "url": "https://jmrplens.github.io/phonometry/guides/levels/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-13",
            "name": "LAr,T",
            "termCode": "LAr,T",
            "description": "Rating level of an impulsive source over a reference interval, LAeq plus the graduated impulse adjustment. Unit: dB. Defined in: NT ACOU 112:2002, clause 8.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Impulse prominence",
              "url": "https://jmrplens.github.io/phonometry/guides/impulse-prominence/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-14",
            "name": "KI",
            "termCode": "KI",
            "description": "Impulse adjustment added to LAeq, graduated by the predicted prominence of the impulses. Unit: dB. Defined in: NT ACOU 112:2002, clause 8.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Impulse prominence",
              "url": "https://jmrplens.github.io/phonometry/guides/impulse-prominence/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-15",
            "name": "E",
            "termCode": "E",
            "description": "Sound exposure: the time integral of the squared A-weighted sound pressure over the exposure period. Unit: Pa²h. Defined in: IEC 61252:1993, 3.1.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Levels",
              "url": "https://jmrplens.github.io/phonometry/guides/levels/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-16",
            "name": "LEX,8h, LEP,d",
            "termCode": "LEX,8h, LEP,d",
            "description": "Daily noise exposure level: the steady level that, sustained over a nominal 8 h day, carries the same A-weighted sound exposure as the measured one. Unit: dB. Defined in: IEC 61252:1993, 3.3.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Occupational exposure",
              "url": "https://jmrplens.github.io/phonometry/guides/occupational-exposure/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-17",
            "name": "Lp,A,eqT",
            "termCode": "Lp,A,eqT",
            "description": "A-weighted equivalent continuous level of a task, a job sample or a full day, the building block LEX,8h is assembled from. Unit: dB. Defined in: ISO 9612:2009, clauses 9 to 11.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Occupational exposure",
              "url": "https://jmrplens.github.io/phonometry/guides/occupational-exposure/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-18",
            "name": "NIPTS",
            "termCode": "NIPTS",
            "description": "Noise-induced permanent threshold shift: the median hearing loss attributable to a stated exposure level, duration and audiometric frequency. Unit: dB. Defined in: ISO 1999:2013.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Noise-induced hearing loss",
              "url": "https://jmrplens.github.io/phonometry/guides/noise-induced-hearing-loss/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-19",
            "name": "HTLAN",
            "termCode": "HTLAN",
            "description": "Hearing threshold level associated with age and noise: the NIPTS combined with the age component. Unit: dB. Defined in: ISO 1999:2013.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Noise-induced hearing loss",
              "url": "https://jmrplens.github.io/phonometry/guides/noise-induced-hearing-loss/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-20",
            "name": "A, C, Z",
            "termCode": "A, C, Z",
            "description": "The normative frequency weightings: the ear-response curves applied before integration, Z being the flat reference. Unit: dB. Defined in: IEC 61672-1:2013, Annex E (acceptance limits in Table 3).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Weighting",
              "url": "https://jmrplens.github.io/phonometry/guides/weighting/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-21",
            "name": "G",
            "termCode": "G",
            "description": "Infrasound weighting, defined by its poles and zeros for the 0.25 Hz to 315 Hz range. Unit: dB. Defined in: ISO 7196:1995, Table 1 (nominal responses in Table 2).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Weighting",
              "url": "https://jmrplens.github.io/phonometry/guides/weighting/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-22",
            "name": "B",
            "termCode": "B",
            "description": "Historical mid-level weighting, withdrawn from the current meter standard. Unit: dB. Defined in: ANSI S1.4-1983, Appendix C (Formula C2).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Weighting",
              "url": "https://jmrplens.github.io/phonometry/guides/weighting/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-23",
            "name": "D",
            "termCode": "D",
            "description": "Historical aircraft-noise weighting, derived from the 40-noy perceived-noisiness contour. Unit: dB. Defined in: IEC 537:1976 (withdrawn).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Weighting",
              "url": "https://jmrplens.github.io/phonometry/guides/weighting/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-24",
            "name": "AU",
            "termCode": "AU",
            "description": "Weighting for audible sound measured in the presence of ultrasound. Unit: dB. Defined in: IEC 61012:1990, subclause 2.2 (Tables 1 and 2).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Weighting",
              "url": "https://jmrplens.github.io/phonometry/guides/weighting/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-25",
            "name": "F, S, I",
            "termCode": "F, S, I",
            "description": "Fast, Slow and Impulse exponential time weightings: the detector ballistics that produce a displayed level. Unit: s (time constant). Defined in: IEC 61672-1:2013.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Time weighting",
              "url": "https://jmrplens.github.io/phonometry/guides/time-weighting/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-26",
            "name": "T20",
            "termCode": "T20",
            "description": "Reverberation time extrapolated to a 60 dB decay from a least-squares fit over −5 dB to −25 dB of the Schroeder curve. Unit: s. Defined in: ISO 3382-2:2008, Clause 6 and Annex C.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Room acoustics",
              "url": "https://jmrplens.github.io/phonometry/guides/room-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-27",
            "name": "T30",
            "termCode": "T30",
            "description": "The same extrapolation from a fit over −5 dB to −35 dB, the usual choice when the decay range allows it. Unit: s. Defined in: ISO 3382-2:2008, Clause 6 and Annex C.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Room acoustics",
              "url": "https://jmrplens.github.io/phonometry/guides/room-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-28",
            "name": "T60, RT",
            "termCode": "T60, RT",
            "description": "Reverberation time as such: the time for the sound energy to fall by 60 dB. Measured in practice as T20 or T30. Unit: s. Defined in: ISO 3382-1:2009.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Room acoustics",
              "url": "https://jmrplens.github.io/phonometry/guides/room-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-29",
            "name": "EDT",
            "termCode": "EDT",
            "description": "Early decay time: the same slope taken over the first 10 dB of decay, which tracks perceived reverberance rather than the tail. Unit: s. Defined in: ISO 3382-1:2009 (just-noticeable difference in Table A.1).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Room acoustics",
              "url": "https://jmrplens.github.io/phonometry/guides/room-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-30",
            "name": "C50",
            "termCode": "C50",
            "description": "Clarity for speech: the energy ratio between the first 50 ms of the impulse response and everything after it. Unit: dB. Defined in: ISO 3382-1:2009.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Room acoustics",
              "url": "https://jmrplens.github.io/phonometry/guides/room-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-31",
            "name": "C80",
            "termCode": "C80",
            "description": "Clarity for music: the same ratio with the boundary at 80 ms. Unit: dB. Defined in: ISO 3382-1:2009 (just-noticeable difference in Table A.1).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Room acoustics",
              "url": "https://jmrplens.github.io/phonometry/guides/room-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-32",
            "name": "D50",
            "termCode": "D50",
            "description": "Definition, or Deutlichkeit: the fraction of the total energy arriving in the first 50 ms. Unit: dimensionless. Defined in: ISO 3382-1:2009 (just-noticeable difference in Table A.1).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Room acoustics",
              "url": "https://jmrplens.github.io/phonometry/guides/room-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-33",
            "name": "Ts",
            "termCode": "Ts",
            "description": "Centre time: the centre of gravity of the squared impulse response in time, a boundary-free alternative to the clarity indices. Unit: s. Defined in: ISO 3382-1:2009, Equation (A.13).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Room acoustics",
              "url": "https://jmrplens.github.io/phonometry/guides/room-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-34",
            "name": "A",
            "termCode": "A",
            "description": "Equivalent sound absorption area of a room: the area of a perfectly absorbing surface that would give the same reverberation time. Unit: m². Defined in: ISO 354:2003, Equations (5) and (7).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Absorption measurement",
              "url": "https://jmrplens.github.io/phonometry/guides/absorption-measurement/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-35",
            "name": "NC",
            "termCode": "NC",
            "description": "Noise criteria rating of a background spectrum: the speech interference level selects the curve, and the tangency method rates the spectrum when a band exceeds it. Unit: dB (index). Defined in: ANSI/ASA S12.2-2019, 5.2.2 and 5.2.3 (curves in Table 1).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Room noise",
              "url": "https://jmrplens.github.io/phonometry/guides/room-noise/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-36",
            "name": "SIL",
            "termCode": "SIL",
            "description": "Speech interference level: the average of the 500, 1000, 2000 and 4000 Hz octave-band levels. Unit: dB. Defined in: ANSI/ASA S12.2-2019, clause 3.2.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Room noise",
              "url": "https://jmrplens.github.io/phonometry/guides/room-noise/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-37",
            "name": "RC",
            "termCode": "RC",
            "description": "Room criteria Mark II rating: the average of the 500, 1000 and 2000 Hz levels, with a rumble, hiss or neutral spectral tag. Unit: dB (index). Defined in: ANSI/ASA S12.2-2019, Annex D (clauses D.3 and D.4).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Room noise",
              "url": "https://jmrplens.github.io/phonometry/guides/room-noise/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-38",
            "name": "NR",
            "termCode": "NR",
            "description": "Noise rating, the European counterpart curve family of NC. Discussed for comparison and deliberately not implemented. Unit: dB (index). Defined in: Kosten and van Os (1962); no governing standard.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Room noise",
              "url": "https://jmrplens.github.io/phonometry/guides/room-noise/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-39",
            "name": "m(F)",
            "termCode": "m(F)",
            "description": "Modulation transfer function: the fraction of the speech envelope modulation depth at modulation frequency F that survives the transmission path. Unit: dimensionless. Defined in: IEC 60268-16:2020.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Speech transmission",
              "url": "https://jmrplens.github.io/phonometry/guides/speech-transmission/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-40",
            "name": "STI",
            "termCode": "STI",
            "description": "Speech transmission index: the modulation transfer matrix converted to effective signal-to-noise ratios and weighted into a single value on 0 to 1. Unit: dimensionless. Defined in: IEC 60268-16:2020, A.5.2 to A.5.6.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Speech transmission",
              "url": "https://jmrplens.github.io/phonometry/guides/speech-transmission/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-41",
            "name": "STIPA",
            "termCode": "STIPA",
            "description": "The direct STI measurement, made by playing a standardised two-modulation-per-band test signal through the real chain. Unit: dimensionless. Defined in: IEC 60268-16:2020, clause 6.3 and Table 3 (direct method, Annex B).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Speech transmission",
              "url": "https://jmrplens.github.io/phonometry/guides/speech-transmission/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-42",
            "name": "SII",
            "termCode": "SII",
            "description": "Speech intelligibility index: the band-importance-weighted audibility of the speech spectrum against noise and the listener's threshold. Unit: dimensionless. Defined in: ANSI S3.5-1997, clause 6 (procedure in clause 5, importance function in Table 3).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Speech intelligibility",
              "url": "https://jmrplens.github.io/phonometry/guides/speech-intelligibility/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-43",
            "name": "STOI",
            "termCode": "STOI",
            "description": "Short-time objective intelligibility: the clipped per-band envelope correlation between clean and degraded speech. Unit: dimensionless. Defined in: Taal et al. (2011), Equations 5 and 6; no governing standard.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Objective intelligibility",
              "url": "https://jmrplens.github.io/phonometry/guides/objective-intelligibility/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-44",
            "name": "ESTOI",
            "termCode": "ESTOI",
            "description": "The extended measure, row- and column-normalised so that it tracks modulated maskers. Unit: dimensionless. Defined in: Jensen and Taal (2016), Equation 8; no governing standard.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Objective intelligibility",
              "url": "https://jmrplens.github.io/phonometry/guides/objective-intelligibility/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-45",
            "name": "D",
            "termCode": "D",
            "description": "Level difference: the energy-averaged source-room level minus the receiving-room level, with no normalisation. Unit: dB. Defined in: ISO 16283-1:2014, 3.12 to 3.15.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Field insulation",
              "url": "https://jmrplens.github.io/phonometry/guides/insulation-field/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-46",
            "name": "DnT",
            "termCode": "DnT",
            "description": "Standardized level difference: the level difference referred to a reference reverberation time, 0.5 s for dwellings. Unit: dB. Defined in: ISO 16283-1:2014, 3.12 to 3.15.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Field insulation",
              "url": "https://jmrplens.github.io/phonometry/guides/insulation-field/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-47",
            "name": "Dn",
            "termCode": "Dn",
            "description": "Normalized level difference: the level difference referred to a reference absorption area of 10 m². Unit: dB. Defined in: ISO 10052:2021.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Field insulation",
              "url": "https://jmrplens.github.io/phonometry/guides/insulation-field/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-48",
            "name": "Dn,e",
            "termCode": "Dn,e",
            "description": "Element-normalized level difference of a small element or air path, referred to a reference area of 10 m². Unit: dB. Defined in: EN 12354-3:2000.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Insulation prediction",
              "url": "https://jmrplens.github.io/phonometry/guides/insulation-prediction/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-49",
            "name": "R",
            "termCode": "R",
            "description": "Sound reduction index: the level difference corrected by the partition area over the receiving-room absorption area, measured in the laboratory with flanking suppressed. Unit: dB. Defined in: ISO 10140-2:2010.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Laboratory insulation",
              "url": "https://jmrplens.github.io/phonometry/guides/insulation-lab/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-50",
            "name": "R′",
            "termCode": "R′",
            "description": "Apparent sound reduction index: the same construction measured in the building, so it includes every flanking path. The prime is the lab-versus-field marker. Unit: dB. Defined in: ISO 16283-1:2014, 3.12 to 3.15.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Field insulation",
              "url": "https://jmrplens.github.io/phonometry/guides/insulation-field/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-51",
            "name": "TL",
            "termCode": "TL",
            "description": "Transmission loss: the airborne insulation of a panel predicted from its physical properties, the same quantity as R in a prediction context. Unit: dB. Defined in: Bies, Hansen and Howard (2017), Section 7.2; no governing standard.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Panel insulation",
              "url": "https://jmrplens.github.io/phonometry/guides/panel-sound-insulation/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-52",
            "name": "Rw, R′w, DnT,w",
            "termCode": "Rw, R′w, DnT,w",
            "description": "The weighted single-number ratings: a fixed reference curve is shifted toward the measured spectrum until the unfavourable deviations reach their allowed sum, and the shifted curve is read at 500 Hz. Unit: dB. Defined in: ISO 717-1:2020.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Field insulation",
              "url": "https://jmrplens.github.io/phonometry/guides/insulation-field/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-53",
            "name": "Dn,e,w",
            "termCode": "Dn,e,w",
            "description": "The same reference-curve rating applied to the element-normalized level difference. Unit: dB. Defined in: ISO 717-1:2020.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Insulation prediction",
              "url": "https://jmrplens.github.io/phonometry/guides/insulation-prediction/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-54",
            "name": "C, Ctr",
            "termCode": "C, Ctr",
            "description": "Spectrum adaptation terms: the corrections that re-rate the measured curve against A-weighted pink noise (C) and against A-weighted urban road traffic (Ctr). Unit: dB. Defined in: ISO 717-1:2020, Annex A.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Field insulation",
              "url": "https://jmrplens.github.io/phonometry/guides/insulation-field/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-55",
            "name": "Ln",
            "termCode": "Ln",
            "description": "Normalized impact sound pressure level: the receiving-room level under the standard tapping machine, referred to a 10 m² absorption area. Unit: dB. Defined in: ISO 10140-3:2010.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Laboratory insulation",
              "url": "https://jmrplens.github.io/phonometry/guides/insulation-lab/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-56",
            "name": "L′nT",
            "termCode": "L′nT",
            "description": "Standardized impact sound pressure level, referred to a reference reverberation time. Note the sign: more reverberation lowers it, the opposite of DnT. Unit: dB. Defined in: ISO 16283-2:2015.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Field insulation",
              "url": "https://jmrplens.github.io/phonometry/guides/insulation-field/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-57",
            "name": "Ln,w, L′nT,w",
            "termCode": "Ln,w, L′nT,w",
            "description": "The weighted impact ratings. The reference curve is shifted the same way, but an unfavourable deviation is now one where the measurement exceeds the reference. Unit: dB. Defined in: ISO 717-2:2020.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Field insulation",
              "url": "https://jmrplens.github.io/phonometry/guides/insulation-field/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-58",
            "name": "CI",
            "termCode": "CI",
            "description": "Impact spectrum adaptation term, from the energetic sum over 100 Hz to 2500 Hz. The enlarged-range CI,50-2500 extends it down to 50 Hz. Unit: dB. Defined in: ISO 717-2:2020 (enlarged range in A.2.1 NOTE).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Field insulation",
              "url": "https://jmrplens.github.io/phonometry/guides/insulation-field/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-59",
            "name": "ΔLw",
            "termCode": "ΔLw",
            "description": "Weighted reduction of impact sound pressure level given by a floor covering, measured as the improvement over the bare reference floor. Unit: dB. Defined in: ISO 717-2:2020 (measurement in ISO 16251-1:2014, Formulae (3) and (4)).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Laboratory insulation",
              "url": "https://jmrplens.github.io/phonometry/guides/insulation-lab/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-60",
            "name": "ΔRw",
            "termCode": "ΔRw",
            "description": "Weighted improvement of airborne insulation contributed by a lining or additional layer, added to the element rating in the prediction. Unit: dB. Defined in: EN 12354-1:2000, Formulae 27 and 28a.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Insulation prediction",
              "url": "https://jmrplens.github.io/phonometry/guides/insulation-prediction/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-61",
            "name": "Kij",
            "termCode": "Kij",
            "description": "Vibration reduction index of a junction: the direction-averaged velocity level difference corrected by the junction length and the equivalent absorption lengths. Unit: dB. Defined in: ISO 10848-1:2006, Formula (13).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Laboratory insulation",
              "url": "https://jmrplens.github.io/phonometry/guides/insulation-lab/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-62",
            "name": "fc",
            "termCode": "fc",
            "description": "Critical frequency: the frequency at which the bending wavelength of a panel equals the wavelength in air, where the coincidence dip appears. Unit: Hz. Defined in: Bies, Hansen and Howard (2017), Equation 7.3; no governing standard.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Panel insulation",
              "url": "https://jmrplens.github.io/phonometry/guides/panel-sound-insulation/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-63",
            "name": "σ",
            "termCode": "σ",
            "description": "Radiation efficiency of a plate: the airborne power radiated per unit mean-square surface velocity, normalised by the plane-wave value. Unit: dimensionless. Defined in: Hopkins (2007), Equations 2.227 to 2.230; no governing standard.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Panel insulation",
              "url": "https://jmrplens.github.io/phonometry/guides/panel-sound-insulation/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-64",
            "name": "α",
            "termCode": "α",
            "description": "Sound absorption coefficient at normal incidence: the fraction of incident energy not returned by the surface, obtained in the impedance tube from the reflection factor. Unit: dimensionless. Defined in: ISO 10534-2:1998, Equations (17) to (19).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Impedance tube",
              "url": "https://jmrplens.github.io/phonometry/guides/impedance-tube/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-65",
            "name": "αs",
            "termCode": "αs",
            "description": "Random-incidence sound absorption coefficient measured in a reverberation room, from the change in equivalent absorption area with and without the specimen. Unit: dimensionless. Defined in: ISO 354:2003, Equations (8) and (9).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Absorption measurement",
              "url": "https://jmrplens.github.io/phonometry/guides/absorption-measurement/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-66",
            "name": "αp",
            "termCode": "αp",
            "description": "Practical sound absorption coefficient: the one-third-octave data grouped into octave bands and rounded to steps of 0.05. Unit: dimensionless. Defined in: ISO 11654:1997, Clause 4.1.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Absorption measurement",
              "url": "https://jmrplens.github.io/phonometry/guides/absorption-measurement/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-67",
            "name": "αw",
            "termCode": "αw",
            "description": "Weighted sound absorption coefficient: the fixed reference curve shifted toward the practical values and read at 500 Hz. Unit: dimensionless. Defined in: ISO 11654:1997, Clause 4.2.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Absorption measurement",
              "url": "https://jmrplens.github.io/phonometry/guides/absorption-measurement/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-68",
            "name": "Absorption class",
            "termCode": "Absorption class",
            "description": "The A to E letter class the weighted coefficient maps to, or \"not classified\". Unit: class letter. Defined in: ISO 11654:1997, Table B.1.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Absorption measurement",
              "url": "https://jmrplens.github.io/phonometry/guides/absorption-measurement/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-69",
            "name": "R",
            "termCode": "R",
            "description": "Airflow resistance: the pressure difference across a specimen divided by the volumetric airflow rate through it. Unit: Pa·s/m³. Defined in: ISO 9053-1:2018, Clause 3.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Airflow resistance",
              "url": "https://jmrplens.github.io/phonometry/guides/airflow-resistance/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-70",
            "name": "Rs",
            "termCode": "Rs",
            "description": "Specific airflow resistance: the airflow resistance referred to the specimen face area. Unit: Pa·s/m. Defined in: ISO 9053-1:2018, Clause 3.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Airflow resistance",
              "url": "https://jmrplens.github.io/phonometry/guides/airflow-resistance/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-71",
            "name": "σ",
            "termCode": "σ",
            "description": "Airflow resistivity: the specific airflow resistance per unit thickness, the primary input to every empirical porous model. Unit: Pa·s/m². Defined in: ISO 9053-1:2018, Clause 3.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Airflow resistance",
              "url": "https://jmrplens.github.io/phonometry/guides/airflow-resistance/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-72",
            "name": "Z",
            "termCode": "Z",
            "description": "Surface impedance: the complex ratio of sound pressure to particle velocity at the face of the sample, usually reported normalised by the characteristic impedance of air. Unit: Pa·s/m. Defined in: ISO 10534-2:1998, Equations (17) to (19).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Impedance tube",
              "url": "https://jmrplens.github.io/phonometry/guides/impedance-tube/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-73",
            "name": "s",
            "termCode": "s",
            "description": "Scattering coefficient: the fraction of reflected energy that is not returned specularly, measured at random incidence on a turntable in a reverberation room. Unit: dimensionless. Defined in: ISO 17497-1:2004+A1:2014, Formula (5).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Surface scattering",
              "url": "https://jmrplens.github.io/phonometry/guides/surface-scattering/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-74",
            "name": "d",
            "termCode": "d",
            "description": "Diffusion coefficient: the uniformity of the polar response of a surface, from the autocorrelation of the free-field goniometer measurement. Unit: dimensionless. Defined in: ISO 17497-2:2012, Formula (5) (normalised form in Formula (7)).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Surface scattering",
              "url": "https://jmrplens.github.io/phonometry/guides/surface-scattering/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-75",
            "name": "s′",
            "termCode": "s′",
            "description": "Dynamic stiffness per unit area of a resilient layer: a dynamic force per unit area divided by the resulting change in thickness. Unit: MN/m³. Defined in: EN 29052-1:1992 (ISO 9052-1:1989), Formula 1.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Dynamic stiffness",
              "url": "https://jmrplens.github.io/phonometry/guides/dynamic-stiffness/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-76",
            "name": "Y",
            "termCode": "Y",
            "description": "Mobility: the complex ratio of a velocity response to the force that produces it. Unit: m/(N·s). Defined in: ISO 7626-1:2011, 3.1.2 and Table 1.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Mechanical mobility",
              "url": "https://jmrplens.github.io/phonometry/guides/mechanical-mobility/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-77",
            "name": "Z",
            "termCode": "Z",
            "description": "Mechanical impedance: the reciprocal of mobility, force per unit velocity. Unit: N·s/m. Defined in: ISO 7626-1:2011, Table 1.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Mechanical mobility",
              "url": "https://jmrplens.github.io/phonometry/guides/mechanical-mobility/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-78",
            "name": "H",
            "termCode": "H",
            "description": "Receptance, or dynamic compliance: displacement response per unit force, the pivot the whole family converts through. Unit: m/N. Defined in: ISO 7626-1:2011, Table 1.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Mechanical mobility",
              "url": "https://jmrplens.github.io/phonometry/guides/mechanical-mobility/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-79",
            "name": "A",
            "termCode": "A",
            "description": "Accelerance, or inertance: acceleration response per unit force. Its reciprocal is the apparent mass. Unit: 1/kg. Defined in: ISO 7626-1:2011, Table 1.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Mechanical mobility",
              "url": "https://jmrplens.github.io/phonometry/guides/mechanical-mobility/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-80",
            "name": "k21",
            "termCode": "k21",
            "description": "Dynamic transfer stiffness of a resilient element: the blocking force on the output side divided by the displacement on the input side. Unit: N/m. Defined in: ISO 10846-1:2008, 3.7.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Transfer stiffness",
              "url": "https://jmrplens.github.io/phonometry/guides/transfer-stiffness/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-81",
            "name": "Lk",
            "termCode": "Lk",
            "description": "Level of the dynamic transfer stiffness, referred to 1 N/m. Unit: dB re 1 N/m. Defined in: ISO 10846-2:2008 and ISO 10846-3:2002, 3.17.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Transfer stiffness",
              "url": "https://jmrplens.github.io/phonometry/guides/transfer-stiffness/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-82",
            "name": "η",
            "termCode": "η",
            "description": "Loss factor of a resilient element: the tangent of the phase angle of its dynamic transfer stiffness. Unit: dimensionless. Defined in: ISO 10846-1:2008, 3.8.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Transfer stiffness",
              "url": "https://jmrplens.github.io/phonometry/guides/transfer-stiffness/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-83",
            "name": "aw",
            "termCode": "aw",
            "description": "Frequency-weighted acceleration: the root sum of squares of the band accelerations after the human-response weightings. Unit: m/s². Defined in: ISO 2631-1:1997, Equation (9).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Human vibration",
              "url": "https://jmrplens.github.io/phonometry/guides/human-vibration/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-84",
            "name": "A(8)",
            "termCode": "A(8)",
            "description": "Daily vibration exposure: the exposure magnitude normalised to a reference 8 h day, combined over the operations of the day. Unit: m/s². Defined in: ISO 5349-1:2001, Equations (2) and (3).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Human vibration",
              "url": "https://jmrplens.github.io/phonometry/guides/human-vibration/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-85",
            "name": "VDV",
            "termCode": "VDV",
            "description": "Vibration dose value: the fourth-power time integral of the weighted acceleration, which weights shocks far more heavily than an r.m.s. does. Unit: m/s^1.75. Defined in: ISO 2631-1:1997, Equation (5).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Human vibration",
              "url": "https://jmrplens.github.io/phonometry/guides/human-vibration/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-86",
            "name": "MTVV",
            "termCode": "MTVV",
            "description": "Maximum transient vibration value: the largest 1 s running r.m.s. of the weighted acceleration. Unit: m/s². Defined in: ISO 2631-1:1997, Equation (4).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Human vibration",
              "url": "https://jmrplens.github.io/phonometry/guides/human-vibration/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-87",
            "name": "R",
            "termCode": "R",
            "description": "Cumulative stress variable of the multiple-shock model: the daily compressive stresses accumulated over the years of exposure, which the lumbar injury probability is read from. Unit: dimensionless. Defined in: ISO 2631-5:2018, Annex C (Formulae C.1 and C.3 to C.5).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Multiple-shock vibration",
              "url": "https://jmrplens.github.io/phonometry/guides/multiple-shock-vibration/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-88",
            "name": "Lv",
            "termCode": "Lv",
            "description": "Velocity level: twenty times the base-10 logarithm of the surface velocity over the reference velocity. Unit: dB. Defined in: ISO/TS 7849-1:2009, Formula 3.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Vibration sound power",
              "url": "https://jmrplens.github.io/phonometry/guides/vibration-sound-power/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-89",
            "name": "ε",
            "termCode": "ε",
            "description": "Radiation factor, or radiation efficiency, of a vibrating machine surface: the airborne power radiated per unit mean-square velocity and area. Unit: dimensionless. Defined in: ISO/TS 7849-1:2009 and ISO/TS 7849-2:2009.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Vibration sound power",
              "url": "https://jmrplens.github.io/phonometry/guides/vibration-sound-power/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-90",
            "name": "LWs",
            "termCode": "LWs",
            "description": "Structure-borne sound power level injected by equipment into a reception plate. Unit: dB re 1 pW. Defined in: EN 15657:2018, Formula 14.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Structure-borne power",
              "url": "https://jmrplens.github.io/phonometry/guides/structure-borne-power/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-91",
            "name": "ηij",
            "termCode": "ηij",
            "description": "Coupling loss factor: the fraction of energy per radian that a statistical energy analysis subsystem loses into a neighbouring one across a junction. Unit: dimensionless. Defined in: Hopkins (2007), Equation 2.154; no governing standard.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Junction transmission",
              "url": "https://jmrplens.github.io/phonometry/guides/junction-transmission/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-92",
            "name": "N",
            "termCode": "N",
            "description": "Loudness: the perceived magnitude of a sound, anchored so that a 1 kHz tone at 40 dB SPL is exactly 1 sone. Unit: sone. Defined in: ISO 532-1:2017, clause 5 (stationary) and clause 6 (time-varying).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Loudness",
              "url": "https://jmrplens.github.io/phonometry/guides/loudness/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-93",
            "name": "N′",
            "termCode": "N′",
            "description": "Specific loudness: the loudness density along the critical-band scale, whose integral is N. Unit: sone/Bark. Defined in: ISO 532-1:2017 (sone/Cam form in ISO 532-2:2017, Formula 7).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Loudness",
              "url": "https://jmrplens.github.io/phonometry/guides/loudness/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-94",
            "name": "LN",
            "termCode": "LN",
            "description": "Loudness level: the level of the 1 kHz free-field tone judged equally loud as the sound. Unit: phon. Defined in: ISO 226:2023, Formula (2) (contours in Formula (1)).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Loudness",
              "url": "https://jmrplens.github.io/phonometry/guides/loudness/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-95",
            "name": "S",
            "termCode": "S",
            "description": "Sharpness: the position of the centre of gravity of the specific loudness on the critical-band scale, normalised so that the reference narrow-band noise is exactly 1 acum. Unit: acum. Defined in: DIN 45692:2009, clause 6.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Sound quality",
              "url": "https://jmrplens.github.io/phonometry/guides/sound-quality/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-96",
            "name": "R",
            "termCode": "R",
            "description": "Roughness: the perceived harshness of fast amplitude modulation, around 70 Hz, normalised so that the reference modulated tone is 1 asper. Unit: asper. Defined in: ECMA-418-2:2025, clause 7 (Formula 104).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Sound quality",
              "url": "https://jmrplens.github.io/phonometry/guides/sound-quality/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-97",
            "name": "F",
            "termCode": "F",
            "description": "Fluctuation strength: the perceived slow amplitude modulation, around 4 Hz, normalised so that the reference modulated tone is 1 vacil. Unit: vacil. Defined in: ECMA-418-2:2025, clause 9 (Formula 163).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Sound quality",
              "url": "https://jmrplens.github.io/phonometry/guides/sound-quality/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-98",
            "name": "T",
            "termCode": "T",
            "description": "Tonality: the perceived tonal content of a sound, derived from the autocorrelation of the band envelopes. Unit: tu. Defined in: ECMA-418-2:2025, clause 6.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Sound quality",
              "url": "https://jmrplens.github.io/phonometry/guides/sound-quality/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-99",
            "name": "TNR",
            "termCode": "TNR",
            "description": "Tone-to-noise ratio: the level of a discrete tone above the masking noise in the critical band around it. Unit: dB. Defined in: ECMA-418-1:2024, clause 11 (Formulae 9 to 11).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Tone prominence",
              "url": "https://jmrplens.github.io/phonometry/guides/tone-prominence/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-100",
            "name": "PR",
            "termCode": "PR",
            "description": "Prominence ratio: the level of the critical band containing the tone above the mean of the two adjacent bands. Unit: dB. Defined in: ECMA-418-1:2024, clause 12 (Formula 23).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Tone prominence",
              "url": "https://jmrplens.github.io/phonometry/guides/tone-prominence/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-101",
            "name": "ΔL",
            "termCode": "ΔL",
            "description": "Audibility of a tone in noise: the tone level minus the critical-band masking level minus the masking index. Unit: dB. Defined in: ISO/PAS 20065:2016, Formula 14.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Tone audibility",
              "url": "https://jmrplens.github.io/phonometry/guides/tone-audibility/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-102",
            "name": "PA",
            "termCode": "PA",
            "description": "Psychoacoustic annoyance: the percentile loudness scaled by sharpness and by a fluctuation-plus-roughness term. Unit: dimensionless. Defined in: Fastl and Zwicker (2007), Equation 16.2; no governing standard.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Psychoacoustic annoyance",
              "url": "https://jmrplens.github.io/phonometry/guides/psychoacoustic-annoyance/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-103",
            "name": "THD",
            "termCode": "THD",
            "description": "Total harmonic distortion: the harmonic content of the output relative to the fundamental (THD_F) or to the total signal (THD_R). Unit: % or dB. Defined in: IEC 60268-3:2013, 14.12.2 to 14.12.11 (the R form in 14.12.3.2).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Electroacoustics",
              "url": "https://jmrplens.github.io/phonometry/guides/electroacoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-104",
            "name": "THD+N",
            "termCode": "THD+N",
            "description": "Total harmonic distortion plus noise: everything left after notching out the fundamental, within the standard measurement bandwidth. Unit: % or dB. Defined in: AES17-2015, clause 6.3.1 (notch and bandwidth in 5.2.5 and 5.2.8).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Electroacoustics",
              "url": "https://jmrplens.github.io/phonometry/guides/electroacoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-105",
            "name": "SINAD",
            "termCode": "SINAD",
            "description": "Signal to noise and distortion ratio, the reciprocal expression of THD+N. Unit: dB. Defined in: AES17-2015, clause 6.3.1.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Electroacoustics",
              "url": "https://jmrplens.github.io/phonometry/guides/electroacoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-106",
            "name": "IMD",
            "termCode": "IMD",
            "description": "Modulation intermodulation distortion: the sidebands a low-frequency tone produces around a high-frequency one. Unit: %. Defined in: IEC 60268-3:2013, 14.12.7.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Electroacoustics",
              "url": "https://jmrplens.github.io/phonometry/guides/electroacoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-107",
            "name": "DIM",
            "termCode": "DIM",
            "description": "Dynamic intermodulation distortion, measured with a 15 kHz sine against a filtered 3.15 kHz square wave. Unit: %. Defined in: IEC 60268-3:2013, 14.12.9.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Electroacoustics",
              "url": "https://jmrplens.github.io/phonometry/guides/electroacoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-108",
            "name": "LK, LUFS",
            "termCode": "LK, LUFS",
            "description": "Programme loudness: the channel-weighted sum of K-weighted mean-square powers, gated in 400 ms blocks. LUFS and LKFS name the same unit. Unit: LUFS. Defined in: ITU-R BS.1770-5, Formula 2 (gating in Formulae 3 to 7).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Programme loudness",
              "url": "https://jmrplens.github.io/phonometry/guides/program-loudness/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-109",
            "name": "LRA",
            "termCode": "LRA",
            "description": "Loudness range: the spread between the 10th and 95th percentiles of the gated short-term loudness distribution. Unit: LU. Defined in: EBU Tech 3342.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Programme loudness",
              "url": "https://jmrplens.github.io/phonometry/guides/program-loudness/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-110",
            "name": "dBTP",
            "termCode": "dBTP",
            "description": "True peak level: the peak of the signal reconstructed by oversampling, which catches the inter-sample peaks a sample-domain maximum misses. Unit: dBTP. Defined in: ITU-R BS.1770-5, Annex 2.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Programme loudness",
              "url": "https://jmrplens.github.io/phonometry/guides/program-loudness/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-111",
            "name": "PNL",
            "termCode": "PNL",
            "description": "Perceived noise level: the 24 one-third-octave band levels converted to noisiness in noys and recombined. Unit: PNdB. Defined in: ICAO Annex 16, Vol. I, Appendix 2 (noisiness law in Table A2-3).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Aircraft noise",
              "url": "https://jmrplens.github.io/phonometry/guides/aircraft-noise/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-112",
            "name": "PNLT",
            "termCode": "PNLT",
            "description": "Tone-corrected perceived noise level: PNL plus the penalty for spectral irregularities such as fan and turbine tones. Unit: PNdB. Defined in: ICAO Annex 16, Vol. I, Appendix 2.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Aircraft noise",
              "url": "https://jmrplens.github.io/phonometry/guides/aircraft-noise/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-113",
            "name": "EPNL",
            "termCode": "EPNL",
            "description": "Effective perceived noise level: the maximum PNLT plus the duration correction over the 10 dB-down window, the noise-certification metric. Unit: EPNdB. Defined in: ICAO Annex 16, Vol. I, Appendix 2.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Aircraft noise",
              "url": "https://jmrplens.github.io/phonometry/guides/aircraft-noise/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-114",
            "name": "Lp (underwater)",
            "termCode": "Lp (underwater)",
            "description": "Underwater sound pressure level, referred to 1 µPa rather than 20 µPa. An airborne level never converts to it by subtraction alone. Unit: dB re 1 µPa. Defined in: ISO 18405:2017 (mean-square level in ISO 18406:2017, Formula 7).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Underwater acoustics",
              "url": "https://jmrplens.github.io/phonometry/guides/underwater-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-115",
            "name": "SEL (underwater)",
            "termCode": "SEL (underwater)",
            "description": "Underwater sound exposure level, the time integral of squared pressure referred to 1 µPa²·s. Unit: dB re 1 µPa²·s. Defined in: ISO 18405:2017.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Underwater acoustics",
              "url": "https://jmrplens.github.io/phonometry/guides/underwater-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-116",
            "name": "LRN",
            "termCode": "LRN",
            "description": "Radiated noise level of a ship: the level of the product of the far-field r.m.s. pressure and the source distance. Unit: dB re 1 µPa·m. Defined in: ISO 17208-1:2016.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Underwater acoustics",
              "url": "https://jmrplens.github.io/phonometry/guides/underwater-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-117",
            "name": "Ls",
            "termCode": "Ls",
            "description": "Equivalent monopole source level: the radiated noise level after the Lloyd's-mirror surface correction, so that one number describes the source itself. Unit: dB re 1 µPa·m. Defined in: ISO 17208-2:2019, Formula 3.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Underwater acoustics",
              "url": "https://jmrplens.github.io/phonometry/guides/underwater-acoustics/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-118",
            "name": "u(y)",
            "termCode": "u(y)",
            "description": "Combined standard uncertainty of a result, propagated from the standard uncertainties of its inputs by the law of propagation of uncertainty. Unit: unit of the result. Defined in: ISO/IEC Guide 98-3:2008 (JCGM 100:2008), clause 5.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "GUM uncertainty",
              "url": "https://jmrplens.github.io/phonometry/guides/gum-uncertainty/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-119",
            "name": "U",
            "termCode": "U",
            "description": "Expanded uncertainty: the combined standard uncertainty multiplied by a coverage factor, which defines a coverage interval. Unit: unit of the result. Defined in: ISO/IEC Guide 98-3:2008 (JCGM 100:2008), clause 6 and Annex G.",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "GUM uncertainty",
              "url": "https://jmrplens.github.io/phonometry/guides/gum-uncertainty/"
            }
          },
          {
            "@type": "DefinedTerm",
            "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#term-120",
            "name": "σR",
            "termCode": "σR",
            "description": "Standard uncertainty of a building-acoustics measurement situation, tabulated per quantity and situation. Unit: dB. Defined in: ISO 12999-1:2020, Clause 5.2 (coverage factors in Table 8).",
            "inDefinedTermSet": {
              "@id": "https://jmrplens.github.io/phonometry/reference/glossary/#glossary"
            },
            "subjectOf": {
              "@type": "TechArticle",
              "name": "Field insulation",
              "url": "https://jmrplens.github.io/phonometry/guides/insulation-field/"
            }
          }
        ]
      }
---

The guides each open with the standard they implement, so a quantity is always
defined where it is used. This page is the other direction: you have a symbol,
from a report, a specification or a colleague's email, and you want to know
what it is, what it is measured in, which document defines it and where in this
documentation it is computed.

Two conventions govern the fourth column, and they matter more than they look.
Where a clause, formula or table number appears, it is the one the
implementation cites, taken from the standard the guide implements. Where only
a designation appears, the standard is established but the defining clause is
not stated anywhere in this documentation, and inventing a plausible one would
be worse than leaving it out. A handful of quantities have no governing
standard at all; their source is the paper or book the model comes from, named
as such.

Symbols collide across domains, and the table does not pretend otherwise. *R*
is the sound reduction index in building acoustics, the airflow resistance in
materials, the roughness in psychoacoustics and the cumulative stress variable
in ISO 2631-5. Sigma is the airflow resistivity of a porous material and the
radiation efficiency of a plate. *A* is an equivalent absorption area and an
accelerance. *L*<sub>N</sub> is a percentile level here and a loudness level in
phon there. Read the symbol together with its domain, which is what the
grouping below is for.

For the source of each definition rather than the definition itself, the
[bibliography](/phonometry/reference/bibliography/) lists every cited work with
a DOI or publisher link, and the
[conformance report](/phonometry/reference/conformance/) shows the numerical
check that pins each quantity to its standard's own expected value.

## Sound pressure, power and intensity levels

| Symbol | Quantity | Unit | Defined in | Guide |
| --- | --- | --- | --- | --- |
| Lp | Sound pressure level: twenty times the base-10 logarithm of the r.m.s. sound pressure over the reference pressure. | dB re 20 µPa | IEC 61672-1:2013 | [Calibration](/phonometry/guides/calibration/) |
| Leq | Equivalent continuous sound pressure level: the level of the steady sound carrying the same mean-square pressure over the interval. | dB | IEC 61672-1:2013 | [Levels](/phonometry/guides/levels/) |
| LAeq | The same integral applied to the A-weighted signal, the default descriptor of environmental and occupational noise. | dB | IEC 61672-1:2013 | [Levels](/phonometry/guides/levels/) |
| LAE, SEL | Sound exposure level: the whole A-weighted energy of a single event normalised to one second. | dB | IEC 61672-1:2013, Equation 8 (Table 4) | [Levels](/phonometry/guides/levels/) |
| LCpeak | C-weighted peak sound level: the absolute maximum of the C-weighted pressure, not a time-weighted maximum. | dB | IEC 61672-1:2013, subclause 5.13 | [Levels](/phonometry/guides/levels/) |
| LN (L10, L50, L90) | Percentile level: the level exceeded N % of the measurement time, read off the time-weighted level distribution. | dB | ISO 1996-2:2017 (Annex I uses L90 as the residual level) | [Levels](/phonometry/guides/levels/) |
| LW, SWL | Sound power level: the power a source radiates, referred to 1 pW. | dB re 1 pW | ISO 3745:2012, Clause 8 | [Sound power](/phonometry/guides/sound-power/) |
| LI | Sound intensity level: the magnitude of the intensity vector referred to 1 pW/m², with the flow direction reported separately as a sign. | dB re 1 pW/m² | IEC 61043:1993 | [Intensity](/phonometry/guides/intensity/) |
| Lp − LI | Pressure-intensity index: the difference between the pressure and intensity levels at a position, the field indicator that qualifies an intensity measurement. | dB | ISO 9614-1:1993, Equation (A.3) | [Intensity](/phonometry/guides/intensity/) |

## Environmental and occupational descriptors

| Symbol | Quantity | Unit | Defined in | Guide |
| --- | --- | --- | --- | --- |
| Lden | Day-evening-night level: the energy mean of the three periods with 5 dB added to the evening and 10 dB to the night. | dB | ISO 1996-1:2016, 3.6.4 | [Levels](/phonometry/guides/levels/) |
| Ldn | Day-night level: the same construction with the 10 dB night penalty only. | dB | ISO 1996-1:2016, 3.6.5 | [Levels](/phonometry/guides/levels/) |
| Lr | Rating level: the whole-day composite level after the source-character and time-of-day adjustments. | dB | ISO 1996-1:2016, clause 6.5 (Formulae 5 and 6) | [Levels](/phonometry/guides/levels/) |
| LAr,T | Rating level of an impulsive source over a reference interval, LAeq plus the graduated impulse adjustment. | dB | NT ACOU 112:2002, clause 8 | [Impulse prominence](/phonometry/guides/impulse-prominence/) |
| KI | Impulse adjustment added to LAeq, graduated by the predicted prominence of the impulses. | dB | NT ACOU 112:2002, clause 8 | [Impulse prominence](/phonometry/guides/impulse-prominence/) |
| E | Sound exposure: the time integral of the squared A-weighted sound pressure over the exposure period. | Pa²h | IEC 61252:1993, 3.1 | [Levels](/phonometry/guides/levels/) |
| LEX,8h, LEP,d | Daily noise exposure level: the steady level that, sustained over a nominal 8 h day, carries the same A-weighted sound exposure as the measured one. | dB | IEC 61252:1993, 3.3 | [Occupational exposure](/phonometry/guides/occupational-exposure/) |
| Lp,A,eqT | A-weighted equivalent continuous level of a task, a job sample or a full day, the building block LEX,8h is assembled from. | dB | ISO 9612:2009, clauses 9 to 11 | [Occupational exposure](/phonometry/guides/occupational-exposure/) |
| NIPTS | Noise-induced permanent threshold shift: the median hearing loss attributable to a stated exposure level, duration and audiometric frequency. | dB | ISO 1999:2013 | [Noise-induced hearing loss](/phonometry/guides/noise-induced-hearing-loss/) |
| HTLAN | Hearing threshold level associated with age and noise: the NIPTS combined with the age component. | dB | ISO 1999:2013 | [Noise-induced hearing loss](/phonometry/guides/noise-induced-hearing-loss/) |

## Frequency and time weighting

| Symbol | Quantity | Unit | Defined in | Guide |
| --- | --- | --- | --- | --- |
| A, C, Z | The normative frequency weightings: the ear-response curves applied before integration, Z being the flat reference. | dB | IEC 61672-1:2013, Annex E (acceptance limits in Table 3) | [Weighting](/phonometry/guides/weighting/) |
| G | Infrasound weighting, defined by its poles and zeros for the 0.25 Hz to 315 Hz range. | dB | ISO 7196:1995, Table 1 (nominal responses in Table 2) | [Weighting](/phonometry/guides/weighting/) |
| B | Historical mid-level weighting, withdrawn from the current meter standard. | dB | ANSI S1.4-1983, Appendix C (Formula C2) | [Weighting](/phonometry/guides/weighting/) |
| D | Historical aircraft-noise weighting, derived from the 40-noy perceived-noisiness contour. | dB | IEC 537:1976 (withdrawn) | [Weighting](/phonometry/guides/weighting/) |
| AU | Weighting for audible sound measured in the presence of ultrasound. | dB | IEC 61012:1990, subclause 2.2 (Tables 1 and 2) | [Weighting](/phonometry/guides/weighting/) |
| F, S, I | Fast, Slow and Impulse exponential time weightings: the detector ballistics that produce a displayed level. | s (time constant) | IEC 61672-1:2013 | [Time weighting](/phonometry/guides/time-weighting/) |

## Room acoustics

| Symbol | Quantity | Unit | Defined in | Guide |
| --- | --- | --- | --- | --- |
| T20 | Reverberation time extrapolated to a 60 dB decay from a least-squares fit over −5 dB to −25 dB of the Schroeder curve. | s | ISO 3382-2:2008, Clause 6 and Annex C | [Room acoustics](/phonometry/guides/room-acoustics/) |
| T30 | The same extrapolation from a fit over −5 dB to −35 dB, the usual choice when the decay range allows it. | s | ISO 3382-2:2008, Clause 6 and Annex C | [Room acoustics](/phonometry/guides/room-acoustics/) |
| T60, RT | Reverberation time as such: the time for the sound energy to fall by 60 dB. Measured in practice as T20 or T30. | s | ISO 3382-1:2009 | [Room acoustics](/phonometry/guides/room-acoustics/) |
| EDT | Early decay time: the same slope taken over the first 10 dB of decay, which tracks perceived reverberance rather than the tail. | s | ISO 3382-1:2009 (just-noticeable difference in Table A.1) | [Room acoustics](/phonometry/guides/room-acoustics/) |
| C50 | Clarity for speech: the energy ratio between the first 50 ms of the impulse response and everything after it. | dB | ISO 3382-1:2009 | [Room acoustics](/phonometry/guides/room-acoustics/) |
| C80 | Clarity for music: the same ratio with the boundary at 80 ms. | dB | ISO 3382-1:2009 (just-noticeable difference in Table A.1) | [Room acoustics](/phonometry/guides/room-acoustics/) |
| D50 | Definition, or Deutlichkeit: the fraction of the total energy arriving in the first 50 ms. | dimensionless | ISO 3382-1:2009 (just-noticeable difference in Table A.1) | [Room acoustics](/phonometry/guides/room-acoustics/) |
| Ts | Centre time: the centre of gravity of the squared impulse response in time, a boundary-free alternative to the clarity indices. | s | ISO 3382-1:2009, Equation (A.13) | [Room acoustics](/phonometry/guides/room-acoustics/) |
| A | Equivalent sound absorption area of a room: the area of a perfectly absorbing surface that would give the same reverberation time. | m² | ISO 354:2003, Equations (5) and (7) | [Absorption measurement](/phonometry/guides/absorption-measurement/) |
| NC | Noise criteria rating of a background spectrum: the speech interference level selects the curve, and the tangency method rates the spectrum when a band exceeds it. | dB (index) | ANSI/ASA S12.2-2019, 5.2.2 and 5.2.3 (curves in Table 1) | [Room noise](/phonometry/guides/room-noise/) |
| SIL | Speech interference level: the average of the 500, 1000, 2000 and 4000 Hz octave-band levels. | dB | ANSI/ASA S12.2-2019, clause 3.2 | [Room noise](/phonometry/guides/room-noise/) |
| RC | Room criteria Mark II rating: the average of the 500, 1000 and 2000 Hz levels, with a rumble, hiss or neutral spectral tag. | dB (index) | ANSI/ASA S12.2-2019, Annex D (clauses D.3 and D.4) | [Room noise](/phonometry/guides/room-noise/) |
| NR | Noise rating, the European counterpart curve family of NC. Discussed for comparison and deliberately not implemented. | dB (index) | Kosten and van Os (1962); no governing standard | [Room noise](/phonometry/guides/room-noise/) |

## Speech and intelligibility

| Symbol | Quantity | Unit | Defined in | Guide |
| --- | --- | --- | --- | --- |
| m(F) | Modulation transfer function: the fraction of the speech envelope modulation depth at modulation frequency F that survives the transmission path. | dimensionless | IEC 60268-16:2020 | [Speech transmission](/phonometry/guides/speech-transmission/) |
| STI | Speech transmission index: the modulation transfer matrix converted to effective signal-to-noise ratios and weighted into a single value on 0 to 1. | dimensionless | IEC 60268-16:2020, A.5.2 to A.5.6 | [Speech transmission](/phonometry/guides/speech-transmission/) |
| STIPA | The direct STI measurement, made by playing a standardised two-modulation-per-band test signal through the real chain. | dimensionless | IEC 60268-16:2020, clause 6.3 and Table 3 (direct method, Annex B) | [Speech transmission](/phonometry/guides/speech-transmission/) |
| SII | Speech intelligibility index: the band-importance-weighted audibility of the speech spectrum against noise and the listener's threshold. | dimensionless | ANSI S3.5-1997, clause 6 (procedure in clause 5, importance function in Table 3) | [Speech intelligibility](/phonometry/guides/speech-intelligibility/) |
| STOI | Short-time objective intelligibility: the clipped per-band envelope correlation between clean and degraded speech. | dimensionless | Taal et al. (2011), Equations 5 and 6; no governing standard | [Objective intelligibility](/phonometry/guides/objective-intelligibility/) |
| ESTOI | The extended measure, row- and column-normalised so that it tracks modulated maskers. | dimensionless | Jensen and Taal (2016), Equation 8; no governing standard | [Objective intelligibility](/phonometry/guides/objective-intelligibility/) |

## Sound insulation

| Symbol | Quantity | Unit | Defined in | Guide |
| --- | --- | --- | --- | --- |
| D | Level difference: the energy-averaged source-room level minus the receiving-room level, with no normalisation. | dB | ISO 16283-1:2014, 3.12 to 3.15 | [Field insulation](/phonometry/guides/insulation-field/) |
| DnT | Standardized level difference: the level difference referred to a reference reverberation time, 0.5 s for dwellings. | dB | ISO 16283-1:2014, 3.12 to 3.15 | [Field insulation](/phonometry/guides/insulation-field/) |
| Dn | Normalized level difference: the level difference referred to a reference absorption area of 10 m². | dB | ISO 10052:2021 | [Field insulation](/phonometry/guides/insulation-field/) |
| Dn,e | Element-normalized level difference of a small element or air path, referred to a reference area of 10 m². | dB | EN 12354-3:2000 | [Insulation prediction](/phonometry/guides/insulation-prediction/) |
| R | Sound reduction index: the level difference corrected by the partition area over the receiving-room absorption area, measured in the laboratory with flanking suppressed. | dB | ISO 10140-2:2010 | [Laboratory insulation](/phonometry/guides/insulation-lab/) |
| R′ | Apparent sound reduction index: the same construction measured in the building, so it includes every flanking path. The prime is the lab-versus-field marker. | dB | ISO 16283-1:2014, 3.12 to 3.15 | [Field insulation](/phonometry/guides/insulation-field/) |
| TL | Transmission loss: the airborne insulation of a panel predicted from its physical properties, the same quantity as R in a prediction context. | dB | Bies, Hansen and Howard (2017), Section 7.2; no governing standard | [Panel insulation](/phonometry/guides/panel-sound-insulation/) |
| Rw, R′w, DnT,w | The weighted single-number ratings: a fixed reference curve is shifted toward the measured spectrum until the unfavourable deviations reach their allowed sum, and the shifted curve is read at 500 Hz. | dB | ISO 717-1:2020 | [Field insulation](/phonometry/guides/insulation-field/) |
| Dn,e,w | The same reference-curve rating applied to the element-normalized level difference. | dB | ISO 717-1:2020 | [Insulation prediction](/phonometry/guides/insulation-prediction/) |
| C, Ctr | Spectrum adaptation terms: the corrections that re-rate the measured curve against A-weighted pink noise (C) and against A-weighted urban road traffic (Ctr). | dB | ISO 717-1:2020, Annex A | [Field insulation](/phonometry/guides/insulation-field/) |
| Ln | Normalized impact sound pressure level: the receiving-room level under the standard tapping machine, referred to a 10 m² absorption area. | dB | ISO 10140-3:2010 | [Laboratory insulation](/phonometry/guides/insulation-lab/) |
| L′nT | Standardized impact sound pressure level, referred to a reference reverberation time. Note the sign: more reverberation lowers it, the opposite of DnT. | dB | ISO 16283-2:2015 | [Field insulation](/phonometry/guides/insulation-field/) |
| Ln,w, L′nT,w | The weighted impact ratings. The reference curve is shifted the same way, but an unfavourable deviation is now one where the measurement exceeds the reference. | dB | ISO 717-2:2020 | [Field insulation](/phonometry/guides/insulation-field/) |
| CI | Impact spectrum adaptation term, from the energetic sum over 100 Hz to 2500 Hz. The enlarged-range CI,50-2500 extends it down to 50 Hz. | dB | ISO 717-2:2020 (enlarged range in A.2.1 NOTE) | [Field insulation](/phonometry/guides/insulation-field/) |
| ΔLw | Weighted reduction of impact sound pressure level given by a floor covering, measured as the improvement over the bare reference floor. | dB | ISO 717-2:2020 (measurement in ISO 16251-1:2014, Formulae (3) and (4)) | [Laboratory insulation](/phonometry/guides/insulation-lab/) |
| ΔRw | Weighted improvement of airborne insulation contributed by a lining or additional layer, added to the element rating in the prediction. | dB | EN 12354-1:2000, Formulae 27 and 28a | [Insulation prediction](/phonometry/guides/insulation-prediction/) |
| Kij | Vibration reduction index of a junction: the direction-averaged velocity level difference corrected by the junction length and the equivalent absorption lengths. | dB | ISO 10848-1:2006, Formula (13) | [Laboratory insulation](/phonometry/guides/insulation-lab/) |
| fc | Critical frequency: the frequency at which the bending wavelength of a panel equals the wavelength in air, where the coincidence dip appears. | Hz | Bies, Hansen and Howard (2017), Equation 7.3; no governing standard | [Panel insulation](/phonometry/guides/panel-sound-insulation/) |
| σ | Radiation efficiency of a plate: the airborne power radiated per unit mean-square surface velocity, normalised by the plane-wave value. | dimensionless | Hopkins (2007), Equations 2.227 to 2.230; no governing standard | [Panel insulation](/phonometry/guides/panel-sound-insulation/) |

## Materials and surfaces

| Symbol | Quantity | Unit | Defined in | Guide |
| --- | --- | --- | --- | --- |
| α | Sound absorption coefficient at normal incidence: the fraction of incident energy not returned by the surface, obtained in the impedance tube from the reflection factor. | dimensionless | ISO 10534-2:1998, Equations (17) to (19) | [Impedance tube](/phonometry/guides/impedance-tube/) |
| αs | Random-incidence sound absorption coefficient measured in a reverberation room, from the change in equivalent absorption area with and without the specimen. | dimensionless | ISO 354:2003, Equations (8) and (9) | [Absorption measurement](/phonometry/guides/absorption-measurement/) |
| αp | Practical sound absorption coefficient: the one-third-octave data grouped into octave bands and rounded to steps of 0.05. | dimensionless | ISO 11654:1997, Clause 4.1 | [Absorption measurement](/phonometry/guides/absorption-measurement/) |
| αw | Weighted sound absorption coefficient: the fixed reference curve shifted toward the practical values and read at 500 Hz. | dimensionless | ISO 11654:1997, Clause 4.2 | [Absorption measurement](/phonometry/guides/absorption-measurement/) |
| Absorption class | The A to E letter class the weighted coefficient maps to, or "not classified". | class letter | ISO 11654:1997, Table B.1 | [Absorption measurement](/phonometry/guides/absorption-measurement/) |
| R | Airflow resistance: the pressure difference across a specimen divided by the volumetric airflow rate through it. | Pa·s/m³ | ISO 9053-1:2018, Clause 3 | [Airflow resistance](/phonometry/guides/airflow-resistance/) |
| Rs | Specific airflow resistance: the airflow resistance referred to the specimen face area. | Pa·s/m | ISO 9053-1:2018, Clause 3 | [Airflow resistance](/phonometry/guides/airflow-resistance/) |
| σ | Airflow resistivity: the specific airflow resistance per unit thickness, the primary input to every empirical porous model. | Pa·s/m² | ISO 9053-1:2018, Clause 3 | [Airflow resistance](/phonometry/guides/airflow-resistance/) |
| Z | Surface impedance: the complex ratio of sound pressure to particle velocity at the face of the sample, usually reported normalised by the characteristic impedance of air. | Pa·s/m | ISO 10534-2:1998, Equations (17) to (19) | [Impedance tube](/phonometry/guides/impedance-tube/) |
| s | Scattering coefficient: the fraction of reflected energy that is not returned specularly, measured at random incidence on a turntable in a reverberation room. | dimensionless | ISO 17497-1:2004+A1:2014, Formula (5) | [Surface scattering](/phonometry/guides/surface-scattering/) |
| d | Diffusion coefficient: the uniformity of the polar response of a surface, from the autocorrelation of the free-field goniometer measurement. | dimensionless | ISO 17497-2:2012, Formula (5) (normalised form in Formula (7)) | [Surface scattering](/phonometry/guides/surface-scattering/) |
| s′ | Dynamic stiffness per unit area of a resilient layer: a dynamic force per unit area divided by the resulting change in thickness. | MN/m³ | EN 29052-1:1992 (ISO 9052-1:1989), Formula 1 | [Dynamic stiffness](/phonometry/guides/dynamic-stiffness/) |

## Vibration and structure-borne sound

| Symbol | Quantity | Unit | Defined in | Guide |
| --- | --- | --- | --- | --- |
| Y | Mobility: the complex ratio of a velocity response to the force that produces it. | m/(N·s) | ISO 7626-1:2011, 3.1.2 and Table 1 | [Mechanical mobility](/phonometry/guides/mechanical-mobility/) |
| Z | Mechanical impedance: the reciprocal of mobility, force per unit velocity. | N·s/m | ISO 7626-1:2011, Table 1 | [Mechanical mobility](/phonometry/guides/mechanical-mobility/) |
| H | Receptance, or dynamic compliance: displacement response per unit force, the pivot the whole family converts through. | m/N | ISO 7626-1:2011, Table 1 | [Mechanical mobility](/phonometry/guides/mechanical-mobility/) |
| A | Accelerance, or inertance: acceleration response per unit force. Its reciprocal is the apparent mass. | 1/kg | ISO 7626-1:2011, Table 1 | [Mechanical mobility](/phonometry/guides/mechanical-mobility/) |
| k21 | Dynamic transfer stiffness of a resilient element: the blocking force on the output side divided by the displacement on the input side. | N/m | ISO 10846-1:2008, 3.7 | [Transfer stiffness](/phonometry/guides/transfer-stiffness/) |
| Lk | Level of the dynamic transfer stiffness, referred to 1 N/m. | dB re 1 N/m | ISO 10846-2:2008 and ISO 10846-3:2002, 3.17 | [Transfer stiffness](/phonometry/guides/transfer-stiffness/) |
| η | Loss factor of a resilient element: the tangent of the phase angle of its dynamic transfer stiffness. | dimensionless | ISO 10846-1:2008, 3.8 | [Transfer stiffness](/phonometry/guides/transfer-stiffness/) |
| aw | Frequency-weighted acceleration: the root sum of squares of the band accelerations after the human-response weightings. | m/s² | ISO 2631-1:1997, Equation (9) | [Human vibration](/phonometry/guides/human-vibration/) |
| A(8) | Daily vibration exposure: the exposure magnitude normalised to a reference 8 h day, combined over the operations of the day. | m/s² | ISO 5349-1:2001, Equations (2) and (3) | [Human vibration](/phonometry/guides/human-vibration/) |
| VDV | Vibration dose value: the fourth-power time integral of the weighted acceleration, which weights shocks far more heavily than an r.m.s. does. | m/s^1.75 | ISO 2631-1:1997, Equation (5) | [Human vibration](/phonometry/guides/human-vibration/) |
| MTVV | Maximum transient vibration value: the largest 1 s running r.m.s. of the weighted acceleration. | m/s² | ISO 2631-1:1997, Equation (4) | [Human vibration](/phonometry/guides/human-vibration/) |
| R | Cumulative stress variable of the multiple-shock model: the daily compressive stresses accumulated over the years of exposure, which the lumbar injury probability is read from. | dimensionless | ISO 2631-5:2018, Annex C (Formulae C.1 and C.3 to C.5) | [Multiple-shock vibration](/phonometry/guides/multiple-shock-vibration/) |
| Lv | Velocity level: twenty times the base-10 logarithm of the surface velocity over the reference velocity. | dB | ISO/TS 7849-1:2009, Formula 3 | [Vibration sound power](/phonometry/guides/vibration-sound-power/) |
| ε | Radiation factor, or radiation efficiency, of a vibrating machine surface: the airborne power radiated per unit mean-square velocity and area. | dimensionless | ISO/TS 7849-1:2009 and ISO/TS 7849-2:2009 | [Vibration sound power](/phonometry/guides/vibration-sound-power/) |
| LWs | Structure-borne sound power level injected by equipment into a reception plate. | dB re 1 pW | EN 15657:2018, Formula 14 | [Structure-borne power](/phonometry/guides/structure-borne-power/) |
| ηij | Coupling loss factor: the fraction of energy per radian that a statistical energy analysis subsystem loses into a neighbouring one across a junction. | dimensionless | Hopkins (2007), Equation 2.154; no governing standard | [Junction transmission](/phonometry/guides/junction-transmission/) |

## Psychoacoustics

| Symbol | Quantity | Unit | Defined in | Guide |
| --- | --- | --- | --- | --- |
| N | Loudness: the perceived magnitude of a sound, anchored so that a 1 kHz tone at 40 dB SPL is exactly 1 sone. | sone | ISO 532-1:2017, clause 5 (stationary) and clause 6 (time-varying) | [Loudness](/phonometry/guides/loudness/) |
| N′ | Specific loudness: the loudness density along the critical-band scale, whose integral is N. | sone/Bark | ISO 532-1:2017 (sone/Cam form in ISO 532-2:2017, Formula 7) | [Loudness](/phonometry/guides/loudness/) |
| LN | Loudness level: the level of the 1 kHz free-field tone judged equally loud as the sound. | phon | ISO 226:2023, Formula (2) (contours in Formula (1)) | [Loudness](/phonometry/guides/loudness/) |
| S | Sharpness: the position of the centre of gravity of the specific loudness on the critical-band scale, normalised so that the reference narrow-band noise is exactly 1 acum. | acum | DIN 45692:2009, clause 6 | [Sound quality](/phonometry/guides/sound-quality/) |
| R | Roughness: the perceived harshness of fast amplitude modulation, around 70 Hz, normalised so that the reference modulated tone is 1 asper. | asper | ECMA-418-2:2025, clause 7 (Formula 104) | [Sound quality](/phonometry/guides/sound-quality/) |
| F | Fluctuation strength: the perceived slow amplitude modulation, around 4 Hz, normalised so that the reference modulated tone is 1 vacil. | vacil | ECMA-418-2:2025, clause 9 (Formula 163) | [Sound quality](/phonometry/guides/sound-quality/) |
| T | Tonality: the perceived tonal content of a sound, derived from the autocorrelation of the band envelopes. | tu | ECMA-418-2:2025, clause 6 | [Sound quality](/phonometry/guides/sound-quality/) |
| TNR | Tone-to-noise ratio: the level of a discrete tone above the masking noise in the critical band around it. | dB | ECMA-418-1:2024, clause 11 (Formulae 9 to 11) | [Tone prominence](/phonometry/guides/tone-prominence/) |
| PR | Prominence ratio: the level of the critical band containing the tone above the mean of the two adjacent bands. | dB | ECMA-418-1:2024, clause 12 (Formula 23) | [Tone prominence](/phonometry/guides/tone-prominence/) |
| ΔL | Audibility of a tone in noise: the tone level minus the critical-band masking level minus the masking index. | dB | ISO/PAS 20065:2016, Formula 14 | [Tone audibility](/phonometry/guides/tone-audibility/) |
| PA | Psychoacoustic annoyance: the percentile loudness scaled by sharpness and by a fluctuation-plus-roughness term. | dimensionless | Fastl and Zwicker (2007), Equation 16.2; no governing standard | [Psychoacoustic annoyance](/phonometry/guides/psychoacoustic-annoyance/) |

## Electroacoustics and programme loudness

| Symbol | Quantity | Unit | Defined in | Guide |
| --- | --- | --- | --- | --- |
| THD | Total harmonic distortion: the harmonic content of the output relative to the fundamental (THD_F) or to the total signal (THD_R). | % or dB | IEC 60268-3:2013, 14.12.2 to 14.12.11 (the R form in 14.12.3.2) | [Electroacoustics](/phonometry/guides/electroacoustics/) |
| THD+N | Total harmonic distortion plus noise: everything left after notching out the fundamental, within the standard measurement bandwidth. | % or dB | AES17-2015, clause 6.3.1 (notch and bandwidth in 5.2.5 and 5.2.8) | [Electroacoustics](/phonometry/guides/electroacoustics/) |
| SINAD | Signal to noise and distortion ratio, the reciprocal expression of THD+N. | dB | AES17-2015, clause 6.3.1 | [Electroacoustics](/phonometry/guides/electroacoustics/) |
| IMD | Modulation intermodulation distortion: the sidebands a low-frequency tone produces around a high-frequency one. | % | IEC 60268-3:2013, 14.12.7 | [Electroacoustics](/phonometry/guides/electroacoustics/) |
| DIM | Dynamic intermodulation distortion, measured with a 15 kHz sine against a filtered 3.15 kHz square wave. | % | IEC 60268-3:2013, 14.12.9 | [Electroacoustics](/phonometry/guides/electroacoustics/) |
| LK, LUFS | Programme loudness: the channel-weighted sum of K-weighted mean-square powers, gated in 400 ms blocks. LUFS and LKFS name the same unit. | LUFS | ITU-R BS.1770-5, Formula 2 (gating in Formulae 3 to 7) | [Programme loudness](/phonometry/guides/program-loudness/) |
| LRA | Loudness range: the spread between the 10th and 95th percentiles of the gated short-term loudness distribution. | LU | EBU Tech 3342 | [Programme loudness](/phonometry/guides/program-loudness/) |
| dBTP | True peak level: the peak of the signal reconstructed by oversampling, which catches the inter-sample peaks a sample-domain maximum misses. | dBTP | ITU-R BS.1770-5, Annex 2 | [Programme loudness](/phonometry/guides/program-loudness/) |

## Aircraft and underwater

| Symbol | Quantity | Unit | Defined in | Guide |
| --- | --- | --- | --- | --- |
| PNL | Perceived noise level: the 24 one-third-octave band levels converted to noisiness in noys and recombined. | PNdB | ICAO Annex 16, Vol. I, Appendix 2 (noisiness law in Table A2-3) | [Aircraft noise](/phonometry/guides/aircraft-noise/) |
| PNLT | Tone-corrected perceived noise level: PNL plus the penalty for spectral irregularities such as fan and turbine tones. | PNdB | ICAO Annex 16, Vol. I, Appendix 2 | [Aircraft noise](/phonometry/guides/aircraft-noise/) |
| EPNL | Effective perceived noise level: the maximum PNLT plus the duration correction over the 10 dB-down window, the noise-certification metric. | EPNdB | ICAO Annex 16, Vol. I, Appendix 2 | [Aircraft noise](/phonometry/guides/aircraft-noise/) |
| Lp (underwater) | Underwater sound pressure level, referred to 1 µPa rather than 20 µPa. An airborne level never converts to it by subtraction alone. | dB re 1 µPa | ISO 18405:2017 (mean-square level in ISO 18406:2017, Formula 7) | [Underwater acoustics](/phonometry/guides/underwater-acoustics/) |
| SEL (underwater) | Underwater sound exposure level, the time integral of squared pressure referred to 1 µPa²·s. | dB re 1 µPa²·s | ISO 18405:2017 | [Underwater acoustics](/phonometry/guides/underwater-acoustics/) |
| LRN | Radiated noise level of a ship: the level of the product of the far-field r.m.s. pressure and the source distance. | dB re 1 µPa·m | ISO 17208-1:2016 | [Underwater acoustics](/phonometry/guides/underwater-acoustics/) |
| Ls | Equivalent monopole source level: the radiated noise level after the Lloyd's-mirror surface correction, so that one number describes the source itself. | dB re 1 µPa·m | ISO 17208-2:2019, Formula 3 | [Underwater acoustics](/phonometry/guides/underwater-acoustics/) |

## Measurement uncertainty

| Symbol | Quantity | Unit | Defined in | Guide |
| --- | --- | --- | --- | --- |
| u(y) | Combined standard uncertainty of a result, propagated from the standard uncertainties of its inputs by the law of propagation of uncertainty. | unit of the result | ISO/IEC Guide 98-3:2008 (JCGM 100:2008), clause 5 | [GUM uncertainty](/phonometry/guides/gum-uncertainty/) |
| U | Expanded uncertainty: the combined standard uncertainty multiplied by a coverage factor, which defines a coverage interval. | unit of the result | ISO/IEC Guide 98-3:2008 (JCGM 100:2008), clause 6 and Annex G | [GUM uncertainty](/phonometry/guides/gum-uncertainty/) |
| σR | Standard uncertainty of a building-acoustics measurement situation, tabulated per quantity and situation. | dB | ISO 12999-1:2020, Clause 5.2 (coverage factors in Table 8) | [Field insulation](/phonometry/guides/insulation-field/) |
