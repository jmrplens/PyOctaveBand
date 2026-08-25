#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Fiches for environmental noise: what reaches the neighbour, and its penalty.

Sound assessed where people have to live with it: the tonal audibility of
ISO 1996-2 Annex J, the impulse prominence of NT ACOU 112 and the wind-turbine
tonality of IEC 61400-11 that add a penalty to a measured level, the outdoor
propagation and barrier attenuation of the ISO 9613-2 family on the way there,
and the RD 1367/2007 assessment of an activity against the limits binding it.
"""

from __future__ import annotations

import numpy as np

import phonometry as ph
from phonometry import ReportMetadata


def _tone_audibility_example() -> tuple[object, ReportMetadata, str]:
    """Tonal audibility fiche: an ISO 1996-2:2017 Annex J tone assessment.

    The narrow-band spectrum is the ISO/PAS 20065:2016 Annex E combustion-engine
    worked example (Table E.1, the 38 lines of Delta f = 2.7 Hz about the
    137.3 Hz tone). analyze_spectrum detects the three tones at 118.4 / 137.3 /
    158.8 Hz and their combined FG group; the decisive tonal audibility is the
    FG entry (Delta L_ta ~ 9.1 dB), which maps to the tonal adjustment K = 5 dB
    by ISO 1996-2:2017 Table J.1 (9 < Delta L <= 12). The complete-spectrum
    printed value is 9.18 dB (K = 5 either way); the small difference is the
    documented edge-tone masking-level underestimate on the truncated band.
    """
    frequencies = np.array(
        [
            96.9,
            99.6,
            102.3,
            105.0,
            107.7,
            110.4,
            113.0,
            115.7,
            118.4,
            121.1,
            123.8,
            126.5,
            129.2,
            131.9,
            134.6,
            137.3,
            140.0,
            142.7,
            145.3,
            148.0,
            150.7,
            153.4,
            156.1,
            158.8,
            161.5,
            164.2,
            166.9,
            169.6,
            172.3,
            175.0,
            177.6,
            180.3,
            183.0,
            185.7,
            188.4,
            191.1,
            193.8,
            196.5,
        ],
        dtype=float,
    )
    levels = np.array(
        [
            49.40,
            50.68,
            50.09,
            53.37,
            44.47,
            50.91,
            51.41,
            59.40,
            64.54,
            57.57,
            51.02,
            50.76,
            59.93,
            62.94,
            58.49,
            65.87,
            62.66,
            50.25,
            51.32,
            52.30,
            52.58,
            53.15,
            67.04,
            67.27,
            57.40,
            57.17,
            52.56,
            51.39,
            52.49,
            47.68,
            51.26,
            49.03,
            61.42,
            59.52,
            48.43,
            50.84,
            48.20,
            55.95,
        ],
        dtype=float,
    )
    result = ph.psychoacoustics.analyze_spectrum(levels, frequencies, 2.7)
    metadata = ReportMetadata(
        specimen="Combustion engine, steady operation",
        client="Example client",
        test_room="Free field, 3 m from source (example)",
        instrumentation="Class 1 analyser, FFT with 2.7 Hz lines (Hann window)",
        measurement_standard="ISO 1996-2",
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-1996-TONE",
        requirement=6.0,
    )
    return result, metadata, "iso1996_tone_audibility_example.pdf"


def _impulse_prominence_example() -> tuple[object, ReportMetadata, str]:
    """Impulse-prominence fiche: an NT ACOU 112:2002 impulsive-sound assessment.

    Three candidate impulses of a pile-driving site, each ``(onset rate dB/s,
    level difference dB)``: (1200, 32), (300, 18) and (60, 11). All three qualify
    (onset rate above 10 dB/s), and the predicted prominence follows Formula 1
    P = 3 lg(OR) + 2 lg(LD): P1 = 3 lg 1200 + 2 lg 32 = 9.2375 + 3.0103 = 12.25,
    P2 = 3 lg 300 + 2 lg 18 = 9.94 and P3 = 3 lg 60 + 2 lg 11 = 7.42. The highest,
    the first impulse, governs (P = 12.25), and its LAeq adjustment is
    KI = 1.8 (12.2478 - 5) = 13.05 dB (Formula 2), rounding to 13.0 dB. The
    requirement is a plausible maximum governing prominence the example exceeds,
    so the optional verdict FAILs.
    """
    result = ph.environment.assessment.impulse_prominence(
        [1200.0, 300.0, 60.0], [32.0, 18.0, 11.0]
    )
    metadata = ReportMetadata(
        specimen="Pile-driving site, intermittent hammering",
        client="Example client",
        test_room="Free field, 25 m from source (example)",
        instrumentation="Class 1 SLM (IEC 61672-1), L_pAF logged at 100 ms",
        measurement_standard="ISO 1996-2",
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-NTACOU112",
        requirement=10.0,
    )
    return result, metadata, "ntacou112_impulse_prominence_example.pdf"


def _wind_turbine_tonality_example() -> tuple[object, ReportMetadata, str]:
    """Wind-turbine tonality fiche: an IEC 61400-11:2012+A1:2018 assessment.

    A clean 500 Hz tone (a gearbox line) 30 dB above a flat 30 dB narrow-band
    floor, sampled at Delta f = 2 Hz over 440-560 Hz. The critical band about
    500 Hz is CBW = 117.256 Hz (Formula 30) and the ENBW is 1.5 * 2 = 3 Hz, so
    the masking-noise level is L_pn = 30 + 10 lg(117.256 / 3) = 45.92 dB
    (Formula 31); the single tone line gives L_pt = 60 dB, hence the tonality
    is Delta L_tn = 14.08 dB (Formula 32). The audibility criterion at 500 Hz
    is L_a = -2 - lg(1 + (500/502)^2.5) = -2.30 dB (Formula 34), so the tonal
    audibility is Delta L_a = 14.08 - (-2.30) = 16.38 dB (Formula 33): the tone
    is audible. The requirement is a plausible maximum acceptable tonal
    audibility the example exceeds, so the optional verdict FAILs.
    """
    df = 2.0
    frequencies = np.arange(440.0, 560.0 + df, df)
    levels = np.full(frequencies.size, 30.0)
    levels[int(np.argmin(np.abs(frequencies - 500.0)))] = 60.0
    result = ph.environment.wind_turbine_tonality(levels, frequencies)
    metadata = ReportMetadata(
        specimen="Horizontal-axis wind turbine, gearbox tone",
        client="Example client",
        test_room="Ground board, downwind reference position (example)",
        instrumentation="Class 1 analyser, FFT with 2 Hz lines (Hann window)",
        measurement_standard="IEC 61400-11",
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-61400-11",
        requirement=6.0,
    )
    return result, metadata, "iec61400_wind_turbine_tonality_example.pdf"


class _WithSourceEmission:
    """Adapter binding a source emission to an ``OutdoorAttenuation.report`` call.

    The uniform generator drives every fiche with ``result.report(path,
    metadata=...)``; the ISO 9613-2 attenuation fiche needs the source emission
    too (a report-time, display-only object), so this adapter carries it while
    keeping the generator loop unchanged.
    """

    def __init__(
        self,
        result: ph.environment.OutdoorAttenuation,
        emission: ph.environment.SourceEmission,
    ) -> None:
        self._result = result
        self._emission = emission

    def report(self, path: str, *, metadata: ReportMetadata | None = None) -> str:
        return str(
            self._result.report(path, metadata=metadata, source_emission=self._emission)
        )


def _outdoor_attenuation_example() -> tuple[object, ReportMetadata, str]:
    """ISO 9613-2 fiche: predicted outdoor propagation attenuation with a barrier.

    An industrial point source (octave-band Lw from 95 dB at 63 Hz to 88 dB at
    8 kHz, supplied at report time via SourceEmission) 200 m upwind of a dwelling
    over porous ground (Gs = Gm = Gr = 1), screened by a noise barrier
    (source-edge = edge-receiver = 105 m, so the diffracted path exceeds the
    direct one). The divergence, atmospheric, ground and barrier terms come from
    the tested clause-7 functions (see
    tests/environment/propagation/test_outdoor_propagation.py).
    """
    freqs = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
    lw = np.array([95, 100, 103, 105, 104, 101, 95, 88], dtype=float)
    barrier = ph.environment.Barrier(source_to_edge=105.0, edge_to_receiver=105.0)
    result = ph.environment.outdoor_propagation_attenuation(
        200.0,
        4.0,
        2.0,
        freqs,
        1.0,
        1.0,
        1.0,
        barrier=barrier,
        temperature=10.0,
        relative_humidity=70.0,
    )
    emission = ph.environment.SourceEmission(sound_power_level=lw)
    metadata = ReportMetadata(
        specimen="Industrial fan plant (point source)",
        client="Example client",
        test_room="Nearest dwelling facade",
        temperature=10.0,
        relative_humidity=70.0,
        pressure=101.3,
        test_date="2026-07-22",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-9613-ATTEN",
        requirement=50.0,  # maximum acceptable A-weighted downwind level
        notes="Point source over porous ground with a noise barrier (ISO 9613-2).",
    )
    return (
        _WithSourceEmission(result, emission),
        metadata,
        "iso9613_outdoor_attenuation_example.pdf",
    )


def _barrier_insertion_loss_example() -> tuple[object, ReportMetadata, str]:
    """ISO 9613-2 family fiche: predicted barrier insertion loss (wave-theoretic).

    A 4 m thin noise barrier 50 m from a source (1 m high), the receiver 1.5 m
    high at 100 m, in the free field. The per-band insertion loss comes from the
    tested wave-theoretic rigid-screen model (see
    tests/environment/propagation/test_ground_barriers.py).
    """
    freqs = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
    result = ph.environment.barrier_insertion_loss(freqs, 1.0, 50.0, 4.0, 100.0, 1.5)
    metadata = ReportMetadata(
        specimen="Roadside noise barrier, 4 m high",
        client="Example client",
        test_room="Dwelling at 100 m",
        test_date="2026-07-22",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-9613-BARRIER",
        requirement=8.0,  # minimum required mean insertion loss
        notes="Thin-screen diffraction, wave-theoretic model.",
    )
    return result, metadata, "iso9613_barrier_insertion_loss_example.pdf"


def _rd1367_example() -> tuple[object, ReportMetadata, str]:
    """RD 1367/2007 fiche: the noise assessment of an activity.

    The published worked case of Aviles Lopez & Perera Martin, Manual de
    acustica ambiental y arquitectonica, Ejemplos 3.1 to 3.3: an activity on
    residential land (acoustic area type a) open from 9 h to 21 h, with a noisy
    machine running from 9 h to 15 h. The two measured noise phases are
    LAeq,5s = 50 dB with Kt = 6 and Kf = 3 dB (so LKeq,5s = 59 dB) and
    LAeq,5s = 48 dB with Kt = 3 and Kf = 3 dB (LKeq,5s = 54 dB). Integrating
    them over the evaluation periods gives LKeq,d = 57 dB and LKeq,e = 51 dB,
    and averaging over the 303 operating days of the year LK,d = 56 dB and
    LK,e = 50 dB. Against the 55 dB of Annex III Table B1 the phase and daily
    criteria are met but the annual LK,d is not, so a new activity does not
    comply.

    The fiche renders in Spanish, the language of the regulation it applies.
    """
    day = [
        ph.environment.NoisePhase(2.0, 0.0, label="Actividad cerrada"),
        ph.environment.NoisePhase(
            6.0, 50.0, kt=6.0, kf=3.0, label="Maquina ruidosa activa"
        ),
        ph.environment.NoisePhase(4.0, 48.0, kt=3.0, kf=3.0, label="Resto de fuentes"),
    ]
    evening = [
        ph.environment.NoisePhase(2.0, 48.0, kt=3.0, kf=3.0, label="Resto de fuentes"),
        ph.environment.NoisePhase(2.0, 0.0, label="Actividad cerrada"),
    ]
    result = ph.environment.assess_activity(
        {"day": day, "evening": evening},
        ph.environment.activity_limits("a"),
        operating_days=303,
    )
    metadata = ReportMetadata(
        specimen="Actividad con maquinaria, horario 9 h a 21 h",
        client="Example client",
        test_room="Ambiente exterior, punto de evaluacion mas desfavorable",
        instrumentation="Sonometro integrador-promediador clase 1",
        calibration="Verificacion antes y despues, desviacion 0,1 dB",
        measurement_standard="RD 1367/2007 Anexo IV",
        test_date="2026-07-29",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-RD1367",
    )
    return result, metadata, "rd1367_activity_example.pdf"
