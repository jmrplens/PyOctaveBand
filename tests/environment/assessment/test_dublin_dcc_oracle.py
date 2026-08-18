#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Period levels of a real monitoring year against a published Lnight.

Source: the Dublin City Council Ambient Sound Monitoring Network open data
(Smart Dublin / data.gov.ie, CC BY 4.0; see ``tests/data/audio/README.md``):
the full 2015 five-minute A-weighted Leq series of two permanent monitors,
Ballymun and Ringsend. The committed npz is the derived series, not audio,
so what it exercises is the machinery downstream of the microphone: sorting
a year of interval levels into day, evening and night, energy-averaging
each period, and combining them into the ISO 1996-1 whole-day descriptors.

The published anchor is the DCC "Ambient Sound Monitoring Network Annual
Report For 2015" (dublincity.ie), p. 11: annual night-time values of
**58 dB(A)** for Ballymun and **59 dB(A)** for Ringsend (four sites named
as exceeding the 55 dB night guideline), with the averaging declared
"logarithmical". Both anchors are integers, so the assertions allow half
of the last published digit (0.5 dB); the two sites reproduce as 57.98
and 58.81 dB, i.e. the rounding convention is pinned from two independent
directions.

The report defines night as 23:00 to 07:00. Day and evening levels per
the END/ISO 1996 split (07-19, 19-23) have no published counterpart in
the report's text, so they are not asserted against it; they feed
:func:`~phonometry.environment.lden` and
:func:`~phonometry.environment.composite_rating_level`, which must agree
with each other and with the hand-written ISO 1996-1:2016 Formula (6).
There is no public function today that takes a timestamped level series
straight to Lden; the period split lives in this test, and a convenience
accepting ``(timestamps, levels)`` could wrap exactly the code below.
"""

from __future__ import annotations

import numpy as np
import oracle_data
import pytest

from phonometry.environment import composite_rating_level, lden, ldn

_NPZ = oracle_data.DATA / "audio" / "dublin" / "dcc_laeq_5min_2015.npz"

#: Published annual night-time (23:00-07:00) levels, dB(A): DCC Annual
#: Report 2015, p. 11.
_PUBLISHED_LNIGHT = {"ballymun": 58.0, "ringsend": 59.0}
#: The report's guideline both sites are named as exceeding.
_NIGHT_GUIDELINE_DB = 55.0

_SECONDS_PER_HOUR = 3600


def _series(site: str) -> tuple[np.ndarray, np.ndarray]:
    """Interval-end timestamps (s since 2015-01-01 00:00 local) and LAeq."""
    with np.load(_NPZ) as f:
        time0 = int(f[f"{site}_time0_s"])
        dt = f[f"{site}_dt_s"]
        laeq = f[f"{site}_laeq_cdb"] / 100.0
    stamps = time0 + np.concatenate(([0], np.cumsum(dt)))
    return stamps, laeq


def _period_levels(site: str) -> tuple[float, float, float]:
    """Energy-averaged (Lday, Levening, Lnight) over the year.

    Each five-minute value is filed by the wall-clock hour its interval
    starts in (the stamps mark interval ends): day 07-19, evening 19-23,
    night 23-07, the default ISO 1996-1 / END split whose night matches
    the report's 23:00-07:00.
    """
    stamps, laeq = _series(site)
    start_hour = ((stamps - 300) // _SECONDS_PER_HOUR) % 24
    night = (start_hour >= 23) | (start_hour < 7)
    day = (start_hour >= 7) & (start_hour < 19)
    evening = (start_hour >= 19) & (start_hour < 23)
    assert int(np.sum(night) + np.sum(day) + np.sum(evening)) == laeq.size

    def energetic(values: np.ndarray) -> float:
        return float(10 * np.log10(np.mean(10 ** (values / 10))))

    return energetic(laeq[day]), energetic(laeq[evening]), energetic(laeq[night])


def test_the_committed_series_is_the_full_year() -> None:
    for site, count in (("ballymun", 104242), ("ringsend", 104657)):
        stamps, laeq = _series(site)
        assert laeq.size == count
        assert stamps.size == count
        # Five-minute cadence with outage gaps: strictly increasing, and
        # the dominant step is 300 s.
        steps = np.diff(stamps)
        assert np.all(steps > 0)
        assert float(np.mean(steps == 300)) > 0.99
        # A year of stamps, within the monitors' start/stop margins.
        assert 364 * 24 <= (stamps[-1] - stamps[0]) / _SECONDS_PER_HOUR <= 366 * 24


@pytest.mark.parametrize("site", ["ballymun", "ringsend"])
def test_annual_lnight_matches_the_published_value(site: str) -> None:
    _, _, lnight = _period_levels(site)
    published = _PUBLISHED_LNIGHT[site]
    # Published as a whole number: half of the last digit either way.
    assert lnight == pytest.approx(published, abs=0.5)
    assert round(lnight) == published
    assert lnight > _NIGHT_GUIDELINE_DB


@pytest.mark.parametrize("site", ["ballymun", "ringsend"])
def test_period_machinery_agrees_on_lden_and_ldn(site: str) -> None:
    """lden/ldn/composite_rating_level over the year's period levels.

    ISO 1996-1:2016 Formula (6) written out by hand must equal both the
    dedicated ``lden`` and the general ``composite_rating_level`` on the
    same three periods (12 h day, 4 h evening + 5 dB, 8 h night + 10 dB).
    """
    lday, levening, lnight = _period_levels(site)
    by_formula = 10 * np.log10(
        (
            12 * 10 ** (0.1 * lday)
            + 4 * 10 ** (0.1 * (levening + 5.0))
            + 8 * 10 ** (0.1 * (lnight + 10.0))
        )
        / 24.0
    )
    value = lden(lday, levening, lnight)
    assert value == pytest.approx(by_formula, abs=1e-12)
    composite = composite_rating_level(
        [(lday, 12.0, 0.0), (levening, 4.0, 5.0), (lnight, 8.0, 10.0)]
    )
    assert composite == pytest.approx(value, abs=1e-12)
    # The night penalty must lift Lden above the plain 24 h level.
    l24 = composite_rating_level(
        [(lday, 12.0, 0.0), (levening, 4.0, 0.0), (lnight, 8.0, 0.0)]
    )
    assert value > l24
    # Ldn uses its own 15 h + 9 h split (day 07-22, night 22-07), so the
    # series is refiled before the identity is asserted.
    stamps, laeq = _series(site)
    start_hour = ((stamps - 300) // _SECONDS_PER_HOUR) % 24
    night_9h = (start_hour >= 22) | (start_hour < 7)
    lday_15h = float(10 * np.log10(np.mean(10 ** (laeq[~night_9h] / 10))))
    lnight_9h = float(10 * np.log10(np.mean(10 ** (laeq[night_9h] / 10))))
    day_night = ldn(lday_15h, lnight_9h, hours=(15.0, 9.0))
    assert day_night == pytest.approx(
        composite_rating_level([(lday_15h, 15.0, 0.0), (lnight_9h, 9.0, 10.0)]),
        abs=1e-12,
    )
