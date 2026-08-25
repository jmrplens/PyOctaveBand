#  Copyright (c) 2026. Jose Manuel Requena Plens
"""EN/ES language option of the hearing ``.plot()`` renderers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from phonometry import hearing

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _legend_labels(ax: Axes) -> list[str]:
    handles, labels = ax.get_legend_handles_labels()
    assert handles
    return labels


def test_fractile_band_legend_follows_the_language() -> None:
    """The shaded fractile band's legend entry is localised like the rest."""
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    mpl.use("Agg")
    for result in (
        hearing.age_threshold(60.0, "male", fractile=0.9),
        hearing.nipts(95.0, 20.0, 0.9),
    ):
        assert "10-90 % fractile band" in _legend_labels(result.plot(language="en"))
        labels_es = _legend_labels(result.plot(language="es"))
        assert "Banda de fractiles 10-90 %" in labels_es
        assert "10-90 % fractile band" not in labels_es


def test_spanish_labels() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    mpl.use("Agg")
    res = hearing.age_threshold(60.0, "male", fractile=0.9)

    ax_en = res.plot(language="en")
    assert ax_en.get_ylabel() == "Threshold deviation from age 18 [dB]"

    ax_es = res.plot(language="es")
    assert ax_es.get_ylabel() == "Desviación del umbral respecto a 18 años [dB]"
    assert "umbral de audición" in ax_es.get_title()


def test_unknown_language_raises() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    mpl.use("Agg")
    result = hearing.age_threshold(60.0, "male")
    with pytest.raises(ValueError, match="Unknown language"):
        result.plot(language="xx")


def test_exposure_plot_needs_per_task_contributions() -> None:
    """The job and full-day strategies carry no tasks, so plot() says so."""
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    mpl.use("Agg")

    result = hearing.full_day_exposure([85.0, 86.0, 84.0], 8.0)
    assert result.tasks == ()
    with pytest.raises(ValueError, match="per-task contributions"):
        result.plot()
