#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Behaviour of the room-to-room chain: closed forms, options and validation.

Oracles: the closed form of Norton & Karczub 2e Equation (4.101),
``NR = TL - 10 lg[S_w / (S_2 alpha_2 + tau S_w)]``, evaluated here with plain
arithmetic; the identity ``NR = TL`` when the receiving-room absorption equals
the partition area; and the exact ``10 lg Q`` of the sound power models of
Table 4.5. The published worked answers live in
``test_room_to_room_norton.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry import DesignCriterion, SourceRoom, room_to_room_transmission
from phonometry.noise_control.room_to_room import RoomToRoomResult

_BANDS = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
_SOURCE = np.array([90.0, 88.0, 86.0, 84.0, 82.0, 80.0])
#: A flat receiving-room absorption equal to the partition area, so the
#: logarithm of Eq. (4.101) vanishes and NR is exactly TL.
_ABSORPTION = np.full(6, 20.0)
_NAN_SPECTRUM = np.full(6, np.nan)
_NO_ABSORPTION = np.zeros(6)
_FLAT_SOURCE = np.full(6, 95.0)
_NAN = float("nan")


def _chain(**kwargs: object) -> RoomToRoomResult:
    """A plain six-band chain with round numbers, for the option tests."""
    defaults: dict[str, object] = {
        "source": SourceRoom(level=_SOURCE),
        "criterion": DesignCriterion(family="NC"),
    }
    defaults.update(kwargs)
    return room_to_room_transmission(
        _BANDS, 40.0, 20.0, _ABSORPTION, **defaults  # type: ignore[arg-type]
    )


def test_noise_reduction_equals_transmission_loss_when_areas_match() -> None:
    """``S_2 alpha_2 = S_w`` makes the logarithm vanish, so ``NR = TL`` exactly."""
    result = _chain()
    assert np.allclose(result.noise_reduction, 40.0)
    assert np.allclose(result.received_level, result.source_level - 40.0)


def test_noise_reduction_closed_form() -> None:
    """Equation (4.101) reproduced band by band with plain arithmetic."""
    absorption = np.array([5.0, 10.0, 20.0, 40.0, 60.0, 80.0])
    tl = np.array([20.0, 25.0, 30.0, 35.0, 40.0, 45.0])
    result = room_to_room_transmission(
        _BANDS, tl, 12.0, absorption, source=SourceRoom(level=100.0)
    )
    expected = [
        t - 10.0 * math.log10(12.0 / a) for t, a in zip(tl, absorption, strict=True)
    ]
    assert np.allclose(result.noise_reduction, expected)


def test_partition_transmission_term_adds_absorption() -> None:
    """``include_partition_transmission`` adds ``tau S_w`` to the denominator."""
    plain = _chain()
    with_term = _chain(include_partition_transmission=True)
    tau = 10.0 ** (-40.0 / 10.0)
    expected = 40.0 - 10.0 * math.log10(20.0 / (20.0 + 20.0 * tau))
    assert np.allclose(with_term.noise_reduction, expected)
    assert np.all(with_term.noise_reduction > plain.noise_reduction)


def test_flanking_penalty_is_a_straight_debit() -> None:
    """The penalty comes off the noise reduction and lands on the received level."""
    plain = _chain()
    penalised = _chain(criterion=DesignCriterion(flanking_penalty=3.0))
    assert np.allclose(penalised.noise_reduction, plain.noise_reduction - 3.0)
    assert np.allclose(penalised.received_level, plain.received_level + 3.0)
    assert penalised.flanking_penalty == pytest.approx(3.0)
    kinds = [row["kind"] for row in penalised.table()]
    assert "flanking" in kinds
    assert "flanking" not in [row["kind"] for row in plain.table()]


@pytest.mark.parametrize(
    ("model", "expected"),
    [("constant_power", 0.0), ("constant_volume", 6.0206), ("constant_pressure", -6.0206)],
)
def test_source_power_models_scale_by_ten_lg_q(model: str, expected: float) -> None:
    """Norton Table 4.5: the radiated power goes as ``Q^0``, ``Q^1``, ``Q^-1``."""
    result = room_to_room_transmission(
        _BANDS,
        40.0,
        20.0,
        _ABSORPTION,
        source=SourceRoom(
            power_level=100.0, room_constant=50.0, directivity=4.0, model=model
        ),
    )
    reverberant = 100.0 + 10.0 * math.log10(4.0 / 50.0)
    assert np.allclose(result.source_level, reverberant + expected, atol=1e-3)
    assert result.source_power_level is not None
    assert np.allclose(result.source_power_level, 100.0)


def test_required_transmission_loss_closes_the_gap() -> None:
    """The required ``TL`` is the current one plus the criterion exceedance."""
    result = _chain(criterion=DesignCriterion(target=45.0))
    required = result.required_transmission_loss
    excess = result.exceedance
    assert required is not None
    assert excess is not None
    assert np.allclose(required - result.transmission_loss, excess)
    # Re-running with that TL lands exactly on the criterion curve.
    tightened = room_to_room_transmission(
        _BANDS, required, 20.0, _ABSORPTION,
        source=SourceRoom(level=result.source_level),
        criterion=DesignCriterion(target=45.0),
    )
    assert np.allclose(tightened.received_level, result.criterion_curve)
    assert tightened.meets_target is True


def test_no_target_leaves_the_verdicts_undefined() -> None:
    """Without a target there is no curve, no exceedance and no verdict."""
    result = _chain()
    assert result.criterion_curve is None
    assert result.exceedance is None
    assert result.meets_target is None
    assert result.required_transmission_loss is None
    assert "criterion" not in [row["kind"] for row in result.table()]


def test_rc_criterion_family() -> None:
    """``family="RC"`` rates against the RC Mark II curves of Annex D."""
    result = _chain(criterion=DesignCriterion(family="RC", target=35.0))
    assert result.criterion == "RC"
    assert result.criterion_curve is not None
    # The six bands of this chain are short of the 31.5 Hz to 4 kHz span
    # clause D.4 asks for, which the rating reports as a warning.
    with pytest.warns(UserWarning, match="D.4"):
        assert result.rating.__class__.__name__ == "RCResult"


def test_nc_rating_of_the_received_spectrum() -> None:
    """``rating`` is the ANSI S12.2 rating of the received spectrum."""
    result = room_to_room_transmission(
        _BANDS,
        [50.0, 55.0, 60.0, 62.0, 62.0, 62.0],
        20.0,
        _ABSORPTION,
        source=SourceRoom(level=_FLAT_SOURCE),
        criterion=DesignCriterion(target=30.0),
    )
    assert result.rating.__class__.__name__ == "NCResult"
    assert np.isfinite(result.rating.rating)


def test_source_description_is_exclusive() -> None:
    """Exactly one of ``source.level`` / ``source.power_level`` is required."""
    with pytest.raises(ValueError, match="exactly one"):
        room_to_room_transmission(_BANDS, 40.0, 20.0, _ABSORPTION)
    with pytest.raises(ValueError, match="exactly one"):
        room_to_room_transmission(
            _BANDS, 40.0, 20.0, _ABSORPTION,
            source=SourceRoom(level=90.0, power_level=100.0),
        )
    with pytest.raises(ValueError, match="room_constant"):
        room_to_room_transmission(
            _BANDS, 40.0, 20.0, _ABSORPTION, source=SourceRoom(power_level=100.0)
        )


def test_validation() -> None:
    """Malformed bands, spectra, areas, absorption and options are rejected."""
    source = SourceRoom(level=90.0)
    with pytest.raises(ValueError, match="non-empty 1-D"):
        room_to_room_transmission([], 40.0, 20.0, 20.0, source=source)
    with pytest.raises(ValueError, match="one value per band"):
        room_to_room_transmission(
            _BANDS, [40.0, 41.0], 20.0, _ABSORPTION, source=source
        )
    with pytest.raises(ValueError, match="finite"):
        room_to_room_transmission(
            _BANDS, _NAN_SPECTRUM, 20.0, _ABSORPTION, source=source
        )
    with pytest.raises(ValueError, match="partition_area"):
        room_to_room_transmission(_BANDS, 40.0, 0.0, _ABSORPTION, source=source)
    with pytest.raises(ValueError, match="receiving_absorption"):
        room_to_room_transmission(_BANDS, 40.0, 20.0, _NO_ABSORPTION, source=source)
    with pytest.raises(ValueError, match="flanking_penalty"):
        _chain(criterion=DesignCriterion(flanking_penalty=-1.0))
    with pytest.raises(ValueError, match="criterion"):
        _chain(criterion=DesignCriterion(family="NR"))
    with pytest.raises(ValueError, match="model"):
        _chain(source=SourceRoom(level=_SOURCE, model="constant_energy"))
    with pytest.raises(ValueError, match="target"):
        _chain(criterion=DesignCriterion(target=_NAN))


def test_plot_smoke() -> None:
    """``.plot()`` draws the two spectra, the criterion curve and the NR axis."""
    import matplotlib

    matplotlib.use("Agg")
    result = _chain(criterion=DesignCriterion(target=45.0, flanking_penalty=2.0))
    ax = result.plot()
    assert ax.get_ylabel() == "Level [dB]"
    assert ax.get_title().startswith("Room-to-room transmission")
    ax_es = result.plot(language="es")
    assert ax_es.get_ylabel() == "Nivel [dB]"
    with pytest.raises(ValueError):
        result.plot(language="fr")
