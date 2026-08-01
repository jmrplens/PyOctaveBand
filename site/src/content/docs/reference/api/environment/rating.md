---
title: "environmental.rating"
description: "Environmental noise descriptors per ISO 1996-1:2016."
sidebar:
  label: "rating"
---

Environmental noise descriptors per ISO 1996-1:2016.

Day-evening-night (Lden, 3.6.4) and day-night (Ldn, 3.6.5) sound levels,
and composite whole-day rating levels (6.5, Formulae 5-6).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## composite_rating_level

```python
composite_rating_level(
    periods: Iterable[tuple[float, float, float]],
) -> float
```

Composite whole-day rating level (ISO 1996-1:2016, 6.5).

Generalizes Formulae (5) and (6): each period contributes its rating
level plus adjustment, weighted by its share of the 24 h day:

$$
10 \log_{10}\left[ \sum_i \frac{h_i}{24} \cdot 10^{0.1 (L_i + K_i)} \right]
$$

**Parameters**

| Name | Description |
| :--- | :--- |
| `periods` | Iterable of `(level_db, hours, adjustment_db)` tuples. `level_db` is the period's (rating) equivalent continuous level, `hours` its duration and `adjustment_db` the time-of-day or source adjustment K (e.g. ISO 1996-1 Table A.1: evening 5 dB, night 10 dB). Hours must be positive and sum to 24. |

**Returns:** Composite rating level in dB.

## lden

```python
lden(
    lday: float,
    levening: float,
    lnight: float,
    hours: tuple[float, float, float] = (12.0, 4.0, 8.0),
) -> float
```

Day-evening-night sound level Lden (ISO 1996-1:2016, 3.6.4).

$$
L_{den} = 10 \log_{10}\left\{ (1/24) \left[ t_d \cdot 10^{0.1 L_{day}} + t_e \cdot 10^{0.1 (L_{evening}+5)} + t_n \cdot 10^{0.1 (L_{night}+10)} \right] \right\}
$$

**Parameters**

| Name | Description |
| :--- | :--- |
| `lday` | LAeq over the day period, in dB. |
| `levening` | LAeq over the evening period, in dB (+5 dB weighting). |
| `lnight` | LAeq over the night period, in dB (+10 dB weighting). |
| `hours` | `(t_day, t_evening, t_night)` in hours, summing to 24. Default (12, 4, 8); countries may define the periods differently (3.6.4 Note 1). |

**Returns:** Lden in dB.

## ldn

```python
ldn(
    lday: float,
    lnight: float,
    hours: tuple[float, float] = (15.0, 9.0),
) -> float
```

Day-night sound level Ldn (ISO 1996-1:2016, 3.6.5).

$$
L_{dn} = 10 \log_{10}\left\{ (1/24) \left[ t_d \cdot 10^{0.1 L_{day}} + t_n \cdot 10^{0.1 (L_{night}+10)} \right] \right\}
$$

**Parameters**

| Name | Description |
| :--- | :--- |
| `lday` | LAeq over the day period, in dB. |
| `lnight` | LAeq over the night period, in dB (+10 dB weighting). |
| `hours` | `(t_day, t_night)` in hours, summing to 24. Default (15, 9). |

**Returns:** Ldn in dB.
