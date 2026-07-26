---
title: "metrology.frequencies"
description: "Frequency calculation logic according to ANSI/IEC standards."
sidebar:
  label: "frequencies"
---

Frequency calculation logic according to ANSI/IEC standards.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## getansifrequencies

```python
getansifrequencies(
    fraction: float,
    limits: list[float] | None = None,
) -> tuple[list[float], list[float], list[float], list[str]]
```

Deprecated alias of [`nominal_frequencies`](/phonometry/reference/api/filters/frequencies/#nominal_frequencies).

## nominal_frequencies

```python
nominal_frequencies(
    fraction: float,
    limits: list[float] | None = None,
) -> tuple[list[float], list[float], list[float], list[str]]
```

Calculate frequencies according to ANSI/IEC standards.

**Parameters**

| Name | Description |
| :--- | :--- |
| `fraction` | Bandwidth fraction (e.g., 1, 3). |
| `limits` | [f_min, f_max] limits. |

**Returns:** Tuple of (center_freqs, lower_edges, upper_edges, nominal_labels).

## normalized_frequencies

```python
normalized_frequencies(fraction: int) -> list[float]
```

Get standardized IEC center frequencies.

**Parameters**

| Name | Description |
| :--- | :--- |
| `fraction` | 1 or 3 (Octave or 1/3 Octave). |

**Returns:** List of standard frequencies.

## normalizedfreq

```python
normalizedfreq(fraction: int) -> list[float]
```

Deprecated alias of [`normalized_frequencies`](/phonometry/reference/api/filters/frequencies/#normalized_frequencies).
