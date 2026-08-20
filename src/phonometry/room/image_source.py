#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Synthetic room impulse response by the image-source method (rectangular room).

A rigid-walled (or absorbing) rectangular room -- a *shoebox* -- reflects a
point source in its six walls. Each reflection is equivalent to the free-field
sound of a mirror image of the source, so the room impulse response (RIR) is
the sum of the direct sound and one delayed, attenuated pulse per image
(Kuttruff, *Room Acoustics* 6th ed., 4.1, Equations (4.4)-(4.5); Vorlander,
*Auralization* 2nd ed., 11.4, the Allen-Berkley/Borish construction). This is
the deterministic complement of the statistical reverberation-time formulae of
:mod:`phonometry.room.reverberation_prediction`: where those give a single
decay rate, the image-source model gives the whole early reflection pattern and
the decay it implies.

**Image lattice.** For a room
:math:`[0, L_x] \times [0, L_y] \times [0, L_z]` with the source at
``(xs, ys, zs)``, mirroring a coordinate in a wall (Vorlander Equation (11.36),
:math:`S_n = S - 2 d n`) turns the source into a regular lattice of images.
Along one axis the images sit at :math:`2 n L \pm x` for every integer ``n``
and the two
mirror parities; the parity index ``p`` and the lattice index ``n`` give the
number of reflections off the two walls of that axis as :math:`|n - p|`
(wall at 0)
and :math:`|n|` (wall at ``L``), so the total reflection order of an image is
:math:`|2 n_x - p_x| + |2 n_y - p_y| + |2 n_z - p_z|` (Allen & Berkley,
*J. Acoust.
Soc. Am.* 65 (1979) 943). The audible images up to order ``i0`` number
:math:`(2/3)(2 i_0^3 + 3 i_0^2 + 4 i_0)` in a shoebox (Kuttruff Equation
(9.23)); the
temporal density of reflections grows as
:math:`dN/dt = 4 \pi c^3 t^2 / V`
(Kuttruff Equation (4.6)).

**Per-image contribution.** Image ``i`` at distance ``r_i`` from the receiver
arrives at :math:`t_i = r_i / c` (Vorlander Equation (11.38)) with amplitude

.. math::

   A_i = \left[ \prod_{\text{walls}} R_{\text{wall}}^{n_{\text{wall}}}
   \right] \frac{e^{-m r_i / 2}}{4 \pi r_i}

with :math:`n_{\text{wall}}` the reflections that image made off each wall:
the :math:`1 / (4 \pi r_i)` spherical spreading, the product of the wall
*pressure* reflection factors :math:`R = \sqrt{1 - \alpha}` (Vorlander
Equation
(11.39); :math:`|R|^2 = 1 - \alpha` in energy, Kuttruff 4.1) each raised to
the
number of reflections that image made off that wall, and the air pressure
attenuation :math:`e^{-m r_i / 2}` over the path (Kuttruff 4.1; ``m`` the
*intensity* attenuation constant, so intensity falls as :math:`e^{-m r}`).
The RIR
is the sum of unit impulses at :math:`t_i` weighted by :math:`A_i` (Kuttruff
Equation
(4.5), :math:`g(t) = \sum_i A_i \delta(t - t_i)`), assembled broadband from a
single
absorption set or one curve per octave band from per-band coefficients.

The Schroeder backward integral of the synthetic RIR (see
:func:`phonometry.room.decay_curve`) reproduces the Eyring reverberation time
:math:`T = -24 V \ln 10 / (c S \ln(1 - \bar{\alpha}))` (Kuttruff Equation
(5.23)) of the
same room to within a few percent, closing the loop between this deterministic
model and the statistical prediction. The construction is exact only for walls
whose reflection factor is real and angle-independent (Kuttruff 4.1: exact for
a specific wall impedance of +-1, a good approximation when the source stands a
few wavelengths from every wall); it captures specular reflections only, with
no diffraction or diffuse scattering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.validation import require_positive

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray

#: Default speed of sound ``c0`` (20 degC dry air), in m/s.
DEFAULT_SPEED_OF_SOUND = 343.0

#: Wall order for a six-tuple of coefficients: the low and high wall of each
#: axis. ``x0``/``xL`` are the walls at ``x = 0`` / ``x = Lx`` and so on.
WALL_ORDER = ("x0", "xL", "y0", "yL", "z0", "zL")


def _resolve_walls(
    absorption: ArrayLike, n_freq: int | None
) -> tuple[NDArray[np.float64], int]:
    r"""Validate the absorption spec into a ``(6, n_bands)`` reflection map.

    Accepts a scalar (uniform on every wall and band), a length-6 per-wall
    vector, a per-band vector (uniform across walls) or a ``(6, n_bands)``
    per-wall per-band array. Returns ``(R, n_bands)`` with the *pressure*
    reflection factor :math:`R = \sqrt{1 - \alpha}` (Vorlander Equation
    (11.39)), shaped ``(6, n_bands)``.

    ``n_freq`` (the length of the caller's ``frequencies``, or ``None``)
    disambiguates a length-6 1D array: with a matching ``frequencies`` it is
    read as a per-band curve (uniform on every wall), otherwise as the six
    per-wall coefficients. A single band cannot be both, so declaring the
    bands via ``frequencies`` resolves the collision.

    :raises ValueError: for a non-finite coefficient, a coefficient outside
        ``[0, 1]`` (a wall cannot reflect more energy than it receives, so
        unlike the Sabine models the edge-effect :math:`\alpha > 1` is rejected
        here), or a shape that is neither a scalar, length 6, a band vector
        nor ``(6, n_bands)``.
    """
    alpha = np.asarray(absorption, dtype=np.float64)
    if not np.all(np.isfinite(alpha)):
        raise ValueError("absorption coefficients must be finite.")
    if np.any(alpha < 0.0) or np.any(alpha > 1.0):
        raise ValueError(
            "absorption coefficients must lie in [0, 1]: the image-source "
            "reflection factor R = sqrt(1 - alpha) has no real value for "
            "alpha > 1, so the edge-effect coefficients above 1 that Sabine "
            "tolerates must be capped at 1 here."
        )
    if alpha.ndim == 0:
        walls = np.full((6, 1), float(alpha), dtype=np.float64)
    elif alpha.ndim == 1 and alpha.shape[0] == 6 and n_freq != 6:
        # Length 6 with no matching frequencies -> the six per-wall values.
        walls = alpha[:, np.newaxis]
    elif alpha.ndim == 1:
        # A per-band curve, uniform across walls (this branch also handles a
        # length-6 curve when frequencies declares six bands).
        walls = np.broadcast_to(alpha, (6, alpha.shape[0])).copy()
    elif alpha.ndim == 2 and alpha.shape[0] == 6:
        walls = alpha
    else:
        raise ValueError(
            "'absorption' must be a scalar, a length-6 per-wall vector, a "
            "per-band vector, or a (6, n_bands) per-wall per-band array; got "
            f"shape {alpha.shape}."
        )
    return np.sqrt(1.0 - walls), walls.shape[1]


def _axis_images(
    coord: float, length: float, max_order: int
) -> tuple[NDArray[np.float64], NDArray[np.int_], NDArray[np.int_]]:
    r"""Image positions and per-wall reflection counts along one axis.

    Enumerates the lattice index ``n`` and parity ``p`` with axial order
    :math:`\lvert 2 n - p \rvert \le \text{max\_order}`. Returns
    ``(positions, counts_low, counts_high)`` where
    :math:`\text{counts\_low} = \lvert n - p \rvert` reflections off the wall
    at ``0`` and :math:`\text{counts\_high} = \lvert n \rvert` off the wall at
    ``length``.
    """
    positions: list[float] = []
    low: list[int] = []
    high: list[int] = []
    for n in range(-max_order, max_order + 1):
        for p in (0, 1):
            n_low = abs(n - p)
            n_high = abs(n)
            if n_low + n_high > max_order:
                continue
            positions.append(2.0 * n * length + (1.0 - 2.0 * p) * coord)
            low.append(n_low)
            high.append(n_high)
    return (
        np.asarray(positions, dtype=np.float64),
        np.asarray(low, dtype=np.int_),
        np.asarray(high, dtype=np.int_),
    )


@dataclass(frozen=True)
class ImageSourceResult:
    r"""Synthetic room impulse response by the image-source method.

    ``ir`` is the sampled RIR: a 1D array for a broadband model, or a
    ``(n_bands, n_samples)`` array with one decay per octave band for per-band
    absorption. Each image contributes a unit impulse at the nearest sample to
    its exact arrival time; the exact, sub-sample reflection table is kept
    separately in ``times`` / ``distances`` / ``orders`` / ``amplitudes`` /
    ``image_positions`` so the geometry stays exact regardless of ``fs``.

    :ivar ir: Sampled impulse response (Kuttruff Equation (4.5)); shape
        ``(n_samples,)`` (broadband) or ``(n_bands, n_samples)`` (per band).
    :ivar fs: Sample rate, Hz.
    :ivar frequencies: Band centre frequencies, Hz, or ``None`` for a
        broadband model.
    :ivar times: Exact arrival time :math:`t_i = r_i / c` of every image, s
        (sorted ascending).
    :ivar distances: Image-to-receiver distance ``r_i``, m (aligned with
        ``times``).
    :ivar orders: Total reflection order of every image (aligned with
        ``times``).
    :ivar amplitudes: Exact per-image amplitude ``A_i``; shape ``(n_images,)``
        (broadband) or ``(n_bands, n_images)`` (per band).
    :ivar image_positions: Image-source coordinates ``(x, y, z)``, m, shape
        ``(n_images, 3)`` (aligned with ``times``).
    :ivar dimensions: Room lengths ``(Lx, Ly, Lz)``, m.
    :ivar source: Source position ``(x, y, z)``, m.
    :ivar receiver: Receiver position ``(x, y, z)``, m.
    :ivar max_order: Reflection-order cut-off used.
    :ivar speed_of_sound: Speed of sound ``c``, m/s.
    """

    ir: np.ndarray
    fs: int
    frequencies: np.ndarray | None
    times: np.ndarray
    distances: np.ndarray
    orders: np.ndarray
    amplitudes: np.ndarray
    image_positions: np.ndarray
    dimensions: tuple[float, float, float]
    source: tuple[float, float, float]
    receiver: tuple[float, float, float]
    max_order: int
    speed_of_sound: float

    @property
    def direct_time(self) -> float:
        """Arrival time of the direct sound (order 0), s."""
        return float(self.times[int(np.argmin(self.distances))])

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        r"""Plot the reflectogram: reflection level in dB against arrival time.

        Stems the per-image amplitudes (in dB re the direct sound), coloured by
        reflection order, with the :math:`1 / r` free-field envelope overlaid.
        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.room import plot_image_source_reflectogram

        check_language(language)
        return plot_image_source_reflectogram(self, ax=ax, language=language, **kwargs)

    def plot_geometry(
        self,
        ax: Axes | None = None,
        *,
        max_order: int | None = None,
        language: str = "en",
        **kwargs: Any,
    ) -> Axes:
        """Draw the image-source lattice in plan view (x-y), to scale.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param max_order: Highest reflection order drawn; ``None`` draws up
            to order 3 (or the result's own maximum if lower).
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the room-outline rectangle.
        """
        from .._i18n import check_language
        from .._plot.geometry import plot_image_source_geometry

        check_language(language)
        return plot_image_source_geometry(
            self, ax=ax, max_order=max_order, language=language, **kwargs
        )


def _validate_point(
    point: ArrayLike, dimensions: tuple[float, float, float], name: str
) -> NDArray[np.float64]:
    """Validate a 3-vector strictly inside the room box."""
    p = np.asarray(point, dtype=np.float64)
    if p.shape != (3,):
        raise ValueError(f"'{name}' must be a 3-vector (x, y, z).")
    if not np.all(np.isfinite(p)):
        raise ValueError(f"'{name}' coordinates must be finite.")
    for value, length, axis in zip(p, dimensions, "xyz"):
        if not 0.0 < value < length:
            raise ValueError(
                f"'{name}' coordinate {axis} = {value} lies outside the room "
                f"(0, {length}); source and receiver must be strictly inside."
            )
    return p


def _resolve_band_maps(
    reflection: NDArray[np.float64],
    n_alpha_bands: int,
    air_attenuation: ArrayLike,
    freq: np.ndarray | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, bool]:
    """Resolve the per-band reflection map, air map, band count and flag.

    Returns ``(reflection, m_bands, n_bands, banded)`` with ``reflection`` and
    ``m_bands`` broadcast to ``n_bands`` bands (``banded`` False for a single
    broadband curve).

    :raises ValueError: for a non-finite/negative air attenuation, or an
        ``absorption`` / ``air_attenuation`` band count that neither is 1 nor
        matches the resolved band count.
    """
    m = np.asarray(air_attenuation, dtype=np.float64)
    if not np.all(np.isfinite(m)) or np.any(m < 0.0):
        raise ValueError("'air_attenuation' must be finite and non-negative.")
    banded = freq is not None or n_alpha_bands > 1 or m.size > 1
    if freq is not None:
        n_bands = freq.size
    elif banded:
        n_bands = max(n_alpha_bands, m.size)
    else:
        n_bands = 1
    if reflection.shape[1] == 1 and n_bands > 1:
        reflection = np.broadcast_to(reflection, (6, n_bands)).copy()
    elif reflection.shape[1] not in (1, n_bands):
        raise ValueError(
            f"'absorption' has {reflection.shape[1]} bands but the result has "
            f"{n_bands}; a per-band 'absorption' must match the band count of "
            "'frequencies' (or the air attenuation)."
        )
    if m.size not in (1, n_bands):
        raise ValueError(
            f"'air_attenuation' has {m.size} bands but the result has "
            f"{n_bands}; they must match."
        )
    return reflection, np.broadcast_to(m, (n_bands,)), n_bands, banded


def _image_lattice(
    src: NDArray[np.float64],
    rcv: NDArray[np.float64],
    dims: tuple[float, float, float],
    max_order: int,
    speed_of_sound: float,
) -> tuple[NDArray[np.float64], ...]:
    """Enumerate the shoebox image lattice up to ``max_order`` reflections.

    Returns, sorted by arrival time, ``(times, distances, orders, positions,
    counts)`` where ``counts`` is the ``(6, n_images)`` per-wall reflection
    count of each image.
    """
    lx, ly, lz = dims
    xs, x_lo, x_hi = _axis_images(src[0], lx, max_order)
    ys, y_lo, y_hi = _axis_images(src[1], ly, max_order)
    zs, z_lo, z_hi = _axis_images(src[2], lz, max_order)
    order_x = x_lo + x_hi
    order_y = y_lo + y_hi
    order_z = z_lo + z_hi
    # Outer combination with the total-order cut-off, kept vectorised.
    total_order = (
        order_x[:, None, None] + order_y[None, :, None] + order_z[None, None, :]
    )
    ix, iy, iz = np.nonzero(np.asarray(total_order <= max_order))

    px, py, pz = xs[ix], ys[iy], zs[iz]
    positions = np.stack([px, py, pz], axis=1)
    distances = np.sqrt((px - rcv[0]) ** 2 + (py - rcv[1]) ** 2 + (pz - rcv[2]) ** 2)
    orders = (order_x[ix] + order_y[iy] + order_z[iz]).astype(np.int_)
    counts = np.stack(
        [x_lo[ix], x_hi[ix], y_lo[iy], y_hi[iy], z_lo[iz], z_hi[iz]], axis=0
    ).astype(np.float64)

    times = distances / speed_of_sound
    order_idx = np.argsort(times, kind="stable")
    return (
        times[order_idx],
        distances[order_idx],
        orders[order_idx],
        positions[order_idx],
        counts[:, order_idx],
    )


def _image_amplitudes(
    reflection: NDArray[np.float64],
    counts: NDArray[np.float64],
    m_bands: NDArray[np.float64],
    distances: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Per-band per-image amplitude
    :math:`\prod R^{\text{count}} \exp(-m r / 2)/(4 \pi r)`.

    The reflection product is formed in log space so a fully absorbing wall
    (:math:`R = 0`) is handled by an explicit annihilation mask rather than a
    ``0 ** 0`` indeterminate form: only images that actually reflect off that
    wall are zeroed.
    """
    with np.errstate(divide="ignore"):
        log_r = np.log(np.where(reflection > 0.0, reflection, 1.0))
    reflection_gain = np.exp(log_r.T @ counts)  # (n_bands, n_images)
    zero_wall = (reflection <= 0.0)[:, :, None]  # (6, n_bands, 1)
    used = counts[:, None, :] > 0  # (6, 1, n_images)
    annihilated = np.any(zero_wall & used, axis=0)  # (n_bands, n_images)
    reflection_gain = np.where(annihilated, 0.0, reflection_gain)
    spreading = 1.0 / (4.0 * np.pi * distances)
    air = np.exp(-0.5 * m_bands[:, None] * distances[None, :])
    return reflection_gain * air * spreading[None, :]


def _sample_ir(
    times: NDArray[np.float64],
    amp: NDArray[np.float64],
    fs: int,
    duration: float | None,
    n_bands: int,
) -> NDArray[np.float64]:
    """Accumulate the per-band impulses at the nearest sample to each delay."""
    if duration is None:
        duration = float(times[-1]) + 1.0 / fs
    else:
        duration = require_positive(duration, "duration")
    n_samples = int(np.ceil(duration * fs))
    if n_samples < 1:
        raise ValueError("'duration' must cover at least one sample.")
    sample_idx = np.rint(times * fs).astype(np.int64)
    inside = sample_idx < n_samples
    ir = np.zeros((n_bands, n_samples), dtype=np.float64)
    for b in range(n_bands):
        np.add.at(ir[b], sample_idx[inside], amp[b, inside])
    return ir


def image_source_rir(
    dimensions: tuple[float, float, float],
    source: ArrayLike,
    receiver: ArrayLike,
    absorption: ArrayLike,
    *,
    fs: int,
    max_order: int = 20,
    speed_of_sound: float = DEFAULT_SPEED_OF_SOUND,
    air_attenuation: ArrayLike = 0.0,
    duration: float | None = None,
    frequencies: ArrayLike | None = None,
) -> ImageSourceResult:
    r"""Synthetic room impulse response of a shoebox by the image-source method.

    Builds every image of the source up to reflection order ``max_order``
    (Vorlander Equation (11.36); Allen & Berkley 1979), then assembles the RIR
    as the sum of the direct sound and one attenuated, delayed unit impulse per
    image (Kuttruff Equations (4.4)-(4.5)). Each image at distance ``r``
    arrives at :math:`r / c` (Vorlander Equation (11.38)) with amplitude
    :math:`\left[ \prod R_{\text{wall}}^{n_{\text{wall}}} \right]
    e^{-m r / 2} / (4 \pi r)`: the :math:`1 / (4 \pi r)`
    spherical spreading, the product of the wall pressure reflection factors
    :math:`R = \sqrt{1 - \alpha}` (Vorlander Equation (11.39)) over the
    reflections
    the image made, and the air pressure attenuation :math:`e^{-m r / 2}`.

    With scalar or per-wall ``absorption`` (and no ``frequencies``) the result
    is a broadband RIR; a per-band ``absorption`` (or a given ``frequencies``)
    produces one RIR per band. The Schroeder decay of the synthetic RIR
    reproduces the Eyring reverberation time of the room (Kuttruff Equation
    (5.23)); feed ``ir`` straight to :func:`phonometry.room.decay_curve` or
    :func:`phonometry.room.room_parameters`.

    :param dimensions: Room lengths ``(Lx, Ly, Lz)``, m.
    :param source: Source position ``(x, y, z)``, m, strictly inside the room.
    :param receiver: Receiver position ``(x, y, z)``, m, strictly inside.
    :param absorption: Wall absorption coefficient(s) in ``[0, 1]``: a scalar
        (uniform), a length-6 per-wall vector (order
        :data:`WALL_ORDER`), a per-band vector, or a ``(6, n_bands)`` per-wall
        per-band array. A length-6 vector is read as the six per-wall values
        *unless* ``frequencies`` declares six bands, in which case it is a
        per-band curve (uniform across walls); use the ``(6, n_bands)`` form
        for six per-wall values that also vary with frequency.
    :param fs: Sample rate, Hz.
    :param max_order: Reflection-order cut-off (total wall reflections). The
        shoebox has :math:`(2/3)(2 i_0^3 + 3 i_0^2 + 4 i_0)` audible images
        up to order ``i0`` (Kuttruff Equation (9.23)). Default 20.
    :param speed_of_sound: Speed of sound ``c``, m/s (default
        :data:`DEFAULT_SPEED_OF_SOUND`).
    :param air_attenuation: Air *intensity* attenuation constant ``m``, in
        neper per metre (scalar or per-band); the pressure amplitude of each
        path is scaled by :math:`e^{-m r / 2}` (Kuttruff 4.1). Default 0 (air
        absorption neglected). Obtain a physical ``m`` from
        :func:`~phonometry.environment.propagation.air_absorption.air_attenuation_m`.
    :param duration: RIR length, s; default the latest image arrival rounded up
        to the next sample.
    :param frequencies: Optional band centre frequencies, Hz, labelling a
        per-band result. When given, its length must match the band count of
        ``absorption`` (or broadcast against it).
    :return: An :class:`ImageSourceResult`.
    :raises ValueError: for a non-positive dimension/sample-rate, a
        source/receiver outside the room, an absorption outside ``[0, 1]`` or
        of an unsupported shape, a negative ``air_attenuation``, or a
        ``frequencies`` length that does not match the band count.
    """
    lx = require_positive(float(dimensions[0]), "Lx")
    ly = require_positive(float(dimensions[1]), "Ly")
    lz = require_positive(float(dimensions[2]), "Lz")
    dims = (lx, ly, lz)
    if fs <= 0:
        raise ValueError("Sample rate 'fs' must be positive.")
    speed_of_sound = require_positive(speed_of_sound, "speed_of_sound")
    if max_order < 0:
        raise ValueError("'max_order' must be non-negative.")
    src = _validate_point(source, dims, "source")
    rcv = _validate_point(receiver, dims, "receiver")
    if float(np.linalg.norm(src - rcv)) <= 0.0:
        raise ValueError(
            "source and receiver coincide; the direct path has zero length "
            "and infinite amplitude. Separate them so the direct distance is "
            "positive."
        )

    freq: np.ndarray | None = None
    n_freq: int | None = None
    if frequencies is not None:
        freq = np.asarray(frequencies, dtype=np.float64)
        n_freq = freq.size
    reflection, n_alpha_bands = _resolve_walls(absorption, n_freq)
    reflection, m_bands, n_bands, banded = _resolve_band_maps(
        reflection, n_alpha_bands, air_attenuation, freq
    )

    times, distances, orders, positions, counts = _image_lattice(
        src, rcv, dims, max_order, speed_of_sound
    )
    amp = _image_amplitudes(reflection, counts, m_bands, distances)
    ir = _sample_ir(times, amp, fs, duration, n_bands)

    if banded:
        ir_out: np.ndarray = ir
        amp_out: np.ndarray = amp
    else:
        ir_out = ir[0]
        amp_out = amp[0]

    return ImageSourceResult(
        ir=ir_out,
        fs=int(fs),
        frequencies=freq,
        times=times,
        distances=distances,
        orders=orders,
        amplitudes=amp_out,
        image_positions=positions,
        dimensions=dims,
        source=(float(src[0]), float(src[1]), float(src[2])),
        receiver=(float(rcv[0]), float(rcv[1]), float(rcv[2])),
        max_order=int(max_order),
        speed_of_sound=float(speed_of_sound),
    )


def audible_image_count(max_order: int) -> int:
    r"""Number of audible shoebox images up to reflection order ``max_order``.

    Kuttruff *Room Acoustics* 6th ed., Equation (9.23):
    :math:`(2/3)(2 i_0^3 + 3 i_0^2 + 4 i_0)`. Every image of a rectangular
    room is
    audible (no visibility test needed), so this is exactly the number of
    impulses :func:`image_source_rir` sums at that order.

    :param max_order: Reflection-order cut-off ``i0`` (non-negative).
    :return: The audible image count.
    """
    if max_order < 0:
        raise ValueError("'max_order' must be non-negative.")
    i0 = int(max_order)
    return (2 * (2 * i0**3 + 3 * i0**2 + 4 * i0)) // 3


def reflection_density(
    time: ArrayLike, volume: float, speed_of_sound: float = DEFAULT_SPEED_OF_SOUND
) -> np.ndarray | float:
    r"""Temporal density of reflections :math:`dN/dt = 4 \pi c^3 t^2 / V`.

    Kuttruff *Room Acoustics* 6th ed., Equation (4.6): the number of image
    sources per unit time whose spheres of radius :math:`c t` sweep the
    receiver.
    Independent of room shape. Useful to judge the reflection-order cut-off of
    :func:`image_source_rir` (the model is complete only while its images keep
    up with this density).

    :param time: Time after the direct sound, s (scalar or array).
    :param volume: Room volume ``V``, m3.
    :param speed_of_sound: Speed of sound ``c``, m/s.
    :return: ``dN/dt`` in reflections per second.
    """
    volume = require_positive(volume, "volume")
    speed_of_sound = require_positive(speed_of_sound, "speed_of_sound")
    t = np.asarray(time, dtype=np.float64)
    # Physical density is zero before the direct sound; clamp negative times
    # so the t**2 squaring does not report a spurious positive density.
    out = 4.0 * np.pi * speed_of_sound**3 * np.maximum(t, 0.0) ** 2 / volume
    return float(out) if out.ndim == 0 else out
