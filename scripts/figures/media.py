#  Copyright (c) 2026. Jose Manuel Requena Plens
"""How an animation becomes files: the timeline, the encoders and the posters.

The clip builders in :mod:`figures.schematics` and :mod:`figures.fields` only
draw frames; this module decides how many there are and what they are written
to. It carries the two animation norms as constants (the 12 s schematic
timeline with its closing hold, the wider budget the wave-field clips size
themselves against), the VP9 and remote AV1/NVENC encoder settings, and
:func:`_save_animation`, which writes each clip in its four language x theme
variants as WebM, extracts the deferred-loading poster still and adds the
English GIF for the GitHub docs.
"""

import os
import sys
from collections.abc import Callable
from typing import Any, cast

import matplotlib.pyplot as plt

from .i18n import (
    _LANG,
    _LANG_SUFFIX,
    _translate_figure,
    audit_figure,
    lookup,
)
from .theme import _FILENAME_SUFFIX

# ====================================================================# Animations (Tier 1 pilot)
# ---------------------------------------------------------------------------
# Deterministic FuncAnimation clips of the level-vs-time phenomena the library
# already computes. Each is rendered to WebM (site <video>) in all four
# language x theme variants, and to an animated GIF for the English GitHub
# docs (both themes). Every WebM also gets a JPEG poster frame (the held
# verdict state near the end of the clip) so the site can embed the video
# with ``preload="none"`` and still show a meaningful still while nothing is
# fetched. Posters are ``.jpg`` on purpose: `make graphs` clears only
# ``*.svg``/``*.png``/``*.webp`` and scripts/check_figures.py
# regenerates/compares only those, so the posters ride with the WebM/GIF
# outside the figure pipeline. Kept out of generate_all()/`make graphs` so
# ordinary figure regeneration stays fast; produced by `make animations` (or `make posters` to re-extract the
# stills from the committed WebMs without re-rendering).
# ====================================================================
_ANIM_FPS = 20
_ANIM_SECONDS = 12
_ANIM_FRAMES = _ANIM_FPS * _ANIM_SECONDS
# Closing hold appended to each schematic timeline: the settled verdict frame
# stays on screen ~2 s so it can be read before the loop restarts.
_ANIM_HOLD = 2 * _ANIM_FPS
# The wave-field clips run on their own frame budgets: on the shared 12 s
# timeline a wavefront advances several cells per frame and the field
# strobes. Each FDTD/elastic clip therefore sizes its own window from the
# two animation norms -- the captured span is 1.2 times the full round-trip
# flight (source to the deepest reflector, wells and cavities included, and
# on to the farthest visible point, at the slowest speed on the path) and
# the capture stride keeps at least 48 frames per period of the highest
# carrier -- and states that arithmetic next to its frame count. This 18 s
# value is the starting budget the older clips were sized against and the
# default for clips whose own note does not override it.
_FDTD_ANIM_FRAMES = 18 * _ANIM_FPS
_ANIM_FIGSIZE = (8.0, 4.5)  # inches at _ANIM_DPI -> 2400 x 1350 px
_ANIM_DPI = 300
# The GitHub-docs GIF is a compact fallback for the smooth site WebM: a lower
# frame rate, smaller frame and capped palette keep even the detail-dense
# clips (the p·u oscillations) near half a megabyte.
_GIF_FPS = 12
_GIF_SCALE = 640
_GIF_COLORS = 64
# Rounded box of the verdict/annotation pills drawn over the clips.
_ANIM_PILL_BOX = "round,pad=0.25"

# VP9 encoding for the site WebM. Beyond the constant-quality base
# (``-crf 40 -b:v 0``) the screen-content tuning, row multithreading and the
# ``good`` deadline cut the encode time roughly in half at the same file size
# (the flat schematic / field content compresses identically). ``-cpu-used``
# trades encode speed for compression effort; on this content the size and
# quality are essentially flat from 2 through 4, so the run uses the faster
# setting. ``-threads`` is capped so eight parallel Python workers do not
# oversubscribe the box. Animations are not byte-compared (unlike the
# SVG/WebP figures), so the mild run-to-run variance these knobs introduce is
# harmless.
_ANIM_CPU_USED = 4
_ANIM_ENCODE_THREADS = 3


def _vp9_extra_args() -> list[str]:
    """ffmpeg args for the tuned constant-quality VP9 WebM encode."""
    return [
        "-b:v",
        "0",
        "-crf",
        "40",
        "-tune-content",
        "screen",
        "-row-mt",
        "1",
        "-deadline",
        "good",
        "-cpu-used",
        str(_ANIM_CPU_USED),
        "-threads",
        str(_ANIM_ENCODE_THREADS),
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-loglevel",
        "error",
    ]


def _translate_str(s: str) -> str:
    """Translate one label to the active language (exact + pattern + comma).

    Mirrors :func:`_translate_figure` for a single string, so animation labels
    -- which are rewritten every frame and never pass through ``save_figure``
    -- can be localised at creation time instead.
    """
    import re as _re

    if _LANG == "en" or not s:
        return s
    out = lookup(s)
    if "$" not in out and _re.search(r"\d\.\d", out):
        out = _re.sub(r"(?<![\d.A-Za-z])(\d+)\.(\d+)(?![.\d])", r"\1,\2", out)
    return out


def _anim_path(output_dir: str, stem: str, ext: str) -> str:
    """Animation output path with the active language + theme suffixes."""
    # The code that draws the clips is hashed into
    # scripts/animation_fingerprints.txt, so rewriting this asks for a
    # re-render that would draw the very same frames.
    return os.path.join(output_dir, f"{stem}{_LANG_SUFFIX}{_FILENAME_SUFFIX}.{ext}")  # noqa: PTH118


def _anim_figure() -> Any:
    """A fixed-size themed figure for a clip (constant canvas across frames)."""
    return plt.figure(figsize=_ANIM_FIGSIZE, dpi=_ANIM_DPI, layout="constrained")


def _render_clip(
    fig: Any,
    update: Callable[[int], tuple[Any, ...]],
    output_dir: str,
    stem: str,
    *,
    frames: int | None = None,
    fps: float | None = None,
    gif_fps: int | None = None,
    poster_ss: float | None = None,
    measure: Callable[[], None] | None = None,
) -> None:
    """Shared clip tail: drive *update* through FuncAnimation and encode.

    Static artists (tick labels included) get the same Spanish pass as the
    figures; per-frame labels are translated by ``_translate_str`` inside
    each clip's update function. ``frames``/``fps`` override the shared
    timeline for clips that need their own budget (the FDTD room modes);
    ``gif_fps`` is forwarded to the GitHub-GIF palette pass.

    ``measure`` runs after the figure is in its final language and before
    the first frame is built. A clip that fits a label against the geometry
    of its axes (``fields._core``) has to read the canvas the frames are
    drawn on, and the Spanish pass can still resize that canvas here: it
    swaps the tick formatters for versions that write a decimal comma and a
    typographic minus, and under constrained layout a tick label that
    changes width moves the axes it belongs to. Measuring from this hook
    reads the settled geometry. Whatever the callback measures must already
    be in its final text, sign included -- these helpers measure the box of
    a rendered string, so a number whose ASCII hyphen is swapped for U+2212
    after the fit was measured against a glyph that is not the one drawn.
    A clip that blanks a pre-translated label until its reveal frame blanks
    it inside this callback too, after the fit.
    """
    from matplotlib.animation import FuncAnimation

    audit_figure(stem)
    _translate_figure(fig)
    if measure is not None:
        measure()
    n_frames = _ANIM_FRAMES if frames is None else frames
    rate = _ANIM_FPS if fps is None else fps
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / rate, blit=False)
    _save_animation(
        anim, fig, output_dir, stem, fps=rate, gif_fps=gif_fps, poster_ss=poster_ss
    )


def _extract_poster(webm: str, poster_ss: float | None = None) -> str:
    """Extract the deferred-loading poster still from a rendered WebM.

    The frame is grabbed half a second before the end of the clip, i.e.
    inside the closing hold, so the poster shows the settled verdict state.
    The output sits next to the WebM as ``<stem>_poster.jpg``; JPEG keeps it
    outside the SVG/WebP figure pipeline (`make graphs` deletion glob and the
    scripts/check_figures.py regeneration compare).
    """
    import subprocess

    poster = webm.removesuffix(".webm") + "_poster.jpg"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error"]
        + (["-ss", f"{poster_ss:.2f}"] if poster_ss is not None else ["-sseof", "-0.5"])
        + ["-i", webm, "-frames:v", "1", "-q:v", "3", "-update", "1", poster],
        check=True,
    )
    return poster


def _gpu_encode_target() -> dict[str, str] | None:
    """The remote AV1 (NVENC) encode target from .env, if enabled and alive.

    Opt-in through ``PHONO_GPU_ENCODE=av1`` next to the ``PHONO_GPU_*``
    host settings: VP9 has no NVIDIA encoder, so the hardware path writes
    AV1 into the same WebM container (the site embeds are codec-agnostic;
    the GitHub GIF is still derived locally). Any doubt (no .env, host
    down, flag off) returns ``None`` and the clips keep the CPU VP9 path.
    """
    import subprocess

    def decline(reason: str) -> None:
        # Say it out loud: the CPU path writes VP9, which is around 40 % larger
        # than the AV1 the committed clips use, and a silent downgrade only
        # shows up as an unexplained size jump in a later diff.
        print(f"note: encoding on the CPU as VP9, {reason}", file=sys.stderr)

    try:
        import fdtd_gpu_remote

        fdtd_gpu_remote.load_env()
    except ImportError:
        decline("fdtd_gpu_remote is not importable")
        return None
    if os.environ.get("PHONO_GPU_ENCODE", "").lower() != "av1":
        decline(
            "PHONO_GPU_ENCODE is not 'av1' (a linked worktree has no .env of "
            "its own, so export the PHONO_GPU_* settings to render there)"
        )
        return None
    host = os.environ.get("PHONO_GPU_HOST", "")
    user = os.environ.get("PHONO_GPU_USER", "root")
    image = os.environ.get("PHONO_GPU_FFMPEG_IMAGE", "linuxserver/ffmpeg:latest")
    if not host:
        decline("PHONO_GPU_HOST is not set")
        return None
    probe = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=4",
            "--",
            f"{user}@{host}",
            "true",
        ],
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        decline(f"{user}@{host} did not answer")
        return None
    workdir = os.environ.get("PHONO_GPU_WORKDIR", "/tmp/phonometry-gpu")
    return {"target": f"{user}@{host}", "image": image, "workdir": workdir}


def _av1_nvenc_extra_args() -> list[str]:
    """ffmpeg args for the remote AV1 NVENC encode (WebM container)."""
    return [
        "-c:v",
        "av1_nvenc",
        # cq 44 calibrated against the VP9 crf-40 size on the metadiffuser
        # clip (cq 42 = parity, 46 = -22 %): smaller files, same resolution.
        "-b:v",
        "0",
        "-cq",
        "44",
        "-preset",
        "p6",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-loglevel",
        "error",
    ]


def _write_gpu_ffmpeg_wrapper(target: str, image: str, workdir: str) -> str:
    """A throwaway ffmpeg shim that runs the encode on the GPU host.

    Matplotlib's writer spawns ``ffmpeg <args> <output>`` and streams raw
    frames on stdin; the shim forwards stdin over ssh into the NVENC
    container, which writes the WebM to a file in the remote work
    directory (the matroska muxer needs a seekable target), and the shim
    then fetches it back with scp and cleans up.
    """
    import shlex
    import tempfile

    q_target = shlex.quote(target)
    q_image = shlex.quote(image)
    q_dir = shlex.quote(workdir)
    script = f"""#!/bin/bash
set -euo pipefail
out="${{@: -1}}"
args=("${{@:1:$#-1}}")
rid="enc-$$-$RANDOM.webm"
quoted=$(printf ' %q' "${{args[@]}}")
ssh -o BatchMode=yes -- {q_target} \
    "mkdir -p {q_dir} && docker run --rm -i --gpus all \
     -v {q_dir}:/work {q_image}$quoted /work/$rid"
scp -q -- {q_target}:{q_dir}/"$rid" "$out"
ssh -o BatchMode=yes -- {q_target} "rm -f {q_dir}/$rid"
"""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".sh", prefix="ffmpeg-gpu-", delete=False
    ) as handle:
        handle.write(script)
    os.chmod(handle.name, 0o755)  # noqa: PTH101
    return handle.name


def _save_animation(
    anim: Any,
    fig: Any,
    output_dir: str,
    stem: str,
    make_gif: bool = True,
    *,
    fps: float | None = None,
    gif_fps: int | None = None,
    poster_ss: float | None = None,
) -> None:
    """Write *anim* to WebM (always), its poster JPEG and, for English, a GIF.

    The GIF is derived from the just-written WebM with an ffmpeg palette pass
    so the GitHub docs get a compact, self-contained loop; the site embeds the
    WebM directly. Every WebM variant also gets a ``_poster.jpg`` still (see
    :func:`_extract_poster`) so the site ``<video>`` embeds can defer loading
    (``preload="none"``) behind a meaningful frame. ``fps`` overrides the
    shared WebM rate and ``gif_fps`` the GIF sampling rate (long clips drop
    GIF frames to stay within GitHub-friendly file sizes). ``savefig.bbox``
    is forced to ``standard`` so every frame keeps the same canvas size (a
    ``tight`` box would jitter and break the encoder).
    """
    import subprocess

    from matplotlib.animation import FFMpegWriter

    webm = _anim_path(output_dir, stem, "webm")
    rate = _ANIM_FPS if fps is None else fps
    # Matplotlib types the writer rate as an int, but the writer only
    # formats it into ffmpeg's ``-r`` and a fractional rate is legal there.
    # A clip whose capture stride was cut by a non-integer factor plays at
    # the matching fractional rate so its pacing stays exact.
    rate_arg = cast(int, rate)
    gpu = _gpu_encode_target()
    saved = False
    if gpu is not None:
        wrapper = _write_gpu_ffmpeg_wrapper(gpu["target"], gpu["image"], gpu["workdir"])
        try:
            writer = FFMpegWriter(
                fps=rate_arg, codec="av1_nvenc", extra_args=_av1_nvenc_extra_args()
            )
            with plt.rc_context(
                {"savefig.bbox": "standard", "animation.ffmpeg_path": wrapper}
            ):
                anim.save(
                    webm,
                    writer=writer,
                    dpi=_ANIM_DPI,
                    savefig_kwargs={"facecolor": fig.get_facecolor()},
                )
            saved = True
        except (RuntimeError, subprocess.CalledProcessError, OSError) as exc:
            print(f"  [gpu-encode] {stem}: {exc}; falling back to VP9")
        finally:
            os.remove(wrapper)  # noqa: PTH107
    if not saved:
        writer = FFMpegWriter(
            fps=rate_arg, codec="libvpx-vp9", extra_args=_vp9_extra_args()
        )
        with plt.rc_context({"savefig.bbox": "standard"}):
            anim.save(
                webm,
                writer=writer,
                dpi=_ANIM_DPI,
                savefig_kwargs={"facecolor": fig.get_facecolor()},
            )
    _extract_poster(webm, poster_ss)
    made_gif = False
    if make_gif and _LANG == "en":
        gif = _anim_path(output_dir, stem, "gif")
        palette = os.path.join(output_dir, f".{stem}{_FILENAME_SUFFIX}_pal.png")  # noqa: PTH118
        vf = (
            f"fps={_GIF_FPS if gif_fps is None else gif_fps},"
            f"scale={_GIF_SCALE}:-1:flags=lanczos"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                webm,
                "-vf",
                f"{vf},palettegen=max_colors={_GIF_COLORS}:stats_mode=diff",
                "-update",
                "1",
                palette,
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                webm,
                "-i",
                palette,
                "-lavfi",
                f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                "-loop",
                "0",
                gif,
            ],
            check=True,
        )
        os.remove(palette)  # noqa: PTH107
        made_gif = True
    plt.close(fig)
    theme = "dark" if _FILENAME_SUFFIX else "light"
    print(
        f"  {stem} [{_LANG} {theme}] -> webm + poster" + (" + gif" if made_gif else "")
    )
