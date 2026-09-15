"""Turning a recorded run into something you can put in a slide.

Two outputs, for two audiences:

    dashboard(run)  a single still image: the flight path, altitude, battery
                    and wind over time.  Good for a report, and it reads
                    correctly in print.
    animate(run)    a video of the flight.  Good whenever the question is
                    "what did it actually do?".

Everything is drawn from the recorded log, never from a live simulation, so the
same code renders a real flight the moment its telemetry is in the same format.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation


def _find_ffmpeg() -> bool:
    """Make MP4 writing work without a system-wide ffmpeg install.

    matplotlib looks for an ffmpeg binary on the PATH.  The pip package
    imageio-ffmpeg ships one, so if it is installed, point matplotlib at it.
    Without either, videos fall back to animated GIF.
    """
    if animation.FFMpegWriter.isAvailable():
        return True
    try:
        import imageio_ffmpeg

        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        return animation.FFMpegWriter.isAvailable()
    except Exception:  # noqa: BLE001 - any failure just means no MP4
        return False
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .recorder import load_run

# Colour-blind safe, and distinguishable in greyscale by lightness.
DRONE_COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def _as_run(run) -> dict:
    return load_run(run) if isinstance(run, (str, Path)) else run


def _by_drone(telemetry: Sequence[dict]) -> "OrderedDict[str, list[dict]]":
    grouped: "OrderedDict[str, list[dict]]" = OrderedDict()
    for row in telemetry:
        grouped.setdefault(row["drone"], []).append(row)
    return grouped


def _series(rows: Sequence[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows], dtype=float)


# ----------------------------------------------------------------------
# Scene furniture
# ----------------------------------------------------------------------


def _draw_box(ax, lower, upper, colour="#999999", alpha=0.18) -> None:
    x0, y0, z0 = lower
    x1, y1, z1 = upper
    corners = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ]
    )
    faces = [
        [corners[i] for i in face]
        for face in (
            (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
            (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4),
        )
    ]
    ax.add_collection3d(
        Poly3DCollection(faces, facecolor=colour, edgecolor="#666666",
                         linewidths=0.5, alpha=alpha)
    )


def _draw_world(ax, manifest: dict) -> tuple:
    world = manifest.get("world", {})
    lower = np.array(world.get("bounds_lower", [-3, -3, 0]), dtype=float)
    upper = np.array(world.get("bounds_upper", [3, 3, 2.5]), dtype=float)

    for obstacle in world.get("obstacles", []):
        if obstacle.get("type") == "box":
            _draw_box(ax, obstacle["lower"], obstacle["upper"])
        elif obstacle.get("type") == "cylinder":
            cx, cy = obstacle["center_xy"]
            r = obstacle["radius"]
            theta = np.linspace(0, 2 * np.pi, 24)
            for z in (obstacle["z_min"], obstacle["z_max"]):
                ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta),
                        np.full_like(theta, z), color="#888888", lw=0.8)

    for pad in world.get("mission_pads", []):
        cx, cy = pad["center"]
        half = pad.get("size", 0.18) / 2.0
        ax.plot(
            [cx - half, cx + half, cx + half, cx - half, cx - half],
            [cy - half, cy - half, cy + half, cy + half, cy - half],
            [0, 0, 0, 0, 0],
            color="#B8860B", lw=1.4,
        )
        ax.text(cx, cy, 0.02, f"m{pad['pad_id']}", color="#B8860B", fontsize=7)

    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.set_xlabel("x  forward (m)", fontsize=8)
    ax.set_ylabel("y  left (m)", fontsize=8)
    ax.set_zlabel("z  up (m)", fontsize=8, labelpad=-4)
    ax.tick_params(labelsize=7)
    try:
        ax.set_box_aspect((upper - lower))
    except (AttributeError, TypeError):
        pass
    return lower, upper


def _wind_caption(manifest: dict) -> str:
    wind = manifest.get("world", {}).get("wind", {})
    kind = wind.get("type", "none")
    if kind == "none":
        return "still air"
    if kind == "constant":
        v = np.array(wind["velocity"], dtype=float)
        return f"steady wind {np.linalg.norm(v):.1f} m/s"
    if kind == "turbulent":
        return f"turbulence, sigma {wind.get('sigma', 0):.1f} m/s"
    if kind == "gust_burst":
        v = np.linalg.norm(np.array(wind["velocity"], dtype=float))
        return f"gust {v:.1f} m/s at t={wind.get('start', 0):.0f}s"
    if kind == "boundary_layer":
        return f"boundary layer {wind.get('speed_at_reference', 0):.1f} m/s"
    if kind == "transient":
        inner = _wind_caption({"world": {"wind": wind["field"]}})
        return f"{inner}, only t={wind['start']:.0f}-{wind['start'] + wind['duration']:.0f}s"
    if kind == "wind_tunnel":
        v = np.array(wind["velocity"], dtype=float)
        return f"draught {np.linalg.norm(v):.1f} m/s over part of the room"
    if kind == "sum":
        return " + ".join(_wind_caption({"world": {"wind": f}}) for f in wind["fields"])
    return kind


# ----------------------------------------------------------------------
# Still dashboard
# ----------------------------------------------------------------------


def dashboard(run, out_path: str | Path | None = None, title: str | None = None) -> Path:
    """One image summarising a whole flight."""
    run = _as_run(run)
    grouped = _by_drone(run["telemetry"])
    manifest = run.get("manifest", {})

    if not grouped:
        raise ValueError("This run has no telemetry to plot.")

    figure = plt.figure(figsize=(13, 7.5))
    grid = figure.add_gridspec(3, 2, width_ratios=[1.25, 1], hspace=0.45, wspace=0.30)

    ax3d = figure.add_subplot(grid[:, 0], projection="3d")
    _draw_world(ax3d, manifest)

    ax_alt = figure.add_subplot(grid[0, 1])
    ax_bat = figure.add_subplot(grid[1, 1])
    ax_wind = figure.add_subplot(grid[2, 1])

    for index, (name, rows) in enumerate(grouped.items()):
        colour = DRONE_COLOURS[index % len(DRONE_COLOURS)]
        t = _series(rows, "t")
        x, y, z = _series(rows, "x"), _series(rows, "y"), _series(rows, "z")

        ax3d.plot(x, y, z, color=colour, lw=1.6, label=name)
        ax3d.scatter(x[:1], y[:1], z[:1], color=colour, marker="o", s=26)
        ax3d.scatter(x[-1:], y[-1:], z[-1:], color=colour, marker="X", s=48)

        ax_alt.plot(t, z, color=colour, lw=1.3, label=name)
        ax_bat.plot(t, _series(rows, "battery"), color=colour, lw=1.3)

    # Wind is a field, not a single number: each drone feels what the air is
    # doing where *it* is. Plot every drone's own wind so a field that only
    # covers part of the room is visible rather than averaged away.
    first_name, first = next(iter(grouped.items()))
    t = _series(first, "t")
    for index, (name, rows) in enumerate(grouped.items()):
        colour = DRONE_COLOURS[index % len(DRONE_COLOURS)] if len(grouped) > 1 else "#444444"
        speed = np.linalg.norm(
            np.column_stack([_series(rows, f"wind_{a}") for a in "xyz"]), axis=1
        )
        ax_wind.plot(_series(rows, "t"), speed, color=colour, lw=1.2)
        if len(grouped) == 1:
            ax_wind.fill_between(_series(rows, "t"), 0, speed, color=colour, alpha=0.12)

    # Mark the moments a command was issued, so the plots line up with intent.
    for command in run.get("commands", []):
        when = command.get("t_started")
        if when is None or command.get("verb") in ("command", "battery?", "rc"):
            continue
        for axis in (ax_alt, ax_bat, ax_wind):
            axis.axvline(when, color="#BBBBBB", lw=0.6, zorder=0)

    ax_alt.set_ylabel("altitude (m)", fontsize=8)
    ax_bat.set_ylabel("battery (%)", fontsize=8)
    ax_wind.set_ylabel(
        "wind at drone (m/s)" if len(grouped) > 1 else "wind (m/s)", fontsize=8
    )
    ax_wind.set_xlabel("time (s)", fontsize=8)
    for axis in (ax_alt, ax_bat, ax_wind):
        axis.grid(alpha=0.25, lw=0.5)
        axis.tick_params(labelsize=7)
        axis.spines[["top", "right"]].set_visible(False)

    if len(grouped) > 1:
        ax3d.legend(fontsize=7, loc="upper left")

    events = run.get("events", [])
    collisions = sum(1 for e in events if e.get("kind") == "collision")
    heading = title or f"{manifest.get('world', {}).get('name', 'run')}"
    subtitle = (
        f"{_wind_caption(manifest)}   |   {len(grouped)} drone(s)   |   "
        f"{t[-1]:.0f} s   |   {collisions} collision(s)"
    )
    figure.suptitle(heading, fontsize=13, y=0.97)
    figure.text(0.5, 0.925, subtitle, ha="center", fontsize=8.5, color="#555555")

    out_path = Path(out_path) if out_path else Path(run["directory"]) / "dashboard.png"
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out_path


# ----------------------------------------------------------------------
# Video
# ----------------------------------------------------------------------


def animate(
    run,
    out_path: str | Path | None = None,
    fps: int = 20,
    trail_seconds: float = 6.0,
    rotate: bool = True,
    speed: float | None = None,
    target_seconds: float = 45.0,
) -> Path:
    """Render the flight as a video.

    Writes MP4 when ffmpeg is available and falls back to an animated GIF when
    it is not, so this works on a fresh machine without extra installs.

    Long flights are played back faster so the video stays watchable: an
    endurance run of ten simulated minutes becomes about forty seconds. Pass
    `speed` to fix the multiplier yourself, or set it to 1.0 for real time.
    """
    run = _as_run(run)
    grouped = _by_drone(run["telemetry"])
    manifest = run.get("manifest", {})
    if not grouped:
        raise ValueError("This run has no telemetry to animate.")

    tracks = {
        name: {
            "t": _series(rows, "t"),
            "p": np.column_stack([_series(rows, a) for a in "xyz"]),
            "battery": _series(rows, "battery"),
            "action": [r["action"] for r in rows],
        }
        for name, rows in grouped.items()
    }

    reference = next(iter(tracks.values()))["t"]
    duration = float(reference[-1])

    if speed is None:
        # Keep the video around target_seconds long, but never slow it down.
        speed = max(1.0, duration / max(target_seconds, 1.0))
    speed = float(speed)

    frame_times = np.arange(0.0, duration, speed / fps)
    trail_frames = max(int(trail_seconds * fps), 2)

    figure = plt.figure(figsize=(7.5, 5.6))
    ax = figure.add_subplot(111, projection="3d")
    figure.subplots_adjust(left=0.0, right=0.94, top=0.94, bottom=0.06)
    _draw_world(ax, manifest)

    lines, heads = {}, {}
    for index, name in enumerate(tracks):
        colour = DRONE_COLOURS[index % len(DRONE_COLOURS)]
        lines[name] = ax.plot([], [], [], color=colour, lw=1.8, label=name)[0]
        heads[name] = ax.plot([], [], [], color=colour, marker="o", ms=7)[0]

    if len(tracks) > 1:
        ax.legend(fontsize=8, loc="upper left")

    caption = figure.text(0.02, 0.03, "", fontsize=9, family="monospace", color="#333333")
    speed_note = "" if speed < 1.05 else f"  |  {speed:.0f}x speed"
    figure.suptitle(
        f"{manifest.get('world', {}).get('name', 'flight')}  |  "
        f"{_wind_caption(manifest)}{speed_note}",
        fontsize=11,
    )

    def frame(i: int):
        now = frame_times[i]
        labels = []
        for name, track in tracks.items():
            end = int(np.searchsorted(track["t"], now))
            # The trail covers a fixed span of *flight* time, not of frames, so
            # it stays the same length however fast the playback is.
            samples_per_second = len(track["t"]) / max(duration, 1e-6)
            start = max(0, end - int(trail_seconds * samples_per_second))
            path = track["p"][start:end]
            if len(path):
                lines[name].set_data(path[:, 0], path[:, 1])
                lines[name].set_3d_properties(path[:, 2])
                heads[name].set_data(path[-1:, 0], path[-1:, 1])
                heads[name].set_3d_properties(path[-1:, 2])
                labels.append(
                    f"{name}  {track['action'][max(end - 1, 0)]:<9s}"
                    f" bat {track['battery'][max(end - 1, 0)]:5.1f}%"
                )
        caption.set_text(f"t = {now:6.2f} s\n" + "\n".join(labels))
        if rotate:
            ax.view_init(elev=24, azim=-60 + 20 * np.sin(now / max(duration, 1) * np.pi))
        return list(lines.values()) + list(heads.values())

    movie = animation.FuncAnimation(
        figure, frame, frames=len(frame_times), interval=1000 / fps, blit=False
    )

    have_ffmpeg = _find_ffmpeg()
    directory = Path(run["directory"])
    if out_path is None:
        out_path = directory / ("flight.mp4" if have_ffmpeg else "flight.gif")
    out_path = Path(out_path)

    if out_path.suffix == ".mp4" and have_ffmpeg:
        # Quality-targeted rather than bitrate-targeted. These frames are mostly
        # flat background, which a fixed bitrate wastes megabytes encoding;
        # constant-quality gets the same picture in a fraction of the size.
        # yuv420p keeps the file playable in QuickTime, Keynote and PowerPoint.
        writer = animation.FFMpegWriter(
            fps=fps,
            bitrate=-1,
            codec="libx264",
            extra_args=[
                # h.264 needs even pixel dimensions; pad rather than depend on
                # the figure size and dpi happening to multiply out that way.
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-crf", "26",
                "-preset", "medium",
                "-pix_fmt", "yuv420p",
            ],
        )
        dpi = 100
    else:
        if out_path.suffix == ".mp4":
            out_path = out_path.with_suffix(".gif")
            print(
                "[render] no ffmpeg, writing a GIF instead (much larger). For MP4: "
                "pip install imageio-ffmpeg"
            )
        writer = animation.PillowWriter(fps=fps)
        # GIFs have no interframe compression, so keep them small deliberately.
        dpi = 70

    # Remove any previous version first: ffmpeg will happily write a shorter
    # file over a longer one without truncating it, leaving tens of megabytes
    # of the old video stuck on the end.
    out_path.unlink(missing_ok=True)
    movie.save(str(out_path), writer=writer, dpi=dpi)
    plt.close(figure)
    return out_path


def report(run, fps: int = 20) -> dict:
    """Dashboard plus video in one call."""
    run = _as_run(run)
    return {"dashboard": dashboard(run), "video": animate(run, fps=fps)}
