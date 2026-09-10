"""Shared 3-D animation and time-history plots for the three playgrounds."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from ..kinematics import body_to_ned
from ..parameters import ModelParameters
from .simulation import FlightHistory, flight_statistics


def craft_segments(vector, parameters: ModelParameters, axis_length: float) -> np.ndarray:
    """Five segments (start/end points): +body x/y/z, left/right thrust axes.

    Return shape (5, 2, 3) in display coordinates north/east/altitude. Engine
    lines originate at their actual body offsets and use actual nacelle states.
    """
    display = np.diag([1.0, 1.0, -1.0])
    rotation = display @ body_to_ned(*vector[3:6])
    origin = display @ vector[:3]
    segments = [(origin, origin + axis_length*rotation[:, i]) for i in range(3)]
    for span, tilt in ((-parameters.engine_span, vector[13]), (parameters.engine_span, vector[12])):
        base = origin + rotation @ np.array([0.0, span, parameters.engine_height])
        direction = rotation @ np.array([np.cos(tilt), 0.0, -np.sin(tilt)])
        segments.append((base, base+axis_length*direction))
    return np.asarray(segments)


def animation_frames(times, fps: float, playback_speed: float) -> np.ndarray:
    """Select recorded samples, always including the beginning and final state."""
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(playback_speed) or playback_speed <= 0:
        raise ValueError("fps and playback_speed must be finite and positive")
    count = min(len(times), max(2, int(np.ceil(times[-1]*fps/playback_speed))+1))
    return np.unique(np.searchsorted(times, np.linspace(0, times[-1], count)))


@dataclass
class FlightDisplay:
    """Keep the animation alive for the lifetime of the figure windows."""

    flight_figure: object
    statistics_figure: object
    animation: FuncAnimation


def create_display(history: FlightHistory, *, fps: float = 30, playback_speed: float = 1,
                   axis_length: float | None = None) -> FlightDisplay:
    params, states, times = history.parameters, history.states, history.times
    length = params.wingspan/2 if axis_length is None else axis_length
    if not np.isfinite(length) or length <= 0:
        raise ValueError("axis_length must be finite and positive")
    frames = animation_frames(times, fps, playback_speed)
    stats = flight_statistics(history)

    stats_figure, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, layout="constrained")
    stats_figure.suptitle(history.title + " — statistics")
    colors = ("tab:red", "tab:green", "tab:blue")
    for i, (label, color) in enumerate(zip(("x (north)", "y (east)", "z (down)"), colors)):
        axes[0, 0].plot(times, states[:, i], label=label, color=color)
    axes[0, 0].set(title="Global position (NED)", ylabel="Position [m]")
    for i, (label, color) in enumerate(zip(("roll", "pitch", "yaw"), colors)):
        axes[0, 1].plot(times, np.rad2deg(states[:, 3+i]), label=label, color=color)
    axes[0, 1].set(title="Craft attitude", ylabel="Angle [deg]")
    axes[1, 0].plot(times, np.rad2deg(stats.alpha), label="alpha")
    axes[1, 0].plot(times, np.rad2deg(stats.beta), label="beta")
    axes[1, 0].set(title="Airflow angles", ylabel="Angle [deg]")
    for label, data in (("Kinetic (translation + rotation)", stats.kinetic_energy),
                         ("Potential", stats.potential_energy), ("Total", stats.total_energy)):
        axes[1, 1].plot(times, data, label=label)
    axes[1, 1].set(title="Mechanical energy", ylabel="Energy [J]")
    for i, (label, color) in enumerate(zip(("u", "v", "w"), colors)):
        axes[2, 0].plot(times, states[:, 6+i], label=label, color=color)
    axes[2, 0].plot(times, stats.airspeed, label="Airspeed", color="black", linestyle="--")
    axes[2, 0].set(title="Body velocities and airspeed", ylabel="Velocity [m/s]")
    for label, column, color in (("Left", 13, "tab:orange"), ("Right", 12, "tab:purple")):
        axes[2, 1].plot(times, np.rad2deg(states[:, column]), label=label, color=color)
    axes[2, 1].set(title="Actual engine tilts", ylabel="Tilt [deg]")
    for ax in axes.flat:
        ax.grid(True, alpha=.3)
        ax.legend(fontsize="small")
        ax.set_xlim(times[0], times[-1])
        ax.set_xlabel("Time [s]")

    flight_figure = plt.figure(figsize=(10, 8), layout="constrained")
    ax = flight_figure.add_subplot(projection="3d")
    ax.set(xlabel="x / north [m]", ylabel="y / east [m]", zlabel="Altitude / -z [m]",
           title=history.title)
    path = states[:, :3]*[1, 1, -1]
    # Bound all craft orientations and engine offsets, including a stationary
    # rotation experiment. Equal scales preserve the geometry of body axes.
    margin = length + np.hypot(params.engine_span, params.engine_height)
    center = (path.max(axis=0)+path.min(axis=0))/2
    radius = max(float(np.ptp(path, axis=0).max())/2 + margin, 1e-3)
    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=-55)
    trail, = ax.plot([], [], [], color="0.4", linewidth=1, label="Flight path")
    labels = ("+body x", "+body y", "+body z", "Left engine", "Right engine")
    lines = [ax.plot([], [], [], color=color, linewidth=2.5, marker="o",
                     markevery=[1], markersize=4, label=label)[0]
             for label, color in zip(labels, (*colors, "tab:orange", "tab:purple"))]
    clock = ax.text2D(.02, .97, "", transform=ax.transAxes)
    ax.text2D(.02, .02, "Dots mark positive axis / thrust direction. Height shown as -z.",
              transform=ax.transAxes, fontsize=8)
    ax.legend(loc="upper right", fontsize="small")

    def update(index):
        trail.set_data_3d(*path[:index+1].T)
        for line, segment in zip(lines, craft_segments(states[index], params, length)):
            line.set_data_3d(*segment.T)
        clock.set_text(f"t = {times[index]:.3f} / {times[-1]:.3f} s")
        return trail, *lines, clock

    interval = 1000*times[-1]/playback_speed/(len(frames)-1)
    animation = FuncAnimation(flight_figure, update, frames=frames,
                              init_func=lambda: update(0), interval=interval,
                              repeat=False, blit=False, cache_frame_data=False)
    # Also retain the reference on the figure for interactive Python callers.
    flight_figure._flight_animation = animation
    return FlightDisplay(flight_figure, stats_figure, animation)
