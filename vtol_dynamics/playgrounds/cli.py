"""Shared argument handling, report output, and optional portable exports."""

import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np

from ..parameters import ModelParameters
from ..types import State
from .simulation import flight_statistics


def finite_number(value):
    number = float(value)
    if not np.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def positive_number(value):
    number = finite_number(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative_number(value):
    number = finite_number(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--time", "--simulation-time", dest="simulation_time", required=True,
                        type=positive_number, help="Simulation duration [s]")
    parser.add_argument("--parameters", type=Path,
                        default=Path(__file__).resolve().parents[2]/"examples/sample_parameters.json",
                        help="Static parameter JSON (default: repository sample)")
    parser.add_argument("--dt", type=positive_number, default=.02, help="Output sample interval [s] (default: .02)")
    parser.add_argument("--fps", type=positive_number, default=30, help="Maximum animation frames/s (default: 30)")
    parser.add_argument("--playback-speed", type=positive_number, default=1,
                        help="Animation speed multiplier (default: 1)")
    parser.add_argument("--axis-length", type=positive_number,
                        help="Displayed body/engine line length [m] (default: half wingspan)")
    parser.add_argument("--output", type=Path, help="Save flight.html, statistics.png and history.csv in this directory")
    parser.add_argument("--no-show", action="store_true", help="Run without GUI windows (can be combined with --output)")
    return parser


def report(history):
    stats = flight_statistics(history)
    state = history.states[-1]
    print(f"{history.title}: {history.times[-1]:g} s, {len(history.times)} samples")
    print(f"Final NED position [m]: {state[0]:.6g}, {state[1]:.6g}, {state[2]:.6g}")
    print("Final roll, pitch, yaw [deg]: " + ", ".join(f"{a:.6g}" for a in np.rad2deg(state[3:6])))
    print(f"Final alpha, beta [deg]: {np.rad2deg(stats.alpha[-1]):.6g}, {np.rad2deg(stats.beta[-1]):.6g}")
    print("Final body u, v, w [m/s]: " + ", ".join(f"{v:.6g}" for v in state[6:9]))
    print(f"Airspeed final / maximum [m/s]: {stats.airspeed[-1]:.6g} / {stats.airspeed.max():.6g}")
    print(f"Final kinetic / potential / total energy [J]: {stats.kinetic_energy[-1]:.6g} / "
          f"{stats.potential_energy[-1]:.6g} / {stats.total_energy[-1]:.6g}")


def export_history(history, path):
    stats = flight_statistics(history)
    names = [f.name for f in fields(stats)]
    data = np.column_stack([history.times, history.states, *(getattr(stats, name) for name in names)])
    columns = ["time_s", *(f.name for f in fields(State)), *names]
    np.savetxt(path, data, delimiter=",", header=",".join(columns), comments="")


def execute(parser, args, run):
    """Load parameters, run a scenario callback, and present the result."""
    try:
        params = ModelParameters.from_json(args.parameters)
        history = run(params)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    report(history)
    if args.no_show and args.output is None:
        return history

    # Select a noninteractive backend before importing pyplot in headless runs.
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plotting import create_display

    display = create_display(history, fps=args.fps, playback_speed=args.playback_speed,
                             axis_length=args.axis_length)
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        display.statistics_figure.savefig(args.output/"statistics.png", dpi=150)
        export_history(history, args.output/"history.csv")
        # Explicit export should include every chosen frame, rather than being
        # silently truncated by Matplotlib's notebook embed-size default.
        with matplotlib.rc_context({"animation.embed_limit": float("inf")}):
            html = display.animation.to_jshtml(default_mode="once")
        (args.output/"flight.html").write_text(html, encoding="utf-8")
        print(f"Saved plots, animation and data to {args.output.resolve()}")
    if args.no_show:
        plt.close(display.flight_figure)
        plt.close(display.statistics_figure)
    else:
        plt.show()
    return history
