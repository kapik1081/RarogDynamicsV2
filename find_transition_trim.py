"""Plot viable transition trims and search for a continuous hover-to-cruise trim path."""

from pathlib import Path

import numpy as np

from vtol_dynamics.transition import TransitionConstraints, transition_trim_manifold
from vtol_dynamics.transition_plotting import plot_transition
from vtol_dynamics.trim_cli import (finite_number, load_parameters, nonnegative_number,
                                    parameter_parser, positive_number)


def main(argv=None):
    parser = parameter_parser(__doc__)
    parser.add_argument("--max-engine-rpm", required=True, type=nonnegative_number,
                        help="Maximum loaded RPM of each engine")
    parser.add_argument("--max-elevator-deflection", required=True, type=nonnegative_number,
                        help="Maximum absolute elevator deflection [deg]")
    parser.add_argument("--cruise-airspeed", required=True, nargs="+", type=positive_number,
                        metavar="V", help="Cruise speed, or MIN MAX interval [m/s], at zero engine tilt")
    parser.add_argument("--airspeed-max", type=positive_number,
                        help="Search upper speed [m/s]; default: upper cruise speed")
    for name, default, description in (
            ("tilt-min", 0, "Minimum common engine tilt"),
            ("tilt-max", 90, "Maximum common engine tilt"),
            ("pitch-min", -30, "Minimum pitch attitude, including hover"),
            ("pitch-max", 30, "Maximum pitch attitude, including hover"),
            ("alpha-min", None, "Minimum valid aerodynamic angle at positive speed"),
            ("alpha-max", None, "Maximum valid aerodynamic angle at positive speed")):
        parser.add_argument("--"+name, type=finite_number, default=default,
                            help=f"{description} [deg]; default: {default if default is not None else 'map domain'}")
    for surface in ("aileron", "rudder"):
        parser.add_argument(f"--max-{surface}-deflection", type=nonnegative_number,
                            help=f"Maximum absolute {surface} travel [deg]; default: unrestricted")
    parser.add_argument("--max-shaft-power", type=nonnegative_number,
                        help="Maximum combined shaft power of both engines [W]")
    parser.add_argument("--monotone-airspeed", action="store_true", help="Require nondecreasing path airspeed")
    parser.add_argument("--monotone-tilt", action="store_true", help="Require nonincreasing path tilt")
    for axis, default in (("speed", 31), ("tilt", 31), ("pitch", 241), ("elevator", 9)):
        parser.add_argument(f"--{axis}-points", type=int, default=default,
                            help=f"{axis.capitalize()} sampling count, at least 3 (default: {default})")
    parser.add_argument("--o", "-o", type=Path, metavar="FILENAME",
                        help="Save the plot to this exact filename (e.g. transition.png or transition.pdf)")
    parser.add_argument("--no-show", action="store_true", help="Suppress the plot window for headless use")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress reports")
    args = parser.parse_args(argv)
    if len(args.cruise_airspeed) not in (1, 2):
        parser.error("--cruise-airspeed takes one speed or MIN MAX")
    params = load_parameters(parser, args.parameters)
    try:
        # Fail before the numerical search if the optional plot dependency is missing.
        if args.no_show:
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        parser.error('Plotting requires matplotlib; install with python -m pip install -e ".[playgrounds]"')
    angular = ("tilt_min", "tilt_max", "pitch_min", "pitch_max", "alpha_min", "alpha_max",
               "max_elevator_deflection", "max_aileron_deflection", "max_rudder_deflection")
    limits = {name: None if getattr(args, name) is None else np.deg2rad(getattr(args, name))
              for name in angular}
    try:
        constraints = TransitionConstraints(**limits, max_engine_rpm=args.max_engine_rpm,
                                            max_shaft_power=args.max_shaft_power,
                                            monotone_airspeed=args.monotone_airspeed,
                                            monotone_tilt=args.monotone_tilt)
        cruise = args.cruise_airspeed[0] if len(args.cruise_airspeed) == 1 else tuple(args.cruise_airspeed)
        result = transition_trim_manifold(
            params, constraints, cruise_airspeed=cruise, airspeed_max=args.airspeed_max,
            **{f"{axis}_points": getattr(args, f"{axis}_points")
               for axis in ("speed", "tilt", "pitch", "elevator")},
            progress=None if args.quiet else lambda message: print(message, flush=True))
        fig = plot_transition(result)
        if args.o is not None:
            # Explicit format prevents matplotlib from appending .png to a filename
            # without an extension: --o always names the actual resulting file.
            fig.savefig(args.o, format=args.o.suffix.lstrip(".") or "png", dpi=180)
            print(f"Saved plot: {args.o}")
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    print(f"Viable trim samples: {len(result.points)}")
    print(f"Unresolved local continuations: {len(result.unresolved)}")
    if result.path:
        end = result.path[-1].solution
        print(f"Proposed steady-trim path: {len(result.path)} verified points; "
              f"hover to {end.airspeed:.6g} m/s at zero engine tilt")
        print(f"Minimum thrust/elevator/power reserve along path: {min(p.reserve for p in result.path):.3%}")
    else:
        print("No hover-to-cruise trim path found at this resolution under the requested limits.")
        print("This numerical result does not prove that no continuous corridor exists.")
    print("The plot shows steady equilibria, not an accelerating trajectory or a stability assessment.")
    if not args.no_show:
        plt.show()
    plt.close(fig)
    return 0 if result.path else 1


if __name__ == "__main__":
    raise SystemExit(main())
