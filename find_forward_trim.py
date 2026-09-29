"""Find symmetric forward straight-level trim by a full-range grid scan and Brent solves."""

import numpy as np

from vtol_dynamics.trim import forward_flight_trim
from vtol_dynamics.trim_cli import (finite_number, load_parameters, nonnegative_number,
                                    parameter_parser, positive_number, print_solution)


def main(argv=None):
    parser = parameter_parser(__doc__)
    parser.add_argument("--engine-tilt", required=True, type=finite_number,
                        help="Prescribed common engine tilt [deg]")
    parser.add_argument("--airspeed", required=True, type=positive_number,
                        help="Prescribed airspeed [m/s]")
    parser.add_argument("--max-elevator-deflection", type=nonnegative_number,
                        help="Maximum absolute elevator deflection [deg]; default: unrestricted")
    parser.add_argument("--max-engine-rpm", type=nonnegative_number,
                        help="Maximum RPM of each engine; default: unrestricted")
    parser.add_argument("--alpha-min", type=finite_number,
                        help="Reject roots below this aerodynamic validity angle [deg]")
    parser.add_argument("--alpha-max", type=finite_number,
                        help="Reject roots above this aerodynamic validity angle [deg]")
    parser.add_argument("--grid-points", type=int, default=4001,
                        help="Uniform alpha-grid points before adding table knots (default: 4001)")
    args = parser.parse_args(argv)
    params = load_parameters(parser, args.parameters)

    def radians(value):
        return None if value is None else float(np.deg2rad(value))

    try:
        result = forward_flight_trim(
            params, radians(args.engine_tilt), args.airspeed,
            max_elevator_deflection=radians(args.max_elevator_deflection),
            max_engine_rpm=args.max_engine_rpm,
            alpha_min=radians(args.alpha_min), alpha_max=radians(args.alpha_max),
            grid_points=args.grid_points)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    low, high = np.rad2deg(result.alpha_range)
    print(f"Scanned full aerodynamic alpha range: [{low:.9g}, {high:.9g}] deg")
    print(f"Grid: {args.grid_points} uniform points plus map knots; solver: Brent (Eq. 2.14)")
    print(f"Candidate roots: {len(result.candidates)}")

    def number(value):
        return "n/a" if value is None else f"{value:.9g}"

    for index, candidate in enumerate(result.candidates, 1):
        elevator = None if candidate.elevator_deflection is None else np.rad2deg(candidate.elevator_deflection)
        print(f"  {index}. alpha={np.rad2deg(candidate.alpha):.9g} deg; "
              f"P={number(candidate.total_thrust)} N; engine={number(candidate.engine_rpm)} RPM; "
              f"elevator={number(elevator)} deg")
        if candidate.solution is None:
            print("     REJECTED: " + "; ".join(candidate.rejection_reasons))
        else:
            branch = "pre-stall" if candidate.solution.pre_stall else "post-stall"
            print(f"     FEASIBLE ({branch})")
    if result.selected is None:
        print("No feasible trim found in the scanned range under the requested limits.")
        return 1
    branch = "pre-stall" if result.selected.pre_stall else "post-stall fallback"
    print(f"Selected trim: {branch}; smallest |alpha| in the preferred branch")
    print_solution(result.selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
