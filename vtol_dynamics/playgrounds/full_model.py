"""Complete VTOL dynamics with constant engine commands and initial body velocity."""

import numpy as np

from ..model import VTOLModel
from ..types import ControlInputs, State
from .cli import common_parser, execute, finite_number, nonnegative_number
from .simulation import simulate


def run(parameters, simulation_time, right_engine_tilt, left_engine_tilt,
        right_propeller_speed, left_propeller_speed, u, v, w, *, dt=.02, instant_tilt=False):
    """Use radians, rev/s, and m/s; engines, aerodynamics, and gravity are active.

    Position, attitude, angular rates, and surface deflections start at zero.
    Engine tilts start at zero unless instant_tilt initializes each at its command.
    Controls remain constant and terrain collision is disabled.
    """
    initial_state = State(u=u, v=v, w=w,
                          right_motor_tilt=right_engine_tilt if instant_tilt else 0.0,
                          left_motor_tilt=left_engine_tilt if instant_tilt else 0.0)
    model = VTOLModel(parameters, initial_state)
    controls = ControlInputs(right_propeller_speed=right_propeller_speed,
                             left_propeller_speed=left_propeller_speed,
                             right_motor_tilt=right_engine_tilt, left_motor_tilt=left_engine_tilt)
    return simulate(model, controls, simulation_time, dt, title="Full VTOL model")


def main(argv=None):
    parser = common_parser(__doc__)
    parser.add_argument("--instant-tilt", action="store_true",
                        help="Start each engine at its requested tilt instead of zero")
    for side in ("right", "left"):
        parser.add_argument(f"--{side}-engine-tilt", required=True, type=finite_number,
                            help=f"{side.capitalize()} commanded engine tilt [deg]")
        parser.add_argument(f"--{side}-propeller-speed", required=True, type=nonnegative_number,
                            help=f"{side.capitalize()} propeller speed [rev/s]")
    for name, axis in (("u", "x"), ("v", "y"), ("w", "z")):
        parser.add_argument(f"--{name}", required=True, type=finite_number,
                            help=f"Initial body {axis} velocity [m/s]")
    args = parser.parse_args(argv)
    return execute(parser, args, lambda params: run(
        params, args.simulation_time, np.deg2rad(args.right_engine_tilt), np.deg2rad(args.left_engine_tilt),
        args.right_propeller_speed, args.left_propeller_speed, args.u, args.v, args.w,
        dt=args.dt, instant_tilt=args.instant_tilt))


if __name__ == "__main__":
    main()
