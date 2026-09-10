"""Independent constant engine commands; CLI tilts are in degrees."""

import numpy as np

from ..testing import BicopterRotationModel
from ..types import ControlInputs, State
from .cli import common_parser, execute, finite_number, nonnegative_number
from .simulation import simulate


def run(parameters, simulation_time, right_engine_tilt, left_engine_tilt,
        right_propeller_speed, left_propeller_speed, *, dt=.02, instant_tilt=False):
    """Use radians and rev/s; instant_tilt initializes each tilt at its command.

    All other states start at zero. Actuator dynamics are unchanged.
    """
    initial_state = State(right_motor_tilt=right_engine_tilt if instant_tilt else 0.0,
                          left_motor_tilt=left_engine_tilt if instant_tilt else 0.0)
    model = BicopterRotationModel(parameters, initial_state)
    controls = ControlInputs(left_propeller_speed=left_propeller_speed,
                             right_propeller_speed=right_propeller_speed,
                             left_motor_tilt=left_engine_tilt, right_motor_tilt=right_engine_tilt)
    return simulate(model, controls, simulation_time, dt, title="Bicopter rotation")


def main(argv=None):
    parser = common_parser(__doc__)
    parser.add_argument("--instant-tilt", action="store_true",
                        help="Start each engine at its requested tilt instead of zero")
    for side in ("right", "left"):
        parser.add_argument(f"--{side}-engine-tilt", required=True, type=finite_number,
                            help=f"{side.capitalize()} commanded engine tilt [deg]")
        parser.add_argument(f"--{side}-propeller-speed", required=True, type=nonnegative_number,
                            help=f"{side.capitalize()} propeller speed [rev/s]")
    args = parser.parse_args(argv)
    return execute(parser, args, lambda params: run(
        params, args.simulation_time, np.deg2rad(args.right_engine_tilt), np.deg2rad(args.left_engine_tilt),
        args.right_propeller_speed, args.left_propeller_speed, dt=args.dt,
        instant_tilt=args.instant_tilt))


if __name__ == "__main__":
    main()
