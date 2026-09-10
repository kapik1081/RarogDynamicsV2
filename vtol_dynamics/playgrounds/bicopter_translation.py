"""Constant symmetric thrust/tilt playground; CLI tilt is in degrees."""

import numpy as np

from ..testing import BicopterTranslationalModel
from ..types import ControlInputs, State
from .cli import common_parser, execute, finite_number, nonnegative_number
from .simulation import simulate


def run(parameters, simulation_time, propeller_speed, engine_tilt, *, dt=.02, instant_tilt=False):
    """Use rev/s and radians; instant_tilt initializes both tilts at their command.

    All other states start at zero. Actuator dynamics are unchanged.
    """
    initial_tilt = engine_tilt if instant_tilt else 0.0
    model = BicopterTranslationalModel(
        parameters, State(left_motor_tilt=initial_tilt, right_motor_tilt=initial_tilt))
    controls = ControlInputs(left_propeller_speed=propeller_speed, right_propeller_speed=propeller_speed,
                             left_motor_tilt=engine_tilt, right_motor_tilt=engine_tilt)
    return simulate(model, controls, simulation_time, dt, title="Bicopter translational flight")


def main(argv=None):
    parser = common_parser(__doc__)
    parser.add_argument("--propeller-speed", required=True, type=nonnegative_number,
                        help="Both propeller speeds [rev/s]")
    parser.add_argument("--engine-tilt", required=True, type=finite_number,
                        help="Both commanded engine tilts [deg]")
    parser.add_argument("--instant-tilt", action="store_true",
                        help="Start both engines at the requested tilt instead of zero")
    args = parser.parse_args(argv)
    return execute(parser, args, lambda params: run(params, args.simulation_time, args.propeller_speed,
                                                    np.deg2rad(args.engine_tilt), dt=args.dt,
                                                    instant_tilt=args.instant_tilt))


if __name__ == "__main__":
    main()
