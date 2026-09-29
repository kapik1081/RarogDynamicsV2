"""Complete VTOL dynamics with configurable initial state and constant controls."""

import numpy as np

from ..model import VTOLModel
from ..types import ControlInputs, State
from .cli import common_parser, execute, finite_number, nonnegative_number
from .simulation import simulate


def run(parameters, simulation_time=1.0, *, dt=.02,
        initial_x=0.0, initial_y=0.0, initial_z=0.0,
        initial_roll=0.0, initial_pitch=0.0, initial_yaw=0.0,
        initial_u=0.0, initial_v=0.0, initial_w=0.0,
        initial_p=0.0, initial_q=0.0, initial_r=0.0,
        initial_right_motor_tilt=0.0, initial_left_motor_tilt=0.0,
        control_right_propeller_speed=0.0, control_left_propeller_speed=0.0,
        control_right_motor_tilt=0.0, control_left_motor_tilt=0.0,
        control_elevator_deflection=0.0, control_aileron_deflection=0.0,
        control_rudder_deflection=0.0):
    """Run with SI units: metres, m/s, radians, rad/s, and propeller rev/s.

    Every initial state and control component defaults to zero. Initial motor
    tilts are independent of the constant tilt commands; actuator dynamics
    remain active. Engines, aerodynamics, and gravity are active, and terrain
    collision is disabled. Duration defaults to one second.
    """
    initial_state = State(
        x=initial_x, y=initial_y, z=initial_z,
        roll=initial_roll, pitch=initial_pitch, yaw=initial_yaw,
        u=initial_u, v=initial_v, w=initial_w,
        p=initial_p, q=initial_q, r=initial_r,
        right_motor_tilt=initial_right_motor_tilt,
        left_motor_tilt=initial_left_motor_tilt)
    model = VTOLModel(parameters, initial_state)
    controls = ControlInputs(
        right_propeller_speed=control_right_propeller_speed,
        left_propeller_speed=control_left_propeller_speed,
        right_motor_tilt=control_right_motor_tilt,
        left_motor_tilt=control_left_motor_tilt,
        elevator_deflection=control_elevator_deflection,
        aileron_deflection=control_aileron_deflection,
        rudder_deflection=control_rudder_deflection)
    return simulate(model, controls, simulation_time, dt, title="Full VTOL model")


def main(argv=None):
    parser = common_parser(__doc__, default_simulation_time=1.0)
    initial = parser.add_argument_group("Initial state (all default to zero)")
    controls = parser.add_argument_group("Constant controls (all default to zero)")
    angular_arguments = []
    model_arguments = []

    def add_argument(group, prefix, name, description, unit, *, nonnegative=False):
        argument = f"{prefix}_{name}"
        group.add_argument(f"--{argument.replace('_', '-')}", default=0.0,
                           type=nonnegative_number if nonnegative else finite_number,
                           help=f"{description} [{unit}] (default: 0)")
        model_arguments.append(argument)
        if unit in ("deg", "deg/s"):
            angular_arguments.append(argument)

    for name, direction in (("x", "north"), ("y", "east"), ("z", "down")):
        add_argument(initial, "initial", name, f"Initial NED {direction} position", "m")
    for angle in ("roll", "pitch", "yaw"):
        add_argument(initial, "initial", angle, f"Initial {angle} angle", "deg")
    for name, axis in (("u", "x"), ("v", "y"), ("w", "z")):
        add_argument(initial, "initial", name, f"Initial body {axis} velocity", "m/s")
    for name, axis in (("p", "x"), ("q", "y"), ("r", "z")):
        add_argument(initial, "initial", name, f"Initial body {axis} angular rate", "deg/s")
    for side in ("right", "left"):
        add_argument(initial, "initial", f"{side}_motor_tilt",
                     f"Initial {side} motor tilt", "deg")
        add_argument(controls, "control", f"{side}_motor_tilt",
                     f"Constant {side} motor tilt command", "deg")
        add_argument(controls, "control", f"{side}_propeller_speed",
                     f"Constant {side} propeller speed", "rev/s", nonnegative=True)
    for surface in ("elevator", "aileron", "rudder"):
        add_argument(controls, "control", f"{surface}_deflection",
                     f"Constant {surface} deflection", "deg")
    args = parser.parse_args(argv)
    values = {name: np.deg2rad(getattr(args, name)) if name in angular_arguments
              else getattr(args, name) for name in model_arguments}
    return execute(parser, args, lambda params: run(
        params, args.simulation_time, dt=args.dt, **values))


if __name__ == "__main__":
    main()
