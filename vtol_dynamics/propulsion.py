"""Pure propeller loads and rate-limited nacelle dynamics."""

import numpy as np

from .parameters import ModelParameters
from .types import ControlInputs, State


def tilt_rate(actual: float, commanded: float, time_constant: float, max_rate: float) -> float:
    """First-order, rate-limited actuator, Eqs. 1.2-1.4."""
    return float(np.clip((commanded-actual)/time_constant, -max_rate, max_rate))


def propeller_thrust(speed: float, params: ModelParameters) -> float:
    """Eq. 1.9; speed is rev/s (neither RPM nor rad/s)."""
    return params.CT * params.air_density * speed**2 * params.propeller_diameter**4


def propeller_torque(speed: float, params: ModelParameters) -> float:
    """Eq. 1.48."""
    return params.Kq * params.air_density * speed**2 * params.propeller_diameter**5


def thrust_forces(state: State, controls: ControlInputs, params: ModelParameters) -> np.ndarray:
    """Individual body thrusts, rows LEFT then RIGHT; Eqs. 1.10-1.12."""
    tilts = np.array([state.left_motor_tilt, state.right_motor_tilt])
    thrusts = np.array([propeller_thrust(controls.left_propeller_speed, params),
                       propeller_thrust(controls.right_propeller_speed, params)])
    return thrusts[:, None] * np.column_stack((np.cos(tilts), np.zeros(2), -np.sin(tilts)))


def thrust_moment(forces: np.ndarray, params: ModelParameters) -> np.ndarray:
    """Eqs. 1.41-1.47, r cross F; engine_height is signed body z."""
    offsets = np.array([[0, -params.engine_span, params.engine_height],
                        [0, params.engine_span, params.engine_height]])
    return np.cross(offsets, forces).sum(axis=0)


def reaction_moment(state: State, controls: ControlInputs, params: ModelParameters) -> np.ndarray:
    """Counter-rotating torque, preserving PDF signs in Eqs. 1.49-1.51.

    In particular, the right propeller contributes POSITIVE sin(tilt) to z.
    This convention is kept explicit for validation against the document.
    """
    left = propeller_torque(controls.left_propeller_speed, params)
    right = propeller_torque(controls.right_propeller_speed, params)
    return np.array([right*np.cos(state.right_motor_tilt)-left*np.cos(state.left_motor_tilt),
                     0.0,
                     right*np.sin(state.right_motor_tilt)-left*np.sin(state.left_motor_tilt)])
