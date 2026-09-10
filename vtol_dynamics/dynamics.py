"""Pure load assembly and 14-state derivative; no simulation state is stored."""

from dataclasses import dataclass

import numpy as np

from .aerodynamics import air_data, aerodynamic_force, aerodynamic_moment
from .kinematics import body_to_ned, euler_rates, gravity_acceleration
from .parameters import Inertia, ModelParameters
from .propulsion import reaction_moment, thrust_forces, thrust_moment, tilt_rate
from .types import ControlInputs, State


@dataclass(frozen=True)
class Loads:
    """Separate body-frame components in N and N m for inspection/testing."""

    aerodynamic_force: np.ndarray
    thrust_force: np.ndarray
    gravity_force: np.ndarray
    aerodynamic_moment: np.ndarray
    thrust_moment: np.ndarray
    reaction_moment: np.ndarray

    @property
    def total_force(self) -> np.ndarray:
        return self.aerodynamic_force + self.thrust_force + self.gravity_force

    @property
    def total_moment(self) -> np.ndarray:
        return self.aerodynamic_moment + self.thrust_moment + self.reaction_moment


def calculate_loads(state: State, controls: ControlInputs, params: ModelParameters) -> Loads:
    air = air_data([state.u, state.v, state.w], params.air_density)
    forces = thrust_forces(state, controls, params)
    return Loads(
        aerodynamic_force(air, params), forces.sum(axis=0),
        params.mass*gravity_acceleration(state.roll, state.pitch, params.gravity),
        aerodynamic_moment(air, [state.p, state.q, state.r], controls, params),
        thrust_moment(forces, params), reaction_moment(state, controls, params))


def translational_acceleration(body_velocity, body_rates, force, mass: float) -> np.ndarray:
    """Eq. 1.5, including the rotating-body transport term."""
    return np.asarray(force)/mass - np.cross(body_rates, body_velocity)


def angular_acceleration(body_rates, moment, inertia: Inertia) -> np.ndarray:
    """Solve Eq. 1.28 directly; equivalent to the expanded Eqs. 1.30-1.39."""
    tensor = inertia.matrix()
    return np.linalg.solve(tensor, np.asarray(moment)-np.cross(body_rates, tensor @ body_rates))


def state_derivative(time: float, vector, controls: ControlInputs,
                     params: ModelParameters) -> np.ndarray:
    """Full Eq. 1.70; time is accepted for SciPy's ODE interface."""
    state = State.from_vector(vector)
    velocity, rates = np.asarray(vector[6:9]), np.asarray(vector[9:12])
    loads = calculate_loads(state, controls, params)
    return np.concatenate((
        body_to_ned(state.roll, state.pitch, state.yaw) @ velocity,
        euler_rates(state.roll, state.pitch, rates),
        translational_acceleration(velocity, rates, loads.total_force, params.mass),
        angular_acceleration(rates, loads.total_moment, params.inertia),
        [tilt_rate(state.right_motor_tilt, controls.right_motor_tilt,
                   params.tilt_time_constant, params.tilt_max_rate),
         tilt_rate(state.left_motor_tilt, controls.left_motor_tilt,
                   params.tilt_time_constant, params.tilt_max_rate)]))
