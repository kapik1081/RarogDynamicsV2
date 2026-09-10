"""Stateless reduced equations for isolating sections of the VTOL model.

All functions use the full 14-state layout. Disabled state derivatives are
zero, and disabled force sources are never evaluated (including their maps).
"""

import numpy as np

from .aerodynamics import air_data, aerodynamic_force, aerodynamic_moment
from .dynamics import angular_acceleration, translational_acceleration
from .kinematics import body_to_ned, euler_rates, gravity_acceleration
from .parameters import ModelParameters
from .propulsion import reaction_moment, thrust_forces, thrust_moment, tilt_rate
from .types import ControlInputs, State


def _derivative(vector, controls, params, *, translation, rotation, engines):
    """Compose shared physics for either engine-only or aerodynamic-only loads."""
    state = State.from_vector(vector)
    velocity = np.array([state.u, state.v, state.w])
    rates = np.array([state.p, state.q, state.r])
    derivative = np.zeros(14)
    force, moment = np.zeros(3), np.zeros(3)

    if engines:
        forces = thrust_forces(state, controls, params)
        if translation:
            force = forces.sum(axis=0)
        if rotation:
            moment = thrust_moment(forces, params) + reaction_moment(state, controls, params)
        derivative[12:] = [
            tilt_rate(state.right_motor_tilt, controls.right_motor_tilt,
                      params.tilt_time_constant, params.tilt_max_rate),
            tilt_rate(state.left_motor_tilt, controls.left_motor_tilt,
                      params.tilt_time_constant, params.tilt_max_rate),
        ]
    else:
        air = air_data(velocity, params.air_density)
        force = aerodynamic_force(air, params)
        moment = aerodynamic_moment(air, rates, controls, params)

    if translation:
        force += params.mass * gravity_acceleration(
            state.roll, state.pitch, params.gravity, params.gravity_model)
        derivative[:3] = body_to_ned(state.roll, state.pitch, state.yaw) @ velocity
        # The translation-only frame is fixed, so its angular rates must be zero.
        derivative[6:9] = translational_acceleration(velocity, rates, force, params.mass)
    if rotation:
        derivative[3:6] = euler_rates(state.roll, state.pitch, rates)
        derivative[9:12] = angular_acceleration(rates, moment, params.inertia)
    return derivative


def bicopter_translational_derivative(time: float, vector, controls: ControlInputs,
                                      params: ModelParameters) -> np.ndarray:
    """Thrust + gravity, fixed attitude, and live nacelle actuators."""
    state = State.from_vector(vector)
    if any(rate != 0 for rate in (state.p, state.q, state.r)):
        raise ValueError("The translation-only model requires p=q=r=0 (fixed attitude)")
    return _derivative(vector, controls, params, translation=True, rotation=False, engines=True)


def bicopter_rotation_derivative(time: float, vector, controls: ControlInputs,
                                 params: ModelParameters) -> np.ndarray:
    """Engine moments and live nacelle actuators; position/linear velocity frozen."""
    return _derivative(vector, controls, params, translation=False, rotation=True, engines=True)


def bicopter_full_derivative(time: float, vector, controls: ControlInputs,
                             params: ModelParameters) -> np.ndarray:
    """Coupled rigid-body motion from thrust, engine moments, and gravity."""
    return _derivative(vector, controls, params, translation=True, rotation=True, engines=True)


def aerodynamics_only_derivative(time: float, vector, controls: ControlInputs,
                                 params: ModelParameters) -> np.ndarray:
    """Glider dynamics: aerodynamic loads + gravity; engine states frozen."""
    return _derivative(vector, controls, params, translation=True, rotation=True, engines=False)
