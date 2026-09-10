"""Stateless point contact with the flat NED z=0 plane.

Contact has no restitution, friction, or attitude constraints. The ground
normal acts at the center of mass and therefore creates no moment.
"""

import numpy as np

from .kinematics import body_to_ned


def project_to_terrain(vector) -> np.ndarray:
    """If on/below the plane, clamp z and remove only inward NED velocity."""
    result = np.array(vector, dtype=float, copy=True)
    if result[2] >= 0.0:
        result[2] = 0.0
        rotation = body_to_ned(*result[3:6])
        velocity_ned = rotation @ result[6:9]
        if velocity_ned[2] > 0.0:
            result[6:9] -= rotation[2, :]*velocity_ned[2]
    return result


def normal_acceleration(vector, derivative) -> float:
    """Unconstrained earth-down acceleration, including the frame transport."""
    rotation = body_to_ned(*vector[3:6])
    return float(rotation[2, :] @ (derivative[6:9] + np.cross(vector[9:12], vector[6:9])))


def contact_derivative(vector, free_derivative) -> np.ndarray:
    """Constrain vertical position and acceleration during sustained contact.

    The integrator uses this only until the required normal force becomes
    negative, at which point the vehicle lifts off into free flight.
    """
    result = np.array(free_derivative, copy=True)
    rotation = body_to_ned(*vector[3:6])
    result[2] = 0.0
    result[6:9] -= rotation[2, :]*normal_acceleration(vector, free_derivative)
    return result
