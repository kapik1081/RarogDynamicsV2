"""Pure frame transforms and position/attitude kinematics."""

import numpy as np


def body_to_ned(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """3-2-1 body-to-earth rotation, Eqs. 1.64-1.66."""
    sp, st, ss = np.sin([roll, pitch, yaw])
    cp, ct, cs = np.cos([roll, pitch, yaw])
    return np.array([[ct*cs, sp*st*cs-cp*ss, cp*st*cs+sp*ss],
                     [ct*ss, sp*st*ss+cp*cs, cp*st*ss-sp*cs],
                     [-st, sp*ct, cp*ct]])


def wind_to_body(alpha: float, beta: float) -> np.ndarray:
    """PDF Eq. 1.24, including its alpha rotation sign convention."""
    sa, sb = np.sin([alpha, beta])
    ca, cb = np.cos([alpha, beta])
    return np.array([[ca*cb, -ca*sb, -sa], [sb, cb, 0], [sa*cb, -sa*sb, ca]])


def euler_rates(roll: float, pitch: float, body_rates) -> np.ndarray:
    """Eqs. 1.67-1.69. Raise near Euler gimbal lock instead of hiding it."""
    if abs(np.cos(pitch)) < 1e-6:
        raise ValueError("Euler attitude is singular near pitch = +/- pi/2; use a quaternion model")
    p, q, r = body_rates
    mixed = q*np.sin(roll) + r*np.cos(roll)
    return np.array([p + mixed*np.tan(pitch),
                     q*np.cos(roll) - r*np.sin(roll), mixed/np.cos(pitch)])


def gravity_acceleration(roll: float, pitch: float, gravity: float) -> np.ndarray:
    """Exact body-frame gravity, PDF Eq. 1.7 (earth NED gravity rotated to body)."""
    return gravity*np.array([-np.sin(pitch), np.sin(roll)*np.cos(pitch),
                             np.cos(roll)*np.cos(pitch)])
