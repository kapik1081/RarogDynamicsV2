"""Input/state value objects. SI units, radians, and propeller revolutions/s."""

from dataclasses import dataclass, fields

import numpy as np


def _finite_fields(obj):
    for field in fields(obj):
        value = float(getattr(obj, field.name))
        if not np.isfinite(value):
            raise ValueError(f"{field.name} must be finite")
        object.__setattr__(obj, field.name, value)


@dataclass(frozen=True)
class State:
    """PDF Eq. 1.1 order, including RIGHT tilt before LEFT tilt.

    Position is earth NED (z <= 0 above terrain). Attitude uses 3-2-1 Euler
    angles. Linear and angular velocities are in body forward/right/down.
    Tilt 0 points thrust forward; pi/2 points thrust upward.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    u: float = 0.0
    v: float = 0.0
    w: float = 0.0
    p: float = 0.0
    q: float = 0.0
    r: float = 0.0
    right_motor_tilt: float = 0.0
    left_motor_tilt: float = 0.0

    def __post_init__(self):
        _finite_fields(self)

    def as_vector(self) -> np.ndarray:
        return np.array([getattr(self, f.name) for f in fields(self)], dtype=float)

    @classmethod
    def from_vector(cls, vector) -> "State":
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (14,):
            raise ValueError("state vector must have shape (14,)")
        return cls(*vector)


@dataclass(frozen=True)
class ControlInputs:
    """Motor speeds are rev/s, surfaces and commanded tilts are radians.

    Commands are held constant throughout a step. Tilt commands drive the
    nacelle actuator states, rather than directly replacing their angles.
    """

    left_propeller_speed: float = 0.0
    right_propeller_speed: float = 0.0
    elevator_deflection: float = 0.0
    aileron_deflection: float = 0.0
    rudder_deflection: float = 0.0
    left_motor_tilt: float = 0.0
    right_motor_tilt: float = 0.0

    def __post_init__(self):
        _finite_fields(self)
        if min(self.left_propeller_speed, self.right_propeller_speed) < 0:
            raise ValueError("propeller speeds must be nonnegative revolutions/s")
