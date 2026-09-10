"""Static, immutable parameters and strict JSON loading."""

from dataclasses import dataclass, field, fields
import json
from pathlib import Path

import numpy as np

from .coefficients import Coefficient, CoefficientMap


@dataclass(frozen=True)
class Inertia:
    """kg m^2; PDF Eq. 1.29 places MINUS Ixz in the off-diagonals."""

    Ixx: float
    Iyy: float
    Izz: float
    Ixz: float = 0.0

    def __post_init__(self):
        if not np.all(np.isfinite(self.matrix())):
            raise ValueError("inertia must be finite")
        if np.any(np.linalg.eigvalsh(self.matrix()) <= 0):
            raise ValueError("inertia matrix must be positive definite")

    def matrix(self) -> np.ndarray:
        return np.array([[self.Ixx, 0, -self.Ixz],
                         [0, self.Iyy, 0], [-self.Ixz, 0, self.Izz]], dtype=float)


@dataclass(frozen=True)
class StallParameters:
    """Eqs. 1.16-1.18. Disable if the tables already describe post-stall flow."""

    enabled: bool = True
    alpha_positive: float = float(np.deg2rad(15))
    alpha_negative: float = float(np.deg2rad(-15))
    transition_width: float = float(np.deg2rad(3))
    drag_max: float = 1.8

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("stall.enabled must be a boolean")
        values = [self.alpha_positive, self.alpha_negative, self.transition_width, self.drag_max]
        if not np.all(np.isfinite(values)):
            raise ValueError("stall parameters must be finite")
        if not self.alpha_negative < 0 < self.alpha_positive:
            raise ValueError("stall angles must straddle zero")
        if self.transition_width <= 0 or self.drag_max < 0:
            raise ValueError("transition_width must be positive and drag_max nonnegative")


@dataclass(frozen=True)
class AerodynamicCoefficients:
    """Each entry may be an alpha/beta map or a scalar constant.

    lift/drag/sideforce correspond to Cl/Cd/Cy in Eqs. 1.19-1.21.
    roll_*, pitch_*, yaw_* correspond to Cl*, Cm*, Cn* in 1.59-1.61.
    Derivatives multiply angles in radians and nondimensional body rates.
    """

    lift: Coefficient = 0.0
    drag: Coefficient = 0.0
    sideforce: Coefficient = 0.0
    roll_0: Coefficient = 0.0
    roll_beta: Coefficient = 0.0
    roll_p: Coefficient = 0.0
    roll_r: Coefficient = 0.0
    roll_aileron: Coefficient = 0.0
    roll_rudder: Coefficient = 0.0
    pitch_0: Coefficient = 0.0
    pitch_alpha: Coefficient = 0.0
    pitch_q: Coefficient = 0.0
    pitch_elevator: Coefficient = 0.0
    yaw_0: Coefficient = 0.0
    yaw_beta: Coefficient = 0.0
    yaw_p: Coefficient = 0.0
    yaw_r: Coefficient = 0.0
    yaw_aileron: Coefficient = 0.0
    yaw_rudder: Coefficient = 0.0

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if not isinstance(value, CoefficientMap):
                value = float(value)
                if not np.isfinite(value):
                    raise ValueError(f"{f.name} must be finite")
                object.__setattr__(self, f.name, value)

    @classmethod
    def from_dict(cls, data: dict) -> "AerodynamicCoefficients":
        return cls(**{key: CoefficientMap.from_dict(value) if isinstance(value, dict) else value
                      for key, value in data.items()})


@dataclass(frozen=True)
class ModelParameters:
    mass: float
    inertia: Inertia
    wing_area: float
    wingspan: float
    mean_chord: float
    propeller_diameter: float
    CT: float
    Kq: float
    engine_span: float
    engine_height: float
    tilt_time_constant: float
    tilt_max_rate: float
    air_density: float = 1.225
    gravity: float = 9.80665
    aerodynamics: AerodynamicCoefficients = field(default_factory=AerodynamicCoefficients)
    stall: StallParameters = field(default_factory=StallParameters)

    def __post_init__(self):
        positive = ("mass", "wing_area", "wingspan", "mean_chord", "propeller_diameter",
                    "tilt_time_constant", "tilt_max_rate")
        nonnegative = ("air_density", "gravity", "CT", "Kq", "engine_span")
        for name in positive + nonnegative + ("engine_height",):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if (name in positive and value <= 0) or (name in nonnegative and value < 0):
                raise ValueError(f"invalid {name}: {value}")
            object.__setattr__(self, name, value)
        for name, expected in (("inertia", Inertia), ("aerodynamics", AerodynamicCoefficients),
                               ("stall", StallParameters)):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}")

    @classmethod
    def from_dict(cls, data: dict) -> "ModelParameters":
        data = dict(data)
        data["inertia"] = Inertia(**data["inertia"])
        if "aerodynamics" in data:
            data["aerodynamics"] = AerodynamicCoefficients.from_dict(data["aerodynamics"])
        if "stall" in data:
            data["stall"] = StallParameters(**data["stall"])
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelParameters":
        """Load a UTF-8 JSON file. Unknown fields fail rather than being ignored."""
        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))
