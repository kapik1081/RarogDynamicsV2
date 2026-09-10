"""VTOL dynamics: immutable inputs, pure physics, and a stateful simulator."""

from .coefficients import CoefficientMap
from .model import VTOLModel
from .parameters import AerodynamicCoefficients, Inertia, ModelParameters, StallParameters
from .types import ControlInputs, State
from .testing import (AerodynamicsOnlyModel, BicopterFullModel, BicopterRotationModel,
                      BicopterTranslationalModel)

__all__ = ["VTOLModel", "State", "ControlInputs", "ModelParameters", "Inertia",
           "AerodynamicCoefficients", "StallParameters", "CoefficientMap",
           "BicopterTranslationalModel", "BicopterRotationModel", "BicopterFullModel",
           "AerodynamicsOnlyModel"]
