"""Stateful models for isolated physics tests, sharing VTOLModel's RK45 API.

These models intentionally remove parts of the aircraft dynamics; use
VTOLModel for the complete plant. Parameters and units are unchanged.
"""

from .model import VTOLModel
from .parameters import ModelParameters
from .testing_dynamics import (aerodynamics_only_derivative, bicopter_full_derivative,
                               bicopter_rotation_derivative, bicopter_translational_derivative)
from .types import State


class BicopterTranslationalModel(VTOLModel):
    """Thrust and gravity translate the craft at its fixed initial attitude.

    Nacelle tilt lag/rate limits are active. Initial p, q, r must be zero.
    Aerodynamic loads, engine moments, and surface deflections are omitted.
    Terrain collision is available.
    """

    _state_derivative = staticmethod(bicopter_translational_derivative)

    def __init__(self, parameters: ModelParameters, initial_state: State, **kwargs):
        if not isinstance(initial_state, State):
            raise TypeError("initial_state must be a State object")
        if any(rate != 0 for rate in (initial_state.p, initial_state.q, initial_state.r)):
            raise ValueError("The translation-only model requires p=q=r=0 (fixed attitude)")
        super().__init__(parameters, initial_state, **kwargs)


class BicopterRotationModel(VTOLModel):
    """Engine thrust/reaction moments rotate the craft; nacelle tilts evolve.

    Position and body linear velocity retain their initial values. Aerodynamic
    loads and surface deflections are omitted. Terrain must remain disabled,
    since there is no translational motion or ground-contact calculation.
    """

    _state_derivative = staticmethod(bicopter_rotation_derivative)

    @VTOLModel.terrain_collision.setter
    def terrain_collision(self, enabled: bool):
        if not isinstance(enabled, bool):
            raise TypeError("terrain_collision must be a boolean")
        if enabled:
            raise ValueError("Terrain collision is unavailable in the rotation-only model")
        self._terrain_collision = False


class BicopterFullModel(VTOLModel):
    """Coupled translation/rotation with engines and gravity, without aerodynamics.

    Thrust, thrust-offset moments, reaction moments, and nacelle dynamics are
    active. Surface deflections have no effect. Terrain collision is available.
    """

    _state_derivative = staticmethod(bicopter_full_derivative)


class AerodynamicsOnlyModel(VTOLModel):
    """Glider with aerodynamic forces/moments, control surfaces, and gravity.

    All engine commands are ignored. Both nacelle states retain their initial
    values. Translation, rotation, and optional terrain collision are active.
    """

    _state_derivative = staticmethod(aerodynamics_only_derivative)
