"""Stateful RK45 wrapper around the pure model functions."""

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import state_derivative
from .kinematics import body_to_ned
from .parameters import ModelParameters
from .terrain import contact_derivative, normal_acceleration, project_to_terrain
from .types import ControlInputs, State


class VTOLModel:
    """Advance a 14-state VTOL model with piecewise-constant controls.

    Only this wrapper stores evolving state/time. A failed step does not
    commit partial results. Solver tolerances and maximum internal step
    are configurable independently of the requested output interval dt.
    """

    # Subclasses can select another pure ODE without duplicating integration,
    # state management, solver failure handling, or terrain transitions.
    _state_derivative = staticmethod(state_derivative)

    def __init__(self, parameters: ModelParameters, initial_state: State, *,
                 terrain_collision: bool = False, rtol: float = 1e-7,
                 atol: float = 1e-9, max_step: float = np.inf):
        if not isinstance(parameters, ModelParameters) or not isinstance(initial_state, State):
            raise TypeError("expected ModelParameters and State objects")
        if not np.isfinite(rtol) or rtol <= 0 or not np.isfinite(atol) or atol <= 0:
            raise ValueError("rtol and atol must be finite and positive")
        if np.isnan(max_step) or max_step <= 0:
            raise ValueError("max_step must be positive")
        self._parameters = parameters
        self._state = initial_state
        self._time = 0.0
        self._rtol, self._atol, self._max_step = rtol, atol, max_step
        self.terrain_collision = terrain_collision

    @property
    def parameters(self) -> ModelParameters:
        return self._parameters

    @property
    def time(self) -> float:
        return self._time

    def get_state(self) -> State:
        """Return the immutable current state (as_vector() returns a copy)."""
        return self._state

    @property
    def terrain_collision(self) -> bool:
        return self._terrain_collision

    @terrain_collision.setter
    def terrain_collision(self, enabled: bool):
        if not isinstance(enabled, bool):
            raise TypeError("terrain_collision must be a boolean")
        if enabled:
            self._state = State.from_vector(project_to_terrain(self._state.as_vector()))
        self._terrain_collision = enabled

    def step(self, controls: ControlInputs, dt: float) -> State:
        """Advance dt seconds with RK45, store and return the resulting state.

        dt=0 is a no-op. Terrain events split integration at impact/liftoff,
        so the remaining part of dt is still simulated after a collision.
        """
        if not isinstance(controls, ControlInputs):
            raise TypeError("controls must be a ControlInputs object")
        dt = float(dt)
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        if dt == 0:
            return self.get_state()
        end_time = self._time + dt
        if not np.isfinite(end_time) or end_time <= self._time:
            raise ValueError("dt cannot be represented at the current simulation time")

        vector = self._state.as_vector()
        local_time = 0.0  # Local time avoids loss of step precision on long runs.

        def free_rhs(t, y):
            return self._state_derivative(t, y, controls, self._parameters)

        # A tiny acceleration tolerance prevents repeated liftoff events at
        # exact zero force (e.g. an ideal hover initialized on the plane).
        release_tolerance = 1e-10
        grounded = False
        if self.terrain_collision and vector[2] >= 0:
            vertical_speed = body_to_ned(*vector[3:6])[2, :] @ vector[6:9]
            grounded = (vertical_speed <= 1e-10 and vertical_speed >= -1e-10
                        and normal_acceleration(vector, free_rhs(0, vector)) >= -release_tolerance)

        for _ in range(1000):
            if local_time >= dt:
                break
            if self.terrain_collision and grounded:
                def rhs(t, y):
                    return contact_derivative(y, free_rhs(t, y))

                def event(t, y):
                    return normal_acceleration(y, free_rhs(t, y)) + release_tolerance

                event.direction = -1
            else:
                rhs = free_rhs

                def event(t, y):
                    # A vehicle starting on the ground with upward velocity
                    # or acceleration must not register a spurious impact.
                    if y[2] == 0 and body_to_ned(*y[3:6])[2, :] @ y[6:9] <= 0:
                        return -1e-12
                    return y[2]

                event.direction = 1
            event.terminal = True
            solution = solve_ivp(rhs, (local_time, dt), vector, method="RK45",
                                 rtol=self._rtol, atol=self._atol, max_step=self._max_step,
                                 events=event if self.terrain_collision else None)
            if not solution.success:
                raise RuntimeError(f"RK45 integration failed: {solution.message}")
            vector = solution.y[:, -1]
            local_time = float(solution.t[-1])
            if solution.status != 1:
                break
            vector[2] = 0.0
            vector = project_to_terrain(vector)
            grounded = not grounded
        else:
            raise RuntimeError("Too many terrain transitions in one step; reduce dt")

        if self.terrain_collision:
            vector = project_to_terrain(vector)
        new_state = State.from_vector(vector)
        self._state, self._time = new_state, end_time
        return new_state
