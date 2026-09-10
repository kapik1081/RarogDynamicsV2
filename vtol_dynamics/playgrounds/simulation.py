"""Simulation sampling and physical statistics, independent of plotting."""

from dataclasses import dataclass

import numpy as np

from ..aerodynamics import air_data
from ..model import VTOLModel
from ..parameters import ModelParameters
from ..types import ControlInputs


@dataclass(frozen=True)
class FlightHistory:
    title: str
    parameters: ModelParameters
    controls: ControlInputs
    times: np.ndarray
    states: np.ndarray


@dataclass(frozen=True)
class FlightStatistics:
    alpha: np.ndarray
    beta: np.ndarray
    airspeed: np.ndarray
    translational_energy: np.ndarray
    rotational_energy: np.ndarray
    kinetic_energy: np.ndarray
    potential_energy: np.ndarray
    total_energy: np.ndarray


def simulate(model: VTOLModel, controls: ControlInputs, simulation_time: float,
             dt: float = 0.02, *, title: str = "VTOL flight") -> FlightHistory:
    """Record the initial state and advance with constant inputs to the exact end.

    dt is the output sampling interval; RK45 takes adaptive internal steps.
    A final partial interval is included when simulation_time is not a dt multiple.
    """
    if not np.isfinite(simulation_time) or simulation_time <= 0:
        raise ValueError("simulation_time must be finite and positive")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    times = np.arange(0.0, simulation_time, dt)
    # Avoid a near-duplicate endpoint caused by floating-point rounding.
    times = times[times < simulation_time - 8*np.finfo(float).eps*simulation_time]
    times = np.append(times, simulation_time)
    states = [model.get_state().as_vector()]
    for start, end in zip(times[:-1], times[1:]):
        try:
            states.append(model.step(controls, float(end-start)).as_vector())
        except (ValueError, RuntimeError) as error:
            raise RuntimeError(f"Simulation failed during {start:.6g} to {end:.6g} s: {error}") from error
    return FlightHistory(title, model.parameters, controls, times, np.asarray(states))


def flight_statistics(history: FlightHistory) -> FlightStatistics:
    """SI energy, including rotational KE and signed PE = -m*g*z_NED."""
    states, params = history.states, history.parameters
    air = [air_data(velocity, params.air_density) for velocity in states[:, 6:9]]
    airspeed = np.array([a.airspeed for a in air])
    translational = .5*params.mass*np.sum(states[:, 6:9]**2, axis=1)
    rotational = .5*np.einsum("ni,ij,nj->n", states[:, 9:12], params.inertia.matrix(), states[:, 9:12])
    kinetic = translational + rotational
    potential = -params.mass*params.gravity*states[:, 2]
    return FlightStatistics(np.array([a.alpha for a in air]), np.array([a.beta for a in air]),
                            airspeed, translational, rotational, kinetic, potential, kinetic+potential)
