"""Pure air-data, stall, force, and moment calculations."""

from dataclasses import dataclass

import numpy as np

from .coefficients import evaluate
from .kinematics import wind_to_body
from .parameters import ModelParameters, StallParameters
from .types import ControlInputs


@dataclass(frozen=True)
class AirData:
    airspeed: float
    alpha: float
    beta: float
    dynamic_pressure: float


def air_data(body_velocity, air_density: float) -> AirData:
    """Eqs. 1.13-1.15, 1.55; still air as assumed in the PDF."""
    u, v, w = body_velocity
    speed = float(np.linalg.norm(body_velocity))
    if speed == 0.0:
        return AirData(0.0, 0.0, 0.0, 0.0)
    return AirData(speed, float(np.arctan2(w, u)), float(np.arctan2(v, np.hypot(u, w))),
                   0.5*air_density*speed**2)


def stall_blend(alpha: float, params: StallParameters) -> float:
    """Eq. 1.16. The negative-stall term has denominator -delta_alpha."""
    if not params.enabled:
        return 0.0
    positive = np.tanh((alpha-params.alpha_positive)/params.transition_width)
    negative = np.tanh((alpha-params.alpha_negative)/(-params.transition_width))
    return float(0.5*(2.0+positive+negative))


def force_coefficients(air: AirData, params: ModelParameters) -> np.ndarray:
    """Return [lift, drag, sideforce] coefficients, including stall blending."""
    aero = params.aerodynamics
    lift = evaluate(aero.lift, air.alpha, air.beta)
    drag = evaluate(aero.drag, air.alpha, air.beta)
    sideforce = evaluate(aero.sideforce, air.alpha, air.beta)
    sigma = stall_blend(air.alpha, params.stall)
    lift = (1-sigma)*lift + sigma*0.5*params.stall.drag_max*np.sin(2*air.alpha)
    drag = (1-sigma)*drag + sigma*params.stall.drag_max*np.sin(air.alpha)**2
    return np.array([lift, drag, sideforce])


def aerodynamic_force(air: AirData, params: ModelParameters) -> np.ndarray:
    """Body force, Eqs. 1.17-1.27; sideforce table gives Cy itself, not Cy_beta."""
    if air.airspeed == 0.0:
        return np.zeros(3)
    lift, drag, sideforce = air.dynamic_pressure*params.wing_area*force_coefficients(air, params)
    return wind_to_body(air.alpha, air.beta) @ np.array([-drag, sideforce, -lift])


def aerodynamic_moment(air: AirData, body_rates, controls: ControlInputs,
                        params: ModelParameters) -> np.ndarray:
    """Eqs. 1.59-1.61: dimensional rates, with the airspeed divisor canceled.

    This formulation tends smoothly to zero at hover without an artificial
    minimum airspeed. Every stability derivative can itself be a map.
    """
    if air.airspeed == 0.0:
        return np.zeros(3)
    p, q, r = body_rates
    a, b = air.alpha, air.beta
    coefficients = params.aerodynamics

    def c(name):
        return evaluate(getattr(coefficients, name), a, b)

    da, de, dr = controls.aileron_deflection, controls.elevator_deflection, controls.rudder_deflection
    static = np.array([c("roll_0")+c("roll_beta")*b+c("roll_aileron")*da+c("roll_rudder")*dr,
                       c("pitch_0")+c("pitch_alpha")*a+c("pitch_elevator")*de,
                       c("yaw_0")+c("yaw_beta")*b+c("yaw_aileron")*da+c("yaw_rudder")*dr])
    damping = np.array([c("roll_p")*p+c("roll_r")*r, c("pitch_q")*q,
                        c("yaw_p")*p+c("yaw_r")*r])
    lengths = np.array([params.wingspan, params.mean_chord, params.wingspan])
    return (air.dynamic_pressure*params.wing_area*lengths*static
            + 0.25*params.air_density*air.airspeed*params.wing_area*lengths**2*damping)
