"""Stateless hover and straight-level trim calculations (PDF chapter 2).

Forward trim scans the full evaluable alpha domain before bracketing roots.
It uses Eq. 2.14, avoiding the tangent poles of Eq. 2.13. Feasibility checks
and full-model residual verification happen after roots have been found.
"""

from dataclasses import dataclass, fields
from typing import Callable

import numpy as np
from scipy.optimize import brentq

from .aerodynamics import AirData, force_coefficients
from .coefficients import CoefficientMap, evaluate
from .dynamics import state_derivative
from .parameters import ModelParameters
from .types import ControlInputs, State


@dataclass(frozen=True)
class TrimSolution:
    state: State
    controls: ControlInputs
    total_thrust: float
    airspeed: float
    alpha: float
    pre_stall: bool
    acceleration_residual: float

    @property
    def engine_rpm(self) -> float:
        return self.controls.right_propeller_speed * 60.0


@dataclass(frozen=True)
class TrimCandidate:
    alpha: float
    total_thrust: float | None
    engine_rpm: float | None
    elevator_deflection: float | None
    rejection_reasons: tuple[str, ...]
    solution: TrimSolution | None


@dataclass(frozen=True)
class ForwardTrimResult:
    alpha_range: tuple[float, float]
    candidates: tuple[TrimCandidate, ...]
    selected: TrimSolution | None


def _thrust_factor(params: ModelParameters) -> float:
    factor = params.CT * params.air_density * params.propeller_diameter**4
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Trim requires a positive finite CT * air_density * propeller_diameter**4")
    return factor


def _solution(params, state, controls, thrust, airspeed, alpha) -> TrimSolution:
    """Verify the entire plant, including lateral force and all three moments."""
    derivative = state_derivative(0.0, state.as_vector(), controls, params)
    # Forward-flight position changes at Va north; the other 11 states are steady.
    expected = np.zeros(14)
    expected[0] = airspeed
    residual = float(np.max(np.abs(derivative[3:])))
    if not np.all(np.isfinite(derivative)) or not np.allclose(
            derivative, expected, rtol=0, atol=1e-6):
        raise ValueError("full-model equilibrium residual exceeds 1e-6 (including lateral dynamics)")
    return TrimSolution(state, controls, float(thrust), float(airspeed), float(alpha),
                        params.stall.alpha_negative < alpha < params.stall.alpha_positive, residual)


def hover_trim(params: ModelParameters) -> TrimSolution:
    """Eq. 2.1: each engine supports mg/2; actual and commanded tilts are pi/2."""
    thrust = params.mass * params.gravity
    speed = float(np.sqrt(thrust / (2*_thrust_factor(params))))
    state = State(right_motor_tilt=np.pi/2, left_motor_tilt=np.pi/2)
    controls = ControlInputs(right_propeller_speed=speed, left_propeller_speed=speed,
                             right_motor_tilt=np.pi/2, left_motor_tilt=np.pi/2)
    return _solution(params, state, controls, thrust, 0.0, 0.0)


def aerodynamic_alpha_range(params: ModelParameters) -> tuple[float, float]:
    """Full atan2 alpha domain [-pi, pi], intersected with strict map bounds."""
    low, high = -np.pi, np.pi
    for field in fields(params.aerodynamics):
        table = getattr(params.aerodynamics, field.name)
        if not isinstance(table, CoefficientMap) or table.bounds != "raise":
            continue
        scale = np.pi/180 if table.angle_unit == "deg" else 1.0
        if not table.betas[0] <= 0 <= table.betas[-1]:
            raise ValueError(f"{field.name} map excludes beta=0")
        low, high = max(low, table.alphas[0]*scale), min(high, table.alphas[-1]*scale)
    if low >= high:
        raise ValueError("Aerodynamic maps have no common nonzero-width alpha interval at beta=0")
    return float(low), float(high)


def longitudinal_forces(alpha: float, airspeed: float, params: ModelParameters) -> tuple[float, float]:
    """Lift and drag at beta=0, using exactly the simulation's blended model."""
    air = AirData(airspeed, alpha, 0.0, .5*params.air_density*airspeed**2)
    lift, drag, _ = air.dynamic_pressure * params.wing_area * force_coefficients(air, params)
    return float(lift), float(drag)


def forward_residual(alpha: float, airspeed: float, engine_tilt: float,
                     params: ModelParameters) -> float:
    """Eq. 2.14: (L-mg)*cos(tilt+alpha) + D*sin(tilt+alpha), in N."""
    lift, drag = longitudinal_forces(alpha, airspeed, params)
    return float((lift-params.mass*params.gravity)*np.cos(engine_tilt+alpha)
                 + drag*np.sin(engine_tilt+alpha))


def bracketed_roots(function: Callable[[float], float], grid) -> tuple[float, ...]:
    """Collect exact grid zeros and solve every sign-changing interval with Brent."""
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
        raise ValueError("Root grid must contain at least two finite, strictly increasing points")
    values = np.array([function(float(alpha)) for alpha in grid])
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite trim residual on the alpha grid")
    if np.all(values == 0):
        raise ValueError("Residual is zero across the grid; isolated trim roots cannot be identified")
    roots = list(grid[values == 0])
    for a, b, fa, fb in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if fa != 0 and fb != 0 and np.signbit(fa) != np.signbit(fb):
            roots.append(brentq(function, a, b, xtol=1e-12, rtol=1e-12))
    unique = []
    for root in sorted(roots):
        if not unique or root-unique[-1] > 1e-9:
            unique.append(float(root))
    return tuple(unique)


def _candidate(alpha, airspeed, engine_tilt, params, *, max_elevator_deflection,
               max_engine_rpm, alpha_min, alpha_max) -> TrimCandidate:
    thrust = rpm = elevator = None
    reasons = []
    if alpha_min is not None and alpha < alpha_min:
        reasons.append("alpha below prescribed validity limit")
    if alpha_max is not None and alpha > alpha_max:
        reasons.append("alpha above prescribed validity limit")
    if abs(np.cos(alpha)) < 1e-6:
        reasons.append("pitch is at the Euler singularity")
    try:
        _, drag = longitudinal_forces(alpha, airspeed, params)
        cosine = np.cos(engine_tilt+alpha)
        if abs(cosine) < 1e-8:
            raise ValueError("singular thrust projection: cos(engine_tilt + alpha) is zero")
        thrust = float(drag/cosine)  # Eq. 2.15
        if not np.isfinite(thrust) or thrust < 0:
            raise ValueError("required total thrust is negative or nonfinite")
        if drag < 0:
            reasons.append("negative aerodynamic drag")
        speed = float(np.sqrt(thrust/(2*_thrust_factor(params))))  # Eq. 2.16, rev/s
        rpm = 60*speed
        if max_engine_rpm is not None and rpm > max_engine_rpm:
            reasons.append("engine RPM exceeds limit")

        def c(name):
            return evaluate(getattr(params.aerodynamics, name), alpha, 0.0)

        qsc = .5*params.air_density*airspeed**2 * params.wing_area * params.mean_chord
        bias = c("pitch_0") + c("pitch_alpha")*alpha + params.engine_height*thrust*np.cos(engine_tilt)/qsc
        effectiveness = c("pitch_elevator")
        # Solving qSc(Cm0 + Cm_alpha*alpha + Cm_deltaE*deltaE)
        # + E_height*P*cos(tilt) = 0 requires a MINUS sign (PDF 2.17 typo).
        if effectiveness == 0:
            if abs(bias*qsc) > 1e-9:
                raise ValueError("zero elevator effectiveness cannot balance pitch moment")
            elevator = 0.0  # Already balanced; pick zero from the undetermined settings.
        else:
            elevator = float(-bias/effectiveness)
        if not np.isfinite(elevator):
            raise ValueError("required elevator deflection is nonfinite")
        if max_elevator_deflection is not None and abs(elevator) > max_elevator_deflection:
            reasons.append("elevator deflection exceeds limit")

        # PDF 2.19-2.20. Minimum-norm settings also handle a consistent singular
        # system; a symmetric model with no lateral effectiveness gives zeros.
        matrix = np.array([[c("roll_aileron"), c("roll_rudder")],
                           [c("yaw_aileron"), c("yaw_rudder")]])
        target = -np.array([c("roll_0"), c("yaw_0")])
        aileron, rudder = np.linalg.lstsq(matrix, target, rcond=None)[0]
        if not np.allclose(matrix @ [aileron, rudder], target, atol=1e-10, rtol=1e-10):
            reasons.append("aileron/rudder cannot balance roll and yaw")
        state = State(pitch=alpha, u=airspeed*np.cos(alpha), w=airspeed*np.sin(alpha),
                      left_motor_tilt=engine_tilt, right_motor_tilt=engine_tilt)
        controls = ControlInputs(left_propeller_speed=speed, right_propeller_speed=speed,
                                 elevator_deflection=elevator, aileron_deflection=aileron,
                                 rudder_deflection=rudder, left_motor_tilt=engine_tilt,
                                 right_motor_tilt=engine_tilt)
        solution = None if reasons else _solution(params, state, controls, thrust, airspeed, alpha)
    except (ValueError, np.linalg.LinAlgError) as error:
        reasons.append(str(error))
        solution = None
    return TrimCandidate(alpha, thrust, rpm, elevator, tuple(reasons), solution)


def forward_flight_trim(params: ModelParameters, engine_tilt: float, airspeed: float, *,
                        max_elevator_deflection: float | None = None,
                        max_engine_rpm: float | None = None,
                        alpha_min: float | None = None, alpha_max: float | None = None,
                        grid_points: int = 4001) -> ForwardTrimResult:
    """Find all bracketed trims, reject infeasible candidates, select low |alpha|."""
    if not np.isfinite(engine_tilt) or not np.isfinite(airspeed) or airspeed <= 0:
        raise ValueError("engine_tilt must be finite and airspeed must be finite and positive")
    _thrust_factor(params)
    for name, value in (("max_elevator_deflection", max_elevator_deflection), ("max_engine_rpm", max_engine_rpm)):
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    for name, value in (("alpha_min", alpha_min), ("alpha_max", alpha_max)):
        if value is not None and not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if alpha_min is not None and alpha_max is not None and alpha_min > alpha_max:
        raise ValueError("alpha_min must not exceed alpha_max")
    if isinstance(grid_points, bool) or not isinstance(grid_points, (int, np.integer)) or grid_points < 3:
        raise ValueError("grid_points must be an integer of at least 3")
    domain = aerodynamic_alpha_range(params)
    grid = np.linspace(*domain, grid_points)
    # Include table knots to avoid stepping across narrow piecewise-linear features.
    knots = []
    for field in fields(params.aerodynamics):
        table = getattr(params.aerodynamics, field.name)
        if isinstance(table, CoefficientMap):
            axis = np.deg2rad(table.alphas) if table.angle_unit == "deg" else table.alphas
            knots.extend(a for a in axis if domain[0] <= a <= domain[1])
    grid = np.unique(np.concatenate((grid, knots)))
    roots = bracketed_roots(lambda alpha: forward_residual(alpha, airspeed, engine_tilt, params), grid)
    candidates = tuple(_candidate(alpha, airspeed, engine_tilt, params,
                                   max_elevator_deflection=max_elevator_deflection,
                                   max_engine_rpm=max_engine_rpm, alpha_min=alpha_min,
                                   alpha_max=alpha_max) for alpha in roots)
    feasible = [candidate.solution for candidate in candidates if candidate.solution is not None]
    selected = min(feasible, key=lambda s: (not s.pre_stall, abs(s.alpha), s.alpha)) if feasible else None
    return ForwardTrimResult(domain, candidates, selected)
