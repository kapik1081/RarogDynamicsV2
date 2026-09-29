"""Transition trim surface and branch-aware path search (PDF section 2.3).

Coordinates are (airspeed, tilt, pitch, total thrust, elevator), in SI units."""

from dataclasses import dataclass, fields
import heapq
from typing import Callable

import numpy as np
from scipy.optimize import brentq, minimize_scalar, root, lsq_linear
from scipy.spatial import cKDTree

from .coefficients import CoefficientMap, evaluate
from .parameters import ModelParameters
from .trim import TrimSolution, _solution, _thrust_factor, aerodynamic_alpha_range, longitudinal_forces
from .types import ControlInputs, State


@dataclass(frozen=True)
class TransitionConstraints:
    max_engine_rpm: float
    max_elevator_deflection: float
    tilt_min: float = 0.0
    tilt_max: float = np.pi/2
    pitch_min: float = -np.pi/6
    pitch_max: float = np.pi/6
    alpha_min: float | None = None
    alpha_max: float | None = None
    max_aileron_deflection: float | None = None
    max_rudder_deflection: float | None = None
    max_shaft_power: float | None = None
    monotone_airspeed: bool = False
    monotone_tilt: bool = False

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
            if field.name.startswith("max_") and value is not None and value < 0:
                raise ValueError(f"{field.name} must be nonnegative")
        for name in ("tilt", "pitch", "alpha"):
            low, high = getattr(self, name+"_min"), getattr(self, name+"_max")
            if low is not None and high is not None and low > high:
                raise ValueError(f"{name}_min must not exceed {name}_max")
        if not -np.pi/2 < self.pitch_min <= self.pitch_max < np.pi/2:
            raise ValueError("pitch limits must lie strictly between -pi/2 and pi/2")


@dataclass(frozen=True)
class TransitionPoint:
    solution: TrimSolution
    shaft_power: float
    reserve: float

    @property
    def coordinates(self):
        s = self.solution
        return np.array([s.airspeed, s.controls.left_motor_tilt, s.state.pitch,
                         s.total_thrust, s.controls.elevator_deflection])


@dataclass(frozen=True)
class TransitionResult:
    points: tuple[TransitionPoint, ...]
    path: tuple[TransitionPoint, ...]
    airspeeds: np.ndarray
    tilts: np.ndarray
    # Number of accepted branches at each sampled (speed, tilt) pair.
    branch_counts: np.ndarray
    unresolved: tuple[tuple[float, float], ...]
    cruise_interval: tuple[float, float]
    constraints: TransitionConstraints


def trim_residual(z, params: ModelParameters):
    """Eq. 2.27/2.57, with exact zero-speed load limits."""
    speed, tilt, pitch, thrust, elevator = z
    weight = params.mass*params.gravity
    if speed == 0:
        lift = drag = aero_moment = 0.0
    else:
        lift, drag = longitudinal_forces(pitch, speed, params)
        c = lambda name: evaluate(getattr(params.aerodynamics, name), pitch, 0.0)
        aero_moment = (.5*params.air_density*speed**2*params.wing_area*params.mean_chord
                       * (c("pitch_0")+c("pitch_alpha")*pitch+c("pitch_elevator")*elevator))
    chi = tilt+pitch
    return np.array([(thrust*np.cos(chi)-drag)/weight,
                     (lift+thrust*np.sin(chi)-weight)/weight,
                     (aero_moment+params.engine_height*thrust*np.cos(tilt))
                     / (weight*params.mean_chord)])


def scalar_roots(function, grid, values=None, tolerance=1e-10):
    """Brent sign changes plus local |H| minima for even-multiplicity roots.

    Table knots belong in grid. Finite sampling cannot certify completeness.
    A sampled continuum is returned as samples, not mislabeled a failed solve.
    """
    grid = np.asarray(grid, dtype=float)
    values = np.array([function(x) for x in grid]) if values is None else np.asarray(values)
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite scalar trim residual")
    roots = list(grid[np.abs(values) <= tolerance])
    for i in range(len(grid)-1):
        if values[i]*values[i+1] < 0:
            roots.append(brentq(function, grid[i], grid[i+1], xtol=1e-13))
    absolute = np.abs(values)
    for i in range(1, len(grid)-1):
        if (absolute[i] < absolute[i-1] and absolute[i] < absolute[i+1]
                and values[i-1]*values[i+1] > 0):
            fit = minimize_scalar(lambda x: function(x)**2, bounds=(grid[i-1], grid[i+1]),
                                  method="bounded", options={"xatol": 1e-14})
            if abs(function(fit.x)) <= tolerance:
                roots.append(float(fit.x))
    unique = []
    for value in sorted(roots):
        if not unique or value-unique[-1] > 1e-7:
            unique.append(float(value))
    return unique


def _jacobian(function, x):
    """Finite differences in fixed nondimensional coordinates; one-sided at map edges."""
    value = function(x)
    columns = []
    for i in range(len(x)):
        step = np.zeros(len(x))
        step[i] = 2e-6
        try:
            right = function(x+step)
        except ValueError:
            right = None
        try:
            left = function(x-step)
        except ValueError:
            left = None
        if right is not None and left is not None:
            columns.append((right-left)/(2*step[i]))
        elif right is not None:
            columns.append((right-value)/step[i])
        elif left is not None:
            columns.append((value-left)/step[i])
        else:
            raise ValueError("No evaluable neighbourhood for continuation")
    return np.column_stack(columns)


class _Surface:
    def __init__(self, params, constraints, max_speed):
        self.params, self.limits = params, constraints
        self.weight = params.mass*params.gravity
        if self.weight <= 0:
            raise ValueError("Transition trim requires positive weight")
        self.factor = _thrust_factor(params)
        self.max_thrust = 2*self.factor*(constraints.max_engine_rpm/60)**2
        self.scale = np.array([max_speed, np.pi/2, .5, self.weight,
                               max(constraints.max_elevator_deflection, .1)])
        low, high = aerodynamic_alpha_range(params)
        self.domain = (max(low, constraints.pitch_min,
                           constraints.alpha_min if constraints.alpha_min is not None else low),
                       min(high, constraints.pitch_max,
                           constraints.alpha_max if constraints.alpha_max is not None else high))
        if self.domain[0] > self.domain[1]:
            raise ValueError("No common pitch / aerodynamic validity domain at positive airspeed")

    def residual(self, x):
        return trim_residual(x*self.scale, self.params)

    def accept(self, z):
        """Check inequalities before full-plant validation; never clip a control."""
        p, c = self.params, self.limits
        speed, tilt, pitch, thrust, elevator = z
        tol = 1e-9
        if (not np.all(np.isfinite(z)) or speed < 0 or speed > self.scale[0]+tol
                or not c.tilt_min-tol <= tilt <= c.tilt_max+tol
                or not c.pitch_min-tol <= pitch <= c.pitch_max+tol
                or not 0 <= thrust <= self.max_thrust+tol
                or abs(elevator) > c.max_elevator_deflection+tol):
            return None
        if speed > 0 and not self.domain[0] <= pitch <= self.domain[1]:
            return None
        try:
            if np.max(np.abs(trim_residual(z, p))) > 1e-8:
                return None
            n = float(np.sqrt(thrust/(2*self.factor)))
            power = float(4*np.pi*p.air_density*p.Kq*p.propeller_diameter**5*n**3)
            if c.max_shaft_power is not None and power > c.max_shaft_power+tol:
                return None
            aileron = rudder = 0.0
            if speed > 0:
                if longitudinal_forces(pitch, speed, p)[1] < 0:
                    return None
                coeff = lambda name: evaluate(getattr(p.aerodynamics, name), pitch, 0.0)
                matrix = np.array([[coeff("roll_aileron"), coeff("roll_rudder")],
                                   [coeff("yaw_aileron"), coeff("yaw_rudder")]])
                target = -np.array([coeff("roll_0"), coeff("yaw_0")])
                controls = np.linalg.lstsq(matrix, target, rcond=None)[0]
                bounds = np.array([c.max_aileron_deflection if c.max_aileron_deflection is not None else np.inf,
                                   c.max_rudder_deflection if c.max_rudder_deflection is not None else np.inf])
                # A minimum-norm solution can violate a bound even when another
                # solution of a rank-deficient lateral system is attainable.
                if np.any(np.abs(controls) > bounds):
                    free = bounds > 0
                    controls = np.zeros(2)
                    if np.any(free):
                        controls[free] = lsq_linear(matrix[:, free], target,
                                                  bounds=(-bounds[free], bounds[free]),
                                                  tol=1e-12).x
                if (np.any(np.abs(controls) > bounds+tol)
                        or not np.allclose(matrix@controls, target, atol=1e-10, rtol=0)):
                    return None
                aileron, rudder = controls
            state = State(pitch=pitch, u=speed*np.cos(pitch), w=speed*np.sin(pitch),
                          left_motor_tilt=tilt, right_motor_tilt=tilt)
            controls = ControlInputs(left_propeller_speed=n, right_propeller_speed=n,
                                     left_motor_tilt=tilt, right_motor_tilt=tilt,
                                     elevator_deflection=elevator, aileron_deflection=aileron,
                                     rudder_deflection=rudder)
            solution = _solution(p, state, controls, thrust, speed, pitch if speed else 0.0)
        except (ValueError, np.linalg.LinAlgError):
            return None
        reserves = [(self.max_thrust-thrust)/self.max_thrust if self.max_thrust else 0,
                    1-abs(elevator)/c.max_elevator_deflection if c.max_elevator_deflection else 0]
        if c.max_shaft_power is not None:
            reserves.append(1-power/c.max_shaft_power if c.max_shaft_power else 0)
        return TransitionPoint(solution, power, float(min(reserves)))

    def reconstruct(self, speed, tilt, pitch):
        """Eq. 2.29 and (2.34)."""
        p = self.params
        lift, drag = longitudinal_forces(pitch, speed, p)
        thrust = drag*np.cos(tilt+pitch)+(self.weight-lift)*np.sin(tilt+pitch)
        coeff = lambda name: evaluate(getattr(p.aerodynamics, name), pitch, 0.0)
        qsc = .5*p.air_density*speed**2*p.wing_area*p.mean_chord
        bias = qsc*(coeff("pitch_0")+coeff("pitch_alpha")*pitch)+p.engine_height*thrust*np.cos(tilt)
        authority = qsc*coeff("pitch_elevator")
        if abs(bias) > abs(authority)*self.limits.max_elevator_deflection+1e-10:
            return None
        if authority == 0:
            if abs(bias) > 1e-10:
                return None
            elevator = 0.0
        else:
            elevator = -bias/authority
        return self.accept(np.array([speed, tilt, pitch, thrust, elevator]))

    def correct_chart(self, start, independent, target):
        """Eq. 2.56"""
        x = np.asarray(start)/self.scale
        independent = list(independent)
        dependent = [i for i in range(5) if i not in independent]
        target = np.asarray(target)/self.scale[independent]
        try:
            jac = _jacobian(self.residual, x)
            predicted = x.copy()
            try:
                predicted[dependent] -= np.linalg.solve(jac[:, dependent],
                                                        jac[:, independent]@(target-x[independent]))
            except np.linalg.LinAlgError:
                # Zero engine height has a genuinely rank-deficient hover
                # boundary. A positive-speed corrector can still leave it;
                # only a verified, continuous correction is accepted by callers.
                pass
            predicted[independent] = target

            def equation(y):
                trial = predicted.copy()
                trial[dependent] = y
                return self.residual(trial)

            fit = root(equation, predicted[dependent], tol=1e-10)
            predicted[dependent] = fit.x
            if np.max(np.abs(self.residual(predicted))) > 1e-9:
                return None
            return predicted*self.scale
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return None

    def arclength(self, seed, step=.04, count=80):
        """Fixed-speed fold traversal, Eqs. 2.58-2.59, in scaled coordinates."""
        for direction in (-1, 1):
            x = seed/self.scale
            previous = None
            distance = step
            for _ in range(count):
                try:
                    jac = _jacobian(self.residual, x)[:, 1:]
                    _, singular, vh = np.linalg.svd(jac, full_matrices=True)
                    if singular[-1] < 1e-10:
                        break
                    tangent = vh[-1]
                    tangent *= direction if previous is None else (1 if tangent@previous >= 0 else -1)
                    predicted = x[1:]+distance*tangent

                    def equation(w):
                        trial = np.r_[x[0], w]
                        return np.r_[self.residual(trial), tangent@(w-predicted)]

                    fit = root(equation, predicted, tol=1e-10)
                    if (np.max(np.abs(equation(fit.x))) > 1e-9
                            or np.linalg.norm(fit.x-predicted) > distance):
                        distance /= 2
                        if distance < step/32:
                            break
                        continue
                    x = np.r_[x[0], fit.x]
                    z = x*self.scale
                    if (not self.domain[0] <= z[2] <= self.domain[1]
                            or z[3] < 0 or abs(x[1]) > 2 or abs(x[4]) > 4):
                        break
                    yield z
                    previous = tangent
                    distance = min(step, distance*1.3)
                except (ValueError, np.linalg.LinAlgError):
                    break


def _connect(surface, first, last, step=.035):
    """Correct interpolated independent coordinates; check every intermediate trim."""
    start, end = first.coordinates, last.coordinates
    limits = surface.limits
    if limits.monotone_airspeed and end[0] < start[0]-1e-10:
        return None
    if limits.monotone_tilt and end[1] > start[1]+1e-10:
        return None
    distance = np.linalg.norm((end-start)/surface.scale)
    if distance > .65:
        return None
    # Use the best-conditioned chart at the start. Pitch as a local
    # coordinate also spans folds where the tilt projection turns back.
    charts = [(0, 4), (0, 1), (0, 2)]
    try:
        jac = _jacobian(surface.residual, start/surface.scale)
        charts.sort(key=lambda ij: np.linalg.cond(jac[:, [k for k in range(5) if k not in ij]]))
    except (ValueError, np.linalg.LinAlgError):
        return None
    for chart in charts:
        current = first
        points = [first]
        fraction = 0.0
        increment = 1/max(2, int(np.ceil(distance/step)))
        while fraction < 1-1e-12:
            next_fraction = min(1.0, fraction+increment)
            target = (1-next_fraction)*start+next_fraction*end
            corrected = surface.correct_chart(current.coordinates, chart, target[list(chart)])
            point = None if corrected is None else surface.accept(corrected)
            if point is not None:
                correction = np.linalg.norm((corrected-target)/surface.scale)
                jump = np.linalg.norm((corrected-current.coordinates)/surface.scale)
                previous_controls = current.solution.controls
                next_controls = point.solution.controls
                lateral_jump = max(abs(next_controls.aileron_deflection-previous_controls.aileron_deflection),
                                   abs(next_controls.rudder_deflection-previous_controls.rudder_deflection))
                valid = (correction <= max(.015, distance*.3) and jump <= step*2
                         and lateral_jump <= step*2
                         and (not limits.monotone_airspeed or corrected[0] >= current.coordinates[0]-1e-10)
                         and (not limits.monotone_tilt or corrected[1] <= current.coordinates[1]+1e-10))
            else:
                valid = False
            if not valid:
                increment /= 2
                if increment*max(distance, .01) < .0003:
                    break
                continue
            points.append(point)
            current, fraction = point, next_fraction
        else:
            if np.linalg.norm((current.coordinates-end)/surface.scale) < 1e-6:
                # Match lateral controls as well as the five reduced coordinates.
                a, b = current.solution.controls, last.solution.controls
                if max(abs(a.aileron_deflection-b.aileron_deflection),
                       abs(a.rudder_deflection-b.rudder_deflection)) < 1e-6:
                    points[-1] = last
                    return tuple(points)
    return None


def _find_path(surface, points, cruise, progress):
    if not points:
        return ()
    coordinates = np.array([point.coordinates for point in points])
    starts = np.flatnonzero((coordinates[:, 0] == 0)
                           & (np.abs(coordinates[:, 1]-np.pi/2) < 1e-8)
                           & (np.abs(coordinates[:, 2]) < 1e-8))
    goals = set(np.flatnonzero((np.abs(coordinates[:, 1]) < 1e-8)
                              & (coordinates[:, 0] >= cruise[0]-1e-9)
                              & (coordinates[:, 0] <= cruise[1]+1e-9)))
    if not len(starts) or not goals:
        return ()
    # Neighbours are chosen in the complete trim space, including lateral controls.
    lateral = np.array([[p.solution.controls.aileron_deflection,
                         p.solution.controls.rudder_deflection] for p in points])
    scaled = np.column_stack((coordinates/surface.scale, lateral))
    tree = cKDTree(scaled)
    _, neighbours = tree.query(scaled, k=min(24, len(points)))
    # Symmetrize the k-nearest graph; constraints determine edge direction later.
    adjacency = [set(row)-{i} for i, row in enumerate(neighbours)]
    for i, row in enumerate(neighbours):
        for j in row:
            if i != j:
                adjacency[j].add(i)
    costs = {int(i): 0.0 for i in starts}
    queue = [(0.0, int(i)) for i in starts]
    heapq.heapify(queue)
    previous, edges, visited = {}, {}, set()
    while queue:
        cost, i = heapq.heappop(queue)
        if i in visited:
            continue
        if i in goals:
            route = []
            while i in previous:
                route.append(edges[i][1:])
                i = previous[i]
            return (points[i],)+tuple(p for segment in reversed(route) for p in segment)
        visited.add(i)
        if len(visited) % 100 == 0 and progress:
            progress(f"Path search: checked connections from {len(visited)} / {len(points)} trims")
        for j in sorted(adjacency[i]):
            if j in visited:
                continue
            length = float(np.linalg.norm(scaled[j]-scaled[i]))
            if cost+length >= costs.get(j, np.inf):
                continue
            connection = _connect(surface, points[i], points[j])
            if connection is None:
                continue
            reserve = max(0.0, min(p.reserve for p in connection))
            ai = np.array([points[i].solution.engine_rpm/surface.limits.max_engine_rpm,
                           scaled[i, 4], scaled[i, 1]])
            aj = np.array([points[j].solution.engine_rpm/surface.limits.max_engine_rpm,
                           scaled[j, 4], scaled[j, 1]])
            candidate = cost+length*(1+.01/(.1+reserve)**2)+.1*float(np.sum((aj-ai)**2))
            if candidate < costs.get(j, np.inf):
                costs[j], previous[j], edges[j] = candidate, i, connection
                heapq.heappush(queue, (candidate, j))
    return ()


def transition_trim_manifold(params: ModelParameters, constraints: TransitionConstraints, *,
                             cruise_airspeed: float | tuple[float, float],
                             airspeed_max: float | None = None, speed_points: int = 31,
                             tilt_points: int = 31, pitch_points: int = 241,
                             elevator_points: int = 9,
                             progress: Callable[[str], None] | None = None) -> TransitionResult:
    """Construct overlapping charts, retain branches, then search validated edges."""
    cruise = ((float(cruise_airspeed),)*2 if np.isscalar(cruise_airspeed)
              else tuple(cruise_airspeed))
    if len(cruise) != 2 or not np.all(np.isfinite(cruise)) or not 0 < cruise[0] <= cruise[1]:
        raise ValueError("cruise_airspeed must be positive, or an ordered positive interval")
    max_speed = cruise[1] if airspeed_max is None else airspeed_max
    if not np.isfinite(max_speed) or max_speed < cruise[1]:
        raise ValueError("airspeed_max must be finite and at least the maximum cruise airspeed")
    for name, value in (("speed_points", speed_points), ("tilt_points", tilt_points),
                        ("pitch_points", pitch_points), ("elevator_points", elevator_points)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 3:
            raise ValueError(f"{name} must be an integer of at least 3")
    surface = _Surface(params, constraints, max_speed)
    speeds = np.unique(np.r_[max_speed*np.linspace(0, 1, speed_points)**2, cruise])
    tilts = np.linspace(constraints.tilt_min, constraints.tilt_max, tilt_points)
    tilts = np.unique(np.r_[tilts, [x for x in (0, np.pi/2)
                                   if constraints.tilt_min <= x <= constraints.tilt_max]])
    pitch_grid = np.linspace(*surface.domain, pitch_points)
    knots = []
    for field in fields(params.aerodynamics):
        table = getattr(params.aerodynamics, field.name)
        if isinstance(table, CoefficientMap):
            axis = np.deg2rad(table.alphas) if table.angle_unit == "deg" else table.alphas
            knots.extend(a for a in axis if surface.domain[0] <= a <= surface.domain[1])
    pitch_grid = np.unique(np.r_[pitch_grid, knots])
    # Loads at unit speed scale exactly with Va squared in the steady model.
    unit_lift, unit_drag = np.array([longitudinal_forces(a, 1.0, params) for a in pitch_grid]).T
    elevator_grid = np.unique(np.linspace(-constraints.max_elevator_deflection,
                                         constraints.max_elevator_deflection, elevator_points))
    if params.engine_height == 0 and surface.domain[0] <= 0 <= surface.domain[1]:
        effectiveness = evaluate(params.aerodynamics.pitch_elevator, 0, 0)
        if effectiveness:
            limit = -evaluate(params.aerodynamics.pitch_0, 0, 0)/effectiveness
            if abs(limit) <= constraints.max_elevator_deflection:
                elevator_grid = np.unique(np.r_[elevator_grid, limit])
    hover_seeds = [np.array([0, np.pi/2, 0, surface.weight, de]) for de in elevator_grid]
    points, keys, unresolved = [], set(), []
    counts = np.zeros((len(speeds), len(tilts)), dtype=int)

    def add(point):
        if point is not None:
            key = tuple(np.round(point.coordinates/surface.scale, 8))
            if key not in keys:
                keys.add(key)
                points.append(point)

    for seed in hover_seeds:
        add(surface.accept(seed))
    for i, speed in enumerate(speeds):
        if progress:
            progress(f"Surface slice {i+1}/{len(speeds)}: {speed:.4g} m/s ({len(points)} viable trims)")
        if speed == 0:
            for j, tilt in enumerate(tilts):
                # E_height=0 has a larger hover family. Its additional equilibria
                # are retained while upright hover remains the requested start.
                z = np.array([0, tilt, np.pi/2-tilt, surface.weight, 0.0])
                point = surface.accept(z)
                if point is not None:
                    counts[i, j] = 1
                    add(point)
            continue
        # Regular (Va, deltaE) chart continued from the entire sampled hover boundary.
        for k, seed in enumerate(hover_seeds):
            corrected = surface.correct_chart(seed, (0, 4), (speed, elevator_grid[k]))
            if corrected is None:
                unresolved.append((float(speed), float(seed[1])))
            else:
                hover_seeds[k] = corrected
                add(surface.accept(corrected))
        fold_seeds = []
        lift, drag = unit_lift*speed**2, unit_drag*speed**2
        for j, tilt in enumerate(tilts):
            def scalar(pitch):
                l, d = longitudinal_forces(pitch, speed, params)
                return ((l-surface.weight)*np.cos(tilt+pitch)+d*np.sin(tilt+pitch))/surface.weight

            values = ((lift-surface.weight)*np.cos(tilt+pitch_grid)
                      + drag*np.sin(tilt+pitch_grid))/surface.weight
            for pitch in scalar_roots(scalar, pitch_grid, values):
                point = surface.reconstruct(speed, tilt, pitch)
                if point is None:
                    continue
                counts[i, j] += 1
                add(point)
                try:
                    jac = _jacobian(surface.residual, point.coordinates/surface.scale)
                    force_det = np.linalg.det(jac[:2, [2, 3]])
                    # Vanishing elevator authority alone makes Jy ill-conditioned
                    # near hover; that is handled by its own regular chart.
                    if np.linalg.cond(jac[:, [2, 3, 4]]) > 100 and abs(force_det) < .05:
                        if all(np.linalg.norm((point.coordinates-z)/surface.scale) > .2 for z in fold_seeds):
                            fold_seeds.append(point.coordinates)
                except (ValueError, np.linalg.LinAlgError):
                    unresolved.append((float(speed), float(tilt)))
        # A force-direction chart supplements the tilt grid. For positive thrust,
        # (Va, pitch) determines P and eta without division by cos(eta+pitch).
        # This samples narrow actuator boundaries and both sides of tilt folds.
        force_samples = []
        for pitch, l, d in zip(pitch_grid, lift, drag):
            tilt = np.arctan2(surface.weight-l, d)-pitch
            point = None
            if constraints.tilt_min <= tilt <= constraints.tilt_max:
                point = surface.reconstruct(speed, tilt, pitch)
                add(point)
            force_samples.append(point)
        # Refine crossings of actuator/validity boundaries in this chart. Keep
        # accepted bisection samples, rather than filling between projected dots.
        for k in range(len(pitch_grid)-1):
            left_ok = force_samples[k] is not None
            right_ok = force_samples[k+1] is not None
            if left_ok == right_ok:
                continue
            low, high = pitch_grid[k:k+2]
            for _ in range(5):
                pitch = (low+high)/2
                l, d = longitudinal_forces(pitch, speed, params)
                tilt = np.arctan2(surface.weight-l, d)-pitch
                point = surface.reconstruct(speed, tilt, pitch)
                add(point)
                if (point is not None) == left_ok:
                    low = pitch
                else:
                    high = pitch
        for seed in fold_seeds:
            for z in surface.arclength(seed):
                add(surface.accept(z))
    if progress:
        progress(f"Surface complete: {len(points)} viable trims; searching for a hover-to-cruise path")
    path = _find_path(surface, points, cruise, progress)
    return TransitionResult(tuple(points), path, speeds, tilts, counts, tuple(unresolved),
                            cruise, constraints)
