Python implementation of VTOL dynamics showcased within the "DERIVATION OF THE MATHEMATICAL MODEL OF A VERTICAL TAKE-OFF UNMANNED AERIAL VEHICLE WITH TILTED-ROTOR DRIVE", with all 14 states, independent
motor tilts, aerodynamic coefficient maps, and RK45 integration. The modules
contain pure functions; `VTOLModel` alone stores the evolving state and time.
The supplied parameters are illustrative and must be replaced with measured
or identified values before aircraft validation.

Hover and forward straight-level trim scripts accept a craft JSON file as a
positional argument and print equilibrium states and controls:

```powershell
.\.venv\Scripts\python.exe find_hover_trim.py examples/sample_parameters.json
.\.venv\Scripts\python.exe find_forward_trim.py examples/sample_parameters.json --engine-tilt 0 --airspeed 12 --max-elevator-deflection 25 --max-engine-rpm 8000
```

Forward trim scans the full aerodynamic alpha range and applies Brent's method
to every detected sign-changing interval. It reports candidates and rejection
reasons before preferring a feasible pre-stall solution with small absolute
alpha. CLI angles use degrees; speed uses m/s; the RPM limit applies to each
engine. Optional `--alpha-min`, `--alpha-max`, and `--grid-points` control
validity filtering and scan resolution. See [trim documentation](docs/trim.md)
for equations, PDF sign corrections, limits, and Python usage.

```powershell
.\.venv\Scripts\python.exe find_transition_trim.py examples/sample_parameters.json --max-engine-rpm 8000 --max-elevator-deflection 25 --cruise-airspeed 12 --o transition.png
```

`--o FILENAME` optionally saves the plot; `--no-show` suppresses the window.
Specify a cruise interval with `--cruise-airspeed 12 18`. Additional tilt,
pitch, aerodynamic-validity, lateral-surface and shaft-power constraints are
available through `--help`. See [transition search documentation](docs/transition.md)
for the numerical method, resolution controls, and interpretation of the path.

From the project directory, using the existing environment:

```powershell
.\.venv\Scripts\python.exe -m examples.hover
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Only NumPy and SciPy are required at runtime. To import the package from other
projects, install it into the desired environment with `python -m pip install -e .`.

Convert an XFLR5 plane polar text export into a parameter JSON template:

```powershell
.\.venv\Scripts\python.exe xflr5_to_parameters.py T1.txt
# Optional: -o aircraft.json --stall-at-extremes
```

Missing values are `"provide-data"` and must be completed before simulation. The supplied polar recovers 7 of 41 parameter slots; the stall flag fills two assumed thresholds and enables stall (10/41). See [the import guide and recovery report](docs/xflr5_import.md) for options, angle selection, units, and the limits of this beta-zero polar.

```python
import math
from vtol_dynamics import ControlInputs, ModelParameters, State, VTOLModel

parameters = ModelParameters.from_json("examples/sample_parameters.json")
initial_state = State(
    z=-2.0,                     # 2 m above the ground in NED
    left_motor_tilt=math.pi/2,
    right_motor_tilt=math.pi/2,
)
model = VTOLModel(parameters, initial_state, terrain_collision=True)

hover_speed = math.sqrt(
    parameters.mass * parameters.gravity
    / (2 * parameters.CT * parameters.air_density * parameters.propeller_diameter**4)
)
controls = ControlInputs(
    left_propeller_speed=hover_speed,
    right_propeller_speed=hover_speed,
    elevator_deflection=0.0,
    aileron_deflection=0.0,
    rudder_deflection=0.0,
    left_motor_tilt=math.pi/2,
    right_motor_tilt=math.pi/2,
)
next_state = model.step(controls, dt=0.01)
current_state = model.get_state()
print(model.time, current_state.z)
model.terrain_collision = False  # Can also be changed between steps.
```

`step` holds the supplied controls constant throughout the interval, integrates
using SciPy `solve_ivp(method="RK45")`, commits the result, and returns a `State`.
It reevaluates loads and coefficient maps at every ODE evaluation. `dt=0` is a
no-op; negative or nonfinite intervals are rejected. A failed integration leaves
the previous state and time intact. `rtol`, `atol`, and `max_step` are optional
constructor arguments (defaults: `1e-7`, `1e-9`, and infinity). These are local
solver controls, not guarantees of accumulated trajectory error; reduce them
and compare converged trajectories for validation, especially near actuator
rate-limit transitions or sharp coefficient changes.

The state and parameter objects are immutable. Use `dataclasses.replace` to
derive modified parameters or initial states; construct another `VTOLModel` to
run a new case. `state.as_vector()` returns a fresh NumPy array and
`State.from_vector(array)` reconstructs the object. State vector order matches
Eq. 1.1, including **right tilt before left tilt**:

```text
[x, y, z, roll, pitch, yaw, u, v, w, p, q, r, right_motor_tilt, left_motor_tilt]
```

| Quantity | Convention / units |
| --- | --- |
| `x, y, z` | Earth north, east, down; metres. Ground is `z=0`; airborne is `z<0`. |
| `u, v, w` | Body forward, right, down velocities; m/s. |
| `roll, pitch, yaw` | 3-2-1 Euler attitude; radians. |
| `p, q, r` | Body angular rates; rad/s. |
| Motor speeds | Nonnegative **revolutions/s**. Divide RPM by 60 or rad/s by `2*pi`. |
| Tilt states / commands | Radians. Zero means forward thrust; `pi/2` means upward thrust. |
| Surface deflections | Radians, with effectiveness signs determined by the coefficients. |
| `mass`, inertia entries | kg and kg m². The inertia matrix uses **negative** `Ixz` off-diagonals. |
| `wing_area`, `wingspan`, `mean_chord` | m², m, m; PDF `S`, `b`, and mean `c`. |
| `propeller_diameter`, `engine_span` | m; span is the distance from the CoM to **each** engine. |
| `engine_height` | Signed body-z engine offset in m; negative means above the CoM. Both engine x offsets are zero. |
| `CT`, `Kq` | Dimensionless thrust and reaction-torque coefficients. |
| `air_density`, `gravity` | kg/m³ and m/s². |
| `tilt_time_constant`, `tilt_max_rate` | s and rad/s. |
| Stall angle thresholds / transition width | Always radians, independently of map axis units. |

Aerodynamic tables have shape **`(len(betas), len(alphas))`**, so beta selects
the row and alpha selects the column. For the grid in the request:

```python
import numpy as np
from vtol_dynamics import CoefficientMap

coefficient = CoefficientMap(
    alphas=[-2, 0, 5],
    betas=[-2, 0, 2],
    values=[[-0.55, 0.25, 1.25],
            [-0.85, 0.85, 2.25],
            [-0.55, 0.25, 1.25]],
    angle_unit="deg",
    bounds="clamp",
)
print(coefficient(*np.deg2rad([0, 0])))    # 0.85
print(coefficient(*np.deg2rad([-1, -1])))  # -0.075 (bilinear interpolation)
```

Map constructors and JSON axes accept `angle_unit="deg"` or `"rad"` (default).
Lookup calls always take radians. Axes must be finite and strictly increasing;
unequal grid spacing is supported. A singleton axis represents no variation
along that axis. `bounds="clamp"` (default) holds the nearest boundary value
outside the map; `bounds="raise"` rejects such queries. There is no implicit
extrapolation or angle wrapping. Map data are copied into immutable tuples.

Every field in `AerodynamicCoefficients` accepts either a `CoefficientMap` or a
scalar constant. The JSON representation of a map uses the same keys as the
constructor. The sample includes maps for all three force coefficients and an
aileron effectiveness derivative. Scalars are convenient for derivatives that
are assumed constant. Omitted aerodynamic fields default to zero.

| Python / JSON coefficient | PDF symbol and use |
| --- | --- |
| `lift`, `drag`, `sideforce` | `Cl`, `Cd`, `Cy`; complete force coefficients before lift/drag stall blending. |
| `roll_0`, `roll_beta`, `roll_p`, `roll_r` | `Cl0`, `Clβ`, `Clp`, `Clr`. |
| `roll_aileron`, `roll_rudder` | `ClδA`, `ClδR`. |
| `pitch_0`, `pitch_alpha`, `pitch_q`, `pitch_elevator` | `Cm0`, `Cmα`, `Cmq`, `CmδE`. |
| `yaw_0`, `yaw_beta`, `yaw_p`, `yaw_r` | `Cn0`, `Cnβ`, `Cnp`, `Cnr`. |
| `yaw_aileron`, `yaw_rudder` | `CnδA`, `CnδR`. |

The moment derivative fields still multiply the indicated angles, rates, or
deflections, even when supplied as maps. For example, `pitch_alpha(alpha,beta)`
is multiplied by alpha; it is not the complete pitch-moment coefficient. To
provide a complete static pitch-moment map, use `pitch_0` and set `pitch_alpha`
to zero. Force `sideforce` is already `Cy`, so it is not multiplied by beta.
Rate derivatives follow the PDF's nondimensional-rate definition; the code
analytically cancels airspeed denominators in the moment equations.

The implementation follows these choices from the PDF:

- Gravity always uses the full body-frame expression in Eq. 1.7:
  `g * [-sin(pitch), sin(roll)*cos(pitch), cos(roll)*cos(pitch)]`.
  Remove the obsolete `gravity_model` field from older JSON files and Python
  parameter constructors; the gravity magnitude remains configurable.
- Lift and drag use the stall blending in Eqs. 1.16–1.18. In Eq. 1.16 the
  negative-stall term divides by **negative** transition width. Set
  `stall.enabled=false` when tables already include the complete stall model.
  Sideforce and moment coefficients are not modified by this blend.
- Angular acceleration solves `I @ omega_dot = moment - cross(omega, I @ omega)`
  directly, using Eq. 1.28 and the inertia matrix of Eq. 1.29. This is equivalent
  to Eqs. 1.30–1.39 and is easier to modify than the ten expanded constants.
- Propeller reaction moments preserve Eqs. 1.49–1.51 exactly, including the
  positive right-propeller `sin(tilt)` term in the z moment. This sign is
  explicit in `reaction_moment` for comparison with the intended hardware.
- The air is stationary: body velocity equals relative air velocity. There
  is no wind, rotor slipstream, motor-speed lag, nacelle gyroscopic torque,
  or added control-surface lift beyond what the PDF specifies. Nacelle tilt
  has first-order lag and a rate limit; mechanical angle stops are not imposed.
- Euler kinematics retain the PDF's pitch singularity. Evaluation very near
  `pitch = +/- pi/2` raises an error. Use a quaternion formulation for trajectories
that need to pass through gimbal lock.

Terrain is a frictionless, nonbouncing point contact at the CoM against `z=0`.
On impact, only the downward **earth-frame** velocity component is removed;
horizontal motion and angular rates continue. Ground contact cancels downward
normal acceleration, and integration resumes in free flight when forces allow
liftoff. RK45 events split a step at impact and liftoff, preserving the rest of
the requested interval. Enabling the toggle also immediately corrects an
initial state below the plane. This does not model landing gear, body extent,
terrain slopes, friction, or ground-induced moments. For complicated contact
trajectories, choose a finite `max_step` so crossings are well resolved.

The files are organized around the document's sections:

| Module | Independently testable responsibility |
| --- | --- |
| `coefficients.py` | Map validation and bilinear lookup. |
| `parameters.py`, `types.py` | JSON loading and immutable input/state objects. |
| `propulsion.py` | Tilt response (§1.2), thrust (§1.5), thrust/reaction moments (§1.8). |
| `aerodynamics.py` | Air data, stall blending, aerodynamic force (§1.6), moment (§1.9). |
| `kinematics.py` | Gravity (§1.4), frame transforms, Euler/position kinematics (§1.10). |
| `dynamics.py` | Newton–Euler accelerations, separate load components, full derivative (§1.11). |
| `terrain.py` | Contact projection and constrained derivative. |
| `model.py` | Stored state/time, integration, terrain mode transitions. |

For component validation, call the functions directly without advancing a model:

```python
from vtol_dynamics.dynamics import calculate_loads, state_derivative

loads = calculate_loads(initial_state, controls, parameters)
print(loads.thrust_force, loads.aerodynamic_force, loads.total_moment)
derivative = state_derivative(0.0, initial_state.as_vector(), controls, parameters)
```

The test suite checks table orientation and interpolation, force/moment signs,
frame transforms, zero-airspeed behavior, Newton–Euler coupling, analytic free
fall and actuator response, hover, integration convergence, terrain impact and
liftoff, JSON validation, and state isolation.

Four additional stateful models are available for testing individual parts of
the plant. They share the same parameter/state/control objects, `get_state()`,
RK45 `step(controls, dt)`, `time`, and solver options as `VTOLModel`.

| Testing class | Active physics | Fixed states |
| --- | --- | --- |
| `BicopterTranslationalModel` | Thrust, nacelle tilt dynamics, gravity, position and linear velocity. | Initial attitude; angular rates must be initialized to zero. |
| `BicopterRotationModel` | Thrust-offset and reaction moments, nacelle tilt dynamics, Euler attitude and angular rates. | Initial position and body linear velocity. |
| `BicopterFullModel` | Coupled translation and rotation, all engine loads, nacelle tilt dynamics, gravity. | None. |
| `AerodynamicsOnlyModel` | Aerodynamic forces and moments, stall blending, control surfaces, gravity, translation and rotation. | Initial nacelle tilts. |

The bicopter variants omit all aerodynamics, including post-stall effects, and
ignore surface deflections. The glider ignores both motor speeds and both tilt
commands, while keeping aerodynamic control surfaces active. Inputs still pass
the usual `ControlInputs` validation. All translating variants use exact gravity;
use `dataclasses.replace(parameters, gravity=0.0)` for an experiment without
gravity. The rotation-only variant has no gravity force calculation.

Terrain collision is available for all variants that translate. Enabling it
on `BicopterRotationModel` raises `ValueError`, because ground contact would
change the position/velocity states that this model holds fixed. Likewise,
`BicopterTranslationalModel` rejects nonzero initial `p`, `q`, or `r` to keep its
fixed-attitude frame consistent with its velocity and terrain calculations.

```python
from vtol_dynamics import (
    BicopterTranslationalModel, BicopterRotationModel,
    BicopterFullModel, AerodynamicsOnlyModel,
    ModelParameters, State, ControlInputs,
)

parameters = ModelParameters.from_json("examples/sample_parameters.json")
models = {
    "translation": BicopterTranslationalModel(parameters, State(z=-10)),
    "rotation": BicopterRotationModel(parameters, State(z=-10)),
    "bicopter": BicopterFullModel(parameters, State(z=-10)),
    "glider": AerodynamicsOnlyModel(parameters, State(z=-30, u=12)),
}
commands = ControlInputs(
    left_propeller_speed=60, right_propeller_speed=65,
    left_motor_tilt=1.0, right_motor_tilt=1.1,
    elevator_deflection=0.02,
)
for name, model in models.items():
    print(name, model.step(commands, dt=0.01))
```

The wrappers live in `vtol_dynamics/testing.py`. Their pure equations live in
`vtol_dynamics/testing_dynamics.py`: `bicopter_translational_derivative`,
`bicopter_rotation_derivative`, `bicopter_full_derivative`, and
`aerodynamics_only_derivative`. Each takes `(time, state_vector, controls,
parameters)` and returns a derivative in the existing 14-state order. Disabled
subsystems are never evaluated, so aerodynamic maps cannot affect bicopter
tests and engine parameters cannot create loads in glider tests. Existing
functions such as `calculate_loads` in `dynamics.py` continue to describe the
complete plant; use the reduced derivatives to inspect these testing models.

Four interactive playgrounds provide an animated 3-D flight path and a separate
statistics window. Run these commands from the repository root:

```powershell
.\.venv\Scripts\python.exe -m vtol_dynamics.playgrounds.bicopter_translation --time 3 --propeller-speed 90 --engine-tilt 90
.\.venv\Scripts\python.exe -m vtol_dynamics.playgrounds.bicopter_rotation --time 1 --right-engine-tilt 90 --left-engine-tilt 80 --right-propeller-speed 65 --left-propeller-speed 60
.\.venv\Scripts\python.exe -m vtol_dynamics.playgrounds.glide --time 3 --u 12 --v 0 --w 0
.\.venv\Scripts\python.exe -m vtol_dynamics.playgrounds.full_model --time 1 --control-right-motor-tilt 30 --control-left-motor-tilt 20 --control-right-propeller-speed 65 --control-left-propeller-speed 60 --initial-u 12
```

All command-line **tilts are in degrees**, propeller speeds are in **rev/s**,
velocities are in m/s, and `--time` is the simulation duration in seconds.
`--simulation-time` is an alias for `--time`. Each module also provides a `run`
function returning sampled states; Python calls use **radians** for tilts, as
in the rest of the model. Use `--help` to inspect a playground's arguments.

The translational playground applies the same constant speed and tilt command
to both motors. The rotation playground commands them independently. By default,
both start with **every state zero**, including actual engine tilts; the actual
tilts approach the commanded values through the actuator dynamics. Add
`--instant-tilt` to either bicopter playground to initialize each engine at its
requested tilt at time zero. All other initial states remain zero, and actuator
dynamics are unchanged. Python callers can pass `instant_tilt=True` to `run()`.
For example:

```powershell
.\.venv\Scripts\python.exe -m vtol_dynamics.playgrounds.bicopter_translation --time 3 --propeller-speed 90 --engine-tilt 90 --instant-tilt
```

The glide playground starts with only `u`, `v`, and `w` set and keeps all control
inputs zero. The `full_model` playground uses the complete `VTOLModel`, with
engine forces/moments, aerodynamic forces/moments, gravity, and coupled
translation/rotation all active. Every argument is optional: duration defaults
to **1 second**, and every omitted initial state or control component is **zero**.
State arguments use `--initial-<field>` and constant commands use
`--control-<field>`, matching the model's field names with hyphens:

| Arguments | Units / meaning |
| --- | --- |
| `--initial-x`, `--initial-y`, `--initial-z` | NED position [m]; negative z is above ground. |
| `--initial-roll`, `--initial-pitch`, `--initial-yaw` | Euler attitude [deg]. |
| `--initial-u`, `--initial-v`, `--initial-w` | Body linear velocity [m/s]. |
| `--initial-p`, `--initial-q`, `--initial-r` | Body angular rates [deg/s]. |
| `--initial-right-motor-tilt`, `--initial-left-motor-tilt` | Actual initial motor tilts [deg]. |
| `--control-right-motor-tilt`, `--control-left-motor-tilt` | Constant commanded motor tilts [deg]. |
| `--control-right-propeller-speed`, `--control-left-propeller-speed` | Constant nonnegative propeller speeds [rev/s]. |
| `--control-elevator-deflection`, `--control-aileron-deflection`, `--control-rudder-deflection` | Constant surface deflections [deg]. |

For the full-model playground, these names replace `--u`, `--v`, `--w`,
`--right-engine-tilt`, `--left-engine-tilt`, the unprefixed propeller-speed
arguments, and the unprefixed surface-deflection arguments. `--instant-tilt`
has been removed: set each initial motor tilt explicitly. Initial tilts and
commands are independent, and actuator dynamics remain active. To start at a
commanded tilt, provide the same value for both, for example
`--initial-right-motor-tilt 30 --control-right-motor-tilt 30`.

Python `full_model.run(parameters, simulation_time=1.0, ...)` accepts the same
state/control names as keyword arguments with underscores in place of hyphens,
such as `initial_z=-10`, `initial_u=12`, and `control_elevator_deflection=-0.02`.
Python angles use **radians** and angular rates use **rad/s**. All omitted state
and control keywords default to zero; `instant_tilt` has been removed.

Terrain collision is disabled in all four experiments, allowing
motion below altitude zero. No trim, initial altitude, or initial attitude is
added automatically.

The animation runs once from the first sample to the last. Three colored lines
extend from the CoM toward the positive body x/y/z axes; two more extend from
the engine offsets along their actual tilted thrust directions. Endpoint dots
indicate the positive direction. The trajectory grows as the craft moves.
For long trajectories, the craft lines and engine offsets are enlarged together
so orientation stays legible. Line length is at least 12% of the largest flight-path
span, with a constant display scale throughout playback. Position data stays in
metres, and the plot notes when the craft is enlarged.
The 3-D view uses north, east, and **altitude = -z** so upward flight appears
upward. The global position statistics retain the model's **NED z** convention.

The statistics window displays global x/y/z; roll/pitch/yaw; alpha/beta;
kinetic, potential, and total mechanical energy; airspeed and body u/v/w; and
actual left/right engine tilt. Angular plots use degrees. Kinetic energy is
`0.5*m*(u²+v²+w²) + 0.5*omega.T @ I @ omega`, including rotational energy and
the inertia cross term. Potential energy is `-m*g*z`, referenced to altitude
zero, and is allowed to be negative. Total energy is their sum; powered flight
and aerodynamic drag need not conserve it.
At zero airspeed, alpha and beta follow the model convention of zero.

Shared optional arguments:

| Argument | Behavior |
| --- | --- |
| `--parameters path.json` | Load custom static parameters; defaults to `examples/sample_parameters.json` in this checkout. Pass an explicit path when using an installed package without the examples. |
| `--dt 0.02` | Sampling interval in seconds. RK45 integrates adaptively between samples; a final partial interval reaches the requested duration. |
| `--fps 30` | Maximum animation frame rate. Plot statistics retain all simulation samples. |
| `--playback-speed 2` | Play the animation at twice simulation speed. |
| `--axis-length 2` | Minimum axis/engine line length in metres; automatically enlarged for long flight paths. Default: half wingspan. |
| `--output directory` | Save `flight.html` (animation with playback controls), `statistics.png`, and `history.csv`. Existing files with these names are replaced. |
| `--no-show` | Suppress GUI windows; combine with `--output` for headless exports, or use alone for a console summary. |

For example, to export a short glide without opening windows:

```powershell
.\.venv\Scripts\python.exe -m vtol_dynamics.playgrounds.glide --time 1 --u 12 --v 0 --w 0 --no-show --output .inspection/glide
```

Open `flight.html` in a browser and press Play to replay or scrub the saved
animation. HTML embeds the frames, so long/high-frame-rate exports can be
large; reduce `--fps` or increase `--playback-speed` to reduce the frame count.
CSV contains time, the 14 states, alpha/beta, airspeed, and the energy components
in SI units (angles in radians). Matplotlib is needed for plotting and is
already present in the supplied environment; it is also declared in the
optional installation extra `python -m pip install -e ".[playgrounds]"`.

Simulation sampling and energy calculations live in
`vtol_dynamics/playgrounds/simulation.py`; five-line craft geometry, animation,
and plots live in `plotting.py`. Euler gimbal-lock limitations of the underlying
models still apply; a solver failure reports the affected time interval instead
of presenting a truncated trajectory as a completed simulation.
