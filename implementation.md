# Using the models in another project

All models are imported from `vtol_dynamics`. They use the same
`ModelParameters`, `State`, and `ControlInputs` objects and expose
`step(controls, dt)`, `get_state()`, and `time`.

## Install and prepare your project

Use Python 3.10 or newer. With your consuming project's virtual environment
activated, install this repository by its local path:

```shell
python -m pip install "/path/to/RarogDynamicsV2"
```

For development, use `python -m pip install -e "/path/to/RarogDynamicsV2"`
so changes in this repository are available without reinstalling. Replace the
path with your checkout's actual location. NumPy and SciPy are installed as
dependencies; plotting is not required to use the models.

Copy [examples/sample_parameters.json](examples/sample_parameters.json) into
your project as `aircraft.json`, next to your simulation script:

```text
your_project/
    aircraft.json
    simulate.py
```

The sample contains illustrative parameters. Replace them with your aircraft's
values for meaningful simulations. The JSON is not bundled with the installed
package. All models require a complete valid `ModelParameters` object, even
when some physics are disabled.

## Full model

Put the following in `simulate.py`, then run `python simulate.py`.
`VTOLModel` includes engine and aerodynamic loads, gravity, translation,
rotation, and motor-tilt actuator dynamics.

```python
import math
from pathlib import Path

from vtol_dynamics import ControlInputs, ModelParameters, State, VTOLModel

parameters = ModelParameters.from_json(
    Path(__file__).with_name("aircraft.json")
)
initial_state = State(
    z=-2.0,
    left_motor_tilt=math.pi / 2,
    right_motor_tilt=math.pi / 2,
)
model = VTOLModel(parameters, initial_state, terrain_collision=True)

# Symmetric thrust balancing weight, in revolutions per second per motor.
# This is an illustrative command, not a feedback controller or general trim.
hover_speed = math.sqrt(
    parameters.mass * parameters.gravity
    / (2 * parameters.CT * parameters.air_density
       * parameters.propeller_diameter**4)
)
controls = ControlInputs(
    left_propeller_speed=hover_speed,
    right_propeller_speed=hover_speed,
    left_motor_tilt=math.pi / 2,
    right_motor_tilt=math.pi / 2,
)

dt = 0.01  # Seconds between command updates / saved samples.
history = [(model.time, model.get_state())]
for _ in range(100):
    # For closed-loop use, compute new ControlInputs from model.get_state()
    # here before each step.
    state = model.step(controls, dt)
    history.append((model.time, state))

print(f"Time: {model.time:.2f} s")
print(f"Altitude: {-model.get_state().z:.3f} m")
```

`step` holds commands constant for `dt` seconds, advances simulation time,
stores the result, and returns a `State`. It uses adaptive RK45 internally;
`dt` is the requested simulation interval, not a fixed internal solver step.
There is no real-time waiting. Omitted state and control fields default to zero.

## Simplified models

Choose a model according to the physics your project needs:

| Class | Active physics | Held fixed / ignored |
| --- | --- | --- |
| `BicopterTranslationalModel` | Thrust, gravity, translation, tilt actuators | Attitude fixed; requires initial `p=q=r=0`; ignores engine moments and aerodynamics. |
| `BicopterRotationModel` | Engine moments, rotation, tilt actuators | Position and body linear velocity fixed; ignores gravity and aerodynamics. |
| `BicopterFullModel` | Engine loads, gravity, coupled translation and rotation, tilt actuators | Ignores aerodynamics. |
| `AerodynamicsOnlyModel` | Aerodynamic loads, control surfaces, gravity, translation and rotation | Motor tilts fixed; ignores all engine commands. |

All bicopter variants ignore control-surface deflections. `BicopterFullModel`
is the complete **bicopter** model; use `VTOLModel` to include aerodynamics too.

The following snippets reuse `parameters`, `initial_state`, and `controls`
from the full-model example. Each creates an independent simulation starting
at time zero and supports the same stepping loop.

```python
from vtol_dynamics import BicopterTranslationalModel

translation = BicopterTranslationalModel(
    parameters, initial_state, terrain_collision=True
)
state = translation.step(controls, dt=0.01)
print(state.z, state.u, state.v, state.w)
```

```python
from dataclasses import replace
from vtol_dynamics import BicopterRotationModel

rotation = BicopterRotationModel(parameters, initial_state)
# Apply unequal motor speeds to excite rotation.
rotation_controls = replace(controls, right_propeller_speed=1.05 * hover_speed)
state = rotation.step(rotation_controls, dt=0.01)
print(state.roll, state.pitch, state.yaw)
```

```python
from vtol_dynamics import BicopterFullModel

bicopter = BicopterFullModel(parameters, initial_state, terrain_collision=True)
state = bicopter.step(controls, dt=0.01)
print(state.z, state.roll, state.pitch)
```

```python
from vtol_dynamics import AerodynamicsOnlyModel

glider = AerodynamicsOnlyModel(
    parameters, State(z=-30.0, u=12.0), terrain_collision=True
)
glider_controls = ControlInputs(elevator_deflection=math.radians(2.0))
state = glider.step(glider_controls, dt=0.01)
print(state.z, state.u, state.pitch)
```

Terrain collision defaults to `False`. When enabled, it models frictionless
point contact at `z=0`. It is available for all translating models;
`BicopterRotationModel` rejects `terrain_collision=True`.

## Units and state handling

| Quantity | Convention |
| --- | --- |
| `x, y, z` | North, east, down position in metres; altitude above ground is `-z`. |
| `u, v, w` | Body forward, right, down velocity in m/s. |
| `roll, pitch, yaw` | Euler angles in radians. |
| `p, q, r` | Body angular rates in rad/s. |
| Motor speeds | Nonnegative revolutions/s; convert RPM with `rpm / 60`. |
| Motor tilts | Radians; `0` is forward thrust, `pi/2` is upward thrust. |
| Elevator, aileron, rudder deflections | Radians. |
| `dt`, `model.time` | Seconds of simulated time. |

Tilt commands drive actuators with lag and rate limits. Set both the initial
tilt states and the commands, as above, to start with engines already vertical.

States, controls, and parameters are immutable. Use `dataclasses.replace` to
derive changed values. To reset a simulation, construct a new model:

```python
from dataclasses import replace

new_controls = replace(controls, elevator_deflection=math.radians(1.0))
new_initial_state = replace(initial_state, z=-10.0)
new_model = VTOLModel(parameters, new_initial_state)

vector = new_model.get_state().as_vector()  # A new NumPy array of shape (14,).
restored_state = State.from_vector(vector)
```

Every model uses the same 14-element vector, including **right tilt before
left tilt**, even when some states are held fixed:

```text
[x, y, z, roll, pitch, yaw, u, v, w, p, q, r, right_motor_tilt, left_motor_tilt]
```

Solver options are constructor keywords: `rtol=1e-7`, `atol=1e-9`, and
`max_step=float("inf")` are the defaults. `dt=0` leaves the model unchanged;
negative or nonfinite intervals are rejected. Failed integration preserves the
previous state and time. Euler kinematics are singular near pitch `+/- pi/2`.

See the [README](README.md) for coefficient maps and detailed physics,
[trim guide](docs/trim.md) for equilibrium states and controls, and
[model structure](docs/dynamics_structure.md) for the component layout.
