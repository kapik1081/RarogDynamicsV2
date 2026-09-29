The transition search implements section 2.3 of the updated `VTOL_dynamics.pdf`.
It returns a sampled admissible trim surface and searches for a continuous
steady-trim schedule from upright hover to zero-tilt cruise. The plot uses
airspeed on the horizontal axis and common engine tilt on the vertical axis,
placing hover at the upper left and forward flight at the lower right.
Color shows pitch, open circles mark sampled coordinates with multiple viable
branches, and a red curve shows the proposed path when one is found.

```powershell
.\.venv\Scripts\python.exe find_transition_trim.py examples/sample_parameters.json --max-engine-rpm 8000 --max-elevator-deflection 25 --cruise-airspeed 12 --o transition.png
```

The JSON file is a positional argument using the existing `ModelParameters`
schema. It must contain numerical craft parameters, including the signed
body-z `engine_height`: engines above the center of mass have negative height.
`T1.json` currently contains `"provide-data"` placeholders for its static
parameters; fill those in before using it for a search. The sample parameters
in the command above are fully populated and illustrative.

The script displays the plot by default. Optional `--o FILENAME` saves it before
display; PNG, PDF, SVG and other Matplotlib formats follow the filename suffix.
A filename without a suffix receives PNG content without changing its name.
Use `--no-show` for headless exports and `--quiet` to hide progress reports.
Plotting uses the existing optional dependency:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[playgrounds]"
```

Required operational arguments are `--max-engine-rpm`,
`--max-elevator-deflection`, and `--cruise-airspeed`. All command-line angles
are **degrees**, speeds are **m/s**, RPM applies **per engine**, and shaft power
is the **combined mechanical power of both engines in watts**.

| Argument | Meaning / default |
| --- | --- |
| `--max-engine-rpm RPM` | Required sustainable loaded speed limit per engine. |
| `--max-elevator-deflection DEG` | Required symmetric absolute elevator travel limit. |
| `--cruise-airspeed V` or `--cruise-airspeed MIN MAX` | Required final speed or inclusive cruise interval, at zero tilt. |
| `--airspeed-max V` | Maximum search speed; defaults to the upper cruise speed and must include it. Search starts at zero. |
| `--tilt-min DEG`, `--tilt-max DEG` | Nacelle travel limits; default 0 to 90. |
| `--pitch-min DEG`, `--pitch-max DEG` | Attitude limits, including hover; default -30 to 30, strictly inside the Euler singularities at +/-90. |
| `--alpha-min DEG`, `--alpha-max DEG` | Additional positive-speed aerodynamic validity bounds; default common evaluable map domain intersected with the pitch bounds. |
| `--max-aileron-deflection DEG`, `--max-rudder-deflection DEG` | Optional symmetric lateral control travel limits; omitted means unrestricted. |
| `--max-shaft-power W` | Optional total shaft-power ceiling, Eq. 2.35. Electrical power must first be converted using efficiency. |
| `--monotone-airspeed` | Require nondecreasing airspeed along checked path connections. |
| `--monotone-tilt` | Require nonincreasing engine tilt along checked path connections. |
| `--speed-points N` | 31 by default; quadratic spacing concentrates samples near hover. Cruise interval endpoints are added exactly. |
| `--tilt-points N` | 31 by default; upright-hover and cruise tilts are added if within the travel limits. |
| `--pitch-points N` | 241 by default, plus aerodynamic table knots. |
| `--elevator-points N` | 9 by default, including both travel limits, for the regular hover chart. |

Every sampling count must be at least 3. An interval with equal tilt or pitch
limits is supported. Zero actuator ceilings are accepted and can produce an
empty feasible set. Excluding upright hover or zero-tilt cruise with the travel
limits still produces the requested surface, but no endpoint-to-endpoint path.

For example, restrict the aerodynamic domain and add power and path constraints:

```powershell
.\.venv\Scripts\python.exe find_transition_trim.py examples/sample_parameters.json --max-engine-rpm 8000 --max-elevator-deflection 25 --cruise-airspeed 12 18 --alpha-min -10 --alpha-max 15 --max-aileron-deflection 20 --max-rudder-deflection 20 --max-shaft-power 1500 --monotone-airspeed --monotone-tilt --no-show --o corridor.pdf
```

The implementation retains the implicit horizontal-force, vertical-force, and
pitch-moment balances of Eq. 2.27, normalized by weight and weight times chord
as in Eq. 2.57. Zero-speed aerodynamic forces and moments are exactly zero;
there is no artificial airspeed floor. Thrust reconstruction uses the regular
projection in Eq. 2.29, so vertical thrust does not cause division by zero.
Elevator feasibility is checked through the moment inequality in Eq. 2.34
before solving for deflection. Zero elevator effectiveness is accepted only
when the remaining pitch moment is balanced.

Surface construction uses overlapping charts:

1. Sample the hover elevator boundary and continue the regular
   **airspeed–elevator** chart using a scaled Jacobian predictor and nonlinear
   corrector. This resolves the `pi/2 - tilt = O(airspeed**2)` neck that a uniform
   tilt grid can miss. Hover elevator settings are retained separately so the
   path can begin at a setting consistent with its outgoing branch.
2. At each speed–tilt pair, scan the accepted pitch domain using Eq. 2.28.
   Brent solves capture sign-changing roots; local minimization of squared
   residual captures sampled tangencies. Include coefficient-table knots and
   retain all admissible roots, including post-stall branches if allowed.
3. Supplement with a **speed–pitch** force-direction chart: required thrust is
   the vector `(D, mg-L)`, and its direction determines tilt. This resolves
   branches between tilt samples and either side of projection folds.
   Bisect detected feasibility boundaries five times to refine actuator edges.
4. Near singular tilt projections, trace fixed-speed slices in both directions
   using scaled SVD tangents and pseudo-arclength correction, Eqs. 2.58–2.59.
   The step decreases when a correction fails. Near-hover degeneracy in
   elevator authority is handled by the hover chart rather than mistaken for
   a tilt fold.
5. Check travel, thrust, power and validity inequalities without clipping
   controls. Solve nonzero constant roll/yaw moments, including a bounded
   lateral solve when necessary. Verify every retained state/control pair in
   the original 14-state plant against `[Va, 0, ..., 0]` within `1e-6` SI units.

Map clamping remains the plant's numerical extension; it does not demonstrate
physical validity outside measured data. The optional alpha bounds constrain
all positive-speed trim calculations. At hover, pitch limits still apply,
but no alpha constraint is treated as a freestream measurement. With zero
engine height the hover boundary has additional rank degeneracy; the code
retains its extra equilibria and attempts positive-speed corrections, while
keeping upright hover as the path start.

The path graph stores all five reduced coordinates plus actual lateral
controls. Neighbours are chosen in this full scaled space. An edge is accepted
only after predictor–corrector continuation between its endpoints: interpolate
independent chart coordinates and recover the dependent variables by solving
equilibrium. Check intermediate actuator limits, full-model residuals,
monotonicity when requested, and the endpoint's branch identity. Adaptive
subdivision and bounded correction distances reject branch jumps. A projected
intersection alone never authorizes an edge. Dijkstra search uses coordinate
length, squared changes in scaled engine speed/elevator/tilt, and a penalty
for low thrust/elevator/power reserve, following Eq. 2.62. The returned path
includes the corrected intermediate samples; it is not an unconstrained spline.

The plot deliberately displays **sampled feasible support**, rather than
filling a convex hull that could hide infeasible holes. Different branches can
overlap in this projection; the Python result retains their full settings.
Unresolved hover-chart or Jacobian calculations are reported separately and
marked with gray crosses where their parameter coordinates are in the plot.

A finite root scan and local graph are not proofs of global completeness.
Increase the four sampling counts and compare results near narrow corridors,
folds and closely spaced roots. Failed local solves and a missing graph route
do not establish physical nonexistence. Exit status is **0** if a path is found,
**1** if no path is found (the plot is still displayed/saved), and **2** for
invalid arguments, parameters, or output errors.

This is a **steady-trim schedule**, as in sections 2.3.7–2.3.8. It has no time
law. The optional monotonicity flags constrain its geometry. Controller
stabilizability, two-sided control authority, actuator rates, feedforward tilt
commands, and the dynamically constrained accelerating trajectory discussed
in section 2.3.9 require a subsequent controller/trajectory calculation.
Those are not certified by the existence of the trim corridor.

The numerical API is independent of Matplotlib:

```python
import numpy as np
from vtol_dynamics import ModelParameters
from vtol_dynamics.transition import TransitionConstraints, transition_trim_manifold
from vtol_dynamics.transition_plotting import plot_transition

params = ModelParameters.from_json("examples/sample_parameters.json")
limits = TransitionConstraints(
    max_engine_rpm=8000,
    max_elevator_deflection=np.deg2rad(25),
    alpha_min=np.deg2rad(-10), alpha_max=np.deg2rad(15),
)
result = transition_trim_manifold(params, limits, cruise_airspeed=(12, 18))
fig = plot_transition(result)
fig.savefig("transition.png")
for point in result.path:
    state, controls = point.solution.state, point.solution.controls
    print(point.coordinates, point.shaft_power, point.reserve)
```

Python angles are **radians**. `TransitionResult.points` contains all retained
trim samples; `path` contains a proposed route or an empty tuple. `branch_counts`
records accepted roots on the regular speed–tilt grid. `unresolved` records
failed local surface calculations. `coordinates` are ordered as
`(airspeed, engine_tilt, pitch, total_thrust, elevator)`.
