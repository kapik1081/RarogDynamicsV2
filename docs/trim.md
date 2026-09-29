The hover and forward trim scripts implement chapter 2 of the updated
`VTOL_dynamics.pdf` and print results to the console. For the section 2.3
surface search and hover-to-cruise plot, see [transition search](transition.md).
The scripts use the same coefficient interpolation,
stall blend, thrust, inertia, gravity, and control-effectiveness functions as
the simulation. No simulation state or parameter file is modified.

```powershell
.\.venv\Scripts\python.exe find_hover_trim.py examples/sample_parameters.json
.\.venv\Scripts\python.exe find_forward_trim.py examples/sample_parameters.json --engine-tilt 0 --airspeed 12 --max-elevator-deflection 25 --max-engine-rpm 8000
```

The craft JSON path is a **positional argument**. Forward-flight tilt is in
**degrees** and airspeed in **m/s**. Optional limits are the maximum **absolute
elevator deflection in degrees** and maximum **RPM of each engine**. Omitted
limits are unrestricted; this does not imply unlimited physical actuators.
The scripts print engine speed in both rev/s (the simulator's input units) and
RPM, along with total and per-engine thrust, attitude, velocities, surface
deflections, and complete `State` / `ControlInputs` values in SI units.

Hover is the closed-form symmetric equilibrium:

```text
u = v = w = p = q = r = roll = pitch = 0
left tilt = right tilt = pi/2
T_left = T_right = m*g/2
N_left = N_right = sqrt(m*g / (2*CT*rho*D_prop^4))   [rev/s]
```

Actual nacelle states and tilt commands are identical, so the actuators are
already at equilibrium. Hover surfaces are zero because freestream
aerodynamic forces and moments vanish at zero airspeed. The computed state
is checked against the complete model's derivative.

For forward flight, the requested airspeed and common engine tilt are fixed.
The symmetric state has `v=beta=roll=p=q=r=0`, `pitch=alpha`,
`u=Va*cos(alpha)`, and `w=Va*sin(alpha)`. The search proceeds as follows:

1. Determine the **full evaluable aerodynamic alpha range**. The simulation's
   `atan2` range is `[-pi, pi]`. Scalar coefficients and `bounds="clamp"`
   tables are defined over that entire range, including post-stall behavior.
   `bounds="raise"` maps restrict the range to their common intersection.
   Strict maps must admit beta zero. The search does not stop at stall.
2. Sample 4,001 uniformly spaced angles and add the coefficient-table alpha
   knots. `--grid-points` changes the uniform-grid resolution.
3. Record zeros at grid points and every adjacent sign change.
4. Apply **Brent's method independently to every sign-changing interval** and
   deduplicate roots. No Newton method or single initial guess is used.
5. At each root, recover total thrust `P`, common propeller speed `N`, and
   elevator deflection. Evaluate moment-derivative maps at the root's alpha
   and beta zero. When needed, solve the PDF's two lateral equations for
   aileron and rudder to cancel roll/yaw trim moments.
6. Reject infeasible candidates and print their rejection reasons. This
   includes negative thrust/drag, singular thrust projection, Euler gimbal
   lock, insufficient control effectiveness, nonfinite results, user limits,
   and failure of the full-model steady-state residual check.
7. Prefer candidates strictly between the configured negative and positive
   stall angles. Select the **smallest absolute alpha** in that group, using
   signed alpha as a deterministic tie-breaker. If none are feasible, select
   the smallest absolute alpha among feasible post-stall candidates and label
   the selection as a post-stall fallback.

The root residual is the tangent-free version in Eq. 2.14:

```text
F(alpha) = (L(alpha)-m*g)*cos(engine_tilt+alpha)
           + D(alpha)*sin(engine_tilt+alpha)
P = D(alpha) / cos(engine_tilt+alpha)
N = sqrt(P / (2*CT*rho*D_prop^4))
elevator = -(Cm0 + Cm_alpha*alpha
             + E_height*P*cos(engine_tilt)/(q_inf*S*c)) / Cm_elevator
```

Using this residual avoids treating the tangent poles of Eq. 2.13 as sign
changes. Multiplying by cosine can still introduce degenerate roots; candidates
with `abs(cos(engine_tilt+alpha)) < 1e-8` are rejected before dividing for thrust.
Negative thrust has no real nonnegative motor-speed solution and is printed
with speed/elevator `n/a`.

Two signs in the PDF's trim chapter conflict with its earlier dynamics and
the equilibrium equations: Eq. 2.8 prints a positive `L*cos(alpha)` in body-z,
where Eq. 1.27 requires **negative** `L*cos(alpha)`; Eq. 2.17 omits the leading
**minus** required when solving its preceding pitch-moment equation for the
elevator. The code uses the consistent signs above and verifies the resulting
forces/moments through the existing full dynamics. No plant equations are
changed by the trim implementation.

Additional aerodynamic validity limits can be prescribed independently of the
search range:

```powershell
.\.venv\Scripts\python.exe find_forward_trim.py examples/sample_parameters.json --engine-tilt 0 --airspeed 12 --alpha-min -10 --alpha-max 15 --max-elevator-deflection 25 --max-engine-rpm 8000 --grid-points 8001
```

`--alpha-min` and `--alpha-max` are inclusive limits in degrees. They **filter
candidates after the full-range scan**, so roots outside them still appear in
the report as rejected. Use these bounds to restrict the admissible branch or
to disallow post-stall fallback. Clamping a coefficient map is a numerical
extension, not evidence that coefficients are physically valid outside the
measured grid; supply validity bounds when the calibrated domain is narrower.
There are no additional aileron/rudder travel limits in this interface.

The lateral solve uses a minimum-norm linear solution, allowing a consistent
rank-deficient system (including zero trim moments with no lateral control
effectiveness). Inconsistent lateral moment equations are rejected. If elevator
effectiveness is zero and pitch is already balanced, the chosen elevator is
zero; otherwise the candidate is rejected. A nonzero sideforce at beta zero
also fails the complete straight-flight equilibrium check in this model.

For every accepted candidate, the expected state derivative is
`[Va, 0, 0, ..., 0]`: motion north with fixed altitude, attitude, body velocities,
rates, and nacelle tilts. All derivative components must agree within `1e-6`
in their respective SI units. The origin and yaw are arbitrary reference values
set to zero. Set another starting position when using the returned trim in a
simulation; ensure its actual motor tilts and commands match.

The default sample at 12 m/s with zero engine tilt produces four candidate
roots, including three feasible branches when using the example limits.
The selected pre-stall solution is approximately alpha `7.30425 deg`,
`P=5.92177 N`, each motor `1843.63 RPM`, and elevator `-4.93597 deg`.
Hover requires about `5305.10 RPM` per motor for these illustrative parameters.

Exit status is 0 for a selected trim, 1 when the scan yields no feasible trim,
and 2 for invalid inputs, unreadable parameters, or a failed search. A finite
grid cannot guarantee detection of arbitrarily close roots or tangential
(even-multiplicity) roots that do not land on a grid point. Increase
`--grid-points` and compare candidate sets when resolving delicate branches.

The numerical functions are available independently of the console scripts:

```python
import numpy as np
from vtol_dynamics import ModelParameters, VTOLModel
from vtol_dynamics.trim import hover_trim, forward_flight_trim

params = ModelParameters.from_json("examples/sample_parameters.json")
hover = hover_trim(params)
result = forward_flight_trim(
    params, engine_tilt=0.0, airspeed=12.0,
    max_elevator_deflection=np.deg2rad(25), max_engine_rpm=8000,
)
if result.selected is not None:
    model = VTOLModel(params, result.selected.state)
    model.step(result.selected.controls, 0.1)
```

Python angles and angular limits are in radians; `max_engine_rpm` remains RPM.
`ForwardTrimResult` retains all candidates and their reasons for rejection as
well as the selected solution. `forward_residual`, `longitudinal_forces`,
`aerodynamic_alpha_range`, and `bracketed_roots` are pure functions for separate
inspection and testing.
