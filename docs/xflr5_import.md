# Importing XFLR5 plane polars

Run from the project root (Python 3.10+; the converter uses only the standard library):

```powershell
python xflr5_to_parameters.py T1.txt
python xflr5_to_parameters.py T1.txt -o aircraft.json --stall-at-extremes
```

The required positional argument is the plane polar text filename. `-o` / `--output`
sets the output filename; otherwise the input suffix is replaced by `.json`, in
the input directory. Existing output files are replaced, but the input cannot
be used as the output. The script locates `examples/sample_parameters.json`
relative to itself and uses its schema without copying its illustrative values.

Every unavailable parameter is the string `"provide-data"`, including booleans,
environment settings, and coefficient fields. Replace these placeholders before
loading the file with `ModelParameters.from_json`; it is an incomplete template.

## Recovery from T1.txt

The supplied file is an XFLR5 v6.61 Type 1 VLM1 plane polar for
`NACA 4418_v9_9`, with 23 rows: alpha = -11 through +11 degrees in 1-degree
steps, beta = 0 degrees, and speed = 11 m/s.

**Seven of the 41 simulation parameter slots can be recovered**, counting a
coefficient table as one slot and each inertia/stall field separately. Six are
directly available; the seventh, `pitch_alpha`, is a numerical estimate.

| Simulation parameter | Export data / result |
| --- | --- |
| `aerodynamics.lift` | All 23 `CL` samples; 0.461302 at alpha = 0 |
| `aerodynamics.drag` | All 23 total `CD` samples; 0.026527 at alpha = 0 |
| `aerodynamics.sideforce` | All 23 `CY` samples, zero at beta = 0 |
| `aerodynamics.roll_0` | `Cl` at alpha = beta = 0: 0 |
| `aerodynamics.pitch_0` | `Cm` at alpha = beta = 0: -0.020901 |
| `aerodynamics.yaw_0` | `Cn` at alpha = beta = 0: 0 |
| `aerodynamics.pitch_alpha` | Local `dCm/dalpha`: approximately -0.362568 per radian |

`CL` (lift) and `Cl` (rolling moment) are distinct, case-sensitive columns.
`CD` already includes induced and viscous drag; neither `CDi` nor `CDv` is added
again. `Cni` is not the total yaw coefficient `Cn`.

With `--stall-at-extremes`, the two stall thresholds are assumed to be
-11 and +11 degrees, written as **-0.19198621771937624 and
0.19198621771937624 radians**. The flag also sets `stall.enabled` to `true`.
Thus **10/41 slots are filled: seven recovered, two assumed thresholds, and one
explicit configuration choice**. `transition_width` and `drag_max` remain
`"provide-data"`: the endpoints do not establish the post-stall model. Without
the flag all five stall fields remain placeholders. An alpha range that does
not straddle zero cannot supply the simulation's two stall thresholds.

The export does not supply mass, the four inertia entries, reference geometry,
propulsion and tilt-actuator data, density, or gravity magnitude. It also
does not supply rate or control-surface derivatives. In particular, zero `Cl`
and `Cn` at beta = 0 do **not** imply zero `roll_beta` or `yaw_beta`; these remain
placeholders. `QInf` is an operating speed, not a static model parameter, and
`XCP` is a center-of-pressure location, not the mean chord or center of mass.

## Maps and zero-angle selection

Force coefficients retain their alpha/beta dependence because the simulation
accepts coefficient maps. Maps use degrees, rows indexed by beta, columns by
alpha, and `bounds="clamp"`, consistent with the sample. A singleton beta axis
means the simulator will use the beta = 0 result at all sideslip angles; this
export provides no information about behavior at nonzero beta. All recovered
values describe the 11 m/s polar; no speed dependence is inferred.

Moment intercepts use alpha = beta = 0. If zero is not sampled but is bracketed,
the script linearly interpolates to zero on each axis. It never substitutes the
nearest nonzero operating point or extrapolates a scalar baseline. A baseline
outside the sampled range stays `"provide-data"`.

`pitch_alpha` uses the nearest negative and positive alpha samples at beta = 0:

```text
(Cm(+1 deg) - Cm(-1 deg)) / radians(2 deg)
= (-0.027305 - (-0.014649)) / radians(2)
```

Both signs of alpha are required. This secant estimates the local slope; it is
not a fit across the entire polar. The moment model uses
`pitch_0 + pitch_alpha * alpha` with alpha in radians. Keeping the intercept
constant avoids counting alpha dependence twice. The simulation also permits
moment maps, but this importer uses the conventional intercept/slope form for
pitch and zero-angle scalar roll/yaw intercepts.

XFLR5's maintainer describes Type 1 moments as wind-axis moments in the
[axis-convention discussion](https://sourceforge.net/p/xflr5/discussion/679398/thread/9fc00666c9/).
At alpha = beta = 0 the wind and simulation body axes coincide; the pitch
component is also unchanged by alpha rotation at beta = 0. This importer does
not copy nonzero-angle roll/yaw moments into body-axis maps or infer beta
derivatives from them. Reference area, span, chord, and moment reference point
must match the XFLR5 model when completing the template.

## Supported input and validation

The parser handles whitespace-separated plane polar exports with `alpha` and
`Beta` headers, optional metadata before the header, blank lines, reordered
columns and rows, and UTF-8 BOMs. Missing coefficient columns leave their
parameters as placeholders. It requires a complete rectangular alpha/beta
grid and rejects malformed/nonfinite rows, duplicate operating points, and
varying `QInf` rather than silently combining incompatible data. It does not
parse Type 7 stability logs or two-dimensional airfoil polars.

Run the importer and simulation regression tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
