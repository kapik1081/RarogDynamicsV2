# Test catalog

This catalog describes all 95 test methods in the project's seven `tests/test_*.py` files. It includes unit, integration, rendering, and command-line regression tests. Each entry names the method and describes its actual assertions; loops and `subTest` cases are included in the description rather than counted separately. The catalog reflects the current working tree, including uncommitted tests.

The suite uses Python's built-in `unittest`, with NumPy assertions and SciPy numerical routines. Matplotlib is required for the plotting tests and is supplied by the `playgrounds` optional dependency. From the repository root, with dependencies installed:

```powershell
python -m unittest discover -s tests -v
```

To run one file or one method:

```powershell
python -m unittest discover -s tests -p test_model.py -v
python -m unittest discover -s tests -k test_hover -v
```

The second command selects all method names containing `test_hover`. Most fixtures load [examples/sample_parameters.json](examples/sample_parameters.json); polar-import tests also read [T1.txt](T1.txt). Temporary files hold generated JSON and plot exports. Plotting tests use the noninteractive Agg backend, while tests of GUI display calls mock `show`. `vtol_dynamics/testing.py` and `vtol_dynamics/testing_dynamics.py` implement reduced simulation models tested below; they are not additional test suites.

| Source | Classes | Test methods |
| --- | --- | ---: |
| [test_model.py](tests/test_model.py) | MapTests, PhysicsTests, IntegrationTests, ParameterTests | 33 |
| [test_testing_models.py](tests/test_testing_models.py) | TestingModelTests | 14 |
| [test_playgrounds.py](tests/test_playgrounds.py) | PlaygroundTests | 11 |
| [test_full_model_playground.py](tests/test_full_model_playground.py) | FullModelPlaygroundTests | 4 |
| [test_trim.py](tests/test_trim.py) | TrimTests | 12 |
| [test_transition.py](tests/test_transition.py) | TransitionTests | 13 |
| [test_xflr5.py](tests/test_xflr5.py) | PolarImportTests | 8 |
| **Total** | **10 classes** | **95** |

## tests/test_model.py

The local `parameters()` helper creates a small aircraft with explicit inertia, propulsion, and actuator values and stall blending disabled. Individual tests replace parameters as needed.

### MapTests — 6 tests

- `test_user_grid_orientation_and_bilinear_interpolation`: Evaluates every point of a degree-based coefficient grid to verify beta rows and alpha columns, then checks two interior bilinear interpolations against -0.075 and 1.15.
- `test_nonuniform_axes_and_units`: Uses nonuniform axes in the default radian units and a linear field `3*alpha + 2*beta`; interpolation at `(2, 1)` must produce 8.
- `test_boundaries`: Checks that the default policy clamps out-of-range angles to an edge value and that `bounds="raise"` rejects an out-of-range query.
- `test_singleton_axes`: Checks a constant one-cell map and interpolation along beta when alpha has only one sample.
- `test_invalid_maps`: Rejects duplicate alpha coordinates, descending beta coordinates, wrong value dimensions, empty axes, unknown bounds policies or angle units, nonfinite axes, and a NaN query.
- `test_copies_source_data`: Mutates the original nested value list after construction and verifies that the coefficient map retains its copied value.

### PhysicsTests — 10 tests

- `test_frame_rotations`: Checks body-to-NED rotation orthogonality, a 90-degree yaw mapping body x to earth y, and reconstruction of a velocity vector from wind-axis airspeed and angles.
- `test_gravity_and_euler_rates`: Compares body gravity with rotated earth gravity, checks direction and magnitude at several attitudes including inverted roll, verifies Euler rates at level attitude, and rejects the pitch singularity at pi/2.
- `test_air_data_and_stall`: Checks speed, dynamic pressure, and angle of attack for velocity `[3, 0, 4]`; verifies negligible stall blending at zero angle, full blending with zero lift and maximum drag at ±pi/2, and blend weights bounded by zero and one across ±pi.
- `test_aerodynamic_force_components`: Independently expands lift, drag, and sideforce into body-axis components for nonzero alpha and beta and compares the calculated force.
- `test_moment_matches_uncancelled_stability_equations`: Reconstructs roll, pitch, and yaw moments from static terms, angle derivatives, normalized angular rates, and all control-surface contributions, including pressure and geometry scaling.
- `test_maps_are_used_for_forces_and_derivatives`: Supplies a coefficient map for lift and aileron effectiveness and checks that both force and moment calculations evaluate it at the current alpha and beta.
- `test_zero_and_near_zero_airspeed`: At speeds zero and 1e-12, checks finite aerodynamic moments and negligible force and moment magnitudes despite nonzero body rates.
- `test_propeller_geometry_and_reaction_signs`: Uses unequal propeller speeds and tilts to check summed thrust, moments from engine offsets, and the signs of counter-rotating propeller reaction moments.
- `test_rigid_body_equations_and_coupling`: Substitutes computed angular accelerations into the coupled inertia equations, checks translational rate coupling, and verifies zero instantaneous rotational-energy change under torque-free motion.
- `test_actuator_rate_limit`: Checks positive and negative tilt-rate saturation and the unsaturated proportional response near the target.

### IntegrationTests — 14 tests

- `test_free_fall_and_dt_zero`: Checks the zero-duration step result, then compares horizontal position, vertical position, vertical speed, and model time after 0.5 seconds with analytic free fall.
- `test_hover`: Computes equal propeller speeds that balance weight with vertical motors and verifies that all 14 state components remain at the initial hover state after two seconds.
- `test_actuator_analytic_response_and_state_order`: Compares both motor tilts with the first-order exponential response and checks that vector index 12 stores the right motor tilt.
- `test_rate_limited_actuator_then_exponential`: Checks a one-radian-per-second linear tilt response followed by the analytic exponential response once the actuator leaves saturation.
- `test_step_subdivision_convergence`: Compares one 0.2-second integration with twenty 0.01-second steps under identical controls and a nontrivial flight state, using `rtol=2e-6` and `atol=1e-8`.
- `test_terrain_impact_and_rest`: Drops a translating aircraft onto the ground, checks preserved horizontal travel and zero ground penetration/vertical speed, then verifies continued ground contact.
- `test_terrain_projection_at_tilted_attitude`: Projects a penetrated, tilted state onto terrain; earth-horizontal velocity must be preserved, earth-vertical velocity removed, and the original state left unchanged.
- `test_contact_derivative_cancels_earth_normal_acceleration`: Applies the contact constraint to a rotating state's derivative and verifies zero earth-normal acceleration and vertical position derivative.
- `test_rotated_ground_contact_with_exact_gravity`: Checks that a stationary aircraft at a nonlevel attitude stays completely stationary on terrain under gravity for two seconds.
- `test_terrain_toggle_and_upward_velocity`: Checks penetration when collision is disabled, immediate position/velocity projection when enabled, renewed falling when disabled again, and permitted upward departure followed by landing with collision enabled.
- `test_takeoff_and_liftoff_during_same_step`: Uses thrust exceeding weight and checks upward departure within one integration step both from initially horizontal motors and from already vertical motors.
- `test_impact_then_takeoff_in_same_step`: Starts just above the ground with downward velocity and verifies that a one-second powered step ends airborne after contact and motor tilting.
- `test_zero_force_ground_equilibrium`: With gravity disabled and no controls, verifies that the terrain-constrained state remains the all-zero vector.
- `test_state_is_immutable_and_failed_step_is_atomic`: Rejects mutation of the frozen state, verifies that modifying an exported array does not change model state, checks unchanged time after a singular-attitude integration failure, and rejects negative, NaN, and infinite durations.

### ParameterTests — 3 tests

- `test_load_sample_json`: Loads the sample parameter file and checks mass 5, lift coefficient 0.2 at zero angles, and a map-valued aileron derivative.
- `test_invalid_parameters`: Rejects zero mass or tilt time constant, negative thrust coefficient or gravity, nonfinite engine height, invalid coupled inertia, negative propeller speed, a nonfinite state component, and zero stall transition width.
- `test_unknown_json_fields_fail`: Adds an unknown top-level field to a temporary parameter JSON file and checks that loading raises `TypeError`.

## tests/test_testing_models.py

### TestingModelTests — 14 tests

The fixture loads the sample parameters and groups the three engine-only models and the aerodynamics-only model for shared checks.

- `test_translation_matches_constant_acceleration_at_fixed_attitude`: Compares thrust-plus-gravity motion with analytic constant-acceleration position and velocity at a tilted attitude; attitude stays fixed and angular rates stay zero.
- `test_translation_rejects_inconsistent_angular_rates`: For each of p, q, and r, checks that both the translation-only constructor and derivative reject nonzero angular rates with a fixed-attitude error.
- `test_tilted_freefall_stays_vertical_in_earth_frame_for_all_translating_models`: With aerodynamic loads disabled, checks analytic earth-frame ballistic position and velocity and conservation of translational, rotational, and gravitational energy for the full model and all three translating reduced models. Models supporting rotation start with nonzero body rates.
- `test_rotation_matches_constant_pitch_acceleration`: Uses symmetric thrust and diagonal inertia to check analytic pitch and pitch-rate evolution while position and translational velocity remain frozen.
- `test_engine_models_never_evaluate_aerodynamics_or_surface_commands`: Gives all engine-only models an aerodynamic map that would reject the flight angles; integration must succeed and produce identical states with and without surface deflections.
- `test_full_bicopter_matches_complete_model_without_aerodynamic_loads`: With aerodynamics disabled, compares the bicopter and full-model derivatives and their integrated states after 0.2 seconds.
- `test_glider_matches_complete_model_with_engines_off`: Compares aerodynamics-only and full-model derivatives and trajectories with zero propeller speeds and motor commands matching initial tilts.
- `test_glider_never_evaluates_engines_and_ignores_all_engine_commands`: Makes thrust, reaction-moment, and tilt-rate helpers raise if called; engine commands must leave the glider trajectory unchanged, and initial motor tilts must remain fixed.
- `test_glider_control_surfaces_are_active`: Independently deflects aileron, elevator, and rudder and checks a measurable change in the associated roll, pitch, or yaw rate.
- `test_nacelle_dynamics_remain_active_in_engine_models`: Checks both motor tilts against the exponential actuator response in each engine-only model with gravity disabled and a high rate limit.
- `test_terrain_available_for_translating_models`: Drops each translating reduced model onto terrain and checks zero final vertical position and velocity.
- `test_rotation_cannot_enable_terrain`: Checks rejection of terrain collision both at rotation-only model construction and through its setter; the rejected setter leaves the flag disabled and state unchanged.
- `test_models_keep_separate_state_and_share_step_validation`: For all four reduced models, checks zero-duration stepping, independent instance state/time, normal time advancement, and preservation of state/time after a rejected negative-duration step.
- `test_rotation_derivative_does_not_evaluate_translational_gravity`: Makes the gravity helper raise if called and verifies that a rotation-only derivative still computes with zero position and translational-velocity derivatives.

## tests/test_playgrounds.py

### PlaygroundTests — 11 tests

These tests load the sample aircraft and exercise reduced-model scenarios, flight statistics, and Matplotlib displays. Figures are closed after rendering checks.

- `test_translation_zero_initial_state_constant_commands_and_partial_step`: Runs for 0.105 seconds at a 0.04-second sampling interval; checks the shortened final step, zero initial state, equal constant engine commands, and equal motor tilts moving toward the target without reaching it instantly.
- `test_rotation_command_sides_and_frozen_translation`: Checks correct left/right assignment of unequal engine commands, zero initial state, frozen position/velocity, and nonzero final angular motion.
- `test_glide_initial_velocity_only_and_zero_controls`: Checks the specified initial velocity and otherwise zero initial state, zero propeller/elevator commands, and initial airspeed, alpha, and beta statistics against analytic values.
- `test_freefall_energy_and_negative_potential_below_zero`: Checks the free-fall trajectory, negative gravitational potential below the NED origin, and zero total mechanical energy throughout an unpowered translation run.
- `test_kinetic_energy_includes_rotational_inertia_coupling`: Computes rotational kinetic energy including the Ixz cross term and compares statistics; rotational energy must be positive and translational energy zero.
- `test_glide_no_aero_matches_ballistic_motion`: Disables aerodynamic forces and stall blending and compares final glide position with the analytic three-dimensional ballistic trajectory.
- `test_positive_axes_and_engine_tilts_in_display_coordinates`: Checks the NED-to-display vertical sign, body-axis directions, left/right engine offsets, horizontal versus vertical motor directions, and rotation under 90-degree yaw.
- `test_animation_includes_both_endpoints`: Checks frame sampling includes the first and final samples with the expected count, including a very short two-sample history.
- `test_craft_stays_visible_and_in_bounds_on_long_paths`: Tests stationary, 1,000-unit, and million-unit paths with automatic or explicit axis length. Checks craft visibility scaling, constant size along each path, preserved directions and relative engine spacing, bounds containing all craft segments, the displayed trajectory, and unchanged source states.
- `test_headless_figures_and_html_animation_render`: Draws both figures and exports JavaScript HTML animation; checks embedded PNG frames, six statistics axes, and six flight lines representing the trail and five craft segments.
- `test_invalid_duration_and_sampling`: Rejects zero, negative, or infinite duration and zero or NaN sample intervals.

## tests/test_full_model_playground.py

### FullModelPlaygroundTests — 4 tests

The CLI helper suppresses output and supplies `--no-show`. The empty-argument test mocks execution to avoid opening a GUI while retaining default scenario construction.

- `test_no_arguments_defaults_to_zero_state_controls_and_one_second`: Calls the CLI with genuinely empty arguments and checks a zero 14-component initial state, zero controls, one-second duration, and the same state history as the Python API defaults.
- `test_entire_initial_state_and_constant_controls`: Supplies every initial-state and control flag, checks degrees-to-radians conversion for angles and angular rates and unchanged numeric propeller speeds, then compares each sampled state with direct `VTOLModel` stepping under those controls.
- `test_partial_arguments_keep_other_components_zero_and_tilts_independent`: Uses the `--simulation-time` alias and only a few state/control flags; checks zero defaults elsewhere and independent left/right tilt responses from different initial and commanded values.
- `test_invalid_cli_values_and_removed_instant_tilt_are_rejected`: Checks argparse exit status 2 for the removed `--instant-tilt` option, zero time or interval, NaN/infinity for every state and control field, and negative speed for either propeller.

## tests/test_trim.py

### TrimTests — 12 tests

These tests use the sample parameters to check hover and forward-flight equilibrium, root enumeration, constraint handling, and both trim CLIs.

- `test_hover_closed_form_and_stationary_trajectory`: Checks equal propeller speeds against the closed-form weight balance, vertical motor tilt with matching commands, and an unchanged full-model state after one second.
- `test_brent_finds_all_sign_changes_and_exact_grid_roots`: Checks recovery of four polynomial roots, including a root exactly on the grid, a separate sequence of exact grid roots, and rejection of a NaN residual.
- `test_full_post_stall_scan_keeps_all_candidates_and_prefers_low_alpha`: Checks a full ±pi search, at least four candidates and three feasible solutions, selection of the pre-stall root near 7.30424617 degrees, retention of post-stall solutions and negative-value rejection reasons, and near-zero residual for every feasible root.
- `test_pitch_sign_force_equilibrium_and_level_trajectory`: At 20-degree engine tilt and 12 m/s, checks zero total force/moment, longitudinal drag/lift/thrust balance, and level travel of six metres in 0.5 seconds with the other state components unchanged.
- `test_rpm_and_absolute_elevator_limits_reject_after_scanning`: Applies restrictive engine/elevator limits, verifies unchanged candidate angles but no selected solution, and checks both rejection reasons for candidates with calculated elevator deflection.
- `test_validity_bounds_do_not_shrink_search_and_allow_fallback`: Restricts acceptable alpha to 10–19 degrees while retaining the original scan range and candidate roots; checks selection of the post-stall fallback near 16.9970791 degrees.
- `test_strict_map_range_and_beta_validity`: Checks that strict coefficient maps restrict the alpha search to ±15 degrees and still permit a trim, then rejects a strict sideforce map whose beta range excludes zero.
- `test_singular_projection_is_rejected_instead_of_false_trim`: With aerodynamics and stall disabled, checks that candidate roots exist but all are rejected for singular thrust projection and no trim is selected.
- `test_zero_elevator_effectiveness_and_balanced_special_case`: Checks rejection when elevator effectiveness is zero and pitch cannot balance, then verifies a zero-elevator solution when engine height and aerodynamic pitching terms are also zero.
- `test_nonzero_lateral_trim_moments_and_unbalanced_sideforce`: Adds static roll/yaw moments and checks nonzero corrective aileron/rudder and balanced total moments; adding sideforce must then prevent selection with a lateral-dynamics rejection reason.
- `test_invalid_trim_inputs`: Rejects hover with zero thrust coefficient and forward trims with zero/NaN speed, negative RPM/elevator limits, too few grid points, or reversed alpha bounds.
- `test_console_scripts_and_no_feasible_exit_status`: Checks successful hover/forward CLI status and RPM/pre-stall output, status 1 and a no-feasible-trim message under an impossible RPM limit, and argparse status 2 for zero airspeed.

## tests/test_transition.py

### TransitionTests — 13 tests

The class fixture builds a shared transition manifold with an 8,000 RPM limit, 25-degree elevator limit, 12 m/s cruise target, and a 13 × 13 speed/tilt grid with 81 pitch and five elevator samples.

- `test_surface_path_and_full_model_equilibria`: Checks a connected hover-to-cruise path, multiple trim branches, correct endpoint speed/tilt/pitch conditions, near-zero trim residuals and full-model acceleration for every stored/path point, RPM/elevator limits, and bounded scaled spacing between path samples.
- `test_tangential_roots_and_close_sign_changes`: Checks the three roots of a polynomial containing a double (tangential) root at 0.12345 and simple roots at -0.41 and 0.7; rejects a NaN residual.
- `test_hover_zero_speed_limits_and_quadratic_neck`: Accepts a zero-speed hover point with nonzero elevator, corrects nearby points at 0.01 and 0.02 m/s, and compares tilt departure divided by speed squared with the analytic coefficient. Also verifies that zero-speed residual evaluation never calls longitudinal aerodynamics.
- `test_vertical_thrust_projection_is_regular`: With zero engine height and aerodynamic loads, reconstructs a 4 m/s point with vertical thrust and checks thrust equal to weight and zero residual despite the vertical projection.
- `test_rpm_power_attitude_and_control_limits`: Separately tightens RPM, shaft power, elevator, tilt, pitch, and alpha constraints to reject the cruise point; also checks that changing elevator after solving causes residual-based rejection.
- `test_lateral_limits_and_sideforce_rejection`: Adds a static roll moment and checks corrective aileron, rejects the point under zero aileron/rudder limits, and rejects an added unbalanced sideforce.
- `test_same_projection_does_not_join_different_branches`: Finds distinct cruise solutions sharing speed and tilt and checks that the connector does not directly join their different branches.
- `test_edge_interior_constraints_not_just_endpoints`: First accepts a connection between two path points, then injects an infeasible strip inside the edge and verifies rejection despite unchanged feasible endpoints.
- `test_monotonicity_constraints`: Enables monotone speed and tilt requirements; a forward path connection must succeed and its reverse must fail.
- `test_pseudo_arclength_traverses_tilt_fold`: Locates a fold where tilt's pitch derivative vanishes, seeds continuation there, and checks more than ten continued points spanning both sides of the fold with residuals within 1e-8.
- `test_no_hover_capacity_means_no_path_but_keeps_forward_trims`: Reduces engine capacity to 3,000 RPM and checks retained positive-speed trims but no hover-to-cruise path.
- `test_invalid_inputs`: Rejects negative RPM, infinite power, invalid tilt/pitch/alpha bounds, a reversed cruise-speed interval, an airspeed maximum below the cruise target, and insufficient pitch sampling.
- `test_cli_plot_export_show_and_no_path_status`: Reuses the solved fixture through a mock; checks PNG output without a filename extension, one display call, and status 0. An empty result must still export PDF, suppress display with `--no-show`, and return 1; reversed pitch bounds must produce argparse status 2.

## tests/test_xflr5.py

### PolarImportTests — 8 tests

The fixture reads the supplied XFLR5 polar. Tests check both recovered coefficients and explicit `MISSING` placeholders for information the polar cannot supply.

- `test_supplied_polar_recovery_and_units`: Checks recovery of 7/41 parameters, the -11…11-degree alpha grid and singleton beta axis, known lift/drag/sideforce and static moment values, and pitch slope converted to per-radian units. Unmeasured derivatives, stall parameters, mass, and inertia remain missing.
- `test_stall_extremes_are_radians_and_missing_shape_is_not_inferred`: With extreme-angle inference enabled, checks 10/41 recovered parameters, enabled stall blending, ±11-degree limits converted to radians, and missing transition width and maximum drag.
- `test_zero_interpolation_and_sorted_grid_orientation`: Converts an unsorted synthetic two-dimensional polar; checks beta-row/alpha-column ordering, interpolation of zero-angle pitch moment, per-radian pitch slope, and missing drag when no drag column exists.
- `test_no_nearest_angle_substitution_or_unmeasured_derivative`: Checks that samples not spanning alpha zero or beta zero cannot supply zero-angle pitch moment or slope. A single zero-angle sample supplies the moment but leaves the slope missing.
- `test_invalid_exports_fail_instead_of_silently_losing_data`: Rejects missing headers/data, NaN values, duplicate grid points, incomplete grids, malformed or short rows, and varying freestream speeds; stall-extreme inference also rejects data without both negative and positive angles.
- `test_completed_template_loads_and_evaluates`: Checks generated keys against the sample schema, fills missing fields from the sample, loads `ModelParameters`, and verifies recovered lift and interpolated drag values.
- `test_cli_default_and_custom_output_from_another_directory`: Runs the converter in subprocesses from a temporary directory using a UTF-8 BOM input; checks status 0, the 7/41 summary, default JSON filename/content, and custom output with stall inference enabled.
- `test_cli_errors_preserve_existing_files`: Checks status 2 for invalid input, using the input as output, a nonexistent input, and missing arguments; verifies that existing input and output contents are preserved.
