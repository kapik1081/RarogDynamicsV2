"""Trim formulas, all-root scanning, feasibility rejection, and console scripts."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import unittest

import numpy as np
from numpy.testing import assert_allclose

from find_forward_trim import main as forward_main
from find_hover_trim import main as hover_main
from vtol_dynamics import (AerodynamicCoefficients, CoefficientMap, ModelParameters,
                           StallParameters, VTOLModel)
from vtol_dynamics.dynamics import calculate_loads
from vtol_dynamics.trim import (aerodynamic_alpha_range, bracketed_roots, forward_flight_trim,
                                forward_residual, hover_trim, longitudinal_forces)


PARAMETERS = Path(__file__).parents[1]/"examples/sample_parameters.json"


class TrimTests(unittest.TestCase):
    def setUp(self):
        self.params = ModelParameters.from_json(PARAMETERS)

    def test_hover_closed_form_and_stationary_trajectory(self):
        trim = hover_trim(self.params)
        expected_speed = np.sqrt(self.params.mass*self.params.gravity /
                                 (2*self.params.CT*self.params.air_density*self.params.propeller_diameter**4))
        self.assertAlmostEqual(trim.controls.left_propeller_speed, expected_speed)
        self.assertEqual(trim.controls.right_propeller_speed, trim.controls.left_propeller_speed)
        self.assertEqual(trim.state.left_motor_tilt, np.pi/2)
        self.assertEqual(trim.controls.left_motor_tilt, trim.state.left_motor_tilt)
        model = VTOLModel(self.params, trim.state)
        assert_allclose(model.step(trim.controls, 1).as_vector(), trim.state.as_vector(), atol=1e-12)

    def test_brent_finds_all_sign_changes_and_exact_grid_roots(self):
        function = lambda x: (x+.73)*x*(x-.24)*(x-.81)
        roots = bracketed_roots(function, np.linspace(-1, 1, 101))
        assert_allclose(roots, [-.73, 0, .24, .81], atol=1e-10)
        assert_allclose(bracketed_roots(lambda x: x*(x-1)*(x-2), [-1, 0, 1, 2, 3]), [0, 1, 2])
        with self.assertRaises(ValueError):
            bracketed_roots(lambda x: np.nan, [-1, 1])

    def test_full_post_stall_scan_keeps_all_candidates_and_prefers_low_alpha(self):
        result = forward_flight_trim(self.params, 0, 12)
        assert_allclose(result.alpha_range, [-np.pi, np.pi])
        feasible = [c.solution for c in result.candidates if c.solution is not None]
        self.assertGreaterEqual(len(result.candidates), 4)
        self.assertGreaterEqual(len(feasible), 3)
        self.assertTrue(result.selected.pre_stall)
        self.assertAlmostEqual(np.rad2deg(result.selected.alpha), 7.30424617, places=6)
        self.assertTrue(any(not s.pre_stall for s in feasible))
        self.assertTrue(any("negative" in " ".join(c.rejection_reasons) for c in result.candidates))
        for solution in feasible:
            self.assertAlmostEqual(forward_residual(solution.alpha, 12, 0, self.params), 0, places=7)

    def test_pitch_sign_force_equilibrium_and_level_trajectory(self):
        solution = forward_flight_trim(self.params, np.deg2rad(20), 12).selected
        self.assertIsNotNone(solution)
        loads = calculate_loads(solution.state, solution.controls, self.params)
        assert_allclose(loads.total_force, 0, atol=1e-8)
        assert_allclose(loads.total_moment, 0, atol=1e-8)
        lift, drag = longitudinal_forces(solution.alpha, 12, self.params)
        angle = np.deg2rad(20)+solution.alpha
        self.assertAlmostEqual(solution.total_thrust*np.cos(angle), drag, places=8)
        self.assertAlmostEqual(lift+solution.total_thrust*np.sin(angle), self.params.mass*self.params.gravity, places=8)
        model = VTOLModel(self.params, solution.state)
        expected = solution.state.as_vector()
        expected[0] = 6.0
        assert_allclose(model.step(solution.controls, .5).as_vector(), expected, atol=1e-8)

    def test_rpm_and_absolute_elevator_limits_reject_after_scanning(self):
        unrestricted = forward_flight_trim(self.params, 0, 12)
        limited = forward_flight_trim(self.params, 0, 12, max_engine_rpm=100,
                                     max_elevator_deflection=np.deg2rad(1))
        self.assertIsNone(limited.selected)
        assert_allclose([c.alpha for c in limited.candidates], [c.alpha for c in unrestricted.candidates])
        calculated = [c for c in limited.candidates if c.elevator_deflection is not None]
        self.assertGreater(len(calculated), 0)
        for candidate in calculated:
            self.assertIn("engine RPM exceeds limit", candidate.rejection_reasons)
            self.assertIn("elevator deflection exceeds limit", candidate.rejection_reasons)

    def test_validity_bounds_do_not_shrink_search_and_allow_fallback(self):
        unrestricted = forward_flight_trim(self.params, 0, 12)
        limited = forward_flight_trim(self.params, 0, 12, alpha_min=np.deg2rad(10), alpha_max=np.deg2rad(19))
        self.assertEqual(limited.alpha_range, unrestricted.alpha_range)
        assert_allclose([c.alpha for c in limited.candidates], [c.alpha for c in unrestricted.candidates])
        self.assertIsNotNone(limited.selected)
        self.assertFalse(limited.selected.pre_stall)
        self.assertAlmostEqual(np.rad2deg(limited.selected.alpha), 16.9970791, places=6)

    def test_strict_map_range_and_beta_validity(self):
        strict = replace(self.params.aerodynamics.lift, bounds="raise")
        params = replace(self.params, aerodynamics=replace(self.params.aerodynamics, lift=strict))
        assert_allclose(aerodynamic_alpha_range(params), np.deg2rad([-15, 15]))
        result = forward_flight_trim(params, 0, 12)
        self.assertIsNotNone(result.selected)
        self.assertTrue(all(-15 <= np.rad2deg(c.alpha) <= 15 for c in result.candidates))
        excluded = CoefficientMap([-1, 1], [.1, .2], [[0, 1], [0, 1]], bounds="raise")
        params = replace(params, aerodynamics=replace(params.aerodynamics, sideforce=excluded))
        with self.assertRaisesRegex(ValueError, "beta=0"):
            forward_flight_trim(params, 0, 12)

    def test_singular_projection_is_rejected_instead_of_false_trim(self):
        params = replace(self.params, aerodynamics=AerodynamicCoefficients(), stall=StallParameters(enabled=False))
        result = forward_flight_trim(params, 0, 12)
        self.assertIsNone(result.selected)
        self.assertGreater(len(result.candidates), 0)
        self.assertTrue(all("singular thrust projection" in " ".join(c.rejection_reasons) for c in result.candidates))

    def test_zero_elevator_effectiveness_and_balanced_special_case(self):
        aero = replace(self.params.aerodynamics, pitch_elevator=0)
        result = forward_flight_trim(replace(self.params, aerodynamics=aero), 0, 12)
        self.assertIsNone(result.selected)
        self.assertTrue(any("zero elevator effectiveness" in " ".join(c.rejection_reasons) for c in result.candidates))
        params = replace(self.params, engine_height=0,
                         aerodynamics=replace(aero, pitch_0=0, pitch_alpha=0))
        self.assertEqual(forward_flight_trim(params, 0, 12).selected.controls.elevator_deflection, 0)

    def test_nonzero_lateral_trim_moments_and_unbalanced_sideforce(self):
        params = replace(self.params, aerodynamics=replace(self.params.aerodynamics, roll_0=.01, yaw_0=-.02))
        solution = forward_flight_trim(params, 0, 12).selected
        self.assertIsNotNone(solution)
        self.assertNotEqual(solution.controls.aileron_deflection, 0)
        self.assertNotEqual(solution.controls.rudder_deflection, 0)
        assert_allclose(calculate_loads(solution.state, solution.controls, params).total_moment, 0, atol=1e-9)
        params = replace(params, aerodynamics=replace(params.aerodynamics, sideforce=.1))
        result = forward_flight_trim(params, 0, 12)
        self.assertIsNone(result.selected)
        self.assertTrue(any("lateral dynamics" in " ".join(c.rejection_reasons) for c in result.candidates))

    def test_invalid_trim_inputs(self):
        with self.assertRaises(ValueError):
            hover_trim(replace(self.params, CT=0))
        for speed, keywords in [(0, {}), (np.nan, {}), (12, {"max_engine_rpm": -1}),
                                (12, {"max_elevator_deflection": -1}), (12, {"grid_points": 2}),
                                (12, {"alpha_min": 1, "alpha_max": 0})]:
            with self.subTest(speed=speed, keywords=keywords), self.assertRaises(ValueError):
                forward_flight_trim(self.params, 0, speed, **keywords)

    def test_console_scripts_and_no_feasible_exit_status(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(hover_main([str(PARAMETERS)]), 0)
            self.assertEqual(forward_main([str(PARAMETERS), "--engine-tilt", "0", "--airspeed", "12",
                                           "--max-elevator-deflection", "25", "--max-engine-rpm", "8000"]), 0)
        self.assertIn("RPM", output.getvalue())
        self.assertIn("Selected trim: pre-stall", output.getvalue())
        output = io.StringIO()
        with redirect_stdout(output):
            status = forward_main([str(PARAMETERS), "--engine-tilt", "0", "--airspeed", "12",
                                   "--max-engine-rpm", "1"])
        self.assertEqual(status, 1)
        self.assertIn("No feasible trim", output.getvalue())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            forward_main([str(PARAMETERS), "--engine-tilt", "0", "--airspeed", "0"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
