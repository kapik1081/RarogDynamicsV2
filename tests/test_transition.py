"""Updated PDF transition equations, constraints, charts, connectivity and CLI."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import brentq

from find_transition_trim import main
from vtol_dynamics import AerodynamicCoefficients, ModelParameters, StallParameters
from vtol_dynamics.dynamics import state_derivative
from vtol_dynamics.transition import (TransitionConstraints, _Surface, _connect,
                                      scalar_roots, transition_trim_manifold, trim_residual)
from vtol_dynamics.trim import longitudinal_forces


PARAMETERS = Path(__file__).parents[1]/"examples/sample_parameters.json"


class TransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = ModelParameters.from_json(PARAMETERS)
        cls.limits = TransitionConstraints(8000, np.deg2rad(25))
        cls.surface = _Surface(cls.params, cls.limits, 12)
        cls.result = transition_trim_manifold(cls.params, cls.limits, cruise_airspeed=12,
                                              speed_points=13, tilt_points=13,
                                              pitch_points=81, elevator_points=5)

    def test_surface_path_and_full_model_equilibria(self):
        result = self.result
        self.assertTrue(result.path)
        self.assertGreater(result.branch_counts.max(), 1)
        assert_allclose(result.path[0].coordinates[:3], [0, np.pi/2, 0], atol=1e-10)
        assert_allclose(result.path[-1].coordinates[:2], [12, 0], atol=1e-10)
        self.assertGreater(abs(result.path[-1].solution.state.pitch), .01)
        for point in result.points+result.path:
            assert_allclose(trim_residual(point.coordinates, self.params), 0, atol=1e-8)
            s = point.solution
            derivative = state_derivative(0, s.state.as_vector(), s.controls, self.params)
            assert_allclose(derivative, np.r_[s.airspeed, np.zeros(13)], atol=1e-6)
            self.assertLessEqual(s.engine_rpm, self.limits.max_engine_rpm+1e-8)
            self.assertLessEqual(abs(s.controls.elevator_deflection), self.limits.max_elevator_deflection+1e-8)
        z = np.array([p.coordinates for p in result.path])
        self.assertLess(np.linalg.norm(np.diff(z, axis=0)/self.surface.scale, axis=1).max(), .071)

    def test_tangential_roots_and_close_sign_changes(self):
        function = lambda x: (x-.12345)**2*(x+.41)*(x-.7)
        roots = scalar_roots(function, np.linspace(-1, 1, 90))
        assert_allclose(roots, [-.41, .12345, .7], atol=1e-7)
        with self.assertRaises(ValueError):
            scalar_roots(lambda x: np.nan, [-1, 0, 1])

    def test_hover_zero_speed_limits_and_quadratic_neck(self):
        de = -.1
        hover = np.array([0, np.pi/2, 0, self.surface.weight, de])
        self.assertIsNotNone(self.surface.accept(hover))
        coefficients = []
        for speed in (.01, .02):
            z = self.surface.correct_chart(hover, (0, 4), (speed, de))
            self.assertIsNotNone(self.surface.accept(z))
            coefficients.append((np.pi/2-z[1])/speed**2)
        expected = (-self.params.air_density*self.params.wing_area*self.params.mean_chord
                    * (.02-.8*de)/(2*self.params.engine_height*self.surface.weight))
        assert_allclose(coefficients, expected, rtol=2e-4)
        # No alpha-map query at zero speed, even with an undefined freestream angle.
        with patch("vtol_dynamics.transition.longitudinal_forces", side_effect=AssertionError):
            assert_allclose(trim_residual(hover, self.params), 0, atol=1e-14)

    def test_vertical_thrust_projection_is_regular(self):
        params = replace(self.params, engine_height=0, aerodynamics=AerodynamicCoefficients(),
                         stall=StallParameters(enabled=False))
        surface = _Surface(params, self.limits, 12)
        point = surface.reconstruct(4, np.pi/2, 0)
        self.assertIsNotNone(point)
        self.assertAlmostEqual(point.solution.total_thrust, surface.weight)
        assert_allclose(trim_residual(point.coordinates, params), 0, atol=1e-14)

    def test_rpm_power_attitude_and_control_limits(self):
        cruise = self.result.path[-1]
        for changes in ({"max_engine_rpm": 1}, {"max_shaft_power": cruise.shaft_power*.5},
                        {"max_elevator_deflection": 0}, {"tilt_min": .1},
                        {"pitch_max": 0}, {"alpha_max": 0}):
            with self.subTest(changes=changes):
                surface = _Surface(self.params, replace(self.limits, **changes), 12)
                self.assertIsNone(surface.accept(cruise.coordinates))
        # Changing a control after the solve must be caught by the residual check.
        changed = cruise.coordinates.copy()
        changed[-1] = 0
        self.assertIsNone(self.surface.accept(changed))

    def test_lateral_limits_and_sideforce_rejection(self):
        params = replace(self.params, aerodynamics=replace(self.params.aerodynamics, roll_0=.01))
        surface = _Surface(params, self.limits, 12)
        point = surface.accept(self.result.path[-1].coordinates)
        self.assertIsNotNone(point)
        self.assertNotEqual(point.solution.controls.aileron_deflection, 0)
        surface = _Surface(params, replace(self.limits, max_aileron_deflection=0,
                                          max_rudder_deflection=0), 12)
        self.assertIsNone(surface.accept(point.coordinates))
        params = replace(self.params, aerodynamics=replace(self.params.aerodynamics, sideforce=.1))
        self.assertIsNone(_Surface(params, self.limits, 12).accept(point.coordinates))

    def test_same_projection_does_not_join_different_branches(self):
        cruise = [p for p in self.result.points if p.coordinates[0] == 12 and p.coordinates[1] == 0]
        self.assertGreaterEqual(len(cruise), 2)
        self.assertIsNone(_connect(self.surface, cruise[0], cruise[1]))

    def test_edge_interior_constraints_not_just_endpoints(self):
        first, last = self.result.path[10], self.result.path[14]
        self.assertIsNotNone(_connect(self.surface, first, last))
        original = self.surface.accept
        low, high = sorted([first.coordinates[0], last.coordinates[0]])

        def forbidden_strip(z):
            if low+.2*(high-low) < z[0] < high-.2*(high-low):
                return None
            return original(z)

        with patch.object(self.surface, "accept", side_effect=forbidden_strip):
            self.assertIsNone(_connect(self.surface, first, last))

    def test_monotonicity_constraints(self):
        surface = _Surface(self.params, replace(self.limits, monotone_airspeed=True,
                                               monotone_tilt=True), 12)
        first, last = self.result.path[5], self.result.path[10]
        self.assertIsNotNone(_connect(surface, first, last))
        self.assertIsNone(_connect(surface, last, first))

    def test_pseudo_arclength_traverses_tilt_fold(self):
        limits = replace(self.limits, max_elevator_deflection=np.deg2rad(85),
                         pitch_min=-1.0, pitch_max=1.0)
        surface = _Surface(self.params, limits, 12)

        def tilt(pitch):
            lift, drag = longitudinal_forces(pitch, 12, self.params)
            return np.arctan2(surface.weight-lift, drag)-pitch

        derivative = lambda pitch: (tilt(pitch+1e-5)-tilt(pitch-1e-5))/2e-5
        grid = np.linspace(.1, .6, 100)
        intervals = [(a, b) for a, b in zip(grid[:-1], grid[1:]) if derivative(a)*derivative(b) < 0]
        self.assertTrue(intervals)
        fold = brentq(derivative, *intervals[-1])
        seed = surface.reconstruct(12, tilt(fold), fold)
        self.assertIsNotNone(seed)
        continued = list(surface.arclength(seed.coordinates, step=.01, count=12))
        self.assertGreater(len(continued), 10)
        self.assertLess(min(z[2] for z in continued), fold)
        self.assertGreater(max(z[2] for z in continued), fold)
        for z in continued:
            assert_allclose(trim_residual(z, self.params), 0, atol=1e-8)

    def test_no_hover_capacity_means_no_path_but_keeps_forward_trims(self):
        limits = replace(self.limits, max_engine_rpm=3000)
        result = transition_trim_manifold(self.params, limits, cruise_airspeed=12,
                                          speed_points=3, tilt_points=3, pitch_points=31,
                                          elevator_points=3)
        self.assertTrue(result.points)
        self.assertFalse(result.path)
        self.assertTrue(all(p.solution.airspeed > 0 for p in result.points))

    def test_invalid_inputs(self):
        for changes in ({"max_engine_rpm": -1}, {"max_shaft_power": np.inf},
                        {"tilt_min": 2}, {"pitch_max": np.pi/2}, {"alpha_min": 1, "alpha_max": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.limits, **changes)
        for keywords in ({"cruise_airspeed": (12, 10)}, {"airspeed_max": 5}, {"pitch_points": 2}):
            with self.subTest(keywords=keywords), self.assertRaises(ValueError):
                transition_trim_manifold(self.params, self.limits, **({"cruise_airspeed": 12}|keywords))

    def test_cli_plot_export_show_and_no_path_status(self):
        args = [str(PARAMETERS), "--max-engine-rpm", "8000", "--max-elevator-deflection", "25",
                "--cruise-airspeed", "12", "--quiet"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"transition image"
            with patch("find_transition_trim.transition_trim_manifold", return_value=self.result), \
                    patch("matplotlib.pyplot.show") as show, redirect_stdout(io.StringIO()):
                self.assertEqual(main(args+["--o", str(output)]), 0)
                show.assert_called_once()
                self.assertTrue(output.read_bytes().startswith(b"\x89PNG"))
            empty = replace(self.result, points=(), path=(), branch_counts=np.zeros_like(self.result.branch_counts))
            with patch("find_transition_trim.transition_trim_manifold", return_value=empty), \
                    patch("matplotlib.pyplot.show") as show, redirect_stdout(io.StringIO()):
                self.assertEqual(main(args+["--no-show", "--o", str(output.with_suffix(".pdf"))]), 1)
                show.assert_not_called()
                self.assertTrue(output.with_suffix(".pdf").read_bytes().startswith(b"%PDF"))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(args+["--pitch-min", "40", "--pitch-max", "20"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
