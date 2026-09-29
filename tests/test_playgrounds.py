"""Analytic statistics, scenario setup, geometry, and rendering checks."""

from dataclasses import replace
from pathlib import Path
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose

from vtol_dynamics import AerodynamicCoefficients, ModelParameters, StallParameters, State
from vtol_dynamics.playgrounds import bicopter_rotation, bicopter_translation, glide
from vtol_dynamics.playgrounds.plotting import animation_frames, craft_segments, create_display
from vtol_dynamics.playgrounds.simulation import flight_statistics


class PlaygroundTests(unittest.TestCase):
    def setUp(self):
        self.params = ModelParameters.from_json(Path(__file__).parents[1]/"examples/sample_parameters.json")

    def test_translation_zero_initial_state_constant_commands_and_partial_step(self):
        history = bicopter_translation.run(self.params, .105, 60, np.pi/2, dt=.04)
        assert_allclose(history.times, [0, .04, .08, .105])
        assert_allclose(history.states[0], np.zeros(14))
        self.assertEqual(history.controls.left_propeller_speed, 60)
        self.assertEqual(history.controls.right_propeller_speed, 60)
        self.assertEqual(history.controls.left_motor_tilt, np.pi/2)
        self.assertGreater(history.states[-1, 12], 0)
        self.assertLess(history.states[-1, 12], np.pi/2)
        assert_allclose(history.states[:, 12], history.states[:, 13])

    def test_rotation_command_sides_and_frozen_translation(self):
        history = bicopter_rotation.run(self.params, .1, .3, .6, 65, 60)
        assert_allclose(history.states[0], np.zeros(14))
        self.assertEqual(history.controls.right_propeller_speed, 65)
        self.assertEqual(history.controls.left_propeller_speed, 60)
        self.assertEqual(history.controls.right_motor_tilt, .3)
        self.assertEqual(history.controls.left_motor_tilt, .6)
        assert_allclose(history.states[:, :3], 0)
        assert_allclose(history.states[:, 6:9], 0)
        self.assertGreater(np.linalg.norm(history.states[-1, 9:12]), 0)

    def test_glide_initial_velocity_only_and_zero_controls(self):
        history = glide.run(self.params, .05, 12, 1, 2)
        assert_allclose(history.states[0], State(u=12, v=1, w=2).as_vector())
        self.assertEqual(history.controls.left_propeller_speed, 0)
        self.assertEqual(history.controls.elevator_deflection, 0)
        stats = flight_statistics(history)
        self.assertAlmostEqual(stats.airspeed[0], np.sqrt(149))
        self.assertAlmostEqual(stats.alpha[0], np.arctan2(2, 12))
        self.assertAlmostEqual(stats.beta[0], np.arctan2(1, np.hypot(12, 2)))

    def test_freefall_energy_and_negative_potential_below_zero(self):
        history = bicopter_translation.run(self.params, .4, 0, 0)
        stats = flight_statistics(history)
        expected_z = .5*self.params.gravity*history.times**2
        assert_allclose(history.states[:, 2], expected_z, atol=1e-12)
        assert_allclose(stats.potential_energy, -self.params.mass*self.params.gravity*expected_z, atol=1e-12)
        assert_allclose(stats.total_energy, 0, atol=1e-11)
        self.assertLess(stats.potential_energy[-1], 0)

    def test_kinetic_energy_includes_rotational_inertia_coupling(self):
        history = bicopter_rotation.run(self.params, .1, .3, .6, 65, 60)
        omega = history.states[-1, 9:12]
        ix = self.params.inertia
        expected = .5*(ix.Ixx*omega[0]**2 + ix.Iyy*omega[1]**2 + ix.Izz*omega[2]**2
                       - 2*ix.Ixz*omega[0]*omega[2])
        stats = flight_statistics(history)
        self.assertAlmostEqual(stats.kinetic_energy[-1], expected)
        self.assertGreater(expected, 0)
        assert_allclose(stats.translational_energy, 0)

    def test_glide_no_aero_matches_ballistic_motion(self):
        params = replace(self.params, aerodynamics=AerodynamicCoefficients(), stall=StallParameters(enabled=False))
        history = glide.run(params, .2, 3, 2, 1)
        assert_allclose(history.states[-1, :3], [.6, .4, .2+.5*params.gravity*.2**2], atol=1e-12)

    def test_positive_axes_and_engine_tilts_in_display_coordinates(self):
        state = State(left_motor_tilt=np.pi/2).as_vector()
        segments = craft_segments(state, self.params, 2)
        assert_allclose(segments[:3, 1]-segments[:3, 0], [[2, 0, 0], [0, 2, 0], [0, 0, -2]], atol=1e-12)
        assert_allclose(segments[3, 0], [0, -self.params.engine_span, -self.params.engine_height])
        assert_allclose(segments[4, 0], [0, self.params.engine_span, -self.params.engine_height])
        assert_allclose(segments[3, 1]-segments[3, 0], [0, 0, 2], atol=1e-12)
        assert_allclose(segments[4, 1]-segments[4, 0], [2, 0, 0], atol=1e-12)
        yawed = craft_segments(State(yaw=np.pi/2).as_vector(), self.params, 2)
        assert_allclose(yawed[0, 1]-yawed[0, 0], [0, 2, 0], atol=1e-12)

    def test_animation_includes_both_endpoints(self):
        frames = animation_frames(np.linspace(0, 1, 101), 20, 2)
        self.assertEqual(frames[0], 0)
        self.assertEqual(frames[-1], 100)
        self.assertEqual(len(frames), 11)
        assert_allclose(animation_frames([0, .001], 30, 1), [0, 1])

    def test_craft_stays_visible_and_in_bounds_on_long_paths(self):
        original = bicopter_rotation.run(self.params, .04, .3, .6, 65, 60)
        for span, axis_length in ((0, None), (1000, None), (1e6, 2)):
            with self.subTest(span=span, axis_length=axis_length):
                states = original.states.copy()
                states[:, :3] = np.linspace([10, -20, -30], [10+span, -20-span, -30-span], len(states))
                expected_states = states.copy()
                history = replace(original, states=states)
                display = create_display(history, fps=10, axis_length=axis_length)
                try:
                    display.flight_figure.canvas.draw()
                    ax = display.flight_figure.axes[0]
                    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
                    base_length = self.params.wingspan/2 if axis_length is None else axis_length
                    lengths = []
                    for index in (0, len(states)-1):
                        display.animation._func(index)
                        segments = np.array([np.array(line.get_data_3d()).T for line in ax.lines[1:]])
                        origin = states[index, :3]*[1, 1, -1]
                        line_lengths = np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1)
                        lengths.append(line_lengths[0])
                        assert_allclose(line_lengths, lengths[-1])
                        if span:
                            self.assertGreater(lengths[-1]/np.ptp(limits[0]), .07)
                        else:
                            self.assertAlmostEqual(lengths[-1], base_length)
                        # Enlargement preserves body directions and engine spacing.
                        natural = craft_segments(states[index], self.params, base_length)
                        assert_allclose((segments-origin)/lengths[-1],
                                        (natural-origin)/base_length, atol=1e-9)
                        self.assertTrue(np.all(segments >= limits[:, 0]))
                        self.assertTrue(np.all(segments <= limits[:, 1]))
                    assert_allclose(lengths[0], lengths[1])
                    assert_allclose(np.array(ax.lines[0].get_data_3d()).T, states[:, :3]*[1, 1, -1])
                    assert_allclose(history.states, expected_states)
                finally:
                    plt.close(display.statistics_figure)
                    plt.close(display.flight_figure)

    def test_headless_figures_and_html_animation_render(self):
        history = bicopter_translation.run(self.params, .04, 50, .3)
        display = create_display(history, fps=10)
        try:
            display.statistics_figure.canvas.draw()
            display.flight_figure.canvas.draw()
            html = display.animation.to_jshtml(default_mode="once")
            self.assertIn("data:image/png;base64", html)
            self.assertEqual(len(display.statistics_figure.axes), 6)
            # Six lines: the flight trail plus the requested five craft axes.
            self.assertEqual(len(display.flight_figure.axes[0].lines), 6)
        finally:
            plt.close(display.statistics_figure)
            plt.close(display.flight_figure)

    def test_invalid_duration_and_sampling(self):
        for time, dt in [(0, .02), (-1, .02), (np.inf, .02), (.1, 0), (.1, np.nan)]:
            with self.subTest(time=time, dt=dt), self.assertRaises(ValueError):
                glide.run(self.params, time, 0, 0, 0, dt=dt)


if __name__ == "__main__":
    unittest.main()
