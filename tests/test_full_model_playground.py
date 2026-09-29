"""Full-model CLI state/control coverage, units, defaults, and validation."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import fields
from io import StringIO
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from vtol_dynamics import ControlInputs, ModelParameters, State, VTOLModel
from vtol_dynamics.playgrounds import full_model


class FullModelPlaygroundTests(unittest.TestCase):
    def setUp(self):
        self.params = ModelParameters.from_json(
            Path(__file__).parents[1]/"examples/sample_parameters.json")

    def cli(self, *arguments):
        with redirect_stdout(StringIO()):
            return full_model.main(["--no-show", *arguments])

    def test_no_arguments_defaults_to_zero_state_controls_and_one_second(self):
        # Exercise a truly empty argv without opening the default GUI.
        with patch.object(full_model, "execute",
                          side_effect=lambda parser, args, run: run(self.params)):
            history = full_model.main([])
        assert_allclose(history.states[0], np.zeros(14))
        self.assertEqual(history.controls, ControlInputs())
        self.assertEqual(history.times[-1], 1.0)
        python_history = full_model.run(self.params)
        assert_allclose(history.states, python_history.states)

    def test_entire_initial_state_and_constant_controls(self):
        history = self.cli(
            "--time", ".04", "--dt", ".02",
            "--initial-x", "1", "--initial-y", "-2", "--initial-z", "-3",
            "--initial-roll", "4", "--initial-pitch", "-5", "--initial-yaw", "6",
            "--initial-u", "7", "--initial-v", "-8", "--initial-w", "9",
            "--initial-p", "10", "--initial-q", "-11", "--initial-r", "12",
            "--initial-right-motor-tilt", "13", "--initial-left-motor-tilt", "14",
            "--control-right-motor-tilt", "15", "--control-left-motor-tilt", "16",
            "--control-right-propeller-speed", "17", "--control-left-propeller-speed", "18",
            "--control-elevator-deflection", "-19", "--control-aileron-deflection", "20",
            "--control-rudder-deflection", "21")
        expected_state = State(
            x=1, y=-2, z=-3, roll=np.deg2rad(4), pitch=np.deg2rad(-5), yaw=np.deg2rad(6),
            u=7, v=-8, w=9, p=np.deg2rad(10), q=np.deg2rad(-11), r=np.deg2rad(12),
            right_motor_tilt=np.deg2rad(13), left_motor_tilt=np.deg2rad(14))
        expected_controls = ControlInputs(
            right_motor_tilt=np.deg2rad(15), left_motor_tilt=np.deg2rad(16),
            right_propeller_speed=17, left_propeller_speed=18,
            elevator_deflection=np.deg2rad(-19), aileron_deflection=np.deg2rad(20),
            rudder_deflection=np.deg2rad(21))
        assert_allclose(history.states[0], expected_state.as_vector())
        self.assertEqual(history.controls, expected_controls)
        model = VTOLModel(self.params, expected_state)
        for actual in history.states[1:]:
            assert_allclose(actual, model.step(expected_controls, .02).as_vector())

    def test_partial_arguments_keep_other_components_zero_and_tilts_independent(self):
        history = self.cli("--simulation-time", ".04", "--initial-z", "-10",
                           "--initial-left-motor-tilt", "20", "--control-right-motor-tilt", "30")
        assert_allclose(history.states[0],
                        State(z=-10, left_motor_tilt=np.deg2rad(20)).as_vector())
        self.assertEqual(history.controls, ControlInputs(right_motor_tilt=np.deg2rad(30)))
        self.assertGreater(history.states[-1, 12], 0)
        self.assertLess(history.states[-1, 12], np.deg2rad(30))
        self.assertGreater(history.states[-1, 13], 0)
        self.assertLess(history.states[-1, 13], np.deg2rad(20))

    def test_invalid_cli_values_and_removed_instant_tilt_are_rejected(self):
        invalid = [["--instant-tilt"], ["--time", "0"], ["--dt", "0"]]
        for prefix, cls in (("initial", State), ("control", ControlInputs)):
            for field in fields(cls):
                flag = f"--{prefix}-{field.name.replace('_', '-')}"
                invalid.extend([[flag, value] for value in ("nan", "inf")])
        invalid.extend([[f"--control-{side}-propeller-speed", "-1"]
                        for side in ("left", "right")])
        for arguments in invalid:
            with self.subTest(arguments=arguments), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as error:
                    full_model.main(arguments)
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
