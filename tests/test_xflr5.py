"""Polar recovery, CLI behavior, and compatibility with the simulation loader."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from vtol_dynamics import ModelParameters
from xflr5_to_parameters import MISSING, convert_polar, main, parameter_counts


ROOT = Path(__file__).resolve().parents[1]
POLAR = (ROOT / "T1.txt").read_text(encoding="utf-8")


class PolarImportTests(unittest.TestCase):
    def test_supplied_polar_recovery_and_units(self):
        result = convert_polar(POLAR)
        self.assertEqual(parameter_counts(result), (7, 41))
        aero = result["aerodynamics"]
        self.assertEqual(aero["lift"]["alphas"], list(range(-11, 12)))
        self.assertEqual(aero["lift"]["betas"], [0])
        self.assertEqual(aero["lift"]["values"][0][11], 0.461302)
        self.assertEqual(aero["drag"]["values"][0][11], 0.026527)
        self.assertEqual(aero["sideforce"]["values"], [[0] * 23])
        self.assertEqual(aero["roll_0"], 0)
        self.assertEqual(aero["yaw_0"], 0)
        self.assertEqual(aero["pitch_0"], -0.020901)
        self.assertAlmostEqual(aero["pitch_alpha"], (-0.027305 + 0.014649) / math.radians(2))
        for key in ("roll_beta", "yaw_beta", "pitch_q", "pitch_elevator", "roll_p"):
            self.assertEqual(aero[key], MISSING)
        self.assertTrue(all(value == MISSING for value in result["stall"].values()))
        self.assertEqual(result["mass"], MISSING)
        self.assertEqual(result["inertia"]["Ixz"], MISSING)

    def test_stall_extremes_are_radians_and_missing_shape_is_not_inferred(self):
        result = convert_polar(POLAR, stall_at_extremes=True)
        self.assertEqual(parameter_counts(result), (10, 41))
        stall = result["stall"]
        self.assertIs(stall["enabled"], True)
        self.assertAlmostEqual(stall["alpha_negative"], math.radians(-11))
        self.assertAlmostEqual(stall["alpha_positive"], math.radians(11))
        self.assertEqual(stall["transition_width"], MISSING)
        self.assertEqual(stall["drag_max"], MISSING)

    def test_zero_interpolation_and_sorted_grid_orientation(self):
        result = convert_polar("alpha Beta CL Cm\n2 3 32 -4\n-2 -3 -32 4\n"
                               "-2 3 28 4\n2 -3 -28 -4\n")
        aero = result["aerodynamics"]
        self.assertEqual(aero["lift"]["values"], [[-32, -28], [28, 32]])
        self.assertEqual(aero["pitch_0"], 0)
        self.assertAlmostEqual(aero["pitch_alpha"], -2 * 180 / math.pi)
        self.assertEqual(aero["drag"], MISSING)

    def test_no_nearest_angle_substitution_or_unmeasured_derivative(self):
        for text in ("alpha Beta Cm\n1 0 2\n2 0 4\n",
                     "alpha Beta Cm\n-1 2 -2\n1 2 2\n"):
            aero = convert_polar(text)["aerodynamics"]
            self.assertEqual(aero["pitch_0"], MISSING)
            self.assertEqual(aero["pitch_alpha"], MISSING)
        aero = convert_polar("alpha Beta Cm\n0 0 -0.02\n")["aerodynamics"]
        self.assertEqual(aero["pitch_0"], -0.02)
        self.assertEqual(aero["pitch_alpha"], MISSING)

    def test_invalid_exports_fail_instead_of_silently_losing_data(self):
        for text in ("not a polar", "alpha Beta CL\n", "alpha Beta CL\n0 0 nan\n",
                     "alpha Beta CL\n0 0 1\n0 0 2\n",
                     "alpha Beta CL\n0 0 1\n1 1 2\n",
                     "alpha Beta CL\n0 0 1\n1 0 broken\n",
                     "alpha Beta CL\n0 0 1\n1 0\n",
                     "alpha Beta CL QInf\n0 0 1 11\n1 0 2 12\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                convert_polar(text)
        with self.assertRaisesRegex(ValueError, "negative and positive"):
            convert_polar("alpha Beta CL\n0 0 1\n1 0 2\n", stall_at_extremes=True)

    def test_completed_template_loads_and_evaluates(self):
        generated = convert_polar(POLAR, stall_at_extremes=True)
        sample = json.loads((ROOT / "examples/sample_parameters.json").read_text())
        self.assertEqual(set(generated), set(sample))
        for name, value in generated.items():
            if isinstance(value, dict):
                self.assertEqual(set(value), set(sample[name]))
                for key, slot in value.items():
                    if slot == MISSING:
                        value[key] = sample[name][key]
            elif value == MISSING:
                generated[name] = sample[name]
        parameters = ModelParameters.from_dict(generated)
        self.assertEqual(parameters.aerodynamics.lift(0, 0), 0.461302)
        self.assertAlmostEqual(parameters.aerodynamics.drag(math.radians(0.5), 0),
                               (0.026527 + 0.030165) / 2)

    def test_cli_default_and_custom_output_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "aircraft.polar.txt"
            source.write_text(POLAR, encoding="utf-8-sig")
            command = [sys.executable, str(ROOT / "xflr5_to_parameters.py"), str(source)]
            run = subprocess.run(command, cwd=directory, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("7/41", run.stdout)
            self.assertEqual(json.loads(source.with_suffix(".json").read_text()), convert_polar(POLAR))
            output = Path(directory) / "custom.json"
            run = subprocess.run(command + ["--output", str(output), "--stall-at-extremes"],
                                 cwd=directory, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue(json.loads(output.read_text())["stall"]["enabled"])

    def test_cli_errors_preserve_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.txt"
            source.write_text("invalid")
            output = Path(directory) / "output.json"
            output.write_text("keep me")
            for arguments in ([str(source), "-o", str(output)], [str(source), "-o", str(source)],
                              [str(Path(directory) / "missing.txt")], []):
                with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        main(arguments)
                self.assertEqual(raised.exception.code, 2)
            self.assertEqual(source.read_text(), "invalid")
            self.assertEqual(output.read_text(), "keep me")


if __name__ == "__main__":
    unittest.main()
