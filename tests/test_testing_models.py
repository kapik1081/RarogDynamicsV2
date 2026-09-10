"""Isolation and analytic checks for the four reduced stateful models."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from vtol_dynamics import (AerodynamicCoefficients, AerodynamicsOnlyModel, BicopterFullModel,
                           BicopterRotationModel, BicopterTranslationalModel, CoefficientMap,
                           ControlInputs, Inertia, ModelParameters, StallParameters, State, VTOLModel)
from vtol_dynamics.dynamics import state_derivative
from vtol_dynamics.kinematics import body_to_ned, gravity_acceleration
from vtol_dynamics.propulsion import propeller_thrust
from vtol_dynamics.testing_dynamics import (aerodynamics_only_derivative, bicopter_full_derivative,
                                           bicopter_rotation_derivative,
                                           bicopter_translational_derivative)


class TestingModelTests(unittest.TestCase):
    def setUp(self):
        self.params = ModelParameters.from_json(
            Path(__file__).parents[1] / "examples/sample_parameters.json")
        self.engine_models = (BicopterTranslationalModel, BicopterRotationModel, BicopterFullModel)
        self.all_models = (*self.engine_models, AerodynamicsOnlyModel)

    def test_translation_matches_constant_acceleration_at_fixed_attitude(self):
        initial = State(z=-20, roll=.2, pitch=-.1, yaw=.3, u=3, v=1, w=-.2,
                        left_motor_tilt=.4, right_motor_tilt=.8)
        controls = ControlInputs(50, 70, left_motor_tilt=.4, right_motor_tilt=.8)
        params = self.params
        model = BicopterTranslationalModel(params, initial)
        tl, tr = [propeller_thrust(n, params) for n in (50, 70)]
        acceleration = np.array([tl*np.cos(.4)+tr*np.cos(.8), 0,
                                 -tl*np.sin(.4)-tr*np.sin(.8)]) / params.mass
        acceleration += gravity_acceleration(initial.roll, initial.pitch, params.gravity)
        dt = .3
        result = model.step(controls, dt).as_vector()
        rotation = body_to_ned(initial.roll, initial.pitch, initial.yaw)
        expected_position = initial.as_vector()[:3] + rotation @ (
            initial.as_vector()[6:9]*dt + .5*acceleration*dt**2)
        assert_allclose(result[:3], expected_position, atol=1e-10)
        assert_allclose(result[6:9], initial.as_vector()[6:9]+acceleration*dt, atol=1e-10)
        assert_allclose(result[3:6], initial.as_vector()[3:6], atol=0)
        assert_allclose(result[9:12], [0, 0, 0], atol=0)

    def test_translation_rejects_inconsistent_angular_rates(self):
        for field in ("p", "q", "r"):
            state = State(**{field: .1})
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "fixed attitude"):
                    BicopterTranslationalModel(self.params, state)
                with self.assertRaisesRegex(ValueError, "fixed attitude"):
                    bicopter_translational_derivative(0, state.as_vector(), ControlInputs(), self.params)

    def test_tilted_freefall_stays_vertical_in_earth_frame_for_all_translating_models(self):
        params = replace(self.params, aerodynamics=AerodynamicCoefficients(),
                         stall=StallParameters(enabled=False))
        for model_type in (VTOLModel, BicopterTranslationalModel, BicopterFullModel, AerodynamicsOnlyModel):
            with self.subTest(model=model_type.__name__):
                initial = State(z=-20, roll=.8, pitch=-.7, yaw=.3, u=3, v=1, w=-.2)
                if model_type is not BicopterTranslationalModel:
                    initial = replace(initial, p=.2, q=-.1, r=.15)
                initial_vector = initial.as_vector()
                initial_velocity = body_to_ned(*initial_vector[3:6]) @ initial_vector[6:9]
                dt = .4
                model = model_type(params, initial, rtol=1e-10, atol=1e-12)
                result = model.step(ControlInputs(), dt).as_vector()
                final_velocity = body_to_ned(*result[3:6]) @ result[6:9]
                assert_allclose(result[:3], initial_vector[:3] + initial_velocity*dt
                                + np.array([0, 0, .5*params.gravity*dt**2]), atol=1e-9)
                assert_allclose(final_velocity, initial_velocity + [0, 0, params.gravity*dt], atol=1e-9)
                energies = [(.5*params.mass*(s[6:9] @ s[6:9])
                             + .5*s[9:12] @ params.inertia.matrix() @ s[9:12]
                             - params.mass*params.gravity*s[2]) for s in (initial_vector, result)]
                self.assertAlmostEqual(energies[0], energies[1], places=7)

    def test_rotation_matches_constant_pitch_acceleration(self):
        params = replace(self.params, inertia=Inertia(.35, .5, .7))
        initial = State(x=1, y=2, z=-10, u=3, v=4, w=5, q=.1)
        model = BicopterRotationModel(params, initial)
        dt = .1
        acceleration = 2*params.engine_height*propeller_thrust(50, params)/params.inertia.Iyy
        result = model.step(ControlInputs(50, 50), dt)
        self.assertAlmostEqual(result.q, .1+acceleration*dt, places=10)
        self.assertAlmostEqual(result.pitch, .1*dt+.5*acceleration*dt**2, places=10)
        assert_allclose(result.as_vector()[:3], initial.as_vector()[:3], atol=0)
        assert_allclose(result.as_vector()[6:9], initial.as_vector()[6:9], atol=0)

    def test_engine_models_never_evaluate_aerodynamics_or_surface_commands(self):
        # This table would reject the alpha/beta query if any aero calculation ran.
        forbidden_map = CoefficientMap([1, 2], [1, 2], [[1, 2], [3, 4]], bounds="raise")
        params = replace(self.params, aerodynamics=AerodynamicCoefficients(lift=forbidden_map),
                         stall=StallParameters(enabled=True))
        initial = State(z=-10, u=10, left_motor_tilt=.2, right_motor_tilt=.3)
        controls = ControlInputs(40, 41, left_motor_tilt=.4, right_motor_tilt=.5)
        surfaces = replace(controls, elevator_deflection=.4, aileron_deflection=.3, rudder_deflection=.2)
        for model_type in self.engine_models:
            with self.subTest(model=model_type.__name__):
                normal, changed = model_type(params, initial), model_type(params, initial)
                assert_allclose(normal.step(controls, .1).as_vector(),
                                changed.step(surfaces, .1).as_vector(), atol=0, rtol=0)

    def test_full_bicopter_matches_complete_model_without_aerodynamic_loads(self):
        params = replace(self.params, aerodynamics=AerodynamicCoefficients(),
                         stall=StallParameters(enabled=False))
        initial = State(z=-10, u=10, v=.5, w=.3, roll=.1, pitch=.2, yaw=.3,
                        p=.01, q=.02, r=.03, left_motor_tilt=.4, right_motor_tilt=.5)
        controls = ControlInputs(40, 42, left_motor_tilt=.6, right_motor_tilt=.7)
        assert_allclose(bicopter_full_derivative(0, initial.as_vector(), controls, params),
                        state_derivative(0, initial.as_vector(), controls, params), atol=1e-12)
        reduced, full = BicopterFullModel(params, initial), VTOLModel(params, initial)
        assert_allclose(reduced.step(controls, .2).as_vector(), full.step(controls, .2).as_vector(),
                        atol=1e-10)

    def test_glider_matches_complete_model_with_engines_off(self):
        initial = State(z=-30, u=12, v=.4, w=.3, p=.01, q=.02, r=.03,
                        left_motor_tilt=.4, right_motor_tilt=.5)
        controls = ControlInputs(elevator_deflection=.02, aileron_deflection=.01,
                                 rudder_deflection=.02, left_motor_tilt=.4, right_motor_tilt=.5)
        assert_allclose(aerodynamics_only_derivative(0, initial.as_vector(), controls, self.params),
                        state_derivative(0, initial.as_vector(), controls, self.params), atol=1e-12)
        glider, full = AerodynamicsOnlyModel(self.params, initial), VTOLModel(self.params, initial)
        assert_allclose(glider.step(controls, .2).as_vector(), full.step(controls, .2).as_vector(), atol=1e-10)

    def test_glider_never_evaluates_engines_and_ignores_all_engine_commands(self):
        initial = State(z=-30, u=12, left_motor_tilt=.4, right_motor_tilt=.5)
        controls = ControlInputs(elevator_deflection=.01)
        powered = replace(controls, left_propeller_speed=200, right_propeller_speed=300,
                          left_motor_tilt=1, right_motor_tilt=1.5)
        unpowered, commanded = [AerodynamicsOnlyModel(self.params, initial) for _ in range(2)]
        with patch("vtol_dynamics.testing_dynamics.thrust_forces", side_effect=AssertionError), \
             patch("vtol_dynamics.testing_dynamics.reaction_moment", side_effect=AssertionError), \
             patch("vtol_dynamics.testing_dynamics.tilt_rate", side_effect=AssertionError):
            assert_allclose(unpowered.step(controls, .2).as_vector(),
                            commanded.step(powered, .2).as_vector(), atol=0, rtol=0)
        assert_allclose(commanded.get_state().as_vector()[12:], initial.as_vector()[12:], atol=0)

    def test_glider_control_surfaces_are_active(self):
        initial = State(z=-30, u=12)
        for surface, rate in (("aileron_deflection", "p"), ("elevator_deflection", "q"),
                              ("rudder_deflection", "r")):
            with self.subTest(surface=surface):
                baseline, deflected = [AerodynamicsOnlyModel(self.params, initial) for _ in range(2)]
                a = baseline.step(ControlInputs(), .01)
                b = deflected.step(ControlInputs(**{surface: .1}), .01)
                self.assertGreater(abs(getattr(a, rate)-getattr(b, rate)), 1e-4)

    def test_nacelle_dynamics_remain_active_in_engine_models(self):
        params = replace(self.params, gravity=0, tilt_max_rate=100)
        controls = ControlInputs(left_motor_tilt=.6, right_motor_tilt=.3)
        for model_type in self.engine_models:
            with self.subTest(model=model_type.__name__):
                result = model_type(params, State()).step(controls, .2)
                expected = (1-np.exp(-.2/params.tilt_time_constant))*np.array([.3, .6])
                assert_allclose(result.as_vector()[12:], expected, rtol=1e-6)

    def test_terrain_available_for_translating_models(self):
        params = replace(self.params, aerodynamics=AerodynamicCoefficients(),
                         stall=StallParameters(enabled=False))
        for model_type in (BicopterTranslationalModel, BicopterFullModel, AerodynamicsOnlyModel):
            with self.subTest(model=model_type.__name__):
                model = model_type(params, State(z=-1, w=2), terrain_collision=True)
                state = model.step(ControlInputs(), 1)
                assert_allclose([state.z, state.w], [0, 0], atol=1e-9)

    def test_rotation_cannot_enable_terrain(self):
        with self.assertRaisesRegex(ValueError, "rotation-only"):
            BicopterRotationModel(self.params, State(), terrain_collision=True)
        model = BicopterRotationModel(self.params, State(z=1, w=2))
        with self.assertRaises(ValueError):
            model.terrain_collision = True
        self.assertFalse(model.terrain_collision)
        self.assertEqual(model.get_state(), State(z=1, w=2))

    def test_models_keep_separate_state_and_share_step_validation(self):
        for model_type in self.all_models:
            with self.subTest(model=model_type.__name__):
                initial = State(z=-10)
                first, second = model_type(self.params, initial), model_type(self.params, initial)
                self.assertEqual(first.step(ControlInputs(), 0), initial)
                first.step(ControlInputs(), .1)
                self.assertEqual(first.time, .1)
                self.assertEqual(second.time, 0)
                self.assertEqual(second.get_state(), initial)
                before = first.get_state()
                with self.assertRaises(ValueError):
                    first.step(ControlInputs(), -1)
                self.assertEqual(first.get_state(), before)
                self.assertEqual(first.time, .1)

    def test_rotation_derivative_does_not_evaluate_translational_gravity(self):
        state = State(z=-10, u=12)
        with patch("vtol_dynamics.testing_dynamics.gravity_acceleration", side_effect=AssertionError):
            derivative = bicopter_rotation_derivative(0, state.as_vector(), ControlInputs(40, 42), self.params)
        assert_allclose(derivative[:3], [0, 0, 0], atol=0)
        assert_allclose(derivative[6:9], [0, 0, 0], atol=0)


if __name__ == "__main__":
    unittest.main()
