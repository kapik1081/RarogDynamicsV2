"""Equation-level and integration regressions; run with unittest discovery."""

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

from vtol_dynamics import (AerodynamicCoefficients, CoefficientMap, ControlInputs, Inertia,
                           ModelParameters, StallParameters, State, VTOLModel)
from vtol_dynamics.aerodynamics import (air_data, aerodynamic_force, aerodynamic_moment,
                                       force_coefficients, stall_blend)
from vtol_dynamics.dynamics import (angular_acceleration, calculate_loads, state_derivative,
                                    translational_acceleration)
from vtol_dynamics.kinematics import body_to_ned, euler_rates, gravity_acceleration, wind_to_body
from vtol_dynamics.propulsion import (propeller_thrust, propeller_torque, reaction_moment,
                                     thrust_forces, thrust_moment, tilt_rate)
from vtol_dynamics.terrain import contact_derivative, normal_acceleration, project_to_terrain


def parameters(**changes):
    base = ModelParameters(mass=2, inertia=Inertia(0.3, 0.5, 0.7, 0.02),
                           wing_area=0.8, wingspan=2, mean_chord=0.4,
                           propeller_diameter=0.4, CT=0.1, Kq=0.01,
                           engine_span=0.5, engine_height=-0.1,
                           tilt_time_constant=0.2, tilt_max_rate=2,
                           stall=StallParameters(enabled=False))
    return replace(base, **changes)


class MapTests(unittest.TestCase):
    def setUp(self):
        self.table = CoefficientMap([-2, 0, 5], [-2, 0, 2],
                                    [[-0.55, 0.25, 1.25], [-0.85, 0.85, 2.25],
                                     [-0.55, 0.25, 1.25]], angle_unit="deg")

    def test_user_grid_orientation_and_bilinear_interpolation(self):
        for j, beta in enumerate(self.table.betas):
            for i, alpha in enumerate(self.table.alphas):
                self.assertAlmostEqual(self.table(*np.deg2rad([alpha, beta])), self.table.values[j][i])
        self.assertAlmostEqual(self.table(*np.deg2rad([-1, -1])), -0.075)
        self.assertAlmostEqual(self.table(*np.deg2rad([2.5, 1])), 1.15)

    def test_nonuniform_axes_and_units(self):
        table = CoefficientMap([0, 1, 4], [-2, 3], [[-4, -1, 8], [6, 9, 18]])
        self.assertAlmostEqual(table(2, 1), 8)  # f(alpha,beta)=3alpha+2beta

    def test_boundaries(self):
        self.assertAlmostEqual(self.table(*np.deg2rad([20, 20])), 1.25)
        with self.assertRaises(ValueError):
            replace(self.table, bounds="raise")(1, 1)

    def test_singleton_axes(self):
        self.assertEqual(CoefficientMap([0], [0], [[2]])(2, 3), 2)
        self.assertEqual(CoefficientMap([0], [0, 2], [[2], [4]])(1, 1), 3)

    def test_invalid_maps(self):
        for changes in ({"alphas": [0, 0, 1]}, {"betas": [2, 0, -2]},
                        {"values": [[1]]}, {"alphas": []}, {"bounds": "oops"},
                        {"angle_unit": "rpm"}, {"alphas": [0, 1, np.nan]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.table, **changes)
        with self.assertRaises(ValueError):
            self.table(np.nan, 0)

    def test_copies_source_data(self):
        values = [[1, 2], [3, 4]]
        table = CoefficientMap([0, 1], [0, 1], values)
        values[0][0] = 99
        self.assertEqual(table(0, 0), 1)


class PhysicsTests(unittest.TestCase):
    def test_frame_rotations(self):
        rotation = body_to_ned(0.3, -0.2, 0.7)
        assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-15)
        assert_allclose(body_to_ned(0, 0, np.pi/2) @ [1, 0, 0], [0, 1, 0], atol=1e-15)
        velocity = np.array([10., 2., 3.])
        air = air_data(velocity, 1.2)
        assert_allclose(wind_to_body(air.alpha, air.beta) @ [air.airspeed, 0, 0], velocity)

    def test_gravity_and_euler_rates(self):
        roll, pitch, yaw = 0.2, -0.4, 0.3
        assert_allclose(gravity_acceleration(roll, pitch, 9.8),
                        body_to_ned(roll, pitch, yaw).T @ [0, 0, 9.8])
        for roll, pitch in ((0, 0), (np.pi, 0), (.8, -.7), (-1.2, 1.1)):
            gravity_body = gravity_acceleration(roll, pitch, 9.8)
            assert_allclose(body_to_ned(roll, pitch, .3) @ gravity_body, [0, 0, 9.8], atol=1e-14)
            self.assertAlmostEqual(np.linalg.norm(gravity_body), 9.8)
        assert_allclose(euler_rates(0, 0, [1, 2, 3]), [1, 2, 3])
        with self.assertRaises(ValueError):
            euler_rates(0, np.pi/2, [0, 0, 0])

    def test_air_data_and_stall(self):
        air = air_data([3, 0, 4], 2)
        self.assertEqual(air.airspeed, 5)
        self.assertEqual(air.dynamic_pressure, 25)
        self.assertAlmostEqual(air.alpha, np.arctan2(4, 3))
        stall = StallParameters()
        self.assertLess(stall_blend(0, stall), 0.001)
        for alpha in [-np.pi/2, np.pi/2]:
            self.assertAlmostEqual(stall_blend(alpha, stall), 1)
            params = parameters(stall=stall)
            coeffs = force_coefficients(air_data([0, 0, np.sign(alpha)], 1), params)
            assert_allclose(coeffs[:2], [0, stall.drag_max], atol=1e-15)
        weights = [stall_blend(a, stall) for a in np.linspace(-np.pi, np.pi, 101)]
        self.assertTrue(all(0 <= value <= 1 for value in weights))

    def test_aerodynamic_force_components(self):
        params = parameters(aerodynamics=AerodynamicCoefficients(lift=0.7, drag=0.1, sideforce=-0.2))
        air = air_data([15, 2, 3], params.air_density)
        lift, drag, side = air.dynamic_pressure*params.wing_area*np.array([0.7, 0.1, -0.2])
        a, b = air.alpha, air.beta
        expected = [-drag*np.cos(a)*np.cos(b)-side*np.cos(a)*np.sin(b)+lift*np.sin(a),
                    -drag*np.sin(b)+side*np.cos(b),
                    -drag*np.sin(a)*np.cos(b)-side*np.sin(a)*np.sin(b)-lift*np.cos(a)]
        assert_allclose(aerodynamic_force(air, params), expected)

    def test_moment_matches_uncancelled_stability_equations(self):
        coeffs = AerodynamicCoefficients(roll_0=.01, roll_beta=-.2, roll_p=-.5, roll_r=.2,
                                         roll_aileron=.3, roll_rudder=.1, pitch_0=.02,
                                         pitch_alpha=-.7, pitch_q=-4, pitch_elevator=-.8,
                                         yaw_0=.03, yaw_beta=.2, yaw_p=-.1, yaw_r=-.3,
                                         yaw_aileron=-.02, yaw_rudder=.12)
        params = parameters(aerodynamics=coeffs)
        air = air_data([14, 2, 3], params.air_density)
        controls = ControlInputs(aileron_deflection=.1, elevator_deflection=-.2, rudder_deflection=.3)
        rates = np.array([.2, -.3, .4])
        ph, qh, rh = rates*np.array([params.wingspan, params.mean_chord, params.wingspan])/(2*air.airspeed)
        expected = air.dynamic_pressure*params.wing_area*np.array([
            params.wingspan*(.01-.2*air.beta-.5*ph+.2*rh+.3*.1+.1*.3),
            params.mean_chord*(.02-.7*air.alpha-4*qh-.8*(-.2)),
            params.wingspan*(.03+.2*air.beta-.1*ph-.3*rh-.02*.1+.12*.3)])
        assert_allclose(aerodynamic_moment(air, rates, controls, params), expected)

    def test_maps_are_used_for_forces_and_derivatives(self):
        table = CoefficientMap([-1, 1], [-1, 1], [[0, 2], [2, 4]])
        params = parameters(aerodynamics=AerodynamicCoefficients(lift=table, roll_aileron=table))
        air = air_data([10, 1, 2], params.air_density)
        expected_coefficient = 2+air.alpha+air.beta
        self.assertAlmostEqual(force_coefficients(air, params)[0], expected_coefficient)
        moment = aerodynamic_moment(air, [0, 0, 0], ControlInputs(aileron_deflection=.2), params)
        self.assertAlmostEqual(moment[0], air.dynamic_pressure*params.wing_area*params.wingspan*.2*expected_coefficient)

    def test_zero_and_near_zero_airspeed(self):
        params = parameters(aerodynamics=AerodynamicCoefficients(lift=1, pitch_q=-4))
        for speed in [0, 1e-12]:
            air = air_data([speed, 0, 0], params.air_density)
            force = aerodynamic_force(air, params)
            moment = aerodynamic_moment(air, [1, 2, 3], ControlInputs(), params)
            self.assertTrue(np.isfinite(moment).all())
            self.assertLess(np.linalg.norm(moment), 1e-10)
            self.assertLess(np.linalg.norm(force), 1e-20)

    def test_propeller_geometry_and_reaction_signs(self):
        params = parameters()
        controls = ControlInputs(left_propeller_speed=60, right_propeller_speed=80)
        state = State(left_motor_tilt=.4, right_motor_tilt=.8)
        left, right = [propeller_thrust(n, params) for n in (60, 80)]
        forces = thrust_forces(state, controls, params)
        assert_allclose(forces.sum(axis=0), [left*np.cos(.4)+right*np.cos(.8), 0,
                                             -left*np.sin(.4)-right*np.sin(.8)])
        assert_allclose(thrust_moment(forces, params), [params.engine_span*(left*np.sin(.4)-right*np.sin(.8)),
                        params.engine_height*(left*np.cos(.4)+right*np.cos(.8)),
                        params.engine_span*(left*np.cos(.4)-right*np.cos(.8))])
        ql, qr = [propeller_torque(n, params) for n in (60, 80)]
        assert_allclose(reaction_moment(state, controls, params),
                        [qr*np.cos(.8)-ql*np.cos(.4), 0, qr*np.sin(.8)-ql*np.sin(.4)])

    def test_rigid_body_equations_and_coupling(self):
        inertia = Inertia(.3, .5, .7, .02)
        p, q, r = rates = np.array([.2, .3, -.4])
        moment = np.array([.1, -.2, .3])
        pd, qd, rd = angular_acceleration(rates, moment, inertia)
        reconstructed = [inertia.Ixx*pd-inertia.Ixz*rd+(inertia.Izz-inertia.Iyy)*q*r-inertia.Ixz*p*q,
                         inertia.Iyy*qd+(inertia.Ixx-inertia.Izz)*p*r+inertia.Ixz*(p*p-r*r),
                         -inertia.Ixz*pd+inertia.Izz*rd+(inertia.Iyy-inertia.Ixx)*p*q+inertia.Ixz*q*r]
        assert_allclose(reconstructed, moment)
        assert_allclose(translational_acceleration([2, 3, 4], rates, [1, 2, 3], 2),
                        [r*3-q*4+.5, p*4-r*2+1, q*2-p*3+1.5])
        torque_free = angular_acceleration(rates, [0, 0, 0], inertia)
        self.assertAlmostEqual(float(rates @ inertia.matrix() @ torque_free), 0)

    def test_actuator_rate_limit(self):
        self.assertEqual(tilt_rate(0, 2, .2, 1), 1)
        self.assertEqual(tilt_rate(2, 0, .2, 1), -1)
        self.assertAlmostEqual(tilt_rate(.9, 1, .2, 1), .5)


class IntegrationTests(unittest.TestCase):
    def test_free_fall_and_dt_zero(self):
        params = parameters()
        model = VTOLModel(params, State(z=-10, u=2, w=1))
        self.assertEqual(model.step(ControlInputs(), 0), model.get_state())
        state = model.step(ControlInputs(), .5)
        assert_allclose([state.x, state.z, state.w], [1, -10+.5+.5*params.gravity*.5**2, 1+params.gravity*.5])
        self.assertEqual(model.time, .5)

    def test_hover(self):
        params = parameters()
        n = np.sqrt(params.mass*params.gravity/(2*params.CT*params.air_density*params.propeller_diameter**4))
        initial = State(z=-2, left_motor_tilt=np.pi/2, right_motor_tilt=np.pi/2)
        model = VTOLModel(params, initial)
        controls = ControlInputs(n, n, left_motor_tilt=np.pi/2, right_motor_tilt=np.pi/2)
        assert_allclose(model.step(controls, 2).as_vector(), initial.as_vector(), atol=1e-12)

    def test_actuator_analytic_response_and_state_order(self):
        model = VTOLModel(parameters(tilt_max_rate=100), State())
        state = model.step(ControlInputs(left_motor_tilt=1, right_motor_tilt=.5), .3)
        response = 1-np.exp(-.3/.2)
        assert_allclose([state.right_motor_tilt, state.left_motor_tilt], [.5*response, response], rtol=1e-6)
        self.assertEqual(state.as_vector()[12], state.right_motor_tilt)

    def test_rate_limited_actuator_then_exponential(self):
        model = VTOLModel(parameters(tilt_time_constant=.2, tilt_max_rate=1), State(),
                          rtol=1e-10, atol=1e-12, max_step=.02)
        controls = ControlInputs(left_motor_tilt=1)
        self.assertAlmostEqual(model.step(controls, .4).left_motor_tilt, .4, places=7)
        state = model.step(controls, .6)
        self.assertAlmostEqual(state.left_motor_tilt, 1-.2*np.exp(-1), places=6)

    def test_step_subdivision_convergence(self):
        params = ModelParameters.from_json(Path(__file__).parents[1]/"examples/sample_parameters.json")
        state = State(z=-30, u=12, v=.3, w=.5, p=.01, q=.02, r=-.01,
                      left_motor_tilt=.2, right_motor_tilt=.25)
        controls = ControlInputs(40, 41, elevator_deflection=.01, aileron_deflection=.005,
                                 rudder_deflection=.005, left_motor_tilt=.3, right_motor_tilt=.35)
        one, many = VTOLModel(params, state), VTOLModel(params, state)
        one.step(controls, .2)
        for _ in range(20):
            many.step(controls, .01)
        assert_allclose(one.get_state().as_vector(), many.get_state().as_vector(), rtol=2e-6, atol=1e-8)

    def test_terrain_impact_and_rest(self):
        model = VTOLModel(parameters(), State(z=-1, u=2, w=3), terrain_collision=True)
        state = model.step(ControlInputs(), 2)
        assert_allclose([state.x, state.z, state.w], [4, 0, 0], atol=1e-9)
        self.assertEqual(model.step(ControlInputs(), 10).z, 0)

    def test_terrain_projection_at_tilted_attitude(self):
        state = State(z=1, roll=.4, pitch=.3, yaw=.2, u=2, v=3, w=4)
        rotation = body_to_ned(state.roll, state.pitch, state.yaw)
        before = rotation @ state.as_vector()[6:9]
        result = project_to_terrain(state.as_vector())
        after = rotation @ result[6:9]
        assert_allclose(after, [*before[:2], 0], atol=1e-14)
        self.assertEqual(result[2], 0)
        self.assertEqual(state.z, 1)

    def test_contact_derivative_cancels_earth_normal_acceleration(self):
        state = State(roll=.3, pitch=.2, yaw=.1, u=2, p=.2, q=.1)
        vector = project_to_terrain(state.as_vector())
        derivative = state_derivative(0, vector, ControlInputs(), parameters())
        constrained = contact_derivative(vector, derivative)
        self.assertAlmostEqual(normal_acceleration(vector, constrained), 0)
        self.assertEqual(constrained[2], 0)

    def test_rotated_ground_contact_with_exact_gravity(self):
        params = parameters()
        initial = State(roll=.4, pitch=.3, yaw=.2)
        model = VTOLModel(params, initial, terrain_collision=True)
        assert_allclose(model.step(ControlInputs(), 2).as_vector(), initial.as_vector(), atol=1e-12)

    def test_terrain_toggle_and_upward_velocity(self):
        model = VTOLModel(parameters(), State(z=1, w=2))
        self.assertGreater(model.step(ControlInputs(), .1).z, 0)
        model.terrain_collision = True
        self.assertEqual(model.get_state().z, 0)
        self.assertEqual(model.get_state().w, 0)
        model.terrain_collision = False
        self.assertGreater(model.step(ControlInputs(), .1).z, 0)
        upward = VTOLModel(parameters(), State(w=-2), terrain_collision=True)
        self.assertLess(upward.step(ControlInputs(), .1).z, 0)
        self.assertEqual(upward.step(ControlInputs(), 1).z, 0)

    def test_takeoff_and_liftoff_during_same_step(self):
        params = parameters(engine_height=0, Kq=0)
        n = np.sqrt(params.mass*params.gravity/(params.CT*params.air_density*params.propeller_diameter**4))
        controls = ControlInputs(n, n, left_motor_tilt=np.pi/2, right_motor_tilt=np.pi/2)
        for initial_tilt in [0, np.pi/2]:
            with self.subTest(tilt=initial_tilt):
                model = VTOLModel(params, State(left_motor_tilt=initial_tilt, right_motor_tilt=initial_tilt),
                                  terrain_collision=True, max_step=.02)
                state = model.step(controls, 1)
                self.assertLess(state.z, -.1)
                self.assertLess(state.w, 0)

    def test_impact_then_takeoff_in_same_step(self):
        params = parameters(engine_height=0, Kq=0)
        n = np.sqrt(params.mass*params.gravity/(params.CT*params.air_density*params.propeller_diameter**4))
        controls = ControlInputs(n, n, left_motor_tilt=np.pi/2, right_motor_tilt=np.pi/2)
        model = VTOLModel(params, State(z=-.01, w=1), terrain_collision=True, max_step=.01)
        state = model.step(controls, 1)
        self.assertLess(state.z, -.1)

    def test_zero_force_ground_equilibrium(self):
        model = VTOLModel(parameters(gravity=0), State(), terrain_collision=True)
        assert_allclose(model.step(ControlInputs(), 1).as_vector(), np.zeros(14))

    def test_state_is_immutable_and_failed_step_is_atomic(self):
        model = VTOLModel(parameters(), State(pitch=np.pi/2))
        with self.assertRaises(FrozenInstanceError):
            model.get_state().z = 10
        array = model.get_state().as_vector()
        array[2] = 10
        self.assertEqual(model.get_state().z, 0)
        with self.assertRaises(ValueError):
            model.step(ControlInputs(), 1)
        self.assertEqual(model.time, 0)
        for dt in [-1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                model.step(ControlInputs(), dt)


class ParameterTests(unittest.TestCase):
    def test_load_sample_json(self):
        params = ModelParameters.from_json(Path(__file__).parents[1]/"examples/sample_parameters.json")
        self.assertEqual(params.mass, 5)
        self.assertAlmostEqual(params.aerodynamics.lift(0, 0), .2)
        self.assertIsInstance(params.aerodynamics.roll_aileron, CoefficientMap)

    def test_invalid_parameters(self):
        for changes in ({"mass": 0}, {"tilt_time_constant": 0}, {"CT": -1},
                        {"engine_height": np.nan}, {"gravity": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                parameters(**changes)
        with self.assertRaises(ValueError):
            Inertia(1, 1, 1, 2)
        with self.assertRaises(ValueError):
            ControlInputs(left_propeller_speed=-1)
        with self.assertRaises(ValueError):
            State(w=np.nan)
        with self.assertRaises(ValueError):
            StallParameters(transition_width=0)

    def test_unknown_json_fields_fail(self):
        path = Path(__file__).parents[1]/"examples/sample_parameters.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["typo_mass"] = 4
        with tempfile.TemporaryDirectory() as directory:
            bad_path = Path(directory)/"parameters.json"
            bad_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(TypeError):
                ModelParameters.from_json(bad_path)


if __name__ == "__main__":
    unittest.main()
