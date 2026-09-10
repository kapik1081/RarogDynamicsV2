"""Ideal symmetric hover using the illustrative JSON, not calibrated hardware."""

from pathlib import Path
import math

from vtol_dynamics import ControlInputs, ModelParameters, State, VTOLModel


def main():
    params = ModelParameters.from_json(Path(__file__).with_name("sample_parameters.json"))
    hover_speed = math.sqrt(params.mass*params.gravity /
                           (2*params.CT*params.air_density*params.propeller_diameter**4))
    model = VTOLModel(params, State(z=-2, left_motor_tilt=math.pi/2, right_motor_tilt=math.pi/2),
                      terrain_collision=True)
    controls = ControlInputs(left_propeller_speed=hover_speed, right_propeller_speed=hover_speed,
                             left_motor_tilt=math.pi/2, right_motor_tilt=math.pi/2)
    for _ in range(100):
        model.step(controls, 0.01)
    print(f"Hover speed: {hover_speed:.3f} rev/s ({hover_speed*60:.1f} RPM)")
    print(f"Time: {model.time:.2f} s")
    print(model.get_state())


if __name__ == "__main__":
    main()
