"""Small shared console helpers for the trim-finding scripts."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .parameters import ModelParameters
from .trim import TrimSolution


def finite_number(value):
    number = float(value)
    if not np.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def nonnegative_number(value):
    number = finite_number(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def positive_number(value):
    number = finite_number(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parameter_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("parameters", type=Path, help="Craft static-parameter JSON file")
    return parser


def load_parameters(parser, path):
    try:
        return ModelParameters.from_json(path)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(f"Cannot load craft parameters: {error}")


def print_solution(solution: TrimSolution):
    print(f"Alpha / pitch: {np.rad2deg(solution.alpha):.9g} deg ({solution.alpha:.12g} rad)")
    print(f"Airspeed: {solution.airspeed:.9g} m/s")
    print(f"Total thrust P: {solution.total_thrust:.9g} N")
    print(f"Thrust per engine: {solution.total_thrust/2:.9g} N")
    print(f"Speed per engine N: {solution.controls.right_propeller_speed:.9g} rev/s = {solution.engine_rpm:.9g} RPM")
    print(f"Tilt per engine: {np.rad2deg(solution.controls.right_motor_tilt):.9g} deg")
    for name in ("elevator_deflection", "aileron_deflection", "rudder_deflection"):
        value = getattr(solution.controls, name)
        print(f"{name}: {np.rad2deg(value):.9g} deg ({value:.12g} rad)")
    print(f"Max absolute steady-state derivative residual: {solution.acceleration_residual:.3g}")
    print("State (SI units, angles in rad; x=y=z=yaw=0 reference):")
    for name, value in asdict(solution.state).items():
        print(f"  {name} = {value:.12g}")
    print("ControlInputs (speeds in rev/s, angles in rad):")
    for name, value in asdict(solution.controls).items():
        print(f"  {name} = {value:.12g}")
