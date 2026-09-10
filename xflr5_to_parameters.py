"""Convert an XFLR5 plane polar text export into a simulation JSON template.

Usage: python xflr5_to_parameters.py T1.txt [-o aircraft.json] [--stall-at-extremes]
Only the Python standard library is required. See docs/xflr5_import.md.
"""

import argparse
from bisect import bisect_left
import json
import math
from pathlib import Path


MISSING = "provide-data"
SAMPLE = Path(__file__).resolve().parent / "examples" / "sample_parameters.json"
FORCES = {"lift": "CL", "drag": "CD", "sideforce": "CY"}
MOMENTS = {"roll_0": "Cl", "pitch_0": "Cm", "yaw_0": "Cn"}


def parse_polar(text):
    """Return a case-sensitive column header and finite numeric polar rows."""
    header = None
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        tokens = line.split()
        if not tokens:
            continue
        if header is None:
            if "alpha" in tokens and "Beta" in tokens:
                header = tokens
                if len(set(header)) != len(header):
                    raise ValueError(f"line {number}: duplicate column names")
            continue
        if tokens == header:  # Some exports repeat the table header.
            continue
        if len(tokens) != len(header):
            raise ValueError(f"line {number}: expected {len(header)} numeric columns")
        try:
            values = [float(token) for token in tokens]
        except ValueError as exc:
            raise ValueError(f"line {number}: invalid numeric polar row") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"line {number}: nonfinite polar value")
        rows.append(dict(zip(header, values)))
    if header is None or not rows:
        raise ValueError("no plane polar table found (expected alpha and Beta columns)")
    if not set(FORCES.values()).union(MOMENTS.values()).intersection(header):
        raise ValueError("polar contains no supported aerodynamic coefficients")
    if "QInf" in header and any(
        not math.isclose(row["QInf"], rows[0]["QInf"], rel_tol=1e-6, abs_tol=1e-9)
        for row in rows
    ):
        raise ValueError("varying QInf cannot be represented by an alpha/beta-only map")
    return header, rows


def at_zero(axis, values):
    """Interpolate at zero, without extrapolating or substituting a nearby angle."""
    if not axis[0] <= 0 <= axis[-1]:
        return None
    index = bisect_left(axis, 0)
    if axis[index] == 0:
        return values[index]
    low, high = axis[index - 1], axis[index]
    weight = -low / (high - low)
    return values[index - 1] * (1 - weight) + values[index] * weight


def slope_at_zero(axis, values):
    """Estimate the zero-angle slope using the nearest bracketing samples, per radian."""
    negative = [i for i, angle in enumerate(axis) if angle < 0]
    positive = [i for i, angle in enumerate(axis) if angle > 0]
    if not negative or not positive:
        return None
    low, high = negative[-1], positive[0]
    return (values[high] - values[low]) / math.radians(axis[high] - axis[low])


def parameter_template():
    """Use the sample's schema, but never copy its illustrative physical values."""
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    return {
        name: {key: MISSING for key in value} if isinstance(value, dict) else MISSING
        for name, value in sample.items()
    }


def convert_polar(text, *, stall_at_extremes=False):
    """Create parameter data; missing physical parameters remain explicit placeholders."""
    header, rows = parse_polar(text)
    alphas = sorted({row["alpha"] for row in rows})
    betas = sorted({row["Beta"] for row in rows})
    grid = {}
    for row in rows:
        key = (row["alpha"], row["Beta"])
        if key in grid:
            raise ValueError(f"duplicate operating point alpha={key[0]}, beta={key[1]}")
        grid[key] = row
    if len(grid) != len(alphas) * len(betas):
        raise ValueError("incomplete alpha/beta grid; export a rectangular polar table")

    result = parameter_template()
    aero = result["aerodynamics"]
    for parameter, column in FORCES.items():
        if column in header:
            aero[parameter] = {
                "angle_unit": "deg",
                "bounds": "clamp",
                "alphas": alphas,
                "betas": betas,
                "values": [[grid[alpha, beta][column] for alpha in alphas] for beta in betas],
            }

    # Moment intercepts are defined at alpha=beta=0, where wind/body axes coincide.
    # Cm's alpha slope at beta=0 is unchanged by rotation about the pitch axis.
    for parameter, column in MOMENTS.items():
        if column not in header:
            continue
        zero_beta = [at_zero(betas, [grid[alpha, beta][column] for beta in betas])
                     for alpha in alphas]
        if any(value is None for value in zero_beta):
            continue
        baseline = at_zero(alphas, zero_beta)
        if baseline is not None:
            aero[parameter] = baseline
        if column == "Cm":
            slope = slope_at_zero(alphas, zero_beta)
            if slope is not None:
                aero["pitch_alpha"] = slope

    if stall_at_extremes:
        if not alphas[0] < 0 < alphas[-1]:
            raise ValueError("--stall-at-extremes requires negative and positive alpha samples")
        result["stall"].update(
            enabled=True,
            alpha_negative=math.radians(alphas[0]),
            alpha_positive=math.radians(alphas[-1]),
        )
    return result


def parameter_counts(parameters):
    """Count each coefficient map as one simulation parameter."""
    slots = []
    for value in parameters.values():
        slots.extend(value.values() if isinstance(value, dict) else [value])
    return sum(value != MISSING for value in slots), len(slots)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xflr_file", type=Path, help="XFLR5 plane polar text export")
    parser.add_argument("-o", "--output", type=Path, help="output JSON (default: input with .json suffix)")
    parser.add_argument("--stall-at-extremes", action="store_true",
                        help="enable stall and use min/max alpha as stall thresholds")
    args = parser.parse_args(argv)
    output = args.output if args.output is not None else args.xflr_file.with_suffix(".json")
    try:
        if output.resolve() == args.xflr_file.resolve() or (
            output.exists() and args.xflr_file.exists() and output.samefile(args.xflr_file)
        ):
            raise ValueError("output must differ from the input file")
        result = convert_polar(args.xflr_file.read_text(encoding="utf-8-sig"),
                               stall_at_extremes=args.stall_at_extremes)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    filled, total = parameter_counts(result)
    print(f"Wrote {output}: {filled}/{total} parameter slots filled; "
          f"{total - filled} require data.")
    if args.stall_at_extremes:
        print("Stall thresholds assumed from alpha extremes; stall.enabled set to true.")
    print('Replace all "provide-data" values before loading this file in the simulation.')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
