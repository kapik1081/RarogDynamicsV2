"""Find symmetric stationary hover trim from a craft parameter JSON file."""

from vtol_dynamics.trim import hover_trim
from vtol_dynamics.trim_cli import load_parameters, parameter_parser, print_solution


def main(argv=None):
    parser = parameter_parser(__doc__)
    args = parser.parse_args(argv)
    params = load_parameters(parser, args.parameters)
    try:
        solution = hover_trim(params)
    except ValueError as error:
        parser.error(str(error))
    print("Hover trim (PDF 2.1; full-model equilibrium verified)")
    print_solution(solution)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
