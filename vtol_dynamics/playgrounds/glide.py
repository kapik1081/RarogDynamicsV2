"""Unpowered glide from specified body velocities and otherwise zero state."""

from ..testing import AerodynamicsOnlyModel
from ..types import ControlInputs, State
from .cli import common_parser, execute, finite_number
from .simulation import simulate


def run(parameters, simulation_time, u, v, w, *, dt=.02):
    """Initial body velocities in m/s; surfaces and all engine commands stay zero."""
    model = AerodynamicsOnlyModel(parameters, State(u=u, v=v, w=w))
    return simulate(model, ControlInputs(), simulation_time, dt, title="Unpowered glide")


def main(argv=None):
    parser = common_parser(__doc__)
    for name, axis in (("u", "x"), ("v", "y"), ("w", "z")):
        parser.add_argument(f"--{name}", required=True, type=finite_number,
                            help=f"Initial body {axis} velocity [m/s]")
    args = parser.parse_args(argv)
    return execute(parser, args, lambda params: run(params, args.simulation_time, args.u, args.v, args.w,
                                                    dt=args.dt))


if __name__ == "__main__":
    main()
