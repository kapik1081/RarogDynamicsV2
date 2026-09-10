"""Immutable alpha/beta tables; rows are beta, columns are alpha."""

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class CoefficientMap:
    """Bilinear lookup with explicit angle units and boundary behavior.

    Constructor axes use ``angle_unit``. Calls always accept radians.
    Outside the grid, ``clamp`` holds the closest edge value; ``raise``
    rejects the query. A single axis entry means constant along that axis.
    Lists/arrays are copied into tuples so callers cannot mutate a table.
    """

    alphas: tuple[float, ...]
    betas: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    angle_unit: Literal["rad", "deg"] = "rad"
    bounds: Literal["clamp", "raise"] = "clamp"

    def __post_init__(self):
        if self.angle_unit not in ("rad", "deg"):
            raise ValueError("angle_unit must be 'rad' or 'deg'")
        if self.bounds not in ("clamp", "raise"):
            raise ValueError("bounds must be 'clamp' or 'raise'")
        for name in ("alphas", "betas"):
            axis = np.asarray(getattr(self, name), dtype=float)
            if axis.ndim != 1 or not axis.size or not np.all(np.isfinite(axis)):
                raise ValueError(f"{name} must be a nonempty, finite 1-D axis")
            if np.any(np.diff(axis) <= 0):
                raise ValueError(f"{name} must be strictly increasing")
            object.__setattr__(self, name, tuple(float(x) for x in axis))
        values = np.asarray(self.values, dtype=float)
        if values.shape != (len(self.betas), len(self.alphas)):
            raise ValueError("values must have shape (len(betas), len(alphas))")
        if not np.all(np.isfinite(values)):
            raise ValueError("coefficient values must be finite")
        object.__setattr__(self, "values", tuple(tuple(float(x) for x in row) for row in values))

    def __call__(self, alpha: float, beta: float) -> float:
        if not np.isfinite(alpha) or not np.isfinite(beta):
            raise ValueError("alpha and beta must be finite")
        if self.angle_unit == "deg":
            alpha, beta = np.rad2deg([alpha, beta])
        if self.bounds == "raise":
            if not (self.alphas[0] <= alpha <= self.alphas[-1]
                    and self.betas[0] <= beta <= self.betas[-1]):
                raise ValueError("alpha/beta query is outside the coefficient map")
        # Interpolate in alpha in each beta row, then between the beta rows.
        rows = [np.interp(alpha, self.alphas, row) for row in self.values]
        return float(np.interp(beta, self.betas, rows))

    @classmethod
    def from_dict(cls, data: dict) -> "CoefficientMap":
        return cls(**data)


Coefficient = float | CoefficientMap


def evaluate(coefficient: Coefficient, alpha: float, beta: float) -> float:
    """Evaluate either a table or a constant (a useful special case)."""
    return coefficient(alpha, beta) if isinstance(coefficient, CoefficientMap) else float(coefficient)
