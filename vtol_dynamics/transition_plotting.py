"""Two-dimensional projection of the sampled transition trim manifold."""

import numpy as np

from .transition import TransitionResult


def plot_transition(result: TransitionResult):
    """Return a figure; callers control display and saving."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    if result.points:
        z = np.array([p.coordinates for p in result.points])
        # Scatter preserves sampled support: filling a projected convex hull
        # would invent feasible cells across holes and disconnected branches.
        dots = ax.scatter(z[:, 0], np.rad2deg(z[:, 1]), c=np.rad2deg(z[:, 2]),
                          cmap="viridis", s=10, alpha=.8, linewidths=0,
                          label="Viable trim samples", rasterized=True)
        fig.colorbar(dots, ax=ax, label="Trim pitch [deg]")
        multiple = np.argwhere(result.branch_counts > 1)
        if len(multiple):
            ax.scatter(result.airspeeds[multiple[:, 0]],
                       np.rad2deg(result.tilts[multiple[:, 1]]), facecolors="none", edgecolors="#555555",
                       s=20, linewidths=.5, label="Multiple viable branches")
    if result.unresolved:
        unresolved = np.array(result.unresolved)
        within = ((unresolved[:, 1] >= result.constraints.tilt_min)
                  & (unresolved[:, 1] <= result.constraints.tilt_max))
        ax.scatter(unresolved[within, 0], np.rad2deg(unresolved[within, 1]),
                   marker="x", c="#b5b5b5", s=10, linewidths=.6,
                   label="Unresolved local continuation", zorder=1)
    if result.path:
        z = np.array([p.coordinates for p in result.path])
        ax.plot(z[:, 0], np.rad2deg(z[:, 1]), color="#d33b32", linewidth=2,
                label="Proposed steady-trim path", zorder=4)
        ax.scatter(z[[0, -1], 0], np.rad2deg(z[[0, -1], 1]), c="#d33b32",
                   marker="*", s=120, edgecolors="white", zorder=5)
        status = "Hover-to-cruise trim corridor found"
    else:
        status = "No hover-to-cruise path found at this resolution"
    if not result.points:
        ax.text(.5, .5, "No viable trims found under these limits",
                transform=ax.transAxes, ha="center", va="center")
    if result.constraints.tilt_min <= 0 <= result.constraints.tilt_max:
        ax.plot(result.cruise_interval, [0, 0], color="#d33b32", marker="s",
                markersize=5, linewidth=4, label="Requested cruise set", zorder=3)
    ax.set(xlabel="Airspeed [m/s]", ylabel="Common engine tilt [deg]",
           title="Transition trim manifold\n"+status,
           ylim=(np.rad2deg(result.constraints.tilt_min)-1,
                 np.rad2deg(result.constraints.tilt_max)+1),
           xlim=(-.02*result.airspeeds[-1], result.airspeeds[-1]*1.03))
    ax.grid(alpha=.2)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best", fontsize=8)
    return fig
