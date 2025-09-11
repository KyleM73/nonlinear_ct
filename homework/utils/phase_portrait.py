from typing import Callable
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def phase_portrait(
    x_max: float,
    n_pts: int,
    dynamics: Callable,
    ics: list[list[float]] | None = None,
    t_max: float | None = None,
    file_name: str = "phase_portrait",
    save: bool = False,
    **kwargs,
) -> None:
    # Grid for vector field
    x1_vals = np.linspace(-x_max, x_max, n_pts)
    x2_vals = np.linspace(-x_max, x_max, n_pts)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # Compute derivatives on the grid
    DX1, DX2 = dynamics(t=None, state=[X1, X2], **kwargs)

    # Magnitude of the vector field
    speed = np.sqrt(DX1**2 + DX2**2)

    # Plot streamplot with color
    fig, ax = plt.subplots(figsize=(7, 6))
    strm = ax.streamplot(
        x=X1, y=X2, u=DX1, v=DX2,
        color=speed,
        linewidth=1,
        cmap="plasma",
        density=3.0,
        arrowsize=1.2
    )

    # Add colorbar for magnitude
    cbar = plt.colorbar(strm.lines, ax=ax)
    # cbar.set_label("Vector field magnitude")

    # Overlay sample trajectories
    if ics is not None:
        t_span = (0, t_max)
        t_eval = np.linspace(*t_span, 1000)

        for ic in ics:
            sol = solve_ivp(dynamics, t_span, ic, t_eval=t_eval)
            ax.plot(sol.y[0], sol.y[1], linewidth=2, label=f"IC={ic}")

    ax.set_title("Phase Portrait")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    # ax.legend()

    plt.tight_layout()
    if save:
        plt.savefig(f"{file_name}.png") 
    plt.show()