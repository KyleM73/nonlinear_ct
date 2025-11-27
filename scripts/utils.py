import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.axes import Axes

def rollout2trajectory(rollout: dict) -> tuple[torch.Tensor, torch.Tensor]:
    obs = []
    actions = []
    for timestep, data in rollout.items():
        obs.append(data["obs"].tolist())
        actions.append(data["action"].tolist())
        # if data["done"]: break
    return torch.tensor(obs), torch.tensor(actions)

def compute_jacobian(policy: torch.jit.ScriptModule, x: torch.Tensor) -> torch.Tensor:
    x = x.detach().requires_grad_(True)
    J = torch.autograd.functional.jacobian(lambda inp: policy(inp), x)
    return J.squeeze()

def compute_sensitivity(
    policy: torch.jit.ScriptModule,
    states: torch.Tensor,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, ...]:
    batch_size = states.shape[0]
    state_dim = states.shape[1]
    policy = policy.to(device)
    states = states.to(device)
    C = torch.zeros(state_dim, state_dim, device=device)
    for i in range(batch_size):
        x = states[i]
        J = compute_jacobian(policy, x)
        C += J.T @ J
    C = C.cpu() / batch_size
    # ascending
    eigvals, eigvecs = torch.linalg.eigh(C)
    # descending
    eigvals = eigvals.flip(0)
    eigvecs = eigvecs.flip(1)
    total = eigvals.sum().item()
    explained_energy = (eigvals.cumsum(0) / (total + 1e-6))
    input_contributions = eigvecs.square() @ eigvals
    input_activity = input_contributions / input_contributions.sum()
    return C, eigvals, eigvecs, explained_energy, input_activity

def compute_multistep_sensitivity(
    policy: torch.jit.ScriptModule,
    states: torch.Tensor,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, ...]:
    H = states.shape[0]
    policy = policy.to(device)
    states = states.to(device)
    C_list = []
    eig_list = []
    for i in range(H):
        x = states[i]
        J = compute_jacobian(policy, x)
        C = J.T @ J
        eigvals = torch.linalg.eigvalsh(C)
        eigvals = eigvals.flip(0)
        C_list.append(C.tolist())
        eig_list.append(eigvals.tolist())
    return torch.tensor(C_list), torch.tensor(eig_list)

def compute_batch_multistep_sensitivity(
    policy: torch.jit.ScriptModule,
    states: torch.Tensor,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, ...]:
    T, N, dim = states.shape
    C_tensor = torch.zeros(T, N, dim, dim)
    eig_tensor = torch.zeros(T, N, dim)
    for i in range(N):
        C, eig = compute_multistep_sensitivity(policy, states[:, i].squeeze(), device)
        C_tensor[:, i] = C
        eig_tensor[:, i] = eig
    return C_tensor, eig_tensor

def plot_matrix(matrix: torch.Tensor, title: str) -> None:
    dim = matrix.shape[0]
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="viridis", interpolation="nearest")
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(dim))
    ax.set_yticks(np.arange(dim))
    max_val = matrix.max()
    for i in range(dim):
        for j in range(dim):
            # Choose text color based on background intensity
            text_color = "white" if matrix[i, j] < max_val / 2 else "black"
            text = ax.text(
                j, i, f"{matrix[i, j]:.2f}",
                ha="center", va="center",
                color=text_color, fontsize=12 if dim < 15 else 8,
            )
    ax.set_title(f"{title}")
    fig.tight_layout()
    fig.show()

def plot_1d(
    data: torch.Tensor,
    title: str,
    xlabel: str,
    ylabel: str,
    grid: bool = True,
) -> None:
    fig, ax = plt.subplots()
    ax.plot(data, marker="o", linewidth=2)
    ax.set_title(f"{title}")
    ax.set_xticks(np.arange(data.shape[0]))
    ax.set_xlabel(f"{xlabel}")
    ax.set_ylabel(f"{ylabel}")
    ax.grid(grid)
    fig.show()

def plot_semilogy(
    data: torch.Tensor,
    title: str,
    xlabel: str,
    ylabel: str,
    grid: bool = True,
) -> None:
    fig, ax = plt.subplots()
    ax.semilogy(data, marker="o", linewidth=2)
    ax.set_title(f"{title}")
    ax.set_xticks(np.arange(data.shape[0]))
    ax.set_xlabel(f"{xlabel}")
    ax.set_ylabel(f"{ylabel}")
    ax.grid(grid)
    fig.show()

def plot_multistep_sensitivity(
    C_data: torch.Tensor,
    eig_data: torch.Tensor,
    grid: bool = True,
    log: bool = False,
) -> None:
    C_norm = C_data.norm(dim=(2,3))
    C_norm_mean = C_norm.mean(dim=1)
    C_norm_std = C_norm.std(dim=1) if C_data.shape[1] > 1 else 0.0

    eig_mean = eig_data.mean(dim=1)
    eig_std = eig_data.std(dim=1) if eig_data.shape[1] > 1 else 0.0

    T = np.arange(C_data.shape[0])

    fig, (ax_norm, ax_eig) = plt.subplots(2, 1, sharex=True)
    ax_norm.plot(C_norm_mean, linewidth=2)
    ax_norm.fill_between(
        T,
        C_norm_mean - C_norm_std,
        C_norm_mean + C_norm_std,
        alpha=0.2
    )
    ax_norm.set_ylabel("Magnitude")
    ax_norm.grid(grid)

    k = eig_mean.shape[1]
    handles = []
    labels = []
    for i in range(k):
        line, = ax_eig.plot(T, eig_mean[:, i], linewidth=2)
        if isinstance(eig_std, float | int):
            std = eig_std
        else:
            std = eig_std[:, i]
        ax_eig.fill_between(
            T,
            eig_mean[:, i] - std,
            eig_mean[:, i] + std,
            alpha=0.2
        )
        handles.append(line)
        labels.append(f"eig {i}")

    ax_eig.set_xlabel("Time [Step]")
    ax_eig.set_ylabel("Eigenvalue")
    ax_eig.grid(grid)
    if log:
        ax_eig.set_yscale("log")
    ax_eig.legend(
        handles, labels,
        ncol=min(4, k), loc="best"
    )

    fig.tight_layout()
    fig.show()
    

def plot_2d(trajectory: torch.Tensor) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots()

    max_extent = trajectory.abs().max().item()
    if max_extent == 0.0:
        max_extent = 1.0

    ax.plot(trajectory[:, 0], trajectory[:, 1], linewidth=2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.grid(True)

    return fig, ax

