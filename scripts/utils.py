import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
plt.rcParams["text.usetex"] = True

import numpy as np
import torch


def rollout2trajectory(rollout: dict) -> tuple[torch.Tensor, ...]:
    obs = []
    actions = []
    dones = []
    for timestep, data in rollout.items():
        obs.append(data["obs"].tolist())
        actions.append(data["action"].tolist())
        dones.append(data["done"].tolist())
    return torch.tensor(obs), torch.tensor(actions), torch.tensor(dones)

def compute_jacobian(policy: torch.jit.ScriptModule, x: torch.Tensor) -> torch.Tensor:
    x = x.detach().requires_grad_(True)
    J_func = torch.func.jacrev(lambda inp: policy(inp))
    J = torch.func.vmap(J_func)(x)
    return J.detach().squeeze(dim=1)

def compute_hessian(policy: torch.jit.ScriptModule, x: torch.Tensor) -> torch.Tensor:
    x = x.detach().requires_grad_(True)
    H_func = torch.func.hessian(lambda inp: policy(inp))
    H = torch.func.vmap(H_func)(x)
    return H.detach().squeeze(dim=1)

def compute_sensitivity(
    policy: torch.jit.ScriptModule,
    states: torch.Tensor,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, ...]:
    policy = policy.to(device)
    states = states.to(device)
    J = compute_jacobian(policy, states)
    C = torch.bmm(J.transpose(1, 2), J).mean(dim=0).cpu()
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
    policy = policy.to(device)
    states = states.to(device)
    J = compute_jacobian(policy, states)
    C = torch.bmm(J.transpose(1, 2), J).cpu()
    eigvals = torch.linalg.eigvalsh(C)
    eigvals = eigvals.flip(1)
    return C, eigvals

def compute_batch_multistep_sensitivity(
    policy: torch.jit.ScriptModule,
    states: torch.Tensor,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, ...]:
    T, N, dim = states.shape
    x = states.reshape(T * N, dim)
    C, eig = compute_multistep_sensitivity(policy, x, device)
    return C.view(T, N, dim, dim), eig.view(T, N, dim)

def compute_perturbation_error(
    policy: torch.jit.ScriptModule,
    states: torch.Tensor,
    eigenvectors: torch.Tensor,
    radii: list[float],
    env_idx: int = 0,
) -> tuple[torch.Tensor, ...]:
    n_eig = eigenvectors.shape[1]
    T = states.shape[0]

    errors = torch.zeros(n_eig, len(radii), T)
    errors_lin = torch.zeros_like(errors)

    for i in range(n_eig):
        v = eigenvectors[:, i]
        for j, a in enumerate(radii):
            a = radii[j]
            x = states[:, env_idx].squeeze()
            with torch.no_grad():
                diff = policy(x + a * v) - policy(x)
            errors[i, j] = diff.norm(dim=1)
            J = compute_jacobian(policy, x)
            errors_lin[i, j] = a * (J @ v).norm(dim=1)
    return errors, errors_lin

def compute_batch_perturbation_error(
    policy: torch.jit.ScriptModule,
    states: torch.Tensor,
    eigenvectors: torch.Tensor,
    radii: list[float],
) -> tuple[torch.Tensor, ...]:
    n_eig = eigenvectors.shape[1]
    dim = states.shape[-1]

    errors = torch.zeros(n_eig, len(radii))
    errors_lin = torch.zeros_like(errors)

    for i in range(n_eig):
        v = eigenvectors[:, i]
        for j, a in enumerate(radii):
            a = radii[j]
            x = states.reshape(-1, dim)
            with torch.no_grad():
                diff = policy(x + a * v) - policy(x)
            errors[i, j] = diff.norm(dim=1).mean(dim=0)
            J = compute_jacobian(policy, x)
            errors_lin[i, j] = a * (J @ v).norm(dim=1).mean(dim=0)
    return errors, errors_lin

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
    # fig.show()

def plot_1d(
    data: torch.Tensor,
    title: str,
    xlabel: str,
    ylabel: str,
    grid: bool = True,
    log: bool = False,
) -> None:
    fig, ax = plt.subplots()
    ax.plot(data, marker="o", linewidth=2)
    ax.set_title(f"{title}")
    ax.set_xticks(np.arange(data.shape[0]))
    ax.set_xlabel(f"{xlabel}")
    ax.set_ylabel(f"{ylabel}")
    if log:
        ax.set_yscale("log")
    ax.grid(grid)
    # fig.show()

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
    # fig.show()

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
    # fig.show()
    
def plot_perturbation_error(
    errors: torch.Tensor,
    errors_linear: torch.Tensor,
    eigenvectors: torch.Tensor,
    radii: list[float],
    log: bool = True,
) -> None:
    n_eig = eigenvectors.shape[1]

    fig, axes = plt.subplots(n_eig, 1, figsize=(10, 2.5 * max(1, n_eig)), sharex=True)
    if n_eig == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        handles = []
        labels = []
        for j, a in enumerate(radii):
            line, = ax.plot(errors[i, j], linewidth=2, label=f"a={a}")
            handles.append(line)
            labels.append(f"a={a}")
            ax.plot(errors_linear[i, j], linewidth=2, linestyle="--", color=line.get_color())
        if log:
            ax.set_yscale("log")
        ax.set_ylabel("Error")
        ax.set_title(f"Eigenvector {i}")
        ax.grid(True)
        # legend shows only solid (nonlinear) curves
        # ax.legend(handles, labels, ncol=min(4, len(radii)), loc="best")

    axes[-1].set_xlabel("Time [Step]")
    fig.tight_layout()
    # fig.show()


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

