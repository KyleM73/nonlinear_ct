import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
plt.rcParams["text.usetex"] = True

import numpy as np
import torch
from typing import Literal


def rollout2trajectory(rollout: dict) -> tuple[torch.Tensor, ...]:
    obs = []
    actions = []
    dones = []
    values = []
    critic_obs = []
    for timestep, data in rollout.items():
        obs.append(data["obs"].tolist())
        actions.append(data["action"].tolist())
        dones.append(data["done"].tolist())
        critic_obs.append(data["critic_obs"].tolist())
        values.append(data["value"].tolist())
    return (
        torch.tensor(obs), torch.tensor(actions), torch.tensor(dones),
        torch.tensor(critic_obs), torch.tensor(values),
    )

def subsample(
    *tensors: torch.Tensor,
    method: Literal["random", "farthest_point"] = "random",
    n_samples: int | None = None,
    min_dist: float | None = None,
    seed: int = 0,
    presample: int | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, ...]:
    """Subsample tensors to reduce density for plotting.

    Args:
        *tensors: Tensors to subsample (must have same first dimension).
        method: "random" for uniform random sampling, "farthest_point" for
            farthest point sampling which maintains spatial coverage.
        n_samples: Target number of samples (used by both methods).
        min_dist: Minimum distance between points (only used by farthest_point).
            If provided, overrides n_samples for farthest_point method.
        seed: Random seed for reproducibility.
        presample: For farthest_point, randomly presample to this size first
            to speed up computation on very large datasets. Default: 10 * n_samples.
        device: Device for distance computation ("cpu" or "cuda").

    Returns:
        Tuple of subsampled tensors with the same ordering.
    """
    n_total = tensors[0].shape[0]
    torch.manual_seed(seed)

    if method == "random":
        n_samples = n_samples or n_total // 10
        n_samples = min(n_samples, n_total)
        indices = torch.randperm(n_total)[:n_samples]

    elif method == "farthest_point":
        n_samples = n_samples or n_total // 10
        n_samples = min(n_samples, n_total)

        # Presample for speed on large datasets
        presample = presample or min(n_total, 10 * n_samples)
        if presample < n_total:
            presample_idx = torch.randperm(n_total)[:presample]
            working_tensors = tuple(t[presample_idx] for t in tensors)
            n_working = presample
        else:
            presample_idx = None
            working_tensors = tensors
            n_working = n_total

        # Use first tensor for distance computation
        coords = working_tensors[0]
        if coords.dim() == 1:
            coords = coords.unsqueeze(1)
        coords = coords.to(device)

        # Initialize
        selected = [torch.randint(n_working, (1,)).item()]
        min_distances = torch.full((n_working,), float("inf"), device=device)

        while True:
            # Update minimum distances
            last_selected = coords[selected[-1]]
            dists = (coords - last_selected).norm(dim=-1)
            min_distances = torch.minimum(min_distances, dists)

            # Check stopping criteria
            if min_dist is not None:
                if min_distances.max() < min_dist:
                    break
            elif len(selected) >= n_samples:
                break

            # Select farthest point
            next_idx = min_distances.argmax().item()
            selected.append(next_idx)

        indices = torch.tensor(selected)

        # Map back to original indices if presampled
        if presample_idx is not None:
            indices = presample_idx[indices]

    else:
        raise ValueError(f"Unknown method: {method}")

    return tuple(t[indices] for t in tensors)


def compute_state_statistics(x: torch.Tensor, decimals: int = 4) -> None:
    dim = x.shape[-1]
    x = x.view(-1, dim)
    for i in range(dim):
        mean = x[..., i].mean(dim=0).item()
        std = x[..., i].std(dim=0).item()
        print(f"State {i}: mean {mean:.{decimals}f} std {std:.{decimals}f}")

def compute_dense_samples(rollout: torch.Tensor, n_grid: int = 100, mode: Literal["point_mass"] = "point_mass") -> tuple[torch.Tensor, ...]:
    if mode == "point_mass":
        dim = rollout.shape[-1]
        states = torch.zeros(n_grid * n_grid, dim)
        x_vals = torch.linspace(-1.0, 1.0, n_grid)
        y_vals = torch.linspace(-1.0, 1.0, n_grid)
        xx, yy = torch.meshgrid(x_vals, y_vals, indexing="xy")
        vx_vals = torch.linspace(-1.0, 1.0, n_grid)
        vy_vals = torch.linspace(-1.0, 1.0, n_grid)
        vxx, vyy = torch.meshgrid(vx_vals, vy_vals, indexing="xy")

        states[:, 0] = xx.flatten()
        states[:, 1] = yy.flatten()
        states[:, 2] = vxx.flatten()
        states[:, 3] = vyy.flatten()
    return states, xx.flatten(), yy.flatten()

def compute_eigenbasis_samples(eigenvectors: torch.Tensor, n_grid: int = 100) -> tuple[torch.Tensor, ...]:
    v1, v2 = eigenvectors[:, 0], eigenvectors[:, 1]

    alpha_vals = torch.linspace(-1.0, 1.0, n_grid)
    beta_vals = torch.linspace(-1.0, 1.0, n_grid)
    aa, bb = torch.meshgrid(alpha_vals, beta_vals, indexing="xy")

    states = aa.flatten().unsqueeze(1) * v1 + bb.flatten().unsqueeze(1) * v2
    return states, aa.flatten(), bb.flatten()

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
    explained_energy = eigvals / (total + 1e-6) # .cumsum(dim=0)
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
    # return C, eigvals

    # ascending
    eigvals, eigvecs = torch.linalg.eigh(C)
    # descending
    eigvals = eigvals.flip(1)
    eigvecs = eigvecs.flip(2)
    total = eigvals.sum(dim=1).unsqueeze(dim=1)
    explained_energy = eigvals / (total + 1e-6) # .cumsum(dim=1)
    input_contributions = torch.einsum("bss,bs->bs", eigvecs.square(), eigvals)
    input_activity = input_contributions / input_contributions.sum(dim=1).unsqueeze(dim=1)
    return C, eigvals, eigvecs, explained_energy, input_activity

def compute_batch_multistep_sensitivity(
    policy: torch.jit.ScriptModule,
    states: torch.Tensor,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, ...]:
    T, N, dim = states.shape
    x = states.reshape(T * N, dim)
    C, eigvals, eigvecs, explained_energy, input_activity = compute_multistep_sensitivity(policy, x, device)
    return (
        C.view(T, N, dim, dim),
        eigvals.view(T, N, dim),
        eigvecs.view(T, N, dim, dim),
        explained_energy.view(T, N, dim),
        input_activity.view(T, N, dim)
    )

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

def compute_projection_distances(states, eigenvectors, mean_center=True) -> torch.Tensor:
    if mean_center:
        mean_state = states.mean(axis=0, keepdims=True)
        x = states - mean_state
    else:
        x = states

    proj_dist = torch.einsum("...s,...ss->...s", x, eigenvectors)
    return proj_dist

def compute_linearization(
    policy: torch.jit.ScriptModule,
    states: torch.Tensor,
    v: torch.Tensor,
    alpha: float = 1.0,
    order: Literal[0, 1, 2] = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    ground_truth = policy(states + alpha * v)
    u0 = policy(states)
    if len(v.shape) == 1:
        v = v.unsqueeze(dim=0)
    match order:
        case 0:
            return ground_truth, u0
        case 1:
            J = compute_jacobian(policy, states)
            Jv = torch.einsum("bas,bs->ba", J, v)
            return ground_truth, u0 + alpha * Jv
        case 2:
            J = compute_jacobian(policy, states)
            Jv = torch.einsum("bas,bs->ba", J, v)
            H = compute_hessian(policy, states)
            Hv = torch.einsum("bass,bs->bas", H, v)
            vHv = torch.einsum("bs,bas->ba", v, Hv)
            return ground_truth, u0 + alpha * Jv + alpha ** 2 * vHv / 2


def plot_matrix(matrix: torch.Tensor, title: str, normalize: bool = False) -> None:
    if normalize:
        matrix /= matrix.sum() + 1e-6
        title_norm = ", Normalized"
    else:
        title_norm = ""
    dim = matrix.shape[0]
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="viridis", interpolation="nearest")
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(torch.arange(dim))
    ax.set_yticks(torch.arange(dim))
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
    ax.set_title(f"{title}"+title_norm)
    # fig.tight_layout()
    # fig.show()

def plot_heatmap(
    x_coords: torch.Tensor,
    y_coords: torch.Tensor,
    values: torch.Tensor,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    level_sets: list[float] | None = None,
    use_scatter: bool = False,
    size: int = 10,
) -> None:
    """Core heatmap plotting function."""
    fig, ax = plt.subplots()

    # Compute symmetric extent
    lim_min = min(x_coords.min().item(), y_coords.min().item())
    lim_max = max(x_coords.max().item(), y_coords.max().item())

    if use_scatter:
        sc = ax.scatter(x_coords.numpy(), y_coords.numpy(), c=values.numpy(), cmap="viridis", s=size)
        ax.figure.colorbar(sc, ax=ax, label=cbar_label)
        if level_sets is not None:
            ax.tricontour(
                x_coords.numpy(), y_coords.numpy(), values.numpy(),
                levels=level_sets, colors="white", linewidths=1.5,
            )
    else:
        n_grid = int(values.numel() ** 0.5)
        values_grid = values.reshape(n_grid, n_grid)
        xx = x_coords.reshape(n_grid, n_grid)
        yy = y_coords.reshape(n_grid, n_grid)
        extent = [lim_min, lim_max, lim_min, lim_max]
        im = ax.imshow(values_grid.T.numpy(), origin="lower", extent=extent, cmap="viridis", aspect="equal")
        ax.figure.colorbar(im, ax=ax, label=cbar_label)
        if level_sets is not None:
            cs = ax.contour(xx.numpy(), yy.numpy(), values_grid.T.numpy(), levels=level_sets, colors="white", linewidths=1.5)
            ax.clabel(cs, inline=True, fontsize=10, fmt="%.2f")

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal")

def plot_1d(
    data: torch.Tensor,
    title: str,
    xlabel: str,
    ylabel: str,
    grid: bool = True,
    logx: bool = False,
    logy: bool = False,
    xticks: bool = False,
    yticks: bool = False,
    linewidth: int = 2,
    marker: str = "",
    prefix: str = "",
    labels: tuple[str, ...] | None = None,
) -> None:
    fig, ax = plt.subplots()
    if len(data.shape) > 1:
        for k in range(data.shape[1]):
            if labels is not None:
                label = f"{prefix}{labels[k]}"
            else:
                label = f"{prefix}{str(k+1)}"
            ax.plot(data[:, k], marker=marker, linewidth=linewidth, label=label)
        ax.legend()
    else:
        ax.plot(data, marker=marker, linewidth=linewidth)
    ax.set_title(f"{title}")
    if xticks:
        ax.set_xticks(torch.arange(data.shape[0]))
    if yticks:
        ax.set_yticks(torch.arange(data.shape[0]))
    ax.set_xlabel(f"{xlabel}")
    ax.set_ylabel(f"{ylabel}")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(grid)
    # fig.tight_layout()
    # fig.show()

def plot_2d(
    data: torch.Tensor,
    title: str,
    xlabel: str,
    ylabel: str,
    grid: bool = True,
    log: bool = False,
    xticks: bool = False,
    yticks: bool = False,
    linewidth: int = 2,
    marker: str = "",
    prefix: str = "",
    temporal: bool = False,
) -> None:
    fig, ax = plt.subplots()
    if temporal:
        n_points = data.shape[0]
        if n_points > 1000:
            step = n_points // 100
            indices = torch.arange(0, n_points, step)
            data = data[indices]
            n_points = data.shape[0]
        colors = plt.cm.plasma(torch.linspace(0, 1, n_points))
        for i in range(n_points - 1):
            ax.plot(
                data[i:i+2, 0], data[i:i+2, 1],
                color=colors[i], linewidth=linewidth
            )
        if marker:
            ax.scatter(data[:, 0], data[:, 1], c=colors, marker=marker, zorder=3)
    else:
        ax.plot(data[:, 0], data[:, 1], marker=marker, linewidth=linewidth)
    ax.set_title(f"{title}")
    if xticks:
        ax.set_xticks(torch.arange(data.shape[0]))
    if yticks:
        ax.set_yticks(torch.arange(data.shape[0]))
    ax.set_xlabel(f"{xlabel}")
    ax.set_ylabel(f"{ylabel}")
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    # ax.set_aspect("equal")
    ax.grid(grid)
    # fig.tight_layout()
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

    T = torch.arange(C_data.shape[0])

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
        labels.append(f"Eigenvalue {i}")

    ax_eig.set_xlabel("Time (Step)")
    ax_eig.set_ylabel("Eigenvalue")
    ax_eig.grid(grid)
    if log:
        ax_eig.set_yscale("log")
    ax_eig.legend(
        handles, labels,
        ncol=min(2, k), loc="best"
    )
    fig.suptitle("Local Sensitivity")
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

    axes[-1].set_xlabel("Time (Step)")
    # fig.tight_layout()
    # fig.show()

def plot_projection_distances(proj, num_vectors=4) -> None:
    T = proj.shape[0]
    n = min(num_vectors, proj.shape[-1])

    fig, ax = plt.subplots()
    for k in range(n):
        ax.plot(proj[:, k], label=f"Eigenvector {k+1}")

    ax.set_xlabel("Time (Step)")
    ax.set_ylabel("Projection Coefficient")
    ax.set_title(f"State Projections onto Top {n} Eigen-Directions")
    ax.legend()
    ax.grid(True)
    # fig.tight_layout()
    # plt.show()


def compute_subspace_alignment(
    global_eigvecs: torch.Tensor,
    local_eigvecs: torch.Tensor,
) -> torch.Tensor:
    """Compute alignment between global and local eigenvector bases over time.

    Args:
        global_eigvecs: Global eigenvectors, shape (dim, dim) with columns as eigenvectors.
        local_eigvecs: Local eigenvectors, shape (T, dim, dim) with columns as eigenvectors.

    Returns:
        Alignment matrix of shape (T, dim, dim) where entry [t, i, j] is |<v_global_i, v_local_j(t)>|^2.
    """
    # global_eigvecs: (dim, dim) -> columns are eigenvectors
    # local_eigvecs: (T, dim, dim) -> columns are eigenvectors
    # Compute dot products: (T, dim, dim) where [t, i, j] = v_global_i · v_local_j(t)
    alignment = torch.einsum("ij,tik->tjk", global_eigvecs, local_eigvecs)
    return alignment.square()


def plot_subspace_alignment(
    alignment: torch.Tensor,
    title: str = "Subspace Alignment",
    grid: bool = True,
) -> None:
    """Plot alignment between global and local eigenvector bases over time.

    Shows how much each local eigenvector aligns with each global eigenvector.
    Perfect alignment with global eigenvector i means row i has value 1.0 in one column.

    Args:
        alignment: Alignment matrix of shape (T, dim, dim) from compute_subspace_alignment.
        title: Plot title.
        grid: Whether to show grid.
    """
    T, dim, _ = alignment.shape

    fig, axes = plt.subplots(dim, 1, figsize=(10, 2.5 * dim), sharex=True)
    if dim == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        for j in range(dim):
            ax.plot(alignment[:, i, j], linewidth=2, label=rf"$|v_{{{i+1}}}^G \cdot v_{{{j+1}}}^L|^2$")
        ax.set_ylabel(rf"Alignment with $v_{{{i+1}}}^G$")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="right", ncol=1, fontsize=8)
        ax.grid(grid)

    axes[-1].set_xlabel("Time (Step)")
    fig.suptitle(title)
    fig.tight_layout()

import torch

def compute_subspace_similarity(
        vecs_A: torch.Tensor,
        vals_A: torch.Tensor,
        vecs_B: torch.Tensor,
        vals_B: torch.Tensor,
        k: int,
) -> tuple[torch.Tensor, ...]:
    """
    Computes subspace similarity between a single basis A and a batch of bases B.
    Returns both Geometric (unweighted) and Energetic (weighted) scores.
    
    Args:
        vecs_A (Tensor): (N, N) Eigenvectors of A.
        vals_A (Tensor): (N,) Eigenvalues of A (sorted descending).
        vecs_B (Tensor): (Batch, N, N) Eigenvectors of B.
        vals_B (Tensor): (Batch, N) Eigenvalues of B (sorted descending).
        k (int): Number of top components to compare.
        
    Returns:
        tuple: (geometric_scores, energetic_scores)
               - geometric_scores: Tensor of shape (Batch,) with values in [0, 1]
               - energetic_scores: Tensor of shape (Batch,) with values in [0, 1]
    """
    N = vecs_A.shape[0]
    B_dim = vecs_B.shape[0]
    k = min(k, N)
    
    # 1. Slice the top k components
    # A shapes: Vecs (N, k), Vals (k,)
    sub_vecs_A = vecs_A[:, :k]
    sub_vals_A = vals_A[:k]
    
    # B shapes: Vecs (Batch, N, k), Vals (Batch, k)
    sub_vecs_B = vecs_B[:, :, :k]
    sub_vals_B = vals_B[:, :k]
    
    # ====================================================
    # Metric 1: Geometric Score (Unweighted Subspace Overlap)
    # ====================================================
    
    # Perform QR to ensure we have a clean orthonormal basis for the subspace
    # even if the input vectors were linearly dependent or defective.
    # Q_A: (N, k)
    Q_A, _ = torch.linalg.qr(sub_vecs_A, mode="reduced")
    
    # Q_B: (Batch, N, k) - Batched QR
    Q_B, _ = torch.linalg.qr(sub_vecs_B, mode="reduced")
    
    # Compute Interaction Matrix: Q_A^T @ Q_B
    # We want to multiply the single matrix Q_A against every matrix in Q_B.
    # Dimensions: (k, N) @ (Batch, N, k)
    # PyTorch matmul broadcasting: If first arg is 2D and second is 3D, 
    # it broadcasts the 2D arg across the batch dimension of the 3D arg.
    # Result: (Batch, k, k)
    interaction_geo = torch.matmul(Q_A.T, Q_B)
    
    # Compute Principal Angles (Singular Values)
    # svdvals runs in batch. Result: (Batch, k)
    # These are the cosines of the principal angles.
    singular_values = torch.linalg.svdvals(interaction_geo)
    
    # Score: Normalized sum of squared cosines
    geo_scores = torch.sum(singular_values ** 2, dim=1) / k
    
    # ====================================================
    # Metric 2: Energetic Score (Spectral Alignment)
    # ====================================================
    
    # 1. Raw Interaction Matrix (Dot products of eigenvectors)
    # Dimensions: (k, N) @ (Batch, N, k) -> (Batch, k, k)
    # C[b, i, j] is the dot product of vector A_i and vector B_{b,j}
    C = torch.matmul(sub_vecs_A.T, sub_vecs_B)
    C_sq = C ** 2
    
    # 2. Compute Weight Grid
    # We weight the match between A_i and B_j by |lambda_A_i * lambda_B_j|
    # w_A: (1, k, 1)  -> View as row aligner for A (rows of C)
    # w_B: (Batch, 1, k) -> View as col aligner for B (cols of C)
    w_A = sub_vals_A.abs().view(1, k, 1)
    w_B = sub_vals_B.abs().view(B_dim, 1, k)
    
    # Broadcast multiply to get grid of weights: (Batch, k, k)
    weights_grid = w_A * w_B 
    
    # 3. Compute Similarity
    # Numerator: Sum of (Weight * cos^2)
    numerator = torch.sum(weights_grid * C_sq, dim=(1, 2))
    
    # Denominator: Normalization by Frobenius norms of the reconstructed matrices
    # Approximation assuming approx orthonormal inputs: sqrt(sum(lambda^2))
    norm_A = torch.sqrt(torch.sum(sub_vals_A ** 2))
    norm_B = torch.sqrt(torch.sum(sub_vals_B ** 2, dim=1))
    
    energetic_scores = numerator / (norm_A * norm_B + 1e-6)
    
    return geo_scores, energetic_scores