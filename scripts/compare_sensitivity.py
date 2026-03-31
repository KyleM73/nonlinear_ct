"""Compare sensitivity structure across point-mass policies with different observation history lengths.

Standalone script — no Isaac Sim required. Downloads checkpoints from WandB, reconstructs
JIT policies, runs sensitivity analysis, and generates comparison plots.

Usage:
    python scripts/compare_sensitivity.py --run_ids RUN_T0 RUN_T1
    python scripts/compare_sensitivity.py --run_ids RUN_T0 RUN_T1 RUN_T2
    python scripts/compare_sensitivity.py --run_ids RUN_T0 RUN_T1 --history_lengths 2 3
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

# Add scripts dir to path so we can import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import compute_sensitivity, plot_matrix, plot_1d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare sensitivity across history-length policies.")
    parser.add_argument("--run_ids", nargs="+", required=True, help="WandB run IDs (one per history length).")
    parser.add_argument("--history_lengths", nargs="+", type=int, default=None,
                        help="History lengths corresponding to each run ID. "
                        "Defaults to [1, 2, ...] matching the number of run IDs.")
    parser.add_argument("--n_samples", type=int, default=10_000, help="Number of synthetic samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device.")
    parser.add_argument("--output_dir", type=str, default="scripts/sensitivity_comparison/",
                        help="Output directory for plots (relative to nonlinear_ct root).")
    parser.add_argument("--checkpoint", type=int, default=500, help="Model iteration to download (e.g. 500 -> model_500.pt).")
    parser.add_argument("--n_trajectories", type=int, default=5, help="Number of rollout trajectories per policy.")
    parser.add_argument("--wandb_project", type=str, default="Point Mass", help="WandB project name.")
    parser.add_argument("--wandb_entity", type=str, default="Apptronik", help="WandB entity.")
    args = parser.parse_args()

    # Default history_lengths to [1, 2, ..., N] based on number of run_ids
    if args.history_lengths is None:
        args.history_lengths = list(range(1, len(args.run_ids) + 1))

    return args


# ---------------------------------------------------------------------------
# WandB checkpoint download (adapted from train.py)
# ---------------------------------------------------------------------------

def download_wandb_checkpoint(run_id: str, project: str, entity: str, iteration: int) -> str:
    """Download a model checkpoint from a WandB run.

    Args:
        run_id: WandB run ID.
        project: WandB project name.
        entity: WandB entity.
        iteration: Model iteration to download (e.g. 500 -> model_500.pt).
    """
    import wandb

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    target_name = f"model_{iteration}.pt"
    files = [f for f in run.files() if f.name == target_name]
    if not files:
        # List available checkpoints for a helpful error message
        all_ckpts = [f.name for f in run.files() if f.name.startswith("model_") and f.name.endswith(".pt")]
        raise FileNotFoundError(
            f"Checkpoint '{target_name}' not found in WandB run {run_id}. "
            f"Available: {sorted(all_ckpts)}"
        )
    latest = files[0]
    print(f"[INFO] Downloading {latest.name} from WandB run {run_id}...")
    download_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wandb_downloads", run_id)
    os.makedirs(download_dir, exist_ok=True)
    latest.download(root=download_dir, replace=True)
    return os.path.join(download_dir, latest.name)


# ---------------------------------------------------------------------------
# Policy reconstruction
# ---------------------------------------------------------------------------

# Known hyperparameters from ppo_cfg.py
ACTOR_HIDDEN_DIMS = [16, 16, 16]
CRITIC_HIDDEN_DIMS = [16, 16, 16]
NUM_ACTIONS = 2
CRITIC_OBS_DIM = 6  # pos(2) + vel(2) + last_action(2)
ACTIVATION = "elu"


class PolicyExporter(nn.Module):
    """Minimal JIT-exportable wrapper: forward(x) = actor(normalizer(x))."""

    def __init__(self, actor: nn.Module, normalizer: nn.Module):
        super().__init__()
        self.actor = copy.deepcopy(actor)
        self.normalizer = copy.deepcopy(normalizer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(x))

    @torch.jit.export
    def reset(self) -> None:
        pass


def reconstruct_policy(checkpoint_path: str, obs_dim: int, device: str = "cpu") -> torch.jit.ScriptModule:
    """Reconstruct a JIT-scriptable policy from a checkpoint.

    Creates a dummy ActorCritic with matching architecture, loads the checkpoint
    weights, then wraps actor + normalizer into a JIT module.
    """
    from air_rl.modules import ActorCritic

    # Build dummy TensorDict with correct shapes
    dummy_obs = TensorDict({
        "policy": torch.zeros(1, obs_dim),
        "critic": torch.zeros(1, CRITIC_OBS_DIM),
    })
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    # Instantiate ActorCritic with known hyperparams
    model = ActorCritic(
        obs=dummy_obs,
        obs_groups=obs_groups,
        num_actions=NUM_ACTIONS,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=ACTOR_HIDDEN_DIMS,
        critic_hidden_dims=CRITIC_HIDDEN_DIMS,
        activation=ACTIVATION,
        init_noise_std=1.0,
    )

    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Wrap actor + normalizer into JIT module
    exporter = PolicyExporter(model.actor, model.actor_obs_normalizer)
    exporter.to(device)
    exporter.eval()
    jit_policy = torch.jit.script(exporter)
    return jit_policy


# ---------------------------------------------------------------------------
# Dimension labels
# ---------------------------------------------------------------------------

def make_dim_labels(history_length: int) -> list[str]:
    """Generate human-readable labels for each observation dimension.

    For history_length=1: [px_t, py_t]
    For history_length=2: [px_t, py_t, px_{t-1}, py_{t-1}]
    etc.
    """
    labels = []
    for t in range(history_length):
        suffix = f"t" if t == 0 else f"t-{t}"
        labels.extend([f"$p_x^{{{suffix}}}$", f"$p_y^{{{suffix}}}$"])
    return labels


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def plot_eigenvalue_spectrum_comparison(
    all_eigvals: list[torch.Tensor],
    labels: list[str],
    output_path: str,
) -> None:
    """Log-scale overlay of eigenvalue spectra from all policies."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (eigvals, label) in enumerate(zip(all_eigvals, labels)):
        indices = torch.arange(1, len(eigvals) + 1)
        ax.plot(indices, eigvals.numpy(), marker="o", linewidth=2, color=COLORS[i % len(COLORS)], label=label)
    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_yscale("log")
    ax.set_title("Eigenvalue Spectrum Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_explained_energy_comparison(
    all_energies: list[torch.Tensor],
    labels: list[str],
    output_path: str,
) -> None:
    """Grouped bar chart of explained energy by eigenvalue index."""
    fig, ax = plt.subplots(figsize=(8, 4))
    n_policies = len(all_energies)
    max_dim = max(len(e) for e in all_energies)
    bar_width = 0.8 / n_policies

    for i, (energy, label) in enumerate(zip(all_energies, labels)):
        indices = np.arange(len(energy)) + i * bar_width
        ax.bar(indices, energy.numpy(), width=bar_width, color=COLORS[i % len(COLORS)], label=label)

    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Explained Energy Fraction")
    ax.set_title("Explained Energy Comparison")
    ax.set_xticks(np.arange(max_dim) + bar_width * (n_policies - 1) / 2)
    ax.set_xticklabels([str(k + 1) for k in range(max_dim)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_input_activity_current_comparison(
    all_activities: list[torch.Tensor],
    all_labels_dim: list[list[str]],
    policy_labels: list[str],
    output_path: str,
) -> None:
    """Grouped bars comparing px_t, py_t activity across policies."""
    fig, ax = plt.subplots(figsize=(6, 4))
    n_policies = len(all_activities)
    bar_width = 0.8 / n_policies
    current_labels = ["$p_x^{t}$", "$p_y^{t}$"]

    for i, (activity, label) in enumerate(zip(all_activities, policy_labels)):
        # First two dims are always current position
        current_activity = activity[:2].numpy()
        indices = np.arange(2) + i * bar_width
        ax.bar(indices, current_activity, width=bar_width, color=COLORS[i % len(COLORS)], label=label)

    ax.set_xlabel("Input Dimension")
    ax.set_ylabel("Input Activity")
    ax.set_title("Current Position Activity Comparison")
    ax.set_xticks(np.arange(2) + bar_width * (n_policies - 1) / 2)
    ax.set_xticklabels(current_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_sensitivity_matrix(
    C: torch.Tensor,
    dim_labels: list[str],
    title: str,
    output_path: str,
) -> None:
    """Heatmap of the sensitivity matrix C (normalized)."""
    C_norm = C / (C.sum() + 1e-6)
    dim = C_norm.shape[0]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(C_norm.numpy(), cmap="viridis", interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(dim))
    ax.set_yticks(range(dim))
    ax.set_xticklabels(dim_labels, fontsize=8)
    ax.set_yticklabels(dim_labels, fontsize=8)
    max_val = C_norm.max()
    for i in range(dim):
        for j in range(dim):
            text_color = "white" if C_norm[i, j] < max_val / 2 else "black"
            ax.text(j, i, f"{C_norm[i, j]:.2f}", ha="center", va="center",
                    color=text_color, fontsize=10 if dim < 8 else 7)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_input_activity_per_policy(
    activity: torch.Tensor,
    dim_labels: list[str],
    title: str,
    output_path: str,
) -> None:
    """Bar chart of per-dimension input activity."""
    fig, ax = plt.subplots(figsize=(6, 4))
    indices = np.arange(len(activity))
    ax.bar(indices, activity.numpy(), color=COLORS[0])
    ax.set_xticks(indices)
    ax.set_xticklabels(dim_labels, fontsize=8)
    ax.set_xlabel("Input Dimension")
    ax.set_ylabel("Input Activity")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Closed-loop trajectory rollout (semi-implicit Euler, double integrator)
# ---------------------------------------------------------------------------

# Double integrator dynamics: mass=1, no gravity, no damping, dt=0.02
# Uses semi-implicit (symplectic) Euler to match PhysX:
#   v' = v + dt * F
#   x' = x + dt * v'
SIM_DT = 0.02
EPISODE_LENGTH_S = 10.0
MAX_STEPS = int(EPISODE_LENGTH_S / SIM_DT)  # 500


def _symplectic_euler_step(
    pos: torch.Tensor,
    vel: torch.Tensor,
    force: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Semi-implicit Euler step for 2D double integrator (m=1).

    Matches PhysX integration: update velocity first, then position.
    """
    vel_next = vel + dt * force
    pos_next = pos + dt * vel_next
    return pos_next, vel_next


def _build_obs(
    pos_history: list[torch.Tensor],
    history_length: int,
) -> torch.Tensor:
    """Build the observation vector from position history.

    Stacks [pos_t, pos_{t-1}, ..., pos_{t-H+1}] matching IsaacLab's
    history buffer ordering (newest first).
    """
    # pos_history[-1] is current, pos_history[-2] is previous, etc.
    parts = []
    for i in range(history_length):
        idx = len(pos_history) - 1 - i
        if idx >= 0:
            parts.append(pos_history[idx])
        else:
            # Before episode start — pad with the earliest available position
            parts.append(pos_history[0])
    return torch.cat(parts, dim=-1)


@torch.no_grad()
def rollout_trajectories(
    policy_jit: torch.jit.ScriptModule,
    history_length: int,
    n_trajectories: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out closed-loop trajectories under RK4 double-integrator dynamics.

    Args:
        policy_jit: JIT policy mapping obs -> action (force).
        history_length: Number of position frames in the observation.
        n_trajectories: Number of trajectories to simulate.
        seed: Random seed for initial conditions.

    Returns:
        positions: (MAX_STEPS+1, n_trajectories, 2) position trajectories.
        actions: (MAX_STEPS, n_trajectories, 2) applied forces.
    """
    torch.manual_seed(seed)

    # Sample initial positions uniformly inside the unit disk
    angles = torch.rand(n_trajectories) * 2 * torch.pi
    radii = torch.sqrt(torch.rand(n_trajectories))  # sqrt for uniform area
    pos = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=-1)  # (N, 2)
    vel = torch.zeros(n_trajectories, 2)

    positions = torch.zeros(MAX_STEPS + 1, n_trajectories, 2)
    actions = torch.zeros(MAX_STEPS, n_trajectories, 2)
    positions[0] = pos

    # Initialise history buffer with the initial position
    pos_history: list[torch.Tensor] = [pos.clone()]

    for step in range(MAX_STEPS):
        obs = _build_obs(pos_history, history_length)  # (N, obs_dim)
        force = policy_jit(obs)  # (N, 2)

        pos, vel = _symplectic_euler_step(pos, vel, force, SIM_DT)

        # Maintain a rolling history (keep at most history_length entries)
        pos_history.append(pos.clone())
        if len(pos_history) > history_length:
            pos_history.pop(0)

        positions[step + 1] = pos
        actions[step] = force

    return positions, actions


def plot_trajectories_comparison(
    all_positions: list[torch.Tensor],
    all_actions: list[torch.Tensor],
    policy_labels: list[str],
    output_path: str,
) -> None:
    """Side-by-side XY trajectory + action magnitude plots, one column per policy."""
    n_policies = len(all_positions)
    fig, axes = plt.subplots(2, n_policies, figsize=(5 * n_policies, 8),
                             squeeze=False)

    time_s = np.arange(MAX_STEPS) * SIM_DT

    for col, (positions, act, label) in enumerate(zip(all_positions, all_actions, policy_labels)):
        ax_xy = axes[0, col]
        ax_act = axes[1, col]
        n_traj = positions.shape[1]

        for i in range(n_traj):
            traj = positions[:, i, :].numpy()
            ax_xy.plot(traj[:, 0], traj[:, 1], linewidth=1, alpha=0.8)
            ax_xy.scatter(traj[0, 0], traj[0, 1], s=20, zorder=5, marker="o")

        ax_xy.scatter([0], [0], s=60, color="black", marker="*", zorder=10)
        ax_xy.set_xlabel("$x$ (m)")
        ax_xy.set_ylabel("$y$ (m)")
        ax_xy.set_title(f"Trajectories — {label}")
        ax_xy.set_aspect("equal")
        ax_xy.grid(True, alpha=0.3)

        for i in range(n_traj):
            act_mag = act[:, i, :].norm(dim=-1).numpy()
            ax_act.plot(time_s, act_mag, linewidth=1, alpha=0.8)

        ax_act.set_xlabel("Time (s)")
        ax_act.set_ylabel(r"$\|F\|$")
        ax_act.set_title(f"Action Magnitude — {label}")
        ax_act.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    assert len(args.run_ids) == len(args.history_lengths), (
        f"Number of run_ids ({len(args.run_ids)}) must match number of history_lengths ({len(args.history_lengths)})"
    )

    # Resolve output dir relative to nonlinear_ct root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(script_dir, "..")
    output_dir = os.path.join(root_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args.seed)

    # Storage for comparison data
    all_C = []
    all_eigvals = []
    all_eigvecs = []
    all_energies = []
    all_activities = []
    all_dim_labels = []
    all_positions = []
    all_actions = []
    policy_labels = []

    for run_id, hist_len in zip(args.run_ids, args.history_lengths):
        obs_dim = 2 * hist_len
        label = f"$t={hist_len - 1}$ (dim={obs_dim})"
        safe_label = f"t{hist_len - 1}"
        dim_labels = make_dim_labels(hist_len)
        policy_labels.append(label)
        all_dim_labels.append(dim_labels)

        print(f"\n{'='*60}")
        print(f"Policy: history_length={hist_len}, obs_dim={obs_dim}, run={run_id}")
        print(f"{'='*60}")

        # 1. Download checkpoint
        print("[1/5] Downloading checkpoint...")
        ckpt_path = download_wandb_checkpoint(run_id, args.wandb_project, args.wandb_entity, args.checkpoint)

        # 2. Reconstruct policy
        print("[2/5] Reconstructing policy...")
        policy_jit = reconstruct_policy(ckpt_path, obs_dim, args.device)

        # 3. Generate synthetic samples
        print("[3/5] Generating synthetic samples...")
        torch.manual_seed(args.seed)
        states = torch.rand(args.n_samples, obs_dim) * 2 - 1  # Uniform in [-1, 1]

        # 4. Sensitivity analysis
        print("[4/5] Computing sensitivity...")
        C, eigvals, eigvecs, explained_energy, input_activity = compute_sensitivity(
            policy_jit, states, device=args.device
        )

        all_C.append(C)
        all_eigvals.append(eigvals)
        all_eigvecs.append(eigvecs)
        all_energies.append(explained_energy)
        all_activities.append(input_activity)

        # 5. Rollout trajectories
        print(f"[5/5] Rolling out {args.n_trajectories} trajectories...")
        positions, actions = rollout_trajectories(
            policy_jit, hist_len, args.n_trajectories, args.seed,
        )
        all_positions.append(positions)
        all_actions.append(actions)

        # Per-policy plots
        print("  Generating per-policy plots...")
        plot_sensitivity_matrix(
            C, dim_labels,
            f"Sensitivity Matrix ({label})",
            os.path.join(output_dir, f"sensitivity_matrix_{safe_label}.png"),
        )
        plot_input_activity_per_policy(
            input_activity, dim_labels,
            f"Input Activity ({label})",
            os.path.join(output_dir, f"input_activity_{safe_label}.png"),
        )

    # Comparison plots
    print(f"\n{'='*60}")
    print("Generating comparison plots...")
    print(f"{'='*60}")

    plot_eigenvalue_spectrum_comparison(
        all_eigvals, policy_labels,
        os.path.join(output_dir, "eigenvalue_spectrum_comparison.png"),
    )

    plot_explained_energy_comparison(
        all_energies, policy_labels,
        os.path.join(output_dir, "explained_energy_comparison.png"),
    )

    plot_input_activity_current_comparison(
        all_activities, all_dim_labels, policy_labels,
        os.path.join(output_dir, "input_activity_current_comparison.png"),
    )

    plot_trajectories_comparison(
        all_positions, all_actions, policy_labels,
        os.path.join(output_dir, "trajectories_comparison.png"),
    )

    # Write metadata
    metadata_path = os.path.join(output_dir, "metadata.txt")
    with open(metadata_path, "w") as f:
        f.write("History-Length Sensitivity Comparison\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Samples: {args.n_samples}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"WandB Project: {args.wandb_project}\n")
        f.write(f"WandB Entity: {args.wandb_entity}\n\n")

        for i, (run_id, hist_len) in enumerate(zip(args.run_ids, args.history_lengths)):
            obs_dim = 2 * hist_len
            f.write(f"--- Policy {i}: history_length={hist_len}, obs_dim={obs_dim} ---\n")
            f.write(f"  Run ID: {run_id}\n")
            f.write(f"  Eigenvalues: {all_eigvals[i].numpy().tolist()}\n")
            f.write(f"  Explained Energy: {all_energies[i].numpy().tolist()}\n")
            f.write(f"  Input Activity: {all_activities[i].numpy().tolist()}\n")
            f.write(f"  Dim Labels: {all_dim_labels[i]}\n\n")

    print(f"  Saved: {metadata_path}")
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
