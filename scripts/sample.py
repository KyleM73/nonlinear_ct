"""Script to collect stability analysis samples for Jacobian estimation.

This script collects perturbation response data needed for the stability
analysis method described in the paper. Uses parallel environments for fast
batch sampling. For each sampled direction d on the unit sphere, it:
1. Resets environments to states +hd, steps once, records responses (batch)
2. Resets environments to states -hd, steps once, records responses (batch)
3. Also records the response from the origin (0)

The collected data allows estimation of:
- Forward difference: y^(k) = (F(hd^(k)) - F(0)) / h
- Second derivative: (F(hd^(k)) - 2F(0) + F(-hd^(k))) / h^2

Usage:
    python scripts/sample.py --task point-mass-stability-v0 --checkpoint <path> \
        --num_samples 100 --perturbation_radius 1e-3 --seed 42
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect stability analysis samples.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments (overrides num_samples).")
parser.add_argument("--task", type=str, default="point-mass-stability-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment and sampling")
parser.add_argument("--num_samples", type=int, default=100, help="Number of perturbation directions to sample (N)")
parser.add_argument("--perturbation_radius", type=float, default=1e-3, help="Perturbation radius h")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True

# suppress logs
if not hasattr(args_cli, "kit_args"):
    args_cli.kit_args = ""
args_cli.kit_args += " --/log/level=error"
args_cli.kit_args += " --/log/fileLogLevel=error"
args_cli.kit_args += " --/log/outputStreamLevel=error"

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

ISAAC_PREFIXES = ("--/log/", "--/app/", "--/renderer=", "--/physics/")
hydra_args = [arg for arg in hydra_args if not arg.startswith(ISAAC_PREFIXES)]
sys.argv = [sys.argv[0]] + hydra_args

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import nonlinear_ct.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def sample_unit_sphere(n_samples: int, dim: int, seed: int, device: str = "cpu") -> torch.Tensor:
    """Sample uniformly from the unit sphere in R^dim.

    Args:
        n_samples: Number of samples to generate.
        dim: Dimension of the sphere.
        seed: Random seed for reproducibility.
        device: Device to create tensor on.

    Returns:
        Tensor of shape (n_samples, dim) with unit norm rows.
    """
    torch.manual_seed(seed)
    # Sample from standard normal
    samples = torch.randn(n_samples, dim, device=device)
    # Normalize to unit sphere
    samples = samples / samples.norm(dim=1, keepdim=True)
    return samples


def collect_batch_sample(
    env,
    policy,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
) -> dict:
    """Collect a batch of perturbation samples using parallel environments.

    Sets all environments to their respective joint states, steps once with
    the policy, and records the observations and actions.

    Args:
        env: The wrapped environment with N parallel envs.
        policy: The inference policy.
        joint_pos: Target joint positions (N, num_joints).
        joint_vel: Target joint velocities (N, num_joints).

    Returns:
        Dictionary with obs_t0, obs_t1, action, joint_t0, joint_t1 for all envs.
    """
    unwrapped = env.unwrapped

    # Set target joint state for reset (one per environment)
    unwrapped._target_joint_pos = joint_pos
    unwrapped._target_joint_vel = joint_vel

    # Reset all environments (triggers reset_joints_to_target event)
    obs, _ = env.reset()

    # Get robot asset for joint state readout
    asset: Articulation = unwrapped.scene["robot"]

    # Record initial state for all envs
    obs_t0 = obs["policy"].clone()
    joint_pos_t0 = asset.data.joint_pos.clone()
    joint_vel_t0 = asset.data.joint_vel.clone()

    # Step with policy (batched inference)
    with torch.inference_mode():
        action = policy(obs)

    # Execute action for all envs
    obs, _, _, _ = env.step(action)

    # Record final state for all envs
    obs_t1 = obs["policy"].clone()
    joint_pos_t1 = asset.data.joint_pos.clone()
    joint_vel_t1 = asset.data.joint_vel.clone()

    return {
        "obs_t0": obs_t0.cpu(),
        "obs_t1": obs_t1.cpu(),
        "action": action.cpu(),
        "joint_pos_t0": joint_pos_t0.cpu(),
        "joint_vel_t0": joint_vel_t0.cpu(),
        "joint_pos_t1": joint_pos_t1.cpu(),
        "joint_vel_t1": joint_vel_t1.cpu(),
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Collect stability analysis samples using parallel environments."""

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    # Number of samples = number of parallel environments
    N = args_cli.num_envs if args_cli.num_envs is not None else args_cli.num_samples
    env_cfg.scene.num_envs = N

    # set the environment seed
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    log_root_path = os.path.join(log_root_path, "logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        log_dir = os.path.dirname(resume_path)
    else:
        try:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            log_dir = os.path.dirname(resume_path)
        except Exception:
            resume_path = None
            log_dir = log_root_path
            print("[INFO] No checkpoint found.")

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment with N parallel envs
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    if resume_path is not None:
        runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Get robot info
    asset: Articulation = env.unwrapped.scene["robot"]
    num_joints = asset.data.joint_pos.shape[-1]
    device = env.unwrapped.device

    print(f"[INFO] Robot has {num_joints} joints")
    print(f"[INFO] Using {N} parallel environments for batch sampling")
    print(f"[INFO] Perturbation radius h = {args_cli.perturbation_radius}")

    # Sample directions on unit sphere in joint state space (pos + vel)
    # State dimension = 2 * num_joints (positions and velocities)
    state_dim = 2 * num_joints
    directions = sample_unit_sphere(N, state_dim, args_cli.seed, device)

    # Split directions into position and velocity components
    dir_pos = directions[:, :num_joints]  # (N, num_joints)
    dir_vel = directions[:, num_joints:]  # (N, num_joints)

    h = args_cli.perturbation_radius

    # Get observation and action dimensions
    obs_dim = env.unwrapped.observation_manager.group_obs_dim["policy"][0]
    act_dim = env.unwrapped.action_manager.total_action_dim

    # Initialize data storage
    data = {
        "h": h,
        "N": N,
        "num_joints": num_joints,
        "state_dim": state_dim,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "seed": args_cli.seed,
        "directions": directions.cpu(),
    }

    # =========================================================================
    # Batch 1: Collect samples from origin (all envs at zero)
    # =========================================================================
    print("[INFO] Collecting sample from origin (batch)...")
    zero_pos = torch.zeros(N, num_joints, device=device)
    zero_vel = torch.zeros(N, num_joints, device=device)
    sample_zero = collect_batch_sample(env, policy, zero_pos, zero_vel)
    # Store just the first env's data as the canonical F(0) response
    # (all should be identical since they start from the same state)
    data["F_zero"] = {k: v[0] for k, v in sample_zero.items()}
    # Also store all for verification
    data["F_zero_all"] = sample_zero

    # =========================================================================
    # Batch 2: Collect samples for positive perturbations (+hd)
    # Each env i is reset to state +h * d^(i)
    # =========================================================================
    print("[INFO] Collecting positive perturbation samples (+hd) in batch...")
    pos_joint_pos = h * dir_pos  # (N, num_joints)
    pos_joint_vel = h * dir_vel  # (N, num_joints)
    sample_pos = collect_batch_sample(env, policy, pos_joint_pos, pos_joint_vel)
    data["F_pos"] = sample_pos

    # =========================================================================
    # Batch 3: Collect samples for negative perturbations (-hd)
    # Each env i is reset to state -h * d^(i)
    # =========================================================================
    print("[INFO] Collecting negative perturbation samples (-hd) in batch...")
    neg_joint_pos = -h * dir_pos  # (N, num_joints)
    neg_joint_vel = -h * dir_vel  # (N, num_joints)
    sample_neg = collect_batch_sample(env, policy, neg_joint_pos, neg_joint_vel)
    data["F_neg"] = sample_neg

    print("[INFO] Batch sampling complete!")

    # Save data
    if resume_path is not None:
        export_dir = os.path.join(os.path.dirname(resume_path), "exported", "stability")
        os.makedirs(export_dir, exist_ok=True)
        save_path = os.path.join(export_dir, "stability_samples.pt")
        torch.save(data, save_path)
        print(f"[INFO] Saved stability samples to: {save_path}")

        # Also save a summary
        summary_path = os.path.join(export_dir, "stability_info.txt")
        with open(summary_path, "w") as f:
            f.write("Stability Sampling Summary\n")
            f.write("==========================\n")
            f.write(f"Perturbation radius h: {h}\n")
            f.write(f"Number of samples N: {N}\n")
            f.write(f"State dimension (joint): {state_dim}\n")
            f.write(f"Observation dimension: {obs_dim}\n")
            f.write(f"Action dimension: {act_dim}\n")
            f.write(f"Random seed: {args_cli.seed}\n")
            f.write(f"\nSampling method: Batch parallel ({N} environments)\n")
            f.write(f"\nData structure:\n")
            f.write(f"  F_pos: Response to +hd perturbations (N samples)\n")
            f.write(f"  F_neg: Response to -hd perturbations (N samples)\n")
            f.write(f"  F_zero: Response from origin (single canonical sample)\n")
            f.write(f"  F_zero_all: Response from origin (all N envs for verification)\n")
            f.write(f"\nFor Jacobian estimation:\n")
            f.write(f"  y^(k) = (F_pos[k].obs_t1 - F_zero.obs_t1) / h\n")
            f.write(f"\nFor Hessian estimation:\n")
            f.write(f"  L2 ~ max_k ||(F_pos[k].obs_t1 - 2*F_zero.obs_t1 + F_neg[k].obs_t1) / h^2||\n")
        print(f"[INFO] Saved summary to: {summary_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
