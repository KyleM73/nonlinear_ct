# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project for nonlinear sensitivity analysis of neural network controllers in robotic systems. Built as an IsaacLab extension using RSL-RL (PPO) for training and torch.func for computing Jacobian/Hessian-based sensitivity metrics.

## Commands

### Setup

**IsaacLab Integration:**
```bash
cd IsaacLab
./isaaclab.sh --uv .venv
source .venv/bin/activate
cd source
git clone git@github.com:KyleM73/nonlinear_ct.git
isaaclab -i rsl_rl
```

**Local Development:**
```bash
uv venv --python=3.11
source .venv/bin/activate
uv pip install -e .
```

### Training and Inference

```bash
# Train point mass environment (headless)
python scripts/train.py --task point-mass-v0 --num_envs 4096 --seed 0 --max_iterations 1050 --headless

# Play back trained policy and export JIT models
python scripts/play.py --task point-mass-play-v0 --checkpoint <path-to-checkpoint> --export
```

### Analysis Pipeline

1. Train policy with `train.py`
2. Run `play.py` with `--export` to generate:
   - `exported/linearization/policy_jit.pt` - JIT-compiled actor
   - `exported/linearization/critic_jit.pt` - JIT-compiled critic
   - `exported/linearization/rollout.pt` - State trajectory data
   - `exported/linearization/policy_limits.pt` - Joint limits
   - `exported/linearization/policy_io.txt` - Action/observation specs
3. Run sensitivity analysis in `scripts/analyze.ipynb`

## Registered Environments

- `point-mass-v0` - Training environment with 8192 envs
- `point-mass-play-v0` - Test environment with fewer envs for inference

## Architecture

### Package Structure

- `nonlinear_ct/assets/` - Robot URDF asset configurations (point mass)
- `nonlinear_ct/mdp/` - MDP components: observation functions (`body_pos_w`, `body_vel_w`) and reward terms (`pos_error`, `vel_error`, `acc_error`)
- `nonlinear_ct/tasks/` - Environment configs using IsaacLab's `@configclass` pattern: scene setup, action/observation spaces, reward functions, PPO hyperparameters

### Scripts

- `scripts/train.py` - RSL-RL training loop with wandb logging
- `scripts/play.py` - Policy inference, exports JIT models and rollout data
- `scripts/exporter.py` - Critic-to-JIT export (supports LSTM/GRU recurrent policies)
- `scripts/utils.py` - Sensitivity analysis utilities (see below)
- `scripts/sample.py` - Standalone sampling script
- `scripts/cli_args.py` - RSL-RL CLI argument parsing
- `scripts/analyze.ipynb` - Interactive sensitivity analysis notebook

### Key Sensitivity Functions (scripts/utils.py)

- `compute_jacobian(policy, x)` - Batched Jacobian via `torch.func.jacrev`
- `compute_hessian(policy, x)` - Batched Hessian via `torch.func.hessian`
- `compute_sensitivity(policy, states)` - Eigendecomposition of `J^T J`, returns eigvals/eigvecs/explained_energy
- `compute_multistep_sensitivity(policy, states)` - Per-timestep sensitivity metrics
- `compute_subspace_alignment(global_eigvecs, local_eigvecs)` - Alignment between global and local eigenbases
- `compute_subspace_similarity(vecs_A, vals_A, vecs_B, vals_B, k)` - Geometric and energetic similarity scores
- `compute_perturbation_error(policy, states, eigenvectors, radii)` - Nonlinear vs linear perturbation comparison

### Key Technical Details

- Uses `torch.func.vmap` for batched Jacobian/Hessian computation
- Sensitivity metric: eigenvalue decomposition of `J^T J` (averaged over states)
- Policies are JIT-compiled (`policy_jit.pt`) for analysis
- Training uses 4096-8192 parallel environments, 0.02s timestep, 10s episodes

## Code Style

- Line length: 120 (isort with black profile)
- Python 3.11
- Type checking: pyright basic mode
- Matplotlib with LaTeX rendering (`plt.rcParams["text.usetex"] = True`)
