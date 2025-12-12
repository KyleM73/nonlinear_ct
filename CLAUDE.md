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
# Train point mass environment
python scripts/train.py --task point-mass-v0 --num_envs 4096 --seed 0 --max_iterations 1050

# Play back trained policy
python scripts/play.py --task point-mass-play-v0 --checkpoint <path-to-checkpoint>
```

### Analysis

Sensitivity analysis is performed in `scripts/analyze.ipynb` using trained policy checkpoints.

## Architecture

### Package Structure

- `nonlinear_ct/assets/` - Robot URDF asset configurations (point mass)
- `nonlinear_ct/mdp/` - MDP components: observation functions (`body_pos_w`, `body_vel_w`) and reward terms (`pos_error`, `vel_error`, `acc_error`)
- `nonlinear_ct/tasks/` - Environment configs using IsaacLab's `@configclass` pattern: scene setup, action/observation spaces, reward functions, PPO hyperparameters

### Scripts

- `scripts/train.py` - RSL-RL training loop with wandb logging
- `scripts/play.py` - Policy inference and video recording
- `scripts/utils.py` - Sensitivity analysis: Jacobian/Hessian computation via `torch.func.jacrev`/`torch.func.hessian`, eigenvalue decomposition, perturbation analysis
- `scripts/cli_args.py` - RSL-RL CLI argument parsing

### Key Technical Details

- Uses `torch.func.vmap` for batched Jacobian/Hessian computation
- Sensitivity metric: eigenvalue decomposition of `J^T J` (averaged over states)
- Policies are JIT-compiled (`policy_jit.pt`) for analysis
- Training uses 8192 parallel environments, 0.02s timestep, 10s episodes

## Code Style

- Line length: 120 (isort with black profile)
- Python 3.11
- Type checking: pyright basic mode
- Matplotlib with LaTeX rendering (`plt.rcParams["text.usetex"] = True`)
