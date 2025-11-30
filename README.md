# Nonlinear Sensitivity Analysis for Learning-based Legged Locomotion

## Setup

### Training with IsaacLab

```bash
cd IsaacLab
./isaaclab.sh --uv .venv
source .venv/bin/activate
cd source
git clone git@github.com:KyleM73/nonlinear_ct.git
isaaclab -i rsl_rl
```

### Local Development

```bash
git clone git@github.com:KyleM73/nonlinear_ct.git
cd nonlinear_ct
uv venv --python=3.11
source .venv/bin/activate
uv pip install -e .
```