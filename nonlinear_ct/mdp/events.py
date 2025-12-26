"""Custom event functions for stability sampling."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_joints_to_target(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints to target positions and velocities stored in env.

    This function reads target joint positions and velocities from environment
    attributes `_target_joint_pos` and `_target_joint_vel` and sets them directly.
    Used for stability analysis sampling where we need precise control over
    initial conditions.

    The target tensors should have shape (num_envs, num_joints) and be set
    before calling env.reset().
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # get target joint state from environment attributes
    # these must be set before reset is called
    if hasattr(env, "_target_joint_pos") and env._target_joint_pos is not None:
        joint_pos = env._target_joint_pos[env_ids].clone()
    else:
        # fallback to default
        joint_pos = asset.data.default_joint_pos[env_ids].clone()

    if hasattr(env, "_target_joint_vel") and env._target_joint_vel is not None:
        joint_vel = env._target_joint_vel[env_ids].clone()
    else:
        # fallback to default
        joint_vel = asset.data.default_joint_vel[env_ids].clone()

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
