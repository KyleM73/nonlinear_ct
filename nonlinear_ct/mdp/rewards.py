from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def pos_error(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalizes distance from the origin."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # access the body poses in world frame
    pose = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    pose -= env.scene.env_origins.unsqueeze(1)
    return pose.norm(dim=2).sum(dim=1)

def vel_error(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalizes velocity."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # access the body poses in world frame
    vel = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]
    return vel.norm(dim=1).sum(dim=1)

def acc_error(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalizes acceleration."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # access the body poses in world frame
    acc = asset.data.body_com_lin_acc_w[:, asset_cfg.body_ids]
    return acc.norm(dim=1).sum(dim=1)