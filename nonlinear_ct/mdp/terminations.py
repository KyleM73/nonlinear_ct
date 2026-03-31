"""Custom termination functions for the point mass environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_out_of_bounds(
    env: ManagerBasedRLEnv,
    radius: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminates if the end-effector position norm exceeds the given radius."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    pos -= env.scene.env_origins.unsqueeze(1)
    return pos[..., :2].view(env.num_envs, -1).norm(dim=-1) > radius
