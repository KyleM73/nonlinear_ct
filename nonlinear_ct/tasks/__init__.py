import gymnasium as gym

gym.register(
    id="point-mass-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.point_mass_cfg:DynamicsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PointMassPPORunnerCfg",
    },
)

gym.register(
    id="point-mass-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.point_mass_cfg:TestDynamicsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PointMassPPORunnerCfg",
    },
)

gym.register(
    id="point-mass-stability-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.point_mass_cfg:StabilitySampleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PointMassPPORunnerCfg",
    },
)