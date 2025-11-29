import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.envs.mdp as mdp

from nonlinear_ct.assets import POINT_MASS_CFG
import nonlinear_ct.mdp as task_mdp

##
# Scene definition
##

@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a point mass robot."""
    # robots
    robot: ArticulationCfg = POINT_MASS_CFG
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )

##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    pass

@configclass
class ActionsCfg:
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        pose = ObsTerm(func=task_mdp.body_pos_w, noise=Unoise(n_min=-0.01, n_max=0.01), params={
            "asset_cfg": SceneEntityCfg("robot", body_names="end_effector")})
        velocity = ObsTerm(func=task_mdp.body_vel_w, noise=Unoise(n_min=-0.1, n_max=0.1), params={
            "asset_cfg": SceneEntityCfg("robot", body_names="end_effector")})
        # last_action = ObsTerm(mdp.last_action)
    
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-2.0, 2.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

@configclass
class TestEventCfg:
    """Configuration for events."""

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-0.0, 0.0),
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    pos_error = RewTerm(func=task_mdp.pos_error, weight=-1.0, params={
        "asset_cfg": SceneEntityCfg("robot", body_names="end_effector")})
    vel_error = RewTerm(func=task_mdp.vel_error, weight=-0.01, params={
        "asset_cfg": SceneEntityCfg("robot", body_names="end_effector")})
    acc_error = RewTerm(func=task_mdp.acc_error, weight=-0.001, params={
        "asset_cfg": SceneEntityCfg("robot", body_names="end_effector")})

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass

##
# Environment configuration
##

@configclass
class DynamicsEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the point mass environment."""

    # Scene settings
    viewer: ViewerCfg = ViewerCfg(eye=(0.0, 0.0, 5.0), origin_type="env", env_index=0)
    scene: SceneCfg = SceneCfg(num_envs=8192, env_spacing=5.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.02
        self.sim.render_interval = self.decimation

@configclass
class TestDynamicsEnvCfg(DynamicsEnvCfg):
    events: TestEventCfg = TestEventCfg()
