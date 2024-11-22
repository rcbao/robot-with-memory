import os
from pathlib import Path
from typing import Any, Dict, Union, List
import numpy as np
import sapien
import torch
import uuid
import base64

from transforms3d.euler import euler2quat
import gymnasium as gym
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.sensors.camera import CameraConfig

from constants import OUTPUT_DIR, ASSET_DIR, CAMERA_CONFIG_DEFAULT, CAMERA_CONFIGS_HIGH_QUALITY, USING_HQ_CAMERA
from utils.camera_utils import StepImageCaptureWrapper


def short_uuid():
    return base64.urlsafe_b64encode(uuid.uuid4().bytes).decode('utf-8').rstrip("=")


def add_object_to_scene(
    table_scene,
    model_file: Union[str, Path],
    position: List[float] = [0, 0, 0],
    orientation_euler: List[float] = [np.pi / 2, 0, np.pi],
    scale: float = 1.0,
    name: str = "object",
    is_static: bool = False,
):
    model_file = Path(model_file)
    
    pose = sapien.Pose(p=position, q=euler2quat(*orientation_euler))
    builder = table_scene.scene.create_actor_builder()
    builder.add_nonconvex_collision_from_file(str(model_file), pose, [scale] * 3)
    builder.add_visual_from_file(str(model_file), pose, [scale] * 3)

    if is_static:
        return builder.build_static(name=name) 
    return builder.build(name=name)


def add_object_to_scene_ycb(
    table_scene,
    model_id:str,
    position: List[float] = [0, 0, 0],
    orientation_euler: List[float] = [0, 0, 0],
):
    builder = actors.get_actor_builder(
        table_scene.scene,
        id=f"ycb:{model_id}",
    )

    builder.initial_pose = sapien.Pose(p=position, q=euler2quat(*orientation_euler))
    short_id = short_uuid()

    return builder.build(name=f"{model_id}-{short_id}")


@register_env("MemoryRobot-v2")
class MemoryRobotEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "fetch", "panda_wristcam"]
    agent: Union[Panda, Fetch]

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.345, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.3455])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        asset_dir = ASSET_DIR

        shelf_model_file = f"{asset_dir}/Shelving_Unit_215_1.glb"

        ## Shelf 1 ##
        self.shelf_1 = add_object_to_scene(
            table_scene=self.table_scene,
            model_file=shelf_model_file,
            position=[-0.32, 0.345, 0],  
            orientation_euler=[np.pi / 2, 0, np.pi],
            scale=0.15,
            name="shelf-1",
            is_static=True
        )

        ## --------------- ##

        ## Shelf 1 Objects ##

        self.pear = add_object_to_scene_ycb(
            table_scene=self.table_scene,
            model_id="016_pear",
            position=[-0.40, 0.345, 0.22]
        )

        self.apple = add_object_to_scene_ycb(
            table_scene=self.table_scene,
            model_id="013_apple",
            position=[-0.24, 0.345, 0.22]
        )

        ## --------------- ##

        self.shelf_2 = add_object_to_scene(
            table_scene=self.table_scene,
            model_file=shelf_model_file,
            position=[-0.32, -0.345, 0],  
            orientation_euler=[np.pi / 2, 0, np.pi],
            scale=0.15,
            name="shelf-2",
            is_static=True
        )

        ## --------------- ##

        ## Shelf 2 Objects ##

        self.tomato_soup = add_object_to_scene_ycb(
            table_scene=self.table_scene,
            model_id="005_tomato_soup_can",
            position=[-0.40, -0.345, 0.24]    
        )

        self.banana = add_object_to_scene_ycb(
            table_scene=self.table_scene,
            model_id="011_banana",
            position=[-0.24, -0.345, 0.22]
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

    def evaluate(self):
        true_value = torch.tensor(True)

        return {
            "is_target_grasped": true_value,
            "is_target_on_base": true_value,
            "is_target_static": true_value,
            "success": true_value,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        return obs


def init_env():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config = CAMERA_CONFIGS_HIGH_QUALITY if USING_HQ_CAMERA else CAMERA_CONFIG_DEFAULT

    env = gym.make(
        "MemoryRobot-v2",
        render_mode="rgb_array",
        obs_mode="rgbd",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sensor_configs=config["sensor_configs"],
        human_render_camera_configs=config["human_render_camera_configs"],
        viewer_camera_configs=config["viewer_camera_configs"],
        enable_shadow=config["enable_shadow"]
    )
    env = StepImageCaptureWrapper(env, capture_frequency=25)
    env = RecordEpisode(env, output_dir=OUTPUT_DIR, save_trajectory=False, save_video=True, video_fps=30, max_steps_per_video=800)
    return env
