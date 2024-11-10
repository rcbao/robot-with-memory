import os
import gymnasium as gym
from mani_skill.utils.wrappers import RecordEpisode
from typing import Any, Dict, Union, List
import numpy as np
import sapien
import torch
from pathlib import Path

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
# mani_skill/utils/scene_builder/replicacad/scene_builder.py
from mani_skill.utils.scene_builder.replicacad.scene_builder import ReplicaCADSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig

from transforms3d.euler import euler2quat


from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv

OUTPUT_DIR = "exp-3/videos"

CAMERA_CONFIGS_HIGH_QUALITY = {
    "sensor_configs": dict(width=1920, height=1088, shader_pack="rt-fast"),
    "human_render_camera_configs":dict(width=1088, height=1088, shader_pack="rt-fast"),
    "viewer_camera_configs": dict(fov=1),
    "enable_shadow": True
}

CAMERA_CONFIG_DEFAULT = {
    "sensor_configs": dict(width=640, height=480, shader_pack="default"),
    "human_render_camera_configs":dict(width=640, height=480, shader_pack="default"),
    "viewer_camera_configs": dict(fov=1),
    "enable_shadow": True
}

USING_HQ_CAMERA = True


def save_camera_image_by_type(env, camera_type: str = "hand_camera"):
    obs = env.get_obs()
    sensor_data = obs['sensor_data']

    if 'sensor_data' in obs:
        sensor_data = obs['sensor_data']
        camera_image = sensor_data[camera_type]['rgb']  
        rgb_image = camera_image.squeeze(0).cpu().numpy() 
        image = Image.fromarray(rgb_image)

        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

        image.save(f'sensor_image-{camera_type}-{timestamp}.png')

def get_camera_image(env) -> np.ndarray:
    """
    Retrieve the current camera image from the simulator.
    
    Returns:
        np.ndarray: The captured image in RGB format.
    """
    obs = env.get_obs()
    if 'sensor_data' in obs:
        sensor_data = obs['sensor_data']
        camera_image = sensor_data["hand_camera"]['rgb']  
        
        save_camera_image_by_type("hand_camera")
    else:
        raise KeyError("Camera observation not found in the environment observations.")

def add_object_to_scene(
    table_scene,
    model_file: Union[str, Path],
    position: List[float] = [0, 0, 0],
    orientation_euler: List[float] = [0, 0, 0],
    scale: float = 1.0,
    name: str = "object",
    is_static: bool = False
):
    """
    Adds a 3D model (.glb) to the scene.

    Args:
        table_scene: Scene builder instance.
        model_file (Union[str, Path]): Path to the .glb model file.
        position (List[float]): Object position [x, y, z].
        orientation_euler (List[float]): Orientation in Euler angles [roll, pitch, yaw].
        scale (float): Uniform scale factor.
        name (str): Name of the object in the scene.
        is_static (bool): If True, makes object static, else dynamic.

    Returns:
        sapien.Actor: The created actor.
    """
    # Set up pose and scale
    pose = sapien.Pose(p=position, q=euler2quat(*orientation_euler))
    scale_factors = [scale] * 3
    model_file = str(Path(model_file))

    # Create the actor with collision and visual
    builder = table_scene.scene.create_actor_builder()
    builder.add_nonconvex_collision_from_file(filename=model_file, pose=pose, scale=scale_factors)
    builder.add_visual_from_file(filename=model_file, pose=pose, scale=scale_factors)

    return builder.build_static(name=name) if is_static else builder.build(name=name)


@register_env("PickApple-v1", max_episode_steps=50)
class PickAppleEnv(PickCubeEnv):
    cube_half_size = 0.02
    goal_thresh = 0.025
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[1, 1, 1, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            # initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

        # add_apple(self.table_scene)
        self.cube = add_object_to_scene(
            table_scene=self.table_scene,
            model_file="exp-2/assets/Cup_10.glb",
            # position=[0.5, 0.5, 0.5],
            orientation_euler=[np.pi / 2, 0, np.pi],
            scale=0.4,
            name="bread-20",
            is_static=False
        )




def init_env():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = CAMERA_CONFIGS_HIGH_QUALITY if USING_HQ_CAMERA else CAMERA_CONFIG_DEFAULT

    env = gym.make(
        "PickApple-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="sparse",
        sensor_configs=config["sensor_configs"],
        human_render_camera_configs=config["human_render_camera_configs"],
        viewer_camera_configs=config["viewer_camera_configs"],
        enable_shadow=config["enable_shadow"]
    )
    env = RecordEpisode(
        env,
        output_dir=f"{OUTPUT_DIR}",
        save_trajectory=False,
        save_video=True,
        video_fps=30,
        max_steps_per_video=800
    )
    return env