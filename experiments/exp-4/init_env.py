import os
from pathlib import Path
from typing import Union, List
import numpy as np
from transforms3d.euler import euler2quat
import sapien
import gymnasium as gym
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from camera_utils import StepImageCaptureWrapper  

OUTPUT_DIR = "exp-4/videos"
USING_HQ_CAMERA = True

CAMERA_CONFIGS_HIGH_QUALITY = {
    "sensor_configs": {"width": 1920, "height": 1088, "shader_pack": "rt-fast"},
    "human_render_camera_configs": {"width": 1088, "height": 1088, "shader_pack": "rt-fast"},
    "viewer_camera_configs": {"fov": 1},
    "enable_shadow": True,
}

CAMERA_CONFIG_DEFAULT = {
    "sensor_configs": {"width": 640, "height": 480, "shader_pack": "default"},
    "human_render_camera_configs": {"width": 640, "height": 480, "shader_pack": "default"},
    "viewer_camera_configs": {"fov": 1},
    "enable_shadow": True,
}

def add_object_to_scene(
    table_scene,
    model_file: Union[str, Path],
    position: List[float] = [0, 0, 0],
    orientation_euler: List[float] = [0, 0, 0],
    scale: float = 1.0,
    name: str = "object",
    is_static: bool = False,
):
    pose = sapien.Pose(p=position, q=euler2quat(*orientation_euler))
    builder = table_scene.scene.create_actor_builder()
    builder.add_nonconvex_collision_from_file(filename=str(Path(model_file)), pose=pose, scale=[scale] * 3)
    builder.add_visual_from_file(filename=str(Path(model_file)), pose=pose, scale=[scale] * 3)
    return builder.build_static(name=name) if is_static else builder.build(name=name)


@register_env("PickApple-v1", max_episode_steps=50)
class PickAppleEnv(PickCubeEnv):
    cube_half_size = 0.02
    goal_thresh = 0.025
    SUPPORTED_ROBOTS = ["panda", "fetch", "panda_wristcam"]

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self.goal_site = actors.build_sphere(
            self.scene, radius=self.goal_thresh, color=[1, 1, 1, 1], name="goal_site", body_type="kinematic",
            add_collision=False
        )
        self._hidden_objects.append(self.goal_site)
        self.cube = add_object_to_scene(
            table_scene=self.table_scene,
            model_file="exp-2/assets/Cup_10.glb",
            orientation_euler=[np.pi / 2, 0, np.pi],
            scale=0.4,
            name="bread-20",
            is_static=False,
        )


def init_env():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config = CAMERA_CONFIGS_HIGH_QUALITY if USING_HQ_CAMERA else CAMERA_CONFIG_DEFAULT

    env = gym.make(
        "PickApple-v1",
        render_mode="rgb_array",
        obs_mode="rgbd",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sensor_configs=config["sensor_configs"],
        human_render_camera_configs=config["human_render_camera_configs"],
        viewer_camera_configs=config["viewer_camera_configs"],
        enable_shadow=config["enable_shadow"]
    )
    env = StepImageCaptureWrapper(env, capture_frequency=25, camera_type="base_camera")
    env = RecordEpisode(env, output_dir=OUTPUT_DIR, save_trajectory=False, save_video=True, video_fps=30, max_steps_per_video=800)
    return env
