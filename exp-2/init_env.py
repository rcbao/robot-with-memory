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

OUTPUT_DIR = "exp-2/videos"


# def add_apple(table_scene):
#     # add apple
#     scale = 1.0
#     builder = table_scene.scene.create_actor_builder()
#     model_dir = "exp-2/assets"
#     apple_model_file = str(f"{model_dir}/Apple_28.glb")

#     apple_pose = sapien.Pose(
#         p=[0.2, 0.2, 0.2], 
#         q=euler2quat(np.pi / 2, 0, np.pi)
#     )
#     builder.add_nonconvex_collision_from_file(
#         filename=apple_model_file, pose=apple_pose, scale=[scale] * 3
#     )
#     builder.add_visual_from_file(
#         filename=apple_model_file, scale=[scale] * 3, pose=apple_pose
#     )
#     table = builder.build_static(name="apple-28")

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

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()


        # self.cube = actors.build_cube(
        #     self.scene,
        #     half_size=self.cube_half_size,
        #     color=[1, 1, 1, 1],
        #     name="cube",
        #     # initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        # )

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
    env = gym.make(
        "PickApple-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="sparse",
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