import os
from pathlib import Path
from typing import Any, Dict, Union, List
import datetime
import numpy as np
import sapien
import torch
from PIL import Image
from transforms3d.euler import euler2quat
from mani_skill.sensors.camera import CameraConfig
from mani_skill.envs.utils import randomization

import gymnasium as gym
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.scene_builder.replicacad.scene_builder import ReplicaCADSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig


OUTPUT_DIR = "exp-5/videos"
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


def save_camera_image_by_type(env, camera_type="base_camera"):
    obs = env.get_obs()
    if 'sensor_data' in obs:
        rgb_image = obs['sensor_data'][camera_type]['rgb'].squeeze(0).cpu().numpy()
        Image.fromarray(rgb_image).save(f"sensor_image-{camera_type}-{datetime.datetime.now():%Y%m%d-%H%M%S}.png")


def get_camera_image(env) -> np.ndarray:
    obs = env.get_obs()
    print("obs::")
    print(obs)
    print("=====")
    if 'sensor_data' in obs:
        save_camera_image_by_type(env, "base_camera")
    else:
        raise KeyError("Camera observation not found in the environment observations.")


class StepImageCaptureWrapper(gym.Wrapper):
    def __init__(self, env, capture_frequency=5):
        super().__init__(env)
        self.capture_frequency = capture_frequency
        self.step_count = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # Capture image every `capture_frequency` steps
        if self.step_count % self.capture_frequency == 0:
            get_camera_image(self.env)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)


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


@register_env("StackCube-v2", max_episode_steps=50)
class StackCubeEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self,
        *args,
        num_cubes: int = 2,  # Added num_cubes parameter with default value 2
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        assert num_cubes >= 2, "num_cubes must be at least 2 (one green and one red cube)"
        self.num_cubes = num_cubes  # Store num_cubes
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options)

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cubes = []  # List to hold all cube actors

        for i in range(self.num_cubes):
            if i == 0:
                color = [0, 1, 0, 1]  # First cube is green (base)
                name = "base_cube"
            elif i == 1:
                color = [1, 0, 0, 1]  # Second cube is red (to be stacked)
                name = "target_cube"
            else:
                # Assign random colors for additional cubes
                color = list(np.random.choice(range(256), size=3) / 255) + [1]
                name = f"cube_{i}"

            cube = actors.build_cube(
                self.scene,
                half_size=0.02,
                color=color,
                name=name
            )
            self.cubes.append(cube)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Sample positions ensuring no collision between cubes
            positions = []
            sampler = randomization.UniformPlacementSampler(
                bounds=[[-0.1, -0.2], [0.1, 0.2]], batch_size=b
            )
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            for _ in range(self.num_cubes):
                while True:
                    xy = sampler.sample(radius, 100)
                    # Check if the new position is sufficiently far from existing positions
                    if all(
                        torch.linalg.norm(xy - torch.tensor(pos[:2])) > (2 * self.cube_half_size[0] + 0.005)
                        for pos in positions
                    ):
                        positions.append(xy.cpu().numpy())
                        break

            for i, cube in enumerate(self.cubes):
                xyz = torch.zeros((b, 3))
                xyz[:, 2] = self.cube_half_size[2]  # Place on the table

                xyz[:, :2] = torch.tensor(positions[i]).to(self.device)

                if i == 1:
                    # Red cube: allow rotation around z-axis only
                    qs = randomization.random_quaternions(
                        b,
                        lock_x=True,
                        lock_y=True,
                        lock_z=False,
                    )
                else:
                    # Other cubes: full random rotation
                    qs = randomization.random_quaternions(b)

                cube.set_pose(Pose.create_from_pq(xyz.clone(), qs))

    def evaluate(self):
        # Identify the green and red cubes
        base_cube = self.cubes[0]
        target_cube = self.cubes[1]

        pos_base = base_cube.pose.p
        pos_target = target_cube.pose.p
        offset = pos_target - pos_base

        # Check if the target cube is on top of the base cube within half cube size
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        )
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[2] * 2) <= 0.005
        is_target_on_base = torch.logical_and(xy_flag, z_flag)

        # Check if the target cube is static
        is_target_static = target_cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)

        # Check if the target cube is not being grasped
        is_target_grasped = self.agent.is_grasping(target_cube)

        # Success condition
        success = is_target_on_base * is_target_static * (~is_target_grasped)

        return {
            "is_target_grasped": is_target_grasped,
            "is_target_on_base": is_target_on_base,
            "is_target_static": is_target_static,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_cubes_pos=[cube.pose.p - self.agent.tcp.pose.p for cube in self.cubes],
                cubes_pose=[cube.pose.raw_pose for cube in self.cubes],
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Reaching reward for the target cube
        tcp_pose = self.agent.tcp.pose.p
        target_cube_pos = self.cubes[1].pose.p
        target_to_tcp_dist = torch.linalg.norm(tcp_pose - target_cube_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * target_to_tcp_dist))

        # Grasp and place reward
        target_cube_pos = self.cubes[1].pose.p
        base_cube_pos = self.cubes[0].pose.p
        goal_xyz = torch.hstack(
            [base_cube_pos[:, 0:2], (base_cube_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]]
        )
        target_to_goal_dist = torch.linalg.norm(goal_xyz - target_cube_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * target_to_goal_dist)

        reward[info["is_target_grasped"]] = (4 + place_reward)[info["is_target_grasped"]]

        # Ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )  # NOTE: hard-coded with panda
        is_target_grasped = info["is_target_grasped"]
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[~is_target_grasped] = 1.0
        v = torch.linalg.norm(self.cubes[1].linear_velocity, axis=1)
        av = torch.linalg.norm(self.cubes[1].angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        reward[info["is_target_on_base"]] = (
            6 + (ungrasp_reward + static_reward) / 2.0
        )[info["is_target_on_base"]]

        reward[info["success"]] = 8

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8



def init_env():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config = CAMERA_CONFIGS_HIGH_QUALITY if USING_HQ_CAMERA else CAMERA_CONFIG_DEFAULT

    env = gym.make(
        # "PickApple-v1",
        "StackCube-v2",
        num_cubes=3,
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
