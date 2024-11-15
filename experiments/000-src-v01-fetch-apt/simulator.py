import datetime
import gymnasium as gym
from typing import Optional, Tuple
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.record import RecordEpisode
from typing import Any, Tuple, Optional
import numpy as np
from mani_skill.sensors.camera import CameraConfig
from PIL import Image
import torch
import logging
from scipy.spatial.transform import Rotation as R


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


CAMERA_CONFIGS_HIGH_QUALITY = {
    "sensor_configs": dict(width=1920, height=1080, shader_pack="rt-fast"),
    "human_render_camera_configs":dict(width=1080, height=1080, shader_pack="rt-fast"),
    "viewer_camera_configs": dict(fov=1),
    "enable_shadow": True
}

CAMERA_CONFIG_DEFAULT = {
    "sensor_configs": dict(width=640, height=480, shader_pack="default"),
    "human_render_camera_configs":dict(width=640, height=480, shader_pack="default"),
    "viewer_camera_configs": dict(fov=1),
    "enable_shadow": True
}

USING_HQ_CAMERA = False

class ManiSkillSimulator:
    def __init__(self, record_video: bool = True):
        """
        Initialize the ManiSkill simulator with a specific environment configuration.

        Args:
            record_video (bool): If True, records the entire episode as a video.
        """
        logging.info("Initializing ManiSkill Simulator.")
        
        config = CAMERA_CONFIGS_HIGH_QUALITY if USING_HQ_CAMERA else CAMERA_CONFIG_DEFAULT

        self.env = gym.make(
            'ReplicaCAD_SceneManipulation-v1', 
            render_mode="rgb_array",
            obs_mode="rgbd",
            sensor_configs=config["sensor_configs"],
            human_render_camera_configs=config["human_render_camera_configs"],
            viewer_camera_configs=config["viewer_camera_configs"],
            enable_shadow=config["enable_shadow"]
        )

        if record_video:
            # self.env = RecordEpisode(self.env, output_dir="./videos", save_video=True, video_fps=30)
            self.env = RecordEpisode(
                self.env,
                output_dir="./videos",
                save_trajectory=False,
                save_video=True,
                video_fps=30
            )
            logging.info(f"Recording enabled.")

        self.env.reset()
        logging.info("ManiSkill Simulator initialized and environment reset.")
        
        # Log the action space for reference
        logging.info(f"Action Space: {self.env.action_space}")

    def start(self):
        """
        Start or reset the ManiSkill environment.
        """
        logging.info("Resetting the ManiSkill environment.")
        self.env.reset()

    def save_camera_image_by_type(self, camera_type: str = "fetch_head"):
        obs = self.env.get_obs()
        sensor_data = obs['sensor_data']

        if 'sensor_data' in obs:
            sensor_data = obs['sensor_data']
            camera_image = sensor_data[camera_type]['rgb']  
            rgb_image = camera_image.squeeze(0).cpu().numpy() 
            image = Image.fromarray(rgb_image)

            timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

            image.save(f'sensor_image-{camera_type}-{timestamp}.png')
    
    def get_camera_image(self) -> np.ndarray:
        """
        Retrieve the current camera image from the simulator.
        
        Returns:
            np.ndarray: The captured image in RGB format.
        """
        obs = self.env.get_obs()
        if 'sensor_data' in obs:
            sensor_data = obs['sensor_data']
            camera_image = sensor_data["fetch_head"]['rgb']  
            
            self.save_camera_image_by_type("fetch_head")
            # self.save_camera_image_by_type("fetch_hand")
        else:
            raise KeyError("Camera observation not found in the environment observations.")
        
        camera_image = camera_image.squeeze(0).cpu().numpy() 
        return camera_image

    def close(self):
        """
        Close the ManiSkill environment.
        """
        logging.info("Closing the ManiSkill environment.")
        self.env.close()
        logging.info("ManiSkill environment closed.")
