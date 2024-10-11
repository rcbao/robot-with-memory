# /simulator/maniskill_simulator.py
import datetime
import gymnasium as gym
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.record import RecordEpisode
from typing import Any, Tuple, Optional
import numpy as np
from mani_skill.sensors.camera import CameraConfig
from PIL import Image



class ManiSkillSimulator:
    def __init__(self):
        """
        Initialize the ManiSkill simulator with a specific environment configuration.
        
        Args:
            config_path (str): Path to the ManiSkill environment configuration file.
        """

        self.env = gym.make(
            'ReplicaCAD_SceneManipulation-v1', 
            render_mode="rgb_array", 
            obs_mode="rgbd",
            sensor_configs=dict(width=1920, height=1440)
            )
        self.env.reset()
    
    def start(self):
        """
        Start the ManiSkill environment.
        """
        self.env.reset()
    
    def home(self):
        """
        Move the robot to the home position.
        """
        home_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        self.env.step(home_action)

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
            print("sensor_data types::")
            print(sensor_data.keys())
            camera_image = sensor_data["fetch_head"]['rgb']  
            
            self.save_camera_image_by_type("fetch_head")
            self.save_camera_image_by_type("fetch_hand")
        else:
            raise KeyError("Camera observation not found in the environment observations.")
        
        camera_image = camera_image.squeeze(0).cpu().numpy() 
        return camera_image
    
    def move_to(self, action_vector: np.ndarray):
        """
        Move the robot by sending an action vector.
        
        Args:
            action_vector (np.ndarray): The action vector conforming to the environment's action space.
        """
        if not isinstance(action_vector, np.ndarray):
            raise TypeError("action_vector must be a NumPy array.")
        
        if action_vector.shape != self.env.action_space.shape:
            raise ValueError(f"action_vector must have shape {self.env.action_space.shape}, but got {action_vector.shape}.")
        
        # Execute the action
        obs, reward, terminated, truncated, info = self.env.step(action_vector)
        print("[DEBUG] Action executed.")
        if terminated or truncated:
            print("[INFO] Episode terminated or truncated.")
            self.env.reset()
    
    def set_base_velocity(self, linear: float, angular: float):
        """
        Set the robot base's linear and angular velocity.
        
        Args:
            linear (float): Linear velocity.
            angular (float): Angular velocity.
        """
        action_vector = np.zeros(self.env.action_space.shape, dtype=np.float32)
        action_vector[0] = linear
        action_vector[1] = angular
        
        self.move_to(action_vector)
    
    def stop(self):
        """
        Stop the ManiSkill environment and perform cleanup.
        """
        self.env.close()
