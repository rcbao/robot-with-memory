# /simulator/maniskill_simulator.py

import gymnasium as gym
from gymnasium.envs.registration import register
from mani_skill.utils.building import URDFLoader
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.record import RecordEpisode
from typing import Any, Tuple, Optional
import numpy as np



class ManiSkillSimulator:
    def __init__(self):
        """
        Initialize the ManiSkill simulator with a specific environment configuration.
        
        Args:
            config_path (str): Path to the ManiSkill environment configuration file.
        """
        self.env = gym.make('ReplicaCAD_SceneManipulation-v1', render_mode="rgb_array")
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
        # Define home position actions based on the environment's API
        home_action = self.env.get_home_action()
        self.env.step(home_action)
    
    def get_camera_image(self) -> np.ndarray:
        """
        Retrieve the current camera image from the simulator.
        
        Returns:
            np.ndarray: The captured image in RGB format.
        """
        obs = self.env.get_obs()
        camera_image = obs.get('image')  # Adjust based on actual observation keys
        return camera_image
    
    def move_to(self, component: str, coords: Tuple[float, float, float]):
        """
        Move the specified robot component to the given coordinates.
        
        Args:
            component (str): The robot component to move (e.g., 'base', 'gripper').
            coords (Tuple[float, float, float]): Target coordinates (x, y, theta).
        """
        # Define movement actions based on component and target coordinates
        action = self.env.get_action_for_movement(component, coords)
        self.env.step(action)
    
    def set_base_velocity(self, linear: float, angular: float):
        """
        Set the robot base's linear and angular velocity.
        
        Args:
            linear (float): Linear velocity.
            angular (float): Angular velocity.
        """
        action = self.env.get_action_for_velocity(linear, angular)
        self.env.step(action)
    
    def stop(self):
        """
        Stop the ManiSkill environment and perform cleanup.
        """
        self.env.close()
