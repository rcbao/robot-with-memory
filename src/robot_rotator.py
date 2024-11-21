from typing import Optional, Tuple
import numpy as np
import logging
import math
import torch
from scipy.spatial.transform import Rotation as R

class RobotRotator:
    def __init__(self, env):
        """
        Initialize the RobotRotator with the simulation environment.
        
        Args:
            env: The robot's simulation environment.
        """
        self.env = env

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def get_current_yaw(self) -> Optional[float]:
        """Retrieve the robot's current yaw angle in radians."""
        obs = self.env.get_obs()
        try:
            qpos = obs['agent']['qpos'].cpu().numpy().flatten()
            quat = qpos[3:7]
            euler = R.from_quat(quat).as_euler('xyz')
            yaw = euler[2]
            return yaw
        except (KeyError, ValueError, AttributeError):
            logging.error("Failed to retrieve yaw from observations.")
            return None

    def get_current_joint_positions(self) -> Optional[np.ndarray]:
        """Extract current joint positions."""
        obs = self.env.get_obs()
        try:
            joint_positions = obs['agent']['qpos'].cpu().numpy().flatten()
            return joint_positions[:8]  # Assuming first 8 joints are controllable
        except (KeyError, AttributeError):
            logging.error("Failed to retrieve joint positions.")
            return None

    def rotate_robot(self, angle_degrees: float, step_size_degrees: float = 3, 
                     tolerance_degrees: float = 0.5, max_steps: int = 15):
        """
        Rotate the robot's base smoothly by a specified angle.
        
        Args:
            angle_degrees (float): Angle to rotate in degrees.
            step_size_degrees (float): Rotation step size per iteration in degrees.
            tolerance_degrees (float): Acceptable deviation in degrees.
            max_steps (int): Maximum rotation steps to prevent infinite loops.
        """
        target_radians = math.radians(angle_degrees)
        step_radians = math.radians(step_size_degrees)
        tolerance_radians = math.radians(tolerance_degrees)

        current_yaw = self.get_current_yaw()
        if current_yaw is None:
            return

        target_yaw = self.normalize_angle(current_yaw + target_radians)

        for _ in range(max_steps):
            current_yaw = self.get_current_yaw()
            if current_yaw is None:
                continue

            angle_diff = self.normalize_angle(target_yaw - current_yaw)
            rotation_step = np.clip(angle_diff, -step_radians, step_radians)

            joint_positions = self.get_current_joint_positions()
            joint_positions[0] = self.normalize_angle(joint_positions[0] + rotation_step)
            action_vector = joint_positions

            self.env.step(action_vector)

        # Maintain the final position
        final_joint_positions = self.get_current_joint_positions()
        if final_joint_positions is not None:
            self.env.step(final_joint_positions)
            logging.info(f"Rotation by {angle_degrees} degrees completed.")
        else:
            logging.error("Failed to maintain final joint positions.")
