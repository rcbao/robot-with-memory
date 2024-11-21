from typing import Optional
import numpy as np
import logging
import math

class RobotRotator:
    def __init__(self, env):
        """
        Initialize the RobotRotator with the simulation environment.
        
        Args:
            env: The robot's simulation environment.
        """
        self.env = env

    def get_current_joint_positions(self) -> Optional[np.ndarray]:
        """
        Extract current joint positions.
        
        Returns:
            A numpy array of joint positions or None if retrieval fails.
        """
        obs = self.env.get_obs()
        try:
            joint_positions = obs['agent']['qpos'].cpu().numpy().flatten()
            return joint_positions[:8]  # Assuming first 8 joints are controllable
        except (KeyError, AttributeError):
            logging.error("Failed to retrieve joint positions.")
            return None

    def rotate_robot(self, step_size_degrees: float = 3, max_steps: int = 15):
        """
        Rotate the robot's base smoothly by a specified step size.
        
        Args:
            step_size_degrees (float): Rotation angle per step in degrees. Positive for left, negative for right.
            max_steps (int): Maximum number of rotation steps to perform.
        
        Example usage:
            rotator = RobotRotator(env)
            rotator.rotate_robot()  # Rotate 45 degrees left (3 degrees * 15 steps)
            rotator.rotate_robot(step_size_degrees=-3, max_steps=30)  # Rotate 90 degrees right
        """
        step_radians = math.radians(step_size_degrees)

        for step in range(max_steps):
            joint_positions = self.get_current_joint_positions()
            if joint_positions is None:
                logging.warning(f"Step {step + 1}: Unable to retrieve joint positions. Skipping step.")
                continue

            # Update the first joint (assumed to control base rotation)
            joint_positions[0] += step_radians
            action_vector = joint_positions

            # Command the environment to perform the action
            self.env.step(action_vector)

            logging.debug(f"Step {step + 1}: Rotated by {step_size_degrees} degrees.")

        # Optionally, maintain the final position
        final_joint_positions = self.get_current_joint_positions()
        if final_joint_positions is not None:
            self.env.step(final_joint_positions)
            total_rotation = step_size_degrees * max_steps
            logging.info(f"Rotation by {total_rotation} degrees completed.")
        else:
            logging.error("Failed to maintain final joint positions.")
