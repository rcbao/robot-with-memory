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
        self.init_joint_pos = self.get_current_joint_positions()

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

    def rotate_robot_to_view(self, view: str):
        """
        Reset the robot to the initial position and rotate it to the specified view.
        
        Args:
            view (str): The desired view to rotate the robot to. 
                        Options are "left", "right".
        
        Example usage:
            rotator = RobotRotator(env)
            rotator.rotate_robot_to_view("left")    # Rotate to -65 degrees
            rotator.rotate_robot_to_view("right")   # Rotate to 65 degrees
        """

        # Define target angles for each view in degrees
        target_angles = {
            "left": 65,
            "right": -65
        }

        # Retrieve the target angle based on the view
        angle = target_angles.get(view.lower())
        if angle is None:
            logging.error(f"Unknown view: '{view}'. Valid options are 'left', 'right'.")
            return

        # Determine the direction and number of steps required
        step_size_degrees = 2  # Degrees per step
        max_steps = int(abs(angle) / step_size_degrees)  # Total steps to reach target angle
        step_direction = 1 if angle > 0 else -1  # 1 for left, -1 for right

        logging.info(f"Rotating robot to '{view}' view: {angle} degrees in {max_steps} steps.")

        # Perform the rotation
        self.rotate_robot(step_size_degrees=step_size_degrees * step_direction, max_steps=max_steps)

        logging.info(f"Rotation to '{view}' view completed.")