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

    def move_to_joint_positions(self, target_joint_positions: np.ndarray, step_size_degrees: float = 1, max_steps: int = 50):
        """
        Move the robot smoothly to the specified joint positions.

        Args:
            target_joint_positions (np.ndarray): Desired joint positions array.
            step_size_degrees (float): Maximum degrees to move per step.
            max_steps (int): Maximum number of steps to reach the target.
        """
        current_joint_positions = self.get_current_joint_positions()
        if current_joint_positions is None:
            logging.error("Current joint positions not available.")
            return

        step_size_radians = math.radians(step_size_degrees)
        for step in range(max_steps):
            error = target_joint_positions - current_joint_positions
            if np.all(np.abs(error) <= step_size_radians):
                current_joint_positions = target_joint_positions
            else:
                step_direction = np.sign(error)
                step_increment = step_direction * step_size_radians
                current_joint_positions += step_increment

            action_vector = current_joint_positions
            self.env.step(action_vector)

            logging.debug(f"Step {step + 1}: Moving joints towards target positions.")

            if np.allclose(current_joint_positions, target_joint_positions, atol=step_size_radians):
                logging.debug("Target joint positions reached.")
                break
    
    def reset_joint_positions(self):
        self.move_to_joint_positions(self.init_joint_pos)

    def rotate_robot_to_view(self, view: str):
        """
        Reset the robot to the initial position and rotate it to the specified view.
        
        Args:
            view (str): The desired view to rotate the robot to. 
                        Options are "left", "center", "right".
        
        Example usage:
            rotator = RobotRotator(env)
            rotator.rotate_robot_to_view("left")    # Rotate to -60 degrees
            rotator.rotate_robot_to_view("center")  # Rotate to 0 degrees
            rotator.rotate_robot_to_view("right")   # Rotate to 60 degrees
        """
        # Reset the robot to the initial joint positions
        self.reset_joint_positions()
        logging.info("Robot joint positions have been reset to the initial configuration.")

        # Define target angles for each view in degrees
        target_angles = {
            "left": 60,
            "center": 0,
            "right": -60
        }

        # Retrieve the target angle based on the view
        angle = target_angles.get(view.lower())
        if angle is None:
            logging.error(f"Unknown view: '{view}'. Valid options are 'left', 'center', 'right'.")
            return

        # Determine the direction and number of steps required
        step_size_degrees = 3  # Degrees per step
        max_steps = int(abs(angle) / step_size_degrees)  # Total steps to reach target angle
        step_direction = 1 if angle > 0 else -1  # 1 for left, -1 for right

        logging.info(f"Rotating robot to '{view}' view: {angle} degrees in {max_steps} steps.")

        # Perform the rotation
        self.rotate_robot(step_size_degrees=step_size_degrees * step_direction, max_steps=max_steps)

        logging.info(f"Rotation to '{view}' view completed.")