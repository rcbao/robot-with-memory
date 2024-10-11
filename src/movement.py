# /src/movement.py

from maniskill_simulator import ManiSkillSimulator
from typing import Tuple, Optional
import numpy as np
import logging
import math
import torch
from scipy.spatial.transform import Rotation as R  # Added for quaternion conversion


# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class MovementSystem:
    def __init__(self, simulator: ManiSkillSimulator):
        """
        Initialize the Movement System with the simulator.

        Args:
            simulator (ManiSkillSimulator): The robot simulator instance.
        """
        self.simulator = simulator
        logging.info("MovementSystem initialized with ManiSkillSimulator.")

    def rotate_robot(self, angle_degrees: float, step_size: float = 0.05, tolerance: float = 0.01):
        """
        Rotate the robot's base by a specific angle.

        Args:
            angle_degrees (float): The angle to rotate in degrees. Positive for clockwise, negative for counter-clockwise.
            step_size (float): The angular velocity step size.
            tolerance (float): The tolerance in radians to stop rotation.
        """
        angle_radians = math.radians(angle_degrees)
        logging.info(f"Rotating robot by {angle_degrees} degrees ({angle_radians} radians).")

        # Initialize cumulative rotation
        cumulative_rotation = 0.0

        # Determine rotation direction
        angular_velocity = step_size if angle_radians > 0 else -step_size

        # Loop until the desired rotation is achieved within the tolerance
        while abs(cumulative_rotation) < abs(angle_radians) - tolerance:
            # Set velocity: linear=0, angular=angular_velocity
            self.simulator.set_base_velocity(linear=0.0, angular=angular_velocity)
            
            # Optionally, add a small sleep to simulate time between actions
            # import time
            # time.sleep(0.1)

            # Retrieve updated location to calculate rotation
            current_location = self.find_current_location()
            if current_location:
                _, _, current_theta = current_location
                cumulative_rotation += angular_velocity  # Simplistic accumulation
                logging.debug(f"Cumulative rotation: {cumulative_rotation} radians.")

        # Stop rotation
        self.simulator.set_base_velocity(linear=0.0, angular=0.0)
        logging.info("Rotation completed.")

    def go_to(self, target_coords: Tuple[float, float, float]):
        """
        Move the robot to the target coordinates.

        Args:
            target_coords (Tuple[float, float, float]): Target (x, y, theta) in meters and radians.
        """
        logging.info(f"Moving robot to coordinates: {target_coords}")
        # Implement a simple proportional controller to reach the target coordinates

        # Retrieve current location
        current_coords = self.find_current_location()
        if not current_coords:
            logging.error("Unable to retrieve current location.")
            return

        current_x, current_y, current_theta = current_coords
        target_x, target_y, target_theta = target_coords

        # Calculate differences
        delta_x = target_x - current_x
        delta_y = target_y - current_y
        delta_theta = target_theta - current_theta

        # Calculate distance and angle to target
        distance = math.hypot(delta_x, delta_y)
        angle_to_target = math.atan2(delta_y, delta_x)
        angle_diff = self.normalize_angle(angle_to_target - current_theta)

        # Define proportional gains
        K_linear = 0.5
        K_angular = 1.0

        # Calculate velocities
        linear_velocity = K_linear * distance
        angular_velocity = K_angular * angle_diff

        # Limit velocities to prevent overshooting
        max_linear = 1.0  # meters per second
        max_angular = 2.0  # radians per second
        linear_velocity = max(-max_linear, min(max_linear, linear_velocity))
        angular_velocity = max(-max_angular, min(max_angular, angular_velocity))

        logging.debug(f"Computed velocities - Linear: {linear_velocity}, Angular: {angular_velocity}")

        # Set velocities
        self.simulator.set_base_velocity(linear=linear_velocity, angular=angular_velocity)

        # Optionally, add logic to determine when to stop
        # For simplicity, we'll assume this method is called iteratively until the robot reaches the target

    def grasp_object(self, target_coords: Tuple[float, float, float], gripper_position: float = 1.0) -> bool:
        """
        Move to the object's location and perform a grasp action.

        Args:
            target_coords (Tuple[float, float, float]): Object's (x, y, theta) in meters and radians.
            gripper_position (float): Desired gripper position (0.0 for open, 1.0 for closed).

        Returns:
            bool: True if grasping was successful, False otherwise.
        """
        logging.info(f"Attempting to grasp object at {target_coords} with gripper position {gripper_position}.")
        # Move to the object's location
        self.go_to(target_coords)

        # Implement grasp logic by setting gripper position
        # Assuming gripper control is part of the action vector, set the gripper to closed
        # This requires knowing the index of the gripper in the action vector

        action_vector = np.zeros(self.simulator.env.action_space.shape, dtype=np.float32)
        
        # Example:
        # If the gripper position is controlled by the first element in the action vector
        gripper_index = 0  # Replace with the actual index based on action space
        action_vector[gripper_index] = gripper_position

        # Execute the grasp action
        self.simulator.move_to(action_vector)

        # Optionally, verify if the object is grasped by checking the environment's state
        # This part depends on the environment's API
        # For simplicity, we'll assume the grasp is always successful
        logging.info("Grasp action executed.")
        return True  # Modify based on actual verification

    def release_object(self, gripper_position: float = 0.0):
        """
        Release the currently held object.

        Args:
            gripper_position (float): Desired gripper position (0.0 for open, 1.0 for closed).
        """
        logging.info(f"Releasing object with gripper position {gripper_position}.")

        # Set gripper to open
        action_vector = np.zeros(self.simulator.env.action_space.shape, dtype=np.float32)
        
        # Example:
        # If the gripper position is controlled by the first element in the action vector
        gripper_index = 0  # Replace with the actual index based on action space
        action_vector[gripper_index] = gripper_position

        # Execute the release action
        self.simulator.move_to(action_vector)
        logging.info("Release action executed.")

    def find_current_location(self) -> Optional[Tuple[float, float, float]]:
        """
        Use the simulator to find the current location of the robot.

        Returns:
            Optional[Tuple[float, float, float]]: Current (x, y, theta) in meters and radians, or None if unavailable.
        """
        logging.info("Retrieving current location from simulator.")
        obs = self.simulator.env.get_obs()

        if 'agent' in obs and 'qpos' in obs['agent']:
            qpos = obs['agent']['qpos']  # Tensor of shape [1, N]
            if isinstance(qpos, torch.Tensor):
                qpos = qpos.cpu().numpy()
                logging.debug(f"qpos shape: {qpos.shape}")
                if qpos.ndim == 2 and qpos.shape[0] == 1:
                    qpos = qpos[0]  # Shape: [N]
                else:
                    logging.error(f"Unexpected qpos shape: {qpos.shape}")
                    return None
            elif isinstance(qpos, np.ndarray):
                logging.debug(f"qpos shape: {qpos.shape}")
                if qpos.ndim == 2 and qpos.shape[0] == 1:
                    qpos = qpos[0]  # Shape: [N]
                else:
                    logging.error(f"Unexpected qpos shape: {qpos.shape}")
                    return None
            else:
                logging.error(f"Unsupported qpos type: {type(qpos)}")
                return None

            # Extract base position (x, y, z)
            base_x = qpos[0]
            base_y = qpos[1]
            base_z = qpos[2]

            # Extract base orientation quaternion (x, y, z, w)
            quat_x = qpos[3]
            quat_y = qpos[4]
            quat_z = qpos[5]
            quat_w = qpos[6]

            # Convert quaternion to Euler angles (roll, pitch, yaw)
            rotation = R.from_quat([quat_x, quat_y, quat_z, quat_w])
            euler = rotation.as_euler('xyz', degrees=False)
            roll, pitch, yaw = euler  # radians

            # Assuming theta is yaw
            theta = yaw

            logging.debug(f"Current location - x: {base_x}, y: {base_y}, theta: {theta}")
            return (base_x, base_y, theta)
        else:
            logging.error("'agent' or 'qpos' not found in observations.")
            return None

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """
        Normalize an angle to the range [-pi, pi].

        Args:
            angle (float): The angle to normalize in radians.

        Returns:
            float: The normalized angle in radians.
        """
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
