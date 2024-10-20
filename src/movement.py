# /src/movement.py

from maniskill_simulator import ManiSkillSimulator
from typing import Tuple, Optional
import numpy as np
import logging
import math
import torch
from scipy.spatial.transform import Rotation as R

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MovementSystem:
    def __init__(self, simulator: ManiSkillSimulator):
        """
        Initialize the Movement System with the simulator.

        Args:
            simulator (ManiSkillSimulator): The robot simulator instance.
        """
        self.simulator = simulator
        logging.info("MovementSystem initialized.")

    def rotate_robot(self, angle_degrees: float, step_size: float = 0.1, tolerance: float = 0.01, max_steps: int = 100):
        """
        Rotate the robot's base by a specific angle.

        Args:
            angle_degrees (float): The angle to rotate in degrees.
            step_size (float): The angular velocity step size in radians per step.
            tolerance (float): The tolerance in radians to stop rotation.
            max_steps (int): Maximum number of steps to prevent infinite loops.
        """
        angle_radians = math.radians(angle_degrees)
        logging.info(f"Rotating robot by {angle_degrees} degrees ({angle_radians:.2f} radians).")

        initial_location = self.find_current_location()
        if not initial_location:
            logging.error("Initial location not found. Aborting rotation.")
            return

        _, _, initial_theta = initial_location
        target_theta = self.normalize_angle(initial_theta + angle_radians)

        for step in range(max_steps):
            current_location = self.find_current_location()
            if not current_location:
                logging.warning("Current location unavailable. Continuing rotation.")
                continue

            _, _, current_theta = current_location
            angle_diff = self.normalize_angle(target_theta - current_theta)

            if abs(angle_diff) < tolerance:
                logging.info("Desired rotation achieved.")
                break

            # Calculate angular velocity towards the target angle
            angular_velocity = np.clip(angle_diff, -step_size, step_size)

            # Create the action vector for rotation
            action_vector = np.zeros(self.simulator.env.action_space.shape, dtype=np.float32)
            base_rotation_index = 12  # 'root_z_rotation_joint' controls base rotation at index 14
            action_vector[base_rotation_index] = angular_velocity

            # Execute the rotation action
            self.simulator.env.step(action_vector)

            # Log the progress
            logging.debug(f"Step {step + 1}: Current theta={current_theta:.2f}, Angle diff={angle_diff:.2f}")

        else:
            logging.warning("Max rotation steps reached without achieving desired angle.")

        # Stop rotation by applying zero velocity
        action_vector = np.zeros(self.simulator.env.action_space.shape, dtype=np.float32)
        self.simulator.env.step(action_vector)
        logging.info("Rotation completed.")


    def go_to(self, target_coords: Tuple[float, float, float]):
        """
        Move the robot to the target coordinates.

        Args:
            target_coords (Tuple[float, float, float]): Target (x, y, theta) in meters and radians.
        """
        logging.info(f"Moving to coordinates: {target_coords}")
        current_coords = self.find_current_location()
        if not current_coords:
            logging.error("Current location not found. Cannot move to target.")
            return

        delta_x = target_coords[0] - current_coords[0]
        delta_y = target_coords[1] - current_coords[1]
        distance = math.hypot(delta_x, delta_y)
        angle_to_target = math.atan2(delta_y, delta_x)
        angle_diff = self.normalize_angle(angle_to_target - current_coords[2])

        # Proportional controller gains
        K_linear = 0.5
        K_angular = 1.0

        # Calculate velocities
        linear_velocity = np.clip(K_linear * distance, -1.0, 1.0)
        angular_velocity = np.clip(K_angular * angle_diff, -2.0, 2.0)

        logging.debug(f"Computed velocities - Linear: {linear_velocity:.2f}, Angular: {angular_velocity:.2f}")

        self.simulator.set_velocity(linear=linear_velocity, angular=angular_velocity)

    def grasp_object(self, gripper_position: float = 1.0) -> bool:
        """
        Perform a grasp action by setting the gripper position.

        Args:
            gripper_position (float): Desired gripper position (0.0 for open, 1.0 for closed).

        Returns:
            bool: True if grasping was successful, False otherwise.
        """
        logging.info(f"Grasping object with gripper position: {gripper_position}")
        return self._set_gripper(gripper_position)

    def release_object(self, gripper_position: float = 0.0):
        """
        Perform a release action by setting the gripper position.

        Args:
            gripper_position (float): Desired gripper position (0.0 for open, 1.0 for closed).
        """
        logging.info(f"Releasing object with gripper position: {gripper_position}")
        self._set_gripper(gripper_position)

    def _set_gripper(self, position: float) -> bool:
        """
        Helper method to set the gripper position.

        Args:
            position (float): Desired gripper position.

        Returns:
            bool: True if action executed, False otherwise.
        """
        action_vector = np.zeros(self.simulator.env.action_space.shape, dtype=np.float32)
        gripper_index = 0  # Update based on actual action space
        action_vector[gripper_index] = position
        self.simulator.move_to(action_vector)
        logging.info("Gripper action executed.")
        return True  # Placeholder for actual verification

    def find_current_location(self) -> Optional[Tuple[float, float, float]]:
        """
        Retrieve the robot's current location.

        Returns:
            Optional[Tuple[float, float, float]]: (x, y, theta) in meters and radians.
        """
        obs = self.simulator.env.get_obs()

        if 'agent' not in obs or 'qpos' not in obs['agent']:
            logging.error("'agent' or 'qpos' not found in observations.")
            return None

        qpos = obs['agent']['qpos']
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.cpu().numpy().flatten()
        elif isinstance(qpos, np.ndarray):
            qpos = qpos.flatten()
        else:
            logging.error(f"Unsupported qpos type: {type(qpos)}")
            return None

        # Extract position and orientation
        base_x, base_y, _ = qpos[:3]
        quat_x, quat_y, quat_z, quat_w = qpos[3:7]
        yaw = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler('xyz')[2]

        logging.debug(f"Current location - x: {base_x:.2f}, y: {base_y:.2f}, theta: {yaw:.2f} radians")
        return (base_x, base_y, yaw)

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """
        Normalize an angle to the range [-pi, pi].

        Args:
            angle (float): The angle to normalize in radians.

        Returns:
            float: The normalized angle in radians.
        """
        return (angle + math.pi) % (2 * math.pi) - math.pi
        
    def move_arm_down_to_clear_view(self):
        """
        Moves the robot's arm down to ensure it does not block the head camera's view.
        This is done by adjusting relevant joints in the arm.

        Returns:
            None
        """
        # Create the action vector with all zeros (no movement for other joints)
        for _ in range(100):
            action_vector = action = np.array([100.0, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 13)
            obs, _, done, _, _ = self.simulator.env.step(action_vector)
            if done:
                break

        logging.info("Moved the arm down to clear the head camera's view.")

        return obs