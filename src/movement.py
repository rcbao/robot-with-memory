# /src/movement.py

from simulator import ManiSkillSimulator
from typing import Tuple, Optional
import numpy as np
import logging
import math
import torch
from scipy.spatial.transform import Rotation as R
from motionplanner import FetchArmMotionPlanningSolver

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MOCK_TARGET_LOCATION = (100, 100, 100)

class MovementSystem:
    def __init__(self, simulator: ManiSkillSimulator):
        """
        Initialize the Movement System with the simulator and motion planner.

        Args:
            simulator (ManiSkillSimulator): The robot simulator instance.
        """
        self.simulator = simulator
        logging.info("MovementSystem initialized.")

        self.DISABLE_ARM_MOVEMENT = True

        # Initialize the motion planner
        if self.DISABLE_ARM_MOVEMENT:
            logging.info("DISABLE_ARM_MOVEMENT set to true. Not initializing planner.")
        else:
            self.motion_planner = FetchArmMotionPlanningSolver(
                env=self.simulator.env,
                # TODO: Implement this Arm motion planner using RRT
            )
            logging.info("FetchArmMotionPlanningSolver initialized.")

        # Initialize base movement parameters
        self.base_speed = 0.5  # meters per second
        self.rotation_speed = 1.0  # radians per second



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

        _, _, _, initial_theta = initial_location
        target_theta = self.normalize_angle(initial_theta + angle_radians)

        for step in range(max_steps):
            current_location = self.find_current_location()
            if not current_location:
                logging.warning("Current location unavailable. Continuing rotation.")
                continue

            _, _, _, current_theta = current_location
            angle_diff = self.normalize_angle(target_theta - current_theta)

            if abs(angle_diff) < tolerance:
                logging.info("Desired rotation achieved.")
                break

            # Calculate angular velocity towards the target angle
            angular_velocity = np.clip(angle_diff, -step_size, step_size)

            # Create the action vector for rotation
            action_vector = np.zeros(self.simulator.env.action_space.shape, dtype=np.float32)
            base_rotation_index = 12  # 'root_z_rotation_joint' controls base rotation at index 12
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

    def navigate_to(self, target_coords: Tuple[float, float, float]):
        """
        Navigate the robot's base to the target coordinates.

        Args:
            target_coords (Tuple[float, float, float]): Target (x, y, z) in meters.
        """
        logging.info(f"Navigating to coordinates: {target_coords}")
        current_coords = self.find_current_location()
        if not current_coords:
            logging.error("Current location not found. Cannot navigate to target.")
            return

        delta_x = target_coords[0] - current_coords[0]
        delta_y = target_coords[1] - current_coords[1]
        delta_z = target_coords[2] - current_coords[2]
        distance = math.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
        angle_to_target = math.atan2(delta_y, delta_x)
        angle_diff = self.normalize_angle(angle_to_target - current_coords[3])  # Assuming current_theta is yaw

        # Proportional controller gains
        K_linear = 0.5
        K_angular = 1.0

        # Calculate velocities
        linear_velocity = np.clip(K_linear * distance, -self.base_speed, self.base_speed)
        angular_velocity = np.clip(K_angular * angle_diff, -self.rotation_speed, self.rotation_speed)

        logging.info(f"Computed velocities - Linear: {linear_velocity:.2f}, Angular: {angular_velocity:.2f}")

        # Create the action vector for base movement
        action_vector = np.zeros(self.simulator.env.action_space.shape, dtype=np.float32)
        # Assuming indices for linear and angular velocities; adjust based on actual action space
        action_vector[-2] = linear_velocity  # Forward/backward
        # action_vector[-1] = angular_velocity  # Rotation

        # Execute the movement action
        # TODO: fix this function
        for _ in range(100):
            obs, _, done, _, _ = self.simulator.env.step(action_vector)
            if done:
                break

    def acquire_object(self, x0: float, y0: float, z0: float):
        """
        Move the robot to the specified coordinates and fetch the object located there.

        Args:
            x0 (float): Target x-coordinate in meters.
            y0 (float): Target y-coordinate in meters.
            z0 (float): Target z-coordinate in meters.
        """
        logging.info(f"Starting fetch operation for object at ({x0}, {y0}, {z0}).")

        # Step 1: Navigate to the vicinity of the target coordinates
        target_position = (x0, y0, z0)
        if MOCK_TARGET_LOCATION:
            target_position = MOCK_TARGET_LOCATION
        self.navigate_to(target_position)
        logging.info(f"Navigated to target position: {target_position}")

        # Step 2: Lower the arm using the predefined action vector
        arm_down_vector = np.array([1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(1, 13)
        logging.info("Lowering the arm to prepare for grasping.")
        self.simulator.env.step(arm_down_vector)
        logging.info("Arm lowered.")

        # Step 3: Plan and execute the motion to grasp the object
        if self.DISABLE_ARM_MOVEMENT:
            logging.info("Arm movement disabled.")
            logging.info("Fetch operation complete.")
        else:
            logging.info("Planning motion to grasp the object.")
            target_grasp_pose = self.calculate_grasp_pose(x0, y0, z0)
            result = self.motion_planner.move_to_pose_with_RRTConnect(pose=target_grasp_pose)

            if result == -1:
                logging.error("Motion planning to grasp pose failed.")
                return

            logging.info("Motion planning and execution to grasp pose succeeded.")

            # Step 4: Close the gripper to grasp the object
            logging.info("Closing the gripper to grasp the object.")
            success = self.grasp_object(gripper_position=1.0)  # Correct usage
            if not success:
                logging.error("Gripper failed to close properly.")
                return
            logging.info("Gripper closed.")

            # Step 5: Lift the arm back to a safe position
            arm_up_vector = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0]).reshape(1, 13)
            logging.info("Lifting the arm with the object.")
            self.simulator.env.step(arm_up_vector)
            logging.info("Arm lifted.")

    def calculate_grasp_pose(self, x0: float, y0: float, z0: float) -> 'sapien.Pose':
        """
        Calculate the target grasp pose based on the object's coordinates.

        Args:
            x0 (float): Object's x-coordinate in meters.
            y0 (float): Object's y-coordinate in meters.
            z0 (float): Object's z-coordinate in meters.

        Returns:
            sapien.Pose: The calculated grasp pose.
        """
        # Define the desired grasp position slightly above the object
        grasp_position = np.array([x0, y0, z0 + 0.1])  # 10 cm above the object
        # Define the desired orientation (e.g., facing downward)
        grasp_orientation = R.from_euler('xyz', [0, math.pi, 0]).as_quat()
        grasp_pose = sapien.Pose(p=grasp_position, q=grasp_orientation)
        logging.debug(f"Calculated grasp pose: Position={grasp_position}, Orientation={grasp_orientation}")
        return grasp_pose

    def grasp_object(self, gripper_position: float = 1.0) -> bool:
        """
        Perform a grasp action by setting the gripper position.

        Args:
            gripper_position (float): Desired gripper position (0.0 for open, 1.0 for closed).

        Returns:
            bool: True if grasping was successful, False otherwise.
        """
        print("grasp_object::")
        print("gripper_position::")
        print(gripper_position)
        if not isinstance(gripper_position, (float, int)):
            logging.error(f"gripper_position must be a float or int, got {type(gripper_position)}")
            return False
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
        try:
            action_vector = np.zeros(self.simulator.env.action_space.shape, dtype=np.float32)
            gripper_index = 12  # Update based on actual action space index for Fetch's gripper
            action_vector[gripper_index] = position
            self.simulator.env.step(action_vector)
            logging.info("Gripper action executed.")
            return True  # Placeholder for actual verification
        except Exception as e:
            logging.error(f"Failed to set gripper position: {e}")
            return False

    def find_current_location(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Retrieve the robot's current location in 3D coordinates with orientation.

        Returns:
            Optional[Tuple[float, float, float, float]]: (x, y, z, yaw) in meters and radians.
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

        if len(qpos) < 7:
            logging.error("qpos does not contain enough elements for position and orientation.")
            return None

        # Extract 3D position and orientation
        base_x, base_y, base_z = qpos[:3]
        quat_x, quat_y, quat_z, quat_w = qpos[3:7]

        try:
            # Adjust the Euler sequence based on simulator's convention
            euler_angles = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler('xyz')
            yaw = euler_angles[2]
        except ValueError as e:
            logging.error(f"Error converting quaternion to euler angles: {e}")
            return None

        logging.debug(f"Current location - x: {base_x:.2f}, y: {base_y:.2f}, z: {base_z:.2f}, yaw: {yaw:.2f} radians")
        return (base_x, base_y, base_z, yaw)

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
        # Create the action vector within the permissible range
        for _ in range(100):
            action_vector = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 13)
            obs, _, done, _, _ = self.simulator.env.step(action_vector)
            if done:
                break

        logging.info("Moved the arm down to clear the head camera's view.")
        return obs