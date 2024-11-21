from typing import Tuple, Optional
import numpy as np
import logging
import math
import torch
from scipy.spatial.transform import Rotation as R

class RobotRotator:
    def __init__(self, env):
        """
        Initialize the Movement System with the simulator and motion planner.

        Args:
            simulator (ManiSkillSimulator): The robot simulator instance.
        """
        self.env = env
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
        
    def find_current_location(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Retrieve the robot's current location in 3D coordinates with orientation.

        Returns:
            Optional[Tuple[float, float, float, float]]: (x, y, z, yaw) in meters and radians.
        """
        obs = self.env.get_obs()

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

    
    def get_current_joint_positions(self):
        """
        Extracts current joint positions from the observation.
        """
        # Assuming joint positions are stored under the key 'joint_positions' or similar
        obs = self.env.get_obs()
        
        print("obs::")
        print(obs)
        joint_positions = obs['agent']['qpos'].cpu().numpy().flatten()


        if joint_positions is not None:
            print("self.env.action_space.shape::")
            print(self.env.action_space.shape)
            return joint_positions[:8]
        else:
            raise KeyError("Joint positions not found in observation data.")
        
    def rotate_robot(self, angle_degrees: float, step_size: float = 0.1, tolerance: float = 0.01, max_steps: int = 25):
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
            curr_joint_pos = self.get_current_joint_positions()
            action_vector = np.copy(curr_joint_pos)
            base_rotation_index = 0 
            action_vector[base_rotation_index] = angular_velocity
    
            # Execute the rotation action
            self.env.step(action_vector)
    
            # Log the progress
            logging.debug(f"Step {step + 1}: Current theta={current_theta:.2f}, Angle diff={angle_diff:.2f}")
    
        else:
            logging.warning("Max rotation steps reached without achieving desired angle.")
    
        # Stop rotation by applying zero velocity
        action_vector = np.zeros(self.env.action_space.shape, dtype=np.float32)
        self.env.step(action_vector)
        logging.info("Rotation completed.")

