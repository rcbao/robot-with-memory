import argparse
import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from init_env import add_object_to_scene_ycb, init_env
from mani_skill.utils.wrappers.record import RecordEpisode
from typing import Optional, Tuple
import logging
import math
import torch
from scipy.spatial.transform import Rotation as R

def move_to_pose(planner, target_pose, dry_run=False):
    result = planner.move_to_pose_with_screw(target_pose, dry_run=dry_run)
    if result == -1:
        print("move_to_pose_with_screw failed, falling back to move_to_pose_with_RRTConnect")
        result = planner.move_to_pose_with_RRTConnect(target_pose, dry_run=dry_run)
    return result != -1


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



def fetch_and_place_target_object(env, target_object, dest_coords, debug=False, vis=False):
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    FINGER_LENGTH = 0.025
    env = env.unwrapped
    approaching = np.array([0, 0, -1])

    # Fetch the target object
    # Compute the grasp pose
    obb = get_actor_obb(target_object)
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # Adjust grasp pose for valid grasp
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose * delta_pose
        res = move_to_pose(planner, grasp_pose2, dry_run=True)
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print("Failed to find a valid grasp pose for the target object.")
        return False

    # Reach, grasp, and initial lift
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    move_to_pose(planner, reach_pose)
    move_to_pose(planner, grasp_pose)
    planner.close_gripper()
    
    # Adjusted initial lift after grasping the object
    # Calculate a dynamic and safe lift height
    safe_lift_height = min(0.1, 0.5 - grasp_pose.p[2])  # Ensure lift stays within workspace bounds
    lift_pose = sapien.Pose([0, 0, safe_lift_height]) * grasp_pose

    move_to_pose(planner, lift_pose)
    
    # Descend to place the object at the target location
    final_destination_pose = sapien.Pose(dest_coords, lift_pose.q)  # Final target position
    move_to_pose(planner, final_destination_pose)
    
    # Release the object
    planner.open_gripper()
    
    planner.close()
    return True




def main():
    env = init_env()  # Initialize the environment
    target_object = env.unwrapped.banana
    dest_coords = [0.05, 0.05, 0]
    try:
        result = fetch_and_place_target_object(env, target_object, dest_coords, vis=False)
        print("Operation result:", "Success" if result else "Failed")
    finally:
        env.close()

if __name__ == "__main__":
    main()
