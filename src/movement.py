# /src/movement.py

from simulator import ManiSkillSimulator
from typing import Tuple, Optional
import numpy as np
import logging
import math
import torch
import os
from mplib import Planner
from scipy.spatial.transform import Rotation as R
from motionplanner import FetchMotionPlanningSolver

# from mplib.examples.demo_setup import DemoSetup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def setup_planner(env_agent):
    """
    Sets up the motion planner using mplib for the Fetch robot.

    Returns:
        mplib.Planner: The initialized motion planner.
    """
    try:
        MOVE_GROUP = "gripper_link"

        link_names = [
            'shoulder_pan_link', 
            'shoulder_lift_link',
            'elbow_flex_link', 
            'wrist_flex_link', 
            'wrist_roll_link', 
            'gripper_link'
        ]
        
        joint_names = [
            'shoulder_pan_joint', 
            'shoulder_lift_joint',
            'elbow_flex_joint', 
            'wrist_flex_joint', 
            'wrist_roll_joint'
        ]
        
        # Path to the meshes directory
        mesh_dir = os.path.dirname(env_agent.urdf_path) + "/fetch_description/meshes/"

        print("mesh dir::")
        print(mesh_dir)
        
        # Filter links to include only those with convex collision meshes
        valid_link_names = []
        for link in link_names:
            collision_mesh = f"{link}_collision.STL.convex.stl"
            collision_mesh_path = os.path.join(mesh_dir, collision_mesh)
            if os.path.exists(collision_mesh_path):
                valid_link_names.append(link)
                logging.info(f"Including link '{link}' with convex collision mesh.")
            else:
                logging.warning(f"Skipping link '{link}': Convex collision mesh '{collision_mesh}' not found.")
        
        if not valid_link_names:
            raise ValueError("No valid links with convex collision meshes found. Planner cannot be initialized.")
        
        planner = Planner(
            urdf=env_agent.urdf_path,
            # srdf=env_agent.urdf_path.replace(".urdf", ".srdf"),
            # user_link_names=valid_link_names,
            # user_joint_names=joint_names,
            move_group=MOVE_GROUP
        )
        logging.info("Motion planner successfully initialized.")
        return planner
    except Exception as e:
        logging.error(f"Failed to set up planner: {e}")
        raise


def setup_planner_v2(env, seed=None, debug=False, vis=False):
    return FetchMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        print_env_info=True,
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
    ) 

class MovementSystem:
    def __init__(self, simulator: ManiSkillSimulator):
        """
        Initialize the Movement System with the simulator and motion planner.

        Args:
            simulator (ManiSkillSimulator): The robot simulator instance.
        """
        self.simulator = simulator
        self.planner = setup_planner_v2(simulator.env)

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

    def navigate_to(self, target_coords: Tuple[float, float, float]):
        target_pose = sapien.Pose(p=list(target_coords), q=[w, x, y, z])

        # Get the current joint positions
        current_qpos = self.planner.robot.get_qpos()

        # Plan the path
        result = self.planner.plan_pose(
            goal_pose=target_pose,
            current_qpos=current_qpos,
            time_step=1 / 250,
            verbose=True
        )

        if result["status"] != "Success":
            print(f"Planning failed with status: {result['status']}")
            print(result["status"])
            return -1
        self.follow_path()

    def follow_path(self, result):
        """Helper function to follow a path generated by the planner"""
        # number of waypoints in the path
        n_step = result["position"].shape[0]
        # this makes sure the robot stays neutrally boyant instead of sagging
        # under gravity
        for i in range(n_step):
            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)
            # set the joint positions and velocities for move group joints only.
            # The others are not the responsibility of the planner
            for j in range(len(self.planner.move_group_joint_indices)):
                self.active_joints[j].set_drive_target(result["position"][i][j])
                self.active_joints[j].set_drive_velocity_target(
                    result["velocity"][i][j]
                )
            # simulation step
            self.scene.step()
            # render every 4 simulation steps to make it faster
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()

    def acquire_object(self, x0: float, y0: float, z0: float):
        """
        Move the robot to the specified coordinates and fetch the object located there.

        Args:
            x0 (float): Target x-coordinate in meters.
            y0 (float): Target y-coordinate in meters.
            z0 (float): Target z-coordinate in meters.
        """
        pass

    def grasp_object(self, gripper_position: float = 1.0) -> bool:
        """
        Perform a grasp action by setting the gripper position.

        Args:
            gripper_position (float): Desired gripper position (0.0 for open, 1.0 for closed).

        Returns:
            bool: True if grasping was successful, False otherwise.
        """
        pass

    def release_object(self, gripper_position: float = 0.0):
        """
        Perform a release action by setting the gripper position.

        Args:
            gripper_position (float): Desired gripper position (0.0 for open, 1.0 for closed).
        """
        pass
