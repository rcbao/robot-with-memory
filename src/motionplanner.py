import os
import mplib
import numpy as np
import sapien
import trimesh
import logging

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.pose import to_sapien_pose
from transforms3d import quaternions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for gripper states
OPEN = 1
CLOSED = -1


class FetchArmMotionPlanningSolver:
    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib might have constraints on robot base
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
        move_group: str = "fetch_gripper",  # Fetch's end-effector group name
    ):
        """
        Initialize the FetchArmMotionPlanningSolver with the simulation environment and parameters.

        Args:
            env (BaseEnv): The simulation environment instance.
            debug (bool): Enables debug mode if True.
            vis (bool): Enables visualization if True.
            base_pose (sapien.Pose, optional): The initial pose of the robot's base.
            visualize_target_grasp_pose (bool): Visualizes the target grasp pose.
            print_env_info (bool): Prints environment information during execution.
            joint_vel_limits (float): Joint velocity limits.
            joint_acc_limits (float): Joint acceleration limits.
            move_group (str): The move group name for the Fetch robot's end-effector.
        """
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.env_agent: BaseAgent = self.base_env.agent
        self.robot = self.env_agent.robot
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits

        self.base_pose = to_sapien_pose(base_pose) if base_pose else sapien.Pose()

        self.move_group = move_group  # Updated for Fetch robot

        self.planner = self.setup_planner()
        self.control_mode = self.base_env.control_mode

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.gripper_state = OPEN
        self.grasp_pose_visual = None
        if self.vis and self.visualize_target_grasp_pose:
            if "grasp_pose_visual" not in self.base_env.scene.actors:
                self.grasp_pose_visual = build_fetch_gripper_grasp_pose_visual(
                    self.base_env.scene
                )
            else:
                self.grasp_pose_visual = self.base_env.scene.actors["grasp_pose_visual"]
            self.grasp_pose_visual.set_pose(self.base_env.agent.tcp.pose)
        self.elapsed_steps = 0

        self.use_point_cloud = False
        self.collision_pts_changed = False
        self.all_collision_pts = None

    def render_wait(self):
        if not self.vis or not self.debug:
            return
        print("Press [c] to continue")
        viewer = self.base_env.render_human()
        while True:
            if viewer.window.key_down("c"):
                break
            self.base_env.render_human()

    def setup_planner(self):
        """
        Sets up the motion planner using mplib for the Fetch robot.

        Returns:
            mplib.Planner: The initialized motion planner.
        """
        try:
            link_names = [link.get_name() for link in self.robot.get_links()]
            joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
            
            # Path to the meshes directory
            mesh_dir = os.path.dirname(self.env_agent.urdf_path) + "/meshes/"
            
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
            
            planner = mplib.Planner(
                urdf=self.env_agent.urdf_path,
                srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
                user_link_names=valid_link_names,
                user_joint_names=joint_names,
                move_group=self.move_group,  # Updated for Fetch
                joint_vel_limits=np.ones(len(joint_names)) * self.joint_vel_limits,
                joint_acc_limits=np.ones(len(joint_names)) * self.joint_acc_limits,
            )
            planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
            logging.info("Motion planner successfully initialized.")
            return planner
        except Exception as e:
            logging.error(f"Failed to set up planner: {e}")
            raise

    def follow_path(self, result, refine_steps: int = 0):
        """
        Executes the planned path by sending actions to the environment.

        Args:
            result (dict): The result dictionary from the planner containing positions and velocities.
            refine_steps (int): Additional steps to refine the path.

        Returns:
            tuple: The last observation, reward, termination flags, and info from the environment.
        """
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action = np.hstack([qpos, qvel, self.gripper_state])
            else:
                action = np.hstack([qpos, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                logging.info(f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}")
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def move_to_pose_with_RRTConnect(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        """
        Plans and optionally executes a motion path to the specified pose using the RRT-Connect algorithm.

        Args:
            pose (sapien.Pose): The target pose to move the robot's end-effector to.
            dry_run (bool): If True, only plan the path without executing it.
            refine_steps (int): Number of additional steps to refine the path.

        Returns:
            dict or int: The result of the environment step if executed, or the planning result if dry_run is True.
                         Returns -1 if planning fails.
        """
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)
        pose = sapien.Pose(p=pose.p, q=pose.q)
        result = self.planner.plan_qpos_to_pose(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
            wrt_world=True,
        )
        if result["status"] != "Success":
            logging.error(f"Motion planning failed with status: {result['status']}")
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_screw(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        """
        Plans and optionally executes a motion path to the specified pose using a screw-based planning approach.

        Args:
            pose (sapien.Pose): The target pose to move the robot's end-effector to.
            dry_run (bool): If True, only plan the path without executing it.
            refine_steps (int): Number of additional steps to refine the path.

        Returns:
            dict or int: The result of the environment step if executed, or the planning result if dry_run is True.
                         Returns -1 if planning fails.
        """
        pose = to_sapien_pose(pose)
        # Attempt screw planning twice before giving up
        for attempt in range(2):
            if self.grasp_pose_visual is not None:
                self.grasp_pose_visual.set_pose(pose)
            pose = sapien.Pose(p=pose.p, q=pose.q)
            result = self.planner.plan_screw(
                np.concatenate([pose.p, pose.q]),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                use_point_cloud=self.use_point_cloud,
            )
            if result["status"] == "Success":
                logging.info(f"Screw planning succeeded on attempt {attempt + 1}.")
                break
            logging.warning(f"Screw planning attempt {attempt + 1} failed with status: {result['status']}")
        else:
            logging.error("Screw planning failed after multiple attempts.")
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def open_gripper(self, steps: int = 6):
        """
        Opens the Fetch robot's gripper.

        Args:
            steps (int): Number of action steps to execute the gripper opening.

        Returns:
            tuple: The last observation, reward, termination flags, and info from the environment.
        """
        self.gripper_state = OPEN
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        for i in range(steps):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                # Assuming Fetch gripper has a single control parameter
                action = np.hstack([qpos, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                logging.info(f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}")
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def close_gripper(self, steps: int = 6):
        """
        Closes the Fetch robot's gripper.

        Args:
            steps (int): Number of action steps to execute the gripper closing.

        Returns:
            tuple: The last observation, reward, termination flags, and info from the environment.
        """
        self.gripper_state = CLOSED
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        for i in range(steps):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                # Assuming Fetch gripper has a single control parameter
                action = np.hstack([qpos, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                logging.info(f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}")
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def add_box_collision(self, extents: np.ndarray, pose: sapien.Pose):
        """
        Adds collision points from a box to the planner's point cloud.

        Args:
            extents (np.ndarray): The dimensions of the box.
            pose (sapien.Pose): The pose of the box in the environment.
        """
        self.use_point_cloud = True
        box = trimesh.creation.box(extents, transform=pose.to_transformation_matrix())
        pts, _ = trimesh.sample.sample_surface(box, 256)
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def add_collision_pts(self, pts: np.ndarray):
        """
        Adds arbitrary collision points to the planner's point cloud.

        Args:
            pts (np.ndarray): Array of collision points.
        """
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def clear_collisions(self):
        """
        Clears all collision points from the planner's point cloud.
        """
        self.all_collision_pts = None
        self.use_point_cloud = False

    def close(self):
        """
        Placeholder for cleanup operations. Implement resource release if necessary.
        """
        pass


def build_fetch_gripper_grasp_pose_visual(scene: ManiSkillScene):
    """
    Builds visual aids for the Fetch robot's grasp pose in the simulation.

    Args:
        scene (ManiSkillScene): The simulation scene.

    Returns:
        sapien.Actor: The visual representation actor for the grasp pose.
    """
    builder = scene.create_actor_builder()
    grasp_pose_visual_width = 0.01
    grasp_width = 0.05

    # Sphere to indicate the grasp point
    builder.add_sphere_visual(
        pose=sapien.Pose(p=[0, 0, 0.0]),
        radius=grasp_pose_visual_width,
        material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, 0.7])
    )

    # Box to represent the gripper fingers
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.05]),
        half_size=[grasp_pose_visual_width, grasp_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )

    # Additional visual elements for clarity
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.02 - grasp_pose_visual_width * 3,
                grasp_width + grasp_pose_visual_width,
                0.02 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.03, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.02 - grasp_pose_visual_width * 3,
                -grasp_width - grasp_pose_visual_width,
                0.02 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.03, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7]),
    )
    grasp_pose_visual = builder.build_kinematic(name="grasp_pose_visual")
    return grasp_pose_visual
