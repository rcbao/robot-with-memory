import mplib
import numpy as np
import sapien
import trimesh

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.pose import to_sapien_pose
import sapien.physx as physx

OPEN = 1
CLOSED = -1

class FetchMotionPlanningSolver:
    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None, 
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
    ):
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.env_agent: BaseAgent = self.base_env.agent
        self.robot = self.env_agent.robot
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits

        self.base_pose = to_sapien_pose(base_pose)

        self.planner = self.setup_planner()
        self.control_mode = self.base_env.control_mode

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.gripper_state = OPEN
        self.grasp_pose_visual = None


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

    def follow_path(self, result, refine_steps: int = 0):
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
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def move_to_pose_with_RRTConnect(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
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
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_screw(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = to_sapien_pose(pose)
        # try screw two times before giving up
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)
        pose = sapien.Pose(p=pose.p , q=pose.q)
        result = self.planner.plan_screw(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
        )
        if result["status"] != "Success":
            result = self.planner.plan_screw(
                np.concatenate([pose.p, pose.q]),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                use_point_cloud=self.use_point_cloud,
            )
            if result["status"] != "Success":
                print(result["status"])
                self.render_wait()
                return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def open_gripper(self):
        self.gripper_state = OPEN
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        for i in range(6):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def close_gripper(self, t=6):
        self.gripper_state = CLOSED
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def add_box_collision(self, extents: np.ndarray, pose: sapien.Pose):
        self.use_point_cloud = True
        box = trimesh.creation.box(extents, transform=pose.to_transformation_matrix())
        pts, _ = trimesh.sample.sample_surface(box, 256)
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def add_collision_pts(self, pts: np.ndarray):
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def clear_collisions(self):
        self.all_collision_pts = None
        self.use_point_cloud = False

    def close(self):
        pass

    def setup_planner(self):
        print("[self.robot.get_links()]")
        print([link.get_name() for link in self.robot.get_links()])
        print("self.robot.get_active_joints()")
        print([joint.get_name() for joint in self.robot.get_active_joints()])
        
        # Retrieve all active joints
        active_joints = self.robot.get_active_joints()
        
        valid_joints = []
        valid_links = set()  # Use a set to avoid duplicates
        
        # Define essential links that must always be included
        essential_links = {
            'estop_link',
            'l_wheel_link',
            'laser_link',
            'r_wheel_link',
            'torso_fixed_link',
            'bellows_link2',
            'head_tilt_link',
            'wrist_roll_link'
        }  # Add other essential links if necessary
        
        for joint in active_joints:
            # Attempt to determine if the joint is continuous
            is_continuous = False
            
            # Method 1: Check for 'joint_type' attribute
            if hasattr(joint, 'joint_type'):
                if joint.joint_type.lower() == 'continuous':
                    is_continuous = True
            # Method 2: Check joint limits (assuming limits are accessible)
            elif hasattr(joint, 'lower_limit') and hasattr(joint, 'upper_limit'):
                lower = joint.lower_limit
                upper = joint.upper_limit
                if lower is None and upper is None:
                    is_continuous = True
                elif lower is not None and upper is not None:
                    # Define a threshold to consider joint as continuous
                    # For example, a full rotation within a small epsilon
                    rotation_threshold = 2 * np.pi - 0.01  # Adjust epsilon as needed
                    if (upper - lower) >= rotation_threshold:
                        is_continuous = True
            # Method 3: Access underlying PhysX articulation to determine if joint is continuous
            elif hasattr(joint, '_physx_articulations') and joint._physx_articulations:
                physx_articulation = joint._physx_articulations[0]
                joints = physx_articulation.get_joints()  # Corrected method name
                joint_index = joint.index.item()
                if 0 <= joint_index < len(joints):
                    physx_joint = joints[joint_index]  # Access the joint using the index
                    if hasattr(physx_joint, 'is_continuous') and callable(getattr(physx_joint, 'is_continuous')):
                        is_continuous = physx_joint.is_continuous()
                    else:
                        print(f"'PhysxJoint' object does not have 'is_continuous' method for joint: {joint.name}")
                else:
                    print(f"Joint index {joint_index} is out of range for PhysxArticulation joints.")
            
            if not is_continuous:
                valid_joints.append(joint.name)
                if joint.parent_link and joint.parent_link.name:
                    valid_links.add(joint.parent_link.name)
            else:
                print(f"Excluding continuous joint: {joint.name}")
                # Ensure the link is included if it's essential
                if joint.parent_link and joint.parent_link.name in essential_links:
                    valid_links.add(joint.parent_link.name)
        
        # Manually add essential links that might have been excluded
        for essential_link in essential_links:
            valid_links.add(essential_link)
        
        # Convert set back to list
        valid_links = list(valid_links)
        
        # Ensure that the move_group is a valid link
        move_group = "gripper_link"
        if move_group not in valid_links:
            raise ValueError(f"Move group '{move_group}' is not in the list of valid links.")
        
        # Set joint velocity and acceleration limits based on the number of valid joints

        num_joints = len(valid_joints)
        joint_vel_limits = np.ones(num_joints) * self.joint_vel_limits
        joint_acc_limits = np.ones(num_joints) * self.joint_acc_limits
        
        # Initialize planner with retry mechanism
        max_retries = 5
        attempt = 0
        
        while attempt < max_retries:
            try:
                valid_joints = list(set(valid_joints))
                valid_links = list(set(valid_links))
                print("Valid joint names:", valid_joints)
                print("Valid link names:", valid_links)
                planner = mplib.Planner(
                    urdf=self.env_agent.urdf_path,
                    srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
                    user_link_names=valid_links,
                    user_joint_names=valid_joints,
                    move_group=move_group,
                    # joint_vel_limits=joint_vel_limits,
                    # joint_acc_limits=joint_acc_limits,
                )
                # If planner is successfully created, exit the loop
                break
            except ValueError as ve:
                print("ValueError encountered during planner initialization:")
                print(ve)
                print("--------")
                
                # Attempt to extract the missing link name from the error message
                error_message = str(ve)
                # Assuming the error message format is "The names does not contain link estop_link"
                # Split the message and extract the last word
                tokens = error_message.split()
                if 'link' in tokens:
                    link_index = tokens.index('link') + 1
                    if link_index < len(tokens):
                        missing_link = tokens[link_index]
                    else:
                        print("Unable to parse missing link from the error message.")
                        raise ve  # Re-raise the exception if parsing fails
                else:
                    print("Error message does not contain 'link'. Unable to determine missing link.")
                    raise ve  # Re-raise the exception if 'link' is not in the message
                
                print(f"Missing link identified: {missing_link}")
                print("Adding the missing link to essential_links and valid_links, then retrying...")
                
                # Add the missing link to essential_links and valid_links
                essential_links.add(missing_link)
                valid_links.append(missing_link)
                
                # Optionally, you might need to adjust joint limits if new joints are added
                # For simplicity, we'll assume the current limits are sufficient
                
                # Increment the attempt counter
                attempt += 1
                
                # Optionally, you can add a small delay before retrying
                # import time
                # time.sleep(1)
            except Exception as e:
                # Handle other unexpected exceptions
                print("An unexpected error occurred during planner initialization:")
                print(e)
                raise e
        else:
            # If all retries failed, raise an exception
            raise Exception(f"Failed to initialize planner after {max_retries} attempts due to missing essential links.")
        
        # Set the base pose after successful planner creation
        planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
        return planner

