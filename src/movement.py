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

def move_to_pose(planner, target_pose, dry_run=False):
    result = planner.move_to_pose_with_screw(target_pose, dry_run=dry_run)
    if result == -1:
        print("move_to_pose_with_screw failed, falling back to move_to_pose_with_RRTConnect")
        result = planner.move_to_pose_with_RRTConnect(target_pose, dry_run=dry_run)
    return result != -1


def rotate_robot_to(env, view, current_joint_positions):
    """
    Rotates the robot's link1 joint to the specified view direction.
    """
    view_angles = {"center": 0, "left": np.pi / 4, "right": -np.pi / 4}
    target_angle = view_angles.get(view, 0)  # Default to 'center' if view is invalid

    action_vector = np.zeros(8, dtype=np.float32)
    action_vector[0] = target_angle

    steps = 100

    for _ in range(steps):        
        obs, _, done, _, _ = env.step(action_vector)
        if done:
            break
    print(f"Rotated to {view}")
    return obs


def get_current_joint_positions(obs):
    """
    Extracts current joint positions from the observation.
    """
    # Assuming joint positions are stored under the key 'joint_positions' or similar
    print("obs::")
    print(obs)
    joint_positions = obs[0]['agent']['qpos'].cpu().numpy().flatten()


    if joint_positions is not None:
        return joint_positions
    else:
        raise KeyError("Joint positions not found in observation data.")


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
