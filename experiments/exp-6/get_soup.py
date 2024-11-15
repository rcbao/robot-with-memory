import argparse
import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import StackCubeEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from mani_skill.utils.wrappers.record import RecordEpisode
from init_env import init_env


def move_to_pose(planner, target_pose, dry_run=False):
    result = planner.move_to_pose_with_screw(target_pose, dry_run=dry_run)
    if result == -1:
        print("move_to_pose_with_screw failed, falling back to move_to_pose_with_RRTConnect")
        result = planner.move_to_pose_with_RRTConnect(target_pose, dry_run=dry_run)
    return result != -1


def fetch_and_place(env, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    
    # Define initial apple object position and target position
    apple_position = np.array([0.08, 0.3, 0.22])
    target_position = np.array([0.2, 0, 0])
    lift_height = 0.4


    # Grasp pose setup for apple
    grasp_pose = sapien.Pose(p=apple_position, q=euler2quat(0, np.pi, 0))

    # Move to grasp pose
    move_to_pose(planner, grasp_pose)
    planner.close_gripper()

    # Lift object to avoid obstacles
    lift_pose = sapien.Pose([0, 0, lift_height]) * grasp_pose
    move_to_pose(planner, lift_pose)

    # Move to target position above the table
    place_pose = sapien.Pose(p=target_position + np.array([0, 0, lift_height]), q=grasp_pose.q)
    move_to_pose(planner, place_pose)

    # Lower to place object
    lower_pose = sapien.Pose(p=target_position, q=grasp_pose.q)
    move_to_pose(planner, lower_pose)
    planner.open_gripper()

    # Lift slightly after releasing
    post_release_lift = sapien.Pose([0, 0, 0.1]) * lower_pose
    move_to_pose(planner, post_release_lift)

    planner.close()
    return True


def main():
    env = init_env()
    try:
        res = fetch_and_place(env, vis=False)
        print("Result of the fetch and place operation:", "Success" if res else "Failed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
