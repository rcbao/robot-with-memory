## stack_cube.py ##
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

def solve(env: StackCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode

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

    # Iterate over all cubes starting from the second cube (index 1) to stack them sequentially
    for i in range(1, len(env.cubes)):
        target_cube = env.cubes[i]
        base_cube = env.cubes[i - 1]  # Cube to stack on

        # Get grasp information for the current cube
        obb = get_actor_obb(target_cube)
        target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].numpy()
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=FINGER_LENGTH,
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
        grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

        # Search for a valid grasp pose
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
            print("Fail to find a valid grasp pose for cube", i)
            return False

        if i < 5:
            # Initial small lift to clear the immediate area after grasping
            initial_lift_pose = sapien.Pose([0, 0, 0.2]) * grasp_pose
            move_to_pose(planner, initial_lift_pose)

        # Intermediate Avoidance Position
        # Calculate avoidance height just above the current stack
        stack_height = (i + 1) * env.cube_half_size[2] * 2  # Height of current stack with buffer
        avoidance_height = stack_height + 0.05  # Add a small buffer above the stack
        avoidance_pose = sapien.Pose([0, 0, avoidance_height]) * grasp_pose
        move_to_pose(planner, avoidance_pose)

        # Reach to cube
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        move_to_pose(planner, reach_pose)

        # Grasp the cube
        move_to_pose(planner, grasp_pose)
        planner.close_gripper()

        # Lift to the intermediate avoidance position after grasping
        lift_pose = sapien.Pose([0, 0, avoidance_height]) * grasp_pose
        move_to_pose(planner, lift_pose)

        # Move to the stack position above the base cube
        goal_pose = base_cube.pose * sapien.Pose([0, 0, env.cube_half_size[2] * 2])
        offset = (goal_pose.p - target_cube.pose.p).numpy()[0]
        align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
        move_to_pose(planner, align_pose)

        # Release the current cube on top of the previous one
        planner.open_gripper()

        # Small upward movement after releasing the cube
        post_release_lift = sapien.Pose([0, 0, 0.2]) * align_pose  # Move slightly up
        move_to_pose(planner, post_release_lift)

    planner.close()  # Close the planner after stacking all cubes
    return True

def main():
    env = init_env()
    # env.set_timestep(0.02)
    try:
        res = solve(env, vis=False)
        print("Result of the stacking operation:", "Success" if res else "Failed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
