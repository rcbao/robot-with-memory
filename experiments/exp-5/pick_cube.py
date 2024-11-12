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

    # Reference the target cube (second cube in the list)
    target_cube = env.cubes[1]

    # Get the OBB (Oriented Bounding Box) of the target cube
    obb = get_actor_obb(target_cube)

    # Define the approach vector (from above)
    approaching = np.array([0, 0, -1])

    # Define the target closing direction based on the robot's TCP orientation
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].numpy()

    # Compute grasp information based on the OBB
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]

    # Build the initial grasp pose
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # Search for a valid grasp pose by rotating around the z-axis
    # angles = np.arange(0, 2 * np.pi, np.pi / 2)  # Full rotation
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    # Alternate rotation directions
    angles[1::2] *= -1
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose * delta_pose
        res = planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print("Failed to find a valid grasp pose")
        return False  # Early exit if no valid grasp pose is found

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)

    # -------------------------------------------------------------------------- #
    # Stack
    # -------------------------------------------------------------------------- #
    # Reference the base cube (first cube in the list)
    base_cube = env.cubes[0]
    goal_pose = base_cube.pose * sapien.Pose([0, 0, env.cube_half_size[2] * 2])

    # Compute the offset to align the target cube above the base cube
    offset = (goal_pose.p - target_cube.pose.p).numpy()[0]  # Assuming batched data
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)

    # Move to the alignment pose
    planner.move_to_pose_with_screw(align_pose)

    # -------------------------------------------------------------------------- #
    # Release
    # -------------------------------------------------------------------------- #
    res = planner.open_gripper()
    planner.close()
    return res


def main():
    env = init_env()
    try:
        res = solve(env, vis=False)
        print("Result of the stacking operation:", res[-1] if res else "No result")
    finally:
        env.close()


if __name__ == "__main__":
    main()
