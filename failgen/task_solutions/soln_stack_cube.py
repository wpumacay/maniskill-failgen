import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)

from failgen.fail_planner_wrapper import FailPlannerWrapper
from failgen.tasks.fail_stack_cube import FailStackCubeEnv
from failgen.fail_planner_wrapper import FailPlannerWrapper


def solve(
    env: FailStackCubeEnv,
    planner_wrapper: FailPlannerWrapper,
    seed=None,
    debug=False,
    vis=False,
):
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

    planner_wrapper.wrap_planner(planner)

    FINGER_LENGTH = 0.025
    env = env.unwrapped
    obb = get_actor_obb(env.cubeA)

    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[
        0, :3, 1
    ].numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # Search a valid pose
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose * delta_pose
        res = planner_wrapper.move_to_pose_with_screw(
            grasp_pose2, dry_run=True, stage=0
        )
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print("Fail to find a valid grasp pose")

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner_wrapper.move_to_pose_with_screw(reach_pose, stage=1)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner_wrapper.move_to_pose_with_screw(grasp_pose, stage=2)
    planner_wrapper.close_gripper(stage=3)

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    planner_wrapper.move_to_pose_with_screw(lift_pose, stage=4)

    # -------------------------------------------------------------------------- #
    # Stack
    # -------------------------------------------------------------------------- #
    goal_pose = env.cubeB.pose * sapien.Pose([0, 0, env.cube_half_size[2] * 2])
    offset = (goal_pose.p - env.cubeA.pose.p).numpy()[
        0
    ]  # remember that all data in ManiSkill is batched and a torch tensor
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner_wrapper.move_to_pose_with_screw(align_pose, stage=5)

    res = planner_wrapper.open_gripper(stage=6)
    planner_wrapper.close()
    return res
