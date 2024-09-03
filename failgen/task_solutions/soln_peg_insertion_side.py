import numpy as np
import sapien

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from mani_skill.utils.structs.pose import to_sapien_pose

from failgen.fail_planner_wrapper import FailPlannerWrapper
from failgen.tasks.fail_peg_insertion_side import FailPegInsertionSideEnv
from failgen.fail_planner_wrapper import FailPlannerWrapper


def solve(
    env: FailPegInsertionSideEnv,
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
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )

    planner_wrapper.wrap_planner(planner)

    env = env.unwrapped
    FINGER_LENGTH = 0.025

    obb = get_actor_obb(env.peg)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[
        0, :3, 1
    ].numpy()

    peg_init_pose = env.peg.pose

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    offset = sapien.Pose(
        [-max(0.05, env.peg_half_sizes[0, 0] / 2 + 0.01), 0, 0]
    )
    grasp_pose = grasp_pose * (offset)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * (sapien.Pose([0, 0, -0.05]))
    planner_wrapper.move_to_pose_with_screw(reach_pose, stage=0)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner_wrapper.move_to_pose_with_screw(grasp_pose, stage=1)
    planner_wrapper.close_gripper(stage=2)

    # -------------------------------------------------------------------------- #
    # Align Peg
    # -------------------------------------------------------------------------- #

    # align the peg with the hole
    insert_pose = env.goal_pose * peg_init_pose.inv() * grasp_pose
    offset = sapien.Pose([-0.01 - env.peg_half_sizes[0, 0], 0, 0])
    pre_insert_pose = insert_pose * (offset)
    planner_wrapper.move_to_pose_with_screw(
        to_sapien_pose(pre_insert_pose), stage=3
    )
    # refine the insertion pose
    for i in range(3):
        delta_pose = env.goal_pose * (offset) * env.peg.pose.inv()
        pre_insert_pose = delta_pose * pre_insert_pose
        planner_wrapper.move_to_pose_with_screw(
            to_sapien_pose(pre_insert_pose), stage=4 + i
        )

    # -------------------------------------------------------------------------- #
    # Insert
    # -------------------------------------------------------------------------- #
    res = planner_wrapper.move_to_pose_with_screw(
        to_sapien_pose(insert_pose * (sapien.Pose([0.05, 0, 0]))), stage=7
    )
    planner_wrapper.close()
    return res


if __name__ == "__main__":
    main()
