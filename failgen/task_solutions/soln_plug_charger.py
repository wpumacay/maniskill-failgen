import numpy as np
import sapien.core as sapien
import trimesh
from transforms3d.euler import euler2quat

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb,
)

from failgen.fail_planner_wrapper import FailPlannerWrapper
from failgen.tasks.fail_plug_charger import FailPlugChargerEnv
from failgen.fail_planner_wrapper import FailPlannerWrapper


def solve(
    env: FailPlugChargerEnv,
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
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )

    planner_wrapper.wrap_planner(planner)

    FINGER_LENGTH = 0.025
    env = env.unwrapped
    charger_base_pose = env.charger_base_pose
    charger_base_size = np.array(env.unwrapped._base_size) * 2

    obb = trimesh.primitives.Box(
        extents=charger_base_size,
        transform=charger_base_pose.sp.to_transformation_matrix(),
    )

    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.sp.to_transformation_matrix()[:3, 1]
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # add a angle to grasp
    grasp_angle = np.deg2rad(15)
    grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0, grasp_angle, 0))

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner_wrapper.move_to_pose_with_screw(reach_pose, stage=0)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner_wrapper.move_to_pose_with_screw(grasp_pose, stage=1)
    planner_wrapper.close_gripper(stage=2)

    # -------------------------------------------------------------------------- #
    # Align
    # -------------------------------------------------------------------------- #
    pre_insert_pose = (
        env.goal_pose.sp
        * sapien.Pose([-0.05, 0.0, 0.0])
        * env.charger.pose.sp.inv()
        * env.agent.tcp.pose.sp
    )
    insert_pose = (
        env.goal_pose.sp * env.charger.pose.sp.inv() * env.agent.tcp.pose.sp
    )
    planner_wrapper.move_to_pose_with_screw(
        pre_insert_pose, refine_steps=0, stage=3
    )
    planner_wrapper.move_to_pose_with_screw(
        pre_insert_pose, refine_steps=5, stage=4
    )
    # -------------------------------------------------------------------------- #
    # Insert
    # -------------------------------------------------------------------------- #
    res = planner_wrapper.move_to_pose_with_screw(insert_pose, stage=5)

    planner_wrapper.close()
    return res


if __name__ == "__main__":
    main()
