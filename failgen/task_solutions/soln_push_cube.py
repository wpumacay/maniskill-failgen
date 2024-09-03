import numpy as np
import sapien

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)

from failgen.fail_planner_wrapper import FailPlannerWrapper
from failgen.tasks.fail_push_cube import FailPushCubeEnv
from failgen.fail_planner_wrapper import FailPlannerWrapper


def solve(
    env: FailPushCubeEnv,
    planner_wrapper: FailPlannerWrapper,
    seed=None,
    debug=False,
    vis=False,
):
    env.reset(seed=seed)
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
    planner_wrapper.close_gripper(stage=0)
    reach_pose = sapien.Pose(
        p=env.obj.pose.sp.p + np.array([-0.05, 0, 0]), q=env.agent.tcp.pose.sp.q
    )
    planner_wrapper.move_to_pose_with_screw(reach_pose, stage=1)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(
        p=env.goal_region.pose.sp.p + np.array([-0.12, 0, 0]),
        q=env.agent.tcp.pose.sp.q,
    )
    res = planner_wrapper.move_to_pose_with_screw(goal_pose, stage=2)

    planner_wrapper.close()
    return res
