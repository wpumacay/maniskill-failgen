from typing import Callable, Optional

import numpy as np
import sapien

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


class FailPlannerWrapper:
    _planner: PandaArmMotionPlanningSolver

    _fail_on_open: bool = False

    _fail_on_close: bool = False

    _fail_on_move: bool = False

    def __init__(
        self,
        planner: PandaArmMotionPlanningSolver,
        fail_on_open: bool,
        fail_on_close: bool,
        fail_on_move: bool,
    ):
        self._planner = planner
        self._fail_on_open = fail_on_open
        self._fail_on_close = fail_on_close
        self._fail_on_move = fail_on_move

    def open_gripper(self) -> None:
        if not self._fail_on_open:
            self._planner.open_gripper()

    def close_gripper(self) -> None:
        if not self._fail_on_close:
            self._planner.close_gripper()

    def move_to_pose_with_screw(
        self,
        target_pose: sapien.Pose,
        dry_run: bool = False,
        refine_steps: int = 0,
    ):
        if self._fail_on_move:
            target_pose.p = target_pose.p + 0.01 * np.random.rand(
                *target_pose.p.shape
            )
        return self._planner.move_to_pose_with_screw(
            target_pose, dry_run=dry_run, refine_steps=refine_steps
        )

    def close(self) -> None:
        return self._planner.close()
