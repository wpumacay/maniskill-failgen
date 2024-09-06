from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import sapien
from omegaconf import DictConfig
from transforms3d.euler import euler2quat, quat2euler

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)

DELTA_TRANS = {
    "trans_x": np.array([1.0, 0.0, 0.0]),
    "trans_y": np.array([0.0, 1.0, 0.0]),
    "trans_z": np.array([0.0, 0.0, 1.0]),
}

DELTA_ROT = {
    "rot_x": np.array([1.0, 0.0, 0.0]),
    "rot_y": np.array([0.0, 1.0, 0.0]),
    "rot_z": np.array([0.0, 0.0, 1.0]),
}


@dataclass
class Failure:
    type: str
    enabled: bool
    stages: List[int]
    rnd_stage: int
    noise: float

    def check_active(self, stage: int, stage_fail: str) -> bool:
        if self.enabled and stage in self.stages and stage_fail == self.type:
            return True
        return False


class FailPlannerWrapper:
    _planner: Optional[PandaArmMotionPlanningSolver] = None

    _failures: Dict[str, Failure] = {}

    _cfg: DictConfig

    _active_fail: Optional[Failure] = None

    _stages: List[int] = []

    def __init__(self, cfg: DictConfig):
        self._planner = None
        self._cfg = cfg.copy()
        self._failures = {}
        self._stages = self._cfg.stages.copy()
        self._fail_stage = np.random.choice(self._stages)

        for fail_cfg in self._cfg.failures:
            self._failures[fail_cfg.type] = Failure(
                type=fail_cfg.type,
                enabled=fail_cfg.enabled,
                stages=fail_cfg.stages.copy(),
                rnd_stage=np.random.choice(fail_cfg.stages),
                noise=fail_cfg.noise if "noise" in fail_cfg else 0,
            )
            if fail_cfg.enabled:
                self._active_fail = self._failures[fail_cfg.type]

    @property
    def stages(self) -> List[int]:
        return self._stages

    def wrap_planner(self, planner: PandaArmMotionPlanningSolver) -> None:
        self._planner = planner

    def set_active_type(self, fail_type: str) -> None:
        for f_type, f_obj in self._failures.items():
            f_obj.enabled = f_type == fail_type
            if f_type == fail_type:
                self._active_fail = f_obj

    def set_active_stage(self, fail_stage: int) -> None:
        self._fail_stage = fail_stage

    def open_gripper(self, stage: int):
        assert self._planner is not None
        should_fail = (
            stage == self._fail_stage
            and self._active_fail
            and self._active_fail.check_active(stage, "grasp")
        )
        if not should_fail:
            return self._planner.open_gripper()
        return False

    def close_gripper(self, stage: int):
        assert self._planner is not None
        should_fail = (
            stage == self._fail_stage
            and self._active_fail
            and self._active_fail.check_active(stage, "grasp")
        )
        if not should_fail:
            return self._planner.close_gripper()
        return False

    def move_to_pose_with_screw(
        self,
        target_pose: sapien.Pose,
        dry_run: bool = False,
        refine_steps: int = 0,
        stage: int = 0,
    ):
        assert self._planner is not None
        position_delta = np.zeros(target_pose.p.shape)
        rotation_delta = np.zeros(target_pose.rpy.shape)
        if self._active_fail is not None:
            if stage == self._fail_stage and self._active_fail.check_active(
                stage, "trans_x"
            ):
                position_delta = (
                    self._active_fail.noise
                    * np.random.rand()
                    * DELTA_TRANS["trans_x"].reshape(target_pose.p.shape)
                )
            if stage == self._fail_stage and self._active_fail.check_active(
                stage, "trans_y"
            ):
                position_delta = (
                    self._active_fail.noise
                    * np.random.rand()
                    * DELTA_TRANS["trans_y"].reshape(target_pose.p.shape)
                )
            if stage == self._fail_stage and self._active_fail.check_active(stage, "trans_z"):
                position_delta = (
                    self._active_fail.noise
                    * np.random.rand()
                    * DELTA_TRANS["trans_z"].reshape(target_pose.p.shape)
                )
            target_pose.p = target_pose.p + position_delta

            if stage == self._fail_stage and self._active_fail.check_active(stage, "rot_x"):
                rotation_delta = (
                    self._active_fail.noise
                    * (np.random.rand() - 0.5)
                    * DELTA_ROT["rot_x"].reshape(target_pose.rpy.shape)
                )
            if stage == self._fail_stage and self._active_fail.check_active(stage, "rot_y"):
                rotation_delta = (
                    self._active_fail.noise
                    * (np.random.rand() - 0.5)
                    * DELTA_ROT["rot_y"].reshape(target_pose.rpy.shape)
                )
            if stage == self._fail_stage and self._active_fail.check_active(stage, "rot_z"):
                rotation_delta = (
                    self._active_fail.noise
                    * (np.random.rand() - 0.5)
                    * DELTA_ROT["rot_z"].reshape(target_pose.rpy.shape)
                )
            target_pose.rpy = target_pose.rpy + rotation_delta

        return self._planner.move_to_pose_with_screw(
            target_pose, dry_run=dry_run, refine_steps=refine_steps
        )

    def close(self) -> None:
        assert self._planner is not None
        return self._planner.close()
