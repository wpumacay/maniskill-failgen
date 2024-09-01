import os
from typing import List, Tuple

import gymnasium as gym
import numpy as np

from omegaconf import OmegaConf

from failgen.task_solutions.soln_peg_insertion_side import (
    solve as solvePegInsertionSide,
)
from failgen.task_solutions.soln_pick_cube import solve as solvePickCube
from failgen.task_solutions.soln_plug_charger import solve as solvePlugCharger
from failgen.task_solutions.soln_push_cube import solve as solvePushCube
from failgen.task_solutions.soln_stack_cube import solve as solveStackCube
from failgen.wrappers.record import RecordEpisode

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(CURRENT_DIR, "configs")

DEFAULT_TASK = "FailPickCube-v1"
MP_SOLUTIONS = {
    "FailPegInsertionSide-v1": solvePegInsertionSide,
    "FailPickCube-v1": solvePickCube,
    "FailPlugCharger-v1": solvePlugCharger,
    "FailPushCube-v1": solvePushCube,
    "FailStackCube-v1": solveStackCube,
}


class FailgenWrapper:
    def __init__(
        self, task_name: str, headless: bool, save_video: bool
    ) -> None:
        self._task_name = task_name
        self._headless = headless
        self._save_video = save_video
        self._solve_fn = MP_SOLUTIONS[task_name]
        self._seed = 0

        self._config = OmegaConf.load(
            os.path.join(CONFIGS_DIR, f"{task_name}.yaml")
        )

        self._env = gym.make(
            task_name,
            obs_mode=self._config.obs_mode,
            control_mode="pd_joint_pos",
            render_mode=self._config.render_mode,
            reward_mode="dense",
            sensor_configs=dict(
                width=self._config.image_size[0],
                height=self._config.image_size[1],
                shader_pack=self._config.shader,
            ),
            human_render_camera_configs=dict(shader_pack=self._config.shader),
            viewer_camera_configs=dict(shader_pack=self._config.shader),
            sim_backend=self._config.sim_backend,
        )

        self._env = RecordEpisode(
            self._env,
            output_dir=os.path.join(self._config.save_path, self._task_name),
            save_video=self._save_video,
            video_fps=30,
            save_on_reset=False,
            image_size=dict(
                width=self._config.image_size[0],
                height=self._config.image_size[1],
            ),
        )

    def get_failure(self) -> bool:
        success = self._solve_fn(
            self._env,
            seed=self._seed,
            debug=False,
            vis=not self._headless,
        )

        self._seed += 1
        self._env.flush_trajectory()
        self._env.flush_video()
        self._env.flush_video_multi()

        return success
