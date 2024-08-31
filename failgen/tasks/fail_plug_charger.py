import numpy as np
import sapien

from mani_skill.envs.tasks import PlugChargerEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env


@register_env("FailPlugCharger-v1", max_episode_steps=200)
class FailPlugChargerEnv(PlugChargerEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose_front = sapien_utils.look_at(
            eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1]
        )
        pose_side = sapien_utils.look_at(
            eye=[0, 0.6, 0.6], target=[-0.1, 0, 0.1]
        )

        return [
            CameraConfig(
                "front_camera", pose_front, 128, 128, np.pi / 2, 0.01, 100
            ),
            CameraConfig(
                "side_camera", pose_side, 128, 128, np.pi / 2, 0.01, 100
            ),
        ]
