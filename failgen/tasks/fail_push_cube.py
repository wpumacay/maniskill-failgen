import numpy as np
import sapien

from mani_skill.envs.tasks import PushCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env


@register_env("FailPushCube-v1", max_episode_steps=50)
class FailPushCubeEnv(PushCubeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("front_camera", pose, 128, 128, np.pi / 2, 0.01, 100),
            CameraConfig("side_camera", pose, 128, 128, np.pi / 2, 0.01, 100),
        ]



