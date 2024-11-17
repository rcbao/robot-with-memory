# memory-bot
Robot with memory. Project for Learning for Interact Robots

## Set up
```
module load anaconda

conda activate maniskill_env  ## Use your maniskill conda environment name

pip install openai
pip install opencv-python
```

## experiments

* exp-1 -- original example `https://github.com/haosulab/ManiSkill/blob/main/mani_skill/examples/motionplanning/panda/solutions/pick_cube.py`
* exp-2 -- exp-1 with cup.glb added to the environment. The robot can pick up the cup
* exp-3 -- exp-2 with pandas robot swapped to pandas-wrist cam. Code saves wrist cam footage to image
* exp-4 -- exp-3 with multiple objects on the table. 
    TODOs: 
        Fix the roaming objects issue


* exp-6 -- **Solved MOTION.**: Set up an env with two shelves and four YCB objects. Robot picks up any object from shelf
* exp-7 -- exp-6 with the robot able to (pick up + drop off) any objects on-command
* exp-8 -- exp-7 with the robot able to take in a recipe, identify objects to re-arrange, and execute


## Work around
For `~/.conda/envs/maniskill_env/lib/python3.1/site-packages/mani_skill/agents/robots/panda/panda_wristcam.py`
Use the below instead: 

```python
import numpy as np
import sapien

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

from .panda import Panda

USE_FRONT_CAMERA = True


@register_agent()
class PandaWristCam(Panda):
    """Panda arm robot with the real sense camera attached to gripper"""

    uid = "panda_wristcam"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v3.urdf"

    @property
    def _sensor_configs(self):
        if USE_FRONT_CAMERA:
            return [
                CameraConfig(
                    uid="front_camera",
                    pose=sapien.Pose(
                        p=[0.1, 0, 0.2],  # 10 cm forward, 20 cm above the base
                        q=[1, 0, 0, 0]  # Pointing straight ahead
                    ),
                    width=640,
                    height=480,
                    fov=np.pi / 4, 
                    near=0.05,
                    far=200,
                    mount=self.robot.links_map["panda_link0"],
                )
            ]
        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],
            )
        ]
```

There's a bug with Maniskill preventing overriding agents. this fixes it using a feature flag.