# Memory-Bot: Robot with Memory
**Project for Learning Interactive Robots**

---

## Setup

1. Load Anaconda:
   ```bash
   module load anaconda
   ```
2. Activate your Maniskill environment:
   ```bash
   conda activate maniskill_env  # Replace "maniskill_env" with your environment name
   ```
3. Install required packages:
   ```bash
   pip install openai
   pip install opencv-python

   python -m mani_skill.utils.download_asset ycb 
   ```

---

## Workaround for Panda Wrist Camera Issue

To fix the bug preventing agents from being overridden, you must update your panda_wristcam.py file. To get this file's location, run: 
```python
import inspect
from mani_skill.agents.robots import PandaWristCam

path = inspect.getfile(PandaWristCam)
print("Absolute path of PandaWristCam:", path)
```

The path will look similar to this, with additional folders in front of lib: 
**`~/lib/python{version}/site-packages/mani_skill/agents/robots/panda/panda_wristcam.py`**

Replace its contents with this code:

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
                        p=[0, 0, -0.1],  # 10 cm forward, 20 cm above the base
                        q=[1, 0, 0, 0]  # Pointing straight ahead
                    ),
                    width=640,
                    height=480,
                    fov=np.pi / 4, 
                    near=0.05,
                    far=200,
                    mount=self.robot.links_map["panda_link1"],
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

### Explanation
- **`USE_FRONT_CAMERA` Flag**: Enables using the front camera mounted on `panda_link2`.
- **Bug Fix**: Adjusts the sensor configuration to avoid overriding issues.
