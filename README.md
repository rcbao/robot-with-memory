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
   ```

---

## Workaround for Panda Wrist Camera Issue

To fix the bug preventing agents from being overridden, update the following file:
**`~/.conda/envs/maniskill_env/lib/python3.1/site-packages/mani_skill/agents/robots/panda/panda_wristcam.py`**

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
    """Panda arm robot with the RealSense camera attached to the gripper."""

    uid = "panda_wristcam"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v3.urdf"

    @property
    def _sensor_configs(self):
        if USE_FRONT_CAMERA:
            return [
                # Camera mounted on panda_link2
                CameraConfig(
                    uid="front_camera",
                    pose=sapien.Pose(
                        p=[0.0, 0.0, 0.15],  # Positioned 15 cm above panda_link2
                        q=[0.7071, 0.7071, 0, 0],  # Facing straight ahead
                    ),
                    width=640,
                    height=480,
                    fov=np.pi / 2,  # 90-degree field of view
                    near=0.05,
                    far=200,
                    mount=self.robot.links_map["panda_link2"],
                ),
            ]
        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=128,
                height=128,
                fov=np.pi / 3,
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],
            )
        ]
```

### Explanation
- **`USE_FRONT_CAMERA` Flag**: Enables using the front camera mounted on `panda_link2`.
- **Bug Fix**: Adjusts the sensor configuration to avoid overriding issues.
