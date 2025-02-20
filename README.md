# Robot with Memory

*Robot with Memory* is a Vision-Language Model (VLM)-based Robotics project. 

The project sets up a customized Franka Panda robot in a structured environment. With the help of the VLM and a lightweight memory store, the robot can observe its surroundings, remember object locations, and retrieve them with its arm on command. 

This project wouldn't have been possible just two years ago (e.g., 2023), and it shows how much the technologies have improved and are enabling applications like this. In an aging society, we think robots with memory -- like a more mature version of this project -- could help many people with memory and mobility issues and make their lives easier. 🙂

<p align="center">
  <img src="https://github.com/user-attachments/assets/61378027-a152-46e5-978a-5a2abf57895e" width="40%" style="margin-right: 5%;" />
  <img src="https://github.com/user-attachments/assets/4f963ab1-80be-4b78-abc5-41458fe6e9c6" width="40%" />
</p>

## Main Components
- **Franka Emika Panda robot**:  A popular arm-only robot. We customized it with a front-facing camera, so it can see its surroundings.
- **Vision-Language Model (VLM)**: Processes camera images and "talks" to user in natural language. We used OpenAI GPT-4o.
- **Motion Planner**:  Helped implement precise robotic arm control. Based on Screw theory & RRT
- **Memory Store**:  A lightweight JSON-based database for storing object location
- **Simulated Environment**:  Two small shelves stocked with everyday objects (e.g., apple, Rubik’s Cube, etc.)

## How It Works
- The robot remembers and retrieves objects through a combination of VLM parsing, motion planning, and stored memory.
- It interprets natural language commands and tries to match inputs like *"get the garbage can"* to objects it has seen.
- If an object is missing from memory, it would *actively scans the environment* to locate it.
- The robot/VLM alsos handle general questions like *“What have you fetched so far?”*, which makes the user experience more interactive.


## Gallery

https://github.com/user-attachments/assets/de868db7-b9d8-44f0-adc9-afe7cd8e22f9

https://github.com/user-attachments/assets/22151713-fd04-4841-a04c-eb98f8d871ea

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

4. Run the simulation
	```bash
	python src/main.py
	```

---

### Workaround for Panda Wrist Camera Issue

A known issue in ManiSkill prevents the Panda wrist camera from correctly overriding agent configurations. To fix this issue, you must directly update your panda_wristcam.py file. To get this file's location, run: 
```python
import inspect
from mani_skill.agents.robots import PandaWristCam

path = inspect.getfile(PandaWristCam)
print("Absolute path of PandaWristCam:", path)
```

Then, replace its contents with the following:

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

**Explanation:**
- **What does the fix do**: It adjusts the sensor configuration directly at a low level, and it helps avoid overriding issues.
- **`USE_FRONT_CAMERA` Flag**: It enables using the front camera mounted on `panda_link2`. By using a feature flag, it avoids changing the original code and makes it easier to revert back to the default behavior if needed.

## Challenges We Encountered
While building this project, we worked through a variety of tough engineering challenges:

1. [Maniskill](https://www.maniskill.ai/), the robotics framework we used, is a new and built by a small team. Thus, at the time of development, it has very limited documentation. As a result, debugging was at times very difficult, and none of the AI coding assistants was helpful due to frequent and severe hallucinations caused by lack of training data. We often had to debug by reading the framework’s source code to figure out how things worked.
2. GPT-4o was helpful for detecting objects, but occasional hallucinations in parsing object descriptions made the system less reliable. To address this issue, we iteratively improved the VLM prompt, and improved command accuracy to be good enough for our use case.

We learned a lot from doing all of this, and got much better at debugging, finding resources online, and writing prompts. We think we became much better engineers after the project. 

It did build character. 😄

## **Acknowledgments**

- Our professor, Yen-Ling Kuo, was extremely helpful and generous with her time in answering questions. Her advice (e.g., using the YCB dataset) sped up our progress by a lot and make it possible to be completed on time.
- The ManySkill maintainers, especially Stone Tao, was super responsive and helpful in answering questions. Thank you, and we are grateful 🙏

This project is a Robert Bao and Jade Gregoire production. It was made for UVA's CS 6501, [Learning for Interative Robots](https://ylkuo.notion.site/Learning-for-Interactive-Robots-Fall-2024-86af804431f24a2ba49925d9a4495d69).

## License
MIT License
