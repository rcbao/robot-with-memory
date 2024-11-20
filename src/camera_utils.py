import datetime
from PIL import Image

import numpy as np
import gymnasium as gym

def save_camera_image_by_type(env, camera_type="base_camera"):
    obs = env.unwrapped.get_obs()
    if 'sensor_data' in obs:
        rgb_image = obs['sensor_data'][camera_type]['rgb'].squeeze(0).cpu().numpy()
        Image.fromarray(rgb_image).save(f"sensor_image-{datetime.datetime.now():%Y%m%d-%H%M%S}-{camera_type}.png")


def get_camera_image(env) -> np.ndarray:
    obs = env.unwrapped.get_obs()
    if 'sensor_data' in obs:
        save_camera_image_by_type(env, "base_camera")
        # save_camera_image_by_type(env, "front_camera")
        save_camera_image_by_type(env, "front_camera_link2")

    else:
        raise KeyError("Camera observation not found in the environment observations.")



class StepImageCaptureWrapper(gym.Wrapper):
    def __init__(self, env, capture_frequency=5):
        super().__init__(env)
        self.capture_frequency = capture_frequency
        self.step_count = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # Capture image every `capture_frequency` steps
        if self.step_count % self.capture_frequency == 0:
            get_camera_image(self.env)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)
