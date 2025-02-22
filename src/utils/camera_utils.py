import datetime
from PIL import Image
import io
import base64
import numpy as np
import gymnasium as gym

def resize_image_by_factor(image, factor: float = 0.5):
    new_size = (int(image.width * factor), int(image.height * factor))
    image = image.resize(new_size)
    return image


def save_camera_image_by_type(env, camera_type="base_camera"):
    obs = env.unwrapped.get_obs()
    if 'sensor_data' in obs:
        rgb_image = obs['sensor_data'][camera_type]['rgb'].squeeze(0).cpu().numpy()

        # Convert the image to PIL format
        image = Image.fromarray(rgb_image)
        image = resize_image_by_factor(image, 0.6)

        # Save the image to a buffer
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        # Encode the image in base64
        encoded_image = base64.b64encode(buffer.read()).decode('utf-8')

        # Optionally, save the image as a file with a timestamp
        image.save(f"sensor_image-{datetime.datetime.now():%Y%m%d-%H%M%S}-{camera_type}.png")

        return encoded_image
    else:
        raise KeyError(f"Camera type '{camera_type}' not found in the environment observations.")


def get_camera_image(env) -> np.ndarray:
    obs = env.unwrapped.get_obs()
    if 'sensor_data' in obs:
        save_camera_image_by_type(env, "base_camera")

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
