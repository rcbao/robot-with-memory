from collections import namedtuple

PromptPath = namedtuple("PromptPath", ["system", "user"])

prompt_paths = {
    "system": "./llm_prompts/general_system.txt",
    "user": {
        "parse_input_json": "./llm_prompts/parse_input_json_user.txt",
        "recall": "./llm_prompts/recall_object_user.txt"
    },
    "task_specific": {
        "parse_image": {
            "system": "./llm_prompts/parse_image_system.txt",
            "user": "./llm_prompts/parse_image_user.txt"
        }
    }
}

OPENAI_MODEL = "gpt-4o"
OPENAI_MAX_TOKENS = 400

OUTPUT_DIR = "videos"
ASSET_DIR = "assets"

USING_HQ_CAMERA = True

CAMERA_CONFIGS_HIGH_QUALITY = {
    "sensor_configs": {
        "width": 1920, "height": 1088, "shader_pack": "rt-fast"
    },
    "human_render_camera_configs": {"width": 1088, "height": 1088, "shader_pack": "rt-fast"},
    "viewer_camera_configs": {"fov": 1},
    "enable_shadow": True,
}

CAMERA_CONFIG_DEFAULT = {
    "sensor_configs": {"width": 1920, "height": 1088, "shader_pack": "default"},
    "human_render_camera_configs": {"width": 1088, "height": 1088, "shader_pack": "default"},
    "viewer_camera_configs": {"fov": 1},
    "enable_shadow": True,
}

VIEWS = ["left", "front", "right"]
TARGET_COORDINATES = [0.05, 0.05, 0]
