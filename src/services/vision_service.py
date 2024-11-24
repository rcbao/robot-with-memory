import json
from typing import Optional, Dict, List
from utils.logger import setup_logger
from utils.camera_utils import save_camera_image_by_type
from openai import OpenAI
from dotenv import load_dotenv
from services.location_oracle_service import Oracle
from prompt_builder import PromptBuilder

logger = setup_logger()

load_dotenv()


class VisionService:
    """
    Service to handle fetching objects.
    """
    def __init__(self, oracle_service: Oracle, env):
        self.prompt_builder = PromptBuilder()
        self.client = OpenAI()
        
        self.oracle_service = oracle_service
        self.env = env

    def get_closest_match(self, object_name, object_names):
        system_prompt = f"You are given an object name and a list of object names.  Your task is to find the cloest match in the list. If there is no match, return an empty string. If there is a match, return the object name ONLY."
        user_prompt = f"Object name: {object_name} \n ------ \n List of Potential Object Names: {object_names} \n ----- \n Return the closest match."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        response_text = response.choices[0].message.content.strip()

        if response_text:
            response_text = response_text.lower()
            return response_text
        return ""
        
    def list_objects_in_scene_image(self, base64_image: str, view: str) -> list:

        # Create the message payload
        system_prompt, user_prompt = self.prompt_builder.build_image_parser_prompts()
        user_prompt = user_prompt.format(view=view)

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt.format(view=view),
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url":  f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ]
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        response_text = response.choices[0].message.content.strip()

        response_text = self.prompt_builder.clean_up_json(response_text)
        response = json.loads(response_text)
        return response

    def detect_object_from_camera_image(self, object_name: str, view: str) -> Optional[Dict]:
        """
        Detect objects from the camera image and find a matching object.

        Args:
            object_name (str): Name of the object to find.

        Returns:
            dict or None: Matched object dictionary if found, else None.
        """
        encoded_image = save_camera_image_by_type(self.env, camera_type="front_camera")
        objects = self.list_objects_in_scene_image(encoded_image, view)
        logger.info(f"Detected objects in image: {objects}.")

        return self.fuzzy_match_item(objects, object_name)

    def find_best_match_name(self, objects_in_view: List[Dict], object_name: str) -> str:
        object_names = [obj["name"].lower() for obj in objects_in_view]
        object_name = object_name.lower()

        closest_match = self.get_closest_match(object_name, object_names)

        if closest_match:
            logger.info(f"closest_match: {closest_match}.")
            return closest_match
        return ""

    def validate_item_name(self, item_name: str):
        if item_name:
            valid_names = self.oracle_service.get_valid_item_names()
            valid_names = [name.lower() for name in valid_names]
            return item_name in valid_names
        print("Name {item_name} is invalid.")
        return False

    def fuzzy_match_item(self, items_in_view: List[Dict], item_name: str) -> Optional[Dict]:
        """
        Perform fuzzy matching to find the best match for the item name.

        Args:
            items_in_view (List[Dict]): List of items detected in view.
            item_name (str): Name of the item to match.

        Returns:
            dict or None: Matched item dictionary if a match is found, else None.
        """
        matched_name = self.find_best_match_name(items_in_view, item_name)
        name_valid = self.validate_item_name(matched_name)
        
        if name_valid:
            matched_items = [item for item in items_in_view if item["name"] == matched_name]
            if matched_items:
                top_match = matched_items[0]
                logger.info(f"Fuzzy matched '{item_name}' to '{matched_name}'.")
                return top_match

        logger.warning(f"No fuzzy match found for '{item_name}'.")
        return None

