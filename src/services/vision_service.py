import json
from typing import Optional, Dict, List
from utils.logger import setup_logger
from utils.camera_utils import save_camera_image_by_type
from openai import OpenAI
from dotenv import load_dotenv
from services.location_oracle_service import Oracle
from prompt_builder import PromptBuilder
from constants import OPENAI_MODEL
logger = setup_logger()

load_dotenv()


class VisionService:
    """
    Service to handle vision-related tasks for object detection.
    """
    def __init__(self, oracle_service: Oracle, env):
        self.prompt_builder = PromptBuilder()
        self.client = OpenAI()
        
        self.oracle_service = oracle_service
        self.env = env

    def get_closest_match(self, object_name: str, object_names: List[str]) -> str:
        system_prompt = (
            "You are given an object name and a list of object names. "
            "Your task is to find the closest match in the list. "
            "If there is no match, return an empty string. "
            "If there is a match, return the object name ONLY."
        )
        user_prompt = (
            f"Object name: {object_name} \n"
            f"------ \n"
            f"List of Potential Object Names: {object_names} \n"
            f"----- \n"
            f"Return the closest match."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,  # Corrected model name
                messages=messages
            )
            response_text = response.choices[0].message.content.strip()

            if response_text:
                response_text = response_text.lower()
                return response_text
            return ""
        except Exception as e:
            logger.error(f"Error in get_closest_match: {e}")
            return ""
    
    def list_objects_in_scene_image(self, base64_image: str, view: str) -> List[Dict]:
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
                        "text": user_prompt,
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
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,  # Corrected model name
                messages=messages
            )
            response_text = response.choices[0].message.content.strip()

            response_text = self.prompt_builder.clean_up_json(response_text)
            try:
                response = json.loads(response_text)
                return response
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from vision response.")
                return []
        except Exception as e:
            logger.error(f"Error in list_objects_in_scene_image: {e}")
            return []

    def detect_objects_from_camera_image(self, object_names: List[str], view: str) -> List[Dict]:
        """
        Detect multiple objects from the camera image and find matching objects.

        Args:
            object_names (List[str]): List of object names to find.
            view (str): Camera view direction.

        Returns:
            List[Dict]: List of matched object dictionaries if found, else empty list.
        """
        encoded_image = save_camera_image_by_type(self.env, camera_type="front_camera")
        objects = self.list_objects_in_scene_image(encoded_image, view)
        logger.info(f"Detected objects in image: {objects}.")

        matched_objects = self.fuzzy_match_items(objects, object_names)
        return matched_objects

    def find_best_match_names(self, objects_in_view: List[Dict], object_names: List[str]) -> Dict[str, str]:
        """
        Find the best matching names for a list of object names.

        Args:
            objects_in_view (List[Dict]): List of objects detected in view.
            object_names (List[str]): List of object names to match.

        Returns:
            Dict[str, str]: Mapping of original object names to their closest matches.
        """
        object_names_lower = [name.lower() for name in object_names]
        object_detected_names = [obj["name"].lower() for obj in objects_in_view]
        
        closest_matches = {}
        for name in object_names_lower:
            closest_match = self.get_closest_match(name, object_detected_names)
            if closest_match:
                closest_matches[name] = closest_match
            else:
                closest_matches[name] = ""
        return closest_matches

    def validate_item_name(self, item_name: str) -> bool:
        if item_name:
            valid_names = self.oracle_service.get_valid_item_names()
            valid_names = [name.lower() for name in valid_names]
            return item_name in valid_names
        logger.warning("Provided item name is empty.")
        return False

    def fuzzy_match_items(self, items_in_view: List[Dict], item_names: List[str]) -> List[Dict]:
        """
        Perform fuzzy matching to find the best matches for multiple item names.

        Args:
            items_in_view (List[Dict]): List of items detected in view.
            item_names (List[str]): List of item names to match.

        Returns:
            List[Dict]: List of matched item dictionaries if found, else empty list.
        """
        closest_matches = self.find_best_match_names(items_in_view, item_names)
        matched_items = []

        for original_name, matched_name in closest_matches.items():
            if self.validate_item_name(matched_name):
                matched = next((item for item in items_in_view if item["name"].lower() == matched_name), None)
                if matched:
                    logger.info(f"Fuzzy matched '{original_name}' to '{matched_name}'.")
                    matched_items.append(matched)
            else:
                logger.warning(f"No valid match found for '{original_name}'.")

        return matched_items
