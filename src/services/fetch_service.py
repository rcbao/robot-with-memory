from typing import Optional, Dict, List
from services.location_oracle_service import Oracle
from services.memory_service import MemoryService
from services.vision_service import VisionService
from utils.logger import setup_logger
from utils.camera_utils import save_camera_image_by_type
from utils.movement import fetch_and_place_target_object
from utils.robot_rotator import RobotRotator
from constants import VIEWS, TARGET_COORDINATES

from difflib import get_close_matches

logger = setup_logger()


class FetchService:
    """
    Service to handle fetching objects.
    """
    def __init__(self, oracle_service: Oracle, memory_service: MemoryService, lang_service, vision_service, env, rotator: RobotRotator):
        self.oracle_service = oracle_service
        self.memory_service = memory_service
        self.lang_service = lang_service
        self.vision_service = vision_service
        self.env = env
        self.rotator = rotator


    def fetch_item(self, item: dict) -> bool:
        """
        Attempt to fetch the object by its name in environment.

        Args:
            name (str): Name of the object to fetch.

        Returns:
            bool: True if the object was fetched successfully, False otherwise.
        """
        name, detail = item["name"], item["detail"]
        target_object = getattr(self.env, name, None)

        if not target_object:
            cleaned_name = name.lower().replace("'", "").replace(" ", "_")
            target_object = getattr(self.env, cleaned_name, None)

        if not target_object:
            logger.error(f"Object '{name}' not found in the environment.")
            return False

        result = fetch_and_place_target_object(self.env, target_object, TARGET_COORDINATES, vis=False)
        if result:
            message = f"Successfully fetched the {name} and placed it on the table."
            logger.info(message)
            print(message)
            existing_object = self.memory_service.get_object(name)
            if not existing_object:
                self.memory_service.add_object(
                    name=name,
                    detail=detail,
                    location={"text": f"{name} is on the table.", "coords": TARGET_COORDINATES}
                )
            else:
                self.memory_service.update_location(
                    name=name,
                    new_location={"text": f"{name} is on the table.", "coords": TARGET_COORDINATES}
                )
            return True

        logger.error(f"Failed to fetch '{name}'.")
        return False

    def fetch_from_camera_view(self, name: str, view: str) -> bool:
        """
        Fetch an object from camera view
        """
        matched_item = self.vision_service.detect_object_from_camera_image(name, view)

        if not matched_item:
            logger.warning(f"No match for item '{name}' in view.")
            return False

        return self.fetch_item(matched_item)

    def fetch_from_memory(self, object_name: str) -> bool:
        """
        Fetches an object from memory and places it on the table if coordinates are available.
        """
        item = self.memory_service.get_object(object_name)
        
        if not item:
            logger.info(f"Object '{object_name}' not found in memory.")
            return False

        coords = item.get("location", {}).get("coords")
        if not coords:
            logger.warning(f"Memory entry for '{object_name}' does not contain coordinates.")
            return False
        
        return self.fetch_item(item)


    def handle_fetch(self, object_name: str, message_history: List[Dict[str, str]]):
        """
        Handle the fetch command by either fetching from memory or exploring the environment.
        """
        if self.fetch_from_memory(object_name):
            return

        item_not_in_memory_message = f"I could not locate '{object_name}' in memory. "
        logger.info(item_not_in_memory_message)
        
        for view in VIEWS:
            logger.info(f"Rotating to '{view}' view.")
            self.rotator.rotate_robot_to_view(view)
            if self.fetch_from_camera_view(object_name, view):
                item_found_message = f"After scanning, I have successfully found {object_name} in my {view} view. "

                full_message = {"role": "assistant", "content": f"{item_not_in_memory_message} {item_found_message}"}
                message_history.append(full_message)
                return

        item_not_found_message = "I could not find the object after scanning the environment."
        logger.info(item_not_found_message)
        
        full_message = {"role": "assistant", "content": f"{item_not_in_memory_message} {item_not_found_message}"}
        message_history.append(full_message)
