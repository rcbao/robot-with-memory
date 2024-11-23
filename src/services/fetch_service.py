from typing import Optional, Dict, List
from services.location_oracle_service import Oracle
from services.memory_service import Memory
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
    def __init__(self, oracle_service: Oracle, memory_service: Memory, lang_service, env, rotator: RobotRotator):
        self.oracle_service = oracle_service
        self.memory_service = memory_service
        self.lang_service = lang_service
        self.env = env
        self.rotator = rotator

    def find_best_match_name(self, objects_in_view: List[Dict], object_name: str) -> "":
        object_names = [obj["name"] for obj in objects_in_view]

        object_name = object_name.lower()

        matches = get_close_matches(object_name, object_names, n=1, cutoff=0.6)

        if matches:
            first_match = matches[0]
            return first_match
        return ""

    def validate_item_name(self, item_name: str):
        if item_name:
            valid_names = self.oracle_service.get_valid_item_names()
            valid_names = [name.lower() for name in valid_names]
            return item_name in valid_names
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

    def detect_object_from_camera_image(self, object_name: str) -> Optional[Dict]:
        """
        Detect objects from the camera image and find a matching object.

        Args:
            object_name (str): Name of the object to find.

        Returns:
            dict or None: Matched object dictionary if found, else None.
        """
        encoded_image = save_camera_image_by_type(self.env, camera_type="front_camera")
        objects = self.lang_service.list_objects_in_scene_image(encoded_image)
        logger.info(f"Detected objects in image: {objects}.")

        return self.fuzzy_match_item(objects, object_name)

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
            logger.error(f"Object '{name}' not found in the environment.")
            return False

        result = fetch_and_place_target_object(self.env, target_object, TARGET_COORDINATES, vis=False)
        if result:
            logger.info(f"Successfully fetched '{name}' and placed it on the table.")
            self.memory_service.add_object(
                name=name,
                detail=detail,
                location={"text": "on the table", "coords": TARGET_COORDINATES}
            )
            return True

        logger.error(f"Failed to fetch '{name}'.")
        return False

    def fetch_from_camera_view(self, name: str) -> bool:
        """
        Fetch an object from camera view
        """
        matched_item = self.detect_object_from_camera_image(name)

        if not matched_item:
            logger.warning(f"No match for item '{name}' in view.")
            return False

        item_coordinates = self.oracle_service.get_object_coordinates(name)

        if not item_coordinates:
            logger.warning(f"Coordinates for '{name}' not found in oracle.")
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
            logger.warning(f"Memory entry for '{name}'does not contain coordinates.")
            return False
        
        return self.fetch_item(item)


    def handle_fetch(self, object_name: str, message_history: List[Dict[str, str]]):
        """
        Handle the fetch command by either fetching from memory or exploring the environment.

        Args:
            object_name (str): Name of the object to fetch.
            message_history (list): List to keep track of message history.
        """
        if self.fetch_from_memory(object_name):
            return

        item_not_in_memory_message = f"I could not find '{object_name}' in memory. I have started to scan the environment. "

        logger.info(item_not_in_memory_message)
        

        for view in VIEWS:
            logger.info(f"Rotating to '{view}' view.")
            self.rotator.rotate_robot_to_view(view)
            if self.fetch_from_camera_view(object_name):
                item_found_message = f"After scanning, I have successfully found {object_name} in my {view} view. "

                full_message = {"role": "assistant", "content": f"{item_not_in_memory_message} {item_found_message}"}
                message_history.append(full_message)
                return

        item_not_found_message = "I can't find the object."
        logger.info(item_not_found_message)
        
        full_message = {"role": "assistant", "content": f"{item_not_in_memory_message} {item_not_found_message}"}
        message_history.append(full_message)
