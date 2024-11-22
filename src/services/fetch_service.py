from typing import Optional, Dict, List
from services.location_oracle_service import OracleService
from services.memory_service import Memory
from utils.logger import setup_logger
from utils.camera_utils import save_camera_image_by_type
from utils.movement import fetch_and_place_target_object
from utils.robot_rotator import RobotRotator

from difflib import get_close_matches

logger = setup_logger()

class FetchService:
    """
    Service to handle fetching objects.
    """
    def __init__(self, oracle_service: OracleService, memory_service: Memory, lang_service, env, rotator: RobotRotator):
        self.oracle_service = oracle_service
        self.memory_service = memory_service
        self.lang_service = lang_service
        self.env = env
        self.rotator = rotator

    def fuzzy_match_object(self, objects_in_view: List[Dict], object_name: str) -> Optional[Dict]:
        """
        Perform fuzzy matching to find the best match for the object name.

        Args:
            objects_in_view (List[Dict]): List of objects detected in view.
            object_name (str): Name of the object to match.

        Returns:
            dict or None: Matched object dictionary if a match is found, else None.
        """
        valid_objects = self.oracle_service.get_valid_objects()
        detected_object_names = [list(obj.keys())[0].lower() for obj in objects_in_view]
        best_match = get_close_matches(object_name.lower(), detected_object_names, n=1, cutoff=0.6)

        if best_match:
            best_detected_match = best_match[0]
            if best_detected_match in [v.lower() for v in valid_objects]:
                matched_obj = next((obj for obj in objects_in_view if list(obj.keys())[0].lower() == best_detected_match), None)
                if matched_obj:
                    logger.info(f"Fuzzy matched '{object_name}' to '{best_detected_match}'.")
                    return matched_obj

        logger.warning(f"No fuzzy match found for '{object_name}'.")
        return None

    def detect_object(self, object_name: str) -> Optional[Dict]:
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

        return self.fuzzy_match_object(objects, object_name)

    def fetch_object(self, object_name: str) -> bool:
        """
        Attempt to fetch the object by locating it in the environment.

        Args:
            object_name (str): Name of the object to fetch.

        Returns:
            bool: True if the object was fetched successfully, False otherwise.
        """
        match = self.detect_object(object_name)
        if match:
            match_name = list(match.keys())[0].lower()
            detail = match[match_name].get("detail", "")
            coords = self.oracle_service.get_object_coordinates(match_name)

            if coords:
                target_object = getattr(self.env, match_name, None)
                if not target_object:
                    logger.error(f"Object '{match_name}' not found in the environment.")
                    return False

                result = fetch_and_place_target_object(self.env, target_object, [0.05, 0.05, 0], vis=False)
                if result:
                    logger.info(f"Successfully fetched '{match_name}' and placed it on the table.")
                    self.memory_service.add_object(
                        name=match_name,
                        detail=detail,
                        location={"text": "on the table", "coords": [0.05, 0.05, 0]}
                    )
                    return True
                else:
                    logger.error(f"Failed to fetch '{match_name}'.")
            else:
                logger.warning(f"Coordinates for '{match_name}' not found in oracle.")
        else:
            logger.info(f"No match found in the current view for '{object_name}'.")

        return False

    def fetch_from_memory(self, object_name: str, message_history: List[Dict[str, str]]) -> bool:
        """
        Fetches an object from memory and places it on the table if coordinates are available.

        Args:
            object_name (str): Name of the object to fetch.
            message_history (list): List to append status messages.

        Returns:
            bool: True if the object was successfully fetched and placed, False otherwise.
        """
        obj = self.memory_service.get_object(object_name)
        if obj:
            coords = obj.get("location", {}).get("coords")
            if coords:
                target_object = getattr(self.env, object_name.lower(), None)
                if not target_object:
                    logger.error(f"Object '{object_name}' not found in the environment.")
                    return False

                logger.info(f"Fetching '{object_name}' from memory.")
                result = fetch_and_place_target_object(self.env, target_object, [0.05, 0.05, 0], vis=False)

                if result:
                    logger.info(f"Successfully fetched '{object_name}' and placed it on the table.")
                    self.memory_service.update_location(
                        name=object_name,
                        new_location={"text": "on the table", "coords": [0.05, 0.05, 0]}
                    )
                    message_history.append({"role": "assistant", "content": f"Successfully fetched '{object_name}' and placed it on the table."})
                    return True
                else:
                    logger.error(f"Failed to fetch '{object_name}'.")
        else:
            logger.info(f"Object '{object_name}' not found in memory.")

        return False

    def handle_fetch(self, object_name: str, message_history: List[Dict[str, str]]):
        """
        Handle the fetch command by either fetching from memory or exploring the environment.

        Args:
            object_name (str): Name of the object to fetch.
            message_history (list): List to keep track of message history.
        """
        if self.fetch_from_memory(object_name, message_history):
            return

        logger.info(f"'{object_name}' not found in memory. Initiating environmental scanning...")
        message_history.append({"role": "assistant", "content": f"'{object_name}' not found in memory. Initiating environmental scanning..."})

        views = ["center", "left", "right"]

        for view in views:
            logger.info(f"Rotating to '{view}' view.")
            self.rotator.rotate_robot_to_view(view)
            if self.fetch_object(object_name):
                return

        logger.info("I can't find the object.")
        message_history.append({"role": "assistant", "content": "I can't find the object."})
