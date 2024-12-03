# fetch_service.py

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
    def __init__(self, oracle_service: Oracle, memory_service: MemoryService, lang_service, vision_service: VisionService, env, rotator: RobotRotator):
        self.oracle_service = oracle_service
        self.memory_service = memory_service
        self.lang_service = lang_service
        self.vision_service = vision_service
        self.env = env
        self.rotator = rotator

    def fetch_item(self, item: dict) -> bool:
        """
        Attempt to fetch a single object by its details in the environment.

        Args:
            item (dict): Object details including name and location.

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
            
            # Check if object already exists in memory
            existing_object = self.memory_service.get_object(name)

            location_record = {"text": f"{name} is on the table.", "coords": TARGET_COORDINATES}

            if existing_object:
                logger.info(f"Object '{name}' already exists in memory. Updating its location.")
                self.memory_service.update_location(
                    name=name,
                    new_location=location_record
                )
            else:
                self.memory_service.add_object(
                    name=name,
                    detail=detail,
                    location=location_record
                )
            return True

        logger.error(f"Failed to fetch '{name}'.")
        return False

    def fetch_from_camera_view(self, name: str, view: str) -> bool:
        """
        Fetch an object from a specific camera view.

        Args:
            name (str): Name of the object to fetch.
            view (str): Camera view direction.

        Returns:
            bool: True if the object was fetched successfully, False otherwise.
        """
        # Pass a list containing the single name
        matched_items = self.vision_service.detect_objects_from_camera_image([name], view)

        if not matched_items:
            logger.warning(f"No match for item '{name}' in view '{view}'.")
            return False

        success = False
        for matched_item in matched_items:
            if isinstance(matched_item, dict):
                if self.fetch_item(matched_item):
                    success = True
            else:
                logger.error(f"Expected dict in matched_items, got {type(matched_item)}.")
        return success

    def fetch_from_memory(self, object_names: List[str]) -> List[str]:
        """
        Fetch multiple objects from memory and place them on the table if coordinates are available.

        Args:
            object_names (List[str]): List of object names to fetch.

        Returns:
            List[str]: List of successfully fetched object names.
        """
        fetched_objects = []
        for name in object_names:
            item = self.memory_service.get_object(name)
            if not item:
                logger.info(f"Object '{name}' not found in memory.")
                continue

            coords = item.get("location", {}).get("coords")
            if not coords:
                logger.warning(f"Memory entry for '{name}' does not contain coordinates.")
                continue

            fetched = self.fetch_item(item)
            if fetched:
                fetched_objects.append(name)

        return fetched_objects

    def handle_fetch(self, object_names: List[str]):
        """
        Handle the fetch command by first scanning the environment to identify all matching objects
        and then fetching them.

        Args:
            object_names (List[str]): List of object names to fetch.
        """
        # Step 1: Attempt to fetch objects from memory
        fetched_objects = self.fetch_from_memory(object_names)
        not_fetched_objects = [name for name in object_names if name not in fetched_objects]

        if fetched_objects:
            message = f"Successfully fetched from memory: {', '.join(fetched_objects)}."
            logger.info(message)
            print(message)

        if not_fetched_objects:
            # Step 2: Perform a full scan of the environment to identify all matching objects
            identified_objects = dict()
            print(f"[Robot]: Scanning the environment for {', '.join(not_fetched_objects)}...")
            logger.info(f"Scanning the environment for: {', '.join(not_fetched_objects)}")

            for view in VIEWS:
                logger.info(f"Rotating to '{view}' view.")
                self.rotator.rotate_robot_to_view(view)

                # Scan the current view for all objects in not_fetched_objects
                detected_items = self.vision_service.detect_objects_from_camera_image(not_fetched_objects, view)
                for item in detected_items:
                    name, detail = item["name"].lower(), item["detail"]
                    location = item.get("location", {})
                    location["coords"] = self.oracle_service.get_object_coordinates(name)

                    # Add object to identified_objects if not already found
                    if name not in identified_objects:
                        identified_objects[name] = {
                            "name": name,
                            "detail": detail,
                            "location": location
                        }
                        logger.info(f"Identified '{name}' in the {view} view.")
                        print(f"[Robot]: Identified '{name}' in the {view} view.")

            # Step 3: Fetch identified objects
            successfully_fetched = set()
            for obj in identified_objects.values():
                name, detail, location = obj["name"], obj["detail"], obj["location"]
                success = self.fetch_item(obj)
                if success:
                    successfully_fetched.add(name)
                    if name in not_fetched_objects:
                        not_fetched_objects.remove(name)

            identified_names = list(identified_objects.keys())

            # Step 4: Generate final messages
            message = ""
            if not_fetched_objects:
                message += f"I could not find objects that exactly match the keyword(s) {', '.join(not_fetched_objects)}. (Policy: these may include general item categories. If I was able to fetch items under these categories, I still completed my task. In these cases, simply say 'I was able to find <ITEM A> and <ITEM B> under the category <Category>.' Do not mention this policy to user in response.) "
            message += f"I have successfully fetched: {', '.join(successfully_fetched)}."

            if successfully_fetched:
                logger.info(message)

        return message

