# robot_controller.py

from typing import List, Dict
from services.location_oracle_service import Oracle
from services.memory_service import MemoryService
from services.language_service import LanguageService
from services.fetch_service import FetchService
from services.vision_service import VisionService
from utils.logger import setup_logger
from utils.robot_rotator import RobotRotator
from utils.init_env import init_env
from constants import VIEWS, TARGET_COORDINATES

logger = setup_logger()

CLEAR_MEMORY_ON_SHUTDOWN = False

class RobotController:
    """
    Controller class to manage robot operations based on user commands.
    """
    def __init__(self):
        self.env = init_env()
        self.env.reset()

        # Initialize services
        self.oracle_service = Oracle()
        if not self.oracle_service.data:
            logger.error("Failed to load Oracle data. Exiting controller initialization.")
            raise SystemExit("Oracle data is essential for operation.")
        self.memory_service = MemoryService()
        self.lang_service = LanguageService()
        self.vision_service = VisionService(self.oracle_service, self.env)

        logger.info("Environment initialized.")

        self.rotator = RobotRotator(self.env)
        self.fetch_service = FetchService(
            oracle_service=self.oracle_service,
            memory_service=self.memory_service,
            lang_service=self.lang_service,
            vision_service=self.vision_service,
            env=self.env,
            rotator=self.rotator
        )
        self.message_history: List[Dict[str, str]] = []

    def rewrite_response(self, input_text: str):
        return self.lang_service.rewrite_response(input_text)

    def handle_recall(self, object_names: List[str]):
        """
        Handle the recall command by first checking memory, and if the object is not found,
        scanning the environment in all directions to locate the requested objects.

        Args:
            object_names (List[str]): List of object names to recall.
        """
        objects_to_recall = [name.lower() for name in object_names]
        recalled_objects = []
        not_found_objects = []

        logger.info(f"Attempting to recall objects: {objects_to_recall}")
        logger.info(f"[Robot]: Checking memory for the requested object(s): {', '.join(objects_to_recall)}...")

        # Step 1: Check memory for each object
        for name in objects_to_recall:
            obj = self.memory_service.get_object(name)
            if obj:
                location_text = obj.get("location", {}).get("text", "unknown location")
                recalled_objects.append(name)
                logger.info(f"Recalled from memory: {name} at {location_text}")
            else:
                not_found_objects.append(name)

        # Step 2: Scan environment for objects not found in memory
        if not_found_objects:
            print(f"[Robot]: Scanning the environment for missing objects: {', '.join(not_found_objects)}")
            for view in VIEWS:
                logger.info(f"Rotating to '{view}' view.")
                self.rotator.rotate_robot_to_view(view)
                detected_items = self.vision_service.detect_objects_from_camera_image(not_found_objects, view)

                for item in detected_items:
                    name, detail = item["name"].lower(), item["detail"]
                    location = item.get("location", {})
                    location["coords"] = self.oracle_service.get_object_coordinates(name)

                    # Add or update object in memory
                    if not self.memory_service.get_object(name):
                        self.memory_service.add_object(name=name, detail=detail, location=location)
                    else:
                        self.memory_service.update_location(name=name, new_location=location)

                    recalled_objects.append(name)
                    not_found_objects.remove(name)
                    logger.info(f"Found from scan: {name} at {location.get('text', 'unknown location')}")

                if not not_found_objects:
                    break  # Stop scanning if all objects are found

        response = ""
        # Step 3: Generate response
        if recalled_objects:
            recall_messages = [f"{name}: {self.memory_service.get_object(name)['location'].get('text', 'unknown location')}" for name in recalled_objects]
            recall_messages = f"Objects:\n" + "\n".join(recall_messages)
            response += recall_messages

        # if not_found_objects:
        #     not_found_message = f"I couldn't find the following objects: {', '.join(not_found_objects)}."
        #     response += not_found_message
            
        return response


    def handle_fetch(self, object_names: List[str]):
        """
        Handle the fetch command by either fetching from memory or exploring the environment.

        Args:
            object_names (List[str]): List of object names to fetch.
            message_history (list): List to keep track of message history.
        """
        # Delegate the fetch handling to fetch_service
        return self.fetch_service.handle_fetch(object_names)

    def process_command(self, user_input: str):
        """
        Process the user's command.

        Args:
            user_input (str): The input string from the user.
        """
        relevancy, action, object_names, detail = self.lang_service.parse_user_input(user_input, self.message_history)

        if not action or not isinstance(action, str):
            response = "Sorry, I didn't understand that command."
            print(response)
            self.message_history.append({"role": "assistant", "content": response})
            return
        
        response = ""

        if relevancy and object_names:
            action_lower = action.lower()
            if action_lower == 'fetch':
                response = self.handle_fetch(object_names)
            elif action_lower == 'recall':
                response = self.handle_recall(object_names)
        else:
            response = self.lang_service.write_generic_response(user_input, self.message_history)

        response = self.rewrite_response(response)
        self.message_history.append({"role": "assistant", "content": response})
        print(response)

    def shutdown(self):
        """
        Clean up resources before shutting down.
        """
        if CLEAR_MEMORY_ON_SHUTDOWN:
            self.memory_service.clear_memory()
        self.env.close()
        logger.info("Environment closed.")
