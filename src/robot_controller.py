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
        Handle the recall command by rotating and observing the environment
        in all directions to locate all requested objects.

        Args:
            object_names (List[str]): List of object names to recall.
        """
        objects_to_recall = [name.lower() for name in object_names]
        recalled_objects = []
        not_found_objects = objects_to_recall.copy()

        recall_messages = []

        logger.info(f"Attempting to recall objects: {objects_to_recall}")

        print(f"> Scanning the environment to find the requested objects: {', '.join(objects_to_recall)}")

        for view in VIEWS:
            logger.info(f"Rotating to '{view}' view.")
            self.rotator.rotate_robot_to_view(view)
            detected_items = self.vision_service.detect_objects_from_camera_image(objects_to_recall, view)

            if detected_items:
                for item in detected_items:
                    name, detail = item["name"].lower(), item["detail"]
                    location = item.get("location", {})

                    if name not in recalled_objects:
                        item_coordinates = self.oracle_service.get_object_coordinates(name)

                        location["coords"] = item_coordinates

                        existing_object = self.memory_service.get_object(name)
                        if not existing_object:
                            self.memory_service.add_object(
                                name=name,
                                detail=detail,
                                location=location
                            )
                        else:
                            self.memory_service.update_location(
                                name=name,
                                new_location=location
                            )
                        recalled_objects.append(name)

                        message = location["text"]
                        logger.info(f"> {message}")
                        recall_messages.append(message)
                        
                    if name in not_found_objects:
                        not_found_objects.remove(name)
                    
                    break
                
        self.rotator.rotate_robot_to_view("left")

        if recalled_objects:
            logger.info(f"Recalled objects: {recalled_objects}")
            full_recall_message = f"Successfully recalled: {', '.join(recalled_objects)}. {''.join(recall_messages)}"
            self.message_history.append({"role": "assistant", "content": full_recall_message})

            full_recall_message = self.rewrite_response(full_recall_message)
            print(full_recall_message)
            return

        if not_found_objects:
            message = f"I couldn't find the following objects: {', '.join(not_found_objects)}."
            logger.info(f"Recall: {message}")
            self.message_history.append({"role": "assistant", "content": message})
            print(f"> {message}")

    def handle_fetch(self, object_names: List[str], message_history: List[Dict[str, str]]):
        """
        Handle the fetch command by either fetching from memory or exploring the environment.

        Args:
            object_names (List[str]): List of object names to fetch.
            message_history (list): List to keep track of message history.
        """
        # Delegate the fetch handling to fetch_service
        self.fetch_service.handle_fetch(object_names, message_history)

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

        if relevancy and object_names:
            action_lower = action.lower()
            if action_lower == 'fetch':
                self.handle_fetch(object_names, self.message_history)
            elif action_lower == 'recall':
                self.handle_recall(object_names)
            else:
                response = self.lang_service.write_generic_response(user_input, self.message_history)
                response = response or "Sorry, I couldn't understand your request."

                response = self.rewrite_response(response)
                print(response)
                self.message_history.append({"role": "assistant", "content": response})
        else:
            response = self.lang_service.write_generic_response(user_input, self.message_history)
            response = response or "Sorry, I couldn't understand your request."

            response = self.rewrite_response(response)
            print(response)
            self.message_history.append({"role": "assistant", "content": response})

    def shutdown(self):
        """
        Clean up resources before shutting down.
        """
        self.memory_service.clear_memory()
        self.env.close()
        logger.info("Environment closed.")
