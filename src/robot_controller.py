from typing import List, Dict
from services.location_oracle_service import Oracle
from services.memory_service import Memory
from services.language_service import LanguageService
from services.fetch_service import FetchService
from utils.logger import setup_logger
from utils.robot_rotator import RobotRotator
from utils.init_env import init_env

logger = setup_logger()

class RobotController:
    """
    Controller class to manage robot operations based on user commands.
    """
    def __init__(self):
        # Initialize services
        self.oracle_service = Oracle()
        if not self.oracle_service.data:
            logger.error("Failed to load Oracle data. Exiting controller initialization.")
            raise SystemExit("Oracle data is essential for operation.")
        
        self.memory_service = Memory()
        self.lang_service = LanguageService()
        self.env = init_env()
        self.env.reset()
        logger.info("Environment initialized.")

        self.rotator = RobotRotator(self.env)
        self.fetch_service = FetchService(
            oracle_service=self.oracle_service,
            memory_service=self.memory_service,
            lang_service=self.lang_service,
            env=self.env,
            rotator=self.rotator
        )
        self.message_history: List[Dict[str, str]] = []

    def handle_recall(self, object_name: str):
        """
        Handle the recall command by first rotating and observing the environment 
        in three directions to locate the object, and then checking memory.

        Args:
            object_name (str): Name of the object to recall.
        """
        found = False
        views = ["left", "center", "right"]

        for view in views:
            logger.info(f"Rotating to '{view}' view.")
            self.rotator.rotate_robot_to_view(view)
            detected_object = self.fetch_service.detect_object_from_camera_image(object_name)
            if detected_object:
                detect_location = detected_object.get("location", {})
                location_text = detect_location.get("text", "")

                match_name, detail, coords = self.get_match_info(detected_object)
                location_coords = self.oracle_service.get_object_coordinates(match_name)

                self.memory_service.add_object(
                    name=object_name,
                    detail=detected_object.get("detail", ""),
                    location={"text": location_text, "coords": location_coords}
                )
                logger.info(f"Recall: {location_text}")
                self.message_history.append({"role": "assistant", "content": location_text})
                print(location_text)
                found = True
                break

        if not found:
            obj = self.memory_service.get_object(object_name)
            if obj:
                location_text = obj.get("location", {}).get("text", "unknown location")
                message = f"The {object_name} is {location_text}."
                logger.info(f"Recall: {message}")
                self.message_history.append({"role": "assistant", "content": message})
                print(message)
            else:
                message = f"I don't have any record of the {object_name}."
                logger.info(f"Recall: {message}")
                self.message_history.append({"role": "assistant", "content": message})
                print(message)


    def process_command(self, user_input: str):
        """
        Process the user's command.

        Args:
            user_input (str): The input string from the user.
        """
        relevancy, action, object_name, detail = self.lang_service.parse_user_input(user_input, self.message_history)

        if not action or not isinstance(action, str):
            response = "Sorry, I didn't understand that command."
            print(response)
            self.message_history.append({"role": "assistant", "content": response})
            return

        if relevancy and object_name:
            action_lower = action.lower()
            if action_lower == 'fetch':
                self.fetch_service.handle_fetch(object_name, self.message_history)
            elif action_lower == 'recall':
                self.handle_recall(object_name)
            else:
                response = self.lang_service.write_generic_response(user_input, self.message_history)
                response = response or "Sorry, I couldn't understand your request."
                print(response)
                self.message_history.append({"role": "assistant", "content": response})
        else:
            response = self.lang_service.write_generic_response(user_input, self.message_history)
            response = response or "Sorry, I couldn't understand your request."
            print(response)
            self.message_history.append({"role": "assistant", "content": response})

    def shutdown(self):
        """
        Clean up resources before shutting down.
        """
        self.memory_service.clear_memory()
        self.env.close()
        logger.info("Environment closed.")
