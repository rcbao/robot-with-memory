from typing import List, Dict
from services.location_oracle_service import OracleService
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
        self.oracle_service = OracleService()
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
        Handle the recall command by retrieving object information from memory.

        Args:
            object_name (str): Name of the object to recall.
        """
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
        self.env.close()
        logger.info("Environment closed.")
