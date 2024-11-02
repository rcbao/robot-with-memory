# /src/command_handler.py

from typing import Optional
from object import Object
from memory import Memory
from vision import VisionSystem
from movement import MovementSystem
from language_processor import LanguageProcessor


class CommandHandler:
    def __init__(
        self,
        memory: Memory,
        vision: VisionSystem,
        movement: MovementSystem,
        language_processor: LanguageProcessor,
    ):
        """
        Initialize the CommandHandler with necessary components.

        Args:
            memory (Memory): Instance managing object memory.
            vision (VisionSystem): Instance handling object detection.
            movement (MovementSystem): Instance controlling robot movement.
            language_processor (LanguageProcessor): Instance processing language.
        """
        self.memory = memory
        self.vision = vision
        self.movement = movement
        self.language_processor = language_processor

    def find_object_from_current_view(
        self, object_name: str, detail: str
    ) -> Optional[Object]:
        """
        Attempt to find the object in the current view.

        Args:
            object_name (str): Name of the object.
            detail (str): Detail/description of the object.

        Returns:
            Optional[Object]: Detected object or None.
        """
        detected_objects = self.vision.identify_objects_in_robot_view()
        for obj in detected_objects:
            print(obj)
            name_match = obj.name.lower() == object_name.lower()
            detail_match = detail and obj.detail and (detail.lower() in obj.detail.lower())
            # TODO: incorporate detail somehow
            if name_match:
                print(f"found object:: {object_name.lower()}")
                return obj
        return None

    def find_object_by_scanning_env(
        self, object_name: str, detail: str
    ) -> Optional[Object]:
        """
        Perform a 360-degree scan to locate the object.

        Args:
            object_name (str): Name of the object.
            detail (str): Detail/description of the object.

        Returns:
            Optional[Object]: Detected object or None.
        """
        # No need to track total_rotation, we are only rotating 360 degrees
        self.movement.move_arm_down_to_clear_view()
        for _ in range(4):  # Rotate in 60-degree increments (6 steps of 60 degrees)
            self.movement.rotate_robot(60)  # Rotate incrementally by 60 degrees
            print("...Trying to find object from current view")
            obj = self.find_object_from_current_view(object_name, detail)
            if obj:
                return obj

        return None

    def find_object(self, object_name: str, detail: str) -> Optional[Object]:
        """
        Locate the object by searching memory, current view, or scanning environment.

        Args:
            object_name (str): Name of the object.
            detail (str): Detail/description of the object.

        Returns:
            Optional[Object]: Located object or None.
        """
        # Search memory
        print("CommandHandler.find_object() -- Searching memory")
        obj = self.memory.find_object_from_past_memory(object_name, detail)
        if obj:
            return obj

        # Search current view
        print("CommandHandler.find_object() -- Searching current view")
        obj = self.find_object_from_current_view(object_name, detail)
        if obj:
            return obj

        # Scan environment
        print("CommandHandler.find_object() -- Scanning environment")
        obj = self.find_object_by_scanning_env(object_name, detail)
        return obj

    def handle_remember(self, object_name: str, detail: str) -> str:
        """
        Handle 'remember' command by locating and saving the object.

        Args:
            object_name (str): Name of the object.
            detail (str): Detail/description of the object.

        Returns:
            str: Feedback message.
        """
        obj = self.find_object(object_name, detail)
        if obj:
            self.memory.save_object_to_memory(obj)
            return f"I have remembered that the {obj.name} is located at {obj.location_description}."
        return "I couldn't locate the object to remember it."

    def handle_recall(self, object_name: str, detail: str) -> str:
        """
        Handle 'recall' command by retrieving object's location from memory.

        Args:
            object_name (str): Name of the object.
            detail (str): Detail/description of the object.

        Returns:
            str: Feedback message.
        """
        obj = self.memory.find_object_from_past_memory(object_name, detail)
        if obj:
            return f"The {obj.name} is located at {obj.location_description}."
        return "I don't have a record of that object."

    def handle_fetch(self, object_name: str, detail: str, destination: tuple) -> str:
        """
        Handle 'fetch' command by locating, grasping, and delivering the object.

        Args:
            object_name (str): Name of the object.
            detail (str): Detail/description of the object.
            destination (tuple): Coordinates to deliver the object.

        Returns:
            str: Feedback message.
        """
        print("running:: handle_fetch")
        obj = self.find_object(object_name, detail)
        if obj and obj.location_3d_coords:
            # Unpack the 3D coordinates tuple into separate variables
            x0, y0, z0 = obj.location_3d_coords
            print("got location_3d_coords")
            print(obj.location_3d_coords)
            print("self.movement.fetch_object")
            success = self.movement.fetch_object(x0, y0, z0)
            if success:
                # Use 'navigate_to' instead of 'go_to' as per movement.py
                self.movement.navigate_to(destination)
                self.movement.release_object()
                return f"I have fetched the {obj.name} and delivered it."
            return "I found the object but couldn't grasp it."
        return "I couldn't locate the object to fetch it."

    def handle_irrelevant_prompt(self, user_prompt: str) -> str:
        """
        Handle prompts unrelated to memory or fetching.

        Args:
            user_prompt (str): User's input prompt.

        Returns:
            str: Generated response.
        """
        return self.language_processor.generate_response(user_prompt)

    def interpret_user_prompt(
        self, user_prompt: str
    ) -> tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """
        Interpret the user's prompt to determine action and object details.

        Args:
            user_prompt (str): User's input prompt.

        Returns:
            Tuple containing relevancy, action, object_name, and detail.
        """
        return self.language_processor.interpret_user_prompt(user_prompt)

    def provide_feedback_to_user(self, response: str):
        """
        Provide feedback to the user.

        Args:
            response (str): Response message.
        """
        print(f"[Robot Response] {response}")