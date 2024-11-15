# /src/command_handler.py

from typing import Optional
from sim_object import Object
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
        """
        self.memory = memory
        self.vision = vision
        self.movement = movement
        self.language_processor = language_processor

    def format_object_location_desc(self, obj):
        print("object::")
        print(obj)
        name = obj.name
        coords, description = obj.location_3d_coords, obj.location_description
        if name:
            if coords and description:
                return f"{obj.name} is located at {obj.location_description} (coordinates: {coords})"
            if coords:
                return f"{obj.name} is located at coordinates {coords}"
            if description:
                return f"{obj.name} is located at {obj.location_description}"
        return "<Object is empty>"

    def find_object_from_current_view(self, name: str, detail: str):
        """
        Attempt to find the object in the current view.

        Args:
            name (str): Name of the object.
            detail (str): Detail/description of the object.

        Returns:
            Optional[Object]: Detected object or None.
        """
        detected_objects = self.vision.identify_objects_in_robot_view()
        for obj in detected_objects:
            print(obj)
            name_match = obj.name.lower() == name.lower()
            detail_match = detail and obj.detail and (detail.lower() in obj.detail.lower())
            # TODO: Incorporate object location details
            if name_match:
                print(f"found object:: {name.lower()}")
                return obj
        return None

    def find_object_by_scanning_env(self, name: str, detail: str) -> Optional[Object]:
        """
        Perform a 360-degree scan to locate the object.

        Args:
            name (str): Name of the object.
            detail (str): Detail/description of the object.

        Returns:
            Optional[Object]: Detected object or None.
        """
        # No need to track total_rotation, we are only rotating 360 degrees
        self.movement.move_arm_down_to_clear_view()
        for _ in range(4):  
            self.movement.rotate_robot(60)  
            print("...Trying to find object from current view")
            obj = self.find_object_from_current_view(name, detail)
            if obj:
                return obj

        return None

    def find_object(self, name: str, detail: str) -> Optional[Object]:
        """
        Locate the object by searching memory, current view, or scanning environment.

        Args:
            name (str): Name of the object.
            detail (str): Detail/description of the object.

        Returns:
            Optional[Object]: Located object or None.
        """
        print("CommandHandler.find_object() :: Searching memory")
        obj = self.memory.find_object_from_past_memory(name, detail)
        if obj:
            return obj

        print("CommandHandler.find_object() :: Searching current view")
        obj = self.find_object_from_current_view(name, detail)
        if obj:
            return obj

        print("CommandHandler.find_object() :: Scanning environment")
        obj = self.find_object_by_scanning_env(name, detail)
        return obj

    def handle_remember(self, name: str, detail: str) -> str:
        """
        Handle 'remember' command by locating and saving the object.

        Args:
            name (str): Name of the object.
            detail (str): Detail/description of the object.

        Returns:
            str: Feedback message.
        """
        obj = self.find_object(name, detail)
        if obj:
            self.memory.save_object_to_memory(obj)
            return f"I have remembered that the {obj.name} is located at {obj.location_description}."
        return "I couldn't locate the object to remember it."

    def handle_recall(self, name: str, detail: str) -> str:
        """
        Handle 'recall' command by retrieving object's location from memory.

        Args:
            name (str): Name of the object.
            detail (str): Detail/description of the object.

        Returns:
            str: Feedback message.
        """
        obj = self.memory.find_object_from_past_memory(name, detail)

        if obj:
            description = self.format_object_location_desc(obj)
            response = f"The {description}."
            return response

        fallback_response = "I don't have a record of that object."
        return fallback_response

    def handle_fetch(self, name: str, detail: str, destination: tuple) -> str:
        """
        Handle 'fetch' command by locating, grasping, and delivering the object.

        Args:
            name (str): Name of the object.
            detail (str): Detail/description of the object.
            destination (tuple): Coordinates to deliver the object.

        Returns:
            str: Feedback message.
        """
        obj = self.find_object(name, detail)

        if obj and obj.location_3d_coords:
            x0, y0, z0 = obj.location_3d_coords
            print("Robot acquired location_3d_coords:: ", obj.location_3d_coords)
            
            # Step 1: Acquire the object (Move close to the object, grasp it, and lift up the arm)
            success = self.movement.acquire_object(x0, y0, z0)
            if success:
                # Step 2: Move back to destination
                self.movement.navigate_to(destination)
                
                response = f"I have fetched the {obj.name} and delivered it."
                return response
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

    def parse_user_input(self, user_prompt: str):
        """
        Interpret the user's prompt to determine action and object details.

        Args:
            user_prompt (str): User's input prompt.

        Returns:
            Tuple containing relevancy, action, name, and detail.
        """
        return self.language_processor.parse_user_input(user_prompt)

    def provide_feedback_to_user(self, response: str):
        """
        Provide feedback to the user.

        Args:
            response (str): Response message.
        """
        print(f"[Robot Response] {response}")