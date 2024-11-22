# main.py
import json
from language_processor import LanguageProcessor
from memory import Memory
from movement import fetch_and_place_target_object, init_env, RobotRotator
from camera_utils import save_camera_image_by_type
from robot_rotator import RobotRotator
import warnings
import logging
from typing import Optional
from difflib import get_close_matches

warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)


def fuzzy_match_object(objects_in_view: list[dict], object_name: str, valid_objects: list[str]) -> Optional[dict]:
    """
    Perform fuzzy matching to find the best match for the object name.
    Use vector-based search to verify if the `object_name` object is in view, then verify if it is in the list of valid object names.
    """
    detected_object_names = [list(obj.keys())[0].lower() for obj in objects_in_view]
    best_detected_match = next(iter(get_close_matches(object_name.lower(), detected_object_names, n=1, cutoff=0.6)), None)

    if best_detected_match and best_detected_match in [v.lower() for v in valid_objects]:
        return next(obj for obj in objects_in_view if list(obj.keys())[0].lower() == best_detected_match)
    
    return None


def get_object_coordinates_oracle():
    try:
        with open("object_coords_oracle.json", "r") as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        logging.error("object_coords_oracle.json file not found.")
        return None

def get_object_coordinates_from_oracle(oracle: dict, object_name: str) -> Optional[list]:
    """
    Retrieve object coordinates from the object_coords_oracle.json file.
    """
    return oracle[object_name]["location"]["coords"]


def robot_fetch_object_process(object_name: str, rotator: RobotRotator, env, memory: Memory, object_coords_oracle: dict, lang_processor) -> bool:
    """
    Process to fetch an object by exploring the environment if it's not in memory.

    Args:
        object_name (str): Name of the object to fetch.
        rotator (RobotRotator): Instance of RobotRotator to control robot rotations.
        env: The robot's simulation environment.
        memory (Memory): Memory instance to manage object locations.

    Returns:
        bool: True if the object was fetched successfully, False otherwise.
    """
    views = ["center", "left", "right"]
    valid_objects = list(object_coords_oracle.keys())

    print("valid_objects::")
    print(valid_objects)

    for view in views:
        logging.info(f"Rotating to '{view}' view.")
        rotator.rotate_robot_to_view(view)

        # Capture image
        encoded_image = save_camera_image_by_type(env, camera_type="front_camera")

        # Analyze image with GPT-4o
        objects = lang_processor.list_objects_in_scene_image(encoded_image)

        logging.debug(f"GPT-4o Response: {objects}")

        # Perform fuzzy matching
        match = fuzzy_match_object(objects, object_name, valid_objects)

        print("match::")
        print(match)
        
        if match:
            match_name = next(iter(match.keys()))
            detail = match[match_name]["detail"]
            logging.info(f"Object '{match_name}' identified in view '{view}'.")
            coords = get_object_coordinates_from_oracle(object_coords_oracle, match_name)
            if coords:
                result = fetch_and_place_target_object(env, getattr(env, match_name, None), [0.05, 0.05, 0], vis=False)
                if result:
                    logging.info(f"Successfully fetched '{match_name}' and placed it on the table.")
                    memory.add_object(
                        name=match_name,
                        detail=detail,
                        location={"text": "on the table", "coords": [0.05, 0.05, 0]}
                    )
                    return True
            else:
                logging.warning(f"Coordinates for '{match_name}' not found in oracle.")
    return False


def handle_fetch(memory: Memory, object_name: str, lang_processor: LanguageProcessor, rotator: RobotRotator, env, message_history: list):
    """
    Handle the fetch command by either fetching from memory or exploring the environment.

    Args:
        memory (Memory): Memory instance to manage object locations.
        object_name (str): Name of the object to fetch.
        lang_processor (LanguageProcessor): LanguageProcessor instance to handle language tasks.
        rotator (RobotRotator): RobotRotator instance to control robot rotations.
        env: The robot's simulation environment.
        message_history (list): List to keep track of message history.
    """
    oracle = get_object_coordinates_oracle()
    obj = memory.get_object(object_name)
    if obj:
        coords = obj.get("location", {}).get("coords", None)
        if coords:
            target_object = getattr(env, object_name, None)
            if not target_object:
                print(f"Object '{object_name}' not found in the environment.")
                return

            print(f"Fetching '{object_name}' from memory...")
            result = fetch_and_place_target_object(env, target_object, [0.05, 0.05, 0], vis=False)

            if result:
                print(f"Successfully fetched '{object_name}' and placed it on the table.")
                memory.update_location(
                    name=object_name,
                    new_location={"text": "on the table", "coords": [0.05, 0.05, 0]}
                )
                message_history.append({"role": "assistant", "content": f"Successfully fetched '{object_name}' and placed it on the table."})
                return
            else:
                print(f"Failed to fetch '{object_name}'.")
    
    # If object is not in memory or fetching failed, proceed to explore
    print(f"'{object_name}' not found in memory. Initiating environmental scanning...")
    success = robot_fetch_object_process(object_name, rotator, env, memory, oracle, lang_processor)
    if not success:
        print("I can't find the object.")
        message_history.append({"role": "assistant", "content": "I can't find the object."})


def main():
    # Initialize components
    memory = Memory()
    lang_processor = LanguageProcessor()
    env = init_env()
    obs = env.reset()
    message_history = []  # Initialize empty message history
    print("----------------------------------------")
    print("Welcome to the Robot Command Interface!")
    print("You can enter commands like 'Fetch the banana for me' or 'Where is the banana?'. Type 'exit' to quit.")
    print("----------------------------------------")

    rotator = RobotRotator(env)
    while True:
        user_input = input(">> ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting. Goodbye!")
            break
        
        # Parse user command
        relevancy, action, object_name, detail = lang_processor.parse_user_input(user_input, message_history)
        
        if not action or not isinstance(action, str):
            print("Sorry, I didn't understand that command.")
            continue
        
        if relevancy and object_name and action.lower() == 'fetch':
            handle_fetch(memory, object_name, lang_processor, rotator, env, message_history)
        elif relevancy and object_name and action.lower() == 'recall':
            # Handle recall command
            handle_recall(memory, object_name, lang_processor, message_history)
        
        else:
            generic_response = lang_processor.write_generic_response(user_input, message_history)
            if not generic_response:
                generic_response = "Sorry, I couldn't understand your request."

            message_history.append({"role": "assistant", "content": generic_response})                
            print(generic_response)

    env.close()

if __name__ == "__main__":
    main()
