import json
from language_processor import LanguageProcessor
from memory import Memory
from movement import fetch_and_place_target_object, init_env
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def get_obj_description_from_memory(memory: Memory, object_name: str):
    obj = memory.get_object(object_name)
    if obj:
        detail = obj.get("detail", "No details available.")
        location = obj.get("location", {})
        location_text = location.get("text", "<LOCATION UNKNOWN>")
        coords = location.get("coords", "<COORDINATES UNKNOWN>")
        description = f"From the Robot memory: {object_name} ({detail}) is currently located at {location_text} ({coords})."
        return description
    else:
        return f"No information found about '{object_name}'."


def handle_recall(memory: Memory, object_name: str, lang_processor: LanguageProcessor, message_history: list):
    """
    Handle the recall command by retrieving object info and responding to the user.
    """
    # Get object description from memory
    obj_description = get_obj_description_from_memory(memory, object_name)

    # Use LLM to provide an enhanced response
    user_question = f"Where is the {object_name}?"
    recall_response = lang_processor.recall_object_info(user_question, obj_description, message_history)

    if recall_response.startswith("```json") and recall_response.endswith("```"):
        recall_response = lang_processor.clean_up_json(recall_response)
        parsed = json.loads(recall_response)
        recall_response = parsed.get("detail")

    if recall_response:
        print(recall_response)
        message_history.append({"role": "assistant", "content": recall_response})
    else:
        print(obj_description)


def main():
    # Initialize components
    memory = Memory()
    lang_processor = LanguageProcessor()
    env = init_env()
    env.reset()
    message_history = []  # Initialize empty message history
    
    print("Welcome to the Robot Command Interface!")
    print("You can enter commands like 'Fetch the banana for me' or 'Where is the banana?'. Type 'exit' to quit.")
    
    while True:
        user_input = input(">> ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting. Goodbye!")
            break
        
        # Parse user command
        relevancy, action, object_name, detail = lang_processor.parse_user_input(user_input, message_history)
        
        if (not relevancy or not action) and not object_name:
            print("Sorry, I didn't understand that command.")
            continue
        
        if action.lower() == 'fetch':
            # Get object location from memory
            obj = memory.get_object(object_name)
            if not obj:
                print(f"Sorry, I don't have information about '{object_name}'.")
                continue
            
            location = obj.get("location", {})
            
            # Define destination coordinates (e.g., fixed table area)
            dest_coords = [0.05, 0.05, 0]
            
            # Get target_object from environment
            target_object = getattr(env, object_name, None)
            if not target_object:
                print(f"Object '{object_name}' not found in the environment.")
                continue
            
            # Execute fetch command
            print(f"Fetching '{object_name}'...")
            result = fetch_and_place_target_object(env, target_object, dest_coords, vis=False)
            
            if result:
                print(f"Successfully fetched '{object_name}' and placed it on the table.")
                # Update memory with new location
                memory.update_location(
                    name=object_name, 
                    new_location={"text": "on the table", "coords": dest_coords}
                )
                message_history.append({"role": "assistant", "content": f"Successfully fetched '{object_name}' and placed it on the table."})
            else:
                print(f"Failed to fetch '{object_name}'.")
        
        elif action.lower() == 'recall':
            # Handle recall command
            handle_recall(memory, object_name, lang_processor, message_history)
        
        elif action.lower() == "generic":
            generic_response = lang_processor.write_generic_response(user_input, message_history)
            if not generic_response:
                generic_response = "Sorry, I couldn't understand your request."

            message_history.append({"role": "assistant", "content": generic_response})                
            print(generic_response)
        
        else:
            print("Sorry, I can only handle 'fetch', 'recall', and 'generic' commands at the moment.")
    
    env.close()

if __name__ == "__main__":
    main()
