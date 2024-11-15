# main.py

from language_processor import LanguageProcessor
from memory import Memory
from movement import fetch_and_place_target_object, init_env

def main():
    # Initialize components
    memory = Memory()
    lang_processor = LanguageProcessor()
    env = init_env()
    
    print("Welcome to the Robot Command Interface!")
    print("You can enter commands like 'Fetch the apple for me'. Type 'exit' to quit.")
    
    while True:
        user_input = input(">> ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting. Goodbye!")
            break
        
        # Parse user command
        relevancy, action, object_name, detail = lang_processor.parse_user_input(user_input)
        
        if not relevancy or action != 'fetch' or not object_name:
            print("Sorry, I didn't understand that command.")
            continue
        
        # Get object location from memory
        obj = memory.get_object(object_name)
        if not obj:
            print(f"Sorry, I don't have information about '{object_name}'.")
            continue
        
        location = obj["location"]
        
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
            # Update memory if needed
            memory.update_location(name=object_name, new_location={"shelf": "table", "position": dest_coords})
        else:
            print(f"Failed to fetch '{object_name}'.")
    
    env.close()

if __name__ == "__main__":
    main()

