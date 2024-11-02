# /src/main.py

import time
from movement import MovementSystem
from memory import Memory
from vision import VisionSystem
from language_processor import LanguageProcessor
from command_handler import CommandHandler
from robot import Robot
from maniskill_simulator import ManiSkillSimulator

def get_robot_request() -> str:
    """
    Placeholder for getting user input.
    Replace with actual input method as needed (e.g., voice command).
    """
    return input("Enter command (e.g., 'Remember dark grey trash can location'): ")

def main():
    # Initialize the simulator
    simulator = ManiSkillSimulator()
    simulator.start()

    # Initialize robot components
    memory = Memory()
    vision = VisionSystem(simulator)
    movement = MovementSystem(simulator)
    language_processor = LanguageProcessor()
    command_handler = CommandHandler(memory, vision, movement, language_processor)
    robot = Robot(simulator)

    print("Robot is ready. Awaiting commands...")

    try:
        while True:
            user_prompt = get_robot_request()
            relevancy, action, object_name, detail = command_handler.parse_user_input(user_prompt)

            if not relevancy:
                response = command_handler.handle_irrelevant_prompt(user_prompt)
            else:
                if action == "remember":
                    response = command_handler.handle_remember(object_name, detail)
                elif action == "recall":
                    response = command_handler.handle_recall(object_name, detail)
                elif action == "fetch":
                    destination = movement.find_current_location()
                    response = command_handler.handle_fetch(object_name, detail, destination)
                else:
                    response = "Command not recognized."

            command_handler.provide_feedback_to_user(response)
            time.sleep(1)  # Brief pause between commands

    except KeyboardInterrupt:
        print("\nShutting down simulation.")
        robot.shutdown()
        simulator.close()

if __name__ == "__main__":
    main()
