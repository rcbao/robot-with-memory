import time
from memory import Memory
from vision import VisionSystem
from movement import MovementSystem
from language_processor import LanguageProcessor
from command_handler import CommandHandler
from robot import Robot
from simulator.mujoco_simulator import StretchMujocoSimulator
from object import Object


def main_poc():
    # Initialize the simulator
    simulator = StretchMujocoSimulator("./simulator/scene.xml")
    simulator.start()
    simulator.home()

    # Initialize the Robot
    robot = Robot(simulator)

    print("Proof-of-Concept: Robot is ready.")

    # Define the object to remember
    poc_object = Object(
        name="red cup",
        detail="ceramic",
        location_description="kitchen table",
        location_3d_coords=(1.0, 2.0, 0.0),  # Example coordinates
    )

    # Simulate placing the object in the environment
    simulator.place_object(poc_object)
    print(f"Placed object: {poc_object.name} at {poc_object.location_description}")

    time.sleep(2)  # Wait for the simulator to update

    # Command 1: Remember the object
    remember_command = "Remember the red cup on the kitchen table."
    print(f"\nUser Command: {remember_command}")
    relevancy, action, object_name, detail = (
        robot.command_handler.interpret_user_prompt(remember_command)
    )

    if relevancy:
        if action == "remember":
            response = robot.command_handler.handle_remember(object_name, detail)
        else:
            response = "Unexpected action for the PoC."
    else:
        response = robot.command_handler.handle_irrelevant_prompt(remember_command)

    robot.command_handler.provide_feedback_to_user(response)

    time.sleep(2)  # Wait before the next command

    # Command 2: Recall the object
    recall_command = "Recall where the red cup is."
    print(f"\nUser Command: {recall_command}")
    relevancy, action, object_name, detail = (
        robot.command_handler.interpret_user_prompt(recall_command)
    )

    if relevancy:
        if action == "recall":
            response = robot.command_handler.handle_recall(object_name, detail)
        else:
            response = "Unexpected action for the PoC."
    else:
        response = robot.command_handler.handle_irrelevant_prompt(recall_command)

    robot.command_handler.provide_feedback_to_user(response)

    # Shutdown the robot and simulator
    print("\nProof-of-Concept completed. Shutting down.")
    robot.shutdown()
    simulator.stop()


if __name__ == "__main__":
    main_poc()
