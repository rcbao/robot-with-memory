import sys
from utils.logger import setup_logger
from robot_controller import RobotController
import logging
import warnings

RUNNING_DEMO = True

if RUNNING_DEMO:
    logging.disable(logging.CRITICAL)

logger = setup_logger()
warnings.filterwarnings("ignore", category=UserWarning)


def display_welcome_message():
    """
    Displays the welcome message to the user.
    """
    print("----------------------------------------")
    print("Welcome to the Robot Command Interface!")
    print("You can enter commands like 'Fetch the banana for me' or 'Where is the banana?'. Type 'exit' to quit.")
    print("----------------------------------------")

def main():
    # Initialize RobotController
    try:
        controller = RobotController()
    except SystemExit as e:
        print(e)
        sys.exit(1)

    display_welcome_message()

    try:
        while True:
            user_input = input(">> ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting. Goodbye!")
                logger.info("User exited the program.")
                break

            controller.process_command(user_input)

    except KeyboardInterrupt:
        print("\nExiting. Goodbye!")
        logger.info("User interrupted the program.")

    finally:
        controller.shutdown()

if __name__ == "__main__":
    main()
