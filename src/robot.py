# /src/robot.py

from memory import Memory
from vision import VisionSystem
from movement import MovementSystem
from language_processor import LanguageProcessor
from command_handler import CommandHandler
from simulator import ManiSkillSimulator


class Robot:
    def __init__(self, simulator: ManiSkillSimulator):
        """
        Initialize the Robot with its core components.

        Args:
            simulator (ManiSkillSimulator): The robot simulator instance.
        """
        self.memory = Memory()
        self.vision = VisionSystem(simulator)
        self.movement = MovementSystem(simulator)
        self.language_processor = LanguageProcessor()
        self.command_handler = CommandHandler(
            self.memory, self.vision, self.movement, self.language_processor
        )

    def shutdown(self):
        """
        Perform any necessary cleanup before shutting down the robot.
        """
        self.memory.close()
        # Add any additional cleanup if necessary
