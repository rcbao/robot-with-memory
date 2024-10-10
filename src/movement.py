# /src/movement.py

from maniskill_simulator import ManiSkillSimulator
from typing import Tuple

class MovementSystem:
    def __init__(self, simulator: ManiSkillSimulator):
        """
        Initialize the Movement System with the simulator.

        Args:
            simulator (ManiSkillSimulator): The robot simulator instance.
        """
        self.simulator = simulator

    def rotate_robot(self, angle: int):
        """
        Rotate the robot's base by a specific angle.

        Args:
            angle (int): The angle to rotate in degrees.
        """
        # Convert angle to radians if necessary
        radians = angle * (3.14159265 / 180)
        self.simulator.set_base_velocity(0, radians * 0.01)  # Adjust scaling as per ManiSkill's API

    def go_to(self, coords: Tuple[float, float, float]):
        """
        Use the simulator to move to the target coordinates.

        Args:
            coords (Tuple[float, float, float]): Target (x, y, theta).
        """
        self.simulator.move_to("base", coords)

    def grasp_object(self, coords: Tuple[float, float, float]) -> bool:
        """
        Move to the object's location and perform a grasp action.

        Args:
            coords (Tuple[float, float, float]): Object's (x, y, theta).

        Returns:
            bool: True if grasping was successful, False otherwise.
        """
        self.go_to(coords)
        # Implement grasp logic based on ManiSkill's API
        success = self.simulator.grasp()
        return success

    def release_object(self):
        """
        Release the currently held object.
        """
        self.simulator.release()

    def find_current_location(self) -> Tuple[float, float, float]:
        """
        Use the simulator to find the current location of the robot.

        Returns:
            Tuple[float, float, float]: Current (x, y, theta).
        """
        status = self.simulator.get_status("base")
        return status["x"], status["y"], status["theta"]
