# /src/movement.py

from typing import Tuple
from simulator.mujoco_simulator import StretchMujocoSimulator


class MovementSystem:
    def __init__(self, simulator: StretchMujocoSimulator):
        """
        Initialize the MovementSystem with the Stretch Mujoco Simulator.

        Args:
            simulator (StretchMujocoSimulator): The robot simulator instance.
        """
        self.simulator = simulator
        self.holding_object = (
            False  # Flag to indicate if the robot is holding an object
        )

    def rotate_robot(self, angle: int):
        """
        Rotate the robot's base by a specific angle.

        Args:
            angle (int): The angle in degrees to rotate. Positive for clockwise, negative for counter-clockwise.
        """
        # Convert angle to radians if needed or use simulator's expected units
        # Assuming the simulator expects degrees
        print(f"Rotating robot by {angle} degrees.")
        self.simulator.set_base_velocity(0, angle * 0.01)  # Simplified rotation control
        # Allow some time for rotation to complete
        self.simulator.sim.step()
        self.simulator.viewer.render()

    def go_to(self, coords: Tuple[float, float, float]):
        """
        Move the robot's base to the target coordinates.

        Args:
            coords (Tuple[float, float, float]): The target (x, y, theta) coordinates.
        """
        print(
            f"Moving robot to coordinates: x={coords[0]}, y={coords[1]}, theta={coords[2]}"
        )
        self.simulator.move_to("base", coords)
        # Allow some time for movement to complete
        self.simulator.sim.step()
        self.simulator.viewer.render()

    def grasp_object(self, coords: Tuple[float, float, float]) -> bool:
        """
        Move to the object's location and perform a grasp action.

        Args:
            coords (Tuple[float, float, float]): The (x, y, theta) coordinates of the object.

        Returns:
            bool: True if grasping was successful, False otherwise.
        """
        print(
            f"Attempting to grasp object at coordinates: x={coords[0]}, y={coords[1]}, theta={coords[2]}"
        )
        try:
            # Move to the object's location
            self.go_to(coords)

            # Perform grasping action
            self.simulator.grasp_object()  # Assumes this method exists in the simulator
            self.holding_object = True
            print("Grasping action performed.")

            # Allow some time for grasping to complete
            self.simulator.sim.step()
            self.simulator.viewer.render()

            return True
        except Exception as e:
            print(f"Error during grasping: {e}")
            self.holding_object = False
            return False

    def release_object(self):
        """
        Release the currently held object.
        """
        if not self.holding_object:
            print("No object is currently being held to release.")
            return

        print("Releasing the held object.")
        try:
            self.simulator.release_object()  # Assumes this method exists in the simulator
            self.holding_object = False

            # Allow some time for releasing to complete
            self.simulator.sim.step()
            self.simulator.viewer.render()

            print("Object released successfully.")
        except Exception as e:
            print(f"Error during releasing: {e}")

    def find_current_location(self) -> Tuple[float, float, float]:
        """
        Retrieve the current location of the robot.

        Returns:
            Tuple[float, float, float]: The (x, y, theta) coordinates of the robot.
        """
        position = self.simulator.get_robot_position()  # Assumes this method exists
        print(
            f"Current robot location: x={position[0]}, y={position[1]}, theta={position[2]}"
        )
        return position
