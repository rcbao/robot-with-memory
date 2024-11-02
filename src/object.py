# /src/object.py

from typing import Optional, Tuple


class Object:
    def __init__(
        self,
        name: str,
        detail: str,
        location_description: str,
        location_3d_coords: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Initialize an Object instance representing an item in the robot's environment.

        Args:
            name (str): The name of the object (e.g., "cup").
            detail (str): Additional details or attributes of the object (e.g., "red", "ceramic").
            location_description (str): A textual description of the object's location (e.g., "kitchen table").
            location_3d_coords (Optional[Tuple[float, float, float]]): The object's 3D coordinates
        """
        self.name = name
        self.detail = detail
        self.location_description = location_description
        self.location_3d_coords = location_3d_coords

    def __repr__(self) -> str:
        """
        Provide a readable string representation of the Object instance.

        Returns:
            str: String representation of the object.
        """
        return (
            f"Object(name='{self.name}', detail='{self.detail}', "
            f"location_description='{self.location_description}', "
            f"location_3d_coords={self.location_3d_coords})"
        )

    def __eq__(self, other) -> bool:
        """
        Compare two Object instances for equality based on their attributes.

        Args:
            other (Object): The other object to compare with.

        Returns:
            bool: True if all attributes are equal, False otherwise.
        """
        if not isinstance(other, Object):
            return False
        return (
            self.name == other.name
            and self.detail == other.detail
            and self.location_description == other.location_description
            and self.location_3d_coords == other.location_3d_coords
        )
