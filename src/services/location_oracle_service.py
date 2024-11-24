import json
from typing import Optional, Dict
from utils.logger import setup_logger

logger = setup_logger()

class Oracle:
    """
    Service to handle loading and retrieving object coordinates from the oracle.
    """
    def __init__(self, filepath: str = "object_coords_oracle.json"):
        self.filepath = filepath
        self.data = self.load_oracle()

    def load_oracle(self) -> Optional[Dict]:
        """
        Load object coordinates from a JSON file.
        """
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
                logger.info("Loaded object coordinates oracle.")
                return data
        except FileNotFoundError:
            logger.error(f"{self.filepath} file not found.")
            return None

    def get_object_coordinates(self, object_name: str) -> Optional[list]:
        """
        Retrieve object coordinates from the oracle.

        Args:
            object_name (str): Name of the object.

        Returns:
            list or None: Coordinates of the object if found, else None.
        """
        if not self.data:
            logger.error("Oracle data is not loaded.")
            return None
        try:
            matches = [item for item in  self.data if item["name"] == object_name.lower()]
            top_match = matches[0]
            coords = top_match["location"]["coords"]
            logger.info(f"Retrieved coordinates for '{object_name}': {coords}.")
            return coords
        except KeyError:
            logger.warning(f"Coordinates for '{object_name}' not found in oracle.")
            return None

    def get_valid_item_names(self) -> list:
        """
        Get a list of valid object names from the oracle.

        Returns:
            list: List of valid object names.
        """
        if not self.data:
            return []
        res = []

        for item in self.data:
            name = item["name"]
            aliases = item["aliases"]

            res.extend(aliases)
            res.append(name)
            
        return res
