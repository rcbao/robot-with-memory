# /src/memory.py

import json
import re
from typing import Optional, Tuple, List
from object import Object  
import logging
import os


class Memory:
    def __init__(self, json_path: str = "memory.json"):
        """
        Initialize the Memory system with a JSON file.
        """
        self.json_path = json_path
        self.objects: List[dict] = []
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as file:
                self.objects = json.load(file)
                logging.info(f"Loaded memory from {self.json_path}.")
        else:
            with open(self.json_path, 'w') as file:
                json.dump(self.objects, file)
                logging.info(f"Created new memory file at {self.json_path}.")

    def save_object_to_memory(self, obj: Object):
        """
        Save an object to the memory JSON file.
        """
        obj_data = {
            "name": obj.name,
            "detail": obj.detail,
            "location_description": obj.location_description,
            "x": obj.location_3d_coords[0],
            "y": obj.location_3d_coords[1],
            "z": obj.location_3d_coords[2]
        }
        self.objects.append(obj_data)
        with open(self.json_path, 'w') as file:
            json.dump(self.objects, file, indent=4)
        logging.info(f"Saved object '{obj.name}' to memory.")

    def convert_name(self, name: str) -> str:
        # Remove numeric suffixes (e.g., "book_01-102" --> "book")
        name = re.sub(r'_\d+-\d+$', '', name)
        # Replace underscores with spaces (e.g., "indoor_plant" --> "indoor plant")
        return name.replace('_', ' ')

    def find_object_from_past_memory(self, object_name: str, object_detail: str) -> Optional[Object]:
        """
        Retrieve the most recent object matching the name and detail from memory, 
        with support for fuzzy name matching.
        """

        converted_object_name = self.convert_name(object_name.lower())

        for obj in reversed(self.objects):
            converted_name = self.convert_name(obj["name"].lower())
            if converted_name == converted_object_name and object_detail.lower() in obj["detail"].lower():
                logging.info(f"Found object '{obj['name']}' in memory.")
                return Object(
                    name=obj["name"],
                    detail=obj["detail"],
                    location_description=obj["location_description"],
                    location_3d_coords=(obj["x"], obj["y"], obj["z"])
                )
        logging.info(f"No matching object found in memory for '{object_name}'.")
        return None

if __name__ == "__main__":
    mem = Memory("../memory.json")
    name1 = mem.convert_name("book_01-102")
    name2 = mem.convert_name("indoor_plant_02-100")

    print(name1)
    print(name2)

    obj_1 = mem.find_object_from_past_memory("indoor plant", "")
    print(obj_1)