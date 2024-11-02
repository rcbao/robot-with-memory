# generate_obj_locations.py

import json
import re
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_objects_from_txt(txt_path: str) -> list:
    """
    Load objects and their coordinates from a text file.
    """
    objects = []
    # Updated pattern to exclude 'frl_apartment_' and allow optional spaces
    pattern = re.compile(
        r"Object 'scs-\[0\]_objects/frl_apartment_([^']+)'\s+Position:\s+tensor\(\[\[\s*([-.\d]+),\s*([-.\d]+),\s*([-.\d]+)\s*\]\]\)"
    )
    if not os.path.exists(txt_path):
        logging.error(f"File {txt_path} does not exist.")
        return objects
    
    with open(txt_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                name, x, y, z = match.groups()
                obj = {
                    "name": name,  # Object name without 'frl_apartment_'
                    "detail": "",
                    "location_description": "",
                    "x": float(x),
                    "y": float(y),
                    "z": float(z)
                }
                objects.append(obj)
    logging.info(f"Loaded {len(objects)} objects from {txt_path}.")
    return objects

def save_to_json(objects: list, json_path: str):
    """
    Save the list of objects to a JSON file.
    """
    with open(json_path, 'w') as file:
        json.dump(objects, file, indent=4)
    logging.info(f"Saved {len(objects)} objects to {json_path}.")

if __name__ == "__main__":
    txt_path = "object_coordinates.txt"  # Path to input text file
    json_path = "memory.json"            # Path to output JSON file

    # Load objects from the text file and save them to JSON
    objects = load_objects_from_txt(txt_path)
    save_to_json(objects, json_path)
