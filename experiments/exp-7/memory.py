import json
import os
from typing import Optional, Dict

MEMORY_FILE = "exp-7/memory.json"

class Memory:
    def __init__(self, file_path: str = MEMORY_FILE):
        self.file_path = file_path
        self.memory = self.load_memory()

    def load_memory(self) -> Dict:
        """Load memory from a JSON file."""
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    print("Memory file is corrupted. Starting with an empty memory.")

        return {}

    def save_memory(self):
        """Save current memory to a JSON file."""
        with open(self.file_path, "w") as f:
            json.dump(self.memory, f, indent=4)

    def add_object(self, name: str, detail: Optional[str], location: Dict):
        """Add a new object to memory."""
        self.memory[name] = {
            "detail": detail,
            "location": location
        }
        self.save_memory()

    def update_location(self, name: str, new_location: Dict):
        """Update the location of an existing object."""
        if name in self.memory:
            self.memory[name]["location"] = new_location
            self.save_memory()
        else:
            print(f"Object '{name}' not found in memory.")

    def get_object(self, name: str) -> Optional[Dict]:
        """Retrieve an object's details and location."""
        return self.memory.get(name, None)

    def remove_object(self, name: str):
        """Remove an object from memory."""
        if name in self.memory:
            del self.memory[name]
            self.save_memory()
        else:
            print(f"Object '{name}' not found in memory.")

    def list_objects(self) -> Dict:
        """List all objects in memory."""
        return self.memory

# Example Usage:
if __name__ == "__main__":
    mem = Memory()

    # # Add an object
    # mem.add_object(
    #     name="apple",
    #     detail="A red apple",
    #     location={"shelf": 1, "position": [0.08, 0.3, 0.22]}
    # )

    # # Update object's location
    # mem.update_location(
    #     name="apple",
    #     new_location={"shelf": 2, "position": [0.10, -0.25, 0.22]}
    # )

    # Retrieve an object
    apple = mem.get_object("banana")
    print("Banana Details:", apple)

    # List all objects
    all_objects = mem.list_objects()
    print("All Objects in Memory:", all_objects)

    # Remove an object
    mem.remove_object("apple")
