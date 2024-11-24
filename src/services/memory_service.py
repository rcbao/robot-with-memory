import json
import os
from typing import Optional, Dict

MEMORY_FILE = "memory.json"

class MemoryService:
    def __init__(self, file_path: str = MEMORY_FILE):
        self.file_path = file_path
        self.memory = self.load_memory()

    def load_memory(self) -> list:
        """Load memory from a JSON file."""
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    print("Memory file is corrupted. Starting with an empty memory.")

        return []

    def save_memory(self):
        """Save current memory to a JSON file."""
        with open(self.file_path, "w") as f:
            json.dump(self.memory, f, indent=4)

    def add_object(self, name: str, detail: Optional[str], location: Dict):
        """Add a new object to memory."""
        # Check if object already exists
        for obj in self.memory:
            if obj["name"] == name:
                return

        # Add new object to memory
        self.memory.append({
            "name": name,
            "detail": detail,
            "location": location
        })
        self.save_memory()

    def update_location(self, name: str, new_location: Dict):
        """Update the location of an existing object."""
        for obj in self.memory:
            if obj["name"] == name:
                obj["location"] = new_location
                self.save_memory()
                return

        print(f"Object '{name}' not found in memory.")

    def get_object(self, name: str) -> Optional[Dict]:
        """Retrieve an object's details and location."""
        for obj in self.memory:
            if obj["name"] == name:
                return obj
        return None

    def remove_object(self, name: str):
        """Remove an object from memory."""
        for obj in self.memory:
            if obj["name"] == name:
                self.memory.remove(obj)
                self.save_memory()
                return

        print(f"Object '{name}' not found in memory.")

    def list_objects(self) -> list:
        """List all objects in memory."""
        return self.memory

    def clear_memory(self):
        """Clear all memory."""
        self.memory = []
        self.save_memory()
