import json
from pathlib import Path

def generate_memory_json(file_path="memory.json"):
    # Define objects and their locations based on init_env
    memory_data = {
        "apple": {
            "detail": "A red apple",
            "location": {
                "text": "left shelf",
                "coords": [0.08, 0.3, 0.22]
            }
        },
        "pear": {
            "detail": "A green pear",
            "location": {
                "text": "left shelf",
                "coords": [-0.08, 0.3, 0.22]
            }
        },
        "tomato_soup": {
            "detail": "A can of tomato soup",
            "location": {
                "text": "right shelf",
                "coords": [-0.08, -0.3, 0.24]
            }
        },
        "banana": {
            "detail": "A yellow banana",
            "location": {
                "text": "right shelf",
                "coords": [0.08, -0.3, 0.22]
            }
        }
    }

    # Save to JSON file
    with open(file_path, "w") as f:
        json.dump(memory_data, f, indent=4)

    print(f"Memory JSON file saved at: {Path(file_path).resolve()}")

# Run the function to create the memory.json
if __name__ == "__main__":
    generate_memory_json()
