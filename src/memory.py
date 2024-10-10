# /src/memory.py

import sqlite3
from typing import Optional, Tuple
from object import Object  # Ensure the Object class is defined in object.py


class Memory:
    def __init__(self, db_path: str = "memory.db"):
        """
        Initialize the Memory system with a SQLite database.

        Args:
            db_path (str): Path to the SQLite database file.
        """
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        """
        Create the objects table if it doesn't exist.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                detail TEXT,
                location_description TEXT,
                x REAL,
                y REAL,
                z REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        self.conn.commit()

    def save_object_to_memory(self, obj: Object):
        """
        Save an object to the memory database.

        Args:
            obj (Object): The object to save.
        """
        cursor = self.conn.cursor()
        x, y, z = (
            obj.location_3d_coords if obj.location_3d_coords else (None, None, None)
        )
        cursor.execute(
            """
            INSERT INTO objects (name, detail, location_description, x, y, z)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (obj.name, obj.detail, obj.location_description, x, y, z),
        )
        self.conn.commit()

    def find_object_from_past_memory(
        self, object_name: str, object_detail: str
    ) -> Optional[Object]:
        """
        Retrieve the most recent object matching the name and detail from memory.

        Args:
            object_name (str): The name of the object.
            object_detail (str): The detail/description of the object.

        Returns:
            Optional[Object]: The found object or None if not found.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT name, detail, location_description, x, y, z FROM objects
            WHERE name = ? AND detail = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """,
            (object_name, object_detail),
        )
        row = cursor.fetchone()
        if row:
            name, detail, location_description, x, y, z = row
            return Object(name, detail, location_description, (x, y, z))
        return None

    def close(self):
        """
        Close the database connection.
        """
        self.conn.close()

    def __del__(self):
        """
        Ensure the database connection is closed when the Memory instance is destroyed.
        """
        self.close()
