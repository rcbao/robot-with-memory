# /src/memory.py

import sqlite3
from typing import Optional, Tuple
from object import Object  # Ensure the Object class is defined in object.py
import logging



class Memory:
    def __init__(self, db_path: str = "memory.db"):
        """
        Initialize the Memory system with a SQLite database.

        Optional argument:
            db_path (str): Path to the SQLite database file.
        """
        self.db_path = db_path
        try:
            self.conn = sqlite3.connect(self.db_path)
            logging.info(f"Connected to SQLite database at {self.db_path}.")
            self.create_table()
        except sqlite3.Error as e:
            logging.error(f"Failed to connect to database: {e}")
            raise

    def create_table(self):
        """
        Create the objects table if it doesn't exist.
        """
        try:
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
            logging.info("Ensured that the 'objects' table exists in the database.")
        except sqlite3.Error as e:
            logging.error(f"Failed to create table: {e}")
            raise

    def save_object_to_memory(self, obj: Object):
        """
        Save an object to the memory database.

        Args:
            obj (Object): The object to save.
        """
        try:
            cursor = self.conn.cursor()
            x, y, z = obj.location_3d_coords
            cursor.execute(
                """
                INSERT INTO objects (name, detail, location_description, x, y, z)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (obj.name, obj.detail, obj.location_description, x, y, z),
            )
            self.conn.commit()
            logging.info(f"Saved object '{obj.name}' to memory.")
        except sqlite3.Error as e:
            logging.error(f"Failed to save object to memory: {e}")
            raise

    def find_object_from_past_memory(self, object_name: str, object_detail: str) -> Optional[Object]:
        """
        Retrieve the most recent object matching the name and detail from memory.

        Arguments:
            object_name (str): The name of the object.
            object_detail (str): The detail/description of the object.

        Returns:
            Optional[Object]: The found object or None if not found.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT name, detail, location_description, x, y, z FROM objects
                WHERE LOWER(name) = LOWER(?) AND LOWER(detail) LIKE LOWER(?)
                ORDER BY timestamp DESC
                LIMIT 1
            """,
                (object_name, f"%{object_detail}%"),
            )
            row = cursor.fetchone()
            if row:
                name, detail, location_description, x, y, z = row
                logging.info(f"Found object '{name}' in past memory.")
                return Object(name, detail, location_description, (x, y, z))
            logging.info(f"No matching object found in past memory for '{object_name}'.")
            return None
        except sqlite3.Error as e:
            logging.error(f"Failed to retrieve object from memory: {e}")
            raise

    def close(self):
        """
        Close the database connection.
        """
        try:
            self.conn.close()
            logging.info("Closed the database connection.")
        except sqlite3.Error as e:
            logging.error(f"Failed to close database connection: {e}")

    def __del__(self):
        """
        Ensure the database connection is closed when the memory instance is destroyed.
        """
        self.close()