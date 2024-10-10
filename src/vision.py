# /src/vision.py

import cv2
from ultralytics import YOLO
from typing import List
from object import Object  # Ensure object.py contains the Object class
from maniskill_simulator import ManiSkillSimulator

class VisionSystem:
    def __init__(self, simulator: ManiSkillSimulator, model_path: str = "models/yolov8n.pt"):
        """
        Initialize the Vision System with YOLOv8 model and simulator.

        Args:
            simulator (ManiSkillSimulator): The robot simulator instance.
            model_path (str): Path to the YOLOv8 model file.
        """
        self.simulator = simulator
        self.model = YOLO(model_path)  # Load the YOLOv8 model

    def get_robot_view_image(self) -> any:
        """
        Capture the current image from the robot's camera feed.

        Returns:
            Any: The captured image in a format compatible with YOLOv8 (e.g., numpy array).
        """
        image = self.simulator.get_camera_image()
        return image

    def identify_objects_in_robot_view(self, confidence_threshold: float = 0.5) -> List[Object]:
        """
        Detect objects in the current robot view using YOLOv8.

        Args:
            confidence_threshold (float): Minimum confidence to consider a detection valid.

        Returns:
            List[Object]: A list of detected Object instances.
        """
        image = self.get_robot_view_image()
        if image is None:
            print("No image captured from the camera.")
            return []

        # Perform object detection
        results = self.model(image)

        detected_objects = []

        for result in results:
            for box in result.boxes:
                confidence = box.confidence
                if confidence < confidence_threshold:
                    continue

                # Extract bounding box coordinates
                xmin, ymin, xmax, ymax = box.xyxy.tolist()[0]
                class_id = int(box.cls)
                class_name = self.model.names[class_id]

                # Optional: Extract additional details if needed
                detail = class_name

                # Simulate 3D coordinates based on bounding box center
                # In ManiSkill, you might have access to more accurate positioning
                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2
                center_z = 0.0  # Placeholder for Z-coordinate

                location_description = f"Detected at pixel coordinates ({center_x:.2f}, {center_y:.2f})"

                detected_object = Object(
                    name=class_name,
                    detail=detail,
                    location_description=location_description,
                    location_3d_coords=(center_x, center_y, center_z)
                )
                detected_objects.append(detected_object)

        return detected_objects
