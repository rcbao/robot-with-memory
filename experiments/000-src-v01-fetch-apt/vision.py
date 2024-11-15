# /src/vision.py

import cv2
from ultralytics import YOLO
from typing import List
from sim_object import Object  # Ensure object.py contains the Object class
from simulator import ManiSkillSimulator
import numpy as np
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisionSystem:
    def __init__(self, 
                 simulator: ManiSkillSimulator, 
                 model_path: str = "models/yolov8x.pt",
                 class_whitelist: List[str] = None):
        """
        Initialize the Vision System with YOLOv8 model and simulator.

        Args:
            simulator (ManiSkillSimulator): The robot simulator instance.
            model_path (str): Path to the YOLOv8 model file. Defaults to 'yolov8m.pt' for better accuracy.
            class_whitelist (List[str], optional): List of classes to detect. If None, detects all classes.
        """
        self.simulator = simulator
        self.model = YOLO(model_path)  # Load the YOLOv8 model
        self.class_whitelist = class_whitelist

        if self.class_whitelist:
            # Filter model's class names to include only whitelisted classes
            original_classes = self.model.names
            self.class_indices = [i for i, name in original_classes.items() if name in self.class_whitelist]
            if not self.class_indices:
                logger.warning("No classes matched the whitelist. All classes will be detected.")
                self.class_indices = list(original_classes.keys())
        else:
            self.class_indices = list(self.model.names.keys())

        logger.info(f"VisionSystem initialized with model '{model_path}'")
        if self.class_whitelist:
            logger.info(f"Class whitelist applied: {self.class_whitelist}")

    def get_robot_view_image(self) -> np.ndarray:
        """
        Capture the current image from the robot's camera feed.

        Returns:
            np.ndarray: The captured image in HWC format (Height, Width, Channels).
        """
        image = self.simulator.get_camera_image()
        return image

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image to ensure it's in the correct format for YOLOv8.

        Args:
            image (np.ndarray): The original image.

        Returns:
            np.ndarray: The preprocessed image.
        """
        # If the image has a batch dimension, remove it
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        
        # Convert image from BGR to RGB if necessary
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image

    def identify_objects_in_robot_view(self, confidence_threshold: float = 0.5, iou_threshold: float = 0.45) -> List[Object]:
        """
        Detect objects in the current robot view using YOLOv8.

        Args:
            confidence_threshold (float): Minimum confidence to consider a detection valid.
            iou_threshold (float): IoU threshold for non-max suppression.

        Returns:
            List[Object]: A list of detected Object instances.
        """
        image = self.get_robot_view_image()
        if image is None:
            logger.warning("No image captured from the camera.")
            return []

        # Preprocess the image
        image = self.preprocess_image(image)

        # Perform object detection with adjusted thresholds
        results = self.model(image, conf=confidence_threshold, iou=iou_threshold)

        detected_objects = []

        for result in results:
            for box in result.boxes:
                confidence = box.conf.item()
                class_id = int(box.cls.item())

                if class_id not in self.class_indices:
                    continue  # Skip classes not in the whitelist

                class_name = self.model.names[class_id]

                # Optional: Extract additional details if needed
                detail = class_name

                # Extract bounding box coordinates
                xmin, ymin, xmax, ymax = box.xyxy.tolist()[0]

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

                logger.debug(f"Detected {class_name} with confidence {confidence:.2f} at ({center_x}, {center_y}, {center_z})")

        logger.info(f"Total detected objects: {len(detected_objects)}")
        return detected_objects
